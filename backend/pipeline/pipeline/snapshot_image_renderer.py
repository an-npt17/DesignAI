from __future__ import annotations

import base64
import binascii
import json
import logging
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Literal
from uuid import uuid4

import httpx
from PIL import Image, UnidentifiedImageError

from config import config_file, load_config, root_config
from pipeline.image_flow_logging import (
    log_image_flow_event,
    redact_for_image_log,
    summarize_gemini_payload,
    summarize_gemini_response,
    summarize_image_data_url,
)
from pipeline.snapshot_prompt_compiler import compile_snapshot_prompt

logger = logging.getLogger(__name__)

GEMINI_IMAGE_API_KEY_ENV = "GEMINI_IMAGE_API_KEY"
GEMINI_IMAGE_BASE_URL_ENV = "GEMINI_IMAGE_BASE_URL"
SNAPSHOT_RENDER_IMAGE_MODEL_ENV = "TKNT_SNAPSHOT_RENDER_IMAGE_MODEL"
SNAPSHOT_RENDER_IMAGE_SIZE_ENV = "TKNT_SNAPSHOT_RENDER_IMAGE_SIZE"
SNAPSHOT_RENDER_MAX_TOKENS_ENV = "TKNT_SNAPSHOT_RENDER_MAX_TOKENS"
SNAPSHOT_RENDER_INCLUDE_LAYOUT_2D_REFERENCE_ENV = (
    "TKNT_SNAPSHOT_RENDER_INCLUDE_LAYOUT_2D_REFERENCE"
)
IMAGE_PROVIDER_ENV = "TKNT_IMAGE_PROVIDER"
OPENAI_IMAGE_API_KEY_ENV = "OPENAI_IMAGE_API_KEY"
OPENAI_IMAGE_BASE_URL_ENV = "OPENAI_IMAGE_BASE_URL"
OPENAI_IMAGE_MODEL_ENV = "TKNT_OPENAI_IMAGE_MODEL"
OPENAI_IMAGE_SIZE_ENV = "TKNT_OPENAI_IMAGE_SIZE"
OPENAI_IMAGE_QUALITY_ENV = "TKNT_OPENAI_IMAGE_QUALITY"

_DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
_DEFAULT_IMAGE_MODEL = "gemini-3.1-flash-image-preview"
_DEFAULT_IMAGE_SIZE = "1K"
_DEFAULT_MAX_TOKENS = 256
_DEFAULT_IMAGE_PROVIDER = "openai"
_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_OPENAI_IMAGE_MODEL = "gpt-image-1.5"
_DEFAULT_OPENAI_IMAGE_SIZE = "1536x1024"
_DEFAULT_OPENAI_IMAGE_QUALITY = "medium"
_SUPPORTED_OPENAI_IMAGE_SIZES = {
    "1024x1024",
    "1024x1536",
    "1536x1024",
    "auto",
}
_SUPPORTED_OPENAI_IMAGE_QUALITIES = {"low", "medium", "high", "auto"}
_GEMINI_INLINE_REQUEST_LIMIT_BYTES = 20_000_000
_BYTES_PER_MB = 1_000_000
_DEFAULT_USER_PROMPT = (
    "Convert the exact visible 3D blockout into a photorealistic interior by "
    "improving material realism, lighting, and finishes. Keep the same camera, "
    "architecture, room shell, openings, object count, object colors, object "
    "shapes, main furniture, and visible layout."
)
_SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".gif")
_SUPPORTED_ASPECT_RATIOS: tuple[tuple[str, float], ...] = (
    ("1:1", 1.0),
    ("1:4", 1 / 4),
    ("1:8", 1 / 8),
    ("2:3", 2 / 3),
    ("3:2", 3 / 2),
    ("3:4", 3 / 4),
    ("4:1", 4.0),
    ("4:3", 4 / 3),
    ("4:5", 4 / 5),
    ("5:4", 5 / 4),
    ("8:1", 8.0),
    ("9:16", 9 / 16),
    ("16:9", 16 / 9),
    ("21:9", 21 / 9),
)
_GEMINI_31_IMAGE_OUTPUT_SIZES: dict[str, dict[str, tuple[int, int]]] = {
    "0.5K": {
        "1:1": (512, 512),
        "1:4": (256, 1024),
        "1:8": (192, 1536),
        "2:3": (424, 632),
        "3:2": (632, 424),
        "3:4": (448, 600),
        "4:1": (1024, 256),
        "4:3": (600, 448),
        "4:5": (464, 576),
        "5:4": (576, 464),
        "8:1": (1536, 192),
        "9:16": (384, 688),
        "16:9": (688, 384),
        "21:9": (792, 336),
    },
    "1K": {
        "1:1": (1024, 1024),
        "1:4": (512, 2048),
        "1:8": (384, 3072),
        "2:3": (848, 1264),
        "3:2": (1264, 848),
        "3:4": (896, 1200),
        "4:1": (2048, 512),
        "4:3": (1200, 896),
        "4:5": (928, 1152),
        "5:4": (1152, 928),
        "8:1": (3072, 384),
        "9:16": (768, 1376),
        "16:9": (1376, 768),
        "21:9": (1584, 672),
    },
    "2K": {
        "1:1": (2048, 2048),
        "1:4": (1024, 4096),
        "1:8": (768, 6144),
        "2:3": (1696, 2528),
        "3:2": (2528, 1696),
        "3:4": (1792, 2400),
        "4:1": (4096, 1024),
        "4:3": (2400, 1792),
        "4:5": (1856, 2304),
        "5:4": (2304, 1856),
        "8:1": (6144, 768),
        "9:16": (1536, 2752),
        "16:9": (2752, 1536),
        "21:9": (3168, 1344),
    },
    "4K": {
        "1:1": (4096, 4096),
        "1:4": (2048, 8192),
        "1:8": (1536, 12288),
        "2:3": (3392, 5056),
        "3:2": (5056, 3392),
        "3:4": (3584, 4800),
        "4:1": (8192, 2048),
        "4:3": (4800, 3584),
        "4:5": (3712, 4608),
        "5:4": (4608, 3712),
        "8:1": (12288, 1536),
        "9:16": (3072, 5504),
        "16:9": (5504, 3072),
        "21:9": (6336, 2688),
    },
}
_GEMINI_3_PRO_IMAGE_OUTPUT_SIZES: dict[str, dict[str, tuple[int, int]]] = {
    size: {
        aspect_ratio: dimensions
        for aspect_ratio, dimensions in aspect_sizes.items()
        if aspect_ratio not in {"1:4", "1:8", "4:1", "8:1"}
    }
    for size, aspect_sizes in _GEMINI_31_IMAGE_OUTPUT_SIZES.items()
    if size != "0.5K"
}
_GEMINI_25_IMAGE_OUTPUT_SIZES: dict[str, dict[str, tuple[int, int]]] = {
    "1K": {
        "1:1": (1024, 1024),
        "2:3": (832, 1248),
        "3:2": (1248, 832),
        "3:4": (864, 1184),
        "4:3": (1184, 864),
        "4:5": (896, 1152),
        "5:4": (1152, 896),
        "9:16": (768, 1344),
        "16:9": (1344, 768),
        "21:9": (1536, 672),
    }
}
_GENERATE_SYSTEM_PROMPT = (
    "You are converting a coarse 3D interior blockout into a final photorealistic render.\n\n"
    "Use the input 3D layout image as the exact source of truth for the camera view, framing, room layout, room shell, openings, furniture placement, object scale, object shape, object color, surface color, and occlusion.\n"
    "The labels, dark structural edge overlays, and guide marks are only there to identify objects and boundaries. Never render labels, guide lines, callouts, colored boxes, black outline strokes, UI marks, numbers, or any other annotation.\n"
    "Object reference images may be provided for appearance memory. Use them only for the paired object's style, material, and identity, never for camera or layout.\n"
    "Do not add, remove, move, resize, rotate, straighten, center, or replace architecture, openings, built-ins, or main furniture unless the prompt explicitly assigns an object reference to it.\n"
    "Preserve the exact visible wall panels and corner seams from the 3D room shell. Do not omit a wall, erase a corner, open a closed side of the room, or move any wall edge.\n"
    "The existing furniture and room surfaces may gain realistic texture and finish detail, but their visible color palette, silhouette, footprint, pose, size, and position stay locked unless the user explicitly changed them in the layout before rendering.\n"
    "Do not add decorative accessories, rugs, plants, wall art, pillows, bedding, curtains, or surface decor unless those objects already exist in the 3D layout or are explicitly requested.\n"
    "Do not invent extra walls, doors, windows, columns, stairs, fireplaces, cabinets, wardrobes, beds, desks, tables, sofas, chairs, or other main furniture.\n"
    "Render a photorealistic version of the exact view shown in the 3D layout image. Treat this as visual refinement, not a redesign.\n"
    "No people, no text, no watermark, no room-type change."
)
_EDIT_SYSTEM_PROMPT = (
    "You are editing an existing photorealistic interior image.\n\n"
    "The first input image is the photorealistic source image to edit.\n"
    "If a labeled 3D layout reference image is provided, use it only as the target camera/view and layout guide. Match that view exactly, but do not reproduce any label, guide line, callout, colored box, black outline stroke, UI mark, number, or annotation in the output.\n"
    "If an edit-region guide image is provided, it contains colored bounding boxes and region ID labels (such as R1, R2, R3) drawn over a copy of the photo. Use the boxes and labels only to locate which regions to edit and what operation applies to each. The boxes, ID text, outlines, colored fills, tinted overlays, and all other guide graphics are instructions only — they must never appear in the output image.\n"
    "If replacement reference images are provided, use them only for the appearance of the explicitly paired object or region.\n"
    "Preserve room identity, materials, style continuity, architecture, openings, and every unedited object unless the prompt explicitly asks for a change.\n"
    "Output a clean photorealistic interior photograph. No colored rectangles, no region IDs, no label text, no guide graphics, no annotations of any kind, no people, no watermarks, no room-type change."
)

RenderMode = Literal["generate", "edit"]
SceneReferenceMode = Literal[
    "none",
    "target_layout_with_scene_reference",
    "scene_reference_camera_transfer",
]


@dataclass(frozen=True)
class SnapshotImageRenderConfig:
    image_model_name: str = _DEFAULT_IMAGE_MODEL
    image_size: str = _DEFAULT_IMAGE_SIZE
    max_output_tokens: int = _DEFAULT_MAX_TOKENS
    gemini_base_url: str = _DEFAULT_GEMINI_BASE_URL
    gemini_api_key: str | None = None
    include_layout_2d_reference: bool = False
    image_provider: str = _DEFAULT_IMAGE_PROVIDER
    openai_api_key: str | None = None
    openai_image_model: str = _DEFAULT_OPENAI_IMAGE_MODEL
    openai_image_size: str = _DEFAULT_OPENAI_IMAGE_SIZE
    openai_image_quality: str = _DEFAULT_OPENAI_IMAGE_QUALITY
    openai_base_url: str = _DEFAULT_OPENAI_BASE_URL

    @classmethod
    def from_env(cls) -> SnapshotImageRenderConfig:
        image_config = root_config.services.gemini_image
        image_size = _string(
            os.getenv(SNAPSHOT_RENDER_IMAGE_SIZE_ENV),
            fallback=image_config.image_size or _DEFAULT_IMAGE_SIZE,
        ).upper()
        max_tokens = _read_int_env(
            SNAPSHOT_RENDER_MAX_TOKENS_ENV,
            default=image_config.max_output_tokens,
        )
        env_api_key = _secret_string(os.getenv(GEMINI_IMAGE_API_KEY_ENV))
        config_api_key = _secret_string(image_config.api_key)

        image_provider = _string(
            os.getenv(IMAGE_PROVIDER_ENV),
            fallback=_DEFAULT_IMAGE_PROVIDER,
        ).lower()
        openai_api_key = _secret_string(os.getenv(OPENAI_IMAGE_API_KEY_ENV))
        if not openai_api_key:
            try:
                from config.openai_config import OpenAIConfig  # noqa: PLC0415

                openai_api_key = OpenAIConfig.OPENAI_API_KEY
            except Exception:
                pass
        openai_image_model = _string(
            os.getenv(OPENAI_IMAGE_MODEL_ENV),
            fallback=_DEFAULT_OPENAI_IMAGE_MODEL,
        )
        openai_image_size = _normalize_openai_image_size(
            os.getenv(OPENAI_IMAGE_SIZE_ENV)
        )
        openai_image_quality = _normalize_openai_image_quality(
            os.getenv(OPENAI_IMAGE_QUALITY_ENV)
        )
        openai_base_url = _string(
            os.getenv(OPENAI_IMAGE_BASE_URL_ENV),
            fallback=_DEFAULT_OPENAI_BASE_URL,
        )

        return cls(
            image_model_name=_normalize_gemini_model_name(
                _string(
                    os.getenv(SNAPSHOT_RENDER_IMAGE_MODEL_ENV),
                    fallback=image_config.model or _DEFAULT_IMAGE_MODEL,
                )
            ),
            image_size=image_size or _DEFAULT_IMAGE_SIZE,
            max_output_tokens=max(32, max_tokens),
            gemini_base_url=_string(
                os.getenv(GEMINI_IMAGE_BASE_URL_ENV),
                fallback=image_config.base_url or _DEFAULT_GEMINI_BASE_URL,
            ),
            gemini_api_key=env_api_key or config_api_key,
            include_layout_2d_reference=_read_bool_env(
                SNAPSHOT_RENDER_INCLUDE_LAYOUT_2D_REFERENCE_ENV,
                default=bool(image_config.include_layout_2d_reference),
            ),
            image_provider=image_provider,
            openai_api_key=openai_api_key,
            openai_image_model=openai_image_model,
            openai_image_size=openai_image_size,
            openai_image_quality=openai_image_quality,
            openai_base_url=openai_base_url,
        )


@dataclass(frozen=True)
class SnapshotEditOperation:
    object_id: str
    object_name: str | None = None
    operation_type: Literal["prompt", "add", "replace", "recolor"] | None = None
    prompt: str | None = None
    bbox_norm: Mapping[str, float] | None = None
    replacement_image_data_url: str | None = None
    target_color: str | None = None


@dataclass(frozen=True)
class _GeminiRequestAttempt:
    name: str
    payload: dict[str, object]
    generation_config_applied: bool
    retry_on_http_400: bool = False


class _GeminiGenerateContentHttpError(RuntimeError):
    def __init__(
        self,
        *,
        message: str,
        status_code: int,
        response_body: object,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


def render_snapshot_image(
    snapshot_payload: Mapping[str, object],
    *,
    snapshot_image_data_url: str,
    user_prompt: str | None = None,
    layout_reference_image_data_url: str | None = None,
    annotated_reference_image_data_url: str | None = None,
    scene_reference_image_data_url: str | None = None,
    scene_reference_mode: SceneReferenceMode = "none",
    render_mode: RenderMode = "generate",
    preset_selection: Mapping[str, object] | None = None,
    edit_operations: list[SnapshotEditOperation] | None = None,
    edit_source_image_data_url: str | None = None,
    config: SnapshotImageRenderConfig | None = None,
    trace_id: str | None = None,
) -> dict[str, object]:
    image_flow_trace_id = trace_id or uuid4().hex
    resolved_config = config or SnapshotImageRenderConfig.from_env()
    structural_image_data_url = _normalize_image_data_url(snapshot_image_data_url)
    normalized_scene_reference = (
        _normalize_image_data_url(scene_reference_image_data_url)
        if scene_reference_image_data_url
        else None
    )
    resolved_scene_reference_mode = _resolve_scene_reference_mode(
        scene_reference_mode=scene_reference_mode,
        render_mode=render_mode,
        scene_reference_image_data_url=normalized_scene_reference,
    )
    reference_only_camera_transfer_used = (
        resolved_scene_reference_mode == "scene_reference_camera_transfer"
    )
    scene_reference_used = (
        resolved_scene_reference_mode == "target_layout_with_scene_reference"
    )
    source_image_data_url = (
        _normalize_image_data_url(edit_source_image_data_url)
        if render_mode == "edit" and edit_source_image_data_url
        else normalized_scene_reference
        if reference_only_camera_transfer_used
        and normalized_scene_reference is not None
        else structural_image_data_url
    )
    visible_objects = _extract_visible_objects(snapshot_payload)
    normalized_operations = _normalize_edit_operations(
        edit_operations or [],
        visible_objects=visible_objects,
    )
    preset_prompt = _build_preset_prompt(preset_selection)
    normalized_annotated_reference = (
        _normalize_image_data_url(annotated_reference_image_data_url)
        if annotated_reference_image_data_url
        else None
    )
    annotated_reference_used = normalized_annotated_reference is not None
    normalized_layout_reference = (
        _normalize_image_data_url(layout_reference_image_data_url)
        if layout_reference_image_data_url
        else None
    )
    layout_reference_enabled = (
        resolved_config.include_layout_2d_reference or render_mode == "edit"
    )
    layout_reference_used = (
        render_mode == "generate"
        and not reference_only_camera_transfer_used
        and not annotated_reference_used
        and resolved_config.include_layout_2d_reference
        and normalized_layout_reference is not None
    ) or (render_mode == "edit" and normalized_layout_reference is not None)
    layout_lock_prompt = (
        _build_generate_layout_lock_prompt(
            scene_reference_mode=resolved_scene_reference_mode,
            snapshot_payload=snapshot_payload,
        )
        if render_mode == "generate"
        else ""
    )
    normalized_user_prompt = _build_final_user_prompt(
        user_prompt=user_prompt,
        render_mode=render_mode,
        preset_prompt=preset_prompt,
        edit_operations=normalized_operations,
        visible_objects=visible_objects,
        layout_lock_prompt=layout_lock_prompt,
        layout_reference_used=layout_reference_used,
        annotated_reference_used=annotated_reference_used,
        scene_reference_mode=resolved_scene_reference_mode,
    )
    aspect_ratio = _resolve_canvas_aspect_ratio(snapshot_payload)
    reference_images = [
        operation.replacement_image_data_url
        for operation in normalized_operations
        if operation.replacement_image_data_url is not None
    ]
    effective_model_name = (
        resolved_config.openai_image_model
        if resolved_config.image_provider == "openai"
        else resolved_config.image_model_name
    )

    log_image_flow_event(
        "snapshot_image.render_input",
        {
            "trace_id": image_flow_trace_id,
            "render_mode": render_mode,
            "image_provider": resolved_config.image_provider,
            "model_name": effective_model_name,
            "image_size": resolved_config.image_size,
            "max_output_tokens": resolved_config.max_output_tokens,
            "aspect_ratio": aspect_ratio,
            "scene_reference_mode": resolved_scene_reference_mode,
            "layout_reference_enabled": layout_reference_enabled,
            "layout_reference_used": layout_reference_used,
            "annotated_reference_used": annotated_reference_used,
            "scene_reference_used": scene_reference_used,
            "reference_only_camera_transfer_used": (
                reference_only_camera_transfer_used
            ),
            "raw_user_prompt": _string(user_prompt) or None,
            "normalized_user_prompt": normalized_user_prompt,
            "preset_prompt": preset_prompt,
            "preset_selection": dict(preset_selection or {}),
            "visible_object_count": len(visible_objects),
            "visible_object_ids": list(visible_objects.keys()),
            "edit_operations": [
                _serialize_edit_operation(operation)
                for operation in normalized_operations
            ],
            "snapshot_payload": dict(snapshot_payload),
            "images": {
                "source": summarize_image_data_url(source_image_data_url),
                "structural": summarize_image_data_url(structural_image_data_url),
                "annotated_reference": summarize_image_data_url(
                    normalized_annotated_reference if annotated_reference_used else None
                ),
                "layout_reference": summarize_image_data_url(
                    normalized_layout_reference if layout_reference_used else None
                ),
                "scene_reference": summarize_image_data_url(
                    normalized_scene_reference
                    if resolved_scene_reference_mode != "none"
                    else None
                ),
                "replacement_references": [
                    summarize_image_data_url(data_url) for data_url in reference_images
                ],
            },
        },
    )

    active_system_prompt = (
        _EDIT_SYSTEM_PROMPT if render_mode == "edit" else _GENERATE_SYSTEM_PROMPT
    )
    try:
        if resolved_config.image_provider == "openai":
            image_result = _generate_image_with_openai(
                trace_id=image_flow_trace_id,
                model_name=resolved_config.openai_image_model,
                aspect_ratio=aspect_ratio,
                system_prompt=active_system_prompt,
                user_prompt=normalized_user_prompt,
                source_image_data_url=source_image_data_url,
                annotated_reference_image_data_url=(
                    normalized_annotated_reference if annotated_reference_used else None
                ),
                scene_reference_image_data_url=(
                    normalized_scene_reference if scene_reference_used else None
                ),
                layout_reference_image_data_url=(
                    normalized_layout_reference if layout_reference_used else None
                ),
                reference_image_data_urls=reference_images,
                edit_operations=normalized_operations,
                openai_api_key=resolved_config.openai_api_key,
                openai_image_size=resolved_config.openai_image_size,
                openai_image_quality=resolved_config.openai_image_quality,
                openai_base_url=resolved_config.openai_base_url,
            )
        else:
            image_result = _generate_image_with_gemini(
                trace_id=image_flow_trace_id,
                model_name=resolved_config.image_model_name,
                image_size=resolved_config.image_size,
                max_output_tokens=resolved_config.max_output_tokens,
                gemini_base_url=resolved_config.gemini_base_url,
                gemini_api_key=resolved_config.gemini_api_key,
                aspect_ratio=aspect_ratio,
                system_prompt=active_system_prompt,
                user_prompt=normalized_user_prompt,
                source_image_data_url=source_image_data_url,
                annotated_reference_image_data_url=(
                    normalized_annotated_reference if annotated_reference_used else None
                ),
                scene_reference_image_data_url=(
                    normalized_scene_reference if scene_reference_used else None
                ),
                layout_reference_image_data_url=(
                    normalized_layout_reference if layout_reference_used else None
                ),
                reference_image_data_urls=reference_images,
            )
    except Exception as exc:
        log_image_flow_event(
            "snapshot_image.render_error",
            {
                "trace_id": image_flow_trace_id,
                "render_mode": render_mode,
                "model_name": effective_model_name,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise

    log_image_flow_event(
        "snapshot_image.render_output",
        {
            "trace_id": image_flow_trace_id,
            "render_mode": render_mode,
            "model_name": effective_model_name,
            "image": redact_for_image_log(image_result),
        },
    )

    return {
        "request": {
            "render_mode": render_mode,
            "system_prompt": active_system_prompt,
            "user_prompt": normalized_user_prompt,
            "raw_user_prompt": _string(user_prompt) or None,
            "preset_prompt": preset_prompt,
            "preset_selection": dict(preset_selection or {}),
            "aspect_ratio": aspect_ratio,
            "source_image_mime_type": _extract_data_url_mime_type(
                source_image_data_url
            ),
            "structural_image_mime_type": _extract_data_url_mime_type(
                structural_image_data_url
            ),
            "layout_reference_enabled": layout_reference_enabled,
            "layout_reference_used": layout_reference_used,
            "layout_reference_image_mime_type": (
                _extract_data_url_mime_type(normalized_layout_reference)
                if layout_reference_used and normalized_layout_reference is not None
                else None
            ),
            "annotated_reference_used": annotated_reference_used,
            "annotated_reference_image_mime_type": (
                _extract_data_url_mime_type(normalized_annotated_reference)
                if annotated_reference_used
                and normalized_annotated_reference is not None
                else None
            ),
            "scene_reference_mode": resolved_scene_reference_mode,
            "scene_reference_used": scene_reference_used,
            "reference_only_camera_transfer_used": (
                reference_only_camera_transfer_used
            ),
            "scene_reference_image_mime_type": (
                _extract_data_url_mime_type(normalized_scene_reference)
                if resolved_scene_reference_mode != "none"
                and normalized_scene_reference is not None
                else None
            ),
            "reference_image_count": len(reference_images),
            "edit_operations": [
                _serialize_edit_operation(operation)
                for operation in normalized_operations
            ],
        },
        "image": image_result,
        "models": {
            "image_model_name": effective_model_name,
            "max_tokens": resolved_config.max_output_tokens,
        },
        "metadata": {
            "camera": snapshot_payload.get("camera"),
            "visible_objects": list(visible_objects.values()),
            "visible_object_ids": list(visible_objects.keys()),
            "render_mode": render_mode,
            "preset_selection": dict(preset_selection or {}),
            "edit_operations": [
                _serialize_edit_operation(operation)
                for operation in normalized_operations
            ],
        },
    }


def render_snapshot_image_from_path(
    snapshot_path: str | Path,
    *,
    user_prompt: str | None = None,
    snapshot_image_data_url: str | None = None,
    layout_reference_image_data_url: str | None = None,
    annotated_reference_image_data_url: str | None = None,
    scene_reference_image_data_url: str | None = None,
    scene_reference_mode: SceneReferenceMode = "none",
    render_mode: RenderMode = "generate",
    preset_selection: Mapping[str, object] | None = None,
    edit_operations: list[SnapshotEditOperation] | None = None,
    edit_source_image_data_url: str | None = None,
    config: SnapshotImageRenderConfig | None = None,
    trace_id: str | None = None,
) -> dict[str, object]:
    path = Path(snapshot_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Snapshot file not found: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Snapshot file must contain a JSON object.")

    image_data_url = snapshot_image_data_url
    if image_data_url is None:
        sibling_image = _find_snapshot_image_path(path)
        if sibling_image is None:
            raise FileNotFoundError(
                "No sibling snapshot image found. Provide snapshot_image_data_url "
                "or save a PNG/JPG beside the snapshot JSON."
            )
        image_data_url = _encode_file_as_data_url(sibling_image)

    return render_snapshot_image(
        payload,
        snapshot_image_data_url=image_data_url,
        user_prompt=user_prompt,
        layout_reference_image_data_url=layout_reference_image_data_url,
        annotated_reference_image_data_url=annotated_reference_image_data_url,
        scene_reference_image_data_url=scene_reference_image_data_url,
        scene_reference_mode=scene_reference_mode,
        render_mode=render_mode,
        preset_selection=preset_selection,
        edit_operations=edit_operations,
        edit_source_image_data_url=edit_source_image_data_url,
        config=config,
        trace_id=trace_id,
    )


def _data_url_to_bytes(data_url: str) -> bytes:
    _, _, image_base64 = data_url.partition(",")
    return base64.b64decode(image_base64)


def _normalize_openai_image_size(value: str | None) -> str:
    normalized = _string(value, fallback=_DEFAULT_OPENAI_IMAGE_SIZE).lower()
    if normalized in _SUPPORTED_OPENAI_IMAGE_SIZES:
        return normalized
    return _DEFAULT_OPENAI_IMAGE_SIZE


def _normalize_openai_image_quality(value: str | None) -> str:
    normalized = _string(value, fallback=_DEFAULT_OPENAI_IMAGE_QUALITY).lower()
    if normalized in _SUPPORTED_OPENAI_IMAGE_QUALITIES:
        return normalized
    return _DEFAULT_OPENAI_IMAGE_QUALITY


def _data_url_to_openai_image_file(
    data_url: str,
    *,
    filename_stem: str,
    force_png: bool = False,
) -> tuple[str, bytes, str]:
    if force_png:
        return (
            f"{filename_stem}.png",
            _data_url_to_png_bytes(data_url),
            "image/png",
        )
    mime_type = _extract_data_url_mime_type(data_url)
    extension = _image_extension_for_mime_type(mime_type)
    return (
        f"{filename_stem}.{extension}",
        _data_url_to_bytes(data_url),
        mime_type,
    )


def _data_url_to_png_bytes(data_url: str) -> bytes:
    image = Image.open(BytesIO(_data_url_to_bytes(data_url))).convert("RGBA")
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _build_openai_edit_mask_file(
    source_image_data_url: str,
    *,
    edit_operations: list[SnapshotEditOperation],
) -> tuple[str, bytes, str] | None:
    region_operations = [
        operation for operation in edit_operations if operation.bbox_norm is not None
    ]
    if not region_operations:
        return None

    source_image = Image.open(BytesIO(_data_url_to_bytes(source_image_data_url)))
    width, height = source_image.size
    if width <= 0 or height <= 0:
        return None

    mask = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    has_edit_region = False
    for operation in region_operations:
        bbox = operation.bbox_norm
        if bbox is None:
            continue
        left = int(max(0.0, min(1.0, float(bbox.get("x", 0.0)))) * width)
        top = int(max(0.0, min(1.0, float(bbox.get("y", 0.0)))) * height)
        right = int(
            max(
                0.0, min(1.0, float(bbox.get("x", 0.0)) + float(bbox.get("width", 0.0)))
            )
            * width
        )
        bottom = int(
            max(
                0.0,
                min(1.0, float(bbox.get("y", 0.0)) + float(bbox.get("height", 0.0))),
            )
            * height
        )
        if right <= left or bottom <= top:
            continue
        mask.paste((0, 0, 0, 0), box=(left, top, right, bottom))
        has_edit_region = True

    if not has_edit_region:
        return None

    output = BytesIO()
    mask.save(output, format="PNG")
    return ("mask.png", output.getvalue(), "image/png")


def _image_extension_for_mime_type(mime_type: str) -> str:
    normalized = mime_type.lower()
    if normalized == "image/jpeg":
        return "jpg"
    if normalized == "image/webp":
        return "webp"
    if normalized == "image/gif":
        return "gif"
    return "png"


def _generate_image_with_openai(
    *,
    trace_id: str,
    model_name: str,
    aspect_ratio: str,
    system_prompt: str,
    user_prompt: str,
    source_image_data_url: str,
    annotated_reference_image_data_url: str | None,
    scene_reference_image_data_url: str | None,
    layout_reference_image_data_url: str | None,
    reference_image_data_urls: list[str],
    edit_operations: list[SnapshotEditOperation],
    openai_api_key: str | None,
    openai_image_size: str,
    openai_image_quality: str,
    openai_base_url: str,
) -> dict[str, object]:
    if not openai_api_key:
        raise ValueError(
            "Missing OpenAI image API key. Set OPENAI_IMAGE_API_KEY or configure "
            "services.openai in app-config.yaml before using OpenAI image rendering."
        )

    size = _normalize_openai_image_size(openai_image_size)
    quality = _normalize_openai_image_quality(openai_image_quality)
    combined_prompt = f"{system_prompt}\n\n{user_prompt}"
    mask_file = _build_openai_edit_mask_file(
        source_image_data_url,
        edit_operations=edit_operations,
    )

    image_files: list[tuple[str, tuple[str, bytes, str]]] = []
    image_files.append(
        (
            "image[]",
            _data_url_to_openai_image_file(
                source_image_data_url,
                filename_stem="source",
                force_png=mask_file is not None,
            ),
        )
    )
    if mask_file is not None:
        image_files.append(("mask", mask_file))

    if layout_reference_image_data_url is not None:
        image_files.append(
            (
                "image[]",
                _data_url_to_openai_image_file(
                    layout_reference_image_data_url,
                    filename_stem="layout_reference",
                ),
            )
        )

    if annotated_reference_image_data_url is not None:
        image_files.append(
            (
                "image[]",
                _data_url_to_openai_image_file(
                    annotated_reference_image_data_url,
                    filename_stem="edit_guide",
                ),
            )
        )

    if scene_reference_image_data_url is not None:
        image_files.append(
            (
                "image[]",
                _data_url_to_openai_image_file(
                    scene_reference_image_data_url,
                    filename_stem="scene_reference",
                ),
            )
        )

    for i, ref_url in enumerate(reference_image_data_urls):
        image_files.append(
            (
                "image[]",
                _data_url_to_openai_image_file(
                    ref_url,
                    filename_stem=f"reference_{i + 1}",
                ),
            )
        )

    endpoint = f"{openai_base_url.rstrip('/')}/images/edits"
    headers = {"Authorization": f"Bearer {openai_api_key}"}
    form_data = {
        "model": model_name,
        "prompt": combined_prompt,
        "n": "1",
        "size": size,
        "quality": quality,
    }

    log_image_flow_event(
        "openai.images_edit.attempt_input",
        {
            "trace_id": trace_id,
            "model_name": model_name,
            "endpoint": endpoint,
            "size": size,
            "quality": quality,
            "aspect_ratio": aspect_ratio,
            "image_count": sum(
                1 for field_name, _ in image_files if field_name == "image[]"
            ),
            "mask_used": mask_file is not None,
        },
    )

    started_at = perf_counter()
    with httpx.Client(timeout=600.0) as client:
        response = client.post(
            endpoint,
            headers=headers,
            data=form_data,
            files=image_files,
        )
    elapsed_ms = round((perf_counter() - started_at) * 1000, 2)

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        body = response.text.strip()
        log_image_flow_event(
            "openai.images_edit.http_error",
            {
                "trace_id": trace_id,
                "model_name": model_name,
                "status_code": response.status_code,
                "elapsed_ms": elapsed_ms,
                "response_body": body[:2000],
            },
        )
        detail = f"{exc}. OpenAI response: {body}" if body else str(exc)
        raise ValueError(detail) from exc

    parsed = response.json()
    data_items = parsed.get("data") or []
    if not data_items:
        raise ValueError(
            f"OpenAI image response contained no image data. Response: {str(parsed)[:400]}"
        )

    b64_data = data_items[0].get("b64_json") or ""
    if not b64_data:
        raise ValueError(
            f"OpenAI image response missing b64_json. Response: {str(parsed)[:400]}"
        )

    image_bytes = base64.b64decode(b64_data)
    mime_type = _guess_image_mime_type(image_bytes)

    log_image_flow_event(
        "openai.images_edit.image_output",
        {
            "trace_id": trace_id,
            "model_name": model_name,
            "elapsed_ms": elapsed_ms,
            "mime_type": mime_type,
            "byte_length": len(image_bytes),
            "size": size,
            "quality": quality,
        },
    )

    return {
        "mime_type": mime_type,
        "image_base64": b64_data,
        "data_url": f"data:{mime_type};base64,{b64_data}",
        "size": size,
        "quality": quality,
        "aspect_ratio": aspect_ratio,
        "request_strategy": "openai_edit",
        "generation_config_applied": True,
        "raw_response": {k: v for k, v in parsed.items() if k != "data"},
    }


def _generate_image_with_gemini(
    *,
    trace_id: str,
    model_name: str,
    image_size: str,
    max_output_tokens: int,
    gemini_base_url: str,
    gemini_api_key: str | None,
    aspect_ratio: str,
    system_prompt: str,
    user_prompt: str,
    source_image_data_url: str,
    annotated_reference_image_data_url: str | None,
    scene_reference_image_data_url: str | None,
    layout_reference_image_data_url: str | None,
    reference_image_data_urls: list[str],
) -> dict[str, object]:
    if not gemini_api_key:
        raise ValueError(
            "Missing Gemini image API key. Set services.gemini_image.api_key in "
            "app-config.yaml or set GEMINI_IMAGE_API_KEY before using AI image rendering."
        )

    endpoint = f"{gemini_base_url.rstrip('/')}/models/{model_name}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": gemini_api_key,
    }
    attempts = _build_gemini_request_attempts(
        model_name=model_name,
        image_size=image_size,
        max_output_tokens=max_output_tokens,
        aspect_ratio=aspect_ratio,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        source_image_data_url=source_image_data_url,
        annotated_reference_image_data_url=annotated_reference_image_data_url,
        scene_reference_image_data_url=scene_reference_image_data_url,
        layout_reference_image_data_url=layout_reference_image_data_url,
        reference_image_data_urls=reference_image_data_urls,
    )
    last_response: Mapping[str, object] | None = None
    with httpx.Client(timeout=600.0) as client:
        for attempt in attempts:
            try:
                parsed = _post_gemini_generate_content(
                    client=client,
                    trace_id=trace_id,
                    model_name=model_name,
                    attempt_name=attempt.name,
                    endpoint=endpoint,
                    headers=headers,
                    payload=attempt.payload,
                )
            except _GeminiGenerateContentHttpError as exc:
                if _should_retry_gemini_http_error_with_minimal_payload(
                    model_name=model_name,
                    attempt=attempt,
                    error=exc,
                ):
                    logger.warning(
                        "Gemini image request for %s failed with HTTP %s on "
                        "%s. Retrying with compatibility payload.",
                        model_name,
                        exc.status_code,
                        attempt.name,
                    )
                    continue
                raise
            last_response = parsed
            image_bytes = _extract_image_bytes(parsed)
            if image_bytes is None:
                log_image_flow_event(
                    "gemini.generate_content.no_image",
                    {
                        "trace_id": trace_id,
                        "model_name": model_name,
                        "attempt": attempt.name,
                        "finish_reason": _extract_gemini_finish_reason(parsed),
                        "finish_message": _extract_gemini_finish_message(parsed),
                        "response_text": _extract_gemini_response_text(parsed),
                        "response": summarize_gemini_response(parsed),
                    },
                )
                if _should_retry_gemini_with_minimal_payload(
                    model_name=model_name,
                    attempt=attempt,
                    response_payload=parsed,
                ):
                    logger.warning(
                        "Gemini image response for %s returned no image with "
                        "finishReason=%s. Retrying with compatibility payload.",
                        model_name,
                        _extract_gemini_finish_reason(parsed) or "unknown",
                    )
                    continue
                raise ValueError(_build_gemini_missing_image_error(parsed))
            mime_type = _guess_image_mime_type(image_bytes)
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            log_image_flow_event(
                "gemini.generate_content.image_output",
                {
                    "trace_id": trace_id,
                    "model_name": model_name,
                    "attempt": attempt.name,
                    "mime_type": mime_type,
                    "byte_length": len(image_bytes),
                    "generation_config_applied": attempt.generation_config_applied,
                    "response": summarize_gemini_response(parsed),
                },
            )
            return {
                "mime_type": mime_type,
                "image_base64": image_base64,
                "data_url": f"data:{mime_type};base64,{image_base64}",
                "size": image_size,
                "aspect_ratio": aspect_ratio,
                "request_strategy": attempt.name,
                "generation_config_applied": attempt.generation_config_applied,
                "raw_response": parsed,
            }

    raise ValueError(
        _build_gemini_missing_image_error(last_response or {"error": "No response"})
    )


def _build_gemini_request_attempts(
    *,
    model_name: str,
    image_size: str,
    max_output_tokens: int,
    aspect_ratio: str,
    system_prompt: str,
    user_prompt: str,
    source_image_data_url: str,
    annotated_reference_image_data_url: str | None,
    scene_reference_image_data_url: str | None,
    layout_reference_image_data_url: str | None,
    reference_image_data_urls: list[str],
) -> list[_GeminiRequestAttempt]:
    resolved_user_prompt = user_prompt
    include_system_instruction = not _model_inlines_system_prompt(model_name)
    if not include_system_instruction:
        resolved_user_prompt = system_prompt + "\n\n" + user_prompt

    normalized_source_image_data_url = _normalize_gemini_input_image_data_url(
        source_image_data_url,
        model_name=model_name,
        image_size=image_size,
        aspect_ratio=aspect_ratio,
    )
    normalized_scene_reference_image_data_url = (
        _normalize_gemini_input_image_data_url(
            scene_reference_image_data_url,
            model_name=model_name,
            image_size=image_size,
            aspect_ratio=aspect_ratio,
        )
        if scene_reference_image_data_url is not None
        else None
    )
    normalized_annotated_reference_image_data_url = (
        _normalize_gemini_input_image_data_url(
            annotated_reference_image_data_url,
            model_name=model_name,
            image_size=image_size,
            aspect_ratio=aspect_ratio,
        )
        if annotated_reference_image_data_url is not None
        else None
    )
    normalized_layout_reference_image_data_url = (
        _normalize_gemini_input_image_data_url(
            layout_reference_image_data_url,
            model_name=model_name,
            image_size=image_size,
            aspect_ratio=aspect_ratio,
        )
        if layout_reference_image_data_url is not None
        else None
    )
    normalized_reference_image_data_urls = [
        _normalize_gemini_input_image_data_url(
            data_url,
            model_name=model_name,
            image_size=image_size,
            aspect_ratio=aspect_ratio,
        )
        for data_url in reference_image_data_urls
    ]

    content: list[dict[str, object]] = [
        {"text": resolved_user_prompt},
        _data_url_to_inline_part(normalized_source_image_data_url),
    ]
    has_supplemental_images = any(
        data_url is not None
        for data_url in (
            normalized_scene_reference_image_data_url,
            normalized_annotated_reference_image_data_url,
            normalized_layout_reference_image_data_url,
        )
    ) or bool(normalized_reference_image_data_urls)
    if normalized_scene_reference_image_data_url is not None:
        content.append(
            _data_url_to_inline_part(normalized_scene_reference_image_data_url)
        )
    if normalized_layout_reference_image_data_url is not None:
        content.append(
            _data_url_to_inline_part(normalized_layout_reference_image_data_url)
        )
    if normalized_annotated_reference_image_data_url is not None:
        content.append(
            _data_url_to_inline_part(normalized_annotated_reference_image_data_url)
        )
    content.extend(
        _data_url_to_inline_part(data_url)
        for data_url in normalized_reference_image_data_urls
    )

    base_payload: dict[str, object] = {
        "contents": [
            {
                "role": "user",
                "parts": content,
            }
        ],
    }
    if include_system_instruction:
        base_payload["systemInstruction"] = {
            "parts": [{"text": system_prompt}],
        }

    configured_payload = {
        **base_payload,
        "generationConfig": _build_gemini_generation_config(
            model_name=model_name,
            image_size=image_size,
            max_output_tokens=max_output_tokens,
            aspect_ratio=aspect_ratio,
        ),
    }
    attempts = [
        _GeminiRequestAttempt(
            name="configured",
            payload=configured_payload,
            generation_config_applied=True,
            retry_on_http_400=True,
        )
    ]
    if _model_uses_compatibility_image_generation_config(model_name):
        attempts.append(
            _GeminiRequestAttempt(
                name="compatibility_no_generation_config",
                payload=base_payload,
                generation_config_applied=False,
                retry_on_http_400=has_supplemental_images,
            )
        )
    elif _model_uses_nano_banana_2_generation_config(model_name):
        attempts.append(
            _GeminiRequestAttempt(
                name="compatibility_no_generation_config",
                payload=base_payload,
                generation_config_applied=False,
                retry_on_http_400=has_supplemental_images,
            )
        )
    if has_supplemental_images and _model_can_retry_with_minimal_image_payload(
        model_name
    ):
        source_only_payload: dict[str, object] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": _remove_supplemental_image_prompt_references(
                                resolved_user_prompt
                            )
                        },
                        _data_url_to_inline_part(normalized_source_image_data_url),
                    ],
                }
            ],
        }
        if include_system_instruction:
            source_only_payload["systemInstruction"] = {
                "parts": [{"text": system_prompt}],
            }
        attempts.append(
            _GeminiRequestAttempt(
                name="compatibility_source_only_no_generation_config",
                payload=source_only_payload,
                generation_config_applied=False,
            )
        )
    return attempts


def _build_gemini_generation_config(
    *,
    model_name: str,
    image_size: str,
    max_output_tokens: int,
    aspect_ratio: str,
) -> dict[str, object]:
    image_config = {
        "aspectRatio": aspect_ratio,
        "imageSize": image_size,
    }
    if _model_uses_nano_banana_2_generation_config(model_name):
        return {
            "responseModalities": ["TEXT", "IMAGE"],
            "imageConfig": image_config,
        }
    if _model_uses_compatibility_image_generation_config(model_name):
        return {
            "responseModalities": ["IMAGE"],
            "imageConfig": image_config,
        }
    return {
        "responseModalities": ["TEXT", "IMAGE"],
        "imageConfig": image_config,
        "maxOutputTokens": max_output_tokens,
    }


def _remove_supplemental_image_prompt_references(prompt: str) -> str:
    prompt_parts: list[str] = []
    for paragraph in prompt.split("\n\n"):
        if paragraph.startswith("Object appearance memory:"):
            continue
        cleaned_lines = [
            line
            for line in paragraph.splitlines()
            if not _line_references_supplemental_image(line)
        ]
        cleaned_paragraph = "\n".join(cleaned_lines).strip()
        if cleaned_paragraph:
            prompt_parts.append(cleaned_paragraph)
    return "\n\n".join(prompt_parts)


def _line_references_supplemental_image(line: str) -> bool:
    if line.startswith("Image ") and (
        "same-camera labeled structural guide" in line
        or "previously rendered same-room reference" in line
        or "labeled 3D target-view reference" in line
        or "user-drawn edit-region guide" in line
    ):
        return True
    return line.startswith("The top-down layout image is supplemental")


def _model_inlines_system_prompt(model_name: str) -> bool:
    normalized = _normalize_gemini_model_name(model_name)
    return normalized == "gemini-3.1-flash-image-preview"


def _model_uses_nano_banana_2_generation_config(model_name: str) -> bool:
    normalized = _normalize_gemini_model_name(model_name)
    return normalized == "gemini-3.1-flash-image-preview"


def _model_uses_compatibility_image_generation_config(model_name: str) -> bool:
    normalized = _normalize_gemini_model_name(model_name)
    if "-image" not in normalized:
        return False
    if _model_uses_nano_banana_2_generation_config(normalized):
        return False
    return normalized.startswith("gemini-3")


def _post_gemini_generate_content(
    *,
    client: httpx.Client,
    trace_id: str,
    model_name: str,
    attempt_name: str,
    endpoint: str,
    headers: Mapping[str, str],
    payload: Mapping[str, object],
) -> dict[str, object]:
    payload_size_bytes = _payload_json_size_bytes(payload)
    inline_image_summaries = _collect_inline_image_summaries(payload)
    log_image_flow_event(
        "gemini.generate_content.attempt_input",
        {
            "trace_id": trace_id,
            "model_name": model_name,
            "attempt": attempt_name,
            "endpoint": endpoint,
            "payload_size_bytes": payload_size_bytes,
            "inline_image_count": len(inline_image_summaries),
            "inline_images": inline_image_summaries,
            "payload": summarize_gemini_payload(dict(payload)),
        },
    )
    if payload_size_bytes >= _GEMINI_INLINE_REQUEST_LIMIT_BYTES:
        log_image_flow_event(
            "gemini.generate_content.payload_too_large",
            {
                "trace_id": trace_id,
                "model_name": model_name,
                "attempt": attempt_name,
                "payload_size_bytes": payload_size_bytes,
                "limit_bytes": _GEMINI_INLINE_REQUEST_LIMIT_BYTES,
                "inline_image_count": len(inline_image_summaries),
                "inline_images": inline_image_summaries,
            },
        )
        payload_size_mb = payload_size_bytes / _BYTES_PER_MB
        limit_mb = _GEMINI_INLINE_REQUEST_LIMIT_BYTES / _BYTES_PER_MB
        raise ValueError(
            "Gemini inline image request is "
            f"{payload_size_mb:.2f} MB, exceeding the {limit_mb:.0f} MB inline "
            "request limit. Reduce canvas/reference image size or switch this "
            "flow to the Gemini Files API for large images."
        )
    started_at = perf_counter()
    response = client.post(endpoint, json=payload, headers=dict(headers))
    elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        body = response.text.strip()
        response_body = _safe_response_body(response)
        log_image_flow_event(
            "gemini.generate_content.http_error",
            {
                "trace_id": trace_id,
                "model_name": model_name,
                "attempt": attempt_name,
                "status_code": response.status_code,
                "elapsed_ms": elapsed_ms,
                "response_body": response_body,
            },
        )
        detail = f"{exc}. Gemini response body: {body}" if body else str(exc)
        raise _GeminiGenerateContentHttpError(
            message=detail,
            status_code=response.status_code,
            response_body=response_body,
        ) from exc

    parsed = response.json()
    if not isinstance(parsed, dict):
        raise ValueError("Gemini image response must be a JSON object.")
    log_image_flow_event(
        "gemini.generate_content.http_output",
        {
            "trace_id": trace_id,
            "model_name": model_name,
            "attempt": attempt_name,
            "status_code": response.status_code,
            "elapsed_ms": elapsed_ms,
            "response": summarize_gemini_response(parsed),
        },
    )
    return parsed


def _payload_json_size_bytes(payload: Mapping[str, object]) -> int:
    encoded = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    return len(encoded)


def _collect_inline_image_summaries(value: object) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    if isinstance(value, Mapping):
        for inline_key in ("inlineData", "inline_data"):
            inline_payload = value.get(inline_key)
            if not isinstance(inline_payload, Mapping):
                continue
            data = inline_payload.get("data")
            mime_type = inline_payload.get("mimeType") or inline_payload.get(
                "mime_type"
            )
            if isinstance(data, str):
                resolved_mime_type = (
                    mime_type
                    if isinstance(mime_type, str)
                    else "application/octet-stream"
                )
                summaries.append(
                    {
                        "mime_type": resolved_mime_type,
                        "data": summarize_image_data_url(
                            f"data:{resolved_mime_type};base64,{data}"
                        ),
                    }
                )
        for item in value.values():
            summaries.extend(_collect_inline_image_summaries(item))
    elif isinstance(value, list):
        for item in value:
            summaries.extend(_collect_inline_image_summaries(item))
    return summaries


def _safe_response_body(response: httpx.Response) -> object:
    try:
        return summarize_gemini_response(response.json())
    except ValueError:
        text = response.text.strip()
        return redact_for_image_log(text)


def _should_retry_gemini_with_minimal_payload(
    *,
    model_name: str,
    attempt: _GeminiRequestAttempt,
    response_payload: Mapping[str, object],
) -> bool:
    if not _model_can_retry_with_minimal_image_payload(model_name):
        return False
    if attempt.name != "configured":
        return False
    _ = response_payload
    return True


def _should_retry_gemini_http_error_with_minimal_payload(
    *,
    model_name: str,
    attempt: _GeminiRequestAttempt,
    error: _GeminiGenerateContentHttpError,
) -> bool:
    if not _model_can_retry_with_minimal_image_payload(model_name):
        return False
    if not attempt.retry_on_http_400:
        return False
    if error.status_code != 400:
        return False
    return True


def _model_can_retry_with_minimal_image_payload(model_name: str) -> bool:
    return _model_uses_compatibility_image_generation_config(
        model_name
    ) or _model_uses_nano_banana_2_generation_config(model_name)


def _build_gemini_missing_image_error(payload: Mapping[str, object]) -> str:
    detail_parts: list[str] = []
    finish_reason = _extract_gemini_finish_reason(payload)
    if finish_reason:
        detail_parts.append(f"finishReason={finish_reason}")
    finish_message = _extract_gemini_finish_message(payload)
    if finish_message:
        detail_parts.append(f"finishMessage={_truncate_text(finish_message)}")
    response_text = _extract_gemini_response_text(payload)
    if response_text:
        detail_parts.append(f"text={_truncate_text(response_text)}")
    detail = f" Details: {'; '.join(detail_parts)}" if detail_parts else ""
    return "Gemini image response did not contain image data." + detail


def _extract_gemini_finish_reason(payload: Mapping[str, object]) -> str | None:
    candidate = _extract_first_gemini_candidate(payload)
    if candidate is None:
        return None
    value = candidate.get("finishReason")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _extract_gemini_finish_message(payload: Mapping[str, object]) -> str | None:
    candidate = _extract_first_gemini_candidate(payload)
    if candidate is None:
        return None
    value = candidate.get("finishMessage")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _extract_gemini_response_text(payload: Mapping[str, object]) -> str:
    candidate = _extract_first_gemini_candidate(payload)
    if candidate is None:
        return ""
    content = candidate.get("content")
    if not isinstance(content, Mapping):
        return ""
    parts = content.get("parts")
    if not isinstance(parts, list):
        return ""

    texts: list[str] = []
    for part in parts:
        if not isinstance(part, Mapping):
            continue
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
    return "\n".join(texts)


def _extract_first_gemini_candidate(
    payload: Mapping[str, object],
) -> Mapping[str, object] | None:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    first = candidates[0]
    if isinstance(first, Mapping):
        return first
    return None


def _truncate_text(value: str, *, limit: int = 240) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def _build_preset_prompt(preset_selection: Mapping[str, object] | None) -> str:
    if preset_selection is None:
        return ""

    preset_config = load_config(config_file).presets
    requested_presets = (
        ("Style", preset_config.styles, _string(preset_selection.get("style"))),
        ("Lighting", preset_config.lights, _string(preset_selection.get("lighting"))),
        (
            "Scenery",
            preset_config.sceneries,
            _string(preset_selection.get("scenery")),
        ),
    )
    prompt_parts: list[str] = []
    for label, options, key in requested_presets:
        if not key:
            continue
        option = options.get(key)
        if option is None:
            raise ValueError(f"Unknown {label.lower()} preset: {key}")
        suffix = _string(option.prompt_suffix)
        if not suffix:
            continue
        prompt_parts.append(f"{label} preset ({option.label}): {suffix}")
    return "\n".join(prompt_parts)


def _build_final_user_prompt(
    *,
    user_prompt: str | None,
    render_mode: RenderMode,
    preset_prompt: str,
    edit_operations: list[SnapshotEditOperation],
    visible_objects: Mapping[str, Mapping[str, object]],
    layout_lock_prompt: str = "",
    layout_reference_used: bool = False,
    annotated_reference_used: bool = False,
    scene_reference_mode: SceneReferenceMode = "none",
) -> str:
    raw_user_prompt = _string(user_prompt)
    if render_mode == "generate":
        prompt_parts: list[str] = []
        image_role_prompt = _build_generate_image_role_prompt(
            scene_reference_mode=scene_reference_mode,
            layout_reference_used=layout_reference_used,
            annotated_reference_used=annotated_reference_used,
        )
        if image_role_prompt:
            prompt_parts.append(image_role_prompt)
        if layout_lock_prompt:
            prompt_parts.append(layout_lock_prompt)
        prompt_parts.append(
            "Material, lighting, and scenery goal, secondary to the exact input view and locked palette: "
            + (raw_user_prompt or _DEFAULT_USER_PROMPT)
        )
        appearance_memory_prompt = _build_object_appearance_memory_prompt(
            edit_operations
        )
        if appearance_memory_prompt:
            prompt_parts.append(appearance_memory_prompt)
        if preset_prompt:
            prompt_parts.append(
                "Apply these selected presets as finish realism, lighting, and exterior scenery only. They must not change architecture, openings, built-ins, object colors, object shape, object count, object size, or object placement:\n"
                + preset_prompt
            )
        return "\n\n".join(prompt_parts)

    if (
        not raw_user_prompt
        and not preset_prompt
        and not edit_operations
        and not layout_reference_used
    ):
        raise ValueError(
            "Provide at least one edit operation, preset change, prompt change, or target-view reference."
        )

    prompt_parts: list[str] = []
    prompt_parts.append("Edit the first image.")
    if layout_reference_used:
        prompt_parts.append(
            "Image 1 is the current photorealistic render. Image 2 is the current 3D layout snapshot reference. If no explicit object operation is provided, perform only the camera/layout transfer from Image 2 while preserving object identity and realistic finishes from Image 1."
        )
        prompt_parts.append(
            "Use Image 2 as the labeled 3D target-view reference. Match that reference's camera angle, framing, room layout, openings, main object placement, main object size, and occlusion. Do not render any label, guide, box, outline, number, or UI mark from the reference. Small decor and material styling may change, but the target structure and main layout must stay locked."
        )
    else:
        prompt_parts.append(
            "Preserve the current camera, room layout, architecture, and all unselected objects."
        )
    if annotated_reference_used:
        bbox_image_number = 3 if layout_reference_used else 2
        prompt_parts.append(
            f"Image {bbox_image_number} is the edit-region guide. "
            "It shows colored bounding boxes drawn over the photo. Each box marks a region to edit; "
            "the small label inside (e.g. R1, R2, R3) is the region ID that matches the instructions below. "
            "Use the boxes and IDs only as your instruction map — they are never part of the scene. "
            "The final output must be a clean photorealistic photo with NONE of the following: "
            "no colored box outlines, no bounding rectangles, no region IDs, no label text, "
            "no tinted overlays, no guide graphics, no annotations of any kind."
        )
    if visible_objects:
        visible_summary = [
            f"{object_id} ({_visible_object_name(payload)})"
            for object_id, payload in visible_objects.items()
        ]
        visible_context = "target reference" if layout_reference_used else "the image"
        prompt_parts.append(
            f"Objects visible in {visible_context}: " + ", ".join(visible_summary)
        )
    target_lock_prompt = _build_edit_target_geometry_prompt(
        edit_operations=edit_operations,
        visible_objects=visible_objects,
    )
    if target_lock_prompt:
        prompt_parts.append(target_lock_prompt)
    if raw_user_prompt:
        prompt_parts.append("Requested edit: " + raw_user_prompt)
    if preset_prompt:
        prompt_parts.append("Preset update:\n" + preset_prompt)

    replacement_index = 1
    for operation in edit_operations:
        if operation.prompt is not None:
            target_text = (
                f"drawn region {operation.object_id}"
                if operation.bbox_norm is not None
                else f"selected object {operation.object_id}"
            )
            prompt_parts.append(
                f"Instruction for {target_text}: {operation.prompt}"
                " Apply this instruction only to that target."
            )
        if operation.replacement_image_data_url is not None:
            if operation.bbox_norm is not None and operation.operation_type == "add":
                prompt_parts.append(
                    f"Drawn region id: {operation.object_id}. Reference image {replacement_index} belongs to this region.\n"
                    f"Add the referenced {operation.object_name or 'object'} inside the matching boxed region from the edit-region guide. "
                    "Fit it to that region's perspective, scale, and occlusion. "
                    "Do not place the new object outside that region. Do not move architecture or existing main furniture outside the boxed region."
                )
            elif operation.bbox_norm is not None:
                prompt_parts.append(
                    f"Drawn region id: {operation.object_id}. Reference image {replacement_index} belongs to this region.\n"
                    f"Use reference image {replacement_index} only for the replacement appearance inside the matching boxed region from the edit-region guide. "
                    "Replace the object/content in that boxed region while preserving the region's placement, scale, perspective, and occlusion. Remove the old object/content in the box instead of duplicating it. "
                    "Do not change architecture or main furniture outside the boxed region."
                )
                replacement_index += 1
                continue
            else:
                prompt_parts.append(
                    "Selected object id: "
                    f"{operation.object_id}. Reference image {replacement_index} belongs to this operation.\n"
                    "Selected object geometry locks from the source image and target layout override reference-image scale, crop, camera, and composition. "
                    "Use the uploaded image as an appearance-only replacement reference, not a scale, camera, crop, layout, or composition reference. "
                    f"Use reference image {replacement_index} only for the selected {operation.object_name}'s appearance. "
                    "Keep the target-view placement, size, angle, depth order, and occlusion from the input images. "
                    "Ignore the reference image background, room, crop, and camera. "
                    "Do not cover, remove, redraw, or relocate any unselected object."
                )
            replacement_index += 1
        if operation.target_color is not None:
            if operation.bbox_norm is not None:
                prompt_parts.append(
                    f"Drawn region id: {operation.object_id}.\n"
                    f"Change only the color/material of the object or surface inside the matching boxed region to {operation.target_color}. "
                    "Keep structure, placement, shape, and everything outside the boxed region unchanged."
                )
            else:
                prompt_parts.append(
                    f"Selected object id: {operation.object_id}.\n"
                    f"Change only the color of the selected {operation.object_name} to {operation.target_color}. "
                    "Keep its shape, material, placement, and all other objects unchanged."
                )

    return "\n\n".join(prompt_parts)


def _build_edit_target_geometry_prompt(
    *,
    edit_operations: list[SnapshotEditOperation],
    visible_objects: Mapping[str, Mapping[str, object]],
) -> str:
    prompt_parts: list[str] = []
    for operation in edit_operations:
        operation_label = _operation_type_label(operation.operation_type)
        if operation.bbox_norm is not None:
            object_name = operation.object_name or operation.object_id
            prompt_parts.append(
                f"{operation.object_id} ({operation_label}, {object_name})"
            )
            continue
        visible_payload = visible_objects.get(operation.object_id)
        object_name = operation.object_name or (
            _visible_object_name(visible_payload)
            if visible_payload is not None
            else "object"
        )
        screen_bbox = (
            _mapping(visible_payload.get("screenBboxPx"))
            if visible_payload is not None
            else {}
        )
        if screen_bbox:
            bbox_json = json.dumps(
                {"screenBboxPx": dict(screen_bbox)},
                ensure_ascii=True,
                separators=(",", ":"),
            )
            prompt_parts.append(
                f"{operation.object_id} ({object_name}) target geometry {bbox_json}"
            )
        else:
            prompt_parts.append(f"{operation.object_id} ({object_name})")

    if not prompt_parts:
        return ""
    return (
        "Selected object ids or drawn edit regions. Only these targets may change:\n"
        + "\n".join(prompt_parts)
    )


def _operation_type_label(
    operation_type: Literal["prompt", "add", "replace", "recolor"] | None,
) -> str:
    if operation_type == "add":
        return "add object"
    if operation_type == "replace":
        return "replace object"
    if operation_type == "recolor":
        return "change color"
    return "prompt edit"


def _build_object_appearance_memory_prompt(
    edit_operations: list[SnapshotEditOperation],
) -> str:
    paired_operations = [
        operation
        for operation in edit_operations
        if operation.replacement_image_data_url is not None
    ]
    if not paired_operations:
        return ""

    prompt_parts = [
        "Object appearance memory:",
        (
            "After the scene/layout images, each reference image is paired with "
            "one object id below. Use these references only for that object's "
            "style, material, silhouette cues, and recognizable appearance. The "
            "labeled 3D layout image still controls camera, placement, size, "
            "visibility, and occlusion."
        ),
    ]
    for index, operation in enumerate(paired_operations, start=1):
        object_name = operation.object_name or "object"
        prompt_parts.append(
            f"Reference image {index}: object id {operation.object_id} ({object_name})."
        )
    return "\n".join(prompt_parts)


def _build_generate_image_role_prompt(
    *,
    scene_reference_mode: SceneReferenceMode,
    layout_reference_used: bool,
    annotated_reference_used: bool,
) -> str:
    prompt_parts: list[str] = []
    next_image_number = 2
    if scene_reference_mode == "scene_reference_camera_transfer":
        prompt_parts.append(
            "Image 1 is a previously rendered source reference. Transfer it to the newly selected target camera only where it does not conflict with the labeled 3D layout guide."
        )
    elif scene_reference_mode == "target_layout_with_scene_reference":
        prompt_parts.append(
            f"Image {next_image_number} is a previously rendered same-room reference. Use it only for realistic finish continuity and object identity when it agrees with the 3D layout."
        )
        next_image_number += 1

    if layout_reference_used:
        prompt_parts.append(
            f"Image {next_image_number} is the top-down layout reference. Use it only to clarify the room footprint and openings."
        )
        next_image_number += 1
    if annotated_reference_used:
        prompt_parts.append(
            f"Image {next_image_number} is a same-camera labeled structural guide. Do not render the guide lines, labels, outlines, or UI marks in the final image."
        )

    return "\n".join(prompt_parts)


def _resolve_scene_reference_mode(
    *,
    scene_reference_mode: SceneReferenceMode,
    render_mode: RenderMode,
    scene_reference_image_data_url: str | None,
) -> SceneReferenceMode:
    if render_mode != "generate" or scene_reference_image_data_url is None:
        return "none"
    if scene_reference_mode == "target_layout_with_scene_reference":
        return "target_layout_with_scene_reference"
    if scene_reference_mode == "scene_reference_camera_transfer":
        return "scene_reference_camera_transfer"
    return "none"


def _build_generate_layout_lock_prompt(
    *,
    scene_reference_mode: SceneReferenceMode,
    snapshot_payload: Mapping[str, object],
) -> str:
    prompt_parts = [
        "Layout lock instructions:",
        "Use the input 3D layout image as the exact target view.",
        "Match the camera angle, framing, perspective, room boundaries, visible wall panels, crisp corner seams, openings, furniture placement, object count, object color, object shape, object size, depth order, and occlusion shown in that image.",
        "The labels, dark structural edge overlays, and guide marks are only for understanding the layout. They must not appear as labels, black outlines, or guide graphics in the final render.",
        "Existing objects may gain photorealistic texture and finish detail, but not a different color, identity, silhouette, footprint, pose, size, or position.",
        "Do not add extra decor or accessories unless they already exist in the 3D layout or the user explicitly requested them.",
        "Do not add extra walls, remove walls, open closed sides, move wall edges, move doors or windows, or invent columns, stairs, fireplaces, cabinets, wardrobes, beds, desks, tables, sofas, chairs, built-ins, or other furniture.",
    ]
    compiled_layout_prompt = _build_compiled_layout_lock_prompt(snapshot_payload)
    if compiled_layout_prompt:
        prompt_parts.append(compiled_layout_prompt)
    if scene_reference_mode == "target_layout_with_scene_reference":
        prompt_parts.append(
            "Use the previous render only for continuity of realistic finish and object identity. Do not copy its camera, layout, room shape, or color palette if they conflict with the 3D layout."
        )
    elif scene_reference_mode == "scene_reference_camera_transfer":
        camera_payload = _mapping(snapshot_payload.get("camera"))
        if camera_payload:
            prompt_parts.append(
                "Target camera metadata: "
                + json.dumps(
                    dict(camera_payload),
                    ensure_ascii=True,
                    separators=(",", ":"),
                    default=str,
                )
            )

    prompt_parts.append(
        "Do not add, remove, move, resize, rotate, recolor, or replace any visible object unless it has an explicit object reference or color edit. Improve photorealistic appearance without changing the locked structure."
    )
    return "\n".join(prompt_parts)


def _build_compiled_layout_lock_prompt(
    snapshot_payload: Mapping[str, object],
) -> str:
    try:
        compilation = compile_snapshot_prompt(snapshot_payload)
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("Snapshot prompt compilation skipped: %s", exc)
        return ""

    strict_facts = [
        item
        for item in _sequence(compilation.get("strict_layout_facts"))
        if isinstance(item, str) and item.strip()
    ]
    layout_constraints = [
        item
        for item in _sequence(compilation.get("layout_constraints"))
        if isinstance(item, str)
        and item.strip()
        and not item.startswith("Minor decor may be omitted")
    ]
    prompt_parts: list[str] = []
    palette_prompt = _build_exact_palette_lock_prompt(snapshot_payload)
    if palette_prompt:
        prompt_parts.append(palette_prompt)
    if strict_facts:
        prompt_parts.append(
            "Exact layout facts to preserve:\n"
            + "\n".join(f"- {fact}" for fact in strict_facts[:18])
        )
    if layout_constraints:
        prompt_parts.append(
            "Strict visual constraints:\n"
            + "\n".join(f"- {constraint}" for constraint in layout_constraints[:14])
        )
    return "\n".join(prompt_parts)


def _build_exact_palette_lock_prompt(
    snapshot_payload: Mapping[str, object],
) -> str:
    room_payload = _mapping(snapshot_payload.get("room"))
    surface_payload = _mapping(room_payload.get("surfaceColors"))
    surface_lines = []
    for label, key in (
        ("walls", "wallColorHex"),
        ("floor", "floorColorHex"),
        ("ceiling", "ceilingColorHex"),
    ):
        color = _string(surface_payload.get(key))
        if _is_hex_color(color):
            surface_lines.append(f"{label} {color}")

    object_lines = []
    for object_id, payload in _extract_visible_objects(snapshot_payload).items():
        color = _string(payload.get("colorHex"))
        if not _is_hex_color(color):
            continue
        object_lines.append(
            f"{object_id} ({_visible_object_name(payload)}) {color}"
        )

    if not surface_lines and not object_lines:
        return ""

    prompt_parts = [
        "Exact color palette from the current 3D layout. Preserve these colors in the photorealistic render unless the user changed them in the layout or requested an explicit color edit."
    ]
    if surface_lines:
        prompt_parts.append("Room surfaces: " + "; ".join(surface_lines) + ".")
    if object_lines:
        prompt_parts.append(
            "Visible objects: " + "; ".join(object_lines[:18]) + "."
        )
    return "\n".join(prompt_parts)


def _normalize_edit_operations(
    operations: list[SnapshotEditOperation],
    *,
    visible_objects: Mapping[str, Mapping[str, object]],
) -> list[SnapshotEditOperation]:
    normalized: list[SnapshotEditOperation] = []
    for operation in operations:
        object_id = _string(operation.object_id)
        if not object_id:
            raise ValueError("Edit operation object_id is required.")
        bbox_norm = _normalize_bbox_norm(operation.bbox_norm)
        is_region_operation = bbox_norm is not None
        visible_payload = visible_objects.get(object_id)
        if visible_payload is None and not is_region_operation:
            raise ValueError(
                f"Object {object_id} is not visible in the source image and cannot be edited."
            )

        replacement_image_data_url = (
            _normalize_image_data_url(operation.replacement_image_data_url)
            if operation.replacement_image_data_url
            else None
        )
        target_color = _string(operation.target_color) or None
        prompt = _string(operation.prompt) or None
        operation_type = operation.operation_type
        if operation_type not in {"prompt", "add", "replace", "recolor", None}:
            operation_type = None
        if operation_type is None:
            operation_type = (
                "replace"
                if replacement_image_data_url
                else "recolor"
                if target_color
                else "prompt"
            )
        if (
            replacement_image_data_url is None
            and target_color is None
            and prompt is None
            and not is_region_operation
        ):
            continue

        object_name = _string(operation.object_name) or (
            _visible_object_name(visible_payload)
            if visible_payload is not None
            else object_id
        )
        normalized.append(
            SnapshotEditOperation(
                object_id=object_id,
                object_name=object_name,
                operation_type=operation_type,
                prompt=prompt,
                bbox_norm=bbox_norm,
                replacement_image_data_url=replacement_image_data_url,
                target_color=target_color,
            )
        )
    return normalized


def _normalize_bbox_norm(
    bbox_norm: Mapping[str, float] | None,
) -> dict[str, float] | None:
    if not bbox_norm:
        return None
    try:
        x = float(bbox_norm.get("x", 0.0))
        y = float(bbox_norm.get("y", 0.0))
        width = float(bbox_norm.get("width", 0.0))
        height = float(bbox_norm.get("height", 0.0))
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    width = max(0.0, min(1.0 - x, width))
    height = max(0.0, min(1.0 - y, height))
    if width <= 0.002 or height <= 0.002:
        return None
    return {
        "x": round(x, 4),
        "y": round(y, 4),
        "width": round(width, 4),
        "height": round(height, 4),
    }


def _extract_visible_objects(
    snapshot_payload: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    raw_visible = snapshot_payload.get("visibleObjects")
    if not isinstance(raw_visible, list):
        return {}

    visible: dict[str, Mapping[str, object]] = {}
    for item in raw_visible:
        if not isinstance(item, Mapping):
            continue
        object_id = _string(item.get("id"))
        if object_id:
            visible[object_id] = item
    return visible


def _visible_object_name(payload: Mapping[str, object]) -> str:
    for key in (
        "label",
        "assetId",
        "asset_id",
        "rawAssetId",
        "raw_asset_id",
        "type",
        "canonicalType",
        "canonical_type",
        "raw_type",
    ):
        value = _string(payload.get(key))
        if value:
            return value.replace("_", " ")
    object_id = _string(payload.get("id"))
    return object_id or "object"


def _serialize_edit_operation(operation: SnapshotEditOperation) -> dict[str, object]:
    return {
        "object_id": operation.object_id,
        "object_name": operation.object_name,
        "operation_type": operation.operation_type,
        "prompt": operation.prompt,
        "bbox_provided": operation.bbox_norm is not None,
        "replacement_image_provided": operation.replacement_image_data_url is not None,
        "target_color": operation.target_color,
    }


def _normalize_gemini_input_image_data_url(
    data_url: str,
    *,
    model_name: str,
    image_size: str,
    aspect_ratio: str,
) -> str:
    target_size = _resolve_gemini_image_output_size(
        model_name=model_name,
        image_size=image_size,
        aspect_ratio=aspect_ratio,
    )
    if target_size is None:
        return data_url

    _, _, image_base64 = data_url.partition(",")
    try:
        image_bytes = base64.b64decode(image_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Image data URL contains invalid base64 image data.") from exc

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            normalized_image = image.convert("RGBA")
    except UnidentifiedImageError as exc:
        raise ValueError("Image data URL contains unsupported image bytes.") from exc

    fitted_image = (
        normalized_image
        if normalized_image.size == target_size
        else _fit_image_to_gemini_canvas(normalized_image, target_size)
    )
    flattened_image = _flatten_image_to_rgb(fitted_image)

    output = BytesIO()
    flattened_image.save(output, format="PNG", optimize=True)
    image_base64 = base64.b64encode(output.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"


def _resolve_gemini_image_output_size(
    *,
    model_name: str,
    image_size: str,
    aspect_ratio: str,
) -> tuple[int, int] | None:
    normalized_model = _normalize_gemini_model_name(model_name)
    normalized_image_size = image_size.upper()
    if normalized_image_size == "512":
        normalized_image_size = "0.5K"

    if normalized_model == "gemini-3.1-flash-image-preview":
        return _GEMINI_31_IMAGE_OUTPUT_SIZES.get(normalized_image_size, {}).get(
            aspect_ratio
        )
    if normalized_model == "gemini-3-pro-image-preview":
        return _GEMINI_3_PRO_IMAGE_OUTPUT_SIZES.get(normalized_image_size, {}).get(
            aspect_ratio
        )
    if normalized_model == "gemini-2.5-flash-image":
        return _GEMINI_25_IMAGE_OUTPUT_SIZES["1K"].get(aspect_ratio)
    return None


def _fit_image_to_gemini_canvas(
    image: Image.Image,
    target_size: tuple[int, int],
) -> Image.Image:
    target_width, target_height = target_size
    width, height = image.size
    scale = min(target_width / width, target_height / height)
    resized_size = (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )
    resized = image.resize(resized_size, Image.Resampling.LANCZOS)
    canvas = Image.new(
        "RGBA",
        target_size,
        _sample_image_background_color(image),
    )
    offset = (
        (target_width - resized.width) // 2,
        (target_height - resized.height) // 2,
    )
    canvas.alpha_composite(resized, offset)
    return canvas


def _flatten_image_to_rgb(image: Image.Image) -> Image.Image:
    rgba_image = image.convert("RGBA")
    background = Image.new(
        "RGBA",
        rgba_image.size,
        _sample_image_background_color(rgba_image),
    )
    background.alpha_composite(rgba_image)
    return background.convert("RGB")


def _sample_image_background_color(image: Image.Image) -> tuple[int, int, int, int]:
    red, green, blue, alpha = image.getpixel((0, 0))
    if alpha < 255:
        return (255, 255, 255, 255)
    return (red, green, blue, alpha)


def _data_url_to_inline_part(data_url: str) -> dict[str, object]:
    mime_type = _extract_data_url_mime_type(data_url)
    _, _, image_base64 = data_url.partition(",")
    return {
        "inline_data": {
            "mime_type": mime_type,
            "data": image_base64,
        }
    }


def _normalize_user_prompt(user_prompt: str | None) -> str:
    normalized = _string(user_prompt)
    if normalized:
        return normalized
    return _DEFAULT_USER_PROMPT


def _resolve_canvas_aspect_ratio(snapshot_payload: Mapping[str, object]) -> str:
    canvas_payload = _mapping(snapshot_payload.get("canvas"))
    width = _positive_float(canvas_payload.get("widthPx"), fallback=1.0)
    height = _positive_float(canvas_payload.get("heightPx"), fallback=1.0)
    ratio = width / max(height, 1.0)
    best_label = "1:1"
    best_distance = math.inf
    for label, candidate_ratio in _SUPPORTED_ASPECT_RATIOS:
        distance = abs(candidate_ratio - ratio)
        if distance < best_distance:
            best_label = label
            best_distance = distance
    return best_label


def _find_snapshot_image_path(snapshot_path: Path) -> Path | None:
    for suffix in _SUPPORTED_IMAGE_EXTENSIONS:
        candidate = snapshot_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def _encode_file_as_data_url(path: Path) -> str:
    image_bytes = path.read_bytes()
    mime_type = _guess_image_mime_type(image_bytes)
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{image_base64}"


def _normalize_image_data_url(data_url: str | None) -> str:
    if data_url is None:
        raise ValueError("Image data URL is required.")
    normalized = data_url.strip()
    if not normalized.startswith("data:image/") or "," not in normalized:
        raise ValueError("Image data URL must be a valid image data URL.")
    _, _, image_base64 = normalized.partition(",")
    try:
        base64.b64decode(image_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Image data URL contains invalid base64 image data.") from exc
    return normalized


def _extract_data_url_mime_type(data_url: str) -> str:
    prefix, _, _ = data_url.partition(",")
    mime_section = prefix.removeprefix("data:")
    mime_type, _, _ = mime_section.partition(";")
    return mime_type or "application/octet-stream"


def _extract_image_bytes(payload: Mapping[str, object]) -> bytes | None:
    for inline_key in ("inlineData", "inline_data"):
        inline_payload = payload.get(inline_key)
        if isinstance(inline_payload, Mapping):
            mime_type = inline_payload.get("mimeType") or inline_payload.get(
                "mime_type"
            )
            data = inline_payload.get("data")
            if isinstance(mime_type, str) and mime_type.startswith("image/"):
                if isinstance(data, str):
                    decoded = _decode_base64_image(data)
                    if decoded is not None:
                        return decoded

    direct_value = payload.get("b64_json") or payload.get("image") or payload.get("url")
    if isinstance(direct_value, str):
        return _decode_base64_image(direct_value)

    for value in payload.values():
        if isinstance(value, Mapping):
            nested = _extract_image_bytes(value)
            if nested is not None:
                return nested
        elif isinstance(value, list):
            nested = _extract_image_bytes_from_list(value)
            if nested is not None:
                return nested
    return None


def _extract_image_bytes_from_list(values: list[object]) -> bytes | None:
    for item in values:
        if isinstance(item, str):
            decoded = _decode_base64_image(item)
            if decoded is not None:
                return decoded
        elif isinstance(item, Mapping):
            nested = _extract_image_bytes(item)
            if nested is not None:
                return nested
    return None


def _decode_base64_image(value: str) -> bytes | None:
    normalized = value.strip()
    if normalized.startswith("data:image/") and "," in normalized:
        normalized = normalized.split(",", maxsplit=1)[1]
    try:
        return base64.b64decode(normalized, validate=True)
    except (binascii.Error, ValueError):
        return None


def _guess_image_mime_type(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    return "application/octet-stream"


def _mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    return {}


def _sequence(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _is_hex_color(value: str | None) -> bool:
    if not value or len(value) != 7 or not value.startswith("#"):
        return False
    return all(char in "0123456789abcdefABCDEF" for char in value[1:])


def _positive_float(value: object, *, fallback: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return fallback
    if numeric <= 0:
        return fallback
    return numeric


def _read_bool_env(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    logger.warning("Invalid boolean in %s=%r. Using default %s.", name, raw, default)
    return default


def _read_int_env(name: str, *, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "Invalid integer in %s=%r. Using default %s.", name, raw, default
        )
        return default


def _string(value: object, *, fallback: str = "") -> str:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return fallback


def _secret_string(value: object) -> str | None:
    normalized = _string(value)
    if not normalized:
        return None
    if normalized.startswith("${") and normalized.endswith("}"):
        return None
    if normalized.lower().startswith("your-"):
        return None
    return normalized


def _normalize_gemini_model_name(value: str) -> str:
    normalized = _string(value, fallback=_DEFAULT_IMAGE_MODEL)
    if "/" in normalized:
        return normalized.rsplit("/", maxsplit=1)[-1]
    return normalized
