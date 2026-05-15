from __future__ import annotations

import json
import logging
import math
from copy import deepcopy
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from adapters.catalog_api import (
    CatalogApiError,
    load_catalog_api_settings,
    load_catalog_inventory_payloads,
)
from api.deps import get_optional_current_user
from config import config_file, load_config
from db.models import UserAccount
from pipeline.orchestrator import (
    run_case,
    case_paths,
    _make_case_id,
    _write_json,
    _now_utc_iso,
)
from pipeline.snapshot_prompt_compiler import (
    compile_snapshot_prompt,
    compile_snapshot_prompt_from_path,
)
from pipeline.snapshot_image_renderer import (
    SnapshotEditOperation,
    render_snapshot_image,
    render_snapshot_image_from_path,
)
from pipeline.image_flow_logging import log_image_flow_event
from services.coordinate_normalization_service import CoordinateNormalizationService
from services.user_content_service import UserContentService

router = APIRouter(prefix="/pipeline", tags=["pipeline"])
logger = logging.getLogger(__name__)


class PipelineRunRequest(BaseModel):
    user_id: str = Field(
        ...,
        min_length=1,
        description="ID of the user initiating the run. Used to namespace the output directory.",
    )
    input_payload: dict[str, Any] = Field(
        ...,
        description="Full pipeline input: floorplan geometry, style preferences, and inventory constraints.",
    )
    description: str | None = Field(
        default=None,
        description="Optional free-text description of the design request (for logging/debugging).",
    )
    special_notes: str | None = Field(
        default=None,
        description="Optional extra instructions injected into the pipeline prompt.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "user_id": "demo_user",
                "description": "Modern bedroom layout",
                "special_notes": "Do not block the window. Keep circulation clear from door to bed.",
                "input_payload": {
                    "tenant_id": "demo_tenant",
                    "user_input": {
                        "description": "Design a modern bedroom for 1-2 people. Keep the layout clean and minimal.",
                        "room_type": "bedroom",
                        "floor_area_m2": 16.74,
                        "height": 2800,
                        "shape_points": [
                            {"x": 0, "y": 0},
                            {"x": 4650, "y": 0},
                            {"x": 4650, "y": 3600},
                            {"x": 0, "y": 3600},
                        ],
                        "style": "modern",
                        "windows": 1,
                        "window_direction": "north",
                    },
                    "doors": [
                        {
                            "id": "door_1",
                            "segment_mm": [{"x": 0, "y": 1600}, {"x": 0, "y": 2600}],
                            "swing_radius_mm": 900,
                            "hinge_hint": "LEFT",
                        }
                    ],
                    "windows": [
                        {
                            "id": "window_1",
                            "segment_mm": [{"x": 2000, "y": 0}, {"x": 4000, "y": 0}],
                            "clearance_mm": 150,
                        }
                    ],
                },
            }
        }
    }


class PipelineRunResponse(BaseModel):
    case_id: str
    case_dir: str
    status_path: str
    status: str


class PipelineNormalizeRunRequest(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {
                "room": {
                    "key": "-3:2.5",
                    "name": "Phòng khách, Ăn + Bếp + Không gian chung",
                    "polygons": [
                        [-2.0019108465826676, -0.8],
                        [-2, 1.5],
                        [-0.2, 1.5],
                        [3.5, 1.5],
                        [3.5, 5],
                        [-7.998060083743152, 5],
                        [-8, -0.8],
                        [-6.5, -0.8],
                    ],
                    "materialId": "classic",
                    "description": "Mô tả chung hoặc chi tiết của căn phòng",
                    "materialLabel": "Cổ điển",
                },
                "walls": [],
                "openings": [],
                "source_unit": "auto",
                "tenant_id": "demo_tenant",
                "user_id": "demo_user",
                "style": "modern",
                "allow_generated_accessories": False,
            }
        },
    )

    room: dict[str, Any] = Field(
        ...,
        description="Single-room payload with polygons/material/description.",
    )
    walls: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Room wall segments in the same coordinate space as room.polygons.",
    )
    openings: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Door/window objects with position, rotation, size, objectRole, and snappedToWall.",
    )
    source_unit: Literal[
        "auto",
        "m",
        "meter",
        "meters",
        "mm",
        "millimeter",
        "millimeters",
    ] = Field(
        default="auto",
        description="Unit of the coordinates in the payload.",
    )
    tenant_id: str | None = Field(
        default="demo_tenant",
        description="Tenant context used for catalog and room configuration lookups.",
    )
    user_id: str | None = Field(
        default="coordinate_normalizer_preview",
        description="User ID attached to generated pipeline cases.",
    )
    description: str | None = Field(
        default=None,
        description="Optional design description forwarded to each room pipeline run.",
    )
    special_notes: str | None = Field(
        default=None,
        description="Optional extra instructions forwarded to each room pipeline run.",
    )
    style: str | None = Field(
        default="modern",
        description="Design style hint used during normalization.",
    )
    split_largest_room: bool = Field(
        default=True,
        description="If true, the largest room is split into living/kitchen sub-zones.",
    )
    allow_generated_accessories: bool = Field(
        default=False,
        description="If true, stylist may auto-add accessory/decor objects.",
    )


class PipelineNormalizeRunPosition(BaseModel):
    x: float
    y: float
    z: float


class PipelineNormalizeRunRotation(BaseModel):
    x: float
    y: float
    z: float
    w: float


class PipelineNormalizeRunObject(BaseModel):
    name: str | None = None
    size: list[float] | None = None
    type: str | None = None
    color: str | None = None
    modelUrl: str
    position: PipelineNormalizeRunPosition
    rotation: PipelineNormalizeRunRotation
    objectRole: str | None = None
    catalogItemId: str | None = None


class PipelineNormalizeRunOption(BaseModel):
    optionId: str
    label: str | None = None
    layoutScore: int | None = None
    hardValid: bool | None = None
    complete: bool | None = None
    coverageRatio: float | None = None
    objects: list[PipelineNormalizeRunObject] = Field(default_factory=list)
    openings: list[dict[str, Any]] = Field(default_factory=list)


class PipelineNormalizeRunResponse(BaseModel):
    objects: list[PipelineNormalizeRunObject] = Field(default_factory=list)
    openings: list[dict[str, Any]] = Field(default_factory=list)
    selectedOptionId: str | None = None
    options: list[PipelineNormalizeRunOption] = Field(default_factory=list)
    selectionSummary: dict[str, Any] | None = None


class PipelineStatusResponse(BaseModel):
    case_id: str = Field(..., description="Unique ID of this pipeline run.")
    stage: str = Field(
        ...,
        description="Current lifecycle stage: 'queued' → 'running' → 'done' | 'error'.",
    )
    updated_at_utc: str = Field(
        ..., description="ISO-8601 timestamp of the last status update."
    )
    error: str | None = Field(
        default=None, description="Error message if stage is 'error', otherwise null."
    )
    message: str | None = Field(
        default=None,
        description="Human-readable progress message from the currently running module.",
    )
    progress_current: int | None = Field(
        default=None, description="Number of completed steps in the current stage."
    )
    progress_total: int | None = Field(
        default=None, description="Total number of steps in the current stage."
    )
    actions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Full history of stage transitions since the run started.",
    )


class PipelineResultResponse(BaseModel):
    case_id: str
    result: dict[str, Any]


class ArtifactResponse(BaseModel):
    case_id: str
    name: str
    payload: dict[str, Any]


class ClusterListResponse(BaseModel):
    case_id: str
    clusters: list[str]


class SnapshotPromptCompileRequest(BaseModel):
    snapshot_payload: dict[str, Any] | None = None
    snapshot_path: str | None = None


class SnapshotPromptCompileResponse(BaseModel):
    compilation: dict[str, Any]


class SnapshotImagePresetSelection(BaseModel):
    style: str | None = None
    lighting: str | None = None
    scenery: str | None = None


class SnapshotImageEditOperationRequest(BaseModel):
    object_id: str
    object_name: str | None = None
    replacement_image_data_url: str | None = None
    target_color: str | None = None


class SnapshotImageRenderRequest(BaseModel):
    snapshot_payload: dict[str, Any] | None = Field(
        default=None,
        description="Snapshot layout payload (objects, room geometry). Required when not using snapshot_path.",
    )
    snapshot_path: str | None = Field(
        default=None,
        description="Server-side file path to a snapshot JSON file. Alternative to snapshot_payload.",
    )
    snapshot_image_data_url: str | None = Field(
        default=None,
        description="Base64 data URL of the current rendered snapshot image. Required when using snapshot_payload.",
    )
    layout_reference_image_data_url: str | None = Field(
        default=None,
        description="Optional reference image for the target layout composition (guides object placement in the render).",
    )
    annotated_reference_image_data_url: str | None = Field(
        default=None,
        description="Optional annotated version of the layout reference image.",
    )
    scene_reference_image_data_url: str | None = Field(
        default=None,
        description="Optional scene/mood reference image. Behaviour depends on scene_reference_mode.",
    )
    scene_reference_mode: Literal[
        "none",
        "target_layout_with_scene_reference",
        "scene_reference_camera_transfer",
    ] = Field(
        default="none",
        description=(
            "How the scene reference image is used:\n"
            "- `none` — ignored\n"
            "- `target_layout_with_scene_reference` — scene mood applied to the target layout\n"
            "- `scene_reference_camera_transfer` — camera angle is transferred from the reference"
        ),
    )
    user_prompt: str | None = Field(
        default=None,
        description="Free-text instructions appended to the image generation prompt.",
    )
    render_mode: Literal["generate", "edit"] = Field(
        default="generate",
        description="'generate' creates a new image from scratch; 'edit' applies edit_operations to edit_source_image_data_url.",
    )
    preset_selection: SnapshotImagePresetSelection = Field(
        default_factory=lambda: SnapshotImagePresetSelection(),
        description="Style, lighting, and scenery presets. Use GET /pipeline/render-presets to see available options.",
    )
    root_layout_id: str | None = Field(
        default=None,
        description="ID of the root layout variant this render is based on (used for metadata).",
    )
    edit_operations: list[SnapshotImageEditOperationRequest] = Field(
        default_factory=list,
        description="List of per-object edits to apply when render_mode is 'edit' (e.g. swap object texture or color).",
    )
    edit_source_image_data_url: str | None = Field(
        default=None,
        description="Base64 data URL of the image to edit. Required when render_mode is 'edit'.",
    )


class SnapshotImageRenderResponse(BaseModel):
    render: dict[str, Any]
    saved_render: dict[str, Any] | None = None


class SnapshotImagePresetOptionResponse(BaseModel):
    label: str
    prompt_suffix: str | None = None
    reference_image: str | None = None


class SnapshotImagePresetsResponse(BaseModel):
    styles: dict[str, SnapshotImagePresetOptionResponse]
    lights: dict[str, SnapshotImagePresetOptionResponse]
    sceneries: dict[str, SnapshotImagePresetOptionResponse]


def get_user_content_service() -> UserContentService:
    return UserContentService()


def _enrich_rotation_ccw(
    *,
    stylist_payload: dict[str, Any],
    absolute_layout_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    UI needs normalized orientation metadata per object.
    Stylist output intentionally omits it; we stitch it from absolute_layout,
    which contains the final semantic rotation and front-facing direction.
    """
    if not isinstance(stylist_payload, dict):
        return stylist_payload
    if not absolute_layout_payload or not isinstance(absolute_layout_payload, dict):
        return stylist_payload

    abs_objects = absolute_layout_payload.get("objects") or []
    if not isinstance(abs_objects, list) or not abs_objects:
        return stylist_payload

    by_id: dict[str, dict[str, Any]] = {}
    by_bbox: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for ao in abs_objects:
        if not isinstance(ao, dict):
            continue
        oid = ao.get("object_id")
        rot = ao.get("rotation_ccw", ao.get("rot"))
        bbox = ao.get("bbox") or {}
        if not isinstance(oid, str) or not oid:
            continue
        try:
            rotation_ccw = int(rot) % 360
        except (TypeError, ValueError):
            continue
        orientation_payload = {
            "rotation_ccw": rotation_ccw,
            "front_world": deepcopy(ao.get("front_world"))
            if isinstance(ao.get("front_world"), dict)
            else None,
            "front_side_world": (
                str(ao.get("front_side_world")).strip().lower()
                if isinstance(ao.get("front_side_world"), str)
                and str(ao.get("front_side_world")).strip()
                else None
            ),
            "axis_world": deepcopy(ao.get("axis_world"))
            if isinstance(ao.get("axis_world"), dict)
            else None,
        }
        by_id[oid] = orientation_payload
        # Best-effort fallback matching: bbox equality.
        # This helps if stylist instance ids don't exactly match absolute layout ids.
        try:
            key = (
                int(bbox.get("min_x", 0)),
                int(bbox.get("min_y", 0)),
                int(bbox.get("max_x", 0)),
                int(bbox.get("max_y", 0)),
            )
            by_bbox[key] = orientation_payload
        except Exception:
            pass

    out = dict(stylist_payload)
    objects = out.get("objects") or []
    if not isinstance(objects, list) or not objects:
        return out

    enriched: list[dict[str, Any]] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        inst = obj.get("instance_id") or obj.get("id")
        bbox = obj.get("bbox") or {}

        orientation_payload: dict[str, Any] | None = None
        if isinstance(inst, str) and inst in by_id:
            orientation_payload = by_id[inst]
        else:
            # bbox match (best-effort)
            try:
                key = (
                    int(bbox.get("min_x", 0)),
                    int(bbox.get("min_y", 0)),
                    int(bbox.get("max_x", 0)),
                    int(bbox.get("max_y", 0)),
                )
                if key in by_bbox:
                    orientation_payload = by_bbox[key]
            except Exception:
                orientation_payload = None

        next_obj = dict(obj)
        next_obj["rotation_ccw"] = (
            int(
                (orientation_payload or {}).get(
                    "rotation_ccw",
                    next_obj.get("rotation_ccw", next_obj.get("rot", 0)),
                )
                or 0
            )
            % 360
        )
        next_obj["front_world"] = deepcopy(
            (orientation_payload or {}).get("front_world")
        )
        next_obj["front_side_world"] = (orientation_payload or {}).get(
            "front_side_world"
        )
        next_obj["axis_world"] = deepcopy((orientation_payload or {}).get("axis_world"))
        enriched.append(next_obj)

    out["objects"] = enriched
    return out


def _run_in_thread(**kwargs: Any) -> None:
    try:
        run_case(**kwargs)
    except Exception as exc:  # noqa: BLE001
        paths = case_paths(kwargs["case_id"], kwargs["cases_root"])
        existing_payload: dict[str, Any] = {}
        if paths.status.exists():
            try:
                existing_payload = json.loads(paths.status.read_text())
            except Exception:
                existing_payload = {}
        actions = existing_payload.get("actions")
        action_history = (
            [item for item in actions if isinstance(item, dict)]
            if isinstance(actions, list)
            else []
        )
        updated_at_utc = _now_utc_iso()
        action_history.append(
            {
                "stage": "error",
                "message": str(exc),
                "updated_at_utc": updated_at_utc,
                "progress_current": existing_payload.get("progress_current"),
                "progress_total": existing_payload.get("progress_total"),
                "error": str(exc),
            }
        )
        _write_json(
            paths.status,
            {
                "case_id": paths.case_id,
                "stage": "error",
                "updated_at_utc": updated_at_utc,
                "error": str(exc),
                "actions": action_history,
            },
        )


_NORMALIZE_RUN_CONTROL_FIELDS = {
    "source_unit",
    "tenant_id",
    "user_id",
    "description",
    "special_notes",
    "style",
    "split_largest_room",
    "allow_generated_accessories",
}


def _normalize_run_floorplan_payload(
    req: PipelineNormalizeRunRequest,
) -> dict[str, Any]:
    return req.model_dump(
        exclude=_NORMALIZE_RUN_CONTROL_FIELDS,
        exclude_none=True,
    )


def _coerce_normalize_run_payload(payload: dict[str, Any]) -> dict[str, Any]:
    room_payload = payload.get("room")
    if isinstance(room_payload, dict):
        out = {
            str(key): deepcopy(value)
            for key, value in payload.items()
            if key not in {"room", "openings", "objects"}
        }
        out["id"] = (
            _string_or_none(payload.get("id"))
            or _string_or_none(room_payload.get("key"))
            or "single-room-design"
        )
        out["name"] = (
            _string_or_none(payload.get("name"))
            or _string_or_none(room_payload.get("name"))
            or "Single Room Design"
        )
        out["rooms"] = [deepcopy(room_payload)]

        objects: list[Any] = []
        existing_objects = payload.get("objects")
        if isinstance(existing_objects, list):
            objects.extend(deepcopy(existing_objects))
        openings = payload.get("openings")
        if isinstance(openings, list):
            objects.extend(deepcopy(openings))
        if objects:
            out["objects"] = objects
        return out

    if isinstance(payload.get("rooms"), list):
        return payload
    if any(key in payload for key in ("polygons", "polygon", "polygon_ccw")):
        room = dict(payload)
        return {
            "id": _string_or_none(payload.get("id")) or "single-room-design",
            "name": _string_or_none(payload.get("name")) or "Single Room Design",
            "rooms": [room],
        }
    return payload


def _safe_case_segment(value: str | None, fallback: str) -> str:
    raw = value or fallback
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw)
    safe = safe.strip("_-")
    return safe[:48] or fallback


def _json_object_from_path(path: Any) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _enriched_case_result(case_id: str, final_output: Any) -> dict[str, Any]:
    paths = case_paths(case_id)
    styled_payload = final_output if isinstance(final_output, dict) else None
    if styled_payload is None:
        styled_payload = _json_object_from_path(paths.stylist)
    if styled_payload is None:
        raise RuntimeError(f"Pipeline case {case_id} did not produce a styled result.")

    absolute_payload = _json_object_from_path(paths.absolute_layout)
    return _enrich_rotation_ccw(
        stylist_payload=styled_payload,
        absolute_layout_payload=absolute_payload,
    )


def _enriched_case_options(case_id: str, final_output: Any) -> list[dict[str, Any]]:
    paths = case_paths(case_id)
    variants_payload = _json_object_from_path(paths.layout_variants)
    variants = (
        variants_payload.get("variants") if isinstance(variants_payload, dict) else None
    )
    out: list[dict[str, Any]] = []
    if isinstance(variants, list):
        for index, variant in enumerate(variants, start=1):
            if not isinstance(variant, dict):
                continue
            styled_payload = variant.get("styled_result")
            if not isinstance(styled_payload, dict):
                continue
            absolute_payload = variant.get("absolute_layout")
            out.append(
                {
                    "option_id": _string_or_none(variant.get("variant_id"))
                    or f"variant_{index}",
                    "label": _string_or_none(variant.get("label")) or f"Option {index}",
                    "layout_score": _number(variant.get("layout_score")),
                    "hard_valid": variant.get("hard_valid")
                    if isinstance(variant.get("hard_valid"), bool)
                    else None,
                    "complete": variant.get("complete")
                    if isinstance(variant.get("complete"), bool)
                    else None,
                    "coverage_ratio": _number(variant.get("coverage_ratio")),
                    "styled_payload": _enrich_rotation_ccw(
                        stylist_payload=styled_payload,
                        absolute_layout_payload=absolute_payload
                        if isinstance(absolute_payload, dict)
                        else None,
                    ),
                }
            )
    if out:
        return out

    return [
        {
            "option_id": "variant_1",
            "label": "Option 1",
            "layout_score": None,
            "hard_valid": None,
            "complete": None,
            "coverage_ratio": None,
            "styled_payload": _enriched_case_result(case_id, final_output),
        }
    ]


def _case_selection_summary(case_id: str) -> dict[str, Any] | None:
    variants_payload = _json_object_from_path(case_paths(case_id).layout_variants)
    if not isinstance(variants_payload, dict):
        return None
    summary = variants_payload.get("selection_summary")
    return summary if isinstance(summary, dict) else None


def _collect_object_types(payloads: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for payload in payloads:
        objects = payload.get("objects")
        if not isinstance(objects, list):
            continue
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            object_type = _string_or_none(obj.get("object_type"))
            if object_type is None:
                object_type = _string_or_none(obj.get("type"))
            normalized = _catalog_key(object_type)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            out.append(object_type or normalized)
            if normalized.endswith("_lamp") and "lamp" not in seen:
                seen.add("lamp")
                out.append("lamp")
    return out


def _collect_catalog_item_ids(payloads: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for payload in payloads:
        objects = payload.get("objects")
        if not isinstance(objects, list):
            continue
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            for value in (
                obj.get("catalogItemId"),
                obj.get("catalog_id"),
                obj.get("inventory_id"),
                obj.get("source_id"),
            ):
                clean = _string_or_none(value)
                if clean is None:
                    continue
                normalized = _catalog_key(clean)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                out.append(clean)
    return out


def _load_catalog_index(
    *,
    object_types: list[str],
    catalog_item_ids: list[str],
) -> dict[str, Any]:
    if not object_types and not catalog_item_ids:
        return {"by_id": {}, "by_type": {}}
    try:
        payloads = load_catalog_inventory_payloads(
            item_ids=catalog_item_ids,
            types=object_types,
            default_rotation_presence=None,
        )
    except CatalogApiError as exc:
        logger.exception("Catalog lookup failed for /pipeline/normalize-run.")
        return {"by_id": {}, "by_type": {}, "error": str(exc)}

    by_id: dict[str, dict[str, Any]] = {}
    by_type: dict[str, list[dict[str, Any]]] = {}
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        attributes = _mapping(payload.get("attributes"))
        for value in (
            payload.get("id"),
            payload.get("inventory_id"),
            payload.get("catalog_id"),
            attributes.get("inventory_id"),
            attributes.get("catalog_id"),
        ):
            key = _catalog_key(value)
            if key:
                by_id.setdefault(key, payload)
        for value in (
            payload.get("object_type"),
            payload.get("type"),
            payload.get("asset_type"),
            attributes.get("semantic_object_type"),
            attributes.get("category"),
            attributes.get("object_role"),
            attributes.get("objectRole"),
            attributes.get("slug"),
            attributes.get("sku_slug"),
        ):
            key = _catalog_key(value)
            if not key:
                continue
            by_type.setdefault(key, []).append(payload)
    return {"by_id": by_id, "by_type": by_type}


def _match_catalog_payload(
    obj: dict[str, Any],
    catalog_index: dict[str, Any],
) -> dict[str, Any] | None:
    by_id = _mapping(catalog_index.get("by_id"))
    by_type = _mapping(catalog_index.get("by_type"))
    for value in (
        obj.get("catalogItemId"),
        obj.get("catalog_id"),
        obj.get("inventory_id"),
        obj.get("source_id"),
    ):
        key = _catalog_key(value)
        match = by_id.get(key) if key else None
        if isinstance(match, dict):
            return match

    type_key = _catalog_key(obj.get("object_type") or obj.get("type"))
    candidates = by_type.get(type_key) if type_key else None
    if isinstance(candidates, list) and candidates:
        first = candidates[0]
        return first if isinstance(first, dict) else None
    if type_key and type_key.endswith("_lamp"):
        lamp_candidates = by_type.get("lamp")
        if isinstance(lamp_candidates, list) and lamp_candidates:
            first = lamp_candidates[0]
            return first if isinstance(first, dict) else None
    return None


def _normalize_run_room_objects(
    *,
    styled_payload: dict[str, Any],
    catalog_index: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[float], list[list[float] | None]]:
    objects = styled_payload.get("objects")
    if not isinstance(objects, list):
        return [], [], []

    out: list[dict[str, Any]] = []
    rotations_ccw: list[float] = []
    default_rotations: list[list[float] | None] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        bbox = _bbox_from_object(obj)
        if bbox is None:
            continue
        catalog_payload = _match_catalog_payload(obj, catalog_index)
        catalog_item_id = _catalog_item_id(catalog_payload, obj)
        if catalog_payload is None and catalog_item_id is None:
            continue
        size_mm = _catalog_size_mm(catalog_payload)
        if size_mm is None:
            size_mm = [
                max(1.0, bbox["max_x"] - bbox["min_x"]),
                300.0,
                max(1.0, bbox["max_y"] - bbox["min_y"]),
            ]
        center_x = (bbox["min_x"] + bbox["max_x"]) / 2.0
        center_z = (bbox["min_y"] + bbox["max_y"]) / 2.0
        rotation_ccw = _number(obj.get("rotation_ccw"))
        if rotation_ccw is None:
            rotation_ccw = _number(obj.get("rot")) or 0.0
        default_rotation = _catalog_default_rotation(catalog_payload)
        model_url = _catalog_model_url(catalog_payload, obj)
        if model_url is None:
            logger.warning(
                "Skipping normalize-run object without modelUrl: catalog_item_id=%s object_type=%s",
                catalog_item_id,
                obj.get("object_type") or obj.get("type"),
            )
            continue
        output_obj: dict[str, Any] = {
            "name": _catalog_name(catalog_payload, obj),
            "size": size_mm,
            "type": _catalog_shape_type(catalog_payload),
            "color": _catalog_color(catalog_payload, obj),
            "modelUrl": model_url,
            "position": {
                "x": center_x,
                "y": size_mm[1] / 2.0,
                "z": center_z,
            },
            "rotation_ccw": rotation_ccw,
            "objectRole": _catalog_object_role(catalog_payload),
            "catalogItemId": catalog_item_id,
        }
        out.append(output_obj)
        rotations_ccw.append(rotation_ccw)
        default_rotations.append(default_rotation)
    return out, rotations_ccw, default_rotations


def _finalize_normalize_run_objects(
    *,
    restored_objects: list[Any],
    rotations_ccw: list[float],
    default_rotations: list[list[float] | None],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, restored in enumerate(restored_objects):
        if not isinstance(restored, dict):
            continue
        rotation_ccw = rotations_ccw[index] if index < len(rotations_ccw) else 0.0
        default_rotation = (
            default_rotations[index] if index < len(default_rotations) else None
        )
        item = {
            "name": restored.get("name"),
            "size": restored.get("size"),
            "type": restored.get("type"),
            "color": restored.get("color"),
            "modelUrl": restored.get("modelUrl"),
            "position": restored.get("position"),
            "rotation": _quaternion_dict(
                _combine_yaw_and_default_rotation(rotation_ccw, default_rotation)
            ),
            "objectRole": restored.get("objectRole"),
            "catalogItemId": restored.get("catalogItemId"),
        }
        out.append(item)
    return out


def _normalize_run_restored_objects(
    *,
    coordinate_service: CoordinateNormalizationService,
    styled_payload: dict[str, Any],
    catalog_index: dict[str, Any],
    transform: dict[str, Any],
    room_id: str,
) -> list[dict[str, Any]]:
    local_objects, rotations_ccw, default_rotations = _normalize_run_room_objects(
        styled_payload=styled_payload,
        catalog_index=catalog_index,
    )
    if not local_objects:
        return []
    try:
        restored = coordinate_service.restore_output(
            local_objects,
            transform,
            coordinate_space="room_local",
            room_id=room_id,
            rotation_input="degrees",
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    restored_payload = restored.get("restored_payload")
    restored_objects = restored_payload if isinstance(restored_payload, list) else []
    return _finalize_normalize_run_objects(
        restored_objects=restored_objects,
        rotations_ccw=rotations_ccw,
        default_rotations=default_rotations,
    )


def _bbox_from_object(obj: dict[str, Any]) -> dict[str, float] | None:
    bbox = _mapping(obj.get("bbox"))
    values = {
        "min_x": _number(bbox.get("min_x")),
        "min_y": _number(bbox.get("min_y")),
        "max_x": _number(bbox.get("max_x")),
        "max_y": _number(bbox.get("max_y")),
    }
    if all(value is not None for value in values.values()):
        return {key: float(value or 0.0) for key, value in values.items()}

    polygon = obj.get("polygon_ccw")
    if not isinstance(polygon, list):
        return None
    points: list[tuple[float, float]] = []
    for point in polygon:
        if not isinstance(point, dict):
            continue
        x = _number(point.get("x"))
        y = _number(point.get("y"))
        if x is not None and y is not None:
            points.append((x, y))
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return {
        "min_x": min(xs),
        "min_y": min(ys),
        "max_x": max(xs),
        "max_y": max(ys),
    }


def _catalog_size_mm(catalog_payload: dict[str, Any] | None) -> list[float] | None:
    if catalog_payload is None:
        return None
    attributes = _mapping(catalog_payload.get("attributes"))
    size_mm = attributes.get("size_mm_xyz")
    if isinstance(size_mm, list) and len(size_mm) == 3:
        values = [_number(item) for item in size_mm]
        if all(value is not None and value > 0 for value in values):
            return [float(value or 0.0) for value in values]

    length = _number(catalog_payload.get("length_mm") or attributes.get("length_mm"))
    height = _number(catalog_payload.get("height_mm") or attributes.get("height_mm"))
    width = _number(catalog_payload.get("width_mm") or attributes.get("width_mm"))
    if length is not None and height is not None and width is not None:
        if length > 0 and height > 0 and width > 0:
            return [float(length), float(height), float(width)]
    return None


def _catalog_default_rotation(
    catalog_payload: dict[str, Any] | None,
) -> list[float] | None:
    if catalog_payload is None:
        return None
    attributes = _mapping(catalog_payload.get("attributes"))
    return _quaternion_from_value(
        attributes.get("defaultRotation") or attributes.get("default_rotation")
    )


def _catalog_name(
    catalog_payload: dict[str, Any] | None,
    obj: dict[str, Any],
) -> str | None:
    if catalog_payload is not None:
        for value in (
            catalog_payload.get("name"),
            catalog_payload.get("inventory_name"),
        ):
            clean = _string_or_none(value)
            if clean is not None:
                return clean
    for value in (obj.get("inventory_name"), obj.get("name"), obj.get("object_type")):
        clean = _string_or_none(value)
        if clean is not None:
            return clean
    return None


def _catalog_shape_type(catalog_payload: dict[str, Any] | None) -> str:
    attributes = _mapping(catalog_payload.get("attributes")) if catalog_payload else {}
    return _string_or_none(attributes.get("shape_type")) or "model"


def _catalog_color(
    catalog_payload: dict[str, Any] | None,
    obj: dict[str, Any],
) -> str | None:
    attributes = _mapping(catalog_payload.get("attributes")) if catalog_payload else {}
    return (
        _string_or_none(attributes.get("color_hex"))
        or _string_or_none(obj.get("color"))
        or _string_or_none(obj.get("color_hex"))
    )


def _catalog_model_url(
    catalog_payload: dict[str, Any] | None,
    obj: dict[str, Any],
) -> str | None:
    attributes = _mapping(catalog_payload.get("attributes")) if catalog_payload else {}
    candidates = (
        attributes.get("modelUrl"),
        attributes.get("model_url"),
        attributes.get("model3d"),
        attributes.get("model_3d"),
        attributes.get("default_model"),
        attributes.get("files"),
        attributes.get("model_variants"),
        attributes.get("modelVariants"),
        catalog_payload.get("modelUrl") if catalog_payload else None,
        catalog_payload.get("model_url") if catalog_payload else None,
        catalog_payload.get("files") if catalog_payload else None,
        obj.get("modelUrl"),
        obj.get("model_url"),
    )
    for candidate in candidates:
        url = _model_url_from_value(candidate)
        if url is not None:
            return _public_asset_url(url)
    return None


def _model_url_from_value(value: Any) -> str | None:
    clean = _string_or_none(value)
    if clean is not None:
        return clean

    if isinstance(value, dict):
        for key in (
            "url",
            "modelUrl",
            "model_url",
            "storage_key",
            "storageKey",
            "src",
            "href",
        ):
            clean = _string_or_none(value.get(key))
            if clean is not None:
                return clean
        for nested in value.values():
            clean = _model_url_from_value(nested)
            if clean is not None:
                return clean

    if isinstance(value, list):
        for item in value:
            item_map = _mapping(item)
            role = _catalog_key(item_map.get("role") or item_map.get("file_kind"))
            if role and role not in {"model", "model_3d", "3d_model", "model_gltf"}:
                continue
            clean = _model_url_from_value(item)
            if clean is not None:
                return clean

    return None


def _public_asset_url(value: str) -> str:
    if value.startswith(("http://", "https://")):
        return value
    settings = load_catalog_api_settings()
    return settings.asset_base_url.rstrip("/") + "/" + value.lstrip("/")


def _catalog_object_role(catalog_payload: dict[str, Any] | None) -> str | None:
    attributes = _mapping(catalog_payload.get("attributes")) if catalog_payload else {}
    return _string_or_none(attributes.get("objectRole")) or _string_or_none(
        attributes.get("object_role")
    )


def _catalog_item_id(
    catalog_payload: dict[str, Any] | None,
    obj: dict[str, Any],
) -> str | None:
    if catalog_payload is not None:
        for value in (
            catalog_payload.get("catalog_id"),
            catalog_payload.get("id"),
            catalog_payload.get("inventory_id"),
        ):
            clean = _string_or_none(value)
            if clean is not None:
                return clean
    return _string_or_none(
        obj.get("catalogItemId") or obj.get("catalog_id") or obj.get("inventory_id")
    )


def _combine_yaw_and_default_rotation(
    rotation_ccw: float,
    default_rotation: list[float] | None,
) -> list[float]:
    yaw_rotation = _yaw_degrees_to_quaternion(rotation_ccw)
    if default_rotation is None:
        return yaw_rotation
    return _normalize_quaternion(_multiply_quaternions(yaw_rotation, default_rotation))


def _yaw_degrees_to_quaternion(degrees: float) -> list[float]:
    half_angle = math.radians(degrees) / 2.0
    return [0.0, math.sin(half_angle), 0.0, math.cos(half_angle)]


def _multiply_quaternions(left: list[float], right: list[float]) -> list[float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return [
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ]


def _quaternion_from_value(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        components = [
            _number(value.get("x")),
            _number(value.get("y")),
            _number(value.get("z")),
            _number(value.get("w")),
        ]
    elif isinstance(value, list) and len(value) == 4:
        components = [_number(item) for item in value]
    else:
        return None
    if any(component is None for component in components):
        return None
    return _normalize_quaternion([float(component or 0.0) for component in components])


def _normalize_quaternion(value: list[float]) -> list[float]:
    norm = math.sqrt(sum(component * component for component in value))
    if norm <= 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return [round(component / norm, 12) for component in value]


def _quaternion_dict(value: list[float]) -> dict[str, float]:
    normalized = _normalize_quaternion(value)
    return {
        "x": normalized[0],
        "y": normalized[1],
        "z": normalized[2],
        "w": normalized[3],
    }


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _string_or_none(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    clean = value.strip()
    return clean or None


def _catalog_key(value: Any) -> str:
    clean = _string_or_none(value)
    if clean is None:
        return ""
    return clean.lower().replace("-", "_").replace(" ", "_")


_PIPELINE_RUN_EXAMPLES = {
    "bedroom_simple": {
        "summary": "Bedroom – simple shape (recommended starting point)",
        "description": (
            "A rectangular 4.65 × 3.6 m bedroom with one door and one window. "
            "All coordinates are in **millimetres**. "
            "Copy this, hit Execute, then poll `GET /pipeline/{case_id}/status`."
        ),
        "value": {
            "user_id": "demo_user",
            "description": "Modern bedroom layout",
            "special_notes": "Do not block the window. Keep circulation clear from door to bed.",
            "input_payload": {
                "tenant_id": "demo_tenant",
                "user_input": {
                    "description": "Design a modern bedroom for 1-2 people. Keep the layout clean and minimal.",
                    "room_type": "bedroom",
                    "floor_area_m2": 16.74,
                    "height": 2800,
                    "shape_points": [
                        {"x": 0, "y": 0},
                        {"x": 4650, "y": 0},
                        {"x": 4650, "y": 3600},
                        {"x": 0, "y": 3600},
                    ],
                    "style": "modern",
                    "windows": 1,
                    "window_direction": "north",
                },
                "doors": [
                    {
                        "id": "door_1",
                        "segment_mm": [{"x": 0, "y": 1600}, {"x": 0, "y": 2600}],
                        "swing_radius_mm": 900,
                        "hinge_hint": "LEFT",
                    }
                ],
                "windows": [
                    {
                        "id": "window_1",
                        "segment_mm": [{"x": 2000, "y": 0}, {"x": 4000, "y": 0}],
                        "clearance_mm": 150,
                    }
                ],
            },
        },
    },
    "living_room_japandi": {
        "summary": "Living room – Japandi style 6 × 4 m",
        "description": "A rectangular living room with Japandi styling. No doors or windows specified — the pipeline will infer reasonable defaults.",
        "value": {
            "user_id": "demo_user",
            "description": "Japandi living room",
            "input_payload": {
                "tenant_id": "demo_tenant",
                "user_input": {
                    "description": "Japandi living room, calm and minimal with natural materials.",
                    "room_type": "living_room",
                    "floor_area_m2": 24.0,
                    "height": 2700,
                    "shape_points": [
                        {"x": 0, "y": 0},
                        {"x": 6000, "y": 0},
                        {"x": 6000, "y": 4000},
                        {"x": 0, "y": 4000},
                    ],
                    "style": "japandi",
                    "windows": 2,
                    "window_direction": "south",
                },
                "doors": [
                    {
                        "id": "door_1",
                        "segment_mm": [{"x": 0, "y": 1700}, {"x": 0, "y": 2700}],
                        "swing_radius_mm": 900,
                        "hinge_hint": "RIGHT",
                    }
                ],
                "windows": [
                    {
                        "id": "window_1",
                        "segment_mm": [{"x": 500, "y": 4000}, {"x": 2500, "y": 4000}],
                        "clearance_mm": 200,
                    },
                    {
                        "id": "window_2",
                        "segment_mm": [{"x": 3500, "y": 4000}, {"x": 5500, "y": 4000}],
                        "clearance_mm": 200,
                    },
                ],
            },
        },
    },
    "office_two_workstations": {
        "summary": "Office – 2 workstations, 5 × 4 m",
        "description": "A square office room that needs two separate work zones.",
        "value": {
            "user_id": "demo_user",
            "description": "Office with two workstations",
            "special_notes": "Need 2 separate work zones. Keep a clear aisle between them.",
            "input_payload": {
                "tenant_id": "demo_tenant",
                "user_input": {
                    "description": "Office for 2 people, modern style, need 2 dedicated work zones.",
                    "room_type": "office",
                    "floor_area_m2": 20.0,
                    "height": 2800,
                    "shape_points": [
                        {"x": 0, "y": 0},
                        {"x": 5000, "y": 0},
                        {"x": 5000, "y": 4000},
                        {"x": 0, "y": 4000},
                    ],
                    "style": "modern",
                    "windows": 1,
                    "window_direction": "east",
                },
                "doors": [
                    {
                        "id": "door_1",
                        "segment_mm": [{"x": 0, "y": 800}, {"x": 0, "y": 1800}],
                        "swing_radius_mm": 900,
                        "hinge_hint": "RIGHT",
                    }
                ],
                "windows": [
                    {
                        "id": "window_1",
                        "segment_mm": [{"x": 5000, "y": 1000}, {"x": 5000, "y": 3000}],
                        "clearance_mm": 150,
                    }
                ],
            },
        },
    },
}


@router.post(
    "/normalize-run",
    response_model=PipelineNormalizeRunResponse,
    summary="Normalize frontend input, run the pipeline, and restore output objects",
)
def normalize_run_pipeline(
    req: PipelineNormalizeRunRequest,
) -> PipelineNormalizeRunResponse:
    """
    One-shot endpoint for frontend floorplan payloads.

    It accepts a single-room frontend payload (`room`, `walls`, `openings`), runs
    the generated room pipeline request synchronously, and returns frontend-ready
    furniture objects in the original coordinate space.
    """
    coordinate_service = CoordinateNormalizationService()
    floorplan_payload = _normalize_run_floorplan_payload(req)
    single_room_payload = isinstance(floorplan_payload.get("room"), dict) or (
        not isinstance(floorplan_payload.get("rooms"), list)
        and any(
            key in floorplan_payload for key in ("polygons", "polygon", "polygon_ccw")
        )
    )
    try:
        normalized = coordinate_service.normalize_input(
            _coerce_normalize_run_payload(floorplan_payload),
            source_unit=req.source_unit,
            tenant_id=req.tenant_id,
            user_id=req.user_id,
            description=req.description,
            special_notes=req.special_notes,
            style=req.style,
            split_largest_room=req.split_largest_room and not single_room_payload,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    system_inputs = normalized.get("system_inputs")
    if not isinstance(system_inputs, list) or not system_inputs:
        raise HTTPException(
            status_code=422,
            detail="No room pipeline inputs were produced from the payload.",
        )

    base_case_id = _make_case_id(req.user_id or "normalize_run")
    room_options: list[tuple[str, list[dict[str, Any]]]] = []
    selection_summary: dict[str, Any] | None = None
    for index, item in enumerate(system_inputs, start=1):
        if not isinstance(item, dict):
            continue
        room_id = _string_or_none(item.get("room_id")) or f"room_{index}"
        pipeline_request = _mapping(item.get("pipeline_run_request"))
        input_payload = _mapping(pipeline_request.get("input_payload"))
        if not input_payload:
            continue
        input_payload = deepcopy(input_payload)
        user_input = dict(_mapping(input_payload.get("user_input")))
        user_input["allow_generated_accessories"] = req.allow_generated_accessories
        user_input[
            "disable_generated_accessories"
        ] = not req.allow_generated_accessories
        input_payload["user_input"] = user_input
        room_case_id = (
            f"{base_case_id}_{index:02d}_{_safe_case_segment(room_id, 'room')}"
        )
        try:
            result = run_case(
                input_payload=input_payload,
                user_id=_string_or_none(pipeline_request.get("user_id"))
                or req.user_id
                or "normalize_run",
                description=_string_or_none(pipeline_request.get("description"))
                or req.description,
                special_notes=_string_or_none(pipeline_request.get("special_notes"))
                or req.special_notes,
                case_id=room_case_id,
            )
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        if result.get("error"):
            raise HTTPException(status_code=502, detail=str(result["error"]))
        try:
            enriched_options = _enriched_case_options(
                room_case_id,
                result.get("final_output"),
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        if selection_summary is None:
            selection_summary = _case_selection_summary(room_case_id)
        room_options.append((room_id, enriched_options))

    styled_payloads = [
        styled_payload
        for _, options in room_options
        for option in options
        if isinstance((styled_payload := option.get("styled_payload")), dict)
    ]
    catalog_index = _load_catalog_index(
        object_types=_collect_object_types(styled_payloads),
        catalog_item_ids=_collect_catalog_item_ids(styled_payloads),
    )
    transform = _mapping(normalized.get("transform"))
    option_count = max((len(options) for _, options in room_options), default=0)
    response_options: list[dict[str, Any]] = []
    for option_index in range(option_count):
        option_objects: list[dict[str, Any]] = []
        option_meta: dict[str, Any] | None = None
        for room_id, options in room_options:
            if not options:
                continue
            option = options[min(option_index, len(options) - 1)]
            if option_meta is None:
                option_meta = option
            styled_payload = option.get("styled_payload")
            if not isinstance(styled_payload, dict):
                continue
            option_objects.extend(
                _normalize_run_restored_objects(
                    coordinate_service=coordinate_service,
                    styled_payload=styled_payload,
                    catalog_index=catalog_index,
                    transform=transform,
                    room_id=room_id,
                )
            )
        if not option_objects:
            continue
        option_id = (
            _string_or_none((option_meta or {}).get("option_id"))
            or f"variant_{option_index + 1}"
        )
        layout_score = _number((option_meta or {}).get("layout_score"))
        coverage_ratio = _number((option_meta or {}).get("coverage_ratio"))
        response_options.append(
            {
                "optionId": option_id,
                "label": _string_or_none((option_meta or {}).get("label"))
                or f"Option {option_index + 1}",
                "layoutScore": int(layout_score) if layout_score is not None else None,
                "hardValid": (option_meta or {}).get("hard_valid")
                if isinstance((option_meta or {}).get("hard_valid"), bool)
                else None,
                "complete": (option_meta or {}).get("complete")
                if isinstance((option_meta or {}).get("complete"), bool)
                else None,
                "coverageRatio": coverage_ratio,
                "objects": option_objects,
                "openings": deepcopy(req.openings),
            }
        )

    selected_objects = response_options[0]["objects"] if response_options else []
    selected_option_id = (
        _string_or_none(response_options[0].get("optionId"))
        if response_options
        else None
    )
    return PipelineNormalizeRunResponse(
        objects=[
            PipelineNormalizeRunObject.model_validate(item) for item in selected_objects
        ],
        openings=deepcopy(req.openings),
        selectedOptionId=selected_option_id,
        options=[
            PipelineNormalizeRunOption.model_validate(item) for item in response_options
        ],
        selectionSummary=selection_summary,
    )


@router.post(
    "/run", response_model=PipelineRunResponse, summary="Start a new pipeline run"
)
def run_pipeline(
    background: BackgroundTasks,
    req: PipelineRunRequest = Body(..., openapi_examples=_PIPELINE_RUN_EXAMPLES),
) -> PipelineRunResponse:
    """
    Kick off the full AI design pipeline asynchronously.

    Returns immediately with a `case_id`. Poll `GET /pipeline/{case_id}/status` to track
    progress, and fetch the final output from `GET /pipeline/{case_id}/result` once the
    stage is `'done'`.

    Pipeline stages (in order): `queued` → `running` → `done` | `error`.

    ---

    ### `input_payload` structure

    | Field | Type | Required | Description |
    |---|---|---|---|
    | `tenant_id` | string | yes | Use `"demo_tenant"` for testing |
    | `user_input.description` | string | yes | Natural-language design brief |
    | `user_input.room_type` | string | yes | `"bedroom"`, `"living_room"`, `"office"`, `"dining_room"`, … |
    | `user_input.style` | string | yes | `"modern"`, `"scandinavian"`, `"japandi"`, `"classic"`, … |
    | `user_input.floor_area_m2` | float | yes | Room area in m² |
    | `user_input.height` | int | yes | Ceiling height in **mm** (e.g. `2800`) |
    | `user_input.shape_points` | `[{x, y}]` | yes | Room outline polygon, coordinates in **mm** |
    | `user_input.windows` | int | no | Number of windows (informational) |
    | `user_input.window_direction` | string | no | Compass direction the window faces |
    | `doors` | array | no | Door openings (position + swing info) |
    | `windows` | array | no | Window openings (position + clearance) |

    **Tip:** pick one of the pre-filled examples from the dropdown above to get started immediately.
    """
    case_id = _make_case_id(req.user_id)
    paths = case_paths(case_id)
    paths.root.mkdir(parents=True, exist_ok=True)

    _write_json(
        paths.status,
        {
            "case_id": case_id,
            "stage": "queued",
            "updated_at_utc": _now_utc_iso(),
        },
    )

    background.add_task(
        _run_in_thread,
        input_payload=req.input_payload,
        user_id=req.user_id,
        description=req.description,
        special_notes=req.special_notes,
        cases_root=str(paths.root.parent),
        case_id=case_id,
    )

    return PipelineRunResponse(
        case_id=case_id,
        case_dir=str(paths.root),
        status_path=str(paths.status),
        status="queued",
    )


@router.post(
    "/compile-snapshot-prompt",
    response_model=SnapshotPromptCompileResponse,
    summary="Compile a snapshot into an image-gen prompt",
)
def compile_snapshot_prompt_route(
    req: SnapshotPromptCompileRequest,
) -> SnapshotPromptCompileResponse:
    """
    Convert a snapshot layout (objects + room geometry) into the structured prompt payload
    that is sent to the image generation model.

    Accepts either an inline `snapshot_payload` dict or a server-side `snapshot_path`.
    Useful for debugging prompt construction without triggering a full render.
    """
    if req.snapshot_payload is None and not req.snapshot_path:
        raise HTTPException(
            status_code=400,
            detail="Provide either snapshot_payload or snapshot_path.",
        )

    try:
        if req.snapshot_payload is not None:
            compilation = compile_snapshot_prompt(req.snapshot_payload)
        else:
            compilation = compile_snapshot_prompt_from_path(req.snapshot_path or "")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return SnapshotPromptCompileResponse(compilation=compilation)


@router.post(
    "/render-snapshot-image",
    response_model=SnapshotImageRenderResponse,
    summary="Render a room image from a snapshot layout",
)
def render_snapshot_image_route(
    req: SnapshotImageRenderRequest,
    current_user: UserAccount | None = Depends(get_optional_current_user),
    user_content_service: UserContentService = Depends(get_user_content_service),
) -> SnapshotImageRenderResponse:
    """
    Generate or edit a photorealistic room image using AI.

    **Two input modes:**
    - **Inline** (`snapshot_payload` + `snapshot_image_data_url`): pass the layout and a
      current render image directly in the request body.
    - **File** (`snapshot_path`): reference a snapshot JSON already on the server.

    **Two render modes** (controlled by `render_mode`):
    - `generate`: produce a brand-new image from the layout and prompt.
    - `edit`: apply `edit_operations` (object swaps, colour changes) to `edit_source_image_data_url`.

    If the request is authenticated, the result is automatically saved to the user's render
    history and returned in `saved_render`.
    """
    trace_id = uuid4().hex
    log_image_flow_event(
        "snapshot_image.api_input",
        {
            "trace_id": trace_id,
            "route": "/pipeline/render-snapshot-image",
            "authenticated": current_user is not None,
            "request": req.model_dump(),
        },
    )
    if req.snapshot_payload is None and not req.snapshot_path:
        log_image_flow_event(
            "snapshot_image.api_error",
            {
                "trace_id": trace_id,
                "status_code": 400,
                "error_message": "Provide either snapshot_payload or snapshot_path.",
            },
        )
        raise HTTPException(
            status_code=400,
            detail="Provide either snapshot_payload or snapshot_path.",
        )

    try:
        edit_operations = [
            SnapshotEditOperation(
                object_id=item.object_id,
                object_name=item.object_name,
                replacement_image_data_url=item.replacement_image_data_url,
                target_color=item.target_color,
            )
            for item in req.edit_operations
        ]
        preset_selection = req.preset_selection.model_dump()
        if req.snapshot_payload is not None:
            if not req.snapshot_image_data_url:
                log_image_flow_event(
                    "snapshot_image.api_error",
                    {
                        "trace_id": trace_id,
                        "status_code": 400,
                        "error_message": (
                            "Provide snapshot_image_data_url when rendering from "
                            "snapshot_payload."
                        ),
                    },
                )
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Provide snapshot_image_data_url when rendering from "
                        "snapshot_payload."
                    ),
                )
            render = render_snapshot_image(
                req.snapshot_payload,
                snapshot_image_data_url=req.snapshot_image_data_url,
                layout_reference_image_data_url=req.layout_reference_image_data_url,
                annotated_reference_image_data_url=(
                    req.annotated_reference_image_data_url
                ),
                scene_reference_image_data_url=req.scene_reference_image_data_url,
                scene_reference_mode=req.scene_reference_mode,
                user_prompt=req.user_prompt,
                render_mode=req.render_mode,
                preset_selection=preset_selection,
                edit_operations=edit_operations,
                edit_source_image_data_url=req.edit_source_image_data_url,
                trace_id=trace_id,
            )
        else:
            render = render_snapshot_image_from_path(
                req.snapshot_path or "",
                snapshot_image_data_url=req.snapshot_image_data_url,
                layout_reference_image_data_url=req.layout_reference_image_data_url,
                annotated_reference_image_data_url=(
                    req.annotated_reference_image_data_url
                ),
                scene_reference_image_data_url=req.scene_reference_image_data_url,
                scene_reference_mode=req.scene_reference_mode,
                user_prompt=req.user_prompt,
                render_mode=req.render_mode,
                preset_selection=preset_selection,
                edit_operations=edit_operations,
                edit_source_image_data_url=req.edit_source_image_data_url,
                trace_id=trace_id,
            )
    except FileNotFoundError as exc:
        log_image_flow_event(
            "snapshot_image.api_error",
            {
                "trace_id": trace_id,
                "status_code": 404,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        log_image_flow_event(
            "snapshot_image.api_error",
            {
                "trace_id": trace_id,
                "status_code": 400,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        log_image_flow_event(
            "snapshot_image.api_error",
            {
                "trace_id": trace_id,
                "status_code": 502,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    saved_render: dict[str, Any] | None = None
    if current_user is not None:
        try:
            metadata = render.get("metadata") if isinstance(render, dict) else None
            metadata_payload = metadata if isinstance(metadata, dict) else {}
            saved = user_content_service.save_snapshot_render_from_data_url(
                user=current_user,
                image_data_url=str(render["image"]["data_url"]),
                model_name=str(render["models"]["image_model_name"]),
                prompt=str(render["request"]["user_prompt"]),
                negative_prompt=None,
                meta={
                    "user_prompt": str(render["request"]["user_prompt"]),
                    "render_mode": str(
                        render["request"].get("render_mode", "generate")
                    ),
                    "preset_selection": render["request"].get("preset_selection"),
                    "edit_operations": render["request"].get("edit_operations"),
                    "aspect_ratio": str(render["image"]["aspect_ratio"]),
                    "source_image_mime_type": str(
                        render["request"]["source_image_mime_type"]
                    ),
                    "camera": metadata_payload.get("camera"),
                    "visible_objects": metadata_payload.get("visible_objects"),
                    "visible_object_ids": metadata_payload.get("visible_object_ids"),
                    "layout_reference_enabled": bool(
                        render["request"]["layout_reference_enabled"]
                    ),
                    "layout_reference_used": bool(
                        render["request"]["layout_reference_used"]
                    ),
                    "annotated_reference_used": bool(
                        render["request"]["annotated_reference_used"]
                    ),
                    "scene_reference_mode": str(
                        render["request"].get("scene_reference_mode", "none")
                    ),
                    "scene_reference_used": bool(
                        render["request"].get("scene_reference_used", False)
                    ),
                    "reference_only_camera_transfer_used": bool(
                        render["request"].get(
                            "reference_only_camera_transfer_used",
                            False,
                        )
                    ),
                    "root_layout_id": req.root_layout_id,
                },
            )
            saved_render = user_content_service.serialize_generated_render(
                saved,
                file_url=f"/account/renders/{saved.id}/file",
            )
            log_image_flow_event(
                "snapshot_image.saved_render.output",
                {
                    "trace_id": trace_id,
                    "render_id": str(saved.id),
                    "model_name": saved.model_name,
                    "mime_type": saved.mime_type,
                    "storage_path": saved.storage_path,
                    "meta": dict(saved.meta or {}),
                },
            )
        except Exception as exc:
            log_image_flow_event(
                "snapshot_image.saved_render.error",
                {
                    "trace_id": trace_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            saved_render = None

    return SnapshotImageRenderResponse(render=render, saved_render=saved_render)


@router.get(
    "/render-presets",
    response_model=SnapshotImagePresetsResponse,
    summary="List available render style/light/scenery presets",
)
def get_render_presets() -> SnapshotImagePresetsResponse:
    """Return all preset options that can be passed in `preset_selection` when calling `POST /pipeline/render-snapshot-image`."""
    presets = load_config(config_file).presets
    return SnapshotImagePresetsResponse(
        styles={
            key: SnapshotImagePresetOptionResponse(**value.model_dump())
            for key, value in presets.styles.items()
        },
        lights={
            key: SnapshotImagePresetOptionResponse(**value.model_dump())
            for key, value in presets.lights.items()
        },
        sceneries={
            key: SnapshotImagePresetOptionResponse(**value.model_dump())
            for key, value in presets.sceneries.items()
        },
    )


@router.get(
    "/{case_id}/status",
    response_model=PipelineStatusResponse,
    summary="Poll the status of a pipeline run",
)
def get_status(case_id: str) -> PipelineStatusResponse:
    """
    Return the current execution state of a pipeline run.

    Recommended polling interval: every 2–5 seconds while `stage` is `'queued'` or `'running'`.
    Use `progress_current` / `progress_total` to render a progress bar.
    Raises `404` if the `case_id` is unknown.
    """
    paths = case_paths(case_id)
    if not paths.status.exists():
        raise HTTPException(status_code=404, detail="case not found")
    payload = json.loads(paths.status.read_text())
    return PipelineStatusResponse(
        case_id=payload.get("case_id", case_id),
        stage=payload.get("stage", "unknown"),
        updated_at_utc=payload.get("updated_at_utc", ""),
        error=payload.get("error"),
        message=payload.get("message"),
        progress_current=payload.get("progress_current"),
        progress_total=payload.get("progress_total"),
        actions=payload.get("actions")
        if isinstance(payload.get("actions"), list)
        else [],
    )


@router.get(
    "/{case_id}/result",
    response_model=PipelineResultResponse,
    summary="Get the final result of a completed pipeline run",
)
def get_result(case_id: str) -> PipelineResultResponse:
    """
    Return the styled layout output once the pipeline stage is `'done'`.

    The `result` object contains placed furniture objects with positions, rotations,
    dimensions, and style metadata. If layout variants were generated, `result.variants`
    holds all alternatives and `result.selected_variant_id` marks the recommended one.

    Raises `404` if no result exists yet (pipeline still running or errored without output).
    """
    paths = case_paths(case_id)
    if not paths.stylist.exists() and not paths.layout_variants.exists():
        raise HTTPException(status_code=404, detail="result not found")
    enriched: dict[str, Any] = {}
    if paths.stylist.exists():
        stylist_payload = json.loads(paths.stylist.read_text())
        absolute_payload: dict[str, Any] | None = None
        try:
            if paths.absolute_layout.exists():
                absolute_payload = json.loads(paths.absolute_layout.read_text())
        except Exception:
            absolute_payload = None

        enriched = _enrich_rotation_ccw(
            stylist_payload=stylist_payload,
            absolute_layout_payload=absolute_payload,
        )
    if paths.layout_variants.exists():
        variants_payload = json.loads(paths.layout_variants.read_text())
        if isinstance(variants_payload, dict):
            variants = variants_payload.get("variants")
            selected_variant_id = variants_payload.get("selected_variant_id")
            if isinstance(variants, list):
                enriched_variants: list[dict[str, Any]] = []
                for variant in variants:
                    if not isinstance(variant, dict):
                        continue
                    styled_variant = variant.get("styled_result")
                    absolute_variant = variant.get("absolute_layout")
                    next_variant = dict(variant)
                    if isinstance(styled_variant, dict):
                        next_variant["styled_result"] = _enrich_rotation_ccw(
                            stylist_payload=styled_variant,
                            absolute_layout_payload=absolute_variant
                            if isinstance(absolute_variant, dict)
                            else None,
                        )
                    enriched_variants.append(next_variant)
                if enriched_variants and not enriched:
                    first_variant = enriched_variants[0]
                    styled_result = first_variant.get("styled_result")
                    if isinstance(styled_result, dict):
                        enriched = dict(styled_result)
                enriched["variants"] = enriched_variants
            if isinstance(selected_variant_id, str) and selected_variant_id:
                enriched["selected_variant_id"] = selected_variant_id
    return PipelineResultResponse(case_id=case_id, result=enriched)


@router.get(
    "/{case_id}/artifact/{name}",
    response_model=ArtifactResponse,
    summary="Get a named intermediate artifact from a pipeline run",
)
def get_artifact(case_id: str, name: str) -> ArtifactResponse:
    """
    Fetch the raw JSON output of a specific pipeline module for debugging or inspection.

    Available artifact names: `module_io_manifest`, `room_interpreter`, `stylist_style_policy`,
    `cluster_forge`, `tier_count`, `tier_count_director`, `cluster_merged`,
    `cluster_output_merger`, `cluster_relation_plan`, `seed_concept_relation_plan`,
    `seed_concept_generator`, `cluster_placer`, `phase2_controller`,
    `macro_cluster_solver`, `macro_cluster_solver_dropped_inventory`,
    `absolute_layout`, `controlled_accessory_refill`, `stylist`,
    `cluster_outlines`, `cluster_outline_bundle`, `layout_variants`.

    Raises `404` if the artifact has not been produced yet or the `case_id` is unknown.
    """
    paths = case_paths(case_id)
    mapping = {
        "module_io_manifest": paths.module_io_manifest,
        "room_interpreter": paths.room_interpreter,
        "stylist_style_policy": paths.module_output("stylist_style_policy"),
        "cluster_forge": paths.cluster_forge,
        "tier_count": paths.tier_count,
        "tier_count_director": paths.module_output("tier_count_director"),
        "cluster_merged": paths.cluster_merged,
        "cluster_output_merger": paths.module_output("cluster_output_merger"),
        "cluster_relation_plan": paths.cluster_relation_plan,
        "seed_concept_relation_plan": paths.module_output("seed_concept_relation_plan"),
        "seed_concept_generator": paths.module_output("seed_concept_generator"),
        "cluster_placer": paths.cluster_placer,
        "phase2_controller": paths.module_output("phase2_controller"),
        "macro_cluster_solver": paths.module_output("macro_cluster_solver"),
        "macro_cluster_solver_dropped_inventory": paths.module_output(
            "macro_cluster_solver_dropped_inventory"
        ),
        "absolute_layout": paths.absolute_layout,
        "controlled_accessory_refill": paths.module_output(
            "controlled_accessory_refill"
        ),
        "stylist": paths.stylist,
        "cluster_outlines": paths.cluster_outlines_all,
        "cluster_outline_bundle": paths.module_output("cluster_outline_bundle"),
        "layout_variants": paths.layout_variants,
    }
    path = mapping.get(name)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="artifact not found")
    payload = json.loads(path.read_text())
    return ArtifactResponse(case_id=case_id, name=name, payload=payload)


@router.get(
    "/{case_id}/clusters",
    response_model=ClusterListResponse,
    summary="List furniture clusters in a pipeline run",
)
def list_clusters(case_id: str) -> ClusterListResponse:
    """
    Return the IDs of all furniture clusters that were composed during the pipeline run.

    A cluster groups related furniture items (e.g. a sofa + coffee table + rug).
    Use the returned IDs with `GET /pipeline/{case_id}/clusters/{cluster_id}`.
    """
    paths = case_paths(case_id)
    if not paths.clusters_dir.exists():
        raise HTTPException(status_code=404, detail="clusters not found")
    clusters = sorted(
        {
            p.name.replace("cluster_composer_", "").replace(".json", "")
            for p in paths.clusters_dir.glob("cluster_composer_*.json")
        }
    )
    return ClusterListResponse(case_id=case_id, clusters=clusters)


@router.get(
    "/{case_id}/clusters/{cluster_id}",
    response_model=ArtifactResponse,
    summary="Get cluster composer output for a specific cluster",
)
def get_cluster_composer(case_id: str, cluster_id: str) -> ArtifactResponse:
    """Return the full cluster_composer JSON for a single cluster (selected items, arrangement, relationships)."""
    paths = case_paths(case_id)
    path = paths.cluster_composer(cluster_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="cluster composer output not found")
    payload = json.loads(path.read_text())
    return ArtifactResponse(
        case_id=case_id, name=f"cluster_composer_{cluster_id}", payload=payload
    )


@router.get(
    "/{case_id}/clusters/{cluster_id}/outline",
    response_model=ArtifactResponse,
    summary="Get the spatial outline of a cluster",
)
def get_cluster_outline(case_id: str, cluster_id: str) -> ArtifactResponse:
    """Return the bounding outline (footprint polygon) of the specified cluster in room-local coordinates."""
    paths = case_paths(case_id)
    path = paths.cluster_outline(cluster_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="cluster outline not found")
    payload = json.loads(path.read_text())
    return ArtifactResponse(
        case_id=case_id, name=f"cluster_outline_{cluster_id}", payload=payload
    )
