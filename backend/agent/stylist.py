from __future__ import annotations

import json
import logging
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from agent.tool_call_parser import extract_tool_calls as parse_tool_calls
from agent_schema.stylist_schema import StylistOutput
from clients.llm_client import get_llm_client
from prompt.stylist import STYLIST_PROMPT
from prompt.system import SYSTEM_PROMPT
from stylist.deterministic_layout import (
    DEFAULT_OBJECT_COLOR,
    DEFAULT_SURFACES,
    build_deterministic_stylist_payload,
)
from stylist.room_essentials_seed import ROOM_SURFACE_GROUPS
from stylist.style_policy import (
    compile_style_policy as compile_layout_style_policy,
)
from stylist.style_policy import (
    final_style_plan_from_surface_payload,
)
from stylist.tools import TOOL_REGISTRY, TOOL_SCHEMAS

logger = logging.getLogger(__name__)

try:
    from config.llm_config import TextLLMConfig
except Exception:  # pragma: no cover
    TextLLMConfig = None  # type: ignore[assignment]


_STYLE_VARIANT_MODEL_NAMES = (
    "gemma-4-31b-it",
    "gemini-3.1-flash-lite-preview",
    "gemma-3-27b-it",
    "gemini-3-flash-preview",
    "gemma-4-31b-it",
)


def stylist_model_name_for_variant(variant_index: int) -> str:
    strict_model_name = _strict_single_model_name()
    if strict_model_name:
        return strict_model_name
    configured_model_name = _configured_agent_model_name("stylist")
    if configured_model_name:
        return configured_model_name
    safe_index = max(1, int(variant_index))
    return _STYLE_VARIANT_MODEL_NAMES[
        (safe_index - 1) % len(_STYLE_VARIANT_MODEL_NAMES)
    ]


@dataclass(frozen=True)
class Stylist:
    system_prompt: str = SYSTEM_PROMPT
    prompt_template: str = STYLIST_PROMPT

    def compile_style_policy(
        self,
        *,
        room_type: str,
        brief_text: str,
        room_model_json: dict[str, Any] | None = None,
        semantic_program_rules: dict[str, Any] | None = None,
        use_llm: bool = True,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        return compile_layout_style_policy(
            room_type=room_type,
            brief_text=brief_text,
            room_model_json=room_model_json,
            semantic_program_rules=semantic_program_rules,
            use_llm=use_llm,
            model_name=model_name,
        )

    def generate_style_plan(
        self,
        *,
        layout_json: dict[str, Any],
        user_context_json: dict[str, Any] | None = None,
        tenant_id: str | None = None,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        _, payload, _, style_context = self._prepare_style_inputs(
            layout_json=layout_json,
            user_context_json=user_context_json,
            tenant_id=tenant_id,
        )
        style_prompt = _build_color_prompt(
            template=self.prompt_template,
            style_context_json=style_context,
            locked_style_payload_json=_build_locked_style_payload(payload),
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": style_prompt},
        ]
        return _generate_style_plan(messages, model_name=model_name)

    def generate_raw(
        self,
        *,
        layout_json: dict[str, Any],
        user_context_json: dict[str, Any] | None = None,
        tenant_id: str | None = None,
    ) -> str:
        payload = self._build_output_payload(
            layout_json=layout_json,
            user_context_json=user_context_json,
            tenant_id=tenant_id,
        )
        return json.dumps(payload, ensure_ascii=True, indent=2)

    def generate(
        self,
        *,
        layout_json: dict[str, Any],
        user_context_json: dict[str, Any] | None = None,
        tenant_id: str | None = None,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        return self.apply_style_plan(
            layout_json=layout_json,
            user_context_json=user_context_json,
            tenant_id=tenant_id,
            model_name=model_name,
        )

    def apply_style_plan(
        self,
        *,
        layout_json: dict[str, Any],
        user_context_json: dict[str, Any] | None = None,
        tenant_id: str | None = None,
        style_plan: dict[str, Any] | None = None,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        normalized_layout, payload, _, style_context = self._prepare_style_inputs(
            layout_json=layout_json,
            user_context_json=user_context_json,
            tenant_id=tenant_id,
        )
        resolved_style_plan = (
            deepcopy(style_plan)
            if isinstance(style_plan, dict)
            else self.generate_style_plan(
                layout_json=normalized_layout,
                user_context_json=user_context_json,
                tenant_id=tenant_id,
                model_name=model_name,
            )
        )
        styled_payload = _apply_style_plan(
            payload=payload,
            style_plan=resolved_style_plan,
            style_context=style_context,
        )
        styled_payload = _fix_place_on_targets(
            styled_payload,
            layout_json=normalized_layout,
        )
        return StylistOutput.model_validate(styled_payload).model_dump()

    def _build_output_payload(
        self,
        *,
        layout_json: dict[str, Any],
        user_context_json: dict[str, Any] | None,
        tenant_id: str | None,
    ) -> dict[str, Any]:
        normalized_layout, payload, _, style_context = self._prepare_style_inputs(
            layout_json=layout_json,
            user_context_json=user_context_json,
            tenant_id=tenant_id,
        )
        style_prompt = _build_color_prompt(
            template=self.prompt_template,
            style_context_json=style_context,
            locked_style_payload_json=_build_locked_style_payload(payload),
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": style_prompt},
        ]
        style_plan = _generate_style_plan(messages)
        styled_payload = _apply_style_plan(
            payload=payload,
            style_plan=style_plan,
            style_context=style_context,
        )
        return styled_payload

    def _prepare_style_inputs(
        self,
        *,
        layout_json: dict[str, Any],
        user_context_json: dict[str, Any] | None,
        tenant_id: str | None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
        normalized_layout = _normalize_layout_json(layout_json)
        merged_user_context = _inject_tenant(user_context_json, tenant_id)
        resolved_tenant_id = _extract_tenant_id(merged_user_context, normalized_layout)
        payload = build_deterministic_stylist_payload(
            normalized_layout,
            tenant_id=resolved_tenant_id,
        )
        style_context = _build_style_context(
            layout_json=normalized_layout,
            user_context_json=merged_user_context,
            payload=payload,
        )
        return normalized_layout, payload, merged_user_context, style_context


def _build_prompt(
    *,
    template: str,
    layout_json: dict[str, Any],
    user_context_json: dict[str, Any] | None,
    surface_requirements_json: dict[str, Any],
) -> str:
    mapping = {
        "LAYOUT_JSON": _json_block(layout_json),
        "USER_CONTEXT_JSON": _json_block(user_context_json),
        "SURFACE_REQUIREMENTS_JSON": _json_block(surface_requirements_json),
    }
    output = template
    for key, value in mapping.items():
        output = output.replace("{" + key + "}", value)
    return output


def _build_color_prompt(
    *,
    template: str,
    style_context_json: dict[str, Any],
    locked_style_payload_json: dict[str, Any],
) -> str:
    mapping = {
        "STYLE_CONTEXT_JSON": _json_block(style_context_json),
        "LOCKED_STYLE_PAYLOAD_JSON": _json_block(locked_style_payload_json),
    }
    output = template
    for key, value in mapping.items():
        output = output.replace("{" + key + "}", value)
    return output


def _inject_tenant(
    user_context_json: dict[str, Any] | None,
    tenant_id: str | None,
) -> dict[str, Any] | None:
    if tenant_id is None:
        return user_context_json
    context = dict(user_context_json or {})
    context.setdefault("tenant_id", tenant_id)
    return context


def _json_block(obj: Any) -> str:
    if obj is None:
        return "null"
    return json.dumps(obj, ensure_ascii=True, indent=2)


def _normalize_layout_json(layout_json: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(layout_json, dict):
        return {}

    out = dict(layout_json)
    room = out.get("room")
    if not isinstance(room, dict):
        room = {}
    out["room"] = room

    if not isinstance(out.get("openings"), dict):
        openings = room.get("openings")
        if isinstance(openings, dict):
            out["openings"] = deepcopy(openings)

    objects = out.get("objects")
    if not isinstance(objects, list):
        return out

    normalized_objects: list[dict[str, Any]] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        row = dict(obj)
        obj_type = row.get("object_type")
        if not isinstance(obj_type, str) or not obj_type:
            obj_id = row.get("object_id")
            if isinstance(obj_id, str) and obj_id:
                row["object_type"] = obj_id
        if not isinstance(row.get("instance_id"), str) or not row.get("instance_id"):
            obj_id = row.get("object_id")
            if isinstance(obj_id, str) and obj_id:
                row["instance_id"] = obj_id
        normalized_objects.append(row)
    out["objects"] = normalized_objects
    return out


def _build_style_context(
    *,
    layout_json: dict[str, Any],
    user_context_json: dict[str, Any] | None,
    payload: dict[str, Any],
) -> dict[str, Any]:
    room = layout_json.get("room") if isinstance(layout_json.get("room"), dict) else {}
    notes = _extract_guidance_notes(user_context_json)
    payload_notes = payload.get("notes")
    if isinstance(payload_notes, list):
        for note in payload_notes:
            if isinstance(note, str) and note.strip() and note.strip() not in notes:
                notes.append(note.strip())

    objects_summary: list[dict[str, Any]] = []
    for row in payload.get("objects") or []:
        if not isinstance(row, dict):
            continue
        place_on = (
            row.get("place_on") if isinstance(row.get("place_on"), dict) else None
        )
        objects_summary.append(
            {
                "instance_id": row.get("instance_id"),
                "object_type": row.get("object_type"),
                "source": row.get("source"),
                "cluster_id": row.get("cluster_id"),
                "place_on": place_on,
            }
        )

    room_info = payload.get("room") if isinstance(payload.get("room"), dict) else {}
    openings = (
        room_info.get("openings") if isinstance(room_info.get("openings"), dict) else {}
    )
    return {
        "room_type": _extract_room_type(layout_json),
        "requested_style": _extract_requested_style(user_context_json, layout_json),
        "style_tags": _extract_style_tags(user_context_json, layout_json),
        "style_policy": _extract_style_policy(user_context_json, layout_json),
        "notes": notes[:8],
        "room_id": room_info.get("room_id") or room.get("room_id") or "room_1",
        "door_ids": [
            row.get("id")
            for row in openings.get("doors", [])
            if isinstance(row, dict) and isinstance(row.get("id"), str)
        ],
        "window_ids": [
            row.get("id")
            for row in openings.get("windows", [])
            if isinstance(row, dict) and isinstance(row.get("id"), str)
        ],
        "objects": objects_summary,
    }


def _extract_guidance_notes(
    user_context_json: dict[str, Any] | None,
) -> list[str]:
    if not isinstance(user_context_json, dict):
        return []

    notes: list[str] = []

    def _add(value: Any) -> None:
        if not isinstance(value, str):
            return
        text = value.strip()
        if text and text not in notes:
            notes.append(text)

    for key in ("guidance_text", "notes_text"):
        _add(user_context_json.get(key))

    for key in ("notes", "room_notes"):
        value = user_context_json.get(key)
        if isinstance(value, list):
            for item in value:
                _add(item)

    user_input = user_context_json.get("user_input")
    if isinstance(user_input, dict):
        for key in (
            "description",
            "special_description",
            "special_notes",
            "notes",
            "feng_shui",
        ):
            _add(user_input.get(key))

    return notes


def _extract_requested_style(
    user_context_json: dict[str, Any] | None,
    layout_json: dict[str, Any],
) -> str:
    if isinstance(user_context_json, dict):
        style = user_context_json.get("style")
        if isinstance(style, str) and style.strip():
            return style.strip()
        user_input = user_context_json.get("user_input")
        if isinstance(user_input, dict):
            style = user_input.get("style")
            if isinstance(style, str) and style.strip():
                return style.strip()

    room = layout_json.get("room") if isinstance(layout_json.get("room"), dict) else {}
    style = room.get("style")
    if isinstance(style, str) and style.strip():
        return style.strip()

    meta = layout_json.get("meta")
    if isinstance(meta, dict):
        style = meta.get("style")
        if isinstance(style, str) and style.strip():
            return style.strip()

    return ""


def _extract_style_policy(
    user_context_json: dict[str, Any] | None,
    layout_json: dict[str, Any],
) -> dict[str, Any] | None:
    if isinstance(user_context_json, dict):
        style_policy = user_context_json.get("style_policy")
        if isinstance(style_policy, dict):
            return deepcopy(style_policy)
        user_input = user_context_json.get("user_input")
        if isinstance(user_input, dict):
            style_policy = user_input.get("style_policy")
            if isinstance(style_policy, dict):
                return deepcopy(style_policy)
    style_policy = layout_json.get("style_policy")
    if isinstance(style_policy, dict):
        return deepcopy(style_policy)
    meta = layout_json.get("meta")
    if isinstance(meta, dict):
        style_policy = meta.get("style_policy")
        if isinstance(style_policy, dict):
            return deepcopy(style_policy)
    return None


def _build_locked_style_payload(payload: dict[str, Any]) -> dict[str, Any]:
    room = payload.get("room") if isinstance(payload.get("room"), dict) else {}
    objects_out: list[dict[str, Any]] = []
    for row in payload.get("objects") or []:
        if not isinstance(row, dict):
            continue
        objects_out.append(
            {
                "instance_id": row.get("instance_id"),
                "object_type": row.get("object_type"),
                "source": row.get("source"),
                "cluster_id": row.get("cluster_id"),
                "place_on": row.get("place_on"),
            }
        )

    return {
        "room": {
            "room_id": room.get("room_id"),
            "room_type": room.get("room_type"),
            "opening_ids": {
                "doors": [
                    item.get("id")
                    for item in (room.get("openings") or {}).get("doors", [])
                    if isinstance(item, dict) and isinstance(item.get("id"), str)
                ],
                "windows": [
                    item.get("id")
                    for item in (room.get("openings") or {}).get("windows", [])
                    if isinstance(item, dict) and isinstance(item.get("id"), str)
                ],
            },
        },
        "objects": objects_out,
    }


def _generate_style_plan(
    messages: list[dict[str, str]],
    *,
    model_name: str | None = None,
) -> dict[str, Any]:
    client = get_llm_client()
    attempted_models: list[str] = []
    last_retryable_error: Exception | None = None

    for candidate_model_name in _style_plan_model_attempt_order(model_name):
        attempted_models.append(candidate_model_name)
        candidate_messages = list(messages)
        try:
            for attempt in range(2):
                response = client.chat_completion(
                    candidate_messages,
                    model_key="primary",
                    model_name=candidate_model_name,
                    fallback_model_names=(),
                    temperature=0.35,
                    max_tokens=2200,
                    thinking_level="medium",
                    response_mime_type="application/json",
                )
                content = _extract_content_from_response(response)
                parsed = _try_parse_json_object(content)
                if parsed is not None and _is_style_plan_shape(parsed):
                    return parsed
                last_retryable_error = ValueError(
                    "Stylist returned invalid JSON: "
                    f"{_truncate_text(content, max_len=300)}"
                )
                if attempt == 1:
                    logger.warning(
                        "Stylist model %s returned invalid JSON after repair.",
                        candidate_model_name,
                    )
                    break
                _record_llm_retry(
                    stage="stylist.style_plan",
                    model_name=candidate_model_name,
                    reason="invalid_json_or_schema",
                )
                candidate_messages.append({"role": "assistant", "content": content})
                candidate_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Return exactly one valid JSON object matching the style "
                            "plan shape. No markdown. No prose outside JSON."
                        ),
                    }
                )
        except Exception as exc:
            if _is_retryable_model_unavailable_error(exc):
                last_retryable_error = exc
                logger.warning(
                    "Stylist model %s is temporarily unavailable; moving to the next configured model if available. Error: %s",
                    candidate_model_name,
                    exc,
                )
                continue
            logger.warning("Stylist color planner fallback triggered: %s", exc)
            return {}

    if last_retryable_error is not None:
        logger.warning(
            "Stylist color planner fallback triggered after exhausting failover models %s: %s",
            attempted_models,
            last_retryable_error,
        )
    return {}


def _style_plan_model_attempt_order(model_name: str | None) -> list[str]:
    strict_model_name = _strict_single_model_name()
    if strict_model_name:
        return [strict_model_name]
    configured_model_name = _configured_agent_model_name("stylist")
    requested_model_name = str(model_name or configured_model_name or "").strip()
    sequence = (
        list(_STYLE_VARIANT_MODEL_NAMES)
        if TextLLMConfig is None or TextLLMConfig.PROVIDER == "gemini"
        else []
    )
    if not requested_model_name:
        primary_model_name = (
            TextLLMConfig.primary_model_name() if TextLLMConfig is not None else ""
        )
        return sequence or ([primary_model_name] if primary_model_name else [])
    if requested_model_name in sequence:
        start_index = sequence.index(requested_model_name)
        return sequence[start_index:] + sequence[:start_index]
    return [
        requested_model_name,
        *[name for name in sequence if name != requested_model_name],
    ]


def _strict_single_model_name() -> str:
    if TextLLMConfig is None:
        return ""
    return str(getattr(TextLLMConfig, "STRICT_SINGLE_TEXT_MODEL", "") or "").strip()


def _configured_agent_model_name(key: str) -> str | None:
    if TextLLMConfig is None:
        return None
    return TextLLMConfig.agent_model(key)


def _record_llm_retry(*, stage: str, model_name: str | None, reason: str) -> None:
    if TextLLMConfig is None or getattr(TextLLMConfig, "PROVIDER", "") != "gemini":
        return
    try:
        from clients.gemini_client import GeminiClient

        GeminiClient.record_retry_event(
            stage=stage,
            model_name=model_name,
            reason=reason,
        )
    except Exception:
        logger.debug("Failed to record Gemini retry event.", exc_info=True)


def _is_retryable_model_unavailable_error(exc: Exception) -> bool:
    error_text = str(exc).lower()
    if not error_text:
        return False
    return (
        "503 service unavailable" in error_text
        or '"code": 503' in error_text
        or '"status": "unavailable"' in error_text
        or "currently experiencing high demand" in error_text
    )


def _apply_style_plan(
    *,
    payload: dict[str, Any],
    style_plan: dict[str, Any],
    style_context: dict[str, Any],
) -> dict[str, Any]:
    out = deepcopy(payload)
    palette = _fallback_palette(style_context)

    room = out.get("room")
    if isinstance(room, dict):
        room["surfaces"] = {
            "wall_color_hex": palette["wall"],
            "floor_color_hex": palette["floor"],
            "ceiling_color_hex": palette["ceiling"],
        }
        opening_colors = room.get("opening_colors")
        if not isinstance(opening_colors, dict):
            opening_colors = {"doors": [], "windows": []}
            room["opening_colors"] = opening_colors
        for item in opening_colors.get("doors", []):
            if isinstance(item, dict):
                item["color_hex"] = palette["trim"]
        for item in opening_colors.get("windows", []):
            if isinstance(item, dict):
                item["color_hex"] = palette["glass"]

    for row in out.get("objects") or []:
        if not isinstance(row, dict):
            continue
        object_type = str(row.get("object_type") or "")
        instance_id = str(row.get("instance_id") or object_type)
        row["color_hex"] = _sanitize_object_color(
            object_type=object_type,
            instance_id=instance_id,
            palette=palette,
            color_hex=_fallback_object_color(
                object_type=object_type,
                instance_id=instance_id,
                palette=palette,
            ),
        )
        material = row.get("material")
        if not isinstance(material, str) or not material.strip():
            row["material"] = _fallback_material_name(object_type)

    if isinstance(style_plan, dict):
        room_surfaces = style_plan.get("room_surfaces")
        if isinstance(room_surfaces, dict) and isinstance(room, dict):
            surfaces = room.get("surfaces")
            if not isinstance(surfaces, dict):
                surfaces = dict(DEFAULT_SURFACES)
                room["surfaces"] = surfaces
            for key in ("wall_color_hex", "floor_color_hex", "ceiling_color_hex"):
                value = room_surfaces.get(key)
                if _is_hex_color(value):
                    surfaces[key] = value

        opening_colors_plan = style_plan.get("opening_colors")
        if isinstance(opening_colors_plan, dict) and isinstance(room, dict):
            opening_colors = room.get("opening_colors")
            if not isinstance(opening_colors, dict):
                opening_colors = {"doors": [], "windows": []}
                room["opening_colors"] = opening_colors
            for key in ("doors", "windows"):
                plan_rows = opening_colors_plan.get(key)
                output_rows = opening_colors.get(key)
                if not isinstance(plan_rows, list) or not isinstance(output_rows, list):
                    continue
                by_id = {
                    item.get("id"): item
                    for item in output_rows
                    if isinstance(item, dict) and isinstance(item.get("id"), str)
                }
                for plan_row in plan_rows:
                    if not isinstance(plan_row, dict):
                        continue
                    opening_id = plan_row.get("id")
                    color_hex = plan_row.get("color_hex")
                    if (
                        isinstance(opening_id, str)
                        and opening_id in by_id
                        and _is_hex_color(color_hex)
                    ):
                        by_id[opening_id]["color_hex"] = color_hex

        object_styles = style_plan.get("object_styles")
        if isinstance(object_styles, list):
            style_by_id = {
                row.get("instance_id"): row
                for row in object_styles
                if isinstance(row, dict) and isinstance(row.get("instance_id"), str)
            }
            for row in out.get("objects") or []:
                if not isinstance(row, dict):
                    continue
                object_type = str(row.get("object_type") or "")
                instance_id = row.get("instance_id")
                if not isinstance(instance_id, str):
                    continue
                planned = style_by_id.get(instance_id)
                if not isinstance(planned, dict):
                    continue
                color_hex = planned.get("color_hex")
                if _is_hex_color(color_hex):
                    row["color_hex"] = _sanitize_object_color(
                        object_type=object_type,
                        instance_id=instance_id,
                        palette=palette,
                        color_hex=color_hex,
                    )
                material = planned.get("material")
                if material is None:
                    row["material"] = None
                elif isinstance(material, str) and material.strip():
                    row["material"] = material.strip()

        combined_notes = list(out.get("notes") or [])
        style_notes = style_plan.get("notes")
        if not isinstance(style_notes, list):
            style_notes = []
        for note in style_notes:
            if (
                isinstance(note, str)
                and note.strip()
                and note.strip() not in combined_notes
            ):
                combined_notes.append(note.strip())
        out["notes"] = combined_notes[:6]

    style_policy = style_context.get("style_policy")
    out["final_style_plan"] = final_style_plan_from_surface_payload(
        style_policy=style_policy if isinstance(style_policy, dict) else None,
        style_plan=style_plan if isinstance(style_plan, dict) else {},
        styled_payload=out,
    )
    return out


def _fallback_palette(style_context: dict[str, Any]) -> dict[str, str]:
    style_text = " ".join(
        [
            str(style_context.get("requested_style") or ""),
            " ".join(
                item
                for item in style_context.get("style_tags", [])
                if isinstance(item, str)
            ),
            " ".join(
                item for item in style_context.get("notes", []) if isinstance(item, str)
            ),
        ]
    ).lower()

    if "japandi" in style_text:
        return {
            "wall": "#F1E8DC",
            "floor": "#B98A61",
            "ceiling": "#FBF5EC",
            "trim": "#8F7259",
            "glass": "#DDE7F0",
            "large": "#B79A7C",
            "soft": "#DCC9B6",
            "accent": "#A46D3F",
            "accent_alt": "#7E8A70",
            "metal": "#5B5651",
            "dark": "#4B4038",
        }
    if "industrial" in style_text:
        return {
            "wall": "#E1DDD7",
            "floor": "#8B6B4F",
            "ceiling": "#F4F1EC",
            "trim": "#4D4D4D",
            "glass": "#C6D4DE",
            "large": "#6D726E",
            "soft": "#B8ADA0",
            "accent": "#B35C3B",
            "accent_alt": "#6C7A7A",
            "metal": "#4D4D4D",
            "dark": "#2E2E2E",
        }
    if "boho" in style_text:
        return {
            "wall": "#F5E6D4",
            "floor": "#B57E50",
            "ceiling": "#FBF5EE",
            "trim": "#99613D",
            "glass": "#D7E4EA",
            "large": "#B48A68",
            "soft": "#D6B79D",
            "accent": "#C46849",
            "accent_alt": "#7B8F65",
            "metal": "#7D665A",
            "dark": "#5B4336",
        }
    if "minimal" in style_text or "scandinav" in style_text:
        return {
            "wall": "#F6F2EC",
            "floor": "#C6A47A",
            "ceiling": "#FCFAF6",
            "trim": "#8D8C84",
            "glass": "#E1EAF0",
            "large": "#C2B8AA",
            "soft": "#DDD6CB",
            "accent": "#8F9C9A",
            "accent_alt": "#BCA08A",
            "metal": "#76736E",
            "dark": "#55514C",
        }
    return {
        "wall": "#F2ECE4",
        "floor": "#BC926A",
        "ceiling": "#FBF7F2",
        "trim": "#8A715B",
        "glass": "#D9E3EC",
        "large": "#B9A690",
        "soft": "#D7C8B8",
        "accent": "#8F7A6A",
        "accent_alt": "#7F8E86",
        "metal": "#6F6962",
        "dark": "#4F463F",
    }


def _fallback_object_color(
    *,
    object_type: str,
    instance_id: str,
    palette: dict[str, str],
) -> str:
    lowered = object_type.lower()
    accent = _pick_palette_color(
        [palette["accent"], palette["accent_alt"], palette["soft"]],
        seed=instance_id,
    )
    if any(token in lowered for token in ("rug", "curtain", "blanket", "cushion")):
        return accent
    if "plant" in lowered:
        return palette["accent_alt"]
    if any(
        token in lowered
        for token in (
            "lamp",
            "speaker",
            "clock",
            "desktop",
            "printer",
            "monitor",
            "tv",
            "air_conditioner",
            "appliance",
            "cooktop",
            "oven",
            "microwave",
            "toaster",
            "blender",
            "coffee_machine",
        )
    ):
        return palette["metal"]
    if any(token in lowered for token in ("mirror", "window", "screen")):
        return palette["glass"]
    if any(token in lowered for token in ("decor", "vase")):
        return accent
    if any(
        token in lowered
        for token in (
            "chair",
            "armchair",
            "sofa",
            "bed",
            "bench",
            "bean_bag",
            "ottoman",
        )
    ):
        return _pick_palette_color(
            [palette["large"], palette["soft"], accent],
            seed=instance_id,
        )
    if any(
        token in lowered
        for token in (
            "desk",
            "table",
            "dresser",
            "wardrobe",
            "cabinet",
            "bookshelf",
            "console",
            "nightstand",
            "storage",
        )
    ):
        return _pick_palette_color(
            [palette["large"], palette["dark"]],
            seed=instance_id,
        )
    return DEFAULT_OBJECT_COLOR


def _pick_palette_color(options: list[str], *, seed: str) -> str:
    if not options:
        return DEFAULT_OBJECT_COLOR
    index = sum(ord(ch) for ch in seed) % len(options)
    return options[index]


def _sanitize_object_color(
    *,
    object_type: str,
    instance_id: str,
    palette: dict[str, str],
    color_hex: Any,
) -> str:
    if _is_hex_color(color_hex) and not _is_near_black_hex(color_hex):
        return color_hex
    return _fallback_object_color(
        object_type=object_type,
        instance_id=instance_id,
        palette=palette,
    )


def _is_hex_color(value: Any) -> bool:
    return isinstance(value, str) and bool(re.match(r"^#[0-9A-Fa-f]{6}$", value))


def _is_near_black_hex(value: str) -> bool:
    if not _is_hex_color(value):
        return False
    red = int(value[1:3], 16)
    green = int(value[3:5], 16)
    blue = int(value[5:7], 16)
    return max(red, green, blue) <= 36 or (red + green + blue) <= 120


def _extract_content_from_response(response: object) -> str:
    choices = getattr(response, "choices", None)
    if isinstance(choices, list) and choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
    raise ValueError("Stylist color planner response missing message content")


def _fallback_material_name(object_type: str) -> str:
    lowered = object_type.lower()
    if any(
        token in lowered
        for token in (
            "rug",
            "curtain",
            "blanket",
            "cushion",
            "bed",
            "sofa",
            "chair",
            "armchair",
            "bean_bag",
        )
    ):
        return "fabric"
    if any(
        token in lowered
        for token in (
            "lamp",
            "clock",
            "speaker",
            "air_conditioner",
            "monitor",
            "laptop",
            "desktop",
            "printer",
            "appliance",
            "microwave",
            "oven",
            "toaster",
            "blender",
            "hood",
            "cooktop",
        )
    ):
        return "metal"
    if any(token in lowered for token in ("mirror", "window", "glass")):
        return "glass"
    if any(token in lowered for token in ("vase", "plant", "decor")):
        return "ceramic"
    return "wood"


def _run_with_tools(
    messages: list[dict[str, Any]],
    *,
    layout_json: dict[str, Any],
    user_context_json: dict[str, Any] | None,
    max_steps: int = 10,
) -> str:
    client = get_llm_client()

    surface_requirements = _compute_surface_requirements(layout_json)
    required_types = surface_requirements.get("required_types") or []
    candidate_types = surface_requirements.get("candidate_types") or []
    logger.info(
        "Stylist surface requirements: room_type=%s required_types=%s candidate_types=%s",
        surface_requirements.get("room_type"),
        required_types,
        candidate_types,
    )

    used_surface_groups = False
    used_inventory = False
    used_knowledge = False
    forced_tools = False

    inventory_items: list[dict[str, Any]] = []
    best_payload: dict[str, Any] | None = None
    best_rank: int | None = None
    best_object_count = 0

    for step in range(1, max_steps + 1):
        logger.info("Stylist step %s/%s", step, max_steps)

        try:
            response = client.chat_completion(
                messages,
                model_key="primary",
                model_name=(
                    TextLLMConfig.agent_model("stylist")
                    if TextLLMConfig is not None
                    else None
                ),
                temperature=0.0,
                thinking_level="medium",
                tools=TOOL_SCHEMAS,
            )
        except Exception as exc:
            if _is_context_length_exceeded(exc):
                logger.warning(
                    "Stylist stopped early due to context length exceeded.",
                    exc_info=exc,
                )
                if best_payload is not None:
                    return json.dumps(best_payload, ensure_ascii=True)
                return json.dumps(
                    {
                        "status": "NEED_INFO",
                        "room": None,
                        "objects": [],
                        "notes": [
                            "Stopped due to context length exceeded before producing a final result."
                        ],
                        "missing": ["context_length_exceeded"],
                    },
                    ensure_ascii=True,
                )
            raise
        message = _extract_message(response)
        tool_calls = _extract_tool_calls(message)

        if tool_calls:
            logger.info(
                "Tool calls requested: %s",
                [(c.get("function", {}) or {}).get("name") for c in tool_calls],
            )

            messages.append(
                {
                    "role": "assistant",
                    "content": getattr(message, "content", "") or "",
                    "tool_calls": tool_calls,
                }
            )

            for idx, call in enumerate(tool_calls):
                call_id = call.get("id") or f"tool_{idx}"
                fn = call.get("function", {}) or {}
                name = fn.get("name")
                args_text = fn.get("arguments") or "{}"
                args = _safe_json_loads(args_text)

                args = _coerce_tool_args(
                    name=name,
                    args=args,
                    layout_json=layout_json,
                    user_context_json=user_context_json,
                    surface_requirements=surface_requirements,
                )

                tool_output = _safe_run_tool(name, args)
                logger.info("Tool %s args: %s", name, _short_json(args))
                logger.info("Tool %s output: %s", name, _short_json(tool_output))

                if name == "GetRoomSurfaceGroups":
                    used_surface_groups = True

                if name == "ListInventoryByTypes":
                    used_inventory = True
                    items = tool_output.get("items")
                    if isinstance(items, list):
                        inventory_items = [i for i in items if isinstance(i, dict)]

                if name == "ListDesignKnowledge":
                    used_knowledge = True

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": str(name),
                        "content": json.dumps(tool_output, ensure_ascii=True),
                    }
                )

            continue

        content = getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            messages.append(
                {
                    "role": "user",
                    "content": "Return valid JSON only. Do not return prose or markdown.",
                }
            )
            continue

        logger.info("Assistant content length: %s", len(content))
        logger.info("Assistant content preview: %s", _truncate_text(content, 500))

        if (
            not (used_surface_groups and used_inventory and used_knowledge)
            and not forced_tools
        ):
            forced_tools = True
            logger.info(
                "Forcing required tools. used_surface_groups=%s used_inventory=%s used_knowledge=%s",
                used_surface_groups,
                used_inventory,
                used_knowledge,
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You MUST call GetRoomSurfaceGroups(room_type), "
                        "ListInventoryByTypes(tenant_id, types[]) and "
                        "ListDesignKnowledge(tenant_id, tags, category) before final output."
                    ),
                }
            )
            continue

        draft = _try_parse_json_object(content)
        if not isinstance(draft, dict):
            messages.append(
                {
                    "role": "user",
                    "content": "Return valid JSON only. Do not include markdown fences.",
                }
            )
            continue

        status = draft.get("status")
        draft_eval = draft
        if status == "OK" and inventory_items:
            draft_eval = _snap_inventory_sizes(dict(draft), inventory_items)

        if status in {"OK", "NEED_INFO", "UNSAT"}:
            candidate = dict(draft_eval)
            if status == "OK":
                missing_required = _validate_required_types(draft_eval, required_types)
                size_issues = (
                    _validate_inventory_sizes(draft_eval, inventory_items)
                    if inventory_items
                    else []
                )
                if size_issues and inventory_items:
                    forced = _force_snap_inventory_sizes(
                        dict(draft_eval), inventory_items
                    )
                    forced_issues = _validate_inventory_sizes(forced, inventory_items)
                    if not forced_issues:
                        draft_eval = forced
                        candidate = dict(draft_eval)
                        size_issues = []
                if missing_required or size_issues:
                    candidate["status"] = "NEED_INFO"
                    notes = candidate.get("notes")
                    if not isinstance(notes, list):
                        notes = []
                    notes.append(
                        "Partial draft does not satisfy required_types and/or inventory sizing; downgraded to NEED_INFO."
                    )
                    candidate["notes"] = notes

                    missing = candidate.get("missing")
                    if not isinstance(missing, list):
                        missing = []
                    if missing_required:
                        missing.extend([f"required_type:{t}" for t in missing_required])
                    if size_issues:
                        missing.append("inventory_size_mismatch")
                    candidate["missing"] = missing

                    rank = _stylist_status_rank("NEED_INFO")
                else:
                    rank = _stylist_status_rank("OK")
            else:
                rank = _stylist_status_rank(str(status))
            objects = candidate.get("objects")
            obj_count = len(objects) if isinstance(objects, list) else 0
            if (
                best_rank is None
                or rank < best_rank
                or (rank == best_rank and obj_count > best_object_count)
            ):
                best_payload = candidate
                best_rank = rank
                best_object_count = obj_count

        logger.info(
            "Stylist draft status=%s objects=%s missing=%s",
            status,
            len(draft.get("objects") or []),
            draft.get("missing"),
        )
        if status not in {"OK", "NEED_INFO", "UNSAT"}:
            messages.append(
                {
                    "role": "user",
                    "content": "Top-level field status must be one of OK, NEED_INFO, UNSAT.",
                }
            )
            continue

        if status == "OK":
            missing_required = _validate_required_types(draft_eval, required_types)
            if missing_required:
                logger.warning(
                    "Stylist final rejected: missing required types = %s",
                    missing_required,
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "SURFACE_REQUIREMENTS_JSON.required_types still has missing types: "
                            f"{missing_required}. "
                            "Add at least one instance for each missing type, or return NEED_INFO/UNSAT if legal placement is impossible."
                        ),
                    }
                )
                continue

            if inventory_items:
                size_issues = _validate_inventory_sizes(draft_eval, inventory_items)
                if size_issues:
                    forced = _force_snap_inventory_sizes(
                        dict(draft_eval), inventory_items
                    )
                    forced_issues = _validate_inventory_sizes(forced, inventory_items)
                    if not forced_issues:
                        draft_eval = forced
                        return json.dumps(draft_eval, ensure_ascii=True)
                    logger.warning(
                        "Stylist final rejected: inventory size issues = %s",
                        size_issues,
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Inventory-sourced objects must match the dimensions of at least one inventory item of the same type "
                                "(rotation swap allowed). Fix these objects: "
                                + "; ".join(size_issues)
                            ),
                        }
                    )
                    continue

        if status == "OK":
            return json.dumps(draft_eval, ensure_ascii=True)
        return content

    if best_payload is not None:
        if inventory_items and isinstance(best_payload, dict):
            forced = _force_snap_inventory_sizes(dict(best_payload), inventory_items)
            forced_issues = _validate_inventory_sizes(forced, inventory_items)
            if not forced_issues:
                best_payload = forced
                if isinstance(best_payload.get("missing"), list):
                    best_payload["missing"] = [
                        x
                        for x in best_payload["missing"]
                        if x != "inventory_size_mismatch"
                    ]
                best_payload["status"] = "OK"
        logger.warning(
            "Stylist hit max_steps; returning best partial payload with status=%s",
            best_payload.get("status"),
        )
        return json.dumps(best_payload, ensure_ascii=True)

    logger.warning("Stylist hit max_steps with no usable draft.")
    return json.dumps(
        {
            "status": "NEED_INFO",
            "room": None,
            "objects": [],
            "notes": ["Stopped due to max_steps with no usable draft."],
            "missing": ["max_steps_exceeded"],
        },
        ensure_ascii=True,
    )


def _coerce_tool_args(
    *,
    name: str | None,
    args: dict[str, Any],
    layout_json: dict[str, Any],
    user_context_json: dict[str, Any] | None,
    surface_requirements: dict[str, Any],
) -> dict[str, Any]:
    out = dict(args)

    room_type = _extract_room_type(layout_json)
    tenant_id = _extract_tenant_id(user_context_json, layout_json)

    if name == "GetRoomSurfaceGroups":
        out["room_type"] = room_type

    elif name == "ListInventoryByTypes":
        out["tenant_id"] = tenant_id
        requested_types = out.get("types")
        if not isinstance(requested_types, list) or not requested_types:
            requested_types = surface_requirements.get("candidate_types") or []
        normalized = []
        for t in requested_types:
            if isinstance(t, str) and t and t not in normalized:
                normalized.append(t)
        out["types"] = normalized

    elif name == "ListDesignKnowledge":
        out["tenant_id"] = tenant_id

        # IMPORTANT: DesignKnowledgeFilter uses "tags @> <tags>" (contains all tags).
        # To avoid over-filtering, we only filter by room_type as requested.
        tags: list[str] = ["styling"]
        if room_type and room_type != "unknown":
            tags.append(room_type)

        out["tags"] = tags
        out["category"] = None

    return out


def _compute_surface_requirements(layout_json: dict[str, Any]) -> dict[str, Any]:
    room_type = _extract_room_type(layout_json)
    groups = ROOM_SURFACE_GROUPS.get(room_type, {})

    no_stack = groups.get("no_stack") or []
    can_stack_map = groups.get("can_stack_or_be_stacked_or_hang_or_soft") or {}

    base_present: set[str] = set()
    for obj in layout_json.get("objects") or []:
        if not isinstance(obj, dict):
            continue
        obj_type = obj.get("object_type")
        if isinstance(obj_type, str) and obj_type in no_stack:
            base_present.add(obj_type)

    active_anchors: list[str] = []
    for base in sorted(base_present):
        if base in can_stack_map:
            active_anchors.append(base)

    openings = layout_json.get("openings") or {}
    has_openings = bool(
        (openings.get("doors") or []) or (openings.get("windows") or [])
    )
    if has_openings and "__opening__" in can_stack_map:
        active_anchors.append("__opening__")
    if "__wall__" in can_stack_map:
        active_anchors.append("__wall__")
    if "__ceiling__" in can_stack_map:
        active_anchors.append("__ceiling__")
    if "__utility_zone__" in can_stack_map:
        active_anchors.append("__utility_zone__")

    required_types: list[str] = []
    candidate_types: list[str] = []

    for anchor in active_anchors:
        spec = can_stack_map.get(anchor)

        # Backward-compatible mode:
        # - list[str] => candidate only
        # - dict => may contain required / optional / recommended
        if isinstance(spec, list):
            for item in spec:
                if isinstance(item, str):
                    candidate_types.append(item)
            continue

        if isinstance(spec, dict):
            for item in spec.get("required", []) or []:
                if isinstance(item, str):
                    required_types.append(item)
                    candidate_types.append(item)

            for key in ("optional", "recommended", "candidates", "items"):
                for item in spec.get(key, []) or []:
                    if isinstance(item, str):
                        candidate_types.append(item)

    required_types = _stable_unique(required_types)
    candidate_types = _stable_unique(candidate_types)

    return {
        "room_type": room_type,
        "base_present": sorted(base_present),
        "active_anchors": active_anchors,
        "required_types": required_types,
        "candidate_types": candidate_types,
    }


def _extract_message(response: object) -> object:
    choices = getattr(response, "choices", None)
    if isinstance(choices, list) and choices:
        return getattr(choices[0], "message", None)
    raise ValueError("OpenAI response missing message")


def _extract_tool_calls(message: object) -> list[dict[str, Any]]:
    return parse_tool_calls(message)


def _safe_json_loads(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _safe_run_tool(name: str | None, args: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(name, str) or name not in TOOL_REGISTRY:
        return {"error": "unknown_tool", "tool": name, "args": args}
    try:
        return TOOL_REGISTRY[name](**args)
    except Exception as exc:
        return {
            "error": "tool_failed",
            "tool": name,
            "message": str(exc),
            "args": args,
        }


def _parse_json(raw: str) -> dict[str, Any]:
    text = _coerce_json_text(raw)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        text = _extract_json_object(text)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("Stylist returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Stylist response must be a JSON object")
    return payload


def _coerce_json_text(raw: str) -> str:
    text = raw.strip()
    if not text.startswith("```"):
        return text
    parts = text.split("```")
    for idx in range(1, len(parts), 2):
        candidate = parts[idx].strip()
        if candidate.startswith("json"):
            candidate = candidate[4:].lstrip("\n").strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            return candidate
    return text


def _truncate_text(text: str, max_len: int = 500) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _short_json(obj: Any, max_len: int = 1200) -> str:
    try:
        dumped = json.dumps(obj, ensure_ascii=True)
    except Exception:
        dumped = str(obj)
    return _truncate_text(dumped, max_len=max_len)


def _is_context_length_exceeded(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    if code == "context_length_exceeded":
        return True
    text = str(exc)
    return ("context_length_exceeded" in text) or ("maximum context length" in text)


def _stylist_status_rank(status: str) -> int:
    return {"OK": 0, "NEED_INFO": 1, "UNSAT": 2}.get(status, 3)


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return text
    return text[start : end + 1]


def _extract_room_type(layout_json: dict[str, Any]) -> str:
    if not isinstance(layout_json, dict):
        return "unknown"
    room = layout_json.get("room") or {}
    return str(room.get("room_type") or "unknown")


def _extract_tenant_id(
    user_context_json: dict[str, Any] | None,
    layout_json: dict[str, Any],
) -> str | None:
    if isinstance(user_context_json, dict):
        tenant_id = user_context_json.get("tenant_id")
        if isinstance(tenant_id, str) and tenant_id.strip():
            return tenant_id.strip()

    tenant_id = layout_json.get("tenant_id")
    if isinstance(tenant_id, str) and tenant_id.strip():
        return tenant_id.strip()

    room = layout_json.get("room") or {}
    tenant_id = room.get("tenant_id")
    if isinstance(tenant_id, str) and tenant_id.strip():
        return tenant_id.strip()

    return None


def _extract_style_tags(
    user_context_json: dict[str, Any] | None,
    layout_json: dict[str, Any],
) -> list[str]:
    tags: list[str] = []

    def _add(value: Any) -> None:
        if isinstance(value, str) and value.strip() and value not in tags:
            tags.append(value.strip())

    if isinstance(user_context_json, dict):
        style = user_context_json.get("style")
        _add(style)

        preferred = user_context_json.get("style_tags")
        if isinstance(preferred, list):
            for item in preferred:
                _add(item)

        user_input = user_context_json.get("user_input")
        if isinstance(user_input, dict):
            _add(user_input.get("style"))
            style_tags = user_input.get("style_tags")
            if isinstance(style_tags, list):
                for item in style_tags:
                    _add(item)

    room = layout_json.get("room") or {}
    _add(room.get("style"))

    meta = layout_json.get("meta") or {}
    if isinstance(meta, dict):
        _add(meta.get("style"))
        style_tags = meta.get("style_tags")
        if isinstance(style_tags, list):
            for item in style_tags:
                _add(item)

    return tags


def _validate_required_types(
    draft: dict[str, Any],
    required_types: list[str],
) -> list[str]:
    if not required_types:
        return []

    objects = draft.get("objects") or []
    present: set[str] = set()
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        obj_type = obj.get("object_type")
        if isinstance(obj_type, str):
            present.add(obj_type)

    return [t for t in required_types if t not in present]


def _snap_inventory_sizes(
    draft: dict[str, Any],
    inventory_items: list[dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(draft, dict):
        return draft

    objects = draft.get("objects")
    if not isinstance(objects, list):
        return draft

    def _to_int_mm(value: Any) -> int | None:
        if isinstance(value, (int, float)):
            return int(round(float(value)))
        return None

    dims_by_type: dict[str, list[tuple[int, int, int]]] = {}
    for item in inventory_items:
        obj_type = item.get("type")
        if not isinstance(obj_type, str) or not obj_type:
            continue
        dims = item.get("dimensions") or {}
        w = None
        h = None
        hh = None
        if isinstance(dims, dict):
            w = _to_int_mm(dims.get("length_mm"))
            h = _to_int_mm(dims.get("width_mm"))
            hh = _to_int_mm(dims.get("height_mm"))
        if w is None or h is None:
            w = _to_int_mm(item.get("length_mm"))
            h = _to_int_mm(item.get("width_mm"))
        if hh is None:
            hh = _to_int_mm(item.get("height_mm"))
        if isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
            dims_by_type.setdefault(obj_type.lower(), []).append((w, h, hh or 0))

    if not dims_by_type:
        return draft

    def _bbox_from_poly(poly: list[dict[str, Any]]) -> tuple[int, int, int, int] | None:
        if not poly:
            return None
        xs, ys = [], []
        for pt in poly:
            try:
                xs.append(int(pt.get("x")))
                ys.append(int(pt.get("y")))
            except Exception:
                return None
        if not xs or not ys:
            return None
        return min(xs), min(ys), max(xs), max(ys)

    def _rect_poly(
        min_x: int, min_y: int, max_x: int, max_y: int
    ) -> list[dict[str, int]]:
        return [
            {"x": min_x, "y": min_y},
            {"x": max_x, "y": min_y},
            {"x": max_x, "y": max_y},
            {"x": min_x, "y": max_y},
        ]

    out_objects: list[dict[str, Any]] = []
    adjusted = 0
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        row = dict(obj)
        if row.get("source") != "inventory":
            out_objects.append(row)
            continue

        obj_type = row.get("object_type")
        if not isinstance(obj_type, str):
            out_objects.append(row)
            continue
        dims_list = dims_by_type.get(obj_type.lower()) or []
        if not dims_list:
            out_objects.append(row)
            continue

        bbox = row.get("bbox")
        if not isinstance(bbox, dict):
            poly = row.get("polygon_ccw") or []
            bbox_vals = _bbox_from_poly(poly)
            if bbox_vals is None:
                out_objects.append(row)
                continue
            min_x, min_y, max_x, max_y = bbox_vals
        else:
            try:
                min_x = int(bbox.get("min_x"))
                min_y = int(bbox.get("min_y"))
                max_x = int(bbox.get("max_x"))
                max_y = int(bbox.get("max_y"))
            except Exception:
                out_objects.append(row)
                continue

        cur_w = max_x - min_x
        cur_h = max_y - min_y
        if cur_w <= 0 or cur_h <= 0:
            out_objects.append(row)
            continue

        place_on = row.get("place_on") if isinstance(row.get("place_on"), dict) else {}
        method = str(place_on.get("method") or "floor")

        def _pairs_for_dims(
            L: int, W: int, H: int, *, method: str
        ) -> list[tuple[int, int]]:
            if method in {"floor", "on_top"}:
                return [(L, W)]
            pairs: list[tuple[int, int]] = []
            if W > 0 and H > 0:
                pairs.append((W, H))
            if L > 0 and W > 0:
                pairs.append((L, W))
            if L > 0 and H > 0:
                pairs.append((L, H))
            return pairs

        best = None
        best_score = None
        for w, h, hh in dims_list:
            for cand_w, cand_h in _pairs_for_dims(w, h, hh, method=method):
                score = abs(cur_w - cand_w) + abs(cur_h - cand_h)
                if best_score is None or score < best_score:
                    best_score = score
                    best = (cand_w, cand_h)

        if best is None:
            out_objects.append(row)
            continue

        new_w, new_h = best
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0
        nmin_x = int(round(cx - new_w / 2.0))
        nmax_x = int(round(cx + new_w / 2.0))
        nmin_y = int(round(cy - new_h / 2.0))
        nmax_y = int(round(cy + new_h / 2.0))

        row["bbox"] = {
            "min_x": nmin_x,
            "min_y": nmin_y,
            "max_x": nmax_x,
            "max_y": nmax_y,
        }
        row["polygon_ccw"] = _rect_poly(nmin_x, nmin_y, nmax_x, nmax_y)
        if cur_w != new_w or cur_h != new_h:
            adjusted += 1
        out_objects.append(row)

    draft["objects"] = out_objects
    if adjusted:
        logger.info("Stylist snapped inventory sizes for %s object(s).", adjusted)
    return draft


def _force_snap_inventory_sizes(
    draft: dict[str, Any],
    inventory_items: list[dict[str, Any]],
) -> dict[str, Any]:
    draft = _snap_inventory_sizes(draft, inventory_items)

    if not isinstance(draft, dict):
        return draft

    objects = draft.get("objects")
    if not isinstance(objects, list):
        return draft

    dims_by_type: dict[str, list[tuple[int, int, int]]] = {}
    for item in inventory_items:
        obj_type = item.get("type")
        dims = item.get("dimensions") or {}
        if not isinstance(obj_type, str) or not isinstance(dims, dict):
            continue
        L = int(dims.get("length_mm") or 0)
        W = int(dims.get("width_mm") or 0)
        H = int(dims.get("height_mm") or 0)
        if L <= 0 or W <= 0:
            continue
        dims_by_type.setdefault(obj_type, []).append((L, W, H))

    if not dims_by_type:
        return draft

    def pairs_for_dims(L: int, W: int, H: int, *, method: str) -> list[tuple[int, int]]:
        if method in {"floor", "on_top"}:
            return [(L, W)]
        pairs: list[tuple[int, int]] = []
        if W > 0 and H > 0:
            pairs.append((W, H))
        if L > 0 and W > 0:
            pairs.append((L, W))
        if L > 0 and H > 0:
            pairs.append((L, H))
        return pairs

    def _bbox_from_poly(poly: list[dict[str, Any]]) -> tuple[int, int, int, int] | None:
        if not poly:
            return None
        xs, ys = [], []
        for pt in poly:
            try:
                xs.append(int(pt.get("x")))
                ys.append(int(pt.get("y")))
            except Exception:
                return None
        if not xs or not ys:
            return None
        return min(xs), min(ys), max(xs), max(ys)

    def _rect_poly(
        min_x: int, min_y: int, max_x: int, max_y: int
    ) -> list[dict[str, int]]:
        return [
            {"x": min_x, "y": min_y},
            {"x": max_x, "y": min_y},
            {"x": max_x, "y": max_y},
            {"x": min_x, "y": max_y},
        ]

    out_objects: list[dict[str, Any]] = []
    forced = 0
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        row = dict(obj)
        if row.get("source") != "inventory":
            out_objects.append(row)
            continue

        obj_type = row.get("object_type")
        if not isinstance(obj_type, str) or obj_type not in dims_by_type:
            out_objects.append(row)
            continue

        place_on = row.get("place_on") if isinstance(row.get("place_on"), dict) else {}
        method = str(place_on.get("method") or "floor")

        dims_list = dims_by_type[obj_type]
        pair = None
        for L, W, H in dims_list:
            pairs = pairs_for_dims(L, W, H, method=method)
            if pairs:
                pair = pairs[0]
                break
        if pair is None:
            out_objects.append(row)
            continue

        bbox = row.get("bbox")
        if not isinstance(bbox, dict):
            poly = row.get("polygon_ccw") or []
            bbox_vals = _bbox_from_poly(poly)
            if bbox_vals is None:
                out_objects.append(row)
                continue
            min_x, min_y, max_x, max_y = bbox_vals
        else:
            try:
                min_x = int(bbox.get("min_x"))
                min_y = int(bbox.get("min_y"))
                max_x = int(bbox.get("max_x"))
                max_y = int(bbox.get("max_y"))
            except Exception:
                out_objects.append(row)
                continue

        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0
        new_w, new_h = pair
        nmin_x = int(round(cx - new_w / 2.0))
        nmax_x = int(round(cx + new_w / 2.0))
        nmin_y = int(round(cy - new_h / 2.0))
        nmax_y = int(round(cy + new_h / 2.0))

        row["bbox"] = {
            "min_x": nmin_x,
            "min_y": nmin_y,
            "max_x": nmax_x,
            "max_y": nmax_y,
        }
        row["polygon_ccw"] = _rect_poly(nmin_x, nmin_y, nmax_x, nmax_y)
        forced += 1
        out_objects.append(row)

    if forced:
        logger.info("Stylist forced inventory sizes for %s object(s).", forced)
    draft["objects"] = out_objects
    return draft


def _validate_inventory_sizes(
    draft: dict[str, Any],
    inventory_items: list[dict[str, Any]],
    tol_mm: int = 5,
) -> list[str]:
    dims_by_type: dict[str, list[tuple[int, int, int]]] = {}
    dims_by_id: dict[str, tuple[str, int, int, int]] = {}

    for item in inventory_items:
        obj_type = item.get("type")
        dims = item.get("dimensions") or {}
        if not isinstance(obj_type, str) or not isinstance(dims, dict):
            continue

        L = int(dims.get("length_mm") or 0)
        W = int(dims.get("width_mm") or 0)
        H = int(dims.get("height_mm") or 0)
        if L <= 0 or W <= 0:
            continue

        dims_by_type.setdefault(obj_type, []).append((L, W, H))
        inv_id = item.get("id")
        if isinstance(inv_id, str) and inv_id:
            dims_by_id[inv_id] = (obj_type, L, W, H)

    if not dims_by_type:
        return []

    def pairs_for_dims(L: int, W: int, H: int, *, method: str) -> list[tuple[int, int]]:
        # For footprint-like placements, we only accept (length,width).
        # For wall/ceiling/lean placements, a 2D representation can use any two axes.
        if method in {"floor", "on_top"}:
            return [(L, W)]
        pairs: list[tuple[int, int]] = []
        if W > 0 and H > 0:
            pairs.append((W, H))
        if L > 0 and W > 0:
            pairs.append((L, W))
        if L > 0 and H > 0:
            pairs.append((L, H))
        return pairs

    def match_pair(a: int, b: int, x: int, y: int) -> bool:
        return (abs(x - a) <= tol_mm and abs(y - b) <= tol_mm) or (
            abs(x - b) <= tol_mm and abs(y - a) <= tol_mm
        )

    issues: list[str] = []

    for obj in draft.get("objects") or []:
        if not isinstance(obj, dict):
            continue
        if obj.get("source") != "inventory":
            continue

        obj_type = obj.get("object_type")
        if not isinstance(obj_type, str) or obj_type not in dims_by_type:
            continue

        w, h = _bbox_dims(obj)
        if w is None or h is None:
            issues.append(f"{obj.get('instance_id') or obj_type}(missing_bbox)")
            continue

        place_on = obj.get("place_on") if isinstance(obj.get("place_on"), dict) else {}
        method = str(place_on.get("method") or "floor")
        instance_id = obj.get("instance_id")

        candidates: list[tuple[int, int, int]] = []
        if isinstance(instance_id, str) and instance_id in dims_by_id:
            _, L0, W0, H0 = dims_by_id[instance_id]
            candidates = [(L0, W0, H0)]
        else:
            candidates = dims_by_type.get(obj_type, [])

        ok = False
        sample_pairs: list[tuple[int, int]] = []
        for L, W, H in candidates:
            pairs = pairs_for_dims(L, W, H, method=method)
            if len(sample_pairs) < 3:
                sample_pairs.extend(pairs[: 3 - len(sample_pairs)])
            if any(match_pair(a, b, w, h) for a, b in pairs):
                ok = True
                break

        if not ok:
            name = str(instance_id or obj_type)
            issues.append(
                f"{name}({w}x{h} not matching inventory dims, sample={sample_pairs})"
            )

    return issues


def _bbox_dims(obj: dict[str, Any]) -> tuple[int | None, int | None]:
    bbox = obj.get("bbox")
    if not isinstance(bbox, dict):
        return None, None

    try:
        min_x = int(bbox.get("min_x"))
        min_y = int(bbox.get("min_y"))
        max_x = int(bbox.get("max_x"))
        max_y = int(bbox.get("max_y"))
    except Exception:
        return None, None

    return max_x - min_x, max_y - min_y


def _stable_unique(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _try_parse_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    for candidate in _json_object_parse_candidates(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _is_style_plan_shape(payload: dict[str, Any]) -> bool:
    room_surfaces = payload.get("room_surfaces")
    opening_colors = payload.get("opening_colors")
    object_styles = payload.get("object_styles")
    return (
        isinstance(room_surfaces, dict)
        and isinstance(opening_colors, dict)
        and isinstance(object_styles, list)
    )


def _json_object_parse_candidates(text: str) -> list[str]:
    coerced = _coerce_json_text(text).strip()
    if not coerced:
        return []
    raw_candidate = _extract_outer_json_slice(coerced).strip()
    candidates = [
        coerced,
        raw_candidate,
        _strip_trailing_json_commas(raw_candidate),
        _close_unterminated_json(_strip_trailing_json_commas(raw_candidate)),
    ]
    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        cleaned = candidate.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out


def _extract_outer_json_slice(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return text

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char in "{[":
            depth += 1
            continue
        if char in "}]":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return text[start:]


def _strip_trailing_json_commas(text: str) -> str:
    previous = text
    while True:
        cleaned = re.sub(r",(\s*[}\]])", r"\1", previous)
        if cleaned == previous:
            return cleaned
        previous = cleaned


def _close_unterminated_json(text: str) -> str:
    if not text:
        return text

    closers: list[str] = []
    in_string = False
    escape = False
    for char in text:
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            closers.append("}")
        elif char == "[":
            closers.append("]")
        elif char in "}]":
            if closers and closers[-1] == char:
                closers.pop()
            else:
                return text
    if in_string:
        return text
    return text + "".join(reversed(closers))


def _fix_place_on_targets(
    payload: dict[str, Any], *, layout_json: dict[str, Any]
) -> dict[str, Any]:
    """
    Stylist sometimes returns place_on.target_instance_id=null even though the schema requires a string.
    We repair this deterministically to keep the pipeline stable.
    """
    if not isinstance(payload, dict):
        return payload

    objects = payload.get("objects")
    if not isinstance(objects, list) or not objects:
        return payload

    supports: list[
        tuple[str, int, int, int, int, int]
    ] = []  # (id, minx, miny, maxx, maxy, area)
    for o in layout_json.get("objects") or []:
        if not isinstance(o, dict):
            continue
        inst = o.get("instance_id")
        bbox = o.get("bbox")
        if not isinstance(inst, str) or not inst or not isinstance(bbox, dict):
            continue
        try:
            min_x = int(bbox.get("min_x"))
            min_y = int(bbox.get("min_y"))
            max_x = int(bbox.get("max_x"))
            max_y = int(bbox.get("max_y"))
        except Exception:
            continue
        area = max(0, max_x - min_x) * max(0, max_y - min_y)
        supports.append((inst, min_x, min_y, max_x, max_y, area))

    # Prefer smallest container when inferring "on_top" supports.
    supports.sort(key=lambda x: x[5])

    def contains(
        a: tuple[str, int, int, int, int, int], b: dict[str, Any], *, tol: int = 5
    ) -> bool:
        w, h = _bbox_dims(b)
        bbox = b.get("bbox")
        if w is None or h is None or not isinstance(bbox, dict):
            return False
        try:
            bmin_x = int(bbox.get("min_x"))
            bmin_y = int(bbox.get("min_y"))
            bmax_x = int(bbox.get("max_x"))
            bmax_y = int(bbox.get("max_y"))
        except Exception:
            return False
        _, amin_x, amin_y, amax_x, amax_y, _ = a
        return (
            bmin_x >= amin_x - tol
            and bmin_y >= amin_y - tol
            and bmax_x <= amax_x + tol
            and bmax_y <= amax_y + tol
        )

    def is_ceiling_type(obj_type: str) -> bool:
        return obj_type in {
            "ceiling_light",
            "pendant_light",
            "track_light",
            "projector",
        }

    for obj in objects:
        if not isinstance(obj, dict):
            continue
        place_on = obj.get("place_on")
        if not isinstance(place_on, dict):
            continue
        target = place_on.get("target_instance_id")
        if isinstance(target, str) and target:
            continue

        method = str(place_on.get("method") or "")
        obj_type = str(obj.get("object_type") or "")

        if method in {"hang_on", "lean_on"}:
            place_on["target_instance_id"] = (
                "ceiling" if is_ceiling_type(obj_type) else "wall"
            )
            continue

        if method == "floor":
            place_on["target_instance_id"] = "floor"
            continue

        if method == "on_top":
            inferred = None
            for s in supports:
                if contains(s, obj):
                    inferred = s[0]
                    break
            if inferred:
                place_on["target_instance_id"] = inferred
            else:
                # If we can't infer a support, downgrade to a floor object for schema + renderer safety.
                place_on["method"] = "floor"
                place_on["target_instance_id"] = "floor"

    payload["objects"] = objects
    return payload
