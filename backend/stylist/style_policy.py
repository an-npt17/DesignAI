from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from clients.llm_client import get_llm_client

try:
    from config.llm_config import TextLLMConfig
except Exception:  # pragma: no cover
    from config.gemini_config import GeminiConfig

    class TextLLMConfig:
        PROVIDER = "gemini"
        STRICT_SINGLE_TEXT_MODEL = str(
            getattr(GeminiConfig, "STRICT_SINGLE_TEXT_MODEL", "") or ""
        ).strip()

        @classmethod
        def primary_model_name(cls) -> str:
            return ""


logger = logging.getLogger(__name__)

ChatMessage = dict[str, str]

STYLE_POLICY_COMPILER_SYSTEM_PROMPT = """You are a Style Policy Compiler for a deterministic 2D interior layout pipeline.

Your job is to convert the style brief into layout-level policy knobs. These knobs steer object count, spacing, center openness, wall loading, daylight preference, and decor tolerance before the geometry solver runs.

Return one compact strict JSON object only.
Do not use markdown.
Do not include comments.
Do not include prose before or after the JSON object.
Do not output partial strings.

You must output exactly these top-level keys:
- style_name
- target_density
- center_openness_bias
- wall_loading_bias
- symmetry_bias
- cluster_spacing_bias
- clutter_tolerance
- decor_tolerance
- daylight_bias
- floating_cluster_tolerance
- material_weight_bias
- visual_balance_bias

Allowed style_name values:
- minimal
- japandi
- scandinavian
- industrial
- boho
- coastal
- formal
- cozy
- social
- balanced

Allowed level-like values:
- low
- low_to_medium
- low_to_balanced
- medium
- balanced
- balanced_to_medium
- medium_high
- high
- very_high

Allowed wall_loading_bias values:
- low
- balanced
- medium
- medium_high
- focal_balanced
- perimeter

Allowed cluster_spacing_bias values:
- airy
- balanced
- structured
- layered
- soft
- social

Allowed material_weight_bias values:
- light
- balanced
- heavy

Allowed visual_balance_bias values:
- calm
- balanced
- bright
- contrast
- expressive
- formal
- cozy
- social

Choose conservative, solver-friendly values. Functional requirements, room capacity, access clearance, and hard geometry constraints always outrank style.
Use only the allowed enum-like values listed above.
Keep every value short and lowercase.
Do not place furniture.
Do not invent geometry.
Do not add object lists.
Do not override functional requirements.
Do not omit any key.
"""

_STYLE_POLICY_REQUIRED_KEYS: tuple[str, ...] = (
    "style_name",
    "target_density",
    "center_openness_bias",
    "wall_loading_bias",
    "symmetry_bias",
    "cluster_spacing_bias",
    "clutter_tolerance",
    "decor_tolerance",
    "daylight_bias",
    "floating_cluster_tolerance",
    "material_weight_bias",
    "visual_balance_bias",
)

_STYLE_NAME_VALUES: tuple[str, ...] = (
    "minimal",
    "japandi",
    "scandinavian",
    "industrial",
    "boho",
    "coastal",
    "formal",
    "cozy",
    "social",
    "balanced",
)
_STYLE_LEVEL_VALUES: tuple[str, ...] = (
    "low",
    "low_to_medium",
    "low_to_balanced",
    "medium",
    "balanced",
    "balanced_to_medium",
    "medium_high",
    "high",
    "very_high",
)
_WALL_LOADING_VALUES: tuple[str, ...] = (
    "low",
    "balanced",
    "medium",
    "medium_high",
    "focal_balanced",
    "perimeter",
)
_CLUSTER_SPACING_VALUES: tuple[str, ...] = (
    "airy",
    "balanced",
    "structured",
    "layered",
    "soft",
    "social",
)
_MATERIAL_WEIGHT_VALUES: tuple[str, ...] = (
    "light",
    "balanced",
    "heavy",
)
_VISUAL_BALANCE_VALUES: tuple[str, ...] = (
    "calm",
    "balanced",
    "bright",
    "contrast",
    "expressive",
    "formal",
    "cozy",
    "social",
)
_STYLE_POLICY_ENUMS_BY_KEY: dict[str, tuple[str, ...]] = {
    "style_name": _STYLE_NAME_VALUES,
    "target_density": _STYLE_LEVEL_VALUES,
    "center_openness_bias": _STYLE_LEVEL_VALUES,
    "wall_loading_bias": _WALL_LOADING_VALUES,
    "symmetry_bias": _STYLE_LEVEL_VALUES,
    "cluster_spacing_bias": _CLUSTER_SPACING_VALUES,
    "clutter_tolerance": _STYLE_LEVEL_VALUES,
    "decor_tolerance": _STYLE_LEVEL_VALUES,
    "daylight_bias": _STYLE_LEVEL_VALUES,
    "floating_cluster_tolerance": _STYLE_LEVEL_VALUES,
    "material_weight_bias": _MATERIAL_WEIGHT_VALUES,
    "visual_balance_bias": _VISUAL_BALANCE_VALUES,
}
_STYLE_POLICY_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "OBJECT",
    "properties": {
        key: {"type": "STRING", "enum": list(values)}
        for key, values in _STYLE_POLICY_ENUMS_BY_KEY.items()
    },
    "required": list(_STYLE_POLICY_REQUIRED_KEYS),
}

_STYLE_POLICY_REPAIR_PROMPT = (
    "Return exactly one valid JSON object with all required top-level keys: "
    + ", ".join(_STYLE_POLICY_REQUIRED_KEYS)
    + ". No markdown. No prose outside JSON."
)

_RESPONSE_MIME_TYPE = "application/json"
_STYLE_POLICY_REQUIRE_LLM_ENV = "TKNT_STYLE_POLICY_REQUIRE_LLM"
_STYLE_POLICY_MAX_TOKENS = 2000
_STYLE_POLICY_MODEL_NAMES: tuple[str, ...] = (
    "gemini-3.1-flash-lite-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
)
_STYLE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("japandi", ("japandi", "wabi", "zen", "japanese", "japan")),
    ("industrial", ("industrial", "loft", "concrete", "steel", "brick")),
    ("boho", ("boho", "bohemian", "eclectic", "layered")),
    ("coastal", ("coastal", "beach", "seaside", "airy coastal")),
    ("scandinavian", ("scandinavian", "scandi", "nordic")),
    ("minimal", ("minimal", "minimalist", "uncluttered", "simple", "clean")),
    ("formal", ("formal", "classic", "symmetrical", "symmetry", "elegant")),
    ("cozy", ("cozy", "warm", "soft", "restful", "calm")),
    ("social", ("social", "hosting", "conversation", "gathering")),
)

_STYLE_ONTOLOGY: dict[str, dict[str, str]] = {
    "minimal": {
        "target_density": "low_to_balanced",
        "center_openness_bias": "high",
        "wall_loading_bias": "balanced",
        "symmetry_bias": "medium",
        "cluster_spacing_bias": "airy",
        "clutter_tolerance": "low",
        "decor_tolerance": "low",
        "daylight_bias": "high",
        "floating_cluster_tolerance": "low_to_medium",
        "material_weight_bias": "light",
        "visual_balance_bias": "calm",
    },
    "japandi": {
        "target_density": "low_to_balanced",
        "center_openness_bias": "high",
        "wall_loading_bias": "balanced",
        "symmetry_bias": "medium",
        "cluster_spacing_bias": "airy",
        "clutter_tolerance": "low",
        "decor_tolerance": "low",
        "daylight_bias": "high",
        "floating_cluster_tolerance": "low_to_medium",
        "material_weight_bias": "light",
        "visual_balance_bias": "calm",
    },
    "scandinavian": {
        "target_density": "balanced",
        "center_openness_bias": "high",
        "wall_loading_bias": "balanced",
        "symmetry_bias": "medium",
        "cluster_spacing_bias": "airy",
        "clutter_tolerance": "low",
        "decor_tolerance": "low_to_medium",
        "daylight_bias": "high",
        "floating_cluster_tolerance": "medium",
        "material_weight_bias": "light",
        "visual_balance_bias": "calm",
    },
    "industrial": {
        "target_density": "balanced_to_medium",
        "center_openness_bias": "medium",
        "wall_loading_bias": "medium_high",
        "symmetry_bias": "low_to_medium",
        "cluster_spacing_bias": "structured",
        "clutter_tolerance": "medium",
        "decor_tolerance": "medium",
        "daylight_bias": "medium",
        "floating_cluster_tolerance": "medium",
        "material_weight_bias": "heavy",
        "visual_balance_bias": "contrast",
    },
    "boho": {
        "target_density": "medium",
        "center_openness_bias": "medium",
        "wall_loading_bias": "balanced",
        "symmetry_bias": "low",
        "cluster_spacing_bias": "layered",
        "clutter_tolerance": "medium_high",
        "decor_tolerance": "high",
        "daylight_bias": "medium",
        "floating_cluster_tolerance": "medium",
        "material_weight_bias": "balanced",
        "visual_balance_bias": "expressive",
    },
    "coastal": {
        "target_density": "low_to_balanced",
        "center_openness_bias": "very_high",
        "wall_loading_bias": "balanced",
        "symmetry_bias": "low_to_medium",
        "cluster_spacing_bias": "airy",
        "clutter_tolerance": "low",
        "decor_tolerance": "low_to_medium",
        "daylight_bias": "high",
        "floating_cluster_tolerance": "low_to_medium",
        "material_weight_bias": "light",
        "visual_balance_bias": "bright",
    },
    "formal": {
        "target_density": "balanced",
        "center_openness_bias": "medium",
        "wall_loading_bias": "focal_balanced",
        "symmetry_bias": "high",
        "cluster_spacing_bias": "structured",
        "clutter_tolerance": "low_to_medium",
        "decor_tolerance": "medium",
        "daylight_bias": "medium",
        "floating_cluster_tolerance": "medium",
        "material_weight_bias": "balanced",
        "visual_balance_bias": "formal",
    },
    "cozy": {
        "target_density": "balanced_to_medium",
        "center_openness_bias": "medium",
        "wall_loading_bias": "balanced",
        "symmetry_bias": "low_to_medium",
        "cluster_spacing_bias": "soft",
        "clutter_tolerance": "medium",
        "decor_tolerance": "medium",
        "daylight_bias": "medium",
        "floating_cluster_tolerance": "medium",
        "material_weight_bias": "balanced",
        "visual_balance_bias": "cozy",
    },
    "social": {
        "target_density": "balanced_to_medium",
        "center_openness_bias": "medium",
        "wall_loading_bias": "balanced",
        "symmetry_bias": "medium",
        "cluster_spacing_bias": "social",
        "clutter_tolerance": "medium",
        "decor_tolerance": "medium",
        "daylight_bias": "medium",
        "floating_cluster_tolerance": "medium_high",
        "material_weight_bias": "balanced",
        "visual_balance_bias": "social",
    },
}

_BASE_POLICY = _STYLE_ONTOLOGY["minimal"] | {
    "target_density": "balanced",
    "center_openness_bias": "medium",
    "wall_loading_bias": "balanced",
    "cluster_spacing_bias": "balanced",
    "clutter_tolerance": "medium",
    "decor_tolerance": "medium",
    "daylight_bias": "medium",
    "floating_cluster_tolerance": "medium",
    "material_weight_bias": "balanced",
    "visual_balance_bias": "balanced",
}


def compile_style_policy(
    *,
    room_type: str,
    brief_text: str,
    room_model_json: Mapping[str, Any] | None = None,
    semantic_program_rules: Mapping[str, Any] | None = None,
    use_llm: bool = True,
    model_name: str | None = None,
) -> dict[str, Any]:
    require_llm = _style_policy_requires_llm()
    llm_seed: dict[str, Any] = {}
    if use_llm and brief_text.strip():
        llm_seed = _try_llm_style_policy_seed(
            room_type=room_type,
            brief_text=brief_text,
            room_model_json=room_model_json,
            semantic_program_rules=semantic_program_rules,
            model_name=model_name,
            require_llm=require_llm,
        )
    elif require_llm:
        raise ValueError(
            "Style policy strict LLM mode is enabled, but no style brief was provided."
        )

    deterministic = (
        {"style_name": "balanced"}
        if require_llm
        else _deterministic_style_policy_seed(brief_text)
    )
    style_name = (
        _canonical_llm_style_name(llm_seed)
        if require_llm
        else _canonical_style_name(llm_seed, deterministic)
    )
    layout_policy = dict(_BASE_POLICY)
    layout_policy.update(_STYLE_ONTOLOGY.get(style_name, {}))
    layout_policy.update(_validated_layout_policy_seed(llm_seed))
    layout_policy = _apply_text_modifiers(layout_policy, brief_text)
    layout_policy, room_notes = _apply_room_type_correction(
        layout_policy,
        room_type=room_type,
    )
    layout_policy = _normalize_layout_policy(layout_policy)
    return {
        "style_name": style_name,
        "style_tags": _style_tags(brief_text, style_name),
        "layout_policy": layout_policy,
        "cluster_policy_overrides": _cluster_policy_overrides(
            layout_policy,
            room_type=room_type,
        ),
        "policy_weights": _policy_weights(layout_policy),
        "search_settings": {
            "style_policy_variants": 1,
            "final_palette_candidates": 3,
            "decor_plan_candidates": 3,
            "max_decor_items_per_room": _max_decor_items_per_room(layout_policy),
            "max_decor_items_per_cluster": 1,
        },
        "controller": {
            "phase_early": "style_policy_compiler",
            "phase_late": "final_stylist_decorator",
            "llm_seed_used": bool(llm_seed),
            "validation": "deterministic_policy_normalization",
        },
        "notes": room_notes,
    }


def build_neutral_style_policy(
    *,
    room_type: str,
    reason: str = "Neutral style policy.",
) -> dict[str, Any]:
    layout_policy = _normalize_layout_policy(_BASE_POLICY)
    return {
        "style_name": "balanced",
        "style_tags": [],
        "layout_policy": layout_policy,
        "cluster_policy_overrides": _cluster_policy_overrides(
            layout_policy,
            room_type=room_type,
        ),
        "policy_weights": _policy_weights(layout_policy),
        "search_settings": {
            "style_policy_variants": 1,
            "final_palette_candidates": 1,
            "decor_plan_candidates": 1,
            "max_decor_items_per_room": _max_decor_items_per_room(layout_policy),
            "max_decor_items_per_cluster": 1,
        },
        "controller": {
            "phase_early": "neutral_ablation_policy",
            "phase_late": "final_stylist_decorator",
            "llm_seed_used": False,
            "validation": "neutral_base_policy",
        },
        "notes": [reason],
    }


def apply_style_policy_to_semantic_program(
    semantic_program: Mapping[str, Any],
    style_policy: Mapping[str, Any] | None,
) -> dict[str, Any]:
    out = deepcopy(dict(semantic_program))
    policy = _layout_policy(style_policy)
    if not policy:
        return out

    out["style_policy"] = deepcopy(dict(style_policy or {}))
    intent = out.get("global_layout_intent")
    if not isinstance(intent, dict):
        intent = {}
        out["global_layout_intent"] = intent
    intent["style_name"] = str((style_policy or {}).get("style_name") or "balanced")
    intent["space_character"] = _space_character(policy, intent)
    intent["prefer_open_center"] = _bias_at_least(
        policy.get("center_openness_bias"),
        {"high", "very_high"},
        default=bool(intent.get("prefer_open_center", True)),
    )
    intent["prefer_core_before_support"] = True
    intent["style_policy_summary"] = {
        "target_density": policy.get("target_density"),
        "center_openness_bias": policy.get("center_openness_bias"),
        "clutter_tolerance": policy.get("clutter_tolerance"),
        "decor_tolerance": policy.get("decor_tolerance"),
    }

    quality = out.get("quality_targets")
    if not isinstance(quality, dict):
        quality = {}
        out["quality_targets"] = quality
    weights = _policy_weights(policy)
    quality["functionality_weight"] = 1.0
    quality["naturalness_weight"] = max(
        float(quality.get("naturalness_weight") or 1.0),
        float(weights["naturalness_weight"]),
    )
    quality["semantic_coherence_weight"] = max(
        float(quality.get("semantic_coherence_weight") or 1.0),
        float(weights["semantic_coherence_weight"]),
    )
    quality["spatial_quality_weight"] = max(
        float(quality.get("spatial_quality_weight") or 1.0),
        float(weights["spatial_quality_weight"]),
    )

    _apply_policy_to_active_clusters(out, policy)
    _apply_policy_to_macro_relations(out, policy)

    notes = _str_list(out.get("notes"))
    note = "Style policy compiled before layout planning; downstream geometry treats it as bias, not a hard override."
    if note not in notes:
        notes.append(note)
    out["notes"] = notes
    return out


def extract_style_policy(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    policy = payload.get("style_policy")
    if isinstance(policy, Mapping):
        return deepcopy(dict(policy))
    semantic = payload.get("semantic_layout_program")
    if isinstance(semantic, Mapping):
        policy = semantic.get("style_policy")
        if isinstance(policy, Mapping):
            return deepcopy(dict(policy))
    return {}


def decor_refill_policy(style_policy: Mapping[str, Any] | None) -> dict[str, Any]:
    policy = _layout_policy(style_policy)
    max_total = _max_decor_items_per_room(policy)
    if _value_level(policy.get("decor_tolerance")) <= 1:
        max_total = min(max_total, 1)
    return {
        "allow_categories": ["accessory", "decor"],
        "max_refills_total": max(0, max_total),
    }


def final_style_plan_from_surface_payload(
    *,
    style_policy: Mapping[str, Any] | None,
    style_plan: Mapping[str, Any] | None,
    styled_payload: Mapping[str, Any],
) -> dict[str, Any]:
    policy = _layout_policy(style_policy)
    style_name = str((style_policy or {}).get("style_name") or "balanced")
    room = styled_payload.get("room") if isinstance(styled_payload, Mapping) else None
    room = room if isinstance(room, Mapping) else {}
    surfaces = room.get("surfaces") if isinstance(room.get("surfaces"), Mapping) else {}
    object_styles = (
        style_plan.get("object_styles") if isinstance(style_plan, Mapping) else []
    )
    return {
        "style_name": style_name,
        "layout_policy_trace": deepcopy(dict(style_policy or {})),
        "palette": _semantic_palette(style_name, surfaces),
        "surface_plan": {
            "walls": _surface_label(surfaces.get("wall_color_hex"), "paint"),
            "floor": _surface_label(surfaces.get("floor_color_hex"), "floor_finish"),
            "ceiling": _surface_label(
                surfaces.get("ceiling_color_hex"), "ceiling_paint"
            ),
        },
        "object_finish_plan": _object_finish_plan(object_styles),
        "decor_plan": _deterministic_decor_plan(
            styled_payload=styled_payload,
            style_policy=style_policy,
        ),
        "lighting_mood": _lighting_mood(policy),
        "rules": {
            "layout_locked": True,
            "max_decor_items_per_room": _max_decor_items_per_room(policy),
            "max_decor_items_per_cluster": 1,
        },
    }


def _try_llm_style_policy_seed(
    *,
    room_type: str,
    brief_text: str,
    room_model_json: Mapping[str, Any] | None,
    semantic_program_rules: Mapping[str, Any] | None,
    model_name: str | None,
    require_llm: bool,
) -> dict[str, Any]:
    payload = {
        "room_type": room_type,
        "brief": {"text": brief_text},
        "room_model": {
            "affordance_map": (room_model_json or {}).get("affordance_map")
            if isinstance(room_model_json, Mapping)
            else {},
            "topology": (room_model_json or {}).get("topology")
            if isinstance(room_model_json, Mapping)
            else {},
        },
        "semantic_program_rules": semantic_program_rules or {},
    }
    messages: list[ChatMessage] = [
        {"role": "system", "content": STYLE_POLICY_COMPILER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "INPUT JSON:\n"
            + json.dumps(payload, ensure_ascii=True, indent=2)
            + "\nReturn strict JSON only with these top-level keys: "
            + ", ".join(_STYLE_POLICY_REQUIRED_KEYS)
            + ". Every key must be present and every value must be a short string.",
        },
    ]
    try:
        client = get_llm_client()
        attempted_models: list[str] = []
        last_error: Exception | None = None
        for candidate_model_name in _style_policy_model_attempt_order(
            client,
            model_name,
        ):
            attempted_models.append(candidate_model_name)
            candidate_messages = list(messages)
            for attempt in range(2):
                raw_text = ""
                try:
                    response = client.chat_completion(
                        candidate_messages,
                        model_key="helper",
                        model_name=candidate_model_name,
                        fallback_model_names=(),
                        temperature=0.0,
                        top_p=0.9,
                        max_tokens=_STYLE_POLICY_MAX_TOKENS,
                        response_mime_type=_RESPONSE_MIME_TYPE,
                        response_schema=_STYLE_POLICY_RESPONSE_SCHEMA,
                    )
                    raw_text = _extract_content(response)
                    parsed = _parse_json_object(raw_text)
                    _validate_style_policy_seed(parsed)
                    return parsed
                except Exception as exc:
                    last_error = exc
                    if _is_retryable_model_unavailable_error(exc):
                        _record_llm_retry(
                            stage="style_policy",
                            model_name=candidate_model_name,
                            reason="model_unavailable",
                        )
                        logger.warning(
                            "Style policy model %s is temporarily unavailable.",
                            candidate_model_name,
                        )
                        break
                    if attempt == 1:
                        logger.warning(
                            "Style policy model %s returned invalid JSON after repair. Error: %s",
                            candidate_model_name,
                            exc,
                        )
                        break
                    _record_llm_retry(
                        stage="style_policy",
                        model_name=candidate_model_name,
                        reason="invalid_json_or_schema",
                    )
                    candidate_messages.append(
                        {"role": "assistant", "content": raw_text}
                    )
                    candidate_messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"{_STYLE_POLICY_REPAIR_PROMPT}\nValidation error: {exc}"
                            ),
                        }
                    )
        if last_error is not None:
            if _is_retryable_model_unavailable_error(last_error):
                raise RuntimeError(
                    "Style policy model unavailable after attempting models: "
                    f"{', '.join(attempted_models)}"
                ) from last_error
            raise ValueError(
                "Style policy compiler could not produce valid JSON after "
                f"attempting models: {', '.join(attempted_models)}"
            ) from last_error
        raise RuntimeError("Style policy compiler produced no model candidates.")
    except Exception as exc:  # pragma: no cover - external LLM/network fallback
        if require_llm:
            logger.warning("Style policy strict LLM mode failed: %s", exc)
            raise
        logger.info(
            "Style policy LLM seed skipped; using deterministic compiler: %s", exc
        )
        return {}


def _deterministic_style_policy_seed(brief_text: str) -> dict[str, str]:
    text = brief_text.lower()
    for style_name, keywords in _STYLE_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return {"style_name": style_name}
    return {"style_name": "balanced"}


def _style_policy_requires_llm() -> bool:
    return os.getenv(_STYLE_POLICY_REQUIRE_LLM_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _record_llm_retry(*, stage: str, model_name: str, reason: str) -> None:
    if getattr(TextLLMConfig, "PROVIDER", "") != "gemini":
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


def _style_policy_model_attempt_order(
    client: object,
    model_name: str | None,
) -> list[str]:
    strict_model_name = str(TextLLMConfig.STRICT_SINGLE_TEXT_MODEL or "").strip()
    if strict_model_name:
        return [strict_model_name]
    helper_model_name = ""
    get_model_name = getattr(client, "get_model_name", None)
    if callable(get_model_name):
        helper_model_name = str(get_model_name("helper")).strip()
    requested_model_name = str(model_name or helper_model_name).strip()
    sequence = (
        list(_STYLE_POLICY_MODEL_NAMES) if TextLLMConfig.PROVIDER == "gemini" else []
    )
    if not requested_model_name:
        return sequence or [TextLLMConfig.primary_model_name()]
    if requested_model_name in sequence:
        start_index = sequence.index(requested_model_name)
        return sequence[start_index:] + sequence[:start_index]
    return [
        requested_model_name,
        *[name for name in sequence if name != requested_model_name],
    ]


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


def _canonical_llm_style_name(llm_seed: Mapping[str, Any]) -> str:
    llm_name = str(llm_seed.get("style_name") or "").strip().lower()
    if llm_name in _STYLE_ONTOLOGY or llm_name == "balanced":
        return llm_name
    return "balanced"


def _canonical_style_name(
    llm_seed: Mapping[str, Any],
    deterministic: Mapping[str, str],
) -> str:
    deterministic_name = str(deterministic.get("style_name") or "").strip().lower()
    if deterministic_name and deterministic_name != "balanced":
        return deterministic_name
    candidates = [
        str(llm_seed.get("style_name") or "").strip().lower(),
        deterministic_name,
    ]
    for candidate in candidates:
        for style_name, keywords in _STYLE_KEYWORDS:
            if candidate == style_name or any(
                keyword in candidate for keyword in keywords
            ):
                return style_name
    return "balanced"


def _validated_layout_policy_seed(seed: Mapping[str, Any]) -> dict[str, str]:
    allowed_keys = set(_BASE_POLICY)
    out: dict[str, str] = {}
    for key in allowed_keys:
        value = seed.get(key)
        if isinstance(value, str) and value.strip():
            out[key] = value.strip().lower()
    return out


def _apply_text_modifiers(policy: dict[str, str], brief_text: str) -> dict[str, str]:
    text = brief_text.lower()
    out = dict(policy)
    if any(token in text for token in ("airy", "bright", "open", "spacious")):
        out["center_openness_bias"] = "very_high"
        out["cluster_spacing_bias"] = "airy"
        out["daylight_bias"] = "high"
    if any(token in text for token in ("uncluttered", "minimal", "clean", "calm")):
        out["clutter_tolerance"] = "low"
        out["decor_tolerance"] = "low"
    if any(token in text for token in ("cozy", "warm", "soft", "restful")):
        out["visual_balance_bias"] = "cozy" if "cozy" in text else "calm"
        out["cluster_spacing_bias"] = "soft"
    if any(token in text for token in ("formal", "symmetrical", "symmetry")):
        out["symmetry_bias"] = "high"
        out["visual_balance_bias"] = "formal"
    if any(token in text for token in ("social", "hosting", "conversation")):
        out["floating_cluster_tolerance"] = "medium_high"
        out["visual_balance_bias"] = "social"
    return out


def _apply_room_type_correction(
    policy: dict[str, str],
    *,
    room_type: str,
) -> tuple[dict[str, str], list[str]]:
    normalized = room_type.strip().lower().replace(" ", "_")
    out = dict(policy)
    notes: list[str] = []
    if "bedroom" in normalized:
        out["visual_balance_bias"] = "calm"
        out["clutter_tolerance"] = _min_level(out["clutter_tolerance"], "medium")
        notes.append("Bedroom correction raised restfulness and clutter control.")
    elif "living" in normalized:
        out["symmetry_bias"] = _max_level(out["symmetry_bias"], "medium")
        notes.append("Living room correction preserved social clarity and focal logic.")
    elif "office" in normalized or "study" in normalized:
        out["daylight_bias"] = "high"
        out["clutter_tolerance"] = _min_level(out["clutter_tolerance"], "medium")
        notes.append("Home office correction raised daylight and productivity clarity.")
    elif "studio" in normalized:
        out["clutter_tolerance"] = "low"
        out["center_openness_bias"] = _max_level(out["center_openness_bias"], "high")
        notes.append("Studio correction raised multifunction clutter control.")
    return out, notes


def _normalize_layout_policy(policy: Mapping[str, str]) -> dict[str, str]:
    out = dict(_BASE_POLICY)
    for key, value in policy.items():
        text = str(value or "").strip().lower()
        if text:
            out[key] = text
    return out


def _cluster_policy_overrides(
    policy: Mapping[str, str],
    *,
    room_type: str,
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if _value_level(policy.get("center_openness_bias")) >= 3:
        overrides["sleep_core"] = {"prefer_breathing_room": True}
        overrides["primary_core"] = {"prefer_breathing_room": True}
    if _value_level(policy.get("clutter_tolerance")) <= 1:
        overrides["support_optional"] = {"allow_only_if_space_surplus": True}
        overrides["decor"] = {"max_items_per_cluster": 1}
    if (
        "living" in room_type.lower()
        and str(policy.get("visual_balance_bias")) == "social"
    ):
        overrides["lounge_reading"] = {"allow_if_social_clearance_surplus": True}
    else:
        overrides.setdefault("lounge_reading", {"allow_only_if_space_surplus": True})
    return overrides


def _policy_weights(policy: Mapping[str, str]) -> dict[str, float]:
    openness = _value_level(policy.get("center_openness_bias"))
    symmetry = _value_level(policy.get("symmetry_bias"))
    clutter = _value_level(policy.get("clutter_tolerance"))
    decor = _value_level(policy.get("decor_tolerance"))
    return {
        "density_multiplier": _density_multiplier(policy.get("target_density")),
        "optional_utility_bias": -0.45
        if clutter <= 1
        else 0.25
        if clutter >= 3
        else 0.0,
        "decor_utility_bias": -0.9 if decor <= 1 else 0.65 if decor >= 3 else 0.0,
        "openness_weight": 1.15 if openness >= 3 else 1.0,
        "symmetry_weight": 1.12 if symmetry >= 3 else 1.0,
        "naturalness_weight": 1.12 if openness >= 3 or decor <= 1 else 1.0,
        "semantic_coherence_weight": 1.06,
        "spatial_quality_weight": 1.15 if openness >= 3 else 1.0,
    }


def _density_multiplier(value: Any) -> float:
    text = str(value or "").lower()
    if "low" in text:
        return 0.82
    if "medium_high" in text or "high" in text:
        return 1.12
    if "medium" in text:
        return 1.06
    return 1.0


def _max_decor_items_per_room(policy: Mapping[str, str]) -> int:
    decor_level = _value_level(policy.get("decor_tolerance"))
    if decor_level <= 1:
        return 1
    if decor_level >= 3:
        return 3
    return 2


def _style_tags(brief_text: str, style_name: str) -> list[str]:
    tags = [style_name] if style_name != "balanced" else []
    text = brief_text.lower()
    for tag, keywords in _STYLE_KEYWORDS:
        if tag not in tags and any(keyword in text for keyword in keywords):
            tags.append(tag)
    return tags[:6]


def _layout_policy(style_policy: Mapping[str, Any] | None) -> dict[str, str]:
    if not isinstance(style_policy, Mapping):
        return {}
    policy = style_policy.get("layout_policy")
    if not isinstance(policy, Mapping):
        return {}
    return {str(key): str(value) for key, value in policy.items()}


def _space_character(policy: Mapping[str, str], intent: Mapping[str, Any]) -> str:
    style = str(policy.get("visual_balance_bias") or "").strip()
    base = str(intent.get("space_character") or "balanced_functional").strip()
    return f"{style}_{base}" if style and style not in base else base


def _apply_policy_to_active_clusters(
    semantic_program: dict[str, Any],
    policy: Mapping[str, str],
) -> None:
    clusters = semantic_program.get("active_clusters")
    if not isinstance(clusters, list):
        return
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        zone_claims = cluster.get("zone_claims")
        if not isinstance(zone_claims, dict):
            zone_claims = {}
            cluster["zone_claims"] = zone_claims
        if _value_level(policy.get("daylight_bias")) >= 3:
            zone_claims["daylight_affinity"] = _max_affinity(
                zone_claims.get("daylight_affinity"),
                "medium",
            )
        if _wall_level(policy.get("wall_loading_bias")) >= 3:
            zone_claims["wall_affinity"] = _max_affinity(
                zone_claims.get("wall_affinity"),
                "high",
            )
        if _value_level(policy.get("floating_cluster_tolerance")) <= 1:
            zone_claims["floating_allowed"] = False
        if (
            _value_level(policy.get("clutter_tolerance")) <= 1
            and cluster.get("priority") != "core"
        ):
            ladder = _str_list(cluster.get("degradation_ladder"))
            note = "drop_if_room_capacity_tight_under_style_policy"
            if note not in ladder:
                ladder.insert(0, note)
            cluster["degradation_ladder"] = ladder[:5]


def _apply_policy_to_macro_relations(
    semantic_program: dict[str, Any],
    policy: Mapping[str, str],
) -> None:
    macro = semantic_program.get("macro_relations")
    if not isinstance(macro, dict):
        macro = {}
        semantic_program["macro_relations"] = macro
    keep_open = macro.get("keep_open_regions")
    if not isinstance(keep_open, list):
        keep_open = []
        macro["keep_open_regions"] = keep_open
    if _value_level(policy.get("center_openness_bias")) >= 3:
        row = {
            "type": "center_lane",
            "near": "room_center",
            "priority": "high",
            "reason": "Style policy reserves center openness.",
        }
        if row not in keep_open:
            keep_open.append(row)
    reserved = macro.get("reserved_regions")
    if not isinstance(reserved, list):
        reserved = []
        macro["reserved_regions"] = reserved
    if _value_level(policy.get("center_openness_bias")) >= 4:
        row = {
            "region": "room_center",
            "reason": "Very high openness bias from style policy.",
        }
        if row not in reserved:
            reserved.append(row)


def _bias_at_least(value: Any, levels: set[str], *, default: bool) -> bool:
    text = str(value or "").strip().lower()
    return True if text in levels else default


def _max_affinity(value: Any, minimum: str) -> str:
    order = {"none": 0, "low": 1, "medium": 2, "high": 3}
    current = str(value or "none").strip().lower()
    return current if order.get(current, 0) >= order[minimum] else minimum


def _value_level(value: Any) -> int:
    text = str(value or "").lower()
    if "very_high" in text or "high" in text:
        return 4
    if "medium_high" in text:
        return 3
    if "low_to_medium" in text or "low_to_balanced" in text:
        return 1
    if "medium" in text or "balanced" in text:
        return 2
    if "low" in text:
        return 0
    return 2


def _wall_level(value: Any) -> int:
    text = str(value or "").lower()
    if "perimeter" in text or "medium_high" in text or "focal" in text:
        return 3
    if "balanced" in text:
        return 2
    if "low" in text:
        return 0
    return 2


def _min_level(current: str, ceiling: str) -> str:
    order = ["low", "low_to_medium", "medium", "medium_high", "high"]
    current_index = order.index(current) if current in order else 2
    ceiling_index = order.index(ceiling) if ceiling in order else 2
    return order[min(current_index, ceiling_index)]


def _max_level(current: str, floor: str) -> str:
    order = ["low", "low_to_medium", "medium", "medium_high", "high"]
    current_index = order.index(current) if current in order else 2
    floor_index = order.index(floor) if floor in order else 2
    return order[max(current_index, floor_index)]


def _lighting_mood(policy: Mapping[str, str]) -> str:
    if _value_level(policy.get("daylight_bias")) >= 3:
        return f"soft_daylit_{policy.get('visual_balance_bias', 'balanced')}"
    return f"layered_warm_{policy.get('visual_balance_bias', 'balanced')}"


def _semantic_palette(style_name: str, surfaces: Mapping[str, Any]) -> dict[str, str]:
    labels = {
        "japandi": {
            "walls": "warm_off_white",
            "floor": "light_oak",
            "primary_wood": "natural_oak",
            "metal": "matte_black_soft",
            "textiles": "beige_sand_gray",
        },
        "industrial": {
            "walls": "warm_gray_plaster",
            "floor": "smoked_wood_or_concrete",
            "primary_wood": "reclaimed_dark_wood",
            "metal": "aged_blackened_steel",
            "textiles": "charcoal_taupe_leather",
        },
        "boho": {
            "walls": "warm_clay_white",
            "floor": "natural_wood",
            "primary_wood": "rattan_and_oak",
            "metal": "antique_brass",
            "textiles": "terracotta_cream_olive",
        },
    }
    palette = dict(labels.get(style_name, labels["japandi"]))
    for key, surface_key in (
        ("wall_color_hex", "walls_hex"),
        ("floor_color_hex", "floor_hex"),
        ("ceiling_color_hex", "ceiling_hex"),
    ):
        value = surfaces.get(key)
        if isinstance(value, str) and value.strip():
            palette[surface_key] = value.strip()
    return palette


def _surface_label(value: Any, fallback: str) -> dict[str, str]:
    color = value if isinstance(value, str) and value.strip() else ""
    return {"finish": fallback, "color_hex": color}


def _object_finish_plan(object_styles: Any) -> list[dict[str, str | None]]:
    if not isinstance(object_styles, list):
        return []
    out: list[dict[str, str | None]] = []
    for row in object_styles:
        if not isinstance(row, Mapping):
            continue
        instance_id = row.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id.strip():
            continue
        material = row.get("material")
        out.append(
            {
                "instance_id": instance_id.strip(),
                "material": material.strip() if isinstance(material, str) else None,
                "color_hex": row.get("color_hex")
                if isinstance(row.get("color_hex"), str)
                else None,
            }
        )
    return out


def _deterministic_decor_plan(
    *,
    styled_payload: Mapping[str, Any],
    style_policy: Mapping[str, Any] | None,
) -> list[dict[str, str]]:
    policy = _layout_policy(style_policy)
    max_items = _max_decor_items_per_room(policy)
    objects = styled_payload.get("objects")
    rows = objects if isinstance(objects, list) else []
    plan: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        object_type = str(row.get("object_type") or "").lower()
        if not any(
            token in object_type
            for token in ("art", "plant", "vase", "decor", "lamp", "rug")
        ):
            continue
        plan.append(
            {
                "type": object_type,
                "placement_zone": str(row.get("cluster_id") or "existing_layout_zone"),
                "style_role": _decor_style_role(policy),
            }
        )
        if len(plan) >= max_items:
            break
    return plan


def _decor_style_role(policy: Mapping[str, str]) -> str:
    if _value_level(policy.get("decor_tolerance")) <= 1:
        return "quiet_minimal_accent"
    if _value_level(policy.get("decor_tolerance")) >= 3:
        return "layered_expressive_accent"
    return "cohesive_light_accent"


def _extract_content(response: object) -> str:
    choices = getattr(response, "choices", None)
    if isinstance(choices, Sequence) and choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
    raise ValueError("Style policy compiler response missing message content")


def _parse_json_object(raw: str) -> dict[str, Any]:
    last_decode_error: json.JSONDecodeError | None = None
    for text in _json_object_parse_candidates(raw):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            last_decode_error = exc
            continue
        if not isinstance(payload, dict):
            raise ValueError("Style policy compiler JSON must be an object")
        return payload

    if last_decode_error is not None:
        raise ValueError(
            f"Style policy compiler returned malformed JSON: {last_decode_error}"
        ) from last_decode_error
    raise ValueError("Style policy compiler response did not contain a JSON object")


def _json_object_parse_candidates(raw: str) -> list[str]:
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        for idx in range(1, len(parts), 2):
            candidate = parts[idx].strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].lstrip()
            if candidate.startswith("{") and candidate.endswith("}"):
                text = candidate
                break
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match is not None:
            text = match.group(0)
    candidates = [
        text,
        _strip_trailing_json_commas(text),
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


def _strip_trailing_json_commas(text: str) -> str:
    previous = text
    while True:
        cleaned = re.sub(r",(\s*[}\]])", r"\1", previous)
        if cleaned == previous:
            return cleaned
        previous = cleaned


def _validate_style_policy_seed(payload: Mapping[str, Any]) -> None:
    missing = [key for key in _STYLE_POLICY_REQUIRED_KEYS if key not in payload]
    if missing:
        raise ValueError(
            "Style policy compiler JSON omitted required keys: " + ", ".join(missing)
        )
    invalid: list[str] = []
    for key, allowed_values in _STYLE_POLICY_ENUMS_BY_KEY.items():
        value = payload.get(key)
        normalized = value.strip().lower() if isinstance(value, str) else ""
        if normalized not in allowed_values:
            invalid.append(f"{key}={value!r}")
    if invalid:
        raise ValueError(
            "Style policy compiler JSON used invalid enum values: " + ", ".join(invalid)
        )


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return out
