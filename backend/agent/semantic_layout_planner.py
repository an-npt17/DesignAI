from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from agent.request_contract import build_request_contract, sanitize_request_contract
from agent_schema.semantic_layout_planner_schema import SemanticLayoutProgram
from clients.base_client import ChatMessage
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
        AGENT_MODELS = dict(getattr(GeminiConfig, "AGENT_MODELS", {}) or {})

        @classmethod
        def primary_model_name(cls) -> str:
            return ""

        @classmethod
        def agent_model(cls, key: str) -> str | None:
            return cls.AGENT_MODELS.get(key)

        @classmethod
        def agent_model_chain(
            cls,
            model_config_keys: Sequence[str],
            default_model_chain: Sequence[str] = (),
        ) -> list[str]:
            if cls.STRICT_SINGLE_TEXT_MODEL:
                return [cls.STRICT_SINGLE_TEXT_MODEL]
            configured = [
                cls.AGENT_MODELS.get(config_key) for config_key in model_config_keys
            ]
            return [
                model_name
                for model_name in (*configured, *default_model_chain)
                if isinstance(model_name, str) and model_name.strip()
            ]


from layout.grid_policy import GLOBAL_LAYOUT_GRID_MM
from layout.room_profiles.registry import (
    is_profile_floating_object,
    is_profile_storage_object,
    is_profile_wall_backed_object,
    is_profile_workflow_object,
    profile_cluster_tag_for_objects,
    profile_layout_role_for_objects,
    profile_layout_trace_for_active_clusters,
    profile_macro_relations_for_active_clusters,
    profile_relation_intents_for_objects,
    profile_semantic_role_for_objects,
    profile_zone_claims_for_objects,
    select_profile_room_rule,
    semantic_placements_for_members,
    semantic_room_rule_for,
)
from layout.semantic_roles import (
    is_bed_like,
    is_bedside_support_like,
    is_bench_like,
    is_lounge_anchor_like,
    is_seat_like,
    is_work_surface_like,
)
from prompt.semantic_layout_planner import (
    SEMANTIC_LAYOUT_PLANNER_SYSTEM_PROMPT,
    SEMANTIC_LAYOUT_PLANNER_USER_PROMPT,
)
from stylist.semantic_program_rules import get_compiled_semantic_room_rule
from stylist.style_policy import apply_style_policy_to_semantic_program

logger = logging.getLogger(__name__)

_RESPONSE_MIME_TYPE = "application/json"
_LLM_TEMPERATURE = 0.2
_CLUSTER_SEMANTICS_TEMPERATURE_CAP = 0.1
_LLM_TOP_P = 0.9
_CLUSTER_CANDIDATE_CAP = 12
_RELATION_INTENT_CAP_PER_CLUSTER = 6
_ZONE_CLAIM_CAP_PER_CLUSTER = 4
_GLOBAL_RELATION_CAP = 20
_DEGRADATION_STEPS_CAP_PER_CLUSTER = 5
_ADAPTIVE_SEMANTIC_LLM_ENV = "TKNT_SEMANTIC_ADAPTIVE_LLM"
_REQUEST_CONTRACT_LLM_ENV = "TKNT_REQUEST_CONTRACT_LLM"
_ADAPTIVE_STAGE_NOTE = (
    "Adaptive semantic LLM mode enabled via "
    f"{_ADAPTIVE_SEMANTIC_LLM_ENV}; deterministic stage fallbacks are disabled."
)
_REQUEST_CONTRACT_STAGE_NOTE = (
    "Request contract LLM mode enabled via "
    f"{_REQUEST_CONTRACT_LLM_ENV}; deterministic validation remains active."
)


@dataclass(frozen=True)
class _AdaptiveStageSpec:
    model_config_keys: tuple[str, str, str]
    default_model_chain: tuple[str, str, str]
    system_prompt: str
    task_prompt: str
    output_contract: str
    repair_prompt: str


@dataclass(frozen=True)
class _RoomRulePlan:
    rules: dict[str, Any]
    trace: dict[str, Any]


_ADAPTIVE_STAGE_SPECS: dict[str, _AdaptiveStageSpec] = {
    "adaptive_request_contract": _AdaptiveStageSpec(
        model_config_keys=(
            "planner_adaptive_request_contract_primary",
            "planner_adaptive_request_contract_fallback_primary",
            "planner_adaptive_request_contract_fallback_secondary",
        ),
        default_model_chain=(
            "gemini-3.1-flash-lite-preview",
            "gemini-2.5-flash-lite",
            "gemini-3-flash-preview",
        ),
        system_prompt=(
            "You extract executable request intent for an interior layout pipeline. "
            "Return exactly one JSON object. No markdown. No prose outside JSON. "
            "Only encode objects explicitly mentioned or clearly referred to in the brief. "
            "Do not invent furniture just because it is common for the room type."
        ),
        task_prompt=(
            "Read the brief and produce a request contract for downstream deterministic tier-count. "
            "Classify each explicitly requested object as hard required, viable target, optional surplus, or forbidden. "
            "Use `must_keep` only when the user directly asks to include/need/use the object without fit-dependent wording. "
            "Use `target_if_viable` for wording such as if possible, if it fits, one or two where possible, or balanced composition goals. "
            "Use `optional_if_surplus` for only-if-space/surplus wording. "
            "Use `max0` for explicit no/avoid/without wording. "
            "Prefer no group over a weak group; include groups only when the brief describes a composition or functional pairing."
        ),
        output_contract=(
            "Top-level keys: `objects`, `groups`, `notes`. "
            "Each object must contain `object_type`, `intent`, `min_keep`, `target_count`, `max_keep`, `evidence`, `reason`, and `confidence`. "
            "`intent` must be one of `must_keep`, `target_if_viable`, `optional_if_surplus`, or `max0`. "
            "`evidence` must be a short exact phrase from the brief. "
            "`min_keep` is a hard floor; set it to 0 for `target_if_viable` and `optional_if_surplus`. "
            "`target_count` is the desired count when spatially viable. "
            "`max_keep` is the upper bound implied by the brief, or equal to `target_count` when no range is implied. "
            "Use only object types from `program_object_types` or `inventory_object_types`, unless the brief explicitly names a common synonym. "
            "Groups are optional and each group must contain `group_id`, `members`, `intent`, `priority`, and `drop_policy`."
        ),
        repair_prompt=(
            "Return exactly one valid JSON object with top-level keys `objects`, `groups`, and `notes`. "
            "No markdown. Do not include objects without evidence from the brief."
        ),
    ),
    "adaptive_room_rule": _AdaptiveStageSpec(
        model_config_keys=(
            "planner_adaptive_room_rule_primary",
            "planner_adaptive_room_rule_fallback_primary",
            "planner_adaptive_room_rule_fallback_secondary",
        ),
        default_model_chain=(
            "gemini-3-flash-preview",
            "gemini-3.1-flash-lite-preview",
            "gemini-2.5-flash-lite",
        ),
        system_prompt=(
            "You adapt semantic room rules for a deterministic interior layout pipeline. "
            "Return exactly one JSON object. No markdown. No prose outside JSON. "
            "Preserve schema validity and prefer minimal solver-friendly edits."
        ),
        task_prompt=(
            "Adapt the canonical room rule to this exact room. "
            "Keep existing cluster ids when possible. "
            "Add a new cluster only when the brief, affordance, and inventory clearly require it. "
            "Prefer the smallest object program that still covers the room's likely use. "
            "Set `tier_count_hints` as an executable keep/drop policy for downstream quantity selection, not as abstract commentary."
        ),
        output_contract=(
            "Top-level keys: `room_rule`, `notes`. "
            "`room_rule` must be a full replacement object, not a patch. "
            "`room_rule` must contain `room_type`, `policy`, `global_program`, and `clusters`. "
            "Each cluster must contain `cluster_id`, `priority`, `activation`, `object_program`, `semantic`, `degradation_hints`, and `tier_count_hints`. "
            "`tier_count_hints` must be numerically usable by a deterministic tier-count step: preserve only what the room can realistically support, but do not force early drops when real spatial surplus exists. "
            "`tier_count_hints.object_hints` should include one entry for every object type named anywhere in that cluster's `object_program`; use cluster-level defaults for neutral object-level hints instead of omitting them. "
            "Use only object types already present in `inventory_summary` or `canonical_room_rule`. "
            "Do not omit a required key just because the value is unchanged."
        ),
        repair_prompt=(
            "Return exactly one valid JSON object with top-level keys `room_rule` and `notes`. "
            "No markdown. Do not omit required fields."
        ),
    ),
    "adaptive_candidate_overrides": _AdaptiveStageSpec(
        model_config_keys=(
            "planner_adaptive_candidate_overrides_primary",
            "planner_adaptive_candidate_overrides_fallback_primary",
            "planner_adaptive_candidate_overrides_fallback_secondary",
        ),
        default_model_chain=(
            "gemini-3.1-flash-lite-preview",
            "gemini-2.5-flash-lite",
            "gemini-3-flash-preview",
        ),
        system_prompt=(
            "You produce executable cluster candidate overrides for a deterministic planner. "
            "Return exactly one JSON object. No markdown. No prose outside JSON. "
            "Be conservative, solver-friendly, and easy for downstream validators. "
            "These outputs directly steer deterministic quantity and drop-order decisions."
        ),
        task_prompt=(
            "For every cluster, decide the executable object program and the tier-count recommendation for this exact room. "
            "Treat `useful` as whether the cluster should realistically survive deterministic filtering. "
            "Use `tier_count_hints` to say what should be kept longer, what can drop earlier, and what should survive when the room still has real surplus. "
            "Base this on the actual room area, affordance headroom, brief priorities, and inventory mix. "
            "Avoid over-pruning small support items when the room still has comfortable surplus."
        ),
        output_contract=(
            "Top-level keys: `candidate_overrides`, `notes`. "
            "Return one entry for every cluster id in `room_rule_clusters`. "
            "Each entry must contain `cluster_id`, `brief_support`, `useful`, and `active_by_rule`. "
            "`object_program` is optional: include it only when changing the current room-rule cluster program; if included, it must be a full replacement object program, not a patch. "
            "`brief_support` must be a float from 0.0 to 1.0. "
            "`tier_count_hints` is optional: include it only when changing the current room-rule keep/drop policy. "
            "When included, `tier_count_hints` must contain `bundle_class`, `preserve_level`, `keep_if_space_surplus`, `space_surplus_threshold`, `drop_order_bias`, and `object_hints`. "
            "Each object hint must contain `object_type`, `min_keep`, `max_keep`, `keep_if_space_surplus`, `space_surplus_threshold`, `drop_order_bias`, `preserve_level`, and `preferred_size_tier`. "
            "`object_hints` should include one entry for every object type named anywhere in that override's `object_program`; use cluster-level defaults for neutral object-level hints instead of omitting them. "
            "Keep the hints executable and internally consistent: `min_keep <= max_keep` when `max_keep` is present, and `keep_if_space_surplus` should only be true when the room plausibly has usable headroom. "
            "Do not omit a required key just because the value is unchanged."
        ),
        repair_prompt=(
            "Return exactly one valid JSON object with top-level keys `candidate_overrides` and `notes`. "
            "Include one override for every cluster id. No markdown."
        ),
    ),
    "adaptive_cluster_semantics": _AdaptiveStageSpec(
        model_config_keys=(
            "planner_adaptive_cluster_semantics_primary",
            "planner_adaptive_cluster_semantics_fallback_primary",
            "planner_adaptive_cluster_semantics_fallback_secondary",
        ),
        default_model_chain=(
            "gemini-3.1-flash-lite-preview",
            "gemini-3-flash-preview",
            "gemini-2.5-flash-lite",
        ),
        system_prompt=(
            "You assign compact semantic intent for already-active clusters in a deterministic interior layout pipeline. "
            "Return exactly one JSON object. No markdown. No prose outside JSON. "
            "Prefer short, precise, solver-friendly semantics. "
            "Treat functional room facts such as openings, daylight, circulation, usable walls, privacy zones, and center openness as stable planning evidence; style may shape taste but must not erase spatial conflicts."
        ),
        task_prompt=(
            "For every active cluster, produce semantic intent only: `layout_role`, `semantic_role`, dominant anchors, zone claims, relation intents, and degradation ladder. "
            "Use concise constraints rather than creative language. "
            "When writing `zone_claims`, consider all relevant affordance groups in the payload, including entry landing zones, primary circulation corridors, daylight/window regions, usable wall anchors, privacy regions, focal surfaces, and center-openness regions. "
            "Encode only the relation you judge functionally relevant through existing `preferred_regions` and `avoid_regions`; leave neutral regions out instead of inventing labels."
        ),
        output_contract=(
            "Top-level keys: `cluster_semantics`, `notes`. "
            "Return one entry for every active cluster. "
            "Each entry must contain `cluster_id`, `layout_role`, `semantic_role`, `dominant_anchor_candidates`, `zone_claims`, `relation_intents`, and `degradation_ladder`. "
            "`layout_role` must be one of `primary`, `secondary`, `support`, or `optional`. "
            "`zone_claims.preferred_regions` and `zone_claims.avoid_regions` must be deliberate judgements from the provided affordance_summary, not copied mechanically from the fallback program. "
            "Do not omit an opening, daylight, circulation, entry, wall, privacy, focal, or center-openness region when it is functionally relevant to the cluster's anchors, access needs, or visual blocking behavior. "
            "Use only affordance region labels that already exist in `affordance_summary`. "
            "Use only active cluster ids in `relation_intents.target_cluster`. "
            "Do not omit a required key just because the value is unchanged."
        ),
        repair_prompt=(
            "Return exactly one valid JSON object with top-level keys `cluster_semantics` and `notes`. "
            "Include one entry for every active cluster. No markdown."
        ),
    ),
}

_MEDIA_TOKENS = ("tv", "media", "screen", "projector", "console")
_WORK_TOKENS = ("work", "study", "desk", "office", "laptop", "computer")
_READING_TOKENS = ("read", "reading", "lounge", "relax", "armchair")
_PET_TOKENS = ("pet", "dog", "cat")
_ENTRY_TOKENS = ("entry", "shoe", "foyer", "landing")
_LAUNDRY_TOKENS = ("laundry", "hamper", "daily storage", "basket")
_LAYOUT_ROLES = {"primary", "secondary", "support", "optional"}

_STRING_SCHEMA: dict[str, object] = {"type": "STRING"}
_NUMBER_SCHEMA: dict[str, object] = {"type": "NUMBER"}
_BOOLEAN_SCHEMA: dict[str, object] = {"type": "BOOLEAN"}
_STRING_ARRAY_SCHEMA: dict[str, object] = {
    "type": "ARRAY",
    "items": _STRING_SCHEMA,
}
_STRING_GROUP_ARRAY_SCHEMA: dict[str, object] = {
    "type": "ARRAY",
    "items": {
        "type": "ARRAY",
        "items": _STRING_SCHEMA,
    },
}
_OBJECT_PROGRAM_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "OBJECT",
    "properties": {
        "required": _STRING_ARRAY_SCHEMA,
        "required_if_kept": _STRING_ARRAY_SCHEMA,
        "optional": _STRING_ARRAY_SCHEMA,
        "choose_exactly_one_from": _STRING_GROUP_ARRAY_SCHEMA,
        "choose_exactly_one_from_if_kept": _STRING_GROUP_ARRAY_SCHEMA,
        "choose_at_least_one_from": _STRING_GROUP_ARRAY_SCHEMA,
        "optional_limits": {"type": "OBJECT"},
    },
}
_TIER_COUNT_HINT_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "OBJECT",
    "properties": {
        "bundle_class": _STRING_SCHEMA,
        "preserve_level": _STRING_SCHEMA,
        "keep_if_space_surplus": _BOOLEAN_SCHEMA,
        "space_surplus_threshold": _NUMBER_SCHEMA,
        "drop_order_bias": _STRING_SCHEMA,
        "object_hints": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "object_type": _STRING_SCHEMA,
                    "min_keep": _NUMBER_SCHEMA,
                    "max_keep": _NUMBER_SCHEMA,
                    "keep_if_space_surplus": _BOOLEAN_SCHEMA,
                    "space_surplus_threshold": _NUMBER_SCHEMA,
                    "drop_order_bias": _STRING_SCHEMA,
                    "preserve_level": _STRING_SCHEMA,
                    "preferred_size_tier": _STRING_SCHEMA,
                },
                "required": [
                    "object_type",
                    "min_keep",
                    "keep_if_space_surplus",
                    "space_surplus_threshold",
                    "drop_order_bias",
                    "preserve_level",
                ],
            },
        },
    },
}
_ADAPTIVE_REQUEST_CONTRACT_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "OBJECT",
    "properties": {
        "objects": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "object_type": _STRING_SCHEMA,
                    "intent": {
                        "type": "STRING",
                        "enum": [
                            "must_keep",
                            "target_if_viable",
                            "optional_if_surplus",
                            "max0",
                        ],
                    },
                    "min_keep": _NUMBER_SCHEMA,
                    "target_count": _NUMBER_SCHEMA,
                    "max_keep": _NUMBER_SCHEMA,
                    "evidence": _STRING_SCHEMA,
                    "reason": _STRING_SCHEMA,
                    "confidence": _NUMBER_SCHEMA,
                },
                "required": [
                    "object_type",
                    "intent",
                    "min_keep",
                    "target_count",
                    "max_keep",
                    "evidence",
                    "reason",
                    "confidence",
                ],
            },
        },
        "groups": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "group_id": _STRING_SCHEMA,
                    "members": _STRING_ARRAY_SCHEMA,
                    "intent": _STRING_SCHEMA,
                    "priority": _STRING_SCHEMA,
                    "drop_policy": _STRING_SCHEMA,
                },
                "required": [
                    "group_id",
                    "members",
                    "intent",
                    "priority",
                    "drop_policy",
                ],
            },
        },
        "notes": _STRING_ARRAY_SCHEMA,
    },
    "required": ["objects", "groups", "notes"],
}
_ADAPTIVE_CANDIDATE_OVERRIDES_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "OBJECT",
    "properties": {
        "candidate_overrides": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "cluster_id": _STRING_SCHEMA,
                    "object_program": _OBJECT_PROGRAM_RESPONSE_SCHEMA,
                    "brief_support": _NUMBER_SCHEMA,
                    "useful": _BOOLEAN_SCHEMA,
                    "active_by_rule": _BOOLEAN_SCHEMA,
                    "tier_count_hints": _TIER_COUNT_HINT_RESPONSE_SCHEMA,
                },
                "required": [
                    "cluster_id",
                    "brief_support",
                    "useful",
                    "active_by_rule",
                ],
            },
        },
        "notes": _STRING_ARRAY_SCHEMA,
    },
    "required": ["candidate_overrides", "notes"],
}
_ADAPTIVE_CLUSTER_SEMANTICS_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "OBJECT",
    "properties": {
        "cluster_semantics": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "cluster_id": _STRING_SCHEMA,
                    "layout_role": {
                        "type": "STRING",
                        "enum": ["primary", "secondary", "support", "optional"],
                    },
                    "semantic_role": _STRING_SCHEMA,
                    "dominant_anchor_candidates": _STRING_ARRAY_SCHEMA,
                    "zone_claims": {
                        "type": "OBJECT",
                        "properties": {
                            "preferred_regions": _STRING_ARRAY_SCHEMA,
                            "avoid_regions": _STRING_ARRAY_SCHEMA,
                            "wall_affinity": _STRING_SCHEMA,
                            "daylight_affinity": _STRING_SCHEMA,
                            "privacy_affinity": _STRING_SCHEMA,
                            "floating_allowed": _BOOLEAN_SCHEMA,
                        },
                    },
                    "relation_intents": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "type": _STRING_SCHEMA,
                                "target_cluster": _STRING_SCHEMA,
                                "strength": _STRING_SCHEMA,
                            },
                        },
                    },
                    "degradation_ladder": _STRING_ARRAY_SCHEMA,
                },
                "required": [
                    "cluster_id",
                    "layout_role",
                    "semantic_role",
                    "dominant_anchor_candidates",
                    "zone_claims",
                    "relation_intents",
                    "degradation_ladder",
                ],
            },
        },
        "notes": _STRING_ARRAY_SCHEMA,
    },
    "required": ["cluster_semantics", "notes"],
}


@dataclass(frozen=True)
class SemanticLayoutPlanner:
    system_prompt: str = SEMANTIC_LAYOUT_PLANNER_SYSTEM_PROMPT
    prompt_template: str = SEMANTIC_LAYOUT_PLANNER_USER_PROMPT

    def generate(
        self,
        *,
        room_model_json: Mapping[str, Any],
        room_type: str,
        brief_text: str,
        inventory_catalog: Sequence[Mapping[str, Any]] | None = None,
        semantic_program_rules: Mapping[str, Any] | None = None,
        style_policy: Mapping[str, Any] | None = None,
        use_llm: bool = True,
        temperature: float = _LLM_TEMPERATURE,
        top_p: float = _LLM_TOP_P,
        max_tokens: int | None = None,
    ) -> SemanticLayoutProgram:
        room_rule_plan = _room_rule_plan_for(
            room_type=room_type,
            semantic_program_rules=semantic_program_rules,
        )
        room_rules = room_rule_plan.rules
        affordance_summary = summarize_room_affordance(room_model_json)
        inventory = _normalize_inventory_catalog(inventory_catalog, room_rules)
        adaptive_llm_mode = use_llm and _adaptive_semantic_llm_enabled()
        adaptive_notes: list[str] = []

        if adaptive_llm_mode:
            logger.info(
                "SemanticLayoutPlanner adaptive LLM mode enabled via %s",
                _ADAPTIVE_SEMANTIC_LLM_ENV,
            )
            room_rules, room_rule_notes = self._generate_adaptive_room_rules(
                room_type=room_type,
                brief_text=brief_text,
                room_rules=room_rules,
                affordance_summary=affordance_summary,
                inventory_catalog=inventory,
                style_policy=style_policy,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            adaptive_notes.extend(room_rule_notes)
            inventory = _normalize_inventory_catalog(inventory_catalog, room_rules)
            candidate_overrides, candidate_notes = (
                self._generate_adaptive_candidate_overrides(
                    room_type=room_type,
                    brief_text=brief_text,
                    room_rules=room_rules,
                    affordance_summary=affordance_summary,
                    inventory_catalog=inventory,
                    style_policy=style_policy,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
            )
            adaptive_notes.extend(candidate_notes)
            candidates = build_cluster_candidates(
                room_rules=room_rules,
                inventory_catalog=inventory,
                brief_text=brief_text,
                room_model_json=room_model_json,
                llm_candidate_overrides=candidate_overrides,
            )
        else:
            candidates = build_cluster_candidates(
                room_rules=room_rules,
                inventory_catalog=inventory,
                brief_text=brief_text,
                room_model_json=room_model_json,
            )
        deterministic_program = _build_deterministic_program(
            room_type=room_type,
            room_rules=room_rules,
            room_model_json=room_model_json,
            affordance_summary=affordance_summary,
            candidates=candidates,
            brief_text=brief_text,
            inventory_catalog=inventory,
        )
        deterministic_program["profile_rule_trace"] = room_rule_plan.trace
        deterministic_program = self._attach_llm_request_contract_if_enabled(
            deterministic_program=deterministic_program,
            room_type=room_type,
            brief_text=brief_text,
            room_model_json=room_model_json,
            inventory_catalog=inventory,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        if adaptive_llm_mode:
            if deterministic_program.get("active_clusters"):
                deterministic_program, cluster_notes = (
                    self._apply_llm_cluster_semantics(
                        deterministic_program=deterministic_program,
                        room_type=room_type,
                        brief_text=brief_text,
                        affordance_summary=affordance_summary,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                    )
                )
                adaptive_notes.extend(cluster_notes)
            deterministic_program["notes"] = _merge_notes(
                [_ADAPTIVE_STAGE_NOTE, *adaptive_notes],
                deterministic_program.get("notes"),
            )
            deterministic_program = apply_style_policy_to_semantic_program(
                deterministic_program,
                style_policy,
            )
            deterministic_program = _normalize_profile_semantic_program(
                deterministic_program,
                affordance_summary=affordance_summary,
            )
            return SemanticLayoutProgram.model_validate(deterministic_program)

        deterministic_program = apply_style_policy_to_semantic_program(
            deterministic_program,
            style_policy,
        )
        deterministic_program = _normalize_profile_semantic_program(
            deterministic_program,
            affordance_summary=affordance_summary,
        )

        if not use_llm or not candidates:
            return SemanticLayoutProgram.model_validate(deterministic_program)

        try:
            llm_payload = self._generate_llm_payload(
                room_type=room_type,
                brief_text=brief_text,
                room_rules=room_rules,
                affordance_summary=affordance_summary,
                candidates=candidates,
                deterministic_program=deterministic_program,
                style_policy=style_policy,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            logger.warning(
                "SemanticLayoutPlanner LLM pass failed; using deterministic program: %s",
                exc,
            )
            return SemanticLayoutProgram.model_validate(deterministic_program)

        merged = _validate_and_normalize_llm_program(
            llm_payload=llm_payload,
            deterministic_program=deterministic_program,
            room_type=room_type,
            room_rules=room_rules,
            candidates=candidates,
            affordance_summary=affordance_summary,
        )
        merged = apply_style_policy_to_semantic_program(merged, style_policy)
        merged = _normalize_profile_semantic_program(
            merged,
            affordance_summary=affordance_summary,
        )
        return SemanticLayoutProgram.model_validate(merged)

    def _generate_llm_payload(
        self,
        *,
        room_type: str,
        brief_text: str,
        room_rules: Mapping[str, Any],
        affordance_summary: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
        deterministic_program: Mapping[str, Any],
        style_policy: Mapping[str, Any] | None,
        temperature: float,
        top_p: float,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        prompt_payload = {
            "room_type": room_type,
            "brief": {"text": brief_text},
            "room_affordance_summary": affordance_summary,
            "canonical_global_rules": room_rules.get("global_program") or {},
            "policy": room_rules.get("policy") or {},
            "style_policy": dict(style_policy or {}),
            "candidate_clusters": list(candidates)[:_CLUSTER_CANDIDATE_CAP],
            "deterministic_seed_program": {
                "active_clusters": deterministic_program.get("active_clusters") or [],
                "global_layout_intent": deterministic_program.get(
                    "global_layout_intent"
                )
                or {},
                "macro_relations": deterministic_program.get("macro_relations") or {},
            },
            "limits": {
                "relation_intent_cap_per_cluster": _RELATION_INTENT_CAP_PER_CLUSTER,
                "zone_claim_cap_per_cluster": _ZONE_CLAIM_CAP_PER_CLUSTER,
                "global_relation_cap": _GLOBAL_RELATION_CAP,
                "degradation_steps_cap_per_cluster": _DEGRADATION_STEPS_CAP_PER_CLUSTER,
                "llm_temperature": temperature,
                "llm_top_p": top_p,
            },
        }
        user_prompt = self.prompt_template.replace(
            "{PAYLOAD_JSON}",
            json.dumps(prompt_payload, ensure_ascii=True, indent=2),
        )
        messages: list[ChatMessage] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        client = get_llm_client()
        model_name = TextLLMConfig.agent_model("forge") or TextLLMConfig.agent_model(
            "planner"
        )
        last_raw = ""
        for attempt in range(2):
            response = client.chat_completion(
                messages,
                model_key="primary",
                model_name=model_name,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                response_mime_type=_RESPONSE_MIME_TYPE,
            )
            last_raw = _extract_content(response)
            try:
                return _parse_json(last_raw)
            except ValueError:
                if attempt == 1:
                    raise
                _record_llm_retry(
                    stage="semantic_layout_planner.main",
                    model_name=model_name,
                    reason="invalid_json",
                )
                messages.append({"role": "assistant", "content": last_raw})
                messages.append(
                    {
                        "role": "user",
                        "content": "Return valid strict JSON only, with no markdown or prose.",
                    }
                )
        raise ValueError("SemanticLayoutPlanner returned invalid JSON")

    def _attach_llm_request_contract_if_enabled(
        self,
        *,
        deterministic_program: Mapping[str, Any],
        room_type: str,
        brief_text: str,
        room_model_json: Mapping[str, Any],
        inventory_catalog: Mapping[str, Mapping[str, Any]],
        temperature: float,
        top_p: float,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        out = deepcopy(dict(deterministic_program))
        if not _request_contract_llm_enabled():
            return out

        program_object_types = _request_contract_program_object_types(out)
        try:
            prompt_payload = {
                "stage": "adaptive_request_contract",
                "room_type": room_type,
                "brief": {"text": brief_text},
                "room_summary": {
                    "area_m2": round(_room_area_m2(room_model_json), 2),
                },
                "program_object_types": program_object_types,
                "inventory_object_types": sorted(inventory_catalog),
                "inventory_summary": _inventory_summary_for_llm(inventory_catalog),
                "active_clusters": out.get("active_clusters") or [],
            }
            response = self._call_adaptive_stage(
                stage_name="adaptive_request_contract",
                prompt_payload=prompt_payload,
                temperature=min(float(temperature), 0.1),
                top_p=top_p,
                max_tokens=max_tokens,
                response_validator=lambda payload: sanitize_request_contract(
                    payload,
                    brief_text=brief_text,
                    available_object_types=program_object_types,
                    fallback_to_heuristic=False,
                ),
            )
            contract = sanitize_request_contract(
                response,
                brief_text=brief_text,
                available_object_types=program_object_types,
                fallback_to_heuristic=True,
            )
            contract["source"] = "llm_request_contract"
            notes = _merge_notes(
                [_REQUEST_CONTRACT_STAGE_NOTE],
                out.get("notes"),
            )
        except Exception as exc:
            logger.warning(
                "Adaptive request contract failed; using heuristic fallback: %s",
                exc,
            )
            contract = build_request_contract(
                brief_text=brief_text,
                available_object_types=program_object_types,
            )
            notes = _merge_notes(
                [
                    (
                        "Request contract LLM failed; heuristic request contract "
                        "fallback was used."
                    )
                ],
                out.get("notes"),
            )

        out["request_contract"] = contract
        out["notes"] = notes
        return out

    def _generate_adaptive_room_rules(
        self,
        *,
        room_type: str,
        brief_text: str,
        room_rules: Mapping[str, Any],
        affordance_summary: Mapping[str, Any],
        inventory_catalog: Mapping[str, Mapping[str, Any]],
        style_policy: Mapping[str, Any] | None,
        temperature: float,
        top_p: float,
        max_tokens: int | None,
    ) -> tuple[dict[str, Any], list[str]]:
        prompt_payload = {
            "stage": "adaptive_room_rule",
            "room_type": room_type,
            "brief": {"text": brief_text},
            "affordance_summary": affordance_summary,
            "inventory_summary": _inventory_summary_for_llm(inventory_catalog),
            "style_policy": dict(style_policy or {}),
            "canonical_room_rule": room_rules,
        }
        response = self._call_adaptive_stage(
            stage_name="adaptive_room_rule",
            prompt_payload=prompt_payload,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            response_validator=lambda payload: _sanitize_adaptive_room_rule(
                payload.get("room_rule"),
                room_type=room_type,
            ),
        )
        room_rule = _sanitize_adaptive_room_rule(
            response.get("room_rule"),
            room_type=room_type,
        )
        return room_rule, _sanitize_notes(response.get("notes"))

    def _generate_adaptive_candidate_overrides(
        self,
        *,
        room_type: str,
        brief_text: str,
        room_rules: Mapping[str, Any],
        affordance_summary: Mapping[str, Any],
        inventory_catalog: Mapping[str, Mapping[str, Any]],
        style_policy: Mapping[str, Any] | None,
        temperature: float,
        top_p: float,
        max_tokens: int | None,
    ) -> tuple[dict[str, dict[str, Any]], list[str]]:
        cluster_rules_by_id = {
            str(cluster.get("cluster_id") or "").strip(): dict(cluster)
            for cluster in room_rules.get("clusters") or []
            if isinstance(cluster, Mapping)
            and str(cluster.get("cluster_id") or "").strip()
        }
        prompt_payload = {
            "stage": "adaptive_candidate_overrides",
            "room_type": room_type,
            "brief": {"text": brief_text},
            "affordance_summary": affordance_summary,
            "inventory_summary": _inventory_summary_for_llm(inventory_catalog),
            "style_policy": dict(style_policy or {}),
            "room_rule_clusters": room_rules.get("clusters") or [],
        }
        response = self._call_adaptive_stage(
            stage_name="adaptive_candidate_overrides",
            prompt_payload=prompt_payload,
            temperature=min(float(temperature), 0.1),
            top_p=top_p,
            max_tokens=max_tokens,
            response_validator=lambda payload: _sanitize_candidate_overrides(
                payload.get("candidate_overrides"),
                expected_clusters=cluster_rules_by_id,
            ),
        )
        overrides = _sanitize_candidate_overrides(
            response.get("candidate_overrides"),
            expected_clusters=cluster_rules_by_id,
        )
        return overrides, _sanitize_notes(response.get("notes"))

    def _apply_llm_cluster_semantics(
        self,
        *,
        deterministic_program: Mapping[str, Any],
        room_type: str,
        brief_text: str,
        affordance_summary: Mapping[str, Any],
        temperature: float,
        top_p: float,
        max_tokens: int | None,
    ) -> tuple[dict[str, Any], list[str]]:
        prompt_payload = {
            "stage": "adaptive_cluster_semantics",
            "room_type": room_type,
            "brief": {"text": brief_text},
            "affordance_summary": affordance_summary,
            "active_clusters": deterministic_program.get("active_clusters") or [],
        }
        response = self._call_adaptive_stage(
            stage_name="adaptive_cluster_semantics",
            prompt_payload=prompt_payload,
            temperature=min(float(temperature), _CLUSTER_SEMANTICS_TEMPERATURE_CAP),
            top_p=top_p,
            max_tokens=max_tokens,
            response_validator=lambda payload: _sanitize_cluster_semantics(
                payload.get("cluster_semantics"),
                deterministic_program=deterministic_program,
                affordance_summary=affordance_summary,
            ),
        )
        merged = _sanitize_cluster_semantics(
            response.get("cluster_semantics"),
            deterministic_program=deterministic_program,
            affordance_summary=affordance_summary,
        )
        return merged, _sanitize_notes(response.get("notes"))

    def _call_adaptive_stage(
        self,
        *,
        stage_name: str,
        prompt_payload: Mapping[str, Any],
        temperature: float,
        top_p: float,
        max_tokens: int | None,
        response_validator: Callable[[dict[str, Any]], object] | None = None,
        allow_repair: bool = True,
    ) -> dict[str, Any]:
        stage_spec = _adaptive_stage_spec(stage_name)
        user_prompt = (
            f"Stage: {stage_name}\n\n"
            f"Task:\n{stage_spec.task_prompt}\n\n"
            f"Output Contract:\n{stage_spec.output_contract}\n\n"
            "Rules:\n"
            "- Return exactly one JSON object.\n"
            "- No markdown.\n"
            "- No commentary outside JSON.\n"
            "- Do not omit required keys.\n\n"
            "Payload:\n"
            f"{json.dumps(prompt_payload, ensure_ascii=True, indent=2)}"
        )
        base_messages: list[ChatMessage] = [
            {"role": "system", "content": stage_spec.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        client = get_llm_client()
        model_chain = _adaptive_stage_model_chain(stage_name)
        last_error: Exception | None = None
        for model_index, model_name in enumerate(model_chain):
            messages = list(base_messages)
            fallback_model_names = model_chain[model_index + 1 :]
            try:
                last_raw = ""
                for attempt in range(2):
                    response = client.chat_completion(
                        messages,
                        model_key="primary",
                        model_name=model_name,
                        fallback_model_names=fallback_model_names,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        response_mime_type=_RESPONSE_MIME_TYPE,
                        response_schema=_adaptive_stage_response_schema(stage_name),
                    )
                    last_raw = _extract_content(response)
                    try:
                        parsed = _parse_json(last_raw)
                        if response_validator is not None:
                            response_validator(parsed)
                        return parsed
                    except ValueError as exc:
                        if not allow_repair:
                            raise ValueError(
                                f"{stage_name} validation failed on {model_name}: {exc}"
                            ) from exc
                        if attempt == 1:
                            raise ValueError(
                                f"{stage_name} validation failed on {model_name}: {exc}"
                            ) from exc
                        _record_llm_retry(
                            stage=f"semantic_layout_planner.{stage_name}",
                            model_name=model_name,
                            reason="invalid_json_or_schema",
                        )
                        if _is_invalid_json_error(exc):
                            messages = list(base_messages)
                            messages.append(
                                {
                                    "role": "user",
                                    "content": (
                                        f"{stage_spec.repair_prompt}\n"
                                        f"Previous response was invalid JSON: {exc}. "
                                        "Regenerate the complete response from the payload. "
                                        "Keep the JSON compact and omit optional unchanged fields."
                                    ),
                                }
                            )
                        else:
                            messages.append({"role": "assistant", "content": last_raw})
                            messages.append(
                                {
                                    "role": "user",
                                    "content": (
                                        f"{stage_spec.repair_prompt}\n"
                                        f"Validation error: {exc}"
                                    ),
                                }
                            )
            except Exception as exc:
                last_error = exc
                if model_index == len(model_chain) - 1:
                    break
                logger.warning(
                    "Adaptive stage %s failed on model %s; retrying with %s: %s",
                    stage_name,
                    model_name,
                    model_chain[model_index + 1],
                    exc,
                )
        if last_error is not None:
            raise last_error
        raise ValueError(f"{stage_name} produced no response")


def build_cluster_candidates(
    *,
    room_rules: Mapping[str, Any],
    inventory_catalog: Mapping[str, Mapping[str, Any]],
    brief_text: str,
    room_model_json: Mapping[str, Any],
    llm_candidate_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    clusters = room_rules.get("clusters")
    if not isinstance(clusters, list):
        return []

    candidates: list[dict[str, Any]] = []
    for cluster in clusters[:_CLUSTER_CANDIDATE_CAP]:
        if not isinstance(cluster, Mapping):
            continue
        candidate = _build_single_candidate(
            cluster_rule=cluster,
            inventory_catalog=inventory_catalog,
            brief_text=brief_text,
            room_model_json=room_model_json,
            llm_candidate_override=(
                llm_candidate_overrides.get(
                    str(cluster.get("cluster_id") or "").strip()
                )
                if llm_candidate_overrides is not None
                else None
            ),
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def summarize_room_affordance(room_model_json: Mapping[str, Any]) -> dict[str, Any]:
    affordance = room_model_json.get("affordance_map")
    affordance = affordance if isinstance(affordance, Mapping) else {}
    room = (
        room_model_json.get("room")
        if isinstance(room_model_json.get("room"), Mapping)
        else {}
    )
    openings = (
        room_model_json.get("openings")
        if isinstance(room_model_json.get("openings"), Mapping)
        else {}
    )
    doors = openings.get("doors") if isinstance(openings.get("doors"), list) else []
    windows = (
        openings.get("windows") if isinstance(openings.get("windows"), list) else []
    )
    polygon = room.get("polygon_ccw") or room_model_json.get("polygon_mm") or []
    wall_count = max(0, len(polygon) if isinstance(polygon, list) else 0)

    return {
        "usable_wall_segments": _region_labels(
            affordance.get("usable_wall_segments"),
            fallback=[f"wall_segment_{index}" for index in range(1, wall_count + 1)],
        ),
        "entry_landing_zones": _region_labels(
            affordance.get("entry_landing_zones"),
            fallback=_opening_labels(doors, "entry_landing_zone"),
        ),
        "primary_circulation_corridors": _region_labels(
            affordance.get("primary_circulation_corridors"),
            fallback=["entry_to_center_corridor"] if doors else ["center_access_lane"],
        ),
        "daylight_regions": _region_labels(
            affordance.get("daylight_regions"),
            fallback=_opening_labels(windows, "daylight_region"),
        ),
        "privacy_regions": _region_labels(
            affordance.get("privacy_regions"),
            fallback=["private_back_zone"] if doors else ["quiet_wall_zone"],
        ),
        "focal_surfaces": _region_labels(
            affordance.get("focal_surfaces"),
            fallback=["primary_focal_wall"] if wall_count else [],
        ),
        "center_openness_regions": _region_labels(
            affordance.get("center_openness_regions"),
            fallback=["room_center"],
        ),
        "wall_anchor_candidates": _region_labels(
            affordance.get("wall_anchor_candidates"),
            fallback=["quiet_wall_zone", "long_wall_zone"] if wall_count else [],
        ),
        "floating_zone_candidates": _region_labels(
            affordance.get("floating_zone_candidates"),
            fallback=["central_floating_zone"],
        ),
        "room_area_m2": round(_room_area_m2(room_model_json), 2),
    }


def semantic_program_to_cluster_forge_payload(
    semantic_program: Mapping[str, Any],
) -> dict[str, Any]:
    clusters: list[dict[str, Any]] = []
    for active_cluster in semantic_program.get("active_clusters") or []:
        if not isinstance(active_cluster, Mapping):
            continue
        cluster_id = str(active_cluster.get("cluster_id") or "").strip()
        if not cluster_id:
            continue
        members = _objects_from_active_cluster(active_cluster)
        if not members:
            continue
        dominant_anchor_candidates = _dominant_anchor_candidates_from_active_cluster(
            active_cluster,
            members,
        )
        anchors = _anchors_from_active_cluster(
            active_cluster,
            members,
            dominant_anchor_candidates=dominant_anchor_candidates,
        )
        rules = {
            "grid_mm": GLOBAL_LAYOUT_GRID_MM,
            "allowed_rotations": {member: [0, 90, 180, 270] for member in members},
            "facing": {
                member: {"front": "top", "notes": "functional front"}
                for member in members
                if _needs_front_access(member)
            },
            "access_requirements": [
                {"id": member, "type": "front_clearance", "required": True}
                for member in members
                if _needs_front_access(member)
            ],
            "semantic_placements": semantic_placements_for_members(
                room_type=semantic_program.get("room_type"),
                cluster_id=cluster_id,
                members=members,
                anchors=anchors,
            ),
            "dominant_anchor_candidates": dominant_anchor_candidates,
            "allow_empty_cluster": active_cluster.get("priority") != "core",
            "zone_claims": active_cluster.get("zone_claims") or {},
            "layout_role": active_cluster.get("layout_role") or "support",
            "degradation_ladder": active_cluster.get("degradation_ladder") or [],
            "tier_count_hints": active_cluster.get("tier_count_hints") or {},
            "anchor_first_policy": _anchor_first_policy_for_active_cluster(
                active_cluster=active_cluster,
                members=members,
                anchors=anchors,
                dominant_anchor_candidates=dominant_anchor_candidates,
            ),
        }
        clusters.append(
            {
                "cluster_id": cluster_id,
                "tag": _cluster_tag(
                    cluster_id,
                    members,
                    room_type=semantic_program.get("room_type"),
                ),
                "members": members,
                "anchors": anchors,
                "hard_constraints": _hard_constraints_for_members(members),
                "soft_constraints": _soft_constraints_for_members(members, anchors),
                "cluster_rules": rules,
                "notes": [str(active_cluster.get("activation_reason") or "").strip()],
            }
        )
    status = "OK" if semantic_program.get("status") != "UNSAT" else "UNSAT"
    return {
        "status": status,
        "clusters": clusters,
        "semantic_layout_program": dict(semantic_program),
        "notes": list(semantic_program.get("notes") or []),
        "missing": list(semantic_program.get("missing") or []),
    }


def _anchor_first_policy_for_active_cluster(
    *,
    active_cluster: Mapping[str, Any],
    members: Sequence[str],
    anchors: Sequence[str],
    dominant_anchor_candidates: Sequence[str],
) -> dict[str, Any]:
    member_set = set(members)
    protected_ids = [
        item for item in (*dominant_anchor_candidates, *anchors) if item in member_set
    ]
    droppable_ids: list[str] = []
    for object_type in _objects_marked_optional(active_cluster):
        if object_type in member_set and object_type not in protected_ids:
            droppable_ids.append(object_type)
    for action in active_cluster.get("degradation_ladder") or []:
        object_type = _drop_action_object_type(action, members)
        if object_type in member_set and object_type not in protected_ids:
            droppable_ids.append(object_type)
    return {
        "dominant_anchor_id": next(iter(protected_ids), ""),
        "dominant_anchor_candidates": list(dominant_anchor_candidates),
        "placement_order": list(members),
        "protected_ids": _uniq(protected_ids),
        "droppable_ids": _uniq(droppable_ids),
    }


def _objects_marked_optional(active_cluster: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    for bundle in active_cluster.get("required_bundles") or []:
        if not isinstance(bundle, Mapping):
            continue
        objects = bundle.get("objects")
        if not isinstance(objects, Sequence) or isinstance(objects, str):
            continue
        for row in objects:
            if not isinstance(row, Mapping):
                continue
            object_type = str(row.get("object_type") or "").strip()
            if object_type and not bool(row.get("required", False)):
                out.append(object_type)
    return out


def _drop_action_object_type(
    action: object,
    member_candidates: Sequence[str] = (),
) -> str:
    text = str(action or "").strip()
    if not text.startswith("drop_"):
        return ""
    object_type = text.removeprefix("drop_")
    for suffix in ("_first", "_last"):
        if object_type.endswith(suffix):
            object_type = object_type[: -len(suffix)]
    if object_type in member_candidates:
        return object_type
    for candidate in member_candidates:
        if candidate.endswith(f"_{object_type}") or object_type.endswith(
            f"_{candidate}"
        ):
            return candidate
    return object_type


def _relation_intent_list(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _merge_relation_intents(
    existing: Sequence[Mapping[str, Any]],
    additions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for item in (*existing, *additions):
        intent = dict(item)
        key = (
            str(intent.get("type") or ""),
            str(intent.get("target") or ""),
            str(intent.get("target_cluster") or ""),
            str(intent.get("strength") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(intent)
        if len(out) >= _RELATION_INTENT_CAP_PER_CLUSTER:
            break
    return out


def _normalize_profile_semantic_program(
    program: Mapping[str, Any],
    *,
    affordance_summary: Mapping[str, Any],
) -> dict[str, Any]:
    out = deepcopy(dict(program))
    room_type = out.get("room_type")
    active_clusters = out.get("active_clusters")
    if not isinstance(active_clusters, list):
        return out

    normalized_clusters: list[Any] = []
    for cluster in active_clusters:
        if not isinstance(cluster, Mapping):
            normalized_clusters.append(cluster)
            continue
        normalized = dict(cluster)
        object_types = _objects_from_active_cluster(normalized)
        semantic_role = profile_semantic_role_for_objects(
            cluster_id=normalized.get("cluster_id"),
            object_types=object_types,
            priority=normalized.get("priority"),
            room_type=room_type,
        )
        layout_role = profile_layout_role_for_objects(
            cluster_id=normalized.get("cluster_id"),
            object_types=object_types,
            room_type=room_type,
        )
        zone_claims = profile_zone_claims_for_objects(
            cluster_id=normalized.get("cluster_id"),
            object_types=object_types,
            room_type=room_type,
            affordance_summary=affordance_summary,
        )
        if layout_role is not None:
            normalized["layout_role"] = layout_role
        if semantic_role is not None:
            normalized["semantic_role"] = semantic_role
        if zone_claims is not None:
            normalized["zone_claims"] = zone_claims
        normalized_relation_intents = profile_relation_intents_for_objects(
            cluster_id=normalized.get("cluster_id"),
            object_types=object_types,
            room_type=room_type,
        )
        if normalized_relation_intents:
            normalized["relation_intents"] = _merge_relation_intents(
                _relation_intent_list(normalized.get("relation_intents")),
                normalized_relation_intents,
            )
        normalized_clusters.append(normalized)
    out["active_clusters"] = normalized_clusters
    out["macro_relations"] = _build_macro_relations(
        active_clusters=[
            cluster for cluster in normalized_clusters if isinstance(cluster, Mapping)
        ],
        room_type=room_type,
        affordance_summary=affordance_summary,
    )
    return out


def _room_rules_for(
    *,
    room_type: str,
    semantic_program_rules: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return _room_rule_plan_for(
        room_type=room_type,
        semantic_program_rules=semantic_program_rules,
    ).rules


def _room_rule_plan_for(
    *,
    room_type: str,
    semantic_program_rules: Mapping[str, Any] | None,
) -> _RoomRulePlan:
    profile_rule = semantic_room_rule_for(room_type)
    legacy_rule = _legacy_room_rule_for(
        room_type=room_type,
        semantic_program_rules=semantic_program_rules,
    )
    selection = select_profile_room_rule(
        room_type=room_type,
        profile_rule=profile_rule,
        legacy_rule=legacy_rule,
    )
    return _RoomRulePlan(rules=selection.rule, trace=selection.trace())


def _legacy_room_rule_for(
    *,
    room_type: str,
    semantic_program_rules: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if semantic_program_rules is not None:
        if semantic_program_rules.get("room_type") == room_type:
            return dict(semantic_program_rules)
        rooms = semantic_program_rules.get("rooms")
        if isinstance(rooms, list):
            for item in rooms:
                if isinstance(item, Mapping) and item.get("room_type") == room_type:
                    return dict(item)
    return get_compiled_semantic_room_rule(room_type)


def _normalize_inventory_catalog(
    inventory_catalog: Sequence[Mapping[str, Any]] | None,
    room_rules: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if inventory_catalog is not None:
        for item in inventory_catalog:
            object_type = _clean_object_type(
                item.get("object_type") or item.get("type") or item.get("category")
            )
            if object_type is None:
                continue
            out[object_type] = {
                "object_type": object_type,
                "available": bool(item.get("available", True)),
                "size_profiles": _string_list(item.get("size_profiles")),
                "functional_tags": _string_list(item.get("functional_tags")),
            }
    if out:
        return out

    for object_type in _all_rule_object_types(room_rules):
        out[object_type] = {
            "object_type": object_type,
            "available": True,
            "size_profiles": ["S", "M", "L"],
            "functional_tags": [],
        }
    return out


def _build_single_candidate(
    *,
    cluster_rule: Mapping[str, Any],
    inventory_catalog: Mapping[str, Mapping[str, Any]],
    brief_text: str,
    room_model_json: Mapping[str, Any],
    llm_candidate_override: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    cluster_id = str(cluster_rule.get("cluster_id") or "").strip()
    priority = _priority(cluster_rule.get("priority"))
    if not cluster_id:
        return None
    object_program = _merged_object_program_for_brief(cluster_rule, brief_text)
    if isinstance(llm_candidate_override, Mapping):
        override_program = _sanitize_object_program(
            llm_candidate_override.get("object_program")
        )
        if _object_program_has_content(override_program):
            object_program = override_program
    objects, missing = _candidate_objects(
        object_program=object_program,
        cluster_rule=cluster_rule,
        inventory_catalog=inventory_catalog,
        priority=priority,
        brief_text=brief_text,
    )
    activation = (
        cluster_rule.get("activation")
        if isinstance(cluster_rule.get("activation"), Mapping)
        else {}
    )
    if isinstance(llm_candidate_override, Mapping):
        active_by_rule = bool(llm_candidate_override.get("active_by_rule"))
        brief_support = _float_0_1(
            llm_candidate_override.get("brief_support"),
            default=0.35,
        )
        useful = bool(llm_candidate_override.get("useful"))
    else:
        active_by_rule = bool(activation.get("always_consider")) or priority == "core"
        brief_support = _brief_support_score(cluster_id, objects, brief_text)
        room_area = _room_area_m2(room_model_json)
        useful = (
            active_by_rule
            or brief_support >= 0.45
            or (priority == "support" and room_area >= 12.0 and objects)
        )
    tier_count_hint_source = (
        llm_candidate_override.get("tier_count_hints")
        if isinstance(llm_candidate_override, Mapping)
        else cluster_rule.get("tier_count_hints")
    )
    tier_count_hints = _sanitize_tier_count_hints(
        tier_count_hint_source,
        object_types=[
            str(item.get("object_type") or "")
            for item in objects
            if isinstance(item, Mapping)
        ],
        default_bundle_class=_default_bundle_class_for_priority(priority),
        default_preserve_level=_default_preserve_level_for_priority(priority),
        required=False,
    )
    return {
        "cluster_id": cluster_id,
        "priority": priority,
        "activation": dict(activation),
        "active_by_rule": active_by_rule,
        "useful": useful,
        "object_program": object_program,
        "objects": objects,
        "missing": missing,
        "semantic": dict(cluster_rule.get("semantic") or {})
        if isinstance(cluster_rule.get("semantic"), Mapping)
        else {},
        "degradation_hints": dict(cluster_rule.get("degradation_hints") or {})
        if isinstance(cluster_rule.get("degradation_hints"), Mapping)
        else {},
        "tier_count_hints": tier_count_hints,
        "viability": {
            "rule_support": 1.0,
            "brief_support": brief_support,
            "inventory_support": 1.0 if not missing else 0.45,
        },
    }


def _build_deterministic_program(
    *,
    room_type: str,
    room_rules: Mapping[str, Any],
    room_model_json: Mapping[str, Any],
    affordance_summary: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    brief_text: str,
    inventory_catalog: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    _ = inventory_catalog
    active_clusters: list[dict[str, Any]] = []
    missing: list[str] = []
    conflicts: list[str] = []
    for candidate in candidates:
        candidate_missing = _string_list(candidate.get("missing"))
        if candidate.get("priority") == "core" and candidate_missing:
            missing.extend(candidate_missing)
            continue
        if not candidate.get("useful"):
            continue
        active_clusters.append(
            _active_cluster_from_candidate(
                candidate=candidate,
                room_type=room_type,
                affordance_summary=affordance_summary,
                brief_text=brief_text,
            )
        )

    global_program = room_rules.get("global_program")
    global_program = global_program if isinstance(global_program, Mapping) else {}
    active_clusters = _enforce_dominant_requirements(
        active_clusters,
        candidates=candidates,
        room_type=room_type,
        global_program=global_program,
        missing=missing,
    )
    active_clusters = _enforce_group_caps(active_clusters, global_program)
    active_clusters = active_clusters[:_CLUSTER_CANDIDATE_CAP]

    macro_relations = _build_macro_relations(
        active_clusters=active_clusters,
        room_type=room_type,
        affordance_summary=affordance_summary,
    )
    controlled_degradation = _build_controlled_degradation(
        active_clusters=active_clusters,
        global_program=global_program,
    )
    status = "OK"
    if missing:
        status = "UNSAT" if not active_clusters else "NEEDS_REVIEW"
    profile_layout_trace = profile_layout_trace_for_active_clusters(
        room_type=room_type,
        active_clusters=active_clusters,
    )
    return {
        "status": status,
        "room_type": room_type,
        "active_clusters": active_clusters,
        "global_layout_intent": _global_layout_intent(
            room_rules=room_rules,
            active_clusters=active_clusters,
            brief_text=brief_text,
        ),
        "macro_relations": macro_relations,
        "selection_constraints": {
            "dominant_anchor_required": _string_list(
                global_program.get("dominant_anchor_required")
            ),
            "dominant_workflow_required": _string_list(
                global_program.get("dominant_workflow_required")
            ),
            "group_caps": list(global_program.get("group_caps") or []),
            "group_minimums": list(global_program.get("group_minimums") or []),
        },
        "controlled_degradation": controlled_degradation,
        "quality_targets": {
            "functionality_weight": 1.0,
            "naturalness_weight": 1.0,
            "semantic_coherence_weight": 1.0,
            "spatial_quality_weight": 1.0,
        },
        "missing": _uniq(missing),
        "conflicts": _uniq(conflicts),
        "confidence": _confidence(active_clusters, missing),
        "profile_layout_trace": profile_layout_trace,
        "profile_shadow_trace": profile_layout_trace,
        "notes": [
            "SemanticLayoutPlanner used the room profile registry as the canonical rule source.",
            "Planner output is cluster/bundle/zone level only; pose is left to downstream composer and solver.",
        ],
    }


def _validate_and_normalize_llm_program(
    *,
    llm_payload: Mapping[str, Any],
    deterministic_program: Mapping[str, Any],
    room_type: str,
    room_rules: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    affordance_summary: Mapping[str, Any],
) -> dict[str, Any]:
    _ = room_rules
    allowed_candidates = {str(row.get("cluster_id")): row for row in candidates}
    deterministic_by_id = {
        str(row.get("cluster_id")): row
        for row in deterministic_program.get("active_clusters") or []
        if isinstance(row, Mapping)
    }
    raw_clusters = llm_payload.get("active_clusters")
    active_clusters: list[dict[str, Any]] = []
    if isinstance(raw_clusters, list):
        for raw_cluster in raw_clusters:
            if not isinstance(raw_cluster, Mapping):
                continue
            cluster_id = str(raw_cluster.get("cluster_id") or "").strip()
            if cluster_id not in allowed_candidates:
                continue
            base = deterministic_by_id.get(cluster_id)
            if base is None:
                continue
            merged = dict(base)
            merged["layout_role"] = _require_layout_role(
                raw_cluster.get("layout_role"),
                context=f"semantic_layout_program cluster `{cluster_id}`",
            )
            if isinstance(raw_cluster.get("semantic_role"), str):
                merged["semantic_role"] = str(raw_cluster["semantic_role"]).strip()
            if isinstance(raw_cluster.get("zone_claims"), Mapping):
                merged["zone_claims"] = _normalize_zone_claims(
                    raw_cluster.get("zone_claims"),
                    fallback=base.get("zone_claims"),
                    affordance_summary=affordance_summary,
                )
            if isinstance(raw_cluster.get("relation_intents"), list):
                merged["relation_intents"] = _normalize_relation_intents(
                    raw_cluster.get("relation_intents"),
                    allowed_cluster_ids=set(allowed_candidates),
                )
            if isinstance(raw_cluster.get("degradation_ladder"), list):
                merged["degradation_ladder"] = _string_list(
                    raw_cluster.get("degradation_ladder")
                )[:_DEGRADATION_STEPS_CAP_PER_CLUSTER]
            active_clusters.append(merged)
    if not active_clusters:
        return dict(deterministic_program)

    out = dict(deterministic_program)
    out["room_type"] = room_type
    out["active_clusters"] = active_clusters
    out["macro_relations"] = _build_macro_relations(
        active_clusters=active_clusters,
        room_type=room_type,
        affordance_summary=affordance_summary,
    )
    out["controlled_degradation"] = _build_controlled_degradation(
        active_clusters=active_clusters,
        global_program=out.get("selection_constraints") or {},
    )
    profile_layout_trace = profile_layout_trace_for_active_clusters(
        room_type=room_type,
        active_clusters=active_clusters,
    )
    out["profile_layout_trace"] = profile_layout_trace
    out["profile_shadow_trace"] = profile_layout_trace
    out["confidence"] = max(float(out.get("confidence") or 0.75), 0.82)
    return out


def _require_layout_role(value: Any, *, context: str) -> str:
    role = str(value or "").strip().lower()
    if role not in _LAYOUT_ROLES:
        raise ValueError(
            f"{context} must set layout_role to one of: "
            + ", ".join(sorted(_LAYOUT_ROLES))
        )
    return role


def _deterministic_layout_role(priority: str) -> str:
    if priority == "optional":
        return "optional"
    return "support"


def _active_cluster_from_candidate(
    *,
    candidate: Mapping[str, Any],
    room_type: object | None,
    affordance_summary: Mapping[str, Any],
    brief_text: str,
) -> dict[str, Any]:
    cluster_id = str(candidate.get("cluster_id") or "")
    priority = _priority(candidate.get("priority"))
    objects = [
        dict(item)
        for item in candidate.get("objects") or []
        if isinstance(item, Mapping)
    ]
    zone_claims = _zone_claims_for(
        cluster_id=cluster_id,
        objects=objects,
        room_type=room_type,
        affordance_summary=affordance_summary,
    )
    relation_intents = _relation_intents_for(
        cluster_id,
        objects,
        room_type=room_type,
    )
    viability = _semantic_viability(
        candidate=candidate,
        zone_claims=zone_claims,
        brief_text=brief_text,
    )
    return {
        "cluster_id": cluster_id,
        "layout_role": _deterministic_layout_role(priority),
        "priority": priority,
        "activation_reason": _activation_reason(candidate),
        "semantic_role": _semantic_role(
            cluster_id,
            objects,
            priority,
            room_type=room_type,
        ),
        "dominant_anchor_candidates": _string_list(
            (candidate.get("semantic") or {}).get("dominant_anchor_candidates")
            if isinstance(candidate.get("semantic"), Mapping)
            else []
        ),
        "required_bundles": [
            {
                "bundle_id": f"{cluster_id}_bundle",
                "objects": objects,
            }
        ],
        "zone_claims": zone_claims,
        "relation_intents": relation_intents[:_RELATION_INTENT_CAP_PER_CLUSTER],
        "degradation_ladder": _degradation_ladder(candidate),
        "tier_count_hints": deepcopy(dict(candidate.get("tier_count_hints") or {})),
        "viability_score": viability,
    }


def _candidate_objects(
    *,
    object_program: Mapping[str, Any],
    cluster_rule: Mapping[str, Any],
    inventory_catalog: Mapping[str, Mapping[str, Any]],
    priority: str,
    brief_text: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    objects: list[dict[str, Any]] = []
    missing: list[str] = []

    def add_object(object_type: str, *, role: str, required: bool) -> None:
        if object_type in {str(row.get("object_type")) for row in objects}:
            return
        if not _is_available(object_type, inventory_catalog):
            if required:
                missing.append(object_type)
            return
        objects.append(
            {
                "object_type": object_type,
                "role": role,
                "required": required,
                "max_keep": _max_keep_for_object(object_type, object_program),
            }
        )

    dominant_candidates = _string_list(
        (cluster_rule.get("semantic") or {}).get("dominant_anchor_candidates")
        if isinstance(cluster_rule.get("semantic"), Mapping)
        else []
    )
    for object_type in _string_list(object_program.get("required")):
        add_object(
            object_type,
            role=_required_object_role(
                object_type,
                dominant_candidates=dominant_candidates,
            ),
            required=True,
        )
    for group in _list_of_string_lists(object_program.get("choose_exactly_one_from")):
        selected = _select_best_object(group, inventory_catalog, brief_text)
        if selected is None:
            missing.extend(group)
        else:
            add_object(selected, role="dominant_anchor", required=True)
    for group in _list_of_string_lists(object_program.get("choose_at_least_one_from")):
        selected = _select_best_object(group, inventory_catalog, brief_text)
        if selected is None:
            missing.extend(group)
        else:
            add_object(selected, role="dominant_anchor", required=True)
    kept_required_if = (
        priority == "core"
        or _brief_support_score(
            str(cluster_rule.get("cluster_id") or ""), [], brief_text
        )
        >= 0.45
    )
    if kept_required_if:
        for object_type in _string_list(object_program.get("required_if_kept")):
            add_object(object_type, role="workflow_anchor", required=True)
        for group in _list_of_string_lists(
            object_program.get("choose_exactly_one_from_if_kept")
        ):
            selected = _select_best_object(group, inventory_catalog, brief_text)
            if selected is not None:
                add_object(selected, role="support", required=True)

    optional = _string_list(object_program.get("optional"))
    optional_limit = _global_optional_limit(object_program)
    for object_type in optional[:optional_limit]:
        add_object(object_type, role=_support_role(object_type), required=False)

    return objects, _uniq(missing)


def _merged_object_program_for_brief(
    cluster_rule: Mapping[str, Any],
    brief_text: str,
) -> dict[str, Any]:
    base = dict(cluster_rule.get("object_program") or {})
    activation = cluster_rule.get("activation")
    conditions = activation.get("conditions") if isinstance(activation, Mapping) else []
    if not isinstance(conditions, list):
        return base
    for condition in conditions:
        if not isinstance(condition, Mapping):
            continue
        predicate = str(condition.get("predicate") or "")
        if not _predicate_supported(predicate, brief_text):
            continue
        effects = condition.get("effects")
        if isinstance(effects, Mapping):
            base = _merge_object_programs(base, effects)
    return base


def _merge_object_programs(
    base: Mapping[str, Any], effects: Mapping[str, Any]
) -> dict[str, Any]:
    out = dict(base)
    for key in (
        "required",
        "required_if_kept",
        "choose_exactly_one_from",
        "choose_exactly_one_from_if_kept",
        "choose_at_least_one_from",
        "optional",
    ):
        if key not in effects:
            continue
        if key.startswith("choose_"):
            existing = _list_of_string_lists(out.get(key))
            incoming = effects.get(key)
            if isinstance(incoming, list) and incoming:
                existing.extend(_list_of_string_lists(incoming))
            out[key] = existing
        else:
            out[key] = _uniq(
                _string_list(out.get(key)) + _string_list(effects.get(key))
            )
    limits = effects.get("optional_limits")
    if isinstance(limits, Mapping):
        out["optional_limits"] = dict(limits)
    return out


def _zone_claims_for(
    *,
    cluster_id: str,
    objects: Sequence[Mapping[str, Any]],
    room_type: object | None,
    affordance_summary: Mapping[str, Any],
) -> dict[str, Any]:
    object_types = [str(row.get("object_type") or "") for row in objects]
    lowered = f"{cluster_id} {' '.join(object_types)}".lower()
    profile_claims = profile_zone_claims_for_objects(
        cluster_id=cluster_id,
        object_types=object_types,
        room_type=room_type,
        affordance_summary=affordance_summary,
    )
    if profile_claims is not None:
        return profile_claims

    entry = _cap_regions(affordance_summary.get("entry_landing_zones"))
    daylight = _cap_regions(affordance_summary.get("daylight_regions"))
    privacy = _cap_regions(affordance_summary.get("privacy_regions"))
    walls = _cap_regions(affordance_summary.get("wall_anchor_candidates"))
    focal = _cap_regions(affordance_summary.get("focal_surfaces"))
    center = _cap_regions(affordance_summary.get("center_openness_regions"))
    corridors = _cap_regions(affordance_summary.get("primary_circulation_corridors"))

    if "sleep" in lowered or any(is_bed_like(obj) for obj in object_types):
        return {
            "preferred_regions": _cap_regions(privacy + walls),
            "avoid_regions": _cap_regions(entry + corridors),
            "wall_affinity": "high",
            "daylight_affinity": "medium",
            "privacy_affinity": "high",
            "floating_allowed": False,
        }
    if "work" in lowered or any(is_work_surface_like(obj) for obj in object_types):
        return {
            "preferred_regions": _cap_regions(daylight + walls),
            "avoid_regions": _cap_regions(entry + center),
            "wall_affinity": "high",
            "daylight_affinity": "high",
            "privacy_affinity": "medium",
            "floating_allowed": False,
        }
    if "media" in lowered or any(
        _contains_any(obj, _MEDIA_TOKENS) for obj in object_types
    ):
        return {
            "preferred_regions": _cap_regions(focal + walls),
            "avoid_regions": _cap_regions(daylight + entry),
            "wall_affinity": "high",
            "daylight_affinity": "low",
            "privacy_affinity": "none",
            "floating_allowed": False,
        }
    if "storage" in lowered or any(_is_storage_like(obj) for obj in object_types):
        return {
            "preferred_regions": _cap_regions(walls + privacy),
            "avoid_regions": _cap_regions(center + corridors + entry),
            "wall_affinity": "high",
            "daylight_affinity": "low",
            "privacy_affinity": "medium",
            "floating_allowed": False,
        }
    if "dining" in lowered:
        return {
            "preferred_regions": _cap_regions(
                affordance_summary.get("floating_zone_candidates")
            ),
            "avoid_regions": _cap_regions(entry + corridors),
            "wall_affinity": "medium",
            "daylight_affinity": "medium",
            "privacy_affinity": "none",
            "floating_allowed": True,
        }
    return {
        "preferred_regions": _cap_regions(focal + center + walls),
        "avoid_regions": _cap_regions(entry + corridors),
        "wall_affinity": "medium",
        "daylight_affinity": "medium",
        "privacy_affinity": "low",
        "floating_allowed": True,
    }


def _relation_intents_for(
    cluster_id: str,
    objects: Sequence[Mapping[str, Any]],
    *,
    room_type: object | None,
) -> list[dict[str, Any]]:
    object_types = [str(row.get("object_type") or "") for row in objects]
    intents: list[dict[str, Any]] = []
    if any(str(row.get("role")) == "dominant_anchor" for row in objects):
        intents.append({"type": "dominance", "target": "room", "strength": "hard"})
    if any(is_bed_like(obj) for obj in object_types):
        intents.append(
            {"type": "separate", "target_cluster": "work_study", "strength": "soft"}
        )
    if any(_contains_any(obj, _MEDIA_TOKENS) for obj in object_types):
        intents.append(
            {"type": "face", "target_cluster": "main_seating", "strength": "soft"}
        )
    intents = _merge_relation_intents(
        intents,
        profile_relation_intents_for_objects(
            cluster_id=cluster_id,
            object_types=object_types,
            room_type=room_type,
        ),
    )
    if "work" in cluster_id:
        intents.append(
            {"type": "claim_daylight", "target": "daylight", "strength": "soft"}
        )
    return intents


def _build_macro_relations(
    *,
    active_clusters: Sequence[Mapping[str, Any]],
    room_type: object | None,
    affordance_summary: Mapping[str, Any],
) -> dict[str, Any]:
    adjacency: list[dict[str, object]] = []
    separation: list[dict[str, object]] = []
    orientation: list[dict[str, object]] = []
    by_id = {str(row.get("cluster_id")): row for row in active_clusters}
    seating = _first_cluster_matching(active_clusters, _cluster_has_seating)
    media = _first_cluster_matching(active_clusters, _cluster_has_media)
    sleep = _first_cluster_matching(active_clusters, _cluster_has_sleep)
    work = _first_cluster_matching(active_clusters, _cluster_has_work)
    profile_relations = profile_macro_relations_for_active_clusters(
        room_type=room_type,
        active_clusters=active_clusters,
    )
    if seating and media:
        adjacency.append(
            {"a": seating, "b": media, "relation": "near", "priority": "high"}
        )
        orientation.append(
            {"a": seating, "b": media, "relation": "face", "priority": "high"}
        )
    if sleep and work and sleep in by_id and work in by_id:
        separation.append(
            {"a": sleep, "b": work, "relation": "separate", "priority": "medium"}
        )
    adjacency.extend(profile_relations.get("adjacency_preferences", []))
    separation.extend(profile_relations.get("separation_preferences", []))
    orientation.extend(profile_relations.get("orientation_preferences", []))
    return {
        "adjacency_preferences": adjacency[:_GLOBAL_RELATION_CAP],
        "separation_preferences": separation[:_GLOBAL_RELATION_CAP],
        "orientation_preferences": orientation[:_GLOBAL_RELATION_CAP],
        "keep_open_regions": [
            {"region": region, "reason": "preserve open center and circulation"}
            for region in _cap_regions(
                affordance_summary.get("center_openness_regions")
            )
        ],
        "reserved_regions": [
            {"region": region, "reason": "entry and primary circulation stay clear"}
            for region in _cap_regions(
                _cap_regions(affordance_summary.get("entry_landing_zones"))
                + _cap_regions(affordance_summary.get("primary_circulation_corridors"))
            )
        ],
    }


def _build_controlled_degradation(
    *,
    active_clusters: Sequence[Mapping[str, Any]],
    global_program: Mapping[str, Any],
) -> dict[str, Any]:
    cluster_rank = {"optional": 0, "support": 1, "core": 2}
    ordered = sorted(
        active_clusters,
        key=lambda row: (
            cluster_rank.get(str(row.get("priority") or "support"), 1),
            str(row.get("cluster_id") or ""),
        ),
    )
    bundle_drop_order: list[str] = []
    never_drop_first = _string_list(global_program.get("dominant_anchor_required"))
    for cluster in ordered:
        for bundle in cluster.get("required_bundles") or []:
            if isinstance(bundle, Mapping):
                bundle_id = str(bundle.get("bundle_id") or "").strip()
                if bundle_id:
                    bundle_drop_order.append(bundle_id)
    return {
        "cluster_drop_order": [str(row.get("cluster_id")) for row in ordered],
        "bundle_drop_order": bundle_drop_order,
        "never_drop_first": never_drop_first,
    }


def _degradation_ladder(candidate: Mapping[str, Any]) -> list[str]:
    hints = candidate.get("degradation_hints")
    hints = hints if isinstance(hints, Mapping) else {}
    steps: list[str] = []
    if _string_list(hints.get("shrink_before_drop")):
        steps.append("shrink_optional_support")
    for object_type in _string_list(hints.get("drop_first")):
        steps.append(f"drop_{object_type}")
    for object_type in _string_list(hints.get("shrink_before_drop")):
        steps.append(f"reduce_{object_type}_count")
    if not steps:
        steps = ["shrink_optional_support", "drop_secondary_support"]
    return _uniq(steps)[:_DEGRADATION_STEPS_CAP_PER_CLUSTER]


def _semantic_viability(
    *,
    candidate: Mapping[str, Any],
    zone_claims: Mapping[str, Any],
    brief_text: str,
) -> dict[str, Any]:
    affordance_support = 1.0 if zone_claims.get("preferred_regions") else 0.55
    inventory_support = float(
        (candidate.get("viability") or {}).get("inventory_support", 1.0)
        if isinstance(candidate.get("viability"), Mapping)
        else 1.0
    )
    brief_support = _brief_support_score(
        str(candidate.get("cluster_id") or ""),
        candidate.get("objects") if isinstance(candidate.get("objects"), list) else [],
        brief_text,
    )
    conflict_penalty = 0.15 if not zone_claims.get("avoid_regions") else 0.0
    score = max(
        0.0,
        min(
            1.0,
            (0.3 * 1.0)
            + (0.25 * affordance_support)
            + (0.2 * max(brief_support, 0.35))
            + (0.25 * inventory_support)
            - conflict_penalty,
        ),
    )
    return {
        "rule_support": 1.0,
        "affordance_support": round(affordance_support, 3),
        "brief_support": round(max(brief_support, 0.35), 3),
        "inventory_support": round(inventory_support, 3),
        "conflict_penalty": round(conflict_penalty, 3),
        "score": round(score, 3),
    }


def _enforce_dominant_requirements(
    active_clusters: list[dict[str, Any]],
    *,
    candidates: Sequence[Mapping[str, Any]],
    room_type: object | None,
    global_program: Mapping[str, Any],
    missing: list[str],
) -> list[dict[str, Any]]:
    required = _string_list(global_program.get("dominant_anchor_required"))
    if not required:
        return active_clusters
    active_objects = {
        str(obj.get("object_type"))
        for cluster in active_clusters
        for bundle in cluster.get("required_bundles") or []
        if isinstance(bundle, Mapping)
        for obj in bundle.get("objects") or []
        if isinstance(obj, Mapping)
    }
    for object_type in required:
        if object_type in active_objects:
            continue
        candidate = _candidate_containing_object(candidates, object_type)
        if candidate is None:
            missing.append(object_type)
            continue
        active_clusters.append(
            _active_cluster_from_candidate(
                candidate=candidate,
                room_type=room_type,
                affordance_summary={
                    "privacy_regions": ["private_back_zone"],
                    "wall_anchor_candidates": ["quiet_wall_zone"],
                    "entry_landing_zones": ["entry_landing_zone"],
                    "primary_circulation_corridors": ["center_access_lane"],
                    "center_openness_regions": ["room_center"],
                },
                brief_text="",
            )
        )
    return active_clusters


def _enforce_group_caps(
    active_clusters: list[dict[str, Any]], global_program: Mapping[str, Any]
) -> list[dict[str, Any]]:
    caps = global_program.get("group_caps")
    if not isinstance(caps, list):
        return active_clusters
    for cap in caps:
        if not isinstance(cap, Mapping):
            continue
        capped = set(_string_list(cap.get("objects")))
        max_keep = _int_value(cap.get("max_keep"), default=len(capped))
        if not capped or max_keep < 0:
            continue
        kept = 0
        for cluster in active_clusters:
            for bundle in cluster.get("required_bundles") or []:
                if not isinstance(bundle, Mapping):
                    continue
                objects = bundle.get("objects")
                if not isinstance(objects, list):
                    continue
                filtered: list[dict[str, Any]] = []
                for obj in objects:
                    if not isinstance(obj, Mapping):
                        continue
                    object_type = str(obj.get("object_type") or "")
                    if object_type in capped:
                        kept += 1
                        if kept > max_keep:
                            continue
                    filtered.append(dict(obj))
                bundle["objects"] = filtered
    return active_clusters


def _global_layout_intent(
    *,
    room_rules: Mapping[str, Any],
    active_clusters: Sequence[Mapping[str, Any]],
    brief_text: str,
) -> dict[str, Any]:
    policy = (
        room_rules.get("policy")
        if isinstance(room_rules.get("policy"), Mapping)
        else {}
    )
    primary_focus = _primary_focus(active_clusters)
    space_character = str(policy.get("intent") or "").strip() or "balanced_functional"
    lowered = brief_text.lower()
    prefer_open_center = not any(
        token in lowered for token in ("dense", "maximal", "full")
    )
    return {
        "primary_focus": primary_focus,
        "space_character": _snake_token(space_character) or "balanced_functional",
        "prefer_open_center": prefer_open_center,
        "prefer_core_before_support": True,
        "prefer_clear_primary_circulation": True,
    }


def _objects_from_active_cluster(active_cluster: Mapping[str, Any]) -> list[str]:
    objects: list[str] = []
    for bundle in active_cluster.get("required_bundles") or []:
        if not isinstance(bundle, Mapping):
            continue
        for obj in bundle.get("objects") or []:
            if not isinstance(obj, Mapping):
                continue
            object_type = _clean_object_type(obj.get("object_type"))
            if object_type is not None and object_type not in objects:
                objects.append(object_type)
    return objects


def _request_contract_program_object_types(
    semantic_program: Mapping[str, Any],
) -> list[str]:
    out: list[str] = []
    active_clusters = semantic_program.get("active_clusters")
    if not isinstance(active_clusters, Sequence) or isinstance(active_clusters, str):
        return out
    for cluster in active_clusters:
        if isinstance(cluster, Mapping):
            out.extend(_objects_from_active_cluster(cluster))
    return _uniq(out)


def _anchors_from_active_cluster(
    active_cluster: Mapping[str, Any],
    members: Sequence[str],
    *,
    dominant_anchor_candidates: Sequence[str] | None = None,
) -> list[str]:
    for candidate in dominant_anchor_candidates or []:
        if candidate in members:
            return [candidate]
    for bundle in active_cluster.get("required_bundles") or []:
        if not isinstance(bundle, Mapping):
            continue
        for obj in bundle.get("objects") or []:
            if not isinstance(obj, Mapping):
                continue
            object_type = _clean_object_type(obj.get("object_type"))
            if object_type in members and obj.get("role") in {
                "dominant_anchor",
                "workflow_anchor",
            }:
                return [object_type]
    return [members[0]] if members else []


def _dominant_anchor_candidates_from_active_cluster(
    active_cluster: Mapping[str, Any],
    members: Sequence[str],
) -> list[str]:
    member_set = set(members)
    candidates = [
        candidate
        for candidate in _string_list(active_cluster.get("dominant_anchor_candidates"))
        if candidate in member_set
    ]
    if candidates:
        return candidates

    role_candidates: list[str] = []
    for bundle in active_cluster.get("required_bundles") or []:
        if not isinstance(bundle, Mapping):
            continue
        for obj in bundle.get("objects") or []:
            if not isinstance(obj, Mapping):
                continue
            object_type = _clean_object_type(obj.get("object_type"))
            if object_type in member_set and obj.get("role") == "dominant_anchor":
                role_candidates.append(object_type)
    return _uniq(role_candidates)


def _hard_constraints_for_members(members: Sequence[str]) -> list[dict[str, object]]:
    constraints: list[dict[str, object]] = []
    for idx, a in enumerate(members):
        for b in members[idx + 1 :]:
            constraints.append({"type": "no_overlap", "a": a, "b": b})
    for member in members:
        if _needs_front_access(member):
            constraints.append(
                {
                    "type": "requires_access",
                    "id": member,
                    "mode": "front_clearance",
                }
            )
    return constraints


def _soft_constraints_for_members(
    members: Sequence[str], anchors: Sequence[str]
) -> list[dict[str, object]]:
    if not anchors:
        return []
    anchor = anchors[0]
    out: list[dict[str, object]] = []
    for member in members:
        if member == anchor:
            continue
        out.append({"type": "prefer_near", "a": member, "b": anchor, "weight": 6})
    return out


def _normalize_zone_claims(
    value: Any,
    *,
    fallback: Any,
    affordance_summary: Mapping[str, Any],
) -> dict[str, Any]:
    fallback = fallback if isinstance(fallback, Mapping) else {}
    value = value if isinstance(value, Mapping) else {}
    allowed_regions = set()
    for regions in affordance_summary.values():
        if isinstance(regions, list):
            allowed_regions.update(str(region) for region in regions)
    preferred = [
        region
        for region in _string_list(value.get("preferred_regions"))
        if region in allowed_regions
    ][:_ZONE_CLAIM_CAP_PER_CLUSTER]
    avoid = [
        region
        for region in _string_list(value.get("avoid_regions"))
        if region in allowed_regions
    ][:_ZONE_CLAIM_CAP_PER_CLUSTER]
    return {
        "preferred_regions": preferred
        or _string_list(fallback.get("preferred_regions"))[
            :_ZONE_CLAIM_CAP_PER_CLUSTER
        ],
        "avoid_regions": avoid
        or _string_list(fallback.get("avoid_regions"))[:_ZONE_CLAIM_CAP_PER_CLUSTER],
        "wall_affinity": _affinity(
            value.get("wall_affinity"), fallback.get("wall_affinity")
        ),
        "daylight_affinity": _affinity(
            value.get("daylight_affinity"), fallback.get("daylight_affinity")
        ),
        "privacy_affinity": _affinity(
            value.get("privacy_affinity"), fallback.get("privacy_affinity")
        ),
        "floating_allowed": bool(
            value.get("floating_allowed", fallback.get("floating_allowed", False))
        ),
    }


def _normalize_relation_intents(
    value: Any, *, allowed_cluster_ids: set[str]
) -> list[dict[str, Any]]:
    allowed_types = {
        "near",
        "separate",
        "face",
        "buffer",
        "claim_wall",
        "claim_daylight",
        "claim_privacy",
        "avoid_entry",
        "preserve_center",
        "dominance",
    }
    out: list[dict[str, Any]] = []
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, Mapping):
            continue
        intent_type = _snake_token(str(item.get("type") or ""))
        if intent_type not in allowed_types:
            continue
        target_cluster = str(item.get("target_cluster") or "").strip() or None
        if target_cluster is not None and target_cluster not in allowed_cluster_ids:
            continue
        strength = (
            "hard" if str(item.get("strength") or "").lower() == "hard" else "soft"
        )
        out.append(
            {
                "type": intent_type,
                "target": str(item.get("target") or "").strip() or None,
                "target_cluster": target_cluster,
                "strength": strength,
            }
        )
    return out[:_RELATION_INTENT_CAP_PER_CLUSTER]


def _adaptive_stage_spec(stage_name: str) -> _AdaptiveStageSpec:
    try:
        return _ADAPTIVE_STAGE_SPECS[stage_name]
    except KeyError as exc:
        raise ValueError(f"Unknown adaptive stage: {stage_name}") from exc


def _adaptive_stage_model_chain(stage_name: str) -> list[str]:
    stage_spec = _adaptive_stage_spec(stage_name)
    return TextLLMConfig.agent_model_chain(
        stage_spec.model_config_keys,
        stage_spec.default_model_chain,
    )


def _adaptive_stage_response_schema(stage_name: str) -> Mapping[str, object] | None:
    if stage_name == "adaptive_request_contract":
        return _ADAPTIVE_REQUEST_CONTRACT_RESPONSE_SCHEMA
    if stage_name == "adaptive_candidate_overrides":
        return _ADAPTIVE_CANDIDATE_OVERRIDES_RESPONSE_SCHEMA
    if stage_name == "adaptive_cluster_semantics":
        return _ADAPTIVE_CLUSTER_SEMANTICS_RESPONSE_SCHEMA
    return None


def _is_invalid_json_error(exc: Exception) -> bool:
    text = str(exc)
    return (
        "returned invalid JSON" in text
        or "invalid JSON" in text
        or isinstance(exc.__cause__, json.JSONDecodeError)
    )


def _record_llm_retry(*, stage: str, model_name: str | None, reason: str) -> None:
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


def _adaptive_semantic_llm_enabled() -> bool:
    return os.getenv(_ADAPTIVE_SEMANTIC_LLM_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _request_contract_llm_enabled() -> bool:
    return os.getenv(_REQUEST_CONTRACT_LLM_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _sanitize_notes(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if text and text not in out:
            out.append(text)
    return out


def _merge_notes(primary: Any, secondary: Any) -> list[str]:
    out: list[str] = []
    for source in (primary, secondary):
        for note in _sanitize_notes(source):
            if note not in out:
                out.append(note)
    if _ADAPTIVE_STAGE_NOTE in out:
        out = [
            note
            for note in out
            if "compiled_semantic_program.json as the canonical rule source" not in note
        ]
    return out


def _inventory_summary_for_llm(
    inventory_catalog: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for object_type in sorted(inventory_catalog):
        item = inventory_catalog.get(object_type) or {}
        rows.append(
            {
                "object_type": object_type,
                "available": bool(item.get("available", True)),
                "size_profiles": _string_list(item.get("size_profiles")),
                "functional_tags": _string_list(item.get("functional_tags")),
            }
        )
    return rows


def _sanitize_adaptive_room_rule(
    value: Any,
    *,
    room_type: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("adaptive_room_rule must return a room_rule object")
    clusters_raw = value.get("clusters")
    if not isinstance(clusters_raw, list):
        raise ValueError("adaptive_room_rule.room_rule.clusters must be a list")

    clusters: list[dict[str, Any]] = []
    seen_cluster_ids: set[str] = set()
    for cluster in clusters_raw:
        sanitized = _sanitize_adaptive_cluster_rule(cluster)
        cluster_id = sanitized["cluster_id"]
        if cluster_id in seen_cluster_ids:
            continue
        seen_cluster_ids.add(cluster_id)
        clusters.append(sanitized)
        if len(clusters) >= _CLUSTER_CANDIDATE_CAP:
            break
    if not clusters:
        raise ValueError("adaptive_room_rule produced no valid clusters")

    policy = value.get("policy")
    return {
        "room_type": room_type,
        "policy": _clone_json_mapping(policy),
        "clusters": clusters,
        "global_program": _sanitize_global_program(value.get("global_program")),
    }


def _sanitize_adaptive_cluster_rule(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("adaptive_room_rule cluster entries must be objects")
    cluster_id = str(value.get("cluster_id") or "").strip()
    if not cluster_id:
        raise ValueError("adaptive_room_rule cluster is missing cluster_id")
    priority = _priority(value.get("priority"))
    object_program = _sanitize_object_program(value.get("object_program"))
    if not _object_program_has_content(object_program):
        raise ValueError(
            f"adaptive_room_rule cluster `{cluster_id}` has no usable object_program"
        )
    return {
        "cluster_id": cluster_id,
        "priority": priority,
        "activation": _sanitize_activation_rule(value.get("activation")),
        "object_program": object_program,
        "semantic": _sanitize_cluster_rule_semantic(value.get("semantic")),
        "degradation_hints": _sanitize_degradation_hints(
            value.get("degradation_hints")
        ),
        "tier_count_hints": _sanitize_tier_count_hints(
            value.get("tier_count_hints"),
            object_types=_object_types_from_program(object_program),
            default_bundle_class=_default_bundle_class_for_priority(priority),
            default_preserve_level=_default_preserve_level_for_priority(priority),
            required=True,
        ),
    }


def _clone_json_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): _clone_jsonish(child)
        for key, child in value.items()
        if isinstance(key, str)
    }


def _clone_jsonish(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _clone_json_mapping(value)
    if isinstance(value, list):
        return [_clone_jsonish(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sanitize_global_program(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {
            "dominant_anchor_required": [],
            "dominant_workflow_required": [],
            "group_caps": [],
            "group_minimums": [],
        }
    return {
        "dominant_anchor_required": _string_list(value.get("dominant_anchor_required")),
        "dominant_workflow_required": _string_list(
            value.get("dominant_workflow_required")
        ),
        "group_caps": _sanitize_group_rules(
            value.get("group_caps"),
            count_key="max_keep",
            default_count=1,
        ),
        "group_minimums": _sanitize_group_rules(
            value.get("group_minimums"),
            count_key="min_keep",
            default_count=1,
        ),
    }


def _sanitize_group_rules(
    value: Any,
    *,
    count_key: str,
    default_count: int,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        objects = _string_list(item.get("objects"))
        if not objects:
            continue
        out.append(
            {
                "objects": objects,
                count_key: max(
                    0,
                    _int_value(item.get(count_key), default=default_count),
                ),
            }
        )
    return out


def _sanitize_activation_rule(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, Any] = {}
    if "always_consider" in value:
        out["always_consider"] = bool(value.get("always_consider"))
    conditions: list[dict[str, Any]] = []
    for item in value.get("conditions") or []:
        if not isinstance(item, Mapping):
            continue
        predicate = str(item.get("predicate") or "").strip()
        effects = _sanitize_object_program(item.get("effects"))
        if predicate and _object_program_has_content(effects):
            conditions.append({"predicate": predicate, "effects": effects})
    if conditions:
        out["conditions"] = conditions
    return out


def _sanitize_object_program(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, Any] = {}
    for key in (
        "required",
        "required_if_kept",
        "optional",
    ):
        values = _string_list(value.get(key))
        if values:
            out[key] = values
    for key in (
        "choose_exactly_one_from",
        "choose_exactly_one_from_if_kept",
        "choose_at_least_one_from",
    ):
        groups = _list_of_string_lists(value.get(key))
        if groups:
            out[key] = groups
    limits = value.get("optional_limits")
    if isinstance(limits, Mapping):
        clean_limits: dict[str, Any] = {}
        if limits.get("global") is not None:
            clean_limits["global"] = max(
                0,
                _int_value(limits.get("global"), default=0),
            )
        by_object = limits.get("by_object")
        if isinstance(by_object, Mapping):
            clean_by_object: dict[str, int] = {}
            for object_type, limit in by_object.items():
                normalized_object_type = _clean_object_type(object_type)
                if normalized_object_type is None:
                    continue
                clean_by_object[normalized_object_type] = max(
                    0,
                    _int_value(limit, default=1),
                )
            if clean_by_object:
                clean_limits["by_object"] = clean_by_object
        if clean_limits:
            out["optional_limits"] = clean_limits
    return out


def _object_program_has_content(value: Mapping[str, Any]) -> bool:
    return any(
        _string_list(value.get(key))
        for key in ("required", "required_if_kept", "optional")
    ) or any(
        _list_of_string_lists(value.get(key))
        for key in (
            "choose_exactly_one_from",
            "choose_exactly_one_from_if_kept",
            "choose_at_least_one_from",
        )
    )


def _sanitize_cluster_rule_semantic(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    dominant_anchor_candidates = _string_list(value.get("dominant_anchor_candidates"))
    if not dominant_anchor_candidates:
        return {}
    return {"dominant_anchor_candidates": dominant_anchor_candidates}


def _sanitize_degradation_hints(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, Any] = {}
    drop_first = _string_list(value.get("drop_first"))
    shrink_before_drop = _string_list(value.get("shrink_before_drop"))
    if drop_first:
        out["drop_first"] = drop_first
    if shrink_before_drop:
        out["shrink_before_drop"] = shrink_before_drop
    return out


def _default_bundle_class_for_priority(priority: str) -> str:
    if priority == "core":
        return "indispensable"
    if priority == "support":
        return "strong_support"
    return "optional"


def _default_preserve_level_for_priority(priority: str) -> str:
    if priority == "core":
        return "highest"
    if priority == "support":
        return "high"
    return "medium"


def _sanitize_tier_count_hints(
    value: Any,
    *,
    object_types: Sequence[str],
    default_bundle_class: str,
    default_preserve_level: str,
    required: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        if required:
            raise ValueError("tier_count_hints must be an object")
        value = {}

    normalized_object_types = [
        object_type for object_type in _uniq(object_types) if object_type
    ]
    bundle_class = _tier_count_bundle_class(
        value.get("bundle_class"),
        default=default_bundle_class,
    )
    preserve_level = _tier_count_preserve_level(
        value.get("preserve_level"),
        default=default_preserve_level,
    )
    keep_if_space_surplus = bool(value.get("keep_if_space_surplus"))
    space_surplus_threshold = _float_0_1(
        value.get("space_surplus_threshold"),
        default=0.45,
    )
    drop_order_bias = _tier_count_drop_order_bias(
        value.get("drop_order_bias"),
        default="neutral",
    )

    raw_object_hints = value.get("object_hints")
    object_hints_list = raw_object_hints if isinstance(raw_object_hints, list) else []
    by_object: dict[str, dict[str, Any]] = {}
    for item in object_hints_list:
        sanitized = _sanitize_tier_count_object_hint(
            item,
            allowed_object_types=set(normalized_object_types),
            default_keep_if_space_surplus=keep_if_space_surplus,
            default_space_surplus_threshold=space_surplus_threshold,
            default_drop_order_bias=drop_order_bias,
            default_preserve_level=preserve_level,
        )
        if sanitized is None:
            continue
        by_object[sanitized["object_type"]] = sanitized

    missing = [
        object_type
        for object_type in normalized_object_types
        if object_type not in by_object
    ]
    for object_type in missing:
        by_object[object_type] = {
            "object_type": object_type,
            "min_keep": 0,
            "max_keep": None,
            "keep_if_space_surplus": keep_if_space_surplus,
            "space_surplus_threshold": space_surplus_threshold,
            "drop_order_bias": drop_order_bias,
            "preserve_level": preserve_level,
            "preferred_size_tier": None,
        }

    out = {
        "bundle_class": bundle_class,
        "preserve_level": preserve_level,
        "keep_if_space_surplus": keep_if_space_surplus,
        "space_surplus_threshold": space_surplus_threshold,
        "drop_order_bias": drop_order_bias,
        "object_hints": [
            by_object[object_type]
            for object_type in normalized_object_types
            if object_type in by_object
        ],
    }
    return out


def _sanitize_tier_count_object_hint(
    value: Any,
    *,
    allowed_object_types: set[str],
    default_keep_if_space_surplus: bool,
    default_space_surplus_threshold: float,
    default_drop_order_bias: str,
    default_preserve_level: str,
) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    object_type = _clean_object_type(value.get("object_type"))
    if object_type is None or object_type not in allowed_object_types:
        return None

    raw_max_keep = value.get("max_keep")
    max_keep = (
        max(0, _int_value(raw_max_keep, default=0))
        if raw_max_keep is not None
        else None
    )
    min_keep = max(0, _int_value(value.get("min_keep"), default=0))
    if max_keep is not None:
        min_keep = min(min_keep, max_keep)

    preferred_size_tier = _tier_count_size_tier(value.get("preferred_size_tier"))
    return {
        "object_type": object_type,
        "min_keep": min_keep,
        "max_keep": max_keep,
        "keep_if_space_surplus": bool(
            value.get("keep_if_space_surplus", default_keep_if_space_surplus)
        ),
        "space_surplus_threshold": _float_0_1(
            value.get("space_surplus_threshold"),
            default=default_space_surplus_threshold,
        ),
        "drop_order_bias": _tier_count_drop_order_bias(
            value.get("drop_order_bias"),
            default=default_drop_order_bias,
        ),
        "preserve_level": _tier_count_preserve_level(
            value.get("preserve_level"),
            default=default_preserve_level,
        ),
        "preferred_size_tier": preferred_size_tier,
    }


def _tier_count_bundle_class(value: Any, *, default: str) -> str:
    text = str(value or default).strip().lower()
    if text in {"indispensable", "strong_support", "optional", "decor_light"}:
        return text
    return default


def _tier_count_preserve_level(value: Any, *, default: str) -> str:
    text = str(value or default).strip().lower()
    if text in {"highest", "high", "medium", "low"}:
        return text
    return default


def _tier_count_drop_order_bias(value: Any, *, default: str) -> str:
    text = str(value or default).strip().lower()
    if text in {"drop_first", "drop_early", "neutral", "drop_late", "drop_last"}:
        return text
    return default


def _tier_count_size_tier(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    if text in {"S", "M", "L"}:
        return text
    return None


def _sanitize_candidate_overrides(
    value: Any,
    *,
    expected_clusters: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("adaptive_candidate_overrides must return a list")
    expected = [cluster_id for cluster_id in expected_clusters if cluster_id]
    expected_set = set(expected)
    out: dict[str, dict[str, Any]] = {}
    for item in value:
        if not isinstance(item, Mapping):
            continue
        cluster_id = str(item.get("cluster_id") or "").strip()
        if cluster_id not in expected_set or cluster_id in out:
            continue
        cluster_rule = expected_clusters.get(cluster_id) or {}
        priority = _priority(cluster_rule.get("priority"))
        object_program = _sanitize_object_program(item.get("object_program"))
        if not _object_program_has_content(object_program):
            object_program = _sanitize_object_program(
                cluster_rule.get("object_program")
            )
        if not _object_program_has_content(object_program):
            raise ValueError(
                f"adaptive_candidate_overrides cluster `{cluster_id}` has no usable object_program"
            )
        if not isinstance(item.get("useful"), bool):
            raise ValueError(
                f"adaptive_candidate_overrides cluster `{cluster_id}` must set useful"
            )
        if not isinstance(item.get("active_by_rule"), bool):
            raise ValueError(
                f"adaptive_candidate_overrides cluster `{cluster_id}` must set active_by_rule"
            )
        if item.get("brief_support") is None:
            raise ValueError(
                f"adaptive_candidate_overrides cluster `{cluster_id}` must set brief_support"
            )
        out[cluster_id] = {
            "object_program": object_program,
            "brief_support": _float_0_1(item.get("brief_support"), default=0.35),
            "useful": bool(item.get("useful")),
            "active_by_rule": bool(item.get("active_by_rule")),
            "tier_count_hints": _sanitize_tier_count_hints(
                item.get("tier_count_hints")
                if isinstance(item.get("tier_count_hints"), Mapping)
                else cluster_rule.get("tier_count_hints"),
                object_types=_object_types_from_program(object_program),
                default_bundle_class=_default_bundle_class_for_priority(priority),
                default_preserve_level=_default_preserve_level_for_priority(priority),
                required=False,
            ),
        }
    missing = [cluster_id for cluster_id in expected if cluster_id not in out]
    if missing:
        raise ValueError(
            "adaptive_candidate_overrides omitted cluster ids: " + ", ".join(missing)
        )
    return out


def _sanitize_cluster_semantics(
    value: Any,
    *,
    deterministic_program: Mapping[str, Any],
    affordance_summary: Mapping[str, Any],
) -> dict[str, Any]:
    semantic_items = _coerce_cluster_semantics_items(value)
    if semantic_items is None:
        raise ValueError("adaptive_cluster_semantics must return a list")
    base_clusters = [
        deepcopy(cluster)
        for cluster in deterministic_program.get("active_clusters") or []
        if isinstance(cluster, Mapping)
    ]
    expected_order = [str(cluster.get("cluster_id") or "") for cluster in base_clusters]
    base_by_id = {
        str(cluster.get("cluster_id") or ""): cluster for cluster in base_clusters
    }
    allowed_cluster_ids = set(base_by_id)
    merged_by_id: dict[str, dict[str, Any]] = {}

    for item in semantic_items:
        if not isinstance(item, Mapping):
            continue
        cluster_id = str(item.get("cluster_id") or "").strip()
        if cluster_id not in allowed_cluster_ids or cluster_id in merged_by_id:
            continue
        semantic_role = str(item.get("semantic_role") or "").strip()
        if not semantic_role:
            raise ValueError(
                f"adaptive_cluster_semantics cluster `{cluster_id}` is missing semantic_role"
            )
        layout_role = _require_layout_role(
            item.get("layout_role"),
            context=f"adaptive_cluster_semantics cluster `{cluster_id}`",
        )
        if not isinstance(item.get("zone_claims"), Mapping):
            raise ValueError(
                f"adaptive_cluster_semantics cluster `{cluster_id}` must set zone_claims"
            )
        if not isinstance(item.get("relation_intents"), list):
            raise ValueError(
                f"adaptive_cluster_semantics cluster `{cluster_id}` must set relation_intents"
            )
        if not isinstance(item.get("degradation_ladder"), list):
            raise ValueError(
                f"adaptive_cluster_semantics cluster `{cluster_id}` must set degradation_ladder"
            )
        base = deepcopy(base_by_id[cluster_id])
        base["layout_role"] = layout_role
        member_types = set(_objects_from_active_cluster(base))
        base["semantic_role"] = semantic_role
        base["dominant_anchor_candidates"] = [
            candidate
            for candidate in _string_list(item.get("dominant_anchor_candidates"))
            if candidate in member_types
        ]
        base["zone_claims"] = _normalize_zone_claims(
            item.get("zone_claims"),
            fallback={},
            affordance_summary=affordance_summary,
        )
        base["relation_intents"] = _normalize_relation_intents(
            item.get("relation_intents"),
            allowed_cluster_ids=allowed_cluster_ids,
        )
        base["degradation_ladder"] = _string_list(item.get("degradation_ladder"))[
            :_DEGRADATION_STEPS_CAP_PER_CLUSTER
        ]
        merged_by_id[cluster_id] = base

    missing = [
        cluster_id for cluster_id in expected_order if cluster_id not in merged_by_id
    ]
    if missing:
        raise ValueError(
            "adaptive_cluster_semantics omitted cluster ids: " + ", ".join(missing)
        )

    out = deepcopy(dict(deterministic_program))
    out["active_clusters"] = [merged_by_id[cluster_id] for cluster_id in expected_order]
    out["macro_relations"] = _build_macro_relations(
        active_clusters=out["active_clusters"],
        room_type=deterministic_program.get("room_type"),
        affordance_summary=affordance_summary,
    )
    out["controlled_degradation"] = _build_controlled_degradation(
        active_clusters=out["active_clusters"],
        global_program=out.get("selection_constraints") or {},
    )
    out["confidence"] = max(float(out.get("confidence") or 0.75), 0.82)
    return out


def _coerce_cluster_semantics_items(value: Any) -> list[Any] | None:
    if isinstance(value, list):
        return value
    if not isinstance(value, Mapping):
        return None

    for key in ("cluster_semantics", "clusters", "items", "entries"):
        nested = value.get(key)
        if isinstance(nested, list):
            return nested

    out: list[Any] = []
    for cluster_id, item in value.items():
        if not isinstance(item, Mapping):
            continue
        entry = dict(item)
        entry.setdefault("cluster_id", str(cluster_id))
        out.append(entry)
    return out or None


def _float_0_1(value: Any, *, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    if isinstance(value, str):
        try:
            return max(0.0, min(1.0, float(value.strip())))
        except ValueError:
            return default
    return default


def _parse_json(raw: str) -> dict[str, Any]:
    text = _coerce_json_text(raw)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        text = _extract_json_object(text)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("SemanticLayoutPlanner returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("SemanticLayoutPlanner response must be a JSON object")
    return payload


def _extract_content(response: object) -> str:
    choices = getattr(response, "choices", None)
    if isinstance(choices, Sequence) and choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
    raise ValueError("OpenAI response missing message content")


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


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return text
    return text[start : end + 1]


def _all_rule_object_types(room_rules: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    clusters = room_rules.get("clusters")
    if not isinstance(clusters, list):
        return out
    for cluster in clusters:
        if not isinstance(cluster, Mapping):
            continue
        for program in (
            cluster.get("object_program"),
            *(condition.get("effects") for condition in _conditions(cluster)),
        ):
            if isinstance(program, Mapping):
                out.extend(_object_types_from_program(program))
    return _uniq(out)


def _object_types_from_program(program: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    for key in ("required", "required_if_kept", "optional"):
        out.extend(_string_list(program.get(key)))
    for key in (
        "choose_exactly_one_from",
        "choose_exactly_one_from_if_kept",
        "choose_at_least_one_from",
    ):
        for group in _list_of_string_lists(program.get(key)):
            out.extend(group)
    return out


def _conditions(cluster: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    activation = cluster.get("activation")
    if not isinstance(activation, Mapping):
        return []
    conditions = activation.get("conditions")
    if not isinstance(conditions, list):
        return []
    return [condition for condition in conditions if isinstance(condition, Mapping)]


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [
            text for item in value if (text := _clean_object_type(item)) is not None
        ]
    text = _clean_object_type(value)
    return [text] if text is not None else []


def _list_of_string_lists(value: Any) -> list[list[str]]:
    if not isinstance(value, list):
        return []
    if all(isinstance(item, str) for item in value):
        return [_string_list(value)]
    return [_string_list(item) for item in value if isinstance(item, list)]


def _clean_object_type(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _is_available(
    object_type: str, inventory_catalog: Mapping[str, Mapping[str, Any]]
) -> bool:
    item = inventory_catalog.get(object_type)
    if item is None:
        return False
    return bool(item.get("available", True))


def _select_best_object(
    options: Sequence[str],
    inventory_catalog: Mapping[str, Mapping[str, Any]],
    brief_text: str,
) -> str | None:
    available = [item for item in options if _is_available(item, inventory_catalog)]
    if not available:
        return None
    lowered = brief_text.lower()
    return max(
        available,
        key=lambda item: (
            1 if item.lower() in lowered else 0,
            _anchor_score(item),
            -available.index(item),
        ),
    )


def _anchor_score(object_type: str) -> int:
    if is_bed_like(object_type):
        return 100
    if is_lounge_anchor_like(object_type):
        return 90
    if is_work_surface_like(object_type):
        return 80
    if _is_storage_like(object_type):
        return 70
    if is_seat_like(object_type):
        return 50
    return 10


def _priority(value: Any) -> str:
    text = str(value or "support").strip().lower()
    if text in {"core", "support", "optional"}:
        return text
    return "support"


def _predicate_supported(predicate: str, brief_text: str) -> bool:
    text = brief_text.lower()
    key = predicate.lower()
    if "tv" in key or "media" in key:
        return any(token in text for token in _MEDIA_TOKENS)
    if "work" in key or "study" in key:
        return any(token in text for token in _WORK_TOKENS)
    if "pet" in key:
        return any(token in text for token in _PET_TOKENS)
    if "entry" in key:
        return any(token in text for token in _ENTRY_TOKENS)
    if "laundry" in key or "storage" in key:
        return any(token in text for token in _LAUNDRY_TOKENS)
    return False


def _brief_support_score(
    cluster_id: str,
    objects: Sequence[Any],
    brief_text: str,
) -> float:
    text = brief_text.lower()
    haystack = f"{cluster_id} " + " ".join(
        str(item.get("object_type") or item) if isinstance(item, Mapping) else str(item)
        for item in objects
    )
    tokens = set(_snake_token(haystack).split("_"))
    if len(tokens) > 1:
        tokens.discard("kitchen")
    if not text:
        return 0.35
    score = 0.35
    if tokens & set(re.findall(r"[a-z0-9]+", text)):
        score += 0.25
    if any(token in text for token in _WORK_TOKENS) and "work" in tokens:
        score += 0.35
    if any(token in text for token in _READING_TOKENS) and (
        "lounge" in tokens or "reading" in tokens
    ):
        score += 0.3
    if any(token in text for token in _MEDIA_TOKENS) and (
        "media" in tokens or "tv" in tokens
    ):
        score += 0.35
    return min(score, 1.0)


def _max_keep_for_object(
    object_type: str, object_program: Mapping[str, Any]
) -> int | None:
    limits = object_program.get("optional_limits")
    if not isinstance(limits, Mapping):
        return None
    by_object = limits.get("by_object")
    if isinstance(by_object, Mapping) and object_type in by_object:
        return _int_value(by_object.get(object_type), default=1)
    return None


def _global_optional_limit(object_program: Mapping[str, Any]) -> int:
    optional = _string_list(object_program.get("optional"))
    limits = object_program.get("optional_limits")
    if isinstance(limits, Mapping):
        global_limit = limits.get("global")
        if global_limit is not None:
            return max(0, _int_value(global_limit, default=len(optional)))
    return len(optional)


def _required_object_role(
    object_type: str,
    *,
    dominant_candidates: Sequence[str],
) -> str:
    if object_type in dominant_candidates:
        return "dominant_anchor"
    if is_profile_workflow_object(object_type):
        return "workflow_anchor"
    if is_profile_storage_object(object_type):
        return "support"
    return "support"


def _support_role(object_type: str) -> str:
    if is_profile_workflow_object(object_type):
        return "workflow_anchor"
    if is_profile_storage_object(object_type):
        return "support"
    if is_profile_floating_object(object_type):
        return "secondary_support"
    if is_bedside_support_like(object_type) or is_bench_like(object_type):
        return "secondary_support"
    if "lamp" in object_type or "plant" in object_type or "decor" in object_type:
        return "decor"
    return "support"


def _semantic_role(
    cluster_id: str,
    objects: Sequence[Mapping[str, Any]],
    priority: str,
    *,
    room_type: object | None,
) -> str:
    lowered = f"{cluster_id} {' '.join(str(row.get('object_type') or '') for row in objects)}".lower()
    object_types = [str(row.get("object_type") or "") for row in objects]
    profile_role = profile_semantic_role_for_objects(
        cluster_id=cluster_id,
        object_types=object_types,
        priority=priority,
        room_type=room_type,
    )
    if profile_role is not None:
        return profile_role
    if priority == "core" and ("sleep" in lowered or "bed" in lowered):
        return "primary_anchor_zone"
    if priority == "core":
        return "core_function_zone"
    if "work" in lowered:
        return "support_workflow_zone"
    if "storage" in lowered:
        return "support_storage_zone"
    return "secondary_support_zone"


def _activation_reason(candidate: Mapping[str, Any]) -> str:
    if candidate.get("active_by_rule"):
        return "required by compiled semantic rule and supported by inventory"
    if candidate.get("useful"):
        return (
            "activated because brief or room capacity makes this support cluster useful"
        )
    return "not activated"


def _primary_focus(active_clusters: Sequence[Mapping[str, Any]]) -> str:
    for cluster in active_clusters:
        if cluster.get("priority") != "core":
            continue
        cluster_id = str(cluster.get("cluster_id") or "").lower()
        if "sleep" in cluster_id:
            return "sleep"
        if "media" in cluster_id:
            return "media"
        if "seating" in cluster_id or "living" in cluster_id:
            return "living"
        if "kitchen" in cluster_id:
            return "kitchen"
        if "work" in cluster_id:
            return "work"
    return "mixed"


def _cluster_tag(
    cluster_id: str,
    members: Sequence[str],
    *,
    room_type: object | None,
) -> str:
    lowered = f"{cluster_id} {' '.join(members)}".lower()
    profile_tag = profile_cluster_tag_for_objects(
        members,
        room_type=room_type,
    )
    if profile_tag is not None:
        return profile_tag
    if "sleep" in lowered or any(is_bed_like(member) for member in members):
        return "sleep"
    if "dining" in lowered:
        return "dining"
    if "work" in lowered or any(
        _is_work_anchor_surface_like(member) for member in members
    ):
        return "work"
    if (
        "seating" in lowered
        or "living" in lowered
        or any(_contains_any(member, _MEDIA_TOKENS) for member in members)
        or any(
            is_lounge_anchor_like(member) or is_seat_like(member) for member in members
        )
    ):
        return "living"
    if "storage" in lowered or any(_is_storage_like(member) for member in members):
        return "storage"
    return "misc"


def _is_work_anchor_surface_like(object_type: str) -> bool:
    key = object_type.lower()
    if any(token in key for token in ("coffee_table", "side_table", "ottoman")):
        return False
    return is_work_surface_like(object_type)


def _is_storage_like(object_type: str) -> bool:
    key = object_type.lower()
    if is_profile_storage_object(key):
        return True
    return any(
        token in key
        for token in ("wardrobe", "dresser", "cabinet", "bookshelf", "shelf", "storage")
    )


def _needs_front_access(object_type: str) -> bool:
    return (
        is_seat_like(object_type)
        or is_work_surface_like(object_type)
        or _is_storage_like(object_type)
        or is_profile_wall_backed_object(object_type)
        or is_profile_floating_object(object_type)
        or is_profile_workflow_object(object_type)
        or _contains_any(object_type, _MEDIA_TOKENS)
        or is_bed_like(object_type)
    )


def _cluster_has_seating(cluster: Mapping[str, Any]) -> bool:
    return any(
        is_lounge_anchor_like(obj) or is_seat_like(obj)
        for obj in _objects_from_active_cluster(cluster)
    )


def _cluster_has_media(cluster: Mapping[str, Any]) -> bool:
    return any(
        _contains_any(obj, _MEDIA_TOKENS)
        for obj in _objects_from_active_cluster(cluster)
    )


def _cluster_has_sleep(cluster: Mapping[str, Any]) -> bool:
    return any(is_bed_like(obj) for obj in _objects_from_active_cluster(cluster))


def _cluster_has_work(cluster: Mapping[str, Any]) -> bool:
    return any(
        is_work_surface_like(obj) for obj in _objects_from_active_cluster(cluster)
    )


def _first_cluster_matching(
    active_clusters: Sequence[Mapping[str, Any]],
    predicate: Any,
) -> str | None:
    for cluster in active_clusters:
        if predicate(cluster):
            return str(cluster.get("cluster_id") or "")
    return None


def _candidate_containing_object(
    candidates: Sequence[Mapping[str, Any]], object_type: str
) -> Mapping[str, Any] | None:
    for candidate in candidates:
        for obj in candidate.get("objects") or []:
            if isinstance(obj, Mapping) and obj.get("object_type") == object_type:
                return candidate
    return None


def _room_area_m2(room_model_json: Mapping[str, Any]) -> float:
    room = (
        room_model_json.get("room")
        if isinstance(room_model_json.get("room"), Mapping)
        else {}
    )
    points = room.get("polygon_ccw") or room_model_json.get("polygon_mm") or []
    if not isinstance(points, list) or len(points) < 3:
        return 0.0
    clean: list[tuple[float, float]] = []
    for point in points:
        if isinstance(point, Mapping):
            clean.append((float(point.get("x") or 0.0), float(point.get("y") or 0.0)))
        elif isinstance(point, list) and len(point) >= 2:
            clean.append((float(point[0]), float(point[1])))
    if len(clean) < 3:
        return 0.0
    area2 = 0.0
    for idx, (x1, y1) in enumerate(clean):
        x2, y2 = clean[(idx + 1) % len(clean)]
        area2 += x1 * y2 - x2 * y1
    return abs(area2) / 2_000_000.0


def _region_labels(value: Any, *, fallback: Sequence[str]) -> list[str]:
    labels = _cap_regions(value)
    return labels if labels else _uniq(list(fallback))[:_ZONE_CLAIM_CAP_PER_CLUSTER]


def _cap_regions(value: Any) -> list[str]:
    if isinstance(value, list):
        labels: list[str] = []
        for index, item in enumerate(value, start=1):
            if isinstance(item, str) and item.strip():
                labels.append(_snake_token(item))
            elif isinstance(item, Mapping):
                label = (
                    item.get("id")
                    or item.get("label")
                    or item.get("name")
                    or f"region_{index}"
                )
                labels.append(_snake_token(str(label)))
        return _uniq(labels)[:_ZONE_CLAIM_CAP_PER_CLUSTER]
    return []


def _opening_labels(value: Sequence[Any], prefix: str) -> list[str]:
    labels: list[str] = []
    for index, item in enumerate(value, start=1):
        opening_id = None
        if isinstance(item, Mapping):
            opening_id = item.get("id") or item.get("door_id") or item.get("window_id")
        labels.append(_snake_token(f"{prefix}_{opening_id or index}"))
    return labels


def _affinity(value: Any, fallback: Any) -> str:
    text = str(value or fallback or "medium").strip().lower()
    if text in {"none", "low", "medium", "high"}:
        return text
    return "medium"


def _confidence(
    active_clusters: Sequence[Mapping[str, Any]], missing: Sequence[str]
) -> float:
    if missing:
        return 0.58 if active_clusters else 0.25
    if len(active_clusters) <= 1:
        return 0.72
    return 0.86


def _contains_any(value: str, tokens: Sequence[str]) -> bool:
    key = value.lower()
    return any(token in key for token in tokens)


def _snake_token(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def _uniq(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def _uniq_model_names(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip().lower()
        if not normalized or normalized in seen:
            continue
        out.append(value)
        seen.add(normalized)
    return out


def _int_value(value: Any, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    if isinstance(value, str):
        try:
            return int(round(float(value.strip())))
        except ValueError:
            return default
    return default
