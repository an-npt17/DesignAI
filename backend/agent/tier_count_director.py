from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

from agent.request_contract import (
    contract_intent,
    contract_item_for_object_type,
    contract_min_keep,
    contract_target_count,
    request_contract_from_payload,
)
from agent.tool_call_parser import extract_tool_calls as parse_tool_calls
from layout.room_profiles.registry import (
    apply_profile_capacity_model,
    canonical_profile_object_type,
    fallback_profile_size,
    is_profile_floating_object,
    is_profile_trait_object,
    is_profile_wall_backed_object,
    is_profile_workflow_object,
)
from prompt.tier_count import TIER_COUNT_DIRECTOR, TIER_COUNT_DIRECTOR_SYSTEM
from stylist.style_policy import extract_style_policy

logger = logging.getLogger(__name__)
_LAYOUT_SPEED_MODE_ENV = "TKNT_LAYOUT_SPEED_MODE"
_HARD_REQUEST_CONTRACT_INTENTS = {"must_keep", "must_try"}
_TARGET_REQUEST_CONTRACT_INTENTS = {"target_if_viable", "preferred_if_fit"}


def _layout_speed_mode() -> str:
    return str(os.getenv(_LAYOUT_SPEED_MODE_ENV) or "").strip().lower()


def _is_fast_layout_mode() -> bool:
    return _layout_speed_mode() in {"fast", "speed", "speed-first", "speed_first"}


def _get_tool_registry() -> dict[str, Any]:
    from tier_count.tools import TOOL_REGISTRY

    return TOOL_REGISTRY


def _get_tool_schemas() -> list[dict[str, Any]]:
    from tier_count.tools import TOOL_SCHEMAS

    return list(TOOL_SCHEMAS)


@dataclass(frozen=True)
class TierCountDirector:
    system_prompt: str = TIER_COUNT_DIRECTOR_SYSTEM
    prompt_template: str = TIER_COUNT_DIRECTOR

    def generate(
        self,
        *,
        description: str,
        special_notes: str,
        room_model_json: dict[str, Any],
        user_intent_json: dict[str, Any],
        clusters_json: dict[str, Any],
        size_profiles_json: dict[str, Any] | None = None,
        layout_failure_report_json: dict[str, Any] | None = None,
        max_steps: int = 20,
        access_clearance_ratio: float = 0.25,
    ) -> dict[str, Any]:
        clusters_json_clean = _strip_raw_text(
            _unwrap_payload(clusters_json, key="clusters")
        )

        tenant_id = ""
        if isinstance(user_intent_json, dict):
            tenant_id = str(user_intent_json.get("tenant_id") or "")
        if not tenant_id and isinstance(room_model_json, dict):
            tenant_id = str(room_model_json.get("tenant_id") or "")

        context = {
            "description": description,
            "room_model_json": room_model_json,
            "user_intent_json": user_intent_json,
            "special_notes": special_notes,
            "clusters_json": clusters_json_clean,
            "layout_failure_report_json": layout_failure_report_json,
            "tenant_id": tenant_id or None,
        }

        size_profiles_by_category: dict[str, Any] | None = None
        if isinstance(size_profiles_json, dict):
            sp = size_profiles_json.get("size_profiles_by_category")
            if isinstance(sp, dict):
                size_profiles_by_category = sp

        return _run_hardcoded_tier_count(
            context=context,
            max_steps=max_steps,
            size_profiles_by_category=size_profiles_by_category,
        )

    def tools(self) -> list[dict[str, Any]]:
        return []


PROFILE_CATEGORY_ALIASES: dict[str, str] = {
    "wardrobe_or_closet": "wardrobe",
    "book_shelf": "bookshelf",
    "desk_chair": "chair",
    "base_cabinet": "kitchen_base_cabinet",
    "counter": "kitchen_base_cabinet",
    "countertop": "kitchen_base_cabinet",
    "kitchen_counter": "kitchen_base_cabinet",
    "prep_counter": "kitchen_base_cabinet",
    "refrigerator": "fridge",
    "tall_cabinet": "kitchen_tall_cabinet",
    "wall_cabinet": "kitchen_wall_cabinet",
}

OPTIONAL_MEMBER_TOKENS = (
    "lamp",
    "basket",
    "bar_cart",
    "bean_bag",
    "pet_bed",
    "stool",
)

SECONDARY_MEMBER_TOKENS = (
    "nightstand",
    "side_table",
    "storage_cabinet",
    "bookshelf",
    "book_shelf",
    "dresser",
    "chair",
    "ottoman",
    "bench",
    "media_shelf",
)

LARGE_MEMBER_TOKENS = (
    "bed",
    "wardrobe",
    "closet",
    "sofa",
    "sectional",
    "tv_console",
    "fridge",
    "stove",
    "dishwasher",
    "sink",
    "pantry_cabinet",
    "kitchen_island",
)

MEDIUM_MEMBER_TOKENS = (
    "desk",
    "dresser",
    "storage_cabinet",
    "bookshelf",
    "book_shelf",
    "coffee_table",
    "console_table",
    "armchair",
    "tv_console",
    "kitchen_base_cabinet",
    "kitchen_tall_cabinet",
)

MINIMAL_NOTE_TOKENS = (
    "airy",
    "uncluttered",
    "minimal",
    "minimalist",
    "clean",
    "open",
    "compact",
    "simple",
)

GENEROUS_NOTE_TOKENS = (
    "cozy",
    "luxury",
    "luxurious",
    "ample",
    "generous",
    "full",
    "comfortable",
    "layered",
)

EXCLUSIVE_OBJECT_FAMILIES: tuple[tuple[str, frozenset[str]], ...] = (
    ("primary_sofa", frozenset(("sofa", "sectional_sofa"))),
)

EXCLUSIVE_FAMILY_DEFAULT_RANK: dict[str, int] = {
    "sofa": 0,
    "sectional_sofa": 1,
}


def _run_hardcoded_tier_count(
    *,
    context: dict[str, Any],
    max_steps: int,
    size_profiles_by_category: dict[str, Any] | None,
) -> dict[str, Any]:
    _ = max_steps
    clusters_json = context.get("clusters_json")
    required_types = _extract_member_types(clusters_json)
    if not required_types:
        return {
            "status": "NEED_INFO",
            "assumptions": [
                "No cluster members were available for deterministic tier counting."
            ],
            "decisions": [],
            "global_notes": ["ClusterForge did not provide usable cluster members."],
        }

    members_by_cluster = _extract_members_by_cluster(clusters_json)
    anchors_by_cluster = _extract_anchors_by_cluster(clusters_json)
    droppable_clusters = _extract_droppable_clusters(clusters_json)
    semantic_support_roles = _extract_semantic_support_roles(clusters_json)
    protected_ids_by_cluster = _extract_anchor_first_ids(
        clusters_json, field="protected_ids"
    )
    droppable_ids_by_cluster = _extract_anchor_first_ids(
        clusters_json, field="droppable_ids"
    )

    furnishing_mode = _infer_furnishing_mode(context)
    available_area_m2 = _estimate_available_area_m2(context.get("room_model_json"))
    room_scale = _classify_room_scale(available_area_m2)
    room_affordances = _extract_room_affordances(context.get("room_model_json"))
    semantic_program = _extract_semantic_program(clusters_json)
    request_contract = request_contract_from_payload(
        clusters_json if isinstance(clusters_json, dict) else {}
    )
    style_policy = extract_style_policy(clusters_json)
    room_type = _infer_room_type(context.get("room_model_json"), semantic_program)
    capacity_model = _compute_capacity_model(
        room_model_json=context.get("room_model_json"),
        available_area_m2=available_area_m2,
        room_scale=room_scale,
        furnishing_mode=furnishing_mode,
    )
    capacity_model = _apply_style_policy_to_capacity_model(
        capacity_model,
        style_policy=style_policy,
    )
    capacity_model = _apply_room_type_capacity_model(
        capacity_model,
        room_type=room_type,
    )

    logger.info(
        "TierCount utility mode: available_area_m2=%.2f room_scale=%s furnishing_mode=%s",
        available_area_m2,
        room_scale,
        furnishing_mode,
    )

    size_profiles = _ensure_size_profiles(
        required_types=required_types,
        tenant_id=context.get("tenant_id"),
        existing=size_profiles_by_category,
    )

    bundles = _build_candidate_decision_set(
        clusters_json=clusters_json,
        semantic_program=semantic_program,
        anchors_by_cluster=anchors_by_cluster,
        droppable_clusters=droppable_clusters,
        semantic_support_roles=semantic_support_roles,
        protected_ids_by_cluster=protected_ids_by_cluster,
        droppable_ids_by_cluster=droppable_ids_by_cluster,
        request_contract=request_contract,
    )
    bundles = _apply_exclusive_family_caps_to_bundles(
        bundles,
        request_contract=request_contract,
    )
    draft = _select_inventory_decision_program(
        bundles=bundles,
        room_type=room_type,
        room_scale=room_scale,
        furnishing_mode=furnishing_mode,
        capacity_model=capacity_model,
        size_profiles_by_category=size_profiles,
        semantic_program=semantic_program,
        style_policy=style_policy,
    )
    decisions = draft["decisions"]
    optional_trial_clusters = _extract_solver_trial_optional_clusters(
        clusters_json=clusters_json,
        semantic_program=semantic_program,
    )
    trial_markers = _mark_optional_solver_trial_decisions(
        draft,
        optional_trial_clusters=optional_trial_clusters,
        size_profiles_by_category=size_profiles,
    )
    if trial_markers:
        draft["solver_trial_optionals"] = trial_markers
    decisions = draft["decisions"]

    valid, detail = _validate_decisions(
        decisions,
        required_types,
        members_by_cluster=members_by_cluster,
        anchors_by_cluster=anchors_by_cluster,
        droppable_clusters=droppable_clusters,
    )
    if not valid:
        draft["status"] = "NEEDS_REVIEW"
        draft.setdefault("conflicts", []).append(detail)

    draft["assumptions"] = _build_deterministic_assumptions(
        context=context,
        furnishing_mode=furnishing_mode,
        available_area_m2=available_area_m2,
        room_affordances=room_affordances,
        style_policy=style_policy,
    )
    draft["global_notes"] = _build_deterministic_notes(
        context=context,
        furnishing_mode=furnishing_mode,
    )
    draft["budget_valid"] = draft["status"] == "OK"

    tool_registry = _get_tool_registry()
    if "estimate_budget" in tool_registry and isinstance(draft.get("decisions"), list):
        budget_out = _run_budget_check_for_result(
            result=draft,
            context=context,
            size_profiles_by_category=size_profiles,
            frozen_cluster_budget_limits_m2=None,
            rescue_mode=False,
        )
        if not _tool_output_has_error(budget_out):
            frozen_cluster_budget_limits_m2 = (
                dict(budget_out.get("cluster_budget_limits_m2"))
                if isinstance(budget_out.get("cluster_budget_limits_m2"), dict)
                else None
            )
            if bool(budget_out.get("input_decisions_fit", False)):
                return _build_ok_result(
                    decisions=list(draft.get("decisions") or []),
                    base_draft=draft,
                    size_profiles_by_category=size_profiles,
                    budget_mode="input",
                )

            recommended_decisions = budget_out.get("recommended_decisions")
            if bool(budget_out.get("recommended_decisions_fit", False)) and isinstance(
                recommended_decisions,
                list,
            ):
                return _build_ok_result(
                    decisions=recommended_decisions,
                    base_draft=draft,
                    size_profiles_by_category=size_profiles,
                    budget_mode="recommended",
                )

            if isinstance(recommended_decisions, list) and recommended_decisions:
                repaired_draft = _merge_recommended_decisions_into_draft(
                    draft,
                    recommended_decisions,
                )
                repair_budget_out = _run_budget_check_for_result(
                    result=repaired_draft,
                    context=context,
                    size_profiles_by_category=size_profiles,
                    frozen_cluster_budget_limits_m2=frozen_cluster_budget_limits_m2,
                    rescue_mode=False,
                )
                if not _tool_output_has_error(repair_budget_out):
                    if bool(repair_budget_out.get("input_decisions_fit", False)):
                        return _build_ok_result(
                            decisions=list(repaired_draft.get("decisions") or []),
                            base_draft=repaired_draft,
                            size_profiles_by_category=size_profiles,
                            budget_mode="repair_pass",
                        )
                    repair_recommended = repair_budget_out.get("recommended_decisions")
                    if bool(
                        repair_budget_out.get("recommended_decisions_fit", False)
                    ) and isinstance(repair_recommended, list):
                        return _build_ok_result(
                            decisions=repair_recommended,
                            base_draft=repaired_draft,
                            size_profiles_by_category=size_profiles,
                            budget_mode="repair_recommended",
                        )
            draft["budget_valid"] = False

    _attach_rep_dims(draft, size_profiles)
    return _repair_overfull_draft_if_needed(
        draft,
        capacity_model=capacity_model,
        size_profiles_by_category=size_profiles,
    )


def _extract_semantic_program(clusters_json: Any) -> dict[str, Any]:
    if not isinstance(clusters_json, dict):
        return {}
    semantic_program = clusters_json.get("semantic_layout_program")
    if isinstance(semantic_program, dict):
        return semantic_program
    if isinstance(clusters_json.get("active_clusters"), list):
        return clusters_json
    return {}


def _infer_room_type(room_model_json: Any, semantic_program: dict[str, Any]) -> str:
    if isinstance(room_model_json, dict):
        for key in ("room_type", "type"):
            value = room_model_json.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        room = room_model_json.get("room")
        if isinstance(room, dict):
            value = room.get("room_type") or room.get("type")
            if isinstance(value, str) and value.strip():
                return value.strip()
        meta = room_model_json.get("meta")
        if isinstance(meta, dict):
            value = meta.get("room_type") or meta.get("type")
            if isinstance(value, str) and value.strip():
                return value.strip()

    value = semantic_program.get("room_type")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "room"


def _compute_capacity_model(
    *,
    room_model_json: Any,
    available_area_m2: float,
    room_scale: str,
    furnishing_mode: str,
) -> dict[str, Any]:
    room_area_m2 = available_area_m2
    room_polygon: list[dict[str, int]] = []
    affordance_map: dict[str, Any] = {}

    if isinstance(room_model_json, dict):
        area_mm2 = room_model_json.get("area_mm2")
        if isinstance(area_mm2, (int, float)) and area_mm2 > 0:
            room_area_m2 = float(area_mm2) / 1_000_000.0

        room = room_model_json.get("room")
        if isinstance(room, dict):
            polygon = room.get("polygon_ccw")
            if isinstance(polygon, list):
                room_polygon = [p for p in polygon if isinstance(p, dict)]

        raw_affordance = room_model_json.get("affordance_map")
        if isinstance(raw_affordance, dict):
            affordance_map = raw_affordance

    perimeter_m = _polygon_perimeter_m(room_polygon)
    usable_wall_segments = _list_value(affordance_map.get("usable_wall_segments"))
    floating_zones = _list_value(affordance_map.get("floating_zone_candidates"))
    center_regions = _list_value(affordance_map.get("center_openness_regions"))
    corridors = _list_value(affordance_map.get("primary_circulation_corridors"))
    entry_zones = _list_value(affordance_map.get("entry_landing_zones"))

    wall_signal = (
        len(usable_wall_segments) if usable_wall_segments else perimeter_m / 2.5
    )
    floating_signal = len(floating_zones) if floating_zones else 1.0
    center_signal = len(center_regions) if center_regions else 1.0
    circulation_signal = len(corridors) if corridors else 1.0

    density_ratio = {"minimal": 0.30, "neutral": 0.38, "generous": 0.44}.get(
        furnishing_mode,
        0.38,
    )
    if room_scale == "small":
        density_ratio -= 0.04
    elif room_scale == "large":
        density_ratio += 0.04
    density_ratio = max(0.24, min(0.50, density_ratio))

    wall_capacity_m2 = max(room_area_m2 * 0.18, wall_signal * 0.75)
    floating_capacity_m2 = max(room_area_m2 * 0.16, floating_signal * 1.15)
    center_openness_budget_m2 = room_area_m2 * (0.26 + 0.03 * center_signal)
    circulation_budget_m2 = room_area_m2 * (0.18 + 0.02 * circulation_signal)
    clutter_budget_m2 = room_area_m2 * density_ratio

    return {
        "available_area_m2": room_area_m2,
        "target_density": "balanced",
        "density_ratio": density_ratio,
        "wall_capacity_m2": wall_capacity_m2,
        "floating_capacity_m2": floating_capacity_m2,
        "center_openness_budget_m2": center_openness_budget_m2,
        "circulation_budget_m2": circulation_budget_m2,
        "clutter_budget_m2": clutter_budget_m2,
        "entry_conflict_sensitivity": "high" if entry_zones else "moderate",
        "center_openness_weight": "high",
        "circulation_penalty_weight": "high",
        "semantic_core_preserve_weight": "very_high",
        "signals": {
            "usable_wall_segments": len(usable_wall_segments),
            "floating_zone_candidates": len(floating_zones),
            "center_openness_regions": len(center_regions),
            "primary_circulation_corridors": len(corridors),
            "entry_landing_zones": len(entry_zones),
        },
    }


def _apply_style_policy_to_capacity_model(
    capacity_model: dict[str, Any],
    *,
    style_policy: dict[str, Any],
) -> dict[str, Any]:
    if not style_policy:
        return capacity_model
    out = dict(capacity_model)
    layout_policy = _style_layout_policy(style_policy)
    weights = style_policy.get("policy_weights")
    weights = weights if isinstance(weights, dict) else {}
    density_multiplier = float(weights.get("density_multiplier") or 1.0)
    density_ratio = float(out.get("density_ratio") or 0.38) * density_multiplier
    if _style_level(layout_policy.get("center_openness_bias")) >= 3:
        density_ratio = min(density_ratio, 0.34)
        out["center_openness_weight"] = "very_high"
        out["circulation_penalty_weight"] = "very_high"
    if _style_level(layout_policy.get("clutter_tolerance")) <= 1:
        density_ratio = min(density_ratio, 0.32)
    if _style_level(layout_policy.get("clutter_tolerance")) >= 3:
        density_ratio = max(density_ratio, 0.40)
    density_ratio = max(0.22, min(0.52, density_ratio))
    room_area = max(0.1, float(out.get("available_area_m2") or 0.1))
    out["target_density"] = _style_target_density(style_policy)
    out["density_ratio"] = density_ratio
    out["clutter_budget_m2"] = room_area * density_ratio
    out["style_policy"] = {
        "style_name": style_policy.get("style_name"),
        "layout_policy": layout_policy,
    }
    return out


def _apply_room_type_capacity_model(
    capacity_model: dict[str, Any],
    *,
    room_type: str,
) -> dict[str, Any]:
    return apply_profile_capacity_model(capacity_model, room_type=room_type)


def _polygon_perimeter_m(points_mm: list[dict[str, int]]) -> float:
    if len(points_mm) < 2:
        return 0.0
    total_mm = 0.0
    for idx, point in enumerate(points_mm):
        nxt = points_mm[(idx + 1) % len(points_mm)]
        dx = float(nxt.get("x", 0)) - float(point.get("x", 0))
        dy = float(nxt.get("y", 0)) - float(point.get("y", 0))
        total_mm += (dx * dx + dy * dy) ** 0.5
    return total_mm / 1000.0


def _list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _build_candidate_decision_set(
    *,
    clusters_json: Any,
    semantic_program: dict[str, Any],
    anchors_by_cluster: dict[str, set[str]],
    droppable_clusters: set[str],
    semantic_support_roles: dict[str, dict[str, dict[str, str]]],
    protected_ids_by_cluster: dict[str, set[str]],
    droppable_ids_by_cluster: dict[str, set[str]],
    request_contract: dict[str, Any],
) -> list[dict[str, Any]]:
    clusters = _cluster_list(clusters_json)
    semantic_clusters = _semantic_clusters_by_id(semantic_program)

    bundles: list[dict[str, Any]] = []
    for cluster in clusters[:8]:
        cluster_id = str(cluster.get("cluster_id") or "").strip()
        if not cluster_id:
            continue
        semantic_cluster = semantic_clusters.get(cluster_id, {})
        cluster_bundles = _bundles_from_semantic_cluster(
            cluster=cluster,
            semantic_cluster=semantic_cluster,
            anchors=anchors_by_cluster.get(cluster_id, set()),
            droppable=cluster_id in droppable_clusters,
            semantic_support_roles=semantic_support_roles.get(cluster_id, {}),
            protected_ids=protected_ids_by_cluster.get(cluster_id, set()),
            droppable_ids=droppable_ids_by_cluster.get(cluster_id, set()),
            request_contract=request_contract,
        )
        if not cluster_bundles:
            cluster_bundles = [
                _fallback_bundle_from_cluster(
                    cluster=cluster,
                    anchors=anchors_by_cluster.get(cluster_id, set()),
                    droppable=cluster_id in droppable_clusters,
                    semantic_support_roles=semantic_support_roles.get(cluster_id, {}),
                    protected_ids=protected_ids_by_cluster.get(cluster_id, set()),
                    droppable_ids=droppable_ids_by_cluster.get(cluster_id, set()),
                    request_contract=request_contract,
                )
            ]
        bundles.extend(cluster_bundles[:4])

    return bundles


def _cluster_list(clusters_json: Any) -> list[dict[str, Any]]:
    if not isinstance(clusters_json, dict):
        return []
    clusters = clusters_json.get("clusters")
    if not isinstance(clusters, list):
        return []
    return [cluster for cluster in clusters if isinstance(cluster, dict)]


def _semantic_clusters_by_id(
    semantic_program: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    active_clusters = semantic_program.get("active_clusters")
    if not isinstance(active_clusters, list):
        return out
    for row in active_clusters:
        if not isinstance(row, dict):
            continue
        cluster_id = row.get("cluster_id")
        if isinstance(cluster_id, str) and cluster_id.strip():
            out[cluster_id] = row
    return out


def _bundles_from_semantic_cluster(
    *,
    cluster: dict[str, Any],
    semantic_cluster: dict[str, Any],
    anchors: set[str],
    droppable: bool,
    semantic_support_roles: dict[str, dict[str, str]],
    protected_ids: set[str],
    droppable_ids: set[str],
    request_contract: dict[str, Any],
) -> list[dict[str, Any]]:
    raw_bundles = semantic_cluster.get("required_bundles")
    if not isinstance(raw_bundles, list):
        return []

    members = _string_list_from_any(cluster.get("members"))
    member_lookup = _members_by_base_type(members)
    cluster_priority = str(semantic_cluster.get("priority") or "").strip().lower()
    tier_count_hints = _cluster_tier_count_hints(
        semantic_cluster,
        cluster_priority=cluster_priority or "support",
    )
    effective_cluster_droppable = (
        droppable and not _tier_count_cluster_relaxes_droppable(tier_count_hints)
    )
    bundles: list[dict[str, Any]] = []

    for raw_bundle in raw_bundles:
        if not isinstance(raw_bundle, dict):
            continue
        raw_objects = raw_bundle.get("objects")
        if not isinstance(raw_objects, list):
            continue

        objects: list[dict[str, Any]] = []
        for raw_object in raw_objects:
            if not isinstance(raw_object, dict):
                continue
            base_type = str(raw_object.get("object_type") or "").strip()
            if not base_type:
                continue
            matched_members = member_lookup.get(
                _profile_category_for_member(base_type),
                [],
            )
            for member in matched_members:
                object_hint = _tier_count_object_hint(
                    member=member,
                    cluster_hints=tier_count_hints,
                )
                member_droppable = effective_cluster_droppable and not (
                    _tier_count_object_relaxes_droppable(object_hint)
                )
                role = _resolved_object_role(
                    member=member,
                    anchors=anchors,
                    explicit_role=str(raw_object.get("role") or "support"),
                    droppable=member_droppable,
                    semantic_support_roles=semantic_support_roles,
                )
                required = (
                    bool(raw_object.get("required", False))
                    or member in protected_ids
                    or member in anchors
                ) and not member_droppable
                object_min_keep = int(object_hint["min_keep"])
                raw_object_max_keep = (
                    int(object_hint["max_keep"])
                    if object_hint["max_keep"] is not None
                    else _positive_int(raw_object.get("max_keep"), default=1)
                )
                objects.append(
                    _apply_request_contract_to_object(
                        {
                            "object_type": member,
                            "base_type": _profile_category_for_member(member),
                            "role": role,
                            "semantic_support_role": str(
                                (semantic_support_roles.get(member) or {}).get(
                                    "support_role"
                                )
                                or ""
                            ),
                            "band_intent": str(
                                (semantic_support_roles.get(member) or {}).get(
                                    "band_intent"
                                )
                                or ""
                            ),
                            "required": required,
                            "protected": member in protected_ids
                            and not member_droppable,
                            "droppable": member in droppable_ids or member_droppable,
                            "max_keep": max(raw_object_max_keep, object_min_keep),
                            "min_keep": object_min_keep,
                            "tier_count_explicit": bool(tier_count_hints["explicit"]),
                            "keep_if_space_surplus": bool(
                                object_hint["keep_if_space_surplus"]
                            ),
                            "space_surplus_threshold": float(
                                object_hint["space_surplus_threshold"]
                            ),
                            "drop_order_bias": str(object_hint["drop_order_bias"]),
                            "preferred_size_tier": object_hint["preferred_size_tier"],
                            "preserve_level": str(object_hint["preserve_level"]),
                        },
                        request_contract=request_contract,
                    )
                )

        if not objects:
            continue
        bundle_id = str(
            raw_bundle.get("bundle_id") or f"{cluster.get('cluster_id')}_bundle"
        )
        bundles.append(
            _apply_request_contract_to_bundle(
                {
                    "cluster_id": str(cluster.get("cluster_id")),
                    "bundle_id": bundle_id,
                    "bundle_class": (
                        str(tier_count_hints["bundle_class"])
                        if bool(tier_count_hints["explicit"])
                        else _bundle_class(
                            cluster_priority,
                            objects,
                            effective_cluster_droppable,
                        )
                    ),
                    "preserve_level": (
                        str(tier_count_hints["preserve_level"])
                        if bool(tier_count_hints["explicit"])
                        else _preserve_level(
                            cluster_priority,
                            objects,
                            effective_cluster_droppable,
                        )
                    ),
                    "objects": _dedupe_bundle_objects(objects, anchors),
                    "cluster_priority": cluster_priority or "support",
                    "droppable": effective_cluster_droppable,
                    "tier_count_explicit": bool(tier_count_hints["explicit"]),
                    "drop_order_bias": str(tier_count_hints["drop_order_bias"]),
                    "keep_if_space_surplus": bool(
                        tier_count_hints["keep_if_space_surplus"]
                    ),
                    "space_surplus_threshold": float(
                        tier_count_hints["space_surplus_threshold"]
                    ),
                },
            )
        )

    return bundles


def _fallback_bundle_from_cluster(
    *,
    cluster: dict[str, Any],
    anchors: set[str],
    droppable: bool,
    semantic_support_roles: dict[str, dict[str, str]],
    protected_ids: set[str],
    droppable_ids: set[str],
    request_contract: dict[str, Any],
) -> dict[str, Any]:
    cluster_id = str(cluster.get("cluster_id") or "cluster")
    tag = str(cluster.get("tag") or "").strip().lower()
    objects: list[dict[str, Any]] = []
    for member in _string_list_from_any(cluster.get("members")):
        role = _resolved_object_role(
            member=member,
            anchors=anchors,
            explicit_role=_fallback_role(member, anchors),
            droppable=droppable,
            semantic_support_roles=semantic_support_roles,
        )
        objects.append(
            _apply_request_contract_to_object(
                {
                    "object_type": member,
                    "base_type": _profile_category_for_member(member),
                    "role": role,
                    "semantic_support_role": str(
                        (semantic_support_roles.get(member) or {}).get("support_role")
                        or ""
                    ),
                    "band_intent": str(
                        (semantic_support_roles.get(member) or {}).get("band_intent")
                        or ""
                    ),
                    "required": (member in anchors or member in protected_ids)
                    and not droppable,
                    "protected": member in protected_ids and not droppable,
                    "droppable": member in droppable_ids or droppable,
                    "max_keep": _fallback_max_keep(member, tag),
                    "min_keep": 0,
                    "tier_count_explicit": False,
                    "keep_if_space_surplus": False,
                    "space_surplus_threshold": 0.45,
                    "drop_order_bias": "neutral",
                    "preferred_size_tier": None,
                    "preserve_level": _preserve_level(
                        "optional" if droppable else ("core" if anchors else "support"),
                        [],
                        droppable,
                    ),
                },
                request_contract=request_contract,
            )
        )

    priority = "optional" if droppable else ("core" if anchors else "support")
    return _apply_request_contract_to_bundle(
        {
            "cluster_id": cluster_id,
            "bundle_id": f"{cluster_id}_bundle",
            "bundle_class": _bundle_class(priority, objects, droppable),
            "preserve_level": _preserve_level(priority, objects, droppable),
            "objects": _dedupe_bundle_objects(objects, anchors),
            "cluster_priority": priority,
            "droppable": droppable,
            "tier_count_explicit": False,
            "drop_order_bias": "neutral",
            "keep_if_space_surplus": False,
            "space_surplus_threshold": 0.45,
        },
    )


def _apply_request_contract_to_object(
    obj: dict[str, Any],
    *,
    request_contract: dict[str, Any],
) -> dict[str, Any]:
    item = contract_item_for_object_type(
        request_contract,
        str(obj.get("base_type") or obj.get("object_type") or ""),
    )
    if item is None:
        return obj

    intent = contract_intent(item)
    min_keep = contract_min_keep(item)
    target_count = contract_target_count(item)
    out = dict(obj)
    out["request_contract_intent"] = intent
    out["request_contract_reason"] = str(item.get("reason") or "")
    out["request_contract_evidence"] = str(item.get("evidence") or "")
    out["request_contract_target_count"] = target_count

    if intent == "max0":
        out["min_keep"] = 0
        out["max_keep"] = 0
        out["required"] = False
        out["protected"] = False
        out["droppable"] = True
        out["tier_count_explicit"] = True
        out["keep_if_space_surplus"] = False
        out["drop_order_bias"] = "drop_first"
        out["preserve_level"] = "low"
        return out

    preferred_count = max(1, target_count)
    current_max = _positive_int(out.get("max_keep"), default=preferred_count)
    if intent in _TARGET_REQUEST_CONTRACT_INTENTS:
        out["min_keep"] = 0
        out["max_keep"] = max(current_max, preferred_count)
        out["tier_count_explicit"] = True
        out["keep_if_space_surplus"] = True
        out["space_surplus_threshold"] = min(
            0.35,
            _coerce_ratio(out.get("space_surplus_threshold"), default=0.45),
        )
        out["drop_order_bias"] = _stronger_drop_order(
            str(out.get("drop_order_bias") or "neutral"),
            "drop_late",
        )
        out["preserve_level"] = _stronger_preserve(
            str(out.get("preserve_level") or "medium"),
            "high",
        )
        if str(out.get("role") or "") not in {"dominant_anchor", "workflow_anchor"}:
            out["required"] = False
            out["protected"] = False
            out["droppable"] = True
        return out

    if intent == "optional_if_surplus":
        out["min_keep"] = 0
        out["max_keep"] = max(current_max, preferred_count)
        out["tier_count_explicit"] = True
        out["keep_if_space_surplus"] = True
        out["space_surplus_threshold"] = min(
            0.32,
            _coerce_ratio(out.get("space_surplus_threshold"), default=0.45),
        )
        out["drop_order_bias"] = "drop_late"
        out["preserve_level"] = _stronger_preserve(
            str(out.get("preserve_level") or "medium"),
            "medium",
        )
        if str(out.get("role") or "") not in {"dominant_anchor", "workflow_anchor"}:
            out["required"] = False
            out["protected"] = False
            out["droppable"] = True
        return out

    if intent not in _HARD_REQUEST_CONTRACT_INTENTS or min_keep <= 0:
        return out

    out["min_keep"] = max(min_keep, _int_value(out.get("min_keep"), default=0))
    out["max_keep"] = max(current_max, preferred_count, int(out["min_keep"]))
    out["required"] = True
    out["protected"] = True
    out["droppable"] = False
    out["tier_count_explicit"] = True
    out["keep_if_space_surplus"] = False
    out["drop_order_bias"] = "drop_last"
    out["preserve_level"] = _stronger_preserve(
        str(out.get("preserve_level") or "medium"),
        "highest",
    )
    if str(out.get("role") or "") in {"optional", "decor_light"}:
        out["role"] = "support"
    return out


def _apply_request_contract_to_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    objects = _bundle_objects(bundle)
    protected_objects = [
        obj
        for obj in objects
        if _request_contract_min_keep_from_object(obj) > 0
        and str(obj.get("request_contract_intent") or "")
        in _HARD_REQUEST_CONTRACT_INTENTS
    ]
    target_objects = [
        obj
        for obj in objects
        if str(obj.get("request_contract_intent") or "")
        in _TARGET_REQUEST_CONTRACT_INTENTS
    ]
    optional_objects = [
        obj
        for obj in objects
        if str(obj.get("request_contract_intent") or "") == "optional_if_surplus"
    ]
    if not protected_objects and not target_objects and not optional_objects:
        return bundle

    out = dict(bundle)
    if protected_objects:
        must_try = any(
            str(obj.get("request_contract_intent") or "")
            in _HARD_REQUEST_CONTRACT_INTENTS
            for obj in protected_objects
        )
        out["droppable"] = False
        out["tier_count_explicit"] = True
        out["bundle_class"] = _stronger_bundle_class(
            str(out.get("bundle_class") or "optional"),
            "indispensable" if must_try else "strong_support",
        )
        out["preserve_level"] = _stronger_preserve(
            str(out.get("preserve_level") or "medium"),
            "highest" if must_try else "high",
        )
        out["drop_order_bias"] = (
            "drop_last"
            if must_try
            else _stronger_drop_order(
                str(out.get("drop_order_bias") or "neutral"),
                "drop_late",
            )
        )
        out["keep_if_space_surplus"] = False
    elif target_objects:
        out["tier_count_explicit"] = True
        out["keep_if_space_surplus"] = True
        out["drop_order_bias"] = _stronger_drop_order(
            str(out.get("drop_order_bias") or "neutral"),
            "drop_late",
        )
        out["preserve_level"] = _stronger_preserve(
            str(out.get("preserve_level") or "medium"),
            "high",
        )
    elif optional_objects:
        out["tier_count_explicit"] = True
        out["keep_if_space_surplus"] = True
        out["drop_order_bias"] = _stronger_drop_order(
            str(out.get("drop_order_bias") or "neutral"),
            "drop_late",
        )
    return out


def _apply_exclusive_family_caps_to_bundles(
    bundles: list[dict[str, Any]],
    *,
    request_contract: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        _apply_exclusive_family_caps_to_bundle(
            bundle,
            request_contract=request_contract,
        )
        for bundle in bundles
    ]


def _apply_exclusive_family_caps_to_bundle(
    bundle: dict[str, Any],
    *,
    request_contract: dict[str, Any],
) -> dict[str, Any]:
    objects = _bundle_objects(bundle)
    if len(objects) <= 1:
        return bundle

    drops_by_object_type: dict[str, tuple[str, str]] = {}
    for family_id, family_types in EXCLUSIVE_OBJECT_FAMILIES:
        family_objects = [
            obj for obj in objects if _family_base_type(obj) in family_types
        ]
        if len(family_objects) <= 1:
            continue
        if _exclusive_family_contract_allows_multiple(
            family_types=family_types,
            request_contract=request_contract,
        ):
            continue

        winner_base_type = _exclusive_family_winner_base_type(
            family_objects=family_objects,
            family_types=family_types,
            request_contract=request_contract,
        )
        if not winner_base_type:
            continue

        for obj in family_objects:
            object_type = str(obj.get("object_type") or "")
            if object_type and _family_base_type(obj) != winner_base_type:
                drops_by_object_type[object_type] = (family_id, winner_base_type)

    if not drops_by_object_type:
        return bundle

    capped_objects: list[dict[str, Any]] = []
    for obj in objects:
        object_type = str(obj.get("object_type") or "")
        drop = drops_by_object_type.get(object_type)
        if drop is None:
            capped_objects.append(obj)
            continue

        family_id, winner_base_type = drop
        capped_objects.append(
            _drop_exclusive_family_object(
                obj,
                family_id=family_id,
                winner_base_type=winner_base_type,
            )
        )

    out = dict(bundle)
    out["objects"] = capped_objects
    return out


def _exclusive_family_contract_allows_multiple(
    *,
    family_types: frozenset[str],
    request_contract: dict[str, Any],
) -> bool:
    requested_types: set[str] = set()
    objects = request_contract.get("objects")
    if not isinstance(objects, list):
        return False

    protected_intents = (
        _HARD_REQUEST_CONTRACT_INTENTS | _TARGET_REQUEST_CONTRACT_INTENTS
    )
    for item in objects:
        if not isinstance(item, dict):
            continue
        intent = contract_intent(item)
        if intent not in protected_intents:
            continue
        target_count = contract_target_count(item)
        if target_count <= 0:
            continue
        requested_type = _profile_category_for_member(
            str(item.get("object_type") or "")
        )
        if requested_type in family_types:
            requested_types.add(requested_type)

    return len(requested_types) > 1


def _exclusive_family_winner_base_type(
    *,
    family_objects: list[dict[str, Any]],
    family_types: frozenset[str],
    request_contract: dict[str, Any],
) -> str:
    requested_type = _exclusive_family_requested_type(
        family_types=family_types,
        request_contract=request_contract,
    )
    if requested_type:
        return requested_type

    ranked = sorted(
        family_objects,
        key=lambda obj: (
            0 if str(obj.get("role") or "") == "dominant_anchor" else 1,
            0 if bool(obj.get("protected")) else 1,
            0 if bool(obj.get("required")) else 1,
            EXCLUSIVE_FAMILY_DEFAULT_RANK.get(_family_base_type(obj), 50),
            str(obj.get("object_type") or ""),
        ),
    )
    return _family_base_type(ranked[0]) if ranked else ""


def _exclusive_family_requested_type(
    *,
    family_types: frozenset[str],
    request_contract: dict[str, Any],
) -> str:
    objects = request_contract.get("objects")
    if not isinstance(objects, list):
        return ""

    candidates: list[tuple[int, int, str]] = []
    protected_intents = (
        _HARD_REQUEST_CONTRACT_INTENTS | _TARGET_REQUEST_CONTRACT_INTENTS
    )
    for index, item in enumerate(objects):
        if not isinstance(item, dict):
            continue
        intent = contract_intent(item)
        if intent not in protected_intents:
            continue
        if contract_target_count(item) <= 0:
            continue

        requested_type = _profile_category_for_member(
            str(item.get("object_type") or "")
        )
        evidence = str(item.get("evidence") or "").lower()
        if "sectional" in evidence and "sectional_sofa" in family_types:
            candidates.append((0, index, "sectional_sofa"))
            continue
        if requested_type in family_types:
            candidates.append((1, index, requested_type))

    if not candidates:
        return ""
    return sorted(candidates)[0][2]


def _drop_exclusive_family_object(
    obj: dict[str, Any],
    *,
    family_id: str,
    winner_base_type: str,
) -> dict[str, Any]:
    out = dict(obj)
    out["min_keep"] = 0
    out["max_keep"] = 0
    out["required"] = False
    out["protected"] = False
    out["droppable"] = True
    out["tier_count_explicit"] = True
    out["keep_if_space_surplus"] = False
    out["drop_order_bias"] = "drop_first"
    out["preserve_level"] = "low"
    out["exclusive_family"] = family_id
    out["exclusive_family_winner"] = winner_base_type
    out["request_contract_intent"] = ""
    out["request_contract_reason"] = ""
    out["request_contract_evidence"] = ""
    out["request_contract_target_count"] = 0
    return out


def _family_base_type(obj: dict[str, Any]) -> str:
    return _profile_category_for_member(
        str(obj.get("object_type") or obj.get("base_type") or "")
    )


def _request_contract_min_keep_from_object(obj: dict[str, Any]) -> int:
    if not obj.get("request_contract_intent"):
        return 0
    return max(0, _int_value(obj.get("min_keep"), default=0))


def _stronger_preserve(current: str, requested: str) -> str:
    values = ("highest", "high", "medium", "low")
    current_rank = (
        values.index(current) if current in values else values.index("medium")
    )
    requested_rank = (
        values.index(requested) if requested in values else values.index("medium")
    )
    return values[min(current_rank, requested_rank)]


def _stronger_drop_order(current: str, requested: str) -> str:
    values = ("drop_first", "drop_early", "neutral", "drop_late", "drop_last")
    current_rank = (
        values.index(current) if current in values else values.index("neutral")
    )
    requested_rank = (
        values.index(requested) if requested in values else values.index("neutral")
    )
    return values[max(current_rank, requested_rank)]


def _stronger_bundle_class(current: str, requested: str) -> str:
    values = ("indispensable", "strong_support", "optional", "decor_light")
    current_rank = (
        values.index(current) if current in values else values.index("optional")
    )
    requested_rank = (
        values.index(requested) if requested in values else values.index("optional")
    )
    return values[min(current_rank, requested_rank)]


def _members_by_base_type(members: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for member in members:
        out.setdefault(_profile_category_for_member(member), []).append(member)
    return out


def _dedupe_bundle_objects(
    objects: list[dict[str, Any]],
    anchors: set[str],
) -> list[dict[str, Any]]:
    by_type: dict[str, dict[str, Any]] = {}
    for obj in objects:
        object_type = str(obj.get("object_type") or "")
        if not object_type:
            continue
        existing = by_type.get(object_type)
        if existing is None:
            by_type[object_type] = obj
            continue
        existing_min = max(0, _int_value(existing.get("min_keep"), default=0))
        obj_min = max(0, _int_value(obj.get("min_keep"), default=0))
        existing_max = (
            max(0, _int_value(existing.get("max_keep"), default=1))
            if existing.get("max_keep") is not None
            else 1
        )
        obj_max = (
            max(0, _int_value(obj.get("max_keep"), default=1))
            if obj.get("max_keep") is not None
            else 1
        )
        existing["min_keep"] = max(existing_min, obj_min)
        existing["max_keep"] = max(existing_max, obj_max, int(existing["min_keep"]))
        existing["required"] = bool(existing.get("required")) or bool(
            obj.get("required")
        )
        existing["protected"] = bool(existing.get("protected")) or bool(
            obj.get("protected")
        )
        existing["droppable"] = bool(existing.get("droppable")) and bool(
            obj.get("droppable")
        )
        if not existing.get("semantic_support_role") and obj.get(
            "semantic_support_role"
        ):
            existing["semantic_support_role"] = obj.get("semantic_support_role")
        if not existing.get("band_intent") and obj.get("band_intent"):
            existing["band_intent"] = obj.get("band_intent")
        if not existing.get("request_contract_intent") and obj.get(
            "request_contract_intent"
        ):
            existing["request_contract_intent"] = obj.get("request_contract_intent")
            existing["request_contract_reason"] = obj.get("request_contract_reason")
            existing["request_contract_evidence"] = obj.get("request_contract_evidence")
        if str(existing.get("request_contract_intent") or "") == "max0":
            existing["min_keep"] = 0
            existing["max_keep"] = 0
            existing["required"] = False
            existing["protected"] = False
            existing["droppable"] = True
            continue
        if int(existing.get("min_keep") or 0) > 0 and existing.get(
            "request_contract_intent"
        ):
            existing["required"] = True
            existing["protected"] = True
            existing["droppable"] = False
            existing["tier_count_explicit"] = True
            existing["drop_order_bias"] = _stronger_drop_order(
                str(existing.get("drop_order_bias") or "neutral"),
                "drop_last",
            )
            existing["preserve_level"] = _stronger_preserve(
                str(existing.get("preserve_level") or "medium"),
                "high",
            )
        if object_type in anchors:
            existing["role"] = "dominant_anchor"
    return list(by_type.values())


def _extract_semantic_support_roles(
    clusters_json: Any,
) -> dict[str, dict[str, dict[str, str]]]:
    if not isinstance(clusters_json, dict):
        return {}
    clusters = clusters_json.get("clusters")
    if not isinstance(clusters, list):
        return {}
    out: dict[str, dict[str, dict[str, str]]] = {}
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        cluster_id = str(cluster.get("cluster_id") or "").strip()
        rules = cluster.get("cluster_rules")
        if not cluster_id or not isinstance(rules, dict):
            continue
        semantic_rows = rules.get("semantic_placements")
        if not isinstance(semantic_rows, list):
            continue
        by_member: dict[str, dict[str, str]] = {}
        for row in semantic_rows:
            if not isinstance(row, dict):
                continue
            object_id = str(row.get("id") or "").strip()
            if not object_id:
                continue
            by_member[object_id] = {
                "support_role": str(row.get("support_role") or "").strip().lower(),
                "band_intent": str(row.get("band_intent") or "").strip().lower(),
                "orientation": str(row.get("orientation") or "").strip().lower(),
            }
        if by_member:
            out[cluster_id] = by_member
    return out


def _extract_anchor_first_ids(clusters_json: Any, *, field: str) -> dict[str, set[str]]:
    if not isinstance(clusters_json, dict):
        return {}
    clusters = clusters_json.get("clusters")
    if not isinstance(clusters, list):
        return {}
    out: dict[str, set[str]] = {}
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        cluster_id = str(cluster.get("cluster_id") or "").strip()
        rules = cluster.get("cluster_rules")
        if not cluster_id or not isinstance(rules, dict):
            continue
        anchor_policy = rules.get("anchor_first_policy")
        values = anchor_policy.get(field) if isinstance(anchor_policy, dict) else None
        if isinstance(values, list):
            out[cluster_id] = {
                str(item).strip()
                for item in values
                if isinstance(item, str) and str(item).strip()
            }
    return out


def _resolved_object_role(
    *,
    member: str,
    anchors: set[str],
    explicit_role: str,
    droppable: bool,
    semantic_support_roles: dict[str, dict[str, str]],
) -> str:
    if droppable:
        return "decor_light"
    if member in anchors:
        return "dominant_anchor"
    semantic_role = (
        str((semantic_support_roles.get(member) or {}).get("support_role") or "")
        .strip()
        .lower()
    )
    if semantic_role == "frontal_support":
        return "support"
    if semantic_role == "secondary_seat":
        return "secondary_support"
    if semantic_role in {"side_support", "wall_support"}:
        return "support"
    if semantic_role == "peripheral_support":
        return "optional"
    role = str(explicit_role or "support").strip().lower()
    return role or "support"


def _semantic_support_role_bias(obj: dict[str, Any]) -> float:
    role = str(obj.get("semantic_support_role") or "").strip().lower()
    band_intent = str(obj.get("band_intent") or "").strip().lower()
    if role == "frontal_support" or band_intent == "front_band":
        return 0.95
    if role == "secondary_seat" or band_intent == "flank_band":
        return 0.55
    if role in {"side_support", "wall_support"}:
        return 0.32
    if role == "peripheral_support":
        return 0.12
    return 0.0


def _fallback_role(member: str, anchors: set[str]) -> str:
    if member in anchors:
        return "dominant_anchor"
    member_key = _norm_key(member)
    if is_profile_workflow_object(member_key):
        return "workflow_anchor"
    if is_profile_trait_object(member_key):
        return "support"
    if any(token in member_key for token in OPTIONAL_MEMBER_TOKENS):
        return "decor_light"
    if any(token in member_key for token in SECONDARY_MEMBER_TOKENS):
        return "support"
    if any(token in member_key for token in MEDIUM_MEMBER_TOKENS):
        return "workflow_anchor"
    return "support"


def _fallback_max_keep(member: str, cluster_tag: str) -> int:
    _ = member, cluster_tag
    return 1


def _bundle_class(
    priority: str,
    objects: list[dict[str, Any]],
    droppable: bool,
) -> str:
    if droppable:
        return "decor_light"
    if priority == "core" or any(
        str(obj.get("role")) == "dominant_anchor" and bool(obj.get("required"))
        for obj in objects
    ):
        return "indispensable"
    if priority in {"support", "strong_support"} or any(
        bool(obj.get("required")) for obj in objects
    ):
        return "strong_support"
    return "optional"


def _preserve_level(
    priority: str,
    objects: list[dict[str, Any]],
    droppable: bool,
) -> str:
    if droppable:
        return "low"
    if priority == "core" or any(
        str(obj.get("role")) == "dominant_anchor" for obj in objects
    ):
        return "highest"
    if priority in {"support", "strong_support"}:
        return "high"
    return "medium"


def _default_tier_count_bundle_class(priority: str) -> str:
    if priority == "core":
        return "indispensable"
    if priority == "support":
        return "strong_support"
    return "optional"


def _default_tier_count_preserve_level(priority: str) -> str:
    if priority == "core":
        return "highest"
    if priority == "support":
        return "high"
    return "medium"


def _cluster_tier_count_hints(
    semantic_cluster: dict[str, Any],
    *,
    cluster_priority: str,
) -> dict[str, Any]:
    raw = semantic_cluster.get("tier_count_hints")
    explicit = isinstance(raw, dict)
    hints = raw if isinstance(raw, dict) else {}
    bundle_class = _tier_count_bundle_class(
        hints.get("bundle_class"),
        default=_default_tier_count_bundle_class(cluster_priority),
    )
    preserve_level = _tier_count_preserve_level(
        hints.get("preserve_level"),
        default=_default_tier_count_preserve_level(cluster_priority),
    )
    keep_if_space_surplus = (
        bool(hints.get("keep_if_space_surplus")) if explicit else False
    )
    space_surplus_threshold = _coerce_ratio(
        hints.get("space_surplus_threshold"),
        default=0.45,
    )
    drop_order_bias = _tier_count_drop_order_bias(
        hints.get("drop_order_bias"),
        default="neutral",
    )
    object_hints_by_type: dict[str, dict[str, Any]] = {}
    object_hints = hints.get("object_hints")
    if isinstance(object_hints, list):
        for item in object_hints:
            if not isinstance(item, dict):
                continue
            object_type = _decision_type_id(item)
            if object_type is None:
                continue
            raw_max_keep = item.get("max_keep")
            max_keep = (
                max(0, _int_value(raw_max_keep, default=0))
                if raw_max_keep is not None
                else None
            )
            min_keep = max(0, _int_value(item.get("min_keep"), default=0))
            if max_keep is not None:
                min_keep = min(min_keep, max_keep)
            object_hints_by_type[object_type] = {
                "object_type": object_type,
                "min_keep": min_keep,
                "max_keep": max_keep,
                "keep_if_space_surplus": bool(
                    item.get("keep_if_space_surplus", keep_if_space_surplus)
                ),
                "space_surplus_threshold": _coerce_ratio(
                    item.get("space_surplus_threshold"),
                    default=space_surplus_threshold,
                ),
                "drop_order_bias": _tier_count_drop_order_bias(
                    item.get("drop_order_bias"),
                    default=drop_order_bias,
                ),
                "preserve_level": _tier_count_preserve_level(
                    item.get("preserve_level"),
                    default=preserve_level,
                ),
                "preferred_size_tier": _tier_count_size_tier(
                    item.get("preferred_size_tier")
                ),
            }

    return {
        "explicit": explicit,
        "bundle_class": bundle_class,
        "preserve_level": preserve_level,
        "keep_if_space_surplus": keep_if_space_surplus,
        "space_surplus_threshold": space_surplus_threshold,
        "drop_order_bias": drop_order_bias,
        "object_hints_by_type": object_hints_by_type,
    }


def _tier_count_object_hint(
    *,
    member: str,
    cluster_hints: dict[str, Any],
) -> dict[str, Any]:
    by_type = (
        cluster_hints.get("object_hints_by_type")
        if isinstance(cluster_hints.get("object_hints_by_type"), dict)
        else {}
    )
    value = by_type.get(member)
    if isinstance(value, dict):
        return _relax_surplus_support_zero_cap(
            member=member,
            object_hint=dict(value),
            cluster_hints=cluster_hints,
        )
    return _relax_surplus_support_zero_cap(
        member=member,
        object_hint={
            "object_type": member,
            "min_keep": 0,
            "max_keep": None,
            "keep_if_space_surplus": bool(
                cluster_hints.get("keep_if_space_surplus") if cluster_hints else False
            ),
            "space_surplus_threshold": float(
                cluster_hints.get("space_surplus_threshold") if cluster_hints else 0.45
            ),
            "drop_order_bias": str(
                cluster_hints.get("drop_order_bias") if cluster_hints else "neutral"
            ),
            "preserve_level": str(
                cluster_hints.get("preserve_level") if cluster_hints else "medium"
            ),
            "preferred_size_tier": None,
        },
        cluster_hints=cluster_hints,
    )


def _relax_surplus_support_zero_cap(
    *,
    member: str,
    object_hint: dict[str, Any],
    cluster_hints: dict[str, Any],
) -> dict[str, Any]:
    if _int_value(object_hint.get("max_keep"), default=1) != 0:
        return object_hint
    if not bool(cluster_hints.get("keep_if_space_surplus")):
        return object_hint
    if not _member_matches_any_token(member, SECONDARY_MEMBER_TOKENS):
        return object_hint
    if _member_matches_any_token(member, OPTIONAL_MEMBER_TOKENS):
        return object_hint
    if _member_matches_any_token(member, LARGE_MEMBER_TOKENS):
        return object_hint
    out = dict(object_hint)
    out["max_keep"] = 1
    out["keep_if_space_surplus"] = True
    out["space_surplus_threshold"] = max(
        0.45,
        _coerce_ratio(out.get("space_surplus_threshold"), default=0.5),
    )
    out["drop_order_bias"] = _stronger_drop_order(
        str(out.get("drop_order_bias") or "neutral"),
        "neutral",
    )
    out["preserve_level"] = _stronger_preserve(
        str(out.get("preserve_level") or "medium"),
        "medium",
    )
    return out


def _member_matches_any_token(member: str, tokens: tuple[str, ...]) -> bool:
    member_key = _norm_key(member)
    return any(token in member_key for token in tokens)


def _tier_count_cluster_relaxes_droppable(cluster_hints: dict[str, Any]) -> bool:
    if not bool(cluster_hints.get("explicit")):
        return False
    if str(cluster_hints.get("bundle_class") or "") != "decor_light":
        return True
    if str(cluster_hints.get("preserve_level") or "") in {"highest", "high", "medium"}:
        return True
    if bool(cluster_hints.get("keep_if_space_surplus")):
        return True
    by_type = cluster_hints.get("object_hints_by_type")
    if isinstance(by_type, dict):
        return any(
            _tier_count_object_relaxes_droppable(value)
            for value in by_type.values()
            if isinstance(value, dict)
        )
    return False


def _tier_count_object_relaxes_droppable(object_hint: dict[str, Any]) -> bool:
    if int(object_hint.get("min_keep") or 0) > 0:
        return True
    if bool(object_hint.get("keep_if_space_surplus")):
        return True
    if str(object_hint.get("preserve_level") or "") in {"highest", "high", "medium"}:
        return True
    return str(object_hint.get("drop_order_bias") or "") in {
        "drop_late",
        "drop_last",
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


def _coerce_ratio(value: Any, *, default: float) -> float:
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


def _effective_preserve_level(*, obj: dict[str, Any], bundle: dict[str, Any]) -> str:
    return _tier_count_preserve_level(
        obj.get("preserve_level"),
        default=str(bundle.get("preserve_level") or "medium"),
    )


def _effective_drop_order_bias(*, obj: dict[str, Any], bundle: dict[str, Any]) -> str:
    return _tier_count_drop_order_bias(
        obj.get("drop_order_bias"),
        default=str(bundle.get("drop_order_bias") or "neutral"),
    )


def _effective_space_surplus_threshold(
    *,
    obj: dict[str, Any],
    bundle: dict[str, Any],
    default: float,
) -> float:
    object_value = obj.get("space_surplus_threshold")
    if object_value is not None:
        return _coerce_ratio(object_value, default=default)
    bundle_value = bundle.get("space_surplus_threshold")
    if bundle_value is not None:
        return _coerce_ratio(bundle_value, default=default)
    return default


def _explicit_space_surplus_keep(
    *, obj: dict[str, Any], bundle: dict[str, Any]
) -> bool:
    if bool(obj.get("keep_if_space_surplus")):
        return True
    return bool(bundle.get("keep_if_space_surplus"))


def _has_explicit_tier_count_guidance(
    *, obj: dict[str, Any], bundle: dict[str, Any]
) -> bool:
    if bool(obj.get("tier_count_explicit")):
        return True
    return bool(bundle.get("tier_count_explicit"))


def _drop_order_rank(value: str) -> int:
    return {
        "drop_first": 0,
        "drop_early": 1,
        "neutral": 2,
        "drop_late": 3,
        "drop_last": 4,
    }.get(value, 2)


def _drop_order_keep_rank(value: str) -> int:
    return {
        "drop_last": 0,
        "drop_late": 1,
        "neutral": 2,
        "drop_early": 3,
        "drop_first": 4,
    }.get(value, 2)


def _preserve_keep_rank(value: str) -> int:
    return {
        "highest": 0,
        "high": 1,
        "medium": 2,
        "low": 3,
    }.get(value, 2)


def _preserve_drop_rank(value: str) -> int:
    return {
        "low": 0,
        "medium": 1,
        "high": 2,
        "highest": 3,
    }.get(value, 1)


def _select_inventory_decision_program(
    *,
    bundles: list[dict[str, Any]],
    room_type: str,
    room_scale: str,
    furnishing_mode: str,
    capacity_model: dict[str, Any],
    size_profiles_by_category: dict[str, Any],
    semantic_program: dict[str, Any],
    style_policy: dict[str, Any],
) -> dict[str, Any]:
    used_footprint_m2 = 0.0
    selected_count = 0
    dropped_types: list[str] = []
    conflicts: list[str] = []
    decisions: list[dict[str, Any]] = []
    cluster_decisions_by_id: dict[str, dict[str, Any]] = {}

    for bundle in sorted(bundles, key=_bundle_sort_key):
        cluster_id = str(bundle.get("cluster_id") or "cluster")
        bundle_decision = {
            "bundle_id": str(bundle.get("bundle_id") or f"{cluster_id}_bundle"),
            "preserve_level": str(bundle.get("preserve_level") or "medium"),
            "bundle_class": str(bundle.get("bundle_class") or "optional"),
            "objects": [],
        }
        cluster_decision = cluster_decisions_by_id.setdefault(
            cluster_id,
            {
                "cluster_id": cluster_id,
                "decision_status": "active",
                "selected_bundles": [],
            },
        )
        cluster_decision["selected_bundles"].append(bundle_decision)

        for obj in sorted(
            _bundle_objects(bundle),
            key=lambda row: _object_selection_sort_key(row, bundle),
        ):
            score = _score_object_utility(
                obj=obj,
                bundle=bundle,
                used_footprint_m2=used_footprint_m2,
                capacity_model=capacity_model,
                size_profiles_by_category=size_profiles_by_category,
                style_policy=style_policy,
            )
            quantity = _select_quantity(
                obj=obj,
                bundle=bundle,
                score=score,
                used_footprint_m2=used_footprint_m2,
                capacity_model=capacity_model,
                size_profiles_by_category=size_profiles_by_category,
                room_scale=room_scale,
                furnishing_mode=furnishing_mode,
            )
            size_tier = _select_size_tier(
                obj=obj,
                quantity=quantity,
                score=score,
                used_footprint_m2=used_footprint_m2,
                capacity_model=capacity_model,
                size_profiles_by_category=size_profiles_by_category,
                room_scale=room_scale,
                furnishing_mode=furnishing_mode,
            )
            footprint_m2 = _footprint_for_object(
                obj=obj,
                quantity=quantity,
                size_tier=size_tier or "S",
                size_profiles_by_category=size_profiles_by_category,
            )

            if quantity > 0:
                used_footprint_m2 += footprint_m2
                selected_count += quantity
            else:
                dropped_types.append(str(obj["object_type"]))

            role = str(obj.get("role") or "support")
            reason = _decision_reason(
                quantity=quantity,
                role=role,
                score=score,
                obj=obj,
            )
            priority = _decision_priority(obj=obj, role=role, bundle=bundle)
            decision_object = {
                "object_type": str(obj["object_type"]),
                "quantity": quantity,
                "size_tier": size_tier if quantity > 0 else None,
                "role": role,
                "preserve_level": _effective_preserve_level(obj=obj, bundle=bundle),
                "drop_order_bias": _effective_drop_order_bias(obj=obj, bundle=bundle),
                "min_keep": max(0, _int_value(obj.get("min_keep"), default=0)),
                "keep_if_space_surplus": _explicit_space_surplus_keep(
                    obj=obj,
                    bundle=bundle,
                ),
                "space_surplus_threshold": _effective_space_surplus_threshold(
                    obj=obj,
                    bundle=bundle,
                    default=0.42,
                ),
                "preferred_size_tier": _tier_count_size_tier(
                    obj.get("preferred_size_tier")
                ),
                "request_contract_intent": str(
                    obj.get("request_contract_intent") or ""
                ),
                "request_contract_reason": str(
                    obj.get("request_contract_reason") or ""
                ),
                "request_contract_evidence": str(
                    obj.get("request_contract_evidence") or ""
                ),
                "request_contract_target_count": max(
                    0,
                    _int_value(obj.get("request_contract_target_count"), default=0),
                ),
                "decision_reason": reason,
                "utility_score": round(float(score["total"]), 3),
                "utility_breakdown": {
                    key: round(float(value), 3) for key, value in score.items()
                },
            }
            _copy_exclusive_family_trace(obj, decision_object)
            bundle_decision["objects"].append(decision_object)

            decision_row = {
                "object_type": str(obj["object_type"]),
                "category": str(obj["object_type"]),
                "cluster_id": cluster_id,
                "quantity": quantity,
                "size_tier": size_tier or "S",
                "priority": priority,
                "preserve_level": _effective_preserve_level(
                    obj=obj,
                    bundle=bundle,
                ),
                "bundle_id": str(bundle.get("bundle_id") or ""),
                "role": role,
                "semantic_support_role": str(obj.get("semantic_support_role") or ""),
                "band_intent": str(obj.get("band_intent") or ""),
                "protected": bool(obj.get("protected")),
                "droppable": bool(obj.get("droppable")),
                "drop_order_bias": _effective_drop_order_bias(
                    obj=obj,
                    bundle=bundle,
                ),
                "min_keep": max(0, _int_value(obj.get("min_keep"), default=0)),
                "keep_if_space_surplus": _explicit_space_surplus_keep(
                    obj=obj,
                    bundle=bundle,
                ),
                "space_surplus_threshold": _effective_space_surplus_threshold(
                    obj=obj,
                    bundle=bundle,
                    default=0.42,
                ),
                "preferred_size_tier": _tier_count_size_tier(
                    obj.get("preferred_size_tier")
                ),
                "request_contract_intent": str(
                    obj.get("request_contract_intent") or ""
                ),
                "request_contract_reason": str(
                    obj.get("request_contract_reason") or ""
                ),
                "request_contract_evidence": str(
                    obj.get("request_contract_evidence") or ""
                ),
                "request_contract_target_count": max(
                    0,
                    _int_value(obj.get("request_contract_target_count"), default=0),
                ),
                "rationale": reason,
                "utility_score": round(float(score["total"]), 3),
            }
            _copy_exclusive_family_trace(obj, decision_row)
            decisions.append(decision_row)

            if quantity > 0 and _is_core_object(obj, bundle) and score["total"] < 0:
                conflicts.append(
                    f"{obj['object_type']} is semantically required but has negative utility under capacity."
                )

    circulation_pressure = _circulation_pressure(
        used_footprint_m2=used_footprint_m2,
        capacity_model=capacity_model,
    )
    status = _decision_status(
        decisions=decisions,
        conflicts=conflicts,
        used_footprint_m2=used_footprint_m2,
        capacity_model=capacity_model,
    )

    return {
        "status": status,
        "room_type": room_type,
        "cluster_decisions": list(cluster_decisions_by_id.values()),
        "global_density_policy": {
            "target_density": _style_target_density(style_policy),
            "clutter_budget_level": _style_clutter_budget_level(
                furnishing_mode,
                style_policy,
            ),
            "center_openness_preserved": circulation_pressure <= 0.72,
            "style_policy_applied": bool(style_policy),
        },
        "capacity_model": capacity_model,
        "decision_summary": {
            "selected_object_count": selected_count,
            "dropped_object_types": sorted(set(dropped_types)),
            "estimated_footprint_mm2": int(round(used_footprint_m2 * 1_000_000)),
            "estimated_circulation_pressure": round(circulation_pressure, 3),
        },
        "degradation_ready_order": _build_degradation_ready_order(decisions),
        "missing": list(semantic_program.get("missing") or []),
        "conflicts": conflicts,
        "confidence": _inventory_confidence(status, conflicts, decisions),
        "notes": [
            "Tier Count used deterministic utility, capacity, and bundle-preserving rules, plus Forge semantic tier-count hints when available.",
            "Tier Count itself did not run an additional LLM call.",
            "Output is synchronized for the object-level anchor-first solver handoff.",
        ],
        "decisions": decisions,
    }


def _bundle_sort_key(bundle: dict[str, Any]) -> tuple[int, int, int, str]:
    rank = {
        "indispensable": 0,
        "strong_support": 1,
        "optional": 2,
        "decor_light": 3,
    }
    return (
        rank.get(str(bundle.get("bundle_class") or "optional"), 4),
        _drop_order_keep_rank(str(bundle.get("drop_order_bias") or "neutral")),
        _preserve_keep_rank(str(bundle.get("preserve_level") or "medium")),
        str(bundle.get("bundle_id") or ""),
    )


def _bundle_objects(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    objects = bundle.get("objects")
    if not isinstance(objects, list):
        return []
    return [obj for obj in objects if isinstance(obj, dict)]


def _object_selection_sort_key(
    obj: dict[str, Any],
    bundle: dict[str, Any],
) -> tuple[int, int, int, int, str]:
    role_rank = {
        "dominant_anchor": 0,
        "workflow_anchor": 1,
        "support": 2,
        "secondary_support": 3,
        "decor": 4,
        "decor_light": 5,
        "optional": 6,
    }
    return (
        0 if _is_core_object(obj, bundle) else 1,
        _preserve_keep_rank(_effective_preserve_level(obj=obj, bundle=bundle)),
        _drop_order_keep_rank(_effective_drop_order_bias(obj=obj, bundle=bundle)),
        role_rank.get(str(obj.get("role") or "support"), 4),
        str(obj.get("object_type") or ""),
    )


def _score_object_utility(
    *,
    obj: dict[str, Any],
    bundle: dict[str, Any],
    used_footprint_m2: float,
    capacity_model: dict[str, Any],
    size_profiles_by_category: dict[str, Any],
    style_policy: dict[str, Any],
) -> dict[str, float]:
    role = str(obj.get("role") or "support")
    object_type = str(obj.get("object_type") or "")
    base_type = str(obj.get("base_type") or _profile_category_for_member(object_type))
    required = bool(obj.get("required"))
    bundle_class = str(bundle.get("bundle_class") or "optional")

    functional = _functional_utility(role, base_type)
    semantic = _semantic_utility(
        role=role,
        required=required,
        bundle_class=bundle_class,
        preserve_level=_effective_preserve_level(obj=obj, bundle=bundle),
    )
    completeness = _completeness_utility(base_type, bundle)
    footprint_m2 = _footprint_for_object(
        obj=obj,
        quantity=1,
        size_tier="M",
        size_profiles_by_category=size_profiles_by_category,
    )
    roomfit = _roomfit_utility(
        base_type=base_type,
        footprint_m2=footprint_m2,
        capacity_model=capacity_model,
    )
    clutter = _clutter_penalty(
        footprint_m2=footprint_m2,
        used_footprint_m2=used_footprint_m2,
        capacity_model=capacity_model,
    )
    circulation = _circulation_penalty(
        base_type=base_type,
        footprint_m2=footprint_m2,
        capacity_model=capacity_model,
    )
    redundancy = _redundancy_penalty(object_type, bundle)
    style_bias = _style_object_utility_bias(
        obj=obj,
        bundle=bundle,
        style_policy=style_policy,
    )
    semantic_support_bias = _semantic_support_role_bias(obj)
    protected_bonus = 0.9 if bool(obj.get("protected")) else 0.0
    tier_count_hint_bias = _tier_count_hint_utility_bias(
        obj=obj,
        bundle=bundle,
        used_footprint_m2=used_footprint_m2,
        capacity_model=capacity_model,
    )
    space_surplus_bonus = _surplus_optional_support_bonus(
        obj=obj,
        bundle=bundle,
        used_footprint_m2=used_footprint_m2,
        capacity_model=capacity_model,
        style_policy=style_policy,
    )
    total = (
        functional
        + semantic
        + completeness
        + roomfit
        + style_bias
        + semantic_support_bias
        + protected_bonus
        + tier_count_hint_bias
        + space_surplus_bonus
        - clutter
        - circulation
        - redundancy
    )

    return {
        "function": functional,
        "semantic": semantic,
        "completeness": completeness,
        "roomfit": roomfit,
        "clutter_penalty": clutter,
        "circulation_penalty": circulation,
        "redundancy_penalty": redundancy,
        "style_policy_bias": style_bias,
        "semantic_support_bias": semantic_support_bias,
        "protected_bonus": protected_bonus,
        "tier_count_hint_bias": tier_count_hint_bias,
        "space_surplus_bonus": space_surplus_bonus,
        "total": total,
    }


def _tier_count_hint_utility_bias(
    *,
    obj: dict[str, Any],
    bundle: dict[str, Any],
    used_footprint_m2: float,
    capacity_model: dict[str, Any],
) -> float:
    preserve_bias = {
        "highest": 0.95,
        "high": 0.55,
        "medium": 0.15,
        "low": -0.15,
    }
    drop_bias = {
        "drop_last": 0.85,
        "drop_late": 0.35,
        "neutral": 0.0,
        "drop_early": -0.3,
        "drop_first": -0.75,
    }
    preserve_level = _effective_preserve_level(obj=obj, bundle=bundle)
    drop_order_bias = _effective_drop_order_bias(obj=obj, bundle=bundle)
    min_keep = max(0, _int_value(obj.get("min_keep"), default=0))
    explicit_guidance = _has_explicit_tier_count_guidance(obj=obj, bundle=bundle)
    if (
        not explicit_guidance
        and min_keep <= 0
        and not _explicit_space_surplus_keep(
            obj=obj,
            bundle=bundle,
        )
    ):
        return 0.0
    bonus = preserve_bias.get(preserve_level, 0.0) + drop_bias.get(
        drop_order_bias,
        0.0,
    )
    if min_keep > 0:
        bonus += 0.45 + min(0.35, 0.18 * float(min_keep - 1))
    if _explicit_space_surplus_keep(obj=obj, bundle=bundle):
        surplus = _space_surplus_ratio(
            capacity_model=capacity_model,
            used_footprint_m2=used_footprint_m2,
        )
        threshold = _effective_space_surplus_threshold(
            obj=obj,
            bundle=bundle,
            default=0.42,
        )
        if surplus >= threshold:
            bonus += 0.55 * min(1.0, (surplus - threshold + 0.18) / 0.36)
        else:
            bonus -= 0.12
    return max(-1.25, min(1.95, bonus))


def _functional_utility(role: str, base_type: str) -> float:
    role_scores = {
        "dominant_anchor": 5.0,
        "workflow_anchor": 4.3,
        "support": 2.7,
        "secondary_support": 2.1,
        "decor_light": 0.9,
        "optional": 0.8,
    }
    score = role_scores.get(role, 2.0)
    if any(
        token in base_type
        for token in ("bed", "desk", "dining_table", "sofa", "tv_console")
    ):
        score += 0.6
    if is_profile_workflow_object(base_type):
        score += 0.75
    elif is_profile_trait_object(base_type):
        score += 0.35
    return score


def _semantic_utility(
    *,
    role: str,
    required: bool,
    bundle_class: str,
    preserve_level: str,
) -> float:
    score = 1.2 if required else 0.0
    if bundle_class == "indispensable":
        score += 2.0
    elif bundle_class == "strong_support":
        score += 1.0
    elif bundle_class == "decor_light":
        score -= 0.3
    if preserve_level == "highest":
        score += 1.0
    if role == "dominant_anchor":
        score += 0.8
    return score


def _completeness_utility(base_type: str, bundle: dict[str, Any]) -> float:
    objects = _bundle_objects(bundle)
    matched = [
        obj
        for obj in objects
        if str(
            obj.get("base_type")
            or _profile_category_for_member(str(obj.get("object_type") or ""))
        )
        == base_type
    ]
    if any(_request_contract_min_keep_from_object(obj) > 0 for obj in matched):
        return 1.45
    if any(
        _int_value(obj.get("request_contract_target_count"), default=0) > 1
        for obj in matched
    ):
        return 1.25
    support_signals = {
        str(obj.get("semantic_support_role") or "") for obj in matched
    } | {str(obj.get("band_intent") or "") for obj in matched}
    if any(signal for signal in support_signals):
        return 1.1
    if any(
        str(obj.get("role") or "") in {"dominant_anchor", "workflow_anchor"}
        for obj in objects
    ) and any(
        str(obj.get("role") or "") in {"support", "secondary_support"}
        for obj in matched
    ):
        return 0.9
    return 0.45


def _roomfit_utility(
    *,
    base_type: str,
    footprint_m2: float,
    capacity_model: dict[str, Any],
) -> float:
    placement_mode = _placement_mode(base_type)
    capacity_key = (
        "wall_capacity_m2"
        if placement_mode == "wall_backed"
        else "floating_capacity_m2"
    )
    capacity = max(0.1, float(capacity_model.get(capacity_key) or 0.1))
    ratio = footprint_m2 / capacity
    if ratio <= 0.22:
        return 1.4
    if ratio <= 0.38:
        return 0.8
    if ratio <= 0.55:
        return 0.2
    return -0.8


def _clutter_penalty(
    *,
    footprint_m2: float,
    used_footprint_m2: float,
    capacity_model: dict[str, Any],
) -> float:
    budget = max(0.1, float(capacity_model.get("clutter_budget_m2") or 0.1))
    after_ratio = (used_footprint_m2 + footprint_m2) / budget
    if after_ratio <= 0.72:
        return 0.15
    if after_ratio <= 1.0:
        return 0.8 + after_ratio * 0.5
    return 1.8 + (after_ratio - 1.0) * 4.0


def _circulation_penalty(
    *,
    base_type: str,
    footprint_m2: float,
    capacity_model: dict[str, Any],
) -> float:
    budget = max(0.1, float(capacity_model.get("circulation_budget_m2") or 0.1))
    pressure = footprint_m2 / budget
    multiplier = 1.4 if _placement_mode(base_type) == "floating" else 0.8
    if any(
        token in base_type for token in ("bed", "sofa", "sectional", "dining_table")
    ):
        multiplier += 0.35
    return pressure * multiplier


def _redundancy_penalty(object_type: str, bundle: dict[str, Any]) -> float:
    base_type = _profile_category_for_member(object_type)
    similar_count = sum(
        1
        for obj in _bundle_objects(bundle)
        if _profile_category_for_member(str(obj.get("object_type") or "")) == base_type
    )
    if similar_count <= 1:
        return 0.0
    multi_allowed = any(
        _positive_int(obj.get("max_keep"), default=1) > 1
        or _int_value(obj.get("request_contract_target_count"), default=0) > 1
        for obj in _bundle_objects(bundle)
        if _profile_category_for_member(str(obj.get("object_type") or "")) == base_type
    )
    if multi_allowed:
        return 0.25 * (similar_count - 1)
    return 0.65 * (similar_count - 1)


def _select_quantity(
    *,
    obj: dict[str, Any],
    bundle: dict[str, Any],
    score: dict[str, float],
    used_footprint_m2: float,
    capacity_model: dict[str, Any],
    size_profiles_by_category: dict[str, Any],
    room_scale: str,
    furnishing_mode: str,
) -> int:
    min_keep = max(0, _int_value(obj.get("min_keep"), default=0))
    if (
        obj.get("max_keep") is not None
        and _int_value(
            obj.get("max_keep"),
            default=1,
        )
        <= 0
    ):
        return 0
    max_keep = _positive_int(obj.get("max_keep"), default=max(1, min_keep or 1))
    if _is_core_object(obj, bundle):
        return _core_quantity(
            obj,
            bundle,
            room_scale,
            furnishing_mode,
            size_profiles_by_category,
        )

    total = float(score["total"])
    explicit_guidance = _has_explicit_tier_count_guidance(obj=obj, bundle=bundle)
    surplus_keep = _allow_surplus_optional_keep(
        obj=obj,
        bundle=bundle,
        score=score,
        used_footprint_m2=used_footprint_m2,
        capacity_model=capacity_model,
    )
    base_type = str(obj.get("base_type") or "")
    if (
        not explicit_guidance
        and (furnishing_mode == "minimal" or _is_fast_layout_mode())
        and base_type in {"coffee_table", "ottoman", "recliner"}
        and not surplus_keep
    ):
        return 0
    preserve_level = _effective_preserve_level(obj=obj, bundle=bundle)
    drop_order_bias = _effective_drop_order_bias(obj=obj, bundle=bundle)
    bundle_class = str(bundle.get("bundle_class") or "optional")
    keep_threshold = 3.35 if bundle_class == "decor_light" else 2.2
    if explicit_guidance or min_keep > 0:
        keep_threshold += {
            "highest": -0.8,
            "high": -0.45,
            "medium": 0.0,
            "low": 0.3,
        }.get(preserve_level, 0.0)
        keep_threshold += {
            "drop_last": -0.75,
            "drop_late": -0.35,
            "neutral": 0.0,
            "drop_early": 0.3,
            "drop_first": 0.65,
        }.get(drop_order_bias, 0.0)
    if surplus_keep:
        keep_threshold -= 0.35
    if (
        not explicit_guidance
        and (furnishing_mode == "minimal" or _is_fast_layout_mode())
        and base_type in {"coffee_table", "ottoman", "recliner"}
    ):
        keep_threshold += 2.0
    keep_threshold = max(0.9, keep_threshold)

    forced_quantity = min(max_keep, min_keep) if min_keep > 0 else 0
    if forced_quantity > 0:
        forced_tier = _tier_count_size_tier(obj.get("preferred_size_tier")) or "M"
        forced_footprint = _footprint_for_object(
            obj=obj,
            quantity=forced_quantity,
            size_tier=forced_tier,
            size_profiles_by_category=size_profiles_by_category,
        )
        clutter_budget = max(0.1, float(capacity_model.get("clutter_budget_m2") or 0.1))
        overflow_ratio = (used_footprint_m2 + forced_footprint) / clutter_budget
        if overflow_ratio <= 1.08 or surplus_keep:
            if total >= keep_threshold - 1.2:
                return _expand_quantity_toward_target(
                    obj=obj,
                    bundle=bundle,
                    current_quantity=forced_quantity,
                    max_keep=max_keep,
                    used_footprint_m2=used_footprint_m2,
                    capacity_model=capacity_model,
                    size_profiles_by_category=size_profiles_by_category,
                    room_scale=room_scale,
                    furnishing_mode=furnishing_mode,
                    preserve_level=preserve_level,
                    surplus_keep=surplus_keep,
                )

    if total < keep_threshold and not surplus_keep:
        return 0

    footprint_m2 = _footprint_for_object(
        obj=obj,
        quantity=1,
        size_tier=_tier_count_size_tier(obj.get("preferred_size_tier")) or "M",
        size_profiles_by_category=size_profiles_by_category,
    )
    clutter_budget = max(0.1, float(capacity_model.get("clutter_budget_m2") or 0.1))
    if (
        used_footprint_m2 + footprint_m2 > clutter_budget * 1.04
        and total < keep_threshold + 1.1
        and not surplus_keep
        and min_keep <= 0
    ):
        return 0

    quantity = max(1, min_keep)
    return _expand_quantity_toward_target(
        obj=obj,
        bundle=bundle,
        current_quantity=quantity,
        max_keep=max_keep,
        used_footprint_m2=used_footprint_m2,
        capacity_model=capacity_model,
        size_profiles_by_category=size_profiles_by_category,
        room_scale=room_scale,
        furnishing_mode=furnishing_mode,
        preserve_level=preserve_level,
        surplus_keep=surplus_keep,
    )


def _expand_quantity_toward_target(
    *,
    obj: dict[str, Any],
    bundle: dict[str, Any],
    current_quantity: int,
    max_keep: int,
    used_footprint_m2: float,
    capacity_model: dict[str, Any],
    size_profiles_by_category: dict[str, Any],
    room_scale: str,
    furnishing_mode: str,
    preserve_level: str,
    surplus_keep: bool,
) -> int:
    current_quantity = max(0, min(current_quantity, max_keep))
    target_quantity = _target_quantity_for_object(
        obj=obj,
        bundle=bundle,
        current_quantity=current_quantity,
        max_keep=max_keep,
        room_scale=room_scale,
        furnishing_mode=furnishing_mode,
        preserve_level=preserve_level,
        surplus_keep=surplus_keep,
    )
    if target_quantity <= current_quantity:
        return current_quantity

    size_tier = _tier_count_size_tier(obj.get("preferred_size_tier")) or "M"
    per_item_footprint = _footprint_for_object(
        obj=obj,
        quantity=1,
        size_tier=size_tier,
        size_profiles_by_category=size_profiles_by_category,
    )
    if per_item_footprint <= 0.0:
        return target_quantity

    clutter_budget = max(0.1, float(capacity_model.get("clutter_budget_m2") or 0.1))
    budget_ratio = 0.94
    if preserve_level in {"highest", "high"}:
        budget_ratio += 0.08
    if surplus_keep or furnishing_mode == "generous":
        budget_ratio += 0.08
    if room_scale == "large":
        budget_ratio += 0.06
    elif room_scale == "small" and preserve_level not in {"highest", "high"}:
        budget_ratio -= 0.08

    remaining_after_current = (
        clutter_budget * max(0.75, budget_ratio)
        - used_footprint_m2
        - per_item_footprint * current_quantity
    )
    additional_capacity = int(max(0.0, remaining_after_current) // per_item_footprint)
    if additional_capacity <= 0:
        return current_quantity
    return min(target_quantity, current_quantity + additional_capacity)


def _target_quantity_for_object(
    *,
    obj: dict[str, Any],
    bundle: dict[str, Any],
    current_quantity: int,
    max_keep: int,
    room_scale: str,
    furnishing_mode: str,
    preserve_level: str,
    surplus_keep: bool,
) -> int:
    target_count = _int_value(obj.get("request_contract_target_count"), default=0)
    if target_count <= 0:
        target_count = _int_value(obj.get("target_count"), default=0)
    if target_count <= current_quantity:
        return current_quantity
    if (
        not _has_explicit_tier_count_guidance(obj=obj, bundle=bundle)
        and not surplus_keep
        and preserve_level not in {"highest", "high"}
    ):
        return current_quantity
    if furnishing_mode == "minimal" and preserve_level not in {"highest", "high"}:
        return current_quantity
    if room_scale == "small" and preserve_level not in {"highest", "high"}:
        return current_quantity
    return min(max_keep, max(current_quantity, target_count))


def _core_quantity(
    obj: dict[str, Any],
    bundle: dict[str, Any],
    room_scale: str,
    furnishing_mode: str,
    size_profiles_by_category: dict[str, Any],
) -> int:
    object_type = str(obj.get("object_type") or "")
    base_type = str(obj.get("base_type") or _profile_category_for_member(object_type))
    role = str(obj.get("role") or "")
    min_keep = max(0, _int_value(obj.get("min_keep"), default=0))
    if (
        obj.get("max_keep") is not None
        and _int_value(
            obj.get("max_keep"),
            default=1,
        )
        <= 0
    ):
        return 0
    max_keep = _positive_int(obj.get("max_keep"), default=1)
    if (
        str(obj.get("role") or "") == "dominant_anchor"
        and not _has_explicit_tier_count_guidance(obj=obj, bundle=bundle)
        and (
            furnishing_mode == "minimal"
            or _is_fast_layout_mode()
            or room_scale != "large"
        )
    ):
        preferred_anchor = _preferred_anchor_object_type(
            bundle=bundle,
            size_profiles_by_category=size_profiles_by_category,
        )
        if preferred_anchor is not None and object_type != preferred_anchor:
            return 0
    if (
        role == "workflow_anchor"
        and base_type == "coffee_table"
        and not _has_explicit_tier_count_guidance(obj=obj, bundle=bundle)
        and (furnishing_mode == "minimal" or _is_fast_layout_mode())
    ):
        return 0
    quantity = max(1, min_keep)
    target_quantity = _target_quantity_for_object(
        obj=obj,
        bundle=bundle,
        current_quantity=quantity,
        max_keep=max_keep,
        room_scale=room_scale,
        furnishing_mode=furnishing_mode,
        preserve_level=_effective_preserve_level(obj=obj, bundle=bundle),
        surplus_keep=_explicit_space_surplus_keep(obj=obj, bundle=bundle),
    )
    return min(max_keep, max(quantity, target_quantity))


def _preferred_anchor_object_type(
    *,
    bundle: dict[str, Any],
    size_profiles_by_category: dict[str, Any],
) -> str | None:
    anchors = [
        obj
        for obj in _bundle_objects(bundle)
        if str(obj.get("role") or "") == "dominant_anchor"
    ]
    if len(anchors) <= 1:
        return None

    ranked: list[tuple[float, str]] = []
    for candidate in anchors:
        object_type = str(candidate.get("object_type") or "")
        if not object_type:
            continue
        base_type = str(
            candidate.get("base_type") or _profile_category_for_member(object_type)
        )
        footprint = _footprint_for_object(
            obj=candidate,
            quantity=1,
            size_tier="M",
            size_profiles_by_category=size_profiles_by_category,
        )
        bias = 0.0
        if "sectional" in base_type:
            bias += 0.25
        if base_type == "sofa":
            bias -= 0.05
        ranked.append((footprint + bias, object_type))
    if not ranked:
        return None
    return min(ranked)[1]


def _select_size_tier(
    *,
    obj: dict[str, Any],
    quantity: int,
    score: dict[str, float],
    used_footprint_m2: float,
    capacity_model: dict[str, Any],
    size_profiles_by_category: dict[str, Any],
    room_scale: str,
    furnishing_mode: str,
) -> str | None:
    if quantity <= 0:
        return None

    candidates = _available_tiers_for_object(obj, size_profiles_by_category)
    if not candidates:
        return "M"

    scored = [
        (
            _score_size_tier(
                obj=obj,
                tier=tier,
                quantity=quantity,
                utility_score=float(score["total"]),
                used_footprint_m2=used_footprint_m2,
                capacity_model=capacity_model,
                size_profiles_by_category=size_profiles_by_category,
                room_scale=room_scale,
                furnishing_mode=furnishing_mode,
            ),
            _tier_tiebreak(tier),
            tier,
        )
        for tier in candidates[:3]
    ]
    return max(scored, key=lambda row: (row[0], row[1]))[2]


def _score_size_tier(
    *,
    obj: dict[str, Any],
    tier: str,
    quantity: int,
    utility_score: float,
    used_footprint_m2: float,
    capacity_model: dict[str, Any],
    size_profiles_by_category: dict[str, Any],
    room_scale: str,
    furnishing_mode: str,
) -> float:
    role = str(obj.get("role") or "support")
    object_type = str(obj.get("object_type") or "")
    base_type = str(obj.get("base_type") or _profile_category_for_member(object_type))
    footprint_m2 = _footprint_for_object(
        obj=obj,
        quantity=quantity,
        size_tier=tier,
        size_profiles_by_category=size_profiles_by_category,
    )
    clutter_budget = max(0.1, float(capacity_model.get("clutter_budget_m2") or 0.1))
    after_ratio = (used_footprint_m2 + footprint_m2) / clutter_budget

    preferred = "M" if role in {"dominant_anchor", "workflow_anchor"} else "S"
    hinted_preferred = _tier_count_size_tier(obj.get("preferred_size_tier"))
    if hinted_preferred is not None:
        preferred = hinted_preferred
    if (
        role == "dominant_anchor"
        and room_scale == "large"
        and furnishing_mode == "generous"
        and hinted_preferred is None
    ):
        preferred = "L"
    if base_type == "bed":
        preferred = "M"
        if (
            room_scale == "large"
            and furnishing_mode == "generous"
            and after_ratio < 0.72
            and hinted_preferred is None
        ):
            preferred = "L"

    dominance = 1.0 if tier == preferred else 0.35
    if hinted_preferred is not None:
        if tier == hinted_preferred:
            dominance += 0.45
        else:
            dominance -= 0.25
    if (
        role not in {"dominant_anchor", "workflow_anchor"}
        and tier == "L"
        and hinted_preferred != "L"
    ):
        dominance -= 0.9
    if tier == "L" and after_ratio > 0.82:
        dominance -= 1.2
    if tier == "M" and after_ratio > 0.95:
        dominance -= 0.7

    fit = 1.2 if after_ratio <= 0.72 else 0.3 if after_ratio <= 1.0 else -1.2
    naturalness = {"S": 0.18, "M": 0.12, "L": -0.18}.get(tier, 0.0)
    return utility_score * 0.12 + dominance + fit + naturalness


def _available_tiers_for_object(
    obj: dict[str, Any],
    size_profiles_by_category: dict[str, Any],
) -> list[str]:
    profile = _profile_for_object(obj, size_profiles_by_category)
    rep_dims = profile.get("rep_dims_m") if isinstance(profile, dict) else None
    if not isinstance(rep_dims, dict):
        return ["S", "M", "L"]
    tiers = [tier for tier in ("S", "M", "L") if isinstance(rep_dims.get(tier), dict)]
    return tiers or ["S", "M", "L"]


def _tier_tiebreak(tier: str) -> int:
    return {"S": 2, "M": 1, "L": 0}.get(tier, 0)


def _profile_for_object(
    obj: dict[str, Any],
    size_profiles_by_category: dict[str, Any],
) -> dict[str, Any]:
    object_type = str(obj.get("object_type") or "")
    base_type = str(obj.get("base_type") or _profile_category_for_member(object_type))
    for key in (object_type, base_type, _profile_category_for_member(object_type)):
        profile = size_profiles_by_category.get(key)
        if isinstance(profile, dict):
            return profile
    generic = size_profiles_by_category.get("__generic__")
    return generic if isinstance(generic, dict) else {}


def _footprint_for_object(
    *,
    obj: dict[str, Any],
    quantity: int,
    size_tier: str,
    size_profiles_by_category: dict[str, Any],
) -> float:
    if quantity <= 0:
        return 0.0
    profile = _profile_for_object(obj, size_profiles_by_category)
    rep_dims = profile.get("rep_dims_m") if isinstance(profile, dict) else None
    rep = rep_dims.get(size_tier.upper()) if isinstance(rep_dims, dict) else None
    if isinstance(rep, dict):
        area = float(rep.get("A") or 0.0)
        if area <= 0:
            area = float(rep.get("L") or 0.0) * float(rep.get("W") or 0.0)
        if area > 0:
            return area * quantity

    object_type = str(obj.get("object_type") or "")
    base_type = str(obj.get("base_type") or _profile_category_for_member(object_type))
    fallback = {
        "bed": 3.6,
        "sofa": 2.1,
        "sectional_sofa": 3.4,
        "desk": 1.2,
        "dining_table": 1.8,
        "wardrobe": 1.3,
        "tv_console": 0.8,
        "coffee_table": 0.7,
        "nightstand": 0.25,
        "chair": 0.45,
        "armchair": 0.75,
        "floor_lamp": 0.12,
        "fridge": 0.53,
        "sink": 0.44,
        "stove": 0.49,
        "cooktop": 0.36,
        "kitchen_base_cabinet": 0.72,
        "kitchen_tall_cabinet": 0.45,
        "kitchen_wall_cabinet": 0.36,
        "pantry_cabinet": 0.45,
        "dishwasher": 0.36,
        "kitchen_island": 1.36,
        "bar_stool": 0.18,
        "dining_chair": 0.23,
    }.get(base_type, 0.55)
    tier_mult = {"S": 0.75, "M": 1.0, "L": 1.35}.get(size_tier.upper(), 1.0)
    return fallback * tier_mult * quantity


def _placement_mode(base_type: str) -> str:
    if is_profile_wall_backed_object(base_type):
        return "wall_backed"
    if is_profile_floating_object(base_type):
        return "floating"
    if any(
        token in base_type
        for token in (
            "bed",
            "wardrobe",
            "closet",
            "bookshelf",
            "cabinet",
            "dresser",
            "tv_console",
            "console",
            "desk",
        )
    ):
        return "wall_backed"
    return "floating"


def _space_surplus_ratio(
    *,
    capacity_model: dict[str, Any],
    used_footprint_m2: float,
) -> float:
    room_area = max(0.1, float(capacity_model.get("available_area_m2") or 0.1))
    clutter_budget = max(0.1, float(capacity_model.get("clutter_budget_m2") or 0.1))
    room_headroom = max(0.0, 1.0 - used_footprint_m2 / room_area)
    budget_headroom = max(0.0, 1.0 - used_footprint_m2 / clutter_budget)
    return max(0.0, min(1.0, 0.45 * room_headroom + 0.55 * budget_headroom))


def _surplus_optional_support_bonus(
    *,
    obj: dict[str, Any],
    bundle: dict[str, Any],
    used_footprint_m2: float,
    capacity_model: dict[str, Any],
    style_policy: dict[str, Any],
) -> float:
    role = str(obj.get("role") or "support")
    bundle_class = str(bundle.get("bundle_class") or "optional")
    if role in {"dominant_anchor", "workflow_anchor"}:
        return 0.0
    if bundle_class not in {
        "optional",
        "decor_light",
        "strong_support",
    } and role not in {
        "support",
        "secondary_support",
        "optional",
    }:
        return 0.0

    explicit_surplus_keep = _explicit_space_surplus_keep(obj=obj, bundle=bundle)
    layout_policy = _style_layout_policy(style_policy)
    target_density = str(layout_policy.get("target_density") or "balanced")
    if not explicit_surplus_keep and target_density not in {
        "low_to_balanced",
        "balanced",
        "medium",
        "moderate_high",
    }:
        return 0.0

    surplus = _space_surplus_ratio(
        capacity_model=capacity_model,
        used_footprint_m2=used_footprint_m2,
    )
    threshold = _effective_space_surplus_threshold(
        obj=obj,
        bundle=bundle,
        default=0.38 if explicit_surplus_keep else 0.4,
    )
    if surplus < threshold:
        return 0.0

    object_type = str(obj.get("object_type") or "")
    base_type = str(obj.get("base_type") or _profile_category_for_member(object_type))
    placement_mode = _placement_mode(base_type)
    role_bonus = 0.0
    if explicit_surplus_keep:
        role_bonus += 0.4
    if placement_mode == "wall_backed":
        role_bonus += 0.45
    elif base_type in {"side_table", "floor_lamp", "ottoman", "bench", "armchair"}:
        role_bonus += 0.18
    if any(
        token in base_type
        for token in (
            "bookshelf",
            "cabinet",
            "shelf",
            "console",
            "shoe",
            "rack",
            "storage",
        )
    ):
        role_bonus += 0.35
    if role in {"support", "secondary_support"}:
        role_bonus += 0.18
    if bundle_class == "decor_light":
        role_bonus -= 0.10 if not explicit_surplus_keep else 0.0

    return max(0.0, min(1.1, role_bonus * min(1.0, surplus / 0.75)))


def _allow_surplus_optional_keep(
    *,
    obj: dict[str, Any],
    bundle: dict[str, Any],
    score: dict[str, float],
    used_footprint_m2: float,
    capacity_model: dict[str, Any],
) -> bool:
    object_type = str(obj.get("object_type") or "")
    base_type = str(obj.get("base_type") or _profile_category_for_member(object_type))
    role = str(obj.get("role") or "support")
    bundle_class = str(bundle.get("bundle_class") or "optional")
    explicit_surplus_keep = _explicit_space_surplus_keep(obj=obj, bundle=bundle)
    if role in {"dominant_anchor", "workflow_anchor"}:
        return False
    if bundle_class not in {
        "optional",
        "decor_light",
        "strong_support",
    } and role not in {
        "support",
        "secondary_support",
        "optional",
    }:
        return False
    surplus = _space_surplus_ratio(
        capacity_model=capacity_model, used_footprint_m2=used_footprint_m2
    )
    threshold = _effective_space_surplus_threshold(
        obj=obj,
        bundle=bundle,
        default=0.36 if explicit_surplus_keep else 0.42,
    )
    if surplus < threshold:
        return False
    circulation_penalty = float(score.get("circulation_penalty") or 0.0)
    clutter_penalty = float(score.get("clutter_penalty") or 0.0)
    if explicit_surplus_keep:
        if circulation_penalty > 0.42:
            return False
        if clutter_penalty > 2.1:
            return False
        return float(score.get("total") or 0.0) >= -0.4
    placement_mode = _placement_mode(base_type)
    if placement_mode != "wall_backed":
        if base_type not in {"side_table", "floor_lamp", "bench"}:
            return False
        if circulation_penalty > 0.22:
            return False
    elif circulation_penalty > 0.28:
        return False
    if clutter_penalty > 1.55:
        return False
    return float(score.get("total") or 0.0) >= 0.0


def _is_core_object(obj: dict[str, Any], bundle: dict[str, Any]) -> bool:
    if bool(bundle.get("droppable")):
        return False
    role = str(obj.get("role") or "")
    if bool(obj.get("protected")):
        return True
    semantic_support_role = str(obj.get("semantic_support_role") or "")
    if semantic_support_role == "frontal_support":
        return True
    return bool(obj.get("required")) or role in {"dominant_anchor", "workflow_anchor"}


def _bundle_contains_base_type(bundle: dict[str, Any], base_type: str) -> bool:
    return any(
        str(
            obj.get("base_type")
            or _profile_category_for_member(str(obj.get("object_type") or ""))
        )
        == base_type
        for obj in _bundle_objects(bundle)
    )


def _positive_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _int_value(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _string_list_from_any(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def _decision_type_id(decision: dict[str, Any]) -> str | None:
    obj_type = decision.get("object_type") or decision.get("category")
    if not isinstance(obj_type, str):
        return None
    obj_type = obj_type.strip()
    return obj_type if obj_type else None


def _legacy_priority(role: str, bundle: dict[str, Any]) -> str:
    _ = bundle
    if role == "dominant_anchor":
        return "anchor"
    if role == "workflow_anchor":
        return "primary"
    if role in {"decor_light", "optional"}:
        return "optional"
    return "secondary"


def _decision_priority(
    *,
    obj: dict[str, Any],
    role: str,
    bundle: dict[str, Any],
) -> str:
    priority = _legacy_priority(role, bundle)
    request_intent = str(obj.get("request_contract_intent") or "")
    if (
        _request_contract_min_keep_from_object(obj) > 0
        and request_intent in _HARD_REQUEST_CONTRACT_INTENTS
        and priority in {"optional", "secondary"}
    ):
        return "primary"
    if request_intent in _TARGET_REQUEST_CONTRACT_INTENTS and priority in {
        "optional",
        "secondary",
    }:
        return "primary"
    return priority


def _decision_reason(
    *,
    quantity: int,
    role: str,
    score: dict[str, float],
    obj: dict[str, Any] | None = None,
) -> str:
    if quantity <= 0:
        if isinstance(obj, dict) and obj.get("exclusive_family_winner"):
            winner = str(obj.get("exclusive_family_winner") or "primary seating")
            return f"dropped because {winner} is the selected seating-family anchor"
        return "dropped by utility ranking under clutter and circulation budget"
    if role == "dominant_anchor":
        return "highest functional utility and semantic anchor value"
    if role == "workflow_anchor":
        return "required workflow utility under room-fit budget"
    if float(score.get("space_surplus_bonus") or 0.0) >= 0.35:
        return "kept because the room still has spatial surplus after core layout needs"
    if float(score["completeness"]) >= 1.0:
        return "high completeness utility under clutter budget"
    return "positive utility after room-fit and circulation penalties"


def _copy_exclusive_family_trace(
    source: dict[str, Any],
    target: dict[str, Any],
) -> None:
    family_id = str(source.get("exclusive_family") or "")
    winner = str(source.get("exclusive_family_winner") or "")
    if not family_id or not winner:
        return
    target["exclusive_family"] = family_id
    target["exclusive_family_winner"] = winner


def _circulation_pressure(
    *,
    used_footprint_m2: float,
    capacity_model: dict[str, Any],
) -> float:
    room_area = max(0.1, float(capacity_model.get("available_area_m2") or 0.1))
    clutter_budget = max(0.1, float(capacity_model.get("clutter_budget_m2") or 0.1))
    raw = 0.6 * (used_footprint_m2 / room_area) + 0.4 * (
        used_footprint_m2 / clutter_budget
    )
    return max(0.0, min(1.0, raw))


def _decision_status(
    *,
    decisions: list[dict[str, Any]],
    conflicts: list[str],
    used_footprint_m2: float,
    capacity_model: dict[str, Any],
) -> str:
    if conflicts:
        return "NEEDS_REVIEW"
    core_kept = [
        decision
        for decision in decisions
        if str(decision.get("priority")) in {"anchor", "primary"}
        and int(decision.get("quantity") or 0) > 0
    ]
    if not core_kept:
        return "UNSAT"
    budget = max(0.1, float(capacity_model.get("clutter_budget_m2") or 0.1))
    if used_footprint_m2 > budget * 1.15:
        return "NEEDS_REVIEW"
    return "OK"


def _build_degradation_ready_order(decisions: list[dict[str, Any]]) -> list[str]:
    active = [
        decision
        for decision in decisions
        if isinstance(decision, dict)
        and int(decision.get("quantity") or 0) > 0
        and bool(decision.get("droppable", True))
    ]
    drop_first = [
        str(row.get("object_type") or "")
        for row in sorted(active, key=_degradation_sort_key)
        if str(row.get("priority")) in {"optional", "secondary"}
    ]
    shrink_first = [
        str(row.get("object_type"))
        for row in sorted(active, key=_degradation_sort_key)
        if str(row.get("size_tier")) in {"M", "L"}
    ]
    order: list[str] = []
    if drop_first:
        order.append(f"drop {drop_first[0]}")
    if shrink_first:
        order.append(f"shrink {shrink_first[0]} tier")
    order.extend(
        f"reduce {str(row.get('object_type'))} count"
        for row in sorted(active, key=_degradation_sort_key)
        if int(row.get("quantity") or 0) > 1
    )
    return _uniq(order)


def _degradation_sort_key(
    row: dict[str, Any],
) -> tuple[int, int, int, float, str]:
    priority_rank = {"optional": 0, "secondary": 1, "primary": 2, "anchor": 3}
    priority = str(row.get("priority") or "secondary")
    drop_order_bias = str(row.get("drop_order_bias") or "neutral")
    preserve_level = str(row.get("preserve_level") or "medium")
    score = float(row.get("utility_score") or 0.0)
    return (
        priority_rank.get(priority, 1),
        _drop_order_rank(drop_order_bias),
        _preserve_drop_rank(preserve_level),
        score,
        str(row.get("object_type") or ""),
    )


def _clutter_budget_level(furnishing_mode: str) -> str:
    if furnishing_mode == "minimal":
        return "low"
    if furnishing_mode == "generous":
        return "moderate_high"
    return "moderate"


def _style_clutter_budget_level(
    furnishing_mode: str,
    style_policy: dict[str, Any],
) -> str:
    layout_policy = _style_layout_policy(style_policy)
    clutter_tolerance = _style_level(layout_policy.get("clutter_tolerance"))
    if clutter_tolerance <= 1:
        return "low"
    if clutter_tolerance >= 3:
        return "moderate_high"
    return _clutter_budget_level(furnishing_mode)


def _style_target_density(style_policy: dict[str, Any]) -> str:
    layout_policy = _style_layout_policy(style_policy)
    value = layout_policy.get("target_density")
    return str(value or "balanced")


def _style_object_utility_bias(
    *,
    obj: dict[str, Any],
    bundle: dict[str, Any],
    style_policy: dict[str, Any],
) -> float:
    if not style_policy:
        return 0.0
    role = str(obj.get("role") or "support")
    bundle_class = str(bundle.get("bundle_class") or "optional")
    object_type = str(obj.get("object_type") or "").lower()
    layout_policy = _style_layout_policy(style_policy)
    weights = style_policy.get("policy_weights")
    weights = weights if isinstance(weights, dict) else {}
    bias = 0.0
    if role in {"decor", "decor_light"} or bundle_class == "decor_light":
        bias += float(weights.get("decor_utility_bias") or 0.0)
    if role in {"secondary_support", "support", "optional"}:
        bias += float(weights.get("optional_utility_bias") or 0.0)
    if _style_level(layout_policy.get("decor_tolerance")) <= 1 and any(
        token in object_type for token in ("art", "plant", "vase", "decor", "cushion")
    ):
        bias -= 0.35
    if (
        _style_level(layout_policy.get("clutter_tolerance")) <= 1
        and bundle_class == "optional"
    ):
        bias -= 0.35
    return bias


def _style_layout_policy(style_policy: dict[str, Any]) -> dict[str, Any]:
    layout_policy = style_policy.get("layout_policy")
    return dict(layout_policy) if isinstance(layout_policy, dict) else {}


def _style_level(value: Any) -> int:
    text = str(value or "").lower()
    if "very_high" in text or text == "high":
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


def _inventory_confidence(
    status: str,
    conflicts: list[str],
    decisions: list[dict[str, Any]],
) -> float:
    if status == "UNSAT":
        return 0.35
    active_count = sum(int(row.get("quantity") or 0) for row in decisions)
    base = 0.86 if active_count else 0.45
    base -= min(0.22, len(conflicts) * 0.07)
    if status == "NEEDS_REVIEW":
        base -= 0.12
    return round(max(0.25, min(0.94, base)), 2)


def _uniq(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _ensure_size_profiles(
    *,
    required_types: list[str],
    tenant_id: str | None,
    existing: dict[str, Any] | None,
) -> dict[str, Any]:
    tool_registry = _get_tool_registry()

    if isinstance(existing, dict) and existing:
        profiles = dict(existing)
        _enrich_profiles_for_required_types(
            profiles=profiles,
            required_types=required_types,
        )
        return profiles

    categories = []
    for member in required_types:
        category = _profile_category_for_member(member)
        if category not in categories:
            categories.append(category)

    result = tool_registry["get_size_profiles"](
        categories=categories,
        tenant_id=tenant_id,
    )
    if _tool_output_has_error(result):
        raise RuntimeError(f"get_size_profiles failed: {_tool_error_text(result)}")

    profiles = result.get("size_profiles_by_category")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("Deterministic tier count could not load size profiles.")
    _enrich_profiles_for_required_types(
        profiles=profiles,
        required_types=required_types,
    )
    return profiles


def _build_deterministic_assumptions(
    *,
    context: dict[str, Any],
    furnishing_mode: str,
    available_area_m2: float,
    room_affordances: dict[str, Any],
    style_policy: dict[str, Any],
) -> list[str]:
    assumptions = [
        "Deterministic tier-count policy applied from semantic bundles, room capacity, and utility scores."
    ]
    if furnishing_mode == "minimal":
        assumptions.append(
            "User notes suggest a clutter-light layout, so optional items start conservative."
        )
    elif furnishing_mode == "generous":
        assumptions.append(
            "User notes suggest a fuller furnishing plan, so anchors and primaries start slightly larger."
        )
    assumptions.append(f"Estimated available room area is {available_area_m2:.2f} m2.")
    fill_ratio = float(room_affordances.get("fill_ratio") or 1.0)
    if fill_ratio < 0.92:
        assumptions.append(
            "Room geometry is irregular enough that room-fit and circulation penalties are weighted more carefully."
        )
    if style_policy:
        assumptions.append(
            "Style policy adjusted density, clutter, and decor utility without overriding required functional anchors."
        )
    return assumptions


def _build_deterministic_notes(
    *,
    context: dict[str, Any],
    furnishing_mode: str,
) -> list[str]:
    notes = ["TierCountDirector used deterministic rules instead of an LLM."]
    style = _infer_style(context)
    if style:
        notes.append(f"Utility scoring considered the requested style: {style}.")
    notes.append(f"Initial furnishing mode: {furnishing_mode}.")
    return notes


def _infer_furnishing_mode(context: dict[str, Any]) -> str:
    text_parts: list[str] = []
    description = context.get("description")
    if isinstance(description, str) and description.strip():
        text_parts.append(description.lower())

    special_notes = context.get("special_notes")
    if isinstance(special_notes, str) and special_notes.strip():
        text_parts.append(special_notes.lower())

    user_intent = context.get("user_intent_json")
    if isinstance(user_intent, dict):
        user_input = user_intent.get("user_input")
        if isinstance(user_input, dict):
            for key in ("description", "special_description", "style"):
                value = user_input.get(key)
                if isinstance(value, str) and value.strip():
                    text_parts.append(value.lower())

    text = " ".join(text_parts)
    minimal_score = sum(token in text for token in MINIMAL_NOTE_TOKENS)
    generous_score = sum(token in text for token in GENEROUS_NOTE_TOKENS)

    if minimal_score > generous_score:
        return "minimal"
    if generous_score > minimal_score:
        return "generous"
    return "neutral"


def _estimate_available_area_m2(room_model_json: Any) -> float:
    if not isinstance(room_model_json, dict):
        return 0.0

    room = room_model_json.get("room")
    room_polygon = room.get("polygon_ccw") if isinstance(room, dict) else None
    room_area = _polygon_area_m2(room_polygon if isinstance(room_polygon, list) else [])

    obstacles = room_model_json.get("obstacles")
    obstacle_area = 0.0
    if isinstance(obstacles, list):
        for obstacle in obstacles:
            if not isinstance(obstacle, dict):
                continue
            if obstacle.get("hard", True) is False:
                continue
            polygon = obstacle.get("polygon_ccw")
            if isinstance(polygon, list):
                obstacle_area += _polygon_area_m2(polygon)

    return max(0.0, room_area - obstacle_area)


def _extract_room_affordances(room_model_json: Any) -> dict[str, Any]:
    if not isinstance(room_model_json, dict):
        return {
            "bbox_width_mm": 0,
            "bbox_height_mm": 0,
            "bbox_area_m2": 0.0,
            "room_area_m2": 0.0,
            "fill_ratio": 1.0,
        }

    room = room_model_json.get("room")
    room_polygon = room.get("polygon_ccw") if isinstance(room, dict) else None
    polygon = room_polygon if isinstance(room_polygon, list) else []
    room_area_m2 = _polygon_area_m2(polygon)
    bbox = _polygon_bbox_mm(polygon)
    bbox_width_mm = max(0, int(bbox["max_x"]) - int(bbox["min_x"]))
    bbox_height_mm = max(0, int(bbox["max_y"]) - int(bbox["min_y"]))
    bbox_area_m2 = (bbox_width_mm * bbox_height_mm) / 1_000_000.0
    fill_ratio = room_area_m2 / bbox_area_m2 if bbox_area_m2 > 0 else 1.0
    return {
        "bbox_width_mm": bbox_width_mm,
        "bbox_height_mm": bbox_height_mm,
        "bbox_area_m2": bbox_area_m2,
        "room_area_m2": room_area_m2,
        "fill_ratio": max(0.0, min(1.0, fill_ratio)),
    }


def _polygon_area_m2(points_ccw_mm: list[dict[str, int]]) -> float:
    if len(points_ccw_mm) < 3:
        return 0.0
    total = 0
    for idx, point in enumerate(points_ccw_mm):
        if not isinstance(point, dict):
            continue
        nxt = points_ccw_mm[(idx + 1) % len(points_ccw_mm)]
        if not isinstance(nxt, dict):
            continue
        total += int(point.get("x", 0)) * int(nxt.get("y", 0)) - int(
            nxt.get("x", 0)
        ) * int(point.get("y", 0))
    return abs(total) / 2.0 / 1_000_000.0


def _polygon_bbox_mm(points_ccw_mm: list[dict[str, int]]) -> dict[str, int]:
    xs: list[int] = []
    ys: list[int] = []
    for point in points_ccw_mm:
        if not isinstance(point, dict):
            continue
        xs.append(int(point.get("x", 0)))
        ys.append(int(point.get("y", 0)))
    if not xs or not ys:
        return {"min_x": 0, "min_y": 0, "max_x": 0, "max_y": 0}
    return {
        "min_x": min(xs),
        "min_y": min(ys),
        "max_x": max(xs),
        "max_y": max(ys),
    }


def _classify_room_scale(area_m2: float) -> str:
    if area_m2 < 10.0:
        return "small"
    if area_m2 < 18.0:
        return "medium"
    return "large"


def _profile_category_for_member(member: str) -> str:
    normalized = _norm_key(member)
    canonical = re.sub(r"(?:_\d+)+$", "", normalized)
    profile_canonical = canonical_profile_object_type(canonical)
    if profile_canonical is not None:
        return profile_canonical
    return PROFILE_CATEGORY_ALIASES.get(canonical, canonical)


def _enrich_profiles_for_required_types(
    *,
    profiles: dict[str, Any],
    required_types: list[str],
) -> None:
    for member in required_types:
        if member in profiles:
            continue
        alias = _profile_category_for_member(member)
        if alias != member and isinstance(profiles.get(alias), dict):
            profiles[member] = profiles[alias]
            continue
        profile_size = fallback_profile_size(alias)
        if profile_size is not None:
            profiles.setdefault(alias, profile_size)
            profiles[member] = profile_size


def _select_next_budget_decisions(
    *,
    current_decisions: list[dict[str, Any]],
    budget_out: dict[str, Any],
) -> list[dict[str, Any]] | None:
    recommended_decisions = budget_out.get("recommended_decisions")
    if (
        isinstance(recommended_decisions, list)
        and recommended_decisions != current_decisions
    ):
        return recommended_decisions

    details = budget_out.get("decision_footprint_details")
    if not isinstance(details, list):
        return None

    detail_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in details:
        if not isinstance(row, dict):
            continue
        cluster_id = row.get("cluster_id")
        category = row.get("category")
        if not isinstance(cluster_id, str) or not cluster_id:
            continue
        if not isinstance(category, str) or not category:
            continue
        detail_by_key[(cluster_id, category)] = row

    changed = False
    next_decisions: list[dict[str, Any]] = []
    for decision in current_decisions:
        if not isinstance(decision, dict):
            continue
        cluster_id = decision.get("cluster_id")
        category = decision.get("category")
        key = (
            str(cluster_id or ""),
            str(category or decision.get("object_type") or ""),
        )
        row = detail_by_key.get(key)
        if row is None:
            next_decisions.append(dict(decision))
            continue

        updated = dict(decision)
        recommended_quantity = row.get("recommended_quantity")
        recommended_size_tier = row.get("recommended_size_tier")

        if isinstance(
            recommended_quantity, int
        ) and recommended_quantity != updated.get("quantity"):
            updated["quantity"] = recommended_quantity
            changed = True

        if isinstance(recommended_size_tier, str):
            normalized_tier = recommended_size_tier.upper()
            if normalized_tier != updated.get("size_tier"):
                updated["size_tier"] = normalized_tier
                changed = True

        next_decisions.append(updated)

    return next_decisions if changed else None


def _norm_key(value: str | None) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


# ============================================================
# Prompt helpers
# ============================================================


def _json_block(obj: Any) -> str:
    if obj is None:
        return "null"
    return json.dumps(obj, ensure_ascii=True, indent=2)


def _build_prompt(
    *,
    template: str,
    description: str,
    special_notes: str,
    room_model_json: dict[str, Any],
    user_intent_json: dict[str, Any],
    clusters_json: dict[str, Any],
    size_profiles_json: dict[str, Any] | None,
    layout_failure_report_json: dict[str, Any] | None,
) -> str:
    mapping = {
        "DESCRIPTION": description or "",
        "SPECIAL_NOTES": special_notes or "",
        "ROOM_MODEL_JSON": _json_block(room_model_json),
        "USER_INTENT_JSON": _json_block(user_intent_json),
        "CLUSTERS_JSON": _json_block(clusters_json),
        "SIZE_PROFILES_JSON": _json_block(size_profiles_json),
        "LAYOUT_FAILURE_REPORT_JSON": _json_block(layout_failure_report_json),
    }
    out = template
    for k, v in mapping.items():
        out = out.replace("{" + k + "}", v)
    return out


def _unwrap_payload(payload: dict[str, Any], *, key: str) -> dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get(key), list):
        return payload
    parsed = payload.get("parsed") if isinstance(payload, dict) else None
    if isinstance(parsed, dict) and isinstance(parsed.get(key), list):
        return parsed
    raw = payload.get("raw") if isinstance(payload, dict) else None
    if isinstance(raw, dict) and isinstance(raw.get(key), list):
        return raw
    return payload


def _strip_raw_text(payload: Any) -> Any:
    if isinstance(payload, dict):
        payload = dict(payload)
        payload.pop("raw_text", None)
        for k, v in list(payload.items()):
            payload[k] = _strip_raw_text(v)
        return payload
    if isinstance(payload, list):
        return [_strip_raw_text(x) for x in payload]
    return payload


# ============================================================
# Core tool loop
# ============================================================


def _run_with_tools(
    *,
    messages: list[dict[str, Any]],
    context: dict[str, Any],
    max_steps: int,
    size_profiles_by_category: dict[str, Any] | None,
) -> dict[str, Any]:
    from clients.llm_client import get_llm_client

    client = get_llm_client()
    tool_registry = _get_tool_registry()

    clusters_json = context.get("clusters_json")
    required_types = _extract_member_types(clusters_json)
    members_by_cluster = _extract_members_by_cluster(clusters_json)
    anchors_by_cluster = _extract_anchors_by_cluster(clusters_json)
    droppable_clusters = _extract_droppable_clusters(clusters_json)

    last_draft: dict[str, Any] | None = None
    frozen_cluster_budget_limits_m2: dict[str, float] | None = None
    rescue_mode_used = False

    for step in range(max_steps):
        logger.info("TierCount step %s/%s", step + 1, max_steps)

        response = client.chat_completion(
            messages,
            model_key="primary",
            temperature=0.0,
            max_tokens=None,
            tools=_get_tool_schemas(),
        )

        message = _extract_message(response)
        tool_calls = _extract_tool_calls(message)
        content = getattr(message, "content", "") or ""

        draft = _try_parse_json_object(content)
        if isinstance(draft, dict):
            last_draft = draft

        # ----------------------------------------------------
        # TOOL CALLS
        # ----------------------------------------------------
        if tool_calls:
            logger.info(
                "Tool calls requested: %s",
                [(c.get("function", {}) or {}).get("name") for c in tool_calls],
            )

            prepared_calls: list[dict[str, Any]] = []
            preflight_error_message: str | None = None

            for idx, call in enumerate(tool_calls):
                call_id = call.get("id") or f"tool_{idx}"
                fn = call.get("function", {}) or {}
                name = fn.get("name")
                args_text = fn.get("arguments") or "{}"
                args = _safe_json_loads(args_text)

                if not isinstance(name, str) or name not in tool_registry:
                    prepared_calls.append(
                        {
                            "call_id": call_id,
                            "name": str(name),
                            "args": args,
                            "kind": "unknown_tool",
                        }
                    )
                    continue

                if name == "get_size_profiles":
                    args = _coerce_get_size_profiles_args(
                        args=args,
                        context=context,
                        last_draft=last_draft,
                        required_types=required_types,
                    )

                    if (
                        not isinstance(args.get("categories"), list)
                        or not args["categories"]
                    ):
                        logger.warning(
                            "get_size_profiles has no categories after coercion. last_draft=%s required_types=%s",
                            isinstance(last_draft, dict),
                            required_types,
                        )
                        preflight_error_message = (
                            "Before calling get_size_profiles, provide categories or decisions. "
                            "You may include a JSON draft with decisions first, or call get_size_profiles "
                            f"with categories derived from cluster members: {required_types}"
                        )
                        break

                elif name == "estimate_budget":
                    args = _coerce_estimate_budget_args(
                        args=args,
                        context=context,
                        size_profiles_by_category=size_profiles_by_category,
                        last_draft=last_draft,
                        frozen_cluster_budget_limits_m2=frozen_cluster_budget_limits_m2,
                    )

                    ok, detail = _validate_decisions(
                        args.get("decisions"),
                        required_types,
                        members_by_cluster=members_by_cluster,
                        anchors_by_cluster=anchors_by_cluster,
                        droppable_clusters=droppable_clusters,
                    )
                    if not ok:
                        logger.warning(
                            "estimate_budget preflight failed: detail=%s | args_keys=%s | has_last_draft=%s",
                            detail,
                            sorted(list(args.keys())),
                            isinstance(last_draft, dict),
                        )
                        preflight_error_message = (
                            "Before calling estimate_budget, you MUST provide a valid decisions list. "
                            "Requirement: exactly one decision per member within its own cluster, "
                            "quantity must be an integer >= 0, and each CORE cluster must keep at least one anchor "
                            "(quantity >= 1, using CLUSTERS_JSON.anchors when present). "
                            "DROPPABLE clusters (for example misc/decor/accent/accessory) may be reduced to all quantities = 0. "
                            f"Issue: {detail}. "
                            "You can either include decisions in the tool arguments, or output a JSON draft "
                            "with decisions first so the controller can reuse it."
                        )
                        break

                    if not isinstance(args.get("size_profiles_by_category"), dict):
                        cats = _extract_categories_from_decisions(
                            {"decisions": args.get("decisions")}
                        )
                        if not cats:
                            cats = required_types[:]

                        if cats:
                            logger.info(
                                "Auto-fetching size profiles before estimate_budget for categories=%s",
                                cats,
                            )
                            sp_out = tool_registry["get_size_profiles"](
                                categories=cats,
                                tenant_id=context.get("tenant_id"),
                            )
                            if _tool_output_has_error(sp_out):
                                raise RuntimeError(
                                    f"get_size_profiles failed during preflight: {_tool_error_text(sp_out)}"
                                )
                            sp = sp_out.get("size_profiles_by_category")
                            if isinstance(sp, dict):
                                size_profiles_by_category = sp
                                args["size_profiles_by_category"] = sp

                    if not isinstance(args.get("size_profiles_by_category"), dict):
                        logger.warning(
                            "estimate_budget still missing size_profiles_by_category after auto-fetch"
                        )
                        preflight_error_message = (
                            "Size profiles are still missing, so estimate_budget cannot run. "
                            "Call get_size_profiles first, or provide size_profiles_by_category."
                        )
                        break

                prepared_calls.append(
                    {
                        "call_id": call_id,
                        "name": name,
                        "args": args,
                        "kind": "normal",
                    }
                )

            if preflight_error_message is not None:
                messages.append({"role": "user", "content": preflight_error_message})
                continue

            messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                }
            )

            for item in prepared_calls:
                call_id = item["call_id"]
                name = item["name"]
                args = item["args"]
                kind = item["kind"]

                if kind == "unknown_tool":
                    tool_output = {
                        "error": "unknown_tool",
                        "tool": name,
                        "args": args,
                    }
                else:
                    tool_output = _safe_run_tool(name, args)

                logger.info(
                    "Tool %s output: %s",
                    name,
                    json.dumps(tool_output, ensure_ascii=True),
                )

                if _tool_output_has_error(tool_output):
                    raise RuntimeError(
                        f"{name} failed: {_tool_error_text(tool_output)}"
                    )

                if name == "get_size_profiles":
                    sp = tool_output.get("size_profiles_by_category")
                    if isinstance(sp, dict):
                        size_profiles_by_category = sp

                if (
                    name == "estimate_budget"
                    and frozen_cluster_budget_limits_m2 is None
                    and isinstance(tool_output.get("cluster_budget_limits_m2"), dict)
                    and not bool(tool_output.get("rescue_mode", False))
                ):
                    frozen_cluster_budget_limits_m2 = tool_output[
                        "cluster_budget_limits_m2"
                    ]

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": str(name),
                        "content": json.dumps(tool_output, ensure_ascii=True),
                    }
                )

                if name != "estimate_budget":
                    continue

                active_budget_out = tool_output
                active_decisions = args.get("decisions")
                if not isinstance(active_decisions, list):
                    active_decisions = []

                input_fit = bool(active_budget_out.get("input_decisions_fit", False))
                recommended_fit = bool(
                    active_budget_out.get("recommended_decisions_fit", False)
                )
                repair_exhausted = _tool_reports_repair_exhausted(active_budget_out)
                hard_unsat = _tool_reports_hard_unsat(active_budget_out)

                repair_hints = _preview_budget_repair_hints(active_budget_out)
                violations_after = active_budget_out.get("violations_after_fit") or []
                recommended_decisions = active_budget_out.get("recommended_decisions")

                if (
                    not input_fit
                    and not recommended_fit
                    and not rescue_mode_used
                    and repair_exhausted
                ):
                    rescue_args = dict(args)
                    if (
                        isinstance(recommended_decisions, list)
                        and recommended_decisions
                    ):
                        rescue_args["decisions"] = recommended_decisions
                    rescue_args["rescue_mode"] = True
                    rescue_args.pop("frozen_cluster_budget_limits_m2", None)

                    rescue_budget_out = _safe_run_tool("estimate_budget", rescue_args)
                    if _tool_output_has_error(rescue_budget_out):
                        raise RuntimeError(
                            f"estimate_budget rescue failed: {_tool_error_text(rescue_budget_out)}"
                        )

                    rescue_mode_used = True

                    logger.info(
                        "Rescue estimate_budget output: %s",
                        json.dumps(rescue_budget_out, ensure_ascii=True),
                    )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": f"{call_id}_rescue",
                            "name": "estimate_budget",
                            "content": json.dumps(rescue_budget_out, ensure_ascii=True),
                        }
                    )

                    active_budget_out = rescue_budget_out
                    input_fit = bool(
                        active_budget_out.get("input_decisions_fit", False)
                    )
                    recommended_fit = bool(
                        active_budget_out.get("recommended_decisions_fit", False)
                    )
                    repair_exhausted = _tool_reports_repair_exhausted(active_budget_out)
                    hard_unsat = _tool_reports_hard_unsat(active_budget_out)

                    repair_hints = _preview_budget_repair_hints(active_budget_out)
                    violations_after = (
                        active_budget_out.get("violations_after_fit") or []
                    )
                    recommended_decisions = active_budget_out.get(
                        "recommended_decisions"
                    )

                if input_fit:
                    result = _build_ok_result(
                        decisions=active_decisions,
                        base_draft=last_draft,
                        size_profiles_by_category=size_profiles_by_category,
                        budget_mode="rescue"
                        if bool(active_budget_out.get("rescue_mode", False))
                        else None,
                    )
                    return result

                if recommended_fit and isinstance(recommended_decisions, list):
                    result = _build_ok_result(
                        decisions=recommended_decisions,
                        base_draft=last_draft,
                        size_profiles_by_category=size_profiles_by_category,
                        budget_mode="rescue"
                        if bool(active_budget_out.get("rescue_mode", False))
                        else None,
                    )
                    return result

                if hard_unsat:
                    result = _build_unsat_result(
                        base_draft=last_draft,
                        fallback_decisions=recommended_decisions
                        if isinstance(recommended_decisions, list)
                        else active_decisions,
                        budget_out=active_budget_out,
                        size_profiles_by_category=size_profiles_by_category,
                    )
                    return result

                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Budget check still fails. "
                            "You must continue revising until input_decisions_fit=true. "
                            "Use the recommended_decisions below as the new base draft and make stronger minimal reductions if needed "
                            "(you may change multiple non-anchor items in one revision). "
                            f"Remaining violations: {json.dumps(_preview_budget_violations(violations_after), ensure_ascii=True)}. "
                            f"Recommended decisions: {json.dumps(recommended_decisions, ensure_ascii=True)}. "
                            f"Repair hints: {json.dumps(repair_hints, ensure_ascii=True)}"
                        ),
                    }
                )

            continue

        # ----------------------------------------------------
        # NO TOOL CALLS => EXPECT FINAL JSON
        # ----------------------------------------------------
        if not isinstance(content, str) or not content.strip():
            messages.append(
                {"role": "user", "content": "Return a JSON object only (no markdown)."}
            )
            continue

        logger.info("Assistant content length: %s", len(content))

        try:
            result = _parse_json(content)
            status = str(result.get("status") or "").upper()
        except ValueError:
            _record_llm_retry(
                client=client,
                stage="tier_count_director",
                reason="invalid_json",
            )
            messages.append(
                {
                    "role": "user",
                    "content": "Your output must be valid JSON only (no markdown, no prose).",
                }
            )
            continue

        # IMPORTANT FIX:
        # Only return NEED_INFO if it truly has no valid decision set to continue with.
        if status == "NEED_INFO":
            ok_needinfo, _ = _validate_decisions(
                result.get("decisions"),
                required_types,
                members_by_cluster=members_by_cluster,
                anchors_by_cluster=anchors_by_cluster,
                droppable_clusters=droppable_clusters,
            )
            if not ok_needinfo:
                return result
            result["status"] = "OK"
            status = "OK"

        ok, detail = _validate_decisions(
            result.get("decisions"),
            required_types,
            members_by_cluster=members_by_cluster,
            anchors_by_cluster=anchors_by_cluster,
            droppable_clusters=droppable_clusters,
        )
        if not ok:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Decisions are required before final output. "
                        "You MUST output exactly one decision per member within its own cluster, "
                        "quantity must be an integer >= 0, and each CORE cluster must keep at least one anchor "
                        "(quantity >= 1, using CLUSTERS_JSON.anchors when present). "
                        "DROPPABLE clusters (for example misc/decor/accent/accessory) may be reduced to all quantities = 0. "
                        f"Issue: {detail}"
                    ),
                }
            )
            continue

        assistant_draft_added = False

        def ensure_assistant_draft_in_messages() -> None:
            nonlocal assistant_draft_added
            if assistant_draft_added:
                return
            assistant_content = (
                content
                if isinstance(content, str) and content.strip()
                else json.dumps(result, ensure_ascii=True)
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_content,
                }
            )
            assistant_draft_added = True

        if not isinstance(size_profiles_by_category, dict):
            cats = _extract_categories_from_decisions(result)
            if not cats:
                cats = required_types[:]
            if cats:
                logger.info(
                    "Auto-fetching size profiles before finalization for categories=%s",
                    cats,
                )
                sp_out = tool_registry["get_size_profiles"](
                    categories=cats,
                    tenant_id=context.get("tenant_id"),
                )
                if _tool_output_has_error(sp_out):
                    raise RuntimeError(
                        f"get_size_profiles failed during finalization: {_tool_error_text(sp_out)}"
                    )
                sp = sp_out.get("size_profiles_by_category")
                if isinstance(sp, dict):
                    size_profiles_by_category = sp
                    logger.info(
                        "Auto tool get_size_profiles output: %s",
                        json.dumps(sp_out, ensure_ascii=True),
                    )
                    ensure_assistant_draft_in_messages()
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": f"auto_get_size_profiles_step_{step + 1}",
                            "name": "get_size_profiles",
                            "content": json.dumps(sp_out, ensure_ascii=True),
                        }
                    )

        if not isinstance(size_profiles_by_category, dict):
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Unable to finalize because size profiles are missing. "
                        "Call get_size_profiles first or provide size_profiles_by_category."
                    ),
                }
            )
            continue

        if status == "UNSAT":
            budget_out_unsat = _run_budget_check_for_result(
                result=result,
                context=context,
                size_profiles_by_category=size_profiles_by_category,
                frozen_cluster_budget_limits_m2=frozen_cluster_budget_limits_m2,
                rescue_mode=False,
            )
            if _tool_output_has_error(budget_out_unsat):
                raise RuntimeError(
                    f"estimate_budget failed while verifying UNSAT: {_tool_error_text(budget_out_unsat)}"
                )
            logger.info("Auto-running estimate_budget to verify UNSAT draft.")
            logger.info(
                "UNSAT verification budget check output: %s",
                json.dumps(budget_out_unsat, ensure_ascii=True),
            )
            ensure_assistant_draft_in_messages()
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"auto_estimate_budget_unsat_step_{step + 1}",
                    "name": "estimate_budget",
                    "content": json.dumps(budget_out_unsat, ensure_ascii=True),
                }
            )
            if _tool_reports_hard_unsat(budget_out_unsat):
                return _build_unsat_result(
                    base_draft=result,
                    fallback_decisions=result.get("decisions", []),
                    budget_out=budget_out_unsat,
                    size_profiles_by_category=size_profiles_by_category,
                )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Do not stop with status='UNSAT' yet. "
                        "Continue revising unless the budget tool explicitly confirms true infeasibility."
                    ),
                }
            )
            continue

        budget_out = _run_budget_check_for_result(
            result=result,
            context=context,
            size_profiles_by_category=size_profiles_by_category,
            frozen_cluster_budget_limits_m2=frozen_cluster_budget_limits_m2,
            rescue_mode=False,
        )
        if _tool_output_has_error(budget_out):
            raise RuntimeError(
                f"estimate_budget failed during finalization: {_tool_error_text(budget_out)}"
            )

        logger.info("Auto-running estimate_budget from JSON draft.")
        logger.info(
            "Finalization budget check output: %s",
            json.dumps(budget_out, ensure_ascii=True),
        )
        ensure_assistant_draft_in_messages()
        messages.append(
            {
                "role": "tool",
                "tool_call_id": f"auto_estimate_budget_step_{step + 1}",
                "name": "estimate_budget",
                "content": json.dumps(budget_out, ensure_ascii=True),
            }
        )

        if (
            frozen_cluster_budget_limits_m2 is None
            and isinstance(budget_out.get("cluster_budget_limits_m2"), dict)
            and not bool(budget_out.get("rescue_mode", False))
        ):
            frozen_cluster_budget_limits_m2 = budget_out["cluster_budget_limits_m2"]

        input_fit = bool(budget_out.get("input_decisions_fit", False))
        recommended_fit = bool(budget_out.get("recommended_decisions_fit", False))
        repair_exhausted = _tool_reports_repair_exhausted(budget_out)
        hard_unsat = _tool_reports_hard_unsat(budget_out)

        if input_fit:
            return _build_ok_result(
                decisions=result.get("decisions", []),
                base_draft=result,
                size_profiles_by_category=size_profiles_by_category,
                budget_mode=None,
            )

        recommended_decisions = budget_out.get("recommended_decisions")
        if recommended_fit and isinstance(recommended_decisions, list):
            return _build_ok_result(
                decisions=recommended_decisions,
                base_draft=result,
                size_profiles_by_category=size_profiles_by_category,
                budget_mode=None,
            )

        if repair_exhausted and not rescue_mode_used:
            rescue_budget_out = _run_budget_check_for_result(
                result=result,
                context=context,
                size_profiles_by_category=size_profiles_by_category,
                frozen_cluster_budget_limits_m2=None,
                rescue_mode=True,
            )
            if _tool_output_has_error(rescue_budget_out):
                raise RuntimeError(
                    f"estimate_budget rescue failed during finalization: {_tool_error_text(rescue_budget_out)}"
                )

            rescue_mode_used = True
            logger.info(
                "Finalization rescue budget check output: %s",
                json.dumps(rescue_budget_out, ensure_ascii=True),
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"auto_estimate_budget_rescue_step_{step + 1}",
                    "name": "estimate_budget",
                    "content": json.dumps(rescue_budget_out, ensure_ascii=True),
                }
            )

            rescue_input_fit = bool(rescue_budget_out.get("input_decisions_fit", False))
            rescue_recommended_fit = bool(
                rescue_budget_out.get("recommended_decisions_fit", False)
            )
            rescue_hard_unsat = _tool_reports_hard_unsat(rescue_budget_out)
            rescue_recommended_decisions = rescue_budget_out.get(
                "recommended_decisions"
            )

            if rescue_input_fit:
                return _build_ok_result(
                    decisions=result.get("decisions", []),
                    base_draft=result,
                    size_profiles_by_category=size_profiles_by_category,
                    budget_mode="rescue",
                )

            if rescue_recommended_fit and isinstance(
                rescue_recommended_decisions, list
            ):
                return _build_ok_result(
                    decisions=rescue_recommended_decisions,
                    base_draft=result,
                    size_profiles_by_category=size_profiles_by_category,
                    budget_mode="rescue",
                )

            if rescue_hard_unsat:
                return _build_unsat_result(
                    base_draft=result,
                    fallback_decisions=rescue_recommended_decisions
                    if isinstance(rescue_recommended_decisions, list)
                    else result.get("decisions", []),
                    budget_out=rescue_budget_out,
                    size_profiles_by_category=size_profiles_by_category,
                )

            budget_out = rescue_budget_out
            recommended_decisions = budget_out.get("recommended_decisions")

        if hard_unsat:
            return _build_unsat_result(
                base_draft=result,
                fallback_decisions=recommended_decisions
                if isinstance(recommended_decisions, list)
                else result.get("decisions", []),
                budget_out=budget_out,
                size_profiles_by_category=size_profiles_by_category,
            )

        violations_after = budget_out.get("violations_after_fit") or []
        repair_hints = _preview_budget_repair_hints(budget_out)

        messages.append(
            {
                "role": "user",
                "content": (
                    "Your current decisions still fail budget. "
                    "Do not return UNSAT. Continue revising until input_decisions_fit=true. "
                    "Use the recommended_decisions below as the next base draft and make stronger minimal corrections if needed "
                    "(you may reduce several non-anchor items in one revision). "
                    f"Remaining violations: {json.dumps(_preview_budget_violations(violations_after), ensure_ascii=True)}. "
                    f"Recommended decisions: {json.dumps(recommended_decisions, ensure_ascii=True)}. "
                    f"Repair hints: {json.dumps(repair_hints, ensure_ascii=True)}"
                ),
            }
        )

    # --------------------------------------------------------
    # Timeout fallback
    # --------------------------------------------------------
    if isinstance(last_draft, dict) and isinstance(size_profiles_by_category, dict):
        try:
            budget_out = _run_budget_check_for_result(
                result=last_draft,
                context=context,
                size_profiles_by_category=size_profiles_by_category,
                frozen_cluster_budget_limits_m2=frozen_cluster_budget_limits_m2,
                rescue_mode=False,
            )
            if _tool_output_has_error(budget_out):
                raise RuntimeError(_tool_error_text(budget_out))

            logger.info(
                "Timeout fallback budget check output: %s",
                json.dumps(budget_out, ensure_ascii=True),
            )

            if bool(budget_out.get("input_decisions_fit", False)):
                return _build_ok_result(
                    decisions=last_draft.get("decisions", []),
                    base_draft=last_draft,
                    size_profiles_by_category=size_profiles_by_category,
                    budget_mode=None,
                )

            recommended_decisions = budget_out.get("recommended_decisions")
            if bool(budget_out.get("recommended_decisions_fit", False)) and isinstance(
                recommended_decisions, list
            ):
                return _build_ok_result(
                    decisions=recommended_decisions,
                    base_draft=last_draft,
                    size_profiles_by_category=size_profiles_by_category,
                    budget_mode=None,
                )

            rescue_budget_out = _run_budget_check_for_result(
                result=last_draft,
                context=context,
                size_profiles_by_category=size_profiles_by_category,
                frozen_cluster_budget_limits_m2=None,
                rescue_mode=True,
            )
            if _tool_output_has_error(rescue_budget_out):
                raise RuntimeError(_tool_error_text(rescue_budget_out))

            logger.info(
                "Timeout fallback rescue budget check output: %s",
                json.dumps(rescue_budget_out, ensure_ascii=True),
            )

            if bool(rescue_budget_out.get("input_decisions_fit", False)):
                return _build_ok_result(
                    decisions=last_draft.get("decisions", []),
                    base_draft=last_draft,
                    size_profiles_by_category=size_profiles_by_category,
                    budget_mode="rescue",
                )

            rescue_recommended_decisions = rescue_budget_out.get(
                "recommended_decisions"
            )
            if bool(
                rescue_budget_out.get("recommended_decisions_fit", False)
            ) and isinstance(rescue_recommended_decisions, list):
                return _build_ok_result(
                    decisions=rescue_recommended_decisions,
                    base_draft=last_draft,
                    size_profiles_by_category=size_profiles_by_category,
                    budget_mode="rescue",
                )

            if _tool_reports_hard_unsat(rescue_budget_out):
                return _build_unsat_result(
                    base_draft=last_draft,
                    fallback_decisions=rescue_recommended_decisions
                    if isinstance(rescue_recommended_decisions, list)
                    else last_draft.get("decisions", []),
                    budget_out=rescue_budget_out,
                    size_profiles_by_category=size_profiles_by_category,
                )
        except Exception as exc:
            logger.warning("Timeout fallback failed: %s", exc)

    raise TimeoutError(
        "TierCountDirector exceeded max_steps without reaching a budget-valid final JSON."
    )


# ============================================================
# Tool / args helpers
# ============================================================


def _safe_json_loads(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _safe_run_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    tool_registry = _get_tool_registry()
    try:
        return tool_registry[name](**args)
    except TypeError as exc:
        return {
            "error": "invalid_tool_arguments",
            "tool": name,
            "message": str(exc),
            "args": args,
        }
    except Exception as exc:
        return {"error": "tool_failed", "tool": name, "message": str(exc), "args": args}


def _record_llm_retry(*, client: object, stage: str, reason: str) -> None:
    get_model_name = getattr(client, "get_model_name", None)
    model_name = ""
    if callable(get_model_name):
        model_name = str(get_model_name("primary") or "").strip()
    recorder = getattr(client, "record_retry_event", None)
    if not callable(recorder):
        return
    try:
        recorder(stage=stage, model_name=model_name, reason=reason)
    except Exception:
        logger.debug("Failed to record Gemini retry event.", exc_info=True)


def _tool_output_has_error(tool_output: dict[str, Any]) -> bool:
    return isinstance(tool_output, dict) and isinstance(tool_output.get("error"), str)


def _tool_error_text(tool_output: dict[str, Any]) -> str:
    if not isinstance(tool_output, dict):
        return "unknown tool error"
    return json.dumps(tool_output, ensure_ascii=True)


def _coerce_get_size_profiles_args(
    *,
    args: dict[str, Any],
    context: dict[str, Any],
    last_draft: dict[str, Any] | None,
    required_types: list[str],
) -> dict[str, Any]:
    out = dict(args)

    # get_size_profiles expects furniture/object categories, not cluster tags (e.g. living/storage/misc).
    # Use CLUSTERS_JSON members as the primary source of truth when available.
    if required_types:
        cluster_tags: set[str] = set()
        clusters_json = context.get("clusters_json")
        if isinstance(clusters_json, dict):
            clusters = clusters_json.get("clusters")
            if isinstance(clusters, list):
                for cluster in clusters:
                    if not isinstance(cluster, dict):
                        continue
                    tag = cluster.get("tag")
                    if isinstance(tag, str) and tag.strip():
                        cluster_tags.add(tag.strip().lower())

        provided = out.get("categories")
        provided_cats: list[str] = []
        if isinstance(provided, list):
            for c in provided:
                if isinstance(c, str) and c.strip():
                    provided_cats.append(c.strip())

        deduped: list[str] = []
        seen: set[str] = set()
        for c in provided_cats:
            key = c.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(c)

        required_set = {t.lower() for t in required_types if t}
        filtered: list[str] = []
        dropped: list[str] = []
        for c in deduped:
            ck = c.lower()
            if ck in cluster_tags and ck not in required_set:
                dropped.append(c)
                continue
            filtered.append(c)

        if dropped:
            logger.info(
                "Dropping cluster tags from get_size_profiles.categories: dropped=%s kept=%s",
                dropped,
                filtered,
            )

        if filtered and all(c.lower() in required_set for c in filtered):
            out["categories"] = filtered
        else:
            if filtered:
                logger.info(
                    "Ignoring get_size_profiles.categories not present in CLUSTERS_JSON.members: %s",
                    filtered,
                )
            out["categories"] = required_types[:]
    else:
        cats = out.get("categories")
        if not isinstance(cats, list) or not cats:
            cats = []
            if isinstance(last_draft, dict):
                cats = _extract_categories_from_decisions(last_draft)
            if not cats:
                cats = required_types[:]
            out["categories"] = cats

    if "tenant_id" not in out:
        out["tenant_id"] = context.get("tenant_id")

    return out


def _coerce_estimate_budget_args(
    *,
    args: dict[str, Any],
    context: dict[str, Any],
    size_profiles_by_category: dict[str, Any] | None,
    last_draft: dict[str, Any] | None,
    frozen_cluster_budget_limits_m2: dict[str, float] | None,
) -> dict[str, Any]:
    out = dict(args)

    if not isinstance(out.get("decisions"), list):
        if isinstance(last_draft, dict) and isinstance(
            last_draft.get("decisions"), list
        ):
            out["decisions"] = last_draft["decisions"]

    if "room_model" not in out:
        out["room_model"] = context.get("room_model_json", {})
    if isinstance(out.get("room_model"), str):
        parsed = _safe_json_loads(out["room_model"])
        if parsed:
            out["room_model"] = parsed

    if "size_profiles_by_category" not in out and isinstance(
        size_profiles_by_category, dict
    ):
        out["size_profiles_by_category"] = size_profiles_by_category
    if isinstance(out.get("size_profiles_by_category"), str):
        parsed = _safe_json_loads(out["size_profiles_by_category"])
        if parsed:
            out["size_profiles_by_category"] = parsed

    if "style" not in out:
        style = ""
        user_intent = context.get("user_intent_json") or {}
        if isinstance(user_intent, dict):
            ui = user_intent.get("user_input") or {}
            if isinstance(ui, dict):
                style = str(ui.get("style") or "")
        if not style:
            meta = (context.get("room_model_json") or {}).get("meta") or {}
            if isinstance(meta, dict):
                style = str(meta.get("style") or "")
        out["style"] = style

    if "user_notes" not in out:
        out["user_notes"] = str(context.get("special_notes") or "")

    if "clusters_json" not in out:
        out["clusters_json"] = context.get("clusters_json")
    if isinstance(out.get("clusters_json"), str):
        parsed = _safe_json_loads(out["clusters_json"])
        if parsed:
            out["clusters_json"] = parsed

    rescue_mode = bool(out.get("rescue_mode", False))
    if (
        "frozen_cluster_budget_limits_m2" not in out
        and isinstance(frozen_cluster_budget_limits_m2, dict)
        and not rescue_mode
    ):
        out["frozen_cluster_budget_limits_m2"] = frozen_cluster_budget_limits_m2

    return out


def _run_budget_check_for_result(
    *,
    result: dict[str, Any],
    context: dict[str, Any],
    size_profiles_by_category: dict[str, Any],
    frozen_cluster_budget_limits_m2: dict[str, float] | None,
    rescue_mode: bool,
) -> dict[str, Any]:
    args = {
        "room_model": context.get("room_model_json", {}),
        "decisions": result.get("decisions", []),
        "size_profiles_by_category": size_profiles_by_category,
        "style": _infer_style(context),
        "user_notes": str(context.get("special_notes") or ""),
        "clusters_json": context.get("clusters_json"),
        "rescue_mode": rescue_mode,
    }
    if frozen_cluster_budget_limits_m2 is not None and not rescue_mode:
        args["frozen_cluster_budget_limits_m2"] = frozen_cluster_budget_limits_m2
    return _safe_run_tool("estimate_budget", args)


def _infer_style(context: dict[str, Any]) -> str:
    user_intent = context.get("user_intent_json") or {}
    if isinstance(user_intent, dict):
        ui = user_intent.get("user_input") or {}
        if isinstance(ui, dict):
            style = str(ui.get("style") or "")
            if style:
                return style

    room_model = context.get("room_model_json") or {}
    if isinstance(room_model, dict):
        meta = room_model.get("meta") or {}
        if isinstance(meta, dict):
            style = str(meta.get("style") or "")
            if style:
                return style

    return ""


# ============================================================
# Parsing helpers
# ============================================================


def _extract_message(response: object) -> object:
    choices = getattr(response, "choices", None)
    if isinstance(choices, list) and choices:
        return getattr(choices[0], "message", None)
    raise ValueError("OpenAI response missing message")


def _extract_tool_calls(message: object) -> list[dict[str, Any]]:
    return parse_tool_calls(message)


def _try_parse_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        parsed = json.loads(_extract_json_object(_coerce_json_text(text)))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_json(raw: str) -> dict[str, Any]:
    text = _coerce_json_text(raw)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        text = _extract_json_object(text)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("TierCountDirector returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("TierCountDirector response must be a JSON object")
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


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return text
    return text[start : end + 1]


# ============================================================
# Draft / result helpers
# ============================================================


def _merge_recommended_decisions_into_draft(
    draft: dict[str, Any] | None,
    recommended_decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    base = dict(draft) if isinstance(draft, dict) else {}
    base["status"] = "OK"
    base["decisions"] = recommended_decisions
    if not isinstance(base.get("assumptions"), list):
        base["assumptions"] = []
    if not isinstance(base.get("global_notes"), list):
        base["global_notes"] = []
    return base


def _build_ok_result(
    *,
    decisions: list[dict[str, Any]],
    base_draft: dict[str, Any] | None,
    size_profiles_by_category: dict[str, Any] | None,
    budget_mode: str | None,
) -> dict[str, Any]:
    draft_decisions = _decision_rows(
        base_draft.get("decisions") if isinstance(base_draft, dict) else None
    )
    trial_decisions, trial_restores = _restore_solver_trial_decisions(
        draft_decisions=draft_decisions,
        final_decisions=decisions,
        size_profiles_by_category=size_profiles_by_category,
    )
    result = _merge_recommended_decisions_into_draft(base_draft, trial_decisions)
    result["status"] = "OK"
    result["budget_valid"] = True
    if budget_mode:
        result["budget_mode"] = (
            f"{budget_mode}_with_solver_trials" if trial_restores else budget_mode
        )
    if trial_restores:
        result["budget_trial_restores"] = trial_restores
    if isinstance(size_profiles_by_category, dict):
        _attach_rep_dims(result, size_profiles_by_category)
    _refresh_budget_adjusted_trace(result, draft_decisions=draft_decisions)
    return result


def _build_unsat_result(
    *,
    base_draft: dict[str, Any] | None,
    fallback_decisions: list[dict[str, Any]],
    budget_out: dict[str, Any],
    size_profiles_by_category: dict[str, Any] | None,
) -> dict[str, Any]:
    result = dict(base_draft) if isinstance(base_draft, dict) else {}
    result["status"] = "OK"
    result["degradation_status"] = "DEGRADED_OK"
    result["decisions"] = (
        fallback_decisions if isinstance(fallback_decisions, list) else []
    )
    if not isinstance(result.get("assumptions"), list):
        result["assumptions"] = []
    if not isinstance(result.get("global_notes"), list):
        result["global_notes"] = []

    core_clusters = budget_out.get("core_minimum_infeasible_clusters")
    blocked_clusters = budget_out.get("repair_blocked_clusters")

    notes = list(result.get("global_notes") or [])
    if isinstance(core_clusters, list) and core_clusters:
        notes.append(
            f"Minimum surviving CORE clusters still exceed budget: {core_clusters}"
        )
    elif isinstance(blocked_clusters, list) and blocked_clusters:
        notes.append(f"Repair blocked clusters: {blocked_clusters}")
    else:
        notes.append("Budget remained infeasible after allowed minimal repairs.")

    result["global_notes"] = notes
    result["budget_valid"] = False
    result["budget_reason"] = "hard_unsat_degraded_to_usable_plan"

    if isinstance(size_profiles_by_category, dict):
        _attach_rep_dims(result, size_profiles_by_category)
    _refresh_budget_adjusted_trace(
        result,
        draft_decisions=_decision_rows(
            base_draft.get("decisions") if isinstance(base_draft, dict) else None
        ),
    )
    return result


def _repair_overfull_draft_if_needed(
    draft: dict[str, Any],
    *,
    capacity_model: dict[str, Any],
    size_profiles_by_category: dict[str, Any],
) -> dict[str, Any]:
    decisions = _decision_rows(draft.get("decisions"))
    if not decisions:
        return draft
    target_pressure = _room_fit_repair_pressure_target(capacity_model)
    current_pressure = _decision_circulation_pressure(
        decisions=decisions,
        capacity_model=capacity_model,
    )
    if current_pressure <= target_pressure:
        return draft

    original_decisions = _decision_rows(draft.get("decisions"))
    minimum_count = _minimum_room_fit_selected_count(
        decisions=decisions,
        capacity_model=capacity_model,
    )
    changed = False
    while (
        current_pressure > target_pressure
        and _selected_decision_count(decisions) > minimum_count
    ):
        candidate = _room_fit_reduction_candidate(
            decisions=decisions,
            minimum_count=minimum_count,
        )
        if candidate is None:
            break
        quantity = _decision_quantity(candidate)
        min_keep = _decision_min_keep(candidate)
        if quantity <= min_keep:
            break
        candidate["quantity"] = quantity - 1
        candidate["budget_adjusted"] = True
        candidate["budget_adjustment_reason"] = (
            "reduced by deterministic room-fit backoff after scalar budget repair failed"
        )
        changed = True
        current_pressure = _decision_circulation_pressure(
            decisions=decisions,
            capacity_model=capacity_model,
        )

    if not changed:
        return draft

    repaired = dict(draft)
    repaired["decisions"] = decisions
    repaired["budget_valid"] = current_pressure <= target_pressure
    notes = list(repaired.get("global_notes") or [])
    notes.append(
        (
            "Tier Count applied deterministic room-fit backoff to keep requested "
            "minimums while trimming surplus quantity for solver feasibility."
        )
    )
    repaired["global_notes"] = _uniq([str(note) for note in notes if str(note)])
    _attach_rep_dims(repaired, size_profiles_by_category)
    _refresh_budget_adjusted_trace(
        repaired,
        draft_decisions=original_decisions,
    )
    return repaired


def _room_fit_repair_pressure_target(capacity_model: dict[str, Any]) -> float:
    area = float(
        capacity_model.get("available_area_m2")
        or capacity_model.get("room_area_m2")
        or 0.0
    )
    if area >= 18.0:
        return 0.78
    if area >= 12.0:
        return 0.74
    return 0.7


def _decision_circulation_pressure(
    *,
    decisions: list[dict[str, Any]],
    capacity_model: dict[str, Any],
) -> float:
    return _circulation_pressure(
        used_footprint_m2=sum(_decision_footprint_m2(row) for row in decisions),
        capacity_model=capacity_model,
    )


def _minimum_room_fit_selected_count(
    *,
    decisions: list[dict[str, Any]],
    capacity_model: dict[str, Any],
) -> int:
    selected_count = _selected_decision_count(decisions)
    required_count = sum(_decision_min_keep(row) for row in decisions)
    area = float(
        capacity_model.get("available_area_m2")
        or capacity_model.get("room_area_m2")
        or 0.0
    )
    if area >= 22.0:
        area_floor = 7
    elif area >= 16.0:
        area_floor = 6
    elif area >= 12.0:
        area_floor = 5
    elif area >= 9.0:
        area_floor = 4
    else:
        area_floor = 3
    return min(selected_count, max(required_count, area_floor))


def _selected_decision_count(decisions: list[dict[str, Any]]) -> int:
    return sum(_decision_quantity(row) for row in decisions)


def _room_fit_reduction_candidate(
    *,
    decisions: list[dict[str, Any]],
    minimum_count: int,
) -> dict[str, Any] | None:
    if _selected_decision_count(decisions) <= minimum_count:
        return None
    candidates: list[tuple[tuple[float, ...], dict[str, Any]]] = []
    for row in decisions:
        quantity = _decision_quantity(row)
        min_keep = _decision_min_keep(row)
        if quantity <= min_keep:
            continue
        candidates.append((_room_fit_reduction_rank(row), row))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def _room_fit_reduction_rank(row: dict[str, Any]) -> tuple[float, ...]:
    intent = str(row.get("request_contract_intent") or "").strip()
    request_rank = 2.0 if intent in _HARD_REQUEST_CONTRACT_INTENTS else 0.0
    if intent in _TARGET_REQUEST_CONTRACT_INTENTS or intent == "optional_if_surplus":
        request_rank = 1.0
    priority = str(row.get("priority") or "").strip().lower()
    role = str(row.get("role") or "").strip().lower()
    priority_rank = {
        "optional": 0.0,
        "secondary": 1.0,
        "primary": 2.0,
        "anchor": 4.0,
    }.get(priority, 2.0)
    role_rank = {
        "decor_light": 0.0,
        "optional": 0.0,
        "secondary_support": 1.0,
        "support": 2.0,
        "workflow_anchor": 3.0,
        "dominant_anchor": 4.0,
    }.get(role, 2.0)
    quantity = max(1, _decision_quantity(row))
    per_item_footprint = _decision_footprint_m2(row) / quantity
    return (
        request_rank,
        float(_drop_order_rank(str(row.get("drop_order_bias") or "neutral"))),
        float(_preserve_drop_rank(str(row.get("preserve_level") or "medium"))),
        priority_rank,
        role_rank,
        -per_item_footprint,
        float(_decision_utility_score(row)),
    )


def _decision_min_keep(row: dict[str, Any]) -> int:
    return max(0, _int_value(row.get("min_keep"), default=0))


def _decision_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, dict)]


def _extract_solver_trial_optional_clusters(
    *,
    clusters_json: Any,
    semantic_program: dict[str, Any],
) -> set[str]:
    out = set(_extract_droppable_clusters(clusters_json))
    for cluster_id, row in _semantic_clusters_by_id(semantic_program).items():
        layout_role = str(row.get("layout_role") or "").strip().lower()
        priority = str(row.get("priority") or "").strip().lower()
        if layout_role == "optional" or priority == "optional":
            out.add(cluster_id)
    return out


def _mark_optional_solver_trial_decisions(
    result: dict[str, Any],
    *,
    optional_trial_clusters: set[str],
    size_profiles_by_category: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not optional_trial_clusters:
        return []
    decisions = _decision_rows(result.get("decisions"))
    marked_keys: set[tuple[str, str]] = set()
    markers: list[dict[str, Any]] = []
    for row in decisions:
        key = _decision_key(row)
        if key is None or key[0] not in optional_trial_clusters:
            continue
        if not _budget_solver_trial_candidate(
            row,
            size_profiles_by_category=size_profiles_by_category,
        ):
            continue
        row["solver_trial"] = True
        row["trial_optional"] = True
        row["droppable"] = True
        row["protected"] = False
        row["solver_trial_reason"] = (
            "optional cluster kept for solver-side geometry validation"
        )
        markers.append(
            {
                "cluster_id": key[0],
                "object_type": key[1],
                "quantity": _decision_quantity(row),
                "size_tier": _decision_size_tier(row) or None,
                "reason": row["solver_trial_reason"],
            }
        )
        marked_keys.add(key)
    if not marked_keys:
        return []
    result["decisions"] = decisions
    _sync_optional_trial_flags_into_cluster_decisions(result, marked_keys=marked_keys)
    return markers


def _sync_optional_trial_flags_into_cluster_decisions(
    result: dict[str, Any],
    *,
    marked_keys: set[tuple[str, str]],
) -> None:
    clusters = result.get("cluster_decisions")
    if not isinstance(clusters, list):
        return
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        cluster_id = str(cluster.get("cluster_id") or "").strip()
        for bundle in cluster.get("selected_bundles") or []:
            if not isinstance(bundle, dict):
                continue
            for obj in bundle.get("objects") or []:
                if not isinstance(obj, dict):
                    continue
                object_type = str(obj.get("object_type") or "").strip()
                if (cluster_id, object_type) not in marked_keys:
                    continue
                obj["solver_trial"] = True
                obj["trial_optional"] = True
                obj["protected"] = False
                obj["droppable"] = True
                obj["decision_reason"] = (
                    "kept for solver-side geometry validation as optional cluster"
                )


def _restore_solver_trial_decisions(
    *,
    draft_decisions: list[dict[str, Any]],
    final_decisions: list[dict[str, Any]],
    size_profiles_by_category: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    final_rows = _decision_rows(final_decisions)
    final_by_key, _ = _final_decision_maps(final_rows)
    candidates_by_cluster: dict[str, list[dict[str, Any]]] = {}
    for draft in draft_decisions:
        key = _decision_key(draft)
        if key is None:
            continue
        final = final_by_key.get(key)
        if _decision_quantity(draft) <= 0 or _decision_quantity(final) > 0:
            continue
        if not _budget_solver_trial_candidate(
            draft,
            size_profiles_by_category=size_profiles_by_category,
        ):
            continue
        candidates_by_cluster.setdefault(key[0], []).append(draft)

    restore_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    restores: list[dict[str, Any]] = []
    for cluster_id, candidates in candidates_by_cluster.items():
        if not _budget_trial_cluster_should_restore(
            candidates,
            size_profiles_by_category=size_profiles_by_category,
        ):
            continue
        for draft in candidates:
            key = _decision_key(draft)
            if key is None:
                continue
            restored = dict(draft)
            restored["budget_trial"] = True
            restored["solver_trial"] = True
            restored["trial_optional"] = True
            restored["droppable"] = True
            restored["protected"] = False
            restored["budget_restore_reason"] = (
                "kept as optional solver trial after scalar budget recommendation"
            )
            restored["rationale"] = (
                "kept for solver-side geometry validation after budget recommendation"
            )
            restore_by_key[key] = restored
            restores.append(
                {
                    "cluster_id": cluster_id,
                    "object_type": key[1],
                    "quantity": _decision_quantity(restored),
                    "size_tier": _decision_size_tier(restored) or None,
                    "reason": restored["budget_restore_reason"],
                }
            )

    if not restore_by_key:
        return final_rows, []

    out: list[dict[str, Any]] = []
    emitted: set[tuple[str, str]] = set()
    for row in final_rows:
        key = _decision_key(row)
        if key is not None and key in restore_by_key:
            out.append(dict(restore_by_key[key]))
            emitted.add(key)
        else:
            out.append(dict(row))
            if key is not None:
                emitted.add(key)
    for key, row in restore_by_key.items():
        if key not in emitted:
            out.append(dict(row))
    return out, restores


def _budget_solver_trial_candidate(
    row: dict[str, Any],
    *,
    size_profiles_by_category: dict[str, Any] | None,
) -> bool:
    if _decision_quantity(row) <= 0:
        return False
    if _request_contract_min_keep_from_object(row) > 0:
        return False
    if str(row.get("request_contract_intent") or "") in _HARD_REQUEST_CONTRACT_INTENTS:
        return False
    role = str(row.get("role") or "").strip().lower()
    priority = str(row.get("priority") or "").strip().lower()
    if role not in {
        "dominant_anchor",
        "workflow_anchor",
        "support",
        "secondary_support",
        "optional",
    } and priority not in {"anchor", "primary", "secondary", "optional"}:
        return False
    footprint_m2 = _decision_footprint_m2_for_trial(
        row,
        size_profiles_by_category=size_profiles_by_category,
    )
    if footprint_m2 <= 0.0 or footprint_m2 > 1.15:
        return False
    utility_score = _decision_utility_score(row)
    if role in {"dominant_anchor", "workflow_anchor"} or priority == "anchor":
        return utility_score >= 6.5
    return utility_score >= 3.2


def _budget_trial_cluster_should_restore(
    candidates: list[dict[str, Any]],
    *,
    size_profiles_by_category: dict[str, Any] | None,
) -> bool:
    if not candidates:
        return False
    total_footprint = sum(
        _decision_footprint_m2_for_trial(
            row,
            size_profiles_by_category=size_profiles_by_category,
        )
        for row in candidates
    )
    if total_footprint > 1.6:
        return False
    return any(
        str(row.get("role") or "").strip().lower()
        in {"dominant_anchor", "workflow_anchor"}
        or str(row.get("priority") or "").strip().lower() == "anchor"
        for row in candidates
    )


def _decision_utility_score(row: dict[str, Any]) -> float:
    value = row.get("utility_score")
    if isinstance(value, (int, float)):
        return float(value)
    breakdown = row.get("utility_breakdown")
    if isinstance(breakdown, dict):
        total = breakdown.get("total")
        if isinstance(total, (int, float)):
            return float(total)
    return 0.0


def _decision_footprint_m2_for_trial(
    row: dict[str, Any],
    *,
    size_profiles_by_category: dict[str, Any] | None,
) -> float:
    existing = _decision_footprint_m2(row)
    if existing > 0.0:
        return existing
    if not isinstance(size_profiles_by_category, dict):
        return 0.0
    object_type = str(row.get("object_type") or row.get("category") or "")
    category = _profile_category_for_member(object_type)
    profile = (
        size_profiles_by_category.get(object_type)
        or size_profiles_by_category.get(category)
        or size_profiles_by_category.get("__generic__")
    )
    if not isinstance(profile, dict):
        return 0.0
    tier = _decision_size_tier(row) or str(row.get("preferred_size_tier") or "S")
    rep_dims = profile.get("rep_dims_m")
    rep = rep_dims.get(tier.upper()) if isinstance(rep_dims, dict) else None
    if not isinstance(rep, dict):
        return 0.0
    try:
        length = float(rep.get("L") or 0.0)
        width = float(rep.get("W") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, length * width * max(1, _decision_quantity(row)))


def _decision_key(row: dict[str, Any]) -> tuple[str, str] | None:
    cluster_id = str(row.get("cluster_id") or "").strip()
    object_type = str(row.get("object_type") or row.get("category") or "").strip()
    if not cluster_id or not object_type:
        return None
    return (cluster_id, object_type)


def _final_decision_maps(
    decisions: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, Any]]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    by_type: dict[str, dict[str, Any]] = {}
    for row in decisions:
        key = _decision_key(row)
        if key is not None:
            by_key[key] = row
            by_type.setdefault(key[1], row)
    return by_key, by_type


def _decision_quantity(row: dict[str, Any] | None) -> int:
    if row is None:
        return 0
    return max(0, _int_value(row.get("quantity"), default=0))


def _decision_size_tier(row: dict[str, Any] | None) -> str:
    if row is None:
        return ""
    tier = row.get("size_tier")
    return str(tier or "").strip().upper()


def _budget_adjustments(
    *,
    draft_decisions: list[dict[str, Any]],
    final_decisions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    final_by_key, _ = _final_decision_maps(final_decisions)
    adjustments: list[dict[str, Any]] = []
    for draft in draft_decisions:
        key = _decision_key(draft)
        if key is None:
            continue
        final = final_by_key.get(key)
        if final is None:
            continue
        from_quantity = _decision_quantity(draft)
        to_quantity = _decision_quantity(final)
        from_size_tier = _decision_size_tier(draft) if from_quantity > 0 else ""
        to_size_tier = _decision_size_tier(final) if to_quantity > 0 else ""
        if from_quantity == to_quantity and from_size_tier == to_size_tier:
            continue
        adjustments.append(
            {
                "cluster_id": key[0],
                "object_type": key[1],
                "from_quantity": from_quantity,
                "to_quantity": to_quantity,
                "from_size_tier": from_size_tier or None,
                "to_size_tier": to_size_tier or None,
            }
        )
    return adjustments


def _post_budget_decision_reason(
    *,
    original_reason: str,
    old_quantity: int,
    new_quantity: int,
    old_size_tier: str,
    new_size_tier: str,
) -> str:
    if old_quantity > 0 and new_quantity <= 0:
        return "removed by budget recommendation after deterministic draft scoring"
    if old_quantity != new_quantity:
        return "quantity adjusted by budget recommendation after deterministic draft scoring"
    if old_size_tier and new_size_tier and old_size_tier != new_size_tier:
        return "size tier adjusted by budget recommendation after deterministic draft scoring"
    return original_reason


def _sync_cluster_decisions_with_final_decisions(
    *,
    cluster_decisions: Any,
    final_decisions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(cluster_decisions, list):
        return []
    final_by_key, final_by_type = _final_decision_maps(final_decisions)
    synced_clusters: list[dict[str, Any]] = []

    for cluster in cluster_decisions:
        if not isinstance(cluster, dict):
            continue
        cluster_id = str(cluster.get("cluster_id") or "").strip()
        synced_cluster = dict(cluster)
        selected_bundles: list[dict[str, Any]] = []
        active_objects = 0

        for bundle in cluster.get("selected_bundles") or []:
            if not isinstance(bundle, dict):
                continue
            synced_bundle = dict(bundle)
            objects: list[dict[str, Any]] = []
            for obj in bundle.get("objects") or []:
                if not isinstance(obj, dict):
                    continue
                object_type = str(obj.get("object_type") or "").strip()
                final = final_by_key.get((cluster_id, object_type))
                if final is None and not cluster_id:
                    final = final_by_type.get(object_type)

                synced_obj = dict(obj)
                if final is not None:
                    old_quantity = _decision_quantity(obj)
                    new_quantity = _decision_quantity(final)
                    old_size_tier = _decision_size_tier(obj) if old_quantity > 0 else ""
                    new_size_tier = (
                        _decision_size_tier(final) if new_quantity > 0 else ""
                    )
                    synced_obj["quantity"] = new_quantity
                    synced_obj["size_tier"] = (
                        new_size_tier if new_quantity > 0 and new_size_tier else None
                    )
                    synced_obj["decision_reason"] = _post_budget_decision_reason(
                        original_reason=str(obj.get("decision_reason") or ""),
                        old_quantity=old_quantity,
                        new_quantity=new_quantity,
                        old_size_tier=old_size_tier,
                        new_size_tier=new_size_tier,
                    )
                    if old_quantity != new_quantity or old_size_tier != new_size_tier:
                        synced_obj["budget_adjusted"] = True
                if _decision_quantity(synced_obj) > 0:
                    active_objects += _decision_quantity(synced_obj)
                objects.append(synced_obj)
            synced_bundle["objects"] = objects
            selected_bundles.append(synced_bundle)

        synced_cluster["selected_bundles"] = selected_bundles
        synced_cluster["decision_status"] = (
            "active" if active_objects > 0 else "dropped"
        )
        synced_clusters.append(synced_cluster)

    return synced_clusters


def _decision_footprint_m2(decision: dict[str, Any]) -> float:
    quantity = _decision_quantity(decision)
    if quantity <= 0:
        return 0.0
    rep_dims = decision.get("rep_dims_m")
    if not isinstance(rep_dims, dict):
        return 0.0
    area = rep_dims.get("A")
    if isinstance(area, (int, float)) and area > 0:
        return float(area) * quantity
    length = rep_dims.get("L")
    width = rep_dims.get("W")
    if isinstance(length, (int, float)) and isinstance(width, (int, float)):
        return max(0.0, float(length) * float(width) * quantity)
    return 0.0


def _decision_summary_from_final_decisions(
    *,
    decisions: list[dict[str, Any]],
    capacity_model: dict[str, Any] | None,
) -> dict[str, Any]:
    selected_count = sum(_decision_quantity(row) for row in decisions)
    dropped_types = sorted(
        {
            str(row.get("object_type") or row.get("category") or "")
            for row in decisions
            if _decision_quantity(row) <= 0
            and str(row.get("object_type") or row.get("category") or "").strip()
        }
    )
    used_footprint_m2 = sum(_decision_footprint_m2(row) for row in decisions)
    summary: dict[str, Any] = {
        "selected_object_count": selected_count,
        "dropped_object_types": dropped_types,
        "estimated_footprint_mm2": int(round(used_footprint_m2 * 1_000_000)),
    }
    if isinstance(capacity_model, dict):
        circulation_pressure = _circulation_pressure(
            used_footprint_m2=used_footprint_m2,
            capacity_model=capacity_model,
        )
        summary["estimated_circulation_pressure"] = round(circulation_pressure, 3)
    return summary


def _refresh_budget_adjusted_trace(
    result: dict[str, Any],
    *,
    draft_decisions: list[dict[str, Any]],
) -> None:
    final_decisions = _decision_rows(result.get("decisions"))
    if not final_decisions:
        return

    result["cluster_decisions"] = _sync_cluster_decisions_with_final_decisions(
        cluster_decisions=result.get("cluster_decisions"),
        final_decisions=final_decisions,
    )
    result["decision_summary"] = _decision_summary_from_final_decisions(
        decisions=final_decisions,
        capacity_model=result.get("capacity_model")
        if isinstance(result.get("capacity_model"), dict)
        else None,
    )
    result["degradation_ready_order"] = _build_degradation_ready_order(final_decisions)
    result["confidence"] = _inventory_confidence(
        str(result.get("status") or "OK"),
        list(result.get("conflicts") or []),
        final_decisions,
    )

    global_density_policy = result.get("global_density_policy")
    decision_summary = result.get("decision_summary")
    if isinstance(global_density_policy, dict) and isinstance(decision_summary, dict):
        pressure = decision_summary.get("estimated_circulation_pressure")
        if isinstance(pressure, (int, float)):
            global_density_policy["center_openness_preserved"] = float(pressure) <= 0.72

    adjustments = _budget_adjustments(
        draft_decisions=draft_decisions,
        final_decisions=final_decisions,
    )
    if adjustments:
        result["budget_adjustments"] = adjustments
    else:
        result.pop("budget_adjustments", None)

    degradation_report = _request_contract_degradation_report(
        draft_decisions=draft_decisions,
        final_decisions=final_decisions,
    )
    if degradation_report["dropped"] or degradation_report["reduced"]:
        result["degradation_status"] = "DEGRADED_OK"
        result["request_contract_degradation_report"] = degradation_report
        notes = list(result.get("global_notes") or [])
        notes.append(
            (
                "Tier Count returned a degraded usable plan because budget repair "
                "reduced explicitly requested objects."
            )
        )
        result["global_notes"] = _uniq([str(note) for note in notes if str(note)])
    else:
        result.pop("degradation_status", None)
        result.pop("request_contract_degradation_report", None)


def _request_contract_degradation_report(
    *,
    draft_decisions: list[dict[str, Any]],
    final_decisions: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    final_by_key = {
        key: row for row in final_decisions if (key := _decision_key(row)) is not None
    }
    dropped: list[dict[str, Any]] = []
    reduced: list[dict[str, Any]] = []
    for draft in draft_decisions:
        intent = str(draft.get("request_contract_intent") or "")
        if intent not in _HARD_REQUEST_CONTRACT_INTENTS:
            continue
        min_keep = max(0, _int_value(draft.get("min_keep"), default=0))
        if min_keep <= 0:
            continue
        key = _decision_key(draft)
        if key is None:
            continue
        final = final_by_key.get(key, {})
        from_quantity = max(0, _int_value(draft.get("quantity"), default=0))
        to_quantity = max(0, _int_value(final.get("quantity"), default=0))
        if to_quantity >= min_keep and to_quantity >= from_quantity:
            continue
        row = {
            "cluster_id": key[0],
            "object_type": key[1],
            "from_quantity": from_quantity,
            "to_quantity": to_quantity,
            "min_keep": min_keep,
            "intent": intent,
            "reason": str(draft.get("request_contract_reason") or ""),
            "evidence": str(draft.get("request_contract_evidence") or ""),
        }
        if to_quantity <= 0:
            row["degradation_reason"] = "dropped_after_budget_repair"
            dropped.append(row)
        else:
            row["degradation_reason"] = "quantity_reduced_after_budget_repair"
            reduced.append(row)
    return {
        "dropped": dropped,
        "reduced": reduced,
    }


def _tool_reports_repair_exhausted(budget_out: dict[str, Any]) -> bool:
    if bool(budget_out.get("repair_exhausted", False)):
        return True
    return _budget_repair_exhausted(budget_out)


def _tool_reports_hard_unsat(budget_out: dict[str, Any]) -> bool:
    if bool(budget_out.get("core_minimum_infeasible", False)):
        return True
    if bool(budget_out.get("repair_exhausted", False)) and bool(
        budget_out.get("global_repair_blocked", False)
    ):
        return True
    return False


# ============================================================
# Post-process: attach rep_dims_m for composer
# ============================================================


def _attach_rep_dims(
    result: dict[str, Any], size_profiles_by_category: dict[str, Any]
) -> None:
    decisions = result.get("decisions")
    if not isinstance(decisions, list):
        return

    for d in decisions:
        if not isinstance(d, dict):
            continue
        if "rep_dims_m" in d and isinstance(d.get("rep_dims_m"), dict):
            continue

        tier = d.get("size_tier")
        if not isinstance(tier, str) or not tier:
            continue

        category = d.get("category")
        object_type = d.get("object_type")

        profile = None
        if isinstance(category, str) and category in size_profiles_by_category:
            profile = size_profiles_by_category.get(category)
        elif isinstance(object_type, str) and object_type in size_profiles_by_category:
            profile = size_profiles_by_category.get(object_type)
        elif isinstance(size_profiles_by_category.get("__generic__"), dict):
            profile = size_profiles_by_category.get("__generic__")

        if not isinstance(profile, dict):
            continue

        rep = (profile.get("rep_dims_m") or {}).get(tier.upper())
        if isinstance(rep, dict):
            d["rep_dims_m"] = rep


def _extract_categories_from_decisions(result: dict[str, Any]) -> list[str]:
    decisions = result.get("decisions")
    if not isinstance(decisions, list):
        return []
    cats: list[str] = []
    for d in decisions:
        if not isinstance(d, dict):
            continue
        for key in ("category", "object_type"):
            v = d.get(key)
            if isinstance(v, str) and v and v not in cats:
                cats.append(v)
    return cats


def _extract_member_types(clusters_json: Any) -> list[str]:
    if not isinstance(clusters_json, dict):
        return []
    clusters = clusters_json.get("clusters")
    if not isinstance(clusters, list):
        return []
    out: list[str] = []
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        members = cluster.get("members")
        if not isinstance(members, list):
            continue
        for m in members:
            if isinstance(m, str) and m and m not in out:
                out.append(m)
    return out


def _extract_members_by_cluster(clusters_json: Any) -> dict[str, set[str]]:
    if not isinstance(clusters_json, dict):
        return {}
    clusters = clusters_json.get("clusters")
    if not isinstance(clusters, list):
        return {}

    out: dict[str, set[str]] = {}
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        cluster_id = cluster.get("cluster_id")
        if not isinstance(cluster_id, str) or not cluster_id:
            continue
        members = cluster.get("members")
        if not isinstance(members, list):
            continue
        out[cluster_id] = {m for m in members if isinstance(m, str) and m}
    return out


def _extract_anchors_by_cluster(clusters_json: Any) -> dict[str, set[str]]:
    if not isinstance(clusters_json, dict):
        return {}
    clusters = clusters_json.get("clusters")
    if not isinstance(clusters, list):
        return {}

    out: dict[str, set[str]] = {}
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        cluster_id = cluster.get("cluster_id")
        if not isinstance(cluster_id, str) or not cluster_id:
            continue
        members = cluster.get("members")
        member_set = (
            {member for member in members if isinstance(member, str) and member}
            if isinstance(members, list)
            else set()
        )
        anchors = cluster.get("anchors")
        if not isinstance(anchors, list):
            out.setdefault(cluster_id, set())
            continue
        out[cluster_id] = {
            anchor
            for anchor in anchors
            if isinstance(anchor, str) and anchor and anchor in member_set
        }
    return out


def _extract_droppable_clusters(clusters_json: Any) -> set[str]:
    if not isinstance(clusters_json, dict):
        return set()

    clusters = clusters_json.get("clusters")
    if not isinstance(clusters, list):
        return set()

    out: set[str] = set()
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue

        cluster_id = cluster.get("cluster_id")
        if not isinstance(cluster_id, str) or not cluster_id:
            continue

        tag = str(cluster.get("tag") or "").strip().lower()
        rules = cluster.get("cluster_rules") or {}
        allow_empty = (
            isinstance(rules, dict) and rules.get("allow_empty_cluster") is True
        )

        if tag in {"misc", "decor", "accent", "accessory"} or allow_empty:
            out.add(cluster_id)

    return out


def _validate_decisions(
    decisions: Any,
    required_types: list[str],
    *,
    members_by_cluster: dict[str, set[str]] | None = None,
    anchors_by_cluster: dict[str, set[str]] | None = None,
    droppable_clusters: set[str] | None = None,
) -> tuple[bool, str]:
    if not isinstance(decisions, list) or not decisions:
        return False, "decisions list is missing or empty"

    types: list[str] = []
    bad_qty: list[str] = []
    bad_cluster_id: list[str] = []
    bad_cluster_assignment: list[str] = []
    unexpected_types: list[str] = []
    decisions_by_cluster: dict[str, dict[str, int]] = {}
    decision_counts_by_cluster: dict[str, dict[str, int]] = {}

    for d in decisions:
        if not isinstance(d, dict):
            continue
        t = d.get("object_type") or d.get("category")
        if not isinstance(t, str) or not t:
            continue

        types.append(t)

        if required_types and t not in required_types:
            unexpected_types.append(t)

        qty = d.get("quantity")
        if not isinstance(qty, int) or qty < 0:
            bad_qty.append(t)

        cluster_id = d.get("cluster_id")
        if not isinstance(cluster_id, str) or not cluster_id:
            bad_cluster_id.append(t)
            continue

        if isinstance(members_by_cluster, dict) and members_by_cluster:
            members = members_by_cluster.get(cluster_id)
            if not isinstance(members, set):
                bad_cluster_assignment.append(f"{t}->{cluster_id}")
            elif t not in members:
                bad_cluster_assignment.append(f"{t}->{cluster_id}")

        decisions_by_cluster.setdefault(cluster_id, {})[t] = (
            qty if isinstance(qty, int) else 0
        )
        cluster_counts = decision_counts_by_cluster.setdefault(cluster_id, {})
        cluster_counts[t] = cluster_counts.get(t, 0) + 1

    if bad_qty:
        return False, f"quantity must be an integer >= 0 for: {sorted(set(bad_qty))}"

    if bad_cluster_id:
        return (
            False,
            f"cluster_id must be a non-empty string for: {sorted(set(bad_cluster_id))}",
        )

    if bad_cluster_assignment:
        return (
            False,
            "each decision must reference a valid cluster_id and a member in that cluster. "
            f"invalid assignments: {sorted(set(bad_cluster_assignment))}",
        )

    if unexpected_types:
        return False, f"unexpected decisions for types: {sorted(set(unexpected_types))}"

    if required_types:
        missing = [t for t in required_types if t not in types]
        if missing:
            return False, f"missing decisions for types: {missing}"

    duplicate_cluster_members = sorted(
        f"{cluster_id}.{object_type}"
        for cluster_id, counts in decision_counts_by_cluster.items()
        for object_type, count in counts.items()
        if count > 1
    )
    if duplicate_cluster_members:
        return (
            False,
            "multiple decisions found for cluster-scoped members: "
            f"{duplicate_cluster_members}",
        )

    droppable_clusters = droppable_clusters or set()

    if isinstance(members_by_cluster, dict) and members_by_cluster:
        for cluster_id in sorted(members_by_cluster.keys()):
            members = members_by_cluster.get(cluster_id) or set()
            anchors = (
                anchors_by_cluster.get(cluster_id, set())
                if isinstance(anchors_by_cluster, dict)
                else set()
            )
            cluster_decisions = decisions_by_cluster.get(cluster_id, {})
            missing_cluster_members = sorted(
                member for member in members if member not in cluster_decisions
            )
            if missing_cluster_members:
                return (
                    False,
                    "each cluster member must have exactly one decision in its own "
                    f"cluster. missing members for {cluster_id}: "
                    f"{missing_cluster_members}",
                )

            if cluster_id in droppable_clusters:
                continue

            if anchors:
                kept = any(cluster_decisions.get(a, 0) >= 1 for a in anchors)
                if not kept:
                    return (
                        False,
                        f"CORE cluster {cluster_id} must keep at least one anchor with quantity >= 1: {sorted(anchors)}",
                    )
            else:
                kept = any(
                    cluster_decisions.get(m, 0) >= 1
                    for m in members
                    if isinstance(m, str)
                )
                if not kept and members:
                    return (
                        False,
                        f"CORE cluster {cluster_id} must keep at least one member with quantity >= 1",
                    )

    return True, "ok"


def _preview_budget_repair_hints(
    budget_out: dict[str, Any],
    limit: int = 8,
) -> list[dict[str, Any]]:
    details = budget_out.get("decision_footprint_details")
    if not isinstance(details, list):
        return []

    out: list[dict[str, Any]] = []
    for row in details:
        if not isinstance(row, dict):
            continue

        orig_qty = row.get("quantity")
        rec_qty = row.get("recommended_quantity")
        orig_tier = row.get("size_tier")
        rec_tier = row.get("recommended_size_tier")

        changed_qty = (
            isinstance(orig_qty, int)
            and isinstance(rec_qty, int)
            and orig_qty != rec_qty
        )
        changed_tier = (
            isinstance(orig_tier, str)
            and isinstance(rec_tier, str)
            and orig_tier != rec_tier
        )

        if not changed_qty and not changed_tier:
            continue

        out.append(
            {
                "cluster_id": row.get("cluster_id"),
                "category": row.get("category"),
                "priority": row.get("priority"),
                "quantity": orig_qty,
                "recommended_quantity": rec_qty,
                "size_tier": orig_tier,
                "recommended_size_tier": rec_tier,
            }
        )

        if len(out) >= limit:
            break

    return out


def _budget_repair_exhausted(budget_out: dict[str, Any]) -> bool:
    if bool(budget_out.get("recommended_decisions_fit", False)):
        return False

    details = budget_out.get("decision_footprint_details")
    if not isinstance(details, list):
        return False

    for row in details:
        if not isinstance(row, dict):
            continue
        if row.get("quantity") != row.get("recommended_quantity"):
            return False
        if row.get("size_tier") != row.get("recommended_size_tier"):
            return False

    return True


def _preview_budget_violations(
    violations: list[dict[str, Any]], limit: int = 5
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for v in violations[:limit]:
        if not isinstance(v, dict):
            continue
        out.append(
            {
                "cluster_id": v.get("cluster_id"),
                "limit_m2": v.get("limit_m2"),
                "footprint_m2": v.get("footprint_m2"),
                "over_by_m2": v.get("over_by_m2"),
            }
        )
    return out
