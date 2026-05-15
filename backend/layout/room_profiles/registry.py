from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from layout.room_profiles.base import RoomProfile, normalize_profile_token
from layout.room_profiles.kitchen import KITCHEN_PROFILE
from layout.room_profiles.legacy import (
    BEDROOM_LEGACY_PROFILE,
    LIVING_ROOM_LEGACY_PROFILE,
)

ROOM_PROFILES: tuple[RoomProfile, ...] = (
    KITCHEN_PROFILE,
    BEDROOM_LEGACY_PROFILE,
    LIVING_ROOM_LEGACY_PROFILE,
)
_PROFILE_READINESS_MIN_COVERAGE = 1.0
_PROFILE_READINESS_MIN_TRAIT_COVERAGE = 1.0
_ROOM_PROFILE_RULE_MODE_ENV = "TKNT_ROOM_PROFILE_RULE_MODE"
_ROOM_PROFILE_FIRST_ROOMS_ENV = "TKNT_ROOM_PROFILE_FIRST_ROOMS"
_DEFAULT_ROOM_PROFILE_RULE_MODE = "unified"
_ROOM_PROFILE_RULE_MODES = frozenset({"legacy", "shadow", "canary", "unified"})
_ROOM_PROFILE_TRAITS_MODE_ENV = "TKNT_ROOM_PROFILE_TRAITS_MODE"
_ROOM_PROFILE_TRAIT_ROOMS_ENV = "TKNT_ROOM_PROFILE_TRAIT_ROOMS"
_DEFAULT_ROOM_PROFILE_TRAITS_MODE = "unified"
_ROOM_PROFILE_TRAITS_MODES = frozenset({"legacy", "shadow", "canary", "unified"})


@dataclass(frozen=True)
class RoomRuleSelection:
    room_type: str
    requested_room_type: str
    profile_id: str | None
    mode: str
    selected_source: str
    rule: dict[str, Any]
    profile_first_requested: bool
    profile_first_active: bool
    fallback_used: bool
    fallback_reason: str | None
    dry_run_diff: dict[str, Any]

    def trace(self) -> dict[str, Any]:
        return {
            "engine": "room_profile_registry",
            "room_type": self.room_type,
            "requested_room_type": self.requested_room_type,
            "profile_id": self.profile_id,
            "mode": self.mode,
            "selected_source": self.selected_source,
            "profile_first_requested": self.profile_first_requested,
            "profile_first_active": self.profile_first_active,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "dry_run_diff": deepcopy(self.dry_run_diff),
        }


def resolve_room_profile(room_type: object) -> RoomProfile | None:
    for profile in ROOM_PROFILES:
        if profile.matches_room_type(room_type):
            return profile
    return None


def semantic_room_rule_for(room_type: object) -> dict[str, Any] | None:
    profile = resolve_room_profile(room_type)
    if profile is None:
        return None
    return profile.semantic_room_rule(profile.canonical_room_type)


def select_profile_room_rule(
    *,
    room_type: object,
    profile_rule: Mapping[str, Any] | None,
    legacy_rule: Mapping[str, Any] | None,
) -> RoomRuleSelection:
    requested_room_type = str(room_type or "").strip()
    profile = resolve_room_profile(room_type)
    canonical_room_type = (
        profile.canonical_room_type if profile is not None else requested_room_type
    )
    mode = _room_profile_rule_mode()
    profile_rule_clean = _clone_rule(profile_rule)
    legacy_rule_clean = _clone_rule(legacy_rule)
    dry_run_diff = _room_rule_dry_run_diff(
        profile_rule=profile_rule_clean,
        legacy_rule=legacy_rule_clean,
    )

    profile_first_requested = _profile_rule_requested(
        room_type=requested_room_type,
        profile=profile,
        mode=mode,
    )
    selected_source = "legacy"
    fallback_used = False
    fallback_reason: str | None = None
    selected_rule = legacy_rule_clean

    if profile_rule_clean is None and legacy_rule_clean is None:
        selected_source = "fallback_empty"
        selected_rule = _fallback_empty_rule(canonical_room_type)
        fallback_used = True
        fallback_reason = "no_profile_or_legacy_rule"
    elif profile_rule_clean is None:
        selected_source = "legacy"
        selected_rule = legacy_rule_clean
        fallback_used = profile_first_requested
        fallback_reason = "profile_rule_missing" if fallback_used else None
    elif legacy_rule_clean is None:
        selected_source = "profile"
        selected_rule = profile_rule_clean
    elif _profile_rule_blocked_by_regression_gate(
        profile=profile,
        mode=mode,
        dry_run_diff=dry_run_diff,
    ):
        selected_source = "legacy"
        selected_rule = legacy_rule_clean
        fallback_used = True
        fallback_reason = "profile_dry_run_diff_not_equivalent"
    elif profile_first_requested:
        selected_source = "profile"
        selected_rule = profile_rule_clean
    else:
        selected_source = "legacy"
        selected_rule = legacy_rule_clean

    if selected_rule is None:
        selected_rule = _fallback_empty_rule(canonical_room_type)
        fallback_used = True
        fallback_reason = fallback_reason or "selected_rule_missing"
        selected_source = "fallback_empty"

    return RoomRuleSelection(
        room_type=canonical_room_type,
        requested_room_type=requested_room_type,
        profile_id=profile.profile_id if profile is not None else None,
        mode=mode,
        selected_source=selected_source,
        rule=selected_rule,
        profile_first_requested=profile_first_requested,
        profile_first_active=selected_source == "profile",
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        dry_run_diff=dry_run_diff,
    )


def apply_profile_capacity_model(
    capacity_model: dict[str, Any],
    *,
    room_type: object,
) -> dict[str, Any]:
    profile = resolve_room_profile(room_type)
    if profile is None:
        return capacity_model
    return profile.apply_capacity_policy(capacity_model, str(room_type or ""))


def semantic_placements_for_members(
    *,
    room_type: object,
    cluster_id: str,
    members: Sequence[str],
    anchors: Sequence[str],
) -> list[dict[str, Any]]:
    profile = resolve_room_profile(room_type)
    if profile is not None:
        rows = profile.semantic_placements_for_members(cluster_id, members, anchors)
        if rows:
            return rows
    for candidate in _profiles_for_members(members):
        rows = candidate.semantic_placements_for_members(cluster_id, members, anchors)
        if rows:
            return rows
    return []


def all_profile_object_aliases() -> dict[str, tuple[str, ...]]:
    out: dict[str, tuple[str, ...]] = {}
    for profile in ROOM_PROFILES:
        for canonical, aliases in profile.object_aliases.items():
            out[normalize_profile_token(canonical)] = _request_alias_variants(
                canonical,
                aliases,
            )
    return out


def all_profile_scoring_aliases() -> dict[str, tuple[str, ...]]:
    out: dict[str, tuple[str, ...]] = {}
    for profile in ROOM_PROFILES:
        aliases = profile.scoring_aliases or profile.object_aliases
        for canonical, values in aliases.items():
            out[normalize_profile_token(canonical)] = _scoring_alias_variants(
                canonical,
                values,
            )
    return out


def all_profile_non_functional_contract_types() -> frozenset[str]:
    values: set[str] = set()
    for profile in ROOM_PROFILES:
        values.update(profile.non_functional_contract_types)
    return frozenset(values)


def all_profile_non_functional_layout_specs() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for profile in ROOM_PROFILES:
        for object_type, spec in profile.non_functional_layout_specs.items():
            if isinstance(spec, Mapping):
                out[normalize_profile_token(object_type)] = deepcopy(dict(spec))
    return out


def canonical_profile_object_type(object_type: object) -> str | None:
    for profile in ROOM_PROFILES:
        canonical = profile.canonical_object_type(object_type)
        if canonical is not None:
            return canonical
    return None


def profile_id_for_objects(
    object_types: Sequence[str],
    *,
    room_type: object | None = None,
) -> str | None:
    profile = _profile_for_objects(object_types, room_type=room_type)
    if profile is None:
        return None
    return profile.profile_id


def profile_room_type_for_objects(
    object_types: Sequence[str],
    *,
    room_type: object | None = None,
) -> str | None:
    profile = _profile_for_objects(object_types, room_type=room_type)
    if profile is None:
        return None
    return profile.canonical_room_type


def is_profile_trait_object(object_type: object) -> bool:
    return any(
        _profile_has_effective_trait_object(profile, object_type)
        for profile in _layout_enabled_profiles()
    )


def is_profile_workflow_object(object_type: object) -> bool:
    canonical = canonical_profile_object_type(object_type)
    if canonical is None:
        return False
    return any(
        canonical in profile.workflow_objects for profile in _layout_enabled_profiles()
    )


def is_profile_wall_backed_object(object_type: object) -> bool:
    canonical = canonical_profile_object_type(object_type)
    if canonical is None:
        return False
    return any(
        canonical in profile.wall_backed_objects
        for profile in _layout_enabled_profiles()
    )


def is_profile_floating_object(object_type: object) -> bool:
    canonical = canonical_profile_object_type(object_type)
    if canonical is None:
        return False
    return any(
        canonical in profile.floating_objects for profile in _layout_enabled_profiles()
    )


def is_profile_mounted_object(object_type: object) -> bool:
    canonical = canonical_profile_object_type(object_type)
    if canonical is None:
        return False
    return any(
        canonical in profile.mounted_objects for profile in _layout_enabled_profiles()
    )


def is_profile_storage_object(object_type: object) -> bool:
    canonical = canonical_profile_object_type(object_type)
    if canonical is None:
        return False
    return any(
        canonical in profile.storage_objects for profile in _layout_enabled_profiles()
    )


def fallback_profile_size(object_type: object) -> dict[str, Any] | None:
    for profile in _layout_enabled_profiles():
        profile_size = profile.fallback_size_profile(object_type)
        if profile_size is not None:
            return profile_size
    return None


def profile_object_traits(
    *,
    room_type: object,
    object_type: object,
    include_shadow: bool = True,
) -> tuple[str, ...]:
    profile = resolve_room_profile(room_type)
    if profile is None:
        return ()
    return profile.object_traits(
        object_type,
        include_shadow=include_shadow or _profile_layout_traits_enabled(profile),
    )


def profile_cluster_tag_for_objects(
    object_types: Sequence[str],
    *,
    room_type: object | None = None,
) -> str | None:
    profile = _profile_for_objects(object_types, room_type=room_type)
    if profile is None or not _profile_layout_semantics_enabled(profile):
        return None
    return profile.cluster_tag


def profile_semantic_role_for_objects(
    *,
    cluster_id: object,
    object_types: Sequence[str],
    priority: object,
    room_type: object | None = None,
) -> str | None:
    profile = _profile_for_objects(object_types, room_type=room_type)
    if profile is None or not _profile_layout_semantics_enabled(profile):
        return None

    canonical_types = _profile_canonical_object_types(profile, object_types)
    explicit_role = _profile_semantic_role_override(profile, canonical_types)
    if explicit_role is not None:
        return explicit_role

    traits = _profile_object_trait_set(profile, object_types)
    if "workflow" in traits:
        return f"{profile.canonical_room_type}_workflow_zone"
    if "storage" in traits and "workflow" not in traits:
        return f"{profile.canonical_room_type}_storage_zone"
    if "floating" in traits:
        return f"{profile.canonical_room_type}_floating_support_zone"

    cluster_text = str(cluster_id or "").strip()
    if normalize_profile_token(priority) == "core" and cluster_text:
        return f"{profile.canonical_room_type}_core_zone"
    return None


def profile_layout_role_for_objects(
    *,
    cluster_id: object,
    object_types: Sequence[str],
    room_type: object | None,
) -> str | None:
    profile = _profile_for_objects(object_types, room_type=room_type)
    if profile is None or not _profile_layout_semantics_enabled(profile):
        return None

    canonical_types = set(_profile_canonical_object_types(profile, object_types))
    cluster_key = normalize_profile_token(cluster_id)
    if (
        profile.canonical_room_type == "kitchen"
        and (
            "dining" in cluster_key
            or canonical_types & {"dining_table", "dining_chair"}
        )
    ):
        return "support"
    return None


def profile_zone_claims_for_objects(
    *,
    cluster_id: object,
    object_types: Sequence[str],
    room_type: object | None,
    affordance_summary: Mapping[str, Any],
) -> dict[str, Any] | None:
    _ = cluster_id
    profile = _profile_for_objects(object_types, room_type=room_type)
    if profile is None or not _profile_layout_semantics_enabled(profile):
        return None

    canonical_types = set(_profile_canonical_object_types(profile, object_types))
    traits = _profile_object_trait_set(profile, object_types)
    if (
        profile.canonical_room_type == "kitchen"
        and "floating" in traits
        and "wall_backed" not in traits
        and canonical_types & {"dining_table", "dining_chair"}
    ):
        return {
            "preferred_regions": _profile_regions(
                _profile_regions(affordance_summary.get("daylight_regions"))
                + _profile_regions(affordance_summary.get("floating_zone_candidates"))
            ),
            "avoid_regions": _profile_regions(
                _profile_regions(affordance_summary.get("entry_landing_zones"))
                + _profile_regions(
                    affordance_summary.get("primary_circulation_corridors")
                )
            ),
            "wall_affinity": "medium",
            "daylight_affinity": "high",
            "privacy_affinity": "none",
            "floating_allowed": True,
        }
    if "floating" in traits and "wall_backed" not in traits:
        return {
            "preferred_regions": _profile_regions(
                affordance_summary.get("floating_zone_candidates")
            ),
            "avoid_regions": _profile_regions(
                _profile_regions(affordance_summary.get("entry_landing_zones"))
                + _profile_regions(
                    affordance_summary.get("primary_circulation_corridors")
                )
            ),
            "wall_affinity": "medium",
            "daylight_affinity": "medium",
            "privacy_affinity": "none",
            "floating_allowed": True,
        }
    if traits & {"workflow", "storage", "wall_backed"}:
        return {
            "preferred_regions": _profile_regions(
                _profile_regions(affordance_summary.get("wall_anchor_candidates"))
                + _profile_regions(affordance_summary.get("focal_surfaces"))
            ),
            "avoid_regions": _profile_regions(
                _profile_regions(affordance_summary.get("center_openness_regions"))
                + _profile_regions(
                    affordance_summary.get("primary_circulation_corridors")
                )
                + _profile_regions(affordance_summary.get("entry_landing_zones"))
            ),
            "wall_affinity": "high",
            "daylight_affinity": "medium",
            "privacy_affinity": "none",
            "floating_allowed": False,
        }
    return None


def profile_relation_intents_for_objects(
    *,
    cluster_id: object,
    object_types: Sequence[str],
    room_type: object | None = None,
) -> list[dict[str, Any]]:
    _ = cluster_id
    profile = _profile_for_objects(object_types, room_type=room_type)
    if profile is None or not _profile_layout_semantics_enabled(profile):
        return []

    canonical_types = set(_profile_canonical_object_types(profile, object_types))
    traits = _profile_object_trait_set(profile, object_types)
    intents: list[dict[str, Any]] = []
    if "workflow" in traits:
        intents.append(
            {"type": "claim_wall", "target": "usable_wall", "strength": "hard"}
        )
        intents.append(
            {"type": "preserve_center", "target": "circulation", "strength": "soft"}
        )
    elif "floating" in traits:
        workflow_target = profile.relation_targets.get("workflow")
        if (
            profile.canonical_room_type == "kitchen"
            and canonical_types & {"dining_table", "dining_chair"}
        ):
            intents.append(
                {"type": "claim_daylight", "target": "window_side", "strength": "soft"}
            )
        if workflow_target:
            intents.append(
                {
                    "type": "near",
                    "target_cluster": workflow_target,
                    "strength": "soft",
                }
            )
    return intents


def profile_macro_relations_for_active_clusters(
    *,
    room_type: object | None,
    active_clusters: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, object]]]:
    adjacency: list[dict[str, object]] = []
    by_id = {
        str(cluster.get("cluster_id") or ""): cluster for cluster in active_clusters
    }

    for profile in _layout_enabled_profiles():
        if not _profile_layout_semantics_enabled(profile):
            continue
        target_cluster = profile.relation_targets.get("workflow")
        if not target_cluster or target_cluster not in by_id:
            continue

        target_objects = _object_types_from_active_cluster(by_id[target_cluster])
        if _profile_for_objects(target_objects, room_type=room_type) != profile:
            continue
        if "workflow" not in _profile_object_trait_set(profile, target_objects):
            continue

        for cluster in active_clusters:
            cluster_id = str(cluster.get("cluster_id") or "")
            if not cluster_id or cluster_id == target_cluster:
                continue
            object_types = _object_types_from_active_cluster(cluster)
            if _profile_for_objects(object_types, room_type=room_type) != profile:
                continue
            traits = _profile_object_trait_set(profile, object_types)
            if "storage" in traits and "workflow" not in traits:
                adjacency.append(
                    {
                        "a": target_cluster,
                        "b": cluster_id,
                        "relation": "near",
                        "priority": "high",
                    }
                )
            elif "floating" in traits:
                adjacency.append(
                    {
                        "a": target_cluster,
                        "b": cluster_id,
                        "relation": "near",
                        "priority": "medium",
                    }
                )

    return {
        "adjacency_preferences": adjacency,
        "separation_preferences": [],
        "orientation_preferences": [],
    }


def profile_layout_trace_for_active_clusters(
    *,
    room_type: object,
    active_clusters: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    profile = resolve_room_profile(room_type)
    if profile is None:
        return None
    layout_traits_enabled = _profile_layout_traits_enabled(profile)

    total_count = 0
    recognized_count = 0
    traited_count = 0
    trait_counts: dict[str, int] = {}
    unrecognized_objects: set[str] = set()
    untraited_objects: set[str] = set()
    clusters: list[dict[str, Any]] = []

    for cluster in active_clusters:
        cluster_id = str(cluster.get("cluster_id") or "").strip()
        object_types = _object_types_from_active_cluster(cluster)
        object_rows: list[dict[str, Any]] = []
        cluster_traits: set[str] = set()
        cluster_recognized = 0

        for object_type in object_types:
            total_count += 1
            canonical = profile.canonical_object_type(object_type)
            traits = profile.object_traits(object_type, include_shadow=True)
            if canonical is not None:
                recognized_count += 1
                cluster_recognized += 1
            else:
                unrecognized_objects.add(object_type)
            if traits:
                traited_count += 1
            else:
                untraited_objects.add(object_type)
            for trait in traits:
                trait_counts[trait] = trait_counts.get(trait, 0) + 1
                cluster_traits.add(trait)

            object_row: dict[str, Any] = {
                "object_type": object_type,
                "canonical_type": canonical,
                "traits": list(traits),
                "recognized": canonical is not None,
            }
            size_profile = profile.fallback_size_profile(object_type)
            if isinstance(size_profile, Mapping):
                rep_dims = size_profile.get("rep_dims_m")
                if isinstance(rep_dims, Mapping):
                    object_row["size_tiers"] = sorted(str(key) for key in rep_dims)
            object_rows.append(object_row)

        clusters.append(
            {
                "cluster_id": cluster_id,
                "object_count": len(object_types),
                "recognized_count": cluster_recognized,
                "dominant_traits": sorted(cluster_traits),
                "objects": object_rows,
            }
        )

    coverage_ratio = (
        round(recognized_count / total_count, 3) if total_count > 0 else 0.0
    )
    trait_coverage_ratio = (
        round(traited_count / total_count, 3) if total_count > 0 else 0.0
    )
    promotion_readiness = _profile_promotion_readiness(
        profile=profile,
        layout_traits_enabled=layout_traits_enabled,
        total_count=total_count,
        coverage_ratio=coverage_ratio,
        trait_coverage_ratio=trait_coverage_ratio,
        unrecognized_objects=sorted(unrecognized_objects),
        untraited_objects=sorted(untraited_objects),
    )
    return {
        "profile_id": profile.profile_id,
        "mode": "active" if layout_traits_enabled else "shadow",
        "layout_traits_enabled": layout_traits_enabled,
        "layout_traits_intrinsic_enabled": profile.layout_traits_enabled,
        "layout_traits_mode": _room_profile_traits_mode(),
        "semantic_rule_enabled": profile.semantic_room_rule_provider is not None,
        "object_count": total_count,
        "recognized_count": recognized_count,
        "coverage_ratio": coverage_ratio,
        "trait_coverage_ratio": trait_coverage_ratio,
        "trait_counts": dict(sorted(trait_counts.items())),
        "promotion_readiness": promotion_readiness,
        "clusters": clusters,
    }


def profile_shadow_trace_for_active_clusters(
    *,
    room_type: object,
    active_clusters: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    return profile_layout_trace_for_active_clusters(
        room_type=room_type,
        active_clusters=active_clusters,
    )


def _profile_promotion_readiness(
    *,
    profile: RoomProfile,
    layout_traits_enabled: bool,
    total_count: int,
    coverage_ratio: float,
    trait_coverage_ratio: float,
    unrecognized_objects: Sequence[str],
    untraited_objects: Sequence[str],
) -> dict[str, Any]:
    blockers: list[str] = []
    if total_count <= 0:
        blockers.append("no_active_objects")
    if coverage_ratio < _PROFILE_READINESS_MIN_COVERAGE:
        blockers.append("coverage_below_threshold")
    if trait_coverage_ratio < _PROFILE_READINESS_MIN_TRAIT_COVERAGE:
        blockers.append("trait_coverage_below_threshold")
    if unrecognized_objects:
        blockers.append("unrecognized_objects")
    if untraited_objects:
        blockers.append("untraited_objects")

    if layout_traits_enabled:
        status = "active"
    elif blockers:
        status = "needs_profile_expansion"
    else:
        status = "ready_for_profile_first"

    return {
        "status": status,
        "eligible_for_profile_first": status == "ready_for_profile_first",
        "minimum_coverage_ratio": _PROFILE_READINESS_MIN_COVERAGE,
        "minimum_trait_coverage_ratio": _PROFILE_READINESS_MIN_TRAIT_COVERAGE,
        "blockers": blockers,
        "unrecognized_objects": list(unrecognized_objects),
        "untraited_objects": list(untraited_objects),
    }


def _room_profile_rule_mode() -> str:
    raw_mode = str(
        os.getenv(_ROOM_PROFILE_RULE_MODE_ENV) or _DEFAULT_ROOM_PROFILE_RULE_MODE
    )
    mode = normalize_profile_token(raw_mode)
    mode_aliases = {
        "profile": "unified",
        "profile_first": "unified",
        "profile_first_default": "unified",
        "dry_run": "shadow",
    }
    mode = mode_aliases.get(mode, mode)
    return mode if mode in _ROOM_PROFILE_RULE_MODES else _DEFAULT_ROOM_PROFILE_RULE_MODE


def _room_profile_traits_mode() -> str:
    raw_mode = str(
        os.getenv(_ROOM_PROFILE_TRAITS_MODE_ENV) or _DEFAULT_ROOM_PROFILE_TRAITS_MODE
    )
    mode = normalize_profile_token(raw_mode)
    mode_aliases = {
        "active": "unified",
        "layout": "unified",
        "profile": "unified",
        "profile_traits": "unified",
        "dry_run": "shadow",
        "legacy": "shadow",
    }
    mode = mode_aliases.get(mode, mode)
    return (
        mode
        if mode in _ROOM_PROFILE_TRAITS_MODES
        else _DEFAULT_ROOM_PROFILE_TRAITS_MODE
    )


def _profile_rule_requested(
    *,
    room_type: str,
    profile: RoomProfile | None,
    mode: str,
) -> bool:
    if profile is None or mode in {"legacy", "shadow"}:
        return False
    if profile.layout_traits_enabled or mode == "unified":
        return True
    if mode != "canary":
        return False

    selectors = _profile_first_selectors()
    if not selectors:
        return False
    match_tokens = {
        normalize_profile_token(room_type),
        normalize_profile_token(profile.profile_id),
        normalize_profile_token(profile.canonical_room_type),
        *(normalize_profile_token(value) for value in profile.room_types),
    }
    return bool({"all", "*"} & selectors or match_tokens & selectors)


def _profile_layout_traits_enabled(profile: RoomProfile) -> bool:
    if profile.layout_traits_enabled:
        return True

    mode = _room_profile_traits_mode()
    if mode in {"legacy", "shadow"}:
        return False
    if mode == "unified":
        return True
    if mode != "canary":
        return False

    selectors = _profile_trait_selectors()
    if not selectors:
        return False
    match_tokens = {
        normalize_profile_token(profile.profile_id),
        normalize_profile_token(profile.canonical_room_type),
        *(normalize_profile_token(value) for value in profile.room_types),
    }
    return bool({"all", "*"} & selectors or match_tokens & selectors)


def _profile_has_effective_trait_object(
    profile: RoomProfile,
    object_type: object,
) -> bool:
    if not _profile_layout_traits_enabled(profile):
        return False
    return profile.has_trait_object(object_type, include_shadow=True)


def _profile_layout_semantics_enabled(profile: RoomProfile) -> bool:
    if not _profile_layout_traits_enabled(profile):
        return False
    return bool(
        profile.cluster_tag
        or profile.semantic_roles_by_object
        or profile.relation_targets
    )


def _profile_canonical_object_types(
    profile: RoomProfile,
    object_types: Sequence[str],
) -> list[str]:
    canonical_types: list[str] = []
    for object_type in object_types:
        canonical = profile.canonical_object_type(object_type)
        if canonical is not None:
            canonical_types.append(canonical)
    return canonical_types


def _profile_object_trait_set(
    profile: RoomProfile,
    object_types: Sequence[str],
) -> set[str]:
    traits: set[str] = set()
    for object_type in object_types:
        traits.update(profile.object_traits(object_type, include_shadow=True))
    return traits


def _profile_semantic_role_override(
    profile: RoomProfile,
    canonical_types: Sequence[str],
) -> str | None:
    roles = [
        profile.semantic_roles_by_object[object_type]
        for object_type in canonical_types
        if object_type in profile.semantic_roles_by_object
    ]
    if not roles:
        return None
    first_role = roles[0]
    if all(role == first_role for role in roles):
        return first_role
    return first_role


def _profile_regions(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    out: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text)
        if len(out) >= 4:
            break
    return out


def _profile_rule_blocked_by_regression_gate(
    *,
    profile: RoomProfile | None,
    mode: str,
    dry_run_diff: Mapping[str, Any],
) -> bool:
    if profile is None or profile.layout_traits_enabled or mode != "unified":
        return False
    return not bool(dry_run_diff.get("equivalent"))


def _profile_first_selectors() -> frozenset[str]:
    raw_value = str(os.getenv(_ROOM_PROFILE_FIRST_ROOMS_ENV) or "")
    tokens = {
        normalize_profile_token(item)
        for item in raw_value.replace(";", ",").split(",")
        if normalize_profile_token(item)
    }
    return frozenset(tokens)


def _profile_trait_selectors() -> frozenset[str]:
    raw_value = str(os.getenv(_ROOM_PROFILE_TRAIT_ROOMS_ENV) or "")
    tokens = {
        normalize_profile_token(item)
        for item in raw_value.replace(";", ",").split(",")
        if normalize_profile_token(item)
    }
    return frozenset(tokens)


def _clone_rule(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return deepcopy(dict(value))


def _fallback_empty_rule(room_type: str) -> dict[str, Any]:
    return {
        "room_type": room_type,
        "policy": {"selection_policy": "fallback"},
        "clusters": [],
        "global_program": {},
    }


def _room_rule_dry_run_diff(
    *,
    profile_rule: Mapping[str, Any] | None,
    legacy_rule: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if profile_rule is None and legacy_rule is None:
        return {"status": "no_rules", "equivalent": True}
    if profile_rule is None:
        return {"status": "legacy_only", "equivalent": False}
    if legacy_rule is None:
        return {"status": "profile_only", "equivalent": False}

    profile_clusters = _room_rule_cluster_signature(profile_rule)
    legacy_clusters = _room_rule_cluster_signature(legacy_rule)
    profile_ids = set(profile_clusters)
    legacy_ids = set(legacy_clusters)
    added_clusters = sorted(profile_ids - legacy_ids)
    removed_clusters = sorted(legacy_ids - profile_ids)
    changed_clusters: list[dict[str, Any]] = []

    for cluster_id in sorted(profile_ids & legacy_ids):
        profile_cluster = profile_clusters[cluster_id]
        legacy_cluster = legacy_clusters[cluster_id]
        profile_objects = set(profile_cluster["object_types"])
        legacy_objects = set(legacy_cluster["object_types"])
        added_objects = sorted(profile_objects - legacy_objects)
        removed_objects = sorted(legacy_objects - profile_objects)
        priority_changed = profile_cluster["priority"] != legacy_cluster["priority"]
        if added_objects or removed_objects or priority_changed:
            changed_clusters.append(
                {
                    "cluster_id": cluster_id,
                    "added_objects": added_objects,
                    "removed_objects": removed_objects,
                    "priority_changed": priority_changed,
                    "profile_priority": profile_cluster["priority"],
                    "legacy_priority": legacy_cluster["priority"],
                }
            )

    equivalent = not (added_clusters or removed_clusters or changed_clusters)
    return {
        "status": "equivalent" if equivalent else "different",
        "equivalent": equivalent,
        "added_clusters": added_clusters,
        "removed_clusters": removed_clusters,
        "changed_clusters": changed_clusters,
        "profile_cluster_count": len(profile_clusters),
        "legacy_cluster_count": len(legacy_clusters),
    }


def _room_rule_cluster_signature(
    rule: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    clusters = rule.get("clusters")
    if not isinstance(clusters, Sequence) or isinstance(clusters, str):
        return {}

    out: dict[str, dict[str, Any]] = {}
    for cluster in clusters:
        if not isinstance(cluster, Mapping):
            continue
        cluster_id = str(cluster.get("cluster_id") or "").strip()
        if not cluster_id:
            continue
        object_program = cluster.get("object_program")
        out[cluster_id] = {
            "priority": str(cluster.get("priority") or "").strip(),
            "object_types": sorted(_object_program_types(object_program)),
        }
    return out


def _object_program_types(value: object) -> tuple[str, ...]:
    object_types: set[str] = set()
    _collect_object_program_types(value, object_types)
    return tuple(sorted(object_types))


def _collect_object_program_types(value: object, out: set[str]) -> None:
    if isinstance(value, str):
        token = normalize_profile_token(value)
        if token:
            out.add(token)
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "by_object" and isinstance(child, Mapping):
                out.update(
                    normalize_profile_token(item)
                    for item in child
                    if normalize_profile_token(item)
                )
                continue
            _collect_object_program_types(child, out)
        return
    if isinstance(value, Sequence):
        for child in value:
            _collect_object_program_types(child, out)


def _profiles_for_members(members: Sequence[str]) -> list[RoomProfile]:
    return [
        profile
        for profile in _layout_enabled_profiles()
        if any(
            _profile_has_effective_trait_object(profile, member) for member in members
        )
    ]


def _profile_for_objects(
    object_types: Sequence[str],
    *,
    room_type: object | None = None,
) -> RoomProfile | None:
    preferred = resolve_room_profile(room_type)
    if (
        preferred is not None
        and _profile_layout_traits_enabled(preferred)
        and any(
            _profile_has_effective_trait_object(preferred, item)
            for item in object_types
        )
    ):
        return preferred

    best_profile: RoomProfile | None = None
    best_score = 0
    for profile in _layout_enabled_profiles():
        score = sum(
            1
            for item in object_types
            if _profile_has_effective_trait_object(profile, item)
        )
        if score > best_score:
            best_profile = profile
            best_score = score
    return best_profile


def _layout_enabled_profiles() -> tuple[RoomProfile, ...]:
    return tuple(
        profile for profile in ROOM_PROFILES if _profile_layout_traits_enabled(profile)
    )


def _object_types_from_active_cluster(cluster: Mapping[str, Any]) -> list[str]:
    object_types: list[str] = []
    bundles = cluster.get("required_bundles")
    if isinstance(bundles, Sequence) and not isinstance(bundles, str):
        for bundle in bundles:
            if not isinstance(bundle, Mapping):
                continue
            objects = bundle.get("objects")
            if not isinstance(objects, Sequence) or isinstance(objects, str):
                continue
            for obj in objects:
                if not isinstance(obj, Mapping):
                    continue
                object_type = str(obj.get("object_type") or "").strip()
                if object_type:
                    object_types.append(object_type)
    if object_types:
        return object_types

    members = cluster.get("members")
    if isinstance(members, Sequence) and not isinstance(members, str):
        return [str(member).strip() for member in members if str(member).strip()]
    return []


def _request_alias_variants(
    canonical: str,
    aliases: Sequence[str],
) -> tuple[str, ...]:
    values: list[str] = []
    for value in (canonical, canonical.replace("_", " "), *aliases):
        text = str(value or "").strip().lower()
        if not text:
            continue
        variants = (text, text.replace("_", " "))
        for variant in variants:
            if variant and variant not in values:
                values.append(variant)
    return tuple(values)


def _scoring_alias_variants(
    canonical: str,
    aliases: Sequence[str],
) -> tuple[str, ...]:
    values: list[str] = []
    for value in (canonical, *aliases):
        token = normalize_profile_token(value)
        if token and token not in values:
            values.append(token)
    return tuple(values)
