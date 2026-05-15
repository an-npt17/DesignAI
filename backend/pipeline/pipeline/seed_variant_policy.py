from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SeedVariantBrief:
    source: str
    model_name: str
    reason: str
    special_notes: str
    focus_mode: str
    center_open_preference: str
    support_cluster_behavior: str
    distribution_mode: str
    constraint_tokens: tuple[str, ...]


def build_seed_variant_briefs(target_count: int) -> list[SeedVariantBrief]:
    briefs = [
        SeedVariantBrief(
            source="seed_stable",
            model_name="gemma-3-27b-it",
            reason="Keep the seed structure stable and improve the macro logic around it.",
            special_notes=(
                "Variant brief: stay close to the current seed, preserve the existing side-of-room distribution where possible, "
                "and only make the smallest macro changes needed to improve plausibility."
            ),
            focus_mode="mixed",
            center_open_preference="medium",
            support_cluster_behavior="balanced",
            distribution_mode="balanced",
            constraint_tokens=(
                "seed_region_preservation",
                "primary_center_anchor",
                "balanced_support",
            ),
        ),
        SeedVariantBrief(
            source="seed_focal_axis",
            model_name="gemini-3.1-flash-lite-preview",
            reason="Strengthen the focal or viewing axis without forcing a full rearrangement.",
            special_notes=(
                "Variant brief: bias the primary pair into a stronger focal-facing arrangement and make this option read as a grouped "
                "viewing composition rather than another balanced seed clone."
            ),
            focus_mode="viewing",
            center_open_preference="medium",
            support_cluster_behavior="integrate",
            distribution_mode="focal_grouped",
            constraint_tokens=(
                "focal_axis_lock",
                "primary_faces_secondary",
                "secondary_wall_backed",
            ),
        ),
        SeedVariantBrief(
            source="seed_open_center",
            model_name="gemma-4-31b-it",
            reason="Protect a cleaner open center and clearer entry circulation.",
            special_notes=(
                "Variant brief: deliberately keep a cleaner center lane and move support pressure away from the central field, "
                "so this option reads differently from the focal-axis and stable variants."
            ),
            focus_mode="mixed",
            center_open_preference="high",
            support_cluster_behavior="recede",
            distribution_mode="balanced",
            constraint_tokens=(
                "center_lane_lock",
                "support_far_from_primary",
                "support_edge_bias",
            ),
        ),
        SeedVariantBrief(
            source="seed_edge_weighted",
            model_name="gemma-3-27b-it",
            reason="Move support pressure toward edges or recesses to free the main zone.",
            special_notes=(
                "Variant brief: push support and storage clusters harder toward perimeter or recess conditions and keep them from "
                "pulling the primary pair back toward the middle."
            ),
            focus_mode="mixed",
            center_open_preference="high",
            support_cluster_behavior="recede",
            distribution_mode="edge_weighted",
            constraint_tokens=(
                "perimeter_bias_lock",
                "all_clusters_avoid_center",
                "wall_backed_primary_pair",
            ),
        ),
        SeedVariantBrief(
            source="seed_zoned",
            model_name="gemma-4-31b-it",
            reason="Create a more clearly zoned arrangement from the current seed.",
            special_notes=(
                "Variant brief: form a visibly separate support zone relative to the primary pair and keep each cluster family in "
                "its own macro territory instead of collapsing back into a single balanced composition."
            ),
            focus_mode="mixed",
            center_open_preference="medium",
            support_cluster_behavior="recede",
            distribution_mode="zoned",
            constraint_tokens=(
                "primary_zone_claim",
                "support_zone_split",
                "support_separate_from_secondary",
            ),
        ),
    ]
    return briefs[: max(1, int(target_count))]


def apply_seed_variant_policy(
    *,
    relation_plan: dict[str, Any],
    planner_clusters_json: dict[str, Any],
    brief: SeedVariantBrief,
) -> dict[str, Any]:
    plan = deepcopy(relation_plan) if isinstance(relation_plan, dict) else {}
    clusters = _planner_clusters_map(planner_clusters_json)
    if not clusters:
        return plan

    primary_cluster_id, secondary_cluster_id = _resolve_primary_pair(plan, clusters)
    support_cluster_ids = [
        cluster_id
        for cluster_id in clusters
        if cluster_id not in {primary_cluster_id, secondary_cluster_id}
    ]
    _set_layout_intent_profile(
        plan,
        primary_cluster_id=primary_cluster_id,
        secondary_cluster_id=secondary_cluster_id,
        brief=brief,
    )
    _append_note(
        plan,
        (
            f"Seed variant policy {brief.source}: distribution={brief.distribution_mode}, "
            f"center_open={brief.center_open_preference}, support_behavior={brief.support_cluster_behavior}."
        ),
    )
    _ensure_guideline(
        plan,
        "Preserve a materially distinct macro geometry from the other post-seed variants.",
    )

    if primary_cluster_id is not None and secondary_cluster_id is not None:
        _ensure_cluster_relation(
            plan,
            a=primary_cluster_id,
            b=secondary_cluster_id,
            relation="near",
            priority="high",
            reason="Keep the primary pair coherent while exploring the seed variant policy.",
        )
        _ensure_cluster_directional_relation(
            plan,
            a=primary_cluster_id,
            b=secondary_cluster_id,
            relation="face_each_other",
            priority="high",
            reason="Primary pair should preserve a legible viewing or conversational relationship.",
        )

    if brief.source == "seed_stable":
        _apply_stable_policy(
            plan,
            clusters=clusters,
            primary_cluster_id=primary_cluster_id,
            secondary_cluster_id=secondary_cluster_id,
            support_cluster_ids=support_cluster_ids,
        )
    elif brief.source == "seed_focal_axis":
        _apply_focal_axis_policy(
            plan,
            primary_cluster_id=primary_cluster_id,
            secondary_cluster_id=secondary_cluster_id,
        )
    elif brief.source == "seed_open_center":
        _apply_open_center_policy(
            plan,
            clusters=clusters,
            primary_cluster_id=primary_cluster_id,
            secondary_cluster_id=secondary_cluster_id,
            support_cluster_ids=support_cluster_ids,
        )
    elif brief.source == "seed_edge_weighted":
        _apply_edge_weighted_policy(
            plan,
            clusters=clusters,
            primary_cluster_id=primary_cluster_id,
            secondary_cluster_id=secondary_cluster_id,
            support_cluster_ids=support_cluster_ids,
        )
    elif brief.source == "seed_zoned":
        _apply_zoned_policy(
            plan,
            clusters=clusters,
            primary_cluster_id=primary_cluster_id,
            secondary_cluster_id=secondary_cluster_id,
            support_cluster_ids=support_cluster_ids,
        )

    _set_seed_variant_policy_metadata(plan, brief=brief)
    return plan


def _apply_stable_policy(
    plan: dict[str, Any],
    *,
    clusters: dict[str, dict[str, Any]],
    primary_cluster_id: str | None,
    secondary_cluster_id: str | None,
    support_cluster_ids: list[str],
) -> None:
    for cluster_id in (primary_cluster_id, secondary_cluster_id):
        if cluster_id is None:
            continue
        region_tags = _seed_region_tags(clusters.get(cluster_id) or {})
        prefer_add = [
            tag
            for tag in region_tags
            if tag in {"entry_side", "far_from_entry", "window_side"}
        ]
        if cluster_id == primary_cluster_id and "near_center" in region_tags:
            prefer_add.append("center")
        if prefer_add:
            _ensure_cluster_affinity(
                plan,
                cluster_id=cluster_id,
                prefer_add=prefer_add,
                avoid_add=[],
                priority="medium",
                reason="Stable variant should preserve the seed-side territory of the primary pair where feasible.",
            )
    for cluster_id in support_cluster_ids:
        region_tags = _seed_region_tags(clusters.get(cluster_id) or {})
        prefer_add = [
            tag
            for tag in region_tags
            if tag in {"entry_side", "far_from_entry", "window_side"}
        ]
        if prefer_add:
            _ensure_cluster_affinity(
                plan,
                cluster_id=cluster_id,
                prefer_add=prefer_add,
                avoid_add=[],
                priority="medium",
                reason="Stable variant should preserve the seed-side territory of support clusters where feasible.",
            )
    if primary_cluster_id is not None and secondary_cluster_id is not None:
        _ensure_cluster_orientation(
            plan,
            cluster_id=primary_cluster_id,
            intents=["face_cluster"],
            target_cluster_id=secondary_cluster_id,
            priority="medium",
            reason="Stable variant should preserve the legible frontality of the seeded primary pair.",
        )
    if secondary_cluster_id is not None:
        _ensure_cluster_orientation(
            plan,
            cluster_id=secondary_cluster_id,
            intents=["back_to_wall"],
            target_cluster_id=None,
            priority="medium",
            reason="Stable variant should keep the secondary focal cluster wall-backed where feasible.",
        )
    _ensure_guideline(
        plan,
        "Stay close to the current seed-side occupancy and avoid unnecessary macro relocation.",
    )


def _apply_focal_axis_policy(
    plan: dict[str, Any],
    *,
    primary_cluster_id: str | None,
    secondary_cluster_id: str | None,
) -> None:
    if primary_cluster_id is not None:
        _ensure_cluster_affinity(
            plan,
            cluster_id=primary_cluster_id,
            prefer_add=["wall", "far_from_entry"],
            avoid_add=["center", "door_swing"],
            priority="high",
            reason="Primary seating should gain a cleaner backed viewing position.",
        )
        _ensure_cluster_orientation(
            plan,
            cluster_id=primary_cluster_id,
            intents=["face_cluster", "inward_to_room"],
            target_cluster_id=secondary_cluster_id,
            priority="high",
            reason="Primary seating should clearly face the focal cluster in the focal-axis variant.",
        )
    if secondary_cluster_id is not None:
        _ensure_cluster_affinity(
            plan,
            cluster_id=secondary_cluster_id,
            prefer_add=["wall", "far_from_entry"],
            avoid_add=["center", "window_blocking"],
            prefer_remove=["window_side"],
            priority="high",
            reason="Focal/media cluster should keep a strong viewing axis instead of drifting toward glazing.",
        )
        _ensure_cluster_orientation(
            plan,
            cluster_id=secondary_cluster_id,
            intents=["back_to_wall", "axis_parallel_wall"],
            target_cluster_id=None,
            priority="high",
            reason="Focal/media cluster should read as the most wall-backed anchored element in the focal-axis variant.",
        )
    _ensure_guideline(
        plan,
        "Make this the strongest focal-axis option, with the primary pair reading as a grouped viewing composition.",
    )


def _apply_open_center_policy(
    plan: dict[str, Any],
    *,
    clusters: dict[str, dict[str, Any]],
    primary_cluster_id: str | None,
    secondary_cluster_id: str | None,
    support_cluster_ids: list[str],
) -> None:
    _ensure_keep_open_region(
        plan,
        region_type="center_lane",
        near="room_center",
        priority="high",
        reason="Open-center variant should preserve a visibly cleaner center lane.",
    )
    primary_pair = {
        cluster_id
        for cluster_id in (primary_cluster_id, secondary_cluster_id)
        if isinstance(cluster_id, str)
    }
    for cluster_id in clusters:
        if cluster_id in primary_pair:
            _ensure_cluster_affinity(
                plan,
                cluster_id=cluster_id,
                prefer_add=[],
                avoid_add=["entry_blocking"],
                priority="medium",
                reason="Primary pair should not pinch the cleaner center or entry corridor.",
            )
            continue
        _ensure_support_edge_bias(
            plan,
            cluster_id=cluster_id,
            region_tags=_seed_region_tags(clusters[cluster_id]),
            priority="high",
            reason="Support clusters should vacate the center to preserve a cleaner open field.",
        )
    for cluster_id in support_cluster_ids:
        _ensure_cluster_relation(
            plan,
            a=cluster_id,
            b=primary_cluster_id or cluster_id,
            relation="far_if_possible",
            priority="medium",
            reason="Open-center variant should let support clusters recede from the main pair.",
        )
    if primary_cluster_id is not None:
        _ensure_cluster_orientation(
            plan,
            cluster_id=primary_cluster_id,
            intents=["access_to_open_space", "inward_to_room"],
            target_cluster_id=None,
            priority="high",
            reason="Primary cluster should front into the clearer central field in the open-center variant.",
        )
    if secondary_cluster_id is not None:
        _ensure_cluster_orientation(
            plan,
            cluster_id=secondary_cluster_id,
            intents=["back_to_wall"],
            target_cluster_id=None,
            priority="medium",
            reason="Secondary cluster should stay wall-backed while leaving the center field cleaner.",
        )
    _ensure_guideline(
        plan,
        "Keep the center visibly more open than the stable and focal-axis variants.",
    )


def _apply_edge_weighted_policy(
    plan: dict[str, Any],
    *,
    clusters: dict[str, dict[str, Any]],
    primary_cluster_id: str | None,
    secondary_cluster_id: str | None,
    support_cluster_ids: list[str],
) -> None:
    _ensure_keep_open_region(
        plan,
        region_type="center_lane",
        near="room_center",
        priority="high",
        reason="Edge-weighted variant should keep the premium center field relatively free.",
    )
    for cluster_id in support_cluster_ids:
        _ensure_support_edge_bias(
            plan,
            cluster_id=cluster_id,
            region_tags=_seed_region_tags(clusters[cluster_id]),
            priority="high",
            reason="Support clusters should bias toward edge or recess conditions in the edge-weighted variant.",
        )
        if primary_cluster_id is not None:
            _ensure_cluster_relation(
                plan,
                a=cluster_id,
                b=primary_cluster_id,
                relation="far_if_possible",
                priority="high",
                reason="Support clusters should stay off the primary field in the edge-weighted variant.",
            )
    for cluster_id in (primary_cluster_id, secondary_cluster_id):
        if cluster_id is None:
            continue
        _ensure_cluster_affinity(
            plan,
            cluster_id=cluster_id,
            prefer_add=["wall", "recess_or_edge", "far_from_entry"],
            avoid_add=["center", "main_path"],
            priority="medium",
            reason="Primary pair should stay coherent while still reading as perimeter-biased.",
        )
        _ensure_cluster_orientation(
            plan,
            cluster_id=cluster_id,
            intents=["back_to_wall", "axis_parallel_wall"],
            target_cluster_id=None,
            priority="medium",
            reason="Edge-weighted variant should keep even the primary pair more wall-backed than other variants.",
        )
    for cluster_id in support_cluster_ids:
        _ensure_cluster_orientation(
            plan,
            cluster_id=cluster_id,
            intents=["back_to_wall", "axis_parallel_wall"],
            target_cluster_id=None,
            priority="medium",
            reason="Edge-weighted variant should align support clusters tightly with perimeter walls.",
        )
    _ensure_guideline(
        plan,
        "Make perimeter pressure stronger than the open-center variant so the main zone reads clearly edge-weighted.",
    )


def _apply_zoned_policy(
    plan: dict[str, Any],
    *,
    clusters: dict[str, dict[str, Any]],
    primary_cluster_id: str | None,
    secondary_cluster_id: str | None,
    support_cluster_ids: list[str],
) -> None:
    if primary_cluster_id is not None:
        _ensure_cluster_affinity(
            plan,
            cluster_id=primary_cluster_id,
            prefer_add=["center"],
            avoid_add=["main_path"],
            priority="high",
            reason="Zoned variant should let the primary cluster claim the main social field.",
        )
        _ensure_cluster_orientation(
            plan,
            cluster_id=primary_cluster_id,
            intents=["inward_to_room", "access_to_open_space"],
            target_cluster_id=None,
            priority="medium",
            reason="Primary zone should front into the room rather than collapsing into the support edge territory.",
        )
    if secondary_cluster_id is not None:
        _ensure_cluster_affinity(
            plan,
            cluster_id=secondary_cluster_id,
            prefer_add=["wall", "far_from_entry"],
            avoid_add=["center"],
            priority="high",
            reason="Zoned variant should keep the secondary cluster as the anchored edge of the main zone.",
        )
        _ensure_cluster_orientation(
            plan,
            cluster_id=secondary_cluster_id,
            intents=["back_to_wall", "axis_parallel_wall"],
            target_cluster_id=None,
            priority="high",
            reason="Secondary cluster should anchor the edge of the main zone in the zoned variant.",
        )
    for cluster_id in support_cluster_ids:
        region_tags = _seed_region_tags(clusters[cluster_id])
        prefer_add = ["wall", "recess_or_edge"]
        if "far_from_entry" in region_tags:
            prefer_add.append("far_from_entry")
        elif "entry_side" in region_tags:
            prefer_add.append("entry_side")
        elif "window_side" in region_tags:
            prefer_add.append("window_side")
        _ensure_cluster_affinity(
            plan,
            cluster_id=cluster_id,
            prefer_add=prefer_add,
            avoid_add=["center", "main_path"],
            priority="high",
            reason="Zoned variant should keep support clusters in a distinct macro territory.",
        )
        if primary_cluster_id is not None:
            _ensure_cluster_relation(
                plan,
                a=cluster_id,
                b=primary_cluster_id,
                relation="far_if_possible",
                priority="high",
                reason="Support zone should stay distinct from the primary cluster zone.",
            )
        if secondary_cluster_id is not None:
            _ensure_cluster_relation(
                plan,
                a=cluster_id,
                b=secondary_cluster_id,
                relation="separate",
                priority="medium",
                reason="Support zone should not collapse back into the primary pair.",
            )
        _ensure_cluster_orientation(
            plan,
            cluster_id=cluster_id,
            intents=["back_to_wall", "axis_parallel_wall"],
            target_cluster_id=None,
            priority="medium",
            reason="Support zone should read as its own wall-backed territory in the zoned variant.",
        )
    _ensure_guideline(
        plan,
        "Create the clearest support-versus-primary zoning split of the post-seed variants.",
    )


def _planner_clusters_map(
    planner_clusters_json: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    clusters = planner_clusters_json.get("clusters")
    if isinstance(clusters, dict):
        return {
            cluster_id: cluster
            for cluster_id, cluster in clusters.items()
            if isinstance(cluster_id, str) and isinstance(cluster, dict)
        }
    return {}


def _resolve_primary_pair(
    plan: dict[str, Any],
    clusters: dict[str, dict[str, Any]],
) -> tuple[str | None, str | None]:
    profile = (
        plan.get("layout_intent_profile")
        if isinstance(plan.get("layout_intent_profile"), dict)
        else {}
    )
    primary_cluster_id = _valid_cluster_id(clusters, profile.get("primary_cluster_id"))
    secondary_cluster_id = _valid_cluster_id(
        clusters, profile.get("secondary_cluster_id")
    )
    if primary_cluster_id is None:
        primary_cluster_id = _pick_cluster_by_role(
            clusters,
            preferred_roles=("lounge", "sleep", "work", "media", "support"),
        )
    if secondary_cluster_id is None or secondary_cluster_id == primary_cluster_id:
        primary_role = _cluster_role(clusters.get(primary_cluster_id) or {})
        if primary_role == "lounge":
            secondary_cluster_id = _pick_cluster_by_role(
                clusters,
                preferred_roles=("media", "support", "work"),
                exclude={primary_cluster_id},
            )
        else:
            secondary_cluster_id = _pick_cluster_by_role(
                clusters,
                preferred_roles=("lounge", "media", "support"),
                exclude={primary_cluster_id},
            )
    return primary_cluster_id, secondary_cluster_id


def _valid_cluster_id(
    clusters: dict[str, dict[str, Any]],
    value: Any,
) -> str | None:
    cluster_id = str(value or "").strip()
    if not cluster_id or cluster_id not in clusters:
        return None
    return cluster_id


def _pick_cluster_by_role(
    clusters: dict[str, dict[str, Any]],
    *,
    preferred_roles: tuple[str, ...],
    exclude: set[str] | None = None,
) -> str | None:
    excluded = exclude or set()
    ranked = sorted(
        (
            (preferred_roles.index(role), cluster_id)
            for cluster_id, cluster in clusters.items()
            if cluster_id not in excluded
            and (role := _cluster_role(cluster)) in preferred_roles
        ),
        key=lambda row: (row[0], row[1]),
    )
    return ranked[0][1] if ranked else None


def _cluster_role(cluster: dict[str, Any]) -> str:
    cluster_id = str(cluster.get("cluster_id") or "").lower()
    object_ids = [
        str(row.get("id") or "").lower()
        for row in cluster.get("local_placements") or []
        if isinstance(row, dict)
    ]
    tokens = " ".join([cluster_id, *object_ids])
    if any(
        token in tokens for token in ("sofa", "sectional", "living", "seat", "lounge")
    ):
        return "lounge"
    if any(token in tokens for token in ("tv", "media", "console", "screen")):
        return "media"
    if any(token in tokens for token in ("bed", "sleep")):
        return "sleep"
    if any(token in tokens for token in ("desk", "work", "office")):
        return "work"
    return "support"


def _seed_region_tags(cluster: dict[str, Any]) -> list[str]:
    seed_state = cluster.get("seed_state")
    region_tags = seed_state.get("region_tags") if isinstance(seed_state, dict) else []
    return [
        str(tag).strip()
        for tag in region_tags
        if isinstance(tag, str) and str(tag).strip()
    ]


def _set_layout_intent_profile(
    plan: dict[str, Any],
    *,
    primary_cluster_id: str | None,
    secondary_cluster_id: str | None,
    brief: SeedVariantBrief,
) -> None:
    profile = (
        deepcopy(plan.get("layout_intent_profile"))
        if isinstance(plan.get("layout_intent_profile"), dict)
        else {}
    )
    profile["focus_mode"] = brief.focus_mode
    profile["primary_cluster_id"] = primary_cluster_id
    profile["secondary_cluster_id"] = secondary_cluster_id
    profile["circulation_priority"] = (
        "high" if brief.center_open_preference == "high" else "medium"
    )
    profile["center_open_preference"] = brief.center_open_preference
    profile["support_cluster_behavior"] = brief.support_cluster_behavior
    profile["distribution_mode"] = brief.distribution_mode
    plan["layout_intent_profile"] = profile


def _set_seed_variant_policy_metadata(
    plan: dict[str, Any],
    *,
    brief: SeedVariantBrief,
) -> None:
    structural_fingerprint = _build_seed_variant_structural_fingerprint(plan)
    plan["seed_variant_policy"] = {
        "source": brief.source,
        "constraint_tokens": list(brief.constraint_tokens),
        "structural_fingerprint": structural_fingerprint,
    }
    _append_note(
        plan,
        (
            f"Seed constraint fingerprint {brief.source}: "
            f"{', '.join(brief.constraint_tokens)}."
        ),
    )


def _append_note(plan: dict[str, Any], text: str) -> None:
    notes = plan.get("notes")
    if not isinstance(notes, list):
        notes = []
        plan["notes"] = notes
    if text not in notes:
        notes.append(text)


def _uniq_keep_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        text = value.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _ensure_guideline(plan: dict[str, Any], text: str) -> None:
    if not text.strip():
        return
    rows = plan.get("placement_guidelines")
    if not isinstance(rows, list):
        rows = []
        plan["placement_guidelines"] = rows
    if text not in rows:
        rows.append(text)


def _ensure_keep_open_region(
    plan: dict[str, Any],
    *,
    region_type: str,
    near: str,
    priority: str,
    reason: str,
) -> None:
    circulation_plan = plan.get("circulation_plan")
    if not isinstance(circulation_plan, dict):
        circulation_plan = {"main_paths": [], "keep_open_regions": []}
        plan["circulation_plan"] = circulation_plan
    keep_open_regions = circulation_plan.get("keep_open_regions")
    if not isinstance(keep_open_regions, list):
        keep_open_regions = []
        circulation_plan["keep_open_regions"] = keep_open_regions
    for row in keep_open_regions:
        if not isinstance(row, dict):
            continue
        if row.get("type") == region_type and row.get("near") == near:
            row["priority"] = _max_priority(
                str(row.get("priority") or "medium"), priority
            )
            if not str(row.get("reason") or "").strip():
                row["reason"] = reason
            return
    keep_open_regions.append(
        {
            "type": region_type,
            "near": near,
            "priority": priority,
            "reason": reason,
        }
    )


def _ensure_cluster_affinity(
    plan: dict[str, Any],
    *,
    cluster_id: str,
    prefer_add: list[str],
    avoid_add: list[str],
    priority: str,
    reason: str,
    prefer_remove: list[str] | None = None,
) -> None:
    affinities = plan.get("cluster_affinities")
    if not isinstance(affinities, list):
        affinities = []
        plan["cluster_affinities"] = affinities
    prefer_remove_set = set(prefer_remove or [])
    for row in affinities:
        if not isinstance(row, dict) or row.get("cluster_id") != cluster_id:
            continue
        prefer = [item for item in row.get("prefer") or [] if isinstance(item, str)]
        avoid = [item for item in row.get("avoid") or [] if isinstance(item, str)]
        row["prefer"] = [
            item
            for item in _uniq_keep_order(prefer + list(prefer_add))
            if item not in prefer_remove_set
        ]
        row["avoid"] = _uniq_keep_order(avoid + list(avoid_add))
        row["priority"] = _max_priority(str(row.get("priority") or "medium"), priority)
        if not str(row.get("reason") or "").strip():
            row["reason"] = reason
        return
    affinities.append(
        {
            "cluster_id": cluster_id,
            "prefer": [
                item
                for item in _uniq_keep_order(list(prefer_add))
                if item not in prefer_remove_set
            ],
            "avoid": _uniq_keep_order(list(avoid_add)),
            "priority": priority,
            "reason": reason,
        }
    )


def _ensure_cluster_relation(
    plan: dict[str, Any],
    *,
    a: str,
    b: str,
    relation: str,
    priority: str,
    reason: str,
) -> None:
    if not a or not b or a == b:
        return
    relations = plan.get("cluster_relations")
    if not isinstance(relations, list):
        relations = []
        plan["cluster_relations"] = relations
    key = tuple(sorted((a, b)))
    for row in relations:
        if not isinstance(row, dict):
            continue
        existing_key = tuple(
            sorted(
                (
                    str(row.get("a") or "").strip(),
                    str(row.get("b") or "").strip(),
                )
            )
        )
        if existing_key != key:
            continue
        row["relation"] = _prefer_stronger_relation(
            str(row.get("relation") or ""), relation
        )
        row["priority"] = _max_priority(str(row.get("priority") or "medium"), priority)
        if not str(row.get("reason") or "").strip():
            row["reason"] = reason
        return
    relations.append(
        {
            "a": a,
            "b": b,
            "relation": relation,
            "priority": priority,
            "reason": reason,
        }
    )


def _ensure_cluster_directional_relation(
    plan: dict[str, Any],
    *,
    a: str,
    b: str,
    relation: str,
    priority: str,
    reason: str,
) -> None:
    if not a or not b or a == b:
        return
    rows = plan.get("cluster_directional_relations")
    if not isinstance(rows, list):
        rows = []
        plan["cluster_directional_relations"] = rows
    key = tuple(sorted((a, b)))
    for row in rows:
        if not isinstance(row, dict):
            continue
        existing_key = tuple(
            sorted(
                (
                    str(row.get("a") or "").strip(),
                    str(row.get("b") or "").strip(),
                )
            )
        )
        if existing_key != key:
            continue
        row["relation"] = relation
        row["priority"] = _max_priority(str(row.get("priority") or "medium"), priority)
        if not str(row.get("reason") or "").strip():
            row["reason"] = reason
        return
    rows.append(
        {
            "a": a,
            "b": b,
            "relation": relation,
            "priority": priority,
            "reason": reason,
        }
    )


def _ensure_cluster_orientation(
    plan: dict[str, Any],
    *,
    cluster_id: str,
    intents: list[str],
    target_cluster_id: str | None,
    priority: str,
    reason: str,
) -> None:
    rows = plan.get("cluster_orientations")
    if not isinstance(rows, list):
        rows = []
        plan["cluster_orientations"] = rows
    for row in rows:
        if not isinstance(row, dict) or row.get("cluster_id") != cluster_id:
            continue
        existing_intents = row.get("intents")
        if not isinstance(existing_intents, list):
            existing_intents = []
        row["intents"] = _uniq_keep_order(
            [item for item in existing_intents if isinstance(item, str)] + intents
        )
        row["priority"] = _max_priority(str(row.get("priority") or "medium"), priority)
        if "face_cluster" in row["intents"]:
            if (
                not str(row.get("target_cluster_id") or "").strip()
                and target_cluster_id is not None
            ):
                row["target_cluster_id"] = target_cluster_id
        else:
            row["target_cluster_id"] = None
        if not str(row.get("reason") or "").strip():
            row["reason"] = reason
        return
    rows.append(
        {
            "cluster_id": cluster_id,
            "intents": _uniq_keep_order(intents),
            "target_cluster_id": target_cluster_id
            if "face_cluster" in intents
            else None,
            "priority": priority,
            "reason": reason,
        }
    )


def _ensure_support_edge_bias(
    plan: dict[str, Any],
    *,
    cluster_id: str,
    region_tags: list[str],
    priority: str,
    reason: str,
) -> None:
    prefer_add = ["wall", "recess_or_edge"]
    if "far_from_entry" in region_tags:
        prefer_add.append("far_from_entry")
    if "window_side" in region_tags:
        prefer_add.append("window_side")
    _ensure_cluster_affinity(
        plan,
        cluster_id=cluster_id,
        prefer_add=prefer_add,
        avoid_add=["center", "main_path", "bottleneck"],
        priority=priority,
        reason=reason,
    )


def _prefer_stronger_relation(existing: str, new_relation: str) -> str:
    priorities = {
        "near": 4,
        "adjacent_if_possible": 3,
        "separate": 2,
        "far_if_possible": 1,
    }
    if priorities.get(new_relation, 0) >= priorities.get(existing, 0):
        return new_relation
    return existing


def _max_priority(left: str, right: str) -> str:
    order = {"low": 0, "medium": 1, "high": 2}
    return left if order.get(left, 1) >= order.get(right, 1) else right


def _build_seed_variant_structural_fingerprint(plan: dict[str, Any]) -> str:
    profile = plan.get("layout_intent_profile")
    profile_dict = profile if isinstance(profile, dict) else {}
    profile_tokens = [
        f"focus={str(profile_dict.get('focus_mode') or '').strip()}",
        f"center={str(profile_dict.get('center_open_preference') or '').strip()}",
        f"support={str(profile_dict.get('support_cluster_behavior') or '').strip()}",
        f"distribution={str(profile_dict.get('distribution_mode') or '').strip()}",
    ]

    circulation = plan.get("circulation_plan")
    circulation_dict = circulation if isinstance(circulation, dict) else {}
    keep_open_rows = circulation_dict.get("keep_open_regions")
    keep_open_tokens = sorted(
        f"{str(row.get('type') or '').strip()}:{str(row.get('near') or '').strip()}"
        for row in keep_open_rows or []
        if isinstance(row, dict)
    )

    affinity_tokens = sorted(
        (
            f"{str(row.get('cluster_id') or '').strip()}|"
            f"p={','.join(sorted(str(item).strip() for item in row.get('prefer') or [] if isinstance(item, str)))}|"
            f"a={','.join(sorted(str(item).strip() for item in row.get('avoid') or [] if isinstance(item, str)))}"
        )
        for row in plan.get("cluster_affinities") or []
        if isinstance(row, dict)
    )
    relation_tokens = sorted(
        (
            f"{'|'.join(sorted((str(row.get('a') or '').strip(), str(row.get('b') or '').strip())))}|"
            f"{str(row.get('relation') or '').strip()}"
        )
        for row in plan.get("cluster_relations") or []
        if isinstance(row, dict)
    )
    directional_tokens = sorted(
        (
            f"{'|'.join(sorted((str(row.get('a') or '').strip(), str(row.get('b') or '').strip())))}|"
            f"{str(row.get('relation') or '').strip()}"
        )
        for row in plan.get("cluster_directional_relations") or []
        if isinstance(row, dict)
    )
    orientation_tokens = sorted(
        (
            f"{str(row.get('cluster_id') or '').strip()}|"
            f"{','.join(sorted(str(item).strip() for item in row.get('intents') or [] if isinstance(item, str)))}|"
            f"{str(row.get('target_cluster_id') or '').strip()}"
        )
        for row in plan.get("cluster_orientations") or []
        if isinstance(row, dict)
    )

    segments = [
        ",".join(profile_tokens),
        ",".join(keep_open_tokens),
        ",".join(affinity_tokens),
        ",".join(relation_tokens),
        ",".join(directional_tokens),
        ",".join(orientation_tokens),
    ]
    return " || ".join(segment for segment in segments if segment)
