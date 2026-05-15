"""Cluster output merge utilities for the anchor-first object-level flow."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def merge_cluster_outputs(
    cluster_forge: dict[str, Any],
    tier_count: dict[str, Any],
) -> dict[str, Any]:
    """
    Merge ClusterForge output with TierCount decisions into a solver-ready object program.

    This stage now acts as the semantic/object program normalizer for the object-level
    anchor-first solver. It keeps only active objects, filters stale semantic references,
    and materializes per-cluster object metadata needed by downstream planning.
    """
    cluster_payload = _unwrap_payload(cluster_forge, key="clusters")
    tier_payload = _unwrap_payload(tier_count, key="decisions")

    clusters = cluster_payload.get("clusters", [])
    decisions = tier_payload.get("decisions", [])

    decision_by_cluster_and_type: dict[tuple[str, str], dict[str, Any]] = {}
    decision_by_type: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        obj_type = _decision_type_id(decision)
        if obj_type is None:
            continue
        qty = decision.get("quantity")
        if not isinstance(qty, int) or qty < 1:
            continue
        cluster_id = str(decision.get("cluster_id") or "").strip()
        if cluster_id:
            decision_by_cluster_and_type[(cluster_id, obj_type)] = deepcopy(decision)
        decision_by_type.setdefault(obj_type, deepcopy(decision))

    merged_clusters: list[dict[str, Any]] = []
    object_program_by_cluster: dict[str, dict[str, Any]] = {}
    active_cluster_ids: list[str] = []
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        cluster_id = str(cluster.get("cluster_id") or "").strip()
        if not cluster_id:
            continue
        members = cluster.get("members")
        if not isinstance(members, list):
            members = []

        kept_members = [
            member
            for member in members
            if isinstance(member, str)
            and member.strip()
            and _decision_for_member(
                cluster_id, member, decision_by_cluster_and_type, decision_by_type
            )
            is not None
        ]
        if not kept_members:
            continue
        kept_set = set(kept_members)

        merged = deepcopy(cluster)
        merged["members"] = kept_members

        anchors = cluster.get("anchors")
        filtered_anchors: list[str] = []
        if isinstance(anchors, list):
            filtered_anchors = [
                anchor
                for anchor in anchors
                if isinstance(anchor, str) and anchor.strip() and anchor in kept_set
            ]
        if not filtered_anchors:
            dominant_candidates = _string_list(
                _cluster_rules(cluster).get("dominant_anchor_candidates")
            )
            filtered_anchors = [
                anchor for anchor in dominant_candidates if anchor in kept_set
            ][:1]
        if not filtered_anchors:
            continue
        merged["anchors"] = filtered_anchors

        merged_decisions: list[dict[str, Any]] = []
        for member in kept_members:
            decision = _decision_for_member(
                cluster_id, member, decision_by_cluster_and_type, decision_by_type
            )
            if decision is None:
                continue
            decision_row = deepcopy(decision)
            decision_row["cluster_id"] = cluster_id
            merged_decisions.append(decision_row)
        merged["decisions"] = merged_decisions

        for key in ("hard_constraints", "soft_constraints"):
            constraints = cluster.get(key)
            if not isinstance(constraints, list):
                continue
            merged[key] = [
                deepcopy(item)
                for item in constraints
                if isinstance(item, dict)
                and _constraint_subjects(item).issubset(kept_set)
            ]

        rules = cluster.get("cluster_rules")
        if isinstance(rules, dict):
            merged["cluster_rules"] = _filter_cluster_rules(
                rules,
                kept_ids=kept_set,
                anchors=set(filtered_anchors),
            )

        object_program = _build_object_program_for_cluster(merged)
        merged["object_program"] = object_program
        merged["members"] = list(object_program.get("members") or [])
        merged["anchors"] = list(object_program.get("anchors") or [])
        merged_clusters.append(merged)
        object_program_by_cluster[cluster_id] = object_program
        active_cluster_ids.append(cluster_id)

    merged_output: dict[str, Any] = {
        "status": cluster_payload.get("status", "OK"),
        "planner_kind": "merged_object_program",
        "clusters": merged_clusters,
        "active_cluster_ids": active_cluster_ids,
        "object_program_by_cluster": object_program_by_cluster,
        "notes": _merged_notes(cluster_payload, tier_payload),
        "missing": deepcopy(cluster_payload.get("missing", [])),
    }

    semantic_program = cluster_payload.get("semantic_layout_program")
    if isinstance(semantic_program, dict):
        merged_output["semantic_layout_program"] = deepcopy(semantic_program)
    style_policy = cluster_payload.get("style_policy")
    if isinstance(style_policy, dict):
        merged_output["style_policy"] = deepcopy(style_policy)

    _remove_key_recursive(merged_output, "raw_text")
    return merged_output


def _build_object_program_for_cluster(cluster: dict[str, Any]) -> dict[str, Any]:
    cluster_id = str(cluster.get("cluster_id") or "").strip()
    base_members = [
        member
        for member in cluster.get("members", [])
        if isinstance(member, str) and member.strip()
    ]
    rules = _cluster_rules(cluster)
    anchor_first_policy = (
        rules.get("anchor_first_policy")
        if isinstance(rules.get("anchor_first_policy"), dict)
        else {}
    )
    decisions = (
        cluster.get("decisions") if isinstance(cluster.get("decisions"), list) else []
    )
    object_specs_by_id: dict[str, dict[str, Any]] = {}
    expanded_ids_by_base: dict[str, list[str]] = {}
    required_ids: list[str] = []
    optional_ids: list[str] = []
    decision_droppable_ids: list[str] = []

    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        base_object_id = _decision_type_id(decision)
        if base_object_id is None:
            continue
        rep_dims = (
            decision.get("rep_dims_m")
            if isinstance(decision.get("rep_dims_m"), dict)
            else {}
        )
        length_mm = int(round(float(rep_dims.get("L") or 0.0) * 1000.0))
        width_mm = int(round(float(rep_dims.get("W") or 0.0) * 1000.0))
        height_mm = int(round(float(rep_dims.get("H") or 0.0) * 1000.0))
        if length_mm <= 0 or width_mm <= 0:
            continue
        preserve_level = str(decision.get("preserve_level") or "").strip().lower()
        role = str(decision.get("role") or "").strip().lower()
        priority = str(decision.get("priority") or "").strip().lower()
        quantity = _decision_quantity(decision)
        min_keep = _decision_min_keep(decision)
        allowed_rotations = (
            rules.get("allowed_rotations")
            if isinstance(rules.get("allowed_rotations"), dict)
            else {}
        )
        facing = rules.get("facing") if isinstance(rules.get("facing"), dict) else {}
        for quantity_index in range(1, quantity + 1):
            object_id = _expanded_object_id(base_object_id, quantity_index)
            anchor_like = role in {"dominant_anchor", "anchor"} or priority == "anchor"
            required = quantity_index <= min_keep or (
                anchor_like
                and not bool(decision.get("droppable"))
                and quantity_index == 1
            )
            droppable = bool(decision.get("droppable")) or not required
            expanded_ids_by_base.setdefault(base_object_id, []).append(object_id)
            facing_row = facing.get(object_id) or facing.get(base_object_id) or {}
            if not isinstance(facing_row, dict):
                facing_row = {}
            object_specs_by_id[object_id] = {
                "object_id": object_id,
                "base_object_id": base_object_id,
                "quantity_index": quantity_index,
                "cluster_id": cluster_id,
                "category": str(decision.get("category") or base_object_id),
                "role": role,
                "priority": priority,
                "preserve_level": preserve_level,
                "size_tier": str(decision.get("size_tier") or ""),
                "rep_dims_mm": {
                    "L": length_mm,
                    "W": width_mm,
                    "H": height_mm,
                },
                "source_id": str(rep_dims.get("source_id") or ""),
                "protected": bool(decision.get("protected")) and required,
                "droppable": droppable,
                "budget_trial": bool(decision.get("budget_trial")),
                "solver_trial": bool(decision.get("solver_trial")),
                "trial_optional": bool(decision.get("trial_optional")),
                "allowed_rotations": deepcopy(
                    allowed_rotations.get(
                        object_id,
                        allowed_rotations.get(base_object_id, [0, 90, 180, 270]),
                    )
                ),
                "front": deepcopy(facing_row.get("front")),
            }
            if required:
                required_ids.append(object_id)
            else:
                optional_ids.append(object_id)
            if droppable:
                decision_droppable_ids.append(object_id)

    members = _expanded_members(base_members, expanded_ids_by_base)
    member_set = set(members)

    support_edges: list[dict[str, Any]] = []
    semantic_rows = (
        rules.get("semantic_placements")
        if isinstance(rules.get("semantic_placements"), list)
        else []
    )
    for row in semantic_rows:
        if not isinstance(row, dict):
            continue
        base_object_id = _placement_object_id(row)
        relative_to = row.get("relative_to")
        if not isinstance(base_object_id, str):
            continue
        object_ids = expanded_ids_by_base.get(base_object_id, [])
        relative_id = _primary_expanded_id(relative_to, expanded_ids_by_base)
        if not object_ids or relative_id is None:
            continue
        side_options = _string_list(row.get("side_options"))
        for index, object_id in enumerate(object_ids):
            if object_id == relative_id:
                continue
            support_edges.append(
                {
                    "object_id": object_id,
                    "relative_to": relative_id,
                    "kind": str(row.get("kind") or "anchor_side"),
                    "side_options": _side_options_for_instance(
                        side_options,
                        support_role=str(row.get("support_role") or ""),
                        band_intent=str(row.get("band_intent") or ""),
                        instance_index=index,
                        instance_count=len(object_ids),
                    ),
                    "gap_min_mm": int(row.get("gap_min") or 0),
                    "gap_max_mm": int(row.get("gap_max") or 0),
                    "proximity": str(row.get("proximity") or "balanced"),
                    "selection": str(row.get("selection") or "best_fit"),
                    "support_role": str(row.get("support_role") or ""),
                    "band_intent": str(row.get("band_intent") or ""),
                    "orientation": str(row.get("orientation") or ""),
                }
            )

    protected_ids = _string_list(anchor_first_policy.get("protected_ids"))
    droppable_ids = _stable_unique(
        [
            *_string_list(anchor_first_policy.get("droppable_ids")),
            *decision_droppable_ids,
        ]
    )
    dominant_anchor_id = _clean_str(anchor_first_policy.get("dominant_anchor_id"))
    if dominant_anchor_id is None:
        dominant_anchor_id = next(iter(cluster.get("anchors") or []), None)
    dominant_anchor_id = _primary_expanded_id(dominant_anchor_id, expanded_ids_by_base)

    placement_order = _expand_id_list(
        _string_list(anchor_first_policy.get("placement_order")) or list(base_members),
        expanded_ids_by_base,
    )
    anchor_candidates = _string_list(
        anchor_first_policy.get("dominant_anchor_candidates")
    ) or _string_list(rules.get("dominant_anchor_candidates"))
    anchors = _expand_id_list(
        _string_list(cluster.get("anchors")),
        expanded_ids_by_base,
        primary_only=True,
    )

    return {
        "cluster_id": cluster_id,
        "members": list(members),
        "anchors": anchors,
        "dominant_anchor_id": dominant_anchor_id,
        "dominant_anchor_candidates": _expand_id_list(
            anchor_candidates,
            expanded_ids_by_base,
            primary_only=True,
        ),
        "placement_order": [item for item in placement_order if item in member_set],
        "support_edges": support_edges,
        "protected_ids": [
            item
            for item in _expand_id_list(protected_ids, expanded_ids_by_base)
            if item in member_set
        ],
        "droppable_ids": [
            item
            for item in _expand_id_list(droppable_ids, expanded_ids_by_base)
            if item in member_set
        ],
        "degradation_ladder": _expand_degradation_ladder(
            _string_list(rules.get("degradation_ladder")),
            expanded_ids_by_base,
        ),
        "zone_claims": deepcopy(rules.get("zone_claims") or {}),
        "access_requirements": _expand_access_requirements(
            rules,
            expanded_ids_by_base,
        ),
        "required_object_ids": _stable_unique(required_ids),
        "optional_object_ids": _stable_unique(optional_ids),
        "object_specs_by_id": object_specs_by_id,
    }


def _remove_key_recursive(value: Any, key: str) -> None:
    if isinstance(value, dict):
        if key in value:
            value.pop(key, None)
        for child in list(value.values()):
            _remove_key_recursive(child, key)
    elif isinstance(value, list):
        for item in value:
            _remove_key_recursive(item, key)


def _unwrap_payload(payload: dict[str, Any], *, key: str) -> dict[str, Any]:
    if isinstance(payload.get(key), list):
        return payload
    parsed = payload.get("parsed")
    if isinstance(parsed, dict) and isinstance(parsed.get(key), list):
        return parsed
    raw = payload.get("raw")
    if isinstance(raw, dict) and isinstance(raw.get(key), list):
        return raw
    return payload


def _decision_type_id(decision: dict[str, Any]) -> str | None:
    obj_type = decision.get("object_type") or decision.get("category")
    if not isinstance(obj_type, str):
        return None
    obj_type = obj_type.strip()
    return obj_type if obj_type else None


def _decision_quantity(decision: dict[str, Any]) -> int:
    quantity = decision.get("quantity")
    if quantity is None:
        return 1
    if isinstance(quantity, bool):
        return 0
    try:
        return max(0, int(quantity))
    except (TypeError, ValueError):
        return 0


def _decision_min_keep(decision: dict[str, Any]) -> int:
    value = decision.get("min_keep")
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _expanded_object_id(base_object_id: str, quantity_index: int) -> str:
    if quantity_index <= 1:
        return base_object_id
    return f"{base_object_id}_{quantity_index}"


def _expanded_members(
    base_members: list[str],
    expanded_ids_by_base: dict[str, list[str]],
) -> list[str]:
    out: list[str] = []
    for member in base_members:
        out.extend(expanded_ids_by_base.get(member, []))
    for expanded_ids in expanded_ids_by_base.values():
        for object_id in expanded_ids:
            if object_id not in out:
                out.append(object_id)
    return out


def _expand_id_list(
    values: list[str],
    expanded_ids_by_base: dict[str, list[str]],
    *,
    primary_only: bool = False,
) -> list[str]:
    out: list[str] = []
    all_expanded_ids = {
        object_id
        for expanded_ids in expanded_ids_by_base.values()
        for object_id in expanded_ids
    }
    for value in values:
        expanded_ids = expanded_ids_by_base.get(value)
        if expanded_ids:
            out.extend(expanded_ids[:1] if primary_only else expanded_ids)
        elif value in all_expanded_ids:
            out.append(value)
    return _stable_unique(out)


def _expand_degradation_ladder(
    values: list[str],
    expanded_ids_by_base: dict[str, list[str]],
) -> list[str]:
    out: list[str] = []
    for value in values:
        if value.startswith("drop_"):
            object_type = _drop_action_object_type(value, expanded_ids_by_base)
            expanded_ids = _expand_id_list([object_type], expanded_ids_by_base)
            if expanded_ids:
                out.extend(f"drop_{object_id}" for object_id in expanded_ids)
                continue
        out.append(value)
    return _stable_unique(out)


def _drop_action_object_type(
    action: str,
    expanded_ids_by_base: dict[str, list[str]],
) -> str:
    object_type = action.removeprefix("drop_")
    for suffix in ("_first", "_last"):
        if object_type.endswith(suffix):
            object_type = object_type[: -len(suffix)]
    if object_type in expanded_ids_by_base:
        return object_type
    for base_object_id in expanded_ids_by_base:
        if base_object_id.endswith(f"_{object_type}") or object_type.endswith(
            f"_{base_object_id}"
        ):
            return base_object_id
    return object_type


def _primary_expanded_id(
    value: Any,
    expanded_ids_by_base: dict[str, list[str]],
) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    expanded_ids = expanded_ids_by_base.get(text)
    if expanded_ids:
        return expanded_ids[0]
    all_expanded_ids = {
        object_id for ids in expanded_ids_by_base.values() for object_id in ids
    }
    return text if text in all_expanded_ids else None


def _placement_object_id(row: dict[str, Any]) -> str | None:
    value = row.get("id") or row.get("object_id") or row.get("target_id")
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _expand_access_requirements(
    rules: dict[str, Any],
    expanded_ids_by_base: dict[str, list[str]],
) -> list[dict[str, Any]]:
    rows = (
        rules.get("access_requirements")
        if isinstance(rules.get("access_requirements"), list)
        else []
    )
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        base_object_id = _placement_object_id(row)
        if base_object_id is None:
            continue
        for object_id in expanded_ids_by_base.get(base_object_id, []):
            expanded_row = deepcopy(row)
            replaced = False
            for key in ("id", "object_id", "target_id"):
                if expanded_row.get(key) == base_object_id:
                    expanded_row[key] = object_id
                    replaced = True
            if not replaced:
                expanded_row["id"] = object_id
            out.append(expanded_row)
    return out


def _side_options_for_instance(
    side_options: list[str],
    *,
    support_role: str,
    band_intent: str,
    instance_index: int,
    instance_count: int,
) -> list[str]:
    if instance_count <= 1:
        return list(side_options)
    role_tokens = {
        support_role.strip().lower(),
        band_intent.strip().lower(),
    }
    if (
        role_tokens & {"side_support", "beside_base", "side_table"}
        and "head_left" in side_options
        and "head_right" in side_options
    ):
        return ["left" if instance_index % 2 == 0 else "right"]
    side_pairs = (
        ("head_left", "head_right"),
        ("front_left", "front_right"),
        ("left", "right"),
    )
    for left_option, right_option in side_pairs:
        if left_option not in side_options or right_option not in side_options:
            continue
        selected = left_option if instance_index % 2 == 0 else right_option
        return [selected]
    return list(side_options)


def _decision_for_member(
    cluster_id: str,
    member: str,
    decision_by_cluster_and_type: dict[tuple[str, str], dict[str, Any]],
    decision_by_type: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    return decision_by_cluster_and_type.get(
        (cluster_id, member)
    ) or decision_by_type.get(member)


def _constraint_subjects(constraint: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for key in ("a", "b", "id"):
        value = constraint.get(key)
        if isinstance(value, str) and value:
            out.add(value)
    return out


def _filter_cluster_rules(
    rules: dict[str, Any],
    kept_ids: set[str],
    anchors: set[str] | None = None,
) -> dict[str, Any]:
    out = deepcopy(rules)
    anchor_ids = {anchor for anchor in (anchors or set()) if anchor in kept_ids}

    allowed_rotations = out.get("allowed_rotations")
    if isinstance(allowed_rotations, dict):
        out["allowed_rotations"] = {
            key: value for key, value in allowed_rotations.items() if key in kept_ids
        }

    facing = out.get("facing")
    if isinstance(facing, dict):
        out["facing"] = {key: value for key, value in facing.items() if key in kept_ids}

    access_requirements = out.get("access_requirements")
    if isinstance(access_requirements, list):
        out["access_requirements"] = [
            deepcopy(item)
            for item in access_requirements
            if isinstance(item, dict)
            and isinstance(item.get("id"), str)
            and item.get("id") in kept_ids
        ]

    semantic_placements = out.get("semantic_placements")
    if isinstance(semantic_placements, list):
        out["semantic_placements"] = [
            deepcopy(item)
            for item in semantic_placements
            if isinstance(item, dict)
            and isinstance(item.get("id"), str)
            and item.get("id") in kept_ids
            and (
                not isinstance(item.get("relative_to"), str)
                or item.get("relative_to") in kept_ids
            )
        ]

    dominant_candidates = out.get("dominant_anchor_candidates")
    if isinstance(dominant_candidates, list):
        out["dominant_anchor_candidates"] = [
            item
            for item in dominant_candidates
            if isinstance(item, str) and item in kept_ids
        ]

    anchor_first_policy = out.get("anchor_first_policy")
    if isinstance(anchor_first_policy, dict):
        out["anchor_first_policy"] = _filter_anchor_first_policy(
            anchor_first_policy,
            kept_ids=kept_ids,
            anchor_ids=anchor_ids,
        )

    return out


def _filter_anchor_first_policy(
    policy: dict[str, Any],
    *,
    kept_ids: set[str],
    anchor_ids: set[str],
) -> dict[str, Any]:
    out = deepcopy(policy)

    dominant_anchor_id = out.get("dominant_anchor_id")
    if not isinstance(dominant_anchor_id, str) or dominant_anchor_id not in kept_ids:
        fallback_anchor = next(iter(anchor_ids), None)
        if fallback_anchor is not None:
            out["dominant_anchor_id"] = fallback_anchor
        else:
            out.pop("dominant_anchor_id", None)

    dominant_candidates = out.get("dominant_anchor_candidates")
    if isinstance(dominant_candidates, list):
        out["dominant_anchor_candidates"] = [
            item
            for item in dominant_candidates
            if isinstance(item, str) and item in kept_ids
        ]

    placement_order = out.get("placement_order")
    if isinstance(placement_order, list):
        out["placement_order"] = [
            item
            for item in placement_order
            if isinstance(item, str) and item in kept_ids
        ]

    support_chain = out.get("support_chain")
    if isinstance(support_chain, list):
        kept_chain: list[dict[str, Any]] = []
        for row in support_chain:
            if not isinstance(row, dict):
                continue
            object_id = row.get("object_id")
            relative_to = row.get("relative_to")
            if not isinstance(object_id, str) or object_id not in kept_ids:
                continue
            if (
                isinstance(relative_to, str)
                and relative_to
                and relative_to not in kept_ids
            ):
                continue
            kept_chain.append(deepcopy(row))
        out["support_chain"] = kept_chain

    for key in ("protected_ids", "droppable_ids"):
        values = out.get(key)
        if isinstance(values, list):
            out[key] = [
                item for item in values if isinstance(item, str) and item in kept_ids
            ]

    return out


def _cluster_rules(cluster: dict[str, Any]) -> dict[str, Any]:
    rules = cluster.get("cluster_rules")
    return rules if isinstance(rules, dict) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


def _clean_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _stable_unique(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _merged_notes(
    cluster_payload: dict[str, Any], tier_payload: dict[str, Any]
) -> list[str]:
    notes: list[str] = []
    for source in (cluster_payload.get("notes"), tier_payload.get("notes")):
        if not isinstance(source, list):
            continue
        for item in source:
            text = str(item).strip()
            if text and text not in notes:
                notes.append(text)
    notes.append(
        "Merged into solver-ready object program for object-level anchor-first placement."
    )
    return notes
