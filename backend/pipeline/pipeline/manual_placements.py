from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from layout.semantic_roles import (
    is_bed_like,
    is_bedside_support_like,
    is_bench_like,
    is_lounge_anchor_like,
    is_seat_like,
    is_work_surface_like,
)

DEFAULT_MANUAL_COLOR_HEX = "#CFCAC2"
MANUAL_CLUSTER_ID = "__manual__"


def prepare_input_payload_for_manual_placements(
    input_payload: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    normalized = extract_manual_placements(input_payload)
    prepared = deepcopy(input_payload)
    if not normalized:
        return prepared, []

    constraints = prepared.get("constraints")
    if not isinstance(constraints, dict):
        constraints = {}
        prepared["constraints"] = constraints

    zones = constraints.get("no_go_zones")
    no_go_zones = list(zones) if isinstance(zones, list) else []
    existing_ids = {
        str(item.get("id"))
        for item in no_go_zones
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }

    for placement in normalized:
        obstacle = _manual_obstacle_from_placement(placement)
        obstacle_id = str(obstacle["id"])
        if obstacle_id in existing_ids:
            continue
        no_go_zones.append(obstacle)
        existing_ids.add(obstacle_id)

    constraints["no_go_zones"] = no_go_zones
    return prepared, normalized


def extract_manual_placements(input_payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw = input_payload.get("manual_placements")
    if not isinstance(raw, list):
        return []

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw, start=1):
        placement = _normalize_manual_placement(item, index=index)
        if placement is None:
            continue
        placement_id = str(placement["placement_id"])
        if placement_id in seen_ids:
            continue
        seen_ids.add(placement_id)
        normalized.append(placement)
    return normalized


def apply_manual_placements_to_tier_output(
    tier_output: dict[str, Any],
    manual_placements: list[dict[str, Any]],
    *,
    clusters_json: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not manual_placements:
        return deepcopy(tier_output)

    decisions = tier_output.get("decisions")
    if not isinstance(decisions, list):
        return deepcopy(tier_output)

    assigned_by_cluster_and_type: dict[tuple[str, str], int] = {}
    unassigned_by_type: dict[str, int] = {}
    cluster_ids_by_placement = _match_manual_placements_to_clusters(
        clusters_json,
        manual_placements,
    )
    for placement in manual_placements:
        placement_id = placement.get("placement_id")
        object_type = _normalize_object_type(placement.get("category"))
        if object_type is None:
            continue
        if isinstance(placement_id, str):
            cluster_id = cluster_ids_by_placement.get(placement_id)
        else:
            cluster_id = None
        if isinstance(cluster_id, str) and cluster_id:
            key = (cluster_id, object_type)
            assigned_by_cluster_and_type[key] = (
                assigned_by_cluster_and_type.get(key, 0) + 1
            )
            continue
        unassigned_by_type[object_type] = unassigned_by_type.get(object_type, 0) + 1

    if not assigned_by_cluster_and_type and not unassigned_by_type:
        return deepcopy(tier_output)

    out = deepcopy(tier_output)
    next_decisions: list[dict[str, Any]] = []
    adjusted_counts: dict[str, int] = {}
    for decision in decisions:
        if not isinstance(decision, dict):
            next_decisions.append(decision)
            continue
        next_decision = dict(decision)
        object_type = _normalize_object_type(
            next_decision.get("object_type") or next_decision.get("category")
        )
        cluster_id = next_decision.get("cluster_id")
        quantity = next_decision.get("quantity")
        if object_type is None or not isinstance(quantity, int) or quantity <= 0:
            next_decisions.append(next_decision)
            continue

        reduction = 0
        if isinstance(cluster_id, str):
            reduction = min(
                quantity,
                assigned_by_cluster_and_type.get((cluster_id, object_type), 0),
            )
            if reduction > 0:
                assigned_by_cluster_and_type[(cluster_id, object_type)] -= reduction
        if reduction <= 0:
            reduction = min(quantity, unassigned_by_type.get(object_type, 0))
            if reduction > 0:
                unassigned_by_type[object_type] -= reduction
        if reduction <= 0:
            next_decisions.append(next_decision)
            continue

        next_decision["quantity"] = quantity - reduction
        adjusted_counts[object_type] = adjusted_counts.get(object_type, 0) + reduction

        rationale = next_decision.get("rationale")
        if isinstance(rationale, str) and rationale.strip():
            next_decision["rationale"] = (
                f"{rationale.strip()} Reduced by {reduction} because the same type was manually "
                "placed in the editor before generation."
            )
        next_decisions.append(next_decision)

    out["decisions"] = next_decisions
    if adjusted_counts:
        notes = out.get("global_notes")
        global_notes = list(notes) if isinstance(notes, list) else []
        summary = ", ".join(
            f"{object_type} x{count}"
            for object_type, count in sorted(adjusted_counts.items())
        )
        note = (
            "Manual placements reserved protected footprint and reduced generated counts for: "
            f"{summary}."
        )
        if note not in global_notes:
            global_notes.append(note)
        out["global_notes"] = global_notes
    return out


def build_manual_placements_guidance_text(
    manual_placements: list[dict[str, Any]],
) -> str:
    if not manual_placements:
        return ""

    lines = [
        "EXISTING MANUAL OBJECTS FROM EDITOR:",
        (
            "Treat these as already-present user-placed furniture. "
            "Do not re-count the same object types, and use them as existing seeds "
            "when inferring cluster semantics if relevant."
        ),
    ]
    for placement in manual_placements:
        category = _normalize_object_type(placement.get("category"))
        position = placement.get("position_mm")
        rotation_deg = float(placement.get("rotation_deg") or 0.0)
        constraint_mode = str(placement.get("constraint_mode") or "pinned")
        if category is None or not isinstance(position, dict):
            continue
        x = int(round(float(position.get("x") or 0.0)))
        y = int(round(float(position.get("y") or 0.0)))
        lines.append(
            f"- {category} at approx ({x}mm, {y}mm), rot {int(round(rotation_deg)) % 360}deg, {constraint_mode}"
        )
    return "\n".join(lines)


def append_manual_placements_guidance(
    base_text: str,
    manual_placements: list[dict[str, Any]],
) -> str:
    manual_text = build_manual_placements_guidance_text(manual_placements)
    if not manual_text:
        return base_text
    if not base_text.strip():
        return manual_text
    return f"{base_text.rstrip()}\n\n{manual_text}"


def augment_cluster_forge_with_manual_placements(
    cluster_output: dict[str, Any],
    manual_placements: list[dict[str, Any]],
) -> dict[str, Any]:
    out = deepcopy(cluster_output)
    if not manual_placements:
        return out

    clusters = out.get("clusters")
    if not isinstance(clusters, list):
        return out

    cluster_by_id = {
        str(cluster.get("cluster_id")): cluster
        for cluster in clusters
        if isinstance(cluster, dict) and isinstance(cluster.get("cluster_id"), str)
    }
    cluster_ids_by_placement = _match_manual_placements_to_clusters(
        out, manual_placements
    )
    matched_count = 0
    unmatched_descriptions: list[str] = []

    for placement in manual_placements:
        placement_id = placement.get("placement_id")
        category = _normalize_object_type(placement.get("category"))
        if not isinstance(placement_id, str) or category is None:
            continue
        cluster_id = cluster_ids_by_placement.get(placement_id)
        if not isinstance(cluster_id, str):
            unmatched_descriptions.append(
                f"{placement_id} ({category})",
            )
            continue
        cluster = cluster_by_id.get(cluster_id)
        if not isinstance(cluster, dict):
            continue
        rules = cluster.get("cluster_rules")
        if not isinstance(rules, dict):
            rules = {}
            cluster["cluster_rules"] = rules
        manual_existing_objects = rules.get("manual_existing_objects")
        existing_rows = (
            list(manual_existing_objects)
            if isinstance(manual_existing_objects, list)
            else []
        )
        if any(
            isinstance(item, dict) and item.get("placement_id") == placement_id
            for item in existing_rows
        ):
            continue
        existing_rows.append(
            {
                "id": category,
                "placement_id": placement_id,
                "asset_id": placement.get("asset_id"),
                "constraint_mode": placement.get("constraint_mode"),
                "position_mm": deepcopy(placement.get("position_mm")),
                "rotation_deg": placement.get("rotation_deg"),
                "footprint_mm": deepcopy(placement.get("footprint_mm")),
                "height_mm": placement.get("height_mm"),
            }
        )
        rules["manual_existing_objects"] = existing_rows
        cluster["notes"] = _append_note(
            cluster.get("notes"),
            (
                f"Manual editor object {placement_id} ({category}) is already present "
                "and should be treated as an existing seed when reasoning about this cluster."
            ),
        )
        matched_count += 1

    if matched_count > 0:
        out["notes"] = _append_note(
            out.get("notes"),
            f"Matched {matched_count} manual editor placement(s) to cluster semantics.",
        )
    if unmatched_descriptions:
        out["notes"] = _append_note(
            out.get("notes"),
            (
                "Manual editor placements preserved as protected objects without a direct "
                f"cluster semantic match: {', '.join(sorted(unmatched_descriptions))}."
            ),
        )
    return out


def merge_manual_placements_into_absolute_layout(
    absolute_layout: dict[str, Any],
    manual_placements: list[dict[str, Any]],
) -> dict[str, Any]:
    out = deepcopy(absolute_layout)
    if not manual_placements:
        return out

    objects = out.get("objects")
    out["objects"] = _merge_object_rows(
        existing_rows=list(objects) if isinstance(objects, list) else [],
        incoming_rows=[
            _manual_absolute_object_from_placement(placement)
            for placement in manual_placements
        ],
        object_key="object_id",
    )
    out["notes"] = _append_note(
        out.get("notes"),
        f"Preserved {len(manual_placements)} manual placement(s) from the editor.",
    )
    return out


def merge_manual_placements_into_styled_output(
    stylist_output: dict[str, Any],
    manual_placements: list[dict[str, Any]],
) -> dict[str, Any]:
    out = deepcopy(stylist_output)
    if not manual_placements:
        return out

    objects = out.get("objects")
    out["objects"] = _merge_object_rows(
        existing_rows=list(objects) if isinstance(objects, list) else [],
        incoming_rows=[
            _manual_styled_object_from_placement(placement)
            for placement in manual_placements
        ],
        object_key="instance_id",
    )
    out["notes"] = _append_note(
        out.get("notes"),
        f"Rendered {len(manual_placements)} protected manual placement(s).",
    )
    return out


def merge_manual_placements_into_layout_variants(
    layout_variants: dict[str, Any],
    manual_placements: list[dict[str, Any]],
) -> dict[str, Any]:
    out = deepcopy(layout_variants)
    if not manual_placements:
        return out

    variants = out.get("variants")
    if not isinstance(variants, list):
        return out

    next_variants: list[dict[str, Any]] = []
    for variant in variants:
        if not isinstance(variant, dict):
            continue
        next_variant = dict(variant)
        absolute_layout = next_variant.get("absolute_layout")
        if isinstance(absolute_layout, dict):
            next_variant["absolute_layout"] = (
                merge_manual_placements_into_absolute_layout(
                    absolute_layout,
                    manual_placements,
                )
            )
        styled_result = next_variant.get("styled_result")
        if isinstance(styled_result, dict):
            next_variant["styled_result"] = merge_manual_placements_into_styled_output(
                styled_result,
                manual_placements,
            )
        next_variants.append(next_variant)

    out["variants"] = next_variants
    return out


def _normalize_manual_placement(
    value: Any,
    *,
    index: int,
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None

    category = _normalize_object_type(value.get("category") or value.get("object_type"))
    if category is None:
        return None

    position = _normalize_point(value.get("position_mm"))
    if position is None:
        return None

    footprint = _normalize_footprint(value.get("footprint_mm"))
    if footprint is None:
        return None

    placement_id = str(
        value.get("placement_id") or value.get("id") or f"manual_{index}"
    )
    asset_id = str(value.get("asset_id") or category or placement_id)
    rotation_deg = float(value.get("rotation_deg") or 0.0)
    height_mm = float(value.get("height_mm") or 0.0)
    color_hex = str(value.get("color_hex") or DEFAULT_MANUAL_COLOR_HEX)
    constraint_mode = str(value.get("constraint_mode") or "pinned").strip().lower()
    if constraint_mode != "locked":
        constraint_mode = "pinned"

    return {
        "placement_id": placement_id,
        "asset_id": asset_id,
        "category": category,
        "position_mm": position,
        "rotation_deg": rotation_deg,
        "footprint_mm": footprint,
        "height_mm": height_mm,
        "color_hex": color_hex,
        "anchor": "center",
        "source": "manual",
        "constraint_mode": constraint_mode,
    }


def _normalize_point(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    x = value.get("x")
    y = value.get("y")
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return None
    return {"x": float(x), "y": float(y)}


def _normalize_footprint(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    width = value.get("w")
    depth = value.get("d")
    if not isinstance(width, (int, float)) or not isinstance(depth, (int, float)):
        return None
    if width <= 0 or depth <= 0:
        return None
    return {"w": float(width), "d": float(depth)}


def _normalize_object_type(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text if text else None


def _match_manual_placements_to_clusters(
    clusters_json: dict[str, Any] | None,
    manual_placements: list[dict[str, Any]],
) -> dict[str, str]:
    if not isinstance(clusters_json, dict):
        return {}
    clusters = clusters_json.get("clusters")
    if not isinstance(clusters, list):
        return {}

    cluster_maps = [
        cluster
        for cluster in clusters
        if isinstance(cluster, dict) and isinstance(cluster.get("cluster_id"), str)
    ]
    matches: dict[str, str] = {}
    for placement in manual_placements:
        placement_id = placement.get("placement_id")
        category = _normalize_object_type(placement.get("category"))
        if not isinstance(placement_id, str) or category is None:
            continue
        cluster_id = _match_manual_object_to_cluster_id(cluster_maps, category)
        if cluster_id is not None:
            matches[placement_id] = cluster_id
    return matches


def _match_manual_object_to_cluster_id(
    clusters: Sequence[Mapping[str, Any]],
    object_type: str,
) -> str | None:
    best_cluster_id: str | None = None
    best_score = -10_000
    preferred_tags = _preferred_cluster_tags_for_object(object_type)

    for cluster in clusters:
        cluster_id = cluster.get("cluster_id")
        tag = cluster.get("tag")
        if not isinstance(cluster_id, str) or not isinstance(tag, str):
            continue
        members = _coerce_string_list(cluster.get("members"))
        anchors = _coerce_string_list(cluster.get("anchors"))
        score = 0
        if object_type in members:
            score += 1_000
        if tag in preferred_tags:
            score += 100 - (preferred_tags.index(tag) * 10)
        score += _compatibility_score_for_members(
            object_type=object_type,
            members=members,
            anchors=anchors,
            tag=tag,
        )
        if score > best_score:
            best_score = score
            best_cluster_id = cluster_id

    if best_score <= 0:
        return None
    return best_cluster_id


def _compatibility_score_for_members(
    *,
    object_type: str,
    members: Sequence[str],
    anchors: Sequence[str],
    tag: str,
) -> int:
    score = 0
    all_known = tuple(dict.fromkeys([*members, *anchors]))
    if is_bed_like(object_type):
        if tag == "sleep":
            score += 80
        if any(is_bedside_support_like(member) for member in all_known):
            score += 40
        return score
    if is_bedside_support_like(object_type) or is_bench_like(object_type):
        if tag == "sleep":
            score += 80
        if any(is_bed_like(member) for member in all_known):
            score += 50
        return score
    if _is_storage_like(object_type):
        if tag == "storage":
            score += 110
        return score
    if _is_task_chair_like(object_type):
        if tag == "work":
            score += 100
        if any(is_work_surface_like(member) for member in all_known):
            score += 60
        return score
    if is_work_surface_like(object_type):
        if _is_dining_surface_like(object_type):
            if tag == "dining":
                score += 100
        elif tag == "work":
            score += 110
        if any(_is_task_chair_like(member) for member in all_known):
            score += 30
        return score
    if _is_lounge_support_like(object_type):
        if tag == "living":
            score += 90
        if any(is_lounge_anchor_like(member) for member in all_known):
            score += 60
        return score
    if is_lounge_anchor_like(object_type):
        if tag == "living":
            score += 110
        if any(_is_lounge_support_like(member) for member in all_known):
            score += 30
        return score
    if is_seat_like(object_type):
        if tag == "living" and any(
            is_lounge_anchor_like(member) for member in all_known
        ):
            score += 90
        if tag == "dining" and any(
            _is_dining_surface_like(member) for member in all_known
        ):
            score += 90
        if tag == "work" and any(is_work_surface_like(member) for member in all_known):
            score += 80
        return score
    if tag == "misc":
        score += 10
    return score


def _preferred_cluster_tags_for_object(object_type: str) -> list[str]:
    if (
        is_bed_like(object_type)
        or is_bedside_support_like(object_type)
        or is_bench_like(object_type)
    ):
        return ["sleep"]
    if _is_storage_like(object_type):
        return ["storage"]
    if _is_dining_surface_like(object_type):
        return ["dining"]
    if _is_task_chair_like(object_type):
        return ["work", "dining", "living"]
    if is_work_surface_like(object_type):
        return ["work", "dining"]
    if _is_lounge_support_like(object_type) or is_lounge_anchor_like(object_type):
        return ["living"]
    if is_seat_like(object_type):
        return ["living", "dining", "work"]
    return ["misc"]


def _coerce_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def _is_task_chair_like(object_type: str) -> bool:
    key = object_type.lower()
    return any(
        pattern in key for pattern in ("office_chair", "desk_chair", "task_chair")
    )


def _is_storage_like(object_type: str) -> bool:
    key = object_type.lower()
    return any(
        pattern in key
        for pattern in (
            "wardrobe",
            "dresser",
            "storage",
            "cabinet",
            "closet",
            "bookshelf",
            "shoe_rack",
            "console",
            "sideboard",
            "buffet",
        )
    )


def _is_dining_surface_like(object_type: str) -> bool:
    key = object_type.lower()
    return any(
        pattern in key for pattern in ("dining_table", "bar_table", "breakfast_table")
    )


def _is_lounge_support_like(object_type: str) -> bool:
    key = object_type.lower()
    return "side_table" in key or "floor_lamp" in key


def _manual_obstacle_from_placement(placement: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"manual_{placement['placement_id']}",
        "type": "no_go",
        "hard": True,
        "polygon_ccw": _placement_polygon(placement),
    }


def _manual_absolute_object_from_placement(placement: dict[str, Any]) -> dict[str, Any]:
    polygon = _placement_polygon(placement)
    bbox = _bbox_from_polygon(polygon)
    rotation_ccw = int(round(float(placement.get("rotation_deg") or 0.0))) % 360
    return {
        "object_id": str(placement["placement_id"]),
        "instance_id": str(placement["placement_id"]),
        "object_type": str(placement["category"]),
        "cluster_id": MANUAL_CLUSTER_ID,
        "polygon_ccw": polygon,
        "bbox": bbox,
        "rotation_ccw": rotation_ccw,
        "manual_input": deepcopy(placement),
        "source": "manual",
    }


def _manual_styled_object_from_placement(placement: dict[str, Any]) -> dict[str, Any]:
    polygon = _placement_polygon(placement)
    bbox = _bbox_from_polygon(polygon)
    rotation_ccw = int(round(float(placement.get("rotation_deg") or 0.0))) % 360
    return {
        "instance_id": str(placement["placement_id"]),
        "object_type": str(placement["category"]),
        "source": "existing",
        "cluster_id": None,
        "polygon_ccw": polygon,
        "bbox": bbox,
        "color_hex": str(placement.get("color_hex") or DEFAULT_MANUAL_COLOR_HEX),
        "material": None,
        "place_on": None,
        "rotation_ccw": rotation_ccw,
        "manual_input": deepcopy(placement),
    }


def _placement_polygon(placement: dict[str, Any]) -> list[dict[str, int]]:
    position = placement["position_mm"]
    footprint = placement["footprint_mm"]
    center_x = float(position["x"])
    center_y = float(position["y"])
    half_w = float(footprint["w"]) / 2.0
    half_d = float(footprint["d"]) / 2.0
    rotation_rad = math.radians(float(placement.get("rotation_deg") or 0.0))
    cos_theta = math.cos(rotation_rad)
    sin_theta = math.sin(rotation_rad)

    corners = [
        (-half_w, -half_d),
        (half_w, -half_d),
        (half_w, half_d),
        (-half_w, half_d),
    ]
    polygon: list[dict[str, int]] = []
    for dx, dy in corners:
        polygon.append(
            {
                "x": int(round(center_x + dx * cos_theta - dy * sin_theta)),
                "y": int(round(center_y + dx * sin_theta + dy * cos_theta)),
            }
        )
    return polygon


def _bbox_from_polygon(polygon: list[dict[str, int]]) -> dict[str, int]:
    xs = [int(point["x"]) for point in polygon]
    ys = [int(point["y"]) for point in polygon]
    return {
        "min_x": min(xs),
        "min_y": min(ys),
        "max_x": max(xs),
        "max_y": max(ys),
    }


def _merge_object_rows(
    *,
    existing_rows: list[dict[str, Any]],
    incoming_rows: list[dict[str, Any]],
    object_key: str,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    index_by_id: dict[str, int] = {}
    for row in existing_rows:
        if not isinstance(row, dict):
            continue
        next_row = deepcopy(row)
        merged.append(next_row)
        identifier = next_row.get(object_key)
        if isinstance(identifier, str) and identifier:
            index_by_id[identifier] = len(merged) - 1

    for row in incoming_rows:
        identifier = row.get(object_key)
        if not isinstance(identifier, str) or not identifier:
            merged.append(deepcopy(row))
            continue
        if identifier in index_by_id:
            idx = index_by_id[identifier]
            current = dict(merged[idx])
            current.update(deepcopy(row))
            if "color_hex" not in row and "color_hex" in merged[idx]:
                current["color_hex"] = merged[idx]["color_hex"]
            merged[idx] = current
            continue
        index_by_id[identifier] = len(merged)
        merged.append(deepcopy(row))
    return merged


def _append_note(value: Any, note: str) -> list[str]:
    notes = list(value) if isinstance(value, list) else []
    if note not in notes:
        notes.append(note)
    return notes
