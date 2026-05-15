from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from layout.orientation_contract import (
    effective_front_side as _effective_front_side_contract,
)
from layout.orientation_contract import (
    normalize_cardinal_side,
    side_to_vec,
)

try:
    from shapely.geometry import LineString, Polygon
    from shapely.ops import unary_union
except Exception:  # pragma: no cover - import guard only
    LineString = None  # type: ignore[assignment]
    Polygon = None  # type: ignore[assignment]
    unary_union = None  # type: ignore[assignment]


_CONTROLLED_REFILL_MAX_REFILLS_TOTAL = 2
_CONTROLLED_REFILL_MAX_CANDIDATES_PER_OBJECT = 6
_CONTROLLED_REFILL_MAX_SAFE_ZONES = 8
_CONTROLLED_REFILL_GRID_MM = 25
_CONTROLLED_REFILL_MAX_ITERATIONS = 1
_CONTROLLED_REFILL_MAX_ACCESSORY_FOOTPRINT_RATIO = 0.015
_CONTROLLED_REFILL_MIN_QUALITY_GAIN = 0.002
_CONTROLLED_REFILL_ZONE_DEPTH_MM = 900
_CONTROLLED_REFILL_CLUSTER_GAP_MM = 100

_ACCESSORY_DECOR_CATEGORIES = frozenset(
    {
        "accessory",
        "accessories",
        "decor",
        "decorative",
        "decoration",
        "soft_decor",
        "accessory_decor",
    }
)
_ACCESSORY_DECOR_TYPES = frozenset(
    {
        "accent_object",
        "art",
        "artwork",
        "cushion",
        "decor",
        "decor_object",
        "decorative_object",
        "pillow",
        "plant",
        "planter",
        "sculpture",
        "small_decor",
        "small_decor_object",
        "small_rug",
        "table_decor",
        "throw",
        "vase",
        "wall_art",
    }
)
_FUNCTIONAL_CORE_TYPES = frozenset(
    {
        "bathroom_core",
        "bed",
        "chair",
        "desk",
        "dining_chair",
        "dining_table",
        "dresser",
        "entry_storage",
        "fridge",
        "laundry_core",
        "office_chair",
        "sectional_sofa",
        "sink",
        "sofa",
        "stove",
        "tv_console",
        "wardrobe",
    }
)
_FUNCTIONAL_SUPPORT_TYPES = frozenset(
    {
        "armchair",
        "bench",
        "bookshelf",
        "coat_rack",
        "coffee_table",
        "floor_lamp",
        "media_shelf",
        "nightstand",
        "ottoman",
        "pantry",
        "pet_bed",
        "shoe_rack",
        "side_table",
        "storage_cabinet",
        "tall_cabinet",
    }
)


def _layout_objects(absolute_layout: Mapping[str, Any]) -> list[dict[str, Any]]:
    for key in ("objects", "object_placements", "placements"):
        value = absolute_layout.get(key)
        if isinstance(value, Sequence) and not isinstance(value, str):
            return [deepcopy(row) for row in value if isinstance(row, Mapping)]
    return []


def collect_dropped_inventory(
    *,
    previous_tier_output: dict[str, Any],
    next_tier_output: dict[str, Any],
    attempt: int,
) -> dict[str, list[dict[str, Any]]]:
    previous_lookup = _decision_lookup(previous_tier_output)
    next_lookup = _decision_lookup(next_tier_output)

    dropped_by_cluster: dict[str, list[dict[str, Any]]] = {}
    for key, previous_row in previous_lookup.items():
        cluster_id, object_type = key
        previous_quantity = max(0, int(previous_row.get("quantity") or 0))
        next_row = next_lookup.get(key)
        next_quantity = (
            max(0, int(next_row.get("quantity") or 0))
            if isinstance(next_row, dict)
            else 0
        )
        removed_count = max(0, previous_quantity - next_quantity)
        if removed_count <= 0:
            continue

        cluster_items = dropped_by_cluster.setdefault(cluster_id, [])
        for removed_index in range(removed_count):
            cluster_items.append(
                {
                    "cluster_id": cluster_id,
                    "object_type": object_type,
                    "category": str(previous_row.get("category") or object_type),
                    "size_tier": str(previous_row.get("size_tier") or "M"),
                    "priority": str(previous_row.get("priority") or "secondary"),
                    "rep_dims_m": deepcopy(previous_row.get("rep_dims_m") or {}),
                    "rationale": str(previous_row.get("rationale") or ""),
                    "drop_attempt": int(attempt),
                    "drop_sequence": removed_index + 1,
                }
            )
    return dropped_by_cluster


def collect_seed_omitted_inventory(
    *,
    tier_output: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    omitted_by_cluster: dict[str, list[dict[str, Any]]] = {}
    decisions = tier_output.get("decisions")
    if not isinstance(decisions, list):
        return omitted_by_cluster

    for row in decisions:
        if not isinstance(row, dict):
            continue
        cluster_id = str(row.get("cluster_id") or "").strip()
        object_type = str(row.get("object_type") or row.get("category") or "").strip()
        quantity = max(0, int(row.get("quantity") or 0))
        if not cluster_id or not object_type or quantity > 0:
            continue
        omitted_by_cluster.setdefault(cluster_id, []).append(
            {
                "cluster_id": cluster_id,
                "object_type": object_type,
                "category": str(row.get("category") or object_type),
                "size_tier": str(row.get("size_tier") or "M"),
                "priority": str(row.get("priority") or "secondary"),
                "rep_dims_m": deepcopy(row.get("rep_dims_m") or {}),
                "rationale": str(row.get("rationale") or ""),
                "drop_reason": "seed_omission",
                "drop_attempt": 0,
                "drop_sequence": 1,
            }
        )
    return omitted_by_cluster


def merge_dropped_inventory(
    base_inventory: dict[str, list[dict[str, Any]]] | None,
    additions: dict[str, list[dict[str, Any]]] | None,
) -> dict[str, list[dict[str, Any]]]:
    merged: dict[str, list[dict[str, Any]]] = {
        str(cluster_id): [deepcopy(item) for item in items if isinstance(item, dict)]
        for cluster_id, items in (base_inventory or {}).items()
        if isinstance(cluster_id, str) and isinstance(items, list)
    }
    for cluster_id, items in (additions or {}).items():
        if not isinstance(cluster_id, str) or not isinstance(items, list):
            continue
        target = merged.setdefault(cluster_id, [])
        target.extend(deepcopy(item) for item in items if isinstance(item, dict))
    return merged


def dropped_inventory_payload(
    dropped_inventory_by_cluster: dict[str, list[dict[str, Any]]] | None,
) -> dict[str, Any]:
    clusters: list[dict[str, Any]] = []
    total_items = 0
    for cluster_id in sorted((dropped_inventory_by_cluster or {}).keys()):
        items = dropped_inventory_by_cluster.get(cluster_id)
        if not isinstance(items, list) or not items:
            continue
        clean_items = [deepcopy(item) for item in items if isinstance(item, dict)]
        if not clean_items:
            continue
        total_items += len(clean_items)
        clusters.append({"cluster_id": cluster_id, "items": clean_items})
    return {
        "status": "OK",
        "total_dropped_items": total_items,
        "clusters": clusters,
    }


def controlled_accessory_refill(
    *,
    room_output: dict[str, Any],
    absolute_layout: dict[str, Any],
    cluster_output: dict[str, Any],
    dropped_inventory_by_cluster: dict[str, list[dict[str, Any]]] | None,
    refined_layout_solution: dict[str, Any] | None = None,
    refill_policy: dict[str, Any] | None = None,
    grid_mm: int = _CONTROLLED_REFILL_GRID_MM,
) -> tuple[dict[str, Any], dict[str, Any]]:
    layout = deepcopy(absolute_layout)
    layout["objects"] = _layout_objects(layout)

    inventory = {
        str(cluster_id): [deepcopy(item) for item in items if isinstance(item, dict)]
        for cluster_id, items in (dropped_inventory_by_cluster or {}).items()
        if isinstance(cluster_id, str) and isinstance(items, list)
    }
    if not inventory:
        return layout, _accessory_refill_summary(
            original_inventory=inventory,
            eligible_inventory={},
            inserted_by_cluster={},
            remaining_by_cluster={},
            rejected_by_cluster={},
            safe_zone_count=0,
            quality_before=_quality_from_refined_solution(refined_layout_solution),
            quality_after=_quality_from_refined_solution(refined_layout_solution),
            notes=[],
        )

    policy = _normalize_refill_policy(refill_policy, grid_mm=grid_mm)
    if int(policy["max_refills_total"]) <= 0:
        notes = ["Controlled Accessory Refill skipped because refill cap is zero."]
        disabled_reason = str(policy.get("disabled_reason") or "").strip()
        missing_types = policy.get("missing_request_object_types")
        if disabled_reason:
            notes.append(f"Disabled reason: {disabled_reason}.")
        if isinstance(missing_types, list) and missing_types:
            notes.append(
                "Missing request object types before refill: "
                f"{', '.join(str(item) for item in missing_types)}."
            )
        return layout, _accessory_refill_summary(
            original_inventory=inventory,
            eligible_inventory={},
            inserted_by_cluster={},
            remaining_by_cluster=inventory,
            rejected_by_cluster={},
            safe_zone_count=0,
            quality_before=_quality_from_refined_solution(refined_layout_solution),
            quality_after=_quality_from_refined_solution(refined_layout_solution),
            notes=notes,
        )

    room_polygon = _room_polygon(room_output)
    if room_polygon is None or Polygon is None or unary_union is None:
        rejected = {
            cluster_id: [
                _rejected_refill_item(item, "room_polygon_unavailable")
                for item in items
            ]
            for cluster_id, items in inventory.items()
        }
        return layout, _accessory_refill_summary(
            original_inventory=inventory,
            eligible_inventory={},
            inserted_by_cluster={},
            remaining_by_cluster=inventory,
            rejected_by_cluster=rejected,
            safe_zone_count=0,
            quality_before=_quality_from_refined_solution(refined_layout_solution),
            quality_after=_quality_from_refined_solution(refined_layout_solution),
            notes=[
                "Controlled Accessory Refill skipped because room geometry is unavailable for object-level layout."
            ],
        )

    cluster_lookup = _cluster_lookup(cluster_output)
    objects = [
        deepcopy(row) for row in layout.get("objects") or [] if isinstance(row, dict)
    ]
    existing_ids = {
        str(row.get("object_id") or "").strip()
        for row in objects
        if isinstance(row.get("object_id"), str) and str(row.get("object_id")).strip()
    }
    eligible_inventory, rejected_by_cluster = _filter_accessory_refill_inventory(
        inventory,
        allow_categories=set(policy["allow_categories"]),
        room_area_mm2=float(getattr(room_polygon, "area", 0.0)),
        max_footprint_ratio=float(policy["max_accessory_footprint_ratio"]),
    )
    remaining = deepcopy(eligible_inventory)
    inserted_by_cluster: dict[str, list[dict[str, Any]]] = {}
    quality_before = _quality_from_refined_solution(refined_layout_solution)
    quality_after = deepcopy(quality_before)
    notes: list[str] = []

    safe_zones = _detect_safe_refill_zones(
        room_polygon=room_polygon,
        room_output=room_output,
        objects=objects,
        cluster_lookup=cluster_lookup,
        max_safe_zones=int(policy["max_safe_zones"]),
    )
    if not safe_zones:
        notes.append("Controlled Accessory Refill found no low-conflict refill zones.")

    inserted_count = 0
    active_cluster_ids = sorted(
        [cluster_id for cluster_id, items in remaining.items() if items]
    )
    for _ in range(int(policy["max_refill_iterations"])):
        for cluster_id in active_cluster_ids:
            if inserted_count >= int(policy["max_refills_total"]):
                break
            item = remaining[cluster_id][0]
            cluster_meta = cluster_lookup.get(cluster_id) or {}
            placed_object, candidate_score = _place_accessory_refill_item(
                room_polygon=room_polygon,
                room_output=room_output,
                existing_objects=objects,
                existing_ids=existing_ids,
                cluster_id=cluster_id,
                cluster_meta=cluster_meta,
                item=item,
                safe_zones=safe_zones,
                grid_mm=int(policy["grid_mm"]),
                max_candidates=int(policy["max_candidates_per_object"]),
            )
            if placed_object is None:
                continue

            candidate_quality = _quality_after_accessory_refill(
                quality_before=quality_after,
                candidate_score=candidate_score,
            )
            if not _strict_refill_accept(
                quality_before=quality_after,
                quality_after=candidate_quality,
                min_quality_gain=float(policy["min_quality_gain_to_accept"]),
            ):
                continue

            objects.append(placed_object)
            existing_ids.add(str(placed_object["object_id"]))
            remaining[cluster_id].pop(0)
            inserted_by_cluster.setdefault(cluster_id, []).append(
                {
                    "object_id": str(placed_object["object_id"]),
                    "object_type": str(item.get("object_type") or ""),
                    "rotation_ccw": int(placed_object.get("rotation_ccw") or 0),
                    "bbox": deepcopy(placed_object.get("bbox") or {}),
                    "reason": str(placed_object.get("refill_reason") or ""),
                }
            )
            quality_after = candidate_quality
            inserted_count += 1
        break

    layout["objects"] = objects
    summary = _accessory_refill_summary(
        original_inventory=inventory,
        eligible_inventory=eligible_inventory,
        inserted_by_cluster=inserted_by_cluster,
        remaining_by_cluster=remaining,
        rejected_by_cluster=rejected_by_cluster,
        safe_zone_count=len(safe_zones),
        quality_before=quality_before,
        quality_after=quality_after,
        notes=notes,
    )
    layout["accessory_refill_objects_by_cluster"] = deepcopy(
        summary["refill_objects_by_cluster"]
    )
    layout["remaining_accessory_refill_objects_by_cluster"] = deepcopy(
        summary["remaining_refill_objects_by_cluster"]
    )
    if summary["refill_count"] > 0:
        notes = [
            str(item).strip() for item in layout.get("notes") or [] if str(item).strip()
        ]
        note = (
            "Controlled Accessory Refill added "
            f"{summary['refill_count']} non-critical accessory object(s) after judge "
            "without changing functional layout."
        )
        if note not in notes:
            notes.append(note)
        layout["notes"] = notes
    return layout, summary


def reintroduce_dropped_inventory(
    *,
    room_output: dict[str, Any],
    absolute_layout: dict[str, Any],
    cluster_output: dict[str, Any],
    dropped_inventory_by_cluster: dict[str, list[dict[str, Any]]] | None,
    refined_layout_solution: dict[str, Any] | None = None,
    refill_policy: dict[str, Any] | None = None,
    grid_mm: int = _CONTROLLED_REFILL_GRID_MM,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return controlled_accessory_refill(
        room_output=room_output,
        absolute_layout=absolute_layout,
        cluster_output=cluster_output,
        dropped_inventory_by_cluster=dropped_inventory_by_cluster,
        refined_layout_solution=refined_layout_solution,
        refill_policy=refill_policy,
        grid_mm=grid_mm,
    )


def _decision_lookup(
    tier_output: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    decisions = tier_output.get("decisions")
    if not isinstance(decisions, list):
        return lookup
    for row in decisions:
        if not isinstance(row, dict):
            continue
        cluster_id = str(row.get("cluster_id") or "").strip()
        object_type = str(row.get("object_type") or row.get("category") or "").strip()
        if not cluster_id or not object_type:
            continue
        lookup[(cluster_id, object_type)] = row
    return lookup


def _normalize_refill_policy(
    refill_policy: dict[str, Any] | None,
    *,
    grid_mm: int,
) -> dict[str, Any]:
    policy = refill_policy if isinstance(refill_policy, dict) else {}
    allowed_categories = {
        _normalized_inventory_label(item)
        for item in policy.get("allow_categories", ["accessory", "decor"])
        if str(item).strip()
    }
    allowed_categories = allowed_categories & _ACCESSORY_DECOR_CATEGORIES
    if not allowed_categories:
        allowed_categories = {"accessory", "decor"}
    return {
        "allow_categories": allowed_categories,
        "max_refills_total": _non_negative_int(
            policy.get("max_refills_total"),
            _CONTROLLED_REFILL_MAX_REFILLS_TOTAL,
        ),
        "max_candidates_per_object": _positive_int(
            policy.get("max_candidates_per_object"),
            _CONTROLLED_REFILL_MAX_CANDIDATES_PER_OBJECT,
        ),
        "max_safe_zones": _positive_int(
            policy.get("max_safe_zones"),
            _CONTROLLED_REFILL_MAX_SAFE_ZONES,
        ),
        "grid_mm": _positive_int(policy.get("grid_mm"), grid_mm),
        "max_refill_iterations": min(
            _positive_int(
                policy.get("max_refill_iterations"),
                _CONTROLLED_REFILL_MAX_ITERATIONS,
            ),
            _CONTROLLED_REFILL_MAX_ITERATIONS,
        ),
        "max_accessory_footprint_ratio": _positive_float(
            policy.get("max_accessory_footprint_ratio"),
            _CONTROLLED_REFILL_MAX_ACCESSORY_FOOTPRINT_RATIO,
        ),
        "min_quality_gain_to_accept": _positive_float(
            policy.get("min_quality_gain_to_accept"),
            _CONTROLLED_REFILL_MIN_QUALITY_GAIN,
        ),
        "disabled_reason": str(policy.get("disabled_reason") or ""),
        "missing_request_object_types": list(
            policy.get("missing_request_object_types") or []
        )
        if isinstance(policy.get("missing_request_object_types"), list)
        else [],
    }


def _non_negative_int(value: Any, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return int(default)
    return number if number >= 0 else int(default)


def _positive_int(value: Any, default: int) -> int:
    try:
        number = int(value)
    except Exception:
        return int(default)
    return number if number > 0 else int(default)


def _positive_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except Exception:
        return float(default)
    return number if number > 0.0 else float(default)


def _filter_accessory_refill_inventory(
    inventory: dict[str, list[dict[str, Any]]],
    *,
    allow_categories: set[str],
    room_area_mm2: float,
    max_footprint_ratio: float,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    eligible: dict[str, list[dict[str, Any]]] = {}
    rejected: dict[str, list[dict[str, Any]]] = {}
    for cluster_id, items in inventory.items():
        for item in items:
            rejection_reason = _accessory_refill_rejection_reason(
                item,
                allow_categories=allow_categories,
                room_area_mm2=room_area_mm2,
                max_footprint_ratio=max_footprint_ratio,
            )
            if rejection_reason is None:
                eligible.setdefault(cluster_id, []).append(deepcopy(item))
                continue
            rejected.setdefault(cluster_id, []).append(
                _rejected_refill_item(item, rejection_reason)
            )
    return eligible, rejected


def _accessory_refill_rejection_reason(
    item: Mapping[str, Any],
    *,
    allow_categories: set[str],
    room_area_mm2: float,
    max_footprint_ratio: float,
) -> str | None:
    object_type = _normalized_inventory_label(
        item.get("object_type") or item.get("category")
    )
    category = _normalized_inventory_label(item.get("category"))
    if object_type in _FUNCTIONAL_CORE_TYPES:
        return "functional_core_never_refilled"
    if object_type in _FUNCTIONAL_SUPPORT_TYPES:
        return "functional_support_never_refilled"
    if category not in allow_categories and object_type not in _ACCESSORY_DECOR_TYPES:
        return "not_accessory_or_decor"

    dims_mm = _object_dims_mm(dict(item))
    if dims_mm is None:
        return "missing_accessory_dimensions"
    footprint_ratio = (float(dims_mm[0]) * float(dims_mm[1])) / max(room_area_mm2, 1.0)
    if footprint_ratio > max_footprint_ratio:
        return "accessory_footprint_too_large"
    return None


def _normalized_inventory_label(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _rejected_refill_item(item: Mapping[str, Any], reason: str) -> dict[str, Any]:
    out = deepcopy(dict(item))
    out["refill_rejected_reason"] = reason
    return out


def _quality_from_refined_solution(
    refined_layout_solution: dict[str, Any] | None,
) -> dict[str, float]:
    raw_quality = (
        refined_layout_solution.get("quality_after")
        if isinstance(refined_layout_solution, dict)
        else None
    )
    quality = raw_quality if isinstance(raw_quality, dict) else {}
    functionality = _quality_float(quality.get("functionality"), 1.0)
    naturalness = _quality_float(quality.get("naturalness"), 0.92)
    semantic = _quality_float(quality.get("semantic"), 0.92)
    spatial = _quality_float(
        quality.get("spatial")
        if "spatial" in quality
        else quality.get("spatial_quality"),
        1.0,
    )
    total = _quality_float(
        quality.get("total"),
        (0.35 * functionality)
        + (0.30 * naturalness)
        + (0.20 * spatial)
        + (0.15 * semantic),
    )
    return {
        "functionality": round(functionality, 4),
        "naturalness": round(naturalness, 4),
        "semantic": round(semantic, 4),
        "spatial": round(spatial, 4),
        "total": round(total, 4),
    }


def _quality_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except Exception:
        return float(default)
    return max(0.0, min(1.0, number))


def _quality_after_accessory_refill(
    *,
    quality_before: Mapping[str, float],
    candidate_score: float,
) -> dict[str, float]:
    gain = min(0.012, max(_CONTROLLED_REFILL_MIN_QUALITY_GAIN, candidate_score * 0.008))
    naturalness = min(1.0, float(quality_before.get("naturalness", 0.0)) + gain * 0.45)
    semantic = min(1.0, float(quality_before.get("semantic", 0.0)) + gain * 0.35)
    total = min(1.0, float(quality_before.get("total", 0.0)) + gain)
    return {
        "functionality": round(float(quality_before.get("functionality", 0.0)), 4),
        "naturalness": round(naturalness, 4),
        "semantic": round(semantic, 4),
        "spatial": round(float(quality_before.get("spatial", 0.0)), 4),
        "total": round(total, 4),
    }


def _strict_refill_accept(
    *,
    quality_before: Mapping[str, float],
    quality_after: Mapping[str, float],
    min_quality_gain: float,
) -> bool:
    if (
        float(quality_after.get("total", 0.0))
        <= float(quality_before.get("total", 0.0)) + min_quality_gain
    ):
        return False
    for key in ("functionality", "spatial", "naturalness"):
        if float(quality_after.get(key, 0.0)) < float(quality_before.get(key, 0.0)):
            return False
    return True


def _detect_safe_refill_zones(
    *,
    room_polygon: Any,
    room_output: dict[str, Any],
    objects: list[dict[str, Any]],
    cluster_lookup: dict[str, dict[str, Any]],
    max_safe_zones: int,
) -> list[dict[str, Any]]:
    protected_polygons = _protected_refill_polygons(room_output)
    blockers = _blocked_polygons(room_output, objects)
    all_blockers = blockers + protected_polygons
    try:
        safe_geometry = (
            room_polygon
            if not all_blockers
            else room_polygon.difference(unary_union(all_blockers))
        )
    except Exception:
        safe_geometry = room_polygon

    zones: list[dict[str, Any]] = []
    affordance_map = _affordance_map(room_output)
    for row in _mapping_rows(affordance_map.get("wall_anchor_candidates")):
        _add_refill_zone(
            zones,
            room_polygon=room_polygon,
            safe_geometry=safe_geometry,
            source_polygon=_polygon_from_rows(row.get("anchor_polygon_ccw")),
            zone_id=str(row.get("id") or "wall_anchor"),
            zone_type="wall_anchor",
            cluster_id=None,
            score=0.72 + (0.2 * _safe_float(row.get("score"), 0.0)),
        )

    for row in _mapping_rows(affordance_map.get("daylight_regions")):
        _add_refill_zone(
            zones,
            room_polygon=room_polygon,
            safe_geometry=safe_geometry,
            source_polygon=_polygon_from_rows(
                row.get("near_polygon_ccw") or row.get("mid_polygon_ccw")
            ),
            zone_id=str(row.get("id") or "daylight_edge"),
            zone_type="daylight_edge",
            cluster_id=None,
            score=0.64 + (0.16 * _safe_float(row.get("score"), 0.0)),
        )

    for index, polygon in enumerate(_corner_refill_polygons(room_polygon), start=1):
        _add_refill_zone(
            zones,
            room_polygon=room_polygon,
            safe_geometry=safe_geometry,
            source_polygon=polygon,
            zone_id=f"corner_dead_zone_{index}",
            zone_type="corner_dead_zone",
            cluster_id=None,
            score=0.68,
        )

    for cluster_id in sorted(cluster_lookup):
        cluster_bbox = _cluster_bbox(objects, cluster_id)
        if cluster_bbox is None:
            continue
        for index, polygon in enumerate(
            _cluster_support_refill_polygons(cluster_bbox), start=1
        ):
            _add_refill_zone(
                zones,
                room_polygon=room_polygon,
                safe_geometry=safe_geometry,
                source_polygon=polygon,
                zone_id=f"{cluster_id}_support_{index}",
                zone_type="cluster_support_edge",
                cluster_id=cluster_id,
                score=0.55,
            )

    zones = _dedupe_refill_zones(zones)
    zones.sort(
        key=lambda row: (
            float(row.get("score") or 0.0),
            float(getattr(row.get("polygon"), "area", 0.0)),
            str(row.get("zone_id") or ""),
        ),
        reverse=True,
    )
    return zones[:max_safe_zones]


def _affordance_map(room_output: Mapping[str, Any]) -> Mapping[str, Any]:
    affordance = room_output.get("affordance_map")
    if isinstance(affordance, Mapping):
        return affordance
    room = room_output.get("room")
    if isinstance(room, Mapping) and isinstance(room.get("affordance_map"), Mapping):
        return room["affordance_map"]
    return {}


def _mapping_rows(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _add_refill_zone(
    zones: list[dict[str, Any]],
    *,
    room_polygon: Any,
    safe_geometry: Any,
    source_polygon: Any | None,
    zone_id: str,
    zone_type: str,
    cluster_id: str | None,
    score: float,
) -> None:
    if source_polygon is None:
        return
    try:
        clipped = source_polygon.intersection(safe_geometry).intersection(room_polygon)
    except Exception:
        return
    for polygon in _iter_polygon_geometries(clipped):
        if float(getattr(polygon, "area", 0.0)) < 10_000.0:
            continue
        if _polygon_wall_distance(room_polygon=room_polygon, polygon=polygon) > 500.0:
            continue
        zones.append(
            {
                "zone_id": zone_id,
                "zone_type": zone_type,
                "cluster_id": cluster_id,
                "score": round(max(0.0, min(1.0, score)), 4),
                "polygon": polygon,
            }
        )


def _iter_polygon_geometries(geometry: Any) -> list[Any]:
    if geometry is None or bool(getattr(geometry, "is_empty", False)):
        return []
    if getattr(geometry, "geom_type", "") == "Polygon":
        return [geometry]
    geoms = getattr(geometry, "geoms", None)
    if geoms is None:
        return []
    return [row for row in geoms if getattr(row, "geom_type", "") == "Polygon"]


def _corner_refill_polygons(room_polygon: Any) -> list[Any]:
    min_x, min_y, max_x, max_y = room_polygon.bounds
    depth = float(_CONTROLLED_REFILL_ZONE_DEPTH_MM)
    boxes = [
        (min_x, min_y, min_x + depth, min_y + depth),
        (max_x - depth, min_y, max_x, min_y + depth),
        (max_x - depth, max_y - depth, max_x, max_y),
        (min_x, max_y - depth, min_x + depth, max_y),
    ]
    polygons: list[Any] = []
    for left, top, right, bottom in boxes:
        try:
            polygons.append(
                Polygon(
                    [
                        (float(left), float(top)),
                        (float(right), float(top)),
                        (float(right), float(bottom)),
                        (float(left), float(bottom)),
                    ]
                )
            )
        except Exception:
            continue
    return polygons


def _cluster_support_refill_polygons(cluster_bbox: Mapping[str, int]) -> list[Any]:
    min_x = int(cluster_bbox["min_x"])
    min_y = int(cluster_bbox["min_y"])
    max_x = int(cluster_bbox["max_x"])
    max_y = int(cluster_bbox["max_y"])
    gap = _CONTROLLED_REFILL_CLUSTER_GAP_MM
    depth = _CONTROLLED_REFILL_ZONE_DEPTH_MM
    boxes = [
        (min_x - depth - gap, min_y, min_x - gap, max_y),
        (max_x + gap, min_y, max_x + depth + gap, max_y),
        (min_x, min_y - depth - gap, max_x, min_y - gap),
        (min_x, max_y + gap, max_x, max_y + depth + gap),
    ]
    polygons: list[Any] = []
    for left, top, right, bottom in boxes:
        try:
            polygons.append(
                Polygon(
                    [
                        (float(left), float(top)),
                        (float(right), float(top)),
                        (float(right), float(bottom)),
                        (float(left), float(bottom)),
                    ]
                )
            )
        except Exception:
            continue
    return polygons


def _dedupe_refill_zones(zones: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[int, int, int, int, str]] = set()
    out: list[dict[str, Any]] = []
    for zone in zones:
        polygon = zone.get("polygon")
        if polygon is None:
            continue
        min_x, min_y, max_x, max_y = polygon.bounds
        key = (
            int(round(min_x / 25.0)),
            int(round(min_y / 25.0)),
            int(round(max_x / 25.0)),
            int(round(max_y / 25.0)),
            str(zone.get("zone_type") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(zone)
    return out


def _protected_refill_polygons(room_output: Mapping[str, Any]) -> list[Any]:
    protected: list[Any] = []
    affordance_map = _affordance_map(room_output)
    for key in ("entry_landing_zones", "opening_guards"):
        for row in _mapping_rows(affordance_map.get(key)):
            polygon = _polygon_from_rows(row.get("polygon_ccw"))
            if polygon is not None:
                protected.append(polygon)

    for key in ("primary_circulation_corridors", "circulation_corridors"):
        for row in _mapping_rows(affordance_map.get(key)):
            corridor = _corridor_polygon(row)
            if corridor is not None:
                protected.append(corridor)

    for row in _mapping_rows(affordance_map.get("center_openness_regions")):
        polygon = _polygon_from_rows(row.get("polygon_ccw"))
        if polygon is None:
            polygon = _bbox_polygon(row.get("bbox_mm"))
        if polygon is not None:
            protected.append(polygon)
    return protected


def _corridor_polygon(row: Mapping[str, Any]) -> Any | None:
    if LineString is None:
        return None
    points = [
        (float(point.get("x") or 0.0), float(point.get("y") or 0.0))
        for point in row.get("polyline_mm") or []
        if isinstance(point, Mapping)
    ]
    if len(points) < 2:
        return None
    width = max(600.0, _safe_float(row.get("width_mm"), 900.0))
    try:
        return LineString(points).buffer(width / 2.0, cap_style=2, join_style=2)
    except Exception:
        return None


def _bbox_polygon(value: Any) -> Any | None:
    if not isinstance(value, Mapping) or Polygon is None:
        return None
    try:
        min_x = float(value.get("min_x") or 0.0)
        min_y = float(value.get("min_y") or 0.0)
        max_x = float(value.get("max_x") or 0.0)
        max_y = float(value.get("max_y") or 0.0)
    except Exception:
        return None
    if max_x <= min_x or max_y <= min_y:
        return None
    return Polygon([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)])


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _cluster_lookup(cluster_output: dict[str, Any]) -> dict[str, dict[str, Any]]:
    clusters = cluster_output.get("clusters")
    if not isinstance(clusters, list):
        return {}
    return {
        str(cluster.get("cluster_id") or "").strip(): cluster
        for cluster in clusters
        if isinstance(cluster, dict) and str(cluster.get("cluster_id") or "").strip()
    }


def _room_polygon(room_output: dict[str, Any]) -> Any | None:
    if Polygon is None:
        return None
    room = room_output.get("room")
    polygon_rows = room.get("polygon_ccw") if isinstance(room, dict) else None
    if not isinstance(polygon_rows, list):
        return None
    points = [
        (float(row.get("x") or 0.0), float(row.get("y") or 0.0))
        for row in polygon_rows
        if isinstance(row, dict)
    ]
    if len(points) < 3:
        return None
    try:
        return Polygon(points)
    except Exception:
        return None


def _blocked_polygons(
    room_output: dict[str, Any], objects: list[dict[str, Any]]
) -> list[Any]:
    blocked: list[Any] = []
    for row in objects:
        polygon = _polygon_from_rows(row.get("polygon_ccw"))
        if polygon is not None:
            blocked.append(polygon)

    obstacle_ids: set[str] = set()
    for obstacle in _obstacle_rows(room_output):
        if not isinstance(obstacle, dict):
            continue
        obstacle_id = str(obstacle.get("id") or "").strip()
        if obstacle_id:
            obstacle_ids.add(obstacle_id)
        polygon = _polygon_from_rows(obstacle.get("polygon_ccw"))
        if polygon is not None:
            blocked.append(polygon)

    blocked.extend(_opening_guard_polygons(room_output, obstacle_ids))
    return blocked


def _obstacle_rows(room_output: dict[str, Any]) -> list[dict[str, Any]]:
    room = room_output.get("room")
    raw_sources = [
        room_output.get("obstacles"),
        room.get("obstacles") if isinstance(room, dict) else None,
    ]

    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw in raw_sources:
        if not isinstance(raw, list):
            continue
        for item in raw:
            if not isinstance(item, dict):
                continue
            obstacle_id = str(item.get("id") or "").strip()
            if obstacle_id and obstacle_id in seen_ids:
                continue
            if obstacle_id:
                seen_ids.add(obstacle_id)
            rows.append(item)
    return rows


def _opening_guard_polygons(
    room_output: dict[str, Any],
    obstacle_ids: set[str],
) -> list[Any]:
    if LineString is None:
        return []

    openings = room_output.get("openings")
    if not isinstance(openings, dict):
        room = room_output.get("room")
        openings = room.get("openings") if isinstance(room, dict) else None
    if not isinstance(openings, dict):
        return []

    guards: list[Any] = []
    for door in openings.get("doors") or []:
        if not isinstance(door, dict):
            continue
        door_id = str(door.get("id") or "").strip()
        if door_id and f"{door_id}_swing" in obstacle_ids:
            continue
        line = _opening_line(door.get("segment_mm"))
        if line is None:
            continue
        depth = float(door.get("swing_radius_mm") or 700.0)
        if depth <= 0.0:
            depth = 700.0
        guards.append(line.buffer(depth, cap_style=2, join_style=2))

    for window in openings.get("windows") or []:
        if not isinstance(window, dict):
            continue
        window_id = str(window.get("id") or "").strip()
        if window_id and f"{window_id}_clearance" in obstacle_ids:
            continue
        line = _opening_line(window.get("segment_mm"))
        if line is None:
            continue
        depth = float(window.get("clearance_mm") or 100.0)
        if depth <= 0.0:
            depth = 100.0
        guards.append(line.buffer(depth, cap_style=2, join_style=2))
    return guards


def _opening_line(rows: Any) -> Any | None:
    if LineString is None or not isinstance(rows, list):
        return None
    points = [
        (float(row.get("x") or 0.0), float(row.get("y") or 0.0))
        for row in rows
        if isinstance(row, dict)
    ]
    if len(points) < 2:
        return None
    try:
        return LineString(points[:2])
    except Exception:
        return None


def _polygon_from_rows(rows: Any) -> Any | None:
    if Polygon is None or not isinstance(rows, list):
        return None
    points = [
        (float(row.get("x") or 0.0), float(row.get("y") or 0.0))
        for row in rows
        if isinstance(row, dict)
    ]
    if len(points) < 3:
        return None
    try:
        return Polygon(points)
    except Exception:
        return None


def _place_accessory_refill_item(
    *,
    room_polygon: Any,
    room_output: dict[str, Any],
    existing_objects: list[dict[str, Any]],
    existing_ids: set[str],
    cluster_id: str,
    cluster_meta: dict[str, Any],
    item: dict[str, Any],
    safe_zones: list[dict[str, Any]],
    grid_mm: int,
    max_candidates: int,
) -> tuple[dict[str, Any] | None, float]:
    dims_mm = _object_dims_mm(item)
    if dims_mm is None:
        return None, 0.0
    source_width, source_height = dims_mm
    object_type = str(
        item.get("object_type") or item.get("category") or "accessory"
    ).strip()
    allowed_rotations = _allowed_rotations(cluster_meta, object_type)
    front_side = _cluster_front_side(cluster_meta, object_type)
    blockers = _blocked_polygons(room_output, existing_objects)
    protected_polygons = _protected_refill_polygons(room_output)
    best_candidate: dict[str, Any] | None = None
    best_score = 0.0
    candidate_count = 0

    for zone in _zones_for_accessory_type(safe_zones, object_type, cluster_id):
        zone_polygon = zone.get("polygon")
        if zone_polygon is None:
            continue
        for rotation in allowed_rotations:
            box_width, box_height = (
                (source_width, source_height)
                if rotation % 180 == 0
                else (source_height, source_width)
            )
            origins = _accessory_candidate_origins(
                zone_polygon=zone_polygon,
                box_width=box_width,
                box_height=box_height,
                grid_mm=grid_mm,
            )
            for min_x, min_y in origins:
                if candidate_count >= max_candidates:
                    break
                candidate_count += 1
                polygon = _box_polygon(
                    min_x=int(min_x),
                    min_y=int(min_y),
                    width=int(box_width),
                    height=int(box_height),
                )
                if polygon is None:
                    continue
                if not _accessory_candidate_hard_valid(
                    room_polygon=room_polygon,
                    zone_polygon=zone_polygon,
                    polygon=polygon,
                    blockers=blockers,
                    protected_polygons=protected_polygons,
                ):
                    continue
                score = _accessory_refill_candidate_score(
                    room_polygon=room_polygon,
                    zone=zone,
                    polygon=polygon,
                    item=item,
                )
                if score <= best_score:
                    continue
                front_side_world = _effective_front_side_contract(
                    base_front=front_side,
                    rotation=rotation,
                )
                front_world_vec = side_to_vec(front_side_world)
                object_id = _next_accessory_refill_object_id(existing_ids, object_type)
                best_candidate = {
                    "object_id": object_id,
                    "instance_id": object_id,
                    "object_type": object_type,
                    "cluster_id": cluster_id,
                    "rotation_ccw": int(rotation % 360),
                    "bbox": {
                        "min_x": int(min_x),
                        "min_y": int(min_y),
                        "max_x": int(min_x + box_width),
                        "max_y": int(min_y + box_height),
                    },
                    "polygon_ccw": [
                        {"x": int(min_x), "y": int(min_y)},
                        {"x": int(min_x + box_width), "y": int(min_y)},
                        {"x": int(min_x + box_width), "y": int(min_y + box_height)},
                        {"x": int(min_x), "y": int(min_y + box_height)},
                    ],
                    "source_rect": {
                        "x": 0,
                        "y": 0,
                        "w": int(source_width),
                        "h": int(source_height),
                    },
                    "front_world": (
                        None
                        if front_world_vec is None
                        else {
                            "dx": int(front_world_vec[0]),
                            "dy": int(front_world_vec[1]),
                        }
                    ),
                    "front_side_world": front_side_world,
                    "refill_source": "controlled_accessory_refill",
                    "refill_zone_id": str(zone.get("zone_id") or ""),
                    "refill_zone_type": str(zone.get("zone_type") or ""),
                    "refill_reason": _accessory_refill_reason(zone, score),
                    "source": "controlled_accessory_refill",
                }
                best_score = score
            if candidate_count >= max_candidates:
                break
        if candidate_count >= max_candidates:
            break
    return best_candidate, best_score


def _zones_for_accessory_type(
    zones: list[dict[str, Any]],
    object_type: str,
    cluster_id: str,
) -> list[dict[str, Any]]:
    normalized_type = _normalized_inventory_label(object_type)

    def zone_rank(zone: dict[str, Any]) -> tuple[float, float, str]:
        zone_type = str(zone.get("zone_type") or "")
        semantic_bonus = 0.0
        if normalized_type in {"plant", "planter"} and zone_type == "daylight_edge":
            semantic_bonus = 0.2
        elif (
            normalized_type in {"wall_art", "art", "artwork"}
            and zone_type == "wall_anchor"
        ):
            semantic_bonus = 0.18
        elif (
            zone.get("cluster_id") == cluster_id and zone_type == "cluster_support_edge"
        ):
            semantic_bonus = 0.12
        return (
            float(zone.get("score") or 0.0) + semantic_bonus,
            float(getattr(zone.get("polygon"), "area", 0.0)),
            str(zone.get("zone_id") or ""),
        )

    return sorted(zones, key=zone_rank, reverse=True)


def _accessory_candidate_origins(
    *,
    zone_polygon: Any,
    box_width: int,
    box_height: int,
    grid_mm: int,
) -> list[tuple[int, int]]:
    min_x, min_y, max_x, max_y = zone_polygon.bounds
    gap = max(0, int(grid_mm))
    raw = [
        (min_x + gap, min_y + gap),
        (max_x - box_width - gap, min_y + gap),
        (min_x + gap, max_y - box_height - gap),
        (max_x - box_width - gap, max_y - box_height - gap),
        (
            (min_x + max_x - box_width) / 2.0,
            (min_y + max_y - box_height) / 2.0,
        ),
        (
            (min_x + max_x - box_width) / 2.0,
            min_y + gap,
        ),
    ]
    seen: set[tuple[int, int]] = set()
    origins: list[tuple[int, int]] = []
    for x_pos, y_pos in raw:
        origin = (
            _snap_to_grid(int(round(x_pos)), grid_mm),
            _snap_to_grid(int(round(y_pos)), grid_mm),
        )
        if origin in seen:
            continue
        seen.add(origin)
        origins.append(origin)
    return origins


def _box_polygon(*, min_x: int, min_y: int, width: int, height: int) -> Any | None:
    if Polygon is None:
        return None
    try:
        return Polygon(
            [
                (float(min_x), float(min_y)),
                (float(min_x + width), float(min_y)),
                (float(min_x + width), float(min_y + height)),
                (float(min_x), float(min_y + height)),
            ]
        )
    except Exception:
        return None


def _accessory_candidate_hard_valid(
    *,
    room_polygon: Any,
    zone_polygon: Any,
    polygon: Any,
    blockers: list[Any],
    protected_polygons: list[Any],
) -> bool:
    try:
        if not room_polygon.buffer(1e-6).covers(polygon):
            return False
        if not zone_polygon.buffer(1.0).covers(polygon):
            return False
    except Exception:
        return False
    if _overlaps_blockers(polygon, blockers):
        return False
    if _overlaps_blockers(polygon, protected_polygons):
        return False
    if _polygon_wall_distance(room_polygon=room_polygon, polygon=polygon) > 260.0:
        return False
    return True


def _accessory_refill_candidate_score(
    *,
    room_polygon: Any,
    zone: Mapping[str, Any],
    polygon: Any,
    item: Mapping[str, Any],
) -> float:
    zone_type = str(zone.get("zone_type") or "")
    zone_score = float(zone.get("score") or 0.0)
    wall_distance = _polygon_wall_distance(room_polygon=room_polygon, polygon=polygon)
    wall_score = max(0.0, 1.0 - (wall_distance / 260.0))
    dead_zone_gain = 1.0 if zone_type in {"corner_dead_zone", "wall_anchor"} else 0.74
    semantic_fit = _accessory_semantic_fit(item, zone_type)
    clutter_penalty = min(
        0.35,
        float(getattr(polygon, "area", 0.0)) / max(float(room_polygon.area), 1.0) * 8.0,
    )
    return max(
        0.0,
        (0.34 * zone_score)
        + (0.22 * wall_score)
        + (0.22 * dead_zone_gain)
        + (0.22 * semantic_fit)
        - clutter_penalty,
    )


def _accessory_semantic_fit(item: Mapping[str, Any], zone_type: str) -> float:
    object_type = _normalized_inventory_label(
        item.get("object_type") or item.get("category")
    )
    if object_type in {"plant", "planter"} and zone_type in {
        "daylight_edge",
        "corner_dead_zone",
        "wall_anchor",
    }:
        return 1.0
    if object_type in {"wall_art", "art", "artwork"} and zone_type == "wall_anchor":
        return 1.0
    if object_type in {"cushion", "pillow", "throw", "table_decor"}:
        return 0.82 if zone_type == "cluster_support_edge" else 0.62
    return 0.74


def _accessory_refill_reason(zone: Mapping[str, Any], score: float) -> str:
    zone_type = str(zone.get("zone_type") or "")
    if zone_type == "corner_dead_zone":
        base = "fills low-conflict edge dead zone without harming circulation"
    elif zone_type == "daylight_edge":
        base = "adds light decor near daylight edge outside protected clearance"
    elif zone_type == "cluster_support_edge":
        base = "adds light cluster-adjacent decor without changing function"
    else:
        base = "uses safe wall-adjacent accessory zone outside protected regions"
    return f"{base}; refill_score={score:.3f}"


def _object_dims_mm(item: dict[str, Any]) -> tuple[int, int] | None:
    dims = item.get("dims_mm")
    if isinstance(dims, Sequence) and not isinstance(dims, str) and len(dims) >= 2:
        try:
            first = int(round(float(dims[0])))
            second = int(round(float(dims[1])))
        except Exception:
            first = 0
            second = 0
        if first > 0 and second > 0:
            return max(first, second), min(first, second)

    rep_dims = item.get("rep_dims_m")
    if isinstance(rep_dims, dict):
        try:
            length_mm = int(round(float(rep_dims.get("L") or 0.0) * 1000.0))
            width_mm = int(round(float(rep_dims.get("W") or 0.0) * 1000.0))
        except Exception:
            return None
        if length_mm <= 0 or width_mm <= 0:
            return None
        return max(length_mm, width_mm), min(length_mm, width_mm)

    try:
        length_mm = int(
            round(
                float(
                    item.get("length_mm")
                    or item.get("w_mm")
                    or item.get("width_mm")
                    or 0.0
                )
            )
        )
        width_mm = int(
            round(
                float(
                    item.get("width_mm")
                    or item.get("d_mm")
                    or item.get("depth_mm")
                    or 0.0
                )
            )
        )
    except Exception:
        return None
    if length_mm <= 0 or width_mm <= 0:
        return None
    return max(length_mm, width_mm), min(length_mm, width_mm)


def _allowed_rotations(cluster_meta: dict[str, Any], object_type: str) -> list[int]:
    cluster_rules = (
        cluster_meta.get("cluster_rules")
        if isinstance(cluster_meta.get("cluster_rules"), dict)
        else {}
    )
    allowed = (
        cluster_rules.get("allowed_rotations")
        if isinstance(cluster_rules, dict)
        else {}
    )
    rotations = allowed.get(object_type) if isinstance(allowed, dict) else None
    clean_rotations = sorted(
        {
            int(rotation) % 360
            for rotation in (rotations or [0, 90, 180, 270])
            if isinstance(rotation, int | float)
        }
    )
    return clean_rotations or [0, 90, 180, 270]


def _cluster_bbox(
    objects: list[dict[str, Any]], cluster_id: str
) -> dict[str, int] | None:
    relevant = [
        row.get("bbox")
        for row in objects
        if isinstance(row, dict)
        and str(row.get("cluster_id") or "").strip() == cluster_id
        and isinstance(row.get("bbox"), dict)
    ]
    if not relevant:
        return None
    return {
        "min_x": min(int(bbox.get("min_x") or 0) for bbox in relevant),
        "min_y": min(int(bbox.get("min_y") or 0) for bbox in relevant),
        "max_x": max(int(bbox.get("max_x") or 0) for bbox in relevant),
        "max_y": max(int(bbox.get("max_y") or 0) for bbox in relevant),
    }


def _cluster_front_side(cluster_meta: dict[str, Any], object_type: str) -> str:
    cluster_rules = (
        cluster_meta.get("cluster_rules")
        if isinstance(cluster_meta.get("cluster_rules"), dict)
        else {}
    )
    facing = cluster_rules.get("facing") if isinstance(cluster_rules, dict) else None
    rule = facing.get(object_type) if isinstance(facing, dict) else None
    front_side = rule.get("front") if isinstance(rule, dict) else None
    return normalize_cardinal_side(front_side, default="top") or "top"


def _polygon_wall_distance(*, room_polygon: Any, polygon: Any) -> float:
    try:
        return float(room_polygon.boundary.distance(polygon.boundary))
    except Exception:
        return float("inf")


def _overlaps_blockers(polygon: Any, blockers: list[Any]) -> bool:
    for blocker in blockers:
        try:
            if float(polygon.intersection(blocker).area) > 1.0:
                return True
        except Exception:
            continue
    return False


def _next_accessory_refill_object_id(existing_ids: set[str], object_type: str) -> str:
    base = f"{object_type}__accessory_refill"
    index = 1
    while f"{base}_{index}" in existing_ids:
        index += 1
    return f"{base}_{index}"


def _snap_to_grid(value: int, grid_mm: int) -> int:
    step = max(1, int(grid_mm))
    return int(round(value / step) * step)


def _accessory_refill_summary(
    *,
    original_inventory: dict[str, list[dict[str, Any]]],
    eligible_inventory: dict[str, list[dict[str, Any]]],
    inserted_by_cluster: dict[str, list[dict[str, Any]]],
    remaining_by_cluster: dict[str, list[dict[str, Any]]],
    rejected_by_cluster: dict[str, list[dict[str, Any]]],
    safe_zone_count: int,
    quality_before: dict[str, float],
    quality_after: dict[str, float],
    notes: list[str],
) -> dict[str, Any]:
    refill_count = sum(len(items) for items in inserted_by_cluster.values())
    total_count = sum(len(items) for items in original_inventory.values())
    eligible_count = sum(len(items) for items in eligible_inventory.values())
    remaining_clean = {
        cluster_id: [deepcopy(item) for item in items if isinstance(item, dict)]
        for cluster_id, items in remaining_by_cluster.items()
        if isinstance(items, list) and items
    }
    rejected_clean = {
        cluster_id: [deepcopy(item) for item in items if isinstance(item, dict)]
        for cluster_id, items in rejected_by_cluster.items()
        if isinstance(items, list) and items
    }
    hard_valid = refill_count == 0 or _strict_refill_accept(
        quality_before=quality_before,
        quality_after=quality_after,
        min_quality_gain=_CONTROLLED_REFILL_MIN_QUALITY_GAIN,
    )
    return {
        "status": "OK" if refill_count > 0 else "UNCHANGED",
        "module": "controlled_accessory_refill",
        "refill_applied": refill_count > 0,
        "total_dropped_items": total_count,
        "eligible_accessory_count": eligible_count,
        "rejected_count": sum(len(items) for items in rejected_clean.values()),
        "refill_count": refill_count,
        "inserted_count": refill_count,
        "remaining_count": sum(len(items) for items in remaining_clean.values()),
        "safe_zone_count": int(safe_zone_count),
        "refill_objects_by_cluster": deepcopy(inserted_by_cluster),
        "remaining_refill_objects_by_cluster": remaining_clean,
        "rejected_refill_objects_by_cluster": rejected_clean,
        "quality_before": deepcopy(quality_before),
        "quality_after": deepcopy(quality_after),
        "hard_valid": bool(hard_valid),
        "notes": [str(note) for note in notes if str(note).strip()],
        "settings": {
            "max_refills_total": _CONTROLLED_REFILL_MAX_REFILLS_TOTAL,
            "max_candidates_per_object": _CONTROLLED_REFILL_MAX_CANDIDATES_PER_OBJECT,
            "max_safe_zones": _CONTROLLED_REFILL_MAX_SAFE_ZONES,
            "grid_mm": _CONTROLLED_REFILL_GRID_MM,
            "max_refill_iterations": _CONTROLLED_REFILL_MAX_ITERATIONS,
            "max_accessory_footprint_ratio": (
                _CONTROLLED_REFILL_MAX_ACCESSORY_FOOTPRINT_RATIO
            ),
            "min_quality_gain_to_accept": _CONTROLLED_REFILL_MIN_QUALITY_GAIN,
        },
    }
