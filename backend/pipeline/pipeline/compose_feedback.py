from __future__ import annotations

from copy import deepcopy
from typing import Any

_PRIORITY_ORDER = {
    "optional": 0,
    "secondary": 1,
    "primary": 2,
    "anchor": 3,
}
_SIZE_ORDER = ("S", "M", "L")


def evaluate_composed_cluster_feasibility(
    *,
    room_output: dict[str, Any],
    merged_output: dict[str, Any],
    cluster_results: dict[str, Any],
    cluster_outlines: dict[str, Any],
) -> dict[str, Any]:
    room_bbox = _room_bbox(room_output)
    room_width = max(0, room_bbox["max_x"] - room_bbox["min_x"])
    room_height = max(0, room_bbox["max_y"] - room_bbox["min_y"])
    room_polygon_area_mm2 = _room_polygon_area_mm2(room_output)
    room_long_side = max(room_width, room_height)
    room_short_side = min(room_width, room_height)

    offenders: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for cluster in _clusters(merged_output):
        cluster_id = str(cluster.get("cluster_id") or "").strip()
        if not cluster_id:
            continue

        active_decisions = _active_cluster_decisions(cluster)
        if not active_decisions:
            continue

        cluster_result = cluster_results.get(cluster_id)
        cluster_outline = cluster_outlines.get(cluster_id)
        status = _cluster_status(cluster_result, cluster_outline)
        bbox = _cluster_local_bbox(cluster_result, cluster_outline)
        span_x = max(0, int(bbox.get("max_x", 0)) - int(bbox.get("min_x", 0)))
        span_y = max(0, int(bbox.get("max_y", 0)) - int(bbox.get("min_y", 0)))
        max_span = max(span_x, span_y)
        bbox_area_mm2 = span_x * span_y
        cluster_outline_area_mm2 = _cluster_outline_area_mm2(
            cluster_result,
            cluster_outline,
        )
        fits_room_envelope = _fits_room_envelope(
            span_x=span_x,
            span_y=span_y,
            room_width=room_width,
            room_height=room_height,
        )
        fits_room_polygon_area = (
            cluster_outline_area_mm2 <= room_polygon_area_mm2
            if cluster_outline_area_mm2 > 0 and room_polygon_area_mm2 > 0
            else True
        )

        summary = {
            "cluster_id": cluster_id,
            "status": status,
            "span_x_mm": span_x,
            "span_y_mm": span_y,
            "max_span_mm": max_span,
            "bbox_area_mm2": bbox_area_mm2,
            "outline_area_mm2": cluster_outline_area_mm2,
            "fits_room_envelope": fits_room_envelope,
            "fits_room_polygon_area": fits_room_polygon_area,
            "active_object_ids": [
                str(row.get("object_type") or row.get("category") or "")
                for row in active_decisions
                if isinstance(row, dict)
            ],
        }
        summaries.append(summary)

        if status != "OK":
            offenders.append(
                {
                    **summary,
                    "reason": "COMPOSER_STATUS_NOT_OK",
                }
            )
            continue

        if not fits_room_envelope or not fits_room_polygon_area:
            offenders.append(
                {
                    **summary,
                    "reason": "CLUSTER_ENVELOPE_EXCEEDS_ROOM",
                    "fit_failure_mode": (
                        "polygon_area_upper_bound"
                        if fits_room_envelope and not fits_room_polygon_area
                        else "envelope"
                    ),
                }
            )

    return {
        "feasible": not offenders,
        "stage": "composer",
        "room_bbox": room_bbox,
        "room_polygon_area_mm2": room_polygon_area_mm2,
        "room_width_mm": room_width,
        "room_height_mm": room_height,
        "room_long_side_mm": room_long_side,
        "room_short_side_mm": room_short_side,
        "clusters": summaries,
        "offenders": offenders,
    }


def evaluate_solver_cluster_feasibility(
    *,
    merged_output: dict[str, Any],
    solver_output: dict[str, Any],
) -> dict[str, Any]:
    solver_status = str(solver_output.get("status") or "").strip().upper()
    skipped_clusters = [
        str(cluster_id).strip()
        for cluster_id in (solver_output.get("skipped_clusters") or [])
        if str(cluster_id).strip()
    ]
    solver_debug = _solver_debug(solver_output)
    candidate_counts = _solver_count_map(solver_debug, "candidate_counts")
    before_policy_counts = _solver_count_map(
        solver_debug,
        "candidate_counts_before_policy",
    )
    after_policy_counts = _solver_count_map(
        solver_debug,
        "candidate_counts_after_policy",
    )
    placer_seed = solver_output.get("placer_seed")
    if not isinstance(placer_seed, dict):
        placer_seed = {}
    placer_seed_reason = str(placer_seed.get("reason") or "").strip()
    conflicts = _solver_text_list(solver_output.get("conflicts"))
    blocked_clusters = {
        *skipped_clusters,
        *_solver_text_list(placer_seed.get("required_clusters_without_candidates")),
    }

    offenders: list[dict[str, Any]] = []
    for cluster in _clusters(merged_output):
        cluster_id = str(cluster.get("cluster_id") or "").strip()
        if not cluster_id or cluster_id not in blocked_clusters:
            continue
        before_count = _solver_cluster_count(before_policy_counts, cluster_id)
        after_count = _solver_cluster_count(after_policy_counts, cluster_id)
        candidate_count = _solver_cluster_count(candidate_counts, cluster_id)
        offenders.append(
            {
                "cluster_id": cluster_id,
                "reason": _solver_cluster_failure_reason(
                    before_count=before_count,
                    after_count=after_count,
                    candidate_count=candidate_count,
                    default_reason="SOLVER_HAS_NO_HARD_VALID_CANDIDATES",
                ),
                "candidate_count": _known_solver_count(
                    candidate_count,
                    after_count,
                    before_count,
                ),
                "candidate_count_before_policy": before_count,
                "candidate_count_after_policy": after_count,
                "solver_status": solver_status,
                "placer_seed_reason": placer_seed_reason,
                "conflicts": conflicts,
            }
        )

    if not offenders and solver_status == "UNSAT":
        if _solver_output_has_complete_hard_valid_layout(solver_output):
            return {
                "feasible": True,
                "stage": "solver_probe",
                "solver_status": solver_status,
                "skipped_clusters": skipped_clusters,
                "offenders": [],
            }
        for cluster_id in _select_unsat_pressure_clusters(
            merged_output=merged_output,
            solver_output=solver_output,
            candidate_counts=candidate_counts,
        ):
            before_count = _solver_cluster_count(before_policy_counts, cluster_id)
            after_count = _solver_cluster_count(after_policy_counts, cluster_id)
            candidate_count = _solver_cluster_count(candidate_counts, cluster_id)
            offenders.append(
                {
                    "cluster_id": cluster_id,
                    "reason": _solver_cluster_failure_reason(
                        before_count=before_count,
                        after_count=after_count,
                        candidate_count=candidate_count,
                        default_reason="SOLVER_EXACT_ASSIGNMENT_UNSAT",
                    ),
                    "candidate_count": _known_solver_count(
                        candidate_count,
                        after_count,
                        before_count,
                    ),
                    "candidate_count_before_policy": before_count,
                    "candidate_count_after_policy": after_count,
                    "solver_status": solver_status,
                    "placer_seed_reason": placer_seed_reason,
                    "conflicts": conflicts,
                }
            )

    return {
        "feasible": not offenders,
        "stage": "solver_probe",
        "solver_status": solver_status,
        "skipped_clusters": skipped_clusters,
        "offenders": offenders,
    }


def _solver_debug(solver_output: dict[str, Any]) -> dict[str, Any]:
    solver_debug = solver_output.get("solver_debug")
    return solver_debug if isinstance(solver_debug, dict) else {}


def _solver_count_map(
    solver_debug: dict[str, Any],
    key: str,
) -> dict[str, int | None]:
    raw = solver_debug.get(key)
    if not isinstance(raw, dict):
        return {}
    return {
        str(cluster_id).strip(): _coerce_int_count(value)
        for cluster_id, value in raw.items()
        if str(cluster_id).strip()
    }


def _solver_cluster_count(
    counts: dict[str, int | None],
    cluster_id: str,
) -> int | None:
    return counts.get(cluster_id)


def _known_solver_count(*counts: int | None) -> int:
    for count in counts:
        if count is not None:
            return max(0, int(count))
    return 0


def _coerce_int_count(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _solver_text_list(value: Any) -> list[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _solver_cluster_failure_reason(
    *,
    before_count: int | None,
    after_count: int | None,
    candidate_count: int | None,
    default_reason: str,
) -> str:
    if before_count is not None and before_count > 0 and after_count == 0:
        return "SOLVER_CONCEPT_POLICY_EMPTY_POOL"
    if before_count == 0:
        return "SOLVER_HAS_NO_HARD_VALID_CANDIDATES"
    if before_count is None and after_count == 0 and candidate_count == 0:
        return "SOLVER_HAS_NO_HARD_VALID_CANDIDATES"
    return default_reason


def _solver_output_has_complete_hard_valid_layout(
    solver_output: dict[str, Any],
) -> bool:
    solver_debug = solver_output.get("solver_debug")
    if not isinstance(solver_debug, dict):
        return False

    for key in ("verify", "best_verify"):
        verify = solver_debug.get(key)
        if not isinstance(verify, dict):
            continue
        if bool(verify.get("hard_valid")) and bool(verify.get("complete")):
            return True
    return False


def apply_compose_backoff(
    *,
    tier_output: dict[str, Any],
    merged_output: dict[str, Any],
    feasibility: dict[str, Any],
) -> tuple[dict[str, Any], list[str], bool]:
    updated = deepcopy(tier_output)
    decisions = updated.get("decisions")
    if not isinstance(decisions, list):
        return updated, [], False

    cluster_lookup = {
        str(cluster.get("cluster_id") or "").strip(): cluster
        for cluster in _clusters(merged_output)
        if isinstance(cluster, dict)
    }

    notes: list[str] = []
    changed = False
    for offender in feasibility.get("offenders") or []:
        if not isinstance(offender, dict):
            continue
        cluster_id = str(offender.get("cluster_id") or "").strip()
        if not cluster_id:
            continue
        cluster = cluster_lookup.get(cluster_id)
        if not isinstance(cluster, dict):
            continue
        cluster_changed, note = _backoff_cluster_decisions(
            decisions=decisions,
            cluster=cluster,
        )
        if cluster_changed:
            changed = True
            if note:
                notes.append(note)
            break

    if changed:
        existing_notes = updated.get("notes")
        if isinstance(existing_notes, list):
            updated["notes"] = [*existing_notes, *notes]
        else:
            updated["notes"] = list(notes)

    return updated, notes, changed


def apply_variant_diversification(
    *,
    tier_output: dict[str, Any],
    merged_output: dict[str, Any],
    variant_index: int,
) -> tuple[dict[str, Any], list[str], bool]:
    updated = deepcopy(tier_output)
    decisions = updated.get("decisions")
    if not isinstance(decisions, list):
        return updated, [], False

    clusters = [
        cluster
        for cluster in _clusters(merged_output)
        if _active_cluster_decisions(cluster)
    ]
    if not clusters:
        return updated, [], False

    ranked_clusters = sorted(
        clusters,
        key=lambda cluster: (
            -_cluster_active_footprint(cluster),
            str(cluster.get("cluster_id") or ""),
        ),
    )
    offset = max(0, int(variant_index) - 1) % len(ranked_clusters)
    ordered_clusters = ranked_clusters[offset:] + ranked_clusters[:offset]

    notes: list[str] = []
    for cluster in ordered_clusters:
        changed, note = _diversify_cluster_decisions(
            decisions=decisions,
            cluster=cluster,
            variant_index=variant_index,
        )
        if changed:
            if note:
                notes.append(note)
            existing_notes = updated.get("notes")
            if isinstance(existing_notes, list):
                updated["notes"] = [*existing_notes, *notes]
            else:
                updated["notes"] = list(notes)
            return updated, notes, True

    return updated, [], False


def _clusters(merged_output: dict[str, Any]) -> list[dict[str, Any]]:
    clusters = merged_output.get("clusters")
    if not isinstance(clusters, list):
        return []
    return [cluster for cluster in clusters if isinstance(cluster, dict)]


def _cluster_active_footprint(cluster: dict[str, Any]) -> float:
    return sum(
        _decision_footprint_m2(row) * max(1, int(row.get("quantity") or 0))
        for row in _active_cluster_decisions(cluster)
    )


def _active_cluster_decisions(cluster: dict[str, Any]) -> list[dict[str, Any]]:
    decisions = cluster.get("decisions")
    if not isinstance(decisions, list):
        return []
    active: list[dict[str, Any]] = []
    for row in decisions:
        if not isinstance(row, dict):
            continue
        quantity = int(row.get("quantity") or 0)
        if quantity <= 0:
            continue
        active.append(row)
    return active


def _room_bbox(room_output: dict[str, Any]) -> dict[str, int]:
    room = room_output.get("room")
    points = room.get("polygon_ccw") if isinstance(room, dict) else None
    xs = [int(point.get("x", 0)) for point in points or [] if isinstance(point, dict)]
    ys = [int(point.get("y", 0)) for point in points or [] if isinstance(point, dict)]
    if not xs or not ys:
        return {"min_x": 0, "min_y": 0, "max_x": 0, "max_y": 0}
    return {
        "min_x": min(xs),
        "min_y": min(ys),
        "max_x": max(xs),
        "max_y": max(ys),
    }


def _room_polygon_area_mm2(room_output: dict[str, Any]) -> int:
    room = room_output.get("room")
    points = room.get("polygon_ccw") if isinstance(room, dict) else None
    return _polygon_area_mm2(points)


def _cluster_status(cluster_result: Any, cluster_outline: Any) -> str:
    for payload in (cluster_result, cluster_outline):
        if not isinstance(payload, dict):
            continue
        status = str(payload.get("status") or "").strip().upper()
        if status:
            return status
    return "UNSAT"


def _cluster_local_bbox(cluster_result: Any, cluster_outline: Any) -> dict[str, int]:
    for payload in (cluster_outline, cluster_result):
        if not isinstance(payload, dict):
            continue
        footprint = payload.get("cluster_footprint")
        if not isinstance(footprint, dict):
            continue
        bbox = footprint.get("local_bbox")
        if not isinstance(bbox, dict):
            continue
        return {
            "min_x": int(bbox.get("min_x", 0)),
            "min_y": int(bbox.get("min_y", 0)),
            "max_x": int(bbox.get("max_x", 0)),
            "max_y": int(bbox.get("max_y", 0)),
        }
    return {"min_x": 0, "min_y": 0, "max_x": 0, "max_y": 0}


def _cluster_outline_area_mm2(cluster_result: Any, cluster_outline: Any) -> int:
    for payload in (cluster_outline, cluster_result):
        if not isinstance(payload, dict):
            continue
        footprint = payload.get("cluster_footprint")
        if not isinstance(footprint, dict):
            continue
        rects = footprint.get("rects")
        area_mm2 = _rect_union_area_mm2(rects)
        if area_mm2 > 0:
            return area_mm2
        outlines = footprint.get("outline_polygons_ccw")
        if isinstance(outlines, list):
            outline_area_mm2 = 0
            for poly in outlines:
                outline_area_mm2 += _polygon_area_mm2(poly)
            if outline_area_mm2 > 0:
                return outline_area_mm2
    return 0


def _polygon_area_mm2(points: Any) -> int:
    if not isinstance(points, list):
        return 0
    vertices: list[tuple[float, float]] = []
    for point in points:
        if not isinstance(point, dict):
            continue
        vertices.append((float(point.get("x", 0.0)), float(point.get("y", 0.0))))
    if len(vertices) < 3:
        return 0
    area2 = 0.0
    for idx, (x1, y1) in enumerate(vertices):
        x2, y2 = vertices[(idx + 1) % len(vertices)]
        area2 += (x1 * y2) - (x2 * y1)
    return int(round(abs(area2) / 2.0))


def _rect_union_area_mm2(rects: Any) -> int:
    if not isinstance(rects, list):
        return 0

    intervals: list[tuple[int, int, int, int]] = []
    x_coords: set[int] = set()
    for rect in rects:
        if not isinstance(rect, dict):
            continue
        x1 = int(rect.get("x", 0))
        y1 = int(rect.get("y", 0))
        width = int(rect.get("w", 0))
        height = int(rect.get("h", 0))
        if width <= 0 or height <= 0:
            continue
        x2 = x1 + width
        y2 = y1 + height
        intervals.append((x1, x2, y1, y2))
        x_coords.add(x1)
        x_coords.add(x2)

    if not intervals or len(x_coords) < 2:
        return 0

    sorted_x = sorted(x_coords)
    area_mm2 = 0
    for left, right in zip(sorted_x, sorted_x[1:]):
        if right <= left:
            continue
        y_segments = [
            (y1, y2)
            for x1, x2, y1, y2 in intervals
            if x1 < right and x2 > left and y2 > y1
        ]
        covered_y = _merged_interval_length(y_segments)
        if covered_y <= 0:
            continue
        area_mm2 += (right - left) * covered_y
    return int(area_mm2)


def _merged_interval_length(segments: list[tuple[int, int]]) -> int:
    if not segments:
        return 0
    ordered = sorted(segments)
    total = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start > current_end:
            total += max(0, current_end - current_start)
            current_start, current_end = start, end
            continue
        current_end = max(current_end, end)
    total += max(0, current_end - current_start)
    return total


def _fits_room_envelope(
    *,
    span_x: int,
    span_y: int,
    room_width: int,
    room_height: int,
) -> bool:
    return (span_x <= room_width and span_y <= room_height) or (
        span_y <= room_width and span_x <= room_height
    )


def _backoff_cluster_decisions(
    *,
    decisions: list[dict[str, Any]],
    cluster: dict[str, Any],
) -> tuple[bool, str | None]:
    cluster_id = str(cluster.get("cluster_id") or "").strip()
    anchors = {
        str(anchor).strip()
        for anchor in (cluster.get("anchors") or [])
        if isinstance(anchor, str) and anchor.strip()
    }
    scoped = [
        row
        for row in decisions
        if isinstance(row, dict) and str(row.get("cluster_id") or "") == cluster_id
    ]
    active = [row for row in scoped if int(row.get("quantity") or 0) > 0]
    if not active:
        return False, None

    shrink_candidates = sorted(
        (row for row in active if _tier_rank(row.get("size_tier")) > 0),
        key=_shrink_sort_key,
    )
    if shrink_candidates:
        target = shrink_candidates[0]
        current_tier = str(target.get("size_tier") or "M").upper()
        target["size_tier"] = _shift_tier_down(current_tier)
        _append_backoff_rationale(
            target,
            "Layout feedback downgraded size tier to improve feasibility.",
        )
        object_type = str(target.get("object_type") or target.get("category") or "")
        return (
            True,
            f"Layout feedback backoff shrank {cluster_id}.{object_type} from {current_tier} to {target['size_tier']}.",
        )

    if len(active) > 1:
        drop_candidates = sorted(
            (
                row
                for row in active
                if _can_drop_decision(row=row, active=active, anchors=anchors)
            ),
            key=lambda row: _backoff_sort_key(row, anchors=anchors),
        )
        if drop_candidates:
            target = drop_candidates[0]
            current_quantity = int(target.get("quantity") or 0)
            target["quantity"] = max(0, current_quantity - 1)
            _append_backoff_rationale(
                target,
                "Layout feedback reduced quantity to improve feasibility.",
            )
            object_type = str(target.get("object_type") or target.get("category") or "")
            return (
                True,
                f"Layout feedback backoff reduced {cluster_id}.{object_type} quantity to {target['quantity']}.",
            )

    return False, None


def _diversify_cluster_decisions(
    *,
    decisions: list[dict[str, Any]],
    cluster: dict[str, Any],
    variant_index: int,
) -> tuple[bool, str | None]:
    cluster_id = str(cluster.get("cluster_id") or "").strip()
    anchors = {
        str(anchor).strip()
        for anchor in (cluster.get("anchors") or [])
        if isinstance(anchor, str) and anchor.strip()
    }
    scoped = [
        row
        for row in decisions
        if isinstance(row, dict) and str(row.get("cluster_id") or "") == cluster_id
    ]
    active = [row for row in scoped if int(row.get("quantity") or 0) > 0]
    if not active:
        return False, None

    strategy = max(0, int(variant_index) - 1) % 3
    object_type = ""
    if strategy == 0:
        shrink_candidates = _sorted_shrink_candidates(active=active, anchors=anchors)
        if shrink_candidates:
            target = shrink_candidates[0]
            current_tier = str(target.get("size_tier") or "M").upper()
            next_tier = _shift_tier_down(current_tier)
            if next_tier != current_tier:
                target["size_tier"] = next_tier
                _append_backoff_rationale(
                    target,
                    "Variant diversification reduced size tier to create a distinct feasible option.",
                )
                object_type = str(
                    target.get("object_type") or target.get("category") or ""
                )
                return (
                    True,
                    f"Variant diversification shrank {cluster_id}.{object_type} from {current_tier} to {next_tier}.",
                )

    if strategy == 1 and len(active) > 1:
        drop_candidates = sorted(
            (
                row
                for row in active
                if _can_drop_decision(row=row, active=active, anchors=anchors)
            ),
            key=lambda row: _backoff_sort_key(row, anchors=anchors),
        )
        if drop_candidates:
            target = drop_candidates[0]
            current_quantity = int(target.get("quantity") or 0)
            target["quantity"] = max(0, current_quantity - 1)
            _append_backoff_rationale(
                target,
                "Variant diversification reduced quantity to explore another feasible layout.",
            )
            object_type = str(target.get("object_type") or target.get("category") or "")
            return (
                True,
                f"Variant diversification reduced {cluster_id}.{object_type} quantity to {target['quantity']}.",
            )

    shrink_candidates = _sorted_shrink_candidates(active=active, anchors=anchors)
    if shrink_candidates:
        target = shrink_candidates[0]
        current_tier = str(target.get("size_tier") or "M").upper()
        next_tier = _shift_tier_down(current_tier)
        if next_tier != current_tier:
            target["size_tier"] = next_tier
            _append_backoff_rationale(
                target,
                "Variant diversification reduced size tier to create a distinct feasible option.",
            )
            object_type = str(target.get("object_type") or target.get("category") or "")
            return (
                True,
                f"Variant diversification shrank {cluster_id}.{object_type} from {current_tier} to {next_tier}.",
            )

    if len(active) > 1:
        drop_candidates = sorted(
            (
                row
                for row in active
                if _can_drop_decision(row=row, active=active, anchors=anchors)
            ),
            key=lambda row: _backoff_sort_key(row, anchors=anchors),
        )
        if drop_candidates:
            target = drop_candidates[0]
            current_quantity = int(target.get("quantity") or 0)
            target["quantity"] = max(0, current_quantity - 1)
            _append_backoff_rationale(
                target,
                "Variant diversification reduced quantity to explore another feasible layout.",
            )
            object_type = str(target.get("object_type") or target.get("category") or "")
            return (
                True,
                f"Variant diversification reduced {cluster_id}.{object_type} quantity to {target['quantity']}.",
            )

    return False, None


def _can_drop_decision(
    *,
    row: dict[str, Any],
    active: list[dict[str, Any]],
    anchors: set[str],
) -> bool:
    object_type = str(row.get("object_type") or row.get("category") or "")
    quantity = int(row.get("quantity") or 0)
    if quantity <= 0:
        return False

    priority = str(row.get("priority") or "secondary").strip().lower()
    active_anchor_count = sum(
        1
        for item in active
        if str(item.get("object_type") or item.get("category") or "") in anchors
        and int(item.get("quantity") or 0) > 0
    )

    if object_type in anchors or priority == "anchor":
        return active_anchor_count > 1

    return len(active) > 1


def _backoff_sort_key(
    row: dict[str, Any], anchors: set[str]
) -> tuple[int, float, int, str]:
    object_type = str(row.get("object_type") or row.get("category") or "")
    priority = str(row.get("priority") or "secondary").strip().lower()
    priority_rank = _PRIORITY_ORDER.get(priority, 99)
    if object_type in anchors or priority == "anchor":
        priority_rank = _PRIORITY_ORDER["anchor"]
    footprint = _decision_footprint_m2(row)
    tier_rank = _tier_rank(row.get("size_tier"))
    return (priority_rank, -footprint, -tier_rank, object_type)


def _sorted_shrink_candidates(
    *,
    active: list[dict[str, Any]],
    anchors: set[str],
) -> list[dict[str, Any]]:
    _ = anchors
    return sorted(
        (row for row in active if _tier_rank(row.get("size_tier")) > 0),
        key=_shrink_sort_key,
    )


def _shrink_sort_key(row: dict[str, Any]) -> tuple[float, int, int, str]:
    object_type = str(row.get("object_type") or row.get("category") or "")
    footprint = _decision_footprint_m2(row)
    tier_rank = _tier_rank(row.get("size_tier"))
    priority = str(row.get("priority") or "secondary").strip().lower()
    priority_rank = _PRIORITY_ORDER.get(priority, 99)
    return (-footprint, -tier_rank, priority_rank, object_type)


def _decision_footprint_m2(row: dict[str, Any]) -> float:
    dims = row.get("rep_dims_m")
    if not isinstance(dims, dict):
        return 0.0
    try:
        if dims.get("A") is not None:
            return float(dims.get("A") or 0.0)
        length = float(dims.get("L") or 0.0)
        width = float(dims.get("W") or 0.0)
    except Exception:
        return 0.0
    return max(0.0, length) * max(0.0, width)


def _tier_rank(value: Any) -> int:
    tier = str(value or "M").upper()
    try:
        return _SIZE_ORDER.index(tier)
    except ValueError:
        return 1


def _shift_tier_down(tier: str) -> str:
    try:
        idx = _SIZE_ORDER.index(tier.upper())
    except ValueError:
        idx = 1
    return _SIZE_ORDER[max(0, idx - 1)]


def _append_backoff_rationale(row: dict[str, Any], note: str) -> None:
    rationale = str(row.get("rationale") or "").strip()
    if note in rationale:
        return
    row["rationale"] = f"{rationale} {note}".strip()


def _select_unsat_pressure_clusters(
    *,
    merged_output: dict[str, Any],
    solver_output: dict[str, Any],
    candidate_counts: dict[str, Any],
) -> list[str]:
    prioritized_clusters = _extract_prioritized_solver_clusters(solver_output)
    if prioritized_clusters:
        return prioritized_clusters

    ranked_clusters: list[tuple[tuple[float, int, int, str], str]] = []
    for cluster in _clusters(merged_output):
        cluster_id = str(cluster.get("cluster_id") or "").strip()
        if not cluster_id:
            continue
        active = _active_cluster_decisions(cluster)
        if not active:
            continue
        footprint_m2 = sum(
            _decision_footprint_m2(row) * max(1, int(row.get("quantity") or 0))
            for row in active
        )
        candidate_count = int(candidate_counts.get(cluster_id) or 0)
        ranked_clusters.append(
            (
                (-footprint_m2, candidate_count, -len(active), cluster_id),
                cluster_id,
            )
        )

    ranked_clusters.sort()
    if not ranked_clusters:
        return []
    return [cluster_id for _, cluster_id in ranked_clusters]


def _extract_prioritized_solver_clusters(
    solver_output: dict[str, Any],
) -> list[str]:
    solver_debug = solver_output.get("solver_debug")
    if not isinstance(solver_debug, dict):
        return []

    for key in ("best_verify", "verify"):
        verify = solver_debug.get(key)
        if not isinstance(verify, dict):
            continue
        repair_guidance = verify.get("repair_guidance")
        if not isinstance(repair_guidance, dict):
            continue
        prioritized = repair_guidance.get("prioritized_clusters")
        if not isinstance(prioritized, list):
            continue

        cluster_ids: list[str] = []
        for item in prioritized:
            if not isinstance(item, dict):
                continue
            cluster_id = str(item.get("cluster_id") or "").strip()
            if cluster_id and cluster_id not in cluster_ids:
                cluster_ids.append(cluster_id)
        if cluster_ids:
            return cluster_ids

    return []
