from __future__ import annotations

import math
from typing import Any, Iterable, Tuple

from layout.grid_policy import normalize_layout_grid_mm

Rect = Tuple[int, int, int, int]
Point = Tuple[int, int]

FIXED_ACCESS_CLEARANCE_RATIO = 0.25


# ============================================================
# Public API
# ============================================================


def score_cluster_from_cluster_json(
    *,
    cluster: dict[str, Any],
    local_placements: list[dict[str, Any]],
    use_clearance: bool = True,
) -> dict[str, Any]:
    """
    High-level entrypoint compatible with your cluster JSON.

    Input:
    - cluster: one cluster JSON object
    - local_placements: [{"id":..., "x":..., "y":..., "rot":...}, ...]

    Output:
    {
      "cluster_id": ...,
      "valid_hard": bool,
      "hard": {...},
      "soft": {...},
      "layout_quality": {...},
      "all_evaluations": [...],
      "hard_evaluations": [...],
      "soft_evaluations": [...]
    }
    """
    hard_constraints = cluster.get("hard_constraints", [])
    soft_constraints = cluster.get("soft_constraints", [])
    cluster_rules = (
        cluster.get("cluster_rules", {}) if isinstance(cluster, dict) else {}
    )

    objects = build_object_specs_from_cluster(cluster)

    return score_cluster_constraints(
        hard_constraints=hard_constraints,
        soft_constraints=soft_constraints,
        objects=objects,
        local_placements=local_placements,
        grid_mm=normalize_layout_grid_mm((cluster_rules or {}).get("grid_mm")),
        cluster_rules=cluster_rules,
        use_clearance=use_clearance,
    )


def score_cluster_constraints(
    *,
    hard_constraints: list[dict[str, Any]] | None,
    soft_constraints: list[dict[str, Any]] | None,
    objects: list[dict[str, Any]] | dict[str, dict[str, Any]] | None,
    local_placements: list[dict[str, Any]] | None,
    grid_mm: int,
    cluster_rules: dict[str, Any] | None = None,
    use_clearance: bool = True,
) -> dict[str, Any]:
    """
    Core scoring engine.

    Hard constraints:
    - still treated as hard
    - each one gets a continuous violation score

    Soft constraints:
    - weighted penalty

    Also returns layout_quality so you can compare valid layouts by compactness.
    """
    if not isinstance(hard_constraints, list):
        hard_constraints = []
    if not isinstance(soft_constraints, list):
        soft_constraints = []
    if objects is None or not isinstance(objects, (list, dict)):
        objects = []
    if not isinstance(local_placements, list):
        local_placements = []
    if cluster_rules is not None and not isinstance(cluster_rules, dict):
        cluster_rules = None

    specs = _normalize_objects(objects)
    placements = _normalize_placements(local_placements)

    facing_map = {}
    if isinstance(cluster_rules, dict) and isinstance(
        cluster_rules.get("facing"), dict
    ):
        facing_map = cluster_rules["facing"]

    rects = _build_rects(placements, specs, use_clearance=False)
    rects_clear = _build_rects(placements, specs, use_clearance=use_clearance)

    access_required_ids = _collect_access_required_ids(
        hard_constraints=hard_constraints,
        cluster_rules=cluster_rules,
    )

    allow_clearance_overlap = _build_allow_clearance_overlap(
        hard_constraints=hard_constraints,
        facing_map=facing_map,
        specs=specs,
    )

    ctx = {
        "grid_mm": int(grid_mm),
        "hard_constraints": hard_constraints,
        "soft_constraints": soft_constraints,
        "cluster_rules": cluster_rules or {},
        "specs": specs,
        "placements": placements,
        "rects": rects,
        "rects_clear": rects_clear,
        "facing_map": facing_map,
        "access_required_ids": access_required_ids,
        "allow_clearance_overlap": allow_clearance_overlap,
        "use_clearance": bool(use_clearance),
    }

    hard_evaluations: list[dict[str, Any]] = []
    soft_evaluations: list[dict[str, Any]] = []

    for idx, c in enumerate(hard_constraints):
        ev = evaluate_constraint(
            constraint=c,
            ctx=ctx,
            index=idx,
            hard=True,
        )
        hard_evaluations.append(ev)

    for idx, c in enumerate(soft_constraints):
        ev = evaluate_constraint(
            constraint=c,
            ctx=ctx,
            index=idx,
            hard=False,
        )
        soft_evaluations.append(ev)

    layout_quality = compute_layout_quality(rects)

    hard_summary = summarize_hard_evaluations(hard_evaluations)
    soft_summary = summarize_soft_evaluations(soft_evaluations)

    # Lexicographic comparison key:
    # 1) fewer hard violations
    # 2) lower total hard weighted violation
    # 3) lower max hard violation
    # 4) lower soft weighted penalty from forge semantics
    # 5) lower compact score
    lexicographic_key = [
        hard_summary["num_violations"],
        hard_summary["total_weighted_violation"],
        hard_summary["max_weighted_violation"],
        soft_summary["total_weighted_penalty"],
        layout_quality["compact_score"],
    ]

    return {
        "cluster_id": None,
        "valid_hard": hard_summary["num_violations"] == 0,
        "hard": hard_summary,
        "soft": soft_summary,
        "layout_quality": layout_quality,
        "lexicographic_key": lexicographic_key,
        "all_evaluations": hard_evaluations + soft_evaluations,
        "hard_evaluations": hard_evaluations,
        "soft_evaluations": soft_evaluations,
        "debug": {
            "rects": {k: _rect_dict(v) for k, v in rects.items()},
            "rects_clear": {k: _rect_dict(v) for k, v in rects_clear.items()},
            "access_required_ids": sorted(list(access_required_ids)),
            "allow_clearance_overlap": {
                k: sorted(list(v)) for k, v in allow_clearance_overlap.items()
            },
        },
    }


def build_object_specs_from_cluster(cluster: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Build object specs from your current cluster JSON schema.

    Notes:
    - id is currently object_type
    - w/h are derived from rep_dims_m["L"], rep_dims_m["W"]
    - rotations/facing come from cluster_rules
    """
    specs: list[dict[str, Any]] = []

    decisions = cluster.get("decisions", [])
    rules = cluster.get("cluster_rules", {}) if isinstance(cluster, dict) else {}
    allowed_raw = rules.get("allowed_rotations") if isinstance(rules, dict) else None
    allowed = allowed_raw if isinstance(allowed_raw, dict) else {}
    facing_raw = rules.get("facing") if isinstance(rules, dict) else None
    facing = facing_raw if isinstance(facing_raw, dict) else {}

    if not isinstance(decisions, list):
        return specs

    for d in decisions:
        if not isinstance(d, dict):
            continue

        obj_id = d.get("object_type") or d.get("category")
        rep = d.get("rep_dims_m") or {}
        if not isinstance(obj_id, str) or not obj_id or not isinstance(rep, dict):
            continue

        L = float(rep.get("L", 0) or 0)
        W = float(rep.get("W", 0) or 0)
        if L <= 0 or W <= 0:
            continue

        spec: dict[str, Any] = {
            "id": obj_id,
            "w": int(round(L * 1000)),
            "h": int(round(W * 1000)),
            "clearance_mm": int(d.get("clearance_mm", 0) or 0),
            "allowed_rotations": allowed.get(obj_id, [0, 90, 180, 270]),
            "collision": d.get("collision", "solid"),
        }

        f = facing.get(obj_id)
        if isinstance(f, dict):
            front = f.get("front")
            if front in {"top", "bottom", "left", "right"}:
                spec["front"] = front

        specs.append(spec)

    return specs


# ============================================================
# Constraint dispatch
# ============================================================


def evaluate_constraint(
    *,
    constraint: dict[str, Any],
    ctx: dict[str, Any],
    index: int,
    hard: bool,
) -> dict[str, Any]:
    ctype = constraint.get("type")
    weight = _constraint_weight(constraint, hard=hard)

    if ctype == "no_overlap":
        base = _eval_no_overlap(constraint, ctx)
    elif ctype == "contain_in":
        base = _eval_contain_in(constraint, ctx)
    elif ctype == "dock_to_edge":
        base = _eval_dock_to_edge(constraint, ctx)
    elif ctype == "anchor_side":
        base = _eval_anchor_side(constraint, ctx)
    elif ctype == "requires_access":
        base = _eval_requires_access(constraint, ctx)
    elif ctype == "prefer_near":
        base = _eval_prefer_near(constraint, ctx)
    elif ctype == "prefer_align_edge":
        base = _eval_prefer_align_edge(constraint, ctx)
    elif ctype == "prefer_facing":
        base = _eval_prefer_facing(constraint, ctx)
    else:
        base = {
            "satisfied": False if hard else True,
            "violation": 1_000_000.0 if hard else 0.0,
            "components": {"unsupported": 1.0},
            "debug": {"detail": f"Unsupported constraint type: {ctype}"},
            "subjects": _subjects_from_constraint(constraint),
        }

    violation = float(base.get("violation", 0.0) or 0.0)
    weighted_violation = violation * weight

    return {
        "index": int(index),
        "type": ctype,
        "hard": bool(hard),
        "weight": float(weight),
        "satisfied": bool(base.get("satisfied", False)) if hard else violation <= 0.0,
        "violation": float(round(violation, 6)),
        "weighted_violation": float(round(weighted_violation, 6)),
        "components": base.get("components", {}),
        "debug": base.get("debug", {}),
        "subjects": base.get("subjects", _subjects_from_constraint(constraint)),
        "constraint": constraint,
    }


def summarize_hard_evaluations(evals: list[dict[str, Any]]) -> dict[str, Any]:
    num_violations = sum(1 for e in evals if not e.get("satisfied", False))
    total_violation = sum(float(e.get("violation", 0.0) or 0.0) for e in evals)
    total_weighted_violation = sum(
        float(e.get("weighted_violation", 0.0) or 0.0) for e in evals
    )
    max_violation = max([float(e.get("violation", 0.0) or 0.0) for e in evals] or [0.0])
    max_weighted_violation = max(
        [float(e.get("weighted_violation", 0.0) or 0.0) for e in evals] or [0.0]
    )

    return {
        "num_constraints": len(evals),
        "num_violations": int(num_violations),
        "total_violation": float(round(total_violation, 6)),
        "total_weighted_violation": float(round(total_weighted_violation, 6)),
        "max_violation": float(round(max_violation, 6)),
        "max_weighted_violation": float(round(max_weighted_violation, 6)),
    }


def summarize_soft_evaluations(evals: list[dict[str, Any]]) -> dict[str, Any]:
    total_penalty = sum(float(e.get("violation", 0.0) or 0.0) for e in evals)
    total_weighted_penalty = sum(
        float(e.get("weighted_violation", 0.0) or 0.0) for e in evals
    )
    max_penalty = max([float(e.get("violation", 0.0) or 0.0) for e in evals] or [0.0])

    return {
        "num_constraints": len(evals),
        "total_penalty": float(round(total_penalty, 6)),
        "total_weighted_penalty": float(round(total_weighted_penalty, 6)),
        "max_penalty": float(round(max_penalty, 6)),
    }


# ============================================================
# Hard constraint evaluators
# ============================================================


def _eval_no_overlap(constraint: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    a = constraint.get("a")
    b = constraint.get("b")
    rects = ctx["rects_clear"] if ctx["use_clearance"] else ctx["rects"]

    if not _has_rects(rects, a, b):
        return _unknown_subject_eval(a=a, b=b)

    ra = rects[a]
    rb = rects[b]
    ix, iy, area = _intersection_stats(ra, rb)
    min_sep = _min_translation_to_separate(ra, rb)

    violation = float(area)
    return {
        "satisfied": area <= 0,
        "violation": violation,
        "components": {
            "overlap_area": float(area),
            "overlap_x": float(ix),
            "overlap_y": float(iy),
            "min_translation_mm": float(min_sep),
        },
        "debug": {
            "a_rect": _rect_dict(ra),
            "b_rect": _rect_dict(rb),
        },
        "subjects": {"a": a, "b": b},
    }


def _eval_contain_in(constraint: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    a = constraint.get("a")
    b = constraint.get("b")
    rects = ctx["rects"]

    if not _has_rects(rects, a, b):
        return _unknown_subject_eval(a=a, b=b)

    ra = rects[a]
    rb = rects[b]

    ax1, ay1, ax2, ay2 = ra
    bx1, by1, bx2, by2 = rb

    left_excess = max(0, bx1 - ax1)
    right_excess = max(0, ax2 - bx2)
    bottom_excess = max(0, by1 - ay1)
    top_excess = max(0, ay2 - by2)

    violation = float(left_excess + right_excess + bottom_excess + top_excess)

    return {
        "satisfied": violation <= 0.0,
        "violation": violation,
        "components": {
            "left_excess_mm": float(left_excess),
            "right_excess_mm": float(right_excess),
            "bottom_excess_mm": float(bottom_excess),
            "top_excess_mm": float(top_excess),
        },
        "debug": {
            "a_rect": _rect_dict(ra),
            "b_rect": _rect_dict(rb),
        },
        "subjects": {"a": a, "b": b},
    }


def _eval_dock_to_edge(
    constraint: dict[str, Any], ctx: dict[str, Any]
) -> dict[str, Any]:
    a = constraint.get("a")
    b = constraint.get("b")
    b_edge = constraint.get("b_edge")
    span = str(constraint.get("span") or "any")
    gap_min = int(constraint.get("gap_min", 0) or 0)
    gap_max = int(constraint.get("gap_max", 0) or 0)

    rects = ctx["rects"]
    placements = ctx["placements"]
    specs = ctx["specs"]
    facing_map = ctx["facing_map"]

    if (
        not _has_rects(rects, a, b)
        or b not in placements
        or not isinstance(b_edge, str)
    ):
        return _unknown_subject_eval(a=a, b=b)

    ra = rects[a]
    rb = rects[b]
    rot_b = placements[b]["rot"] % 360

    b_front_base = _get_front_base(b, facing_map, specs)
    base_side = _resolve_edge_token_to_base_side(b_edge, b_front_base)
    mapped_side = _rotate_side(base_side, rot_b)

    gap = _edge_gap(ra, rb, mapped_side)
    perp_overlap = _perpendicular_overlap_len(ra, rb, mapped_side)
    span_ok = _dock_span_ok(span=span, mapped_side=mapped_side, ra=ra, rb=rb)
    span_penalty = 0.0 if span_ok else 1.0
    edge_family_ok = _dock_edge_family_ok(span=span, mapped_side=mapped_side, rb=rb)
    edge_family_penalty = 0.0 if edge_family_ok else 1.0

    gap_penalty = max(0, gap_min - gap) + max(0, gap - gap_max)
    perp_penalty = 0 if perp_overlap > 0 else 1.0

    # Side penalty is mostly implicit in gap penalty already, but keeping it visible helps debugging.
    side_penalty = float(max(0, -gap)) if gap < 0 else 0.0

    violation = float(
        gap_penalty * 100.0
        + perp_penalty * 1000.0
        + span_penalty * 1000.0
        + edge_family_penalty * 500.0
        + side_penalty
    )

    return {
        "satisfied": (
            (gap_min <= gap <= gap_max)
            and (perp_overlap > 0)
            and span_ok
            and edge_family_ok
        ),
        "violation": violation,
        "components": {
            "gap_mm": float(gap),
            "gap_penalty_mm": float(gap_penalty),
            "perpendicular_overlap_mm": float(perp_overlap),
            "perpendicular_penalty": float(perp_penalty),
            "span_penalty": float(span_penalty),
            "edge_family_penalty": float(edge_family_penalty),
            "side_penalty_mm": float(side_penalty),
        },
        "debug": {
            "a_rect": _rect_dict(ra),
            "b_rect": _rect_dict(rb),
            "b_edge": b_edge,
            "span": span,
            "mapped_side": mapped_side,
            "b_front_base": b_front_base,
        },
        "subjects": {"a": a, "b": b},
    }


def _eval_anchor_side(
    constraint: dict[str, Any], ctx: dict[str, Any]
) -> dict[str, Any]:
    a = constraint.get("a")
    b = constraint.get("b")
    side = constraint.get("side")
    gap_min = int(constraint.get("gap_min", 0) or 0)
    gap_max = int(constraint.get("gap_max", 0) or 0)

    rects = ctx["rects"]
    placements = ctx["placements"]

    if not _has_rects(rects, a, b) or b not in placements or not isinstance(side, str):
        return _unknown_subject_eval(a=a, b=b)

    ra = rects[a]
    rb = rects[b]
    rot_b = placements[b]["rot"] % 360

    base_side, qualifier_local = _resolve_anchor_side(side)
    if base_side == "head":
        base_side = "top"
    elif base_side == "foot":
        base_side = "bottom"

    mapped_side = _rotate_side(base_side, rot_b)
    gap = _edge_gap(ra, rb, mapped_side)
    gap_penalty = max(0, gap_min - gap) + max(0, gap - gap_max)

    qualifier_world = None
    if qualifier_local in {"left", "right"}:
        qualifier_world = _rotate_side(qualifier_local, rot_b)

    qualifier_penalty = 0.0
    qualifier_ok = _qualifier_ok(
        qualifier_world=qualifier_world,
        mapped_side=mapped_side,
        ra=ra,
        rb=rb,
    )
    qualifier_penalty = 0.0 if qualifier_ok else 1.0

    violation = float(gap_penalty * 100.0 + qualifier_penalty * 1000.0)

    return {
        "satisfied": (gap_min <= gap <= gap_max) and qualifier_ok,
        "violation": violation,
        "components": {
            "gap_mm": float(gap),
            "gap_penalty_mm": float(gap_penalty),
            "qualifier_penalty": float(qualifier_penalty),
        },
        "debug": {
            "a_rect": _rect_dict(ra),
            "b_rect": _rect_dict(rb),
            "side": side,
            "mapped_side": mapped_side,
            "qualifier_local": qualifier_local,
            "qualifier_world": qualifier_world,
        },
        "subjects": {"a": a, "b": b},
    }


def _eval_requires_access(
    constraint: dict[str, Any], ctx: dict[str, Any]
) -> dict[str, Any]:
    oid = constraint.get("id")
    mode = constraint.get("mode", "front_clearance")

    if not isinstance(oid, str) or mode != "front_clearance":
        return _unknown_subject_eval(a=oid, b=None)

    placements = ctx["placements"]
    rects = ctx["rects"]
    rects_eff = ctx["rects_clear"] if ctx["use_clearance"] else ctx["rects"]
    specs = ctx["specs"]
    facing_map = ctx["facing_map"]
    grid_mm = ctx["grid_mm"]
    allow_clearance_overlap = ctx["allow_clearance_overlap"]

    if oid not in placements or oid not in rects:
        return _unknown_subject_eval(a=oid, b=None)

    clearance_rect = _build_front_clearance_rects(
        access_required_ids={oid},
        placements=placements,
        specs=specs,
        rects=rects,
        facing_map=facing_map,
        grid_mm=grid_mm,
        access_clearance_ratio=FIXED_ACCESS_CLEARANCE_RATIO,
    ).get(oid)

    if clearance_rect is None:
        return _unknown_subject_eval(a=oid, b=None)

    blockers: list[dict[str, Any]] = []
    total_block_area = 0
    max_block_area = 0

    for other_id, other_rect in rects_eff.items():
        if other_id == oid:
            continue
        if other_id in allow_clearance_overlap.get(oid, set()):
            continue
        if not _is_floor_occupying(specs.get(other_id, {})):
            continue

        ix, iy, area = _intersection_stats(clearance_rect, other_rect)
        if area > 0:
            blockers.append(
                {
                    "blocker": other_id,
                    "intersection_area": int(area),
                    "intersection_x": int(ix),
                    "intersection_y": int(iy),
                }
            )
            total_block_area += area
            max_block_area = max(max_block_area, area)

    # Penalize both total blocked area and blocker count.
    violation = float(total_block_area + len(blockers) * 1000.0)

    return {
        "satisfied": len(blockers) == 0,
        "violation": violation,
        "components": {
            "num_blockers": float(len(blockers)),
            "total_blocked_area": float(total_block_area),
            "max_blocked_area": float(max_block_area),
        },
        "debug": {
            "clearance_rect": _rect_dict(clearance_rect),
            "blockers": blockers,
        },
        "subjects": {"id": oid},
    }


# ============================================================
# Soft constraint evaluators
# ============================================================


def _eval_prefer_near(
    constraint: dict[str, Any], ctx: dict[str, Any]
) -> dict[str, Any]:
    a = constraint.get("a")
    b = constraint.get("b")
    rects = ctx["rects"]

    if not _has_rects(rects, a, b):
        return _unknown_subject_eval(a=a, b=b, hard=False)

    ra = rects[a]
    rb = rects[b]

    edge_gap = _rect_edge_gap(ra, rb)
    center_dist = _center_distance(ra, rb)

    # Prefer using edge gap as main penalty.
    penalty = float(edge_gap)

    return {
        "satisfied": penalty <= 0.0,
        "violation": penalty,
        "components": {
            "edge_gap_mm": float(edge_gap),
            "center_distance_mm": float(round(center_dist, 6)),
        },
        "debug": {
            "a_rect": _rect_dict(ra),
            "b_rect": _rect_dict(rb),
        },
        "subjects": {"a": a, "b": b},
    }


def _eval_prefer_align_edge(
    constraint: dict[str, Any], ctx: dict[str, Any]
) -> dict[str, Any]:
    a = constraint.get("a")
    b = constraint.get("b")
    edge = constraint.get("edge")

    rects = ctx["rects"]
    placements = ctx["placements"]
    specs = ctx["specs"]
    facing_map = ctx["facing_map"]

    if not _has_rects(rects, a, b) or b not in placements or not isinstance(edge, str):
        return _unknown_subject_eval(a=a, b=b, hard=False)

    ra = rects[a]
    rb = rects[b]
    rot_b = placements[b]["rot"] % 360

    b_front_base = _get_front_base(b, facing_map, specs)
    base_side = _resolve_edge_token_to_base_side(edge, b_front_base)
    mapped_side = _rotate_side(base_side, rot_b)

    gap = _edge_gap(ra, rb, mapped_side)
    perp_offset = _perpendicular_center_offset(ra, rb, mapped_side)

    # Soft preference:
    # - prefer centered alignment on the target edge
    # - prefer small positive/zero gap
    gap_penalty = abs(gap)
    align_penalty = abs(perp_offset)

    penalty = float(gap_penalty + align_penalty)

    return {
        "satisfied": penalty <= 0.0,
        "violation": penalty,
        "components": {
            "gap_mm": float(gap),
            "gap_penalty_mm": float(gap_penalty),
            "perpendicular_center_offset_mm": float(perp_offset),
            "alignment_penalty_mm": float(align_penalty),
        },
        "debug": {
            "a_rect": _rect_dict(ra),
            "b_rect": _rect_dict(rb),
            "edge": edge,
            "mapped_side": mapped_side,
        },
        "subjects": {"a": a, "b": b},
    }


def _eval_prefer_facing(
    constraint: dict[str, Any], ctx: dict[str, Any]
) -> dict[str, Any]:
    a = constraint.get("a")
    b = constraint.get("b")
    mode = constraint.get("mode", "face_each_other")

    rects = ctx["rects"]
    placements = ctx["placements"]
    specs = ctx["specs"]
    facing_map = ctx["facing_map"]

    if (
        not _has_rects(rects, a, b)
        or a not in placements
        or b not in placements
        or not isinstance(mode, str)
    ):
        return _unknown_subject_eval(a=a, b=b, hard=False)

    ra = rects[a]
    rb = rects[b]

    front_a = _global_front_side(
        obj_id=a,
        placement=placements[a],
        specs=specs,
        facing_map=facing_map,
    )
    front_b = _global_front_side(
        obj_id=b,
        placement=placements[b],
        specs=specs,
        facing_map=facing_map,
    )

    va = _side_to_unit_vector(front_a)
    vb = _side_to_unit_vector(front_b)

    penalty = 0.0
    detail = {}

    if mode == "face_each_other":
        # exact opposite directions => dot = -1 => penalty 0
        dot = va[0] * vb[0] + va[1] * vb[1]
        facing_penalty = dot + 1.0  # -1 -> 0, 0 -> 1, 1 -> 2

        # also prefer that A lies roughly in front of B and vice versa
        ca = _rect_center(ra)
        cb = _rect_center(rb)
        ab = (cb[0] - ca[0], cb[1] - ca[1])

        dir_penalty_a = 0.0 if _dot_nonneg(va, ab) else 1.0
        dir_penalty_b = 0.0 if _dot_nonneg(vb, (-ab[0], -ab[1])) else 1.0

        penalty = float(
            facing_penalty * 1000.0 + dir_penalty_a * 500.0 + dir_penalty_b * 500.0
        )
        detail = {
            "front_a": front_a,
            "front_b": front_b,
            "dot": float(dot),
            "facing_penalty": float(facing_penalty),
            "direction_penalty_a": float(dir_penalty_a),
            "direction_penalty_b": float(dir_penalty_b),
        }
    else:
        # Generic fallback
        penalty = 0.0
        detail = {
            "front_a": front_a,
            "front_b": front_b,
            "mode": mode,
        }

    return {
        "satisfied": penalty <= 0.0,
        "violation": penalty,
        "components": detail,
        "debug": {
            "a_rect": _rect_dict(ra),
            "b_rect": _rect_dict(rb),
        },
        "subjects": {"a": a, "b": b},
    }


# ============================================================
# Layout quality
# ============================================================


def compute_layout_quality(rects: dict[str, Rect]) -> dict[str, Any]:
    if not rects:
        return {
            "bbox": {
                "min_x": 0,
                "min_y": 0,
                "max_x": 0,
                "max_y": 0,
                "span_x": 0,
                "span_y": 0,
                "area_mm2": 0,
                "aspect_ratio": 1.0,
                "max_span_mm": 0,
            },
            "outline": {
                "area_mm2": 0,
                "perimeter_mm": 0,
            },
            "convex_hull": {
                "area_mm2": 0,
                "perimeter_mm": 0.0,
                "points": [],
            },
            "fill_ratio_bbox": 1.0,
            "fill_ratio_hull": 1.0,
            "compactness_perimeter2_over_4piA": 1.0,
            "compact_score": 0,
        }

    xs_all: list[int] = []
    ys_all: list[int] = []
    points: list[Point] = []

    for r in rects.values():
        x1, y1, x2, y2 = r
        xs_all.extend([x1, x2])
        ys_all.extend([y1, y2])
        points.extend([(x1, y1), (x1, y2), (x2, y1), (x2, y2)])

    min_x = min(xs_all)
    min_y = min(ys_all)
    max_x = max(xs_all)
    max_y = max(ys_all)

    span_x = max_x - min_x
    span_y = max_y - min_y
    bbox_area = max(0, span_x) * max(0, span_y)
    min_span_pos = max(1, min(span_x, span_y))
    aspect_ratio = max(span_x, span_y) / float(min_span_pos)

    outline_area, outline_perimeter = _union_area_perimeter(list(rects.values()))
    hull_points = _convex_hull(points)
    hull_area = _polygon_area(hull_points)
    hull_perimeter = _polygon_perimeter(hull_points)

    fill_ratio_bbox = float(outline_area) / float(bbox_area) if bbox_area > 0 else 1.0
    fill_ratio_hull = float(outline_area) / float(hull_area) if hull_area > 0 else 1.0

    compactness = (
        (outline_perimeter * outline_perimeter) / (4.0 * math.pi * outline_area)
        if outline_area > 0
        else 1.0
    )

    compact_score = int(
        bbox_area * 1000
        + max(span_x, span_y) * 100
        + outline_perimeter * 10
        + (1.0 - min(1.0, fill_ratio_bbox)) * 1_000_000
        + (1.0 - min(1.0, fill_ratio_hull)) * 500_000
        + max(0.0, aspect_ratio - 1.0) * 100_000
    )

    return {
        "bbox": {
            "min_x": int(min_x),
            "min_y": int(min_y),
            "max_x": int(max_x),
            "max_y": int(max_y),
            "span_x": int(span_x),
            "span_y": int(span_y),
            "area_mm2": int(bbox_area),
            "aspect_ratio": float(round(aspect_ratio, 6)),
            "max_span_mm": int(max(span_x, span_y)),
        },
        "outline": {
            "area_mm2": int(outline_area),
            "perimeter_mm": int(outline_perimeter),
        },
        "convex_hull": {
            "area_mm2": int(round(hull_area)),
            "perimeter_mm": float(round(hull_perimeter, 6)),
            "points": [{"x": int(x), "y": int(y)} for x, y in hull_points],
        },
        "fill_ratio_bbox": float(round(fill_ratio_bbox, 6)),
        "fill_ratio_hull": float(round(fill_ratio_hull, 6)),
        "compactness_perimeter2_over_4piA": float(round(compactness, 6)),
        "compact_score": int(compact_score),
    }


# ============================================================
# Utilities
# ============================================================


def _constraint_weight(constraint: dict[str, Any], *, hard: bool) -> float:
    if hard:
        priority = constraint.get("priority")
        if isinstance(priority, (int, float)):
            return float(priority)
        return 1.0

    weight = constraint.get("weight")
    if isinstance(weight, (int, float)):
        return float(weight)
    return 1.0


def _subjects_from_constraint(constraint: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in ("a", "b", "id"):
        if k in constraint:
            out[k] = constraint.get(k)
    return out


def _unknown_subject_eval(
    *,
    a: Any,
    b: Any,
    hard: bool = True,
) -> dict[str, Any]:
    return {
        "satisfied": False if hard else True,
        "violation": 1_000_000.0 if hard else 0.0,
        "components": {"unknown_subject": 1.0},
        "debug": {"detail": "Unknown or missing object reference"},
        "subjects": {"a": a, "b": b},
    }


def _has_rects(rects: dict[str, Rect], a: Any, b: Any) -> bool:
    return isinstance(a, str) and isinstance(b, str) and a in rects and b in rects


def _normalize_objects(
    objects: list[dict[str, Any]] | dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if isinstance(objects, dict):
        return {str(k): dict(v) for k, v in objects.items() if isinstance(v, dict)}

    output: dict[str, dict[str, Any]] = {}
    for item in objects:
        if not isinstance(item, dict):
            continue
        obj_id = item.get("id") or item.get("object_id") or item.get("type")
        if isinstance(obj_id, str) and obj_id:
            spec = dict(item)
            spec["id"] = obj_id
            output[obj_id] = spec
    return output


def _normalize_placements(
    placements: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for item in placements:
        if not isinstance(item, dict):
            continue
        obj_id = item.get("id") or item.get("instance_id") or item.get("object_id")
        if not isinstance(obj_id, str) or not obj_id:
            continue
        output[obj_id] = {
            "x": int(item.get("x", 0)),
            "y": int(item.get("y", 0)),
            "rot": int(item.get("rot", 0)) % 360,
        }
    return output


def _build_rects(
    placements: dict[str, dict[str, Any]],
    specs: dict[str, dict[str, Any]],
    *,
    use_clearance: bool,
) -> dict[str, Rect]:
    rects: dict[str, Rect] = {}
    for obj_id, placement in placements.items():
        spec = specs.get(obj_id)
        if spec is None:
            continue

        w = int(spec.get("w", 0) or 0)
        h = int(spec.get("h", 0) or 0)
        rot = int(placement.get("rot", 0)) % 360

        if rot in (90, 270):
            w, h = h, w

        x = int(placement.get("x", 0))
        y = int(placement.get("y", 0))
        clearance = int(spec.get("clearance_mm", 0) or 0) if use_clearance else 0

        rects[obj_id] = (
            x - clearance,
            y - clearance,
            x + w + clearance,
            y + h + clearance,
        )
    return rects


def _rect_dict(r: Rect) -> dict[str, int]:
    return {
        "x1": int(r[0]),
        "y1": int(r[1]),
        "x2": int(r[2]),
        "y2": int(r[3]),
        "w": int(r[2] - r[0]),
        "h": int(r[3] - r[1]),
    }


def _rect_area(r: Rect) -> int:
    w = max(0, int(r[2] - r[0]))
    h = max(0, int(r[3] - r[1]))
    return w * h


def _intersection_stats(a: Rect, b: Rect) -> Tuple[int, int, int]:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    ix = max(0, ix2 - ix1)
    iy = max(0, iy2 - iy1)
    return ix, iy, ix * iy


def _min_translation_to_separate(a: Rect, b: Rect) -> int:
    ix, iy, area = _intersection_stats(a, b)
    if area <= 0:
        return 0
    return min(ix, iy)


def _rect_center(rect: Rect) -> tuple[float, float]:
    x1, y1, x2, y2 = rect
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _center_distance(a: Rect, b: Rect) -> float:
    ax, ay = _rect_center(a)
    bx, by = _rect_center(b)
    return math.hypot(ax - bx, ay - by)


def _rect_edge_gap(a: Rect, b: Rect) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    dx = max(0, max(bx1 - ax2, ax1 - bx2))
    dy = max(0, max(by1 - ay2, ay1 - by2))

    if dx == 0 and dy == 0:
        return 0
    if dx == 0:
        return dy
    if dy == 0:
        return dx
    return int(round(math.hypot(dx, dy)))


def _resolve_anchor_side(side: str) -> tuple[str, str | None]:
    if side in {"left", "right", "top", "bottom"}:
        return side, None
    if side.startswith("head_"):
        return "head", side.split("_", 1)[1]
    if side.startswith("foot_"):
        return "foot", side.split("_", 1)[1]
    if side in {"head", "foot"}:
        return side, None
    return side, None


def _rotate_side(side: str, rot: int) -> str:
    mapping = {
        0: {"top": "top", "right": "right", "bottom": "bottom", "left": "left"},
        90: {"top": "left", "right": "top", "bottom": "right", "left": "bottom"},
        180: {"top": "bottom", "right": "left", "bottom": "top", "left": "right"},
        270: {"top": "right", "right": "bottom", "bottom": "left", "left": "top"},
    }
    return mapping.get(rot % 360, mapping[0]).get(side, side)


def _edge_gap(ra: Rect, rb: Rect, side: str) -> int:
    ax1, ay1, ax2, ay2 = ra
    bx1, by1, bx2, by2 = rb

    if side == "left":
        return bx1 - ax2
    if side == "right":
        return ax1 - bx2
    if side == "top":
        return ay1 - by2
    if side == "bottom":
        return by1 - ay2
    return 0


def _perpendicular_overlap_len(ra: Rect, rb: Rect, side: str) -> int:
    ax1, ay1, ax2, ay2 = ra
    bx1, by1, bx2, by2 = rb

    if side in {"left", "right"}:
        return max(0, min(ay2, by2) - max(ay1, by1))
    if side in {"top", "bottom"}:
        return max(0, min(ax2, bx2) - max(ax1, bx1))
    return 0


def _perpendicular_center_offset(ra: Rect, rb: Rect, side: str) -> float:
    ax, ay = _rect_center(ra)
    bx, by = _rect_center(rb)

    if side in {"left", "right"}:
        return ay - by
    if side in {"top", "bottom"}:
        return ax - bx
    return 0.0


def _anchor_zone_bounds(
    free_span: int,
    qualifier_world: str | None,
) -> tuple[int, int]:
    free_span = max(0, int(free_span))

    if qualifier_world in {"left", "bottom"}:
        return 0, free_span // 3
    if qualifier_world in {"right", "top"}:
        return (2 * free_span + 2) // 3, free_span

    return 0, free_span


def _dock_span_zone_bounds(free_span: int, span: str) -> tuple[int, int]:
    free_span = max(0, int(free_span))
    if span == "center":
        return free_span // 3, (2 * free_span + 2) // 3
    if span == "left":
        return 0, free_span // 3
    if span == "right":
        return (2 * free_span + 2) // 3, free_span
    return 0, free_span


def _dock_span_ok(*, span: str, mapped_side: str, ra: Rect, rb: Rect) -> bool:
    if span not in {"center", "left", "right"}:
        return True

    ax1, ay1, ax2, ay2 = ra
    bx1, by1, bx2, by2 = rb
    aw = ax2 - ax1
    ah = ay2 - ay1
    bw = bx2 - bx1
    bh = by2 - by1

    if mapped_side in {"top", "bottom"}:
        free_span = max(0, bw - aw)
        rel = ax1 - bx1
        z_lo, z_hi = _dock_span_zone_bounds(free_span, span)
        return z_lo <= rel <= z_hi

    if mapped_side in {"left", "right"}:
        free_span = max(0, bh - ah)
        rel = ay1 - by1
        z_lo, z_hi = _dock_span_zone_bounds(free_span, span)
        return z_lo <= rel <= z_hi

    return True


def _dock_edge_family_ok(*, span: str, mapped_side: str, rb: Rect) -> bool:
    if span not in {"short_edge", "long_edge"}:
        return True

    bx1, by1, bx2, by2 = rb
    bw = bx2 - bx1
    bh = by2 - by1
    edge_len = bh if mapped_side in {"left", "right"} else bw
    other_len = bw if mapped_side in {"left", "right"} else bh

    if span == "short_edge":
        return edge_len <= other_len
    return edge_len >= other_len


def _qualifier_ok(
    qualifier_world: str | None,
    mapped_side: str,
    ra: Rect,
    rb: Rect,
) -> bool:
    ax1, ay1, ax2, ay2 = ra
    bx1, by1, bx2, by2 = rb

    aw = ax2 - ax1
    ah = ay2 - ay1
    bw = bx2 - bx1
    bh = by2 - by1

    if mapped_side in {"top", "bottom"}:
        if qualifier_world not in {"left", "right"}:
            return True
        free_span = max(0, bw - aw)
        rel = ax1 - bx1
        z_lo, z_hi = _anchor_zone_bounds(free_span, qualifier_world)
        return z_lo <= rel <= z_hi

    if mapped_side in {"left", "right"}:
        if qualifier_world not in {"bottom", "top"}:
            return True
        free_span = max(0, bh - ah)
        rel = ay1 - by1
        z_lo, z_hi = _anchor_zone_bounds(free_span, qualifier_world)
        return z_lo <= rel <= z_hi

    return True


def _get_front_base(
    obj_id: str,
    facing_map: dict[str, Any],
    specs: dict[str, dict[str, Any]],
) -> str:
    f = facing_map.get(obj_id) if isinstance(facing_map, dict) else None
    if isinstance(f, dict):
        front = f.get("front")
        if front in {"top", "bottom", "left", "right"}:
            return str(front)

    spec_front = specs.get(obj_id, {}).get("front")
    if spec_front in {"top", "bottom", "left", "right"}:
        return str(spec_front)

    return "top"


def _resolve_edge_token_to_base_side(edge_token: str, front_base: str) -> str:
    if edge_token in {"top", "bottom", "left", "right"}:
        return edge_token
    if edge_token == "front":
        return front_base
    if edge_token == "back":
        return _opposite_base_side(front_base)
    return "top"


def _opposite_base_side(side: str) -> str:
    return {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}.get(
        side, side
    )


def _global_front_side(
    *,
    obj_id: str,
    placement: dict[str, Any],
    specs: dict[str, dict[str, Any]],
    facing_map: dict[str, Any],
) -> str:
    base_front = _get_front_base(obj_id, facing_map, specs)
    return _rotate_side(base_front, int(placement.get("rot", 0)) % 360)


def _side_to_unit_vector(side: str) -> tuple[int, int]:
    if side == "top":
        return (0, 1)
    if side == "bottom":
        return (0, -1)
    if side == "left":
        return (-1, 0)
    if side == "right":
        return (1, 0)
    return (0, 1)


def _dot_nonneg(v: tuple[int, int], w: tuple[float, float]) -> bool:
    return (v[0] * w[0] + v[1] * w[1]) >= 0


def _is_floor_occupying(spec: dict[str, Any]) -> bool:
    c = spec.get("collision")
    if not isinstance(c, str):
        return True
    return c.lower() != "on_top"


def _collect_access_required_ids(
    *,
    hard_constraints: list[dict[str, Any]],
    cluster_rules: dict[str, Any] | None,
) -> set[str]:
    ids: set[str] = set()

    if isinstance(cluster_rules, dict):
        ar = cluster_rules.get("access_requirements")
        if isinstance(ar, list):
            for item in ar:
                if not isinstance(item, dict):
                    continue
                if item.get("type") != "front_clearance":
                    continue
                if item.get("required", False) is not True:
                    continue
                oid = item.get("id")
                if isinstance(oid, str) and oid:
                    ids.add(oid)

    for c in hard_constraints:
        if not isinstance(c, dict):
            continue
        if c.get("type") != "requires_access":
            continue
        if c.get("mode") != "front_clearance":
            continue
        oid = c.get("id")
        if isinstance(oid, str) and oid:
            ids.add(oid)

    return ids


def _build_allow_clearance_overlap(
    *,
    hard_constraints: list[dict[str, Any]],
    facing_map: dict[str, Any],
    specs: dict[str, dict[str, Any]],
) -> dict[str, set[str]]:
    allow: dict[str, set[str]] = {}

    for c in hard_constraints:
        if not isinstance(c, dict):
            continue
        if c.get("type") != "dock_to_edge":
            continue

        a = c.get("a")
        b = c.get("b")
        b_edge = c.get("b_edge")

        if not (isinstance(a, str) and isinstance(b, str) and isinstance(b_edge, str)):
            continue

        b_front_base = _get_front_base(b, facing_map, specs)
        base_side = _resolve_edge_token_to_base_side(b_edge, b_front_base)

        if base_side == b_front_base:
            allow.setdefault(b, set()).add(a)

    return allow


def _build_front_clearance_rects(
    *,
    access_required_ids: set[str],
    placements: dict[str, dict[str, Any]],
    specs: dict[str, dict[str, Any]],
    rects: dict[str, Rect],
    facing_map: dict[str, Any],
    grid_mm: int,
    access_clearance_ratio: float,
) -> dict[str, Rect]:
    out: dict[str, Rect] = {}
    r = max(0.0, min(2.0, float(access_clearance_ratio)))

    for oid in access_required_ids:
        if oid not in placements or oid not in rects:
            continue

        spec = specs.get(oid, {})
        w0 = int(spec.get("w", 0) or 0)
        h0 = int(spec.get("h", 0) or 0)
        if w0 <= 0 or h0 <= 0:
            continue

        long_edge = max(w0, h0)
        side_len = max(1, int(long_edge * r))
        side_len = _snap_up(side_len, grid_mm)

        x1, y1, x2, y2 = rects[oid]
        cx, cy = _rect_center((x1, y1, x2, y2))

        rot = placements[oid]["rot"] % 360
        front_base = _get_front_base(oid, facing_map, specs)
        front_global = _rotate_side(front_base, rot)

        if front_global == "top":
            half = side_len / 2.0
            rx1 = int(round(cx - half))
            rx2 = rx1 + side_len
            ry1 = y2
            ry2 = y2 + side_len
        elif front_global == "bottom":
            half = side_len / 2.0
            rx1 = int(round(cx - half))
            rx2 = rx1 + side_len
            ry2 = y1
            ry1 = y1 - side_len
        elif front_global == "right":
            half = side_len / 2.0
            ry1 = int(round(cy - half))
            ry2 = ry1 + side_len
            rx1 = x2
            rx2 = x2 + side_len
        else:  # left
            half = side_len / 2.0
            ry1 = int(round(cy - half))
            ry2 = ry1 + side_len
            rx2 = x1
            rx1 = x1 - side_len

        out[oid] = (rx1, ry1, rx2, ry2)

    return out


def _snap_up(v: int, grid_mm: int) -> int:
    if grid_mm <= 0:
        return v
    if v % grid_mm == 0:
        return v
    return ((v // grid_mm) + 1) * grid_mm


def _snap_down(v: int, grid_mm: int) -> int:
    if grid_mm <= 0:
        return v
    return v - (v % grid_mm)


def _snap_nearest(v: int, grid_mm: int) -> int:
    if grid_mm <= 0:
        return v
    down = _snap_down(v, grid_mm)
    up = _snap_up(v, grid_mm)
    if abs(v - down) < abs(up - v):
        return down
    return up


def _snap_delta(delta: int, grid_mm: int) -> int:
    if delta == 0:
        return 0
    if grid_mm <= 0:
        return int(delta)
    sign = 1 if delta > 0 else -1
    return sign * _snap_up(abs(int(delta)), grid_mm)


def _union_area_perimeter(rects: list[Rect]) -> tuple[int, int]:
    if not rects:
        return 0, 0

    xs = sorted({x for r in rects for x in (r[0], r[2])})
    ys = sorted({y for r in rects for y in (r[1], r[3])})

    if len(xs) < 2 or len(ys) < 2:
        return 0, 0

    x_index = {x: i for i, x in enumerate(xs)}
    y_index = {y: i for i, y in enumerate(ys)}
    covered = [[False for _ in range(len(ys) - 1)] for _ in range(len(xs) - 1)]

    for x1, y1, x2, y2 in rects:
        if x2 <= x1 or y2 <= y1:
            continue
        ix1 = x_index[x1]
        ix2 = x_index[x2]
        iy1 = y_index[y1]
        iy2 = y_index[y2]
        for i in range(ix1, ix2):
            row = covered[i]
            for j in range(iy1, iy2):
                row[j] = True

    area = 0
    perimeter = 0

    for i in range(len(xs) - 1):
        dx = xs[i + 1] - xs[i]
        for j in range(len(ys) - 1):
            if not covered[i][j]:
                continue
            dy = ys[j + 1] - ys[j]
            area += dx * dy

            if i == 0 or not covered[i - 1][j]:
                perimeter += dy
            if i == len(xs) - 2 or not covered[i + 1][j]:
                perimeter += dy
            if j == 0 or not covered[i][j - 1]:
                perimeter += dx
            if j == len(ys) - 2 or not covered[i][j + 1]:
                perimeter += dx

    return int(area), int(perimeter)


def _cross(o: Point, a: Point, b: Point) -> int:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _convex_hull(points: Iterable[Point]) -> list[Point]:
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts

    lower: list[Point] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[Point] = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def _polygon_area(poly: list[Point]) -> float:
    if len(poly) < 3:
        return 0.0
    s = 0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _polygon_perimeter(poly: list[Point]) -> float:
    if len(poly) <= 1:
        return 0.0
    total = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        total += math.hypot(x2 - x1, y2 - y1)
    return total


# ============================================================
# Public aliases / exports for tool integration
# ============================================================


def normalize_objects(
    objects: list[dict[str, Any]] | dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return _normalize_objects(objects)


def normalize_placements(
    placements: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return _normalize_placements(placements)


def build_rects(
    placements: dict[str, dict[str, Any]],
    specs: dict[str, dict[str, Any]],
    *,
    use_clearance: bool,
) -> dict[str, Rect]:
    return _build_rects(placements, specs, use_clearance=use_clearance)


def rect_dict(r: Rect) -> dict[str, int]:
    return _rect_dict(r)


def intersection_stats(a: Rect, b: Rect) -> Tuple[int, int, int]:
    return _intersection_stats(a, b)


def rect_area(rect: Rect) -> int:
    return _rect_area(rect)


def resolve_anchor_side(side: str) -> tuple[str, str | None]:
    return _resolve_anchor_side(side)


def rotate_side(side: str, rot: int) -> str:
    return _rotate_side(side, rot)


def edge_gap(ra: Rect, rb: Rect, side: str) -> int:
    return _edge_gap(ra, rb, side)


def perpendicular_overlap_len(ra: Rect, rb: Rect, side: str) -> int:
    return _perpendicular_overlap_len(ra, rb, side)


def get_front_base(
    obj_id: str,
    facing_map: dict[str, Any],
    specs: dict[str, dict[str, Any]],
) -> str:
    return _get_front_base(obj_id, facing_map, specs)


def resolve_edge_token_to_base_side(edge_token: str, front_base: str) -> str:
    return _resolve_edge_token_to_base_side(edge_token, front_base)


def snap_delta(delta: int, grid_mm: int) -> int:
    return _snap_delta(delta, grid_mm)


def snap_nearest(v: int, grid_mm: int) -> int:
    return _snap_nearest(v, grid_mm)


def snap_up(v: int, grid_mm: int) -> int:
    return _snap_up(v, grid_mm)


def snap_down(v: int, grid_mm: int) -> int:
    return _snap_down(v, grid_mm)


def build_front_clearance_rects(
    *,
    access_required_ids: set[str],
    placements: dict[str, dict[str, Any]],
    specs: dict[str, dict[str, Any]],
    rects: dict[str, Rect],
    facing_map: dict[str, Any],
    grid_mm: int,
    access_clearance_ratio: float = FIXED_ACCESS_CLEARANCE_RATIO,
) -> dict[str, Rect]:
    access_clearance_ratio = FIXED_ACCESS_CLEARANCE_RATIO
    return _build_front_clearance_rects(
        access_required_ids=access_required_ids,
        placements=placements,
        specs=specs,
        rects=rects,
        facing_map=facing_map,
        grid_mm=grid_mm,
        access_clearance_ratio=access_clearance_ratio,
    )


def collect_access_required_ids(
    *,
    hard_constraints: list[dict[str, Any]],
    cluster_rules: dict[str, Any] | None,
) -> set[str]:
    return _collect_access_required_ids(
        hard_constraints=hard_constraints,
        cluster_rules=cluster_rules,
    )


def build_allow_clearance_overlap(
    *,
    hard_constraints: list[dict[str, Any]],
    facing_map: dict[str, Any],
    specs: dict[str, dict[str, Any]],
) -> dict[str, set[str]]:
    return _build_allow_clearance_overlap(
        hard_constraints=hard_constraints,
        facing_map=facing_map,
        specs=specs,
    )


def is_floor_occupying(spec: dict[str, Any]) -> bool:
    return _is_floor_occupying(spec)


__all__ = [
    "FIXED_ACCESS_CLEARANCE_RATIO",
    "score_cluster_from_cluster_json",
    "score_cluster_constraints",
    "build_object_specs_from_cluster",
    "normalize_objects",
    "normalize_placements",
    "build_rects",
    "rect_dict",
    "intersection_stats",
    "rect_area",
    "resolve_anchor_side",
    "rotate_side",
    "edge_gap",
    "perpendicular_overlap_len",
    "get_front_base",
    "resolve_edge_token_to_base_side",
    "snap_delta",
    "snap_nearest",
    "snap_up",
    "snap_down",
    "build_front_clearance_rects",
    "collect_access_required_ids",
    "build_allow_clearance_overlap",
    "is_floor_occupying",
]
