from __future__ import annotations

from typing import Any, Tuple

from cluster_composer.constraint_score import (
    FIXED_ACCESS_CLEARANCE_RATIO,
    _anchor_zone_bounds,
    _build_allow_clearance_overlap,
    _build_rects,
    _collect_access_required_ids,
    _get_front_base,
    _intersection_stats,
    _is_floor_occupying,
    _normalize_objects,
    _normalize_placements,
    _perpendicular_center_offset,
    _perpendicular_overlap_len,
    _rect_area,
    _rect_dict,
    _rect_edge_gap,
    _resolve_anchor_side,
    _resolve_edge_token_to_base_side,
    _rotate_side,
    _snap_delta,
    _snap_down,
    _snap_nearest,
    _snap_up,
    score_cluster_constraints,
)

Rect = Tuple[int, int, int, int]


def local_cluster_verifier(
    *,
    hard_constraints: list[dict[str, Any]] | None,
    soft_constraints: list[dict[str, Any]] | None = None,
    objects: list[dict[str, Any]] | dict[str, dict[str, Any]] | None,
    local_placements: list[dict[str, Any]] | None,
    grid_mm: int,
    use_clearance: bool = True,
    # optional:
    cluster_rules: dict[str, Any] | None = None,
    # kept only for backward compatibility; ignored internally
    access_clearance_ratio: float | None = FIXED_ACCESS_CLEARANCE_RATIO,
) -> dict[str, Any]:
    """
    Validate local cluster placements against constraints in local coordinates.

    This version delegates constraint scoring/evaluation to constraint_score.py,
    while keeping the old verifier-facing API and move suggestion interface.

    Conventions:
    - rot is CCW in degrees (0/90/180/270).
    - Rectangles are axis-aligned after applying rot by swapping (w,h) for 90/270.
    - Coordinates are in mm integers.
    - (x,y) is the bottom-left corner of the rotated rectangle.
    - w is extent in +X, h is extent in +Y (after rotation swap).

    Supported hard constraints:
    - no_overlap(a,b)
    - contain_in(a,b)
    - anchor_side(a,b,side,gap_min,gap_max)
    - dock_to_edge(a,b,b_edge,span,gap_min,gap_max)
    - requires_access(id, mode=front_clearance)

    Supported soft constraints:
    - prefer_near(a,b,weight)
    - prefer_align_edge(a,b,edge,weight)
    - prefer_facing(...)

    Important behavior:
    - Even if no explicit no_overlap(a,b) is provided, floor-occupying objects
      are still checked for implicit body overlap globally.
    - Objects with collision="on_top" are excluded from implicit floor-body overlap checks.
    - If A is docked to the FRONT of B, A is allowed inside B's front_clearance
      and is not treated as a front_clearance blocker for B.
    - access_clearance_ratio is FIXED internally to 0.25.
    Orientation behavior:
    - orientation_meta is NOT an input of this verifier and is NOT hard-validated here.
    - This verifier DOES return orientation_inference debug derived from the actual verified local layout.
    - orientation_inference exposes per-object effective front_local / axis_local after applying the final object rotation.
    - The composer must build final orientation_meta from this verified geometric/semantic evidence, not from hardcoded defaults.
    """
    del access_clearance_ratio  # force fixed policy internally

    errors: list[dict[str, Any]] = []
    debug: dict[str, Any] = {}

    effective_access_clearance_ratio = FIXED_ACCESS_CLEARANCE_RATIO

    # -------------------------
    # Defensive coercion
    # -------------------------
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

    # -------------------------
    # SAFETY: missing specs / invalid dims -> fail fast
    # -------------------------
    missing_specs: list[str] = [oid for oid in placements.keys() if oid not in specs]

    invalid_dims: list[dict[str, Any]] = []
    for oid in placements.keys():
        spec = specs.get(oid)
        if not isinstance(spec, dict):
            continue
        w = int(spec.get("w", 0) or 0)
        h = int(spec.get("h", 0) or 0)
        if w <= 0 or h <= 0:
            invalid_dims.append({"id": oid, "w": w, "h": h})

    if missing_specs or invalid_dims:
        if missing_specs:
            errors.append(
                {
                    "code": "MISSING_OBJECT_SPECS",
                    "a": None,
                    "b": None,
                    "detail": f"Missing specs for ids: {missing_specs}",
                }
            )
        if invalid_dims:
            errors.append(
                {
                    "code": "INVALID_DIMS",
                    "a": None,
                    "b": None,
                    "detail": f"Objects have non-positive w/h: {invalid_dims}",
                }
            )
        errors = _sort_errors(errors)
        debug = {
            "grid_mm": int(grid_mm),
            "missing_specs": missing_specs,
            "invalid_dims": invalid_dims,
            "spec_ids": sorted(list(specs.keys())),
            "placement_ids": sorted(list(placements.keys())),
            "access_clearance_ratio": float(effective_access_clearance_ratio),
        }
        return {
            "result": "INVALID",
            "errors": errors,
            "debug": debug,
            "quality": _empty_quality(),
            "suggested_moves": [],
        }

    # -------------------------
    # Facing map / access collection
    # -------------------------
    facing_map: dict[str, Any] = {}
    if isinstance(cluster_rules, dict):
        f = cluster_rules.get("facing")
        if isinstance(f, dict):
            facing_map = f

    access_required_ids = _collect_access_required_ids(
        hard_constraints=hard_constraints,
        cluster_rules=cluster_rules,
    )

    allow_clearance_overlap = _build_allow_clearance_overlap(
        hard_constraints=hard_constraints,
        facing_map=facing_map,
        specs=specs,
    )

    dock_evals: list[dict[str, Any]] = []
    for c in hard_constraints:
        if not isinstance(c, dict):
            continue
        if c.get("type") != "dock_to_edge":
            continue
        a_id = c.get("a")
        b_id = c.get("b")
        b_edge = c.get("b_edge")
        if not (
            isinstance(a_id, str) and isinstance(b_id, str) and isinstance(b_edge, str)
        ):
            continue

        b_front_base = _get_front_base(b_id, facing_map, specs)
        base_side = _resolve_edge_token_to_base_side(b_edge, b_front_base)

        dock_evals.append(
            {
                "type": "dock_to_edge",
                "a": a_id,
                "b": b_id,
                "b_edge": b_edge,
                "b_front_base": b_front_base,
                "resolved_base_side": base_side,
                "is_front_dock": base_side == b_front_base,
            }
        )

    # -------------------------
    # Grid / rotation checks
    # -------------------------
    grid_violations: list[dict[str, Any]] = []
    rot_violations: list[dict[str, Any]] = []

    for pid, placement in placements.items():
        x = placement["x"]
        y = placement["y"]
        rot = placement["rot"] % 360

        if grid_mm > 0 and (x % grid_mm != 0 or y % grid_mm != 0):
            grid_violations.append({"id": pid, "x": x, "y": y})
            errors.append(
                {
                    "code": "GRID_VIOLATION",
                    "a": pid,
                    "b": None,
                    "detail": f"x,y must be multiples of grid_mm={grid_mm}",
                }
            )

        spec = specs.get(pid) or {}
        allowed = spec.get("allowed_rotations") or spec.get("rotations")
        if isinstance(allowed, list) and allowed and rot not in allowed:
            rot_violations.append({"id": pid, "rot": rot, "allowed": allowed})
            errors.append(
                {
                    "code": "ROTATION_NOT_ALLOWED",
                    "a": pid,
                    "b": None,
                    "detail": f"rot={rot} not in allowed rotations.",
                }
            )

    # -------------------------
    # Build base rects and clearance rects
    # -------------------------
    rects = _build_rects(placements, specs, use_clearance=False)
    rects_clear = _build_rects(placements, specs, use_clearance=use_clearance)

    debug["rects"] = {k: _rect_dict(v) for k, v in rects.items()}
    debug["rects_clear"] = {k: _rect_dict(v) for k, v in rects_clear.items()}
    debug["allow_clearance_overlap"] = {
        k: sorted(list(v)) for k, v in allow_clearance_overlap.items()
    }
    debug["dock_to_edge_meta"] = dock_evals

    relation_ctx = _build_relation_context(
        hard_constraints=hard_constraints,
        placements=placements,
        rects=rects,
        specs=specs,
        facing_map=facing_map,
    )
    debug["relation_context"] = {
        "dependent_to_bases": {
            k: sorted(list(v)) for k, v in relation_ctx["dependent_to_bases"].items()
        },
        "base_to_dependents": {
            k: sorted(list(v)) for k, v in relation_ctx["base_to_dependents"].items()
        },
    }
    orientation_inference = _build_orientation_inference_debug(
        specs=specs,
        placements=placements,
        rects=rects,
        relation_ctx=relation_ctx,
        access_required_ids=access_required_ids,
        facing_map=facing_map,
    )
    debug["orientation_inference"] = orientation_inference

    # -------------------------
    # Score-based hard constraint evaluation
    # Add synthetic requires_access constraints from cluster_rules
    # so scoring sees them as hard constraints too.
    # -------------------------
    scoring_hard_constraints = _augment_hard_constraints_with_access(
        hard_constraints=hard_constraints,
        access_required_ids=access_required_ids,
    )
    canonical_local_placements = _placements_dict_to_list(placements)

    scoring = score_cluster_constraints(
        hard_constraints=scoring_hard_constraints,
        soft_constraints=soft_constraints,
        objects=specs,
        local_placements=canonical_local_placements,
        grid_mm=grid_mm,
        cluster_rules=cluster_rules,
        use_clearance=use_clearance,
    )

    debug["constraint_scores"] = {
        "hard_summary": scoring.get("hard", {}),
        "soft_summary": scoring.get("soft", {}),
        "lexicographic_key": scoring.get("lexicographic_key", []),
        "hard_evaluations": scoring.get("hard_evaluations", []),
        "soft_evaluations": scoring.get("soft_evaluations", []),
    }
    hard_summary = scoring.get("hard", {}) if isinstance(scoring, dict) else {}
    debug["hard_violated_count"] = int(hard_summary.get("violated_count", 0) or 0)
    debug["hard_satisfied_count"] = int(hard_summary.get("satisfied_count", 0) or 0)
    debug["hard_total_count"] = int(hard_summary.get("count", 0) or 0)

    shape_quality = scoring.get("layout_quality", _empty_quality())

    # -------------------------
    # Convert scored hard constraints to verifier-style errors/debug
    # -------------------------
    explicit_overlap_debug: list[dict[str, Any]] = []
    implicit_overlap_debug: list[dict[str, Any]] = []
    contain_debug: list[dict[str, Any]] = []
    dock_debug: list[dict[str, Any]] = []
    anchor_debug: list[dict[str, Any]] = []

    clearance_rects: dict[str, Rect] = {}
    clearance_block_debug: list[dict[str, Any]] = []

    hard_evaluations = scoring.get("hard_evaluations", [])
    if not isinstance(hard_evaluations, list):
        hard_evaluations = []

    for ev in hard_evaluations:
        if not isinstance(ev, dict):
            continue

        ctype = ev.get("type")
        satisfied = bool(ev.get("satisfied", False))
        subjects = (
            ev.get("subjects", {}) if isinstance(ev.get("subjects"), dict) else {}
        )
        components = (
            ev.get("components", {}) if isinstance(ev.get("components"), dict) else {}
        )
        ev_debug = ev.get("debug", {}) if isinstance(ev.get("debug"), dict) else {}

        # Unknown subject / unsupported -> UNKNOWN_OBJECT
        if components.get("unknown_subject") or components.get("unsupported"):
            a = subjects.get("a")
            b = subjects.get("b")
            errors.append(
                {
                    "code": "UNKNOWN_OBJECT",
                    "a": a,
                    "b": b,
                    "detail": f"Constraint references unknown object id or unsupported type: {ctype}",
                }
            )
            continue

        if satisfied:
            # keep access debug even when satisfied if available
            if ctype == "requires_access":
                cid = subjects.get("id")
                clear_rect = ev_debug.get("clearance_rect")
                if isinstance(clear_rect, dict) and isinstance(cid, str):
                    clearance_rects[cid] = (
                        int(clear_rect["x1"]),
                        int(clear_rect["y1"]),
                        int(clear_rect["x2"]),
                        int(clear_rect["y2"]),
                    )
            continue

        if ctype == "no_overlap":
            a = subjects.get("a")
            b = subjects.get("b")
            explicit_overlap_debug.append(
                {
                    "a": a,
                    "b": b,
                    "ix": int(components.get("overlap_x", 0) or 0),
                    "iy": int(components.get("overlap_y", 0) or 0),
                    "area": int(components.get("overlap_area", 0) or 0),
                }
            )
            errors.append(
                {
                    "code": "OVERLAP",
                    "a": a,
                    "b": b,
                    "detail": "Rectangles overlap.",
                }
            )
            continue

        if ctype == "contain_in":
            a = subjects.get("a")
            b = subjects.get("b")
            contain_debug.append(
                {
                    "a": a,
                    "b": b,
                    "a_rect": ev_debug.get("a_rect"),
                    "b_rect": ev_debug.get("b_rect"),
                }
            )
            errors.append(
                {
                    "code": "CONTAIN_VIOLATION",
                    "a": a,
                    "b": b,
                    "detail": "Object a must be fully inside object b.",
                }
            )
            continue

        if ctype == "dock_to_edge":
            a = subjects.get("a")
            b = subjects.get("b")
            dock_debug.append(
                {
                    "a": a,
                    "b": b,
                    "b_edge": ev_debug.get("b_edge"),
                    "mapped_side": ev_debug.get("mapped_side"),
                    "gap": components.get("gap_mm", 0),
                    "gap_min": _constraint_int(
                        scoring_hard_constraints,
                        ctype="dock_to_edge",
                        a=a,
                        b=b,
                        field="gap_min",
                        default=0,
                    ),
                    "gap_max": _constraint_int(
                        scoring_hard_constraints,
                        ctype="dock_to_edge",
                        a=a,
                        b=b,
                        field="gap_max",
                        default=0,
                    ),
                    "perp_ok": (components.get("perpendicular_overlap_mm", 0) or 0) > 0,
                    "span": _constraint_str(
                        scoring_hard_constraints,
                        ctype="dock_to_edge",
                        a=a,
                        b=b,
                        field="span",
                        default="any",
                    ),
                }
            )
            errors.append(
                {
                    "code": "DOCK_VIOLATION",
                    "a": a,
                    "b": b,
                    "detail": f"dock_to_edge failed for b_edge={ev_debug.get('b_edge')}",
                }
            )
            continue

        if ctype == "anchor_side":
            a = subjects.get("a")
            b = subjects.get("b")
            anchor_debug.append(
                {
                    "a": a,
                    "b": b,
                    "side": ev_debug.get("side"),
                    "mapped_side": ev_debug.get("mapped_side"),
                    "gap": components.get("gap_mm", 0),
                    "gap_min": _constraint_int(
                        scoring_hard_constraints,
                        ctype="anchor_side",
                        a=a,
                        b=b,
                        field="gap_min",
                        default=0,
                    ),
                    "gap_max": _constraint_int(
                        scoring_hard_constraints,
                        ctype="anchor_side",
                        a=a,
                        b=b,
                        field="gap_max",
                        default=0,
                    ),
                }
            )
            errors.append(
                {
                    "code": "ANCHOR_VIOLATION",
                    "a": a,
                    "b": b,
                    "detail": f"anchor_side failed for side={ev_debug.get('side')}",
                }
            )
            continue

        if ctype == "requires_access":
            owner_id = subjects.get("id")
            clear_rect = ev_debug.get("clearance_rect")
            if isinstance(clear_rect, dict) and isinstance(owner_id, str):
                clearance_rects[owner_id] = (
                    int(clear_rect["x1"]),
                    int(clear_rect["y1"]),
                    int(clear_rect["x2"]),
                    int(clear_rect["y2"]),
                )

            blockers = ev_debug.get("blockers", [])
            if isinstance(blockers, list):
                for blk in blockers:
                    if not isinstance(blk, dict):
                        continue
                    blocker = blk.get("blocker")
                    clearance_block_debug.append(
                        {
                            "owner": owner_id,
                            "blocker": blocker,
                            "ix": int(blk.get("intersection_x", 0) or 0),
                            "iy": int(blk.get("intersection_y", 0) or 0),
                            "area": int(blk.get("intersection_area", 0) or 0),
                        }
                    )
                    errors.append(
                        {
                            "code": "ACCESS_BLOCKED",
                            "a": owner_id,
                            "b": blocker,
                            "detail": "front_clearance area is blocked by another object",
                        }
                    )
            continue

    debug["pair_overlaps"] = explicit_overlap_debug
    debug["contain_debug"] = contain_debug
    debug["dock_debug"] = dock_debug
    debug["anchor_debug"] = anchor_debug

    # -------------------------
    # Implicit global body overlap check (tool-level, separate from scoring script)
    # -------------------------
    implicit_overlap_debug = _find_implicit_body_overlaps(
        rects=rects,
        specs=specs,
        hard_constraints=hard_constraints,
    )

    for ov in implicit_overlap_debug:
        errors.append(
            {
                "code": "OVERLAP",
                "a": ov["a"],
                "b": ov["b"],
                "detail": "Body rectangles overlap (implicit global check).",
            }
        )

    debug["implicit_pair_overlaps"] = implicit_overlap_debug
    debug["grid_violations"] = grid_violations
    debug["rotation_violations"] = rot_violations
    debug["front_clearance_rects"] = {
        k: _rect_dict(v) for k, v in clearance_rects.items()
    }
    debug["front_clearance_blocks"] = clearance_block_debug
    debug["access_required_ids"] = sorted(list(access_required_ids))
    debug["access_clearance_ratio"] = float(effective_access_clearance_ratio)

    # -------------------------
    # Budget + quality
    # -------------------------
    budget_errors = _evaluate_layout_budget(
        shape_quality=shape_quality,
        cluster_rules=cluster_rules,
    )
    if budget_errors:
        errors.extend(budget_errors)

    normalized_preview = _normalized_preview_from_rects(
        placements=placements,
        rects=rects,
    )

    debug["shape_quality"] = shape_quality
    debug["normalized_preview"] = normalized_preview
    debug["layout_budget"] = (
        dict(cluster_rules.get("layout_budget"))
        if isinstance(cluster_rules, dict)
        and isinstance(cluster_rules.get("layout_budget"), dict)
        else None
    )

    # -------------------------
    # Suggested moves
    # -------------------------
    suggested_moves: list[dict[str, Any]] = []

    suggested_moves.extend(
        _suggest_moves_for_grid_violations(
            placements=placements,
            rects=rects,
            relation_ctx=relation_ctx,
            grid_mm=grid_mm,
        )
    )

    all_overlap_debug: list[dict[str, Any]] = []
    all_overlap_debug.extend(explicit_overlap_debug)
    all_overlap_debug.extend(implicit_overlap_debug)

    suggested_moves.extend(
        _suggest_moves_for_overlaps(
            placements=placements,
            rects=(rects_clear if use_clearance else rects),
            overlap_debug=all_overlap_debug,
            relation_ctx=relation_ctx,
            grid_mm=grid_mm,
        )
    )

    suggested_moves.extend(
        _suggest_moves_for_contain_violations(
            placements=placements,
            rects=rects,
            contain_debug=contain_debug,
            grid_mm=grid_mm,
        )
    )

    suggested_moves.extend(
        _suggest_moves_for_access_blocks(
            placements=placements,
            rects=(rects_clear if use_clearance else rects),
            clearance_rects=clearance_rects,
            blocks=clearance_block_debug,
            grid_mm=grid_mm,
        )
    )

    suggested_moves.extend(
        _suggest_moves_for_dock_violations(
            placements=placements,
            rects=rects,
            dock_debug=dock_debug,
            grid_mm=grid_mm,
        )
    )

    suggested_moves.extend(
        _suggest_moves_for_anchor_violations(
            placements=placements,
            rects=rects,
            anchor_debug=anchor_debug,
            grid_mm=grid_mm,
        )
    )

    suggested_moves.extend(
        _suggest_moves_for_rotation_violations(
            placements=placements,
            rotation_violations=rot_violations,
        )
    )

    if not errors:
        suggested_moves.extend(
            _suggest_moves_for_relation_refinement(
                placements=placements,
                rects=rects,
                relation_ctx=relation_ctx,
                grid_mm=grid_mm,
            )
        )

        suggested_moves.extend(
            _suggest_moves_for_soft_constraints(
                soft_constraints=soft_constraints,
                placements=placements,
                rects=rects,
                relation_ctx=relation_ctx,
                specs=specs,
                facing_map=facing_map,
                grid_mm=grid_mm,
            )
        )

        suggested_moves.extend(
            _suggest_moves_for_compaction(
                placements=placements,
                rects=rects,
                relation_ctx=relation_ctx,
                grid_mm=grid_mm,
                shape_quality=shape_quality,
            )
        )

    suggested_moves = _dedup_moves(suggested_moves)
    errors = _sort_errors(errors)
    debug["suggested_moves_count"] = len(suggested_moves)
    rank_key = _tool_rank_key_from_scoring(
        grid_violations=grid_violations,
        rot_violations=rot_violations,
        scoring=scoring,
        shape_quality=shape_quality,
    )

    preferred_patches = _build_preferred_patches(
        hard_constraints=scoring_hard_constraints,
        soft_constraints=soft_constraints,
        objects=specs,
        placements=canonical_local_placements,
        grid_mm=grid_mm,
        cluster_rules=cluster_rules,
        use_clearance=use_clearance,
        suggested_moves=suggested_moves,
        current_rank_key=rank_key,
        top_k=5,
    )

    debug["rank_key"] = list(rank_key)
    debug["preferred_patches_count"] = len(preferred_patches)

    return {
        "result": "INVALID" if errors else "VALID",
        "errors": errors,
        "debug": debug,
        "quality": shape_quality,
        "rank_key": list(rank_key),
        "suggested_moves": suggested_moves,
        "preferred_patches": preferred_patches,
    }


# ============================================================
# Tool-local helpers
# ============================================================


def _empty_quality() -> dict[str, Any]:
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
            "perimeter_mm": 0,
            "points": [],
        },
        "fill_ratio_bbox": 1.0,
        "fill_ratio_hull": 1.0,
        "compactness_perimeter2_over_4piA": 1.0,
        "compact_score": 0,
    }


def _placements_dict_to_list(
    placements: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for oid in sorted(placements.keys()):
        p = placements[oid]
        out.append(
            {
                "id": oid,
                "x": int(p.get("x", 0)),
                "y": int(p.get("y", 0)),
                "rot": int(p.get("rot", 0)) % 360,
            }
        )
    return out


def _augment_hard_constraints_with_access(
    *,
    hard_constraints: list[dict[str, Any]],
    access_required_ids: set[str],
) -> list[dict[str, Any]]:
    out = [dict(c) for c in hard_constraints if isinstance(c, dict)]

    explicit_access_ids: set[str] = set()
    for c in out:
        if c.get("type") == "requires_access" and c.get("mode") == "front_clearance":
            oid = c.get("id")
            if isinstance(oid, str) and oid:
                explicit_access_ids.add(oid)

    for oid in sorted(access_required_ids):
        if oid in explicit_access_ids:
            continue
        out.append(
            {
                "type": "requires_access",
                "id": oid,
                "mode": "front_clearance",
            }
        )
    return out


def _constraint_int(
    constraints: list[dict[str, Any]],
    *,
    ctype: str,
    a: Any,
    b: Any,
    field: str,
    default: int,
) -> int:
    for c in constraints:
        if not isinstance(c, dict):
            continue
        if c.get("type") != ctype:
            continue
        if c.get("a") == a and c.get("b") == b:
            return int(c.get(field, default) or default)
    return int(default)


def _constraint_str(
    constraints: list[dict[str, Any]],
    *,
    ctype: str,
    a: Any,
    b: Any,
    field: str,
    default: str,
) -> str:
    for c in constraints:
        if not isinstance(c, dict):
            continue
        if c.get("type") != ctype:
            continue
        if c.get("a") == a and c.get("b") == b:
            value = c.get(field, default)
            return str(value) if value is not None else default
    return default


def _normalized_preview_from_rects(
    *,
    placements: dict[str, dict[str, Any]],
    rects: dict[str, Rect],
) -> dict[str, Any]:
    if not placements or not rects:
        return {
            "shift_dx": 0,
            "shift_dy": 0,
            "placements": [],
        }

    min_x = min(r[0] for r in rects.values())
    min_y = min(r[1] for r in rects.values())

    preview = []
    for oid in sorted(placements.keys()):
        p = placements[oid]
        preview.append(
            {
                "id": oid,
                "x": int(p["x"] - min_x),
                "y": int(p["y"] - min_y),
                "rot": int(p["rot"]) % 360,
            }
        )

    return {
        "shift_dx": int(-min_x),
        "shift_dy": int(-min_y),
        "placements": preview,
    }


def _evaluate_layout_budget(
    *,
    shape_quality: dict[str, Any],
    cluster_rules: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not isinstance(cluster_rules, dict):
        return []

    budget = cluster_rules.get("layout_budget")
    if not isinstance(budget, dict):
        return []

    errors: list[dict[str, Any]] = []

    bbox = shape_quality.get("bbox", {})
    outline = shape_quality.get("outline", {})
    hull = shape_quality.get("convex_hull", {})

    bbox_w = int(bbox.get("span_x", 0) or 0)
    bbox_h = int(bbox.get("span_y", 0) or 0)
    bbox_area = int(bbox.get("area_mm2", 0) or 0)
    max_span = int(bbox.get("max_span_mm", 0) or 0)
    outline_area = int(outline.get("area_mm2", 0) or 0)
    hull_area = float(hull.get("area_mm2", 0) or 0)
    fill_ratio_bbox = float(shape_quality.get("fill_ratio_bbox", 1.0) or 0.0)

    max_bbox_w = budget.get("max_bbox_w_mm")
    max_bbox_h = budget.get("max_bbox_h_mm")
    max_bbox_area = budget.get("max_bbox_area_mm2")
    max_outline_area = budget.get("max_outline_area_mm2")
    max_outline_span = budget.get("max_outline_span_mm")
    max_hull_area = budget.get("max_hull_area_mm2")
    min_fill_ratio = budget.get("min_fill_ratio_bbox")

    if isinstance(max_bbox_w, (int, float)) and bbox_w > int(max_bbox_w):
        errors.append(
            {
                "code": "LOCAL_BBOX_BUDGET_EXCEEDED",
                "a": None,
                "b": None,
                "detail": f"bbox width {bbox_w} exceeds max_bbox_w_mm={int(max_bbox_w)}",
            }
        )
    if isinstance(max_bbox_h, (int, float)) and bbox_h > int(max_bbox_h):
        errors.append(
            {
                "code": "LOCAL_BBOX_BUDGET_EXCEEDED",
                "a": None,
                "b": None,
                "detail": f"bbox height {bbox_h} exceeds max_bbox_h_mm={int(max_bbox_h)}",
            }
        )
    if isinstance(max_bbox_area, (int, float)) and bbox_area > int(max_bbox_area):
        errors.append(
            {
                "code": "LOCAL_BBOX_BUDGET_EXCEEDED",
                "a": None,
                "b": None,
                "detail": f"bbox area {bbox_area} exceeds max_bbox_area_mm2={int(max_bbox_area)}",
            }
        )
    if isinstance(max_outline_area, (int, float)) and outline_area > int(
        max_outline_area
    ):
        errors.append(
            {
                "code": "LOCAL_OUTLINE_BUDGET_EXCEEDED",
                "a": None,
                "b": None,
                "detail": f"outline area {outline_area} exceeds max_outline_area_mm2={int(max_outline_area)}",
            }
        )
    if isinstance(max_outline_span, (int, float)) and max_span > int(max_outline_span):
        errors.append(
            {
                "code": "LOCAL_OUTLINE_BUDGET_EXCEEDED",
                "a": None,
                "b": None,
                "detail": f"max span {max_span} exceeds max_outline_span_mm={int(max_outline_span)}",
            }
        )
    if isinstance(max_hull_area, (int, float)) and hull_area > float(max_hull_area):
        errors.append(
            {
                "code": "LOCAL_HULL_BUDGET_EXCEEDED",
                "a": None,
                "b": None,
                "detail": f"hull area {int(round(hull_area))} exceeds max_hull_area_mm2={int(max_hull_area)}",
            }
        )
    if isinstance(min_fill_ratio, (int, float)) and fill_ratio_bbox < float(
        min_fill_ratio
    ):
        errors.append(
            {
                "code": "LOCAL_FILL_RATIO_TOO_LOW",
                "a": None,
                "b": None,
                "detail": f"fill_ratio_bbox={round(fill_ratio_bbox, 6)} below min_fill_ratio_bbox={float(min_fill_ratio)}",
            }
        )

    return errors


# ============================================================
# Relation context
# ============================================================


def _build_relation_context(
    *,
    hard_constraints: list[dict[str, Any]],
    placements: dict[str, dict[str, Any]],
    rects: dict[str, Rect],
    specs: dict[str, dict[str, Any]],
    facing_map: dict[str, Any],
) -> dict[str, Any]:
    ctx: dict[str, Any] = {
        "dock_by_a": {},
        "anchor_by_a": {},
        "contain_by_a": {},
        "dependent_to_bases": {},
        "base_to_dependents": {},
    }

    def add_dep(a: str, b: str) -> None:
        ctx["dependent_to_bases"].setdefault(a, set()).add(b)
        ctx["base_to_dependents"].setdefault(b, set()).add(a)

    for c in hard_constraints:
        if not isinstance(c, dict):
            continue
        ctype = c.get("type")

        if ctype == "dock_to_edge":
            a = c.get("a")
            b = c.get("b")
            b_edge = c.get("b_edge")
            if not (
                isinstance(a, str) and isinstance(b, str) and isinstance(b_edge, str)
            ):
                continue
            if a not in rects or b not in placements:
                continue

            rot_b = placements[b]["rot"] % 360
            b_front_base = _get_front_base(b, facing_map, specs)
            base_side = _resolve_edge_token_to_base_side(b_edge, b_front_base)
            mapped_side = _rotate_side(base_side, rot_b)

            rel = {
                "type": "dock_to_edge",
                "a": a,
                "b": b,
                "b_edge": b_edge,
                "mapped_side": mapped_side,
                "gap_min": int(c.get("gap_min", 0)),
                "gap_max": int(c.get("gap_max", 0)),
                "span": str(c.get("span", "any")),
            }
            ctx["dock_by_a"].setdefault(a, []).append(rel)
            add_dep(a, b)
            continue

        if ctype == "anchor_side":
            a = c.get("a")
            b = c.get("b")
            side = c.get("side")
            if not (
                isinstance(a, str) and isinstance(b, str) and isinstance(side, str)
            ):
                continue
            if a not in rects or b not in placements:
                continue

            rot_b = placements[b]["rot"] % 360
            base_side, qualifier = _resolve_anchor_side(side)
            if base_side == "head":
                base_side = "top"
            elif base_side == "foot":
                base_side = "bottom"
            mapped_side = _rotate_side(base_side, rot_b)

            rel = {
                "type": "anchor_side",
                "a": a,
                "b": b,
                "side": side,
                "mapped_side": mapped_side,
                "gap_min": int(c.get("gap_min", 0)),
                "gap_max": int(c.get("gap_max", 0)),
                "qualifier": qualifier,
            }
            ctx["anchor_by_a"].setdefault(a, []).append(rel)
            add_dep(a, b)
            continue

        if ctype == "contain_in":
            a = c.get("a")
            b = c.get("b")
            if not (isinstance(a, str) and isinstance(b, str)):
                continue
            rel = {"type": "contain_in", "a": a, "b": b}
            ctx["contain_by_a"].setdefault(a, []).append(rel)
            add_dep(a, b)

    return ctx


# ============================================================
# Tool-local implicit overlap
# ============================================================


def _has_explicit_binary_constraint(
    hard_constraints: list[dict[str, Any]],
    *,
    a: str,
    b: str,
    types: set[str],
) -> bool:
    for c in hard_constraints:
        if not isinstance(c, dict):
            continue
        ctype = c.get("type")
        if ctype not in types:
            continue
        ca = c.get("a")
        cb = c.get("b")
        if not isinstance(ca, str) or not isinstance(cb, str):
            continue
        if (ca == a and cb == b) or (ca == b and cb == a):
            return True
    return False


def _find_implicit_body_overlaps(
    *,
    rects: dict[str, Rect],
    specs: dict[str, dict[str, Any]],
    hard_constraints: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ids = sorted(rects.keys())
    overlaps: list[dict[str, Any]] = []

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a = ids[i]
            b = ids[j]

            if not _is_floor_occupying(specs.get(a, {})):
                continue
            if not _is_floor_occupying(specs.get(b, {})):
                continue

            if _has_explicit_binary_constraint(
                hard_constraints,
                a=a,
                b=b,
                types={"no_overlap"},
            ):
                continue

            if _has_explicit_binary_constraint(
                hard_constraints,
                a=a,
                b=b,
                types={"contain_in"},
            ):
                continue

            ix, iy, area = _intersection_stats(rects[a], rects[b])
            if area > 0:
                overlaps.append(
                    {
                        "a": a,
                        "b": b,
                        "ix": ix,
                        "iy": iy,
                        "area": area,
                    }
                )

    return overlaps


# ============================================================
# Interval / grid helpers
# ============================================================


def _intersect_optional_interval(
    base: tuple[int, int] | None, new_lo: int, new_hi: int
) -> tuple[int, int] | None:
    lo = min(new_lo, new_hi)
    hi = max(new_lo, new_hi)
    if base is None:
        return (lo, hi)
    return (max(base[0], lo), min(base[1], hi))


def _choose_grid_value_in_interval(
    *,
    current: int,
    interval: tuple[int, int] | None,
    grid_mm: int,
) -> int:
    if grid_mm <= 0:
        if interval is None:
            return current
        lo, hi = interval
        return min(max(current, lo), hi)

    if interval is None:
        return _snap_nearest(current, grid_mm)

    lo, hi = interval
    lo, hi = min(lo, hi), max(lo, hi)

    first = _snap_up(lo, grid_mm)
    last = _snap_down(hi, grid_mm)

    if first <= last:
        if current <= first:
            return first
        if current >= last:
            return last

        k = int(round((current - first) / grid_mm))
        cand = first + k * grid_mm
        cand = min(max(cand, first), last)

        candidates = {cand}
        if cand - grid_mm >= first:
            candidates.add(cand - grid_mm)
        if cand + grid_mm <= last:
            candidates.add(cand + grid_mm)

        best = None
        for v in sorted(candidates):
            score = (abs(v - current), v)
            if best is None or score < best[0]:
                best = (score, v)
        return best[1]

    cands = {
        _snap_nearest(current, grid_mm),
        _snap_up(lo, grid_mm),
        _snap_down(hi, grid_mm),
    }

    best = None
    for v in cands:
        dist_to_interval = 0
        if v < lo:
            dist_to_interval = lo - v
        elif v > hi:
            dist_to_interval = v - hi
        score = (dist_to_interval, abs(v - current), v)
        if best is None or score < best[0]:
            best = (score, v)
    return best[1]


def _side_axis_interval(
    *,
    ra: Rect,
    rb: Rect,
    mapped_side: str,
    gap_min: int,
    gap_max: int,
) -> tuple[str, int, int]:
    ax1, ay1, ax2, ay2 = ra
    bx1, by1, bx2, by2 = rb
    aw = ax2 - ax1
    ah = ay2 - ay1

    g0 = min(gap_min, gap_max)
    g1 = max(gap_min, gap_max)

    if mapped_side == "top":
        return ("y", by2 + g0, by2 + g1)
    if mapped_side == "bottom":
        return ("y", by1 - ah - g1, by1 - ah - g0)
    if mapped_side == "right":
        return ("x", bx2 + g0, bx2 + g1)
    if mapped_side == "left":
        return ("x", bx1 - aw - g1, bx1 - aw - g0)
    return ("x", ax1, ax1)


def _perpendicular_positive_overlap_interval(
    *,
    ra: Rect,
    rb: Rect,
    mapped_side: str,
) -> tuple[str, int, int]:
    ax1, ay1, ax2, ay2 = ra
    bx1, by1, bx2, by2 = rb
    aw = ax2 - ax1
    ah = ay2 - ay1

    if mapped_side in {"top", "bottom"}:
        return ("x", bx1 - aw + 1, bx2 - 1)
    return ("y", by1 - ah + 1, by2 - 1)


def _perpendicular_overlap_positive(ra: Rect, rb: Rect, side: str) -> bool:
    return _perpendicular_overlap_len(ra, rb, side) > 0


def _constraint_aware_snap_for_object(
    *,
    oid: str,
    placements: dict[str, dict[str, Any]],
    rects: dict[str, Rect],
    relation_ctx: dict[str, Any],
    grid_mm: int,
) -> tuple[int, int] | None:
    if oid not in placements or oid not in rects:
        return None

    p = placements[oid]
    ra = rects[oid]
    cur_x = int(p["x"])
    cur_y = int(p["y"])

    x_interval: tuple[int, int] | None = None
    y_interval: tuple[int, int] | None = None

    for rel in relation_ctx.get("dock_by_a", {}).get(oid, []):
        b = rel["b"]
        if b not in rects:
            continue
        rb = rects[b]
        axis, lo, hi = _side_axis_interval(
            ra=ra,
            rb=rb,
            mapped_side=rel["mapped_side"],
            gap_min=int(rel["gap_min"]),
            gap_max=int(rel["gap_max"]),
        )
        if axis == "x":
            x_interval = _intersect_optional_interval(x_interval, lo, hi)
        else:
            y_interval = _intersect_optional_interval(y_interval, lo, hi)

        if not _perpendicular_overlap_positive(ra, rb, rel["mapped_side"]):
            pax, plo, phi = _perpendicular_positive_overlap_interval(
                ra=ra,
                rb=rb,
                mapped_side=rel["mapped_side"],
            )
            if pax == "x":
                x_interval = _intersect_optional_interval(x_interval, plo, phi)
            else:
                y_interval = _intersect_optional_interval(y_interval, plo, phi)

    for rel in relation_ctx.get("anchor_by_a", {}).get(oid, []):
        b = rel["b"]
        if b not in rects:
            continue
        rb = rects[b]
        axis, lo, hi = _side_axis_interval(
            ra=ra,
            rb=rb,
            mapped_side=rel["mapped_side"],
            gap_min=int(rel["gap_min"]),
            gap_max=int(rel["gap_max"]),
        )
        if axis == "x":
            x_interval = _intersect_optional_interval(x_interval, lo, hi)
        else:
            y_interval = _intersect_optional_interval(y_interval, lo, hi)

    if x_interval is None and y_interval is None:
        return None

    new_x = cur_x
    new_y = cur_y

    if grid_mm > 0 and cur_x % grid_mm != 0:
        new_x = _choose_grid_value_in_interval(
            current=cur_x,
            interval=x_interval,
            grid_mm=grid_mm,
        )
    elif x_interval is not None and not (x_interval[0] <= cur_x <= x_interval[1]):
        new_x = _choose_grid_value_in_interval(
            current=cur_x,
            interval=x_interval,
            grid_mm=grid_mm,
        )

    if grid_mm > 0 and cur_y % grid_mm != 0:
        new_y = _choose_grid_value_in_interval(
            current=cur_y,
            interval=y_interval,
            grid_mm=grid_mm,
        )
    elif y_interval is not None and not (y_interval[0] <= cur_y <= y_interval[1]):
        new_y = _choose_grid_value_in_interval(
            current=cur_y,
            interval=y_interval,
            grid_mm=grid_mm,
        )

    return new_x, new_y


def _soft_step_toward(delta: float, grid_mm: int) -> int:
    if abs(delta) <= 1e-6:
        return 0
    step = abs(delta)
    if grid_mm > 0:
        step = min(step, float(grid_mm))
    snapped = int(round(step))
    if snapped <= 0 and grid_mm > 0:
        snapped = int(grid_mm)
    return snapped if delta > 0 else -snapped


def _suggest_moves_for_soft_constraints(
    *,
    soft_constraints: list[dict[str, Any]],
    placements: dict[str, dict[str, Any]],
    rects: dict[str, Rect],
    relation_ctx: dict[str, Any],
    specs: dict[str, dict[str, Any]],
    facing_map: dict[str, Any],
    grid_mm: int,
) -> list[dict[str, Any]]:
    moves: list[dict[str, Any]] = []
    base_to_dependents = relation_ctx.get("base_to_dependents", {})

    for constraint in soft_constraints:
        if not isinstance(constraint, dict):
            continue
        ctype = str(constraint.get("type") or "").strip().lower()

        if ctype == "prefer_near":
            a = constraint.get("a")
            b = constraint.get("b")
            if not (isinstance(a, str) and isinstance(b, str)):
                continue
            if a not in placements or a not in rects or b not in rects:
                continue
            if len(base_to_dependents.get(a, set())) > 0:
                continue

            ra = rects[a]
            rb = rects[b]
            edge_gap = _rect_edge_gap(ra, rb)
            if edge_gap <= 0:
                continue

            ax1, ay1, ax2, ay2 = ra
            bx1, by1, bx2, by2 = rb
            center_dx = ((bx1 + bx2) / 2.0) - ((ax1 + ax2) / 2.0)
            center_dy = ((by1 + by2) / 2.0) - ((ay1 + ay2) / 2.0)
            x_gap = max(0, max(bx1 - ax2, ax1 - bx2))
            y_gap = max(0, max(by1 - ay2, ay1 - by2))

            candidate_offsets: list[tuple[int, int]] = []
            if x_gap > 0:
                candidate_offsets.append((_soft_step_toward(center_dx, grid_mm), 0))
            if y_gap > 0:
                candidate_offsets.append((0, _soft_step_toward(center_dy, grid_mm)))
            if x_gap > 0 and y_gap > 0:
                candidate_offsets.append(
                    (
                        _soft_step_toward(center_dx, grid_mm),
                        _soft_step_toward(center_dy, grid_mm),
                    )
                )

            seen_offsets: set[tuple[int, int]] = set()
            for dx, dy in candidate_offsets:
                if (dx, dy) == (0, 0) or (dx, dy) in seen_offsets:
                    continue
                seen_offsets.add((dx, dy))
                base_x = int(placements[a]["x"])
                base_y = int(placements[a]["y"])
                moves.append(
                    {
                        "reason": "SOFT_PREFER_NEAR",
                        "a": a,
                        "b": b,
                        "move_object": a,
                        "dx": int(dx),
                        "dy": int(dy),
                        "new_x": int(base_x + dx),
                        "new_y": int(base_y + dy),
                        "note": f"Reduce the prefer_near gap between {a} and {b}.",
                    }
                )
            continue

        if ctype == "prefer_align_edge":
            a = constraint.get("a")
            b = constraint.get("b")
            edge = constraint.get("edge")
            if not (
                isinstance(a, str) and isinstance(b, str) and isinstance(edge, str)
            ):
                continue
            if a not in placements or a not in rects or b not in rects:
                continue
            if len(base_to_dependents.get(a, set())) > 0:
                continue

            ra = rects[a]
            rb = rects[b]
            rot_b = placements[b]["rot"] % 360
            b_front_base = _get_front_base(b, facing_map, specs)
            base_side = _resolve_edge_token_to_base_side(edge, b_front_base)
            mapped_side = _rotate_side(base_side, rot_b)
            axis, lo, hi = _side_axis_interval(
                ra=ra,
                rb=rb,
                mapped_side=mapped_side,
                gap_min=0,
                gap_max=0,
            )

            base_x = int(placements[a]["x"])
            base_y = int(placements[a]["y"])
            new_x = base_x
            new_y = base_y

            if axis == "x":
                new_x = _choose_grid_value_in_interval(
                    current=base_x,
                    interval=(lo, hi),
                    grid_mm=grid_mm,
                )
            else:
                new_y = _choose_grid_value_in_interval(
                    current=base_y,
                    interval=(lo, hi),
                    grid_mm=grid_mm,
                )

            perp_offset = _perpendicular_center_offset(ra, rb, mapped_side)
            if mapped_side in {"top", "bottom"}:
                desired_x = int(round(base_x - perp_offset))
                new_x = _choose_grid_value_in_interval(
                    current=desired_x,
                    interval=None,
                    grid_mm=grid_mm,
                )
            else:
                desired_y = int(round(base_y - perp_offset))
                new_y = _choose_grid_value_in_interval(
                    current=desired_y,
                    interval=None,
                    grid_mm=grid_mm,
                )

            dx = int(new_x - base_x)
            dy = int(new_y - base_y)
            if dx == 0 and dy == 0:
                continue

            moves.append(
                {
                    "reason": "SOFT_ALIGN_EDGE",
                    "a": a,
                    "b": b,
                    "move_object": a,
                    "dx": dx,
                    "dy": dy,
                    "new_x": int(new_x),
                    "new_y": int(new_y),
                    "note": f"Improve soft edge alignment between {a} and {b}.",
                }
            )

    return moves


def _suggest_moves_for_relation_refinement(
    *,
    placements: dict[str, dict[str, Any]],
    rects: dict[str, Rect],
    relation_ctx: dict[str, Any],
    grid_mm: int,
) -> list[dict[str, Any]]:
    moves: list[dict[str, Any]] = []

    for oid, relations in (relation_ctx.get("anchor_by_a") or {}).items():
        if oid not in placements or oid not in rects:
            continue
        for rel in relations or []:
            if not isinstance(rel, dict):
                continue
            b = rel.get("b")
            side = rel.get("side")
            mapped_side = rel.get("mapped_side")
            if not (
                isinstance(b, str)
                and isinstance(side, str)
                and isinstance(mapped_side, str)
                and b in placements
                and b in rects
            ):
                continue

            ra = rects[oid]
            rb = rects[b]
            base_x = int(placements[oid]["x"])
            base_y = int(placements[oid]["y"])
            rot_b = int(placements[b]["rot"]) % 360
            base_side, qualifier_local = _resolve_anchor_side(side)
            qualifier_world = None
            if qualifier_local in {"left", "right"}:
                qualifier_world = _rotate_side(qualifier_local, rot_b)

            new_x = base_x
            new_y = base_y
            axis, lo, hi = _side_axis_interval(
                ra=ra,
                rb=rb,
                mapped_side=mapped_side,
                gap_min=int(rel.get("gap_min", 0)),
                gap_max=int(rel.get("gap_max", 0)),
            )
            if axis == "x":
                new_x = _choose_grid_value_in_interval(
                    current=base_x,
                    interval=(lo, hi),
                    grid_mm=grid_mm,
                )
            else:
                new_y = _choose_grid_value_in_interval(
                    current=base_y,
                    interval=(lo, hi),
                    grid_mm=grid_mm,
                )

            ax1, ay1, ax2, ay2 = ra
            bx1, by1, bx2, by2 = rb
            aw = ax2 - ax1
            ah = ay2 - ay1
            bw = bx2 - bx1
            bh = by2 - by1

            if mapped_side in {"top", "bottom"}:
                free_span = max(0, bw - aw)
                z_lo, z_hi = _anchor_zone_bounds(free_span, qualifier_world)
                desired = bx1 + (z_lo + z_hi) // 2
                new_x = _choose_grid_value_in_interval(
                    current=desired,
                    interval=(bx1 + z_lo, bx1 + z_hi),
                    grid_mm=grid_mm,
                )
            else:
                free_span = max(0, bh - ah)
                z_lo, z_hi = _anchor_zone_bounds(free_span, qualifier_world)
                desired = by1 + (z_lo + z_hi) // 2
                new_y = _choose_grid_value_in_interval(
                    current=desired,
                    interval=(by1 + z_lo, by1 + z_hi),
                    grid_mm=grid_mm,
                )

            dx = int(new_x - base_x)
            dy = int(new_y - base_y)
            if dx == 0 and dy == 0:
                continue

            moves.append(
                {
                    "reason": "ANCHOR_REFINEMENT",
                    "a": oid,
                    "b": b,
                    "move_object": oid,
                    "dx": dx,
                    "dy": dy,
                    "new_x": int(new_x),
                    "new_y": int(new_y),
                    "note": f"Refine {oid} inside its anchor-side band relative to {b}.",
                }
            )

    for oid, relations in (relation_ctx.get("dock_by_a") or {}).items():
        if oid not in placements or oid not in rects:
            continue
        for rel in relations or []:
            if not isinstance(rel, dict):
                continue
            b = rel.get("b")
            mapped_side = rel.get("mapped_side")
            if not (
                isinstance(b, str)
                and isinstance(mapped_side, str)
                and b in placements
                and b in rects
            ):
                continue

            ra = rects[oid]
            rb = rects[b]
            base_x = int(placements[oid]["x"])
            base_y = int(placements[oid]["y"])
            new_x = base_x
            new_y = base_y

            axis, lo, hi = _side_axis_interval(
                ra=ra,
                rb=rb,
                mapped_side=mapped_side,
                gap_min=int(rel.get("gap_min", 0)),
                gap_max=int(rel.get("gap_max", 0)),
            )
            if axis == "x":
                new_x = _choose_grid_value_in_interval(
                    current=base_x,
                    interval=(lo, hi),
                    grid_mm=grid_mm,
                )
            else:
                new_y = _choose_grid_value_in_interval(
                    current=base_y,
                    interval=(lo, hi),
                    grid_mm=grid_mm,
                )

            perp_offset = _perpendicular_center_offset(ra, rb, mapped_side)
            if mapped_side in {"top", "bottom"}:
                desired_x = int(round(base_x - perp_offset))
                new_x = _choose_grid_value_in_interval(
                    current=desired_x,
                    interval=None,
                    grid_mm=grid_mm,
                )
            else:
                desired_y = int(round(base_y - perp_offset))
                new_y = _choose_grid_value_in_interval(
                    current=desired_y,
                    interval=None,
                    grid_mm=grid_mm,
                )

            dx = int(new_x - base_x)
            dy = int(new_y - base_y)
            if dx == 0 and dy == 0:
                continue

            moves.append(
                {
                    "reason": "DOCK_REFINEMENT",
                    "a": oid,
                    "b": b,
                    "move_object": oid,
                    "dx": dx,
                    "dy": dy,
                    "new_x": int(new_x),
                    "new_y": int(new_y),
                    "note": f"Refine {oid} along its docked edge relative to {b}.",
                }
            )

    return moves


# ============================================================
# Suggested move generators
# ============================================================


def _suggest_moves_for_grid_violations(
    *,
    placements: dict[str, dict[str, Any]],
    rects: dict[str, Rect],
    relation_ctx: dict[str, Any],
    grid_mm: int,
) -> list[dict[str, Any]]:
    if grid_mm <= 0:
        return []

    moves: list[dict[str, Any]] = []
    for oid, p in placements.items():
        x = int(p["x"])
        y = int(p["y"])

        if x % grid_mm == 0 and y % grid_mm == 0:
            continue

        special = _constraint_aware_snap_for_object(
            oid=oid,
            placements=placements,
            rects=rects,
            relation_ctx=relation_ctx,
            grid_mm=grid_mm,
        )

        if special is not None:
            xs, ys = special
        else:
            xs = _snap_nearest(x, grid_mm)
            ys = _snap_nearest(y, grid_mm)

        dx = xs - x
        dy = ys - y
        if dx == 0 and dy == 0:
            continue

        note = "Snap (x,y) to nearest grid multiple."
        if special is not None:
            note = "Snap to a grid point that also respects current dock/anchor constraints when possible."

        moves.append(
            {
                "reason": "GRID_VIOLATION",
                "move_object": oid,
                "dx": int(dx),
                "dy": int(dy),
                "new_x": int(xs),
                "new_y": int(ys),
                "note": note,
            }
        )

    return moves


def _choose_overlap_move_object(
    *,
    a: str,
    b: str,
    relation_ctx: dict[str, Any],
    rects: dict[str, Rect],
) -> str:
    dep_to_base: dict[str, set[str]] = relation_ctx.get("dependent_to_bases", {})
    base_to_dep: dict[str, set[str]] = relation_ctx.get("base_to_dependents", {})

    a_depends_on_b = b in dep_to_base.get(a, set())
    b_depends_on_a = a in dep_to_base.get(b, set())

    if a_depends_on_b and not b_depends_on_a:
        return a
    if b_depends_on_a and not a_depends_on_b:
        return b

    a_has_dependents = len(base_to_dep.get(a, set())) > 0
    b_has_dependents = len(base_to_dep.get(b, set())) > 0

    if a_has_dependents and not b_has_dependents:
        return b
    if b_has_dependents and not a_has_dependents:
        return a

    area_a = _rect_area(rects[a])
    area_b = _rect_area(rects[b])
    if area_a != area_b:
        return a if area_a < area_b else b

    return a


def _minimal_separating_move(
    *,
    move_rect: Rect,
    other_rect: Rect,
    grid_mm: int,
) -> tuple[int, int, str]:
    mx1, my1, mx2, my2 = move_rect
    ox1, oy1, ox2, oy2 = other_rect

    candidates = [
        ("left", ox1 - mx2, 0),
        ("right", ox2 - mx1, 0),
        ("down", 0, oy1 - my2),
        ("up", 0, oy2 - my1),
    ]

    best = None
    for axis_name, dx0, dy0 in candidates:
        if dx0 == 0 and dy0 == 0:
            continue
        dx = _snap_delta(dx0, grid_mm) if dx0 != 0 else 0
        dy = _snap_delta(dy0, grid_mm) if dy0 != 0 else 0

        if dx0 != 0 and dx == 0:
            dx = grid_mm if dx0 > 0 else -grid_mm
        if dy0 != 0 and dy == 0:
            dy = grid_mm if dy0 > 0 else -grid_mm

        cost = abs(dx) + abs(dy)
        rank = 0 if axis_name in {"left", "right"} else 1
        score = (cost, rank, axis_name)
        if best is None or score < best[0]:
            best = (score, dx, dy, axis_name)

    if best is None:
        return (_snap_delta(grid_mm, grid_mm), 0, "right")

    _, dx, dy, axis_name = best
    return dx, dy, axis_name


def _suggest_moves_for_overlaps(
    *,
    placements: dict[str, dict[str, Any]],
    rects: dict[str, Rect],
    overlap_debug: list[dict[str, Any]],
    relation_ctx: dict[str, Any],
    grid_mm: int,
) -> list[dict[str, Any]]:
    moves: list[dict[str, Any]] = []
    for ov in overlap_debug:
        a = ov["a"]
        b = ov["b"]
        if (
            a not in rects
            or b not in rects
            or a not in placements
            or b not in placements
        ):
            continue

        ix, iy, area = _intersection_stats(rects[a], rects[b])
        if area <= 0:
            continue

        who = _choose_overlap_move_object(
            a=a,
            b=b,
            relation_ctx=relation_ctx,
            rects=rects,
        )
        other = b if who == a else a

        dx, dy, axis_name = _minimal_separating_move(
            move_rect=rects[who],
            other_rect=rects[other],
            grid_mm=grid_mm,
        )

        base_x = int(placements[who]["x"])
        base_y = int(placements[who]["y"])

        note = f"Resolve overlap by shifting dependent/smaller object {who} along {axis_name} (touch allowed)."
        moves.append(
            {
                "reason": "OVERLAP",
                "a": a,
                "b": b,
                "move_object": who,
                "dx": int(dx),
                "dy": int(dy),
                "new_x": int(base_x + dx),
                "new_y": int(base_y + dy),
                "note": note,
            }
        )
    return moves


def _suggest_moves_for_contain_violations(
    *,
    placements: dict[str, dict[str, Any]],
    rects: dict[str, Rect],
    contain_debug: list[dict[str, Any]],
    grid_mm: int,
) -> list[dict[str, Any]]:
    moves: list[dict[str, Any]] = []
    for cd in contain_debug:
        a = cd.get("a")
        b = cd.get("b")
        if not isinstance(a, str) or not isinstance(b, str):
            continue
        if a not in placements or a not in rects or b not in rects:
            continue

        ra = rects[a]
        rb = rects[b]

        ax1, ay1, ax2, ay2 = ra
        bx1, by1, bx2, by2 = rb

        dx_req = 0
        dy_req = 0
        if ax1 < bx1:
            dx_req += bx1 - ax1
        if ax2 > bx2:
            dx_req += bx2 - ax2
        if ay1 < by1:
            dy_req += by1 - ay1
        if ay2 > by2:
            dy_req += by2 - ay2

        dx = _snap_delta(int(dx_req), grid_mm) if dx_req != 0 else 0
        dy = _snap_delta(int(dy_req), grid_mm) if dy_req != 0 else 0

        if dx_req != 0 and dx == 0:
            dx = _snap_delta(grid_mm if dx_req > 0 else -grid_mm, grid_mm)
        if dy_req != 0 and dy == 0:
            dy = _snap_delta(grid_mm if dy_req > 0 else -grid_mm, grid_mm)

        base_x = int(placements[a]["x"])
        base_y = int(placements[a]["y"])

        if dx == 0 and dy == 0:
            continue

        moves.append(
            {
                "reason": "CONTAIN_VIOLATION",
                "a": a,
                "b": b,
                "move_object": a,
                "dx": int(dx),
                "dy": int(dy),
                "new_x": int(base_x + dx),
                "new_y": int(base_y + dy),
                "note": f"Shift {a} to be contained in {b} (touch allowed).",
            }
        )

    return moves


def _suggest_moves_for_access_blocks(
    *,
    placements: dict[str, dict[str, Any]],
    rects: dict[str, Rect],
    clearance_rects: dict[str, Rect],
    blocks: list[dict[str, Any]],
    grid_mm: int,
) -> list[dict[str, Any]]:
    moves: list[dict[str, Any]] = []
    for bl in blocks:
        owner = bl["owner"]
        blocker = bl["blocker"]
        if (
            blocker not in rects
            or blocker not in placements
            or owner not in clearance_rects
        ):
            continue

        cr = clearance_rects[owner]
        br = rects[blocker]
        _, _, area = _intersection_stats(cr, br)
        if area <= 0:
            continue

        cx1, cy1, cx2, cy2 = cr
        bx1, by1, bx2, by2 = br

        cand = [
            ("left", cx1 - bx2, 0),
            ("right", cx2 - bx1, 0),
            ("down", 0, cy1 - by2),
            ("up", 0, cy2 - by1),
        ]

        best = None
        for direction, dx, dy in cand:
            if dx == 0 and dy == 0:
                continue
            dx_s = _snap_delta(int(dx), grid_mm)
            dy_s = _snap_delta(int(dy), grid_mm)

            if dx_s == 0 and dy_s == 0:
                if direction in {"left", "right"}:
                    dx_s = _snap_delta(
                        grid_mm if direction == "right" else -grid_mm, grid_mm
                    )
                else:
                    dy_s = _snap_delta(
                        grid_mm if direction == "up" else -grid_mm, grid_mm
                    )

            cost = abs(dx_s) + abs(dy_s)
            if best is None or cost < best[0]:
                best = (cost, direction, dx_s, dy_s)

        if best is None:
            continue

        _, direction, dx_s, dy_s = best
        base_x = placements[blocker]["x"]
        base_y = placements[blocker]["y"]
        moves.append(
            {
                "reason": "ACCESS_BLOCKED",
                "owner": owner,
                "blocker": blocker,
                "move_object": blocker,
                "dx": int(dx_s),
                "dy": int(dy_s),
                "new_x": int(base_x + dx_s),
                "new_y": int(base_y + dy_s),
                "note": f"Move blocker {direction} to clear {owner}'s front_clearance (touch allowed).",
            }
        )
    return moves


def _suggest_moves_for_dock_violations(
    *,
    placements: dict[str, dict[str, Any]],
    rects: dict[str, Rect],
    dock_debug: list[dict[str, Any]],
    grid_mm: int,
) -> list[dict[str, Any]]:
    moves: list[dict[str, Any]] = []
    for d in dock_debug:
        a = d.get("a")
        b = d.get("b")
        mapped_side = d.get("mapped_side")
        gap_min = d.get("gap_min")
        gap_max = d.get("gap_max")
        per_ok = d.get("perp_ok")

        if not (
            isinstance(a, str) and isinstance(b, str) and isinstance(mapped_side, str)
        ):
            continue
        if a not in placements or a not in rects or b not in rects:
            continue
        if not isinstance(gap_min, int) or not isinstance(gap_max, int):
            continue

        ra = rects[a]
        rb = rects[b]

        axis, lo, hi = _side_axis_interval(
            ra=ra,
            rb=rb,
            mapped_side=mapped_side,
            gap_min=gap_min,
            gap_max=gap_max,
        )

        base_x = int(placements[a]["x"])
        base_y = int(placements[a]["y"])
        new_x = base_x
        new_y = base_y

        if axis == "x":
            new_x = _choose_grid_value_in_interval(
                current=base_x,
                interval=(lo, hi),
                grid_mm=grid_mm,
            )
        else:
            new_y = _choose_grid_value_in_interval(
                current=base_y,
                interval=(lo, hi),
                grid_mm=grid_mm,
            )

        if not per_ok:
            pax, plo, phi = _perpendicular_positive_overlap_interval(
                ra=ra,
                rb=rb,
                mapped_side=mapped_side,
            )
            if pax == "x":
                new_x = _choose_grid_value_in_interval(
                    current=new_x,
                    interval=(plo, phi),
                    grid_mm=grid_mm,
                )
            else:
                new_y = _choose_grid_value_in_interval(
                    current=new_y,
                    interval=(plo, phi),
                    grid_mm=grid_mm,
                )

        dx = new_x - base_x
        dy = new_y - base_y

        if dx == 0 and dy == 0:
            if axis == "x":
                dx = _snap_delta(grid_mm if lo >= base_x else -grid_mm, grid_mm)
                new_x = base_x + dx
            else:
                dy = _snap_delta(grid_mm if lo >= base_y else -grid_mm, grid_mm)
                new_y = base_y + dy

        moves.append(
            {
                "reason": "DOCK_VIOLATION",
                "a": a,
                "b": b,
                "move_object": a,
                "dx": int(dx),
                "dy": int(dy),
                "new_x": int(new_x),
                "new_y": int(new_y),
                "note": f"Move {a} to the nearest grid-valid dock position relative to {b}.",
            }
        )

    return moves


def _suggest_moves_for_anchor_violations(
    *,
    placements: dict[str, dict[str, Any]],
    rects: dict[str, Rect],
    anchor_debug: list[dict[str, Any]],
    grid_mm: int,
) -> list[dict[str, Any]]:
    moves: list[dict[str, Any]] = []
    for ad in anchor_debug:
        a = ad.get("a")
        b = ad.get("b")
        mapped_side = ad.get("mapped_side")
        gap_min = ad.get("gap_min")
        gap_max = ad.get("gap_max")

        if not (
            isinstance(a, str) and isinstance(b, str) and isinstance(mapped_side, str)
        ):
            continue
        if a not in placements or a not in rects or b not in rects:
            continue
        if not isinstance(gap_min, int) or not isinstance(gap_max, int):
            continue

        ra = rects[a]
        rb = rects[b]
        axis, lo, hi = _side_axis_interval(
            ra=ra,
            rb=rb,
            mapped_side=mapped_side,
            gap_min=gap_min,
            gap_max=gap_max,
        )

        base_x = int(placements[a]["x"])
        base_y = int(placements[a]["y"])
        new_x = base_x
        new_y = base_y

        if axis == "x":
            new_x = _choose_grid_value_in_interval(
                current=base_x,
                interval=(lo, hi),
                grid_mm=grid_mm,
            )
        else:
            new_y = _choose_grid_value_in_interval(
                current=base_y,
                interval=(lo, hi),
                grid_mm=grid_mm,
            )

        dx = new_x - base_x
        dy = new_y - base_y

        if dx == 0 and dy == 0:
            if axis == "x":
                dx = _snap_delta(grid_mm if lo >= base_x else -grid_mm, grid_mm)
                new_x = base_x + dx
            else:
                dy = _snap_delta(grid_mm if lo >= base_y else -grid_mm, grid_mm)
                new_y = base_y + dy

        moves.append(
            {
                "reason": "ANCHOR_VIOLATION",
                "a": a,
                "b": b,
                "move_object": a,
                "dx": int(dx),
                "dy": int(dy),
                "new_x": int(new_x),
                "new_y": int(new_y),
                "note": f"Move {a} to the nearest grid-valid anchor position relative to {b}.",
            }
        )

    return moves


def _suggest_moves_for_rotation_violations(
    *,
    placements: dict[str, dict[str, Any]],
    rotation_violations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    moves: list[dict[str, Any]] = []
    for rv in rotation_violations:
        oid = rv.get("id")
        allowed = rv.get("allowed")
        if not isinstance(oid, str) or oid not in placements:
            continue
        if not isinstance(allowed, list) or not allowed:
            continue
        suggested_rot = allowed[0]
        if not isinstance(suggested_rot, int):
            continue
        moves.append(
            {
                "reason": "ROTATION_NOT_ALLOWED",
                "move_object": oid,
                "new_rot": int(suggested_rot),
                "note": "Rotation not allowed; switch to first allowed rotation.",
            }
        )
    return moves


def _suggest_moves_for_compaction(
    *,
    placements: dict[str, dict[str, Any]],
    rects: dict[str, Rect],
    relation_ctx: dict[str, Any],
    grid_mm: int,
    shape_quality: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Lightweight compaction hints for VALID states.
    These are only hints, not guaranteed fixes. The controller should re-verify them.
    Strategy:
    - Prefer moving leaf / non-base objects.
    - If an object lies on an outer bbox edge, try nudging it inward by one grid.
    """
    if not placements or not rects or grid_mm <= 0:
        return []

    bbox = shape_quality.get("bbox", {})
    min_x = int(bbox.get("min_x", 0))
    min_y = int(bbox.get("min_y", 0))
    max_x = int(bbox.get("max_x", 0))
    max_y = int(bbox.get("max_y", 0))

    base_to_deps = relation_ctx.get("base_to_dependents", {})
    dep_to_bases = relation_ctx.get("dependent_to_bases", {})
    dock_by_a = relation_ctx.get("dock_by_a", {})
    anchor_by_a = relation_ctx.get("anchor_by_a", {})
    contain_by_a = relation_ctx.get("contain_by_a", {})

    def movable_rank(oid: str) -> tuple[int, int, int]:
        has_deps = len(base_to_deps.get(oid, set())) > 0
        depends = len(dep_to_bases.get(oid, set())) > 0
        area = _rect_area(rects[oid])
        return (
            1 if has_deps else 0,
            0 if depends else 1,
            area,
        )

    moves: list[dict[str, Any]] = []
    for oid in sorted(rects.keys(), key=movable_rank):
        if len(base_to_deps.get(oid, set())) > 0:
            continue
        if oid in dock_by_a or oid in anchor_by_a or oid in contain_by_a:
            continue
        r = rects[oid]
        px = int(placements[oid]["x"])
        py = int(placements[oid]["y"])

        if r[0] == min_x:
            moves.append(
                {
                    "reason": "COMPACT_LEFT_EDGE",
                    "move_object": oid,
                    "dx": int(grid_mm),
                    "dy": 0,
                    "new_x": int(px + grid_mm),
                    "new_y": int(py),
                    "note": "Compaction hint: move bbox-left-edge object inward by one grid.",
                }
            )
        if r[2] == max_x:
            moves.append(
                {
                    "reason": "COMPACT_RIGHT_EDGE",
                    "move_object": oid,
                    "dx": int(-grid_mm),
                    "dy": 0,
                    "new_x": int(px - grid_mm),
                    "new_y": int(py),
                    "note": "Compaction hint: move bbox-right-edge object inward by one grid.",
                }
            )
        if r[1] == min_y:
            moves.append(
                {
                    "reason": "COMPACT_BOTTOM_EDGE",
                    "move_object": oid,
                    "dx": 0,
                    "dy": int(grid_mm),
                    "new_x": int(px),
                    "new_y": int(py + grid_mm),
                    "note": "Compaction hint: move bbox-bottom-edge object inward by one grid.",
                }
            )
        if r[3] == max_y:
            moves.append(
                {
                    "reason": "COMPACT_TOP_EDGE",
                    "move_object": oid,
                    "dx": 0,
                    "dy": int(-grid_mm),
                    "new_x": int(px),
                    "new_y": int(py - grid_mm),
                    "note": "Compaction hint: move bbox-top-edge object inward by one grid.",
                }
            )

    return moves


def _dedup_moves(moves: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for m in moves:
        key = (
            m.get("reason"),
            m.get("move_object"),
            m.get("dx"),
            m.get("dy"),
            m.get("new_x"),
            m.get("new_y"),
            m.get("new_rot"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(m)
    return out


def _tool_rank_key_from_scoring(
    *,
    grid_violations: list[dict[str, Any]],
    rot_violations: list[dict[str, Any]],
    scoring: dict[str, Any],
    shape_quality: dict[str, Any],
) -> tuple[float, ...]:
    hard = scoring.get("hard", {}) if isinstance(scoring, dict) else {}
    quality = shape_quality if isinstance(shape_quality, dict) else {}
    bbox = quality.get("bbox", {}) if isinstance(quality, dict) else {}
    lexi = scoring.get("lexicographic_key", []) if isinstance(scoring, dict) else []

    violated_count = int(hard.get("violated_count", 0) or 0)
    grid_count = len(grid_violations)
    rot_count = len(rot_violations)

    compact_score = float(quality.get("compact_score", 10**15) or 10**15)
    bbox_area = float(bbox.get("area_mm2", 10**15) or 10**15)
    max_span = float(bbox.get("max_span_mm", 10**15) or 10**15)
    fill_ratio_bbox = float(quality.get("fill_ratio_bbox", 0.0) or 0.0)
    fill_ratio_hull = float(quality.get("fill_ratio_hull", 0.0) or 0.0)
    aspect_ratio = float(bbox.get("aspect_ratio", 10**9) or 10**9)

    lexi_vals: list[float] = []
    if isinstance(lexi, list):
        for x in lexi[:5]:
            try:
                lexi_vals.append(float(x))
            except Exception:
                lexi_vals.append(10**12)
    while len(lexi_vals) < 5:
        lexi_vals.append(10**12)

    return (
        0.0 if (grid_count == 0 and rot_count == 0 and violated_count == 0) else 1.0,
        float(grid_count + rot_count + violated_count),
        *lexi_vals,
        compact_score,
        bbox_area,
        max_span,
        -fill_ratio_bbox,
        -fill_ratio_hull,
        aspect_ratio,
    )


def _count_grid_rotation_violations(
    *,
    placements: list[dict[str, Any]],
    objects: dict[str, dict[str, Any]],
    grid_mm: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grid_violations: list[dict[str, Any]] = []
    rot_violations: list[dict[str, Any]] = []

    for p in placements:
        if not isinstance(p, dict):
            continue
        oid = p.get("id")
        if not isinstance(oid, str) or oid not in objects:
            continue

        x = int(p.get("x", 0))
        y = int(p.get("y", 0))
        rot = int(p.get("rot", 0)) % 360

        if grid_mm > 0 and (x % grid_mm != 0 or y % grid_mm != 0):
            grid_violations.append({"id": oid, "x": x, "y": y})

        allowed = objects[oid].get("allowed_rotations") or objects[oid].get("rotations")
        if isinstance(allowed, list) and allowed and rot not in allowed:
            rot_violations.append({"id": oid, "rot": rot, "allowed": allowed})

    return grid_violations, rot_violations


def _apply_move_to_placements(
    placements: list[dict[str, Any]],
    move: dict[str, Any],
) -> list[dict[str, Any]]:
    move_object = move.get("move_object")
    if not isinstance(move_object, str):
        return [dict(p) for p in placements if isinstance(p, dict)]

    out: list[dict[str, Any]] = []
    for p in placements:
        if not isinstance(p, dict):
            continue

        q = {
            "id": str(p.get("id")),
            "x": int(p.get("x", 0)),
            "y": int(p.get("y", 0)),
            "rot": int(p.get("rot", 0)) % 360,
        }

        if q["id"] == move_object:
            if move.get("new_x") is not None:
                q["x"] = int(move["new_x"])
            else:
                q["x"] += int(move.get("dx", 0) or 0)

            if move.get("new_y") is not None:
                q["y"] = int(move["new_y"])
            else:
                q["y"] += int(move.get("dy", 0) or 0)

            if move.get("new_rot") is not None:
                q["rot"] = int(move["new_rot"]) % 360

        out.append(q)

    out.sort(key=lambda item: item["id"])
    return out


def _build_preferred_patches(
    *,
    hard_constraints: list[dict[str, Any]],
    soft_constraints: list[dict[str, Any]],
    objects: dict[str, dict[str, Any]],
    placements: list[dict[str, Any]],
    grid_mm: int,
    cluster_rules: dict[str, Any] | None,
    use_clearance: bool,
    suggested_moves: list[dict[str, Any]],
    current_rank_key: tuple[float, ...],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    ranked: list[tuple[tuple[float, ...], dict[str, Any]]] = []

    seen_patch_sig: set[str] = set()

    for move in suggested_moves:
        if not isinstance(move, dict):
            continue

        patched = _apply_move_to_placements(placements, move)

        patch_sig = str(patched)
        if patch_sig in seen_patch_sig:
            continue
        seen_patch_sig.add(patch_sig)

        grid_v, rot_v = _count_grid_rotation_violations(
            placements=patched,
            objects=objects,
            grid_mm=grid_mm,
        )

        scoring = score_cluster_constraints(
            hard_constraints=hard_constraints,
            soft_constraints=soft_constraints,
            objects=objects,
            local_placements=patched,
            grid_mm=grid_mm,
            cluster_rules=cluster_rules,
            use_clearance=use_clearance,
        )

        quality = scoring.get("layout_quality", _empty_quality())
        rank_key = _tool_rank_key_from_scoring(
            grid_violations=grid_v,
            rot_violations=rot_v,
            scoring=scoring,
            shape_quality=quality,
        )

        if rank_key >= current_rank_key:
            continue

        ranked.append(
            (
                rank_key,
                {
                    "selected_move": move,
                    "patched_local_placements": patched,
                    "rank_key": list(rank_key),
                    "result": "VALID" if rank_key[0] == 0.0 else "INVALID",
                    "hard_violated_count": int(
                        (
                            scoring.get("hard", {}) if isinstance(scoring, dict) else {}
                        ).get("violated_count", 0)
                        or 0
                    ),
                    "compact_score": quality.get("compact_score"),
                },
            )
        )

    ranked.sort(key=lambda x: x[0])
    return [item for _, item in ranked[:top_k]]


# ============================================================
# Error sorting
# ============================================================


def _error_priority(code: Any) -> int:
    order = {
        "MISSING_OBJECT_SPECS": 0,
        "INVALID_DIMS": 0,
        "UNKNOWN_OBJECT": 1,
        "GRID_VIOLATION": 2,
        "ROTATION_NOT_ALLOWED": 3,
        "CONTAIN_VIOLATION": 4,
        "DOCK_VIOLATION": 5,
        "ANCHOR_VIOLATION": 5,
        "OVERLAP": 6,
        "ACCESS_BLOCKED": 7,
        "LOCAL_BBOX_BUDGET_EXCEEDED": 8,
        "LOCAL_OUTLINE_BUDGET_EXCEEDED": 8,
        "LOCAL_HULL_BUDGET_EXCEEDED": 8,
        "LOCAL_FILL_RATIO_TOO_LOW": 8,
    }
    return order.get(code, 99)


def _sort_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        errors,
        key=lambda e: (
            _error_priority(e.get("code")),
            str(e.get("a") or ""),
            str(e.get("b") or ""),
            str(e.get("detail") or ""),
        ),
    )


def _side_to_vec(side: str | None) -> dict[str, int] | None:
    if not isinstance(side, str):
        return None
    side = side.lower().strip()
    if side == "right":
        return {"dx": 1, "dy": 0}
    if side == "left":
        return {"dx": -1, "dy": 0}
    if side == "top":
        return {"dx": 0, "dy": 1}
    if side == "bottom":
        return {"dx": 0, "dy": -1}
    return None


def _rotate_vec_ccw_90s_local(
    vec: dict[str, int] | None, rot: int
) -> dict[str, int] | None:
    if not isinstance(vec, dict):
        return None
    dx = int(vec.get("dx", 0))
    dy = int(vec.get("dy", 0))
    r = rot % 360

    if r == 0:
        return {"dx": dx, "dy": dy}
    if r == 90:
        return {"dx": -dy, "dy": dx}
    if r == 180:
        return {"dx": -dx, "dy": -dy}
    if r == 270:
        return {"dx": dy, "dy": -dx}
    return {"dx": dx, "dy": dy}


def _infer_base_axis_from_spec(spec: dict[str, Any]) -> dict[str, int]:
    w = int(spec.get("w", 0) or 0)
    h = int(spec.get("h", 0) or 0)
    # deterministic:
    # - object dài theo ngang -> axis = +X
    # - object dài theo dọc -> axis = +Y
    # - vuông -> mặc định +X
    if h > w:
        return {"dx": 0, "dy": 1}
    return {"dx": 1, "dy": 0}


def _rect_center(rect: Rect) -> dict[str, int]:
    x1, y1, x2, y2 = rect
    return {
        "x": int((x1 + x2) // 2),
        "y": int((y1 + y2) // 2),
    }


def _dominant_cluster_axis_from_objects(
    object_meta: dict[str, dict[str, Any]],
) -> dict[str, int]:
    score_x = 0
    score_y = 0

    for item in object_meta.values():
        axis = item.get("axis_local")
        area = int(item.get("area_mm2", 0) or 0)
        if not isinstance(axis, dict):
            continue
        if int(axis.get("dx", 0)) != 0:
            score_x += max(area, 1)
        if int(axis.get("dy", 0)) != 0:
            score_y += max(area, 1)

    if score_y > score_x:
        return {"dx": 0, "dy": 1}
    return {"dx": 1, "dy": 0}


def _dominant_cluster_front_from_objects(
    object_meta: dict[str, dict[str, Any]],
    cluster_axis_local: dict[str, int],
) -> dict[str, int]:
    scores = {
        (1, 0): 0,
        (-1, 0): 0,
        (0, 1): 0,
        (0, -1): 0,
    }

    for item in object_meta.values():
        front = item.get("front_local")
        if not isinstance(front, dict):
            continue

        key = (int(front.get("dx", 0)), int(front.get("dy", 0)))
        if key not in scores:
            continue

        area = int(item.get("area_mm2", 0) or 0)
        weight = max(area, 1)

        if bool(item.get("has_front_access")):
            weight *= 2
        if bool(item.get("is_anchor_like")):
            weight = int(weight * 1.5)

        scores[key] += weight

    best_key = max(scores.items(), key=lambda kv: (kv[1], kv[0]))[0]
    if scores[best_key] > 0:
        return {"dx": best_key[0], "dy": best_key[1]}

    # fallback deterministic, but NOT semantic hardcode per object
    # just for debug when nothing directional exists
    if int(cluster_axis_local.get("dx", 0)) != 0:
        return {"dx": 0, "dy": 1}
    return {"dx": 1, "dy": 0}


def _build_orientation_inference_debug(
    *,
    specs: dict[str, dict[str, Any]],
    placements: dict[str, dict[str, Any]],
    rects: dict[str, Rect],
    relation_ctx: dict[str, Any],
    access_required_ids: set[str],
    facing_map: dict[str, Any],
) -> dict[str, Any]:
    object_meta: dict[str, dict[str, Any]] = {}

    dep_to_base = relation_ctx.get("dependent_to_bases", {}) or {}
    base_to_dep = relation_ctx.get("base_to_dependents", {}) or {}

    for oid in sorted(placements.keys()):
        spec = specs.get(oid, {})
        p = placements.get(oid, {})
        rect = rects.get(oid)
        if rect is None:
            continue

        rot = int(p.get("rot", 0)) % 360
        area = _rect_area(rect)

        # front gốc lấy từ facing/spec, rồi rotate theo object rot thực tế
        base_front_side = _get_front_base(oid, facing_map, specs)
        base_front_vec = _side_to_vec(base_front_side)
        front_local = _rotate_vec_ccw_90s_local(base_front_vec, rot)

        # axis gốc lấy từ kích thước base object, rồi rotate theo rot thực tế
        base_axis_vec = _infer_base_axis_from_spec(spec)
        axis_local = _rotate_vec_ccw_90s_local(base_axis_vec, rot)

        object_meta[oid] = {
            "rect": _rect_dict(rect),
            "center": _rect_center(rect),
            "rot": rot,
            "area_mm2": int(area),
            "base_front_side": base_front_side,
            "front_local": front_local,
            "axis_local": axis_local,
            "has_front_access": oid in access_required_ids,
            "depends_on": sorted(list(dep_to_base.get(oid, set()))),
            "anchors": sorted(list(base_to_dep.get(oid, set()))),
            "is_anchor_like": len(base_to_dep.get(oid, set())) > 0,
            "important_for_orientation": bool(front_local)
            or (oid in access_required_ids)
            or (len(base_to_dep.get(oid, set())) > 0),
        }

    cluster_axis_local = _dominant_cluster_axis_from_objects(object_meta)
    cluster_front_local = _dominant_cluster_front_from_objects(
        object_meta,
        cluster_axis_local,
    )

    return {
        "cluster_axis_local_candidate": cluster_axis_local,
        "cluster_front_local_candidate": cluster_front_local,
        "objects": object_meta,
    }


# ------------------------------------------------------------
# Tool registry + schema
# ------------------------------------------------------------

TOOL_REGISTRY: dict[str, Any] = {"LocalClusterVerifier": local_cluster_verifier}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "LocalClusterVerifier",
            "description": (
                "Validate local cluster placements against hard constraints: overlap, grid, clearance_mm, "
                "dock_to_edge, contain_in, anchor_side, and front_clearance (requires_access). Rotation is CCW. "
                "Also performs implicit global body-overlap checks for floor-occupying objects. "
                "Objects docked to a target's front are allowed inside that target's front_clearance. "
                "Uses fixed access_clearance_ratio = 0.25. "
                "Delegates hard-constraint scoring to constraint_score.py and returns debug rects, "
                "quality metrics (outline/bbox/hull/fill ratio/compact score), suggested moves, and preferred patches. "
                "This tool validates local_placements only; orientation_meta is not validated here. "
                "It also returns debug.orientation_inference, which contains per-object effective front_local/axis_local "
                "derived from the actual verified layout so the composer can build final orientation_meta without hardcoded defaults."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "hard_constraints": {"type": "array", "items": {"type": "object"}},
                    "soft_constraints": {"type": "array", "items": {"type": "object"}},
                    "objects": {
                        "oneOf": [
                            {"type": "array", "items": {"type": "object"}},
                            {"type": "object"},
                        ]
                    },
                    "local_placements": {"type": "array", "items": {"type": "object"}},
                    "grid_mm": {"type": "integer"},
                    "use_clearance": {"type": "boolean"},
                    "cluster_rules": {"type": ["object", "null"]},
                },
                "required": [
                    "hard_constraints",
                    "objects",
                    "local_placements",
                    "grid_mm",
                ],
            },
        },
    }
]
