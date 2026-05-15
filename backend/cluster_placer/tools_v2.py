from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple, TypedDict

try:
    from shapely.geometry import LineString, Point, Polygon
    from shapely.ops import nearest_points, unary_union
except Exception as e:  # pragma: no cover
    raise RuntimeError("phase2_placer_tools_v3 requires shapely") from e

from cluster_composer.constraint_score import (
    build_object_specs_from_cluster,
    build_rects,
    edge_gap,
    get_front_base,
    normalize_objects,
    normalize_placements,
    perpendicular_overlap_len,
    resolve_anchor_side,
    resolve_edge_token_to_base_side,
    rotate_side,
    score_cluster_constraints,
)
from layout.grid_policy import GLOBAL_LAYOUT_GRID_MM, normalize_layout_grid_mm
from layout.orientation_contract import rotation_from_front_world, vec_to_side


class ClusterPreferenceProfile(TypedDict):
    prefer: set[str]
    avoid: set[str]
    intents: set[str]
    viewing_partner: str | None
    viewing_role: str | None


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str | Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------
def _fix_geom(geom: Any) -> Any:
    try:
        if hasattr(geom, "is_valid") and not geom.is_valid:
            geom = geom.buffer(0)
    except Exception:
        pass
    return geom


def rotate_point_ccw_90s(x: float, y: float, rot: int) -> tuple[float, float]:
    r = int(rot) % 360
    if r == 0:
        return x, y
    if r == 90:
        return -y, x
    if r == 180:
        return -x, -y
    if r == 270:
        return y, -x
    raise ValueError(f"Unsupported rot={rot}")


def rotate_vec_ccw_90s(
    vec: tuple[float, float] | None, rot: int
) -> tuple[float, float] | None:
    if vec is None:
        return None
    return rotate_point_ccw_90s(float(vec[0]), float(vec[1]), int(rot))


def normalize_vec(vec: tuple[float, float] | None) -> tuple[float, float] | None:
    if vec is None:
        return None
    x, y = float(vec[0]), float(vec[1])
    norm = (x * x + y * y) ** 0.5
    if norm <= 1e-9:
        return None
    return (x / norm, y / norm)


def parse_vec2(value: Any) -> tuple[float, float] | None:
    if isinstance(value, dict):
        dx = value.get("dx")
        dy = value.get("dy")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        dx, dy = value[0], value[1]
    else:
        return None
    try:
        return normalize_vec((float(dx), float(dy)))
    except Exception:
        return None


def mirror_vec(
    vec: tuple[float, float] | None, axis: str
) -> tuple[float, float] | None:
    v = normalize_vec(vec)
    if v is None:
        return None
    axis = str(axis).lower()
    if axis == "x":
        return (-v[0], v[1])
    if axis == "y":
        return (v[0], -v[1])
    return v


def transform_rect_world(
    rect: dict[str, Any], tx: int, ty: int, rot: int
) -> dict[str, Any]:
    x = float(rect.get("x", 0))
    y = float(rect.get("y", 0))
    w = float(rect.get("w", 0))
    h = float(rect.get("h", 0))
    corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    world = [rotate_point_ccw_90s(px, py, rot) for px, py in corners]
    world = [(px + tx, py + ty) for px, py in world]
    xs = [p[0] for p in world]
    ys = [p[1] for p in world]
    return {
        "polygon_ccw": [{"x": int(round(px)), "y": int(round(py))} for px, py in world],
        "bbox": {
            "min_x": int(round(min(xs))),
            "min_y": int(round(min(ys))),
            "max_x": int(round(max(xs))),
            "max_y": int(round(max(ys))),
        },
        "world_center": {
            "x": int(round(sum(xs) / 4.0)),
            "y": int(round(sum(ys) / 4.0)),
        },
    }


def _polygon_from_points(points: List[Dict[str, Any]]) -> Any:
    return _fix_geom(Polygon([(float(p["x"]), float(p["y"])) for p in points]))


def _bbox_poly(bbox: Dict[str, Any]) -> Any:
    return _fix_geom(
        Polygon(
            [
                (float(bbox["min_x"]), float(bbox["min_y"])),
                (float(bbox["max_x"]), float(bbox["min_y"])),
                (float(bbox["max_x"]), float(bbox["max_y"])),
                (float(bbox["min_x"]), float(bbox["max_y"])),
            ]
        )
    )


# -----------------------------------------------------------------------------
# Payload helpers
# -----------------------------------------------------------------------------
def _cluster_cards_as_map(cluster_cards: Any) -> Dict[str, Dict[str, Any]]:
    if isinstance(cluster_cards, dict):
        return {
            str(k): deepcopy(v)
            for k, v in cluster_cards.items()
            if isinstance(k, str) and isinstance(v, dict)
        }
    if isinstance(cluster_cards, list):
        out: Dict[str, Dict[str, Any]] = {}
        for row in cluster_cards:
            if isinstance(row, dict) and isinstance(row.get("cluster_id"), str):
                out[row["cluster_id"]] = deepcopy(row)
        return out
    return {}


def _seed_transform_map(seed_layout: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for item in seed_layout.get("cluster_transforms") or []:
        if isinstance(item, dict) and isinstance(item.get("cluster_id"), str):
            out[item["cluster_id"]] = deepcopy(item)
    return out


def _seed_variant_map(seed_layout: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for item in seed_layout.get("selected_variants") or []:
        if isinstance(item, dict) and isinstance(item.get("cluster_id"), str):
            out[item["cluster_id"]] = deepcopy(item)
    return out


def _resolved_seed_layout(
    payload: Dict[str, Any], repair: Dict[str, Any] | None = None
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    seed_layout = payload.get("seed_layout") or {}
    tmap = _seed_transform_map(seed_layout)
    vmap = _seed_variant_map(seed_layout)

    for row in (repair or {}).get("cluster_transforms") or []:
        if isinstance(row, dict) and isinstance(row.get("cluster_id"), str):
            tmap[row["cluster_id"]] = deepcopy(row)

    for row in (repair or {}).get("selected_variants") or []:
        if isinstance(row, dict) and isinstance(row.get("cluster_id"), str):
            vmap[row["cluster_id"]] = deepcopy(row)

    cluster_ids = sorted(tmap.keys())
    return (
        [deepcopy(tmap[cid]) for cid in cluster_ids],
        [deepcopy(vmap[cid]) for cid in cluster_ids if cid in vmap],
    )


def _full_seed_repair(
    payload: Dict[str, Any],
    repair: Dict[str, Any] | None = None,
    *,
    status: str = "REPAIRED",
    notes: list[str] | None = None,
) -> Dict[str, Any]:
    cluster_transforms, selected_variants = _resolved_seed_layout(payload, repair)
    merged_notes: list[str] = []
    for source in (((repair or {}).get("notes") or []), notes or []):
        for item in source:
            text = str(item).strip()
            if text and text not in merged_notes:
                merged_notes.append(text)
    return {
        "status": status,
        "cluster_transforms": cluster_transforms,
        "selected_variants": selected_variants,
        "object_repairs": deepcopy((repair or {}).get("object_repairs") or []),
        "notes": merged_notes,
    }


def _room_model(payload: Dict[str, Any]) -> Dict[str, Any]:
    return deepcopy(((payload.get("room_context") or {}).get("room_model_used") or {}))


def _room_notes(payload: Dict[str, Any]) -> List[str]:
    notes = _room_model(payload).get("notes") or []
    out: List[str] = []
    for item in notes:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return out


def _repair_phase(payload: Dict[str, Any]) -> str:
    phase_control = payload.get("phase_control") or {}
    value = str(phase_control.get("repair_phase") or "").strip().lower()
    if value == "object_refine":
        return "object_refine"
    return "macro_layout"


def _room_polygon(payload: Dict[str, Any]) -> Any:
    room_model = _room_model(payload)
    pts = ((room_model.get("room") or {}).get("polygon_ccw")) or []
    return _polygon_from_points(pts)


def _hard_obstacles(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for ob in _room_model(payload).get("obstacles") or []:
        if not isinstance(ob, dict) or not ob.get("hard", True):
            continue
        pts = ob.get("polygon_ccw") or []
        if not isinstance(pts, list) or len(pts) < 3:
            continue
        out.append(
            {
                "id": str(ob.get("id") or "obstacle"),
                "type": str(ob.get("type") or "unknown"),
                "poly": _polygon_from_points(pts),
            }
        )
    return out


def _openings(payload: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    openings = _room_model(payload).get("openings") or {}
    out = {"doors": [], "windows": []}
    for key in ("doors", "windows"):
        for item in openings.get(key) or []:
            if not isinstance(item, dict):
                continue
            seg = item.get("segment_mm") or []
            if not isinstance(seg, list) or len(seg) != 2:
                continue
            p1 = (float(seg[0]["x"]), float(seg[0]["y"]))
            p2 = (float(seg[1]["x"]), float(seg[1]["y"]))
            out[key].append(
                {
                    "id": str(item.get("id") or ""),
                    "midpoint": ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0),
                    "line": LineString([p1, p2]),
                }
            )
    return out


def _nearest_midpoint(
    point: Tuple[float, float], items: List[Dict[str, Any]]
) -> Tuple[float, float] | None:
    best = None
    best_d = None
    px, py = point
    for item in items:
        mid = item.get("midpoint")
        if not isinstance(mid, tuple):
            continue
        d = math.hypot(mid[0] - px, mid[1] - py)
        if best_d is None or d < best_d:
            best = mid
            best_d = d
    return best


def _vec_from_to(
    a: Tuple[float, float], b: Tuple[float, float]
) -> Tuple[float, float] | None:
    return normalize_vec((float(b[0]) - float(a[0]), float(b[1]) - float(a[1])))


def _dot(a: Tuple[float, float] | None, b: Tuple[float, float] | None) -> float | None:
    va = normalize_vec(a)
    vb = normalize_vec(b)
    if va is None or vb is None:
        return None
    return max(-1.0, min(1.0, va[0] * vb[0] + va[1] * vb[1]))


# -----------------------------------------------------------------------------
# Unified object state materialization
# -----------------------------------------------------------------------------
def _cluster_local_bbox(card: Dict[str, Any]) -> Dict[str, int]:
    fp = card.get("cluster_footprint") or {}
    bbox = fp.get("local_bbox") or {}
    try:
        min_x = int(round(float(bbox.get("min_x", 0))))
        min_y = int(round(float(bbox.get("min_y", 0))))
        max_x = int(round(float(bbox.get("max_x", 0))))
        max_y = int(round(float(bbox.get("max_y", 0))))
    except Exception:
        min_x = min_y = max_x = max_y = 0
    if max_x <= min_x or max_y <= min_y:
        xs: List[int] = []
        ys: List[int] = []
        for rect in fp.get("rects") or []:
            if not isinstance(rect, dict):
                continue
            x = int(round(float(rect.get("x", 0))))
            y = int(round(float(rect.get("y", 0))))
            w = int(round(float(rect.get("w", 0))))
            h = int(round(float(rect.get("h", 0))))
            xs.extend([x, x + w])
            ys.extend([y, y + h])
        if xs and ys:
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
    return {"min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y}


def _rotate_rect_about_center(rect: Dict[str, Any], rot: int) -> Dict[str, Any]:
    rot = int(rot) % 360
    x = float(rect.get("x", 0))
    y = float(rect.get("y", 0))
    w = float(rect.get("w", 0))
    h = float(rect.get("h", 0))
    if rot in {0, 180}:
        return {
            **deepcopy(rect),
            "x": int(round(x)),
            "y": int(round(y)),
            "w": int(round(w)),
            "h": int(round(h)),
        }
    cx = x + w / 2.0
    cy = y + h / 2.0
    nw, nh = h, w
    return {
        **deepcopy(rect),
        "x": int(round(cx - nw / 2.0)),
        "y": int(round(cy - nh / 2.0)),
        "w": int(round(nw)),
        "h": int(round(nh)),
    }


def _mirror_rect_in_bbox(
    rect: Dict[str, Any], bbox: Dict[str, int], axis: str
) -> Dict[str, Any]:
    x = int(round(float(rect.get("x", 0))))
    y = int(round(float(rect.get("y", 0))))
    w = int(round(float(rect.get("w", 0))))
    h = int(round(float(rect.get("h", 0))))
    out = deepcopy(rect)
    axis = str(axis).lower()
    if axis == "x":
        out["x"] = int(bbox["max_x"] - (x + w) + bbox["min_x"])
        out["y"] = y
    else:
        out["x"] = x
        out["y"] = int(bbox["max_y"] - (y + h) + bbox["min_y"])
    out["w"] = w
    out["h"] = h
    return out


def _current_cluster_variant_card(
    payload: Dict[str, Any], cluster_id: str, variant_id: str | None
) -> Dict[str, Any]:
    base = _cluster_cards_as_map(payload.get("cluster_cards") or {}).get(cluster_id)
    if not isinstance(base, dict):
        return {}
    if not variant_id:
        return deepcopy(base)
    variants = base.get("available_variants") or base.get("variants") or []
    for item in variants:
        if (
            isinstance(item, dict)
            and item.get("variant_id") == variant_id
            and isinstance(item.get("card"), dict)
        ):
            out = deepcopy(item["card"])
            out.setdefault("cluster_id", cluster_id)
            return out
    return deepcopy(base)


def materialize_phase2_state(
    payload: Dict[str, Any], repair: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    repair = deepcopy(repair or {})
    seed_layout = payload.get("seed_layout") or {}
    tmap = _seed_transform_map(seed_layout)
    vmap = _seed_variant_map(seed_layout)
    for row in repair.get("cluster_transforms") or []:
        if isinstance(row, dict) and isinstance(row.get("cluster_id"), str):
            tmap[row["cluster_id"]] = deepcopy(row)
    for row in repair.get("selected_variants") or []:
        if isinstance(row, dict) and isinstance(row.get("cluster_id"), str):
            vmap[row["cluster_id"]] = deepcopy(row)

    repairs_by_cluster: Dict[str, List[Dict[str, Any]]] = {}
    for rep in repair.get("object_repairs") or []:
        if isinstance(rep, dict) and isinstance(rep.get("cluster_id"), str):
            repairs_by_cluster.setdefault(rep["cluster_id"], []).append(deepcopy(rep))

    room_poly = _room_polygon(payload)
    room_center = (float(room_poly.centroid.x), float(room_poly.centroid.y))
    obstacles = _hard_obstacles(payload)

    clusters: List[Dict[str, Any]] = []
    objects: List[Dict[str, Any]] = []
    object_overlap_pairs: List[Dict[str, Any]] = []
    hard_violations: List[Dict[str, Any]] = []

    for cid in sorted(tmap.keys()):
        tf = tmap[cid]
        variant_id = None
        if cid in vmap:
            variant_id = str(vmap[cid].get("variant_id") or "") or None
        card = _current_cluster_variant_card(payload, cid, variant_id)
        if not card:
            hard_violations.append({"code": "MISSING_CLUSTER_CARD", "cluster_id": cid})
            continue
        bbox = _cluster_local_bbox(card)
        rects = []
        for rect in (card.get("cluster_footprint") or {}).get("rects") or []:
            if isinstance(rect, dict) and isinstance(rect.get("id"), str):
                rects.append(deepcopy(rect))
        rect_map = {r["id"]: deepcopy(r) for r in rects}
        local_placement_map: Dict[str, Dict[str, Any]] = {}
        for row in card.get("local_placements") or []:
            if isinstance(row, dict) and isinstance(row.get("id"), str):
                local_placement_map[row["id"]] = {
                    "id": row["id"],
                    "x": int(round(float(row.get("x", 0)))),
                    "y": int(round(float(row.get("y", 0)))),
                    "rot": int(round(float(row.get("rot", 0)))) % 360,
                }
        for oid, rect in rect_map.items():
            local_placement_map.setdefault(
                oid,
                {
                    "id": oid,
                    "x": int(round(float(rect.get("x", 0)))),
                    "y": int(round(float(rect.get("y", 0)))),
                    "rot": 0,
                },
            )
        important = deepcopy(
            (card.get("orientation_meta") or {}).get("important_objects") or {}
        )
        cluster_front_local = parse_vec2(
            (card.get("orientation_meta") or {}).get("cluster_front_local")
        )
        cluster_axis_local = parse_vec2(
            (card.get("orientation_meta") or {}).get("cluster_axis_local")
        )
        cluster_meta = {"anchor_override": None}

        for rep in repairs_by_cluster.get(cid, []):
            oid = rep.get("object_id")
            op = rep.get("op")
            params = rep.get("params") or {}
            if op == "set_anchor":
                cluster_meta["anchor_override"] = params.get("anchor")
                continue
            if not isinstance(oid, str) or oid not in rect_map:
                continue
            rect = rect_map[oid]
            placement = deepcopy(
                local_placement_map.get(oid)
                or {
                    "id": oid,
                    "x": int(round(float(rect.get("x", 0)))),
                    "y": int(round(float(rect.get("y", 0)))),
                    "rot": 0,
                }
            )
            obj_meta = (
                important.get(oid)
                if isinstance(important, dict) and isinstance(important.get(oid), dict)
                else {}
            )
            front_local = (
                parse_vec2((obj_meta or {}).get("front_local")) or cluster_front_local
            )
            axis_local = parse_vec2((obj_meta or {}).get("axis_local"))

            if op == "rotate_object":
                rr = int(params.get("rot", 0)) % 360
                rect = _rotate_rect_about_center(rect, rr)
                placement["x"] = int(round(float(rect.get("x", 0))))
                placement["y"] = int(round(float(rect.get("y", 0))))
                placement["rot"] = (int(placement.get("rot") or 0) + rr) % 360
                front_local = rotate_vec_ccw_90s(front_local, rr)
                axis_local = rotate_vec_ccw_90s(axis_local, rr)
            elif op == "mirror_object":
                axis = str(params.get("axis") or "x").lower()
                rect = _mirror_rect_in_bbox(rect, bbox, axis)
                placement["x"] = int(round(float(rect.get("x", 0))))
                placement["y"] = int(round(float(rect.get("y", 0))))
                front_local = mirror_vec(front_local, axis)
                axis_local = mirror_vec(axis_local, axis)
            elif op == "nudge_object":
                rect = {
                    **rect,
                    "x": int(
                        round(float(rect.get("x", 0)) + float(params.get("dx", 0)))
                    ),
                    "y": int(
                        round(float(rect.get("y", 0)) + float(params.get("dy", 0)))
                    ),
                }
                placement["x"] = int(round(float(rect.get("x", 0))))
                placement["y"] = int(round(float(rect.get("y", 0))))
            elif op == "swap_objects":
                other_id = str(params.get("other_object_id") or "")
                other = rect_map.get(other_id)
                if isinstance(other, dict):
                    ox, oy = other.get("x"), other.get("y")
                    other["x"], other["y"] = rect.get("x"), rect.get("y")
                    rect["x"], rect["y"] = ox, oy
                    rect_map[other_id] = other
                    other_placement = deepcopy(
                        local_placement_map.get(other_id)
                        or {
                            "id": other_id,
                            "x": int(round(float(other.get("x", 0)))),
                            "y": int(round(float(other.get("y", 0)))),
                            "rot": 0,
                        }
                    )
                    px = int(other_placement.get("x") or 0)
                    py = int(other_placement.get("y") or 0)
                    other_placement["x"], other_placement["y"] = (
                        int(placement.get("x") or 0),
                        int(placement.get("y") or 0),
                    )
                    placement["x"], placement["y"] = px, py
                    local_placement_map[other_id] = other_placement
            elif op == "set_front_override":
                front_local = normalize_vec(
                    (float(params.get("dx", 0.0)), float(params.get("dy", 0.0)))
                )

            rect_map[oid] = rect
            local_placement_map[oid] = placement
            if oid not in important:
                important[oid] = {}
            if front_local is not None:
                important[oid]["front_local"] = {
                    "dx": round(front_local[0], 3),
                    "dy": round(front_local[1], 3),
                }
            if axis_local is not None:
                important[oid]["axis_local"] = {
                    "dx": round(axis_local[0], 3),
                    "dy": round(axis_local[1], 3),
                }

        tx = int(tf.get("x") or 0)
        ty = int(tf.get("y") or 0)
        crot = int(tf.get("rot") or 0) % 360

        cluster_object_indices: List[int] = []
        for oid, rect in rect_map.items():
            geom = transform_rect_world(rect, tx, ty, crot)
            poly = _bbox_poly(geom["bbox"])
            placement = local_placement_map.get(oid) or {}
            obj_meta = (
                important.get(oid) if isinstance(important.get(oid), dict) else {}
            )
            front_local = (
                parse_vec2((obj_meta or {}).get("front_local")) or cluster_front_local
            )
            axis_local = parse_vec2((obj_meta or {}).get("axis_local"))
            front_world = normalize_vec(rotate_vec_ccw_90s(front_local, crot))
            axis_world = normalize_vec(rotate_vec_ccw_90s(axis_local, crot))
            w = int(round(float(rect.get("w", 0))))
            h = int(round(float(rect.get("h", 0))))
            if front_local is None:
                size_along_access = max(w, h)
            else:
                dx, dy = abs(front_local[0]), abs(front_local[1])
                size_along_access = w if dx >= dy else h
            required_clearance = int(round(0.25 * size_along_access))
            object_rot = (int(placement.get("rot") or 0) + crot) % 360
            row = {
                "cluster_id": cid,
                "variant_id": variant_id,
                "object_id": oid,
                "cluster_rot": crot,
                "rotation_ccw": object_rot,
                "local_rect": {
                    "x": int(round(float(rect.get("x", 0)))),
                    "y": int(round(float(rect.get("y", 0)))),
                    "w": w,
                    "h": h,
                },
                "polygon_ccw": geom["polygon_ccw"],
                "bbox": geom["bbox"],
                "world_center": geom["world_center"],
                "poly": poly,
                "front_world": None
                if front_world is None
                else {"dx": round(front_world[0], 3), "dy": round(front_world[1], 3)},
                "axis_world": None
                if axis_world is None
                else {"dx": round(axis_world[0], 3), "dy": round(axis_world[1], 3)},
                "required_clearance_mm": required_clearance,
            }
            cluster_object_indices.append(len(objects))
            objects.append(row)

            if not room_poly.buffer(1e-6).covers(poly):
                hard_violations.append(
                    {
                        "code": "OBJECT_OUT_OF_BOUNDS",
                        "cluster_id": cid,
                        "object_id": oid,
                    }
                )
            for ob in obstacles:
                if poly.intersects(ob["poly"]):
                    hard_violations.append(
                        {
                            "code": "OBJECT_HITS_OBSTACLE",
                            "cluster_id": cid,
                            "object_id": oid,
                            "obstacle_id": ob["id"],
                        }
                    )

        # cluster bbox from member objects
        if cluster_object_indices:
            xs1 = [objects[i]["bbox"]["min_x"] for i in cluster_object_indices]
            ys1 = [objects[i]["bbox"]["min_y"] for i in cluster_object_indices]
            xs2 = [objects[i]["bbox"]["max_x"] for i in cluster_object_indices]
            ys2 = [objects[i]["bbox"]["max_y"] for i in cluster_object_indices]
            cluster_poly = _fix_geom(
                unary_union([objects[i]["poly"] for i in cluster_object_indices])
            )
            if cluster_poly is None or getattr(cluster_poly, "is_empty", True):
                cluster_poly = _bbox_poly(
                    {
                        "min_x": min(xs1),
                        "min_y": min(ys1),
                        "max_x": max(xs2),
                        "max_y": max(ys2),
                    }
                )
            cluster_bbox = {
                "min_x": min(xs1),
                "min_y": min(ys1),
                "max_x": max(xs2),
                "max_y": max(ys2),
            }
            centroid = (float(cluster_poly.centroid.x), float(cluster_poly.centroid.y))
            clusters.append(
                {
                    "cluster_id": cid,
                    "variant_id": variant_id,
                    "bbox": cluster_bbox,
                    "world_center": {
                        "x": int(round(centroid[0])),
                        "y": int(round(centroid[1])),
                    },
                    "front_world": None
                    if cluster_front_local is None
                    else {
                        "dx": round(
                            (
                                rotate_vec_ccw_90s(cluster_front_local, crot)
                                or (0.0, 0.0)
                            )[0],
                            3,
                        ),
                        "dy": round(
                            (
                                rotate_vec_ccw_90s(cluster_front_local, crot)
                                or (0.0, 0.0)
                            )[1],
                            3,
                        ),
                    },
                    "axis_world": None
                    if cluster_axis_local is None
                    else {
                        "dx": round(
                            (
                                rotate_vec_ccw_90s(cluster_axis_local, crot)
                                or (0.0, 0.0)
                            )[0],
                            3,
                        ),
                        "dy": round(
                            (
                                rotate_vec_ccw_90s(cluster_axis_local, crot)
                                or (0.0, 0.0)
                            )[1],
                            3,
                        ),
                    },
                    "anchor_override": cluster_meta.get("anchor_override"),
                    "anchors": [
                        str(anchor).strip()
                        for anchor in (card.get("anchors") or [])
                        if isinstance(anchor, str) and str(anchor).strip()
                    ],
                    "poly": cluster_poly,
                    "area_mm2": float(cluster_poly.area),
                    "local_placements": sorted(
                        [
                            deepcopy(local_placement_map[oid])
                            for oid in local_placement_map
                            if isinstance(oid, str)
                        ],
                        key=lambda item: str(item.get("id") or ""),
                    ),
                    "hard_constraints": deepcopy(card.get("hard_constraints") or []),
                    "soft_constraints": deepcopy(card.get("soft_constraints") or []),
                    "cluster_rules": deepcopy(card.get("cluster_rules") or {}),
                    "decisions": deepcopy(card.get("decisions") or []),
                }
            )

    protected_cluster_ids = _anchor_contract_cluster_ids(payload)
    cluster_by_id = {
        str(cluster.get("cluster_id") or ""): cluster for cluster in clusters if cluster
    }
    seed_tmap = _seed_transform_map(seed_layout)
    for cluster_id in sorted(protected_cluster_ids):
        cluster = cluster_by_id.get(cluster_id)
        if not isinstance(cluster, dict):
            continue
        seed_rot = int((seed_tmap.get(cluster_id) or {}).get("rot") or 0) % 360
        current_rot = int((tmap.get(cluster_id) or {}).get("rot") or 0) % 360
        if current_rot != seed_rot:
            hard_violations.append(
                {
                    "code": "ANCHOR_CONTRACT_CLUSTER_ROTATION",
                    "cluster_id": cluster_id,
                    "seed_rot": seed_rot,
                    "candidate_rot": current_rot,
                }
            )

        anchor_ids = _cluster_anchor_ids_from_state_cluster(cluster)
        for rep in repairs_by_cluster.get(cluster_id, []):
            if not isinstance(rep, dict):
                continue
            op = str(rep.get("op") or "").strip()
            object_id = str(rep.get("object_id") or "").strip()
            if op == "set_anchor":
                hard_violations.append(
                    {
                        "code": "ANCHOR_CONTRACT_ANCHOR_OVERRIDE",
                        "cluster_id": cluster_id,
                    }
                )
                continue
            if op in {"rotate_object", "mirror_object", "set_front_override"}:
                if object_id and object_id in anchor_ids:
                    hard_violations.append(
                        {
                            "code": "ANCHOR_CONTRACT_OBJECT_ORIENTATION",
                            "cluster_id": cluster_id,
                            "object_id": object_id,
                            "op": op,
                        }
                    )
                continue
            if op == "swap_objects":
                other_object_id = str(
                    (rep.get("params") or {}).get("other_object_id") or ""
                ).strip()
                if object_id in anchor_ids or other_object_id in anchor_ids:
                    hard_violations.append(
                        {
                            "code": "ANCHOR_CONTRACT_OBJECT_SWAP",
                            "cluster_id": cluster_id,
                            "object_id": object_id or None,
                            "other_object_id": other_object_id or None,
                        }
                    )

    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            a = objects[i]
            b = objects[j]
            inter = float(a["poly"].intersection(b["poly"]).area)
            if inter > 1e-6:
                object_overlap_pairs.append(
                    {
                        "a": {
                            "cluster_id": a["cluster_id"],
                            "object_id": a["object_id"],
                        },
                        "b": {
                            "cluster_id": b["cluster_id"],
                            "object_id": b["object_id"],
                        },
                        "intersection_area_mm2": round(inter, 2),
                    }
                )
                hard_violations.append(
                    {
                        "code": "OBJECT_OVERLAP",
                        "a_cluster_id": a["cluster_id"],
                        "a_object_id": a["object_id"],
                        "b_cluster_id": b["cluster_id"],
                        "b_object_id": b["object_id"],
                        "intersection_area_mm2": round(inter, 2),
                    }
                )

    return {
        "room_polygon": room_poly,
        "room_center": room_center,
        "obstacles": obstacles,
        "clusters": clusters,
        "objects": objects,
        "hard_violations": hard_violations,
        "object_overlap_pairs": object_overlap_pairs,
        "hard_valid": len(hard_violations) == 0,
    }


# -----------------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------------
def _project_distance_along_ray(
    origin: Tuple[float, float],
    direction: Tuple[float, float],
    point: Tuple[float, float],
) -> float | None:
    v = normalize_vec(direction)
    if v is None:
        return None
    ox, oy = float(origin[0]), float(origin[1])
    px, py = float(point[0]), float(point[1])
    t = ((px - ox) * v[0]) + ((py - oy) * v[1])
    if t <= 1e-6:
        return None
    return t


def _collect_intersection_points(geom: Any, out: List[Tuple[float, float]]) -> None:
    if geom is None:
        return
    try:
        if geom.is_empty:
            return
    except Exception:
        return
    gtype = getattr(geom, "geom_type", "")
    if gtype == "Point":
        out.append((float(geom.x), float(geom.y)))
        return
    if gtype == "MultiPoint":
        for sub in geom.geoms:
            _collect_intersection_points(sub, out)
        return
    if gtype in {"LineString", "LinearRing"}:
        coords = list(geom.coords)
        if coords:
            out.append((float(coords[0][0]), float(coords[0][1])))
            out.append((float(coords[-1][0]), float(coords[-1][1])))
        return
    if gtype == "MultiLineString":
        for sub in geom.geoms:
            _collect_intersection_points(sub, out)
        return
    if gtype == "GeometryCollection":
        for sub in geom.geoms:
            _collect_intersection_points(sub, out)
        return
    if gtype == "Polygon":
        coords = list(geom.exterior.coords)
        if coords:
            out.append((float(coords[0][0]), float(coords[0][1])))
            out.append((float(coords[-1][0]), float(coords[-1][1])))
        return
    if gtype == "MultiPolygon":
        for sub in geom.geoms:
            _collect_intersection_points(sub, out)


def _directional_clearance_mm(
    point: Tuple[float, float],
    direction: Tuple[float, float] | None,
    room_poly: Any,
    blockers: List[Any],
    max_distance: float,
) -> float:
    v = normalize_vec(direction)
    if v is None:
        return 0.0
    start = (float(point[0]) + v[0] * 1e-3, float(point[1]) + v[1] * 1e-3)
    end = (float(point[0]) + v[0] * max_distance, float(point[1]) + v[1] * max_distance)
    ray = LineString([start, end])
    best = max_distance

    pts: List[Tuple[float, float]] = []
    _collect_intersection_points(ray.intersection(room_poly.boundary), pts)
    for pt in pts:
        dist = _project_distance_along_ray(point, v, pt)
        if dist is not None and dist < best:
            best = dist

    for geom in blockers:
        try:
            inter = ray.intersection(geom.boundary)
        except Exception:
            try:
                inter = ray.intersection(geom)
            except Exception:
                continue
        pts2: List[Tuple[float, float]] = []
        _collect_intersection_points(inter, pts2)
        for pt in pts2:
            dist = _project_distance_along_ray(point, v, pt)
            if dist is not None and dist < best:
                best = dist

    return max(0.0, float(best))


def _candidate_open_dirs(
    point: Tuple[float, float], room_center: Tuple[float, float]
) -> List[Tuple[float, float]]:
    base = [
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (1.0, 1.0),
        (1.0, -1.0),
        (-1.0, 1.0),
        (-1.0, -1.0),
    ]
    to_center = _vec_from_to(point, room_center)
    if to_center is not None:
        base = [to_center, (-to_center[0], -to_center[1])] + base
    out: List[Tuple[float, float]] = []
    seen = set()
    for d in base:
        v = normalize_vec(d)
        if v is None:
            continue
        key = (round(v[0], 3), round(v[1], 3))
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def _distance_between_points(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _distance_to_room_center(
    center: Tuple[float, float], room_center: Tuple[float, float]
) -> int:
    return int(round(_distance_between_points(center, room_center)))


def _distance_to_nearest_opening(
    center: Tuple[float, float], items: List[Dict[str, Any]]
) -> int | None:
    midpoint = _nearest_midpoint(center, items)
    if midpoint is None:
        return None
    return int(round(_distance_between_points(center, midpoint)))


def _door_window_metrics(
    center: Tuple[float, float], openings: Dict[str, List[Dict[str, Any]]]
) -> tuple[int | None, int | None]:
    door_distance = _distance_to_nearest_opening(center, openings.get("doors") or [])
    window_distance = _distance_to_nearest_opening(
        center, openings.get("windows") or []
    )
    return door_distance, window_distance


def _distance_poly_to_nearest_opening(
    poly: Any, items: List[Dict[str, Any]]
) -> float | None:
    best: float | None = None
    for item in items:
        line = item.get("line")
        if line is None:
            continue
        try:
            distance = float(poly.distance(line))
        except Exception:
            continue
        if best is None or distance < best:
            best = distance
    return best


def _nearest_opening_line_direction(
    point: Tuple[float, float], items: List[Dict[str, Any]]
) -> tuple[float, float] | None:
    best_dir = None
    best_distance = None
    px, py = point
    for item in items:
        line = item.get("line")
        midpoint = item.get("midpoint")
        if line is None or not isinstance(midpoint, tuple):
            continue
        coords = list(getattr(line, "coords", []))
        if len(coords) < 2:
            continue
        direction = normalize_vec(
            (
                float(coords[-1][0]) - float(coords[0][0]),
                float(coords[-1][1]) - float(coords[0][1]),
            )
        )
        if direction is None:
            continue
        distance = math.hypot(float(midpoint[0]) - px, float(midpoint[1]) - py)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_dir = direction
    return best_dir


def _nearest_boundary_point(
    point: Tuple[float, float], room_poly: Any
) -> tuple[float, float] | None:
    try:
        boundary_point, _ = nearest_points(room_poly.boundary, Point(point))
    except Exception:
        return None
    return (float(boundary_point.x), float(boundary_point.y))


def _nearest_wall_inward_dir(
    point: Tuple[float, float], room_poly: Any
) -> tuple[float, float] | None:
    boundary_point = _nearest_boundary_point(point, room_poly)
    if boundary_point is None:
        return None
    return _vec_from_to(boundary_point, point)


def _back_to_wall_penalty(
    *,
    point: Tuple[float, float],
    front: Tuple[float, float] | None,
    room_poly: Any,
    blockers: List[Any],
    desired_back_clear_mm: float,
    desired_front_advantage_mm: float,
    weight_scale: float,
) -> tuple[float, float | None, dict[str, int]]:
    if front is None:
        return 0.0, None, {}
    max_distance = max(1800.0, room_poly.length)
    front_clear = _directional_clearance_mm(
        point, front, room_poly, blockers, max_distance
    )
    back = (-front[0], -front[1])
    back_clear = _directional_clearance_mm(
        point, back, room_poly, blockers, max_distance
    )
    front_advantage = front_clear - back_clear
    penalty = (
        max(0.0, back_clear - desired_back_clear_mm) * 0.55
        + max(0.0, desired_front_advantage_mm - front_advantage) * 0.85
        + max(0.0, back_clear - front_clear) * 0.35
    ) * weight_scale
    debug = {
        "front_clear_mm": int(round(front_clear)),
        "back_clear_mm": int(round(back_clear)),
        "front_advantage_mm": int(round(front_advantage)),
    }
    return penalty, front_advantage, debug


def _room_wall_segments(room_poly: Any) -> list[dict[str, Any]]:
    try:
        coords = list(room_poly.exterior.coords)
    except Exception:
        return []
    segments: list[dict[str, Any]] = []
    for idx in range(max(0, len(coords) - 1)):
        p1 = coords[idx]
        p2 = coords[idx + 1]
        line = LineString([p1, p2])
        segments.append(
            {
                "line": line,
                "length": float(line.length),
                "direction": normalize_vec(
                    (
                        float(p2[0]) - float(p1[0]),
                        float(p2[1]) - float(p1[1]),
                    )
                ),
            }
        )
    return segments


def _longest_wall_segment(room_poly: Any) -> dict[str, Any] | None:
    segments = _room_wall_segments(room_poly)
    if not segments:
        return None
    return max(segments, key=lambda item: float(item.get("length") or 0.0))


def _distance_to_longest_wall(
    point: Tuple[float, float], room_poly: Any
) -> float | None:
    longest = _longest_wall_segment(room_poly)
    if not longest:
        return None
    try:
        return float(longest["line"].distance(Point(point)))
    except Exception:
        return None


def _window_clearance_hits(state: Dict[str, Any], cluster_id: str | None = None) -> int:
    window_clearance_polys = [
        ob["poly"]
        for ob in state.get("obstacles") or []
        if str(ob.get("type") or "").lower() == "window_clearance"
    ]
    if not window_clearance_polys:
        return 0

    hits = 0
    for row in state.get("objects") or []:
        if cluster_id is not None and row.get("cluster_id") != cluster_id:
            continue
        poly = row.get("poly")
        if poly is None:
            continue
        for clearance in window_clearance_polys:
            if poly.intersects(clearance):
                hits += 1
                break
    return hits


def _classify_cluster_zone(
    center: Tuple[float, float],
    room_center: Tuple[float, float],
    openings: Dict[str, List[Dict[str, Any]]],
) -> str:
    door_distance, window_distance = _door_window_metrics(center, openings)
    center_distance = _distance_to_room_center(center, room_center)
    if window_distance is not None and window_distance + 200 < min(
        door_distance or 10**9, center_distance
    ):
        return "window_side"
    if door_distance is not None and door_distance + 200 < min(
        window_distance or 10**9, center_distance
    ):
        return "entry_side"
    return "interior"


def _cluster_anchor_ids_from_state_cluster(cluster: Dict[str, Any]) -> set[str]:
    anchors = {
        str(anchor).strip()
        for anchor in (cluster.get("anchors") or [])
        if isinstance(anchor, str) and str(anchor).strip()
    }
    if anchors:
        return anchors

    anchor_override = cluster.get("anchor_override")
    if isinstance(anchor_override, str) and anchor_override.strip():
        return {anchor_override.strip()}

    for row in cluster.get("decisions") or []:
        if not isinstance(row, dict):
            continue
        priority = str(row.get("priority") or "").strip().lower()
        if priority != "anchor":
            continue
        object_id = str(row.get("object_type") or row.get("category") or "").strip()
        if object_id:
            anchors.add(object_id)
    return anchors


def _cluster_intents_by_id(payload: Dict[str, Any]) -> Dict[str, set[str]]:
    relation_plan = (payload.get("goals") or {}).get("relation_plan_used") or {}
    out: Dict[str, set[str]] = {}

    for row in relation_plan.get("cluster_orientations") or []:
        if not isinstance(row, dict):
            continue
        cid = row.get("cluster_id")
        if not isinstance(cid, str):
            continue
        slot = out.setdefault(cid, set())
        for intent in row.get("intents") or []:
            text = str(intent).strip().lower()
            if text:
                slot.add(text)

    for row in relation_plan.get("object_orientations") or []:
        if not isinstance(row, dict):
            continue
        cid = row.get("cluster_id")
        if not isinstance(cid, str):
            continue
        slot = out.setdefault(cid, set())
        for intent in row.get("intents") or []:
            text = str(intent).strip().lower()
            if text:
                slot.add(text)

    return out


def _cluster_preference_profile(
    payload: Dict[str, Any],
) -> dict[str, ClusterPreferenceProfile]:
    relation_plan = (payload.get("goals") or {}).get("relation_plan_used") or {}
    out: dict[str, ClusterPreferenceProfile] = {}

    def _slot(cluster_id: str) -> ClusterPreferenceProfile:
        return out.setdefault(
            cluster_id,
            {
                "prefer": set(),
                "avoid": set(),
                "intents": set(),
                "viewing_partner": None,
                "viewing_role": None,
            },
        )

    for row in relation_plan.get("cluster_affinities") or []:
        if not isinstance(row, dict):
            continue
        cluster_id = row.get("cluster_id")
        if not isinstance(cluster_id, str):
            continue
        slot = _slot(cluster_id)
        for tag in row.get("prefer") or []:
            text = str(tag).strip().lower()
            if text:
                slot["prefer"].add(text)
        for tag in row.get("avoid") or []:
            text = str(tag).strip().lower()
            if text:
                slot["avoid"].add(text)

    for row in relation_plan.get("cluster_orientations") or []:
        if not isinstance(row, dict):
            continue
        cluster_id = row.get("cluster_id")
        if not isinstance(cluster_id, str):
            continue
        slot = _slot(cluster_id)
        for intent in row.get("intents") or []:
            text = str(intent).strip().lower()
            if text:
                slot["intents"].add(text)

    for row in relation_plan.get("object_orientations") or []:
        if not isinstance(row, dict):
            continue
        cluster_id = row.get("cluster_id")
        if not isinstance(cluster_id, str):
            continue
        slot = _slot(cluster_id)
        for intent in row.get("intents") or []:
            text = str(intent).strip().lower()
            if text:
                slot["intents"].add(text)

    layout_intent = relation_plan.get("layout_intent_profile") or {}
    if isinstance(layout_intent, dict):
        focus_mode = str(layout_intent.get("focus_mode") or "mixed").strip().lower()
        primary_cluster_id = layout_intent.get("primary_cluster_id")
        secondary_cluster_id = layout_intent.get("secondary_cluster_id")
        support_behavior = str(
            layout_intent.get("support_cluster_behavior") or "balanced"
        ).strip()
        center_open_preference = str(
            layout_intent.get("center_open_preference") or "medium"
        ).strip()
        distribution_mode = str(
            layout_intent.get("distribution_mode") or "balanced"
        ).strip()

        if isinstance(primary_cluster_id, str) and primary_cluster_id:
            for cluster_id in list(out.keys()):
                if cluster_id == primary_cluster_id:
                    continue
                slot = _slot(cluster_id)
                if support_behavior == "recede":
                    slot["prefer"].add("recess_or_edge")
                    slot["avoid"].add("center")
                if center_open_preference == "high":
                    slot["avoid"].add("center")
                if distribution_mode == "edge_weighted":
                    slot["prefer"].add("recess_or_edge")

        if (
            focus_mode in {"viewing", "mixed"}
            and isinstance(primary_cluster_id, str)
            and primary_cluster_id
            and isinstance(secondary_cluster_id, str)
            and secondary_cluster_id
            and secondary_cluster_id != primary_cluster_id
        ):
            primary_slot = _slot(primary_cluster_id)
            secondary_slot = _slot(secondary_cluster_id)
            primary_slot["viewing_partner"] = secondary_cluster_id
            primary_slot["viewing_role"] = "primary"
            secondary_slot["viewing_partner"] = primary_cluster_id
            secondary_slot["viewing_role"] = "secondary"
            primary_slot["prefer"].update({"wall", "far_from_entry"})
            secondary_slot["prefer"].update({"wall", "far_from_entry"})
            secondary_slot["avoid"].add("center")
            secondary_slot["prefer"].discard("window_side")

    for row in relation_plan.get("cluster_directional_relations") or []:
        if not isinstance(row, dict):
            continue
        relation = str(row.get("relation") or "").strip().lower()
        if relation not in {"face_each_other", "access_faces_other"}:
            continue
        a = row.get("a")
        b = row.get("b")
        if not isinstance(a, str) or not isinstance(b, str) or not a or not b:
            continue
        slot_a = _slot(a)
        slot_b = _slot(b)
        if not slot_a.get("viewing_partner"):
            slot_a["viewing_partner"] = b
        if not slot_b.get("viewing_partner"):
            slot_b["viewing_partner"] = a

    for row in relation_plan.get("anchor_pairs") or []:
        if not isinstance(row, dict):
            continue
        pair_type = str(row.get("pair_type") or "").strip().lower()
        if pair_type not in {
            "face_each_other",
            "supports_use_axis",
            "access_faces_other",
        }:
            continue
        a = str(row.get("cluster_a") or "").strip()
        b = str(row.get("cluster_b") or "").strip()
        if not a or not b or a == b:
            continue
        slot_a = _slot(a)
        slot_b = _slot(b)
        if not slot_a.get("viewing_partner"):
            slot_a["viewing_partner"] = b
        if not slot_b.get("viewing_partner"):
            slot_b["viewing_partner"] = a
        if not slot_a.get("viewing_role"):
            slot_a["viewing_role"] = "primary"
        if not slot_b.get("viewing_role"):
            slot_b["viewing_role"] = "secondary"
        if pair_type in {"supports_use_axis", "access_faces_other"}:
            slot_a["intents"].add("access_to_open_space")
            slot_b["intents"].add("access_to_open_space")

    return out


def _viewing_cluster_pairs(payload: Dict[str, Any]) -> list[tuple[str, str]]:
    relation_plan = (payload.get("goals") or {}).get("relation_plan_used") or {}
    pairs: list[tuple[str, str]] = []

    def _append_pair(a: str, b: str) -> None:
        if not a or not b or a == b:
            return
        pair = (a, b)
        reverse_pair = (b, a)
        if pair not in pairs and reverse_pair not in pairs:
            pairs.append(pair)

    layout_intent = relation_plan.get("layout_intent_profile") or {}
    if isinstance(layout_intent, dict):
        focus_mode = str(layout_intent.get("focus_mode") or "").strip().lower()
        primary_cluster_id = str(layout_intent.get("primary_cluster_id") or "").strip()
        secondary_cluster_id = str(
            layout_intent.get("secondary_cluster_id") or ""
        ).strip()
        if (
            focus_mode in {"viewing", "mixed"}
            and primary_cluster_id
            and secondary_cluster_id
            and primary_cluster_id != secondary_cluster_id
        ):
            _append_pair(primary_cluster_id, secondary_cluster_id)

    for row in relation_plan.get("cluster_directional_relations") or []:
        if not isinstance(row, dict):
            continue
        relation = str(row.get("relation") or "").strip().lower()
        if relation not in {"face_each_other", "access_faces_other"}:
            continue
        a = str(row.get("a") or "").strip()
        b = str(row.get("b") or "").strip()
        _append_pair(a, b)

    for row in relation_plan.get("anchor_pairs") or []:
        if not isinstance(row, dict):
            continue
        pair_type = str(row.get("pair_type") or "").strip().lower()
        if pair_type not in {
            "face_each_other",
            "supports_use_axis",
            "access_faces_other",
        }:
            continue
        a = str(row.get("cluster_a") or "").strip()
        b = str(row.get("cluster_b") or "").strip()
        _append_pair(a, b)

    return pairs


def _anchor_layout_hints_by_cluster(
    payload: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    relation_plan = (payload.get("goals") or {}).get("relation_plan_used") or {}
    hints = relation_plan.get("anchor_layout_hints_by_cluster") or {}
    if not isinstance(hints, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for cluster_id, row in hints.items():
        if isinstance(cluster_id, str) and isinstance(row, dict):
            out[cluster_id] = deepcopy(row)
    return out


def _cluster_anchor_state(
    *,
    cluster_id: str,
    state_objects: List[Dict[str, Any]],
    cluster_state_by_id: Dict[str, Dict[str, Any]],
    anchor_hints_by_cluster: Dict[str, Dict[str, Any]],
) -> Dict[str, Any] | None:
    hint = anchor_hints_by_cluster.get(cluster_id) or {}
    dominant_anchor_id = str(hint.get("dominant_anchor_object_id") or "").strip()
    if dominant_anchor_id:
        for row in state_objects:
            if (
                str(row.get("cluster_id") or "") == cluster_id
                and str(row.get("object_id") or "") == dominant_anchor_id
            ):
                return row
    cluster = cluster_state_by_id.get(cluster_id) or {}
    anchor_ids = _cluster_anchor_ids_from_state_cluster(cluster)
    for row in state_objects:
        if (
            str(row.get("cluster_id") or "") == cluster_id
            and str(row.get("object_id") or "") in anchor_ids
        ):
            return row
    return None


def _anchor_contract_cluster_ids(payload: Dict[str, Any]) -> set[str]:
    relation_plan = (payload.get("goals") or {}).get("relation_plan_used") or {}
    protected_cluster_ids: set[str] = set()

    for left, right in _viewing_cluster_pairs(payload):
        protected_cluster_ids.add(left)
        protected_cluster_ids.add(right)

    for row in relation_plan.get("cluster_orientations") or []:
        if not isinstance(row, dict):
            continue
        cluster_id = str(row.get("cluster_id") or "").strip()
        target_cluster_id = str(row.get("target_cluster_id") or "").strip()
        intents = {
            str(intent).strip().lower()
            for intent in (row.get("intents") or [])
            if str(intent).strip()
        }
        if "face_cluster" not in intents:
            continue
        if cluster_id:
            protected_cluster_ids.add(cluster_id)
        if target_cluster_id:
            protected_cluster_ids.add(target_cluster_id)

    return protected_cluster_ids


def _preferred_zone_for_cluster(intents: set[str]) -> str | None:
    lowered = {str(item).lower() for item in intents}
    if "face_window" in lowered:
        return "window_side"
    if "face_entry" in lowered:
        return "entry_side"
    if lowered & {"back_to_wall", "access_to_open_space"}:
        return "edge_side"
    return None


def _opening_bands(
    payload: Dict[str, Any],
    openings: Dict[str, List[Dict[str, Any]]],
    room_poly: Any,
) -> List[Dict[str, Any]]:
    grid_mm = normalize_layout_grid_mm(
        (payload.get("room_context") or {}).get("grid_mm")
    )
    out: List[Dict[str, Any]] = []

    for kind, depth in (
        ("doors", max(500, grid_mm * 8)),
        ("windows", max(300, grid_mm * 5)),
    ):
        opening_type = "door" if kind == "doors" else "window"
        for item in openings.get(kind) or []:
            line = item.get("line")
            if line is None:
                continue
            band = _fix_geom(room_poly.intersection(line.buffer(depth, cap_style=2)))
            try:
                if band.is_empty:
                    continue
            except Exception:
                continue
            out.append(
                {
                    "id": str(item.get("id") or ""),
                    "type": opening_type,
                    "depth_mm": int(depth),
                    "poly": band,
                    "area_mm2": float(band.area),
                }
            )
    return out


def _main_path_corridors(
    payload: Dict[str, Any],
    openings: Dict[str, List[Dict[str, Any]]],
    room_center: Tuple[float, float],
    room_poly: Any,
    state: Dict[str, Any],
) -> List[Dict[str, Any]]:
    grid_mm = normalize_layout_grid_mm(
        (payload.get("room_context") or {}).get("grid_mm")
    )
    corridors: List[Dict[str, Any]] = []
    relation_plan = (payload.get("goals") or {}).get("relation_plan_used") or {}
    cluster_by_id = {
        str(row.get("cluster_id") or ""): row for row in state.get("clusters") or []
    }

    def _opening_by_id(opening_id: str) -> dict[str, Any] | None:
        for key in ("doors", "windows"):
            for item in openings.get(key) or []:
                if str(item.get("id") or "") == opening_id:
                    return item
        return None

    def _cluster_access_point(cluster: Dict[str, Any]) -> tuple[float, float]:
        center = (
            float((cluster.get("world_center") or {}).get("x") or room_center[0]),
            float((cluster.get("world_center") or {}).get("y") or room_center[1]),
        )
        bbox = cluster.get("bbox") or {}
        span_x = float(bbox.get("max_x", 0.0)) - float(bbox.get("min_x", 0.0))
        span_y = float(bbox.get("max_y", 0.0)) - float(bbox.get("min_y", 0.0))
        front = parse_vec2(cluster.get("front_world"))
        if front is None:
            return center
        offset_mm = max(450.0, (max(span_x, span_y) / 2.0) + 150.0)
        return (
            center[0] + front[0] * offset_mm,
            center[1] + front[1] * offset_mm,
        )

    def _dedupe_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for point in points:
            if not out or point != out[-1]:
                out.append(point)
        return out

    def _best_corridor_line(
        start: tuple[float, float], end: tuple[float, float]
    ) -> Any:
        sx, sy = float(start[0]), float(start[1])
        ex, ey = float(end[0]), float(end[1])
        cx, cy = float(room_center[0]), float(room_center[1])

        candidates: list[list[tuple[float, float]]] = [
            [(sx, sy), (ex, ey)],
            [(sx, sy), (sx, ey), (ex, ey)],
            [(sx, sy), (ex, sy), (ex, ey)],
            [(sx, sy), (cx, sy), (cx, ey), (ex, ey)],
            [(sx, sy), (sx, cy), (ex, cy), (ex, ey)],
        ]

        best_line = LineString([start, end])
        best_key: tuple[float, int, float] | None = None
        room_cover = room_poly.buffer(1e-6)

        for raw_points in candidates:
            points = _dedupe_points(raw_points)
            if len(points) < 2:
                continue
            line = LineString(points)
            try:
                outside_length = float(line.difference(room_cover).length)
            except Exception:
                outside_length = float(line.length)
            bend_count = max(0, len(points) - 2)
            rank_key = (outside_length, bend_count, float(line.length))
            if best_key is None or rank_key < best_key:
                best_key = rank_key
                best_line = line
        return best_line

    for row in relation_plan.get("circulation_plan", {}).get("main_paths") or []:
        if not isinstance(row, dict):
            continue
        opening_id = str(row.get("from") or "").strip()
        target_cluster_id = str(row.get("to_cluster") or "").strip()
        opening = _opening_by_id(opening_id)
        cluster = cluster_by_id.get(target_cluster_id)
        midpoint = opening.get("midpoint") if isinstance(opening, dict) else None
        if not isinstance(midpoint, tuple):
            continue
        target_point = (
            _cluster_access_point(cluster) if isinstance(cluster, dict) else room_center
        )
        priority = str(row.get("priority") or "medium").strip().lower()
        corridor_width = {
            "high": max(850, grid_mm * 12),
            "medium": max(700, grid_mm * 10),
            "low": max(600, grid_mm * 8),
        }.get(priority, max(700, grid_mm * 10))
        line = _best_corridor_line(midpoint, target_point)
        corridor = _fix_geom(
            room_poly.intersection(line.buffer(corridor_width / 2.0, cap_style=2))
        )
        try:
            if corridor.is_empty:
                continue
        except Exception:
            continue
        corridors.append(
            {
                "id": f"{opening_id}->{target_cluster_id or 'room_center'}",
                "opening_id": opening_id,
                "target_cluster_id": target_cluster_id or None,
                "priority": priority,
                "desired_clearance_mm": int(corridor_width),
                "width_mm": int(corridor_width),
                "line": line,
                "poly": corridor,
                "area_mm2": float(corridor.area),
            }
        )

    if corridors:
        return corridors

    corridor_width = max(700, grid_mm * 10)
    for item in openings.get("doors") or []:
        midpoint = item.get("midpoint")
        if not isinstance(midpoint, tuple):
            continue
        line = _best_corridor_line(midpoint, room_center)
        corridor = _fix_geom(
            room_poly.intersection(line.buffer(corridor_width / 2.0, cap_style=2))
        )
        try:
            if corridor.is_empty:
                continue
        except Exception:
            continue
        corridors.append(
            {
                "id": str(item.get("id") or ""),
                "opening_id": str(item.get("id") or ""),
                "target_cluster_id": None,
                "priority": "medium",
                "desired_clearance_mm": int(corridor_width),
                "width_mm": int(corridor_width),
                "line": line,
                "poly": corridor,
                "area_mm2": float(corridor.area),
            }
        )
    return corridors


def _free_space_polygon(state: Dict[str, Any]) -> Any:
    room_poly = state["room_polygon"]
    blockers = [row["poly"] for row in state.get("objects") or []]
    blockers.extend(ob["poly"] for ob in state.get("obstacles") or [])
    if not blockers:
        return room_poly
    try:
        blocked = unary_union(blockers)
        return _fix_geom(room_poly.difference(blocked))
    except Exception:
        return room_poly


def _corridor_clearance_summary(
    state: Dict[str, Any], corridor: Dict[str, Any]
) -> Dict[str, Any]:
    free_space = _free_space_polygon(state)
    line = corridor.get("line")
    if line is None:
        return {
            "min_clearance_mm": 0,
            "blocked_samples": 0,
            "sample_count": 0,
            "clearance_shortage_mm": int(corridor.get("desired_clearance_mm") or 0),
        }

    try:
        length = float(line.length)
    except Exception:
        length = 0.0
    sample_count = max(5, min(13, int(length / 450.0) + 1))
    blocked_samples = 0
    min_clearance = float(corridor.get("desired_clearance_mm") or 0.0)
    free_space_cover = free_space.buffer(1e-6)

    for idx in range(sample_count):
        fraction = (idx + 1) / float(sample_count + 1)
        sample_point = line.interpolate(fraction, normalized=True)
        if not bool(free_space_cover.covers(sample_point)):
            clearance = 0.0
            blocked_samples += 1
        else:
            clearance = float(free_space.boundary.distance(sample_point))
        min_clearance = min(min_clearance, clearance)

    desired_clearance = float(corridor.get("desired_clearance_mm") or 0.0) / 2.0
    shortage = max(0.0, desired_clearance - min_clearance)
    return {
        "min_clearance_mm": int(round(min_clearance)),
        "blocked_samples": blocked_samples,
        "sample_count": sample_count,
        "clearance_shortage_mm": int(round(shortage)),
    }


def _central_zone(room_poly: Any) -> Any:
    min_x, min_y, max_x, max_y = room_poly.bounds
    max_span = max(max_x - min_x, max_y - min_y)
    radius = max(800.0, max_span * 0.18)
    center = (float(room_poly.centroid.x), float(room_poly.centroid.y))
    return _fix_geom(room_poly.intersection(Point(center).buffer(radius)))


def _cluster_poly(cluster: Dict[str, Any]) -> Any:
    poly = cluster.get("poly")
    if poly is not None:
        return poly
    return _bbox_poly(cluster.get("bbox") or {})


def _normalize_semantic_proximity(value: Any) -> str:
    if not isinstance(value, str):
        return "balanced"
    token = value.strip().lower()
    if token in {"compact", "balanced", "loose"}:
        return token
    return "balanced"


def _target_gap_for_proximity(gap_min: int, gap_max: int, proximity: str) -> int:
    if proximity == "compact":
        return gap_min
    if proximity == "loose":
        return gap_max
    return int(round((gap_min + gap_max) / 2.0))


def _qualifier_world_for_anchor_option(
    side_token: str, cluster_rot: int
) -> tuple[str, str | None]:
    base_side, qualifier_local = resolve_anchor_side(side_token)
    if base_side == "head":
        base_side = "top"
    elif base_side == "foot":
        base_side = "bottom"
    mapped_side = rotate_side(base_side, cluster_rot)
    qualifier_world = None
    if qualifier_local in {"left", "right"}:
        qualifier_world = rotate_side(qualifier_local, cluster_rot)
    return mapped_side, qualifier_world


def _qualifier_matches_centers(
    *,
    qualifier_world: str | None,
    subject_rect: tuple[int, int, int, int],
    base_rect: tuple[int, int, int, int],
) -> bool:
    if qualifier_world is None:
        return True
    subject_center = (
        (subject_rect[0] + subject_rect[2]) / 2.0,
        (subject_rect[1] + subject_rect[3]) / 2.0,
    )
    base_center = (
        (base_rect[0] + base_rect[2]) / 2.0,
        (base_rect[1] + base_rect[3]) / 2.0,
    )
    if qualifier_world == "left":
        return subject_center[0] <= base_center[0]
    if qualifier_world == "right":
        return subject_center[0] >= base_center[0]
    if qualifier_world == "top":
        return subject_center[1] >= base_center[1]
    if qualifier_world == "bottom":
        return subject_center[1] <= base_center[1]
    return True


def _best_anchor_side_option(
    *,
    side_options: list[str],
    cluster_rot: int,
    subject_rect: tuple[int, int, int, int],
    base_rect: tuple[int, int, int, int],
) -> tuple[str, str, str | None] | None:
    best: tuple[float, str, str, str | None] | None = None
    for side_token in side_options:
        mapped_side, qualifier_world = _qualifier_world_for_anchor_option(
            side_token, cluster_rot
        )
        gap_mm = edge_gap(subject_rect, base_rect, mapped_side)
        qualifier_penalty = 0.0
        if not _qualifier_matches_centers(
            qualifier_world=qualifier_world,
            subject_rect=subject_rect,
            base_rect=base_rect,
        ):
            qualifier_penalty = 10_000.0
        score = qualifier_penalty + abs(float(gap_mm))
        candidate = (score, side_token, mapped_side, qualifier_world)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        return None
    return best[1], best[2], best[3]


def _measure_semantic_proximity(
    *,
    record: Dict[str, Any],
    placements: Dict[str, Dict[str, Any]],
    rects: Dict[str, tuple[int, int, int, int]],
    facing_map: Dict[str, Any],
    specs: Dict[str, Dict[str, Any]],
) -> Dict[str, Any] | None:
    subject_id = str(record.get("id") or "").strip()
    base_id = str(record.get("relative_to") or "").strip()
    kind = str(record.get("kind") or "").strip().lower()
    if not subject_id or not base_id or subject_id not in rects or base_id not in rects:
        return None

    gap_min = int(record.get("gap_min") or 0)
    gap_max = int(record.get("gap_max") or gap_min)
    if gap_max < gap_min:
        gap_min, gap_max = gap_max, gap_min
    proximity = _normalize_semantic_proximity(record.get("proximity"))
    target_gap = _target_gap_for_proximity(gap_min, gap_max, proximity)

    subject_rect = rects[subject_id]
    base_rect = rects[base_id]
    cluster_rot = int((placements.get(base_id) or {}).get("rot") or 0) % 360

    mapped_side: str | None = None
    relation_token: str | None = None
    qualifier_world: str | None = None
    overlap_mm: int | None = None

    if kind == "dock_to_edge":
        base_front = get_front_base(base_id, facing_map, specs)
        base_side = resolve_edge_token_to_base_side(
            str(record.get("b_edge") or "front"), base_front
        )
        mapped_side = rotate_side(base_side, cluster_rot)
        relation_token = str(record.get("b_edge") or "")
        overlap_mm = perpendicular_overlap_len(subject_rect, base_rect, mapped_side)
    elif kind == "anchor_side":
        side_options = record.get("side_options")
        if not isinstance(side_options, list) or not side_options:
            return None
        chosen = _best_anchor_side_option(
            side_options=[
                str(item).strip().lower() for item in side_options if str(item).strip()
            ],
            cluster_rot=cluster_rot,
            subject_rect=subject_rect,
            base_rect=base_rect,
        )
        if chosen is None:
            return None
        relation_token, mapped_side, qualifier_world = chosen
    else:
        return None

    if mapped_side is None:
        return None

    gap_mm = edge_gap(subject_rect, base_rect, mapped_side)
    penalty_mm = abs(int(gap_mm) - int(target_gap))
    if penalty_mm <= 0:
        return None

    return {
        "subject_id": subject_id,
        "base_id": base_id,
        "kind": kind,
        "proximity": proximity,
        "mapped_side": mapped_side,
        "relation_token": relation_token,
        "qualifier_world": qualifier_world,
        "gap_mm": int(gap_mm),
        "gap_min_mm": int(gap_min),
        "gap_max_mm": int(gap_max),
        "target_gap_mm": int(target_gap),
        "penalty_mm": int(penalty_mm),
        "overlap_mm": None if overlap_mm is None else int(overlap_mm),
    }


def _score_semantic_proximity_preferences(
    cluster: Dict[str, Any],
) -> tuple[float, Dict[tuple[str, str], float], List[Dict[str, Any]]]:
    cluster_rules = cluster.get("cluster_rules") or {}
    if not isinstance(cluster_rules, dict):
        return 0.0, {}, []
    semantic_placements = cluster_rules.get("semantic_placements")
    if not isinstance(semantic_placements, list) or not semantic_placements:
        return 0.0, {}, []

    local_placements = deepcopy(cluster.get("local_placements") or [])
    if not local_placements:
        return 0.0, {}, []

    specs = normalize_objects(build_object_specs_from_cluster(cluster))
    placements = normalize_placements(local_placements)
    rects = build_rects(placements, specs, use_clearance=False)
    facing_map = (
        cluster_rules.get("facing")
        if isinstance(cluster_rules.get("facing"), dict)
        else {}
    )

    cluster_penalty = 0.0
    object_penalty: Dict[tuple[str, str], float] = {}
    debug_rows: List[Dict[str, Any]] = []
    cluster_id = str(cluster.get("cluster_id") or "")

    for raw_record in semantic_placements:
        if not isinstance(raw_record, dict):
            continue
        measurement = _measure_semantic_proximity(
            record=raw_record,
            placements=placements,
            rects=rects,
            facing_map=facing_map,
            specs=specs,
        )
        if measurement is None:
            continue
        penalty_mm = float(measurement["penalty_mm"])
        cluster_penalty += penalty_mm
        key = (cluster_id, str(measurement["subject_id"]))
        object_penalty[key] = object_penalty.get(key, 0.0) + penalty_mm
        debug_rows.append(
            {
                "kind": "cluster_internal_constraint",
                "cluster_id": cluster_id,
                "constraint_type": "semantic_proximity",
                "penalty_mm": int(round(penalty_mm)),
                "hard": False,
                "subjects": {
                    "a": measurement["subject_id"],
                    "b": measurement["base_id"],
                },
                "proximity": measurement["proximity"],
                "mapped_side": measurement["mapped_side"],
                "relation_token": measurement["relation_token"],
                "qualifier_world": measurement["qualifier_world"],
                "gap_mm": measurement["gap_mm"],
                "gap_min_mm": measurement["gap_min_mm"],
                "gap_max_mm": measurement["gap_max_mm"],
                "target_gap_mm": measurement["target_gap_mm"],
                "overlap_mm": measurement["overlap_mm"],
            }
        )

    return cluster_penalty, object_penalty, debug_rows


def _score_cluster_internal_fidelity(
    state: Dict[str, Any],
) -> tuple[Dict[str, float], Dict[Tuple[str, str], float], List[Dict[str, Any]]]:
    cluster_penalty: Dict[str, float] = {}
    object_penalty: Dict[Tuple[str, str], float] = {}
    debug_rows: List[Dict[str, Any]] = []

    for cluster in state.get("clusters") or []:
        cid = str(cluster.get("cluster_id") or "")
        local_placements = deepcopy(cluster.get("local_placements") or [])
        if not cid or not local_placements:
            continue

        cluster_payload = {
            "cluster_id": cid,
            "hard_constraints": deepcopy(cluster.get("hard_constraints") or []),
            "soft_constraints": deepcopy(cluster.get("soft_constraints") or []),
            "cluster_rules": deepcopy(cluster.get("cluster_rules") or {}),
            "decisions": deepcopy(cluster.get("decisions") or []),
        }
        cluster_rules = cluster_payload.get("cluster_rules") or {}
        grid_mm = normalize_layout_grid_mm(cluster_rules.get("grid_mm"))
        scoring = score_cluster_constraints(
            hard_constraints=cluster_payload["hard_constraints"],
            soft_constraints=cluster_payload["soft_constraints"],
            objects=build_object_specs_from_cluster(cluster_payload),
            local_placements=local_placements,
            grid_mm=grid_mm,
            cluster_rules=cluster_rules,
            use_clearance=True,
        )

        hard_summary = scoring.get("hard") or {}
        soft_summary = scoring.get("soft") or {}
        cluster_total = (
            float(hard_summary.get("total_weighted_violation") or 0.0)
            + float(soft_summary.get("total_weighted_penalty") or 0.0) * 0.6
        )
        if cluster_total > 0:
            cluster_penalty[cid] = cluster_penalty.get(cid, 0.0) + cluster_total

        for row in (scoring.get("all_evaluations") or [])[:20]:
            if not isinstance(row, dict):
                continue
            weighted = float(row.get("weighted_violation") or 0.0)
            if weighted <= 0:
                continue
            debug_rows.append(
                {
                    "kind": "cluster_internal_constraint",
                    "cluster_id": cid,
                    "constraint_type": str(row.get("type") or ""),
                    "penalty_mm": int(round(weighted)),
                    "hard": bool(row.get("hard")),
                    "subjects": deepcopy(row.get("subjects") or {}),
                }
            )
            subjects = row.get("subjects") or {}
            member_ids: List[str] = []
            for key in ("a", "b", "id"):
                value = subjects.get(key)
                if isinstance(value, str) and value:
                    member_ids.append(value)
            if not member_ids:
                continue
            share = weighted / float(len(member_ids))
            for oid in member_ids:
                object_penalty[(cid, oid)] = object_penalty.get((cid, oid), 0.0) + share

        semantic_cluster_penalty, semantic_object_penalty, semantic_debug_rows = (
            _score_semantic_proximity_preferences(cluster)
        )
        if semantic_cluster_penalty > 0:
            cluster_penalty[cid] = (
                cluster_penalty.get(cid, 0.0) + semantic_cluster_penalty
            )
        for key, value in semantic_object_penalty.items():
            object_penalty[key] = object_penalty.get(key, 0.0) + value
        debug_rows.extend(semantic_debug_rows)

    debug_rows.sort(
        key=lambda row: (
            -int(row.get("penalty_mm") or 0),
            str(row.get("cluster_id") or ""),
            str(row.get("constraint_type") or ""),
        )
    )
    return cluster_penalty, object_penalty, debug_rows


def _score_global_layout_metrics(
    payload: Dict[str, Any],
    state: Dict[str, Any],
    openings: Dict[str, List[Dict[str, Any]]],
    room_center: Tuple[float, float],
) -> Dict[str, Any]:
    room_poly = state["room_polygon"]
    room_area = max(float(room_poly.area), 1.0)
    room_bounds = room_poly.bounds
    max_span = max(room_bounds[2] - room_bounds[0], room_bounds[3] - room_bounds[1])
    cluster_preferences = _cluster_preference_profile(payload)
    opening_bands = _opening_bands(payload, openings, room_poly)
    path_corridors = _main_path_corridors(
        payload, openings, room_center, room_poly, state
    )
    central_zone = _central_zone(room_poly)
    central_zone_area = max(float(central_zone.area), 1.0)

    cluster_penalty: Dict[str, float] = {}
    preferred_zone_rows: List[Dict[str, Any]] = []
    opening_block_rows: List[Dict[str, Any]] = []
    path_rows: List[Dict[str, Any]] = []
    path_object_rows: List[Dict[str, Any]] = []
    central_rows: List[Dict[str, Any]] = []
    edge_fit_rows: List[Dict[str, Any]] = []

    for cluster in state.get("clusters") or []:
        cid = str(cluster.get("cluster_id") or "")
        center = (
            float((cluster.get("world_center") or {}).get("x") or 0.0),
            float((cluster.get("world_center") or {}).get("y") or 0.0),
        )
        poly = _cluster_poly(cluster)
        area_mm2 = max(float(getattr(poly, "area", 0.0) or 0.0), 1.0)
        area_ratio = area_mm2 / room_area
        edge_distance = float(room_poly.boundary.distance(poly))
        actual_zone = _classify_cluster_zone(center, room_center, openings)
        profile = cluster_preferences.get(
            cid, {"prefer": set(), "avoid": set(), "intents": set()}
        )
        prefer_tags = set(profile.get("prefer") or set())
        avoid_tags = set(profile.get("avoid") or set())
        intents = set(profile.get("intents") or set())
        edge_pref = bool(
            prefer_tags & {"recess_or_edge", "long_wall"}
            or intents & {"back_to_wall", "access_to_open_space"}
        )
        window_pref = bool(
            "window_side" in prefer_tags
            or intents & {"face_window", "front_to_window", "axis_parallel_window"}
        )
        entry_pref = bool("entry_side" in prefer_tags or intents & {"face_entry"})

        affinity_rows: list[tuple[str, float, str]] = []
        if window_pref:
            distance = float(
                _distance_poly_to_nearest_opening(poly, openings.get("windows") or [])
                or max_span
            )
            affinity_rows.append(("window_side", distance, "window_side"))
        if entry_pref:
            distance = float(
                _distance_poly_to_nearest_opening(poly, openings.get("doors") or [])
                or max_span
            )
            affinity_rows.append(("entry_side", distance, "entry_side"))
        if "far_from_entry" in prefer_tags:
            distance = float(
                _distance_poly_to_nearest_opening(poly, openings.get("doors") or [])
                or 0.0
            )
            desired_clear = max(1500.0, max_span * 0.26)
            affinity_rows.append(
                (
                    "far_from_entry",
                    max(0.0, desired_clear - distance),
                    "far_from_entry",
                )
            )
        if "recess_or_edge" in prefer_tags:
            affinity_rows.append(("recess_or_edge", edge_distance, "edge_side"))
        if "long_wall" in prefer_tags:
            affinity_rows.append(
                (
                    "long_wall",
                    float(
                        _distance_to_longest_wall(center, room_poly) or edge_distance
                    ),
                    "edge_side",
                )
            )
        if not affinity_rows:
            fallback_zone = _preferred_zone_for_cluster(intents)
            if fallback_zone == "window_side":
                affinity_rows.append(
                    (
                        fallback_zone,
                        float(
                            _distance_poly_to_nearest_opening(
                                poly, openings.get("windows") or []
                            )
                            or max_span
                        ),
                        actual_zone,
                    )
                )
            elif fallback_zone == "entry_side":
                affinity_rows.append(
                    (
                        fallback_zone,
                        float(
                            _distance_poly_to_nearest_opening(
                                poly, openings.get("doors") or []
                            )
                            or max_span
                        ),
                        actual_zone,
                    )
                )
            elif fallback_zone == "edge_side":
                affinity_rows.append((fallback_zone, edge_distance, actual_zone))

        for preferred_zone, affinity_distance, actual_zone_label in affinity_rows:
            if preferred_zone == "window_side":
                zone_penalty = affinity_distance * 0.38
            elif preferred_zone == "entry_side":
                zone_penalty = affinity_distance * 0.32
            elif preferred_zone == "far_from_entry":
                zone_penalty = affinity_distance * 0.72
            elif preferred_zone == "long_wall":
                zone_penalty = affinity_distance * 0.44
            else:
                zone_penalty = affinity_distance * 0.42
            if zone_penalty <= 0:
                continue
            cluster_penalty[cid] = cluster_penalty.get(cid, 0.0) + zone_penalty
            preferred_zone_rows.append(
                {
                    "cluster_id": cid,
                    "preferred_zone": preferred_zone,
                    "actual_zone": actual_zone_label,
                    "distance_mm": int(round(affinity_distance)),
                    "penalty_mm": int(round(zone_penalty)),
                }
            )

        opening_penalty = 0.0
        for band in opening_bands:
            band_poly = band.get("poly")
            if band_poly is None:
                continue
            intersection_area = float(poly.intersection(band_poly).area)
            if intersection_area <= 1e-6:
                continue
            area_ratio_in_band = intersection_area / max(
                float(band.get("area_mm2") or 1.0), 1.0
            )
            weight = 1000.0 if band.get("type") == "door" else 640.0
            if band.get("type") == "window" and (
                "window_blocking" in avoid_tags or window_pref
            ):
                weight *= 1.7
            if band.get("type") == "door" and (
                avoid_tags & {"door_swing", "entry_blocking"}
                or "far_from_entry" in prefer_tags
            ):
                weight *= 1.8
            penalty = area_ratio_in_band * weight
            opening_penalty += penalty
            opening_block_rows.append(
                {
                    "cluster_id": cid,
                    "opening_id": band.get("id"),
                    "opening_type": band.get("type"),
                    "intersection_area_mm2": int(round(intersection_area)),
                    "penalty_mm": int(round(penalty)),
                }
            )
        if opening_penalty > 0:
            cluster_penalty[cid] = cluster_penalty.get(cid, 0.0) + opening_penalty

        central_area = float(poly.intersection(central_zone).area)
        if central_area > 1e-6:
            central_ratio = central_area / central_zone_area
            weight = 950.0
            if edge_pref or window_pref or "far_from_entry" in prefer_tags:
                weight = 1450.0
            penalty = central_ratio * weight
            cluster_penalty[cid] = cluster_penalty.get(cid, 0.0) + penalty
            central_rows.append(
                {
                    "cluster_id": cid,
                    "intersection_area_mm2": int(round(central_area)),
                    "central_ratio": round(central_ratio, 3),
                    "penalty_mm": int(round(penalty)),
                }
            )

        edge_weight = (
            2.8 if edge_pref or window_pref or "far_from_entry" in prefer_tags else 1.2
        )
        edge_fit_penalty = edge_distance * max(area_ratio, 0.08) * edge_weight
        if edge_fit_penalty > 0:
            cluster_penalty[cid] = cluster_penalty.get(cid, 0.0) + edge_fit_penalty
            edge_fit_rows.append(
                {
                    "cluster_id": cid,
                    "area_ratio": round(area_ratio, 4),
                    "edge_distance_mm": int(round(edge_distance)),
                    "penalty_mm": int(round(edge_fit_penalty)),
                }
            )

    for corridor in path_corridors:
        corridor_poly = corridor.get("poly")
        if corridor_poly is None:
            continue
        corridor_area = max(float(corridor.get("area_mm2") or 1.0), 1.0)
        corridor_summary = _corridor_clearance_summary(state, corridor)
        blocked_sample_ratio = float(
            corridor_summary.get("blocked_samples") or 0.0
        ) / max(float(corridor_summary.get("sample_count") or 1.0), 1.0)
        desired_half_clearance = max(
            float(corridor.get("desired_clearance_mm") or 0.0) / 2.0, 1.0
        )
        shortage_ratio = min(
            1.6,
            float(corridor_summary.get("clearance_shortage_mm") or 0.0)
            / desired_half_clearance,
        )
        priority = str(corridor.get("priority") or "medium").strip().lower()
        priority_weight = {"high": 1.3, "medium": 1.0, "low": 0.82}.get(priority, 1.0)

        cluster_blockers: List[Dict[str, Any]] = []
        total_cluster_blocked_area = 0.0
        for cluster in state.get("clusters") or []:
            blocked_area = float(
                _cluster_poly(cluster).intersection(corridor_poly).area
            )
            if blocked_area <= 1e-6:
                continue
            blocked_ratio = blocked_area / corridor_area
            total_cluster_blocked_area += blocked_area
            cluster_blockers.append(
                {
                    "cluster_id": str(cluster.get("cluster_id") or ""),
                    "blocked_area_mm2": blocked_area,
                    "blocked_ratio": blocked_ratio,
                }
            )

        object_blockers: List[Dict[str, Any]] = []
        total_object_blocked_area = 0.0
        for obj in state.get("objects") or []:
            blocked_area = float(obj["poly"].intersection(corridor_poly).area)
            if blocked_area <= 1e-6:
                continue
            blocked_ratio = blocked_area / corridor_area
            total_object_blocked_area += blocked_area
            object_blockers.append(
                {
                    "cluster_id": str(obj.get("cluster_id") or ""),
                    "object_id": str(obj.get("object_id") or ""),
                    "blocked_area_mm2": blocked_area,
                    "blocked_ratio": blocked_ratio,
                }
            )

        total_blocked_ratio = min(
            1.0,
            total_cluster_blocked_area / corridor_area if corridor_area > 0 else 0.0,
        )
        path_penalty_total = priority_weight * (
            (total_blocked_ratio * 1180.0)
            + (blocked_sample_ratio * 1560.0)
            + (shortage_ratio * 1420.0)
        )

        if (
            total_cluster_blocked_area <= 1e-6
            and total_object_blocked_area <= 1e-6
            and float(corridor_summary.get("clearance_shortage_mm") or 0.0) <= 0.0
            and float(corridor_summary.get("blocked_samples") or 0.0) <= 0.0
        ):
            continue

        for blocker in cluster_blockers:
            cluster_id = blocker.get("cluster_id")
            if not isinstance(cluster_id, str) or not cluster_id:
                continue
            if total_cluster_blocked_area > 0:
                share = path_penalty_total * (
                    float(blocker.get("blocked_area_mm2") or 0.0)
                    / total_cluster_blocked_area
                )
            else:
                share = path_penalty_total / max(float(len(cluster_blockers)), 1.0)
            cluster_penalty[cluster_id] = cluster_penalty.get(cluster_id, 0.0) + share

        if not cluster_blockers and path_penalty_total > 0.0:
            target_cluster_id = corridor.get("target_cluster_id")
            if isinstance(target_cluster_id, str) and target_cluster_id:
                cluster_penalty[target_cluster_id] = (
                    cluster_penalty.get(target_cluster_id, 0.0) + path_penalty_total
                )

        for blocker in object_blockers:
            if total_object_blocked_area > 0:
                share = path_penalty_total * (
                    float(blocker.get("blocked_area_mm2") or 0.0)
                    / total_object_blocked_area
                )
            else:
                share = path_penalty_total / max(float(len(object_blockers)), 1.0)
            path_object_rows.append(
                {
                    "path_id": corridor.get("id"),
                    "opening_id": corridor.get("opening_id"),
                    "target_cluster_id": corridor.get("target_cluster_id"),
                    "cluster_id": blocker.get("cluster_id"),
                    "object_id": blocker.get("object_id"),
                    "blocked_area_mm2": int(
                        round(float(blocker.get("blocked_area_mm2") or 0.0))
                    ),
                    "blocked_ratio": round(
                        float(blocker.get("blocked_ratio") or 0.0), 3
                    ),
                    "min_clearance_mm": int(
                        corridor_summary.get("min_clearance_mm") or 0
                    ),
                    "clearance_shortage_mm": int(
                        corridor_summary.get("clearance_shortage_mm") or 0
                    ),
                    "penalty_mm": int(round(share)),
                }
            )

        path_rows.append(
            {
                "path_id": corridor.get("id"),
                "opening_id": corridor.get("opening_id"),
                "target_cluster_id": corridor.get("target_cluster_id"),
                "priority": priority,
                "desired_clearance_mm": int(corridor.get("desired_clearance_mm") or 0),
                "min_clearance_mm": int(corridor_summary.get("min_clearance_mm") or 0),
                "clearance_shortage_mm": int(
                    corridor_summary.get("clearance_shortage_mm") or 0
                ),
                "blocked_samples": int(corridor_summary.get("blocked_samples") or 0),
                "sample_count": int(corridor_summary.get("sample_count") or 0),
                "blocked_ratio": round(total_blocked_ratio, 3),
                "blocking_clusters": [
                    blocker.get("cluster_id")
                    for blocker in sorted(
                        cluster_blockers,
                        key=lambda row: -float(row.get("blocked_area_mm2") or 0.0),
                    )[:4]
                    if isinstance(blocker.get("cluster_id"), str)
                ],
                "blocking_objects": [
                    f"{blocker.get('cluster_id')}.{blocker.get('object_id')}"
                    for blocker in sorted(
                        object_blockers,
                        key=lambda row: -float(row.get("blocked_area_mm2") or 0.0),
                    )[:4]
                    if isinstance(blocker.get("cluster_id"), str)
                    and isinstance(blocker.get("object_id"), str)
                ],
                "penalty_mm": int(round(path_penalty_total)),
            }
        )

    preferred_zone_rows.sort(key=lambda row: -int(row.get("penalty_mm") or 0))
    opening_block_rows.sort(key=lambda row: -int(row.get("penalty_mm") or 0))
    path_rows.sort(key=lambda row: -int(row.get("penalty_mm") or 0))
    path_object_rows.sort(key=lambda row: -int(row.get("penalty_mm") or 0))
    central_rows.sort(key=lambda row: -int(row.get("penalty_mm") or 0))
    edge_fit_rows.sort(key=lambda row: -int(row.get("penalty_mm") or 0))

    return {
        "cluster_penalty": cluster_penalty,
        "cluster_affinity_to_preferred_zone": preferred_zone_rows[:12],
        "opening_band_blocking": opening_block_rows[:16],
        "main_path_clearance": {
            "paths": path_rows[:16],
            "blocking_objects": path_object_rows[:20],
            "worst_blocked_ratio": max(
                [float(row.get("blocked_ratio") or 0.0) for row in path_rows] or [0.0]
            ),
            "worst_clearance_shortage_mm": max(
                [int(row.get("clearance_shortage_mm") or 0) for row in path_rows] or [0]
            ),
            "lowest_min_clearance_mm": min(
                [int(row.get("min_clearance_mm") or 0) for row in path_rows] or [0]
            ),
        },
        "central_congestion": central_rows[:12],
        "cluster_edge_vs_center_fit": edge_fit_rows[:12],
    }


def _best_open_dir(
    point: Tuple[float, float],
    room_center: Tuple[float, float],
    room_poly: Any,
    blockers: List[Any],
) -> Tuple[Tuple[float, float] | None, float]:
    max_distance = max(1800.0, room_poly.length)
    best_dir = None
    best_clear = -1.0
    for d in _candidate_open_dirs(point, room_center):
        clear = _directional_clearance_mm(point, d, room_poly, blockers, max_distance)
        if clear > best_clear:
            best_dir = d
            best_clear = clear
    return best_dir, max(0.0, best_clear)


def _scoring_object_blockers(state: Dict[str, Any], current_index: int) -> List[Any]:
    blockers = [ob["poly"] for ob in state.get("obstacles") or []]
    for i, row in enumerate(state.get("objects") or []):
        if i == current_index:
            continue
        blockers.append(row["poly"])
    return blockers


def _orientation_debug_index(
    score: Dict[str, Any],
) -> tuple[
    Dict[Tuple[str, str], Dict[str, Any]],
    Dict[Tuple[str, str], Dict[str, Any]],
    Dict[str, Dict[str, Any]],
]:
    best_object = {}
    front_clearance = {}
    best_cluster = {}

    for item in score.get("orientation_debug") or []:
        if not isinstance(item, dict):
            continue
        cid = item.get("cluster_id")
        if not isinstance(cid, str):
            continue

        pen = int(item.get("penalty_mm") or 0)
        if item.get("kind") == "object_orientation":
            oid = item.get("object_id")
            if isinstance(oid, str):
                key = (cid, oid)
                prev = best_object.get(key)
                if prev is None or pen > int(prev.get("penalty_mm") or 0):
                    best_object[key] = deepcopy(item)
                if str(item.get("intent") or "") == "front_to_open_space":
                    front_clearance[key] = deepcopy(item)
        else:
            prev_cluster = best_cluster.get(cid)
            if prev_cluster is None or pen > int(prev_cluster.get("penalty_mm") or 0):
                best_cluster[cid] = deepcopy(item)

    return best_object, front_clearance, best_cluster


def _build_materialized_layout(
    payload: Dict[str, Any], state: Dict[str, Any]
) -> Dict[str, Any]:
    room_model = _room_model(payload)
    room = deepcopy(room_model.get("room") or {})
    if isinstance(room_model.get("openings"), dict):
        room["openings"] = deepcopy(room_model.get("openings") or {})
    if isinstance(room_model.get("obstacles"), list):
        room["obstacles"] = deepcopy(room_model.get("obstacles") or [])

    clusters = []
    for row in state.get("clusters") or []:
        clusters.append(
            {
                "cluster_id": row.get("cluster_id"),
                "variant_id": row.get("variant_id"),
                "bbox": deepcopy(row.get("bbox") or {}),
                "world_center": deepcopy(row.get("world_center") or {}),
                "front_world": deepcopy(row.get("front_world")),
                "axis_world": deepcopy(row.get("axis_world")),
                "anchor_override": row.get("anchor_override"),
            }
        )

    objects = []
    for row in state.get("objects") or []:
        objects.append(
            {
                "object_id": row.get("object_id"),
                "cluster_id": row.get("cluster_id"),
                "variant_id": row.get("variant_id"),
                "bbox": deepcopy(row.get("bbox") or {}),
                "world_center": deepcopy(row.get("world_center") or {}),
                "front_world": deepcopy(row.get("front_world")),
                "axis_world": deepcopy(row.get("axis_world")),
                "required_clearance_mm": int(row.get("required_clearance_mm") or 0),
            }
        )

    return {"room": room, "clusters": clusters, "objects": objects}


def _build_goal_alignment_summary(
    payload: Dict[str, Any], score: Dict[str, Any]
) -> Dict[str, Any]:
    cluster_intents: List[Dict[str, Any]] = []
    object_intents: List[Dict[str, Any]] = []
    global_issues: List[Dict[str, Any]] = []

    for row in score.get("orientation_debug") or []:
        if not isinstance(row, dict):
            continue
        item = {
            "cluster_id": row.get("cluster_id"),
            "intent": row.get("intent"),
            "penalty_mm": int(row.get("penalty_mm") or 0),
            "dot": row.get("dot"),
        }
        if row.get("kind") == "object_orientation":
            item["object_id"] = row.get("object_id")
            object_intents.append(item)
        else:
            cluster_intents.append(item)

    for row in score.get("global_layout_debug") or []:
        if not isinstance(row, dict):
            continue
        item = {
            "kind": row.get("kind"),
            "cluster_id": row.get("cluster_id"),
            "penalty_mm": int(row.get("penalty_mm") or 0),
        }
        if isinstance(row.get("object_id"), str):
            item["object_id"] = row.get("object_id")
        if isinstance(row.get("constraint_type"), str):
            item["constraint_type"] = row.get("constraint_type")
        global_issues.append(item)

    cluster_intents.sort(
        key=lambda row: (
            -int(row.get("penalty_mm") or 0),
            str(row.get("cluster_id") or ""),
            str(row.get("intent") or ""),
        )
    )
    object_intents.sort(
        key=lambda row: (
            -int(row.get("penalty_mm") or 0),
            str(row.get("cluster_id") or ""),
            str(row.get("object_id") or ""),
            str(row.get("intent") or ""),
        )
    )
    global_issues.sort(
        key=lambda row: (
            -int(row.get("penalty_mm") or 0),
            str(row.get("cluster_id") or ""),
            str(row.get("kind") or ""),
        )
    )

    return {
        "room_notes": _room_notes(payload)[:8],
        "top_cluster_intents": cluster_intents[:8],
        "top_object_intents": object_intents[:10],
        "top_global_layout_issues": global_issues[:10],
    }


def _build_objects_world_from_state(
    state: Dict[str, Any], score: Dict[str, Any]
) -> List[Dict[str, Any]]:
    dominant_by_object, front_clear_by_object, _ = _orientation_debug_index(score)
    out: List[Dict[str, Any]] = []

    for row in state.get("objects") or []:
        cid = str(row.get("cluster_id") or "")
        oid = str(row.get("object_id") or "")
        dominant = dominant_by_object.get((cid, oid), {})
        front_clear = front_clear_by_object.get((cid, oid), {})
        bbox = row.get("bbox") or {}
        out.append(
            {
                "cluster_id": cid,
                "object_id": oid,
                "cluster_rot": int(row.get("rotation_ccw") or 0),
                "local_rect": deepcopy(row.get("local_rect") or {}),
                "polygon_ccw": deepcopy(row.get("polygon_ccw") or []),
                "bbox": deepcopy(bbox),
                "world_center": deepcopy(row.get("world_center") or {}),
                "size_mm": {
                    "w": int((row.get("local_rect") or {}).get("w") or 0),
                    "h": int((row.get("local_rect") or {}).get("h") or 0),
                },
                "front_world": deepcopy(row.get("front_world")),
                "axis_world": deepcopy(row.get("axis_world")),
                "required_clearance_mm": int(row.get("required_clearance_mm") or 0),
                "current_front_clear_mm": front_clear.get("front_clear_mm"),
                "current_back_clear_mm": front_clear.get("back_clear_mm"),
                "best_clear_mm": front_clear.get("best_clear_mm"),
                "distance_to_nearest_wall_mm": None,
                "dominant_debug_intent": dominant.get("intent"),
                "dominant_penalty_mm": dominant.get("penalty_mm"),
                "dominant_dot": dominant.get("dot"),
            }
        )

    out.sort(
        key=lambda item: (
            item.get("cluster_id") or "",
            -int(item.get("dominant_penalty_mm") or 0),
            item.get("object_id") or "",
        )
    )
    return out


def _build_metrics(
    payload: Dict[str, Any], state: Dict[str, Any], score: Dict[str, Any]
) -> Dict[str, Any]:
    openings = _openings(payload)
    room_center = state["room_center"]
    dominant_by_object, front_clear_by_object, dominant_by_cluster = (
        _orientation_debug_index(score)
    )

    object_front_clearance = []
    for row in state.get("objects") or []:
        cid = str(row.get("cluster_id") or "")
        oid = str(row.get("object_id") or "")
        center = (
            float((row.get("world_center") or {}).get("x") or 0.0),
            float((row.get("world_center") or {}).get("y") or 0.0),
        )
        door_distance, window_distance = _door_window_metrics(center, openings)
        front_debug = front_clear_by_object.get((cid, oid), {})
        current_front = front_debug.get("front_clear_mm")
        best_clear = front_debug.get("best_clear_mm")
        required = int(row.get("required_clearance_mm") or 0)
        shortage = None
        if current_front is not None:
            shortage = max(0, required - int(current_front))
        object_front_clearance.append(
            {
                "cluster_id": cid,
                "object_id": oid,
                "required_clearance_mm": required,
                "current_front_clear_mm": current_front,
                "best_open_clear_mm": best_clear,
                "shortage_mm": shortage,
                "dominant_penalty_mm": dominant_by_object.get((cid, oid), {}).get(
                    "penalty_mm"
                ),
                "distance_to_nearest_door_mm": door_distance,
                "distance_to_nearest_window_mm": window_distance,
            }
        )

    cluster_entry_proximity = []
    cluster_window_alignment = []
    cluster_window_band_occupancy = []
    zone_usage = []
    near_entry_clusters = []

    for cluster in state.get("clusters") or []:
        cid = str(cluster.get("cluster_id") or "")
        center = (
            float((cluster.get("world_center") or {}).get("x") or 0.0),
            float((cluster.get("world_center") or {}).get("y") or 0.0),
        )
        door_distance, window_distance = _door_window_metrics(center, openings)
        zone = _classify_cluster_zone(center, room_center, openings)
        entry_row = {
            "cluster_id": cid,
            "distance_to_nearest_door_mm": door_distance,
        }
        cluster_entry_proximity.append(entry_row)
        if door_distance is not None and door_distance < 1200:
            near_entry_clusters.append({**entry_row, "zone": zone})

        dominant_cluster = dominant_by_cluster.get(cid, {})
        cluster_window_alignment.append(
            {
                "cluster_id": cid,
                "distance_to_nearest_window_mm": window_distance,
                "dominant_intent": dominant_cluster.get("intent"),
                "dominant_penalty_mm": dominant_cluster.get("penalty_mm"),
                "dot": dominant_cluster.get("dot"),
            }
        )
        cluster_window_band_occupancy.append(
            {
                "cluster_id": cid,
                "distance_to_nearest_window_mm": window_distance,
                "window_clearance_hits": _window_clearance_hits(state, cluster_id=cid),
            }
        )
        zone_usage.append({"cluster_id": cid, "zone": zone})

    obstructing_objects = []
    for row in object_front_clearance:
        distance = row.get("distance_to_nearest_door_mm")
        if distance is not None and int(distance) < 900:
            obstructing_objects.append(
                {
                    "cluster_id": row["cluster_id"],
                    "object_id": row["object_id"],
                    "distance_to_nearest_door_mm": distance,
                }
            )

    return {
        "score_summary": {
            "score": int(score.get("score") or 0),
            "penalties": deepcopy(score.get("penalties") or {}),
            "hard_violation_count": int(score.get("hard_violation_count") or 0),
            "quality_gate_result": str(
                ((score.get("quality_gate") or {}).get("result") or "PASS")
            ),
            "quality_gate_reasons": deepcopy(
                ((score.get("quality_gate") or {}).get("reasons") or [])
            ),
        },
        "goal_alignment_summary": _build_goal_alignment_summary(payload, score),
        "cluster_internal_constraint_fidelity": [
            deepcopy(row)
            for row in (score.get("global_layout_debug") or [])
            if isinstance(row, dict)
            and str(row.get("kind") or "") == "cluster_internal_constraint"
        ][:16],
        "cluster_affinity_to_preferred_zone": deepcopy(
            (score.get("global_layout_metrics") or {}).get(
                "cluster_affinity_to_preferred_zone"
            )
            or []
        ),
        "opening_band_blocking": deepcopy(
            (score.get("global_layout_metrics") or {}).get("opening_band_blocking")
            or []
        ),
        "main_path_clearance": deepcopy(
            (score.get("global_layout_metrics") or {}).get("main_path_clearance") or {}
        ),
        "central_congestion": deepcopy(
            (score.get("global_layout_metrics") or {}).get("central_congestion") or []
        ),
        "cluster_edge_vs_center_fit": deepcopy(
            (score.get("global_layout_metrics") or {}).get("cluster_edge_vs_center_fit")
            or []
        ),
        "object_front_clearance": object_front_clearance,
        "cluster_entry_proximity": cluster_entry_proximity,
        "cluster_window_alignment": cluster_window_alignment,
        "cluster_window_band_occupancy": cluster_window_band_occupancy,
        "path_obstruction_summary": {
            "near_entry_clusters": near_entry_clusters,
            "near_entry_objects": obstructing_objects,
            "worst_paths": [
                {
                    "path_id": row.get("path_id"),
                    "target_cluster_id": row.get("target_cluster_id"),
                    "priority": row.get("priority"),
                    "min_clearance_mm": row.get("min_clearance_mm"),
                    "clearance_shortage_mm": row.get("clearance_shortage_mm"),
                    "blocked_samples": row.get("blocked_samples"),
                    "sample_count": row.get("sample_count"),
                    "blocking_clusters": deepcopy(row.get("blocking_clusters") or []),
                    "blocking_objects": deepcopy(row.get("blocking_objects") or []),
                }
                for row in (
                    (
                        (score.get("global_layout_metrics") or {}).get(
                            "main_path_clearance"
                        )
                        or {}
                    ).get("paths")
                    or []
                )[:4]
                if isinstance(row, dict)
            ],
            "main_path_blockers": [
                deepcopy(row)
                for row in (
                    (
                        (score.get("global_layout_metrics") or {}).get(
                            "main_path_clearance"
                        )
                        or {}
                    ).get("blocking_objects")
                    or []
                )[:8]
                if isinstance(row, dict)
            ],
        },
        "zone_usage_summary": zone_usage,
        "orientation_debug": deepcopy(score.get("orientation_debug") or [])[:16],
        "global_layout_debug": deepcopy(score.get("global_layout_debug") or [])[:16],
    }


def _violations_by_cluster(errors: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for error in errors:
        if not isinstance(error, dict):
            continue
        cluster_ids = []
        for key in ("cluster_id", "a_cluster_id", "b_cluster_id"):
            value = error.get(key)
            if isinstance(value, str) and value:
                cluster_ids.append(value)
        for cid in cluster_ids:
            slot = out.setdefault(cid, {"error_count": 0, "codes": [], "details": []})
            code = str(error.get("code") or "UNKNOWN")
            slot["error_count"] += 1
            if code not in slot["codes"]:
                slot["codes"].append(code)
            slot["details"].append(deepcopy(error))
    return out


def _build_key_findings(
    score: Dict[str, Any], metrics: Dict[str, Any], errors: List[Dict[str, Any]]
) -> List[str]:
    findings: List[str] = []

    for error in errors[:4]:
        if not isinstance(error, dict):
            continue
        code = str(error.get("code") or "UNKNOWN")
        if code == "OBJECT_OVERLAP":
            findings.append(
                "Hard invalid: overlap between "
                f"{error.get('a_cluster_id')}.{error.get('a_object_id')} and "
                f"{error.get('b_cluster_id')}.{error.get('b_object_id')}."
            )
        elif code == "OBJECT_OUT_OF_BOUNDS":
            findings.append(
                "Hard invalid: "
                f"{error.get('cluster_id')}.{error.get('object_id')} leaves the room."
            )
        elif code == "OBJECT_HITS_OBSTACLE":
            findings.append(
                "Hard invalid: "
                f"{error.get('cluster_id')}.{error.get('object_id')} intersects "
                f"{error.get('obstacle_id')}."
            )

    if findings:
        return findings[:6]

    for row in (score.get("prioritized_clusters") or [])[:3]:
        cid = row.get("cluster_id")
        cluster_score = row.get("score")
        findings.append(
            f"Cluster {cid} carries the largest remaining soft penalty ({cluster_score})."
        )

    for row in (metrics.get("cluster_internal_constraint_fidelity") or [])[:2]:
        if not isinstance(row, dict):
            continue
        findings.append(
            "Cluster "
            f"{row.get('cluster_id')} drifts from local constraint "
            f"{row.get('constraint_type')} (penalty={row.get('penalty_mm')})."
        )
        if len(findings) >= 6:
            return findings[:6]

    for row in (metrics.get("cluster_affinity_to_preferred_zone") or [])[:2]:
        if not isinstance(row, dict):
            continue
        findings.append(
            "Cluster "
            f"{row.get('cluster_id')} sits away from its preferred zone "
            f"{row.get('preferred_zone')} (penalty={row.get('penalty_mm')})."
        )
        if len(findings) >= 6:
            return findings[:6]

    for row in ((metrics.get("main_path_clearance") or {}).get("paths") or [])[:2]:
        if not isinstance(row, dict):
            continue
        shortage = int(row.get("clearance_shortage_mm") or 0)
        blocked_samples = int(row.get("blocked_samples") or 0)
        if shortage <= 0 and blocked_samples <= 0:
            continue
        target = str(row.get("target_cluster_id") or "room target")
        findings.append(
            f"Main path to {target} narrows to {row.get('min_clearance_mm')}mm "
            f"(shortage={shortage}, blocked_samples={blocked_samples})."
        )
        if len(findings) >= 6:
            return findings[:6]

    for row in (metrics.get("object_front_clearance") or [])[:4]:
        current_front = row.get("current_front_clear_mm")
        best_clear = row.get("best_open_clear_mm")
        shortage = row.get("shortage_mm")
        if current_front is None or best_clear is None:
            continue
        findings.append(
            f"Object {row.get('cluster_id')}.{row.get('object_id')} has "
            f"{current_front}mm front clearance vs {best_clear}mm best open space "
            f"(shortage={shortage})."
        )
        if len(findings) >= 6:
            break

    return findings[:6]


def _build_repair_debug_from_score(
    score: Dict[str, Any], candidate_counts: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    orientation_debug = deepcopy(score.get("orientation_debug") or [])
    global_layout_debug = deepcopy(score.get("global_layout_debug") or [])
    critical_orientation_debug = [
        row for row in orientation_debug if int(row.get("penalty_mm") or 0) >= 400
    ]
    return {
        "seed_metrics": {
            "score": int(score.get("score") or 0),
            "penalties": deepcopy(score.get("penalties") or {}),
            "global_layout_metrics": deepcopy(score.get("global_layout_metrics") or {}),
        },
        "seed_verify": {
            "hard_valid": bool(score.get("hard_valid")),
            "quality": {
                "layout_score": int(score.get("score") or 0),
                "orientation_debug": orientation_debug,
                "critical_orientation_debug": critical_orientation_debug,
                "global_layout_debug": global_layout_debug,
            },
            "repair_guidance": {
                "prioritized_clusters": deepcopy(
                    score.get("prioritized_clusters") or []
                ),
                "prioritized_objects": deepcopy(score.get("prioritized_objects") or []),
            },
            "quality_gate": {
                "result": "PASS"
                if bool(score.get("hard_valid")) and not critical_orientation_debug
                else "REVISE"
            },
        },
        "repair_targets": {
            "prioritized_clusters": deepcopy(score.get("prioritized_clusters") or []),
            "prioritized_objects": deepcopy(score.get("prioritized_objects") or []),
        },
        "candidate_counts": deepcopy(candidate_counts or {}),
        "quality_gate": {
            "result": "PASS"
            if bool(score.get("hard_valid")) and not critical_orientation_debug
            else "REVISE"
        },
    }


def PromotePhase2RepairToSeedPayload(
    *, payload: Dict[str, Any], repair: Dict[str, Any]
) -> Dict[str, Any]:
    promoted = deepcopy(payload)
    cluster_transforms, selected_variants = _resolved_seed_layout(payload, repair)
    promoted["seed_layout"] = {
        "cluster_transforms": cluster_transforms,
        "selected_variants": selected_variants,
    }

    variant_by_cluster = {
        str(item.get("cluster_id")): str(item.get("variant_id") or "")
        for item in selected_variants
        if isinstance(item, dict) and isinstance(item.get("cluster_id"), str)
    }
    repairs_by_cluster: Dict[str, List[Dict[str, Any]]] = {}
    for row in repair.get("object_repairs") or []:
        if isinstance(row, dict) and isinstance(row.get("cluster_id"), str):
            repairs_by_cluster.setdefault(row["cluster_id"], []).append(deepcopy(row))

    promoted_cards = []
    for transform in cluster_transforms:
        cid = str(transform.get("cluster_id") or "")
        card = _current_cluster_variant_card(
            payload, cid, variant_by_cluster.get(cid) or None
        )
        if not card:
            continue

        bbox = _cluster_local_bbox(card)
        rects = []
        rect_order = []
        for rect in (card.get("cluster_footprint") or {}).get("rects") or []:
            if isinstance(rect, dict) and isinstance(rect.get("id"), str):
                rects.append(deepcopy(rect))
                rect_order.append(rect["id"])
        rect_map = {row["id"]: row for row in rects}
        placements = []
        for placement in card.get("local_placements") or []:
            if isinstance(placement, dict) and isinstance(placement.get("id"), str):
                placements.append(deepcopy(placement))
        placement_map = {row["id"]: row for row in placements}

        orientation_meta = deepcopy(card.get("orientation_meta") or {})
        important = deepcopy(orientation_meta.get("important_objects") or {})
        cluster_front_local = parse_vec2(orientation_meta.get("cluster_front_local"))

        for rep in repairs_by_cluster.get(cid, []):
            oid = rep.get("object_id")
            op = rep.get("op")
            params = rep.get("params") or {}
            if op == "set_anchor":
                card["anchor_override"] = params.get("anchor")
                continue
            if not isinstance(oid, str) or oid not in rect_map:
                continue

            rect = rect_map[oid]
            placement = placement_map.get(
                oid, {"id": oid, "x": rect.get("x", 0), "y": rect.get("y", 0), "rot": 0}
            )
            obj_meta = (
                important.get(oid) if isinstance(important.get(oid), dict) else {}
            )
            front_local = (
                parse_vec2((obj_meta or {}).get("front_local")) or cluster_front_local
            )
            axis_local = parse_vec2((obj_meta or {}).get("axis_local"))

            if op == "rotate_object":
                rr = int(params.get("rot", 0)) % 360
                rect = _rotate_rect_about_center(rect, rr)
                placement["rot"] = int((placement.get("rot") or 0) + rr) % 360
                front_local = rotate_vec_ccw_90s(front_local, rr)
                axis_local = rotate_vec_ccw_90s(axis_local, rr)
            elif op == "mirror_object":
                axis = str(params.get("axis") or "x").lower()
                rect = _mirror_rect_in_bbox(rect, bbox, axis)
                placement["x"] = rect.get("x", placement.get("x"))
                placement["y"] = rect.get("y", placement.get("y"))
                front_local = mirror_vec(front_local, axis)
                axis_local = mirror_vec(axis_local, axis)
            elif op == "nudge_object":
                dx = int(round(float(params.get("dx", 0))))
                dy = int(round(float(params.get("dy", 0))))
                rect = {
                    **rect,
                    "x": int(round(float(rect.get("x", 0)) + dx)),
                    "y": int(round(float(rect.get("y", 0)) + dy)),
                }
                placement["x"] = int(round(float(placement.get("x", 0)) + dx))
                placement["y"] = int(round(float(placement.get("y", 0)) + dy))
            elif op == "swap_objects":
                other_id = str(params.get("other_object_id") or "")
                other_rect = rect_map.get(other_id)
                if isinstance(other_rect, dict):
                    ox, oy = other_rect.get("x"), other_rect.get("y")
                    other_rect["x"], other_rect["y"] = rect.get("x"), rect.get("y")
                    rect["x"], rect["y"] = ox, oy
                    other_place = placement_map.get(other_id)
                    if isinstance(other_place, dict):
                        px, py = other_place.get("x"), other_place.get("y")
                        other_place["x"], other_place["y"] = (
                            placement.get("x"),
                            placement.get("y"),
                        )
                        placement["x"], placement["y"] = px, py
                        placement_map[other_id] = other_place
                    rect_map[other_id] = other_rect
            elif op == "set_front_override":
                front_local = normalize_vec(
                    (float(params.get("dx", 0.0)), float(params.get("dy", 0.0)))
                )

            placement["x"] = int(round(float(rect.get("x", placement.get("x", 0)))))
            placement["y"] = int(round(float(rect.get("y", placement.get("y", 0)))))
            rect_map[oid] = rect
            placement_map[oid] = placement
            important.setdefault(oid, {})
            if front_local is not None:
                important[oid]["front_local"] = {
                    "dx": round(front_local[0], 3),
                    "dy": round(front_local[1], 3),
                }
            if axis_local is not None:
                important[oid]["axis_local"] = {
                    "dx": round(axis_local[0], 3),
                    "dy": round(axis_local[1], 3),
                }

        card.setdefault("cluster_footprint", {})
        card["cluster_footprint"]["rects"] = [
            deepcopy(rect_map[oid]) for oid in rect_order if oid in rect_map
        ]
        card["cluster_footprint"]["local_bbox"] = _cluster_local_bbox(card)
        card["orientation_meta"] = orientation_meta
        card["orientation_meta"]["important_objects"] = important
        ordered_placements = []
        for oid in rect_order:
            placement = placement_map.get(oid)
            if isinstance(placement, dict):
                ordered_placements.append(placement)
        card["local_placements"] = ordered_placements
        promoted_cards.append(card)

    promoted["cluster_cards"] = promoted_cards
    state = materialize_phase2_state(promoted, repair=None)
    score = score_phase2_state(promoted, state)
    promoted["objects_world"] = _build_objects_world_from_state(state, score)
    previous_debug = payload.get("repair_debug") or {}
    promoted["repair_debug"] = _build_repair_debug_from_score(
        score, candidate_counts=previous_debug.get("candidate_counts") or {}
    )
    promoted.pop("tool_context", None)
    return promoted


def score_phase2_state(
    payload: Dict[str, Any], state: Dict[str, Any]
) -> Dict[str, Any]:
    room_poly = state["room_polygon"]
    room_center = state["room_center"]
    openings = _openings(payload)
    goals = payload.get("goals") or {}
    relation_plan = goals.get("relation_plan_used") or {}
    repair_debug = payload.get("repair_debug") or {}

    orientation_debug: List[Dict[str, Any]] = []
    global_layout_debug: List[Dict[str, Any]] = []
    cluster_penalty: Dict[str, float] = {}
    object_penalty: Dict[Tuple[str, str], float] = {}
    macro_penalty = 0.0
    micro_penalty = 0.0

    cluster_state_by_id = {
        str(row.get("cluster_id") or ""): row for row in state.get("clusters") or []
    }
    anchor_hints_by_cluster = _anchor_layout_hints_by_cluster(payload)

    # object-level generic access/open-space score
    for idx, row in enumerate(state.get("objects") or []):
        cid = row["cluster_id"]
        oid = row["object_id"]
        center = (float(row["world_center"]["x"]), float(row["world_center"]["y"]))
        front_world = parse_vec2(row.get("front_world"))
        blockers = _scoring_object_blockers(state, idx)
        max_distance = max(1800.0, room_poly.length)
        front_clear = _directional_clearance_mm(
            center, front_world, room_poly, blockers, max_distance
        )
        best_dir, best_clear = _best_open_dir(center, room_center, room_poly, blockers)
        required = float(row.get("required_clearance_mm") or 0.0)
        dot = _dot(front_world, best_dir)
        shortage = max(0.0, required - front_clear)
        misalign = max(0.0, 1.0 - (dot if dot is not None else -1.0))
        pen = shortage * 2.2 + misalign * 600.0
        object_penalty[(cid, oid)] = object_penalty.get((cid, oid), 0.0) + pen
        cluster_penalty[cid] = cluster_penalty.get(cid, 0.0) + pen
        micro_penalty += pen
        orientation_debug.append(
            {
                "kind": "object_orientation",
                "cluster_id": cid,
                "object_id": oid,
                "intent": "front_to_open_space",
                "penalty_mm": int(round(pen)),
                "front_clear_mm": int(round(front_clear)),
                "best_clear_mm": int(round(best_clear)),
                "dot": None if dot is None else round(dot, 3),
            }
        )

    state_objects = [
        row for row in (state.get("objects") or []) if isinstance(row, dict)
    ]

    def _resolve_relation_target_object(
        *,
        cluster_id: str,
        object_id: str,
        target_object_id: str,
        target_object_cluster_id: str | None,
    ) -> Dict[str, Any] | None:
        if not target_object_id:
            return None
        preferred_cluster_id = (
            target_object_cluster_id.strip()
            if isinstance(target_object_cluster_id, str)
            and target_object_cluster_id.strip()
            else None
        )
        fallback_match: Dict[str, Any] | None = None
        for row in state_objects:
            if str(row.get("object_id") or "") != target_object_id:
                continue
            row_cluster_id = str(row.get("cluster_id") or "")
            if row_cluster_id == cluster_id and target_object_id == object_id:
                continue
            if preferred_cluster_id is not None:
                if row_cluster_id == preferred_cluster_id:
                    return row
                continue
            if fallback_match is None:
                fallback_match = row
        return fallback_match

    def _reference_anchor_or_target_object(
        *,
        cluster_id: str,
        object_id: str,
        target_object_id: str,
        target_object_cluster_id: str | None,
    ) -> Dict[str, Any] | None:
        target_object = _resolve_relation_target_object(
            cluster_id=cluster_id,
            object_id=object_id,
            target_object_id=target_object_id,
            target_object_cluster_id=target_object_cluster_id,
        )
        if isinstance(target_object, dict):
            return target_object
        hint = anchor_hints_by_cluster.get(cluster_id) or {}
        hinted_target_cluster_id = str(hint.get("target_cluster_id") or "").strip()
        if hinted_target_cluster_id and hinted_target_cluster_id != cluster_id:
            target_anchor = _cluster_anchor_state(
                cluster_id=hinted_target_cluster_id,
                state_objects=state_objects,
                cluster_state_by_id=cluster_state_by_id,
                anchor_hints_by_cluster=anchor_hints_by_cluster,
            )
            if isinstance(target_anchor, dict):
                return target_anchor
        anchor_row = _cluster_anchor_state(
            cluster_id=cluster_id,
            state_objects=state_objects,
            cluster_state_by_id=cluster_state_by_id,
            anchor_hints_by_cluster=anchor_hints_by_cluster,
        )
        if (
            isinstance(anchor_row, dict)
            and str(anchor_row.get("object_id") or "") != object_id
        ):
            return anchor_row
        return None

    # relation-plan specific object intents
    for item in relation_plan.get("object_orientations") or []:
        if not isinstance(item, dict):
            continue
        cid = item.get("cluster_id")
        oid = item.get("object_id")
        intents = item.get("intents") or []
        if not isinstance(cid, str) or not isinstance(oid, str):
            continue
        target = None
        for row in state.get("objects") or []:
            if row["cluster_id"] == cid and row["object_id"] == oid:
                target = row
                break
        if target is None:
            continue
        anchor_ids = _cluster_anchor_ids_from_state_cluster(
            cluster_state_by_id.get(cid) or {}
        )
        center = (
            float(target["world_center"]["x"]),
            float(target["world_center"]["y"]),
        )
        front_world = parse_vec2(target.get("front_world"))
        target_object_id = str(item.get("target_object_id") or "")
        target_object_cluster_id = item.get("target_object_cluster_id")
        for intent in intents:
            intent = str(intent).lower()
            pen = 0.0
            dot = None
            back_dbg: dict[str, int] | None = None
            resolved_target_cluster_id = None
            if intent in {"face_window", "front_to_window"}:
                mid = _nearest_midpoint(center, openings.get("windows") or [])
                if mid is not None:
                    dot = _dot(front_world, _vec_from_to(center, mid))
                    weight = 620.0 if intent == "front_to_window" else 520.0
                    pen = max(0.0, 1.0 - (dot if dot is not None else -1.0)) * weight
            elif intent in {"face_entry", "front_to_entry"}:
                mid = _nearest_midpoint(center, openings.get("doors") or [])
                if mid is not None:
                    dot = _dot(front_world, _vec_from_to(center, mid))
                    pen = max(0.0, 1.0 - (dot if dot is not None else -1.0)) * 520.0
            elif intent == "back_to_wall":
                blockers = [ob["poly"] for ob in state.get("obstacles") or []]
                for other in state_objects:
                    if (
                        str(other.get("cluster_id") or "") == cid
                        and str(other.get("object_id") or "") == oid
                    ):
                        continue
                    blockers.append(other["poly"])
                pen, dot, back_dbg = _back_to_wall_penalty(
                    point=center,
                    front=front_world,
                    room_poly=room_poly,
                    blockers=blockers,
                    desired_back_clear_mm=260.0,
                    desired_front_advantage_mm=220.0,
                    weight_scale=1.0,
                )
            elif intent == "preserve_front_access":
                front_clear = next(
                    (
                        d["front_clear_mm"]
                        for d in orientation_debug
                        if d.get("cluster_id") == cid
                        and d.get("object_id") == oid
                        and d.get("intent") == "front_to_open_space"
                    ),
                    0,
                )
                req = int(target.get("required_clearance_mm") or 0)
                pen = max(0.0, req - front_clear) * 2.0
            elif intent in {"face_object", "face_away_from_object"}:
                target_object = _resolve_relation_target_object(
                    cluster_id=cid,
                    object_id=oid,
                    target_object_id=target_object_id,
                    target_object_cluster_id=target_object_cluster_id
                    if isinstance(target_object_cluster_id, str)
                    else None,
                )
                if target_object is not None:
                    resolved_target_cluster_id = str(
                        target_object.get("cluster_id") or ""
                    ).strip()
                    is_cross_cluster = (
                        bool(resolved_target_cluster_id)
                        and resolved_target_cluster_id != cid
                    )
                    if is_cross_cluster and oid not in anchor_ids:
                        continue
                    target_center = (
                        float(target_object["world_center"]["x"]),
                        float(target_object["world_center"]["y"]),
                    )
                    desired_vec = _vec_from_to(center, target_center)
                    if desired_vec is not None and intent == "face_away_from_object":
                        desired_vec = (-desired_vec[0], -desired_vec[1])
                    dot = _dot(front_world, desired_vec)
                    weight = 860.0 if intent == "face_object" else 780.0
                    pen = max(0.0, 1.0 - (dot if dot is not None else -1.0)) * weight
            elif intent in {
                "same_direction_as_anchor",
                "same_view_side_as_primary_pair",
            }:
                reference_object = _reference_anchor_or_target_object(
                    cluster_id=cid,
                    object_id=oid,
                    target_object_id=target_object_id,
                    target_object_cluster_id=target_object_cluster_id
                    if isinstance(target_object_cluster_id, str)
                    else None,
                )
                if isinstance(reference_object, dict):
                    resolved_target_cluster_id = str(
                        reference_object.get("cluster_id") or ""
                    ).strip()
                    reference_front = parse_vec2(reference_object.get("front_world"))
                    dot = _dot(front_world, reference_front)
                    weight = (
                        680.0 if intent == "same_view_side_as_primary_pair" else 560.0
                    )
                    pen = max(0.0, 1.0 - (dot if dot is not None else -1.0)) * weight
            elif intent == "align_with_anchor_axis":
                reference_object = _reference_anchor_or_target_object(
                    cluster_id=cid,
                    object_id=oid,
                    target_object_id=target_object_id,
                    target_object_cluster_id=target_object_cluster_id
                    if isinstance(target_object_cluster_id, str)
                    else None,
                )
                if isinstance(reference_object, dict):
                    resolved_target_cluster_id = str(
                        reference_object.get("cluster_id") or ""
                    ).strip()
                    reference_axis = parse_vec2(
                        reference_object.get("axis_world")
                    ) or parse_vec2(reference_object.get("front_world"))
                    current_axis = parse_vec2(target.get("axis_world")) or front_world
                    raw_dot = _dot(current_axis, reference_axis)
                    dot = None if raw_dot is None else abs(raw_dot)
                    pen = max(0.0, 1.0 - (dot if dot is not None else -1.0)) * 520.0
            elif intent in {"not_behind_anchor_view", "in_front_of_anchor"}:
                reference_object = _reference_anchor_or_target_object(
                    cluster_id=cid,
                    object_id=oid,
                    target_object_id=target_object_id,
                    target_object_cluster_id=target_object_cluster_id
                    if isinstance(target_object_cluster_id, str)
                    else None,
                )
                if isinstance(reference_object, dict):
                    resolved_target_cluster_id = str(
                        reference_object.get("cluster_id") or ""
                    ).strip()
                    reference_center = (
                        float(
                            (reference_object.get("world_center") or {}).get("x") or 0.0
                        ),
                        float(
                            (reference_object.get("world_center") or {}).get("y") or 0.0
                        ),
                    )
                    reference_front = parse_vec2(reference_object.get("front_world"))
                    rel_vec = _vec_from_to(reference_center, center)
                    raw_dot = _dot(reference_front, rel_vec)
                    dist_mm = _distance_between_points(reference_center, center)
                    signed_offset_mm = 0.0 if raw_dot is None else raw_dot * dist_mm
                    threshold_mm = 120.0 if intent == "in_front_of_anchor" else 0.0
                    pen = max(0.0, threshold_mm - signed_offset_mm) * 1.25
                    dot = raw_dot
                    back_dbg = {"signed_offset_mm": int(round(signed_offset_mm))}
            if pen > 0:
                object_penalty[(cid, oid)] = object_penalty.get((cid, oid), 0.0) + pen
                cluster_penalty[cid] = cluster_penalty.get(cid, 0.0) + pen
                if (
                    intent in {"face_object", "face_away_from_object"}
                    and isinstance(resolved_target_cluster_id, str)
                    and resolved_target_cluster_id
                    and resolved_target_cluster_id != cid
                    and oid in anchor_ids
                ):
                    macro_penalty += pen
                else:
                    micro_penalty += pen
                orientation_debug.append(
                    {
                        "kind": "object_orientation",
                        "cluster_id": cid,
                        "object_id": oid,
                        "intent": intent,
                        "penalty_mm": int(round(pen)),
                        "dot": None if dot is None else round(dot, 3),
                        **(back_dbg or {}),
                        "target_object_id": target_object_id or None,
                        "target_object_cluster_id": (
                            target_object_cluster_id
                            if isinstance(target_object_cluster_id, str)
                            and target_object_cluster_id
                            else None
                        ),
                    }
                )

    # cluster-level intents
    for item in relation_plan.get("cluster_orientations") or []:
        if not isinstance(item, dict):
            continue
        cid = item.get("cluster_id")
        intents = item.get("intents") or []
        if not isinstance(cid, str):
            continue
        cluster = next(
            (x for x in state.get("clusters") or [] if x["cluster_id"] == cid), None
        )
        if cluster is None:
            continue
        center = (
            float(cluster["world_center"]["x"]),
            float(cluster["world_center"]["y"]),
        )
        front = parse_vec2(cluster.get("front_world"))
        axis = parse_vec2(cluster.get("axis_world"))
        for intent in intents:
            intent = str(intent).lower()
            pen = 0.0
            dot = None
            back_dbg: dict[str, int] | None = None
            if intent == "access_to_open_space":
                blockers = [ob["poly"] for ob in state.get("obstacles") or []]
                for row in state.get("objects") or []:
                    if row["cluster_id"] != cid:
                        blockers.append(row["poly"])
                best_dir, _ = _best_open_dir(center, room_center, room_poly, blockers)
                dot = _dot(front, best_dir)
                pen = max(0.0, 1.0 - (dot if dot is not None else -1.0)) * 860.0
            elif intent == "face_window":
                mid = _nearest_midpoint(center, openings.get("windows") or [])
                if mid is not None:
                    dot = _dot(front, _vec_from_to(center, mid))
                    pen = max(0.0, 1.0 - (dot if dot is not None else -1.0)) * 760.0
            elif intent == "back_to_wall":
                blockers = [ob["poly"] for ob in state.get("obstacles") or []]
                for row in state.get("objects") or []:
                    if row["cluster_id"] != cid:
                        blockers.append(row["poly"])
                pen, dot, back_dbg = _back_to_wall_penalty(
                    point=center,
                    front=front,
                    room_poly=room_poly,
                    blockers=blockers,
                    desired_back_clear_mm=320.0,
                    desired_front_advantage_mm=260.0,
                    weight_scale=1.0,
                )
            elif intent == "axis_parallel_window":
                window_dir = _nearest_opening_line_direction(
                    center, openings.get("windows") or []
                )
                if axis is not None and window_dir is not None:
                    parallel = abs(_dot(axis, window_dir) or -1.0)
                    dot = parallel
                    pen = max(0.0, 1.0 - parallel) * 620.0
            if pen > 0:
                cluster_penalty[cid] = cluster_penalty.get(cid, 0.0) + pen
                macro_penalty += pen
                orientation_debug.append(
                    {
                        "kind": "cluster_orientation",
                        "cluster_id": cid,
                        "intent": intent,
                        "penalty_mm": int(round(pen)),
                        "dot": None if dot is None else round(dot, 3),
                        **(back_dbg or {}),
                    }
                )

    cluster_preferences = _cluster_preference_profile(payload)
    for cluster_a_id, cluster_b_id in _viewing_cluster_pairs(payload):
        cluster_a = cluster_state_by_id.get(cluster_a_id)
        cluster_b = cluster_state_by_id.get(cluster_b_id)
        if not isinstance(cluster_a, dict) or not isinstance(cluster_b, dict):
            continue

        profile_a = cluster_preferences.get(cluster_a_id, {})
        profile_b = cluster_preferences.get(cluster_b_id, {})
        role_a = str(profile_a.get("viewing_role") or "").strip().lower()
        role_b = str(profile_b.get("viewing_role") or "").strip().lower()
        if role_b == "primary" and role_a != "primary":
            primary_cluster_id, secondary_cluster_id = cluster_b_id, cluster_a_id
            primary_cluster = cluster_b
            secondary_cluster = cluster_a
        else:
            primary_cluster_id, secondary_cluster_id = cluster_a_id, cluster_b_id
            primary_cluster = cluster_a
            secondary_cluster = cluster_b

        primary_center = (
            float((primary_cluster.get("world_center") or {}).get("x") or 0.0),
            float((primary_cluster.get("world_center") or {}).get("y") or 0.0),
        )
        secondary_center = (
            float((secondary_cluster.get("world_center") or {}).get("x") or 0.0),
            float((secondary_cluster.get("world_center") or {}).get("y") or 0.0),
        )
        primary_front = parse_vec2(primary_cluster.get("front_world"))
        secondary_front = parse_vec2(secondary_cluster.get("front_world"))
        primary_target = _vec_from_to(primary_center, secondary_center)
        secondary_target = _vec_from_to(secondary_center, primary_center)

        primary_dot = _dot(primary_front, primary_target)
        secondary_dot = _dot(secondary_front, secondary_target)
        primary_penalty = (
            max(0.0, 1.0 - (primary_dot if primary_dot is not None else -1.0)) * 980.0
        )
        secondary_penalty = (
            max(0.0, 1.0 - (secondary_dot if secondary_dot is not None else -1.0))
            * 880.0
        )

        primary_edge_distance = float(
            room_poly.boundary.distance(Point(primary_center))
        )
        secondary_edge_distance = float(
            room_poly.boundary.distance(Point(secondary_center))
        )
        depth_penalty = max(0.0, primary_edge_distance - secondary_edge_distance) * 0.55

        pair_axis = _vec_from_to(primary_center, secondary_center)
        axis_alignment = max(abs(pair_axis[0]), abs(pair_axis[1])) if pair_axis else 0.0
        axis_penalty = max(0.0, 0.94 - axis_alignment) * 520.0

        pair_penalties = (
            (
                primary_cluster_id,
                primary_penalty + (depth_penalty * 0.65) + (axis_penalty * 0.5),
                primary_dot,
                "primary",
            ),
            (
                secondary_cluster_id,
                secondary_penalty + (depth_penalty * 0.35) + (axis_penalty * 0.5),
                secondary_dot,
                "secondary",
            ),
        )
        for cluster_id, penalty, dot, role in pair_penalties:
            if penalty <= 0.0:
                continue
            cluster_penalty[cluster_id] = cluster_penalty.get(cluster_id, 0.0) + penalty
            macro_penalty += penalty
            orientation_debug.append(
                {
                    "kind": "cluster_directional_relation",
                    "cluster_id": cluster_id,
                    "other_cluster_id": (
                        secondary_cluster_id
                        if cluster_id == primary_cluster_id
                        else primary_cluster_id
                    ),
                    "relation": "face_each_other",
                    "role": role,
                    "penalty_mm": int(round(penalty)),
                    "dot": None if dot is None else round(dot, 3),
                }
            )

        if depth_penalty > 0.0:
            global_layout_debug.append(
                {
                    "kind": "viewing_pair_depth",
                    "primary_cluster_id": primary_cluster_id,
                    "secondary_cluster_id": secondary_cluster_id,
                    "penalty_mm": int(round(depth_penalty)),
                    "primary_edge_distance_mm": int(round(primary_edge_distance)),
                    "secondary_edge_distance_mm": int(round(secondary_edge_distance)),
                }
            )

    internal_cluster_penalty, internal_object_penalty, internal_debug = (
        _score_cluster_internal_fidelity(state)
    )
    for cid, penalty in internal_cluster_penalty.items():
        cluster_penalty[cid] = cluster_penalty.get(cid, 0.0) + penalty
        micro_penalty += penalty
    for key, penalty in internal_object_penalty.items():
        object_penalty[key] = object_penalty.get(key, 0.0) + penalty
        cluster_penalty[key[0]] = cluster_penalty.get(key[0], 0.0) + penalty * 0.2
        micro_penalty += penalty * 0.2
    global_layout_debug.extend(internal_debug)

    global_layout_metrics = _score_global_layout_metrics(
        payload, state, openings, room_center
    )
    for cid, penalty in (global_layout_metrics.get("cluster_penalty") or {}).items():
        cluster_penalty[cid] = cluster_penalty.get(cid, 0.0) + float(penalty or 0.0)
        macro_penalty += float(penalty or 0.0)
    for row in (global_layout_metrics.get("main_path_clearance") or {}).get(
        "blocking_objects"
    ) or []:
        if not isinstance(row, dict):
            continue
        cid = row.get("cluster_id")
        oid = row.get("object_id")
        penalty = float(row.get("penalty_mm") or 0.0)
        if isinstance(cid, str) and isinstance(oid, str) and penalty > 0.0:
            object_penalty[(cid, oid)] = object_penalty.get((cid, oid), 0.0) + penalty
            macro_penalty += penalty
    for key in (
        "cluster_affinity_to_preferred_zone",
        "opening_band_blocking",
        "central_congestion",
        "cluster_edge_vs_center_fit",
    ):
        for row in global_layout_metrics.get(key) or []:
            if isinstance(row, dict):
                global_layout_debug.append({"kind": key, **deepcopy(row)})
    for row in (global_layout_metrics.get("main_path_clearance") or {}).get(
        "paths"
    ) or []:
        if isinstance(row, dict):
            global_layout_debug.append({"kind": "main_path_clearance", **deepcopy(row)})
    for row in (global_layout_metrics.get("main_path_clearance") or {}).get(
        "blocking_objects"
    ) or []:
        if isinstance(row, dict):
            global_layout_debug.append(
                {"kind": "main_path_object_blocker", **deepcopy(row)}
            )

    # amplify seeded failing objects/clusters a bit to keep focus aligned with solver
    seed_verify = repair_debug.get("seed_verify") or {}
    quality = seed_verify.get("quality") or {}
    for item in (quality.get("critical_orientation_debug") or [])[:20]:
        if not isinstance(item, dict):
            continue
        cid = item.get("cluster_id")
        oid = item.get("object_id")
        pen = float(item.get("penalty_mm") or 0.0) * 0.10
        if isinstance(cid, str):
            cluster_penalty[cid] = cluster_penalty.get(cid, 0.0) + pen
            if isinstance(oid, str):
                micro_penalty += pen
            else:
                macro_penalty += pen
        if isinstance(cid, str) and isinstance(oid, str):
            object_penalty[(cid, oid)] = object_penalty.get((cid, oid), 0.0) + pen

    hard_penalty = 1_000_000.0 * len(state.get("hard_violations") or [])
    soft_penalty = sum(cluster_penalty.values())
    total_penalty = hard_penalty + soft_penalty

    prioritized_clusters = [
        {"cluster_id": cid, "score": round(score, 2)}
        for cid, score in sorted(cluster_penalty.items(), key=lambda kv: -kv[1])
    ]
    prioritized_objects = [
        {"cluster_id": cid, "object_id": oid, "score": round(score, 2)}
        for (cid, oid), score in sorted(object_penalty.items(), key=lambda kv: -kv[1])
    ]

    critical_orientation_rows = [
        row
        for row in orientation_debug
        if isinstance(row, dict)
        and int(row.get("penalty_mm") or 0) >= 300
        and str(row.get("kind") or "")
        in {"cluster_directional_relation", "cluster_orientation", "object_orientation"}
    ]
    focal_pair_rows = [
        row
        for row in orientation_debug
        if isinstance(row, dict)
        and int(row.get("penalty_mm") or 0) >= 220
        and (
            str(row.get("relation") or "") in {"face_each_other", "access_faces_other"}
            or str(row.get("kind") or "") == "cluster_directional_relation"
        )
    ]
    quality_gate_reasons: List[str] = []
    if not bool(state.get("hard_valid")):
        quality_gate_reasons.append("hard_validity_failed")
    if critical_orientation_rows:
        quality_gate_reasons.append("critical_orientation_penalty_too_high")
    if focal_pair_rows:
        quality_gate_reasons.append("focal_pair_penalty_too_high")

    return {
        "hard_valid": bool(state.get("hard_valid")),
        "hard_violation_count": len(state.get("hard_violations") or []),
        "hard_violations": deepcopy(state.get("hard_violations") or []),
        "orientation_debug": sorted(
            orientation_debug, key=lambda x: -int(x.get("penalty_mm") or 0)
        ),
        "global_layout_debug": sorted(
            global_layout_debug, key=lambda x: -int(x.get("penalty_mm") or 0)
        ),
        "prioritized_clusters": prioritized_clusters[:8],
        "prioritized_objects": prioritized_objects[:16],
        "penalties": {
            "hard_penalty": int(round(hard_penalty)),
            "macro_penalty": int(round(macro_penalty)),
            "micro_penalty": int(round(micro_penalty)),
            "soft_penalty": int(round(soft_penalty)),
            "total_penalty": int(round(total_penalty)),
        },
        "score": -int(round(total_penalty)),
        "global_layout_metrics": {
            key: deepcopy(value)
            for key, value in global_layout_metrics.items()
            if key != "cluster_penalty"
        },
        "quality_gate": {
            "result": "PASS"
            if bool(state.get("hard_valid")) and not quality_gate_reasons
            else "REVISE",
            "reasons": quality_gate_reasons,
        },
    }


# -----------------------------------------------------------------------------
# Public tool API
# -----------------------------------------------------------------------------
def _toward_base_direction_for_side(side: str) -> tuple[int, int]:
    if side == "top":
        return (0, -1)
    if side == "bottom":
        return (0, 1)
    if side == "left":
        return (1, 0)
    if side == "right":
        return (-1, 0)
    return (0, 0)


def _spacing_offsets_for_debug_row(
    row: Dict[str, Any], *, grid_mm: int
) -> list[tuple[int, int]]:
    mapped_side = str(row.get("mapped_side") or "").strip().lower()
    if mapped_side not in {"top", "bottom", "left", "right"}:
        return []
    gap_mm = int(row.get("gap_mm") or 0)
    target_gap_mm = int(row.get("target_gap_mm") or 0)
    if gap_mm == target_gap_mm:
        return []

    toward_dx, toward_dy = _toward_base_direction_for_side(mapped_side)
    if toward_dx == 0 and toward_dy == 0:
        return []

    if gap_mm > target_gap_mm:
        step_dx, step_dy = toward_dx, toward_dy
    else:
        step_dx, step_dy = -toward_dx, -toward_dy

    distance_mm = abs(gap_mm - target_gap_mm)
    multipliers = [1]
    if distance_mm >= int(grid_mm * 1.5):
        multipliers.append(2)

    return [
        (step_dx * grid_mm * factor, step_dy * grid_mm * factor)
        for factor in multipliers
    ]


def DiagnosePhase2Seed(*, payload: Dict[str, Any]) -> Dict[str, Any]:
    state = materialize_phase2_state(payload, repair=None)
    score = score_phase2_state(payload, state)
    return {
        "result": "OK",
        "search_phase": _repair_phase(payload),
        "hard_valid": score["hard_valid"],
        "score": score["score"],
        "penalties": score["penalties"],
        "prioritized_clusters": score["prioritized_clusters"],
        "prioritized_objects": score["prioritized_objects"],
        "hard_violations": score["hard_violations"],
        "orientation_debug": score["orientation_debug"][:16],
        "global_layout_debug": score.get("global_layout_debug") or [],
        "global_layout_metrics": deepcopy(score.get("global_layout_metrics") or {}),
    }


def EnumeratePhase2RepairMoves(
    *, payload: Dict[str, Any], limit: int = 24
) -> Dict[str, Any]:
    state = materialize_phase2_state(payload, repair=None)
    score = score_phase2_state(payload, state)
    search_phase = _repair_phase(payload)
    seed_layout = payload.get("seed_layout") or {}
    tmap = _seed_transform_map(seed_layout)
    vmap = _seed_variant_map(seed_layout)
    allowed_variants = {}
    for cid, card in _cluster_cards_as_map(payload.get("cluster_cards") or {}).items():
        variant_ids = []
        for item in card.get("available_variants") or card.get("variants") or []:
            if isinstance(item, dict) and isinstance(item.get("variant_id"), str):
                variant_ids.append(item["variant_id"])
            elif isinstance(item, str):
                variant_ids.append(item)
        if variant_ids:
            allowed_variants[cid] = sorted(set(variant_ids))

    grid_mm = normalize_layout_grid_mm(
        (payload.get("room_context") or {}).get("grid_mm")
    )
    cluster_preferences = _cluster_preference_profile(payload)
    anchor_hints_by_cluster = _anchor_layout_hints_by_cluster(payload)
    cluster_state_by_id = {
        str(row.get("cluster_id") or ""): row for row in state.get("clusters") or []
    }
    protected_cluster_ids = _anchor_contract_cluster_ids(payload)
    protected_anchor_ids_by_cluster = {
        cluster_id: _cluster_anchor_ids_from_state_cluster(cluster)
        for cluster_id, cluster in cluster_state_by_id.items()
        if cluster_id in protected_cluster_ids and isinstance(cluster, dict)
    }
    room_poly = state["room_polygon"]
    room_center = state["room_center"]
    state_objects = [
        row for row in (state.get("objects") or []) if isinstance(row, dict)
    ]
    object_state_by_key = {
        (
            str(row.get("cluster_id") or ""),
            str(row.get("object_id") or ""),
        ): row
        for row in state_objects
        if isinstance(row.get("cluster_id"), str)
        and isinstance(row.get("object_id"), str)
    }
    object_index_by_key = {
        (
            str(row.get("cluster_id") or ""),
            str(row.get("object_id") or ""),
        ): index
        for index, row in enumerate(state_objects)
        if isinstance(row.get("cluster_id"), str)
        and isinstance(row.get("object_id"), str)
    }
    openings = _openings(payload)
    orientation_debug = score.get("orientation_debug") or []
    viewing_orientation_pressure = any(
        isinstance(row, dict)
        and int(row.get("penalty_mm") or 0) >= 220
        and (
            str(row.get("kind") or "") == "cluster_directional_relation"
            or str(row.get("relation") or "")
            in {"face_each_other", "access_faces_other"}
            or str(row.get("intent") or "")
            in {
                "face_object",
                "same_view_side_as_primary_pair",
                "same_direction_as_anchor",
                "not_behind_anchor_view",
                "align_with_anchor_axis",
            }
        )
        for row in orientation_debug
    )

    macro_moves: List[Dict[str, Any]] = []
    object_moves: List[Dict[str, Any]] = []

    priority_cluster_ids: List[str] = []
    for row in score["prioritized_clusters"][:4]:
        cid = row.get("cluster_id")
        if isinstance(cid, str) and cid not in priority_cluster_ids:
            priority_cluster_ids.append(cid)
    for row in (score.get("global_layout_metrics") or {}).get(
        "cluster_affinity_to_preferred_zone"
    ) or []:
        cid = row.get("cluster_id")
        if isinstance(cid, str) and cid not in priority_cluster_ids:
            priority_cluster_ids.append(cid)
        if len(priority_cluster_ids) >= 4:
            break

    def _offsets_from_vector(
        raw_dx: float, raw_dy: float, *, include_far_steps: bool
    ) -> list[tuple[int, int]]:
        sx = 0 if abs(raw_dx) < (grid_mm * 0.5) else (1 if raw_dx > 0 else -1)
        sy = 0 if abs(raw_dy) < (grid_mm * 0.5) else (1 if raw_dy > 0 else -1)
        if sx == 0 and sy == 0:
            return []
        multipliers = (2, 4, 6) if include_far_steps else (1, 2, 3)
        prefer_x = abs(raw_dx) >= abs(raw_dy)
        out: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for mult in multipliers:
            step = mult * grid_mm
            candidates: list[tuple[int, int]] = []
            if sx and sy:
                candidates.append((sx * step, sy * step))
            if prefer_x:
                if sx:
                    candidates.append((sx * step, 0))
                if sy:
                    candidates.append((0, sy * step))
            else:
                if sy:
                    candidates.append((0, sy * step))
                if sx:
                    candidates.append((sx * step, 0))
            for offset in candidates:
                if offset not in seen:
                    seen.add(offset)
                    out.append(offset)
        return out

    def _targeted_macro_offsets(cluster_id: str) -> list[tuple[tuple[int, int], str]]:
        cluster = cluster_state_by_id.get(cluster_id)
        if not isinstance(cluster, dict):
            return []
        profile = cluster_preferences.get(
            cluster_id,
            {
                "prefer": set(),
                "avoid": set(),
                "intents": set(),
                "viewing_partner": None,
                "viewing_role": None,
            },
        )
        prefer_tags = set(profile.get("prefer") or set())
        avoid_tags = set(profile.get("avoid") or set())
        intents = set(profile.get("intents") or set())
        viewing_partner_id = profile.get("viewing_partner")
        viewing_role = str(profile.get("viewing_role") or "").strip().lower()
        center = (
            float((cluster.get("world_center") or {}).get("x") or 0.0),
            float((cluster.get("world_center") or {}).get("y") or 0.0),
        )
        targets: list[tuple[tuple[float, float], str]] = []
        room_bounds = room_poly.bounds
        max_distance = (
            max(
                float(room_bounds[2] - room_bounds[0]),
                float(room_bounds[3] - room_bounds[1]),
            )
            * 2.0
        )

        if isinstance(viewing_partner_id, str) and viewing_partner_id:
            partner_cluster = cluster_state_by_id.get(viewing_partner_id)
            if isinstance(partner_cluster, dict):
                partner_center = (
                    float((partner_cluster.get("world_center") or {}).get("x") or 0.0),
                    float((partner_cluster.get("world_center") or {}).get("y") or 0.0),
                )
                away_vector = (
                    center[0] - partner_center[0],
                    center[1] - partner_center[1],
                )
                if normalize_vec(away_vector) is not None:
                    targets.append((away_vector, "away from the viewing partner"))
                    clearance = _directional_clearance_mm(
                        center,
                        away_vector,
                        room_poly,
                        [],
                        max_distance=max_distance,
                    )
                    if clearance > float(grid_mm):
                        unit = normalize_vec(away_vector)
                        if unit is not None:
                            targets.append(
                                (
                                    (
                                        unit[0] * clearance,
                                        unit[1] * clearance,
                                    ),
                                    "toward the opposite viewing wall",
                                )
                            )

        suppress_window_side = viewing_role == "secondary" and isinstance(
            viewing_partner_id, str
        )
        if not suppress_window_side and (
            "window_side" in prefer_tags
            or intents
            & {
                "face_window",
                "front_to_window",
                "axis_parallel_window",
            }
        ):
            midpoint = _nearest_midpoint(center, openings.get("windows") or [])
            if midpoint is not None:
                targets.append(
                    (
                        (
                            float(midpoint[0]) - center[0],
                            float(midpoint[1]) - center[1],
                        ),
                        "toward window-side zone",
                    )
                )
        if "entry_side" in prefer_tags or intents & {"face_entry", "front_to_entry"}:
            midpoint = _nearest_midpoint(center, openings.get("doors") or [])
            if midpoint is not None:
                targets.append(
                    (
                        (
                            float(midpoint[0]) - center[0],
                            float(midpoint[1]) - center[1],
                        ),
                        "toward entry-side zone",
                    )
                )
        if "far_from_entry" in prefer_tags or avoid_tags & {
            "door_swing",
            "entry_blocking",
        }:
            midpoint = _nearest_midpoint(center, openings.get("doors") or [])
            if midpoint is not None:
                targets.append(
                    (
                        (
                            center[0] - float(midpoint[0]),
                            center[1] - float(midpoint[1]),
                        ),
                        "away from the entry zone",
                    )
                )
        if prefer_tags & {"recess_or_edge", "long_wall"} or intents & {
            "back_to_wall",
            "access_to_open_space",
        }:
            edge_point = _nearest_boundary_point(center, room_poly)
            if edge_point is not None:
                targets.append(
                    (
                        (
                            float(edge_point[0]) - center[0],
                            float(edge_point[1]) - center[1],
                        ),
                        "toward the room edge",
                    )
                )
        if "long_wall" in prefer_tags:
            longest = _longest_wall_segment(room_poly)
            if longest:
                try:
                    wall_point, _ = nearest_points(longest["line"], Point(center))
                except Exception:
                    wall_point = None
                if wall_point is not None:
                    targets.append(
                        (
                            (
                                float(wall_point.x) - center[0],
                                float(wall_point.y) - center[1],
                            ),
                            "toward the longest wall",
                        )
                    )

        out: list[tuple[tuple[int, int], str]] = []
        seen: set[tuple[int, int, str]] = set()
        for vector, reason in targets:
            for offset in _offsets_from_vector(
                float(vector[0]), float(vector[1]), include_far_steps=True
            ):
                signature = (offset[0], offset[1], reason)
                if signature in seen or offset == (0, 0):
                    continue
                seen.add(signature)
                out.append((offset, reason))
        return out

    def _ordered_viewing_pairs() -> list[tuple[str, str]]:
        ordered: list[tuple[str, str]] = []
        for cluster_a_id, cluster_b_id in _viewing_cluster_pairs(payload):
            profile_a = cluster_preferences.get(cluster_a_id, {})
            profile_b = cluster_preferences.get(cluster_b_id, {})
            role_a = str(profile_a.get("viewing_role") or "").strip().lower()
            role_b = str(profile_b.get("viewing_role") or "").strip().lower()
            if role_b == "primary" and role_a != "primary":
                pair = (cluster_b_id, cluster_a_id)
            else:
                pair = (cluster_a_id, cluster_b_id)
            if pair not in ordered:
                ordered.append(pair)
        return ordered

    def _pair_priority_offsets(
        cluster_id: str, *, primary: bool
    ) -> list[tuple[tuple[int, int], str]]:
        offsets = _targeted_macro_offsets(cluster_id)
        if not offsets:
            return []
        preferred_reasons = (
            (
                "opposite viewing wall",
                "longest wall",
                "room edge",
                "away from the entry zone",
            )
            if primary
            else (
                "away from the viewing partner",
                "away from the entry zone",
                "room edge",
            )
        )
        filtered = [
            item
            for item in offsets
            if any(reason in item[1] for reason in preferred_reasons)
        ]
        if filtered:
            return filtered[:2]
        return offsets[:2]

    viewing_pairs = _ordered_viewing_pairs()
    for primary_cluster_id, secondary_cluster_id in viewing_pairs:
        for cluster_id in (primary_cluster_id, secondary_cluster_id):
            if cluster_id not in priority_cluster_ids:
                priority_cluster_ids.append(cluster_id)

    for primary_cluster_id, secondary_cluster_id in viewing_pairs:
        primary_transform = tmap.get(primary_cluster_id)
        secondary_transform = tmap.get(secondary_cluster_id)
        if not isinstance(primary_transform, dict) or not isinstance(
            secondary_transform, dict
        ):
            continue
        primary_offsets = _pair_priority_offsets(primary_cluster_id, primary=True)
        secondary_offsets = _pair_priority_offsets(secondary_cluster_id, primary=False)
        if not primary_offsets or not secondary_offsets:
            continue

        for (primary_dx, primary_dy), primary_reason in primary_offsets:
            for (secondary_dx, secondary_dy), secondary_reason in secondary_offsets:
                macro_moves.append(
                    {
                        "kind": "cluster_pose",
                        "cluster_id": primary_cluster_id,
                        "proposal": {
                            "cluster_transforms": [
                                {
                                    **deepcopy(primary_transform),
                                    "x": int(primary_transform.get("x") or 0)
                                    + primary_dx,
                                    "y": int(primary_transform.get("y") or 0)
                                    + primary_dy,
                                },
                                {
                                    **deepcopy(secondary_transform),
                                    "x": int(secondary_transform.get("x") or 0)
                                    + secondary_dx,
                                    "y": int(secondary_transform.get("y") or 0)
                                    + secondary_dy,
                                },
                            ],
                            "selected_variants": [],
                            "object_repairs": [],
                        },
                        "reason": (
                            f"Realign viewing pair {primary_cluster_id} and "
                            f"{secondary_cluster_id}: move focal cluster "
                            f"{primary_reason} and viewing partner {secondary_reason}"
                        ),
                    }
                )

    # Macro-first: variants, cluster rotation, then small translations.
    for cid in priority_cluster_ids:
        tf = tmap.get(cid) or {"cluster_id": cid, "x": 0, "y": 0, "rot": 0}
        current_variant = str((vmap.get(cid) or {}).get("variant_id") or "")

        for variant_id in allowed_variants.get(cid, []):
            if variant_id == current_variant:
                continue
            macro_moves.append(
                {
                    "kind": "cluster_variant",
                    "cluster_id": cid,
                    "proposal": {
                        "cluster_transforms": [],
                        "selected_variants": [
                            {"cluster_id": cid, "variant_id": variant_id}
                        ],
                        "object_repairs": [],
                    },
                    "reason": f"Try variant switch for failing cluster {cid}",
                }
            )

        if cid not in protected_cluster_ids:
            for rot in (0, 90, 180, 270):
                if rot == int(tf.get("rot") or 0):
                    continue
                macro_moves.append(
                    {
                        "kind": "cluster_pose",
                        "cluster_id": cid,
                        "proposal": {
                            "cluster_transforms": [{**deepcopy(tf), "rot": rot}],
                            "selected_variants": [],
                            "object_repairs": [],
                        },
                        "reason": f"Try cluster rotation for failing cluster {cid}",
                    }
                )
        for (dx, dy), move_reason in _targeted_macro_offsets(cid):
            macro_moves.append(
                {
                    "kind": "cluster_pose",
                    "cluster_id": cid,
                    "proposal": {
                        "cluster_transforms": [
                            {
                                **deepcopy(tf),
                                "x": int(tf.get("x") or 0) + dx,
                                "y": int(tf.get("y") or 0) + dy,
                            }
                        ],
                        "selected_variants": [],
                        "object_repairs": [],
                    },
                    "reason": f"Move {cid} {move_reason}",
                }
            )
        for dx, dy in (
            (grid_mm, 0),
            (-grid_mm, 0),
            (0, grid_mm),
            (0, -grid_mm),
        ):
            macro_moves.append(
                {
                    "kind": "cluster_pose",
                    "cluster_id": cid,
                    "proposal": {
                        "cluster_transforms": [
                            {
                                **deepcopy(tf),
                                "x": int(tf.get("x") or 0) + dx,
                                "y": int(tf.get("y") or 0) + dy,
                            }
                        ],
                        "selected_variants": [],
                        "object_repairs": [],
                    },
                    "reason": f"Try a small grid translation for failing cluster {cid}",
                }
            )

    # Object-level only after macro moves.
    front_access_intents = {
        "preserve_front_access",
        "front_to_open_space",
        "face_window",
        "front_to_window",
        "face_entry",
        "front_to_entry",
        "back_to_wall",
    }
    facing_intents = {"face_object", "face_away_from_object"}
    support_axis_intents = {
        "same_direction_as_anchor",
        "same_view_side_as_primary_pair",
        "align_with_anchor_axis",
        "not_behind_anchor_view",
        "in_front_of_anchor",
    }
    relation_plan = (payload.get("goals") or {}).get("relation_plan_used") or {}
    relation_object_intents: dict[tuple[str, str], set[str]] = {}
    relation_object_targets: dict[tuple[str, str], dict[str, str]] = {}
    spacing_rows_by_object: dict[tuple[str, str], list[dict[str, Any]]] = {}
    ordered_object_keys: list[tuple[str, str]] = []
    object_move_entries: list[dict[str, Any]] = []
    seen_object_move_signatures: set[str] = set()

    def _register_object_key(
        cluster_id: str, object_id: str, intents: set[str] | None = None
    ) -> None:
        key = (cluster_id, object_id)
        if key not in ordered_object_keys:
            ordered_object_keys.append(key)
        if intents:
            relation_object_intents.setdefault(key, set()).update(intents)

    def _append_object_move(
        *,
        cluster_id: str,
        object_id: str | None,
        proposal: dict[str, Any],
        reason: str,
        planner_priority: bool,
        priority_index: int,
    ) -> None:
        repairs = deepcopy((proposal.get("object_repairs") or []))
        if not repairs:
            return
        move_signature = json.dumps(repairs, sort_keys=True, ensure_ascii=True)
        if move_signature in seen_object_move_signatures:
            return
        seen_object_move_signatures.add(move_signature)
        involved_clusters = sorted(
            {
                str(item.get("cluster_id") or "")
                for item in repairs
                if isinstance(item, dict) and str(item.get("cluster_id") or "").strip()
            }
        )
        move = {
            "kind": "object_pose",
            "cluster_id": cluster_id,
            "proposal": {
                "cluster_transforms": [],
                "selected_variants": [],
                "object_repairs": repairs,
            },
            "reason": reason,
        }
        if object_id is not None:
            move["object_id"] = object_id
        if len(involved_clusters) > 1:
            move["secondary_cluster_ids"] = involved_clusters[1:]
        object_move_entries.append(
            {
                "move": move,
                "planner_priority": planner_priority,
                "priority_index": priority_index,
                "repair_count": len(repairs),
                "cluster_count": len(involved_clusters),
            }
        )

    def _resolve_relation_target_object(
        *,
        cluster_id: str,
        object_id: str,
        target_object_id: str,
        target_object_cluster_id: str | None,
    ) -> dict[str, Any] | None:
        if not target_object_id:
            return None
        preferred_cluster_id = (
            target_object_cluster_id.strip()
            if isinstance(target_object_cluster_id, str)
            and target_object_cluster_id.strip()
            else None
        )
        fallback_match: dict[str, Any] | None = None
        for row in state_objects:
            if str(row.get("object_id") or "") != target_object_id:
                continue
            row_cluster_id = str(row.get("cluster_id") or "")
            if row_cluster_id == cluster_id and target_object_id == object_id:
                continue
            if preferred_cluster_id is not None:
                if row_cluster_id == preferred_cluster_id:
                    return row
                continue
            if fallback_match is None:
                fallback_match = row
        return fallback_match

    def _world_vector_to_local(
        cluster_rot: int, vector: tuple[float, float] | None
    ) -> tuple[float, float] | None:
        if vector is None:
            return None
        return normalize_vec(rotate_vec_ccw_90s(vector, (-int(cluster_rot or 0)) % 360))

    def _cluster_anchor_state_for_enum(cluster_id: str) -> dict[str, Any] | None:
        return _cluster_anchor_state(
            cluster_id=cluster_id,
            state_objects=state_objects,
            cluster_state_by_id=cluster_state_by_id,
            anchor_hints_by_cluster=anchor_hints_by_cluster,
        )

    def _reference_object_for_enum(
        *,
        cluster_id: str,
        object_id: str,
        relation_target: dict[str, str] | None,
    ) -> dict[str, Any] | None:
        relation_target = relation_target or {}
        target_object_id = str(relation_target.get("target_object_id") or "").strip()
        target_object_cluster_id = (
            str(relation_target.get("target_object_cluster_id") or "").strip() or None
        )
        target_object = _resolve_relation_target_object(
            cluster_id=cluster_id,
            object_id=object_id,
            target_object_id=target_object_id,
            target_object_cluster_id=target_object_cluster_id,
        )
        if isinstance(target_object, dict):
            return target_object
        hint = anchor_hints_by_cluster.get(cluster_id) or {}
        hinted_target_cluster_id = str(hint.get("target_cluster_id") or "").strip()
        if hinted_target_cluster_id and hinted_target_cluster_id != cluster_id:
            ref = _cluster_anchor_state_for_enum(hinted_target_cluster_id)
            if isinstance(ref, dict):
                return ref
        ref = _cluster_anchor_state_for_enum(cluster_id)
        if isinstance(ref, dict) and str(ref.get("object_id") or "") != object_id:
            return ref
        return None

    def _targeted_object_nudge_offsets(
        cluster_id: str, object_id: str, intents: set[str]
    ) -> list[tuple[int, int]]:
        obj = object_state_by_key.get((cluster_id, object_id))
        if not isinstance(obj, dict):
            return []
        cluster_rot = int(obj.get("cluster_rot") or obj.get("rotation_ccw") or 0)
        relation_target = relation_object_targets.get((cluster_id, object_id)) or {}
        reference_object = _reference_object_for_enum(
            cluster_id=cluster_id,
            object_id=object_id,
            relation_target=relation_target,
        )
        if not isinstance(reference_object, dict):
            return []
        reference_front = parse_vec2(reference_object.get("front_world"))
        if reference_front is None:
            return []
        obj_center = (
            float((obj.get("world_center") or {}).get("x") or 0.0),
            float((obj.get("world_center") or {}).get("y") or 0.0),
        )
        ref_center = (
            float((reference_object.get("world_center") or {}).get("x") or 0.0),
            float((reference_object.get("world_center") or {}).get("y") or 0.0),
        )
        world_vectors: list[tuple[float, float] | None] = []
        if intents & {"not_behind_anchor_view", "in_front_of_anchor"}:
            rel_vec = _vec_from_to(ref_center, obj_center)
            raw_dot = _dot(reference_front, rel_vec)
            signed = (
                0.0
                if raw_dot is None
                else raw_dot * _distance_between_points(ref_center, obj_center)
            )
            threshold = 120.0 if "in_front_of_anchor" in intents else 0.0
            if signed < threshold:
                world_vectors.append(reference_front)
        seen: set[tuple[int, int]] = set()
        out: list[tuple[int, int]] = []
        for vector in world_vectors:
            local_dir = _world_vector_to_local(cluster_rot, vector)
            if local_dir is None:
                continue
            for dx, dy in _offsets_from_vector(
                local_dir[0], local_dir[1], include_far_steps=False
            ):
                if (dx, dy) == (0, 0) or (dx, dy) in seen:
                    continue
                seen.add((dx, dy))
                out.append((dx, dy))
        return out[:4]

    def _front_override_dirs(
        cluster_id: str, object_id: str, intents: set[str]
    ) -> list[tuple[float, float]]:
        obj = object_state_by_key.get((cluster_id, object_id))
        if not isinstance(obj, dict):
            return []
        center = (
            float((obj.get("world_center") or {}).get("x") or 0.0),
            float((obj.get("world_center") or {}).get("y") or 0.0),
        )
        cluster_rot = int(obj.get("cluster_rot") or obj.get("rotation_ccw") or 0)
        blockers = _scoring_object_blockers(
            state, object_index_by_key.get((cluster_id, object_id), -1)
        )
        world_targets: list[tuple[float, float] | None] = []

        if intents & {"face_window", "front_to_window"}:
            midpoint = _nearest_midpoint(center, openings.get("windows") or [])
            if midpoint is not None:
                world_targets.append(_vec_from_to(center, midpoint))
        if intents & {"face_entry", "front_to_entry"}:
            midpoint = _nearest_midpoint(center, openings.get("doors") or [])
            if midpoint is not None:
                world_targets.append(_vec_from_to(center, midpoint))
        if intents & {"front_to_open_space", "preserve_front_access"}:
            best_dir, _ = _best_open_dir(center, room_center, room_poly, blockers)
            world_targets.append(best_dir)
        if "back_to_wall" in intents:
            world_targets.append(_nearest_wall_inward_dir(center, room_poly))
        relation_target = relation_object_targets.get((cluster_id, object_id)) or {}
        if intents & {"face_object", "face_away_from_object"}:
            target = _resolve_relation_target_object(
                cluster_id=cluster_id,
                object_id=object_id,
                target_object_id=str(relation_target.get("target_object_id") or ""),
                target_object_cluster_id=(
                    str(relation_target.get("target_object_cluster_id") or "")
                    if relation_target.get("target_object_cluster_id")
                    else None
                ),
            )
            if isinstance(target, dict):
                target_center = (
                    float((target.get("world_center") or {}).get("x") or 0.0),
                    float((target.get("world_center") or {}).get("y") or 0.0),
                )
                target_vector = _vec_from_to(center, target_center)
                if "face_away_from_object" in intents and target_vector is not None:
                    world_targets.append((-target_vector[0], -target_vector[1]))
                else:
                    world_targets.append(target_vector)

        if intents & {
            "same_direction_as_anchor",
            "same_view_side_as_primary_pair",
            "align_with_anchor_axis",
            "not_behind_anchor_view",
            "in_front_of_anchor",
        }:
            reference_object = _reference_object_for_enum(
                cluster_id=cluster_id,
                object_id=object_id,
                relation_target=relation_target,
            )
            if isinstance(reference_object, dict):
                reference_front = parse_vec2(reference_object.get("front_world"))
                reference_axis = (
                    parse_vec2(reference_object.get("axis_world")) or reference_front
                )
                if intents & {
                    "same_direction_as_anchor",
                    "same_view_side_as_primary_pair",
                    "not_behind_anchor_view",
                    "in_front_of_anchor",
                }:
                    world_targets.append(reference_front)
                if "align_with_anchor_axis" in intents:
                    world_targets.append(reference_axis)

        out: list[tuple[float, float]] = []
        seen_dirs: set[tuple[float, float]] = set()
        for vector in world_targets:
            local_dir = _world_vector_to_local(cluster_rot, vector)
            if local_dir is None:
                continue
            signature = (round(local_dir[0], 3), round(local_dir[1], 3))
            if signature in seen_dirs:
                continue
            seen_dirs.add(signature)
            out.append(local_dir)
        return out[:3]

    for row in relation_plan.get("object_orientations") or []:
        if not isinstance(row, dict):
            continue
        cid = str(row.get("cluster_id") or "").strip()
        oid = str(row.get("object_id") or "").strip()
        if not cid or not oid:
            continue
        intents = {
            str(intent).strip().lower()
            for intent in (row.get("intents") or [])
            if str(intent).strip()
        }
        if intents & (front_access_intents | facing_intents | support_axis_intents):
            _register_object_key(cid, oid, intents)
            relation_object_targets[(cid, oid)] = {
                "target_object_id": str(row.get("target_object_id") or "").strip(),
                "target_object_cluster_id": str(
                    row.get("target_object_cluster_id") or ""
                ).strip(),
            }

    for row in score.get("global_layout_debug") or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("kind") or "") != "cluster_internal_constraint":
            continue
        if str(row.get("constraint_type") or "") != "semantic_proximity":
            continue
        subjects = row.get("subjects") or {}
        cid = str(row.get("cluster_id") or "").strip()
        oid = str(subjects.get("a") or "").strip()
        if not cid or not oid:
            continue
        spacing_rows_by_object.setdefault((cid, oid), []).append(deepcopy(row))
        _register_object_key(cid, oid)

    for row in score["prioritized_objects"][:6]:
        cid = str(row.get("cluster_id") or "").strip()
        oid = str(row.get("object_id") or "").strip()
        if not cid or not oid:
            continue
        _register_object_key(cid, oid)

    same_cluster_swap_pairs: list[tuple[str, str, str]] = []
    for priority_index, (cid, oid) in enumerate(ordered_object_keys):
        dominant_rows = [
            entry
            for entry in score.get("orientation_debug") or []
            if isinstance(entry, dict)
            and str(entry.get("cluster_id") or "") == cid
            and str(entry.get("object_id") or "") == oid
        ]
        dominant_intents = {
            str(entry.get("intent") or "").strip().lower()
            for entry in dominant_rows
            if int(entry.get("penalty_mm") or 0) >= 200
        }
        spacing_rows = spacing_rows_by_object.get((cid, oid), [])
        effective_intents = dominant_intents | relation_object_intents.get(
            (cid, oid), set()
        )
        planner_priority = bool(relation_object_intents.get((cid, oid)))
        protected_anchor_object = oid in protected_anchor_ids_by_cluster.get(cid, set())
        if viewing_orientation_pressure and not (
            effective_intents
            & (front_access_intents | facing_intents | support_axis_intents)
            or spacing_rows
        ):
            continue

        if effective_intents & (
            front_access_intents | facing_intents | support_axis_intents
        ):
            if not protected_anchor_object:
                for rot in (90, 180, 270):
                    _append_object_move(
                        cluster_id=cid,
                        object_id=oid,
                        proposal={
                            "object_repairs": [
                                {
                                    "cluster_id": cid,
                                    "object_id": oid,
                                    "op": "rotate_object",
                                    "params": {"rot": rot},
                                }
                            ]
                        },
                        reason=f"Try rotating {cid}.{oid}",
                        planner_priority=planner_priority,
                        priority_index=priority_index,
                    )
                for axis in ("x", "y"):
                    _append_object_move(
                        cluster_id=cid,
                        object_id=oid,
                        proposal={
                            "object_repairs": [
                                {
                                    "cluster_id": cid,
                                    "object_id": oid,
                                    "op": "mirror_object",
                                    "params": {"axis": axis},
                                }
                            ]
                        },
                        reason=f"Try mirroring {cid}.{oid} across the local {axis} axis",
                        planner_priority=planner_priority,
                        priority_index=priority_index,
                    )
                for local_dir in _front_override_dirs(cid, oid, effective_intents):
                    _append_object_move(
                        cluster_id=cid,
                        object_id=oid,
                        proposal={
                            "object_repairs": [
                                {
                                    "cluster_id": cid,
                                    "object_id": oid,
                                    "op": "set_front_override",
                                    "params": {"dx": local_dir[0], "dy": local_dir[1]},
                                }
                            ]
                        },
                        reason=f"Try a planner-driven front override for {cid}.{oid}",
                        planner_priority=True,
                        priority_index=priority_index,
                    )
        for spacing_row in spacing_rows[:2]:
            proximity = str(spacing_row.get("proximity") or "balanced")
            mapped_side = str(spacing_row.get("mapped_side") or "")
            gap_mm = int(spacing_row.get("gap_mm") or 0)
            target_gap_mm = int(spacing_row.get("target_gap_mm") or 0)
            for dx, dy in _spacing_offsets_for_debug_row(spacing_row, grid_mm=grid_mm):
                _append_object_move(
                    cluster_id=cid,
                    object_id=oid,
                    proposal={
                        "object_repairs": [
                            {
                                "cluster_id": cid,
                                "object_id": oid,
                                "op": "nudge_object",
                                "params": {"dx": dx, "dy": dy},
                            }
                        ]
                    },
                    reason=(
                        f"Adjust {cid}.{oid} {proximity} spacing on the {mapped_side} side "
                        f"(current gap {gap_mm}mm, target {target_gap_mm}mm)"
                    ),
                    planner_priority=False,
                    priority_index=priority_index,
                )
        allow_nudges = (
            bool(effective_intents & (front_access_intents | support_axis_intents))
            or bool(spacing_rows)
            or (not viewing_orientation_pressure and not dominant_intents)
        )
        if allow_nudges:
            for dx, dy in _targeted_object_nudge_offsets(cid, oid, effective_intents):
                _append_object_move(
                    cluster_id=cid,
                    object_id=oid,
                    proposal={
                        "object_repairs": [
                            {
                                "cluster_id": cid,
                                "object_id": oid,
                                "op": "nudge_object",
                                "params": {"dx": dx, "dy": dy},
                            }
                        ]
                    },
                    reason=f"Try an anchor-relative nudge for {cid}.{oid}",
                    planner_priority=True,
                    priority_index=priority_index,
                )
            for dx, dy in (
                (GLOBAL_LAYOUT_GRID_MM, 0),
                (-GLOBAL_LAYOUT_GRID_MM, 0),
                (0, GLOBAL_LAYOUT_GRID_MM),
                (0, -GLOBAL_LAYOUT_GRID_MM),
                (GLOBAL_LAYOUT_GRID_MM, GLOBAL_LAYOUT_GRID_MM),
                (GLOBAL_LAYOUT_GRID_MM, -GLOBAL_LAYOUT_GRID_MM),
                (-GLOBAL_LAYOUT_GRID_MM, GLOBAL_LAYOUT_GRID_MM),
                (-GLOBAL_LAYOUT_GRID_MM, -GLOBAL_LAYOUT_GRID_MM),
            ):
                _append_object_move(
                    cluster_id=cid,
                    object_id=oid,
                    proposal={
                        "object_repairs": [
                            {
                                "cluster_id": cid,
                                "object_id": oid,
                                "op": "nudge_object",
                                "params": {"dx": dx, "dy": dy},
                            }
                        ]
                    },
                    reason=f"Try nudging {cid}.{oid}",
                    planner_priority=planner_priority,
                    priority_index=priority_index,
                )
        for other_cid, other_oid in ordered_object_keys[priority_index + 1 :]:
            if other_cid != cid:
                continue
            if other_oid == oid:
                continue
            same_cluster_swap_pairs.append((cid, oid, other_oid))
            break

    for priority_index, (cid, oid, other_oid) in enumerate(same_cluster_swap_pairs[:4]):
        if oid in protected_anchor_ids_by_cluster.get(
            cid, set()
        ) or other_oid in protected_anchor_ids_by_cluster.get(cid, set()):
            continue
        planner_priority = bool(relation_object_intents.get((cid, oid))) or bool(
            relation_object_intents.get((cid, other_oid))
        )
        _append_object_move(
            cluster_id=cid,
            object_id=oid,
            proposal={
                "object_repairs": [
                    {
                        "cluster_id": cid,
                        "object_id": oid,
                        "op": "swap_objects",
                        "params": {"other_object_id": other_oid},
                    }
                ]
            },
            reason=f"Try swapping {cid}.{oid} with {cid}.{other_oid}",
            planner_priority=planner_priority,
            priority_index=priority_index,
        )

    single_object_moves = [
        entry for entry in object_move_entries if entry["repair_count"] == 1
    ]
    pair_budget = 12
    for left_index, left in enumerate(single_object_moves[:18]):
        left_move = left["move"]
        left_repairs = (left_move.get("proposal") or {}).get("object_repairs") or []
        if len(left_repairs) != 1 or not isinstance(left_repairs[0], dict):
            continue
        left_repair = left_repairs[0]
        left_cluster_id = str(left_repair.get("cluster_id") or "")
        left_object_id = str(left_repair.get("object_id") or "")
        for right in single_object_moves[left_index + 1 : 18]:
            right_move = right["move"]
            right_repairs = (right_move.get("proposal") or {}).get(
                "object_repairs"
            ) or []
            if len(right_repairs) != 1 or not isinstance(right_repairs[0], dict):
                continue
            right_repair = right_repairs[0]
            right_cluster_id = str(right_repair.get("cluster_id") or "")
            right_object_id = str(right_repair.get("object_id") or "")
            if not left_cluster_id or not right_cluster_id:
                continue
            if left_cluster_id == right_cluster_id:
                continue
            ordered_repairs = sorted(
                [deepcopy(left_repair), deepcopy(right_repair)],
                key=lambda row: (
                    str(row.get("cluster_id") or ""),
                    str(row.get("object_id") or ""),
                    str(row.get("op") or ""),
                    json.dumps(
                        row.get("params") or {}, sort_keys=True, ensure_ascii=True
                    ),
                ),
            )
            _append_object_move(
                cluster_id=left_cluster_id,
                object_id=left_object_id,
                proposal={"object_repairs": ordered_repairs},
                reason=(
                    "Try synchronized planner-first object refinement for "
                    f"{left_cluster_id}.{left_object_id} and "
                    f"{right_cluster_id}.{right_object_id}"
                ),
                planner_priority=bool(
                    left["planner_priority"] or right["planner_priority"]
                ),
                priority_index=min(
                    int(left["priority_index"]), int(right["priority_index"])
                ),
            )
            pair_budget -= 1
            if pair_budget <= 0:
                break
        if pair_budget <= 0:
            break

    object_move_entries.sort(
        key=lambda entry: (
            -int(bool(entry["planner_priority"])),
            -int(entry["cluster_count"]),
            -int(entry["repair_count"]),
            int(entry["priority_index"]),
            str((entry["move"].get("proposal") or {}).get("object_repairs") or ""),
        )
    )
    object_moves = [deepcopy(entry["move"]) for entry in object_move_entries]

    moves = object_moves if search_phase == "object_refine" else macro_moves

    return {
        "result": "OK",
        "search_phase": search_phase,
        "moves": moves[:limit],
        "baseline_score": score["score"],
    }


def EvaluatePhase2Proposal(
    *, payload: Dict[str, Any], repair: Dict[str, Any], move_limit: int = 16
) -> Dict[str, Any]:
    baseline_state = materialize_phase2_state(payload, repair=None)
    baseline_score = score_phase2_state(payload, baseline_state)

    candidate_state = materialize_phase2_state(payload, repair=repair)
    candidate_score = score_phase2_state(payload, candidate_state)
    errors = deepcopy(candidate_score.get("hard_violations") or [])
    metrics = _build_metrics(payload, candidate_state, candidate_score)

    next_payload = PromotePhase2RepairToSeedPayload(payload=payload, repair=repair)
    next_moves = []
    if bool(candidate_score.get("hard_valid")):
        next_moves = (
            EnumeratePhase2RepairMoves(payload=next_payload, limit=move_limit).get(
                "moves"
            )
            or []
        )

    diagnosis = {
        "prioritized_clusters": deepcopy(
            candidate_score.get("prioritized_clusters") or []
        ),
        "prioritized_objects": deepcopy(
            candidate_score.get("prioritized_objects") or []
        ),
        "key_findings": _build_key_findings(candidate_score, metrics, errors),
        "enumerated_moves": next_moves,
    }

    return {
        "result": "OK",
        "hard_valid": bool(candidate_score.get("hard_valid")),
        "errors": errors,
        "violations_by_cluster": _violations_by_cluster(errors),
        "materialized_layout": _build_materialized_layout(payload, candidate_state),
        "metrics": metrics,
        "diagnosis": diagnosis,
        "baseline_comparison": {
            "baseline_score": int(baseline_score.get("score") or 0),
            "baseline_hard_valid": bool(baseline_score.get("hard_valid")),
            "candidate_score": int(candidate_score.get("score") or 0),
            "delta_score": int(
                int(candidate_score.get("score") or 0)
                - int(baseline_score.get("score") or 0)
            ),
            "improved": bool(
                candidate_score.get("hard_valid")
                and int(candidate_score.get("score") or 0)
                > int(baseline_score.get("score") or 0)
            ),
        },
    }


def ResolvePhase2HardViolations(
    *,
    payload: Dict[str, Any],
    repair: Dict[str, Any],
    max_rounds: int = 40,
    max_radius_steps: int = 6,
) -> Dict[str, Any]:
    working_repair = _full_seed_repair(payload, repair)
    best_repair = deepcopy(working_repair)
    best_evaluation = EvaluatePhase2Proposal(payload=payload, repair=working_repair)
    if bool(best_evaluation.get("hard_valid")):
        return {
            "result": "OK",
            "repair": best_repair,
            "evaluation": best_evaluation,
            "attempts": [],
        }

    grid_mm = normalize_layout_grid_mm(
        (payload.get("room_context") or {}).get("grid_mm")
    )
    attempted_signatures = {
        json.dumps(
            {
                "cluster_transforms": best_repair.get("cluster_transforms"),
                "selected_variants": best_repair.get("selected_variants"),
                "object_repairs": best_repair.get("object_repairs"),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    }
    attempts: list[dict[str, Any]] = []

    def _repair_rank(evaluation: Dict[str, Any]) -> tuple[int, int, int]:
        return (
            1 if bool(evaluation.get("hard_valid")) else 0,
            -len(evaluation.get("errors") or []),
            int(
                ((evaluation.get("baseline_comparison") or {}).get("candidate_score"))
                or 0
            ),
        )

    def _target_clusters(errors: list[dict[str, Any]]) -> list[str]:
        ordered: list[str] = []
        for error in errors:
            if not isinstance(error, dict):
                continue
            code = str(error.get("code") or "")
            cluster_ids: list[str] = []
            if code == "OBJECT_OVERLAP":
                for key in ("a_cluster_id", "b_cluster_id"):
                    value = error.get(key)
                    if isinstance(value, str) and value:
                        cluster_ids.append(value)
            else:
                value = error.get("cluster_id")
                if isinstance(value, str) and value:
                    cluster_ids.append(value)
            for cluster_id in cluster_ids:
                if cluster_id not in ordered:
                    ordered.append(cluster_id)
        return ordered

    offsets: list[tuple[int, int]] = []
    for radius in range(1, max(1, int(max_radius_steps)) + 1):
        step = radius * grid_mm
        offsets.extend(
            [
                (step, 0),
                (-step, 0),
                (0, step),
                (0, -step),
                (step, step),
                (step, -step),
                (-step, step),
                (-step, -step),
            ]
        )

    for round_idx in range(1, max(1, int(max_rounds)) + 1):
        candidate_clusters = _target_clusters(best_evaluation.get("errors") or [])
        if not candidate_clusters:
            break

        local_best_repair: Dict[str, Any] | None = None
        local_best_evaluation: Dict[str, Any] | None = None
        local_best_cluster: str | None = None
        local_best_offset = (0, 0)

        for cluster_id in candidate_clusters:
            current_transform = next(
                (
                    row
                    for row in working_repair.get("cluster_transforms") or []
                    if isinstance(row, dict) and row.get("cluster_id") == cluster_id
                ),
                None,
            )
            if not isinstance(current_transform, dict):
                continue

            for dx, dy in offsets:
                candidate_repair = deepcopy(working_repair)
                for row in candidate_repair.get("cluster_transforms") or []:
                    if isinstance(row, dict) and row.get("cluster_id") == cluster_id:
                        row["x"] = int(row.get("x") or 0) + dx
                        row["y"] = int(row.get("y") or 0) + dy
                        break
                candidate_repair["notes"] = [
                    *[
                        str(note)
                        for note in candidate_repair.get("notes") or []
                        if str(note).strip()
                    ],
                    f"Hard-fix nudge for {cluster_id}: dx={dx}, dy={dy}.",
                ]
                signature = json.dumps(
                    {
                        "cluster_transforms": candidate_repair.get(
                            "cluster_transforms"
                        ),
                        "selected_variants": candidate_repair.get("selected_variants"),
                        "object_repairs": candidate_repair.get("object_repairs"),
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                )
                if signature in attempted_signatures:
                    continue
                attempted_signatures.add(signature)

                candidate_evaluation = EvaluatePhase2Proposal(
                    payload=payload,
                    repair=candidate_repair,
                )
                if local_best_evaluation is None or _repair_rank(
                    candidate_evaluation
                ) > _repair_rank(local_best_evaluation):
                    local_best_repair = candidate_repair
                    local_best_evaluation = candidate_evaluation
                    local_best_cluster = cluster_id
                    local_best_offset = (dx, dy)
                if bool(candidate_evaluation.get("hard_valid")):
                    attempts.append(
                        {
                            "round": round_idx,
                            "cluster_id": cluster_id,
                            "dx": dx,
                            "dy": dy,
                            "hard_valid": True,
                            "error_count": len(
                                candidate_evaluation.get("errors") or []
                            ),
                        }
                    )
                    return {
                        "result": "OK",
                        "repair": candidate_repair,
                        "evaluation": candidate_evaluation,
                        "attempts": attempts,
                    }

        if local_best_repair is None or local_best_evaluation is None:
            break

        attempts.append(
            {
                "round": round_idx,
                "cluster_id": local_best_cluster,
                "dx": local_best_offset[0],
                "dy": local_best_offset[1],
                "hard_valid": bool(local_best_evaluation.get("hard_valid")),
                "error_count": len(local_best_evaluation.get("errors") or []),
            }
        )
        if _repair_rank(local_best_evaluation) <= _repair_rank(best_evaluation):
            break

        working_repair = local_best_repair
        best_repair = local_best_repair
        best_evaluation = local_best_evaluation

    return {
        "result": "OK",
        "repair": best_repair,
        "evaluation": best_evaluation,
        "attempts": attempts,
    }


_DIRECTIONAL_OBJECT_KEYWORDS = (
    "tv_console",
    "media_shelf",
    "shelf",
    "console",
    "sofa",
    "sectional",
    "recliner",
    "armchair",
    "chair",
    "desk",
    "vanity",
    "bed",
    "wardrobe",
    "cabinet",
    "dresser",
    "nightstand",
)


def _is_directional_object_id(object_id: str) -> bool:
    text = str(object_id or "").strip().lower()
    return any(keyword in text for keyword in _DIRECTIONAL_OBJECT_KEYWORDS)


def CompileAcceptedPhase2Proposal(
    *, payload: Dict[str, Any], repair: Dict[str, Any]
) -> Dict[str, Any]:
    state = materialize_phase2_state(payload, repair=repair)
    score = score_phase2_state(payload, state)
    room_model = _room_model(payload)
    room = deepcopy(room_model.get("room") or {})
    if isinstance(room_model.get("openings"), dict):
        room["openings"] = deepcopy(room_model.get("openings") or {})
    if isinstance(room_model.get("obstacles"), list):
        room["obstacles"] = deepcopy(room_model.get("obstacles") or [])
    objects = []
    for row in state.get("objects") or []:
        object_id = str(row.get("object_id") or "")
        rotation_ccw = int(row.get("rotation_ccw") or 0)
        if _is_directional_object_id(object_id):
            semantic_rotation = rotation_from_front_world(row.get("front_world"))
            if semantic_rotation is not None:
                rotation_ccw = semantic_rotation
        front_world = deepcopy(row.get("front_world"))
        axis_world = deepcopy(row.get("axis_world"))
        front_side_world = None
        if front_world is not None:
            front_side_world = vec_to_side(
                (
                    float(front_world.get("dx", 0)),
                    float(front_world.get("dy", 0)),
                )
                if isinstance(front_world, dict)
                else None
            )
        objects.append(
            {
                "object_id": object_id,
                "cluster_id": row["cluster_id"],
                "polygon_ccw": deepcopy(row["polygon_ccw"]),
                "bbox": deepcopy(row["bbox"]),
                "rotation_ccw": rotation_ccw,
                "source_rect": deepcopy(row.get("local_rect") or {}),
                "front_world": front_world,
                "front_side_world": front_side_world,
                "axis_world": axis_world,
            }
        )
    return {
        "status": "OK" if score["hard_valid"] else "INVALID",
        "room": room,
        "objects": objects,
        "notes": deepcopy(repair.get("notes") or []),
        "missing": deepcopy(score.get("hard_violations") or []),
    }


def ScorePhase2Repair(
    *, payload: Dict[str, Any], repair: Dict[str, Any]
) -> Dict[str, Any]:
    evaluation = EvaluatePhase2Proposal(payload=payload, repair=repair)
    baseline = evaluation.get("baseline_comparison") or {}
    candidate = evaluation.get("diagnosis") or {}
    materialized = evaluation.get("materialized_layout") or {}
    return {
        "result": "OK",
        "baseline": {
            "score": int(baseline.get("baseline_score") or 0),
            "hard_valid": bool(baseline.get("baseline_hard_valid")),
        },
        "candidate": {
            "score": int(baseline.get("candidate_score") or 0),
            "hard_valid": bool(evaluation.get("hard_valid")),
            "hard_violations": deepcopy(evaluation.get("errors") or []),
            "prioritized_clusters": deepcopy(
                candidate.get("prioritized_clusters") or []
            ),
            "prioritized_objects": deepcopy(candidate.get("prioritized_objects") or []),
            "orientation_debug": deepcopy(
                (evaluation.get("metrics") or {}).get("orientation_debug") or []
            ),
            "materialized_layout": materialized,
        },
        "delta_score": int(baseline.get("delta_score") or 0),
        "improved": bool(baseline.get("improved")),
    }


def PreviewPhase2Repair(
    *, payload: Dict[str, Any], repair: Dict[str, Any]
) -> Dict[str, Any]:
    compiled = CompileAcceptedPhase2Proposal(payload=payload, repair=repair)
    evaluation = EvaluatePhase2Proposal(payload=payload, repair=repair)
    return {
        "result": "OK",
        "compiled": compiled,
        "score": {
            "hard_valid": bool(evaluation.get("hard_valid")),
            "hard_violations": deepcopy(evaluation.get("errors") or []),
            "orientation_debug": deepcopy(
                (evaluation.get("metrics") or {}).get("orientation_debug") or []
            ),
            "prioritized_clusters": deepcopy(
                (evaluation.get("diagnosis") or {}).get("prioritized_clusters") or []
            ),
            "prioritized_objects": deepcopy(
                (evaluation.get("diagnosis") or {}).get("prioritized_objects") or []
            ),
            "score": int(
                ((evaluation.get("baseline_comparison") or {}).get("candidate_score"))
                or 0
            ),
        },
    }


TOOL_REGISTRY: Dict[str, Any] = {
    "DiagnosePhase2Seed": DiagnosePhase2Seed,
    "EnumeratePhase2RepairMoves": EnumeratePhase2RepairMoves,
    "ScorePhase2Repair": ScorePhase2Repair,
    "PreviewPhase2Repair": PreviewPhase2Repair,
    "EvaluatePhase2Proposal": EvaluatePhase2Proposal,
    "ResolvePhase2HardViolations": ResolvePhase2HardViolations,
    "CompileAcceptedPhase2Proposal": CompileAcceptedPhase2Proposal,
    "PromotePhase2RepairToSeedPayload": PromotePhase2RepairToSeedPayload,
}

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "EvaluatePhase2Proposal",
            "description": "Materialize and evaluate a phase-2 repair proposal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "payload": {"type": "object"},
                    "repair": {"type": "object"},
                    "move_limit": {"type": "integer"},
                },
                "required": ["payload", "repair"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "CompileAcceptedPhase2Proposal",
            "description": "Compile an accepted proposal into the absolute layout output shape.",
            "parameters": {
                "type": "object",
                "properties": {
                    "payload": {"type": "object"},
                    "repair": {"type": "object"},
                },
                "required": ["payload", "repair"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ResolvePhase2HardViolations",
            "description": "Apply deterministic grid nudges to reduce overlap and out-of-bounds violations on a chosen candidate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "payload": {"type": "object"},
                    "repair": {"type": "object"},
                    "max_rounds": {"type": "integer"},
                    "max_radius_steps": {"type": "integer"},
                },
                "required": ["payload", "repair"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "PromotePhase2RepairToSeedPayload",
            "description": "Promote a repair proposal into a new phase-2 seed payload for the next iteration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "payload": {"type": "object"},
                    "repair": {"type": "object"},
                },
                "required": ["payload", "repair"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "DiagnosePhase2Seed",
            "description": "Diagnose the current phase-2 seed using the unified object-pose scorer.",
            "parameters": {
                "type": "object",
                "properties": {"payload": {"type": "object"}},
                "required": ["payload"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "EnumeratePhase2RepairMoves",
            "description": "Enumerate small candidate repair moves for the current phase-2 seed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "payload": {"type": "object"},
                    "limit": {"type": "integer"},
                },
                "required": ["payload"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ScorePhase2Repair",
            "description": "Score a repair proposal using the same unified state as preview and final compile.",
            "parameters": {
                "type": "object",
                "properties": {
                    "payload": {"type": "object"},
                    "repair": {"type": "object"},
                },
                "required": ["payload", "repair"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "PreviewPhase2Repair",
            "description": "Preview the compiled final output for a repair proposal using the unified state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "payload": {"type": "object"},
                    "repair": {"type": "object"},
                },
                "required": ["payload", "repair"],
                "additionalProperties": False,
            },
        },
    },
]
