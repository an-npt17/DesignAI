from __future__ import annotations

from typing import Any, Dict, List, Tuple


def compute_cluster_outline(
    cluster_output: Dict[str, Any],
    *,
    normalize_non_negative: bool = True,
    snap_to_grid: bool = True,
    outline_mode: str = "union",  # "envelope" | "union" | "buffered_union"
    gap_fill_mm: int = 0,  # only used for buffered_union (e.g. 100~300)
    use_shapely: bool = True,
) -> Dict[str, Any]:
    """
    Reads ClusterComposer output JSON and computes:
      - cluster_footprint.local_bbox (AABB)
      - cluster_footprint.outline_polygons_ccw (based on outline_mode)

    Also normalizes negative coordinates by translating everything so that bbox min_x/min_y >= 0.

    outline_mode:
      - "envelope": staircase envelope polygon (over-approx; may fill gaps). No shapely required.
      - "union": true outline of union-of-rects. May return multiple polygons if disjoint.
      - "buffered_union": union after buffering by gap_fill_mm (connect disjoint pieces), then buffer(-gap_fill_mm).
    """
    out = dict(cluster_output)
    fp = dict(out.get("cluster_footprint") or {})
    rects = fp.get("rects") or []
    if not isinstance(rects, list):
        rects = []
    if not rects:
        rects = _rects_from_variant_bundle(out)
        if rects:
            fp["rects"] = rects
            out["cluster_footprint"] = fp

    if not rects:
        outlines = _coerce_outline_polygons(
            fp.get("outline_polygons_ccw")
            or fp.get("tight_hull_polygons_mm")
            or out.get("tight_hull_polygons_mm")
        )
        if not outlines:
            outline = _coerce_outline_polygon(
                fp.get("tight_hull_polygon_mm") or out.get("tight_hull_polygon_mm")
            )
            outlines = [outline] if outline else []
        if outlines:
            fp["local_bbox"] = _compute_local_bbox_from_polygons(outlines)
            fp["outline_polygons_ccw"] = outlines
            fp["outline_meta"] = {
                "mode": "provided_hull",
                "normalized_translation": {"dx": 0, "dy": 0},
                "gap_fill_mm": 0,
                "used_shapely": False,
            }
            out["cluster_footprint"] = fp
            out["orientation_meta"] = (
                cluster_output.get("orientation_meta")
                if isinstance(cluster_output.get("orientation_meta"), dict)
                else {}
            )
            return out

    # 1) bbox from rects
    bbox = _compute_local_bbox_from_rects(rects)
    fp["local_bbox"] = bbox
    out["cluster_footprint"] = fp

    # 2) normalize coords to non-negative if needed
    dx = 0
    dy = 0
    if normalize_non_negative:
        dx = -bbox["min_x"] if bbox["min_x"] < 0 else 0
        dy = -bbox["min_y"] if bbox["min_y"] < 0 else 0
        if dx != 0 or dy != 0:
            grid = int((out.get("local_frame") or {}).get("grid_mm") or 0)
            out = _translate_cluster(out, dx, dy, grid_mm=grid if snap_to_grid else 0)
            fp = dict(out.get("cluster_footprint") or {})
            rects = fp.get("rects") or []
            bbox = _compute_local_bbox_from_rects(rects)
            fp["local_bbox"] = bbox
            out["cluster_footprint"] = fp

    # 3) compute outline
    outlines: List[List[Dict[str, int]]] = []
    mode = outline_mode.strip().lower()

    if mode == "envelope":
        poly = _staircase_envelope_polygon_ccw(rects)
        outlines = [poly] if poly else []

    elif mode in ("union", "buffered_union"):
        if not use_shapely:
            outlines = _outline_polygons_union_grid(rects)
            mode = "union_grid_fallback"
        else:
            try:
                outlines = _outline_polygons_union_shapely(
                    rects,
                    gap_fill_mm=gap_fill_mm if mode == "buffered_union" else 0,
                )
            except Exception:
                outlines = _outline_polygons_union_grid(rects)
                mode = "union_grid_fallback"

    else:
        # unknown mode -> envelope
        poly = _staircase_envelope_polygon_ccw(rects)
        outlines = [poly] if poly else []
        mode = "envelope_fallback"

    fp = dict(out.get("cluster_footprint") or {})
    fp["outline_polygons_ccw"] = outlines
    fp["outline_meta"] = {
        "mode": mode,
        "normalized_translation": {"dx": dx, "dy": dy},
        "gap_fill_mm": int(gap_fill_mm)
        if outline_mode.strip().lower() == "buffered_union"
        else 0,
        "used_shapely": bool(use_shapely and mode in ("union", "buffered_union")),
    }
    out["cluster_footprint"] = fp
    out["orientation_meta"] = (
        cluster_output.get("orientation_meta")
        if isinstance(cluster_output.get("orientation_meta"), dict)
        else {}
    )
    return out


# ----------------------------
# Helpers: translation / bbox
# ----------------------------


def _translate_cluster(
    cluster_output: Dict[str, Any], dx: int, dy: int, *, grid_mm: int = 0
) -> Dict[str, Any]:
    out = dict(cluster_output)

    def snap(v: int) -> int:
        if grid_mm and grid_mm > 0:
            return int(round(v / grid_mm)) * grid_mm
        return v

    # translate placements
    placements = out.get("local_placements") or []
    if isinstance(placements, list):
        new_pl = []
        for p in placements:
            if not isinstance(p, dict):
                continue
            x = int(p.get("x", 0)) + dx
            y = int(p.get("y", 0)) + dy
            new_p = dict(p)
            new_p["x"] = snap(x)
            new_p["y"] = snap(y)
            new_pl.append(new_p)
        out["local_placements"] = new_pl

    # translate rects + bbox + existing outlines
    fp = dict(out.get("cluster_footprint") or {})
    rects = fp.get("rects") or []
    if isinstance(rects, list):
        new_rects = []
        for r in rects:
            if not isinstance(r, dict):
                continue
            x = int(r.get("x", 0)) + dx
            y = int(r.get("y", 0)) + dy
            new_r = dict(r)
            new_r["x"] = snap(x)
            new_r["y"] = snap(y)
            new_rects.append(new_r)
        fp["rects"] = new_rects

    bbox = fp.get("local_bbox")
    if isinstance(bbox, dict):
        fp["local_bbox"] = {
            "min_x": snap(int(bbox.get("min_x", 0)) + dx),
            "min_y": snap(int(bbox.get("min_y", 0)) + dy),
            "max_x": snap(int(bbox.get("max_x", 0)) + dx),
            "max_y": snap(int(bbox.get("max_y", 0)) + dy),
        }

    outlines = fp.get("outline_polygons_ccw")
    if isinstance(outlines, list):
        new_outlines = []
        for poly in outlines:
            if not isinstance(poly, list):
                continue
            new_poly = []
            for pt in poly:
                if not isinstance(pt, dict):
                    continue
                x = int(pt.get("x", 0)) + dx
                y = int(pt.get("y", 0)) + dy
                new_poly.append({"x": snap(x), "y": snap(y)})
            if new_poly:
                new_outlines.append(new_poly)
        fp["outline_polygons_ccw"] = new_outlines

    out["cluster_footprint"] = fp
    return out


def _compute_local_bbox_from_rects(rects: List[Dict[str, Any]]) -> Dict[str, int]:
    if not rects:
        return {"min_x": 0, "min_y": 0, "max_x": 0, "max_y": 0}

    min_x = 10**18
    min_y = 10**18
    max_x = -(10**18)
    max_y = -(10**18)

    for r in rects:
        if not isinstance(r, dict):
            continue
        x = int(r.get("x", 0))
        y = int(r.get("y", 0))
        w = int(r.get("w", 0))
        h = int(r.get("h", 0))
        if w <= 0 or h <= 0:
            continue
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)

    if min_x == 10**18:
        return {"min_x": 0, "min_y": 0, "max_x": 0, "max_y": 0}

    return {
        "min_x": int(min_x),
        "min_y": int(min_y),
        "max_x": int(max_x),
        "max_y": int(max_y),
    }


def _rects_from_variant_bundle(cluster_output: Dict[str, Any]) -> List[Dict[str, Any]]:
    variants = cluster_output.get("variant_bundle")
    if not isinstance(variants, list):
        return []
    for variant in variants:
        if not isinstance(variant, dict):
            continue
        rects = _coerce_rects(variant.get("interaction_placements"))
        if rects:
            return rects
    return []


def _coerce_rects(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rects: List[Dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            x = int(round(float(item.get("x", 0))))
            y = int(round(float(item.get("y", 0))))
            w = int(round(float(item.get("w", 0))))
            h = int(round(float(item.get("h", 0))))
        except (TypeError, ValueError):
            continue
        if w <= 0 or h <= 0:
            continue
        rect = dict(item)
        rect["x"] = x
        rect["y"] = y
        rect["w"] = w
        rect["h"] = h
        rects.append(rect)
    return rects


def _coerce_outline_polygons(value: Any) -> List[List[Dict[str, int]]]:
    if not isinstance(value, list):
        return []
    polygons: List[List[Dict[str, int]]] = []
    for item in value:
        polygon = _coerce_outline_polygon(item)
        if polygon:
            polygons.append(polygon)
    return polygons


def _coerce_outline_polygon(value: Any) -> List[Dict[str, int]]:
    if not isinstance(value, list):
        return []
    polygon: List[Dict[str, int]] = []
    for point in value:
        if not isinstance(point, dict):
            continue
        try:
            polygon.append(
                {
                    "x": int(round(float(point.get("x", 0)))),
                    "y": int(round(float(point.get("y", 0)))),
                }
            )
        except (TypeError, ValueError):
            continue
    return polygon if len(polygon) >= 3 else []


def _compute_local_bbox_from_polygons(
    polygons: List[List[Dict[str, int]]],
) -> Dict[str, int]:
    xs = [point["x"] for polygon in polygons for point in polygon]
    ys = [point["y"] for polygon in polygons for point in polygon]
    if not xs or not ys:
        return {"min_x": 0, "min_y": 0, "max_x": 0, "max_y": 0}
    return {
        "min_x": min(xs),
        "min_y": min(ys),
        "max_x": max(xs),
        "max_y": max(ys),
    }


# ----------------------------
# Outline mode: envelope (no shapely)
# ----------------------------


def _staircase_envelope_polygon_ccw(
    rects: List[Dict[str, Any]],
) -> List[Dict[str, int]]:
    cleaned: List[Tuple[int, int, int, int]] = []
    for r in rects:
        if not isinstance(r, dict):
            continue
        x = int(r.get("x", 0))
        y = int(r.get("y", 0))
        w = int(r.get("w", 0))
        h = int(r.get("h", 0))
        if w <= 0 or h <= 0:
            continue
        cleaned.append((x, y, x + w, y + h))  # x1,y1,x2,y2

    if not cleaned:
        return []

    bottom = min(y1 for _, y1, _, _ in cleaned)
    top = max(y2 for _, _, _, y2 in cleaned)

    y_breaks = sorted({bottom} | {y2 for _, _, _, y2 in cleaned})
    if y_breaks[-1] != top:
        y_breaks.append(top)

    segs: List[Tuple[int, int, int, int]] = []  # (y0,y1,left_x,right_x)
    for i in range(len(y_breaks) - 1):
        y0 = y_breaks[i]
        y1 = y_breaks[i + 1]
        y_probe = y0 + 0.5
        active = [r for r in cleaned if r[3] > y_probe]  # top > y
        if not active:
            continue
        left_x = min(x1 for x1, _, _, _ in active)
        right_x = max(x2 for _, _, x2, _ in active)
        segs.append((y0, y1, left_x, right_x))

    if not segs:
        return []

    pts: List[Tuple[int, int]] = []
    y0, _, left0, right0 = segs[0]
    pts.append((left0, y0))
    pts.append((right0, y0))

    cur_right = right0
    for idx in range(len(segs)):
        y0, y1, _, _ = segs[idx]
        pts.append((cur_right, y1))
        if idx < len(segs) - 1:
            next_right = segs[idx + 1][3]
            if next_right != cur_right:
                pts.append((next_right, y1))
                cur_right = next_right

    _, top_y, left_last, _ = segs[-1]
    pts.append((left_last, top_y))

    cur_left = left_last
    for idx in range(len(segs) - 1, -1, -1):
        y0, _, _, _ = segs[idx]
        pts.append((cur_left, y0))
        if idx > 0:
            prev_left = segs[idx - 1][2]
            if prev_left != cur_left:
                pts.append((prev_left, y0))
                cur_left = prev_left

    pts = _simplify_orthogonal(pts)
    if pts and pts[0] == pts[-1]:
        pts = pts[:-1]
    if _signed_area(pts) < 0:
        pts = list(reversed(pts))

    return [{"x": int(x), "y": int(y)} for x, y in pts]


def _simplify_orthogonal(points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for p in points:
        if not out or out[-1] != p:
            out.append(p)
    if len(out) < 3:
        return out
    if out[0] != out[-1]:
        out.append(out[0])

    def collinear(a: Tuple[int, int], b: Tuple[int, int], c: Tuple[int, int]) -> bool:
        return (a[0] == b[0] == c[0]) or (a[1] == b[1] == c[1])

    simp: List[Tuple[int, int]] = [out[0]]
    for i in range(1, len(out) - 1):
        prev = simp[-1]
        cur = out[i]
        nxt = out[i + 1]
        if collinear(prev, cur, nxt):
            continue
        simp.append(cur)
    simp.append(out[-1])
    return simp


def _signed_area(points: List[Tuple[int, int]]) -> float:
    if len(points) < 3:
        return 0.0
    s = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return 0.5 * s


# ----------------------------
# Outline mode: union / buffered_union (shapely)
# ----------------------------


def _outline_polygons_union_shapely(
    rects: List[Dict[str, Any]],
    *,
    gap_fill_mm: int = 0,
) -> List[List[Dict[str, int]]]:
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    polys = []
    for r in rects:
        if not isinstance(r, dict):
            continue
        x = int(r.get("x", 0))
        y = int(r.get("y", 0))
        w = int(r.get("w", 0))
        h = int(r.get("h", 0))
        if w <= 0 or h <= 0:
            continue
        polys.append(Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)]))

    if not polys:
        return []

    geom = unary_union(polys)

    # If requested, buffer to connect gaps then shrink back
    if gap_fill_mm and gap_fill_mm > 0:
        # join_style=2 keeps corners square-ish
        geom = geom.buffer(gap_fill_mm, join_style=2, cap_style=2).buffer(
            -gap_fill_mm, join_style=2, cap_style=2
        )

    geoms = []
    if geom.geom_type == "Polygon":
        geoms = [geom]
    elif geom.geom_type == "MultiPolygon":
        geoms = list(geom.geoms)
    else:
        # GeometryCollection -> keep polygons only
        try:
            geoms = [g for g in geom.geoms if g.geom_type == "Polygon"]
        except Exception:
            geoms = []

    outlines: List[List[Dict[str, int]]] = []
    for g in geoms:
        coords = list(g.exterior.coords)  # closed
        pts = [(float(px), float(py)) for px, py in coords[:-1]]
        if _signed_area([(int(round(x)), int(round(y))) for x, y in pts]) < 0:
            pts = list(reversed(pts))
        outlines.append([{"x": int(round(px)), "y": int(round(py))} for px, py in pts])

    return outlines


def _outline_polygons_union_grid(
    rects: List[Dict[str, Any]],
) -> List[List[Dict[str, int]]]:
    cleaned: List[Tuple[int, int, int, int]] = []
    x_breaks: set[int] = set()
    y_breaks: set[int] = set()

    for r in rects:
        if not isinstance(r, dict):
            continue
        x1 = int(r.get("x", 0))
        y1 = int(r.get("y", 0))
        x2 = x1 + int(r.get("w", 0))
        y2 = y1 + int(r.get("h", 0))
        if x2 <= x1 or y2 <= y1:
            continue
        cleaned.append((x1, y1, x2, y2))
        x_breaks.update((x1, x2))
        y_breaks.update((y1, y2))

    if not cleaned:
        return []

    xs = sorted(x_breaks)
    ys = sorted(y_breaks)
    occupied: set[Tuple[int, int]] = set()

    for ix in range(len(xs) - 1):
        cx1 = xs[ix]
        cx2 = xs[ix + 1]
        if cx2 <= cx1:
            continue
        for iy in range(len(ys) - 1):
            cy1 = ys[iy]
            cy2 = ys[iy + 1]
            if cy2 <= cy1:
                continue
            for rx1, ry1, rx2, ry2 in cleaned:
                if rx1 <= cx1 and cx2 <= rx2 and ry1 <= cy1 and cy2 <= ry2:
                    occupied.add((ix, iy))
                    break

    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for ix, iy in sorted(occupied):
        x1 = xs[ix]
        x2 = xs[ix + 1]
        y1 = ys[iy]
        y2 = ys[iy + 1]
        if (ix, iy - 1) not in occupied:
            edges.append(((x1, y1), (x2, y1)))
        if (ix + 1, iy) not in occupied:
            edges.append(((x2, y1), (x2, y2)))
        if (ix, iy + 1) not in occupied:
            edges.append(((x2, y2), (x1, y2)))
        if (ix - 1, iy) not in occupied:
            edges.append(((x1, y2), (x1, y1)))

    loops = _trace_directed_edge_loops(edges)
    outlines: List[List[Dict[str, int]]] = []
    for loop in loops:
        simplified = _simplify_orthogonal(loop)
        if simplified and simplified[0] == simplified[-1]:
            simplified = simplified[:-1]
        if len(simplified) < 3:
            continue
        if _signed_area(simplified) < 0:
            simplified = list(reversed(simplified))
        outlines.append([{"x": int(x), "y": int(y)} for x, y in simplified])

    outlines.sort(
        key=lambda poly: (
            -abs(_signed_area([(p["x"], p["y"]) for p in poly])),
            poly[0]["x"] if poly else 0,
            poly[0]["y"] if poly else 0,
        )
    )
    return outlines


def _trace_directed_edge_loops(
    edges: List[Tuple[Tuple[int, int], Tuple[int, int]]],
) -> List[List[Tuple[int, int]]]:
    starts: dict[Tuple[int, int], list[Tuple[int, int]]] = {}
    for start, end in edges:
        starts.setdefault(start, []).append(end)
    for ends in starts.values():
        ends.sort()

    unused = set(edges)
    loops: List[List[Tuple[int, int]]] = []

    while unused:
        start, end = min(unused)
        unused.remove((start, end))
        loop = [start, end]
        current = end

        while current != start:
            next_candidates = [
                candidate
                for candidate in starts.get(current, [])
                if (current, candidate) in unused
            ]
            if not next_candidates:
                break
            next_point = next_candidates[0]
            unused.remove((current, next_point))
            loop.append(next_point)
            current = next_point

        if len(loop) >= 4 and loop[-1] == start:
            loops.append(loop)

    return loops
