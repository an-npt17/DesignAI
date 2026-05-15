from __future__ import annotations

from typing import Any, Dict, List, Tuple


def build_absolute_layout(
    *,
    room_model: Dict[str, Any],
    clusters_outlines: Dict[str, Any],
    cluster_transforms: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build absolute (room-space) object polygons from cluster outlines + transforms.

    Returns:
    {
      "status": "OK|NEED_INFO",
      "room": {"room_id": str, "room_type": str, "polygon_ccw": [...], "obstacles": [...]},
      "objects": [
        {
          "object_id": str,
          "cluster_id": str,
          "polygon_ccw": [{"x":int,"y":int},...],
          "bbox": {"min_x":int,"min_y":int,"max_x":int,"max_y":int},
          "rotation_ccw": int,
          "source_rect": {"x":int,"y":int,"w":int,"h":int}
        }
      ],
      "notes": ["..."],
      "missing": ["..."]
    }
    """
    notes: list[str] = []
    missing: list[str] = []

    room_model = _unwrap(room_model)
    clusters_outlines = _unwrap(clusters_outlines)

    room = room_model.get("room") or {}
    room_poly = room.get("polygon_ccw") or []
    if not isinstance(room_poly, list) or len(room_poly) < 3:
        missing.append("room.polygon_ccw")

    room_type = (
        (room_model.get("meta") or {}).get("room_type")
        or (room_model.get("user_input") or {}).get("room_type")
        or "unknown"
    )

    obstacles = room_model.get("obstacles") or []

    tf_by_id: Dict[str, Dict[str, Any]] = {}
    for t in cluster_transforms:
        if not isinstance(t, dict):
            continue
        cid = t.get("cluster_id")
        if isinstance(cid, str) and cid:
            tf_by_id[cid] = t

    objects_out: list[dict[str, Any]] = []

    if not isinstance(clusters_outlines, dict) or not clusters_outlines:
        missing.append("clusters_outlines")
    else:
        for cid, cinfo in clusters_outlines.items():
            if not isinstance(cid, str) or not cid:
                continue
            t = tf_by_id.get(cid)
            if not isinstance(t, dict):
                missing.append(f"cluster_transform:{cid}")
                continue

            x = int(t.get("x", 0))
            y = int(t.get("y", 0))
            rot = int(t.get("rot", 0)) % 360

            rects = (cinfo.get("cluster_footprint") or {}).get("rects") or []
            if not isinstance(rects, list) or not rects:
                missing.append(f"cluster_rects:{cid}")
                continue

            for r in rects:
                if not isinstance(r, dict):
                    continue
                obj_id = r.get("id") or r.get("object_id")
                if not isinstance(obj_id, str) or not obj_id:
                    continue
                poly = _transform_rect_to_polygon(r, x=x, y=y, rot=rot)
                if not poly:
                    continue
                bbox = _bbox_from_points(poly)
                objects_out.append(
                    {
                        "object_id": obj_id,
                        "cluster_id": cid,
                        "polygon_ccw": [
                            {"x": int(px), "y": int(py)} for px, py in poly
                        ],
                        "bbox": bbox,
                        "rotation_ccw": rot,
                        "source_rect": {
                            "x": int(r.get("x", 0)),
                            "y": int(r.get("y", 0)),
                            "w": int(r.get("w", 0)),
                            "h": int(r.get("h", 0)),
                        },
                    }
                )

    status = "OK" if not missing else "NEED_INFO"
    return {
        "status": status,
        "room": {
            "room_id": room.get("room_id") or "room_0",
            "room_type": str(room_type),
            "polygon_ccw": room_poly if isinstance(room_poly, list) else [],
            "obstacles": obstacles,
            "openings": room_model.get("openings") or {},
        },
        "objects": objects_out,
        "notes": notes,
        "missing": missing,
    }


def _unwrap(obj: Any) -> Any:
    if not isinstance(obj, dict):
        return obj
    parsed = obj.get("parsed")
    if isinstance(parsed, dict):
        return parsed
    raw = obj.get("raw")
    if isinstance(raw, dict):
        return raw
    return obj


def _rotate_ccw_90s(x: int, y: int, rot: int) -> Tuple[int, int]:
    r = rot % 360
    if r == 0:
        return x, y
    if r == 90:
        return -y, x
    if r == 180:
        return -x, -y
    if r == 270:
        return y, -x
    return x, y


def _transform_rect_to_polygon(
    rect: Dict[str, Any], *, x: int, y: int, rot: int
) -> List[Tuple[int, int]]:
    rx = int(rect.get("x", 0))
    ry = int(rect.get("y", 0))
    w = int(rect.get("w", 0))
    h = int(rect.get("h", 0))
    if w <= 0 or h <= 0:
        return []
    corners = [
        (rx, ry),
        (rx + w, ry),
        (rx + w, ry + h),
        (rx, ry + h),
    ]
    pts: list[Tuple[int, int]] = []
    for px, py in corners:
        rx2, ry2 = _rotate_ccw_90s(px, py, rot)
        pts.append((rx2 + x, ry2 + y))
    # ensure CCW
    if _signed_area(pts) < 0:
        pts = list(reversed(pts))
    return pts


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


def _bbox_from_points(points: List[Tuple[int, int]]) -> Dict[str, int]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return {
        "min_x": int(min(xs)),
        "min_y": int(min(ys)),
        "max_x": int(max(xs)),
        "max_y": int(max(ys)),
    }


TOOL_REGISTRY: Dict[str, Any] = {
    "BuildAbsoluteLayout": build_absolute_layout,
}

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "BuildAbsoluteLayout",
            "description": "Build absolute object polygons in room coordinates from cluster outlines + transforms.",
            "parameters": {
                "type": "object",
                "properties": {
                    "room_model": {"type": "object"},
                    "clusters_outlines": {"type": "object"},
                    "cluster_transforms": {
                        "type": "array",
                        "items": {"type": "object"},
                    },
                },
                "required": ["room_model", "clusters_outlines", "cluster_transforms"],
            },
        },
    }
]
