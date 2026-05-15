from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict

from agent_schema.roomI_schema import RoomInterpreterOutput
from prompt.room_interpreter import ROOM_INTERPRETER_PROMPT
from prompt.system import SYSTEM_PROMPT

PointDict = dict[str, int]
ChatMessage = dict[str, str]
OpeningKind = Literal["door", "window"]

GRID_MM = 50
OPENING_SNAP_TOLERANCE_MM = 120
CORNER_EXCLUSION_MM = 200
DOOR_FRONT_CLEARANCE_MM = 900
DOOR_SIDE_MARGIN_MM = 100
WINDOW_CLEARANCE_MM = 150
DAYLIGHT_NEAR_DEPTH_MM = 1600
DAYLIGHT_NEAR_SIDE_MARGIN_MM = 150
DAYLIGHT_MID_DEPTH_MM = 3000
DAYLIGHT_MID_SIDE_MARGIN_MM = 300
MIN_WALL_LEN_MM = 150
MIN_USABLE_WALL_LEN_MM = 700
AFFORDANCE_FIELD_RESOLUTION_MM = 200
CORRIDOR_CANDIDATE_MAX = 4
ZONE_SPLIT_MAX = 4
LLM_TEMPERATURE = 0.1
LLM_RETRY_MAX = 1
DEFAULT_DOOR_SWING_RADIUS_MM = 900
DEFAULT_ROOM_HEIGHT_MM = 2800


class FloatPoint(TypedDict):
    x: float
    y: float


class Wall(TypedDict):
    id: str
    index: int
    start_mm: PointDict
    end_mm: PointDict
    length_mm: int
    direction: FloatPoint
    inward_normal: FloatPoint
    adjacent_wall_ids: list[str]


class NormalizedOpening(TypedDict):
    id: str
    kind: str
    wall_id: str
    segment_mm: list[PointDict]
    original_segment_mm: list[PointDict]
    width_mm: int
    wall_t_start_mm: int
    wall_t_end_mm: int
    snap_distance_mm: int
    swing_radius_mm: NotRequired[int]
    hinge_hint: NotRequired[str]
    clearance_mm: NotRequired[int]


@dataclass(frozen=True)
class RoomInterpreter:
    system_prompt: str = SYSTEM_PROMPT
    prompt_template: str = ROOM_INTERPRETER_PROMPT

    def build_messages(
        self,
        input_payload: Mapping[str, object],
        *,
        description: str | None = None,
        special_notes: str | None = None,
    ) -> list[ChatMessage]:
        payload = json.dumps(
            _build_summary_input(
                input_payload,
                description=description,
                special_notes=special_notes,
            ),
            ensure_ascii=True,
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"{self.prompt_template}\n\nINPUT_JSON:\n{payload}",
            },
        ]

    def generate_raw(
        self,
        input_payload: Mapping[str, object],
        *,
        description: str | None = None,
        special_notes: str | None = None,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int | None = None,
    ) -> str:
        _ = temperature, max_tokens
        payload = self._build_output_payload(
            input_payload,
            description=description,
            special_notes=special_notes,
        )
        return json.dumps(payload, ensure_ascii=True, indent=2)

    def generate(
        self,
        input_payload: Mapping[str, object],
        *,
        description: str | None = None,
        special_notes: str | None = None,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int | None = None,
    ) -> RoomInterpreterOutput:
        _ = temperature, max_tokens
        payload = self._build_output_payload(
            input_payload,
            description=description,
            special_notes=special_notes,
        )
        return RoomInterpreterOutput.model_validate(payload)

    def _build_output_payload(
        self,
        input_payload: Mapping[str, object],
        *,
        description: str | None,
        special_notes: str | None,
    ) -> dict[str, object]:
        return _build_room_interpreter_payload(
            input_payload,
            description=description,
            special_notes=special_notes,
        )


def _build_room_interpreter_payload(
    input_payload: Mapping[str, object],
    *,
    description: str | None,
    special_notes: str | None,
) -> dict[str, object]:
    missing: list[str] = []
    conflicts: list[str] = []
    notes: list[str] = []

    raw_polygon = _resolve_room_polygon(input_payload)
    sanitized_polygon, sanitize_notes, sanitize_conflicts = _sanitize_polygon(
        raw_polygon
    )
    notes.extend(sanitize_notes)
    conflicts.extend(sanitize_conflicts)
    if len(sanitized_polygon) < 3:
        missing.append("polygon_mm")

    if _room_polygon_conflicts(input_payload):
        conflicts.append(
            "floorplan_geometry.room.polygon_mm conflicts with user_input.shape_points; using floorplan geometry."
        )

    walls = _build_wall_graph(sanitized_polygon)
    openings_input = _resolve_floorplan_openings(input_payload)
    explicit_doors = _opening_kind_explicitly_provided(input_payload, kind="doors")
    explicit_windows = _opening_kind_explicitly_provided(
        input_payload,
        kind="windows",
    )
    doors = _normalize_openings_for_kind(
        openings_input.get("doors"),
        walls=walls,
        kind="door",
        conflicts=conflicts,
    )
    windows = _normalize_openings_for_kind(
        openings_input.get("windows"),
        walls=walls,
        kind="window",
        conflicts=conflicts,
    )

    if sanitized_polygon and not doors and not explicit_doors:
        default_door = _default_opening(walls, kind="door", existing_ids=set())
        if default_door is not None:
            doors.append(default_door)
            notes.append(
                "No door was provided; generated a deterministic default entry on the longest wall."
            )

    if sanitized_polygon and not windows and not explicit_windows:
        default_window = _default_opening(
            walls,
            kind="window",
            existing_ids={opening["id"] for opening in doors},
        )
        if default_window is not None:
            windows.append(default_window)
            notes.append(
                "No window was provided; generated a deterministic default window on a secondary long wall."
            )

    fixed_obstacles = _resolve_fixed_obstacles(input_payload)
    hard_obstacles = _build_hard_obstacles(
        fixed_obstacles=fixed_obstacles,
        doors=doors,
        windows=windows,
        walls=walls,
    )
    usable_walls, blocked_walls = _extract_usable_wall_segments(
        walls=walls,
        doors=doors,
        windows=windows,
        hard_obstacles=hard_obstacles,
    )

    free_points = _sample_free_points(
        polygon=sanitized_polygon,
        hard_obstacles=hard_obstacles,
        resolution_mm=AFFORDANCE_FIELD_RESOLUTION_MM,
    )
    centroid = _polygon_centroid(sanitized_polygon)
    principal_axis = _principal_axis(sanitized_polygon)
    room_area_mm2 = int(round(abs(_signed_area(sanitized_polygon))))
    room_perimeter_mm = int(round(sum(wall["length_mm"] for wall in walls)))
    room_bbox = _bbox_from_points(sanitized_polygon)
    soft_usage_hints = _soft_usage_hints(
        input_payload,
        room_type=_resolve_room_type(input_payload),
        description=description,
        special_notes=special_notes,
    )
    affordance_map = _build_affordance_map(
        polygon=sanitized_polygon,
        walls=walls,
        doors=doors,
        windows=windows,
        hard_obstacles=hard_obstacles,
        usable_walls=usable_walls,
        blocked_walls=blocked_walls,
        free_points=free_points,
        centroid=centroid,
        soft_usage_hints=soft_usage_hints,
    )
    topology = _build_topology(
        walls=walls,
        doors=doors,
        windows=windows,
        affordance_map=affordance_map,
        free_points=free_points,
        centroid=centroid,
    )

    if not windows and not explicit_windows:
        missing.append("windows")
    if not usable_walls:
        conflicts.append(
            "No usable wall segment longer than minimum threshold was found."
        )

    status = _status_from_quality(missing=missing, conflicts=conflicts)
    confidence = _confidence_score(
        polygon=sanitized_polygon,
        doors=doors,
        windows=windows,
        usable_walls=usable_walls,
        missing=missing,
        conflicts=conflicts,
    )
    quality_meta = {
        "missing": missing,
        "conflicts": conflicts,
        "confidence": confidence,
        "notes": _dedupe_strings(notes),
        "settings": _settings_payload(),
    }
    room = {
        "unit": "mm",
        "room_id": _resolve_room_id(input_payload),
        "name": _resolve_room_name(input_payload),
        "room_type": _resolve_room_type(input_payload),
        "polygon_ccw": sanitized_polygon,
        "area_mm2": room_area_mm2,
        "area_m2": round(room_area_mm2 / 1_000_000.0, 3),
        "perimeter_mm": room_perimeter_mm,
        "centroid_mm": _point_from_float(centroid),
        "principal_axis": principal_axis,
        "bbox_mm": room_bbox,
        "ceiling_height_mm": _resolve_height_mm(input_payload),
    }
    openings = {"doors": doors, "windows": windows}
    meta = {
        "room_type": _resolve_room_type(input_payload),
        "style": _resolve_style(input_payload),
        "height_mm": _resolve_height_mm(input_payload),
        "window_direction": _resolve_window_direction(input_payload),
        "grid_mm": GRID_MM,
        "llm_temperature": LLM_TEMPERATURE,
        "llm_retry_max": LLM_RETRY_MAX,
    }

    legacy_notes = _legacy_notes(
        input_payload=input_payload,
        room_type=_resolve_room_type(input_payload),
        style=_resolve_style(input_payload),
        description=description,
        special_notes=special_notes,
        soft_usage_hints=soft_usage_hints,
    )
    return {
        "status": status,
        "room": room,
        "openings": openings,
        "hard_obstacles": hard_obstacles,
        "affordance_map": affordance_map,
        "topology": topology,
        "quality_meta": quality_meta,
        "obstacles": hard_obstacles,
        "meta": meta,
        "notes": legacy_notes,
        "missing": missing,
        "conflicts": conflicts,
    }


def _sanitize_polygon(
    raw_polygon: object,
) -> tuple[list[PointDict], list[str], list[str]]:
    notes: list[str] = []
    conflicts: list[str] = []
    points = _normalize_polygon_points(
        raw_polygon if isinstance(raw_polygon, list) else []
    )
    snapped = [_snap_point(point) for point in points]
    deduped = _remove_duplicate_points(snapped)
    if len(deduped) != len(points):
        notes.append(
            "Removed duplicate or near-duplicate polygon points during sanitize."
        )
    if len(deduped) >= 3 and _signed_area(deduped) < 0:
        deduped = list(reversed(deduped))
        notes.append("Reoriented room polygon to CCW.")
    if _has_self_intersection(deduped):
        conflicts.append("Room polygon has self-intersections after sanitize.")
    return deduped, notes, conflicts


def _snap_point(point: Mapping[str, int]) -> PointDict:
    return {
        "x": int(round(int(point["x"]) / GRID_MM) * GRID_MM),
        "y": int(round(int(point["y"]) / GRID_MM) * GRID_MM),
    }


def _remove_duplicate_points(points: Sequence[PointDict]) -> list[PointDict]:
    deduped: list[PointDict] = []
    for point in points:
        if deduped and _point_distance(deduped[-1], point) < GRID_MM * 0.5:
            continue
        deduped.append(dict(point))
    if len(deduped) > 1 and _point_distance(deduped[0], deduped[-1]) < GRID_MM * 0.5:
        deduped.pop()
    return deduped


def _has_self_intersection(points: Sequence[PointDict]) -> bool:
    if len(points) < 4:
        return False
    count = len(points)
    for idx in range(count):
        a1 = points[idx]
        a2 = points[(idx + 1) % count]
        for jdx in range(idx + 1, count):
            if abs(idx - jdx) <= 1 or {idx, jdx} == {0, count - 1}:
                continue
            b1 = points[jdx]
            b2 = points[(jdx + 1) % count]
            if _segments_intersect(a1, a2, b1, b2):
                return True
    return False


def _segments_intersect(
    a1: Mapping[str, int],
    a2: Mapping[str, int],
    b1: Mapping[str, int],
    b2: Mapping[str, int],
) -> bool:
    def orient(p: Mapping[str, int], q: Mapping[str, int], r: Mapping[str, int]) -> int:
        value = (q["y"] - p["y"]) * (r["x"] - q["x"]) - (q["x"] - p["x"]) * (
            r["y"] - q["y"]
        )
        if abs(value) < 1e-9:
            return 0
        return 1 if value > 0 else 2

    o1 = orient(a1, a2, b1)
    o2 = orient(a1, a2, b2)
    o3 = orient(b1, b2, a1)
    o4 = orient(b1, b2, a2)
    return o1 != o2 and o3 != o4


def _build_wall_graph(points: Sequence[PointDict]) -> list[Wall]:
    walls: list[Wall] = []
    count = len(points)
    if count < 3:
        return walls
    for idx, start in enumerate(points):
        end = points[(idx + 1) % count]
        dx = end["x"] - start["x"]
        dy = end["y"] - start["y"]
        length = math.hypot(dx, dy)
        if length < MIN_WALL_LEN_MM:
            continue
        direction = {"x": dx / length, "y": dy / length}
        wall: Wall = {
            "id": f"wall_{idx + 1}",
            "index": idx,
            "start_mm": dict(start),
            "end_mm": dict(end),
            "length_mm": int(round(length)),
            "direction": direction,
            "inward_normal": {"x": -direction["y"], "y": direction["x"]},
            "adjacent_wall_ids": [],
        }
        walls.append(wall)
    wall_count = len(walls)
    for idx, wall in enumerate(walls):
        wall["adjacent_wall_ids"] = [
            walls[(idx - 1) % wall_count]["id"],
            walls[(idx + 1) % wall_count]["id"],
        ]
    return walls


def _normalize_openings_for_kind(
    value: object,
    *,
    walls: Sequence[Wall],
    kind: OpeningKind,
    conflicts: list[str],
) -> list[NormalizedOpening]:
    if not isinstance(value, list):
        return []
    normalized: list[NormalizedOpening] = []
    existing_ids: set[str] = set()
    for index, item in enumerate(value, start=1):
        opening = _normalize_single_opening(
            item,
            walls=walls,
            kind=kind,
            fallback_id=f"{kind}_{index}",
            conflicts=conflicts,
        )
        if opening is None:
            continue
        opening_id = _ensure_unique_identifier(opening["id"], existing_ids)
        opening["id"] = opening_id
        existing_ids.add(opening_id)
        normalized.append(opening)
    return normalized


def _normalize_single_opening(
    value: object,
    *,
    walls: Sequence[Wall],
    kind: OpeningKind,
    fallback_id: str,
    conflicts: list[str],
) -> NormalizedOpening | None:
    if not isinstance(value, dict):
        return None
    segment = _normalize_segment(
        value.get("segment_mm"),
        start=value.get("start_mm"),
        end=value.get("end_mm"),
    )
    if segment is None:
        return None

    opening_id = _coerce_identifier(
        value.get("id") or value.get(f"{kind}_id"),
        fallback=fallback_id,
    )
    snapped = _snap_segment_to_wall(segment, walls=walls)
    if snapped is None:
        conflicts.append(f"{opening_id} could not be assigned to a wall.")
        return None
    if snapped["snap_distance_mm"] > OPENING_SNAP_TOLERANCE_MM:
        conflicts.append(
            f"{opening_id} is {snapped['snap_distance_mm']}mm away from nearest wall; snapped to {snapped['wall_id']}."
        )

    opening: NormalizedOpening = {
        "id": opening_id,
        "kind": kind,
        "wall_id": snapped["wall_id"],
        "segment_mm": snapped["segment_mm"],
        "original_segment_mm": segment,
        "width_mm": snapped["width_mm"],
        "wall_t_start_mm": snapped["wall_t_start_mm"],
        "wall_t_end_mm": snapped["wall_t_end_mm"],
        "snap_distance_mm": snapped["snap_distance_mm"],
    }
    if kind == "door":
        opening["swing_radius_mm"] = _coerce_positive_int(
            value.get("swing_radius_mm") or value.get("leaf_width_mm"),
            fallback=DEFAULT_DOOR_SWING_RADIUS_MM,
        )
        opening["hinge_hint"] = _normalize_hinge_hint(
            value.get("hinge_hint") or value.get("hinge_side")
        )
    else:
        opening["clearance_mm"] = _coerce_positive_int(
            value.get("clearance_mm"),
            fallback=WINDOW_CLEARANCE_MM,
        )
    return opening


def _snap_segment_to_wall(
    segment: Sequence[PointDict],
    *,
    walls: Sequence[Wall],
) -> dict[str, object] | None:
    if len(segment) != 2 or not walls:
        return None
    best: dict[str, object] | None = None
    for wall in walls:
        start = wall["start_mm"]
        end = wall["end_mm"]
        p1 = _project_point_to_segment(segment[0], start, end)
        p2 = _project_point_to_segment(segment[1], start, end)
        distance = max(p1["distance_mm"], p2["distance_mm"])
        if best is not None and distance >= best["snap_distance_mm"]:
            continue
        t1 = float(p1["t_mm"])
        t2 = float(p2["t_mm"])
        t_start = max(0.0, min(t1, t2))
        t_end = min(float(wall["length_mm"]), max(t1, t2))
        if abs(t_end - t_start) < 1.0:
            center_t = (t1 + t2) / 2.0
            half_width = 450.0
            t_start = max(0.0, center_t - half_width)
            t_end = min(float(wall["length_mm"]), center_t + half_width)
            t1 = t_start
            t2 = t_end
        snapped_start = _point_on_wall(wall, t1)
        snapped_end = _point_on_wall(wall, t2)
        best = {
            "wall_id": wall["id"],
            "segment_mm": [snapped_start, snapped_end],
            "width_mm": int(round(abs(t_end - t_start))),
            "wall_t_start_mm": int(round(t_start)),
            "wall_t_end_mm": int(round(t_end)),
            "snap_distance_mm": int(round(float(distance))),
        }
    return best


def _project_point_to_segment(
    point: Mapping[str, int],
    start: Mapping[str, int],
    end: Mapping[str, int],
) -> dict[str, float]:
    dx = end["x"] - start["x"]
    dy = end["y"] - start["y"]
    length_sq = dx * dx + dy * dy
    if length_sq <= 0:
        return {"t_mm": 0.0, "distance_mm": _point_distance(point, start)}
    raw_t = (
        (point["x"] - start["x"]) * dx + (point["y"] - start["y"]) * dy
    ) / length_sq
    clamped = max(0.0, min(1.0, raw_t))
    proj = {"x": start["x"] + dx * clamped, "y": start["y"] + dy * clamped}
    return {
        "t_mm": math.sqrt(length_sq) * clamped,
        "distance_mm": math.hypot(point["x"] - proj["x"], point["y"] - proj["y"]),
    }


def _default_opening(
    walls: Sequence[Wall],
    *,
    kind: OpeningKind,
    existing_ids: set[str],
) -> NormalizedOpening | None:
    if not walls:
        return None
    sorted_walls = sorted(walls, key=lambda wall: wall["length_mm"], reverse=True)
    wall = (
        sorted_walls[0] if kind == "door" or len(sorted_walls) == 1 else sorted_walls[1]
    )
    width = 900 if kind == "door" else 1200
    usable_width = min(width, max(300, int(wall["length_mm"] * 0.7)))
    center_t = wall["length_mm"] / 2.0
    t_start = max(0.0, center_t - usable_width / 2.0)
    t_end = min(float(wall["length_mm"]), center_t + usable_width / 2.0)
    opening_id = _ensure_unique_identifier(f"{kind}_1", existing_ids)
    opening: NormalizedOpening = {
        "id": opening_id,
        "kind": kind,
        "wall_id": wall["id"],
        "segment_mm": [_point_on_wall(wall, t_start), _point_on_wall(wall, t_end)],
        "original_segment_mm": [
            _point_on_wall(wall, t_start),
            _point_on_wall(wall, t_end),
        ],
        "width_mm": int(round(t_end - t_start)),
        "wall_t_start_mm": int(round(t_start)),
        "wall_t_end_mm": int(round(t_end)),
        "snap_distance_mm": 0,
    }
    if kind == "door":
        opening["swing_radius_mm"] = DEFAULT_DOOR_SWING_RADIUS_MM
        opening["hinge_hint"] = "UNKNOWN"
    else:
        opening["clearance_mm"] = WINDOW_CLEARANCE_MM
    return opening


def _resolve_fixed_obstacles(
    input_payload: Mapping[str, object],
) -> list[dict[str, object]]:
    candidates: list[object] = []
    for key in ("fixed_elements", "hard_obstacles", "obstacles"):
        value = input_payload.get(key)
        if isinstance(value, list):
            candidates.extend(value)

    floorplan = input_payload.get("floorplan_geometry")
    if isinstance(floorplan, dict):
        for key in ("fixed_elements", "hard_obstacles", "obstacles"):
            value = floorplan.get(key)
            if isinstance(value, list):
                candidates.extend(value)
        openings = floorplan.get("openings")
        if isinstance(openings, dict) and isinstance(openings.get("obstacles"), list):
            candidates.extend(openings["obstacles"])

    constraints = input_payload.get("constraints")
    if isinstance(constraints, dict):
        for key in ("fixed_elements", "obstacles", "no_go_zones"):
            value = constraints.get(key)
            if isinstance(value, list):
                candidates.extend(value)

    normalized: list[dict[str, object]] = []
    existing_ids: set[str] = set()
    for index, item in enumerate(candidates, start=1):
        obstacle = _normalize_fixed_obstacle(item, fallback_id=f"fixed_{index}")
        if obstacle is None:
            continue
        obstacle_id = _ensure_unique_identifier(str(obstacle["id"]), existing_ids)
        obstacle["id"] = obstacle_id
        existing_ids.add(obstacle_id)
        normalized.append(obstacle)
    return normalized


def _normalize_fixed_obstacle(
    value: object,
    *,
    fallback_id: str,
) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    polygon = value.get("polygon_ccw")
    if not isinstance(polygon, list):
        polygon = value.get("polygon_mm")
    if not isinstance(polygon, list):
        polygon = _polygon_from_bbox(value.get("bbox_mm") or value.get("bbox"))
    if not isinstance(polygon, list):
        return None
    normalized_polygon = _ensure_ccw(
        _remove_duplicate_points(
            [_snap_point(p) for p in _normalize_polygon_points(polygon)]
        )
    )
    if len(normalized_polygon) < 3:
        return None
    return {
        "id": _coerce_identifier(
            value.get("id") or value.get("element_id") or value.get("obstacle_id"),
            fallback=fallback_id,
        ),
        "type": _normalize_fixed_type(value.get("type")),
        "polygon_ccw": normalized_polygon,
        "hard": bool(value.get("hard", True)),
        "source_id": str(value.get("source_id") or value.get("id") or fallback_id),
    }


def _polygon_from_bbox(value: object) -> list[PointDict] | None:
    if not isinstance(value, dict):
        return None
    min_x = _coerce_optional_int(value.get("min_x") or value.get("x"))
    min_y = _coerce_optional_int(value.get("min_y") or value.get("y"))
    max_x = _coerce_optional_int(value.get("max_x"))
    max_y = _coerce_optional_int(value.get("max_y"))
    width = _coerce_optional_int(value.get("w") or value.get("width_mm"))
    height = _coerce_optional_int(value.get("h") or value.get("height_mm"))
    if min_x is None or min_y is None:
        return None
    if max_x is None and width is not None:
        max_x = min_x + width
    if max_y is None and height is not None:
        max_y = min_y + height
    if max_x is None or max_y is None:
        return None
    return [
        {"x": min_x, "y": min_y},
        {"x": max_x, "y": min_y},
        {"x": max_x, "y": max_y},
        {"x": min_x, "y": max_y},
    ]


def _normalize_fixed_type(value: object) -> str:
    if isinstance(value, str) and value.strip():
        normalized = value.strip().lower()
        if normalized in {
            "door_swing",
            "entry_clearance",
            "opening_guard",
            "window_clearance",
        }:
            return normalized
        return "fixed_element" if normalized in {"fixed", "fixed_element"} else "no_go"
    return "fixed_element"


def _build_hard_obstacles(
    *,
    fixed_obstacles: Sequence[dict[str, object]],
    doors: Sequence[NormalizedOpening],
    windows: Sequence[NormalizedOpening],
    walls: Sequence[Wall],
) -> list[dict[str, object]]:
    obstacles = [dict(obstacle) for obstacle in fixed_obstacles]
    wall_by_id = {wall["id"]: wall for wall in walls}
    existing_ids = {str(obstacle.get("id")) for obstacle in obstacles}
    for door in doors:
        wall = wall_by_id.get(door["wall_id"])
        if wall is None:
            continue
        for obstacle in (
            _opening_clearance_obstacle(
                door,
                wall=wall,
                obstacle_type="door_swing",
                clearance_mm=int(
                    door.get("swing_radius_mm") or DEFAULT_DOOR_SWING_RADIUS_MM
                ),
                side_margin_mm=0,
                suffix="swing",
            ),
            _opening_clearance_obstacle(
                door,
                wall=wall,
                obstacle_type="entry_clearance",
                clearance_mm=DOOR_FRONT_CLEARANCE_MM,
                side_margin_mm=DOOR_SIDE_MARGIN_MM,
                suffix="entry_clearance",
            ),
        ):
            if obstacle["id"] not in existing_ids:
                obstacles.append(obstacle)
                existing_ids.add(str(obstacle["id"]))

    for window in windows:
        wall = wall_by_id.get(window["wall_id"])
        if wall is None:
            continue
        obstacle = _opening_clearance_obstacle(
            window,
            wall=wall,
            obstacle_type="window_clearance",
            clearance_mm=int(window.get("clearance_mm") or WINDOW_CLEARANCE_MM),
            side_margin_mm=0,
            suffix="clearance",
        )
        if obstacle["id"] not in existing_ids:
            obstacles.append(obstacle)
            existing_ids.add(str(obstacle["id"]))
    return obstacles


def _opening_clearance_obstacle(
    opening: NormalizedOpening,
    *,
    wall: Wall,
    obstacle_type: str,
    clearance_mm: int,
    side_margin_mm: int,
    suffix: str,
) -> dict[str, object]:
    t_start = max(0.0, float(opening["wall_t_start_mm"]) - side_margin_mm)
    t_end = min(
        float(wall["length_mm"]), float(opening["wall_t_end_mm"]) + side_margin_mm
    )
    normal = wall["inward_normal"]
    p1 = _point_on_wall(wall, t_start)
    p2 = _point_on_wall(wall, t_end)
    p3 = _offset_point(p2, normal, clearance_mm)
    p4 = _offset_point(p1, normal, clearance_mm)
    return {
        "id": f"{opening['id']}_{suffix}",
        "type": obstacle_type,
        "polygon_ccw": _ensure_ccw([p1, p2, p3, p4]),
        "hard": True,
        "source_id": opening["id"],
    }


def _extract_usable_wall_segments(
    *,
    walls: Sequence[Wall],
    doors: Sequence[NormalizedOpening],
    windows: Sequence[NormalizedOpening],
    hard_obstacles: Sequence[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    opening_intervals_by_wall: dict[str, list[tuple[float, float, str]]] = {}
    for opening in [*doors, *windows]:
        margin = (
            DOOR_SIDE_MARGIN_MM if opening["kind"] == "door" else WINDOW_CLEARANCE_MM
        )
        opening_intervals_by_wall.setdefault(opening["wall_id"], []).append(
            (
                max(0.0, float(opening["wall_t_start_mm"]) - margin),
                float(opening["wall_t_end_mm"]) + margin,
                f"{opening['kind']}:{opening['id']}",
            )
        )

    usable: list[dict[str, object]] = []
    blocked: list[dict[str, object]] = []
    for wall in walls:
        intervals: list[tuple[float, float, str]] = [
            (
                0.0,
                min(float(CORNER_EXCLUSION_MM), float(wall["length_mm"])),
                "corner_start",
            ),
            (
                max(0.0, float(wall["length_mm"]) - CORNER_EXCLUSION_MM),
                float(wall["length_mm"]),
                "corner_end",
            ),
        ]
        intervals.extend(opening_intervals_by_wall.get(wall["id"], []))
        intervals.extend(_obstacle_wall_intervals(wall, hard_obstacles))
        merged = _merge_intervals(
            [
                (
                    max(0.0, min(float(wall["length_mm"]), start)),
                    max(0.0, min(float(wall["length_mm"]), end)),
                    reason,
                )
                for start, end, reason in intervals
                if end > start
            ]
        )
        cursor = 0.0
        for start, end, _reason in merged:
            if start - cursor >= MIN_USABLE_WALL_LEN_MM:
                usable.append(_usable_wall_segment(wall, cursor, start))
            cursor = max(cursor, end)
        if float(wall["length_mm"]) - cursor >= MIN_USABLE_WALL_LEN_MM:
            usable.append(_usable_wall_segment(wall, cursor, float(wall["length_mm"])))

        if not any(segment["wall_id"] == wall["id"] for segment in usable):
            blocked.append(
                {
                    "wall_id": wall["id"],
                    "length_mm": wall["length_mm"],
                    "blocked_intervals_mm": [
                        {
                            "start_mm": int(round(start)),
                            "end_mm": int(round(end)),
                            "reason": reason,
                        }
                        for start, end, reason in merged
                    ],
                }
            )
    return usable, blocked


def _obstacle_wall_intervals(
    wall: Wall,
    hard_obstacles: Sequence[dict[str, object]],
) -> list[tuple[float, float, str]]:
    intervals: list[tuple[float, float, str]] = []
    for obstacle in hard_obstacles:
        if obstacle.get("type") in {
            "door_swing",
            "entry_clearance",
            "window_clearance",
        }:
            continue
        polygon = obstacle.get("polygon_ccw")
        if not isinstance(polygon, list):
            continue
        projected: list[float] = []
        for point in _normalize_polygon_points(polygon):
            projection = _project_point_to_segment(
                point, wall["start_mm"], wall["end_mm"]
            )
            if projection["distance_mm"] <= 250:
                projected.append(projection["t_mm"])
        if len(projected) >= 2:
            intervals.append(
                (
                    min(projected) - DOOR_SIDE_MARGIN_MM,
                    max(projected) + DOOR_SIDE_MARGIN_MM,
                    f"fixed:{obstacle.get('id')}",
                )
            )
    return intervals


def _merge_intervals(
    intervals: Sequence[tuple[float, float, str]],
) -> list[tuple[float, float, str]]:
    if not intervals:
        return []
    sorted_intervals = sorted(intervals, key=lambda item: item[0])
    merged: list[tuple[float, float, str]] = []
    for start, end, reason in sorted_intervals:
        if not merged or start > merged[-1][1]:
            merged.append((start, end, reason))
            continue
        prev_start, prev_end, prev_reason = merged[-1]
        merged[-1] = (prev_start, max(prev_end, end), f"{prev_reason},{reason}")
    return merged


def _usable_wall_segment(wall: Wall, start: float, end: float) -> dict[str, object]:
    length = int(round(end - start))
    return {
        "id": f"{wall['id']}_usable_{int(round(start))}_{int(round(end))}",
        "wall_id": wall["id"],
        "segment_mm": [_point_on_wall(wall, start), _point_on_wall(wall, end)],
        "length_mm": length,
        "wall_t_start_mm": int(round(start)),
        "wall_t_end_mm": int(round(end)),
        "inward_normal": wall["inward_normal"],
    }


def _build_affordance_map(
    *,
    polygon: Sequence[PointDict],
    walls: Sequence[Wall],
    doors: Sequence[NormalizedOpening],
    windows: Sequence[NormalizedOpening],
    hard_obstacles: Sequence[dict[str, object]],
    usable_walls: Sequence[dict[str, object]],
    blocked_walls: Sequence[dict[str, object]],
    free_points: Sequence[FloatPoint],
    centroid: FloatPoint,
    soft_usage_hints: dict[str, object],
) -> dict[str, object]:
    _ = polygon, hard_obstacles
    entry_landing_zones = [
        obstacle
        for obstacle in hard_obstacles
        if isinstance(obstacle.get("type"), str)
        and obstacle["type"] == "entry_clearance"
    ]
    daylight_regions = _daylight_regions(windows=windows, walls=walls)
    privacy_regions = _privacy_regions(doors=doors, free_points=free_points)
    focal_surfaces = _focal_surfaces(
        usable_walls=usable_walls,
        blocked_walls=blocked_walls,
        windows=windows,
    )
    center_regions = _center_openness_regions(
        free_points=free_points,
        centroid=centroid,
    )
    wall_anchors = _wall_anchor_candidates(usable_walls=usable_walls)
    floating_zones = _floating_zone_candidates(
        center_regions=center_regions,
        free_points=free_points,
        doors=doors,
    )
    corridors = _circulation_corridors(
        doors=doors,
        windows=windows,
        free_points=free_points,
        centroid=centroid,
    )
    return {
        "usable_walls": list(usable_walls),
        "blocked_walls": list(blocked_walls),
        "entry_landing_zones": entry_landing_zones,
        "circulation_corridors": corridors,
        "daylight_regions": daylight_regions,
        "privacy_regions": privacy_regions,
        "focal_surfaces": focal_surfaces,
        "center_openness_regions": center_regions,
        "wall_anchor_candidates": wall_anchors,
        "floating_zone_candidates": floating_zones,
        "soft_usage_hints": soft_usage_hints,
    }


def _daylight_regions(
    *,
    windows: Sequence[NormalizedOpening],
    walls: Sequence[Wall],
) -> list[dict[str, object]]:
    wall_by_id = {wall["id"]: wall for wall in walls}
    regions: list[dict[str, object]] = []
    for window in windows:
        wall = wall_by_id.get(window["wall_id"])
        if wall is None:
            continue
        near = _opening_clearance_obstacle(
            window,
            wall=wall,
            obstacle_type="daylight_near",
            clearance_mm=DAYLIGHT_NEAR_DEPTH_MM,
            side_margin_mm=DAYLIGHT_NEAR_SIDE_MARGIN_MM,
            suffix="daylight_near",
        )
        mid = _opening_clearance_obstacle(
            window,
            wall=wall,
            obstacle_type="daylight_mid",
            clearance_mm=DAYLIGHT_MID_DEPTH_MM,
            side_margin_mm=DAYLIGHT_MID_SIDE_MARGIN_MM,
            suffix="daylight_mid",
        )
        regions.append(
            {
                "id": f"{window['id']}_daylight",
                "source_window_id": window["id"],
                "near_polygon_ccw": near["polygon_ccw"],
                "mid_polygon_ccw": mid["polygon_ccw"],
                "score": 1.0,
            }
        )
    return regions


def _privacy_regions(
    *,
    doors: Sequence[NormalizedOpening],
    free_points: Sequence[FloatPoint],
) -> list[dict[str, object]]:
    if not free_points:
        return []
    entry = _primary_entry_point(doors)
    scored: list[tuple[float, FloatPoint]] = []
    for point in free_points:
        distance = _float_distance(point, entry) if entry is not None else 0.0
        scored.append((distance, point))
    scored.sort(key=lambda item: item[0], reverse=True)
    top_points = [point for _distance, point in scored[: max(1, min(12, len(scored)))]]
    bbox = _bbox_from_float_points(top_points)
    max_distance = scored[0][0] if scored else 0.0
    return [
        {
            "id": "privacy_back_zone",
            "bbox_mm": bbox,
            "centroid_mm": _point_from_float(_average_float_points(top_points)),
            "score": round(min(1.0, max_distance / 5000.0), 3),
        }
    ]


def _focal_surfaces(
    *,
    usable_walls: Sequence[dict[str, object]],
    blocked_walls: Sequence[dict[str, object]],
    windows: Sequence[NormalizedOpening],
) -> list[dict[str, object]]:
    blocked_ids = {str(row.get("wall_id")) for row in blocked_walls}
    window_wall_ids = {window["wall_id"] for window in windows}
    candidates: list[dict[str, object]] = []
    for segment in usable_walls:
        wall_id = str(segment.get("wall_id"))
        length = int(segment.get("length_mm") or 0)
        opening_penalty = 0.25 if wall_id in window_wall_ids else 0.0
        blocked_penalty = 0.2 if wall_id in blocked_ids else 0.0
        score = min(1.0, length / 3500.0) - opening_penalty - blocked_penalty
        candidates.append(
            {
                "id": f"{segment.get('id')}_focal",
                "wall_id": wall_id,
                "segment_mm": segment.get("segment_mm") or [],
                "score": round(max(0.0, score), 3),
                "reason": "long usable wall with low opening interruption",
            }
        )
    return sorted(candidates, key=lambda row: float(row["score"]), reverse=True)[:4]


def _center_openness_regions(
    *,
    free_points: Sequence[FloatPoint],
    centroid: FloatPoint,
) -> list[dict[str, object]]:
    if not free_points:
        return []
    sorted_points = sorted(
        free_points, key=lambda point: _float_distance(point, centroid)
    )
    core = sorted_points[: max(1, min(16, len(sorted_points)))]
    center = _average_float_points(core)
    return [
        {
            "id": "center_openness_core",
            "centroid_mm": _point_from_float(center),
            "bbox_mm": _bbox_from_float_points(core),
            "score": round(max(0.2, min(1.0, len(core) / 16.0)), 3),
        }
    ]


def _wall_anchor_candidates(
    *,
    usable_walls: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for segment in usable_walls:
        normal = segment.get("inward_normal")
        segment_points = segment.get("segment_mm")
        if (
            not isinstance(normal, dict)
            or not isinstance(segment_points, list)
            or len(segment_points) != 2
        ):
            continue
        p1 = _normalize_point(segment_points[0])
        p2 = _normalize_point(segment_points[1])
        if p1 is None or p2 is None:
            continue
        depth = 700
        polygon = _ensure_ccw(
            [
                p1,
                p2,
                _offset_point(p2, normal, depth),
                _offset_point(p1, normal, depth),
            ]
        )
        length = int(segment.get("length_mm") or 0)
        candidates.append(
            {
                "id": f"{segment.get('id')}_anchor",
                "wall_id": segment.get("wall_id"),
                "segment_mm": segment_points,
                "anchor_polygon_ccw": polygon,
                "max_backed_width_mm": length,
                "score": round(min(1.0, length / 3000.0), 3),
            }
        )
    return sorted(candidates, key=lambda row: float(row["score"]), reverse=True)[:8]


def _floating_zone_candidates(
    *,
    center_regions: Sequence[dict[str, object]],
    free_points: Sequence[FloatPoint],
    doors: Sequence[NormalizedOpening],
) -> list[dict[str, object]]:
    if not center_regions:
        return []
    entry = _primary_entry_point(doors)
    region = center_regions[0]
    center = _normalize_point(region.get("centroid_mm"))
    if center is None:
        center = _point_from_float(_average_float_points(free_points))
    entry_distance = (
        _point_distance(center, _point_from_float(entry)) if entry is not None else 9999
    )
    return [
        {
            "id": "floating_center_zone",
            "centroid_mm": center,
            "bbox_mm": region.get("bbox_mm") or {},
            "score": round(min(1.0, max(0.2, entry_distance / 3500.0)), 3),
        }
    ]


def _circulation_corridors(
    *,
    doors: Sequence[NormalizedOpening],
    windows: Sequence[NormalizedOpening],
    free_points: Sequence[FloatPoint],
    centroid: FloatPoint,
) -> list[dict[str, object]]:
    entry = _primary_entry_point(doors)
    if entry is None:
        return []
    targets: list[tuple[str, FloatPoint]] = [("room_center", centroid)]
    if free_points:
        farthest = max(free_points, key=lambda point: _float_distance(point, entry))
        targets.append(("deep_room", farthest))
    for window in windows:
        targets.append(
            (f"window:{window['id']}", _segment_midpoint(window["segment_mm"]))
        )
    corridors: list[dict[str, object]] = []
    seen: set[str] = set()
    for target_id, target in targets:
        if target_id in seen:
            continue
        seen.add(target_id)
        width = 900
        corridors.append(
            {
                "id": f"corridor_{len(corridors) + 1}",
                "from": "entry",
                "to": target_id,
                "polyline_mm": [_point_from_float(entry), _point_from_float(target)],
                "width_mm": width,
                "score": round(min(1.0, _float_distance(entry, target) / 4000.0), 3),
            }
        )
        if len(corridors) >= CORRIDOR_CANDIDATE_MAX:
            break
    return corridors


def _build_topology(
    *,
    walls: Sequence[Wall],
    doors: Sequence[NormalizedOpening],
    windows: Sequence[NormalizedOpening],
    affordance_map: Mapping[str, object],
    free_points: Sequence[FloatPoint],
    centroid: FloatPoint,
) -> dict[str, object]:
    entry = _primary_entry_point(doors)
    corridor_rows = affordance_map.get("circulation_corridors")
    corridors = corridor_rows if isinstance(corridor_rows, list) else []
    nodes: list[dict[str, object]] = []
    if entry is not None:
        nodes.append(
            {"id": "entry", "type": "entry", "point_mm": _point_from_float(entry)}
        )
    nodes.append(
        {"id": "room_center", "type": "center", "point_mm": _point_from_float(centroid)}
    )
    for window in windows:
        nodes.append(
            {
                "id": f"window:{window['id']}",
                "type": "window",
                "point_mm": _point_from_float(_segment_midpoint(window["segment_mm"])),
                "opening_id": window["id"],
            }
        )
    return {
        "wall_graph": {
            "nodes": [
                {
                    "id": wall["id"],
                    "start_mm": wall["start_mm"],
                    "end_mm": wall["end_mm"],
                    "length_mm": wall["length_mm"],
                    "inward_normal": wall["inward_normal"],
                }
                for wall in walls
            ],
            "edges": [
                {"from": wall["id"], "to": adjacent_id, "type": "adjacent"}
                for wall in walls
                for adjacent_id in wall["adjacent_wall_ids"]
                if wall["id"] < adjacent_id
            ],
        },
        "entry_node": nodes[0] if entry is not None else None,
        "window_nodes": [node for node in nodes if node["type"] == "window"],
        "passage_graph": {
            "nodes": nodes,
            "edges": [
                {
                    "from": corridor.get("from"),
                    "to": corridor.get("to"),
                    "corridor_id": corridor.get("id"),
                    "width_mm": corridor.get("width_mm"),
                }
                for corridor in corridors
                if isinstance(corridor, dict)
            ],
        },
        "room_subzones": _room_subzones(free_points=free_points, centroid=centroid),
    }


def _room_subzones(
    *,
    free_points: Sequence[FloatPoint],
    centroid: FloatPoint,
) -> list[dict[str, object]]:
    if not free_points:
        return []
    quadrants: dict[str, list[FloatPoint]] = {
        "front_left": [],
        "front_right": [],
        "back_left": [],
        "back_right": [],
    }
    for point in free_points:
        horizontal = "left" if point["x"] <= centroid["x"] else "right"
        vertical = "front" if point["y"] <= centroid["y"] else "back"
        quadrants[f"{vertical}_{horizontal}"].append(point)
    zones: list[dict[str, object]] = []
    for name, points in quadrants.items():
        if not points:
            continue
        zones.append(
            {
                "id": f"zone_{name}",
                "label": name,
                "centroid_mm": _point_from_float(_average_float_points(points)),
                "bbox_mm": _bbox_from_float_points(points),
                "sample_count": len(points),
            }
        )
    return sorted(zones, key=lambda zone: int(zone["sample_count"]), reverse=True)[
        :ZONE_SPLIT_MAX
    ]


def _sample_free_points(
    *,
    polygon: Sequence[PointDict],
    hard_obstacles: Sequence[dict[str, object]],
    resolution_mm: int,
) -> list[FloatPoint]:
    if len(polygon) < 3:
        return []
    bbox = _bbox_from_points(polygon)
    obstacle_polygons = [
        _normalize_polygon_points(obstacle.get("polygon_ccw"))
        for obstacle in hard_obstacles
        if isinstance(obstacle.get("polygon_ccw"), list)
    ]
    points: list[FloatPoint] = []
    min_x = int(bbox["min_x"])
    max_x = int(bbox["max_x"])
    min_y = int(bbox["min_y"])
    max_y = int(bbox["max_y"])
    for x in range(min_x, max_x + 1, resolution_mm):
        for y in range(min_y, max_y + 1, resolution_mm):
            candidate = {"x": float(x), "y": float(y)}
            if not _point_in_polygon(candidate, polygon):
                continue
            if any(
                _point_in_polygon(candidate, obstacle) for obstacle in obstacle_polygons
            ):
                continue
            points.append(candidate)
    if points:
        return points
    centroid = _polygon_centroid(polygon)
    if _point_in_polygon(centroid, polygon):
        return [centroid]
    return []


def _soft_usage_hints(
    input_payload: Mapping[str, object],
    *,
    room_type: str,
    description: str | None,
    special_notes: str | None,
) -> dict[str, object]:
    text_parts: list[str] = [room_type, description or "", special_notes or ""]
    brief_hints = input_payload.get("brief_hints")
    if isinstance(brief_hints, str):
        text_parts.append(brief_hints)
    elif isinstance(brief_hints, list):
        text_parts.extend(str(item) for item in brief_hints if isinstance(item, str))
    user_input = input_payload.get("user_input")
    if isinstance(user_input, dict):
        for key in (
            "brief_hints",
            "description",
            "special_description",
            "special_notes",
            "notes",
        ):
            value = user_input.get(key)
            if isinstance(value, str):
                text_parts.append(value)
    text = " ".join(text_parts).lower()
    hints: dict[str, object] = {
        "active_entry": bool(
            "entry" in text or "entrance" in text or "circulation" in text
        ),
        "prefer_private_back_zone": bool(
            room_type in {"bedroom", "bathroom"}
            or any(term in text for term in ("private", "privacy", "bed", "sleep"))
        ),
        "prefer_daylit_work_zone": bool(
            any(
                term in text
                for term in ("work", "desk", "study", "reading", "daylight", "window")
            )
        ),
        "prefer_open_center": bool(
            any(
                term in text
                for term in (
                    "open",
                    "center",
                    "clear circulation",
                    "uncluttered",
                    "avoid clutter",
                )
            )
        ),
        "likely_primary_focus": _likely_primary_focus(text=text, room_type=room_type),
    }
    return hints


def _likely_primary_focus(*, text: str, room_type: str) -> str:
    if any(term in text for term in ("tv", "media", "screen")):
        return "media_wall"
    if any(term in text for term in ("view", "window", "daylight")):
        return "window_view"
    if any(term in text for term in ("bed", "sleep")) or room_type == "bedroom":
        return "bed_back_wall"
    if (
        any(term in text for term in ("sofa", "living", "conversation"))
        or room_type == "living_room"
    ):
        return "seating_focus"
    return "longest_usable_wall"


def _legacy_notes(
    *,
    input_payload: Mapping[str, object],
    room_type: str,
    style: str,
    description: str | None,
    special_notes: str | None,
    soft_usage_hints: Mapping[str, object],
) -> list[str]:
    summary = _build_summary_input(
        input_payload,
        description=description,
        special_notes=special_notes,
    )
    notes = _fallback_guidance_notes(summary)
    if room_type and room_type != "unknown":
        notes.insert(0, f"Design target room type: {room_type}.")
    if style and style != "unspecified":
        notes.insert(1, f"Requested style: {style}.")
    for key, value in soft_usage_hints.items():
        if isinstance(value, bool) and value:
            notes.append(f"Soft usage hint: {key}.")
        elif key == "likely_primary_focus" and isinstance(value, str) and value:
            notes.append(f"Soft usage hint: likely_primary_focus={value}.")
    return _dedupe_strings(notes)[:8]


def _build_summary_input(
    input_payload: Mapping[str, object],
    *,
    description: str | None,
    special_notes: str | None,
) -> dict[str, object]:
    user_input = input_payload.get("user_input")
    user_input = user_input if isinstance(user_input, dict) else {}

    guidance: dict[str, str] = {}
    if description and description.strip():
        guidance["description"] = description.strip()
    if special_notes and special_notes.strip():
        guidance["special_notes"] = special_notes.strip()

    for key, value in user_input.items():
        if isinstance(value, str) and value.strip():
            guidance[key] = value.strip()

    constraints = input_payload.get("constraints")
    if isinstance(constraints, dict):
        feng_shui = constraints.get("feng_shui")
        if isinstance(feng_shui, str) and feng_shui.strip():
            guidance.setdefault("feng_shui", feng_shui.strip())

    return {
        "room_context": {
            "room_type": _resolve_room_type(input_payload),
            "style": _resolve_style(input_payload),
            "window_direction": _resolve_window_direction(input_payload),
        },
        "user_guidance": guidance,
    }


def _fallback_guidance_notes(summary_input: Mapping[str, object]) -> list[str]:
    notes: list[str] = []
    guidance = summary_input.get("user_guidance")
    if isinstance(guidance, dict):
        for key in (
            "description",
            "special_description",
            "special_notes",
            "notes",
            "feng_shui",
        ):
            value = guidance.get(key)
            if isinstance(value, str) and value.strip():
                notes.append(value.strip())
    return _dedupe_strings(notes)[:6]


def _resolve_floorplan_openings(
    input_payload: Mapping[str, object],
) -> dict[str, list[dict[str, object]]]:
    top_doors = input_payload.get("doors")
    top_windows = input_payload.get("windows")
    if isinstance(top_doors, list) or isinstance(top_windows, list):
        return {
            "doors": top_doors if isinstance(top_doors, list) else [],
            "windows": top_windows if isinstance(top_windows, list) else [],
        }
    floorplan = input_payload.get("floorplan_geometry")
    if isinstance(floorplan, dict):
        openings = floorplan.get("openings")
        if isinstance(openings, dict):
            doors = openings.get("doors")
            windows = openings.get("windows")
            return {
                "doors": doors if isinstance(doors, list) else [],
                "windows": windows if isinstance(windows, list) else [],
            }
    constraints = input_payload.get("constraints")
    if isinstance(constraints, dict):
        openings = constraints.get("openings")
        if isinstance(openings, dict):
            doors = openings.get("doors")
            windows = openings.get("windows")
            return {
                "doors": doors if isinstance(doors, list) else [],
                "windows": windows if isinstance(windows, list) else [],
            }
    return {"doors": [], "windows": []}


def _opening_kind_explicitly_provided(
    input_payload: Mapping[str, object],
    *,
    kind: Literal["doors", "windows"],
) -> bool:
    top_value = input_payload.get(kind)
    if isinstance(top_value, list):
        return True

    for container_key in ("floorplan_geometry", "constraints"):
        container = input_payload.get(container_key)
        if not isinstance(container, Mapping):
            continue
        openings = container.get("openings")
        if isinstance(openings, Mapping) and isinstance(openings.get(kind), list):
            return True

    if kind != "windows":
        return False

    user_input = input_payload.get("user_input")
    if not isinstance(user_input, Mapping):
        return False
    raw_count = user_input.get("windows")
    if isinstance(raw_count, bool):
        return False
    try:
        return int(raw_count) <= 0
    except (TypeError, ValueError):
        return False


def _resolve_room_polygon(input_payload: Mapping[str, object]) -> list[object] | None:
    polygon = input_payload.get("polygon_mm")
    if isinstance(polygon, list) and len(polygon) >= 3:
        return polygon
    floorplan = input_payload.get("floorplan_geometry")
    if isinstance(floorplan, dict):
        room = floorplan.get("room")
        if isinstance(room, dict):
            poly = room.get("polygon_mm")
            if isinstance(poly, list) and len(poly) >= 3:
                return poly
    user_input = input_payload.get("user_input")
    if isinstance(user_input, dict):
        poly = user_input.get("shape_points")
        if isinstance(poly, list) and len(poly) >= 3:
            return poly
    return None


def _room_polygon_conflicts(input_payload: Mapping[str, object]) -> bool:
    floorplan = input_payload.get("floorplan_geometry")
    user_input = input_payload.get("user_input")
    if not isinstance(floorplan, dict) or not isinstance(user_input, dict):
        return False
    room = floorplan.get("room")
    if not isinstance(room, dict):
        return False
    floorplan_polygon = room.get("polygon_mm")
    shape_points = user_input.get("shape_points")
    if not isinstance(floorplan_polygon, list) or not isinstance(shape_points, list):
        return False
    normalized_floorplan, _notes, _conflicts = _sanitize_polygon(floorplan_polygon)
    normalized_shape, _shape_notes, _shape_conflicts = _sanitize_polygon(shape_points)
    return (
        len(normalized_floorplan) >= 3
        and len(normalized_shape) >= 3
        and normalized_floorplan != normalized_shape
    )


def _resolve_room_id(input_payload: Mapping[str, object]) -> str:
    floorplan = input_payload.get("floorplan_geometry")
    if isinstance(floorplan, dict):
        room = floorplan.get("room")
        if isinstance(room, dict):
            room_id = room.get("room_id")
            if isinstance(room_id, str) and room_id:
                return room_id
    room_id = input_payload.get("room_id")
    return room_id if isinstance(room_id, str) and room_id else "room_1"


def _resolve_room_name(input_payload: Mapping[str, object]) -> str:
    floorplan = input_payload.get("floorplan_geometry")
    if isinstance(floorplan, dict):
        room = floorplan.get("room")
        if isinstance(room, dict):
            name = room.get("name")
            if isinstance(name, str) and name:
                return name
    name = input_payload.get("name")
    return name if isinstance(name, str) and name else "Main"


def _resolve_room_type(input_payload: Mapping[str, object]) -> str:
    value = input_payload.get("room_type")
    if isinstance(value, str) and value:
        return value
    user_input = input_payload.get("user_input")
    if isinstance(user_input, dict):
        room_type = user_input.get("room_type")
        if isinstance(room_type, str) and room_type:
            return room_type
    return "unknown"


def _resolve_style(input_payload: Mapping[str, object]) -> str:
    user_input = input_payload.get("user_input")
    if isinstance(user_input, dict):
        style = user_input.get("style")
        if isinstance(style, str) and style:
            return style
    style = input_payload.get("style")
    return style if isinstance(style, str) and style else "unspecified"


def _resolve_height_mm(input_payload: Mapping[str, object]) -> int:
    for key in ("ceiling_height_mm", "height_mm", "height"):
        value = input_payload.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return int(round(value))
    user_input = input_payload.get("user_input")
    if isinstance(user_input, dict):
        for key in ("ceiling_height_mm", "height_mm", "height"):
            height = user_input.get(key)
            if isinstance(height, (int, float)) and height > 0:
                return int(round(height))
    return DEFAULT_ROOM_HEIGHT_MM


def _resolve_window_direction(input_payload: Mapping[str, object]) -> str:
    user_input = input_payload.get("user_input")
    if isinstance(user_input, dict):
        direction = user_input.get("window_direction")
        if isinstance(direction, str):
            return direction
    direction = input_payload.get("window_direction")
    return direction if isinstance(direction, str) else ""


def _normalize_polygon_points(value: object) -> list[PointDict]:
    if not isinstance(value, list):
        return []
    points: list[PointDict] = []
    for item in value:
        point = _normalize_point(item)
        if point is not None:
            points.append(point)
    return points


def _normalize_point(item: object) -> PointDict | None:
    if not isinstance(item, dict):
        return None
    x = item.get("x")
    y = item.get("y")
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return {"x": int(round(x)), "y": int(round(y))}
    return None


def _normalize_segment(
    value: object,
    *,
    start: object | None = None,
    end: object | None = None,
) -> list[PointDict] | None:
    if isinstance(value, list) and len(value) == 2:
        p1 = _normalize_point(value[0])
        p2 = _normalize_point(value[1])
        if p1 is not None and p2 is not None:
            return [p1, p2]
    p1 = _normalize_point(start)
    p2 = _normalize_point(end)
    if p1 is not None and p2 is not None:
        return [p1, p2]
    return None


def _ensure_ccw(points: Sequence[PointDict]) -> list[PointDict]:
    out = [dict(point) for point in points]
    if len(out) < 3:
        return out
    return list(reversed(out)) if _signed_area(out) < 0 else out


def _signed_area(points: Sequence[Mapping[str, int]]) -> float:
    total = 0.0
    for idx, point in enumerate(points):
        nxt = points[(idx + 1) % len(points)]
        total += point["x"] * nxt["y"] - nxt["x"] * point["y"]
    return 0.5 * total


def _polygon_centroid(points: Sequence[PointDict]) -> FloatPoint:
    if len(points) < 3:
        return {"x": 0.0, "y": 0.0}
    area_factor = 0.0
    cx = 0.0
    cy = 0.0
    for idx, point in enumerate(points):
        nxt = points[(idx + 1) % len(points)]
        cross = point["x"] * nxt["y"] - nxt["x"] * point["y"]
        area_factor += cross
        cx += (point["x"] + nxt["x"]) * cross
        cy += (point["y"] + nxt["y"]) * cross
    if abs(area_factor) < 1e-9:
        return _average_float_points(
            [{"x": float(p["x"]), "y": float(p["y"])} for p in points]
        )
    return {"x": cx / (3.0 * area_factor), "y": cy / (3.0 * area_factor)}


def _principal_axis(points: Sequence[PointDict]) -> dict[str, object]:
    if len(points) < 2:
        return {"angle_deg": 0, "vector": {"x": 1.0, "y": 0.0}, "confidence": 0.0}
    best_wall = max(
        (
            (
                math.hypot(
                    points[(idx + 1) % len(points)]["x"] - point["x"],
                    points[(idx + 1) % len(points)]["y"] - point["y"],
                ),
                point,
                points[(idx + 1) % len(points)],
            )
            for idx, point in enumerate(points)
        ),
        key=lambda item: item[0],
    )
    length, start, end = best_wall
    if length <= 0:
        return {"angle_deg": 0, "vector": {"x": 1.0, "y": 0.0}, "confidence": 0.0}
    vx = (end["x"] - start["x"]) / length
    vy = (end["y"] - start["y"]) / length
    angle = math.degrees(math.atan2(vy, vx))
    perimeter = sum(
        math.hypot(
            points[(idx + 1) % len(points)]["x"] - point["x"],
            points[(idx + 1) % len(points)]["y"] - point["y"],
        )
        for idx, point in enumerate(points)
    )
    return {
        "angle_deg": int(round(angle)),
        "vector": {"x": round(vx, 4), "y": round(vy, 4)},
        "confidence": round(length / perimeter if perimeter > 0 else 0.0, 3),
    }


def _bbox_from_points(points: Sequence[Mapping[str, int]]) -> dict[str, int]:
    if not points:
        return {"min_x": 0, "min_y": 0, "max_x": 0, "max_y": 0}
    xs = [int(point["x"]) for point in points]
    ys = [int(point["y"]) for point in points]
    return {"min_x": min(xs), "min_y": min(ys), "max_x": max(xs), "max_y": max(ys)}


def _bbox_from_float_points(points: Sequence[FloatPoint]) -> dict[str, int]:
    if not points:
        return {"min_x": 0, "min_y": 0, "max_x": 0, "max_y": 0}
    xs = [point["x"] for point in points]
    ys = [point["y"] for point in points]
    return {
        "min_x": int(round(min(xs))),
        "min_y": int(round(min(ys))),
        "max_x": int(round(max(xs))),
        "max_y": int(round(max(ys))),
    }


def _point_on_wall(wall: Wall, t_mm: float) -> PointDict:
    start = wall["start_mm"]
    direction = wall["direction"]
    return {
        "x": int(round(start["x"] + direction["x"] * t_mm)),
        "y": int(round(start["y"] + direction["y"] * t_mm)),
    }


def _offset_point(
    point: Mapping[str, int], normal: Mapping[str, object], distance_mm: int
) -> PointDict:
    nx = float(normal.get("x") or 0.0)
    ny = float(normal.get("y") or 0.0)
    return {
        "x": int(round(point["x"] + nx * distance_mm)),
        "y": int(round(point["y"] + ny * distance_mm)),
    }


def _point_distance(a: Mapping[str, int], b: Mapping[str, int]) -> float:
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def _float_distance(a: FloatPoint, b: FloatPoint) -> float:
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def _point_from_float(point: FloatPoint | None) -> PointDict:
    if point is None:
        return {"x": 0, "y": 0}
    return {"x": int(round(point["x"])), "y": int(round(point["y"]))}


def _average_float_points(points: Sequence[FloatPoint]) -> FloatPoint:
    if not points:
        return {"x": 0.0, "y": 0.0}
    return {
        "x": sum(point["x"] for point in points) / len(points),
        "y": sum(point["y"] for point in points) / len(points),
    }


def _point_in_polygon(point: FloatPoint, polygon: Sequence[Mapping[str, int]]) -> bool:
    if len(polygon) < 3:
        return False
    x = point["x"]
    y = point["y"]
    inside = False
    for idx, p1 in enumerate(polygon):
        p2 = polygon[(idx + 1) % len(polygon)]
        y1 = p1["y"]
        y2 = p2["y"]
        if (y1 > y) == (y2 > y):
            continue
        denom = y2 - y1
        if abs(denom) < 1e-9:
            continue
        x_intersection = (p2["x"] - p1["x"]) * (y - y1) / denom + p1["x"]
        if x < x_intersection:
            inside = not inside
    return inside


def _segment_midpoint(segment: Sequence[Mapping[str, int]]) -> FloatPoint:
    if len(segment) != 2:
        return {"x": 0.0, "y": 0.0}
    return {
        "x": (segment[0]["x"] + segment[1]["x"]) / 2.0,
        "y": (segment[0]["y"] + segment[1]["y"]) / 2.0,
    }


def _primary_entry_point(doors: Sequence[NormalizedOpening]) -> FloatPoint | None:
    if not doors:
        return None
    return _segment_midpoint(doors[0]["segment_mm"])


def _coerce_identifier(value: object, *, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def _coerce_positive_int(value: object, *, fallback: int) -> int:
    if isinstance(value, (int, float)) and float(value) > 0:
        return int(round(float(value)))
    return fallback


def _coerce_optional_int(value: object) -> int | None:
    if isinstance(value, (int, float)):
        return int(round(value))
    return None


def _normalize_hinge_hint(value: object) -> Literal["UNKNOWN", "LEFT", "RIGHT"]:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"left", "l"}:
            return "LEFT"
        if normalized in {"right", "r"}:
            return "RIGHT"
    return "UNKNOWN"


def _ensure_unique_identifier(identifier: str, existing_ids: set[str]) -> str:
    if identifier not in existing_ids:
        return identifier
    counter = 1
    while True:
        candidate = f"{identifier}_{counter}"
        if candidate not in existing_ids:
            return candidate
        counter += 1


def _dedupe_strings(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = value.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _status_from_quality(
    *,
    missing: Sequence[str],
    conflicts: Sequence[str],
) -> Literal["OK", "NEED_INFO", "CONFLICT"]:
    if missing:
        return "NEED_INFO"
    if conflicts:
        return "CONFLICT"
    return "OK"


def _confidence_score(
    *,
    polygon: Sequence[PointDict],
    doors: Sequence[NormalizedOpening],
    windows: Sequence[NormalizedOpening],
    usable_walls: Sequence[dict[str, object]],
    missing: Sequence[str],
    conflicts: Sequence[str],
) -> float:
    score = 1.0
    if len(polygon) < 3:
        score -= 0.5
    if not doors:
        score -= 0.15
    if not windows:
        score -= 0.1
    if not usable_walls:
        score -= 0.2
    score -= min(0.3, len(missing) * 0.1)
    score -= min(0.25, len(conflicts) * 0.05)
    return round(max(0.0, min(1.0, score)), 3)


def _settings_payload() -> dict[str, int | float]:
    return {
        "grid_mm": GRID_MM,
        "opening_snap_tolerance_mm": OPENING_SNAP_TOLERANCE_MM,
        "corner_exclusion_mm": CORNER_EXCLUSION_MM,
        "door_front_clearance_mm": DOOR_FRONT_CLEARANCE_MM,
        "door_side_margin_mm": DOOR_SIDE_MARGIN_MM,
        "window_clearance_mm": WINDOW_CLEARANCE_MM,
        "daylight_near_depth_mm": DAYLIGHT_NEAR_DEPTH_MM,
        "daylight_near_side_margin_mm": DAYLIGHT_NEAR_SIDE_MARGIN_MM,
        "daylight_mid_depth_mm": DAYLIGHT_MID_DEPTH_MM,
        "daylight_mid_side_margin_mm": DAYLIGHT_MID_SIDE_MARGIN_MM,
        "min_wall_len_mm": MIN_WALL_LEN_MM,
        "min_usable_wall_len_mm": MIN_USABLE_WALL_LEN_MM,
        "affordance_field_resolution_mm": AFFORDANCE_FIELD_RESOLUTION_MM,
        "corridor_candidate_max": CORRIDOR_CANDIDATE_MAX,
        "zone_split_max": ZONE_SPLIT_MAX,
        "llm_temperature": LLM_TEMPERATURE,
        "llm_retry_max": LLM_RETRY_MAX,
    }
