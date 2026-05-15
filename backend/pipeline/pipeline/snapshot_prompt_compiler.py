from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


PROMPT_COMPILER_INCLUDE_MINOR_DECOR_ENV = "TKNT_PROMPT_COMPILER_INCLUDE_MINOR_DECOR"

ViewElevation = Literal["top_down", "high_oblique", "oblique", "eye_level"]
ViewAzimuth = Literal[
    "front",
    "front_right",
    "right",
    "back_right",
    "back",
    "back_left",
    "left",
    "front_left",
    "centered",
]
RoomShape = Literal["rectangular", "L_shaped", "T_shaped", "irregular", "unknown"]
ShellMode = Literal["canonical", "localized", "omit"]
ScreenRegion = Literal["left", "center", "right"]
DepthBand = Literal["foreground", "midground", "background"]
PromptRole = Literal["anchor", "supporting", "minor"]

_DECOR_TYPES: set[str] = {
    "art",
    "books",
    "candle",
    "ceiling_light",
    "decor",
    "figurine",
    "picture_frame",
    "plant",
    "sculpture",
    "table_lamp",
    "vase",
    "wall_art",
}

_NON_ANCHOR_LARGE_TYPES: set[str] = {
    "ceiling_light",
    "curtain",
    "tv",
    "vase",
    "wall_art",
}

_SEMANTIC_WEIGHTS: dict[str, float] = {
    "armchair": 5.2,
    "bed": 9.6,
    "bookshelf": 4.2,
    "ceiling_light": 1.1,
    "chair": 4.8,
    "coffee_table": 5.2,
    "console_table": 4.1,
    "curtain": 3.2,
    "desk": 7.1,
    "dining_table": 8.2,
    "floor_lamp": 3.4,
    "media_shelf": 4.3,
    "ottoman": 3.6,
    "pet_bed": 2.4,
    "recliner": 5.4,
    "rug": 4.5,
    "sectional_sofa": 9.4,
    "shelf": 4.2,
    "side_table": 2.8,
    "sofa": 8.7,
    "storage_cabinet": 4.8,
    "table": 5.0,
    "tv": 4.7,
    "tv_console": 8.0,
    "vase": 0.8,
    "wall_art": 0.9,
    "wardrobe": 7.8,
}

_TYPE_LABELS: dict[str, str] = {
    "armchair": "armchair",
    "bookshelf": "bookshelf",
    "ceiling_light": "ceiling light",
    "coffee_table": "coffee table",
    "console_table": "console table",
    "floor_lamp": "floor lamp",
    "media_shelf": "media shelf",
    "pet_bed": "pet bed",
    "recliner": "recliner",
    "sectional_sofa": "sectional sofa",
    "side_table": "side table",
    "storage_cabinet": "storage cabinet",
    "tv_console": "TV console",
    "wall_art": "wall art",
}


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float


@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class Bounds2D:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def width(self) -> float:
        return max(0.0, self.max_x - self.min_x)

    @property
    def height(self) -> float:
        return max(0.0, self.max_y - self.min_y)

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(frozen=True)
class Dimensions3D:
    width: float
    depth: float
    height: float


@dataclass(frozen=True)
class OpeningRecord:
    id: str
    segment: tuple[Point2D, Point2D]


@dataclass(frozen=True)
class SnapshotObject:
    id: str
    raw_type: str
    canonical_type: str
    label: str
    color_hex: str | None
    color_phrase: str
    material: str | None
    plan_position_mm: Point3D
    dimensions_mm: Dimensions3D
    plan_rotation_deg: float
    room_compass: str
    room_position_phrase: str
    facing_direction: str | None
    distance_to_camera_m: float
    visible_sample_fraction: float
    screen_bbox_px: Bounds2D | None
    screen_center_px: Point2D | None
    screen_region: ScreenRegion
    depth_band: DepthBand
    semantic_weight: float
    salience: float
    role: PromptRole
    layout_critical: bool
    place_on_method: str | None
    place_on_target_id: str | None


@dataclass(frozen=True)
class OcclusionRelation:
    occluder_id: str
    occluded_id: str
    score: float
    overlap_ratio: float
    region: str
    sentence: str


@dataclass(frozen=True)
class SnapshotPromptCompilerConfig:
    include_minor_decor: bool = False

    @classmethod
    def from_env(cls) -> SnapshotPromptCompilerConfig:
        return cls(
            include_minor_decor=_read_env_flag(
                PROMPT_COMPILER_INCLUDE_MINOR_DECOR_ENV,
                default=False,
            )
        )


def compile_snapshot_prompt(
    snapshot_payload: Mapping[str, object],
    *,
    config: SnapshotPromptCompilerConfig | None = None,
) -> dict[str, object]:
    compiler_config = config or SnapshotPromptCompilerConfig.from_env()
    normalized = _normalize_snapshot(snapshot_payload)
    scene_objects = _build_scene_objects(normalized)
    occlusions = _extract_occlusion_relations(scene_objects)
    scene_objects = _apply_layout_critical_overrides(scene_objects, occlusions)
    selection = _select_prompt_objects(scene_objects, compiler_config=compiler_config)
    view_summary = _build_view_summary(normalized)
    room_summary = _build_room_summary(normalized, elevation=view_summary["elevation"])
    opening_summary = _build_opening_summary(normalized)
    layout_facts = _build_layout_fact_bundle(
        normalized=normalized,
        objects=scene_objects,
        view_summary=view_summary,
        room_summary=room_summary,
        opening_summary=opening_summary,
    )
    prompt_sections = _render_prompt_sections(
        view_summary=view_summary,
        room_summary=room_summary,
        opening_summary=opening_summary,
        selection=selection,
        occlusions=occlusions,
        normalized=normalized,
        compiler_config=compiler_config,
        layout_facts=layout_facts,
    )

    scene_ir = {
        "view": view_summary,
        "room_shell": room_summary,
        "openings": opening_summary,
        "visible_object_count": normalized["visible_object_count"],
        "total_object_count": normalized["total_object_count"],
        "anchor_objects": [asdict(item) for item in selection["anchors"]],
        "supporting_objects": [asdict(item) for item in selection["supporting"]],
        "minor_objects_included": [
            asdict(item) for item in selection["included_minor"]
        ],
        "minor_objects_dropped": [asdict(item) for item in selection["dropped_minor"]],
        "occlusion_relations": [asdict(item) for item in occlusions],
        "opening_facts": list(layout_facts["opening_facts"]),
        "object_facts": list(layout_facts["object_facts"]),
        "spatial_relation_facts": list(layout_facts["spatial_relation_facts"]),
        "strict_layout_facts": list(layout_facts["strict_layout_facts"]),
    }

    return {
        "config": {
            "include_minor_decor": compiler_config.include_minor_decor,
            "minor_decor_env_var": PROMPT_COMPILER_INCLUDE_MINOR_DECOR_ENV,
            "minor_decor_policy": (
                "keep_visible_minor_decor"
                if compiler_config.include_minor_decor
                else "drop_non_layout_critical_minor_decor"
            ),
        },
        "scene_ir": scene_ir,
        "primary_prompt": prompt_sections["primary_prompt"],
        "layout_constraints": prompt_sections["layout_constraints"],
        "strict_layout_facts": list(layout_facts["strict_layout_facts"]),
        "negative_prompt": prompt_sections["negative_prompt"],
        "prompt_sections": {
            "camera": prompt_sections["camera"],
            "room_shell": prompt_sections["room_shell"],
            "openings": prompt_sections["openings"],
            "anchors": prompt_sections["anchors"],
            "supporting": prompt_sections["supporting"],
            "occlusion": prompt_sections["occlusion"],
            "layout_facts": prompt_sections["layout_facts"],
        },
    }


def compile_snapshot_prompt_from_path(
    snapshot_path: str | Path,
    *,
    config: SnapshotPromptCompilerConfig | None = None,
) -> dict[str, object]:
    path = Path(snapshot_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Snapshot file not found: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Snapshot file must contain a JSON object.")
    return compile_snapshot_prompt(payload, config=config)


def _normalize_snapshot(snapshot_payload: Mapping[str, object]) -> dict[str, object]:
    room_payload = _mapping(snapshot_payload.get("room"))
    room_bounds = _bounds_from_payload(room_payload.get("boundsMm"))
    visible_objects_payload = _sequence(snapshot_payload.get("visibleObjects"))
    all_objects_payload = _sequence(snapshot_payload.get("allObjects"))
    canvas_payload = _mapping(snapshot_payload.get("canvas"))
    camera_payload = _mapping(snapshot_payload.get("camera"))
    openings_payload = _mapping(room_payload.get("openings"))

    polygon = _polygon_from_payload(
        room_payload.get("polygonMm")
        or room_payload.get("polygon_mm")
        or room_payload.get("polygonCcw")
        or room_payload.get("polygon_ccw")
        or room_payload.get("polygon")
    )
    doors = _parse_openings(openings_payload.get("doors"))
    windows = _parse_openings(openings_payload.get("windows"))

    return {
        "captured_at": _string(snapshot_payload.get("capturedAt")),
        "variant_id": _string(snapshot_payload.get("variantId")),
        "total_object_count": _int(
            snapshot_payload.get("totalObjectCount"), len(all_objects_payload)
        ),
        "visible_object_count": _int(
            snapshot_payload.get("visibleObjectCount"),
            len(visible_objects_payload),
        ),
        "room_bounds": room_bounds,
        "room_polygon": polygon,
        "room_surface_colors": {
            "wall_color_hex": _string(
                _mapping(room_payload.get("surfaceColors")).get("wallColorHex")
            ),
            "floor_color_hex": _string(
                _mapping(room_payload.get("surfaceColors")).get("floorColorHex")
            ),
            "ceiling_color_hex": _string(
                _mapping(room_payload.get("surfaceColors")).get("ceilingColorHex")
            ),
        },
        "camera_position_room_mm": _point3d_from_payload(
            camera_payload.get("positionRoomMm")
        ),
        "camera_target_room_mm": _point3d_from_payload(
            camera_payload.get("targetRoomMm")
        ),
        "camera_fov_deg": _float(camera_payload.get("fovDeg"), 45.0),
        "camera_aspect": _float(camera_payload.get("aspect"), 1.6),
        "canvas_width_px": max(1, _int(canvas_payload.get("widthPx"), 1)),
        "canvas_height_px": max(1, _int(canvas_payload.get("heightPx"), 1)),
        "doors": doors,
        "windows": windows,
        "visible_objects_payload": visible_objects_payload,
    }


def _build_scene_objects(normalized: Mapping[str, object]) -> list[SnapshotObject]:
    visible_payload = _sequence(normalized.get("visible_objects_payload"))
    canvas_width_px = max(1, _int(normalized.get("canvas_width_px"), 1))
    canvas_height_px = max(1, _int(normalized.get("canvas_height_px"), 1))
    canvas_area = float(canvas_width_px * canvas_height_px)
    room_bounds = (
        normalized.get("room_bounds")
        if isinstance(normalized.get("room_bounds"), Bounds2D)
        else None
    )
    raw_objects: list[dict[str, object]] = []

    for item in visible_payload:
        payload = _mapping(item)
        canonical_type = _canonicalize_object_type(_string(payload.get("type")))
        label = _type_label(canonical_type)
        screen_bbox = _bounds_from_payload(payload.get("screenBboxPx"))
        center_payload = _mapping(payload.get("screenCenterPx"))
        screen_center = (
            Point2D(
                x=_float(center_payload.get("x")),
                y=_float(center_payload.get("y")),
            )
            if center_payload
            else None
        )
        if screen_center is None and screen_bbox is not None:
            screen_center = Point2D(
                x=(screen_bbox.min_x + screen_bbox.max_x) / 2.0,
                y=(screen_bbox.min_y + screen_bbox.max_y) / 2.0,
            )
        area_ratio = (
            (screen_bbox.area / canvas_area) if screen_bbox is not None else 0.0
        )
        center_score = _center_score(screen_center, canvas_width_px, canvas_height_px)
        semantic_weight = _SEMANTIC_WEIGHTS.get(canonical_type, 3.2)
        distance_to_camera_m = max(0.1, _float(payload.get("distanceToCameraM"), 1.0))
        proximity_score = 1.0 / distance_to_camera_m
        salience = round(
            semantic_weight * 10.0
            + area_ratio * 900.0
            + center_score * 8.0
            + proximity_score * 25.0
            + _float(payload.get("visibleSampleFraction"), 0.0) * 8.0,
            4,
        )
        raw_objects.append(
            {
                "id": _string(payload.get("id"), fallback="object"),
                "raw_type": _string(payload.get("type"), fallback="object"),
                "canonical_type": canonical_type,
                "label": label,
                "color_hex": _string(payload.get("colorHex")),
                "color_phrase": _describe_color(_string(payload.get("colorHex"))),
                "material": _string(payload.get("material")),
                "plan_position_mm": _point3d_from_payload(
                    payload.get("planPositionMm")
                ),
                "dimensions_mm": _dimensions_from_payload(payload.get("dimensionsMm")),
                "plan_rotation_deg": _float(payload.get("planRotationDeg"), 0.0),
                "distance_to_camera_m": distance_to_camera_m,
                "visible_sample_fraction": _float(
                    payload.get("visibleSampleFraction"), 0.0
                ),
                "screen_bbox_px": screen_bbox,
                "screen_center_px": screen_center,
                "screen_region": _screen_region(screen_center, canvas_width_px),
                "semantic_weight": semantic_weight,
                "salience": salience,
                "area_ratio": area_ratio,
                "center_score": center_score,
                "place_on_method": _string(
                    _mapping(payload.get("placeOn")).get("method")
                ),
                "place_on_target_id": _string(
                    _mapping(payload.get("placeOn")).get("targetId")
                ),
            }
        )

    raw_objects.sort(key=lambda item: float(item["distance_to_camera_m"]))
    distance_values = [float(item["distance_to_camera_m"]) for item in raw_objects]
    distance_bands = _depth_bands(distance_values)
    scene_objects: list[SnapshotObject] = []

    for index, item in enumerate(raw_objects):
        canonical_type = str(item["canonical_type"])
        semantic_weight = float(item["semantic_weight"])
        area_ratio = float(item["area_ratio"])
        center_score = float(item["center_score"])
        plan_position = _point3d_or_origin(item["plan_position_mm"])
        dimensions = _dimensions_or_default(item["dimensions_mm"])
        plan_rotation_deg = float(item["plan_rotation_deg"])
        room_compass = _room_compass(plan_position, room_bounds)
        room_position_phrase = _room_position_phrase(
            position=plan_position,
            dimensions=dimensions,
            room_bounds=room_bounds,
        )
        facing_direction = _facing_direction_from_rotation(plan_rotation_deg)
        role: PromptRole
        if canonical_type in _NON_ANCHOR_LARGE_TYPES:
            role = "supporting" if canonical_type == "curtain" else "minor"
        elif semantic_weight >= 7.5 or (area_ratio >= 0.018 and semantic_weight >= 4.0):
            role = "anchor"
        elif canonical_type in _DECOR_TYPES and area_ratio < 0.012:
            role = "minor"
        elif semantic_weight <= 1.5 and area_ratio < 0.008:
            role = "minor"
        else:
            role = "supporting"

        layout_critical = bool(
            role == "anchor"
            or area_ratio >= 0.02
            or (center_score >= 0.82 and semantic_weight >= 4.0)
            or canonical_type in {"curtain", "rug", "tv"}
        )
        scene_objects.append(
            SnapshotObject(
                id=str(item["id"]),
                raw_type=str(item["raw_type"]),
                canonical_type=canonical_type,
                label=str(item["label"]),
                color_hex=item["color_hex"]
                if isinstance(item["color_hex"], str)
                else None,
                color_phrase=str(item["color_phrase"]),
                material=item["material"]
                if isinstance(item["material"], str)
                else None,
                plan_position_mm=plan_position,
                dimensions_mm=dimensions,
                plan_rotation_deg=plan_rotation_deg,
                room_compass=room_compass,
                room_position_phrase=room_position_phrase,
                facing_direction=facing_direction,
                distance_to_camera_m=float(item["distance_to_camera_m"]),
                visible_sample_fraction=float(item["visible_sample_fraction"]),
                screen_bbox_px=item["screen_bbox_px"]
                if isinstance(item["screen_bbox_px"], Bounds2D)
                else None,
                screen_center_px=item["screen_center_px"]
                if isinstance(item["screen_center_px"], Point2D)
                else None,
                screen_region=item["screen_region"]
                if item["screen_region"] in {"left", "center", "right"}
                else "center",
                depth_band=distance_bands[index],
                semantic_weight=semantic_weight,
                salience=float(item["salience"]),
                role=role,
                layout_critical=layout_critical,
                place_on_method=item["place_on_method"]
                if isinstance(item["place_on_method"], str)
                else None,
                place_on_target_id=item["place_on_target_id"]
                if isinstance(item["place_on_target_id"], str)
                else None,
            )
        )

    return scene_objects


def _extract_occlusion_relations(
    objects: Sequence[SnapshotObject],
) -> list[OcclusionRelation]:
    relations: list[OcclusionRelation] = []
    for left_index, occluder in enumerate(objects):
        if occluder.screen_bbox_px is None or _should_skip_occlusion_object(occluder):
            continue
        for occluded in objects[left_index + 1 :]:
            if occluded.screen_bbox_px is None or _should_skip_occlusion_object(
                occluded
            ):
                continue
            overlap_area = _intersection_area(
                occluder.screen_bbox_px, occluded.screen_bbox_px
            )
            if overlap_area <= 0.0:
                continue
            occluded_area = max(1.0, occluded.screen_bbox_px.area)
            overlap_ratio = overlap_area / occluded_area
            if overlap_ratio < 0.08:
                continue
            region = _occluded_region(occluder.screen_bbox_px, occluded.screen_bbox_px)
            intensity = (
                "heavily occludes"
                if overlap_ratio >= 0.35
                else "partially occludes"
                if overlap_ratio >= 0.18
                else "slightly occludes"
            )
            sentence = (
                f"The {_object_prompt_phrase(occluder, include_depth=True)} {intensity} "
                f"the {region} portion of the {_object_prompt_phrase(occluded, include_depth=False)} behind it."
            )
            score = round(
                overlap_ratio * 100.0
                + occluder.semantic_weight
                + occluded.semantic_weight
                + abs(occluded.distance_to_camera_m - occluder.distance_to_camera_m)
                * 2.0,
                4,
            )
            relations.append(
                OcclusionRelation(
                    occluder_id=occluder.id,
                    occluded_id=occluded.id,
                    score=score,
                    overlap_ratio=round(overlap_ratio, 4),
                    region=region,
                    sentence=sentence,
                )
            )
    relations.sort(key=lambda item: item.score, reverse=True)
    return relations[:4]


def _apply_layout_critical_overrides(
    objects: Sequence[SnapshotObject],
    occlusions: Sequence[OcclusionRelation],
) -> list[SnapshotObject]:
    occlusion_ids = {item.occluder_id for item in occlusions} | {
        item.occluded_id for item in occlusions
    }
    updated: list[SnapshotObject] = []
    for item in objects:
        layout_critical = item.layout_critical or item.id in occlusion_ids
        role = (
            "supporting"
            if item.role == "minor" and item.id in occlusion_ids
            else item.role
        )
        updated.append(
            SnapshotObject(
                id=item.id,
                raw_type=item.raw_type,
                canonical_type=item.canonical_type,
                label=item.label,
                color_hex=item.color_hex,
                color_phrase=item.color_phrase,
                material=item.material,
                plan_position_mm=item.plan_position_mm,
                dimensions_mm=item.dimensions_mm,
                plan_rotation_deg=item.plan_rotation_deg,
                room_compass=item.room_compass,
                room_position_phrase=item.room_position_phrase,
                facing_direction=item.facing_direction,
                distance_to_camera_m=item.distance_to_camera_m,
                visible_sample_fraction=item.visible_sample_fraction,
                screen_bbox_px=item.screen_bbox_px,
                screen_center_px=item.screen_center_px,
                screen_region=item.screen_region,
                depth_band=item.depth_band,
                semantic_weight=item.semantic_weight,
                salience=item.salience,
                role=role,
                layout_critical=layout_critical,
                place_on_method=item.place_on_method,
                place_on_target_id=item.place_on_target_id,
            )
        )
    return updated


def _select_prompt_objects(
    objects: Sequence[SnapshotObject],
    *,
    compiler_config: SnapshotPromptCompilerConfig,
) -> dict[str, list[SnapshotObject]]:
    sorted_objects = sorted(objects, key=lambda item: item.salience, reverse=True)
    anchors = [item for item in sorted_objects if item.role == "anchor"][:6]
    if len(anchors) < 3:
        for item in sorted_objects:
            if item in anchors or item.role == "minor":
                continue
            anchors.append(item)
            if len(anchors) >= 3:
                break

    included_minor: list[SnapshotObject] = []
    dropped_minor: list[SnapshotObject] = []
    supporting: list[SnapshotObject] = []

    for item in sorted_objects:
        if item in anchors:
            continue
        if item.role == "minor":
            if compiler_config.include_minor_decor or item.layout_critical:
                included_minor.append(item)
            else:
                dropped_minor.append(item)
            continue
        supporting.append(item)

    supporting = supporting[:6]
    included_minor = included_minor[:6]
    return {
        "anchors": anchors,
        "supporting": supporting,
        "included_minor": included_minor,
        "dropped_minor": dropped_minor,
    }


def _build_view_summary(normalized: Mapping[str, object]) -> dict[str, object]:
    camera = _point3d_or_origin(normalized.get("camera_position_room_mm"))
    target = _point3d_or_origin(normalized.get("camera_target_room_mm"))
    horizontal_distance = math.hypot(camera.x - target.x, camera.y - target.y)
    vertical_delta = max(0.0, camera.z - target.z)
    pitch_deg = math.degrees(math.atan2(vertical_delta, max(1.0, horizontal_distance)))
    if pitch_deg >= 72.0:
        elevation: ViewElevation = "top_down"
    elif pitch_deg >= 48.0:
        elevation = "high_oblique"
    elif pitch_deg >= 12.0:
        elevation = "oblique"
    else:
        elevation = "eye_level"

    azimuth = _azimuth_class(camera, target)
    shot_scale = _shot_scale(
        room_bounds=normalized.get("room_bounds"),
        camera=camera,
        target=target,
    )
    summary = {
        "elevation": elevation,
        "azimuth": azimuth,
        "shot_scale": shot_scale,
        "pitch_deg": round(pitch_deg, 2),
        "camera_summary": _camera_summary(elevation, azimuth, shot_scale),
        "fov_deg": round(_float(normalized.get("camera_fov_deg"), 45.0), 2),
    }
    return summary


def _build_room_summary(
    normalized: Mapping[str, object],
    *,
    elevation: ViewElevation,
) -> dict[str, object]:
    polygon = _sequence(normalized.get("room_polygon"))
    shape = _room_shape(polygon)
    room_bounds = (
        normalized.get("room_bounds")
        if isinstance(normalized.get("room_bounds"), Bounds2D)
        else None
    )
    missing_corner_compass = _room_missing_corner_compass(
        polygon=[item for item in polygon if isinstance(item, Point2D)],
        room_bounds=room_bounds,
    )
    if elevation == "top_down":
        mode: ShellMode = "canonical" if shape != "unknown" else "localized"
    elif elevation == "high_oblique":
        mode = (
            "canonical"
            if shape in {"rectangular", "L_shaped", "T_shaped"}
            else "localized"
        )
    elif elevation == "oblique":
        mode = "localized" if shape in {"L_shaped", "T_shaped", "irregular"} else "omit"
    else:
        mode = "omit" if shape in {"rectangular", "unknown"} else "localized"

    summary = _room_shell_summary(
        shape=shape,
        mode=mode,
        missing_corner_compass=missing_corner_compass,
    )
    shape_fact = _room_shape_fact(shape, missing_corner_compass)
    plan_orientation_fact = (
        "For plan-oriented layout facts, north is at the top, east is on the right, south is at the bottom, and west is on the left."
        if elevation in {"top_down", "high_oblique"}
        else ""
    )
    return {
        "canonical_shape": shape,
        "visible_shell_mode": mode,
        "summary": summary,
        "missing_corner_compass": missing_corner_compass,
        "shape_fact": shape_fact,
        "plan_orientation_fact": plan_orientation_fact,
    }


def _build_opening_summary(normalized: Mapping[str, object]) -> dict[str, object]:
    room_bounds = (
        normalized.get("room_bounds")
        if isinstance(normalized.get("room_bounds"), Bounds2D)
        else None
    )
    doors = [
        item
        for item in _sequence(normalized.get("doors"))
        if isinstance(item, OpeningRecord)
    ]
    windows = [
        item
        for item in _sequence(normalized.get("windows"))
        if isinstance(item, OpeningRecord)
    ]
    facts: list[str] = []
    short_descriptions: list[str] = []
    opening_records: list[dict[str, object]] = []

    for opening_type, items in (("door", doors), ("window", windows)):
        for index, item in enumerate(items, start=1):
            opening_info = _opening_layout_info(
                opening=item,
                opening_type=opening_type,
                room_bounds=room_bounds,
                ordinal=index,
            )
            facts.append(opening_info["fact"])
            short_descriptions.append(opening_info["short"])
            opening_records.append(opening_info)

    summary = ""
    if short_descriptions:
        summary = "The room shell includes " + _comma_join(short_descriptions) + "."
    return {
        "door_count": len(doors),
        "window_count": len(windows),
        "summary": summary,
        "facts": facts,
        "records": opening_records,
    }


def _build_layout_fact_bundle(
    *,
    normalized: Mapping[str, object],
    objects: Sequence[SnapshotObject],
    view_summary: Mapping[str, object],
    room_summary: Mapping[str, object],
    opening_summary: Mapping[str, object],
) -> dict[str, list[str]]:
    room_facts: list[str] = []
    shape_fact = _string(room_summary.get("shape_fact"))
    if shape_fact:
        room_facts.append(shape_fact)
    plan_orientation_fact = _string(room_summary.get("plan_orientation_fact"))
    if plan_orientation_fact:
        room_facts.append(plan_orientation_fact)

    room_bounds = (
        normalized.get("room_bounds")
        if isinstance(normalized.get("room_bounds"), Bounds2D)
        else None
    )
    if room_bounds is not None:
        room_facts.append(
            f"The room measures {round(room_bounds.width / 1000.0, 2)} m east-to-west by {round(room_bounds.height / 1000.0, 2)} m south-to-north."
        )

    surface_colors = _mapping(normalized.get("room_surface_colors"))
    room_facts.append(
        "Room surface palette: "
        f"walls {_string(surface_colors.get('wall_color_hex'), fallback='unknown')}, "
        f"floor {_string(surface_colors.get('floor_color_hex'), fallback='unknown')}, "
        f"ceiling {_string(surface_colors.get('ceiling_color_hex'), fallback='unknown')}."
    )
    room_facts.append(
        f"There are {_int(normalized.get('visible_object_count'), 0)} visible objects in frame."
    )

    opening_facts = [
        item
        for item in _sequence(opening_summary.get("facts"))
        if isinstance(item, str) and item.strip()
    ]

    opening_records = [
        item
        for item in _sequence(opening_summary.get("records"))
        if isinstance(item, Mapping)
    ]
    room_center = _room_center(room_bounds)
    object_facts = [
        _object_layout_fact(
            item,
            room_center=room_center,
            opening_records=opening_records,
        )
        for item in _objects_for_layout_facts(objects)
    ]
    spatial_relation_facts = _build_spatial_relation_facts(
        objects,
        room_bounds=room_bounds,
    )

    strict_layout_facts = (
        room_facts + opening_facts + object_facts + spatial_relation_facts
    )
    return {
        "room_facts": room_facts,
        "opening_facts": opening_facts,
        "object_facts": object_facts,
        "spatial_relation_facts": spatial_relation_facts,
        "strict_layout_facts": strict_layout_facts,
    }


def _render_prompt_sections(
    *,
    view_summary: Mapping[str, object],
    room_summary: Mapping[str, object],
    opening_summary: Mapping[str, object],
    selection: Mapping[str, list[SnapshotObject]],
    occlusions: Sequence[OcclusionRelation],
    normalized: Mapping[str, object],
    compiler_config: SnapshotPromptCompilerConfig,
    layout_facts: Mapping[str, list[str]],
) -> dict[str, object]:
    room_bounds = normalized.get("room_bounds")
    room_width_mm = room_bounds.width if isinstance(room_bounds, Bounds2D) else 0.0
    room_height_mm = room_bounds.height if isinstance(room_bounds, Bounds2D) else 0.0
    surface_colors = _mapping(normalized.get("room_surface_colors"))
    wall_phrase = _describe_color(_string(surface_colors.get("wall_color_hex")))
    floor_phrase = _describe_color(_string(surface_colors.get("floor_color_hex")))
    ceiling_phrase = _describe_color(_string(surface_colors.get("ceiling_color_hex")))
    camera_section = (
        f"{_string(view_summary.get('camera_summary'), fallback='Architectural interior view')}, "
        f"{round(room_width_mm / 1000.0, 2)} m by {round(room_height_mm / 1000.0, 2)} m room."
    )
    room_section_parts = [
        _string(room_summary.get("summary")),
        f"{wall_phrase} walls, {floor_phrase} floor, and {ceiling_phrase} ceiling.",
    ]
    room_section = " ".join(part for part in room_section_parts if part).strip()
    openings_section = _string(opening_summary.get("summary"))
    layout_fact_lines = [
        item
        for item in _sequence(layout_facts.get("strict_layout_facts"))
        if isinstance(item, str) and item.strip()
    ]

    anchor_sentences = [
        _anchor_sentence(item, first=index == 0)
        for index, item in enumerate(selection["anchors"])
    ]
    supporting_objects = selection["supporting"] + selection["included_minor"]
    supporting_section = ""
    if supporting_objects:
        supporting_names = [_supporting_label(item) for item in supporting_objects[:8]]
        supporting_section = (
            "Supporting visible items include " + _comma_join(supporting_names) + "."
        )

    occlusion_section = " ".join(item.sentence for item in occlusions[:3])
    primary_parts = [
        camera_section,
        room_section,
        openings_section,
        " ".join(anchor_sentences).strip(),
        occlusion_section,
        supporting_section,
        "Key layout facts: " + _comma_join(layout_fact_lines[:8]) + "."
        if layout_fact_lines
        else "",
        (
            "Preserve the relative spacing, visibility order, and approximate object count from the snapshot."
        ),
    ]
    primary_prompt = " ".join(part for part in primary_parts if part).strip()

    layout_constraints = [
        f"Visible object count in frame: {_int(normalized.get('visible_object_count'), 0)}.",
        (
            "Keep the anchor furniture in the same relative depth order: "
            + _comma_join(
                [
                    f"{item.label} ({item.depth_band}, {item.screen_region})"
                    for item in selection["anchors"][:5]
                ]
            )
            + "."
        ),
        (
            "Keep room surface colors aligned with the snapshot palette: "
            f"walls {_string(surface_colors.get('wall_color_hex'), fallback='unknown')}, "
            f"floor {_string(surface_colors.get('floor_color_hex'), fallback='unknown')}, "
            f"ceiling {_string(surface_colors.get('ceiling_color_hex'), fallback='unknown')}."
        ),
    ]
    layout_constraints.extend(layout_fact_lines[:14])
    if occlusions:
        layout_constraints.append(
            "Preserve the key occlusion relationships: "
            + _comma_join(
                [
                    f"{_object_prompt_phrase(_find_object(selection, item.occluder_id), include_depth=False)} over "
                    f"{_object_prompt_phrase(_find_object(selection, item.occluded_id), include_depth=False)}"
                    for item in occlusions[:3]
                    if _find_object(selection, item.occluder_id) is not None
                    and _find_object(selection, item.occluded_id) is not None
                ]
            )
            + "."
        )
    door_count = _int(opening_summary.get("door_count"), 0)
    window_count = _int(opening_summary.get("window_count"), 0)
    if door_count or window_count:
        layout_constraints.append(
            f"Keep the room shell openings consistent with the snapshot: {door_count} door(s) and {window_count} window(s)."
        )
    if not compiler_config.include_minor_decor:
        dropped_minor = selection["dropped_minor"]
        if dropped_minor:
            layout_constraints.append(
                "Minor decor may be omitted if it does not change the composition: "
                + _comma_join([item.label for item in dropped_minor[:6]])
                + "."
            )

    negative_prompt = (
        "no extra furniture, no duplicated objects, no major room-shape distortion, "
        "no moving anchor furniture to new walls, no moving doors or windows to different sides, "
        "no swapping object compass positions, no rotating anchor furniture away from its intended facing, "
        "no removing visible anchor items, no making occluded furniture fully exposed in front of its blocker"
    )

    return {
        "camera": camera_section,
        "room_shell": room_section,
        "openings": openings_section,
        "anchors": anchor_sentences,
        "supporting": supporting_section,
        "occlusion": occlusion_section,
        "layout_facts": layout_fact_lines,
        "primary_prompt": primary_prompt,
        "layout_constraints": layout_constraints,
        "negative_prompt": negative_prompt,
    }


def _objects_for_layout_facts(
    objects: Sequence[SnapshotObject],
) -> list[SnapshotObject]:
    def _priority(item: SnapshotObject) -> tuple[int, float]:
        role_rank = {"anchor": 0, "supporting": 1, "minor": 2}[item.role]
        return role_rank, -item.salience

    sorted_objects = sorted(objects, key=_priority)
    return sorted_objects[: min(len(sorted_objects), 16)]


def _object_layout_fact(
    item: SnapshotObject,
    *,
    room_center: Point2D | None,
    opening_records: Sequence[Mapping[str, object]],
) -> str:
    size_band = _size_band(item.dimensions_mm)
    size_prefix = f"{size_band} " if size_band else ""
    position_phrase = item.room_position_phrase
    facing_phrase = _facing_phrase_toward_center(
        facing_direction=item.facing_direction,
        room_center=room_center,
        plan_position=item.plan_position_mm,
    )
    sentence = f"The {size_prefix}{item.label} is {position_phrase}"
    if facing_phrase:
        sentence += f", facing {facing_phrase}"
    opening_relation = _opening_relation_phrase(
        item,
        opening_records=opening_records,
    )
    if opening_relation:
        sentence += f", {opening_relation}"
    return sentence + "."


def _build_spatial_relation_facts(
    objects: Sequence[SnapshotObject],
    *,
    room_bounds: Bounds2D | None,
) -> list[str]:
    facts: list[tuple[float, str]] = []
    relevant_objects = [
        item for item in objects if item.role != "minor" or item.layout_critical
    ]
    seen_sentences: set[str] = set()

    for left in relevant_objects:
        for right in relevant_objects:
            if left.id == right.id:
                continue
            relation = _pairwise_relation_fact(
                source=left,
                target=right,
                room_bounds=room_bounds,
            )
            if relation is None:
                continue
            score, sentence = relation
            if sentence in seen_sentences:
                continue
            facts.append((score, sentence))
            seen_sentences.add(sentence)

    facts.sort(key=lambda item: item[0], reverse=True)
    return [sentence for _, sentence in facts[:12]]


def _anchor_sentence(item: SnapshotObject, *, first: bool) -> str:
    lead = (
        f"{item.depth_band.capitalize()} {item.screen_region}, "
        if first
        else f"Also in the {item.depth_band} {item.screen_region}, "
    )
    return lead + _object_sentence_fragment(item) + "."


def _object_sentence_fragment(item: SnapshotObject) -> str:
    phrase = _object_prompt_phrase(item, include_depth=False)
    if item.place_on_method == "on_top" and item.place_on_target_id:
        return f"{phrase} placed on {item.place_on_target_id.replace('_', ' ')}"
    if item.place_on_method == "hang_on":
        if item.place_on_target_id == "ceiling":
            return f"{phrase} suspended from the ceiling"
        if item.place_on_target_id:
            return (
                f"{phrase} mounted against {item.place_on_target_id.replace('_', ' ')}"
            )
    return phrase


def _object_prompt_phrase(item: SnapshotObject | None, *, include_depth: bool) -> str:
    if item is None:
        return "object"
    descriptors = [item.color_phrase]
    if item.material:
        descriptors.append(item.material.replace("_", " "))
    size_band = _size_band(item.dimensions_mm)
    if size_band:
        descriptors.insert(0, size_band)
    descriptor_text = " ".join(part for part in descriptors if part).strip()
    phrase = f"{descriptor_text} {item.label}".strip()
    if include_depth:
        return f"{phrase} in the {item.depth_band} {item.screen_region}"
    return phrase


def _supporting_label(item: SnapshotObject) -> str:
    base = item.label
    if item.canonical_type in _DECOR_TYPES:
        return f"{item.color_phrase} {base}"
    return base


def _find_object(
    selection: Mapping[str, list[SnapshotObject]],
    object_id: str,
) -> SnapshotObject | None:
    for group in ("anchors", "supporting", "included_minor", "dropped_minor"):
        for item in selection.get(group, []):
            if item.id == object_id:
                return item
    return None


def _should_skip_occlusion_object(item: SnapshotObject) -> bool:
    return item.canonical_type in {"ceiling_light", "curtain", "tv", "wall_art"} or (
        item.role == "minor"
        and item.screen_bbox_px is not None
        and item.screen_bbox_px.area < 600.0
    )


def _occluded_region(occluder: Bounds2D, occluded: Bounds2D) -> str:
    overlap_min_x = max(occluder.min_x, occluded.min_x)
    overlap_max_x = min(occluder.max_x, occluded.max_x)
    overlap_min_y = max(occluder.min_y, occluded.min_y)
    overlap_max_y = min(occluder.max_y, occluded.max_y)
    overlap_center_x = (overlap_min_x + overlap_max_x) / 2.0
    overlap_center_y = (overlap_min_y + overlap_max_y) / 2.0
    horizontal = _relative_band(
        overlap_center_x,
        start=occluded.min_x,
        end=occluded.max_x,
        low="left",
        mid="middle",
        high="right",
    )
    vertical = _relative_band(
        overlap_center_y,
        start=occluded.min_y,
        end=occluded.max_y,
        low="upper",
        mid="middle",
        high="lower",
    )
    return f"{vertical} {horizontal}"


def _depth_bands(distances: Sequence[float]) -> list[DepthBand]:
    if not distances:
        return []
    min_distance = min(distances)
    max_distance = max(distances)
    spread = max(0.001, max_distance - min_distance)
    bands: list[DepthBand] = []
    for value in distances:
        ratio = (value - min_distance) / spread
        if ratio <= 0.33:
            bands.append("foreground")
        elif ratio <= 0.68:
            bands.append("midground")
        else:
            bands.append("background")
    return bands


def _camera_summary(
    elevation: ViewElevation,
    azimuth: ViewAzimuth,
    shot_scale: str,
) -> str:
    base = {
        "top_down": "Top-down architectural room view",
        "high_oblique": "High-angle oblique architectural view",
        "oblique": "Elevated oblique interior view",
        "eye_level": "Eye-level interior perspective",
    }[elevation]
    if azimuth == "centered":
        direction = "from a centered vantage"
    elif "left" in azimuth or "right" in azimuth:
        direction = f"from a {azimuth.replace('_', ' ')} corner vantage"
    else:
        direction = f"from a {azimuth.replace('_', ' ')} vantage"
    return f"{base} {direction}, {shot_scale} framing"


def _shot_scale(
    *,
    room_bounds: object,
    camera: Point3D,
    target: Point3D,
) -> str:
    room_span_mm = 0.0
    if isinstance(room_bounds, Bounds2D):
        room_span_mm = max(room_bounds.width, room_bounds.height)
    horizontal_distance_mm = math.hypot(camera.x - target.x, camera.y - target.y)
    if room_span_mm <= 0.0:
        return "wide"
    ratio = horizontal_distance_mm / room_span_mm
    if ratio >= 2.0:
        return "wide"
    if ratio >= 1.2:
        return "medium"
    return "close"


def _azimuth_class(camera: Point3D, target: Point3D) -> ViewAzimuth:
    dx = camera.x - target.x
    dy = camera.y - target.y
    if abs(dx) < 1.0 and abs(dy) < 1.0:
        return "centered"
    angle = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
    sectors: list[tuple[float, float, ViewAzimuth]] = [
        (337.5, 360.0, "right"),
        (0.0, 22.5, "right"),
        (22.5, 67.5, "back_right"),
        (67.5, 112.5, "back"),
        (112.5, 157.5, "back_left"),
        (157.5, 202.5, "left"),
        (202.5, 247.5, "front_left"),
        (247.5, 292.5, "front"),
        (292.5, 337.5, "front_right"),
    ]
    for start, end, label in sectors:
        if start <= angle < end:
            return label
    return "front"


def _room_shape(polygon: Sequence[object]) -> RoomShape:
    points = [item for item in polygon if isinstance(item, Point2D)]
    if len(points) < 3:
        return "unknown"
    if len(points) == 4:
        return "rectangular"
    area = abs(_polygon_area(points))
    bounds = _bounds_from_points(points)
    bounds_area = max(1.0, bounds.area)
    area_ratio = area / bounds_area
    if len(points) >= 6 and _room_missing_corner_compass(
        polygon=points, room_bounds=bounds
    ):
        return "L_shaped"
    if len(points) >= 8 and 0.45 <= area_ratio <= 0.72:
        return "T_shaped"
    if len(points) >= 6 and 0.55 <= area_ratio <= 0.94:
        return "L_shaped"
    return "irregular"


def _room_shape_fact(
    shape: RoomShape,
    missing_corner_compass: str | None,
) -> str:
    if shape == "L_shaped" and missing_corner_compass:
        return f"The room footprint is L-shaped with a missing {missing_corner_compass} corner."
    if shape == "rectangular":
        return "The room footprint is rectangular."
    if shape == "T_shaped":
        return "The room footprint is T-shaped."
    if shape == "irregular":
        return "The room footprint has visible recesses and offsets."
    return "The room shell should remain consistent with the captured layout."


def _room_shell_summary(
    shape: RoomShape,
    mode: ShellMode,
    *,
    missing_corner_compass: str | None,
) -> str:
    if mode == "omit":
        return ""
    if mode == "canonical":
        return {
            "rectangular": "The room has a rectangular shell.",
            "L_shaped": (
                f"The room is L-shaped with a missing {missing_corner_compass} corner."
                if missing_corner_compass
                else "The room is L-shaped."
            ),
            "T_shaped": "The room has a T-shaped footprint.",
            "irregular": "The room has an irregular footprint.",
            "unknown": "The room shell should stay consistent with the captured layout.",
        }[shape]
    return {
        "rectangular": "The room reads as a simple single-volume interior.",
        "L_shaped": (
            f"The room reads as a main volume with a recessed {missing_corner_compass} corner and a visible side wing."
            if missing_corner_compass
            else "The room reads as a main volume with a visible secondary side wing."
        ),
        "T_shaped": "The room reads as a main volume with branching side extensions.",
        "irregular": "The room shell should keep its non-rectilinear recesses and offsets.",
        "unknown": "Keep the visible room shell consistent with the captured view.",
    }[shape]


def _room_missing_corner_compass(
    *,
    polygon: Sequence[Point2D],
    room_bounds: Bounds2D | None,
) -> str | None:
    if room_bounds is None or len(polygon) < 6:
        return None
    inset_x = max(room_bounds.width * 0.12, 120.0)
    inset_y = max(room_bounds.height * 0.12, 120.0)
    samples = {
        "southwest": Point2D(room_bounds.min_x + inset_x, room_bounds.min_y + inset_y),
        "southeast": Point2D(room_bounds.max_x - inset_x, room_bounds.min_y + inset_y),
        "northwest": Point2D(room_bounds.min_x + inset_x, room_bounds.max_y - inset_y),
        "northeast": Point2D(room_bounds.max_x - inset_x, room_bounds.max_y - inset_y),
    }
    missing = [
        label
        for label, point in samples.items()
        if not _point_in_polygon(point, polygon)
    ]
    return missing[0] if len(missing) == 1 else None


def _parse_openings(value: object) -> list[OpeningRecord]:
    items: list[OpeningRecord] = []
    for raw_item in _sequence(value):
        payload = _mapping(raw_item)
        segment_payload = _sequence(
            payload.get("segmentMm") or payload.get("segment_mm")
        )
        if len(segment_payload) < 2:
            continue
        first = _point2d_from_payload(segment_payload[0])
        second = _point2d_from_payload(segment_payload[1])
        if first is None or second is None:
            continue
        items.append(
            OpeningRecord(
                id=_string(payload.get("id"), fallback="opening") or "opening",
                segment=(first, second),
            )
        )
    return items


def _polygon_from_payload(value: object) -> list[Point2D]:
    points: list[Point2D] = []
    for raw_point in _sequence(value):
        point = _point2d_from_payload(raw_point)
        if point is not None:
            points.append(point)
    return points


def _read_env_flag(name: str, *, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _type_label(value: str | None) -> str:
    if not value:
        return "object"
    return _TYPE_LABELS.get(value, value.replace("_", " "))


def _canonicalize_object_type(value: str | None) -> str:
    lowered = (value or "object").strip().lower()
    if "__accessory_refill_" in lowered:
        return lowered.split("__accessory_refill_", 1)[0]
    if "__reintroduced_" in lowered:
        return lowered.split("__reintroduced_", 1)[0]
    if lowered == "storage_cabin":
        return "storage_cabinet"
    return lowered


def _describe_color(color_hex: str | None) -> str:
    if not color_hex or not color_hex.startswith("#") or len(color_hex) != 7:
        return "neutral"
    red = int(color_hex[1:3], 16) / 255.0
    green = int(color_hex[3:5], 16) / 255.0
    blue = int(color_hex[5:7], 16) / 255.0
    max_channel = max(red, green, blue)
    min_channel = min(red, green, blue)
    delta = max_channel - min_channel
    lightness = (max_channel + min_channel) / 2.0
    saturation = (
        0.0 if delta == 0 else delta / max(0.001, 1.0 - abs(2.0 * lightness - 1.0))
    )
    hue = 0.0
    if delta > 0:
        if max_channel == red:
            hue = ((green - blue) / delta) % 6.0
        elif max_channel == green:
            hue = ((blue - red) / delta) + 2.0
        else:
            hue = ((red - green) / delta) + 4.0
        hue *= 60.0

    if saturation <= 0.12:
        if lightness >= 0.92:
            return "off-white"
        if lightness >= 0.72:
            return "light warm gray"
        if lightness >= 0.48:
            return "taupe"
        return "charcoal"
    if 18.0 <= hue < 48.0:
        return "warm beige" if lightness >= 0.6 else "warm brown"
    if 48.0 <= hue < 70.0:
        return "muted olive" if lightness < 0.55 else "sand"
    if 70.0 <= hue < 160.0:
        return "muted green" if lightness < 0.55 else "sage"
    if 160.0 <= hue < 245.0:
        return "muted blue" if lightness < 0.55 else "dusty blue"
    if 245.0 <= hue < 330.0:
        return "muted mauve" if lightness < 0.55 else "pale mauve"
    return "terracotta" if lightness < 0.6 else "soft peach"


def _center_score(
    screen_center: Point2D | None, width_px: int, height_px: int
) -> float:
    if screen_center is None:
        return 0.0
    dx = abs(screen_center.x - (width_px / 2.0)) / max(1.0, width_px / 2.0)
    dy = abs(screen_center.y - (height_px / 2.0)) / max(1.0, height_px / 2.0)
    distance = math.hypot(dx, dy)
    return max(0.0, 1.0 - distance)


def _screen_region(screen_center: Point2D | None, width_px: int) -> ScreenRegion:
    if screen_center is None:
        return "center"
    ratio = screen_center.x / max(1.0, float(width_px))
    if ratio < 0.33:
        return "left"
    if ratio > 0.67:
        return "right"
    return "center"


def _room_center(room_bounds: Bounds2D | None) -> Point2D | None:
    if room_bounds is None:
        return None
    return Point2D(
        x=(room_bounds.min_x + room_bounds.max_x) / 2.0,
        y=(room_bounds.min_y + room_bounds.max_y) / 2.0,
    )


def _room_compass(position: Point3D, room_bounds: Bounds2D | None) -> str:
    if room_bounds is None or room_bounds.width <= 0.0 or room_bounds.height <= 0.0:
        return "center"
    x_ratio = (position.x - room_bounds.min_x) / max(1.0, room_bounds.width)
    y_ratio = (position.y - room_bounds.min_y) / max(1.0, room_bounds.height)
    horizontal = "west" if x_ratio <= 0.33 else "east" if x_ratio >= 0.67 else "center"
    vertical = "south" if y_ratio <= 0.33 else "north" if y_ratio >= 0.67 else "center"
    if horizontal == "center" and vertical == "center":
        return "center"
    if horizontal == "center":
        return vertical
    if vertical == "center":
        return horizontal
    return f"{vertical}{horizontal}"


def _room_position_phrase(
    *,
    position: Point3D,
    dimensions: Dimensions3D,
    room_bounds: Bounds2D | None,
) -> str:
    if room_bounds is None:
        return "near the room center"
    near_west = (position.x - room_bounds.min_x) <= max(
        room_bounds.width * 0.14, dimensions.width * 0.6 + 140.0
    )
    near_east = (room_bounds.max_x - position.x) <= max(
        room_bounds.width * 0.14, dimensions.width * 0.6 + 140.0
    )
    near_south = (position.y - room_bounds.min_y) <= max(
        room_bounds.height * 0.14, dimensions.depth * 0.6 + 140.0
    )
    near_north = (room_bounds.max_y - position.y) <= max(
        room_bounds.height * 0.14, dimensions.depth * 0.6 + 140.0
    )
    if near_south and near_west:
        return "in the southwest corner"
    if near_south and near_east:
        return "in the southeast corner"
    if near_north and near_west:
        return "in the northwest corner"
    if near_north and near_east:
        return "in the northeast corner"
    if near_south:
        return "along the south wall"
    if near_north:
        return "along the north wall"
    if near_west:
        return "along the west wall"
    if near_east:
        return "along the east wall"
    compass = _room_compass(position, room_bounds)
    if compass == "center":
        return "near the room center"
    return f"in the {compass} zone"


def _facing_direction_from_rotation(plan_rotation_deg: float) -> str | None:
    normalized = plan_rotation_deg % 360.0
    sectors: list[tuple[float, float, str]] = [
        (337.5, 360.0, "north"),
        (0.0, 22.5, "north"),
        (22.5, 67.5, "northwest"),
        (67.5, 112.5, "west"),
        (112.5, 157.5, "southwest"),
        (157.5, 202.5, "south"),
        (202.5, 247.5, "southeast"),
        (247.5, 292.5, "east"),
        (292.5, 337.5, "northeast"),
    ]
    for start, end, label in sectors:
        if start <= normalized < end:
            return label
    return None


def _facing_phrase_toward_center(
    *,
    facing_direction: str | None,
    room_center: Point2D | None,
    plan_position: Point3D,
) -> str:
    if not facing_direction:
        return ""
    if room_center is None:
        return facing_direction
    facing_vector = _direction_vector(facing_direction)
    if facing_vector is None:
        return facing_direction
    to_center_x = room_center.x - plan_position.x
    to_center_y = room_center.y - plan_position.y
    magnitude = math.hypot(to_center_x, to_center_y)
    if magnitude <= 1.0:
        return facing_direction
    alignment = facing_vector.x * (to_center_x / magnitude) + facing_vector.y * (
        to_center_y / magnitude
    )
    if alignment >= 0.68:
        return f"{facing_direction} toward the room center"
    return facing_direction


def _direction_vector(direction: str | None) -> Point2D | None:
    mapping = {
        "north": Point2D(0.0, 1.0),
        "northeast": Point2D(0.7071, 0.7071),
        "east": Point2D(1.0, 0.0),
        "southeast": Point2D(0.7071, -0.7071),
        "south": Point2D(0.0, -1.0),
        "southwest": Point2D(-0.7071, -0.7071),
        "west": Point2D(-1.0, 0.0),
        "northwest": Point2D(-0.7071, 0.7071),
    }
    return mapping.get(direction or "")


def _opening_layout_info(
    *,
    opening: OpeningRecord,
    opening_type: str,
    room_bounds: Bounds2D | None,
    ordinal: int,
) -> dict[str, object]:
    wall_direction, wall_position = _opening_wall_location(
        opening.segment,
        room_bounds=room_bounds,
    )
    label = opening_type if ordinal == 1 else f"{opening_type} {ordinal}"
    article = _indefinite_article(f"{wall_direction}-wall {opening_type}")
    short = (
        f"{article if ordinal == 1 else 'another'} {wall_direction}-wall {opening_type}"
        + (f" {wall_position}" if wall_position else "")
    ).strip()
    fact = (
        f"The {label} is on the {wall_direction} wall"
        + (f" {wall_position}" if wall_position else "")
        + "."
    )
    return {
        "id": opening.id,
        "type": opening_type,
        "wall_direction": wall_direction,
        "wall_position": wall_position,
        "fact": fact,
        "short": short,
        "segment": opening.segment,
    }


def _opening_wall_location(
    segment: tuple[Point2D, Point2D],
    *,
    room_bounds: Bounds2D | None,
) -> tuple[str, str]:
    first, second = segment
    delta_x = abs(first.x - second.x)
    delta_y = abs(first.y - second.y)
    if room_bounds is None:
        if delta_x >= delta_y:
            return "south", "near the center"
        return "east", "near the center"
    if delta_x >= delta_y:
        wall_direction = (
            "south"
            if abs(first.y - room_bounds.min_y) <= abs(first.y - room_bounds.max_y)
            else "north"
        )
        center_value = (first.x + second.x) / 2.0
        position = _relative_band(
            center_value,
            start=room_bounds.min_x,
            end=room_bounds.max_x,
            low="toward the west side",
            mid="near the center",
            high="toward the east side",
        )
        return wall_direction, position
    wall_direction = (
        "west"
        if abs(first.x - room_bounds.min_x) <= abs(first.x - room_bounds.max_x)
        else "east"
    )
    center_value = (first.y + second.y) / 2.0
    position = _relative_band(
        center_value,
        start=room_bounds.min_y,
        end=room_bounds.max_y,
        low="toward the south side",
        mid="near the center",
        high="toward the north side",
    )
    return wall_direction, position


def _opening_relation_phrase(
    item: SnapshotObject,
    *,
    opening_records: Sequence[Mapping[str, object]],
) -> str:
    best_score = 0.0
    best_phrase = ""
    for record in opening_records:
        segment = record.get("segment")
        if not (
            isinstance(segment, tuple)
            and len(segment) == 2
            and isinstance(segment[0], Point2D)
            and isinstance(segment[1], Point2D)
        ):
            continue
        wall_direction = _string(record.get("wall_direction"))
        opening_type = _string(record.get("type"), fallback="opening")
        relation = _opening_relation_score(
            item,
            segment=segment,
            wall_direction=wall_direction,
            opening_type=opening_type,
        )
        if relation is None:
            continue
        score, phrase = relation
        if score > best_score:
            best_score = score
            best_phrase = phrase
    return best_phrase


def _opening_relation_score(
    item: SnapshotObject,
    *,
    segment: tuple[Point2D, Point2D],
    wall_direction: str | None,
    opening_type: str,
) -> tuple[float, str] | None:
    if wall_direction is None:
        return None
    center_x = (segment[0].x + segment[1].x) / 2.0
    center_y = (segment[0].y + segment[1].y) / 2.0
    span_x = abs(segment[0].x - segment[1].x)
    span_y = abs(segment[0].y - segment[1].y)
    delta_x = item.plan_position_mm.x - center_x
    delta_y = item.plan_position_mm.y - center_y

    if wall_direction in {"south", "north"}:
        lateral_limit = max(span_x / 2.0 + item.dimensions_mm.width * 0.6, 500.0)
        if abs(delta_x) > lateral_limit:
            return None
        forward_distance = delta_y if wall_direction == "south" else -delta_y
        if forward_distance < 0.0 or forward_distance > max(
            item.dimensions_mm.depth * 1.8, 1200.0
        ):
            return None
    else:
        lateral_limit = max(span_y / 2.0 + item.dimensions_mm.depth * 0.6, 500.0)
        if abs(delta_y) > lateral_limit:
            return None
        forward_distance = delta_x if wall_direction == "west" else -delta_x
        if forward_distance < 0.0 or forward_distance > max(
            item.dimensions_mm.width * 1.8, 1200.0
        ):
            return None

    score = max(0.1, 2000.0 - forward_distance - abs(delta_x) - abs(delta_y))
    phrase = f"positioned in front of the {wall_direction}-wall {opening_type}"
    return score, phrase


def _pairwise_relation_fact(
    *,
    source: SnapshotObject,
    target: SnapshotObject,
    room_bounds: Bounds2D | None,
) -> tuple[float, str] | None:
    if source.facing_direction is None:
        return None
    direction = _direction_vector(source.facing_direction)
    if direction is None:
        return None

    delta_x = target.plan_position_mm.x - source.plan_position_mm.x
    delta_y = target.plan_position_mm.y - source.plan_position_mm.y
    distance = math.hypot(delta_x, delta_y)
    if distance <= 1.0:
        return None

    room_span = 0.0
    if room_bounds is not None:
        room_span = max(room_bounds.width, room_bounds.height)
    max_relation_distance = max(
        room_span * 0.48 if room_span > 0.0 else 0.0,
        max(source.dimensions_mm.width, source.dimensions_mm.depth) * 3.2,
        max(target.dimensions_mm.width, target.dimensions_mm.depth) * 2.6,
        1200.0,
    )
    if distance > max_relation_distance:
        return None

    normalized_dx = delta_x / distance
    normalized_dy = delta_y / distance
    alignment = direction.x * normalized_dx + direction.y * normalized_dy
    lateral = abs(direction.x * normalized_dy - direction.y * normalized_dx)

    if (
        source.canonical_type
        in {"sofa", "sectional_sofa", "armchair", "recliner", "chair"}
        and target.canonical_type in {"tv_console", "tv"}
        and alignment >= 0.72
    ):
        score = (
            180.0 - distance / 30.0 + source.semantic_weight + target.semantic_weight
        )
        return score, f"The {source.label} faces the {target.label}."
    if alignment >= 0.8 and lateral <= 0.5:
        score = (
            176.0 - distance / 25.0 + source.semantic_weight + target.semantic_weight
        )
        return score, f"The {target.label} sits in front of the {source.label}."
    if alignment <= -0.8 and lateral <= 0.5:
        score = (
            170.0 - distance / 25.0 + source.semantic_weight + target.semantic_weight
        )
        return score, f"The {target.label} sits behind the {source.label}."
    if min(abs(delta_x), abs(delta_y)) <= max(
        min(source.dimensions_mm.width, source.dimensions_mm.depth) * 0.55,
        min(target.dimensions_mm.width, target.dimensions_mm.depth) * 0.55,
        320.0,
    ) and max(abs(delta_x), abs(delta_y)) <= max(
        max(source.dimensions_mm.width, source.dimensions_mm.depth) * 2.1,
        max(target.dimensions_mm.width, target.dimensions_mm.depth) * 2.1,
        1300.0,
    ):
        score = (
            160.0 - distance / 28.0 + source.semantic_weight + target.semantic_weight
        )
        return score, f"The {target.label} sits beside the {source.label}."
    if abs(alignment) <= 0.35 and lateral >= 0.78:
        score = (
            118.0 - distance / 24.0 + source.semantic_weight + target.semantic_weight
        )
        return score, f"The {target.label} sits beside the {source.label}."
    return None


def _point_in_polygon(point: Point2D, polygon: Sequence[Point2D]) -> bool:
    inside = False
    if len(polygon) < 3:
        return inside
    previous = polygon[-1]
    for current in polygon:
        intersects = (current.y > point.y) != (previous.y > point.y) and point.x < (
            previous.x - current.x
        ) * (point.y - current.y) / max(previous.y - current.y, 1e-9) + current.x
        if intersects:
            inside = not inside
        previous = current
    return inside


def _relative_band(
    value: float,
    *,
    start: float,
    end: float,
    low: str,
    mid: str,
    high: str,
) -> str:
    span = max(1.0, end - start)
    ratio = (value - start) / span
    if ratio <= 0.33:
        return low
    if ratio >= 0.67:
        return high
    return mid


def _size_band(dimensions: Dimensions3D) -> str:
    footprint_m2 = (dimensions.width * dimensions.depth) / 1_000_000.0
    if footprint_m2 >= 1.8:
        return "large"
    if footprint_m2 >= 0.55:
        return "medium"
    return ""


def _intersection_area(left: Bounds2D, right: Bounds2D) -> float:
    width = max(0.0, min(left.max_x, right.max_x) - max(left.min_x, right.min_x))
    height = max(0.0, min(left.max_y, right.max_y) - max(left.min_y, right.min_y))
    return width * height


def _polygon_area(points: Sequence[Point2D]) -> float:
    area = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        area += point.x * next_point.y - next_point.x * point.y
    return area / 2.0


def _bounds_from_points(points: Sequence[Point2D]) -> Bounds2D:
    xs = [point.x for point in points]
    ys = [point.y for point in points]
    return Bounds2D(min(xs), min(ys), max(xs), max(ys))


def _count_phrase(count: int, *, singular: str, plural: str) -> str:
    return f"{count} {singular if count == 1 else plural}"


def _comma_join(values: Sequence[str]) -> str:
    filtered = [value.strip() for value in values if value and value.strip()]
    if not filtered:
        return ""
    if len(filtered) == 1:
        return filtered[0]
    if len(filtered) == 2:
        return f"{filtered[0]} and {filtered[1]}"
    return ", ".join(filtered[:-1]) + f", and {filtered[-1]}"


def _indefinite_article(value: str) -> str:
    stripped = value.strip().lower()
    return "an" if stripped[:1] in {"a", "e", "i", "o", "u"} else "a"


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    return (
        value
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes))
        else []
    )


def _string(value: object, fallback: str | None = None) -> str | None:
    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed or fallback
    return fallback


def _float(value: object, fallback: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return fallback
    return fallback


def _int(value: object, fallback: int = 0) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return fallback
    return fallback


def _point2d_from_payload(value: object) -> Point2D | None:
    payload = _mapping(value)
    if not payload:
        return None
    return Point2D(
        x=_float(payload.get("x")),
        y=_float(payload.get("y")),
    )


def _point3d_from_payload(value: object) -> Point3D:
    payload = _mapping(value)
    return Point3D(
        x=_float(payload.get("x")),
        y=_float(payload.get("y")),
        z=_float(payload.get("z")),
    )


def _bounds_from_payload(value: object) -> Bounds2D | None:
    payload = _mapping(value)
    if not payload:
        return None
    return Bounds2D(
        min_x=_float(
            payload.get("minX") if "minX" in payload else payload.get("min_x")
        ),
        min_y=_float(
            payload.get("minY") if "minY" in payload else payload.get("min_y")
        ),
        max_x=_float(
            payload.get("maxX") if "maxX" in payload else payload.get("max_x")
        ),
        max_y=_float(
            payload.get("maxY") if "maxY" in payload else payload.get("max_y")
        ),
    )


def _dimensions_from_payload(value: object) -> Dimensions3D:
    payload = _mapping(value)
    return Dimensions3D(
        width=_float(payload.get("width")),
        depth=_float(payload.get("depth")),
        height=_float(payload.get("height")),
    )


def _point3d_or_origin(value: object) -> Point3D:
    return value if isinstance(value, Point3D) else Point3D(0.0, 0.0, 0.0)


def _dimensions_or_default(value: object) -> Dimensions3D:
    return value if isinstance(value, Dimensions3D) else Dimensions3D(0.0, 0.0, 0.0)
