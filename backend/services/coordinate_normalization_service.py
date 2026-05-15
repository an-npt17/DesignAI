from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Mapping

from shapely.geometry import MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry


JsonObject = dict[str, Any]
CoordinateSpace = Literal["apartment_normalized", "room_local"]
RotationInput = Literal["auto", "degrees", "radians", "quaternion"]
SourceUnit = Literal["auto", "m", "meter", "meters", "mm", "millimeter", "millimeters"]

_POINT_TREE_KEYS = {
    "polygon",
    "polygons",
    "polygon_ccw",
    "polygon_mm",
    "polyline",
    "polyline_mm",
    "points",
    "shape_points",
}
_POINT_KEYS = {
    "center",
    "end",
    "endPoint",
    "p1_mm",
    "p2_mm",
    "position_mm",
    "start",
    "startPoint",
}
_POSITION_3D_KEYS = {"position", "translation"}
_QUATERNION_KEYS = {"rotation", "rotation_quaternion"}
_DIMENSION_KEYS = {"dimensions", "footprint", "footprint_mm", "size", "size_mm"}
_SKIP_BOUNDS_KEYS = {"objects"}


@dataclass(frozen=True)
class PlanarPoint:
    x: float
    y: float

    def as_dict(self) -> JsonObject:
        return {"x": _clean_float(self.x), "y": _clean_float(self.y)}


@dataclass(frozen=True)
class PlanarBBox:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @classmethod
    def from_points(cls, points: list[PlanarPoint]) -> "PlanarBBox":
        if not points:
            raise ValueError("At least one planar point is required.")
        xs = [point.x for point in points]
        ys = [point.y for point in points]
        return cls(min_x=min(xs), min_y=min(ys), max_x=max(xs), max_y=max(ys))

    def shifted(self, dx: float, dy: float) -> "PlanarBBox":
        return PlanarBBox(
            min_x=self.min_x + dx,
            min_y=self.min_y + dy,
            max_x=self.max_x + dx,
            max_y=self.max_y + dy,
        )

    def as_dict(self) -> JsonObject:
        return {
            "min_x": _clean_float(self.min_x),
            "min_y": _clean_float(self.min_y),
            "max_x": _clean_float(self.max_x),
            "max_y": _clean_float(self.max_y),
        }


class CoordinateNormalizationService:
    def normalize_input(
        self,
        payload: Mapping[str, Any],
        *,
        source_unit: SourceUnit = "auto",
        tenant_id: str | None = None,
        user_id: str | None = None,
        description: str | None = None,
        special_notes: str | None = None,
        style: str | None = None,
        split_largest_room: bool = True,
    ) -> JsonObject:
        raw_payload = deepcopy(dict(payload))
        source_scale_to_mm = self._source_scale_to_mm(raw_payload, source_unit)
        source_payload = self._scale_json(raw_payload, factor=source_scale_to_mm)
        apartment_bbox = self._apartment_bbox(source_payload)
        split_payload, room_split = self._split_largest_room_payload(
            source_payload,
            enabled=split_largest_room,
        )
        shift = PlanarPoint(x=-apartment_bbox.min_x, y=-apartment_bbox.min_y)
        normalized_payload = self._translate_json(
            split_payload,
            dx=shift.x,
            dy=shift.y,
        )
        room_transforms = self._build_room_transforms(
            split_payload,
            apartment_shift=shift,
        )
        room_views = self._build_room_views(
            normalized_payload,
            room_transforms=room_transforms,
        )
        normalized_bbox = apartment_bbox.shifted(shift.x, shift.y)
        system_inputs = self._build_system_inputs(
            payload=split_payload,
            room_views=room_views,
            room_transforms=room_transforms,
            tenant_id=tenant_id,
            user_id=user_id,
            description=description,
            special_notes=special_notes,
            style=style,
        )

        return {
            "normalized_payload": normalized_payload,
            "transform": {
                "version": "tknt-coordinate-normalizer-v1",
                "unit": "mm",
                "source_scale_to_mm": source_scale_to_mm,
                "apartment": {
                    "bbox_original": apartment_bbox.as_dict(),
                    "bbox_normalized": normalized_bbox.as_dict(),
                    "shift": shift.as_dict(),
                    "inverse_shift": {"x": -shift.x, "y": -shift.y},
                },
                "rooms": room_transforms,
            },
            "apartment": {
                "bbox_original": apartment_bbox.as_dict(),
                "bbox_normalized": normalized_bbox.as_dict(),
                "origin_original": {
                    "x": _clean_float(apartment_bbox.min_x),
                    "y": _clean_float(apartment_bbox.min_y),
                },
                "origin_normalized": {"x": 0.0, "y": 0.0},
                "shift": shift.as_dict(),
            },
            "rooms": room_views,
            "system_inputs": system_inputs,
            "room_split": room_split,
        }

    def restore_output(
        self,
        output_payload: Any,
        transform: Mapping[str, Any],
        *,
        coordinate_space: CoordinateSpace,
        room_id: str | None = None,
        rotation_input: RotationInput = "auto",
    ) -> JsonObject:
        dx, dy, resolved_room_id = self._restore_delta(
            transform,
            coordinate_space=coordinate_space,
            room_id=room_id,
        )
        restored_payload = self._translate_json(deepcopy(output_payload), dx=dx, dy=dy)
        source_scale_to_mm = self._source_scale_from_transform(transform)
        output_scale = 1.0 / source_scale_to_mm
        if abs(output_scale - 1.0) > 1e-12:
            restored_payload = self._scale_json(restored_payload, factor=output_scale)
        restored_payload = self._convert_rotations(restored_payload, rotation_input)
        restored_payload = self._format_frontend_positions(restored_payload)
        return {
            "restored_payload": restored_payload,
            "transform_applied": {
                "coordinate_space": coordinate_space,
                "room_id": resolved_room_id,
                "delta": {"x": dx, "y": dy},
                "source_scale_to_mm": source_scale_to_mm,
                "output_scale": output_scale,
                "rotation_format": "quaternion_xyzw",
            },
        }

    def _source_scale_from_transform(self, transform: Mapping[str, Any]) -> float:
        source_scale_to_mm = self._number(transform.get("source_scale_to_mm"))
        if source_scale_to_mm is None or source_scale_to_mm <= 0:
            return 1.0
        return float(source_scale_to_mm)

    def _apartment_bbox(self, payload: Mapping[str, Any]) -> PlanarBBox:
        points = self._collect_bounds_points(payload, skip_keys=_SKIP_BOUNDS_KEYS)
        if not points:
            points = self._collect_bounds_points(payload, skip_keys=set())
        if not points:
            raise ValueError("No apartment coordinates were found in the payload.")
        return PlanarBBox.from_points(points)

    def _source_scale_to_mm(
        self,
        payload: Mapping[str, Any],
        source_unit: SourceUnit,
    ) -> float:
        normalized_unit = source_unit.strip().lower()
        if normalized_unit in {"mm", "millimeter", "millimeters"}:
            return 1.0
        if normalized_unit in {"m", "meter", "meters"}:
            return 1000.0
        bbox = self._apartment_bbox(payload)
        span = max(bbox.max_x - bbox.min_x, bbox.max_y - bbox.min_y)
        return 1000.0 if span <= 200.0 else 1.0

    def _split_largest_room_payload(
        self,
        payload: Mapping[str, Any],
        *,
        enabled: bool,
    ) -> tuple[JsonObject, JsonObject]:
        split_payload = deepcopy(dict(payload))
        split_meta: JsonObject = {
            "enabled": enabled,
            "applied": False,
            "ratio": {"living_room": 0.6, "kitchen": 0.4},
        }
        if not enabled:
            return split_payload, split_meta

        rooms = split_payload.get("rooms")
        if not isinstance(rooms, list) or len(rooms) < 1:
            split_meta["reason"] = "No rooms were provided."
            return split_payload, split_meta

        largest_index: int | None = None
        largest_area = 0.0
        for index, raw_room in enumerate(rooms):
            if not isinstance(raw_room, Mapping):
                continue
            points = self._room_points(raw_room)
            if len(points) < 3:
                continue
            area = abs(self._polygon_area([point.as_dict() for point in points]))
            if area > largest_area:
                largest_area = area
                largest_index = index

        if largest_index is None:
            split_meta["reason"] = "No splittable room polygon was found."
            return split_payload, split_meta

        largest_room = rooms[largest_index]
        if not isinstance(largest_room, Mapping):
            split_meta["reason"] = "Largest room payload was not an object."
            return split_payload, split_meta

        split_result = self._split_room_mapping(largest_room)
        if split_result is None:
            split_meta["reason"] = "Largest room polygon could not be split cleanly."
            return split_payload, split_meta

        living_room, kitchen_room, partition_wall, split_details = split_result
        split_payload["rooms"] = [
            *rooms[:largest_index],
            living_room,
            kitchen_room,
            *rooms[largest_index + 1 :],
        ]
        if partition_wall is not None:
            partition_wall["height"] = self._payload_height_mm(split_payload)
            walls = split_payload.get("walls")
            if not isinstance(walls, list):
                walls = []
            split_payload["walls"] = [*walls, partition_wall]

        split_meta.update(split_details)
        split_meta["applied"] = True
        return split_payload, split_meta

    def _split_room_mapping(
        self,
        raw_room: Mapping[str, Any],
    ) -> tuple[JsonObject, JsonObject, JsonObject | None, JsonObject] | None:
        room_points = self._room_points(raw_room)
        polygon = self._polygon_from_points(room_points)
        if polygon is None:
            return None

        split_geometries = self._split_polygon_by_ratio(polygon, ratio=0.6)
        if split_geometries is None:
            return None

        living_polygon, kitchen_polygon = split_geometries
        base_id = self._string_or_none(raw_room.get("key") or raw_room.get("room_id"))
        if base_id is None:
            base_id = self._slugify(self._string_or_none(raw_room.get("name")) or "room")
        base_name = self._string_or_none(raw_room.get("name")) or base_id

        living_room = self._split_child_room(
            raw_room,
            room_id=f"{base_id}__living",
            name=f"{base_name} - Phòng khách",
            room_type="living_room",
            polygon=living_polygon,
            split_role="living_room",
        )
        kitchen_room = self._split_child_room(
            raw_room,
            room_id=f"{base_id}__kitchen",
            name=f"{base_name} - Bếp",
            room_type="kitchen",
            polygon=kitchen_polygon,
            split_role="kitchen",
        )
        partition_segment = self._shared_boundary_segment(
            living_polygon,
            kitchen_polygon,
        )
        partition_wall = None
        if partition_segment is not None:
            partition_wall = {
                "id": f"split-wall-{base_id}",
                "height": None,
                "thickness": 120,
                "startPoint": partition_segment[0].as_dict(),
                "endPoint": partition_segment[1].as_dict(),
                "generatedBy": "coordinate_normalizer_largest_room_split",
                "parentRoomId": base_id,
            }

        split_details = {
            "parent_room_id": base_id,
            "parent_room_name": base_name,
            "parent_area_m2": round(float(polygon.area) / 1_000_000.0, 3),
            "children": [
                {
                    "room_id": living_room["key"],
                    "room_type": "living_room",
                    "area_m2": round(float(living_polygon.area) / 1_000_000.0, 3),
                },
                {
                    "room_id": kitchen_room["key"],
                    "room_type": "kitchen",
                    "area_m2": round(float(kitchen_polygon.area) / 1_000_000.0, 3),
                },
            ],
        }
        return living_room, kitchen_room, partition_wall, split_details

    def _split_polygon_by_ratio(
        self,
        polygon: Polygon,
        *,
        ratio: float,
    ) -> tuple[Polygon, Polygon] | None:
        min_x, min_y, max_x, max_y = polygon.bounds
        width = max_x - min_x
        height = max_y - min_y
        axes = ["x", "y"] if width >= height else ["y", "x"]
        for axis in axes:
            result = self._split_polygon_along_axis(polygon, axis=axis, ratio=ratio)
            if result is not None:
                return result
        return None

    def _split_polygon_along_axis(
        self,
        polygon: Polygon,
        *,
        axis: Literal["x", "y"],
        ratio: float,
    ) -> tuple[Polygon, Polygon] | None:
        min_x, min_y, max_x, max_y = polygon.bounds
        lower = min_x if axis == "x" else min_y
        upper = max_x if axis == "x" else max_y
        if upper - lower <= 1e-6:
            return None
        target_area = float(polygon.area) * ratio
        best: tuple[Polygon, Polygon] | None = None
        best_delta: float | None = None
        left = lower
        right = upper

        for _ in range(64):
            cut = (left + right) / 2.0
            first, second = self._clip_polygon_pair(polygon, axis=axis, cut=cut)
            if first is None or second is None:
                return None
            area = float(first.area)
            delta = abs(area - target_area)
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best = (first, second)
            if area < target_area:
                left = cut
            else:
                right = cut

        if best is None:
            return None
        first, second = best
        if first.area <= 1.0 or second.area <= 1.0:
            return None
        if first.area >= second.area:
            return first, second
        return second, first

    def _clip_polygon_pair(
        self,
        polygon: Polygon,
        *,
        axis: Literal["x", "y"],
        cut: float,
    ) -> tuple[Polygon | None, Polygon | None]:
        min_x, min_y, max_x, max_y = polygon.bounds
        pad = max(max_x - min_x, max_y - min_y, 1000.0) * 2.0
        if axis == "x":
            first_box = box(min_x - pad, min_y - pad, cut, max_y + pad)
            second_box = box(cut, min_y - pad, max_x + pad, max_y + pad)
        else:
            first_box = box(min_x - pad, min_y - pad, max_x + pad, cut)
            second_box = box(min_x - pad, cut, max_x + pad, max_y + pad)
        first = self._single_polygon(polygon.intersection(first_box))
        second = self._single_polygon(polygon.intersection(second_box))
        if first is None or second is None:
            return None, None
        return first, second

    def _polygon_from_points(self, points: list[PlanarPoint]) -> Polygon | None:
        if len(points) < 3:
            return None
        polygon = Polygon([(point.x, point.y) for point in points])
        if polygon.is_empty or polygon.area <= 1.0:
            return None
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        return self._single_polygon(polygon)

    def _single_polygon(self, geometry: BaseGeometry) -> Polygon | None:
        if geometry.is_empty:
            return None
        if isinstance(geometry, Polygon):
            return geometry
        if isinstance(geometry, MultiPolygon):
            polygons = [item for item in geometry.geoms if item.area > 1.0]
            if len(polygons) != 1:
                return None
            return polygons[0]
        return None

    def _split_child_room(
        self,
        raw_room: Mapping[str, Any],
        *,
        room_id: str,
        name: str,
        room_type: str,
        polygon: Polygon,
        split_role: str,
    ) -> JsonObject:
        child = deepcopy(dict(raw_room))
        child["key"] = room_id
        child["room_id"] = room_id
        child["name"] = name
        child["roomType"] = room_type
        child["splitRole"] = split_role
        child["splitGenerated"] = True
        child["polygons"] = self._points_from_polygon(polygon)
        return child

    def _points_from_polygon(self, polygon: Polygon) -> list[list[float]]:
        coords = list(polygon.exterior.coords)
        if len(coords) > 1 and coords[0] == coords[-1]:
            coords = coords[:-1]
        return [[_clean_float(float(x)), _clean_float(float(y))] for x, y in coords]

    def _shared_boundary_segment(
        self,
        first: Polygon,
        second: Polygon,
    ) -> tuple[PlanarPoint, PlanarPoint] | None:
        boundary = first.boundary.intersection(second.boundary)
        lines = []
        if boundary.geom_type == "LineString":
            lines = [boundary]
        elif boundary.geom_type == "MultiLineString":
            lines = list(boundary.geoms)
        if not lines:
            return None
        longest = max(lines, key=lambda item: item.length)
        if longest.length <= 1.0:
            return None
        coords = list(longest.coords)
        return (
            PlanarPoint(float(coords[0][0]), float(coords[0][1])),
            PlanarPoint(float(coords[-1][0]), float(coords[-1][1])),
        )

    def _build_room_transforms(
        self,
        payload: Mapping[str, Any],
        *,
        apartment_shift: PlanarPoint,
    ) -> dict[str, JsonObject]:
        rooms = payload.get("rooms")
        if not isinstance(rooms, list):
            return {}

        transforms: dict[str, JsonObject] = {}
        used_ids: set[str] = set()
        for index, raw_room in enumerate(rooms):
            if not isinstance(raw_room, Mapping):
                continue
            room_id = self._room_id(raw_room, index, used_ids)
            room_points = self._room_points(raw_room)
            if not room_points:
                continue
            bbox_original = PlanarBBox.from_points(room_points)
            bbox_apartment = bbox_original.shifted(apartment_shift.x, apartment_shift.y)
            transforms[room_id] = {
                "room_id": room_id,
                "name": str(raw_room.get("name") or room_id),
                "source_key": str(raw_room.get("key") or room_id),
                "bbox_original": bbox_original.as_dict(),
                "bbox_apartment": bbox_apartment.as_dict(),
                "origin_in_original": {
                    "x": _clean_float(bbox_original.min_x),
                    "y": _clean_float(bbox_original.min_y),
                },
                "origin_in_apartment": {
                    "x": _clean_float(bbox_apartment.min_x),
                    "y": _clean_float(bbox_apartment.min_y),
                },
            }
        return transforms

    def _build_system_inputs(
        self,
        *,
        payload: Mapping[str, Any],
        room_views: list[JsonObject],
        room_transforms: Mapping[str, JsonObject],
        tenant_id: str | None,
        user_id: str | None,
        description: str | None,
        special_notes: str | None,
        style: str | None,
    ) -> list[JsonObject]:
        room_openings = self._build_room_openings(
            payload=payload,
            room_transforms=room_transforms,
        )
        system_inputs: list[JsonObject] = []
        resolved_tenant_id = tenant_id or self._string_or_none(payload.get("tenant_id"))
        resolved_user_id = user_id or "coordinate_normalizer_preview"
        resolved_style = style or self._string_or_none(payload.get("style")) or "modern"
        default_height_mm = self._payload_height_mm(payload)

        for room_view in room_views:
            room_id = self._string_or_none(room_view.get("room_id"))
            if room_id is None:
                continue
            local_payload = self._mapping(room_view.get("local_payload"))
            shape_points = self._shape_points_from_room_payload(local_payload)
            if len(shape_points) < 3:
                continue
            room_name = self._string_or_none(room_view.get("name")) or room_id
            room_description = self._string_or_none(local_payload.get("description"))
            room_type = (
                self._string_or_none(local_payload.get("roomType"))
                or self._string_or_none(local_payload.get("room_type"))
                or self._infer_room_type(room_name)
            )
            floor_area_m2 = round(
                abs(self._polygon_area(shape_points)) / 1_000_000.0,
                3,
            )
            openings = room_openings.get(room_id, {"doors": [], "windows": []})
            user_input: JsonObject = {
                "description": description
                or room_description
                or f"Design {room_name} as a {room_type.replace('_', ' ')}.",
                "room_type": room_type,
                "floor_area_m2": floor_area_m2,
                "height": default_height_mm,
                "shape_points": shape_points,
                "windows": len(openings["windows"]),
                "window_direction": "",
                "style": resolved_style,
                "source_room_id": room_id,
                "source_room_name": room_name,
            }
            material_id = self._string_or_none(local_payload.get("materialId"))
            material_label = self._string_or_none(local_payload.get("materialLabel"))
            if material_id is not None:
                user_input["material_id"] = material_id
            if material_label is not None:
                user_input["material_label"] = material_label

            input_payload: JsonObject = {"user_input": user_input}
            if resolved_tenant_id is not None:
                input_payload["tenant_id"] = resolved_tenant_id
            if openings["doors"]:
                input_payload["doors"] = openings["doors"]
            if openings["windows"]:
                input_payload["windows"] = openings["windows"]

            pipeline_request: JsonObject = {
                "user_id": resolved_user_id,
                "input_payload": input_payload,
                "description": user_input["description"],
                "special_notes": special_notes,
            }
            system_inputs.append(
                {
                    "room_id": room_id,
                    "room_name": room_name,
                    "room_type": room_type,
                    "input_payload": input_payload,
                    "pipeline_run_request": pipeline_request,
                }
            )
        return system_inputs

    def _build_room_views(
        self,
        normalized_payload: Mapping[str, Any],
        *,
        room_transforms: Mapping[str, JsonObject],
    ) -> list[JsonObject]:
        rooms = normalized_payload.get("rooms")
        if not isinstance(rooms, list):
            return []

        room_views: list[JsonObject] = []
        used_ids: set[str] = set()
        for index, raw_room in enumerate(rooms):
            if not isinstance(raw_room, Mapping):
                continue
            room_id = self._room_id(raw_room, index, used_ids)
            transform = room_transforms.get(room_id)
            if not isinstance(transform, Mapping):
                continue
            origin = transform.get("origin_in_apartment")
            if not isinstance(origin, Mapping):
                continue
            origin_x = self._number(origin.get("x"))
            origin_y = self._number(origin.get("y"))
            if origin_x is None or origin_y is None:
                continue
            apartment_payload = deepcopy(dict(raw_room))
            local_payload = self._translate_json(
                apartment_payload,
                dx=-origin_x,
                dy=-origin_y,
            )
            room_views.append(
                {
                    "room_id": room_id,
                    "name": transform.get("name"),
                    "origin_in_apartment": {"x": origin_x, "y": origin_y},
                    "origin_in_original": transform.get("origin_in_original"),
                    "bbox_apartment": transform.get("bbox_apartment"),
                    "bbox_original": transform.get("bbox_original"),
                    "apartment_payload": apartment_payload,
                    "local_payload": local_payload,
                    "local_origin": {"x": 0.0, "y": 0.0},
                }
            )
        return room_views

    def _restore_delta(
        self,
        transform: Mapping[str, Any],
        *,
        coordinate_space: CoordinateSpace,
        room_id: str | None,
    ) -> tuple[float, float, str | None]:
        if coordinate_space == "apartment_normalized":
            apartment = self._mapping(transform.get("apartment"))
            inverse_shift = self._mapping(apartment.get("inverse_shift"))
            dx = self._number(inverse_shift.get("x"))
            dy = self._number(inverse_shift.get("y"))
            if dx is None or dy is None:
                shift = self._mapping(apartment.get("shift"))
                shift_x = self._number(shift.get("x"))
                shift_y = self._number(shift.get("y"))
                if shift_x is None or shift_y is None:
                    raise ValueError("Missing apartment shift in coordinate transform.")
                dx = -shift_x
                dy = -shift_y
            return dx, dy, None

        rooms = self._mapping(transform.get("rooms"))
        resolved_room_id = room_id
        if resolved_room_id is None:
            if len(rooms) != 1:
                raise ValueError(
                    "room_id is required when restoring room-local coordinates."
                )
            resolved_room_id = next(iter(rooms))

        room_transform = self._mapping(rooms.get(resolved_room_id))
        origin = self._mapping(room_transform.get("origin_in_original"))
        dx = self._number(origin.get("x"))
        dy = self._number(origin.get("y"))
        if dx is None or dy is None:
            raise ValueError(f"Missing original origin for room_id={resolved_room_id}.")
        return dx, dy, resolved_room_id

    def _build_room_openings(
        self,
        *,
        payload: Mapping[str, Any],
        room_transforms: Mapping[str, JsonObject],
    ) -> dict[str, dict[str, list[JsonObject]]]:
        out: dict[str, dict[str, list[JsonObject]]] = {
            room_id: {"doors": [], "windows": []} for room_id in room_transforms
        }
        openings = self._extract_explicit_openings(payload)
        openings.extend(self._extract_object_openings(payload))

        for opening in openings:
            segment = opening.get("segment_mm")
            if not isinstance(segment, list) or len(segment) != 2:
                continue
            segment_points = [self._point_from_mapping(item) for item in segment]
            if segment_points[0] is None or segment_points[1] is None:
                continue
            center = PlanarPoint(
                x=(segment_points[0].x + segment_points[1].x) / 2.0,
                y=(segment_points[0].y + segment_points[1].y) / 2.0,
            )
            opening_kind = str(opening.get("kind") or "door")
            bucket = "windows" if opening_kind == "window" else "doors"
            for room_id, transform in room_transforms.items():
                bbox = self._bbox_from_mapping(transform.get("bbox_original"))
                origin = self._point_from_mapping(transform.get("origin_in_original"))
                if bbox is None or origin is None:
                    continue
                if not self._bbox_contains_point(bbox, center, tolerance=300.0):
                    continue
                local_opening = deepcopy(opening)
                local_opening.pop("kind", None)
                local_opening["segment_mm"] = [
                    {
                        "x": _clean_float(segment_points[0].x - origin.x),
                        "y": _clean_float(segment_points[0].y - origin.y),
                    },
                    {
                        "x": _clean_float(segment_points[1].x - origin.x),
                        "y": _clean_float(segment_points[1].y - origin.y),
                    },
                ]
                position = self._point_from_mapping(opening.get("position_mm"))
                if position is not None:
                    local_opening["position_mm"] = {
                        "x": _clean_float(position.x - origin.x),
                        "y": _clean_float(position.y - origin.y),
                    }
                out[room_id][bucket].append(local_opening)
        return out

    def _extract_explicit_openings(self, payload: Mapping[str, Any]) -> list[JsonObject]:
        openings: list[JsonObject] = []
        for kind, key in (("door", "doors"), ("window", "windows")):
            rows = payload.get(key)
            if not isinstance(rows, list):
                continue
            for index, row in enumerate(rows, start=1):
                if not isinstance(row, Mapping):
                    continue
                segment = row.get("segment_mm") or row.get("segment")
                if not isinstance(segment, list):
                    continue
                points = [self._point_from_mapping(item) for item in segment[:2]]
                if len(points) < 2 or points[0] is None or points[1] is None:
                    continue
                opening: JsonObject = {
                    "id": self._string_or_none(row.get("id")) or f"{kind}_{index}",
                    "kind": kind,
                    "segment_mm": [points[0].as_dict(), points[1].as_dict()],
                }
                if kind == "door":
                    opening["swing_radius_mm"] = self._opening_dimension(
                        row.get("swing_radius_mm"),
                        fallback=900.0,
                    )
                    hinge_hint = self._string_or_none(row.get("hinge_hint"))
                    if hinge_hint is not None:
                        opening["hinge_hint"] = hinge_hint
                else:
                    opening["clearance_mm"] = self._opening_dimension(
                        row.get("clearance_mm"),
                        fallback=150.0,
                    )
                openings.append(opening)
        return openings

    def _extract_object_openings(self, payload: Mapping[str, Any]) -> list[JsonObject]:
        objects = payload.get("objects")
        if not isinstance(objects, list):
            return []
        wall_by_id = self._wall_map(payload)
        openings: list[JsonObject] = []
        for index, row in enumerate(objects, start=1):
            if not isinstance(row, Mapping):
                continue
            kind = self._object_opening_kind(row)
            if kind is None:
                continue
            position = self._position_point(row.get("position"))
            if position is None:
                continue
            wall = wall_by_id.get(str(row.get("snappedToWall") or ""))
            if wall is None:
                wall = self._nearest_wall(wall_by_id.values(), position)
            if wall is None:
                continue
            width = self._object_opening_width(row, kind=kind)
            segment = self._opening_segment_on_wall(wall, position, width)
            opening: JsonObject = {
                "id": self._string_or_none(row.get("id")) or f"{kind}_{index}",
                "kind": kind,
                "segment_mm": [segment[0].as_dict(), segment[1].as_dict()],
                "position_mm": position.as_dict(),
            }
            rotation = self._quaternion_from_value(row.get("rotation"))
            if rotation is not None:
                opening["rotation"] = self._quaternion_dict(rotation)
                opening["rotation_yaw_deg"] = _clean_float(
                    self._yaw_degrees_from_quaternion(rotation)
                )
            if kind == "door":
                opening["swing_radius_mm"] = width
            else:
                opening["clearance_mm"] = 150.0
            openings.append(opening)
        return openings

    def _wall_map(self, payload: Mapping[str, Any]) -> dict[str, tuple[PlanarPoint, PlanarPoint]]:
        walls = payload.get("walls")
        if not isinstance(walls, list):
            return {}
        out: dict[str, tuple[PlanarPoint, PlanarPoint]] = {}
        for index, wall in enumerate(walls, start=1):
            if not isinstance(wall, Mapping):
                continue
            start = self._point_from_mapping(wall.get("startPoint") or wall.get("start"))
            end = self._point_from_mapping(wall.get("endPoint") or wall.get("end"))
            if start is None or end is None:
                continue
            wall_id = self._string_or_none(wall.get("id")) or f"wall_{index}"
            out[wall_id] = (start, end)
        return out

    def _object_opening_kind(self, row: Mapping[str, Any]) -> str | None:
        text_parts = [
            self._string_or_none(row.get("objectRole")),
            self._string_or_none(row.get("name")),
            self._string_or_none(row.get("category")),
        ]
        text = " ".join(part or "" for part in text_parts).lower()
        if any(term in text for term in ("window", "cua so", "cửa sổ")):
            return "window"
        if any(term in text for term in ("door", "cua", "cửa")):
            return "door"
        return None

    def _object_opening_width(self, row: Mapping[str, Any], *, kind: str) -> float:
        size = row.get("size")
        if isinstance(size, list) and size:
            width = self._number(size[0])
            if width is not None and width > 0:
                return width
        return 1200.0 if kind == "window" else 900.0

    def _opening_segment_on_wall(
        self,
        wall: tuple[PlanarPoint, PlanarPoint],
        center_hint: PlanarPoint,
        width: float,
    ) -> tuple[PlanarPoint, PlanarPoint]:
        start, end = wall
        dx = end.x - start.x
        dy = end.y - start.y
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            return center_hint, center_hint
        ux = dx / length
        uy = dy / length
        projected_t = (
            (center_hint.x - start.x) * ux + (center_hint.y - start.y) * uy
        )
        half_width = min(max(width / 2.0, 1.0), length / 2.0)
        center_t = min(max(projected_t, half_width), max(half_width, length - half_width))
        center = PlanarPoint(x=start.x + ux * center_t, y=start.y + uy * center_t)
        return (
            PlanarPoint(x=center.x - ux * half_width, y=center.y - uy * half_width),
            PlanarPoint(x=center.x + ux * half_width, y=center.y + uy * half_width),
        )

    def _nearest_wall(
        self,
        walls: Any,
        point: PlanarPoint,
    ) -> tuple[PlanarPoint, PlanarPoint] | None:
        best_wall: tuple[PlanarPoint, PlanarPoint] | None = None
        best_distance: float | None = None
        for wall in walls:
            distance = self._distance_to_segment(point, wall)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_wall = wall
        return best_wall

    def _distance_to_segment(
        self,
        point: PlanarPoint,
        wall: tuple[PlanarPoint, PlanarPoint],
    ) -> float:
        start, end = wall
        dx = end.x - start.x
        dy = end.y - start.y
        length_sq = dx * dx + dy * dy
        if length_sq <= 1e-9:
            return math.hypot(point.x - start.x, point.y - start.y)
        t = ((point.x - start.x) * dx + (point.y - start.y) * dy) / length_sq
        clamped_t = min(max(t, 0.0), 1.0)
        px = start.x + clamped_t * dx
        py = start.y + clamped_t * dy
        return math.hypot(point.x - px, point.y - py)

    def _shape_points_from_room_payload(self, room_payload: Mapping[str, Any]) -> list[JsonObject]:
        points = self._room_points(room_payload)
        return [
            {
                "x": _clean_float(point.x),
                "y": _clean_float(point.y),
            }
            for point in points
        ]

    def _polygon_area(self, points: list[JsonObject]) -> float:
        if len(points) < 3:
            return 0.0
        total = 0.0
        for index, point in enumerate(points):
            next_point = points[(index + 1) % len(points)]
            x1 = self._number(point.get("x")) or 0.0
            y1 = self._number(point.get("y")) or 0.0
            x2 = self._number(next_point.get("x")) or 0.0
            y2 = self._number(next_point.get("y")) or 0.0
            total += x1 * y2 - x2 * y1
        return total / 2.0

    def _payload_height_mm(self, payload: Mapping[str, Any]) -> int:
        for key in ("ceiling_height_mm", "height_mm", "height"):
            value = self._number(payload.get(key))
            if value is not None and value > 0:
                return int(round(value))
        walls = payload.get("walls")
        if isinstance(walls, list):
            heights = [
                self._number(wall.get("height"))
                for wall in walls
                if isinstance(wall, Mapping)
            ]
            valid_heights = [height for height in heights if height is not None and height > 0]
            if valid_heights:
                return int(round(max(valid_heights)))
        return 2800

    def _infer_room_type(self, room_name: str) -> str:
        normalized = room_name.strip().lower()
        if any(term in normalized for term in ("ngủ", "bed")):
            return "bedroom"
        if any(term in normalized for term in ("tắm", "wc", "bath", "toilet")):
            return "bathroom"
        if any(term in normalized for term in ("sinh hoạt", "khách", "living")):
            return "living_room"
        if any(term in normalized for term in ("bếp", "kitchen")):
            return "kitchen"
        if any(term in normalized for term in ("logia", "ban công", "balcony")):
            return "balcony"
        return "room"

    def _point_from_mapping(self, value: Any) -> PlanarPoint | None:
        if isinstance(value, Mapping):
            x = self._number(value.get("x"))
            if "z" in value:
                y = self._number(value.get("z"))
            else:
                y = self._number(value.get("y"))
            if x is not None and y is not None:
                return PlanarPoint(x=x, y=y)
        if isinstance(value, list):
            return self._point_from_list(value, parent_key=None)
        return None

    def _position_point(self, value: Any) -> PlanarPoint | None:
        if isinstance(value, Mapping):
            x = self._number(value.get("x"))
            z = self._number(value.get("z"))
            if x is not None and z is not None:
                return PlanarPoint(x=x, y=z)
        if isinstance(value, list):
            return self._point_from_list(value, parent_key="position")
        return None

    def _bbox_from_mapping(self, value: Any) -> PlanarBBox | None:
        if not isinstance(value, Mapping) or not self._is_bbox_mapping(value):
            return None
        return PlanarBBox(
            min_x=float(value["min_x"]),
            min_y=float(value["min_y"]),
            max_x=float(value["max_x"]),
            max_y=float(value["max_y"]),
        )

    def _bbox_contains_point(
        self,
        bbox: PlanarBBox,
        point: PlanarPoint,
        *,
        tolerance: float,
    ) -> bool:
        return (
            bbox.min_x - tolerance <= point.x <= bbox.max_x + tolerance
            and bbox.min_y - tolerance <= point.y <= bbox.max_y + tolerance
        )

    def _opening_dimension(self, value: Any, *, fallback: float) -> float:
        dimension = self._number(value)
        if dimension is not None and dimension > 0:
            return dimension
        return fallback

    def _string_or_none(self, value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        clean = value.strip()
        return clean or None

    def _collect_bounds_points(
        self,
        value: Any,
        *,
        parent_key: str | None = None,
        skip_keys: set[str],
    ) -> list[PlanarPoint]:
        if isinstance(value, Mapping):
            points: list[PlanarPoint] = []
            if parent_key in _QUATERNION_KEYS or parent_key in _DIMENSION_KEYS:
                return points
            if self._is_bbox_mapping(value):
                points.extend(
                    [
                        PlanarPoint(float(value["min_x"]), float(value["min_y"])),
                        PlanarPoint(float(value["max_x"]), float(value["max_y"])),
                    ]
                )
            if parent_key in _POSITION_3D_KEYS and self._is_xyz_mapping(value):
                points.append(PlanarPoint(float(value["x"]), float(value["z"])))
                return points
            if self._is_xy_mapping(value):
                points.append(PlanarPoint(float(value["x"]), float(value["y"])))
            for key, child in value.items():
                key_text = str(key)
                if key_text in skip_keys:
                    continue
                points.extend(
                    self._collect_bounds_points(
                        child,
                        parent_key=key_text,
                        skip_keys=skip_keys,
                    )
                )
            return points

        if isinstance(value, list):
            point = self._point_from_list(value, parent_key=parent_key)
            if point is not None:
                return [point]
            points: list[PlanarPoint] = []
            for child in value:
                points.extend(
                    self._collect_bounds_points(
                        child,
                        parent_key=parent_key,
                        skip_keys=skip_keys,
                    )
                )
            return points

        return []

    def _room_points(self, room: Mapping[str, Any]) -> list[PlanarPoint]:
        points: list[PlanarPoint] = []
        for key in ("polygons", "polygon", "polygon_ccw", "shape_points"):
            value = room.get(key)
            if value is None:
                continue
            points.extend(
                self._collect_bounds_points(
                    value,
                    parent_key=key,
                    skip_keys=set(),
                )
            )
        return points

    def _translate_json(self, value: Any, *, dx: float, dy: float) -> Any:
        return self._translate_json_inner(value, dx=dx, dy=dy, parent_key=None)

    def _scale_json(self, value: Any, *, factor: float) -> Any:
        if abs(factor - 1.0) <= 1e-12:
            return deepcopy(value)
        return self._scale_json_inner(value, factor=factor, parent_key=None)

    def _scale_json_inner(
        self,
        value: Any,
        *,
        factor: float,
        parent_key: str | None,
    ) -> Any:
        if isinstance(value, Mapping):
            scaled: JsonObject = {}
            for key, child in value.items():
                key_text = str(key)
                scaled[key_text] = self._scale_json_inner(
                    child,
                    factor=factor,
                    parent_key=key_text,
                )
            if parent_key in _QUATERNION_KEYS:
                return scaled
            if self._is_bbox_mapping(scaled):
                scaled["min_x"] = _clean_float(float(scaled["min_x"]) * factor)
                scaled["max_x"] = _clean_float(float(scaled["max_x"]) * factor)
                scaled["min_y"] = _clean_float(float(scaled["min_y"]) * factor)
                scaled["max_y"] = _clean_float(float(scaled["max_y"]) * factor)
            elif parent_key in _POSITION_3D_KEYS and self._is_xyz_mapping(scaled):
                scaled["x"] = _clean_float(float(scaled["x"]) * factor)
                scaled["y"] = _clean_float(float(scaled["y"]) * factor)
                scaled["z"] = _clean_float(float(scaled["z"]) * factor)
            elif self._is_xy_mapping(scaled):
                scaled["x"] = _clean_float(float(scaled["x"]) * factor)
                scaled["y"] = _clean_float(float(scaled["y"]) * factor)
            return scaled

        if isinstance(value, list):
            if parent_key in _QUATERNION_KEYS:
                return list(value)
            if parent_key in _DIMENSION_KEYS and self._is_numeric_list(value, 1):
                return [
                    _clean_float(float(item) * factor)
                    if self._number(item) is not None
                    else item
                    for item in value
                ]
            if parent_key in _POSITION_3D_KEYS and self._is_numeric_list(value, 3):
                return [
                    _clean_float(float(value[0]) * factor),
                    _clean_float(float(value[1]) * factor),
                    _clean_float(float(value[2]) * factor),
                    *value[3:],
                ]
            if parent_key in _POINT_KEYS and self._is_numeric_list(value, 2):
                return [
                    _clean_float(float(value[0]) * factor),
                    _clean_float(float(value[1]) * factor),
                    *value[2:],
                ]
            if parent_key in _POINT_TREE_KEYS:
                return self._scale_point_tree(value, factor=factor)
            return [
                self._scale_json_inner(child, factor=factor, parent_key=parent_key)
                for child in value
            ]

        if (
            parent_key
            in {
                "height",
                "height_mm",
                "ceiling_height_mm",
                "thickness",
                "thickness_mm",
                "clearance_mm",
                "swing_radius_mm",
                "leaf_width_mm",
            }
            and self._number(value) is not None
        ):
            return _clean_float(float(value) * factor)
        return value

    def _scale_point_tree(self, value: list[Any], *, factor: float) -> list[Any]:
        if self._is_numeric_list(value, 2):
            return [
                _clean_float(float(value[0]) * factor),
                _clean_float(float(value[1]) * factor),
                *value[2:],
            ]
        return [
            self._scale_point_tree(child, factor=factor)
            if isinstance(child, list)
            else self._scale_json_inner(child, factor=factor, parent_key=None)
            for child in value
        ]

    def _translate_json_inner(
        self,
        value: Any,
        *,
        dx: float,
        dy: float,
        parent_key: str | None,
    ) -> Any:
        if isinstance(value, Mapping):
            translated: JsonObject = {}
            for key, child in value.items():
                key_text = str(key)
                translated[key_text] = self._translate_json_inner(
                    child,
                    dx=dx,
                    dy=dy,
                    parent_key=key_text,
                )

            if parent_key in _QUATERNION_KEYS or parent_key in _DIMENSION_KEYS:
                return translated
            if parent_key in _POSITION_3D_KEYS and self._is_xyz_mapping(translated):
                translated["x"] = _clean_float(float(translated["x"]) + dx)
                translated["z"] = _clean_float(float(translated["z"]) + dy)
            elif self._is_bbox_mapping(translated):
                translated["min_x"] = _clean_float(float(translated["min_x"]) + dx)
                translated["max_x"] = _clean_float(float(translated["max_x"]) + dx)
                translated["min_y"] = _clean_float(float(translated["min_y"]) + dy)
                translated["max_y"] = _clean_float(float(translated["max_y"]) + dy)
            elif self._is_xy_mapping(translated):
                translated["x"] = _clean_float(float(translated["x"]) + dx)
                translated["y"] = _clean_float(float(translated["y"]) + dy)
            return translated

        if isinstance(value, list):
            if parent_key in _POSITION_3D_KEYS and self._is_numeric_list(value, 3):
                out = list(value)
                out[0] = _clean_float(float(out[0]) + dx)
                out[2] = _clean_float(float(out[2]) + dy)
                return out
            if parent_key in _POSITION_3D_KEYS and self._is_numeric_list(value, 2):
                return [
                    _clean_float(float(value[0]) + dx),
                    _clean_float(float(value[1]) + dy),
                    *value[2:],
                ]
            if parent_key in _POINT_KEYS and self._is_numeric_list(value, 2):
                return [
                    _clean_float(float(value[0]) + dx),
                    _clean_float(float(value[1]) + dy),
                    *value[2:],
                ]
            if parent_key in _POINT_TREE_KEYS:
                return self._translate_point_tree(value, dx=dx, dy=dy)
            return [
                self._translate_json_inner(
                    child,
                    dx=dx,
                    dy=dy,
                    parent_key=parent_key,
                )
                for child in value
            ]

        return value

    def _translate_point_tree(
        self,
        value: list[Any],
        *,
        dx: float,
        dy: float,
    ) -> list[Any]:
        if self._is_numeric_list(value, 2):
            return [
                _clean_float(float(value[0]) + dx),
                _clean_float(float(value[1]) + dy),
                *value[2:],
            ]
        return [
            self._translate_point_tree(child, dx=dx, dy=dy)
            if isinstance(child, list)
            else self._translate_json_inner(
                child,
                dx=dx,
                dy=dy,
                parent_key=None,
            )
            for child in value
        ]

    def _convert_rotations(self, value: Any, rotation_input: RotationInput) -> Any:
        if isinstance(value, list):
            return [self._convert_rotations(child, rotation_input) for child in value]
        if not isinstance(value, Mapping):
            return value

        converted = {
            str(key): self._convert_rotations(child, rotation_input)
            for key, child in value.items()
        }
        quaternion = self._rotation_quaternion(converted, rotation_input)
        if quaternion is not None:
            converted["rotation"] = self._quaternion_dict(quaternion)
        return converted

    def _format_frontend_positions(
        self,
        value: Any,
        parent_key: str | None = None,
    ) -> Any:
        if isinstance(value, list):
            if parent_key in _POSITION_3D_KEYS and self._is_numeric_list(value, 3):
                return {
                    "x": float(value[0]),
                    "y": float(value[1]),
                    "z": float(value[2]),
                }
            if parent_key in _POSITION_3D_KEYS and self._is_numeric_list(value, 2):
                return {
                    "x": float(value[0]),
                    "y": 0.0,
                    "z": float(value[1]),
                }
            return [
                self._format_frontend_positions(child, parent_key=parent_key)
                for child in value
            ]

        if isinstance(value, Mapping):
            if parent_key in _POSITION_3D_KEYS and self._is_xyz_mapping(value):
                return {
                    "x": float(value["x"]),
                    "y": float(value["y"]),
                    "z": float(value["z"]),
                }
            if parent_key in _POSITION_3D_KEYS and self._is_xy_mapping(value):
                return {
                    "x": float(value["x"]),
                    "y": 0.0,
                    "z": float(value["y"]),
                }
            return {
                str(key): self._format_frontend_positions(child, parent_key=str(key))
                for key, child in value.items()
            }

        return value

    def _rotation_quaternion(
        self,
        item: Mapping[str, Any],
        rotation_input: RotationInput,
    ) -> list[float] | None:
        existing_rotation = item.get("rotation")
        if rotation_input in {"auto", "quaternion"}:
            quaternion = self._quaternion_from_value(existing_rotation)
            if quaternion is not None:
                return quaternion
            quaternion = self._quaternion_from_value(item.get("rotation_quaternion"))
            if quaternion is not None:
                return quaternion

        degrees = self._rotation_degrees(item, rotation_input)
        if degrees is None:
            return None
        radians = math.radians(degrees)
        return self._yaw_radians_to_quaternion(radians)

    def _rotation_degrees(
        self,
        item: Mapping[str, Any],
        rotation_input: RotationInput,
    ) -> float | None:
        for key in ("rotation_ccw", "rotation_deg", "rot"):
            value = self._number(item.get(key))
            if value is not None:
                return value if rotation_input != "radians" else math.degrees(value)

        rotation_value = item.get("rotation")
        numeric_rotation = self._number(rotation_value)
        if numeric_rotation is not None:
            return (
                math.degrees(numeric_rotation)
                if rotation_input == "radians"
                else numeric_rotation
            )

        if isinstance(rotation_value, list) and self._is_numeric_list(rotation_value, 3):
            yaw = float(rotation_value[1])
            return math.degrees(yaw) if rotation_input != "degrees" else yaw
        return None

    def _quaternion_from_value(self, value: Any) -> list[float] | None:
        if isinstance(value, Mapping):
            x = self._number(value.get("x"))
            y = self._number(value.get("y"))
            z = self._number(value.get("z"))
            w = self._number(value.get("w"))
            if None in {x, y, z, w}:
                return None
            return self._normalize_quaternion(
                [x or 0.0, y or 0.0, z or 0.0, w or 0.0]
            )
        if isinstance(value, list) and self._is_numeric_list(value, 4):
            return self._normalize_quaternion([float(item) for item in value[:4]])
        return None

    def _normalize_quaternion(self, value: list[float]) -> list[float] | None:
        norm = math.sqrt(sum(item * item for item in value))
        if norm <= 1e-12:
            return None
        return [self._round_float(item / norm) for item in value]

    def _quaternion_dict(self, value: list[float]) -> JsonObject:
        return {
            "x": value[0],
            "y": value[1],
            "z": value[2],
            "w": value[3],
        }

    def _yaw_radians_to_quaternion(self, radians: float) -> list[float]:
        half_angle = radians / 2.0
        return [
            0.0,
            self._round_float(math.sin(half_angle)),
            0.0,
            self._round_float(math.cos(half_angle)),
        ]

    def _yaw_degrees_from_quaternion(self, value: list[float]) -> float:
        x, y, z, w = value
        radians = math.atan2(
            2.0 * (w * y + x * z),
            1.0 - 2.0 * (x * x + y * y),
        )
        return math.degrees(radians) % 360.0

    def _point_from_list(
        self,
        value: list[Any],
        *,
        parent_key: str | None,
    ) -> PlanarPoint | None:
        if parent_key in _POSITION_3D_KEYS and self._is_numeric_list(value, 3):
            return PlanarPoint(x=float(value[0]), y=float(value[2]))
        if self._is_numeric_list(value, 2):
            return PlanarPoint(x=float(value[0]), y=float(value[1]))
        return None

    def _room_id(
        self,
        room: Mapping[str, Any],
        index: int,
        used_ids: set[str],
    ) -> str:
        raw_id = room.get("room_id") or room.get("id") or room.get("key")
        base_id = str(raw_id).strip() if raw_id is not None else ""
        if not base_id:
            base_id = f"room_{index + 1}"
        room_id = base_id
        suffix = 2
        while room_id in used_ids:
            room_id = f"{base_id}_{suffix}"
            suffix += 1
        used_ids.add(room_id)
        return room_id

    def _is_bbox_mapping(self, value: Mapping[str, Any]) -> bool:
        return all(self._number(value.get(key)) is not None for key in _BBOX_KEYS)

    def _is_xy_mapping(self, value: Mapping[str, Any]) -> bool:
        if "z" in value and "w" in value:
            return False
        return self._number(value.get("x")) is not None and self._number(
            value.get("y")
        ) is not None

    def _is_xyz_mapping(self, value: Mapping[str, Any]) -> bool:
        return (
            self._number(value.get("x")) is not None
            and self._number(value.get("y")) is not None
            and self._number(value.get("z")) is not None
        )

    def _is_numeric_list(self, value: list[Any], min_length: int) -> bool:
        return len(value) >= min_length and all(
            self._number(item) is not None for item in value[:min_length]
        )

    def _mapping(self, value: Any) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            return {}
        return value

    def _number(self, value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int | float):
            return float(value)
        return None

    def _round_float(self, value: float) -> float:
        rounded = round(value, 12)
        if abs(rounded) <= 1e-12:
            return 0.0
        return rounded


_BBOX_KEYS = {"min_x", "min_y", "max_x", "max_y"}


def _clean_float(value: float) -> float:
    rounded = round(value, 12)
    if abs(rounded) <= 1e-12:
        return 0.0
    nearest_integer = round(rounded)
    if abs(rounded - nearest_integer) <= 1e-9:
        return float(nearest_integer)
    return rounded
