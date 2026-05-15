from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

from stylist.room_essentials_seed import ROOM_SURFACE_GROUPS
from stylist.tools import ListInventoryByTypes

STACK_GAP_MM = 50
SUPPORT_INSET_MM = 40
UNDERLAY_MARGIN_MM = 150
MOUNT_INSET_MM = 20
OPENING_HANG_HEIGHT_MM = 1800
CEILING_FOOTPRINT_MM = 450
UTILITY_GAP_MM = 80

DEFAULT_SURFACES = {
    "wall_color_hex": "#F3EEE6",
    "floor_color_hex": "#C6A27A",
    "ceiling_color_hex": "#FAF7F1",
}
DEFAULT_DOOR_COLOR = "#8F6F58"
DEFAULT_WINDOW_COLOR = "#D7E3EE"
DEFAULT_OBJECT_COLOR = "#CFCAC2"

STRUCTURAL_SKIP_TYPES = {"mattress", "headboard"}
UNDERLAY_TYPES = {"rug"}
OPENING_MOUNTED_TYPES = {"curtain", "blind"}
CEILING_MOUNTED_TYPES = {"ceiling_light", "pendant_light", "track_light"}
WALL_MOUNTED_TYPES = {
    "mirror",
    "wall_art",
    "clock",
    "wall_sconce",
    "air_conditioner",
    "whiteboard",
    "projector_screen",
    "bathroom_shelf",
    "medicine_cabinet",
    "towel_rack",
    "shower_niche",
    "kitchen_wall_cabinet",
    "range_hood",
}
FLOOR_SIDE_TYPES = {"pet_bed", "dehumidifier", "heater"}

MAX_ITEMS_BY_ANCHOR: dict[str, int] = {
    "bed": 2,
    "sofa": 2,
    "sectional_sofa": 2,
    "desk": 2,
    "tv_console": 2,
    "nightstand": 1,
    "side_table": 1,
    "coffee_table": 1,
    "console_table": 1,
    "dresser": 1,
    "dining_table": 1,
    "buffet_sideboard": 1,
    "bathroom_vanity": 2,
    "__opening__": 1,
    "__wall__": 1,
    "__ceiling__": 1,
    "__utility_zone__": 1,
}

ANCHOR_PRIORITY: dict[str, tuple[str, ...]] = {
    "bed": ("rug", "throw_blanket", "cushion", "pet_bed"),
    "sofa": ("rug", "cushion", "throw_blanket"),
    "sectional_sofa": ("rug", "cushion", "throw_blanket"),
    "nightstand": ("bedside_lamp",),
    "desk": (
        "desk_lamp",
        "laptop",
        "monitor",
        "keyboard",
        "mouse",
        "speaker",
        "smart_speaker",
        "printer",
        "desktop_pc",
    ),
    "tv_console": ("speaker", "smart_speaker"),
    "dresser": ("mirror", "decor"),
    "side_table": ("table_lamp", "plant", "decor", "vase", "smart_speaker", "speaker"),
    "console_table": ("mirror", "decor", "table_lamp", "plant", "vase"),
    "coffee_table": ("vase", "decor"),
    "dining_table": ("rug", "vase"),
    "buffet_sideboard": ("mirror", "decor", "speaker", "smart_speaker"),
    "bathroom_vanity": (
        "mirror",
        "medicine_cabinet",
        "bathroom_shelf",
        "wall_sconce",
        "decor",
        "plant",
    ),
    "__opening__": ("curtain", "blind"),
    "__wall__": ("wall_art", "clock", "whiteboard", "wall_sconce", "air_conditioner"),
    "__ceiling__": ("ceiling_light", "pendant_light", "track_light"),
    "__utility_zone__": ("dehumidifier", "heater"),
}

CEILING_PRIORITY_BY_ROOM: dict[str, tuple[str, ...]] = {
    "dining_room": ("pendant_light", "ceiling_light", "track_light"),
    "home_office": ("track_light", "ceiling_light", "pendant_light"),
    "kitchen": ("track_light", "ceiling_light", "pendant_light"),
}

OPENING_PRIORITY_BY_ROOM: dict[str, tuple[str, ...]] = {
    "bathroom": ("blind", "curtain"),
    "kitchen": ("blind", "curtain"),
    "home_office": ("blind", "curtain"),
}

DEFAULT_DIMENSIONS_MM: dict[str, tuple[int, int, int]] = {
    "rug": (1800, 1200, 10),
    "laptop": (350, 240, 20),
    "monitor": (620, 180, 420),
    "desktop_pc": (220, 420, 450),
    "keyboard": (450, 150, 30),
    "mouse": (120, 90, 40),
    "printer": (420, 320, 200),
    "desk_lamp": (220, 220, 480),
    "bedside_lamp": (220, 220, 420),
    "table_lamp": (240, 240, 420),
    "speaker": (180, 160, 260),
    "smart_speaker": (160, 160, 220),
    "decor": (220, 220, 220),
    "vase": (180, 180, 320),
    "plant": (260, 260, 420),
    "tv": (1200, 120, 700),
    "mirror": (800, 80, 900),
    "wall_art": (900, 70, 650),
    "clock": (300, 60, 300),
    "wall_sconce": (160, 80, 320),
    "air_conditioner": (950, 260, 260),
    "whiteboard": (1200, 70, 800),
    "projector_screen": (1500, 70, 900),
    "curtain": (1800, 140, OPENING_HANG_HEIGHT_MM),
    "blind": (1600, 80, OPENING_HANG_HEIGHT_MM),
    "ceiling_light": (CEILING_FOOTPRINT_MM, CEILING_FOOTPRINT_MM, 180),
    "pendant_light": (420, 420, 280),
    "track_light": (1200, 180, 220),
    "throw_blanket": (650, 500, 40),
    "cushion": (420, 320, 120),
    "pet_bed": (700, 550, 180),
    "bathroom_shelf": (700, 120, 260),
    "medicine_cabinet": (700, 120, 750),
    "towel_rack": (600, 80, 180),
    "shower_niche": (500, 120, 300),
    "kitchen_wall_cabinet": (900, 180, 700),
    "range_hood": (900, 180, 700),
    "dehumidifier": (320, 260, 520),
    "heater": (280, 220, 420),
    "cooktop": (600, 520, 60),
    "microwave": (540, 420, 330),
    "rice_cooker": (260, 260, 280),
    "electric_kettle": (220, 180, 260),
    "coffee_machine": (320, 240, 340),
    "toaster": (340, 220, 220),
    "air_fryer": (340, 320, 360),
    "blender": (180, 180, 420),
    "oven": (600, 560, 600),
}


@dataclass(frozen=True)
class _Rect:
    min_x: int
    min_y: int
    max_x: int
    max_y: int

    @property
    def width(self) -> int:
        return max(0, self.max_x - self.min_x)

    @property
    def height(self) -> int:
        return max(0, self.max_y - self.min_y)

    @property
    def center_x(self) -> float:
        return (self.min_x + self.max_x) / 2.0

    @property
    def center_y(self) -> float:
        return (self.min_y + self.max_y) / 2.0

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_bbox(self) -> dict[str, int]:
        return {
            "min_x": self.min_x,
            "min_y": self.min_y,
            "max_x": self.max_x,
            "max_y": self.max_y,
        }

    def to_polygon(self) -> list[dict[str, int]]:
        return [
            {"x": self.min_x, "y": self.min_y},
            {"x": self.max_x, "y": self.min_y},
            {"x": self.max_x, "y": self.max_y},
            {"x": self.min_x, "y": self.max_y},
        ]


@dataclass(frozen=True)
class _Support:
    instance_id: str
    object_type: str
    cluster_id: str | None
    rect: _Rect


@dataclass(frozen=True)
class _Profile:
    length_mm: int
    width_mm: int
    height_mm: int

    def footprint(self) -> tuple[int, int]:
        return self.length_mm, self.width_mm

    def wall_projection(self) -> tuple[int, int]:
        width = max(self.length_mm, self.width_mm)
        height = self.height_mm or min(self.length_mm, self.width_mm)
        return width, max(60, height)


def build_deterministic_stylist_payload(
    layout_json: dict[str, Any],
    *,
    tenant_id: str | None = None,
) -> dict[str, Any]:
    room = layout_json.get("room") if isinstance(layout_json.get("room"), dict) else {}
    room_polygon = _normalize_polygon(room.get("polygon_ccw"))
    room_rect = (
        _rect_from_polygon(room_polygon)
        or _rect_from_layout_objects(layout_json.get("objects"))
        or _Rect(0, 0, 4000, 4000)
    )

    room_type = str(room.get("room_type") or "unknown")
    openings = _normalize_openings(layout_json.get("openings") or room.get("openings"))

    existing_rows, supports = _build_existing_objects(layout_json.get("objects") or [])
    anchor_map = _anchor_map_for_room(room_type)
    inventory_profiles = _load_inventory_profiles(
        tenant_id=tenant_id,
        candidate_types=_collect_candidate_types(anchor_map, supports, openings),
    )

    generated_rows = _build_generated_objects(
        room_type=room_type,
        room_rect=room_rect,
        supports=supports,
        openings=openings,
        anchor_map=anchor_map,
        inventory_profiles=inventory_profiles,
        existing_rows=existing_rows,
    )

    door_opening_colors = [
        {"id": row["id"], "color_hex": DEFAULT_DOOR_COLOR}
        for row in openings.get("doors", [])
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    ]
    window_opening_colors = [
        {"id": row["id"], "color_hex": DEFAULT_WINDOW_COLOR}
        for row in openings.get("windows", [])
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    ]

    missing: list[str] = []
    status = "OK"
    if len(room_polygon) < 3:
        status = "NEED_INFO"
        missing.append("room.polygon_ccw")

    objects, validation = _validate_styled_objects(
        room_rect=room_rect,
        objects=existing_rows + generated_rows,
    )

    return {
        "status": status,
        "room": {
            "room_id": str(room.get("room_id") or "room_1"),
            "room_type": room_type,
            "polygon_ccw": deepcopy(room_polygon),
            "obstacles": deepcopy(room.get("obstacles") or []),
            "openings": openings,
            "surfaces": deepcopy(DEFAULT_SURFACES),
            "opening_colors": {
                "doors": door_opening_colors,
                "windows": window_opening_colors,
            },
        },
        "objects": objects,
        "validation": validation,
        "notes": [
            "Accessory placement used deterministic support rules and fixed 50 mm spacing."
        ],
        "missing": missing,
    }


def _anchor_map_for_room(room_type: str) -> dict[str, list[str]]:
    groups = ROOM_SURFACE_GROUPS.get(room_type) or {}
    raw = groups.get("can_stack_or_be_stacked_or_hang_or_soft")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, list[str]] = {}
    for anchor, values in raw.items():
        if not isinstance(anchor, str) or not isinstance(values, list):
            continue
        out[anchor] = [value for value in values if isinstance(value, str) and value]
    return out


def _collect_candidate_types(
    anchor_map: dict[str, list[str]],
    supports: list[_Support],
    openings: dict[str, list[dict[str, Any]]],
) -> list[str]:
    active_anchors = {support.object_type for support in supports}
    if openings.get("doors") or openings.get("windows"):
        active_anchors.add("__opening__")
    if "__wall__" in anchor_map:
        active_anchors.add("__wall__")
    if "__ceiling__" in anchor_map:
        active_anchors.add("__ceiling__")
    if "__utility_zone__" in anchor_map:
        active_anchors.add("__utility_zone__")

    out: list[str] = []
    seen: set[str] = set()
    for anchor in active_anchors:
        for value in anchor_map.get(anchor, []):
            if value in seen:
                continue
            seen.add(value)
            out.append(value)
    return out


def _load_inventory_profiles(
    *,
    tenant_id: str | None,
    candidate_types: list[str],
) -> dict[str, _Profile]:
    if not candidate_types:
        return {}
    try:
        payload = ListInventoryByTypes(
            tenant_id=tenant_id, types=candidate_types, limit=200
        )
    except Exception:
        return {}

    items = payload.get("items")
    if not isinstance(items, list):
        return {}

    profiles: dict[str, _Profile] = {}
    for row in items:
        if not isinstance(row, dict):
            continue
        profile = _profile_from_inventory_row(row)
        if profile is None:
            continue
        for key in (
            row.get("type"),
            row.get("asset_type"),
            row.get("name"),
        ):
            if not isinstance(key, str):
                continue
            normalized = key.strip().lower()
            if not normalized or normalized in profiles:
                continue
            profiles[normalized] = profile
    return profiles


def _profile_from_inventory_row(row: dict[str, Any]) -> _Profile | None:
    dims = row.get("dimensions")
    if isinstance(dims, dict):
        length = _to_positive_int(dims.get("length_mm"))
        width = _to_positive_int(dims.get("width_mm"))
        height = _to_non_negative_int(dims.get("height_mm"))
        if length and width:
            return _Profile(length_mm=length, width_mm=width, height_mm=height)
    length = _to_positive_int(row.get("length_mm"))
    width = _to_positive_int(row.get("width_mm"))
    height = _to_non_negative_int(row.get("height_mm"))
    if length and width:
        return _Profile(length_mm=length, width_mm=width, height_mm=height)
    return None


def _build_existing_objects(
    raw_objects: Any,
) -> tuple[list[dict[str, Any]], list[_Support]]:
    if not isinstance(raw_objects, list):
        return [], []

    styled_objects: list[dict[str, Any]] = []
    supports: list[_Support] = []
    for row in raw_objects:
        if not isinstance(row, dict):
            continue
        instance_id = row.get("instance_id")
        object_type = row.get("object_type")
        bbox = _rect_from_bbox(row.get("bbox"))
        if not isinstance(instance_id, str) or not instance_id:
            continue
        if not isinstance(object_type, str) or not object_type:
            continue
        if bbox is None:
            bbox = _rect_from_polygon(_normalize_polygon(row.get("polygon_ccw")))
        if bbox is None:
            continue

        polygon = _normalize_polygon(row.get("polygon_ccw")) or bbox.to_polygon()
        cluster_id = row.get("cluster_id")
        cluster_id = cluster_id if isinstance(cluster_id, str) and cluster_id else None
        place_on = (
            deepcopy(row.get("place_on"))
            if isinstance(row.get("place_on"), dict)
            else None
        )
        collision_layer = row.get("collision_layer")
        if not isinstance(collision_layer, str) or not collision_layer:
            collision_layer = _existing_object_collision_layer(
                object_type=object_type,
                place_on=place_on,
            )

        styled_objects.append(
            {
                "instance_id": instance_id,
                "object_type": object_type,
                "source": "existing",
                "cluster_id": cluster_id,
                "polygon_ccw": polygon,
                "bbox": bbox.to_bbox(),
                "color_hex": DEFAULT_OBJECT_COLOR,
                "material": _fallback_material(object_type),
                "place_on": place_on,
                "collision_layer": collision_layer,
            }
        )
        supports.append(
            _Support(
                instance_id=instance_id,
                object_type=object_type,
                cluster_id=cluster_id,
                rect=bbox,
            )
        )
    return styled_objects, supports


def _existing_object_collision_layer(
    *,
    object_type: str,
    place_on: dict[str, Any] | None,
) -> str:
    if isinstance(place_on, dict):
        method = str(place_on.get("method") or "floor").strip().lower()
        target_instance_id = str(place_on.get("target_instance_id") or "")
        if method == "on_top":
            return _collision_layer_for_object(
                item_type=object_type,
                method="on_top",
                target_instance_id=target_instance_id,
            )
        if method in {"hang_on", "wall", "ceiling"}:
            return _collision_layer_for_object(
                item_type=object_type,
                method="hang_on",
                target_instance_id=target_instance_id or "ceiling",
            )
    if object_type.strip().lower() in UNDERLAY_TYPES:
        return "floor_underlay"
    return "floor_solid"


def _build_generated_objects(
    *,
    room_type: str,
    room_rect: _Rect,
    supports: list[_Support],
    openings: dict[str, list[dict[str, Any]]],
    anchor_map: dict[str, list[str]],
    inventory_profiles: dict[str, _Profile],
    existing_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    existing_ids = {
        row["instance_id"]
        for row in existing_rows
        if isinstance(row, dict) and isinstance(row.get("instance_id"), str)
    }
    counters: dict[str, int] = {}
    generated: list[dict[str, Any]] = []
    existing_families = _object_families(existing_rows)

    room_center = (room_rect.center_x, room_rect.center_y)

    for support in supports:
        candidates = anchor_map.get(support.object_type)
        if not candidates:
            continue
        selected = _select_anchor_candidates(
            anchor=support.object_type,
            room_type=room_type,
            candidates=candidates,
        )
        selected = _filter_already_present_item_families(
            selected,
            existing_families=existing_families,
        )
        generated.extend(
            _place_support_attached_items(
                support=support,
                room_rect=room_rect,
                room_center=room_center,
                item_types=selected,
                inventory_profiles=inventory_profiles,
                existing_ids=existing_ids,
                counters=counters,
            )
        )

    opening_candidates = _select_anchor_candidates(
        anchor="__opening__",
        room_type=room_type,
        candidates=anchor_map.get("__opening__", []),
    )
    opening_candidates = _filter_already_present_item_families(
        opening_candidates,
        existing_families=existing_families,
    )
    if opening_candidates:
        generated.extend(
            _place_opening_items(
                room_rect=room_rect,
                openings=openings,
                item_types=opening_candidates,
                inventory_profiles=inventory_profiles,
                existing_ids=existing_ids,
                counters=counters,
            )
        )

    wall_candidates = _select_anchor_candidates(
        anchor="__wall__",
        room_type=room_type,
        candidates=anchor_map.get("__wall__", []),
    )
    wall_candidates = _filter_already_present_item_families(
        wall_candidates,
        existing_families=existing_families,
    )
    if wall_candidates:
        generated.extend(
            _place_wall_items(
                room_rect=room_rect,
                supports=supports,
                item_types=wall_candidates,
                inventory_profiles=inventory_profiles,
                existing_ids=existing_ids,
                counters=counters,
            )
        )

    ceiling_candidates = _select_anchor_candidates(
        anchor="__ceiling__",
        room_type=room_type,
        candidates=anchor_map.get("__ceiling__", []),
    )
    ceiling_candidates = _filter_already_present_item_families(
        ceiling_candidates,
        existing_families=existing_families,
    )
    if ceiling_candidates:
        generated.extend(
            _place_ceiling_items(
                room_rect=room_rect,
                supports=supports,
                item_types=ceiling_candidates,
                inventory_profiles=inventory_profiles,
                existing_ids=existing_ids,
                counters=counters,
            )
        )

    utility_candidates = _select_anchor_candidates(
        anchor="__utility_zone__",
        room_type=room_type,
        candidates=anchor_map.get("__utility_zone__", []),
    )
    utility_candidates = _filter_already_present_item_families(
        utility_candidates,
        existing_families=existing_families,
    )
    if utility_candidates:
        generated.extend(
            _place_utility_items(
                room_rect=room_rect,
                item_types=utility_candidates,
                inventory_profiles=inventory_profiles,
                existing_ids=existing_ids,
                counters=counters,
                occupied=[
                    _rect_from_bbox(row.get("bbox"))
                    for row in existing_rows + generated
                    if isinstance(row, dict)
                ],
            )
        )

    return generated


def _object_families(rows: list[dict[str, Any]]) -> set[str]:
    return {
        _object_family(str(row.get("object_type") or ""))
        for row in rows
        if isinstance(row, dict) and str(row.get("object_type") or "").strip()
    }


def _filter_already_present_item_families(
    item_types: list[str],
    *,
    existing_families: set[str],
) -> list[str]:
    out: list[str] = []
    for item_type in item_types:
        family = _object_family(item_type)
        if family in existing_families:
            continue
        existing_families.add(family)
        out.append(item_type)
    return out


def _object_family(object_type: str) -> str:
    normalized = object_type.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"ceiling_lamp", "ceiling_light", "overhead_light"}:
        return "ceiling_lamp"
    if normalized in {"rug", "carpet"}:
        return "rug"
    if normalized in {"stove", "cooktop", "hob", "range"}:
        return "cooking_appliance"
    if normalized in {"range_hood", "hood", "extractor_hood", "vent_hood"}:
        return "range_hood"
    if normalized in {"kitchen_wall_cabinet", "wall_cabinet", "upper_cabinet"}:
        return "kitchen_wall_cabinet"
    return normalized


def _select_anchor_candidates(
    *,
    anchor: str,
    room_type: str,
    candidates: list[str],
) -> list[str]:
    if not candidates:
        return []
    ordered = _ordered_candidates(
        anchor=anchor, room_type=room_type, candidates=candidates
    )
    limit = MAX_ITEMS_BY_ANCHOR.get(anchor, 1)
    selected: list[str] = []
    for item_type in ordered:
        if item_type in STRUCTURAL_SKIP_TYPES:
            continue
        if _is_disallowed_anchor_attachment(anchor=anchor, item_type=item_type):
            continue
        if not _is_supported_item_type(item_type):
            continue
        selected.append(item_type)
        if len(selected) >= limit:
            break
    return selected


def _is_disallowed_anchor_attachment(*, anchor: str, item_type: str) -> bool:
    normalized_anchor = anchor.strip().lower()
    normalized_item_type = item_type.strip().lower()
    if normalized_item_type != "tv":
        return False
    return normalized_anchor == "tv_console"


def _ordered_candidates(
    *,
    anchor: str,
    room_type: str,
    candidates: list[str],
) -> list[str]:
    priority = ANCHOR_PRIORITY.get(anchor, ())
    if anchor == "__ceiling__":
        priority = CEILING_PRIORITY_BY_ROOM.get(room_type, priority)
    if anchor == "__opening__":
        priority = OPENING_PRIORITY_BY_ROOM.get(room_type, priority)

    seen: set[str] = set()
    ordered: list[str] = []
    for item_type in priority:
        if item_type in candidates and item_type not in seen:
            seen.add(item_type)
            ordered.append(item_type)
    for item_type in candidates:
        if item_type not in seen:
            seen.add(item_type)
            ordered.append(item_type)
    return ordered


def _place_support_attached_items(
    *,
    support: _Support,
    room_rect: _Rect,
    room_center: tuple[float, float],
    item_types: list[str],
    inventory_profiles: dict[str, _Profile],
    existing_ids: set[str],
    counters: dict[str, int],
) -> list[dict[str, Any]]:
    generated: list[dict[str, Any]] = []
    on_top_types: list[str] = []

    for item_type in item_types:
        if item_type == "throw_blanket":
            row = _place_throw_blanket_item(
                support=support,
                room_center=room_center,
                item_type=item_type,
                inventory_profiles=inventory_profiles,
                existing_ids=existing_ids,
                counters=counters,
            )
            if row is not None:
                generated.append(row)
                continue
        method = _infer_method(item_type)
        if method == "floor" and item_type in UNDERLAY_TYPES:
            row = _place_underlay_item(
                support=support,
                item_type=item_type,
                inventory_profiles=inventory_profiles,
                existing_ids=existing_ids,
                counters=counters,
            )
            if row is not None:
                generated.append(row)
            continue
        if method == "hang_on":
            row = _place_wall_item_over_support(
                support=support,
                room_rect=room_rect,
                item_type=item_type,
                inventory_profiles=inventory_profiles,
                existing_ids=existing_ids,
                counters=counters,
            )
            if row is not None:
                generated.append(row)
            continue
        if method == "floor":
            row = _place_adjacent_floor_item(
                support=support,
                room_center=room_center,
                item_type=item_type,
                inventory_profiles=inventory_profiles,
                existing_ids=existing_ids,
                counters=counters,
            )
            if row is not None:
                generated.append(row)
            continue
        on_top_types.append(item_type)

    generated.extend(
        _pack_on_support(
            support=support,
            room_center=room_center,
            item_types=on_top_types,
            inventory_profiles=inventory_profiles,
            existing_ids=existing_ids,
            counters=counters,
        )
    )
    return generated


def _pack_on_support(
    *,
    support: _Support,
    room_center: tuple[float, float],
    item_types: list[str],
    inventory_profiles: dict[str, _Profile],
    existing_ids: set[str],
    counters: dict[str, int],
) -> list[dict[str, Any]]:
    if not item_types:
        return []

    support_rect = support.rect
    run_axis: Literal["x", "y"] = (
        "x" if support_rect.width >= support_rect.height else "y"
    )
    room_cx, room_cy = room_center
    run_dir = _outward_direction(
        support_center=support_rect.center_x
        if run_axis == "x"
        else support_rect.center_y,
        room_center=room_cx if run_axis == "x" else room_cy,
    )
    cross_axis: Literal["x", "y"] = "y" if run_axis == "x" else "x"
    cross_dir = _outward_direction(
        support_center=support_rect.center_y
        if cross_axis == "y"
        else support_rect.center_x,
        room_center=room_cy if cross_axis == "y" else room_cx,
    )

    inner_min_x = support_rect.min_x + SUPPORT_INSET_MM
    inner_max_x = support_rect.max_x - SUPPORT_INSET_MM
    inner_min_y = support_rect.min_y + SUPPORT_INSET_MM
    inner_max_y = support_rect.max_y - SUPPORT_INSET_MM
    if inner_max_x <= inner_min_x or inner_max_y <= inner_min_y:
        return []

    cursor = (
        inner_min_x
        if run_axis == "x" and run_dir > 0
        else inner_max_x
        if run_axis == "x"
        else inner_min_y
        if run_dir > 0
        else inner_max_y
    )
    rows: list[dict[str, Any]] = []
    for item_type in item_types:
        profile = _profile_for_type(item_type, inventory_profiles)
        run_size, cross_size = _oriented_on_top_dims(profile, run_axis=run_axis)
        cross_available = (
            (inner_max_y - inner_min_y)
            if cross_axis == "y"
            else (inner_max_x - inner_min_x)
        )
        if cross_size > cross_available:
            continue

        if run_axis == "x":
            if run_dir > 0:
                min_x = cursor
                max_x = cursor + run_size
            else:
                min_x = cursor - run_size
                max_x = cursor
            if min_x < inner_min_x or max_x > inner_max_x:
                continue
            if cross_dir > 0:
                min_y = inner_min_y
                max_y = min_y + cross_size
            else:
                max_y = inner_max_y
                min_y = max_y - cross_size
        else:
            if run_dir > 0:
                min_y = cursor
                max_y = cursor + run_size
            else:
                min_y = cursor - run_size
                max_y = cursor
            if min_y < inner_min_y or max_y > inner_max_y:
                continue
            if cross_dir > 0:
                min_x = inner_min_x
                max_x = min_x + cross_size
            else:
                max_x = inner_max_x
                min_x = max_x - cross_size

        rect = _Rect(
            min_x=int(round(min_x)),
            min_y=int(round(min_y)),
            max_x=int(round(max_x)),
            max_y=int(round(max_y)),
        )
        row = _make_generated_object(
            item_type=item_type,
            rect=rect,
            method="on_top",
            target_instance_id=support.instance_id,
            cluster_id=support.cluster_id,
            inventory_profiles=inventory_profiles,
            existing_ids=existing_ids,
            counters=counters,
        )
        rows.append(row)

        cursor = (
            rect.max_x + STACK_GAP_MM
            if run_axis == "x" and run_dir > 0
            else rect.min_x - STACK_GAP_MM
            if run_axis == "x"
            else rect.max_y + STACK_GAP_MM
            if run_dir > 0
            else rect.min_y - STACK_GAP_MM
        )

    return rows


def _place_throw_blanket_item(
    *,
    support: _Support,
    room_center: tuple[float, float],
    item_type: str,
    inventory_profiles: dict[str, _Profile],
    existing_ids: set[str],
    counters: dict[str, int],
) -> dict[str, Any] | None:
    if support.object_type not in {"bed", "sofa", "sectional_sofa"}:
        return None

    support_rect = support.rect
    inner_min_x = support_rect.min_x + SUPPORT_INSET_MM
    inner_max_x = support_rect.max_x - SUPPORT_INSET_MM
    inner_min_y = support_rect.min_y + SUPPORT_INSET_MM
    inner_max_y = support_rect.max_y - SUPPORT_INSET_MM
    if inner_max_x <= inner_min_x or inner_max_y <= inner_min_y:
        return None

    profile = _profile_for_type(item_type, inventory_profiles)
    available_width = inner_max_x - inner_min_x
    available_height = inner_max_y - inner_min_y
    horizontal = support_rect.width >= support_rect.height

    if horizontal:
        blanket_width = _clamp(
            int(round(support_rect.width * 0.68)),
            min(available_width, max(profile.width_mm, 450)),
            available_width,
        )
        blanket_height = _clamp(
            int(round(support_rect.height * 0.34)),
            min(available_height, max(profile.height_mm * 6, 260)),
            available_height,
        )
        long_shift_limit = max(0, (available_width - blanket_width) // 2)
        shift_direction = _outward_direction(
            support_center=support_rect.center_x,
            room_center=room_center[0],
        )
        center_x = support_rect.center_x + shift_direction * min(
            long_shift_limit,
            int(round(support_rect.width * 0.12)),
        )
        center_y = support_rect.center_y
    else:
        blanket_width = _clamp(
            int(round(support_rect.width * 0.34)),
            min(available_width, max(profile.height_mm * 6, 260)),
            available_width,
        )
        blanket_height = _clamp(
            int(round(support_rect.height * 0.68)),
            min(available_height, max(profile.width_mm, 450)),
            available_height,
        )
        long_shift_limit = max(0, (available_height - blanket_height) // 2)
        shift_direction = _outward_direction(
            support_center=support_rect.center_y,
            room_center=room_center[1],
        )
        center_x = support_rect.center_x
        center_y = support_rect.center_y + shift_direction * min(
            long_shift_limit,
            int(round(support_rect.height * 0.12)),
        )

    rect = _centered_rect(
        center_x=center_x,
        center_y=center_y,
        width=blanket_width,
        height=blanket_height,
    )
    rect = _fit_rect_within_bounds(
        rect=rect,
        min_x=inner_min_x,
        max_x=inner_max_x,
        min_y=inner_min_y,
        max_y=inner_max_y,
    )
    return _make_generated_object(
        item_type=item_type,
        rect=rect,
        method="on_top",
        target_instance_id=support.instance_id,
        cluster_id=support.cluster_id,
        inventory_profiles=inventory_profiles,
        existing_ids=existing_ids,
        counters=counters,
    )


def _place_underlay_item(
    *,
    support: _Support,
    item_type: str,
    inventory_profiles: dict[str, _Profile],
    existing_ids: set[str],
    counters: dict[str, int],
) -> dict[str, Any] | None:
    profile = _profile_for_type(item_type, inventory_profiles)
    base_w, base_h = profile.footprint()
    width = max(base_w, support.rect.width + UNDERLAY_MARGIN_MM * 2)
    height = max(base_h, support.rect.height + UNDERLAY_MARGIN_MM * 2)
    rect = _centered_rect(
        center_x=support.rect.center_x,
        center_y=support.rect.center_y,
        width=width,
        height=height,
    )
    return _make_generated_object(
        item_type=item_type,
        rect=rect,
        method="floor",
        target_instance_id=support.instance_id,
        cluster_id=support.cluster_id,
        inventory_profiles=inventory_profiles,
        existing_ids=existing_ids,
        counters=counters,
    )


def _place_adjacent_floor_item(
    *,
    support: _Support,
    room_center: tuple[float, float],
    item_type: str,
    inventory_profiles: dict[str, _Profile],
    existing_ids: set[str],
    counters: dict[str, int],
) -> dict[str, Any] | None:
    profile = _profile_for_type(item_type, inventory_profiles)
    width, height = profile.footprint()
    horizontal = support.rect.width >= support.rect.height
    room_cx, room_cy = room_center
    if horizontal:
        side = _outward_direction(support.rect.center_y, room_cy)
        if side > 0:
            min_y = support.rect.min_y - height - STACK_GAP_MM
            max_y = min_y + height
        else:
            max_y = support.rect.max_y + height + STACK_GAP_MM
            min_y = max_y - height
        center_x = support.rect.center_x
        min_x = int(round(center_x - width / 2.0))
        max_x = min_x + width
    else:
        side = _outward_direction(support.rect.center_x, room_cx)
        if side > 0:
            min_x = support.rect.min_x - width - STACK_GAP_MM
            max_x = min_x + width
        else:
            max_x = support.rect.max_x + width + STACK_GAP_MM
            min_x = max_x - width
        center_y = support.rect.center_y
        min_y = int(round(center_y - height / 2.0))
        max_y = min_y + height

    rect = _Rect(
        min_x=int(round(min_x)),
        min_y=int(round(min_y)),
        max_x=int(round(max_x)),
        max_y=int(round(max_y)),
    )
    return _make_generated_object(
        item_type=item_type,
        rect=rect,
        method="floor",
        target_instance_id=support.instance_id,
        cluster_id=support.cluster_id,
        inventory_profiles=inventory_profiles,
        existing_ids=existing_ids,
        counters=counters,
    )


def _place_wall_item_over_support(
    *,
    support: _Support,
    room_rect: _Rect,
    item_type: str,
    inventory_profiles: dict[str, _Profile],
    existing_ids: set[str],
    counters: dict[str, int],
) -> dict[str, Any] | None:
    profile = _profile_for_type(item_type, inventory_profiles)
    width, plan_height = profile.wall_projection()
    wall = _nearest_room_edge(room_rect=room_rect, rect=support.rect)

    if wall in {"top", "bottom"}:
        center_x = _clamp(
            int(round(support.rect.center_x)),
            room_rect.min_x + width // 2 + MOUNT_INSET_MM,
            room_rect.max_x - width // 2 - MOUNT_INSET_MM,
        )
        min_x = center_x - width // 2
        max_x = min_x + width
        if wall == "top":
            min_y = room_rect.min_y + MOUNT_INSET_MM
            max_y = min_y + plan_height
        else:
            max_y = room_rect.max_y - MOUNT_INSET_MM
            min_y = max_y - plan_height
    else:
        center_y = _clamp(
            int(round(support.rect.center_y)),
            room_rect.min_y + plan_height // 2 + MOUNT_INSET_MM,
            room_rect.max_y - plan_height // 2 - MOUNT_INSET_MM,
        )
        min_y = center_y - plan_height // 2
        max_y = min_y + plan_height
        if wall == "left":
            min_x = room_rect.min_x + MOUNT_INSET_MM
            max_x = min_x + width
        else:
            max_x = room_rect.max_x - MOUNT_INSET_MM
            min_x = max_x - width

    rect = _Rect(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y)
    return _make_generated_object(
        item_type=item_type,
        rect=rect,
        method="hang_on",
        target_instance_id="wall",
        cluster_id=support.cluster_id,
        inventory_profiles=inventory_profiles,
        existing_ids=existing_ids,
        counters=counters,
    )


def _place_opening_items(
    *,
    room_rect: _Rect,
    openings: dict[str, list[dict[str, Any]]],
    item_types: list[str],
    inventory_profiles: dict[str, _Profile],
    existing_ids: set[str],
    counters: dict[str, int],
) -> list[dict[str, Any]]:
    if not item_types:
        return []
    target_openings = openings.get("windows") or openings.get("doors") or []
    if not target_openings:
        return []

    item_type = item_types[0]
    profile = _profile_for_type(item_type, inventory_profiles)
    _, plan_height = profile.wall_projection()
    rows: list[dict[str, Any]] = []
    for opening in target_openings:
        segment = opening.get("segment_mm") if isinstance(opening, dict) else None
        opening_id = opening.get("id") if isinstance(opening, dict) else None
        if not isinstance(segment, list) or len(segment) != 2:
            continue
        if not isinstance(opening_id, str) or not opening_id:
            continue
        p1 = _point_tuple(segment[0])
        p2 = _point_tuple(segment[1])
        if p1 is None or p2 is None:
            continue
        seg_length = max(1, int(round(_segment_length(p1, p2))))
        width = max(profile.wall_projection()[0], seg_length + 200)
        center_x = int(round((p1[0] + p2[0]) / 2.0))
        center_y = int(round((p1[1] + p2[1]) / 2.0))
        rect = _centered_rect(
            center_x=center_x,
            center_y=center_y,
            width=width,
            height=plan_height,
        )
        rows.append(
            _make_generated_object(
                item_type=item_type,
                rect=rect,
                method="hang_on",
                target_instance_id=opening_id,
                cluster_id=None,
                inventory_profiles=inventory_profiles,
                existing_ids=existing_ids,
                counters=counters,
            )
        )
    return rows


def _place_wall_items(
    *,
    room_rect: _Rect,
    supports: list[_Support],
    item_types: list[str],
    inventory_profiles: dict[str, _Profile],
    existing_ids: set[str],
    counters: dict[str, int],
) -> list[dict[str, Any]]:
    if not item_types:
        return []
    item_type = item_types[0]
    profile = _profile_for_type(item_type, inventory_profiles)
    width, plan_height = profile.wall_projection()
    focus = max(supports, key=lambda row: row.rect.area, default=None)
    center_x = int(round(focus.rect.center_x if focus else room_rect.center_x))
    wall = "top"
    if focus is not None:
        wall = _nearest_room_edge(room_rect=room_rect, rect=focus.rect)

    if wall in {"top", "bottom"}:
        center_x = _clamp(
            center_x,
            room_rect.min_x + width // 2 + MOUNT_INSET_MM,
            room_rect.max_x - width // 2 - MOUNT_INSET_MM,
        )
        min_x = center_x - width // 2
        max_x = min_x + width
        if wall == "top":
            min_y = room_rect.min_y + MOUNT_INSET_MM
            max_y = min_y + plan_height
        else:
            max_y = room_rect.max_y - MOUNT_INSET_MM
            min_y = max_y - plan_height
    else:
        center_y = int(round(focus.rect.center_y if focus else room_rect.center_y))
        center_y = _clamp(
            center_y,
            room_rect.min_y + plan_height // 2 + MOUNT_INSET_MM,
            room_rect.max_y - plan_height // 2 - MOUNT_INSET_MM,
        )
        min_y = center_y - plan_height // 2
        max_y = min_y + plan_height
        if wall == "left":
            min_x = room_rect.min_x + MOUNT_INSET_MM
            max_x = min_x + width
        else:
            max_x = room_rect.max_x - MOUNT_INSET_MM
            min_x = max_x - width

    return [
        _make_generated_object(
            item_type=item_type,
            rect=_Rect(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y),
            method="hang_on",
            target_instance_id="wall",
            cluster_id=focus.cluster_id if focus else None,
            inventory_profiles=inventory_profiles,
            existing_ids=existing_ids,
            counters=counters,
        )
    ]


def _place_ceiling_items(
    *,
    room_rect: _Rect,
    supports: list[_Support],
    item_types: list[str],
    inventory_profiles: dict[str, _Profile],
    existing_ids: set[str],
    counters: dict[str, int],
) -> list[dict[str, Any]]:
    if not item_types:
        return []
    item_type = item_types[0]
    profile = _profile_for_type(item_type, inventory_profiles)
    width, height = profile.footprint()
    anchor = _preferred_ceiling_anchor(supports)
    center_x = int(round(anchor.rect.center_x if anchor else room_rect.center_x))
    center_y = int(round(anchor.rect.center_y if anchor else room_rect.center_y))
    rect = _centered_rect(
        center_x=center_x, center_y=center_y, width=width, height=height
    )
    rect = _fit_rect_within_bounds(
        rect=rect,
        min_x=room_rect.min_x + MOUNT_INSET_MM,
        max_x=room_rect.max_x - MOUNT_INSET_MM,
        min_y=room_rect.min_y + MOUNT_INSET_MM,
        max_y=room_rect.max_y - MOUNT_INSET_MM,
    )
    return [
        _make_generated_object(
            item_type=item_type,
            rect=rect,
            method="hang_on",
            target_instance_id="ceiling",
            cluster_id=anchor.cluster_id if anchor else None,
            inventory_profiles=inventory_profiles,
            existing_ids=existing_ids,
            counters=counters,
        )
    ]


def _place_utility_items(
    *,
    room_rect: _Rect,
    item_types: list[str],
    inventory_profiles: dict[str, _Profile],
    existing_ids: set[str],
    counters: dict[str, int],
    occupied: list[_Rect | None],
) -> list[dict[str, Any]]:
    if not item_types:
        return []
    item_type = item_types[0]
    profile = _profile_for_type(item_type, inventory_profiles)
    width, height = profile.footprint()
    candidates = [
        _Rect(
            room_rect.max_x - width - UTILITY_GAP_MM,
            room_rect.max_y - height - UTILITY_GAP_MM,
            room_rect.max_x - UTILITY_GAP_MM,
            room_rect.max_y - UTILITY_GAP_MM,
        ),
        _Rect(
            room_rect.min_x + UTILITY_GAP_MM,
            room_rect.max_y - height - UTILITY_GAP_MM,
            room_rect.min_x + width + UTILITY_GAP_MM,
            room_rect.max_y - UTILITY_GAP_MM,
        ),
    ]
    chosen = None
    occupied_rects = [rect for rect in occupied if isinstance(rect, _Rect)]
    for candidate in candidates:
        if any(_rects_overlap(candidate, rect) for rect in occupied_rects):
            continue
        chosen = candidate
        break
    if chosen is None:
        chosen = candidates[0]
    return [
        _make_generated_object(
            item_type=item_type,
            rect=chosen,
            method="floor",
            target_instance_id="floor",
            cluster_id=None,
            inventory_profiles=inventory_profiles,
            existing_ids=existing_ids,
            counters=counters,
        )
    ]


def _make_generated_object(
    *,
    item_type: str,
    rect: _Rect,
    method: Literal["on_top", "hang_on", "floor"],
    target_instance_id: str,
    cluster_id: str | None,
    inventory_profiles: dict[str, _Profile],
    existing_ids: set[str],
    counters: dict[str, int],
) -> dict[str, Any]:
    instance_id = _make_instance_id(
        item_type, existing_ids=existing_ids, counters=counters
    )
    profile = _profile_for_type(item_type, inventory_profiles)
    return {
        "instance_id": instance_id,
        "object_type": item_type,
        "source": "inventory",
        "cluster_id": cluster_id,
        "polygon_ccw": rect.to_polygon(),
        "bbox": rect.to_bbox(),
        "color_hex": DEFAULT_OBJECT_COLOR,
        "material": _fallback_material(item_type),
        "place_on": {
            "target_instance_id": target_instance_id,
            "method": method,
        },
        "collision_layer": _collision_layer_for_object(
            item_type=item_type,
            method=method,
            target_instance_id=target_instance_id,
        ),
        "meta_height_mm": profile.height_mm,
    }


def _collision_layer_for_object(
    *,
    item_type: str,
    method: Literal["on_top", "hang_on", "floor"],
    target_instance_id: str,
) -> str:
    lowered = item_type.strip().lower()
    if method == "on_top":
        return "surface_child"
    if method == "hang_on":
        if target_instance_id == "ceiling" or lowered in CEILING_MOUNTED_TYPES:
            return "ceiling"
        return "wall_mounted"
    if lowered in UNDERLAY_TYPES:
        return "floor_underlay"
    return "floor_solid"


def _validate_styled_objects(
    *,
    room_rect: _Rect,
    objects: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, str]] = []
    floor_solids: list[tuple[str, _Rect, str]] = []

    for row in objects:
        if not isinstance(row, dict):
            continue
        rect = _rect_from_bbox(row.get("bbox"))
        if rect is None:
            kept.append(row)
            continue
        layer = str(row.get("collision_layer") or "floor_solid")
        object_type = str(row.get("object_type") or "")
        instance_id = str(row.get("instance_id") or object_type)
        source = str(row.get("source") or "")

        if layer == "floor_underlay":
            fitted = _fit_rect_within_bounds(
                rect=rect,
                min_x=room_rect.min_x,
                max_x=room_rect.max_x,
                min_y=room_rect.min_y,
                max_y=room_rect.max_y,
            )
            next_row = dict(row)
            next_row["bbox"] = fitted.to_bbox()
            next_row["polygon_ccw"] = fitted.to_polygon()
            kept.append(next_row)
            continue

        if layer == "floor_solid":
            if any(
                _rects_overlap_local(rect, other_rect)
                for _, other_rect, _ in floor_solids
            ):
                if source == "inventory":
                    dropped.append(
                        {
                            "instance_id": instance_id,
                            "reason": "floor_solid_overlap",
                        }
                    )
                    continue
            floor_solids.append((instance_id, rect, object_type))

        if layer == "wall_mounted" and object_type in OPENING_MOUNTED_TYPES:
            if any(
                _rect_overlap_ratio_local(rect, other_rect) > 0.15
                for _, other_rect, _ in floor_solids
            ):
                dropped.append(
                    {
                        "instance_id": instance_id,
                        "reason": "opening_mount_over_floor_solid",
                    }
                )
                continue

        kept.append(row)

    return kept, {
        "collision_model": "layer_aware_v1",
        "dropped_objects": dropped,
    }


def _rects_overlap_local(left: _Rect, right: _Rect) -> bool:
    return (
        left.min_x < right.max_x
        and left.max_x > right.min_x
        and left.min_y < right.max_y
        and left.max_y > right.min_y
    )


def _rect_overlap_ratio_local(left: _Rect, right: _Rect) -> float:
    overlap_x = max(0, min(left.max_x, right.max_x) - max(left.min_x, right.min_x))
    overlap_y = max(0, min(left.max_y, right.max_y) - max(left.min_y, right.min_y))
    overlap_area = overlap_x * overlap_y
    if overlap_area <= 0:
        return 0.0
    return overlap_area / max(1, min(left.area, right.area))


def _profile_for_type(
    item_type: str,
    inventory_profiles: dict[str, _Profile],
) -> _Profile:
    lowered = item_type.strip().lower()
    if lowered in inventory_profiles:
        return inventory_profiles[lowered]
    if lowered.endswith("_lamp") and "lamp" in inventory_profiles:
        return inventory_profiles["lamp"]
    dims = DEFAULT_DIMENSIONS_MM.get(lowered)
    if dims is None:
        return _Profile(length_mm=300, width_mm=300, height_mm=300)
    return _Profile(length_mm=dims[0], width_mm=dims[1], height_mm=dims[2])


def _make_instance_id(
    item_type: str,
    *,
    existing_ids: set[str],
    counters: dict[str, int],
) -> str:
    base = item_type.strip().lower()
    counters[base] = counters.get(base, 0) + 1
    candidate = f"{base}_{counters[base]:03d}"
    while candidate in existing_ids:
        counters[base] += 1
        candidate = f"{base}_{counters[base]:03d}"
    existing_ids.add(candidate)
    return candidate


def _preferred_ceiling_anchor(supports: list[_Support]) -> _Support | None:
    preferred_types = {
        "dining_table",
        "coffee_table",
        "bed",
        "desk",
        "sofa",
        "sectional_sofa",
        "kitchen_island",
    }
    preferred = [row for row in supports if row.object_type in preferred_types]
    if preferred:
        return max(preferred, key=lambda row: row.rect.area)
    return max(supports, key=lambda row: row.rect.area, default=None)


def _infer_method(item_type: str) -> Literal["on_top", "hang_on", "floor"]:
    if item_type in UNDERLAY_TYPES or item_type in FLOOR_SIDE_TYPES:
        return "floor"
    if (
        item_type in OPENING_MOUNTED_TYPES
        or item_type in CEILING_MOUNTED_TYPES
        or item_type in WALL_MOUNTED_TYPES
    ):
        return "hang_on"
    return "on_top"


def _is_supported_item_type(item_type: str) -> bool:
    return item_type not in STRUCTURAL_SKIP_TYPES


def _fallback_material(object_type: str) -> str:
    lowered = object_type.lower()
    if any(
        token in lowered
        for token in (
            "rug",
            "curtain",
            "blanket",
            "cushion",
            "bed",
            "sofa",
            "chair",
            "armchair",
            "bean_bag",
        )
    ):
        return "fabric"
    if any(
        token in lowered
        for token in (
            "lamp",
            "clock",
            "speaker",
            "air_conditioner",
            "monitor",
            "laptop",
            "desktop",
            "printer",
            "appliance",
            "microwave",
            "oven",
            "toaster",
            "blender",
            "hood",
            "cooktop",
        )
    ):
        return "metal"
    if any(token in lowered for token in ("mirror", "window", "glass")):
        return "glass"
    if any(token in lowered for token in ("vase", "plant", "decor")):
        return "ceramic"
    return "wood"


def _normalize_openings(value: Any) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(value, dict):
        return {"doors": [], "windows": []}
    out: dict[str, list[dict[str, Any]]] = {"doors": [], "windows": []}
    for key in ("doors", "windows"):
        rows = value.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            segment = row.get("segment_mm")
            if not isinstance(segment, list) or len(segment) != 2:
                continue
            opening_id = row.get("id") or row.get(f"{key[:-1]}_id")
            if not isinstance(opening_id, str) or not opening_id:
                continue
            p1 = _normalize_point(segment[0])
            p2 = _normalize_point(segment[1])
            if p1 is None or p2 is None:
                continue
            out[key].append(
                {
                    "id": opening_id,
                    "segment_mm": [p1, p2],
                }
            )
    return out


def _normalize_polygon(value: Any) -> list[dict[str, int]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, int]] = []
    for row in value:
        point = _normalize_point(row)
        if point is not None:
            out.append(point)
    return out


def _normalize_point(value: Any) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    x = _to_int(value.get("x"))
    y = _to_int(value.get("y"))
    if x is None or y is None:
        return None
    return {"x": x, "y": y}


def _rect_from_bbox(value: Any) -> _Rect | None:
    if not isinstance(value, dict):
        return None
    min_x = _to_int(value.get("min_x"))
    min_y = _to_int(value.get("min_y"))
    max_x = _to_int(value.get("max_x"))
    max_y = _to_int(value.get("max_y"))
    if None in {min_x, min_y, max_x, max_y}:
        return None
    if max_x <= min_x or max_y <= min_y:
        return None
    return _Rect(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y)


def _rect_from_polygon(value: list[dict[str, int]]) -> _Rect | None:
    if len(value) < 3:
        return None
    xs = [point["x"] for point in value]
    ys = [point["y"] for point in value]
    return _Rect(min_x=min(xs), min_y=min(ys), max_x=max(xs), max_y=max(ys))


def _rect_from_layout_objects(value: Any) -> _Rect | None:
    if not isinstance(value, list):
        return None
    rects = [_rect_from_bbox(row.get("bbox")) for row in value if isinstance(row, dict)]
    rects = [row for row in rects if isinstance(row, _Rect)]
    if not rects:
        return None
    return _Rect(
        min_x=min(row.min_x for row in rects),
        min_y=min(row.min_y for row in rects),
        max_x=max(row.max_x for row in rects),
        max_y=max(row.max_y for row in rects),
    )


def _centered_rect(
    *,
    center_x: float,
    center_y: float,
    width: int,
    height: int,
) -> _Rect:
    min_x = int(round(center_x - width / 2.0))
    min_y = int(round(center_y - height / 2.0))
    return _Rect(
        min_x=min_x,
        min_y=min_y,
        max_x=min_x + width,
        max_y=min_y + height,
    )


def _fit_rect_within_bounds(
    *,
    rect: _Rect,
    min_x: int,
    max_x: int,
    min_y: int,
    max_y: int,
) -> _Rect:
    dx = 0
    dy = 0
    if rect.min_x < min_x:
        dx = min_x - rect.min_x
    elif rect.max_x > max_x:
        dx = max_x - rect.max_x
    if rect.min_y < min_y:
        dy = min_y - rect.min_y
    elif rect.max_y > max_y:
        dy = max_y - rect.max_y
    return _Rect(
        min_x=rect.min_x + dx,
        min_y=rect.min_y + dy,
        max_x=rect.max_x + dx,
        max_y=rect.max_y + dy,
    )


def _nearest_room_edge(
    *, room_rect: _Rect, rect: _Rect
) -> Literal["top", "bottom", "left", "right"]:
    distances = {
        "top": abs(rect.min_y - room_rect.min_y),
        "bottom": abs(room_rect.max_y - rect.max_y),
        "left": abs(rect.min_x - room_rect.min_x),
        "right": abs(room_rect.max_x - rect.max_x),
    }
    return min(distances, key=distances.get)  # type: ignore[return-value]


def _oriented_on_top_dims(
    profile: _Profile,
    *,
    run_axis: Literal["x", "y"],
) -> tuple[int, int]:
    long_side = max(profile.length_mm, profile.width_mm)
    short_side = min(profile.length_mm, profile.width_mm)
    if run_axis == "x":
        return long_side, short_side
    return long_side, short_side


def _outward_direction(*, support_center: float, room_center: float) -> Literal[1, -1]:
    return 1 if support_center <= room_center else -1


def _segment_length(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return (dx * dx + dy * dy) ** 0.5


def _point_tuple(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, dict):
        return None
    x = _to_int(value.get("x"))
    y = _to_int(value.get("y"))
    if x is None or y is None:
        return None
    return x, y


def _rects_overlap(a: _Rect, b: _Rect) -> bool:
    return not (
        a.max_x <= b.min_x
        or a.min_x >= b.max_x
        or a.max_y <= b.min_y
        or a.min_y >= b.max_y
    )


def _clamp(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(max_value, value))


def _to_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(round(float(value)))
    return None


def _to_positive_int(value: Any) -> int:
    number = _to_int(value)
    return number if isinstance(number, int) and number > 0 else 0


def _to_non_negative_int(value: Any) -> int:
    number = _to_int(value)
    return number if isinstance(number, int) and number >= 0 else 0
