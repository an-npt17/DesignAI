from __future__ import annotations

from copy import deepcopy
from typing import Any

from layout.room_profiles.base import RoomProfile
from layout.room_profiles.base import normalize_profile_token

BEDROOM_LEGACY_ALIASES: dict[str, tuple[str, ...]] = {
    "bed": ("bed", "queen bed", "king bed", "single bed", "double bed"),
    "bedside_lamp": ("bedside lamp", "nightstand lamp"),
    "nightstand": ("nightstand", "night stand", "bedside table"),
    "wardrobe": ("wardrobe", "closet", "armoire"),
    "dresser": ("dresser", "chest of drawers"),
    "desk": ("desk", "work desk", "study desk", "vanity", "vanity table"),
    "desk_lamp": ("desk lamp", "task lamp"),
    "chair": ("chair", "desk chair", "office chair", "vanity chair"),
    "bookshelf": ("bookshelf", "book shelf", "bookcase", "shelf"),
    "rug": ("rug", "carpet"),
    "ceiling_lamp": ("ceiling lamp", "ceiling light", "overhead light"),
    "floor_lamp": ("floor lamp", "standing lamp", "reading lamp"),
    "mirror": ("mirror", "vanity mirror"),
    "side_table": ("side table", "end table"),
    "throw_blanket": ("throw blanket", "blanket"),
    "wall_art": ("wall art", "art print", "painting"),
}

LIVING_ROOM_LEGACY_ALIASES: dict[str, tuple[str, ...]] = {
    "armchair": ("armchair", "arm chair", "lounge chair", "reading chair"),
    "bookshelf": ("bookshelf", "book shelf", "bookcase", "shelf"),
    "ceiling_lamp": ("ceiling lamp", "ceiling light", "overhead light"),
    "chair": ("chair", "desk chair", "office chair", "vanity chair"),
    "coffee_table": ("coffee table", "tea table", "center table"),
    "console_table": ("console table", "entry console"),
    "floor_lamp": ("floor lamp", "standing lamp", "reading lamp"),
    "media_shelf": (
        "media shelf",
        "media shelving",
        "av shelf",
        "entertainment shelf",
    ),
    "ottoman": ("ottoman", "pouf", "footstool"),
    "plant": ("plant", "indoor plant"),
    "rug": ("rug", "carpet"),
    "side_table": ("side table", "end table"),
    "sofa": ("sofa", "couch", "loveseat", "sectional"),
    "table_lamp": ("table lamp", "side table lamp"),
    "tv": ("tv", "television"),
    "tv_console": ("tv console", "media console", "tv stand"),
    "wall_art": ("wall art", "art print", "painting"),
}

BEDROOM_ANCHOR_OBJECTS = frozenset({"bed", "desk", "wardrobe"})
BEDROOM_SUPPORT_OBJECTS = frozenset(
    {
        "bedside_lamp",
        "bookshelf",
        "chair",
        "desk_lamp",
        "dresser",
        "floor_lamp",
        "mirror",
        "nightstand",
        "side_table",
    }
)
BEDROOM_SEATING_OBJECTS = frozenset({"chair"})
BEDROOM_SURFACE_OBJECTS = frozenset({"desk", "dresser", "nightstand", "side_table"})
BEDROOM_LIGHTING_OBJECTS = frozenset(
    {"bedside_lamp", "ceiling_lamp", "desk_lamp", "floor_lamp"}
)
BEDROOM_DECOR_OBJECTS = frozenset({"rug", "throw_blanket", "wall_art"})
BEDROOM_STORAGE_OBJECTS = frozenset({"bookshelf", "dresser", "wardrobe"})
BEDROOM_WALL_BACKED_OBJECTS = frozenset(
    {"bed", "bookshelf", "desk", "dresser", "wardrobe"}
)
BEDROOM_FLOATING_OBJECTS = frozenset({"chair", "rug"})
BEDROOM_MOUNTED_OBJECTS = frozenset({"ceiling_lamp", "mirror", "wall_art"})

LIVING_ROOM_ANCHOR_OBJECTS = frozenset({"sofa", "tv_console"})
LIVING_ROOM_SUPPORT_OBJECTS = frozenset(
    {
        "armchair",
        "bookshelf",
        "chair",
        "coffee_table",
        "console_table",
        "floor_lamp",
        "media_shelf",
        "ottoman",
        "plant",
        "side_table",
        "table_lamp",
        "tv",
    }
)
LIVING_ROOM_SEATING_OBJECTS = frozenset({"armchair", "chair", "ottoman", "sofa"})
LIVING_ROOM_SURFACE_OBJECTS = frozenset(
    {"coffee_table", "console_table", "media_shelf", "side_table", "tv_console"}
)
LIVING_ROOM_LIGHTING_OBJECTS = frozenset({"ceiling_lamp", "floor_lamp", "table_lamp"})
LIVING_ROOM_DECOR_OBJECTS = frozenset({"plant", "rug", "wall_art"})
LIVING_ROOM_STORAGE_OBJECTS = frozenset({"bookshelf", "console_table", "media_shelf"})
LIVING_ROOM_WALL_BACKED_OBJECTS = frozenset(
    {"bookshelf", "console_table", "media_shelf", "tv", "tv_console"}
)
LIVING_ROOM_FLOATING_OBJECTS = frozenset(
    {
        "armchair",
        "chair",
        "coffee_table",
        "ottoman",
        "rug",
        "side_table",
        "sofa",
    }
)
LIVING_ROOM_MOUNTED_OBJECTS = frozenset({"ceiling_lamp", "tv", "wall_art"})

_BEDROOM_DECLARED_OBJECTS = (
    frozenset(BEDROOM_LEGACY_ALIASES)
    | BEDROOM_ANCHOR_OBJECTS
    | BEDROOM_SUPPORT_OBJECTS
    | BEDROOM_SEATING_OBJECTS
    | BEDROOM_SURFACE_OBJECTS
    | BEDROOM_LIGHTING_OBJECTS
    | BEDROOM_DECOR_OBJECTS
    | BEDROOM_STORAGE_OBJECTS
    | BEDROOM_WALL_BACKED_OBJECTS
    | BEDROOM_FLOATING_OBJECTS
    | BEDROOM_MOUNTED_OBJECTS
)

_LIVING_ROOM_DECLARED_OBJECTS = (
    frozenset(LIVING_ROOM_LEGACY_ALIASES)
    | LIVING_ROOM_ANCHOR_OBJECTS
    | LIVING_ROOM_SUPPORT_OBJECTS
    | LIVING_ROOM_SEATING_OBJECTS
    | LIVING_ROOM_SURFACE_OBJECTS
    | LIVING_ROOM_LIGHTING_OBJECTS
    | LIVING_ROOM_DECOR_OBJECTS
    | LIVING_ROOM_STORAGE_OBJECTS
    | LIVING_ROOM_WALL_BACKED_OBJECTS
    | LIVING_ROOM_FLOATING_OBJECTS
    | LIVING_ROOM_MOUNTED_OBJECTS
)

_BEDROOM_SIZE_PROFILES: dict[str, dict[str, Any]] = {
    "bed": {
        "rep_dims_m": {
            "S": {"L": 1.90, "W": 0.95, "A": 1.81},
            "M": {"L": 2.00, "W": 1.50, "A": 3.00},
            "L": {"L": 2.10, "W": 1.80, "A": 3.78},
        }
    },
    "nightstand": {
        "rep_dims_m": {
            "S": {"L": 0.35, "W": 0.35, "A": 0.12},
            "M": {"L": 0.45, "W": 0.40, "A": 0.18},
            "L": {"L": 0.55, "W": 0.45, "A": 0.25},
        }
    },
    "wardrobe": {
        "rep_dims_m": {
            "S": {"L": 0.80, "W": 0.55, "A": 0.44},
            "M": {"L": 1.20, "W": 0.60, "A": 0.72},
            "L": {"L": 1.80, "W": 0.65, "A": 1.17},
        }
    },
    "desk": {
        "rep_dims_m": {
            "S": {"L": 0.90, "W": 0.50, "A": 0.45},
            "M": {"L": 1.20, "W": 0.60, "A": 0.72},
            "L": {"L": 1.50, "W": 0.70, "A": 1.05},
        }
    },
    "chair": {
        "rep_dims_m": {
            "S": {"L": 0.42, "W": 0.42, "A": 0.18},
            "M": {"L": 0.50, "W": 0.50, "A": 0.25},
            "L": {"L": 0.60, "W": 0.60, "A": 0.36},
        }
    },
    "bookshelf": {
        "rep_dims_m": {
            "S": {"L": 0.55, "W": 0.30, "A": 0.17},
            "M": {"L": 0.80, "W": 0.35, "A": 0.28},
            "L": {"L": 1.20, "W": 0.40, "A": 0.48},
        }
    },
}

_LIVING_ROOM_SIZE_PROFILES: dict[str, dict[str, Any]] = {
    "sofa": {
        "rep_dims_m": {
            "S": {"L": 1.50, "W": 0.85, "A": 1.28},
            "M": {"L": 2.10, "W": 0.90, "A": 1.89},
            "L": {"L": 2.80, "W": 1.00, "A": 2.80},
        }
    },
    "armchair": {
        "rep_dims_m": {
            "S": {"L": 0.70, "W": 0.70, "A": 0.49},
            "M": {"L": 0.85, "W": 0.85, "A": 0.72},
            "L": {"L": 1.00, "W": 0.95, "A": 0.95},
        }
    },
    "coffee_table": {
        "rep_dims_m": {
            "S": {"L": 0.70, "W": 0.45, "A": 0.32},
            "M": {"L": 1.00, "W": 0.55, "A": 0.55},
            "L": {"L": 1.30, "W": 0.70, "A": 0.91},
        }
    },
    "tv_console": {
        "rep_dims_m": {
            "S": {"L": 0.90, "W": 0.35, "A": 0.32},
            "M": {"L": 1.40, "W": 0.40, "A": 0.56},
            "L": {"L": 1.90, "W": 0.45, "A": 0.86},
        }
    },
    "side_table": {
        "rep_dims_m": {
            "S": {"L": 0.35, "W": 0.35, "A": 0.12},
            "M": {"L": 0.45, "W": 0.45, "A": 0.20},
            "L": {"L": 0.60, "W": 0.50, "A": 0.30},
        }
    },
    "bookshelf": {
        "rep_dims_m": {
            "S": {"L": 0.55, "W": 0.30, "A": 0.17},
            "M": {"L": 0.80, "W": 0.35, "A": 0.28},
            "L": {"L": 1.20, "W": 0.40, "A": 0.48},
        }
    },
    "media_shelf": {
        "rep_dims_m": {
            "S": {"L": 0.70, "W": 0.28, "A": 0.20},
            "M": {"L": 1.00, "W": 0.35, "A": 0.35},
            "L": {"L": 1.40, "W": 0.40, "A": 0.56},
        }
    },
}


def _canonical_from_aliases(
    object_type: object,
    *,
    aliases: dict[str, tuple[str, ...]],
    declared_objects: frozenset[str],
) -> str | None:
    normalized = normalize_profile_token(object_type).rstrip("0123456789").rstrip("_")
    if not normalized:
        return None
    if normalized in declared_objects:
        return normalized
    for canonical, values in aliases.items():
        alias_keys = {normalize_profile_token(canonical)}
        alias_keys.update(normalize_profile_token(value) for value in values)
        if normalized in alias_keys:
            return normalize_profile_token(canonical)
    return None


def bedroom_shadow_size_profile(object_type: object) -> dict[str, Any] | None:
    canonical = _canonical_from_aliases(
        object_type,
        aliases=BEDROOM_LEGACY_ALIASES,
        declared_objects=_BEDROOM_DECLARED_OBJECTS,
    )
    if canonical is None:
        return None
    profile = _BEDROOM_SIZE_PROFILES.get(canonical)
    return deepcopy(profile) if profile is not None else None


def living_room_shadow_size_profile(object_type: object) -> dict[str, Any] | None:
    canonical = _canonical_from_aliases(
        object_type,
        aliases=LIVING_ROOM_LEGACY_ALIASES,
        declared_objects=_LIVING_ROOM_DECLARED_OBJECTS,
    )
    if canonical is None:
        return None
    profile = _LIVING_ROOM_SIZE_PROFILES.get(canonical)
    return deepcopy(profile) if profile is not None else None


def bedroom_legacy_semantic_room_rule(room_type: str) -> dict[str, Any] | None:
    _ = room_type
    from stylist.semantic_program_rules import get_compiled_semantic_room_rule

    return get_compiled_semantic_room_rule("bedroom")


def living_room_legacy_semantic_room_rule(room_type: str) -> dict[str, Any] | None:
    _ = room_type
    from stylist.semantic_program_rules import get_compiled_semantic_room_rule

    return get_compiled_semantic_room_rule("living_room")


BEDROOM_LEGACY_PROFILE = RoomProfile(
    profile_id="bedroom_legacy",
    room_types=frozenset({"bedroom", "guest_bedroom", "primary_bedroom"}),
    canonical_room_type="bedroom",
    layout_traits_enabled=False,
    object_aliases=BEDROOM_LEGACY_ALIASES,
    wall_backed_objects=BEDROOM_WALL_BACKED_OBJECTS,
    floating_objects=BEDROOM_FLOATING_OBJECTS,
    mounted_objects=BEDROOM_MOUNTED_OBJECTS,
    storage_objects=BEDROOM_STORAGE_OBJECTS,
    anchor_objects=BEDROOM_ANCHOR_OBJECTS,
    support_objects=BEDROOM_SUPPORT_OBJECTS,
    seating_objects=BEDROOM_SEATING_OBJECTS,
    surface_objects=BEDROOM_SURFACE_OBJECTS,
    lighting_objects=BEDROOM_LIGHTING_OBJECTS,
    decor_objects=BEDROOM_DECOR_OBJECTS,
    semantic_room_rule_provider=bedroom_legacy_semantic_room_rule,
    size_profile_provider=bedroom_shadow_size_profile,
)

LIVING_ROOM_LEGACY_PROFILE = RoomProfile(
    profile_id="living_room_legacy",
    room_types=frozenset({"living_room", "living", "lounge", "family_room"}),
    canonical_room_type="living_room",
    layout_traits_enabled=False,
    object_aliases=LIVING_ROOM_LEGACY_ALIASES,
    wall_backed_objects=LIVING_ROOM_WALL_BACKED_OBJECTS,
    floating_objects=LIVING_ROOM_FLOATING_OBJECTS,
    mounted_objects=LIVING_ROOM_MOUNTED_OBJECTS,
    storage_objects=LIVING_ROOM_STORAGE_OBJECTS,
    anchor_objects=LIVING_ROOM_ANCHOR_OBJECTS,
    support_objects=LIVING_ROOM_SUPPORT_OBJECTS,
    seating_objects=LIVING_ROOM_SEATING_OBJECTS,
    surface_objects=LIVING_ROOM_SURFACE_OBJECTS,
    lighting_objects=LIVING_ROOM_LIGHTING_OBJECTS,
    decor_objects=LIVING_ROOM_DECOR_OBJECTS,
    semantic_room_rule_provider=living_room_legacy_semantic_room_rule,
    size_profile_provider=living_room_shadow_size_profile,
)
