from __future__ import annotations

from copy import deepcopy
from typing import Any

KITCHEN_ROOM_TYPES = frozenset(
    {
        "kitchen",
        "kitchenette",
        "open_kitchen",
        "open_plan_kitchen",
        "cooking_area",
        "cook_space",
        "eat_in_kitchen",
        "kitchen_diner",
        "kitchen_dining",
    }
)

KITCHEN_OBJECT_ALIASES: dict[str, tuple[str, ...]] = {
    "fridge": ("refrigerator", "fridge_freezer", "freezer"),
    "sink": ("kitchen_sink", "wash_sink"),
    "stove": ("range", "oven_range", "cooker", "hob"),
    "cooktop": ("hob", "induction_hob", "gas_hob"),
    "range_hood": ("hood", "extractor_hood", "vent_hood"),
    "kitchen_base_cabinet": (
        "base_cabinet",
        "counter",
        "countertop",
        "prep_counter",
        "worktop",
        "kitchen_counter",
    ),
    "kitchen_tall_cabinet": ("tall_cabinet", "utility_cabinet"),
    "kitchen_wall_cabinet": ("wall_cabinet", "upper_cabinet"),
    "pantry_cabinet": ("pantry", "larder"),
    "dishwasher": ("dish_washer",),
    "kitchen_island": ("island", "prep_island"),
    "dining_table": (
        "breakfast_table",
        "small_dining_table",
        "eat_in_table",
        "kitchen_table",
    ),
    "dining_chair": ("dining_seat", "breakfast_chair", "kitchen_chair"),
    "bar_stool": ("counter_stool", "island_stool", "breakfast_stool"),
    "microwave": ("microwave_oven",),
}

KITCHEN_WORKFLOW_OBJECTS = frozenset(
    {
        "fridge",
        "sink",
        "stove",
        "cooktop",
        "kitchen_base_cabinet",
        "dishwasher",
    }
)

KITCHEN_WALL_BACKED_OBJECTS = frozenset(
    {
        "fridge",
        "sink",
        "stove",
        "kitchen_base_cabinet",
        "kitchen_tall_cabinet",
        "pantry_cabinet",
        "dishwasher",
    }
)

KITCHEN_FLOATING_OBJECTS = frozenset(
    {
        "kitchen_island",
        "dining_table",
        "dining_chair",
        "bar_stool",
    }
)

KITCHEN_MOUNTED_OBJECTS = frozenset(
    {
        "cooktop",
        "range_hood",
        "kitchen_wall_cabinet",
        "microwave",
        "pendant_light",
    }
)

KITCHEN_STORAGE_OBJECTS = frozenset(
    {
        "kitchen_base_cabinet",
        "kitchen_tall_cabinet",
        "kitchen_wall_cabinet",
        "pantry_cabinet",
    }
)

_KITCHEN_SIZE_PROFILES: dict[str, dict[str, Any]] = {
    "fridge": {
        "rep_dims_m": {
            "S": {"L": 0.65, "W": 0.65, "A": 0.42},
            "M": {"L": 0.75, "W": 0.70, "A": 0.53},
            "L": {"L": 0.90, "W": 0.75, "A": 0.68},
        }
    },
    "sink": {
        "rep_dims_m": {
            "S": {"L": 0.60, "W": 0.50, "A": 0.30},
            "M": {"L": 0.80, "W": 0.55, "A": 0.44},
            "L": {"L": 1.00, "W": 0.60, "A": 0.60},
        }
    },
    "stove": {
        "rep_dims_m": {
            "S": {"L": 0.60, "W": 0.60, "A": 0.36},
            "M": {"L": 0.75, "W": 0.65, "A": 0.49},
            "L": {"L": 0.90, "W": 0.70, "A": 0.63},
        }
    },
    "cooktop": {
        "rep_dims_m": {
            "S": {"L": 0.55, "W": 0.50, "A": 0.28},
            "M": {"L": 0.65, "W": 0.55, "A": 0.36},
            "L": {"L": 0.80, "W": 0.60, "A": 0.48},
        }
    },
    "kitchen_base_cabinet": {
        "rep_dims_m": {
            "S": {"L": 0.90, "W": 0.55, "A": 0.50},
            "M": {"L": 1.20, "W": 0.60, "A": 0.72},
            "L": {"L": 1.80, "W": 0.65, "A": 1.17},
        }
    },
    "kitchen_tall_cabinet": {
        "rep_dims_m": {
            "S": {"L": 0.55, "W": 0.55, "A": 0.30},
            "M": {"L": 0.75, "W": 0.60, "A": 0.45},
            "L": {"L": 1.00, "W": 0.65, "A": 0.65},
        }
    },
    "pantry_cabinet": {
        "rep_dims_m": {
            "S": {"L": 0.55, "W": 0.55, "A": 0.30},
            "M": {"L": 0.75, "W": 0.60, "A": 0.45},
            "L": {"L": 1.00, "W": 0.65, "A": 0.65},
        }
    },
    "dishwasher": {
        "rep_dims_m": {
            "S": {"L": 0.55, "W": 0.55, "A": 0.30},
            "M": {"L": 0.60, "W": 0.60, "A": 0.36},
            "L": {"L": 0.70, "W": 0.65, "A": 0.46},
        }
    },
    "kitchen_island": {
        "rep_dims_m": {
            "S": {"L": 1.20, "W": 0.70, "A": 0.84},
            "M": {"L": 1.60, "W": 0.85, "A": 1.36},
            "L": {"L": 2.20, "W": 1.00, "A": 2.20},
        }
    },
    "dining_table": {
        "rep_dims_m": {
            "S": {"L": 0.80, "W": 0.70, "A": 0.56},
            "M": {"L": 1.40, "W": 0.85, "A": 1.19},
            "L": {"L": 1.80, "W": 0.95, "A": 1.71},
        }
    },
    "dining_chair": {
        "rep_dims_m": {
            "S": {"L": 0.42, "W": 0.42, "A": 0.18},
            "M": {"L": 0.48, "W": 0.48, "A": 0.23},
            "L": {"L": 0.55, "W": 0.55, "A": 0.30},
        }
    },
    "bar_stool": {
        "rep_dims_m": {
            "S": {"L": 0.38, "W": 0.38, "A": 0.14},
            "M": {"L": 0.42, "W": 0.42, "A": 0.18},
            "L": {"L": 0.48, "W": 0.48, "A": 0.23},
        }
    },
}

_KITCHEN_SEMANTIC_ROOM_RULE: dict[str, Any] = {
    "room_type": "kitchen",
    "policy": {
        "intent": "kitchen_workflow",
        "use_all_objects": False,
        "selection_policy": "gated_kitchen_profile",
    },
    "clusters": [
        {
            "cluster_id": "kitchen_workflow_core",
            "priority": "core",
            "activation": {
                "base_rule": "must_include",
                "always_consider": True,
                "requires_usefulness_test": False,
                "conditions": [],
            },
            "object_program": {
                "required": [
                    "kitchen_base_cabinet",
                    "fridge",
                    "sink",
                    "stove",
                ],
                "required_if_kept": [],
                "choose_exactly_one_from": [],
                "choose_exactly_one_from_if_kept": [],
                "choose_at_least_one_from": [],
                "optional": ["dishwasher"],
                "optional_limits": {
                    "global": 1,
                    "by_object": {"dishwasher": 1},
                },
            },
            "semantic": {
                "dominant_anchor_candidates": ["kitchen_base_cabinet"],
                "notes": [
                    "Keep fridge, sink, stove, and prep counter in one wall-led workflow cluster."
                ],
            },
            "degradation_hints": {
                "preserve_first": [
                    "kitchen_base_cabinet",
                    "fridge",
                    "sink",
                    "stove",
                ],
                "shrink_before_drop": ["kitchen_base_cabinet"],
                "drop_first": ["dishwasher"],
            },
            "tier_count_hints": {
                "bundle_class": "indispensable",
                "preserve_level": "highest",
                "keep_if_space_surplus": False,
                "space_surplus_threshold": 0.0,
                "drop_order_bias": "drop_last",
                "object_hints": [
                    {
                        "object_type": "kitchen_base_cabinet",
                        "min_keep": 1,
                        "max_keep": 1,
                        "keep_if_space_surplus": False,
                        "space_surplus_threshold": 0.0,
                        "drop_order_bias": "drop_last",
                        "preserve_level": "highest",
                        "preferred_size_tier": "M",
                    },
                    {
                        "object_type": "fridge",
                        "min_keep": 1,
                        "max_keep": 1,
                        "keep_if_space_surplus": False,
                        "space_surplus_threshold": 0.0,
                        "drop_order_bias": "drop_last",
                        "preserve_level": "highest",
                        "preferred_size_tier": "M",
                    },
                    {
                        "object_type": "sink",
                        "min_keep": 1,
                        "max_keep": 1,
                        "keep_if_space_surplus": False,
                        "space_surplus_threshold": 0.0,
                        "drop_order_bias": "drop_last",
                        "preserve_level": "highest",
                        "preferred_size_tier": "M",
                    },
                    {
                        "object_type": "stove",
                        "min_keep": 1,
                        "max_keep": 1,
                        "keep_if_space_surplus": False,
                        "space_surplus_threshold": 0.0,
                        "drop_order_bias": "drop_last",
                        "preserve_level": "highest",
                        "preferred_size_tier": "M",
                    },
                    {
                        "object_type": "dishwasher",
                        "min_keep": 0,
                        "max_keep": 1,
                        "keep_if_space_surplus": True,
                        "space_surplus_threshold": 0.55,
                        "drop_order_bias": "drop_early",
                        "preserve_level": "medium",
                        "preferred_size_tier": "S",
                    },
                ],
            },
        },
        {
            "cluster_id": "kitchen_storage_support",
            "priority": "support",
            "activation": {
                "base_rule": "may_include_if_useful",
                "always_consider": False,
                "requires_usefulness_test": True,
                "conditions": [
                    {
                        "predicate": "storage",
                        "effects": {
                            "required_if_kept": ["pantry_cabinet"],
                        },
                    }
                ],
            },
            "object_program": {
                "required": [],
                "required_if_kept": [],
                "choose_exactly_one_from": [],
                "choose_exactly_one_from_if_kept": [],
                "choose_at_least_one_from": [],
                "optional": ["kitchen_tall_cabinet", "pantry_cabinet"],
                "optional_limits": {
                    "global": 1,
                    "by_object": {
                        "kitchen_tall_cabinet": 1,
                        "pantry_cabinet": 1,
                    },
                },
            },
            "semantic": {
                "dominant_anchor_candidates": ["kitchen_tall_cabinet"],
                "notes": [
                    "Supplemental tall storage should stay wall-backed and drop before the workflow core."
                ],
            },
            "degradation_hints": {
                "preserve_first": ["kitchen_tall_cabinet"],
                "shrink_before_drop": [],
                "drop_first": ["pantry_cabinet"],
            },
            "tier_count_hints": {
                "bundle_class": "strong_support",
                "preserve_level": "high",
                "keep_if_space_surplus": True,
                "space_surplus_threshold": 0.45,
                "drop_order_bias": "drop_late",
                "object_hints": [
                    {
                        "object_type": "kitchen_tall_cabinet",
                        "min_keep": 0,
                        "max_keep": 1,
                        "keep_if_space_surplus": True,
                        "space_surplus_threshold": 0.45,
                        "drop_order_bias": "drop_late",
                        "preserve_level": "high",
                        "preferred_size_tier": "S",
                    },
                    {
                        "object_type": "pantry_cabinet",
                        "min_keep": 0,
                        "max_keep": 1,
                        "keep_if_space_surplus": True,
                        "space_surplus_threshold": 0.55,
                        "drop_order_bias": "drop_early",
                        "preserve_level": "medium",
                        "preferred_size_tier": "S",
                    },
                ],
            },
        },
        {
            "cluster_id": "kitchen_island_optional",
            "priority": "optional",
            "activation": {
                "base_rule": "may_include_if_useful",
                "always_consider": False,
                "requires_usefulness_test": True,
                "conditions": [
                    {
                        "predicate": "island",
                        "effects": {
                            "required_if_kept": ["kitchen_island"],
                            "optional": ["bar_stool"],
                        },
                    }
                ],
            },
            "object_program": {
                "required": [],
                "required_if_kept": ["kitchen_island"],
                "choose_exactly_one_from": [],
                "choose_exactly_one_from_if_kept": [],
                "choose_at_least_one_from": [],
                "optional": ["bar_stool"],
                "optional_limits": {
                    "global": 2,
                    "by_object": {"kitchen_island": 1, "bar_stool": 2},
                },
            },
            "semantic": {
                "dominant_anchor_candidates": ["kitchen_island"],
                "notes": [
                    "Only keep an island when the room and brief leave enough center clearance."
                ],
            },
            "degradation_hints": {
                "preserve_first": ["kitchen_island"],
                "shrink_before_drop": ["kitchen_island"],
                "drop_first": ["bar_stool", "kitchen_island"],
            },
            "tier_count_hints": {
                "bundle_class": "optional",
                "preserve_level": "medium",
                "keep_if_space_surplus": True,
                "space_surplus_threshold": 0.62,
                "drop_order_bias": "drop_early",
                "object_hints": [
                    {
                        "object_type": "kitchen_island",
                        "min_keep": 0,
                        "max_keep": 1,
                        "keep_if_space_surplus": True,
                        "space_surplus_threshold": 0.62,
                        "drop_order_bias": "drop_early",
                        "preserve_level": "medium",
                        "preferred_size_tier": "S",
                    },
                    {
                        "object_type": "bar_stool",
                        "min_keep": 0,
                        "max_keep": 2,
                        "keep_if_space_surplus": True,
                        "space_surplus_threshold": 0.70,
                        "drop_order_bias": "drop_first",
                        "preserve_level": "low",
                        "preferred_size_tier": "S",
                    },
                ],
            },
        },
        {
            "cluster_id": "kitchen_dining_core",
            "priority": "core",
            "activation": {
                "base_rule": "must_include",
                "always_consider": True,
                "requires_usefulness_test": False,
                "conditions": [
                    {
                        "predicate": "dining",
                        "effects": {
                            "required": ["dining_table"],
                            "required_if_kept": ["dining_chair"],
                        },
                    },
                    {
                        "predicate": "breakfast",
                        "effects": {
                            "required": ["dining_table"],
                            "required_if_kept": ["dining_chair"],
                        },
                    },
                ],
            },
            "object_program": {
                "required": ["dining_table"],
                "required_if_kept": ["dining_chair"],
                "choose_exactly_one_from": [],
                "choose_exactly_one_from_if_kept": [],
                "choose_at_least_one_from": [],
                "optional": [],
                "optional_limits": {
                    "global": 2,
                    "by_object": {"dining_table": 1, "dining_chair": 2},
                },
            },
            "semantic": {
                "dominant_anchor_candidates": ["dining_table"],
                "notes": [
                    "Every kitchen profile includes a compact eat-in setting; keep it near but not inside the cooking workflow."
                ],
            },
            "degradation_hints": {
                "preserve_first": ["dining_table"],
                "shrink_before_drop": ["dining_table", "dining_chair"],
                "drop_first": ["dining_chair", "dining_table"],
            },
            "tier_count_hints": {
                "bundle_class": "strong_support",
                "preserve_level": "high",
                "keep_if_space_surplus": False,
                "space_surplus_threshold": 0.0,
                "drop_order_bias": "drop_late",
                "object_hints": [
                    {
                        "object_type": "dining_table",
                        "min_keep": 1,
                        "max_keep": 1,
                        "keep_if_space_surplus": False,
                        "space_surplus_threshold": 0.0,
                        "drop_order_bias": "drop_late",
                        "preserve_level": "high",
                        "preferred_size_tier": "S",
                    },
                    {
                        "object_type": "dining_chair",
                        "min_keep": 1,
                        "max_keep": 2,
                        "keep_if_space_surplus": True,
                        "space_surplus_threshold": 0.45,
                        "drop_order_bias": "drop_first",
                        "preserve_level": "medium",
                        "preferred_size_tier": "S",
                    },
                ],
            },
        },
    ],
    "global_program": {
        "dominant_anchor_required": [],
        "dominant_workflow_required": [["fridge", "sink", "stove"]],
        "group_caps": [
            {
                "objects": ["kitchen_tall_cabinet", "pantry_cabinet"],
                "max_keep": 1,
            },
            {
                "objects": ["kitchen_island"],
                "max_keep": 1,
            },
            {
                "objects": ["dining_table"],
                "max_keep": 1,
            },
            {
                "objects": ["bar_stool", "dining_chair"],
                "max_keep": 2,
            },
        ],
        "group_minimums": [],
        "soft_preferences": [
            "preserve a compact fridge-sink-stove workflow and a small eat-in dining setting before adding optional island furniture",
            "prefer wall-backed kitchen equipment and keep clear circulation between cooking and dining zones",
        ],
    },
}


def kitchen_key(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def is_kitchen_room_type(room_type: object) -> bool:
    return kitchen_key(room_type) in KITCHEN_ROOM_TYPES


def canonical_kitchen_object_type(object_type: object) -> str | None:
    normalized = kitchen_key(object_type).rstrip("0123456789").rstrip("_")
    if not normalized:
        return None
    if normalized in KITCHEN_OBJECT_ALIASES:
        return normalized
    for canonical, aliases in KITCHEN_OBJECT_ALIASES.items():
        alias_keys = {canonical, *aliases}
        if normalized in alias_keys:
            return canonical
    return normalized if normalized in _ALL_KITCHEN_OBJECTS else None


def is_kitchen_object_like(object_type: object) -> bool:
    return canonical_kitchen_object_type(object_type) is not None


def is_kitchen_workflow_object(object_type: object) -> bool:
    canonical = canonical_kitchen_object_type(object_type)
    return canonical in KITCHEN_WORKFLOW_OBJECTS


def is_kitchen_wall_backed_object(object_type: object) -> bool:
    canonical = canonical_kitchen_object_type(object_type)
    return canonical in KITCHEN_WALL_BACKED_OBJECTS


def is_kitchen_floating_object(object_type: object) -> bool:
    canonical = canonical_kitchen_object_type(object_type)
    return canonical in KITCHEN_FLOATING_OBJECTS


def is_kitchen_mounted_object(object_type: object) -> bool:
    canonical = canonical_kitchen_object_type(object_type)
    return canonical in KITCHEN_MOUNTED_OBJECTS


def is_kitchen_storage_object(object_type: object) -> bool:
    canonical = canonical_kitchen_object_type(object_type)
    return canonical in KITCHEN_STORAGE_OBJECTS


def kitchen_fallback_size_profile(object_type: object) -> dict[str, Any] | None:
    canonical = canonical_kitchen_object_type(object_type)
    if canonical is None:
        return None
    profile = _KITCHEN_SIZE_PROFILES.get(canonical)
    return deepcopy(profile) if profile is not None else None


def kitchen_semantic_room_rule(room_type: object) -> dict[str, Any]:
    rule = deepcopy(_KITCHEN_SEMANTIC_ROOM_RULE)
    rule["room_type"] = kitchen_key(room_type) or "kitchen"
    return rule


_ALL_KITCHEN_OBJECTS = (
    frozenset(KITCHEN_OBJECT_ALIASES)
    | KITCHEN_WORKFLOW_OBJECTS
    | KITCHEN_WALL_BACKED_OBJECTS
    | KITCHEN_FLOATING_OBJECTS
    | KITCHEN_MOUNTED_OBJECTS
    | KITCHEN_STORAGE_OBJECTS
)
