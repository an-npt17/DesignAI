from __future__ import annotations

from typing import Any

ROOM_SURFACE_GROUPS: dict[str, dict[str, Any]] = {
    "bedroom": {
        "no_stack": [
            "bed",
            "wardrobe",
            "dresser",
            "nightstand",
            "bench",
            "desk",
            "chair",
            "office_chair",
            "bookshelf",
            "storage_cabinet",
            "armchair",
            "side_table",
            "floor_lamp",
            "laundry_basket",
            "tv_console",
        ],
        "can_stack_or_be_stacked_or_hang_or_soft": {
            "bed": [
                "mattress",
                "headboard",
                "rug",
                "throw_blanket",
                "cushion",
                "pet_bed",
            ],
            "nightstand": [
                "bedside_lamp",
            ],
            "tv_console": [],
            "desk": [
                "laptop",
                "monitor",
                "desktop_pc",
                "keyboard",
                "mouse",
                "printer",
                "desk_lamp",
                "speaker",
                "smart_speaker",
            ],
            "dresser": [
                "mirror",
                "decor",
            ],
            "side_table": [
                "table_lamp",
                "plant",
            ],
            "__opening__": [
                "curtain",
                "blind",
            ],
            "__wall__": [
                "wall_art",
                "clock",
                "wall_sconce",
                "air_conditioner",
            ],
            "__ceiling__": [
                "ceiling_light",
                "pendant_light",
            ],
        },
    },
    "guest_room": {
        "no_stack": [
            "bed",
            "wardrobe",
            "dresser",
            "nightstand",
            "bench",
            "desk",
            "chair",
            "bookshelf",
            "storage_cabinet",
            "armchair",
            "side_table",
            "floor_lamp",
            "laundry_basket",
        ],
        "can_stack_or_be_stacked_or_hang_or_soft": {
            "bed": [
                "mattress",
                "headboard",
                "rug",
                "cushion",
                "throw_blanket",
            ],
            "nightstand": [
                "bedside_lamp",
            ],
            "desk": [
                "desk_lamp",
                "speaker",
                "smart_speaker",
            ],
            "dresser": [
                "mirror",
                "decor",
            ],
            "side_table": [
                "table_lamp",
                "plant",
            ],
            "__opening__": [
                "curtain",
                "blind",
            ],
            "__wall__": [
                "wall_art",
                "clock",
                "wall_sconce",
                "air_conditioner",
            ],
            "__ceiling__": [
                "ceiling_light",
                "pendant_light",
            ],
        },
    },
    "kids_room": {
        "no_stack": [
            "bed",
            "wardrobe",
            "dresser",
            "nightstand",
            "desk",
            "chair",
            "bookshelf",
            "storage_cabinet",
            "bean_bag",
            "side_table",
            "floor_lamp",
            "laundry_basket",
        ],
        "can_stack_or_be_stacked_or_hang_or_soft": {
            "bed": [
                "mattress",
                "headboard",
                "rug",
                "throw_blanket",
                "pet_bed",
            ],
            "nightstand": [
                "bedside_lamp",
            ],
            "desk": [
                "laptop",
                "monitor",
                "keyboard",
                "mouse",
                "desk_lamp",
                "speaker",
            ],
            "dresser": [
                "mirror",
                "decor",
            ],
            "side_table": [
                "table_lamp",
                "plant",
            ],
            "bean_bag": [
                "cushion",
            ],
            "__opening__": [
                "curtain",
                "blind",
            ],
            "__wall__": [
                "wall_art",
                "clock",
                "wall_sconce",
                "air_conditioner",
            ],
            "__ceiling__": [
                "ceiling_light",
                "pendant_light",
            ],
        },
    },
    "living_room": {
        "no_stack": [
            "sofa",
            "sectional_sofa",
            "armchair",
            "recliner",
            "ottoman",
            "coffee_table",
            "side_table",
            "console_table",
            "tv_console",
            "media_shelf",
            "bookshelf",
            "storage_cabinet",
            "floor_lamp",
            "bean_bag",
            "pet_bed",
            "shoe_rack",
        ],
        "can_stack_or_be_stacked_or_hang_or_soft": {
            "sofa": [
                "rug",
                "cushion",
                "throw_blanket",
            ],
            "tv_console": [
                "speaker",
            ],
            "media_shelf": [
                "projector",
            ],
            "side_table": [
                "table_lamp",
                "plant",
                "smart_speaker",
            ],
            "console_table": [
                "mirror",
                "decor",
            ],
            "coffee_table": [
                "vase",
            ],
            "__opening__": [
                "curtain",
                "blind",
            ],
            "__wall__": [
                "wall_art",
                "clock",
                "projector_screen",
                "wall_sconce",
                "air_conditioner",
            ],
            "__ceiling__": [
                "ceiling_light",
                "pendant_light",
                "track_light",
            ],
        },
    },
    "dining_room": {
        "no_stack": [
            "dining_table",
            "dining_chair",
            "bar_stool",
            "buffet_sideboard",
            "china_cabinet",
            "console_table",
            "side_table",
            "bookshelf",
            "storage_cabinet",
            "bar_cart",
            "plant",
        ],
        "can_stack_or_be_stacked_or_hang_or_soft": {
            "dining_table": [
                "rug",
                "vase",
            ],
            "buffet_sideboard": [
                "mirror",
                "decor",
                "speaker",
                "smart_speaker",
            ],
            "__opening__": [
                "curtain",
                "blind",
            ],
            "__wall__": [
                "wall_art",
                "clock",
                "wall_sconce",
                "air_conditioner",
            ],
            "__ceiling__": [
                "pendant_light",
                "ceiling_light",
                "track_light",
            ],
        },
    },
    "kitchen": {
        "no_stack": [
            "kitchen_base_cabinet",
            "kitchen_tall_cabinet",
            "kitchen_island",
            "fridge",
            "stove",
            "sink",
            "dishwasher",
            "pantry_cabinet",
            "wine_cabinet",
            "bar_cart",
            "bathroom_stool",
        ],
        "can_stack_or_be_stacked_or_hang_or_soft": {
            "stove": [
                "range_hood",
            ],
            "kitchen_tall_cabinet": [
                "oven",
            ],
            "kitchen_base_cabinet": [
                "cooktop",
                "microwave",
                "rice_cooker",
                "electric_kettle",
                "coffee_machine",
                "toaster",
                "air_fryer",
                "blender",
            ],
            "kitchen_island": [
                "decor",
                "vase",
            ],
            "bar_cart": [
                "plant",
                "speaker",
                "smart_speaker",
            ],
            "__wall__": [
                "kitchen_wall_cabinet",
                "clock",
                "wall_sconce",
                "air_conditioner",
            ],
            "__ceiling__": [
                "ceiling_light",
                "pendant_light",
                "track_light",
            ],
        },
    },
    "bathroom": {
        "no_stack": [
            "bathroom_vanity",
            "toilet",
            "shower",
            "bathtub",
            "bidet",
            "bathroom_stool",
            "laundry_basket",
        ],
        "can_stack_or_be_stacked_or_hang_or_soft": {
            "bathroom_vanity": [
                "mirror",
                "medicine_cabinet",
                "bathroom_shelf",
                "wall_sconce",
                "decor",
                "plant",
            ],
            "shower": [
                "towel_rack",
                "shower_niche",
            ],
            "__ceiling__": [
                "ceiling_light",
                "pendant_light",
            ],
        },
    },
    "laundry_room": {
        "no_stack": [
            "washing_machine",
            "dryer",
            "utility_sink",
            "laundry_basket",
            "drying_rack",
            "ironing_board",
            "storage_cabinet",
        ],
        "can_stack_or_be_stacked_or_hang_or_soft": {
            "ironing_board": [
                "iron",
            ],
            "utility_sink": [
                "bathroom_shelf",
            ],
            "__utility_zone__": [
                "dehumidifier",
                "heater",
            ],
            "__wall__": [
                "wall_sconce",
                "clock",
                "air_conditioner",
            ],
            "__ceiling__": [
                "ceiling_light",
            ],
        },
    },
    "home_office": {
        "no_stack": [
            "desk",
            "office_chair",
            "chair",
            "filing_cabinet",
            "office_pedestal",
            "bookshelf",
            "storage_cabinet",
            "side_table",
            "armchair",
            "floor_lamp",
            "printer",
        ],
        "can_stack_or_be_stacked_or_hang_or_soft": {
            "desk": [
                "laptop",
                "monitor",
                "desktop_pc",
                "keyboard",
                "mouse",
                "desk_lamp",
                "speaker",
                "smart_speaker",
                "rug",
            ],
            "side_table": [
                "table_lamp",
                "plant",
                "decor",
            ],
            "__opening__": [
                "curtain",
                "blind",
            ],
            "__wall__": [
                "whiteboard",
                "wall_art",
                "clock",
                "wall_sconce",
                "air_conditioner",
            ],
            "__ceiling__": [
                "ceiling_light",
                "pendant_light",
                "track_light",
            ],
        },
    },
    "entryway": {
        "no_stack": [
            "entry_bench",
            "shoe_rack",
            "coat_rack",
            "console_table",
            "side_table",
            "umbrella_stand",
            "storage_cabinet",
            "bench",
        ],
        "can_stack_or_be_stacked_or_hang_or_soft": {
            "console_table": [
                "mirror",
                "decor",
                "table_lamp",
                "plant",
                "vase",
            ],
            "entry_bench": [
                "rug",
            ],
            "__wall__": [
                "wall_art",
                "clock",
                "wall_sconce",
                "air_conditioner",
            ],
            "__ceiling__": [
                "ceiling_light",
            ],
        },
    },
    "studio": {
        "no_stack": [
            "bed",
            "wardrobe",
            "dresser",
            "nightstand",
            "desk",
            "office_chair",
            "chair",
            "sofa",
            "sectional_sofa",
            "armchair",
            "ottoman",
            "coffee_table",
            "side_table",
            "tv_console",
            "bookshelf",
            "storage_cabinet",
            "dining_table",
            "dining_chair",
            "kitchen_base_cabinet",
            "kitchen_tall_cabinet",
            "fridge",
            "stove",
            "sink",
            "floor_lamp",
            "laundry_basket",
        ],
        "can_stack_or_be_stacked_or_hang_or_soft": {
            "bed": [
                "mattress",
                "headboard",
                "bedside_lamp",
                "cushion",
                "throw_blanket",
            ],
            "desk": [
                "monitor",
                "laptop",
                "keyboard",
                "mouse",
                "printer",
                "desk_lamp",
            ],
            "sofa": [
                "rug",
            ],
            "tv_console": [
                "speaker",
                "smart_speaker",
            ],
            "side_table": [
                "table_lamp",
                "plant",
            ],
            "dresser": [
                "mirror",
                "decor",
            ],
            "kitchen_tall_cabinet": [
                "oven",
            ],
            "kitchen_base_cabinet": [
                "cooktop",
                "microwave",
                "rice_cooker",
                "electric_kettle",
                "coffee_machine",
                "toaster",
                "air_fryer",
                "blender",
            ],
            "__opening__": [
                "curtain",
                "blind",
            ],
            "__wall__": [
                "wall_art",
                "clock",
                "wall_sconce",
                "kitchen_wall_cabinet",
                "range_hood",
                "air_conditioner",
            ],
            "__ceiling__": [
                "ceiling_light",
                "pendant_light",
                "track_light",
            ],
        },
    },
    "balcony": {
        "no_stack": [
            "chair",
            "armchair",
            "side_table",
            "bench",
            "plant",
            "storage_cabinet",
        ],
        "can_stack_or_be_stacked_or_hang_or_soft": {
            "side_table": [
                "decor",
                "floor_lamp",
            ],
            "bench": [
                "rug",
            ],
            "__wall__": [
                "clock",
                "wall_sconce",
            ],
            "__ceiling__": [
                "ceiling_light",
            ],
        },
    },
}


def _build_room_essentials() -> dict[str, list[dict[str, object]]]:
    essentials: dict[str, list[dict[str, object]]] = {}
    for room_type, groups in ROOM_SURFACE_GROUPS.items():
        items: list[dict[str, object]] = []
        for key in ("no_stack", "can_stack_or_be_stacked_or_hang_or_soft"):
            for name in groups.get(key, []):
                items.append({"item": name, "recommended": 1})
        essentials[room_type] = items
    return essentials


ROOM_ESSENTIALS_SEED = _build_room_essentials()
