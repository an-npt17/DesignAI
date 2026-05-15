from __future__ import annotations

from typing import Any


def get_soft_rule_templates() -> dict[str, dict[str, Any]]:
    return {
        "color_composition": {
            "context_fields_to_find": [
                "palette",
                "accent_ratio",
                "dominant_materials",
            ]
        },
        "style_specific_aesthetics": {
            "context_fields_to_find": [
                "materials",
                "forms",
                "textures",
            ]
        },
        "room_specific_feng_shui": {
            "context_fields_to_find": [
                "circulation",
                "entry_path",
                "primary_anchor",
            ]
        },
        "personal_feng_shui": {
            "context_fields_to_find": [
                "element_colors",
                "balance",
                "avoidances",
            ]
        },
        "lighting_color_interaction": {
            "context_fields_to_find": [
                "cct",
                "daylight_balance",
                "accent_lighting",
            ]
        },
    }
