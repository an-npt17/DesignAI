from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

SemanticRuleProvider = Callable[[str], dict[str, Any] | None]
SizeProfileProvider = Callable[[object], dict[str, Any] | None]
CapacityPolicy = Callable[[dict[str, Any], str], dict[str, Any]]
SemanticPlacementProvider = Callable[
    [str, Sequence[str], Sequence[str]], list[dict[str, Any]]
]


def normalize_profile_token(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


@dataclass(frozen=True)
class RoomProfile:
    profile_id: str
    room_types: frozenset[str]
    canonical_room_type: str
    layout_traits_enabled: bool = False
    object_aliases: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    scoring_aliases: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    non_functional_contract_types: frozenset[str] = field(default_factory=frozenset)
    non_functional_layout_specs: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict
    )
    workflow_objects: frozenset[str] = field(default_factory=frozenset)
    wall_backed_objects: frozenset[str] = field(default_factory=frozenset)
    floating_objects: frozenset[str] = field(default_factory=frozenset)
    mounted_objects: frozenset[str] = field(default_factory=frozenset)
    storage_objects: frozenset[str] = field(default_factory=frozenset)
    anchor_objects: frozenset[str] = field(default_factory=frozenset)
    support_objects: frozenset[str] = field(default_factory=frozenset)
    seating_objects: frozenset[str] = field(default_factory=frozenset)
    surface_objects: frozenset[str] = field(default_factory=frozenset)
    lighting_objects: frozenset[str] = field(default_factory=frozenset)
    decor_objects: frozenset[str] = field(default_factory=frozenset)
    cluster_tag: str | None = None
    semantic_roles_by_object: Mapping[str, str] = field(default_factory=dict)
    relation_targets: Mapping[str, str] = field(default_factory=dict)
    semantic_room_rule_provider: SemanticRuleProvider | None = None
    size_profile_provider: SizeProfileProvider | None = None
    capacity_policy: CapacityPolicy | None = None
    semantic_placement_provider: SemanticPlacementProvider | None = None

    def matches_room_type(self, room_type: object) -> bool:
        return normalize_profile_token(room_type) in self.room_types

    def canonical_object_type(self, object_type: object) -> str | None:
        normalized = (
            normalize_profile_token(object_type).rstrip("0123456789").rstrip("_")
        )
        if not normalized:
            return None

        all_objects = self._all_declared_objects()
        if normalized in all_objects:
            return normalized

        for canonical, aliases in self.object_aliases.items():
            canonical_key = normalize_profile_token(canonical)
            alias_keys = {canonical_key}
            alias_keys.update(normalize_profile_token(alias) for alias in aliases)
            if normalized in alias_keys:
                return canonical_key
        return None

    def has_declared_object(self, object_type: object) -> bool:
        return self.canonical_object_type(object_type) is not None

    def object_traits(
        self,
        object_type: object,
        *,
        include_shadow: bool = True,
    ) -> tuple[str, ...]:
        canonical = self.canonical_object_type(object_type)
        if canonical is None:
            return ()
        if not include_shadow and not self.layout_traits_enabled:
            return ()

        traits: list[str] = []
        trait_sets = (
            ("workflow", self.workflow_objects),
            ("wall_backed", self.wall_backed_objects),
            ("floating", self.floating_objects),
            ("mounted", self.mounted_objects),
            ("storage", self.storage_objects),
            ("anchor", self.anchor_objects),
            ("support", self.support_objects),
            ("seating", self.seating_objects),
            ("surface", self.surface_objects),
            ("lighting", self.lighting_objects),
            ("decor", self.decor_objects),
        )
        for trait_name, object_set in trait_sets:
            if canonical in object_set:
                traits.append(trait_name)
        return tuple(traits)

    def has_trait_object(
        self,
        object_type: object,
        *,
        include_shadow: bool = False,
    ) -> bool:
        return bool(self.object_traits(object_type, include_shadow=include_shadow))

    def semantic_room_rule(self, room_type: str) -> dict[str, Any] | None:
        if self.semantic_room_rule_provider is None:
            return None
        return self.semantic_room_rule_provider(room_type)

    def fallback_size_profile(self, object_type: object) -> dict[str, Any] | None:
        if self.size_profile_provider is None:
            return None
        return self.size_profile_provider(object_type)

    def apply_capacity_policy(
        self,
        capacity_model: dict[str, Any],
        room_type: str,
    ) -> dict[str, Any]:
        if self.capacity_policy is None:
            return capacity_model
        return self.capacity_policy(capacity_model, room_type)

    def semantic_placements_for_members(
        self,
        cluster_id: str,
        members: Sequence[str],
        anchors: Sequence[str],
    ) -> list[dict[str, Any]]:
        if self.semantic_placement_provider is None:
            return []
        return self.semantic_placement_provider(cluster_id, members, anchors)

    def _all_declared_objects(self) -> frozenset[str]:
        return frozenset(
            normalize_profile_token(key) for key in self.object_aliases
        ) | (
            self.workflow_objects
            | self.wall_backed_objects
            | self.floating_objects
            | self.mounted_objects
            | self.storage_objects
            | self.anchor_objects
            | self.support_objects
            | self.seating_objects
            | self.surface_objects
            | self.lighting_objects
            | self.decor_objects
            | self.non_functional_contract_types
        )
