from __future__ import annotations

from typing import Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

Status = Literal["OK", "NEED_INFO", "UNSAT"]
Priority = Literal["high", "medium", "low"]

AffinityPrefer = Literal[
    "wall",
    "center",
    "window_side",
    "entry_side",
    "far_from_entry",
    "recess_or_edge",
    "long_wall",
    "short_wall",
]

AffinityAvoid = Literal[
    "door_swing",
    "window_clearance",
    "entry_blocking",
    "center",
    "bottleneck",
    "window_blocking",
    "main_path",
]

RelationType = Literal[
    "near",
    "separate",
    "adjacent_if_possible",
    "far_if_possible",
]

KeepOpenRegionType = Literal[
    "entry_buffer",
    "window_buffer",
    "center_lane",
    "work_lane",
]

ClusterOrientationIntent = Literal[
    "face_center",
    "face_window",
    "face_entry",
    "face_cluster",
    "back_to_wall",
    "access_to_open_space",
    "axis_parallel_wall",
    "axis_perpendicular_wall",
    "axis_parallel_window",
    "axis_perpendicular_window",
    "inward_to_room",
    "outward_to_wall",
]

ObjectOrientationIntent = Literal[
    "front_to_open_space",
    "front_to_cluster_center",
    "front_to_room_center",
    "front_to_window",
    "front_to_entry",
    "back_to_wall",
    "side_to_wall",
    "long_axis_parallel_wall",
    "long_axis_perpendicular_wall",
    "align_with_cluster_axis",
    "face_object",
    "face_away_from_object",
    "preserve_front_access",
]

ClusterDirectionalRelationType = Literal[
    "face_each_other",
    "avoid_facing_each_other",
    "same_axis",
    "parallel_alignment",
    "perpendicular_alignment",
    "access_faces_other",
    "turn_toward",
    "turn_away",
]

LayoutFocusMode = Literal[
    "viewing",
    "conversation",
    "rest",
    "work",
    "dining",
    "display",
    "mixed",
]

SupportClusterBehavior = Literal["recede", "balanced", "integrate"]

DistributionMode = Literal["balanced", "edge_weighted", "focal_grouped", "zoned"]


def _strip_required_str(value: str, field_name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _strip_optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        raise ValueError("optional string fields must be null or non-empty strings")
    return value


def _ensure_unique_list(values: list[str], field_name: str) -> list[str]:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return values


class ClusterAffinity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_id: str
    prefer: list[AffinityPrefer] = Field(default_factory=list)
    avoid: list[AffinityAvoid] = Field(default_factory=list)
    priority: Priority
    reason: str

    @field_validator("cluster_id", "reason")
    @classmethod
    def _non_empty_str(cls, value: str, info) -> str:
        return _strip_required_str(value, info.field_name)

    @field_validator("prefer")
    @classmethod
    def _unique_prefer(cls, value: list[AffinityPrefer]) -> list[AffinityPrefer]:
        return _ensure_unique_list(value, "prefer")

    @field_validator("avoid")
    @classmethod
    def _unique_avoid(cls, value: list[AffinityAvoid]) -> list[AffinityAvoid]:
        return _ensure_unique_list(value, "avoid")


class ClusterOrientation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_id: str
    intents: list[ClusterOrientationIntent] = Field(default_factory=list)
    target_cluster_id: str | None = None
    priority: Priority
    reason: str

    @field_validator("cluster_id", "reason")
    @classmethod
    def _non_empty_str(cls, value: str, info) -> str:
        return _strip_required_str(value, info.field_name)

    @field_validator("target_cluster_id")
    @classmethod
    def _optional_non_empty_str(cls, value: str | None) -> str | None:
        return _strip_optional_str(value)

    @field_validator("intents")
    @classmethod
    def _unique_intents(
        cls, value: list[ClusterOrientationIntent]
    ) -> list[ClusterOrientationIntent]:
        return _ensure_unique_list(value, "intents")

    @model_validator(mode="after")
    def _validate_target_logic(self) -> "ClusterOrientation":
        has_face_cluster = "face_cluster" in self.intents

        if has_face_cluster and self.target_cluster_id is None:
            raise ValueError(
                "target_cluster_id is required when intents contains 'face_cluster'"
            )

        if not has_face_cluster and self.target_cluster_id is not None:
            raise ValueError(
                "target_cluster_id must be null unless intents contains 'face_cluster'"
            )

        if (
            self.target_cluster_id is not None
            and self.target_cluster_id == self.cluster_id
        ):
            raise ValueError("target_cluster_id must be different from cluster_id")

        return self


class ObjectOrientation(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    cluster_id: str
    object_id: str
    intents: list[ObjectOrientationIntent] = Field(default_factory=list)
    target_object_id: str | None = None
    target_object_cluster_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("target_object_cluster_id", "target_cluster_id"),
    )
    priority: Priority
    reason: str

    @field_validator("cluster_id", "object_id", "reason")
    @classmethod
    def _non_empty_str(cls, value: str, info) -> str:
        return _strip_required_str(value, info.field_name)

    @field_validator("target_object_id", "target_object_cluster_id")
    @classmethod
    def _optional_non_empty_str(cls, value: str | None) -> str | None:
        return _strip_optional_str(value)

    @field_validator("intents")
    @classmethod
    def _unique_intents(
        cls, value: list[ObjectOrientationIntent]
    ) -> list[ObjectOrientationIntent]:
        return _ensure_unique_list(value, "intents")

    @model_validator(mode="after")
    def _validate_target_logic(self) -> "ObjectOrientation":
        has_targeted_face = any(
            x in self.intents for x in ("face_object", "face_away_from_object")
        )

        if has_targeted_face and self.target_object_id is None:
            raise ValueError(
                "target_object_id is required when intents contains "
                "'face_object' or 'face_away_from_object'"
            )

        if not has_targeted_face:
            if self.target_object_id is not None:
                raise ValueError(
                    "target_object_id must be null unless intents contains "
                    "'face_object' or 'face_away_from_object'"
                )
            if self.target_object_cluster_id is not None:
                raise ValueError(
                    "target_object_cluster_id must be null unless intents contains "
                    "'face_object' or 'face_away_from_object'"
                )

        if (
            self.target_object_id is not None
            and self.target_object_id == self.object_id
            and self.target_object_cluster_id in {None, self.cluster_id}
        ):
            raise ValueError(
                "target object must be different from source object "
                "(same cluster + same object_id is not allowed)"
            )

        return self


class ClusterRelation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    a: str
    b: str
    relation: RelationType
    priority: Priority
    reason: str

    @field_validator("a", "b", "reason")
    @classmethod
    def _non_empty_str(cls, value: str, info) -> str:
        return _strip_required_str(value, info.field_name)

    @model_validator(mode="after")
    def _a_b_not_same(self) -> "ClusterRelation":
        if self.a == self.b:
            raise ValueError("cluster_relations a and b must be different")
        return self


class ClusterDirectionalRelation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    a: str
    b: str
    relation: ClusterDirectionalRelationType
    priority: Priority
    reason: str

    @field_validator("a", "b", "reason")
    @classmethod
    def _non_empty_str(cls, value: str, info) -> str:
        return _strip_required_str(value, info.field_name)

    @model_validator(mode="after")
    def _a_b_not_same(self) -> "ClusterDirectionalRelation":
        if self.a == self.b:
            raise ValueError("cluster_directional_relations a and b must be different")
        return self


class MainPath(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    from_: str = Field(alias="from")
    to_cluster: str
    priority: Priority
    reason: str

    @field_validator("from_", "to_cluster", "reason")
    @classmethod
    def _non_empty_str(cls, value: str, info) -> str:
        return _strip_required_str(value, info.field_name)

    @model_validator(mode="after")
    def _from_not_equal_to_cluster(self) -> "MainPath":
        if self.from_ == self.to_cluster:
            raise ValueError("main_path from and to_cluster must be different")
        return self


class KeepOpenRegion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: KeepOpenRegionType
    near: str
    priority: Priority
    reason: str

    @field_validator("near", "reason")
    @classmethod
    def _non_empty_str(cls, value: str, info) -> str:
        return _strip_required_str(value, info.field_name)


class CirculationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    main_paths: list[MainPath] = Field(default_factory=list)
    keep_open_regions: list[KeepOpenRegion] = Field(default_factory=list)


class LayoutIntentProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    focus_mode: LayoutFocusMode
    primary_cluster_id: str
    secondary_cluster_id: str | None = None
    circulation_priority: Priority = "medium"
    center_open_preference: Priority = "medium"
    support_cluster_behavior: SupportClusterBehavior = "balanced"
    distribution_mode: DistributionMode = "balanced"

    @field_validator("primary_cluster_id")
    @classmethod
    def _primary_cluster_non_empty(cls, value: str) -> str:
        return _strip_required_str(value, "primary_cluster_id")

    @field_validator("secondary_cluster_id")
    @classmethod
    def _secondary_cluster_optional_non_empty(cls, value: str | None) -> str | None:
        return _strip_optional_str(value)

    @model_validator(mode="after")
    def _secondary_cluster_differs(self) -> "LayoutIntentProfile":
        if (
            self.secondary_cluster_id is not None
            and self.secondary_cluster_id == self.primary_cluster_id
        ):
            raise ValueError(
                "secondary_cluster_id must be different from primary_cluster_id"
            )
        return self


class ClusterRelationPlannerOutput(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    status: Status
    room_id: str

    cluster_affinities: list[ClusterAffinity] = Field(default_factory=list)
    cluster_orientations: list[ClusterOrientation] = Field(default_factory=list)
    object_orientations: list[ObjectOrientation] = Field(default_factory=list)
    cluster_relations: list[ClusterRelation] = Field(default_factory=list)
    cluster_directional_relations: list[ClusterDirectionalRelation] = Field(
        default_factory=list
    )

    circulation_plan: CirculationPlan = Field(default_factory=CirculationPlan)
    layout_intent_profile: LayoutIntentProfile | None = None
    placement_guidelines: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)

    @field_validator("room_id")
    @classmethod
    def _room_id_non_empty(cls, value: str) -> str:
        return _strip_required_str(value, "room_id")

    @field_validator("placement_guidelines", "notes", "missing")
    @classmethod
    def _strip_string_lists(cls, value: list[str], info) -> list[str]:
        out: list[str] = []
        for item in value:
            out.append(_strip_required_str(item, info.field_name))
        return out

    @field_validator("cluster_affinities")
    @classmethod
    def _unique_cluster_affinities(
        cls, value: list[ClusterAffinity]
    ) -> list[ClusterAffinity]:
        ids = [item.cluster_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("cluster_affinities cluster_id must be unique")
        return value

    @field_validator("cluster_orientations")
    @classmethod
    def _unique_cluster_orientations(
        cls, value: list[ClusterOrientation]
    ) -> list[ClusterOrientation]:
        ids = [item.cluster_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("cluster_orientations cluster_id must be unique")
        return value

    @field_validator("object_orientations")
    @classmethod
    def _unique_object_orientations(
        cls, value: list[ObjectOrientation]
    ) -> list[ObjectOrientation]:
        seen: set[tuple[str, str]] = set()
        for item in value:
            key = (item.cluster_id, item.object_id)
            if key in seen:
                raise ValueError(
                    "object_orientations must be unique by (cluster_id, object_id): "
                    f"{item.cluster_id}/{item.object_id}"
                )
            seen.add(key)
        return value

    @field_validator("cluster_relations")
    @classmethod
    def _unique_cluster_relations(
        cls, value: list[ClusterRelation]
    ) -> list[ClusterRelation]:
        seen: set[tuple[str, str]] = set()
        for item in value:
            pair = tuple(sorted((item.a, item.b)))
            if pair in seen:
                raise ValueError(
                    f"cluster_relations contains duplicate pair: {pair[0]}-{pair[1]}"
                )
            seen.add(pair)
        return value

    @field_validator("cluster_directional_relations")
    @classmethod
    def _unique_cluster_directional_relations(
        cls, value: list[ClusterDirectionalRelation]
    ) -> list[ClusterDirectionalRelation]:
        seen: set[tuple[str, str]] = set()
        for item in value:
            pair = tuple(sorted((item.a, item.b)))
            if pair in seen:
                raise ValueError(
                    "cluster_directional_relations contains duplicate pair: "
                    f"{pair[0]}-{pair[1]}"
                )
            seen.add(pair)
        return value

    @model_validator(mode="after")
    def _missing_only_for_need_info(self) -> "ClusterRelationPlannerOutput":
        if self.status != "NEED_INFO" and self.missing:
            raise ValueError("missing must be empty unless status=NEED_INFO")
        return self
