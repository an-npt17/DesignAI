from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator
from pydantic.config import ConfigDict


class NoOverlapConstraint(BaseModel):
    type: Literal["no_overlap"]
    a: str
    b: str


class ContainInConstraint(BaseModel):
    type: Literal["contain_in"]
    a: str
    b: str


class AnchorSideConstraint(BaseModel):
    type: Literal["anchor_side"]
    a: str
    b: str
    side: Literal[
        "head_left",
        "head_right",
        "foot_left",
        "foot_right",
        "head",
        "foot",
        "left",
        "right",
        "top",
        "bottom",
    ]
    gap_min: int
    gap_max: int


class DockToEdgeConstraint(BaseModel):
    type: Literal["dock_to_edge"]
    a: str
    b: str
    b_edge: Literal["front", "back", "left", "right", "top", "bottom"]
    span: Literal["any", "center", "left", "right", "short_edge", "long_edge"]
    gap_min: int
    gap_max: int


class RequiresAccessConstraint(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["requires_access"]
    id: str | None = None
    mode: str | None = None
    required: bool | None = None


HardConstraint = (
    NoOverlapConstraint
    | ContainInConstraint
    | AnchorSideConstraint
    | DockToEdgeConstraint
    | RequiresAccessConstraint
)


class PreferNearConstraint(BaseModel):
    type: Literal["prefer_near"]
    a: str
    b: str
    weight: int


class PreferAlignEdgeConstraint(BaseModel):
    type: Literal["prefer_align_edge"]
    a: str
    b: str
    edge: Literal["left", "right", "top", "bottom", "front", "back", "head", "foot"]
    weight: int


class PreferFacingConstraint(BaseModel):
    type: Literal["prefer_facing"]
    a: str
    b: str
    mode: Literal["face_each_other", "face_same_direction"]
    weight: int


SoftConstraint = (
    PreferNearConstraint | PreferAlignEdgeConstraint | PreferFacingConstraint
)


class FacingRule(BaseModel):
    front: Literal["top", "bottom", "left", "right"]
    notes: str | None = None


class AccessRequirement(BaseModel):
    id: str
    type: str
    required: bool = True


class ClusterRules(BaseModel):
    model_config = ConfigDict(extra="allow")
    grid_mm: int
    allowed_rotations: dict[str, list[int]] = Field(default_factory=dict)
    facing: dict[str, FacingRule] | None = None
    access_requirements: list[AccessRequirement] = Field(default_factory=list)

    @field_validator("allowed_rotations")
    @classmethod
    def _validate_rotations(cls, value: dict[str, list[int]]) -> dict[str, list[int]]:
        allowed = {0, 90, 180, 270}
        for rotations in value.values():
            if any(rotation not in allowed for rotation in rotations):
                raise ValueError("allowed_rotations must be within {0,90,180,270}")
        return value


class Cluster(BaseModel):
    cluster_id: str
    tag: Literal["sleep", "work", "living", "dining", "storage", "kitchen", "misc"]
    members: list[str] = Field(default_factory=list)
    anchors: list[str] = Field(default_factory=list)
    hard_constraints: list[HardConstraint] = Field(default_factory=list)
    soft_constraints: list[SoftConstraint] = Field(default_factory=list)
    cluster_rules: ClusterRules


class ClusterForgeOutput(BaseModel):
    status: Literal["OK", "NEED_INFO", "UNSAT"]
    clusters: list[Cluster] = Field(default_factory=list)
    semantic_layout_program: dict[str, object] | None = None
    style_policy: dict[str, object] | None = None
    notes: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
