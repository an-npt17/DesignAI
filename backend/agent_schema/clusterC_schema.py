from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


AllowedRotation = Literal[0, 90, 180, 270]


class ResolvedInventoryItem(BaseModel):
    type: str
    instance_id: str
    count: int
    dims_mm: dict[Literal["w", "h"], int]
    rotations: list[AllowedRotation]
    clearance_mm: int
    collision: Literal["solid", "floor", "on_top"]

    @field_validator("rotations")
    @classmethod
    def _validate_rotations(cls, value: list[AllowedRotation]) -> list[AllowedRotation]:
        if not value:
            raise ValueError("rotations must be non-empty")
        return value


class LocalFrame(BaseModel):
    unit: Literal["mm"]
    grid_mm: int
    origin_note: str


class LocalPlacement(BaseModel):
    instance_id: str
    type: str
    x: int
    y: int
    rot: AllowedRotation


class ClusterFootprintRect(BaseModel):
    instance_id: str
    type: str
    x: int
    y: int
    w: int
    h: int


class ClusterFootprintBBox(BaseModel):
    min_x: int
    min_y: int
    max_x: int
    max_y: int


class ClusterFootprint(BaseModel):
    type: Literal["union_of_rects"]
    rects: list[ClusterFootprintRect] = Field(default_factory=list)
    local_bbox: ClusterFootprintBBox


class ClusterComposerOutput(BaseModel):
    status: Literal["OK", "UNSAT", "NEED_INFO"]
    cluster_id: str
    resolved_inventory: list[ResolvedInventoryItem] = Field(default_factory=list)
    local_frame: LocalFrame
    local_placements: list[LocalPlacement] = Field(default_factory=list)
    cluster_footprint: ClusterFootprint
    notes: list[str] = Field(default_factory=list)
