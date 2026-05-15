from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

AllowedRotation = Literal[0, 90, 180, 270]
PlacerStatus = Literal["REPAIRED", "NO_IMPROVEMENT", "NEED_INFO"]
ObjectRepairOp = Literal[
    "rotate_object",
    "mirror_object",
    "nudge_object",
    "swap_objects",
    "set_anchor",
    "set_front_override",
]


def _strip_required_str(value: str, field_name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


class ClusterTransform(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_id: str
    x: int
    y: int
    rot: AllowedRotation

    @field_validator("cluster_id")
    @classmethod
    def _cluster_id_non_empty(cls, value: str) -> str:
        return _strip_required_str(value, "cluster_id")


class ClusterVariantSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_id: str
    variant_id: str

    @field_validator("cluster_id")
    @classmethod
    def _cluster_id_non_empty(cls, value: str) -> str:
        return _strip_required_str(value, "cluster_id")

    @field_validator("variant_id")
    @classmethod
    def _variant_id_non_empty(cls, value: str) -> str:
        return _strip_required_str(value, "variant_id")


class ObjectRepair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_id: str
    object_id: str
    op: ObjectRepairOp
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("cluster_id")
    @classmethod
    def _cluster_id_non_empty(cls, value: str) -> str:
        return _strip_required_str(value, "cluster_id")

    @field_validator("object_id")
    @classmethod
    def _object_id_non_empty(cls, value: str) -> str:
        return _strip_required_str(value, "object_id")


class MacroClusterPlacerOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: PlacerStatus
    cluster_transforms: list[ClusterTransform] = Field(default_factory=list)
    selected_variants: list[ClusterVariantSelection] = Field(default_factory=list)
    object_repairs: list[ObjectRepair] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @field_validator("notes")
    @classmethod
    def _notes_non_empty(cls, value: list[str]) -> list[str]:
        out: list[str] = []
        for item in value:
            out.append(_strip_required_str(item, "notes item"))
        return out

    @field_validator("cluster_transforms")
    @classmethod
    def _unique_cluster_transform_ids(
        cls, value: list[ClusterTransform]
    ) -> list[ClusterTransform]:
        ids = [item.cluster_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("cluster_transforms cluster_id must be unique")
        return value

    @field_validator("selected_variants")
    @classmethod
    def _unique_selected_variant_ids(
        cls, value: list[ClusterVariantSelection]
    ) -> list[ClusterVariantSelection]:
        ids = [item.cluster_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("selected_variants cluster_id must be unique")
        return value

    @model_validator(mode="after")
    def _require_full_layout(self) -> "MacroClusterPlacerOutput":
        if self.status in {"REPAIRED", "NO_IMPROVEMENT", "NEED_INFO"}:
            if not self.cluster_transforms:
                raise ValueError("cluster_transforms required for all statuses")
            if not self.selected_variants:
                raise ValueError("selected_variants required for all statuses")

            transform_ids = {item.cluster_id for item in self.cluster_transforms}
            variant_ids = {item.cluster_id for item in self.selected_variants}

            if transform_ids != variant_ids:
                raise ValueError(
                    "selected_variants cluster_id set must match cluster_transforms cluster_id set"
                )

        return self
