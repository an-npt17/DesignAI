from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Status = Literal["OK", "NEED_INFO", "UNSAT"]
Priority = Literal["high", "medium", "low"]
LayoutFocusMode = Literal[
    "viewing",
    "conversation",
    "rest",
    "work",
    "dining",
    "cooking",
    "display",
    "mixed",
]
SupportClusterBehavior = Literal["recede", "balanced", "integrate"]
DistributionMode = Literal["balanced", "edge_weighted", "focal_grouped", "zoned"]
IntentClusterTag = Literal[
    "sleep", "work", "living", "dining", "storage", "kitchen", "misc"
]


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


class InitialLayoutIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent_id: str
    label: str
    summary: str
    focus_mode: LayoutFocusMode
    primary_tag: IntentClusterTag
    secondary_tag: IntentClusterTag | None = None
    circulation_priority: Priority = "medium"
    center_open_preference: Priority = "medium"
    support_cluster_behavior: SupportClusterBehavior = "balanced"
    distribution_mode: DistributionMode = "balanced"
    forge_guidance: list[str] = Field(default_factory=list)
    composer_guidance: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @field_validator("intent_id", "label", "summary")
    @classmethod
    def _non_empty_str(cls, value: str, info) -> str:
        return _strip_required_str(value, info.field_name)

    @field_validator("secondary_tag")
    @classmethod
    def _optional_non_empty_str(cls, value: str | None) -> str | None:
        return _strip_optional_str(value)

    @field_validator("forge_guidance", "composer_guidance", "notes")
    @classmethod
    def _strip_string_lists(cls, value: list[str], info) -> list[str]:
        out: list[str] = []
        for item in value:
            text = _strip_required_str(item, info.field_name)
            if text not in out:
                out.append(text)
        return out

    @model_validator(mode="after")
    def _secondary_tag_differs(self) -> "InitialLayoutIntent":
        if self.secondary_tag is not None and self.secondary_tag == self.primary_tag:
            raise ValueError("secondary_tag must be different from primary_tag")
        return self


class InitialIntentPlannerOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Status
    room_id: str
    intents: list[InitialLayoutIntent] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)

    @field_validator("room_id")
    @classmethod
    def _room_id_non_empty(cls, value: str) -> str:
        return _strip_required_str(value, "room_id")

    @field_validator("notes", "missing")
    @classmethod
    def _strip_string_lists(cls, value: list[str], info) -> list[str]:
        out: list[str] = []
        for item in value:
            text = _strip_required_str(item, info.field_name)
            if text not in out:
                out.append(text)
        return out
