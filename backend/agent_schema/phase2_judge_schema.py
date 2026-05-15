from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

JudgeVerdict = Literal["ACCEPT", "REVISE", "REJECT"]
JudgeNextStepMode = Literal["macro_layout", "object_refine", "stop"]


def _strip_required_str(value: str, field_name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


class Phase2JudgeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasonableness_score: int
    verdict: JudgeVerdict
    next_step_mode: JudgeNextStepMode = "macro_layout"
    top_issues: list[str] = Field(default_factory=list)
    repair_advice: list[str] = Field(default_factory=list)
    priority_clusters: list[str] = Field(default_factory=list)

    @field_validator("reasonableness_score")
    @classmethod
    def _score_in_range(cls, value: int) -> int:
        if value < 0 or value > 100:
            raise ValueError("reasonableness_score must be between 0 and 100")
        return value

    @field_validator("top_issues", "repair_advice", "priority_clusters")
    @classmethod
    def _normalize_strings(cls, value: list[str], info) -> list[str]:
        out: list[str] = []
        for item in value:
            out.append(_strip_required_str(item, info.field_name or "field"))
        return out
