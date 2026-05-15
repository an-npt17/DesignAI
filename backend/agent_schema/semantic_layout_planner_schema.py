from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

PlannerStatus = Literal["OK", "NEEDS_REVIEW", "UNSAT"]
ClusterPriority = Literal["core", "support", "optional"]
LayoutRole = Literal["primary", "secondary", "support", "optional"]
BundleClass = Literal["indispensable", "strong_support", "optional", "decor_light"]
PreserveLevel = Literal["highest", "high", "medium", "low"]
DropOrderBias = Literal[
    "drop_first",
    "drop_early",
    "neutral",
    "drop_late",
    "drop_last",
]
SizeTier = Literal["S", "M", "L"]
ObjectRole = Literal[
    "dominant_anchor",
    "workflow_anchor",
    "support",
    "secondary_support",
    "decor",
]
AffinityLevel = Literal["none", "low", "medium", "high"]
RelationIntentType = Literal[
    "near",
    "separate",
    "face",
    "buffer",
    "claim_wall",
    "claim_daylight",
    "claim_privacy",
    "avoid_entry",
    "preserve_center",
    "dominance",
]
IntentStrength = Literal["hard", "soft"]


class SemanticObject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_type: str
    role: ObjectRole = "support"
    required: bool = False
    max_keep: int | None = None


class SemanticBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bundle_id: str
    objects: list[SemanticObject] = Field(default_factory=list)


class ZoneClaims(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preferred_regions: list[str] = Field(default_factory=list)
    avoid_regions: list[str] = Field(default_factory=list)
    wall_affinity: AffinityLevel = "medium"
    daylight_affinity: AffinityLevel = "none"
    privacy_affinity: AffinityLevel = "none"
    floating_allowed: bool = False


class RelationIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: RelationIntentType
    target: str | None = None
    target_cluster: str | None = None
    strength: IntentStrength = "soft"


class ViabilityScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_support: float = 1.0
    affordance_support: float = 1.0
    brief_support: float = 0.5
    inventory_support: float = 1.0
    conflict_penalty: float = 0.0
    score: float = 1.0


class TierCountObjectHint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_type: str
    min_keep: int = 0
    max_keep: int | None = None
    keep_if_space_surplus: bool = False
    space_surplus_threshold: float = 0.45
    drop_order_bias: DropOrderBias = "neutral"
    preserve_level: PreserveLevel = "medium"
    preferred_size_tier: SizeTier | None = None


class TierCountHints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bundle_class: BundleClass
    preserve_level: PreserveLevel
    keep_if_space_surplus: bool = False
    space_surplus_threshold: float = 0.45
    drop_order_bias: DropOrderBias = "neutral"
    object_hints: list[TierCountObjectHint] = Field(default_factory=list)


class ActiveSemanticCluster(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_id: str
    layout_role: LayoutRole
    priority: ClusterPriority
    activation_reason: str
    semantic_role: str
    dominant_anchor_candidates: list[str] = Field(default_factory=list)
    required_bundles: list[SemanticBundle] = Field(default_factory=list)
    zone_claims: ZoneClaims = Field(default_factory=ZoneClaims)
    relation_intents: list[RelationIntent] = Field(default_factory=list)
    degradation_ladder: list[str] = Field(default_factory=list)
    tier_count_hints: TierCountHints | None = None
    viability_score: ViabilityScore = Field(default_factory=ViabilityScore)


class GlobalLayoutIntent(BaseModel):
    model_config = ConfigDict(extra="allow")

    primary_focus: str = "mixed"
    space_character: str = "balanced_functional"
    prefer_open_center: bool = True
    prefer_core_before_support: bool = True
    prefer_clear_primary_circulation: bool = True


class MacroRelations(BaseModel):
    model_config = ConfigDict(extra="forbid")

    adjacency_preferences: list[dict[str, object]] = Field(default_factory=list)
    separation_preferences: list[dict[str, object]] = Field(default_factory=list)
    orientation_preferences: list[dict[str, object]] = Field(default_factory=list)
    keep_open_regions: list[dict[str, object]] = Field(default_factory=list)
    reserved_regions: list[dict[str, object]] = Field(default_factory=list)


class SelectionConstraints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dominant_anchor_required: list[str] = Field(default_factory=list)
    dominant_workflow_required: list[str] = Field(default_factory=list)
    group_caps: list[dict[str, object]] = Field(default_factory=list)
    group_minimums: list[dict[str, object]] = Field(default_factory=list)


class ControlledDegradation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_drop_order: list[str] = Field(default_factory=list)
    bundle_drop_order: list[str] = Field(default_factory=list)
    never_drop_first: list[str] = Field(default_factory=list)


class QualityTargets(BaseModel):
    model_config = ConfigDict(extra="forbid")

    functionality_weight: float = 1.0
    naturalness_weight: float = 1.0
    semantic_coherence_weight: float = 1.0
    spatial_quality_weight: float = 1.0


class SemanticLayoutProgram(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: PlannerStatus
    room_type: str
    style_policy: dict[str, object] | None = None
    request_contract: dict[str, object] | None = None
    profile_rule_trace: dict[str, object] | None = None
    profile_layout_trace: dict[str, object] | None = None
    profile_shadow_trace: dict[str, object] | None = None
    active_clusters: list[ActiveSemanticCluster] = Field(default_factory=list)
    global_layout_intent: GlobalLayoutIntent = Field(default_factory=GlobalLayoutIntent)
    macro_relations: MacroRelations = Field(default_factory=MacroRelations)
    selection_constraints: SelectionConstraints = Field(
        default_factory=SelectionConstraints
    )
    controlled_degradation: ControlledDegradation = Field(
        default_factory=ControlledDegradation
    )
    quality_targets: QualityTargets = Field(default_factory=QualityTargets)
    missing: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    confidence: float = 0.75
    notes: list[str] = Field(default_factory=list)

    @field_validator("missing", "conflicts", "notes")
    @classmethod
    def _strip_strings(cls, value: list[str]) -> list[str]:
        out: list[str] = []
        for item in value:
            text = item.strip()
            if text and text not in out:
                out.append(text)
        return out
