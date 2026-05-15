from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

AllowedRotation = Literal[0, 90, 180, 270]
ComposerStatus = Literal["OK", "NEEDS_REVIEW", "UNSAT", "NEED_INFO", "SEMANTIC_FAIL"]


def _strip_required_str(value: str, field_name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


class CardinalAxisVector(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dx: int
    dy: int

    @model_validator(mode="after")
    def _validate_cardinal_axis(self) -> "CardinalAxisVector":
        allowed = {(1, 0), (-1, 0), (0, 1), (0, -1)}
        if (self.dx, self.dy) not in allowed:
            raise ValueError(
                "axis vector must be exactly one of {(1,0), (-1,0), (0,1), (0,-1)}"
            )
        return self


class ImportantObjectOrientationMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    front_local: CardinalAxisVector
    effective_front_side: Literal["top", "bottom", "left", "right"] | None = None
    axis_local: CardinalAxisVector


class OrientationMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_front_local: CardinalAxisVector
    cluster_effective_front_side: Literal["top", "bottom", "left", "right"] | None = (
        None
    )
    cluster_axis_local: CardinalAxisVector
    important_objects: dict[str, ImportantObjectOrientationMeta] = Field(
        default_factory=dict
    )

    @field_validator("important_objects")
    @classmethod
    def _validate_important_object_keys(
        cls, value: dict[str, ImportantObjectOrientationMeta]
    ) -> dict[str, ImportantObjectOrientationMeta]:
        out: dict[str, ImportantObjectOrientationMeta] = {}
        for key, meta in value.items():
            k = _strip_required_str(key, "important_objects key")
            out[k] = meta
        return out


class LocalPlacement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    x: int
    y: int
    rot: AllowedRotation

    @field_validator("id")
    @classmethod
    def _id_non_empty(cls, value: str) -> str:
        return _strip_required_str(value, "id")


class ClusterFootprintRect(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    x: int
    y: int
    w: int
    h: int

    @field_validator("id")
    @classmethod
    def _id_non_empty(cls, value: str) -> str:
        return _strip_required_str(value, "id")

    @field_validator("w", "h")
    @classmethod
    def _positive_dims(cls, value: int, info) -> int:
        if value <= 0:
            raise ValueError(f"{info.field_name} must be > 0")
        return value


class ClusterFootprintBBox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_x: int
    min_y: int
    max_x: int
    max_y: int

    @model_validator(mode="after")
    def _validate_bbox(self) -> "ClusterFootprintBBox":
        if self.max_x < self.min_x:
            raise ValueError("local_bbox.max_x must be >= min_x")
        if self.max_y < self.min_y:
            raise ValueError("local_bbox.max_y must be >= min_y")
        return self


class BBoxMinMax(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min: tuple[int, int]
    max: tuple[int, int]


class PolygonPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: int
    y: int


class AccessZone(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    x: int
    y: int
    w: int
    h: int
    kind: str = "front_clearance"

    @field_validator("id", "kind")
    @classmethod
    def _non_empty_str(cls, value: str, info) -> str:
        return _strip_required_str(value, info.field_name)


class AnchorSupportChainItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_id: str
    relative_to: str | None = None
    support_role: str = ""
    band_intent: str = ""
    orientation: str = ""

    @field_validator("object_id")
    @classmethod
    def _object_id_non_empty(cls, value: str) -> str:
        return _strip_required_str(value, "object_id")

    @field_validator("relative_to")
    @classmethod
    def _relative_to_non_empty(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required_str(value, "relative_to")

    @field_validator("support_role", "band_intent", "orientation")
    @classmethod
    def _optional_token_strip(cls, value: str) -> str:
        return value.strip()


class AnchorLayoutHints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    anchor_first_enabled: bool = False
    dominant_anchor_id: str | None = None
    placement_order: list[str] = Field(default_factory=list)
    support_chain: list[AnchorSupportChainItem] = Field(default_factory=list)
    anchor_chain_integrity_score: float = 0.0

    @field_validator("dominant_anchor_id")
    @classmethod
    def _dominant_anchor_id_non_empty(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required_str(value, "dominant_anchor_id")

    @field_validator("placement_order")
    @classmethod
    def _placement_order_non_empty(cls, value: list[str]) -> list[str]:
        out: list[str] = []
        for item in value:
            out.append(_strip_required_str(item, "placement_order"))
        return out

    @field_validator("anchor_chain_integrity_score")
    @classmethod
    def _integrity_score_range(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("anchor_chain_integrity_score must be between 0 and 1")
        return value


class WallContactEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_id: str
    edge: Literal["top", "bottom", "left", "right"]

    @field_validator("object_id")
    @classmethod
    def _object_id_non_empty(cls, value: str) -> str:
        return _strip_required_str(value, "object_id")


class LocalQuality(BaseModel):
    model_config = ConfigDict(extra="forbid")

    functional_score: float
    naturalness_score: float
    semantic_coherence_score: float
    compactness_score: float
    family_fidelity_score: float = 0.0
    awkwardness_penalty: float = 0.0
    solver_friendliness_score: float = 0.0
    split_cluster_penalty: float = 0.0
    awkward_grouping_penalty: float = 0.0
    fake_support_penalty: float = 0.0
    compaction_semantic_penalty: float = 0.0

    @field_validator(
        "functional_score",
        "naturalness_score",
        "semantic_coherence_score",
        "compactness_score",
        "family_fidelity_score",
        "awkwardness_penalty",
        "solver_friendliness_score",
        "split_cluster_penalty",
        "awkward_grouping_penalty",
        "fake_support_penalty",
        "compaction_semantic_penalty",
    )
    @classmethod
    def _score_range(cls, value: float, info) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{info.field_name} must be between 0 and 1")
        return value


class ClusterVariant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variant_id: str
    variant_family: str
    canonical_variant_family: str | None = None
    source_type: Literal["family_native", "fallback_generic"] = "family_native"
    family_fidelity: float = 1.0
    semantic_confidence: float = 1.0
    fallback_heavy: bool = False
    solver_friendliness: float = 0.0
    semantic_signature: list[str] = Field(default_factory=list)
    local_placements: list[LocalPlacement] = Field(default_factory=list)
    interaction_placements: list[ClusterFootprintRect] = Field(default_factory=list)
    tight_hull_polygon_mm: list[PolygonPoint] = Field(default_factory=list)
    tight_hull_polygons_mm: list[list[PolygonPoint]] = Field(default_factory=list)
    interaction_hull_polygon_mm: list[PolygonPoint] = Field(default_factory=list)
    interaction_hull_polygons_mm: list[list[PolygonPoint]] = Field(default_factory=list)
    family_contract_reasons: list[str] = Field(default_factory=list)
    anchor_layout_hints: AnchorLayoutHints | None = None
    anchor_chain_integrity_score: float = 0.0
    local_bbox_mm: BBoxMinMax
    wall_contact_edges: list[WallContactEdge] = Field(default_factory=list)
    required_access_zones: list[AccessZone] = Field(default_factory=list)
    orientation_meta: OrientationMeta | None = None
    local_quality: LocalQuality
    hard_valid: bool

    @field_validator("variant_id", "variant_family")
    @classmethod
    def _variant_str_non_empty(cls, value: str, info) -> str:
        return _strip_required_str(value, info.field_name)

    @field_validator("canonical_variant_family")
    @classmethod
    def _canonical_variant_family_non_empty(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required_str(value, "canonical_variant_family")

    @field_validator("semantic_signature", "family_contract_reasons")
    @classmethod
    def _semantic_signature_non_empty(cls, value: list[str]) -> list[str]:
        out: list[str] = []
        for item in value:
            out.append(_strip_required_str(item, "semantic_signature"))
        return out

    @field_validator("local_placements")
    @classmethod
    def _unique_variant_placement_ids(
        cls, value: list[LocalPlacement]
    ) -> list[LocalPlacement]:
        ids = [p.id for p in value]
        if len(ids) != len(set(ids)):
            raise ValueError("variant local_placements ids must be unique")
        return value

    @field_validator(
        "family_fidelity",
        "semantic_confidence",
        "solver_friendliness",
        "anchor_chain_integrity_score",
    )
    @classmethod
    def _variant_score_range(cls, value: float, info) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{info.field_name} must be between 0 and 1")
        return value


class ClusterFootprint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["union_of_rects"]
    rects: list[ClusterFootprintRect] = Field(default_factory=list)
    local_bbox: ClusterFootprintBBox
    tight_hull_polygon_mm: list[PolygonPoint] = Field(default_factory=list)
    tight_hull_polygons_mm: list[list[PolygonPoint]] = Field(default_factory=list)
    interaction_hull_polygon_mm: list[PolygonPoint] = Field(default_factory=list)
    interaction_hull_polygons_mm: list[list[PolygonPoint]] = Field(default_factory=list)
    variant_family: str | None = None
    family_fidelity: float | None = None

    @field_validator("rects")
    @classmethod
    def _unique_rect_ids(
        cls, value: list[ClusterFootprintRect]
    ) -> list[ClusterFootprintRect]:
        ids = [r.id for r in value]
        if len(ids) != len(set(ids)):
            raise ValueError("cluster_footprint.rects ids must be unique")
        return value

    @model_validator(mode="after")
    def _bbox_contains_rects(self) -> "ClusterFootprint":
        for r in self.rects:
            if r.x < self.local_bbox.min_x:
                raise ValueError(f"rect {r.id} lies left of local_bbox.min_x")
            if r.y < self.local_bbox.min_y:
                raise ValueError(f"rect {r.id} lies below local_bbox.min_y")
            if r.x + r.w > self.local_bbox.max_x:
                raise ValueError(f"rect {r.id} exceeds local_bbox.max_x")
            if r.y + r.h > self.local_bbox.max_y:
                raise ValueError(f"rect {r.id} exceeds local_bbox.max_y")
        return self


class LocalFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unit: Literal["mm"]
    grid_mm: int
    origin_note: str

    @field_validator("grid_mm")
    @classmethod
    def _grid_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("grid_mm must be > 0")
        return value

    @field_validator("origin_note")
    @classmethod
    def _origin_note_non_empty(cls, value: str) -> str:
        return _strip_required_str(value, "origin_note")


class ClusterComposerOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: ComposerStatus
    cluster_id: str
    local_frame: LocalFrame | None = None
    local_placements: list[LocalPlacement] = Field(default_factory=list)
    cluster_footprint: ClusterFootprint | None = None
    orientation_meta: OrientationMeta | None = None
    variant_bundle: list[ClusterVariant] = Field(default_factory=list)
    canonical_variant_id: str | None = None
    canonical_variant_family: str | None = None
    variant_summary: dict[str, object] = Field(default_factory=dict)
    family_coverage: dict[str, object] = Field(default_factory=dict)
    tight_hull_polygon_mm: list[PolygonPoint] = Field(default_factory=list)
    tight_hull_polygons_mm: list[list[PolygonPoint]] = Field(default_factory=list)
    interaction_hull_polygon_mm: list[PolygonPoint] = Field(default_factory=list)
    interaction_hull_polygons_mm: list[list[PolygonPoint]] = Field(default_factory=list)
    variant_family: str | None = None
    source_type: Literal["family_native", "fallback_generic"] | None = None
    family_fidelity: float | None = None
    semantic_confidence: float | None = None
    fallback_heavy: bool | None = None
    solver_friendliness: float | None = None
    family_contract_reasons: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)

    @field_validator("cluster_id")
    @classmethod
    def _cluster_id_non_empty(cls, value: str) -> str:
        return _strip_required_str(value, "cluster_id")

    @field_validator("notes", "missing", "conflicts", "family_contract_reasons")
    @classmethod
    def _string_lists_non_empty(cls, value: list[str], info) -> list[str]:
        out: list[str] = []
        for item in value:
            out.append(_strip_required_str(item, info.field_name))
        return out

    @field_validator("local_placements")
    @classmethod
    def _unique_local_placement_ids(
        cls, value: list[LocalPlacement]
    ) -> list[LocalPlacement]:
        ids = [p.id for p in value]
        if len(ids) != len(set(ids)):
            raise ValueError("local_placements ids must be unique")
        return value

    @model_validator(mode="after")
    def _require_fields_for_ok(self) -> "ClusterComposerOutput":
        if self.status == "OK":
            if self.local_frame is None:
                raise ValueError("local_frame is required when status=OK")
            if self.cluster_footprint is None:
                raise ValueError("cluster_footprint is required when status=OK")
            if self.orientation_meta is None:
                raise ValueError("orientation_meta is required when status=OK")
            if not self.local_placements:
                raise ValueError("local_placements required when status=OK")

            placement_ids = {p.id for p in self.local_placements}
            rect_ids = {r.id for r in self.cluster_footprint.rects}

            if placement_ids != rect_ids:
                raise ValueError(
                    "For status=OK, cluster_footprint.rect ids must match "
                    "local_placements ids exactly"
                )

            important_object_ids = set(self.orientation_meta.important_objects.keys())
            if not important_object_ids.issubset(placement_ids):
                extra = sorted(important_object_ids - placement_ids)
                raise ValueError(
                    f"orientation_meta.important_objects contains unknown ids: {extra}"
                )

        return self

    @model_validator(mode="after")
    def _missing_only_for_need_info(self) -> "ClusterComposerOutput":
        if self.status != "NEED_INFO" and self.missing:
            raise ValueError("missing must be empty unless status=NEED_INFO")
        return self
