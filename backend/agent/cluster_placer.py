from __future__ import annotations

import json
import logging
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Protocol

# type: ignore
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from cluster_placer.tools_v2 import (  # type: ignore
    CompileAcceptedPhase2Proposal,
    DiagnosePhase2Seed,
    EnumeratePhase2RepairMoves,
    EvaluatePhase2Proposal,
)
from layout.grid_policy import normalize_layout_grid_mm, normalize_room_context_grid

try:
    from clients.llm_client import get_llm_client  # type: ignore
except Exception:  # pragma: no cover
    get_llm_client = None  # type: ignore

try:
    from config.llm_config import TextLLMConfig  # type: ignore
except Exception:  # pragma: no cover
    TextLLMConfig = None  # type: ignore

try:
    from prompt.cluster_placer import (
        MACRO_CLUSTER_PLACER_PROMPT as DEFAULT_PHASE2_PLACER_PROMPT,  # type: ignore
    )
except Exception:  # pragma: no cover
    DEFAULT_PHASE2_PLACER_PROMPT = "You are a phase-2 placer. Return JSON only."

logger = logging.getLogger(__name__)

_DEFAULT_PLACER_FALLBACK_MODELS = (
    "gemma-3-27b-it",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
)
_PLACER_MODEL_ROTATION_KEYS = (
    "placer",
    "placer_fallback_primary",
    "placer_fallback_helper",
)
_placer_model_cursor = 0

_PLANNER_OBJECT_AXIS_INTENTS = {
    "same_direction_as_anchor",
    "same_view_side_as_primary_pair",
    "align_with_anchor_axis",
    "not_behind_anchor_view",
    "in_front_of_anchor",
}


AllowedRotation = Literal[0, 90, 180, 270]
RepairStatus = Literal["REPAIRED", "NO_IMPROVEMENT", "NEED_INFO"]
AllowedObjectOp = Literal[
    "rotate_object",
    "mirror_object",
    "nudge_object",
    "swap_objects",
    "set_anchor",
    "set_front_override",
]


class LLMCallable(Protocol):
    def __call__(
        self, *, system_prompt: str, user_payload_json: str
    ) -> str | Dict[str, Any]: ...


class RetryablePlacerModelError(RuntimeError):
    def __init__(
        self,
        *,
        failed_model: str,
        next_model: str,
        retry_delay_s: float | None = None,
        detail: str,
    ) -> None:
        self.failed_model = failed_model
        self.next_model = next_model
        self.retry_delay_s = retry_delay_s
        super().__init__(detail)


# -----------------------------------------------------------------------------
# Output schema
# -----------------------------------------------------------------------------
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
    op: AllowedObjectOp
    params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("cluster_id")
    @classmethod
    def _cluster_id_non_empty(cls, value: str) -> str:
        return _strip_required_str(value, "cluster_id")

    @field_validator("object_id")
    @classmethod
    def _object_id_non_empty(cls, value: str) -> str:
        return _strip_required_str(value, "object_id")


class Phase2RepairOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: RepairStatus
    cluster_transforms: List[ClusterTransform] = Field(default_factory=list)
    selected_variants: List[ClusterVariantSelection] = Field(default_factory=list)
    object_repairs: List[ObjectRepair] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)

    @field_validator("notes")
    @classmethod
    def _notes_non_empty(cls, value: List[str]) -> List[str]:
        out: List[str] = []
        for item in value:
            out.append(_strip_required_str(item, "notes item"))
        return out

    @field_validator("cluster_transforms")
    @classmethod
    def _unique_cluster_transform_ids(
        cls, value: List[ClusterTransform]
    ) -> List[ClusterTransform]:
        ids = [item.cluster_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("cluster_transforms cluster_id must be unique")
        return value

    @field_validator("selected_variants")
    @classmethod
    def _unique_selected_variant_ids(
        cls, value: List[ClusterVariantSelection]
    ) -> List[ClusterVariantSelection]:
        ids = [item.cluster_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("selected_variants cluster_id must be unique")
        return value


# -----------------------------------------------------------------------------
# Payload helpers
# -----------------------------------------------------------------------------
def build_phase2_payload(
    room_01: Dict[str, Any],
    clusters_04: Dict[str, Any],
    outlines_05: Dict[str, Any],
    relation_05b: Dict[str, Any],
    placer_06: Dict[str, Any],
    cluster_constraints: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    from copy import deepcopy

    # -------------------------------------------------------------------------
    # local helpers
    # -------------------------------------------------------------------------
    def _normalize_vec(vec: tuple[float, float] | None) -> tuple[float, float] | None:
        if vec is None:
            return None
        x, y = float(vec[0]), float(vec[1])
        norm = (x * x + y * y) ** 0.5
        if norm <= 1e-9:
            return None
        return (x / norm, y / norm)

    def _parse_vec2(value: Any) -> tuple[float, float] | None:
        if isinstance(value, dict):
            dx = value.get("dx")
            dy = value.get("dy")
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            dx, dy = value[0], value[1]
        else:
            return None
        try:
            return _normalize_vec((float(dx), float(dy)))
        except Exception:
            return None

    def _rotate_point_ccw_90s(x: float, y: float, rot: int) -> tuple[float, float]:
        r = int(rot) % 360
        if r == 0:
            return x, y
        if r == 90:
            return -y, x
        if r == 180:
            return -x, -y
        if r == 270:
            return y, -x
        raise ValueError(f"Unsupported rot={rot}")

    def _rotate_vec_ccw_90s(
        vec: tuple[float, float] | None, rot: int
    ) -> tuple[float, float] | None:
        if vec is None:
            return None
        return _normalize_vec(
            _rotate_point_ccw_90s(float(vec[0]), float(vec[1]), int(rot))
        )

    def _transform_rect_world(
        rect: Dict[str, Any], tx: int, ty: int, rot: int
    ) -> Dict[str, Any]:
        x = float(rect.get("x", 0))
        y = float(rect.get("y", 0))
        w = float(rect.get("w", 0))
        h = float(rect.get("h", 0))
        corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        world = [_rotate_point_ccw_90s(px, py, rot) for px, py in corners]
        world = [(px + tx, py + ty) for px, py in world]
        xs = [p[0] for p in world]
        ys = [p[1] for p in world]
        return {
            "polygon_ccw": [
                {"x": int(round(px)), "y": int(round(py))} for px, py in world
            ],
            "bbox": {
                "min_x": int(round(min(xs))),
                "min_y": int(round(min(ys))),
                "max_x": int(round(max(xs))),
                "max_y": int(round(max(ys))),
            },
            "world_center": {
                "x": int(round(sum(xs) / 4.0)),
                "y": int(round(sum(ys) / 4.0)),
            },
        }

    def _extract_room_context() -> Dict[str, Any]:
        seed = (placer_06 or {}).get("placer_seed") or {}
        room_model_used = seed.get("room_model_used")
        grid_mm = seed.get("grid_mm")
        normalized_grid = normalize_layout_grid_mm(grid_mm)
        if isinstance(room_model_used, dict) and room_model_used:
            return normalize_room_context_grid(
                {
                    "room_model_used": deepcopy(room_model_used),
                    "grid_mm": normalized_grid,
                }
            )
        return normalize_room_context_grid(
            {"room_model_used": deepcopy(room_01), "grid_mm": normalized_grid}
        )

    def _extract_phase_control() -> Dict[str, Any]:
        seed = (placer_06 or {}).get("placer_seed") or {}
        solver_debug = (placer_06 or {}).get("solver_debug") or {}
        best_verify = solver_debug.get("best_verify") or {}

        explicit_ready = seed.get("ready")
        if isinstance(explicit_ready, bool):
            ready = explicit_ready
        else:
            ready = bool(best_verify)
        needed = bool(seed.get("needed", True))
        seed_kind = str(seed.get("seed_kind") or "")
        if not seed_kind:
            if best_verify.get("hard_valid") and best_verify.get("complete"):
                seed_kind = "hard_valid_complete"
            elif best_verify.get("hard_valid"):
                seed_kind = "hard_valid_partial"
            else:
                seed_kind = "none"

        return {
            "phase": "phase2_repair",
            "repair_phase": "macro_layout",
            "ready": ready,
            "needed": needed,
            "seed_kind": seed_kind,
            "solver_status": str((placer_06 or {}).get("status") or "UNKNOWN"),
        }

    def _extract_seed_layout() -> Dict[str, Any]:
        seed = (placer_06 or {}).get("placer_seed") or {}
        if isinstance(seed.get("seed_layout"), dict):
            data = seed["seed_layout"]
            return {
                "cluster_transforms": deepcopy(data.get("cluster_transforms") or []),
                "selected_variants": deepcopy(data.get("selected_variants") or []),
            }

        solver_debug = (placer_06 or {}).get("solver_debug") or {}
        return {
            "cluster_transforms": deepcopy(
                solver_debug.get("best_transforms")
                or (placer_06 or {}).get("cluster_transforms")
                or []
            ),
            "selected_variants": deepcopy(
                solver_debug.get("best_selected_variants")
                or (placer_06 or {}).get("selected_variants")
                or []
            ),
        }

    def _extract_goals() -> Dict[str, Any]:
        seed = (placer_06 or {}).get("placer_seed") or {}
        relation_plan_used = seed.get("relation_plan_used")
        cluster_constraints_used = seed.get("cluster_constraints_used")

        if not isinstance(relation_plan_used, dict) or not relation_plan_used:
            relation_plan_used = deepcopy(relation_05b)
        if not isinstance(cluster_constraints_used, dict):
            cluster_constraints_used = (
                deepcopy(cluster_constraints)
                if isinstance(cluster_constraints, dict)
                else None
            )

        return {
            "relation_plan_used": relation_plan_used,
            "cluster_constraints_used": cluster_constraints_used,
        }

    def _extract_repair_debug() -> Dict[str, Any]:
        seed = (placer_06 or {}).get("placer_seed") or {}
        solver_debug = (placer_06 or {}).get("solver_debug") or {}
        seed_verify = deepcopy(
            seed.get("seed_verify") or solver_debug.get("best_verify") or {}
        )
        return {
            "seed_metrics": deepcopy(seed.get("seed_metrics") or {}),
            "seed_verify": seed_verify,
            "repair_targets": deepcopy(
                seed.get("repair_targets") or seed_verify.get("repair_guidance") or {}
            ),
            "candidate_counts": deepcopy(
                seed.get("candidate_counts")
                or solver_debug.get("candidate_counts")
                or {}
            ),
            "quality_gate": deepcopy(seed_verify.get("quality_gate") or {}),
        }

    def _extract_edit_contract() -> Dict[str, Any]:
        seed = (placer_06 or {}).get("placer_seed") or {}
        contract = seed.get("edit_contract")
        if isinstance(contract, dict) and contract:
            return deepcopy(contract)
        return {
            "must_preserve_hard_valid": True,
            "hard_constraints_must_hold": [
                "out_of_bounds",
                "overlap",
                "door_swing",
            ],
            "primary_soft_targets": [
                "critical_orientation_penalty_mm",
                "focal_pair_penalty_mm",
                "max_critical_item_penalty_mm",
            ],
            "allowed_edit_levels": [
                "cluster_variant",
                "cluster_pose",
                "object_pose",
                "object_swap",
                "object_nudge",
            ],
        }

    def _index_clusters_by_id(
        clusters_json: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for row in (clusters_json or {}).get("clusters") or []:
            if isinstance(row, dict) and isinstance(row.get("cluster_id"), str):
                out[row["cluster_id"]] = deepcopy(row)
        return out

    def _index_outlines_by_id() -> Dict[str, Dict[str, Any]]:
        seed = (placer_06 or {}).get("placer_seed") or {}
        mat = seed.get("materialized_clusters_outlines")
        if isinstance(mat, dict) and mat:
            return deepcopy(mat)

        if isinstance(outlines_05, dict):
            # already keyed by cluster_id
            direct = {}
            for cid, row in outlines_05.items():
                if isinstance(cid, str) and isinstance(row, dict):
                    direct[cid] = deepcopy(row)
            if direct:
                return direct
            # or wrapped in {"clusters": [...]}
            if isinstance(outlines_05.get("clusters"), list):
                out = {}
                for row in outlines_05["clusters"]:
                    if isinstance(row, dict) and isinstance(row.get("cluster_id"), str):
                        out[row["cluster_id"]] = deepcopy(row)
                return out
        return {}

    def _build_cluster_cards(seed_layout: Dict[str, Any]) -> List[Dict[str, Any]]:
        merged = _index_clusters_by_id(clusters_04)
        outlines = _index_outlines_by_id()

        active_ids = {
            str(x.get("cluster_id"))
            for x in seed_layout.get("cluster_transforms") or []
            if isinstance(x, dict) and isinstance(x.get("cluster_id"), str)
        }
        if not active_ids:
            active_ids = set(merged.keys()) | set(outlines.keys())

        def _build_generic_variant_rows() -> Dict[str, List[Dict[str, Any]]]:
            if not outlines or not active_ids:
                return {}
            try:
                from cluster_placer.tools import BuildGenericClusterVariants

                info = BuildGenericClusterVariants(
                    clusters_outlines=outlines,
                    cluster_ids=sorted(active_ids),
                    max_variants_per_cluster=8,
                    include_variant_payloads=True,
                )
            except Exception:
                return {}
            if info.get("result") != "OK":
                return {}

            out: Dict[str, List[Dict[str, Any]]] = {}
            for row in info.get("clusters") or []:
                cid = str(row.get("cluster_id") or "")
                if not cid:
                    continue
                variants: List[Dict[str, Any]] = []
                for variant in row.get("variants") or []:
                    if not isinstance(variant, dict):
                        continue
                    if not isinstance(variant.get("cluster_payload"), dict):
                        continue
                    variants.append(deepcopy(variant))
                if variants:
                    out[cid] = variants
            return out

        generic_variant_rows_by_cluster = _build_generic_variant_rows()

        def _merge_variant_card_geometry(
            base_card: Dict[str, Any],
            variant_row: Dict[str, Any],
        ) -> Dict[str, Any] | None:
            if isinstance(variant_row.get("card"), dict):
                return deepcopy(variant_row["card"])

            def _point_outline(points: Any) -> List[Dict[str, int]]:
                out: List[Dict[str, int]] = []
                if not isinstance(points, list):
                    return out
                for point in points:
                    if not isinstance(point, dict):
                        continue
                    try:
                        out.append(
                            {
                                "x": int(round(float(point.get("x")))),
                                "y": int(round(float(point.get("y")))),
                            }
                        )
                    except Exception:
                        continue
                return out if len(out) >= 3 else []

            def _polygon_outlines(polygons: Any) -> List[List[Dict[str, int]]]:
                if not isinstance(polygons, list):
                    return []
                out: List[List[Dict[str, int]]] = []
                for polygon in polygons:
                    outline = _point_outline(polygon)
                    if outline:
                        out.append(outline)
                return out

            def _bbox_from_variant(
                variant: Dict[str, Any],
                rects: List[Dict[str, Any]],
                outlines: List[List[Dict[str, int]]],
            ) -> Dict[str, int]:
                raw_bbox = variant.get("local_bbox_mm")
                if isinstance(raw_bbox, dict):
                    try:
                        if isinstance(raw_bbox.get("min"), list) and isinstance(
                            raw_bbox.get("max"), list
                        ):
                            min_xy = raw_bbox["min"]
                            max_xy = raw_bbox["max"]
                            return {
                                "min_x": int(round(float(min_xy[0]))),
                                "min_y": int(round(float(min_xy[1]))),
                                "max_x": int(round(float(max_xy[0]))),
                                "max_y": int(round(float(max_xy[1]))),
                            }
                        return {
                            "min_x": int(round(float(raw_bbox.get("min_x")))),
                            "min_y": int(round(float(raw_bbox.get("min_y")))),
                            "max_x": int(round(float(raw_bbox.get("max_x")))),
                            "max_y": int(round(float(raw_bbox.get("max_y")))),
                        }
                    except Exception:
                        pass

                xs: List[int] = []
                ys: List[int] = []
                for rect in rects:
                    try:
                        x = int(round(float(rect.get("x", 0))))
                        y = int(round(float(rect.get("y", 0))))
                        w = int(round(float(rect.get("w", 0))))
                        h = int(round(float(rect.get("h", 0))))
                    except Exception:
                        continue
                    xs.extend([x, x + max(w, 0)])
                    ys.extend([y, y + max(h, 0)])
                for outline in outlines:
                    for point in outline:
                        xs.append(point["x"])
                        ys.append(point["y"])
                if xs and ys:
                    return {
                        "min_x": min(xs),
                        "min_y": min(ys),
                        "max_x": max(xs),
                        "max_y": max(ys),
                    }
                return {"min_x": 0, "min_y": 0, "max_x": 0, "max_y": 0}

            def _rects_from_composer_variant(
                variant: Dict[str, Any],
            ) -> List[Dict[str, Any]]:
                placement_ids = {
                    str(row.get("id") or "")
                    for row in variant.get("local_placements") or []
                    if isinstance(row, dict) and str(row.get("id") or "").strip()
                }
                rects: List[Dict[str, Any]] = []
                for row in variant.get("interaction_placements") or []:
                    if not isinstance(row, dict):
                        continue
                    oid = str(row.get("id") or "").strip()
                    if not oid or oid.startswith("access:"):
                        continue
                    if placement_ids and oid not in placement_ids:
                        continue
                    try:
                        w = int(round(float(row.get("w", 0))))
                        h = int(round(float(row.get("h", 0))))
                        if w <= 0 or h <= 0:
                            continue
                        rects.append(
                            {
                                "id": oid,
                                "x": int(round(float(row.get("x", 0)))),
                                "y": int(round(float(row.get("y", 0)))),
                                "w": w,
                                "h": h,
                            }
                        )
                    except Exception:
                        continue
                if rects:
                    return rects

                base_rect_sizes = {
                    str(rect.get("id")): rect
                    for rect in (
                        (base_card.get("cluster_footprint") or {}).get("rects") or []
                    )
                    if isinstance(rect, dict) and isinstance(rect.get("id"), str)
                }
                for row in variant.get("local_placements") or []:
                    if not isinstance(row, dict):
                        continue
                    oid = str(row.get("id") or "").strip()
                    if not oid:
                        continue
                    size = base_card.get("object_sizes", {}).get(oid) or {}
                    source_rect = base_rect_sizes.get(oid) or {}
                    try:
                        w = int(round(float(size.get("w_mm", source_rect.get("w", 0)))))
                        h = int(round(float(size.get("h_mm", source_rect.get("h", 0)))))
                        if int(row.get("rot", 0)) % 180 == 90:
                            w, h = h, w
                        if w <= 0 or h <= 0:
                            continue
                        rects.append(
                            {
                                "id": oid,
                                "x": int(round(float(row.get("x", 0)))),
                                "y": int(round(float(row.get("y", 0)))),
                                "w": w,
                                "h": h,
                            }
                        )
                    except Exception:
                        continue
                return rects

            def _card_from_composer_variant(
                variant: Dict[str, Any],
            ) -> Dict[str, Any] | None:
                local_placements = variant.get("local_placements")
                if not isinstance(local_placements, list):
                    return None
                rects = _rects_from_composer_variant(variant)
                if not rects:
                    return None

                outlines = _polygon_outlines(
                    variant.get("interaction_hull_polygons_mm")
                ) or _polygon_outlines(variant.get("tight_hull_polygons_mm"))
                outline = _point_outline(
                    variant.get("interaction_hull_polygon_mm")
                    or variant.get("tight_hull_polygon_mm")
                )
                if outline and not outlines:
                    outlines = [outline]
                card = deepcopy(base_card)
                card["local_placements"] = deepcopy(local_placements)
                if isinstance(variant.get("orientation_meta"), dict):
                    card["orientation_meta"] = deepcopy(
                        variant.get("orientation_meta") or {}
                    )
                card["cluster_footprint"] = {
                    "type": "union_of_rects",
                    "rects": rects,
                    "local_bbox": _bbox_from_variant(variant, rects, outlines),
                    "outline_polygons_ccw": outlines,
                }
                card["composer_variant"] = {
                    "variant_id": str(variant.get("variant_id") or ""),
                    "variant_family": str(variant.get("variant_family") or ""),
                    "semantic_signature": deepcopy(
                        variant.get("semantic_signature") or []
                    ),
                    "wall_contact_edges": deepcopy(
                        variant.get("wall_contact_edges") or []
                    ),
                    "required_access_zones": deepcopy(
                        variant.get("required_access_zones") or []
                    ),
                    "local_quality": deepcopy(variant.get("local_quality") or {}),
                }
                card.pop("available_variants", None)
                card.pop("variants", None)
                return card

            if isinstance(variant_row.get("variant_bundle"), list):
                return None
            if isinstance(variant_row.get("local_placements"), list):
                card = _card_from_composer_variant(variant_row)
                if card is not None:
                    return card

            cluster_payload = variant_row.get("cluster_payload")
            if not isinstance(cluster_payload, dict):
                if isinstance(variant_row.get("cluster_footprint"), dict):
                    cluster_payload = variant_row
                else:
                    return None

            card = deepcopy(base_card)
            for key in (
                "cluster_id",
                "local_frame",
                "local_placements",
                "cluster_footprint",
                "orientation_meta",
                "notes",
                "missing",
            ):
                value = cluster_payload.get(key)
                if value is not None:
                    card[key] = deepcopy(value)
            card.pop("available_variants", None)
            card.pop("variants", None)
            return card

        def _normalize_available_variants(
            *,
            base_card: Dict[str, Any],
            explicit_rows: List[Dict[str, Any]],
            fallback_rows: List[Dict[str, Any]],
        ) -> List[Dict[str, Any]]:
            rows = explicit_rows or fallback_rows
            out: List[Dict[str, Any]] = []
            seen_ids: set[str] = set()
            for row in rows:
                if not isinstance(row, dict):
                    continue
                variant_id = str(row.get("variant_id") or "").strip()
                if not variant_id or variant_id in seen_ids:
                    continue
                card = _merge_variant_card_geometry(base_card, row)
                if not isinstance(card, dict):
                    continue
                seen_ids.add(variant_id)
                out.append(
                    {
                        "variant_id": variant_id,
                        "family": str(
                            row.get("family") or row.get("variant_family") or ""
                        ),
                        "priority": float(row.get("priority") or 0.0),
                        "ops": deepcopy(row.get("ops") or []),
                        "signature": row.get("signature"),
                        "card": card,
                    }
                )
            return out

        cards: List[Dict[str, Any]] = []
        for cid in sorted(active_ids):
            merged_row = merged.get(cid, {})
            outline_row = outlines.get(cid, {})
            if not merged_row and not outline_row:
                continue

            decisions = (
                merged_row.get("decisions")
                if isinstance(merged_row.get("decisions"), list)
                else []
            )

            object_sizes: Dict[str, Dict[str, int]] = {}
            for d in decisions:
                if not isinstance(d, dict):
                    continue
                oid = d.get("id")
                dims = d.get("rep_dims_m")
                if isinstance(oid, str) and isinstance(dims, dict):
                    try:
                        object_sizes[oid] = {
                            "w_mm": int(round(float(dims.get("w", 0)) * 1000.0)),
                            "h_mm": int(round(float(dims.get("h", 0)) * 1000.0)),
                        }
                    except Exception:
                        pass

            explicit_variant_rows: List[Dict[str, Any]] = []
            for key in ("available_variants", "variants", "variant_bundle"):
                src = None
                if isinstance(outline_row.get(key), list):
                    src = outline_row.get(key)
                elif isinstance(merged_row.get(key), list):
                    src = merged_row.get(key)
                if isinstance(src, list):
                    explicit_variant_rows = deepcopy(src)
                    break

            base_card = {
                "cluster_id": cid,
                "tag": merged_row.get("tag"),
                "members": deepcopy(merged_row.get("members") or []),
                "anchors": deepcopy(merged_row.get("anchors") or []),
                "hard_constraints": deepcopy(merged_row.get("hard_constraints") or []),
                "soft_constraints": deepcopy(merged_row.get("soft_constraints") or []),
                "cluster_rules": deepcopy(merged_row.get("cluster_rules") or {}),
                "decisions": deepcopy(decisions),
                "object_sizes": object_sizes,
                "local_placements": deepcopy(outline_row.get("local_placements") or []),
                "cluster_footprint": deepcopy(
                    outline_row.get("cluster_footprint") or {}
                ),
                "orientation_meta": deepcopy(outline_row.get("orientation_meta") or {}),
                "source": {
                    "merged_from_04": bool(merged_row),
                    "outline_from_05_or_solver": bool(outline_row),
                },
            }
            available_variants = _normalize_available_variants(
                base_card=base_card,
                explicit_rows=explicit_variant_rows,
                fallback_rows=generic_variant_rows_by_cluster.get(cid) or [],
            )
            base_card["available_variants"] = available_variants

            cards.append(base_card)
        return cards

    def _top_object_debug(
        seed_verify: Dict[str, Any], max_items: int = 24
    ) -> Dict[tuple[str, str], Dict[str, Any]]:
        out: Dict[tuple[str, str], Dict[str, Any]] = {}
        quality = (seed_verify or {}).get("quality") or {}
        dbg = (quality.get("orientation_debug") or []) + (
            quality.get("critical_orientation_debug") or []
        )
        for item in dbg:
            if not isinstance(item, dict):
                continue
            cid = item.get("cluster_id")
            oid = item.get("object_id")
            if not isinstance(cid, str) or not isinstance(oid, str):
                continue
            key = (cid, oid)
            prev = out.get(key)
            pen = int(item.get("penalty_mm") or 0)
            if prev is None or pen > int(prev.get("penalty_mm") or 0):
                out[key] = {
                    "intent": item.get("intent"),
                    "penalty_mm": pen,
                    "front_clear_mm": item.get("front_clear_mm"),
                    "back_clear_mm": item.get("back_clear_mm"),
                    "best_clear_mm": item.get("best_clear_mm"),
                    "dot": item.get("dot"),
                }
        ranked = sorted(out.items(), key=lambda kv: -int(kv[1].get("penalty_mm") or 0))[
            :max_items
        ]
        return {k: v for k, v in ranked}

    def _build_objects_world(
        cluster_cards: List[Dict[str, Any]],
        seed_layout: Dict[str, Any],
        seed_verify: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        tf_by_id = {
            str(x.get("cluster_id")): x
            for x in seed_layout.get("cluster_transforms") or []
            if isinstance(x, dict) and isinstance(x.get("cluster_id"), str)
        }
        object_dbg = _top_object_debug(seed_verify)
        out: List[Dict[str, Any]] = []

        for card in cluster_cards:
            cid = str(card.get("cluster_id"))
            tf = tf_by_id.get(cid)
            if not isinstance(tf, dict):
                continue
            tx = int(tf.get("x") or 0)
            ty = int(tf.get("y") or 0)
            rot = int(tf.get("rot") or 0) % 360

            fp = card.get("cluster_footprint") or {}
            rects = fp.get("rects") or []
            important = (card.get("orientation_meta") or {}).get(
                "important_objects"
            ) or {}
            cluster_front = _parse_vec2(
                (card.get("orientation_meta") or {}).get("cluster_front_local")
            )

            for rect in rects:
                if not isinstance(rect, dict) or not isinstance(rect.get("id"), str):
                    continue
                oid = rect["id"]
                world_geom = _transform_rect_world(rect, tx, ty, rot)

                obj_meta = important.get(oid) if isinstance(important, dict) else {}
                front_local = (
                    _parse_vec2((obj_meta or {}).get("front_local")) or cluster_front
                )
                axis_local = _parse_vec2((obj_meta or {}).get("axis_local"))
                front_world = _rotate_vec_ccw_90s(front_local, rot)
                axis_world = _rotate_vec_ccw_90s(axis_local, rot)

                w = int(round(float(rect.get("w", 0))))
                h = int(round(float(rect.get("h", 0))))
                if front_local is None:
                    size_along_access = max(w, h)
                else:
                    dx, dy = abs(front_local[0]), abs(front_local[1])
                    size_along_access = w if dx >= dy else h
                required_clearance = int(round(0.25 * size_along_access))

                dbg = object_dbg.get((cid, oid), {})
                out.append(
                    {
                        "cluster_id": cid,
                        "object_id": oid,
                        "cluster_rot": rot,
                        "local_rect": {
                            "x": int(round(float(rect.get("x", 0)))),
                            "y": int(round(float(rect.get("y", 0)))),
                            "w": w,
                            "h": h,
                        },
                        "polygon_ccw": world_geom["polygon_ccw"],
                        "bbox": world_geom["bbox"],
                        "world_center": world_geom["world_center"],
                        "size_mm": {"w": w, "h": h},
                        "front_world": None
                        if front_world is None
                        else {
                            "dx": round(front_world[0], 3),
                            "dy": round(front_world[1], 3),
                        },
                        "axis_world": None
                        if axis_world is None
                        else {
                            "dx": round(axis_world[0], 3),
                            "dy": round(axis_world[1], 3),
                        },
                        "required_clearance_mm": required_clearance,
                        "current_front_clear_mm": dbg.get("front_clear_mm"),
                        "current_back_clear_mm": dbg.get("back_clear_mm"),
                        "best_clear_mm": dbg.get("best_clear_mm"),
                        "dominant_debug_intent": dbg.get("intent"),
                        "dominant_penalty_mm": dbg.get("penalty_mm"),
                        "dominant_dot": dbg.get("dot"),
                    }
                )

        out.sort(
            key=lambda x: (
                x["cluster_id"],
                -int(x.get("dominant_penalty_mm") or 0),
                x["object_id"],
            )
        )
        return out

    # -------------------------------------------------------------------------
    # build payload
    # -------------------------------------------------------------------------
    room_context = _extract_room_context()
    phase_control = _extract_phase_control()
    seed_layout = _extract_seed_layout()
    goals = _extract_goals()
    repair_debug = _extract_repair_debug()
    edit_contract = _extract_edit_contract()
    cluster_cards = _build_cluster_cards(seed_layout)
    objects_world = _build_objects_world(
        cluster_cards,
        seed_layout,
        repair_debug.get("seed_verify") or {},
    )

    return {
        "phase_control": phase_control,
        "room_context": room_context,
        "seed_layout": seed_layout,
        "cluster_cards": cluster_cards,
        "goals": goals,
        "repair_debug": repair_debug,
        "edit_contract": edit_contract,
        "objects_world": objects_world,
    }


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str | Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _cluster_cards_as_map(cluster_cards: Any) -> Dict[str, Dict[str, Any]]:
    if isinstance(cluster_cards, dict):
        return {
            str(k): deepcopy(v)
            for k, v in cluster_cards.items()
            if isinstance(k, str) and isinstance(v, dict)
        }
    if isinstance(cluster_cards, list):
        out: Dict[str, Dict[str, Any]] = {}
        for row in cluster_cards:
            if isinstance(row, dict) and isinstance(row.get("cluster_id"), str):
                out[row["cluster_id"]] = deepcopy(row)
        return out
    return {}


def _seed_transform_map(seed_layout: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for item in seed_layout.get("cluster_transforms") or []:
        if isinstance(item, dict) and isinstance(item.get("cluster_id"), str):
            out[item["cluster_id"]] = deepcopy(item)
    return out


def _seed_variant_map(seed_layout: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for item in seed_layout.get("selected_variants") or []:
        if isinstance(item, dict) and isinstance(item.get("cluster_id"), str):
            out[item["cluster_id"]] = deepcopy(item)
    return out


def _allowed_object_ids_by_cluster(payload: Dict[str, Any]) -> Dict[str, set[str]]:
    out: Dict[str, set[str]] = {}
    for row in payload.get("objects_world") or []:
        if not isinstance(row, dict):
            continue
        cid = row.get("cluster_id")
        oid = row.get("object_id")
        if isinstance(cid, str) and isinstance(oid, str):
            out.setdefault(cid, set()).add(oid)
    return out


def _allowed_variant_ids_by_cluster(payload: Dict[str, Any]) -> Dict[str, set[str]]:
    out: Dict[str, set[str]] = {}
    cards = _cluster_cards_as_map(payload.get("cluster_cards") or {})
    current_seed = _seed_variant_map(payload.get("seed_layout") or {})
    for cid, card in cards.items():
        allowed: set[str] = set()
        variants = card.get("available_variants") or card.get("variants") or []
        if isinstance(variants, list):
            for item in variants:
                if isinstance(item, dict) and isinstance(item.get("variant_id"), str):
                    allowed.add(item["variant_id"])
                elif isinstance(item, str):
                    allowed.add(item)
        if not allowed and cid in current_seed:
            allowed.add(str(current_seed[cid].get("variant_id") or ""))
        if allowed:
            out[cid] = allowed
    return out


def _allowed_cluster_rotations(payload: Dict[str, Any]) -> Dict[str, set[int]]:
    out: Dict[str, set[int]] = {}
    cards = _cluster_cards_as_map(payload.get("cluster_cards") or {})
    for cid, card in cards.items():
        vals: set[int] = set()
        rules = card.get("cluster_rules") or {}
        rots = rules.get("allowed_rotations") or []
        for r in rots:
            try:
                rv = int(r)
            except Exception:
                continue
            if rv in {0, 90, 180, 270}:
                vals.add(rv)
        if not vals:
            vals = {0, 90, 180, 270}
        out[cid] = vals
    return out


def _allowed_object_ops(payload: Dict[str, Any]) -> set[str]:
    contract = payload.get("edit_contract") or {}
    levels = contract.get("allowed_edit_levels") or {}
    allowed = set()
    if isinstance(levels, list):
        levels = {str(x): True for x in levels}
    if levels.get("object_pose", False) or levels.get("object_rotation", True):
        allowed.update({"rotate_object", "mirror_object", "set_front_override"})
    if levels.get("object_nudge", False) or levels.get("object_local_nudge", True):
        allowed.add("nudge_object")
    if levels.get("object_swap", False) or levels.get(
        "object_swap_within_cluster", True
    ):
        allowed.add("swap_objects")
    if levels.get("object_anchor", False):
        allowed.add("set_anchor")
    return allowed


def _repair_phase(payload: Dict[str, Any]) -> str:
    phase_control = payload.get("phase_control") or {}
    value = str(phase_control.get("repair_phase") or "").strip().lower()
    if value == "object_refine":
        return "object_refine"
    return "macro_layout"


def _enumerated_object_move_signatures(
    payload: Dict[str, Any],
) -> set[tuple[str, str, str, str]]:
    out: set[tuple[str, str, str, str]] = set()
    tool_context = payload.get("tool_context") or {}
    for move in tool_context.get("enumerated_moves") or []:
        if not isinstance(move, dict):
            continue
        proposal = move.get("proposal") or {}
        repairs = proposal.get("object_repairs") or []
        for repair in repairs:
            if not isinstance(repair, dict):
                continue
            cluster_id = repair.get("cluster_id")
            object_id = repair.get("object_id")
            op = repair.get("op")
            if not (
                isinstance(cluster_id, str)
                and isinstance(object_id, str)
                and isinstance(op, str)
            ):
                continue
            params_json = json.dumps(
                repair.get("params") or {}, sort_keys=True, ensure_ascii=True
            )
            out.add((cluster_id, object_id, op, params_json))
    return out


# -----------------------------------------------------------------------------
# Compact payload + tool context
# -----------------------------------------------------------------------------
def prepare_phase2_llm_payload(
    full_payload: Dict[str, Any], move_limit: int = 20
) -> Dict[str, Any]:
    compact = deepcopy(full_payload)
    diagnosis = DiagnosePhase2Seed(payload=compact)
    moves = EnumeratePhase2RepairMoves(payload=compact, limit=move_limit)
    layout_metrics = diagnosis.get("global_layout_metrics") or {}
    compact["tool_context"] = {
        "diagnosis": diagnosis,
        "enumerated_moves": moves.get("moves") or [],
        "scoring_hints": {
            "baseline_score": diagnosis.get("score"),
            "search_phase": (compact.get("phase_control") or {}).get("repair_phase"),
            "hard_valid": diagnosis.get("hard_valid"),
            "priority_clusters": diagnosis.get("prioritized_clusters") or [],
            "priority_objects": diagnosis.get("prioritized_objects") or [],
            "main_path_clearance": deepcopy(
                ((layout_metrics.get("main_path_clearance") or {}).get("paths") or [])[
                    :4
                ]
            ),
            "main_path_blockers": deepcopy(
                (
                    (layout_metrics.get("main_path_clearance") or {}).get(
                        "blocking_objects"
                    )
                    or []
                )[:8]
            ),
            "opening_band_blocking": deepcopy(
                (layout_metrics.get("opening_band_blocking") or [])[:6]
            ),
        },
    }
    return compact


# -----------------------------------------------------------------------------
# JSON extraction / parse
# -----------------------------------------------------------------------------
def _extract_json_like(text: str) -> str:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def parse_llm_output(raw: str | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    blob = _extract_json_like(str(raw))
    return json.loads(blob)


# -----------------------------------------------------------------------------
# Proposal normalization / validation
# -----------------------------------------------------------------------------
def _normalize_object_repair_params(op: str, params: Dict[str, Any]) -> Dict[str, Any]:
    params = deepcopy(params or {})

    def _first(*keys: str) -> Any:
        for key in keys:
            if key in params and params.get(key) is not None:
                return params.get(key)
        return None

    def _parse_int(value: Any, field_name: str) -> int:
        if value is None:
            raise ValueError(f"{field_name} is required")
        if isinstance(value, str):
            value = value.strip().lower().replace("deg", "").strip()
        try:
            return int(float(value))
        except Exception as e:
            raise ValueError(f"{field_name} must be numeric") from e

    def _parse_float(value: Any, field_name: str) -> float:
        if value is None:
            raise ValueError(f"{field_name} is required")
        try:
            return float(value)
        except Exception as e:
            raise ValueError(f"{field_name} must be numeric") from e

    if op == "rotate_object":
        rot_raw = _first("rot", "rotation", "angle", "degrees")
        rot = _parse_int(rot_raw, "rotate_object.params.rot") % 360
        if rot not in {0, 90, 180, 270}:
            raise ValueError("rotate_object.params.rot must be one of 0/90/180/270")
        return {"rot": rot}

    if op == "mirror_object":
        axis = str(_first("axis", "mirror_axis", "direction") or "x").strip().lower()
        if axis not in {"x", "y"}:
            raise ValueError("mirror_object.params.axis must be 'x' or 'y'")
        return {"axis": axis}

    if op == "nudge_object":
        dx = _parse_int(_first("dx", "x", "offset_x"), "nudge_object.params.dx")
        dy = _parse_int(_first("dy", "y", "offset_y"), "nudge_object.params.dy")
        return {"dx": dx, "dy": dy}

    if op == "swap_objects":
        other = str(
            _first("other_object_id", "target_object_id", "swap_with", "other") or ""
        ).strip()
        if not other:
            raise ValueError("swap_objects.params.other_object_id is required")
        return {"other_object_id": other}

    if op == "set_anchor":
        anchor = str(_first("anchor", "anchor_id", "anchor_name", "side") or "").strip()
        if not anchor:
            raise ValueError("set_anchor.params.anchor is required")
        return {"anchor": anchor}

    if op == "set_front_override":
        fx = _first("dx", "fx", "front_dx", "x")
        fy = _first("dy", "fy", "front_dy", "y")
        return {
            "dx": _parse_float(fx, "set_front_override.params.dx"),
            "dy": _parse_float(fy, "set_front_override.params.dy"),
        }

    return params


def normalize_and_validate_repair_output(
    payload: Dict[str, Any], raw_output: str | Dict[str, Any]
) -> Dict[str, Any]:
    parsed = parse_llm_output(raw_output)
    try:
        proposal = Phase2RepairOutput.model_validate(parsed)
    except ValidationError as e:
        raise ValueError(f"LLM repair output schema invalid: {e}") from e

    seed_layout = payload.get("seed_layout") or {}
    seed_tmap = _seed_transform_map(seed_layout)
    seed_vmap = _seed_variant_map(seed_layout)
    seed_cluster_ids = set(seed_tmap.keys())

    proposal_transform_ids = {item.cluster_id for item in proposal.cluster_transforms}
    extra_transform_ids = sorted(proposal_transform_ids - seed_cluster_ids)
    if extra_transform_ids:
        raise ValueError(
            "cluster_transforms references unknown cluster ids. "
            f"extra={extra_transform_ids}"
        )

    proposal_variant_ids = {item.cluster_id for item in proposal.selected_variants}
    extra_variant_ids = sorted(proposal_variant_ids - seed_cluster_ids)
    if extra_variant_ids:
        raise ValueError(
            "selected_variants references unknown cluster ids. "
            f"extra={extra_variant_ids}"
        )

    out_tmap = {cid: deepcopy(item) for cid, item in seed_tmap.items()}
    for item in proposal.cluster_transforms:
        out_tmap[item.cluster_id] = item.model_dump()

    out_vmap = {cid: deepcopy(item) for cid, item in seed_vmap.items()}
    for item in proposal.selected_variants:
        out_vmap[item.cluster_id] = item.model_dump()

    if set(out_tmap.keys()) != seed_cluster_ids:
        missing = sorted(seed_cluster_ids - set(out_tmap.keys()))
        extra = sorted(set(out_tmap.keys()) - seed_cluster_ids)
        raise ValueError(
            "merged cluster_transforms must contain the full cluster-id set. "
            f"missing={missing}, extra={extra}"
        )

    if set(out_vmap.keys()) != seed_cluster_ids:
        missing = sorted(seed_cluster_ids - set(out_vmap.keys()))
        extra = sorted(set(out_vmap.keys()) - seed_cluster_ids)
        raise ValueError(
            "merged selected_variants must contain the full cluster-id set. "
            f"missing={missing}, extra={extra}"
        )

    allowed_rots = _allowed_cluster_rotations(payload)
    for cid, item in out_tmap.items():
        rot = int(item.get("rot") or 0)
        if cid in allowed_rots and rot not in allowed_rots[cid]:
            raise ValueError(f"cluster {cid} rot={rot} not allowed by cluster rules")

    allowed_variants = _allowed_variant_ids_by_cluster(payload)
    for cid, item in out_vmap.items():
        vid = str(item.get("variant_id") or "")
        if cid in allowed_variants and vid not in allowed_variants[cid]:
            raise ValueError(f"cluster {cid} variant_id={vid} not allowed")

    allowed_obj_ids = _allowed_object_ids_by_cluster(payload)
    allowed_ops = _allowed_object_ops(payload)
    search_phase = _repair_phase(payload)
    enumerated_object_moves = _enumerated_object_move_signatures(payload)

    normalized_object_repairs: List[Dict[str, Any]] = []
    seen_object_repairs: set[tuple] = set()

    for item in proposal.object_repairs:
        row = item.model_dump()
        cid = row["cluster_id"]
        oid = row["object_id"]
        op = row["op"]

        if cid not in seed_cluster_ids:
            raise ValueError(f"object_repair references unknown cluster_id={cid}")
        cluster_obj_ids = allowed_obj_ids.get(cid, set())
        if oid not in cluster_obj_ids:
            raise ValueError(
                f"object_repair references unknown object_id={oid} in cluster={cid}"
            )
        if op not in allowed_ops:
            raise ValueError(f"object_repair op={op} not allowed by edit_contract")

        row["params"] = _normalize_object_repair_params(op, row.get("params") or {})

        if op == "swap_objects":
            other = str(row["params"]["other_object_id"]).strip()
            if other == oid:
                raise ValueError(
                    f"swap_objects cannot swap object_id={oid} with itself in cluster={cid}"
                )
            if other not in cluster_obj_ids:
                for other_cid, obj_ids in allowed_obj_ids.items():
                    if other in obj_ids and other_cid != cid:
                        raise ValueError(
                            f"swap_objects cross-cluster swap is not allowed: {cid}.{oid} cannot swap with {other_cid}.{other}"
                        )
                raise ValueError(
                    f"swap_objects.params.other_object_id={other} not found in cluster={cid}"
                )

        repair_sig = (
            cid,
            oid,
            op,
            json.dumps(row["params"], sort_keys=True, ensure_ascii=True),
        )
        if (
            search_phase == "object_refine"
            and repair_sig not in enumerated_object_moves
        ):
            raise ValueError(
                "object_refine proposals must stay inside the deterministic object-move "
                f"neighborhood; unsupported repair={cid}.{oid}.{op}."
            )
        if repair_sig in seen_object_repairs:
            continue
        seen_object_repairs.add(repair_sig)
        normalized_object_repairs.append(row)

    if search_phase == "macro_layout" and normalized_object_repairs:
        raise ValueError("macro_layout proposals must keep object_repairs empty")

    if search_phase == "object_refine":
        if out_tmap != seed_tmap:
            raise ValueError(
                "object_refine proposals must preserve cluster_transforms from the seed"
            )
        if out_vmap != seed_vmap:
            raise ValueError(
                "object_refine proposals must preserve selected_variants from the seed"
            )
        if not normalized_object_repairs:
            raise ValueError(
                "object_refine proposals must include at least one object repair"
            )
        if len(normalized_object_repairs) > 2:
            raise ValueError(
                "object_refine proposals may include at most two object repairs"
            )

    notes: List[str] = []
    seen_notes = set()
    for note in proposal.notes:
        if note not in seen_notes:
            seen_notes.add(note)
            notes.append(note)

    return {
        "status": proposal.status,
        "cluster_transforms": [out_tmap[cid] for cid in sorted(out_tmap.keys())],
        "selected_variants": [out_vmap[cid] for cid in sorted(out_vmap.keys())],
        "object_repairs": normalized_object_repairs,
        "notes": notes,
    }


# -----------------------------------------------------------------------------
# Compile using the same scorer/preview state
# -----------------------------------------------------------------------------
def compile_phase2_to_final_output(full_payload: Dict[str, Any]) -> Dict[str, Any]:
    phase2 = full_payload.get("phase2_placer") or {}
    repair = phase2.get("phase2_repair")
    if not isinstance(repair, dict):
        repair = make_no_improvement_repair(full_payload)
    compiled = CompileAcceptedPhase2Proposal(payload=full_payload, repair=repair)
    _ensure_room_type(compiled, full_payload)
    return compiled


def _ensure_room_type(compiled: Dict[str, Any], full_payload: Dict[str, Any]) -> None:
    room = compiled.get("room")
    if not isinstance(room, dict):
        return
    if room.get("room_type"):
        return

    room_model = (full_payload.get("room_context") or {}).get("room_model_used") or {}
    meta = room_model.get("meta") if isinstance(room_model, dict) else None
    user_input = room_model.get("user_input") if isinstance(room_model, dict) else None

    room_type = None
    if isinstance(meta, dict):
        room_type = meta.get("room_type")
    if not room_type and isinstance(user_input, dict):
        room_type = user_input.get("room_type")

    if isinstance(room_type, str) and room_type:
        room["room_type"] = room_type
    else:
        room["room_type"] = "unknown"


# -----------------------------------------------------------------------------
# LLM runtime
# -----------------------------------------------------------------------------
def _extract_message(response: object) -> object:
    choices = getattr(response, "choices", None)
    if isinstance(choices, list) and choices:
        return getattr(choices[0], "message", None)
    raise ValueError("OpenAI response missing message")


def _configured_placer_models() -> list[str]:
    if TextLLMConfig is None:
        return list(_DEFAULT_PLACER_FALLBACK_MODELS)
    configured = TextLLMConfig.agent_model_chain(
        _PLACER_MODEL_ROTATION_KEYS,
        _DEFAULT_PLACER_FALLBACK_MODELS,
    )
    return configured or list(_DEFAULT_PLACER_FALLBACK_MODELS)


def _current_placer_model_name() -> str:
    global _placer_model_cursor
    candidates = _configured_placer_models()
    _placer_model_cursor %= len(candidates)
    return candidates[_placer_model_cursor]


def _advance_placer_model_name() -> str:
    global _placer_model_cursor
    candidates = _configured_placer_models()
    _placer_model_cursor = (_placer_model_cursor + 1) % len(candidates)
    return candidates[_placer_model_cursor]


def _extract_retry_delay_seconds(exc: Exception) -> float | None:
    text = str(exc)
    match = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", text, flags=re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def _is_quota_exhausted_error(exc: Exception) -> bool:
    text = str(exc)
    lowered = text.lower()
    return (
        "429" in lowered
        or "resource_exhausted" in lowered
        or "quota exceeded" in lowered
    ) and ("generativelanguage" in lowered or "gemini" in lowered or "quota" in lowered)


def _call_openai(*, system_prompt: str, user_payload_json: str) -> str | Dict[str, Any]:
    if get_llm_client is None:
        raise RuntimeError("LLM client is unavailable")
    client = get_llm_client()
    model_name = _current_placer_model_name()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_payload_json},
    ]
    try:
        response = client.chat_completion(
            messages,
            model_key="primary",
            model_name=model_name,
            temperature=0.0,
        )
    except Exception as exc:
        if _is_quota_exhausted_error(exc):
            next_model = _advance_placer_model_name()
            raise RetryablePlacerModelError(
                failed_model=model_name,
                next_model=next_model,
                retry_delay_s=_extract_retry_delay_seconds(exc),
                detail=str(exc),
            ) from exc
        raise
    message = _extract_message(response)
    content = getattr(message, "content", None)
    if not isinstance(content, str):
        raise ValueError("OpenAI response content missing")
    return content


# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------
def _default_no_improvement(
    payload: Dict[str, Any], note: str | None = None
) -> Dict[str, Any]:
    return make_no_improvement_repair(payload, note=note)


def _repair_signature(repair: Dict[str, Any]) -> str:
    return json.dumps(
        {
            "cluster_transforms": repair.get("cluster_transforms") or [],
            "selected_variants": repair.get("selected_variants") or [],
            "object_repairs": repair.get("object_repairs") or [],
        },
        ensure_ascii=True,
        sort_keys=True,
    )


def _has_macro_orientation_pressure(evaluation: Dict[str, Any]) -> bool:
    metrics = evaluation.get("metrics") or {}
    for row in metrics.get("orientation_debug") or []:
        if not isinstance(row, dict):
            continue
        if int(row.get("penalty_mm") or 0) < 220:
            continue
        kind = str(row.get("kind") or "")
        intent = str(row.get("intent") or "")
        relation = str(row.get("relation") or "")
        if kind in {"cluster_directional_relation", "cluster_orientation"}:
            return True
        if (
            relation in {"face_each_other", "access_faces_other"}
            or intent == "face_object"
        ):
            return True
    return False


def _has_planner_object_axis_pressure(evaluation: Dict[str, Any]) -> bool:
    metrics = evaluation.get("metrics") or {}
    for row in metrics.get("orientation_debug") or []:
        if not isinstance(row, dict):
            continue
        if int(row.get("penalty_mm") or 0) < 180:
            continue
        kind = str(row.get("kind") or "")
        intent = str(row.get("intent") or "").strip().lower()
        if kind == "object_orientation" and intent in _PLANNER_OBJECT_AXIS_INTENTS:
            return True
    return False


def _deterministic_move_rank(
    evaluation: Dict[str, Any],
    *,
    move_kind: str,
) -> tuple[int, int, int, int, int, int]:
    comparison = evaluation.get("baseline_comparison") or {}
    delta_score = int(comparison.get("delta_score") or 0)
    candidate_score = int(comparison.get("candidate_score") or 0)
    hard_valid = 1 if bool(evaluation.get("hard_valid")) else 0
    positive_delta = 1 if delta_score > 0 else 0
    orientation_priority = 0
    if _has_macro_orientation_pressure(evaluation):
        if move_kind in {"cluster_variant", "cluster_pose"}:
            orientation_priority = 1
        elif move_kind == "object_pose":
            orientation_priority = -1
    elif _has_planner_object_axis_pressure(evaluation):
        if move_kind == "object_pose":
            orientation_priority = 1
        elif move_kind in {"cluster_variant", "cluster_pose"}:
            orientation_priority = -1
    move_priority = 0
    if move_kind == "cluster_variant":
        move_priority = 3
    elif move_kind == "cluster_pose":
        move_priority = 2
    elif move_kind == "object_pose":
        move_priority = 1
    return (
        hard_valid,
        positive_delta,
        delta_score,
        candidate_score,
        orientation_priority,
        move_priority,
    )


def _deterministic_phase2_proposal(
    payload: Dict[str, Any],
    *,
    move_limit: int,
) -> Dict[str, Any]:
    compact_payload = prepare_phase2_llm_payload(payload, move_limit=move_limit)
    baseline = make_no_improvement_repair(compact_payload)
    baseline_evaluation = EvaluatePhase2Proposal(
        payload=compact_payload, repair=baseline
    )
    moves = (compact_payload.get("tool_context") or {}).get("enumerated_moves") or []

    best_proposal = baseline
    best_rank = _deterministic_move_rank(
        baseline_evaluation,
        move_kind="seed_retain",
    )
    best_reason = "Retained the current seed because no deterministic move improved it."
    seen_signatures = {_repair_signature(best_proposal)}

    for move in moves:
        if not isinstance(move, dict):
            continue
        reason = str(move.get("reason") or "Deterministic repair move.")
        raw_proposal = {
            "status": "REPAIRED",
            "cluster_transforms": deepcopy(
                ((move.get("proposal") or {}).get("cluster_transforms") or [])
            ),
            "selected_variants": deepcopy(
                ((move.get("proposal") or {}).get("selected_variants") or [])
            ),
            "object_repairs": deepcopy(
                ((move.get("proposal") or {}).get("object_repairs") or [])
            ),
            "notes": [f"Deterministic placer: {reason}"],
        }
        try:
            proposal = normalize_and_validate_repair_output(
                compact_payload, raw_proposal
            )
        except ValueError:
            continue
        signature = _repair_signature(proposal)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        evaluation = EvaluatePhase2Proposal(payload=compact_payload, repair=proposal)
        rank = _deterministic_move_rank(
            evaluation,
            move_kind=str(move.get("kind") or ""),
        )
        if rank > best_rank:
            best_rank = rank
            best_proposal = proposal
            best_reason = reason

    if best_proposal["status"] == "NO_IMPROVEMENT":
        best_proposal["notes"] = [best_reason]
    elif not best_proposal.get("notes"):
        best_proposal["notes"] = [f"Deterministic placer selected: {best_reason}"]

    return best_proposal


def make_no_improvement_repair(
    payload: Dict[str, Any], note: str | None = None
) -> Dict[str, Any]:
    seed = payload.get("seed_layout") or {}
    out = {
        "status": "NO_IMPROVEMENT",
        "cluster_transforms": deepcopy(seed.get("cluster_transforms") or []),
        "selected_variants": deepcopy(seed.get("selected_variants") or []),
        "object_repairs": [],
        "notes": [],
    }
    if note:
        out["notes"].append(str(note))
    return out


def run_phase2_placer(
    *,
    full_payload: Dict[str, Any],
    llm_call: LLMCallable,
    system_prompt: str = DEFAULT_PHASE2_PLACER_PROMPT,
    attach_result_to_payload: bool = True,
    max_attempts: int = 3,
) -> Dict[str, Any]:
    compact_payload = prepare_phase2_llm_payload(full_payload)
    logger.info(
        "Phase2Placer input: repair_phase=%s seed_kind=%s hard_valid=%s priority_clusters=%s",
        ((compact_payload.get("phase_control") or {}).get("repair_phase")),
        ((compact_payload.get("phase_control") or {}).get("seed_kind")),
        (
            ((compact_payload.get("tool_context") or {}).get("diagnosis") or {}).get(
                "hard_valid"
            )
        ),
        (
            ((compact_payload.get("tool_context") or {}).get("diagnosis") or {}).get(
                "prioritized_clusters"
            )
        ),
    )
    attempt_errors: List[str] = []
    proposal: Dict[str, Any] | None = None
    attempt = 0
    rotation_retries = 0
    max_rotation_retries = max(3, len(_configured_placer_models()) * 2)
    while attempt < max(1, int(max_attempts)):
        try:
            raw = llm_call(
                system_prompt=system_prompt,
                user_payload_json=json.dumps(
                    compact_payload, ensure_ascii=False, indent=2
                ),
            )
            proposal = normalize_and_validate_repair_output(compact_payload, raw)
            logger.info(
                "Phase2Placer accepted output on attempt %s: status=%s",
                attempt + 1,
                proposal.get("status"),
            )
            break
        except RetryablePlacerModelError as exc:
            rotation_retries += 1
            retry_note = (
                f"Phase2Placer rate-limited on model={exc.failed_model}; "
                f"rotating to model={exc.next_model} without consuming an attempt."
            )
            if exc.retry_delay_s is not None:
                retry_note += f" retry_delay_s={exc.retry_delay_s:.3f}"
            logger.warning(retry_note)
            if rotation_retries >= max_rotation_retries:
                attempt_errors.append(
                    f"model_rotation_exhausted: {exc.failed_model} -> {exc.next_model}"
                )
                logger.warning(
                    "Phase2Placer exhausted model rotation retries after quota errors"
                )
                break
            continue
        except Exception as e:
            attempt += 1
            logger.warning("Phase2Placer attempt %s failed: %s", attempt, e)
            attempt_errors.append(f"attempt_{attempt}: {e}")

    if proposal is None:
        proposal = make_no_improvement_repair(
            compact_payload,
            note="Fallback to NO_IMPROVEMENT because the placer output was invalid.",
        )
        logger.warning("Phase2Placer fallback to NO_IMPROVEMENT after invalid outputs")

    if not attach_result_to_payload:
        return proposal

    attached = deepcopy(full_payload)
    attached["phase2_placer"] = {
        "phase2_input": compact_payload,
        "phase2_repair": proposal,
        "attempt_errors": attempt_errors,
    }
    return attached


@dataclass(frozen=True)
class MacroClusterPlacer:
    system_prompt: str = DEFAULT_PHASE2_PLACER_PROMPT
    llm_call: LLMCallable = _call_openai

    def generate_from_payload(
        self,
        *,
        payload: Dict[str, Any],
        max_attempts: int = 3,
    ) -> Dict[str, Any]:
        if self.llm_call is _call_openai:
            move_limit = max(24, max_attempts * 24)
            return _deterministic_phase2_proposal(payload, move_limit=move_limit)
        result = run_phase2_placer(
            full_payload=payload,
            llm_call=self.llm_call,
            system_prompt=self.system_prompt,
            attach_result_to_payload=False,
            max_attempts=max_attempts,
        )
        if not isinstance(result, dict):
            raise ValueError("Phase-2 placer must return a JSON object")
        return result

    def generate_raw(
        self,
        *,
        room_interpreter_json: Dict[str, Any],
        cluster_merged_json: Dict[str, Any],
        cluster_outlines_json: Dict[str, Any],
        relation_plan_json: Dict[str, Any],
        placer_output_json: Dict[str, Any],
        cluster_constraints_json: Dict[str, Any] | None = None,
    ) -> str:
        payload = build_phase2_payload(
            room_01=room_interpreter_json,
            clusters_04=cluster_merged_json,
            outlines_05=cluster_outlines_json,
            relation_05b=relation_plan_json,
            placer_06=placer_output_json,
            cluster_constraints=cluster_constraints_json,
        )
        proposal = self.generate_from_payload(payload=payload)
        return json.dumps(proposal, ensure_ascii=True)

    def generate(
        self,
        *,
        room_interpreter_json: Dict[str, Any],
        cluster_merged_json: Dict[str, Any],
        cluster_outlines_json: Dict[str, Any],
        relation_plan_json: Dict[str, Any],
        placer_output_json: Dict[str, Any],
        cluster_constraints_json: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        raw = self.generate_raw(
            room_interpreter_json=room_interpreter_json,
            cluster_merged_json=cluster_merged_json,
            cluster_outlines_json=cluster_outlines_json,
            relation_plan_json=relation_plan_json,
            placer_output_json=placer_output_json,
            cluster_constraints_json=cluster_constraints_json,
        )
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("MacroClusterPlacer response must be a JSON object")
        return payload


# -----------------------------------------------------------------------------
# Testing adapter
# -----------------------------------------------------------------------------
def make_file_response_adapter(path: str | Path) -> LLMCallable:
    def _call(*, system_prompt: str, user_payload_json: str) -> str | Dict[str, Any]:
        _ = system_prompt
        _ = user_payload_json
        return load_json(path)

    return _call


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Phase-2 Placer v3 with a prebuilt payload and a file-based mock LLM response."
    )
    parser.add_argument("--payload", required=True, help="Path to phase-2 payload JSON")
    parser.add_argument(
        "--mock-response", required=True, help="Path to mock LLM repair JSON"
    )
    parser.add_argument("--out", required=True, help="Path to write merged output JSON")
    args = parser.parse_args()

    payload = load_json(args.payload)
    adapter = make_file_response_adapter(args.mock_response)
    out = run_phase2_placer(
        full_payload=payload,
        llm_call=adapter,
        attach_result_to_payload=False,
    )
    dump_json(args.out, out)
