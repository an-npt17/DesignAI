"""
Finite-pose bedroom/room layout solver using OR-Tools CP-SAT.

What this solver does
---------------------
- Builds generic mirrored variants for every cluster.
- Enumerates a finite set of hard-valid pose candidates for each cluster.
- Solves exact selection over that candidate set with CP-SAT:
  - exactly one pose per cluster
  - no pairwise overlap between selected poses
  - objective combines local pose quality + pairwise interaction bonuses
- Verifies the full layout.
- If the best exact solution is still not acceptable, refines only the most
  problematic clusters and solves again.

Important note
--------------
This is an exact solver over a finite candidate set, not a continuous global
optimizer over arbitrary x/y. It is more stable than an LLM search loop, but
still limited by the richness of the generated poses.

Expected companion file
-----------------------
This solver expects tools from cluster_placer/tools.py to be available, or
a path can be passed explicitly. That file provides robust geometry,
verification, canonicalization, mirroring, and candidate enumeration helpers.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from layout.grid_policy import GLOBAL_LAYOUT_GRID_MM
from layout.room_profiles.registry import is_profile_trait_object
from layout.variant_family import FALLBACK_GENERIC_VARIANT_FAMILY
from layout.variant_family import GENERIC_VARIANT_FAMILIES
from layout.variant_family import ROLE_VARIANT_FAMILY_ALLOWLISTS
from layout.variant_family import (
    SEMANTIC_VARIANT_FAMILIES as CORE_SEMANTIC_VARIANT_FAMILIES,
)
from layout.variant_family import SLEEP_VARIANT_FAMILIES
from layout.variant_family import normalize_variant_family

try:
    from ortools.sat.python import cp_model
except Exception as exc:  # pragma: no cover - import guard only
    raise RuntimeError(
        "OR-Tools is required. Install with: python -m pip install ortools"
    ) from exc

logger = logging.getLogger(__name__)

_CLUSTER_WALL_PREFER_TAGS = frozenset(
    {"wall", "long_wall", "short_wall", "recess_or_edge"}
)
_CLUSTER_CENTER_PREFER_TAGS = frozenset({"center", "near_center"})
_CLUSTER_CENTER_AVOID_TAGS = frozenset({"center"})
_CLUSTER_WINDOW_AVOID_TAGS = frozenset({"window_blocking", "window_clearance"})
_CLUSTER_ENTRY_AVOID_TAGS = frozenset({"entry_blocking", "door_swing", "main_path"})
_CLUSTER_WINDOW_PREFER_TAGS = frozenset({"window_side"})
_CLUSTER_ENTRY_PREFER_TAGS = frozenset({"entry_side", "near_entry"})
_WALL_ANCHOR_TOKENS = ("wall", "top", "bottom", "left", "right", "edge", "recess")
_CENTER_ANCHOR_TOKENS = ("center", "quadrant")
_WINDOW_ANCHOR_TOKENS = ("window",)
_ENTRY_ANCHOR_TOKENS = ("entry", "door")
_WINDOW_TREATMENT_OBJECT_TOKENS = frozenset(
    {
        "blind",
        "blinds",
        "curtain",
        "curtain_rod",
        "curtains",
        "drape",
        "drapes",
        "drapery",
        "drapery_rod",
        "shade",
        "shades",
        "sheer",
        "valance",
        "window_covering",
        "window_treatment",
    }
)

DEFAULT_GRID_MM = 50
DEFAULT_MAX_CONCEPTS = 5
DEFAULT_MAX_VARIANTS_PER_CLUSTER = 6
DEFAULT_MAX_POSE_CANDIDATES_PER_VARIANT = 40
DEFAULT_MAX_POSE_CANDIDATES_PER_CLUSTER = 120
DEFAULT_MAX_TOTAL_BINARY_CANDIDATES = 1200
DEFAULT_MAX_FEASIBLE_SOLUTIONS_PER_CONCEPT = 12
DEFAULT_MAX_CROSS_CONCEPT_FINALISTS = 20
DEFAULT_SOLVER_TIME_LIMIT_SEC_PER_CONCEPT = 8.0
DEFAULT_SOLVER_TOTAL_TIME_LIMIT_SEC = 40.0
DEFAULT_MAX_DEGRADATION_ROUNDS = 3
OBJECT_LEVEL_MAX_ANCHOR_CANDIDATES_PER_CLUSTER = 36
OBJECT_LEVEL_MAX_SUPPORT_SOLUTIONS_PER_ANCHOR = 16
OBJECT_LEVEL_MAX_OBJECT_SOLUTIONS = 24
OBJECT_LEVEL_MAX_SUPPORT_SLOT_CANDIDATES = 36
OBJECT_LEVEL_GEOMETRY_REPAIR_MAX_CANDIDATES = 120
OBJECT_LEVEL_GEOMETRY_REPAIR_LOCAL_RADIUS_MM = 1800
OBJECT_LEVEL_REQUIRED_FACE_PAIR_MIN_DOT = 0.8
OBJECT_LEVEL_SUPPORT_ALIGNMENT_FACTORS = (-0.35, 0.0, 0.35)
OBJECT_LEVEL_WALL_CONTACT_TOLERANCE_MM = 1
OBJECT_LEVEL_FRONT_ALIGNMENT_MIN_DOT = 0.45
OBJECT_LEVEL_FRONT_ACCESS_DEPTH_MM = 750
OBJECT_LEVEL_VIEW_CORRIDOR_WIDTH_MM = 900
OBJECT_LEVEL_BLOCKING_PROTECTED_PRIORITIES = frozenset({"critical", "high"})
OBJECT_LEVEL_BLOCKING_PROTECTED_ENFORCEMENT = frozenset({"hard", "hard_soft"})
OBJECT_LEVEL_BLOCKING_PROTECTED_SEVERITIES = frozenset({"blocking", "critical"})
OBJECT_LEVEL_LOW_HEIGHT_DAYLIGHT_SOFT_MAX_MM = 1400
OBJECT_LEVEL_LOW_HEIGHT_DAYLIGHT_MAX_OVERLAP_RATIO = 0.35
OBJECT_LEVEL_HARD_SOFT_ANCHOR_REJECT_RATIO = 0.16

QUALITY_WEIGHTS = {
    "functionality": 0.32,
    "naturalness": 0.24,
    "semantic": 0.24,
    "spatial": 0.20,
}

PUBLISHABLE_BLOCKING_QUALITY_REASONS = frozenset(
    {
        "critical_orientation_penalty_too_high",
        "focal_pair_penalty_too_high",
        "critical_item_penalty_too_high",
    }
)

SEMANTIC_VARIANT_FAMILIES = CORE_SEMANTIC_VARIANT_FAMILIES | frozenset(
    {FALLBACK_GENERIC_VARIANT_FAMILY}
)

STRICT_ROLE_KINDS = frozenset({"focal", "kitchen", "media", "work", "workflow"})


@dataclass(frozen=True)
class Candidate:
    cluster_id: str
    variant_id: str
    variant_family: str
    variant_priority: float
    x: int
    y: int
    rot: int
    anchor_kind: str
    anchor_priority: float
    stage: str
    hard_valid: bool
    acceptable_valid: bool
    rough_score: int
    macro_penalty_mm: int
    micro_penalty_mm: int
    orientation_penalty_mm: int
    critical_orientation_penalty_mm: int
    focal_orientation_penalty_mm: int
    quality_gate_reasons: tuple[str, ...]
    hard_error_codes: tuple[str, ...]
    state_signature: str

    @property
    def key(self) -> tuple[str, str, int, int, int]:
        return (self.cluster_id, self.variant_id, self.x, self.y, self.rot)


@dataclass(frozen=True)
class MacroClusterSolver:
    tools_path: str = ""
    max_variants_per_cluster: int = DEFAULT_MAX_VARIANTS_PER_CLUSTER
    initial_candidates_per_cluster: int = DEFAULT_MAX_POSE_CANDIDATES_PER_VARIANT
    max_rounds: int = DEFAULT_MAX_DEGRADATION_ROUNDS
    time_limit_s: float = DEFAULT_SOLVER_TIME_LIMIT_SEC_PER_CONCEPT
    num_workers: int = 8
    max_feasible_solutions_per_concept: int = DEFAULT_MAX_FEASIBLE_SOLUTIONS_PER_CONCEPT

    def generate(
        self,
        *,
        room_model_json: dict[str, Any],
        clusters_outlines_json: dict[str, Any] | list[Any],
        relation_plan_json: dict[str, Any] | None,
        cluster_constraints_json: dict[str, Any] | None = None,
        grid_mm: int = GLOBAL_LAYOUT_GRID_MM,
    ) -> dict[str, Any]:
        tools_path = self.tools_path or _default_tools_path()
        return solve_layout(
            room_model=room_model_json,
            clusters_outlines=clusters_outlines_json,
            relation_plan=relation_plan_json,
            cluster_constraints=cluster_constraints_json,
            grid_mm=grid_mm,
            tools_path=tools_path,
            max_variants_per_cluster=self.max_variants_per_cluster,
            initial_candidates_per_cluster=self.initial_candidates_per_cluster,
            max_rounds=self.max_rounds,
            time_limit_s=self.time_limit_s,
            num_workers=self.num_workers,
            max_feasible_solutions_per_concept=self.max_feasible_solutions_per_concept,
        )

    def generate_bundle(
        self,
        payload: dict[str, Any],
        *,
        grid_mm: int = DEFAULT_GRID_MM,
    ) -> dict[str, Any]:
        tools_path = self.tools_path or _default_tools_path()
        return solve_global_layout_bundle(
            payload=payload,
            grid_mm=grid_mm,
            tools_path=tools_path,
            max_variants_per_cluster=self.max_variants_per_cluster,
            initial_candidates_per_cluster=self.initial_candidates_per_cluster,
            max_rounds=self.max_rounds,
            time_limit_s=self.time_limit_s,
            num_workers=self.num_workers,
            max_feasible_solutions_per_concept=self.max_feasible_solutions_per_concept,
        )

    def generate_object_layout(
        self,
        *,
        room_model_json: dict[str, Any],
        merged_clusters_json: dict[str, Any],
        relation_plan_json: dict[str, Any] | None,
        cluster_constraints_json: dict[str, Any] | None = None,
        grid_mm: int = GLOBAL_LAYOUT_GRID_MM,
    ) -> dict[str, Any]:
        return solve_object_level_layout(
            room_model=room_model_json,
            merged_clusters=merged_clusters_json,
            relation_plan=relation_plan_json,
            cluster_constraints=cluster_constraints_json,
            grid_mm=grid_mm,
            max_rounds=self.max_rounds,
        )


def _load_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: str | os.PathLike[str], data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_tools_module(path: str | os.PathLike[str]):
    spec = importlib.util.spec_from_file_location("layout_tools", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import tools module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _require_attr(mod: Any, name: str) -> Any:
    if not hasattr(mod, name):
        raise RuntimeError(f"tools module missing required symbol: {name}")
    return getattr(mod, name)


def _candidate_from_dict(item: dict[str, Any]) -> Candidate:
    return Candidate(
        cluster_id=str(item["cluster_id"]),
        variant_id=str(item["variant_id"]),
        variant_family=str(item.get("variant_family") or "base"),
        variant_priority=float(item.get("variant_priority") or 1.0),
        x=int(item["x"]),
        y=int(item["y"]),
        rot=int(item["rot"]),
        anchor_kind=str(item.get("anchor_kind") or "unknown"),
        anchor_priority=float(item.get("anchor_priority") or 0.0),
        stage=str(item.get("stage") or "seed"),
        hard_valid=bool(item.get("hard_valid", False)),
        acceptable_valid=bool(item.get("acceptable_valid", False)),
        rough_score=int(item.get("rough_score") or 0),
        macro_penalty_mm=int(item.get("macro_penalty_mm") or 0),
        micro_penalty_mm=int(item.get("micro_penalty_mm") or 0),
        orientation_penalty_mm=int(item.get("orientation_penalty_mm") or 0),
        critical_orientation_penalty_mm=int(
            item.get("critical_orientation_penalty_mm") or 0
        ),
        focal_orientation_penalty_mm=int(item.get("focal_orientation_penalty_mm") or 0),
        quality_gate_reasons=tuple(item.get("quality_gate_reasons") or []),
        hard_error_codes=tuple(item.get("hard_error_codes") or []),
        state_signature=str(item.get("state_signature") or ""),
    )


def _primary_seed_problem_clusters(
    problem_clusters: list[str],
    *,
    cluster_ids: list[str],
    max_clusters: int = 2,
) -> list[str]:
    ordered: list[str] = []
    for cluster_id in problem_clusters or cluster_ids:
        clean_cluster_id = str(cluster_id or "").strip()
        if not clean_cluster_id or clean_cluster_id in ordered:
            continue
        ordered.append(clean_cluster_id)
        if len(ordered) >= max(1, int(max_clusters)):
            break
    return ordered


def _expanded_candidate_limit(initial_candidates_per_cluster: int) -> int:
    base_limit = max(12, int(initial_candidates_per_cluster))
    return max(24, base_limit + 8)


def _round_candidate_growth_limit(initial_candidates_per_cluster: int) -> int:
    base_limit = max(12, int(initial_candidates_per_cluster))
    return max(8, min(12, base_limit // 2))


def _candidate_score(c: Candidate) -> int:
    score = int(c.rough_score)
    score += int(round(2200.0 * c.anchor_priority))
    score += int(round(1600.0 * c.variant_priority))
    if c.hard_valid:
        score += 3200
    if c.acceptable_valid:
        score += 5200
    score -= 16 * c.macro_penalty_mm
    score -= 3 * c.micro_penalty_mm
    score -= 8 * c.critical_orientation_penalty_mm
    score -= 6 * c.focal_orientation_penalty_mm
    score -= 1 * c.orientation_penalty_mm
    return score


def _candidate_effective_score(
    c: Candidate,
    candidate_bonus_by_key: Mapping[tuple[str, str, int, int, int], int] | None = None,
) -> int:
    bonus = 0
    if candidate_bonus_by_key is not None:
        bonus = int(candidate_bonus_by_key.get(c.key, 0))
    return _candidate_score(c) + bonus


def _verify_quality_metrics(verify: Mapping[str, Any] | None) -> dict[str, int]:
    if not isinstance(verify, Mapping):
        return {
            "layout_score": 0,
            "macro_penalty_mm": 0,
            "micro_penalty_mm": 0,
            "critical_orientation_penalty_mm": 0,
            "focal_pair_penalty_mm": 0,
        }
    quality = verify.get("quality")
    summary = verify.get("summary")
    quality_dict = quality if isinstance(quality, Mapping) else {}
    summary_dict = summary if isinstance(summary, Mapping) else {}
    return {
        "layout_score": int(quality_dict.get("layout_score") or 0),
        "macro_penalty_mm": int(
            quality_dict.get("macro_penalty_mm")
            or summary_dict.get("macro_penalty_mm")
            or 0
        ),
        "micro_penalty_mm": int(
            quality_dict.get("micro_penalty_mm")
            or summary_dict.get("micro_penalty_mm")
            or 0
        ),
        "critical_orientation_penalty_mm": int(
            quality_dict.get("critical_orientation_penalty_mm")
            or summary_dict.get("critical_orientation_penalty_mm")
            or 0
        ),
        "focal_pair_penalty_mm": int(
            quality_dict.get("focal_pair_penalty_mm")
            or summary_dict.get("focal_pair_penalty_mm")
            or 0
        ),
    }


def _verify_search_score(verify: Mapping[str, Any] | None) -> int:
    metrics = _verify_quality_metrics(verify)
    score = int(metrics["layout_score"])
    if isinstance(verify, Mapping) and bool(verify.get("hard_valid")):
        score += 3200
    if isinstance(verify, Mapping) and bool(verify.get("acceptable_valid")):
        score += 5200
    score -= 16 * metrics["macro_penalty_mm"]
    score -= 3 * metrics["micro_penalty_mm"]
    score -= 8 * metrics["critical_orientation_penalty_mm"]
    score -= 6 * metrics["focal_pair_penalty_mm"]
    return score


def _verify_rank_key(verify: Mapping[str, Any] | None) -> tuple[int, ...]:
    metrics = _verify_quality_metrics(verify)
    hard_valid = (
        1 if isinstance(verify, Mapping) and bool(verify.get("hard_valid")) else 0
    )
    acceptable_valid = (
        1 if isinstance(verify, Mapping) and bool(verify.get("acceptable_valid")) else 0
    )
    complete = 1 if isinstance(verify, Mapping) and bool(verify.get("complete")) else 0
    return (
        acceptable_valid,
        hard_valid,
        complete,
        -metrics["critical_orientation_penalty_mm"],
        -metrics["focal_pair_penalty_mm"],
        -metrics["macro_penalty_mm"],
        -metrics["micro_penalty_mm"],
        metrics["layout_score"],
    )


def _verify_macro_ready(verify: Mapping[str, Any] | None) -> bool:
    if not isinstance(verify, Mapping):
        return False
    if not bool(verify.get("hard_valid")) or not bool(verify.get("complete")):
        return False
    quality_gate = verify.get("quality_gate")
    reasons = []
    if isinstance(quality_gate, Mapping):
        reasons = [str(item or "") for item in (quality_gate.get("reasons") or [])]
    blocking = {
        "hard_constraints_failed",
        "layout_incomplete",
        "critical_orientation_penalty_too_high",
        "focal_pair_penalty_too_high",
    }
    return not any(reason in blocking for reason in reasons)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _penalty_score(penalty_mm: Any, scale_mm: float) -> float:
    penalty = max(0.0, _as_float(penalty_mm))
    return _clamp_score(1.0 / (1.0 + (penalty / max(1.0, scale_mm))))


def _quality_dict(verify: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(verify, Mapping):
        return {}
    quality = verify.get("quality")
    return quality if isinstance(quality, Mapping) else {}


def _summary_dict(verify: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(verify, Mapping):
        return {}
    summary = verify.get("summary")
    return summary if isinstance(summary, Mapping) else {}


def _concept_from_relation_plan(
    relation_plan: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if not isinstance(relation_plan, Mapping):
        return {}
    concept = relation_plan.get("macro_concept")
    return concept if isinstance(concept, Mapping) else {}


def _concept_id_from_relation_plan(relation_plan: Mapping[str, Any] | None) -> str:
    concept = _concept_from_relation_plan(relation_plan)
    for key in ("concept_id", "id", "concept_family"):
        value = str(concept.get(key) or "").strip()
        if value:
            return value
    if isinstance(relation_plan, Mapping):
        value = str(relation_plan.get("concept_id") or "").strip()
        if value:
            return value
    return "concept_01"


def _concept_prior_score(relation_plan: Mapping[str, Any] | None) -> float:
    concept = _concept_from_relation_plan(relation_plan)
    value = _as_float(concept.get("concept_score_prior"), 0.78)
    if value > 1.0:
        value /= 100.0
    return _clamp_score(value)


def _center_openness_strength(relation_plan: Mapping[str, Any] | None) -> str:
    concept = _concept_from_relation_plan(relation_plan)
    topology_policy = concept.get("topology_policy")
    if isinstance(topology_policy, Mapping):
        strength = str(topology_policy.get("reserve_center_degree") or "").strip()
        if strength:
            return strength
    if isinstance(relation_plan, Mapping):
        intent = relation_plan.get("layout_intent_profile")
        if isinstance(intent, Mapping):
            return str(intent.get("center_open_preference") or "").strip()
    return ""


def _normalized_policy_token(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _placement_behavior_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _placement_behavior_value(
    placement_behavior: Mapping[str, Any] | None,
    key: str,
) -> str:
    if not isinstance(placement_behavior, Mapping):
        return ""
    return _normalized_policy_token(placement_behavior.get(key))


def _placement_behavior_from_rows(
    *rows: Mapping[str, Any] | object,
) -> Mapping[str, Any]:
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        placement_behavior = _placement_behavior_mapping(row.get("placement_behavior"))
        if placement_behavior:
            return placement_behavior
    return {}


def _placement_behavior_for_cluster(
    relation_plan: Mapping[str, Any] | None,
    cluster_id: str,
) -> Mapping[str, Any]:
    if not isinstance(relation_plan, Mapping):
        return {}
    for section_name in (
        "anchor_layout_hints_by_cluster",
        "anchor_region_preferences_by_cluster",
    ):
        section = relation_plan.get(section_name)
        if not isinstance(section, Mapping):
            continue
        row = section.get(cluster_id)
        placement_behavior = _placement_behavior_from_rows(row)
        if placement_behavior:
            return placement_behavior
    concept = _concept_from_relation_plan(relation_plan)
    for row in concept.get("cluster_zone_plan") or []:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("cluster_id") or "").strip() != cluster_id:
            continue
        return _placement_behavior_from_rows(row)
    return {}


def _placement_behavior_orientation_intents(
    placement_behavior: Mapping[str, Any] | None,
) -> set[str]:
    intents: set[str] = set()
    wall_backing = _placement_behavior_value(placement_behavior, "wall_backing")
    front_space = _placement_behavior_value(placement_behavior, "front_space")
    if wall_backing in {"preferred", "required"}:
        intents.add("back_to_wall")
    if wall_backing == "required":
        intents.add("axis_parallel_wall")
    if front_space in {"preferred", "required"}:
        intents.update({"access_to_open_space", "front_to_open_space"})
    return intents


def _concept_family_from_relation_plan(relation_plan: Mapping[str, Any] | None) -> str:
    concept = _concept_from_relation_plan(relation_plan)
    return _normalized_policy_token(concept.get("concept_family"))


def _center_openness_is_strong(relation_plan: Mapping[str, Any] | None) -> bool:
    return _normalized_policy_token(_center_openness_strength(relation_plan)) in {
        "high",
        "very_high",
    }


def _center_openness_is_very_high(relation_plan: Mapping[str, Any] | None) -> bool:
    return _normalized_policy_token(_center_openness_strength(relation_plan)) == (
        "very_high"
    )


def _cluster_zone_plan_by_id(
    relation_plan: Mapping[str, Any] | None,
) -> dict[str, Mapping[str, Any]]:
    concept = _concept_from_relation_plan(relation_plan)
    zone_plan = concept.get("cluster_zone_plan")
    if not isinstance(zone_plan, list):
        return {}
    out: dict[str, Mapping[str, Any]] = {}
    for row in zone_plan:
        if not isinstance(row, Mapping):
            continue
        cluster_id = str(row.get("cluster_id") or "").strip()
        if cluster_id:
            out[cluster_id] = row
    return out


def _cluster_role_kind(
    relation_plan: Mapping[str, Any] | None,
    cluster_id: str,
) -> str:
    row = _cluster_zone_plan_by_id(relation_plan).get(cluster_id, {})
    explicit = _normalized_policy_token(
        row.get("role_kind")
        or row.get("cluster_role_kind")
        or row.get("role")
        or row.get("cluster_role")
    )
    if explicit:
        return explicit
    tokens = " ".join(
        str(value or "")
        for value in (
            cluster_id,
            row.get("semantic_role"),
            row.get("zone_assignment"),
            row.get("priority"),
        )
    ).lower()
    token_parts = {
        part
        for value in (
            cluster_id,
            row.get("semantic_role"),
            row.get("zone_assignment"),
            row.get("priority"),
        )
        for part in _normalized_policy_token(value).split("_")
        if part
    }
    if "kitchen" in tokens or any(
        is_profile_trait_object(part) for part in token_parts
    ):
        return "kitchen"
    if any(token in tokens for token in ("sofa", "sectional", "lounge", "seating")):
        return "social_anchor"
    if any(token in tokens for token in ("tv", "media", "screen", "fireplace")):
        return "media"
    if {"sleep", "bed", "headboard"} & token_parts:
        return "sleep"
    if any(token in tokens for token in ("desk", "work", "office", "study")):
        return "work"
    if any(token in tokens for token in ("storage", "cabinet", "shelf", "wardrobe")):
        return "storage"
    if any(token in tokens for token in ("focal", "feature_wall")):
        return "focal"
    return "support"


def _cluster_priority_kind(
    relation_plan: Mapping[str, Any] | None,
    cluster_id: str,
) -> str:
    row = _cluster_zone_plan_by_id(relation_plan).get(cluster_id, {})
    priority = _normalized_policy_token(row.get("priority"))
    if priority in {"core", "support", "optional"}:
        return priority
    role_kind = _cluster_role_kind(relation_plan, cluster_id)
    if role_kind in {"social_anchor", "media", "focal", "sleep", "kitchen"}:
        return "core"
    if role_kind in {"work", "storage"}:
        return "support"
    return "support"


def _cluster_is_core_or_protected(
    relation_plan: Mapping[str, Any] | None,
    cluster_id: str,
) -> bool:
    if cluster_id in {
        _extract_layout_primary_cluster_id(
            dict(relation_plan) if isinstance(relation_plan, dict) else None
        ),
        _extract_layout_secondary_cluster_id(
            dict(relation_plan) if isinstance(relation_plan, dict) else None
        ),
    }:
        return True
    return _cluster_priority_kind(relation_plan, cluster_id) == "core"


def _explicit_allowed_variant_families_by_cluster(
    relation_plan: Mapping[str, Any] | None,
) -> dict[str, set[str]]:
    containers: list[Any] = []
    if isinstance(relation_plan, Mapping):
        containers.append(relation_plan.get("allowed_variant_families_by_cluster"))
        policy = relation_plan.get("variant_family_policy")
        if isinstance(policy, Mapping):
            containers.append(policy.get("allowed_variant_families_by_cluster"))
            containers.append(policy.get("allowed_by_cluster"))
    concept = _concept_from_relation_plan(relation_plan)
    containers.append(concept.get("allowed_variant_families_by_cluster"))
    policy = concept.get("variant_family_policy")
    if isinstance(policy, Mapping):
        containers.append(policy.get("allowed_variant_families_by_cluster"))
        containers.append(policy.get("allowed_by_cluster"))

    out: dict[str, set[str]] = {}
    for container in containers:
        if not isinstance(container, Mapping):
            continue
        for cluster_id_raw, raw_value in container.items():
            cluster_id = str(cluster_id_raw or "").strip()
            if not cluster_id:
                continue
            values: Any
            if isinstance(raw_value, Mapping):
                values = (
                    raw_value.get("families")
                    or raw_value.get("allowed")
                    or raw_value.get("variant_families")
                )
            else:
                values = raw_value
            if isinstance(values, str):
                families = {normalize_variant_family(values)}
            elif isinstance(values, Sequence):
                families = {
                    normalize_variant_family(value)
                    for value in values
                    if normalize_variant_family(value)
                }
            else:
                families = set()
            if families:
                out.setdefault(cluster_id, set()).update(families)
    return out


def _concept_allows_fallback_generic(
    relation_plan: Mapping[str, Any] | None,
    cluster_id: str,
) -> bool:
    explicit = _explicit_allowed_variant_families_by_cluster(relation_plan).get(
        cluster_id,
        set(),
    )
    if "fallback_generic" in explicit:
        return True
    concept = _concept_from_relation_plan(relation_plan)
    for key in (
        "allow_fallback_generic",
        "fallback_generic_allowed",
        "allow_generic_fallback",
    ):
        value = concept.get(key)
        if isinstance(value, bool):
            return value
    policy = concept.get("variant_family_policy")
    if isinstance(policy, Mapping):
        for key in (
            "allow_fallback_generic",
            "fallback_generic_allowed",
            "allow_generic_fallback",
        ):
            value = policy.get(key)
            if isinstance(value, bool):
                return value
    return False


def _role_allowed_variant_families(role_kind: str) -> set[str]:
    normalized = _normalized_policy_token(role_kind)
    return set(ROLE_VARIANT_FAMILY_ALLOWLISTS.get(normalized, frozenset()))


def _variant_family_compatibility_score(
    *,
    concept_family: str,
    role_kind: str,
    variant_family: str,
) -> float:
    concept = _normalized_policy_token(concept_family)
    role = _normalized_policy_token(role_kind)
    family = normalize_variant_family(variant_family)
    if family == "fallback_generic":
        return -1.2 if role in {"social_anchor", "media", "focal", "work"} else -0.7
    if family in GENERIC_VARIANT_FAMILIES:
        return 0.0

    role_allowed = _role_allowed_variant_families(role)
    score = 0.0
    if role_allowed:
        score += 0.75 if family in role_allowed else -0.65

    if concept == "open_center":
        if role in {"social_anchor", "lounge"} and family in {
            "open_center",
            "perimeter_facing",
            "conversation_facing",
        }:
            score += 0.8
        elif role in {"media", "focal"} and family in {
            "media_facing",
            "wall_backed_focal",
            "focal_media",
        }:
            score += 0.45
        elif family in {"storage_wall", "edge_storage", "perimeter_storage"}:
            score += 0.25
    elif concept == "focal_axis":
        if role in {"media", "focal"} and family in {
            "media_facing",
            "wall_backed_focal",
            "focal_media",
            "focal_axis",
        }:
            score += 0.9
        elif role in {"social_anchor", "lounge"} and family in {
            "conversation_facing",
            "perimeter_facing",
        }:
            score += 0.35
    elif concept == "daylight_oriented":
        if role in {"work", "workflow"} and family in {
            "daylight_work",
            "work_core",
            "window_oriented",
        }:
            score += 0.95
        elif role == "sleep" and family == "bed_plus_window_side_bench":
            score += 0.95
        elif role == "sleep" and family in {
            "headboard_wall_balanced",
            "headboard_wall_single_side",
        }:
            score += 0.35
        elif family == "window_oriented":
            score += 0.35
    elif concept == "edge_weighted":
        if family in {"storage_wall", "edge_storage", "perimeter_storage"}:
            score += 0.65
        elif role == "sleep" and family == "bed_plus_storage_buffer":
            score += 0.8
        elif role == "sleep" and family in {
            "headboard_wall_balanced",
            "headboard_wall_single_side",
        }:
            score += 0.3
        elif family in {"open_center", "conversation_facing"}:
            score -= 0.25
    elif concept == "zoned":
        if role_allowed and family in role_allowed:
            score += 0.35
        if role == "sleep" and family in SLEEP_VARIANT_FAMILIES:
            score += 0.2
    return max(-1.5, min(1.5, score))


def _variant_family_allowed_for_cluster(
    *,
    relation_plan: Mapping[str, Any] | None,
    cluster_id: str,
    variant_family: str,
    semantic_families_available: bool,
) -> bool:
    family = normalize_variant_family(variant_family)
    if not family:
        return True
    if family == "fallback_generic":
        return _concept_allows_fallback_generic(relation_plan, cluster_id)

    explicit = _explicit_allowed_variant_families_by_cluster(relation_plan).get(
        cluster_id,
        set(),
    )
    if explicit:
        return family in explicit

    role_kind = _cluster_role_kind(relation_plan, cluster_id)
    role_allowed = _role_allowed_variant_families(role_kind)
    if role_allowed and family in role_allowed:
        return True
    if _normalized_policy_token(role_kind) in STRICT_ROLE_KINDS:
        return False
    if family in GENERIC_VARIANT_FAMILIES:
        return not semantic_families_available
    if role_allowed:
        return False
    return True


def _filter_candidates_by_concept_variant_policy(
    candidates_by_cluster: dict[str, list[Candidate]],
    relation_plan: Mapping[str, Any] | None,
) -> dict[str, list[Candidate]]:
    out: dict[str, list[Candidate]] = {}
    for cluster_id, candidates in candidates_by_cluster.items():
        semantic_families_available = any(
            normalize_variant_family(candidate.variant_family)
            in SEMANTIC_VARIANT_FAMILIES
            for candidate in candidates
        )
        out[cluster_id] = [
            candidate
            for candidate in candidates
            if _variant_family_allowed_for_cluster(
                relation_plan=relation_plan,
                cluster_id=cluster_id,
                variant_family=candidate.variant_family,
                semantic_families_available=semantic_families_available,
            )
        ]
    return out


def _filter_variant_bundle_rows_for_concept(
    *,
    cluster_id: str,
    rows: Any,
    relation_plan: Mapping[str, Any] | None,
) -> Any:
    if not isinstance(rows, list):
        return rows
    variant_rows = [row for row in rows if isinstance(row, Mapping)]
    if not variant_rows:
        return rows
    semantic_families_available = any(
        normalize_variant_family(row.get("variant_family")) in SEMANTIC_VARIANT_FAMILIES
        for row in variant_rows
    )
    filtered = [
        dict(row)
        for row in variant_rows
        if _variant_family_allowed_for_cluster(
            relation_plan=relation_plan,
            cluster_id=cluster_id,
            variant_family=str(row.get("variant_family") or ""),
            semantic_families_available=semantic_families_available,
        )
    ]
    return filtered or rows


def _filter_clusters_outlines_by_concept_variant_policy(
    clusters_outlines: Any,
    relation_plan: Mapping[str, Any] | None,
) -> Any:
    if isinstance(clusters_outlines, dict):
        clusters = clusters_outlines.get("clusters")
        if isinstance(clusters, list):
            out = dict(clusters_outlines)
            out_clusters = []
            for item in clusters:
                if not isinstance(item, dict):
                    out_clusters.append(item)
                    continue
                cluster_id = str(item.get("cluster_id") or "").strip()
                next_item = dict(item)
                next_item["variant_bundle"] = _filter_variant_bundle_rows_for_concept(
                    cluster_id=cluster_id,
                    rows=item.get("variant_bundle"),
                    relation_plan=relation_plan,
                )
                out_clusters.append(next_item)
            out["clusters"] = out_clusters
            return out

        out: dict[str, Any] = {}
        for cluster_id, payload in clusters_outlines.items():
            if not isinstance(payload, dict):
                out[cluster_id] = payload
                continue
            next_payload = dict(payload)
            next_payload["variant_bundle"] = _filter_variant_bundle_rows_for_concept(
                cluster_id=str(cluster_id),
                rows=payload.get("variant_bundle"),
                relation_plan=relation_plan,
            )
            out[cluster_id] = next_payload
        return out

    if isinstance(clusters_outlines, list):
        out_list = []
        for item in clusters_outlines:
            if not isinstance(item, dict):
                out_list.append(item)
                continue
            cluster_id = str(item.get("cluster_id") or "").strip()
            next_item = dict(item)
            next_item["variant_bundle"] = _filter_variant_bundle_rows_for_concept(
                cluster_id=cluster_id,
                rows=item.get("variant_bundle"),
                relation_plan=relation_plan,
            )
            out_list.append(next_item)
        return out_list

    return clusters_outlines


def _filter_protected_topology_candidates(
    candidates_by_cluster: dict[str, list[Candidate]],
    relation_plan: Mapping[str, Any] | None,
) -> dict[str, list[Candidate]]:
    if not _center_openness_is_strong(relation_plan):
        return candidates_by_cluster
    profiles, _adjacency = _build_cluster_macro_profiles(
        dict(relation_plan) if isinstance(relation_plan, dict) else None,
        cluster_ids=list(candidates_by_cluster.keys()),
    )
    out: dict[str, list[Candidate]] = {}
    for cluster_id, candidates in candidates_by_cluster.items():
        profile = profiles.get(cluster_id, {})
        filtered = list(candidates)
        if _cluster_avoids_center(profile):
            non_center = [
                candidate
                for candidate in filtered
                if not _anchor_kind_is_center(candidate.anchor_kind)
            ]
            if non_center:
                filtered = non_center
        if _cluster_avoids_entry(profile):
            non_entry = [
                candidate
                for candidate in filtered
                if not _anchor_kind_is_entry(candidate.anchor_kind)
            ]
            if non_entry:
                filtered = non_entry
        out[cluster_id] = filtered
    return out


def _solution_signature(
    cluster_transforms: Sequence[dict[str, Any]],
    selected_variants: Sequence[dict[str, str]],
) -> str:
    variant_pairs = sorted(
        (
            str(item.get("cluster_id") or ""),
            str(item.get("variant_id") or ""),
        )
        for item in selected_variants
        if isinstance(item, dict)
    )
    transform_pairs = sorted(
        (
            str(item.get("cluster_id") or ""),
            _as_int(item.get("x")),
            _as_int(item.get("y")),
            _as_int(item.get("rot")) % 360,
        )
        for item in cluster_transforms
        if isinstance(item, dict)
    )
    return json.dumps(
        {"variants": variant_pairs, "transforms": transform_pairs},
        sort_keys=True,
        separators=(",", ":"),
    )


def _dominant_anchor_correct(
    *,
    relation_plan: Mapping[str, Any] | None,
    chosen: Mapping[str, Candidate],
) -> bool:
    primary_cluster_id = _extract_layout_primary_cluster_id(
        dict(relation_plan) if isinstance(relation_plan, dict) else None
    )
    if primary_cluster_id is None:
        return True
    candidate = chosen.get(primary_cluster_id)
    if candidate is None:
        return False
    profiles, _adjacency = _build_cluster_macro_profiles(
        dict(relation_plan) if isinstance(relation_plan, dict) else None,
        cluster_ids=[primary_cluster_id],
    )
    profile = profiles.get(primary_cluster_id, {})
    if _cluster_prefers_wall(profile):
        return _anchor_kind_is_wall_pinned(candidate.anchor_kind)
    if _cluster_prefers_center(profile):
        return _anchor_kind_is_center(candidate.anchor_kind)
    return True


def _solution_diagnostics(
    *,
    verify: Mapping[str, Any] | None,
    relation_plan: Mapping[str, Any] | None,
    chosen: Mapping[str, Candidate],
) -> dict[str, bool]:
    quality = _quality_dict(verify)
    circulation_penalty = _as_int(quality.get("circulation_penalty_mm"))
    critical_penalty = _as_int(quality.get("critical_orientation_penalty_mm"))
    center_strength = _center_openness_strength(relation_plan)
    center_is_strong = center_strength in {"high", "very_high"}
    center_ok_threshold = 120 if center_is_strong else 260
    return {
        "circulation_clear": circulation_penalty <= 80,
        "entry_preserved": circulation_penalty <= 120,
        "center_openness_preserved": circulation_penalty <= center_ok_threshold,
        "dominant_anchor_correct": _dominant_anchor_correct(
            relation_plan=relation_plan,
            chosen=chosen,
        ),
        "workflow_preserved": critical_penalty <= 220,
    }


def _quality_gate_reasons_from_verify(
    verify: Mapping[str, Any] | None,
) -> list[str]:
    if not isinstance(verify, Mapping):
        return []
    quality_gate = verify.get("quality_gate")
    if not isinstance(quality_gate, Mapping):
        return []
    return [
        str(reason or "")
        for reason in (quality_gate.get("reasons") or [])
        if str(reason or "").strip()
    ]


def _quality_gate_reasons_from_solution(solution: Mapping[str, Any]) -> list[str]:
    verify = solution.get("_verify")
    if isinstance(verify, Mapping):
        reasons = _quality_gate_reasons_from_verify(verify)
        if reasons:
            return reasons
    summary = solution.get("verify_summary")
    if not isinstance(summary, Mapping):
        return []
    return [
        str(reason or "")
        for reason in (summary.get("quality_gate_reasons") or [])
        if str(reason or "").strip()
    ]


def _selected_variant_looks_like_family(
    selected_variant: Mapping[str, Any],
    family: str,
) -> bool:
    family_token = _normalized_policy_token(family)
    variant_family = _normalized_policy_token(selected_variant.get("variant_family"))
    variant_id = _normalized_policy_token(selected_variant.get("variant_id"))
    return variant_family == family_token or family_token in variant_id


def _protected_topology_violations_from_verify(
    verify: Mapping[str, Any] | None,
    relation_plan: Mapping[str, Any] | None,
) -> list[str]:
    if not isinstance(verify, Mapping):
        return []
    quality = _quality_dict(verify)
    circulation_debug = quality.get("circulation_debug")
    if not isinstance(circulation_debug, list):
        return []

    concept = _concept_from_relation_plan(relation_plan)
    topology_policy = concept.get("topology_policy")
    preserve_entry = True
    preserve_corridor = True
    if isinstance(topology_policy, Mapping):
        preserve_entry = bool(topology_policy.get("preserve_entry_landing", True))
        preserve_corridor = bool(topology_policy.get("preserve_primary_corridor", True))

    violations: list[str] = []
    for item in circulation_debug:
        if not isinstance(item, Mapping):
            continue
        penalty = _as_int(item.get("penalty_mm"))
        if penalty <= 0:
            continue
        kind = str(item.get("kind") or "").strip()
        if kind == "entry_buffer" and preserve_entry:
            violations.append("entry_landing_zone_blocked")
        elif kind == "main_path" and preserve_corridor:
            violations.append("primary_circulation_corridor_blocked")
        elif kind == "center_lane" and _center_openness_is_very_high(relation_plan):
            violations.append("center_openness_region_blocked")
    return sorted(set(violations))


def _solution_publishability_violations(
    solution: Mapping[str, Any],
    *,
    relation_plan: Mapping[str, Any] | None,
) -> list[str]:
    violations: list[str] = []
    if not bool(solution.get("hard_valid")):
        violations.append("not_hard_valid")

    gate_reasons = set(_quality_gate_reasons_from_solution(solution))
    violations.extend(sorted(gate_reasons & PUBLISHABLE_BLOCKING_QUALITY_REASONS))

    diagnostics_raw = solution.get("diagnostics")
    diagnostics = diagnostics_raw if isinstance(diagnostics_raw, Mapping) else {}
    for key in ("circulation_clear", "entry_preserved", "workflow_preserved"):
        if diagnostics.get(key) is False:
            violations.append(f"{key}_false")
    if (
        _center_openness_is_strong(relation_plan)
        and diagnostics.get("center_openness_preserved") is False
    ):
        violations.append("center_openness_preserved_false")

    verify = solution.get("_verify")
    if isinstance(verify, Mapping):
        violations.extend(
            _protected_topology_violations_from_verify(verify, relation_plan)
        )

    selected_variants = [
        item
        for item in (solution.get("selected_variants") or [])
        if isinstance(item, Mapping)
    ]
    for item in selected_variants:
        cluster_id = str(item.get("cluster_id") or "").strip()
        if not cluster_id:
            continue
        if not _selected_variant_looks_like_family(item, "fallback_generic"):
            continue
        if _concept_allows_fallback_generic(relation_plan, cluster_id):
            continue
        if _cluster_is_core_or_protected(relation_plan, cluster_id):
            violations.append("fallback_generic_core_variant")

    return sorted(set(violations))


def _is_publishable_solution(
    solution: Mapping[str, Any],
    *,
    relation_plan: Mapping[str, Any] | None,
) -> bool:
    return not _solution_publishability_violations(
        solution,
        relation_plan=relation_plan,
    )


def _global_quality_scores(
    *,
    verify: Mapping[str, Any] | None,
    relation_plan: Mapping[str, Any] | None,
    chosen: Mapping[str, Candidate],
) -> dict[str, float]:
    quality = _quality_dict(verify)
    summary = _summary_dict(verify)

    circulation = _as_int(quality.get("circulation_penalty_mm"))
    tight_gap = _as_int(quality.get("tight_gap_penalty_mm"))
    affinity = _as_int(quality.get("affinity_penalty_mm"))
    relation = _as_int(quality.get("relation_penalty_mm"))
    spread = _as_int(quality.get("spread_penalty_mm"))
    orientation = _as_int(quality.get("orientation_penalty_mm"))
    critical = _as_int(quality.get("critical_orientation_penalty_mm"))
    focal = _as_int(quality.get("focal_pair_penalty_mm"))
    macro = _as_int(quality.get("macro_penalty_mm"))
    micro = _as_int(quality.get("micro_penalty_mm"))

    density = _as_float(quality.get("density_ratio"), 0.28)
    target_density = _style_target_density_ratio(relation_plan)
    density_score = _clamp_score(1.0 - (abs(density - target_density) / 0.32))
    min_gap = summary.get("min_cluster_gap_mm")
    gap_score = 0.72 if min_gap is None else _clamp_score(_as_float(min_gap) / 450.0)

    diagnostics = _solution_diagnostics(
        verify=verify,
        relation_plan=relation_plan,
        chosen=chosen,
    )
    dominant_anchor_score = 1.0 if diagnostics["dominant_anchor_correct"] else 0.35
    concept_family = _concept_family_from_relation_plan(relation_plan)
    compatibility_values = [
        _variant_family_compatibility_score(
            concept_family=concept_family,
            role_kind=_cluster_role_kind(relation_plan, cluster_id),
            variant_family=candidate.variant_family,
        )
        for cluster_id, candidate in chosen.items()
    ]
    compatibility_score = _clamp_score(
        0.5 + (sum(compatibility_values) / max(1, len(compatibility_values))) / 2.4
    )
    gate_reasons = set(_quality_gate_reasons_from_verify(verify))
    primary_pair_ok = "focal_pair_penalty_too_high" not in gate_reasons
    critical_items_ok = "critical_item_penalty_too_high" not in gate_reasons

    functionality = (
        0.34 * _penalty_score(circulation, 420.0)
        + 0.22 * _penalty_score(tight_gap, 520.0)
        + 0.24 * _penalty_score(critical, 220.0)
        + 0.12 * _penalty_score(micro, 520.0)
        + 0.08 * gap_score
    )
    naturalness = (
        0.28 * _penalty_score(affinity, 520.0)
        + 0.24 * _penalty_score(spread, 680.0)
        + 0.18 * _penalty_score(orientation, 760.0)
        + 0.16 * _penalty_score(macro, 900.0)
        + 0.14 * density_score
    )
    semantic = (
        0.20 * _penalty_score(relation, 520.0)
        + 0.15 * _penalty_score(affinity, 520.0)
        + 0.20 * _penalty_score(critical, 220.0)
        + 0.18 * _penalty_score(focal, 220.0)
        + 0.08 * dominant_anchor_score
        + 0.08 * compatibility_score
        + 0.06 * (1.0 if primary_pair_ok else 0.0)
        + 0.05 * (1.0 if critical_items_ok else 0.0)
    )
    spatial = (
        0.24 * _penalty_score(tight_gap, 520.0)
        + 0.22 * _penalty_score(spread, 680.0)
        + 0.18 * _penalty_score(circulation, 500.0)
        + 0.18 * density_score
        + 0.18 * gap_score
    )

    if not (isinstance(verify, Mapping) and bool(verify.get("hard_valid"))):
        functionality *= 0.35
        semantic *= 0.45
        spatial *= 0.45

    quality_weights = _style_quality_weights(relation_plan)
    total = (
        quality_weights["functionality"] * functionality
        + quality_weights["naturalness"] * naturalness
        + quality_weights["semantic"] * semantic
        + quality_weights["spatial"] * spatial
    )
    return {
        "functionality_score": round(_clamp_score(functionality), 3),
        "naturalness_score": round(_clamp_score(naturalness), 3),
        "semantic_coherence_score": round(_clamp_score(semantic), 3),
        "spatial_quality_score": round(_clamp_score(spatial), 3),
        "total_score": round(_clamp_score(total), 3),
    }


def _style_target_density_ratio(relation_plan: Mapping[str, Any] | None) -> float:
    concept = _concept_from_relation_plan(relation_plan)
    topology_policy = concept.get("topology_policy")
    if not isinstance(topology_policy, Mapping):
        return 0.28
    text = str(topology_policy.get("style_density_target") or "").lower()
    if "low" in text:
        return 0.24
    if "medium_high" in text or "high" in text:
        return 0.34
    if "medium" in text:
        return 0.31
    return 0.28


def _style_quality_weights(relation_plan: Mapping[str, Any] | None) -> dict[str, float]:
    weights = dict(QUALITY_WEIGHTS)
    concept = _concept_from_relation_plan(relation_plan)
    topology_policy = concept.get("topology_policy")
    if not isinstance(topology_policy, Mapping):
        return weights
    reserve_center = str(topology_policy.get("reserve_center_degree") or "")
    if reserve_center in {"high", "very_high"}:
        weights["naturalness"] += 0.03
        weights["spatial"] += 0.03
        weights["functionality"] -= 0.03
        weights["semantic"] -= 0.03
    visual_balance = str(topology_policy.get("style_visual_balance") or "")
    if visual_balance in {"formal", "calm"}:
        weights["semantic"] += 0.02
        weights["naturalness"] += 0.02
        weights["functionality"] -= 0.02
        weights["spatial"] -= 0.02
    total = sum(weights.values()) or 1.0
    return {key: value / total for key, value in weights.items()}


def _normalized_text_set(values: Any) -> set[str]:
    out: set[str] = set()
    if not isinstance(values, list):
        return out
    for item in values:
        text = str(item or "").strip().lower()
        if text:
            out.add(text)
    return out


def _string_sequence(values: Any) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, str):
        return []
    out: list[str] = []
    for item in values:
        text = str(item or "").strip()
        if text:
            out.append(text)
    return out


def _dedupe_string_sequence(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _relation_priority_weight(value: Any) -> float:
    text = str(value or "").strip().lower()
    if text == "critical":
        return 1.4
    if text == "high":
        return 1.0
    if text == "medium":
        return 0.7
    if text == "low":
        return 0.4
    return 0.8


def _cluster_relation_weight(relation: str) -> float:
    lowered = relation.strip().lower()
    if lowered == "near":
        return 4.0
    if lowered in {"adjacent", "next_to"}:
        return 3.4
    if lowered in {"aligned", "parallel"}:
        return 3.0
    return 2.6


def _cluster_directional_weight(relation: str) -> float:
    lowered = relation.strip().lower()
    if lowered == "face_each_other":
        return 8.0
    if lowered in {"face_same_direction", "axis_parallel"}:
        return 5.5
    if lowered in {"perpendicular", "orthogonal"}:
        return 2.4
    return 4.0


def _extract_layout_primary_cluster_id(
    relation_plan: dict[str, Any] | None,
) -> str | None:
    if not isinstance(relation_plan, dict):
        return None
    layout_intent = relation_plan.get("layout_intent_profile")
    if not isinstance(layout_intent, dict):
        return None
    cluster_id = str(layout_intent.get("primary_cluster_id") or "").strip()
    return cluster_id or None


def _extract_layout_secondary_cluster_id(
    relation_plan: dict[str, Any] | None,
) -> str | None:
    if not isinstance(relation_plan, dict):
        return None
    layout_intent = relation_plan.get("layout_intent_profile")
    if not isinstance(layout_intent, dict):
        return None
    cluster_id = str(layout_intent.get("secondary_cluster_id") or "").strip()
    return cluster_id or None


def _normalized_cluster_pair(a: str, b: str) -> tuple[str, str] | None:
    left = str(a or "").strip()
    right = str(b or "").strip()
    if not left or not right or left == right:
        return None
    return tuple(sorted((left, right)))


def _viewing_contract_pairs(
    relation_plan: dict[str, Any] | None,
) -> set[tuple[str, str]]:
    if not isinstance(relation_plan, dict):
        return set()

    pairs: set[tuple[str, str]] = set()
    primary_cluster_id = _extract_layout_primary_cluster_id(relation_plan)
    secondary_cluster_id = _extract_layout_secondary_cluster_id(relation_plan)
    layout_intent = relation_plan.get("layout_intent_profile") or {}
    focus_mode = str(layout_intent.get("focus_mode") or "").strip().lower()
    if focus_mode == "viewing":
        pair = _normalized_cluster_pair(
            primary_cluster_id or "",
            secondary_cluster_id or "",
        )
        if pair is not None:
            pairs.add(pair)

    for row in relation_plan.get("cluster_directional_relations") or []:
        if not isinstance(row, dict):
            continue
        relation = str(row.get("relation") or "").strip().lower()
        if relation != "face_each_other":
            continue
        pair = _normalized_cluster_pair(
            str(row.get("a") or ""),
            str(row.get("b") or ""),
        )
        if pair is not None:
            pairs.add(pair)

    return pairs


def _viewing_contract_cluster_ids(
    relation_plan: dict[str, Any] | None,
) -> set[str]:
    cluster_ids: set[str] = set()
    for left, right in _viewing_contract_pairs(relation_plan):
        cluster_ids.add(left)
        cluster_ids.add(right)
    return cluster_ids


def _cluster_face_target_id(
    relation_plan: dict[str, Any] | None,
    cluster_id: str,
) -> str | None:
    if not isinstance(relation_plan, dict):
        return None
    target: str | None = None
    for row in relation_plan.get("cluster_orientations") or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("cluster_id") or "").strip() != cluster_id:
            continue
        intents = _normalized_text_set(row.get("intents"))
        if "face_cluster" not in intents:
            continue
        target = str(row.get("target_cluster_id") or "").strip() or None
        if target is not None:
            return target
    return None


def _build_cluster_macro_profiles(
    relation_plan: dict[str, Any] | None,
    *,
    cluster_ids: Sequence[str],
) -> tuple[dict[str, dict[str, set[str]]], dict[str, dict[str, float]]]:
    clean_cluster_ids = [str(cluster_id or "").strip() for cluster_id in cluster_ids]
    clean_cluster_ids = [cluster_id for cluster_id in clean_cluster_ids if cluster_id]
    profiles = {
        cluster_id: {"prefer": set(), "avoid": set(), "intents": set()}
        for cluster_id in clean_cluster_ids
    }
    adjacency: dict[str, dict[str, float]] = {
        cluster_id: {} for cluster_id in clean_cluster_ids
    }
    if not isinstance(relation_plan, dict):
        return profiles, adjacency

    def ensure(cluster_id: str) -> dict[str, set[str]] | None:
        if cluster_id not in profiles:
            return None
        return profiles[cluster_id]

    def add_edge(a: str, b: str, weight: float) -> None:
        if a == b or a not in adjacency or b not in adjacency or weight <= 0.0:
            return
        adjacency[a][b] = adjacency[a].get(b, 0.0) + float(weight)
        adjacency[b][a] = adjacency[b].get(a, 0.0) + float(weight)

    for row in relation_plan.get("cluster_affinities") or []:
        if not isinstance(row, dict):
            continue
        cluster_id = str(row.get("cluster_id") or "").strip()
        slot = ensure(cluster_id)
        if slot is None:
            continue
        slot["prefer"].update(_normalized_text_set(row.get("prefer")))
        slot["avoid"].update(_normalized_text_set(row.get("avoid")))

    _apply_macro_concept_profiles(
        relation_plan,
        profiles=profiles,
    )

    for row in relation_plan.get("cluster_orientations") or []:
        if not isinstance(row, dict):
            continue
        cluster_id = str(row.get("cluster_id") or "").strip()
        slot = ensure(cluster_id)
        if slot is None:
            continue
        intents = _normalized_text_set(row.get("intents"))
        slot["intents"].update(intents)
        target_cluster_id = str(row.get("target_cluster_id") or "").strip()
        if target_cluster_id and "face_cluster" in intents:
            add_edge(
                cluster_id,
                target_cluster_id,
                5.5 * _relation_priority_weight(row.get("priority")),
            )

    for row in relation_plan.get("cluster_relations") or []:
        if not isinstance(row, dict):
            continue
        a = str(row.get("a") or "").strip()
        b = str(row.get("b") or "").strip()
        if not a or not b:
            continue
        add_edge(
            a,
            b,
            _cluster_relation_weight(str(row.get("relation") or ""))
            * _relation_priority_weight(row.get("priority")),
        )

    for row in relation_plan.get("cluster_directional_relations") or []:
        if not isinstance(row, dict):
            continue
        a = str(row.get("a") or "").strip()
        b = str(row.get("b") or "").strip()
        if not a or not b:
            continue
        add_edge(
            a,
            b,
            _cluster_directional_weight(str(row.get("relation") or ""))
            * _relation_priority_weight(row.get("priority")),
        )
    return profiles, adjacency


def _apply_macro_concept_profiles(
    relation_plan: dict[str, Any],
    *,
    profiles: dict[str, dict[str, set[str]]],
) -> None:
    macro_concept = relation_plan.get("macro_concept")
    if not isinstance(macro_concept, dict):
        return

    topology_policy = macro_concept.get("topology_policy")
    reserve_center = False
    if isinstance(topology_policy, dict):
        reserve_center = str(
            topology_policy.get("reserve_center_degree") or ""
        ).strip() in {"high", "very_high"}

    zone_plan = macro_concept.get("cluster_zone_plan")
    if not isinstance(zone_plan, list):
        return

    for row in zone_plan:
        if not isinstance(row, dict):
            continue
        cluster_id = str(row.get("cluster_id") or "").strip()
        profile = profiles.get(cluster_id)
        if profile is None:
            continue

        zone_assignment = str(row.get("zone_assignment") or "").strip()
        wall_claim = str(row.get("wall_claim") or "").strip()
        center_usage = str(row.get("center_usage") or "").strip()
        entry_relation = str(row.get("entry_relation") or "").strip()
        daylight_relation = str(row.get("daylight_relation") or "").strip()
        placement_behavior = _placement_behavior_from_rows(row)
        wall_backing = _placement_behavior_value(placement_behavior, "wall_backing")
        front_space = _placement_behavior_value(placement_behavior, "front_space")
        daylight_blocking = _placement_behavior_value(
            placement_behavior,
            "daylight_blocking",
        )

        if (
            wall_backing in {"preferred", "required"}
            or wall_claim in {"strong", "medium"}
            or any(token in zone_assignment for token in ("wall", "edge", "storage"))
        ):
            profile["prefer"].update({"wall", "recess_or_edge"})
            profile["intents"].add("back_to_wall")
        if wall_backing == "required" or wall_claim == "strong":
            profile["prefer"].add("long_wall")
            profile["intents"].add("axis_parallel_wall")
        if front_space in {"preferred", "required"}:
            profile["intents"].add("access_to_open_space")
        if (
            daylight_blocking == "prefer"
            or "daylight" in zone_assignment
            or daylight_relation
            in {
                "claim_daylight",
                "daylight_preferred",
            }
        ):
            profile["prefer"].add("window_side")
        if daylight_blocking == "avoid" or daylight_relation == "avoid_window_blocking":
            profile["avoid"].add("window_blocking")
        if (
            "private" in zone_assignment
            or entry_relation == "avoid_direct_entry_conflict"
        ):
            profile["prefer"].add("far_from_entry")
            profile["avoid"].add("entry_blocking")
        if reserve_center or center_usage in {"none", "open_reserved"}:
            profile["avoid"].add("center")
        elif center_usage in {"partial", "primary"}:
            profile["prefer"].add("center")


def _cluster_prefers_wall(profile: Mapping[str, set[str]]) -> bool:
    prefer = set(profile.get("prefer") or set())
    intents = set(profile.get("intents") or set())
    return bool((prefer & _CLUSTER_WALL_PREFER_TAGS) or {"back_to_wall"} & intents)


def _cluster_prefers_center(profile: Mapping[str, set[str]]) -> bool:
    prefer = set(profile.get("prefer") or set())
    return bool(prefer & _CLUSTER_CENTER_PREFER_TAGS)


def _cluster_avoids_center(profile: Mapping[str, set[str]]) -> bool:
    avoid = set(profile.get("avoid") or set())
    return bool(avoid & _CLUSTER_CENTER_AVOID_TAGS)


def _cluster_avoids_window(profile: Mapping[str, set[str]]) -> bool:
    avoid = set(profile.get("avoid") or set())
    return bool(avoid & _CLUSTER_WINDOW_AVOID_TAGS)


def _cluster_avoids_entry(profile: Mapping[str, set[str]]) -> bool:
    avoid = set(profile.get("avoid") or set())
    prefer = set(profile.get("prefer") or set())
    return bool(avoid & _CLUSTER_ENTRY_AVOID_TAGS) or ("far_from_entry" in prefer)


def _cluster_prefers_window(profile: Mapping[str, set[str]]) -> bool:
    prefer = set(profile.get("prefer") or set())
    intents = set(profile.get("intents") or set())
    return bool(prefer & _CLUSTER_WINDOW_PREFER_TAGS) or ("face_window" in intents)


def _cluster_prefers_entry(profile: Mapping[str, set[str]]) -> bool:
    prefer = set(profile.get("prefer") or set())
    intents = set(profile.get("intents") or set())
    return bool(prefer & _CLUSTER_ENTRY_PREFER_TAGS) or ("face_entry" in intents)


def _anchor_kind_contains(anchor_kind: str, tokens: Sequence[str]) -> bool:
    lowered = anchor_kind.strip().lower()
    return any(token in lowered for token in tokens)


def _anchor_kind_is_wall_pinned(anchor_kind: str) -> bool:
    return _anchor_kind_contains(anchor_kind, _WALL_ANCHOR_TOKENS)


def _anchor_kind_is_center(anchor_kind: str) -> bool:
    return _anchor_kind_contains(anchor_kind, _CENTER_ANCHOR_TOKENS)


def _anchor_kind_is_window(anchor_kind: str) -> bool:
    return _anchor_kind_contains(anchor_kind, _WINDOW_ANCHOR_TOKENS)


def _anchor_kind_is_entry(anchor_kind: str) -> bool:
    return _anchor_kind_contains(anchor_kind, _ENTRY_ANCHOR_TOKENS)


def _cluster_root_score(
    *,
    cluster_id: str,
    profile: Mapping[str, set[str]],
    adjacency: Mapping[str, Mapping[str, float]],
    candidate_count: int,
    original_index: int,
    primary_cluster_id: str | None,
    secondary_cluster_id: str | None,
) -> float:
    prefer = set(profile.get("prefer") or set())
    avoid = set(profile.get("avoid") or set())
    intents = set(profile.get("intents") or set())
    score = 0.0
    if cluster_id == primary_cluster_id:
        score += 120.0
    if cluster_id == secondary_cluster_id:
        score += 26.0
    if "back_to_wall" in intents:
        score += 22.0
    if "wall" in prefer:
        score += 16.0
    if "long_wall" in prefer:
        score += 12.0
    if "short_wall" in prefer:
        score += 6.0
    if "recess_or_edge" in prefer:
        score += 6.0
    if "access_to_open_space" in intents:
        score += 5.0
    if "inward_to_room" in intents:
        score += 4.0
    if "far_from_entry" in prefer:
        score += 4.0
    score += min(18.0, 3.0 * float(len(avoid)))
    score += min(34.0, float(sum(adjacency.get(cluster_id, {}).values())))
    score += min(14.0, 32.0 / max(2.0, float(candidate_count)))
    score += max(0.0, 3.0 - (0.18 * float(original_index)))
    return score


def _select_root_cluster_id(
    candidates_by_cluster: dict[str, list[Candidate]],
    *,
    relation_plan: dict[str, Any] | None,
    cluster_order: Sequence[str] | None = None,
) -> str | None:
    ordered_ids = _ordered_cluster_ids(
        candidates_by_cluster, cluster_order=cluster_order
    )
    if not ordered_ids:
        return None
    primary_cluster_id = _extract_layout_primary_cluster_id(relation_plan)
    if primary_cluster_id in candidates_by_cluster:
        return primary_cluster_id

    secondary_cluster_id = _extract_layout_secondary_cluster_id(relation_plan)
    profiles, adjacency = _build_cluster_macro_profiles(
        relation_plan,
        cluster_ids=ordered_ids,
    )
    original_index = {cluster_id: idx for idx, cluster_id in enumerate(ordered_ids)}
    return max(
        ordered_ids,
        key=lambda cluster_id: (
            _cluster_root_score(
                cluster_id=cluster_id,
                profile=profiles.get(cluster_id, {}),
                adjacency=adjacency,
                candidate_count=len(candidates_by_cluster.get(cluster_id, ())),
                original_index=original_index.get(cluster_id, 0),
                primary_cluster_id=primary_cluster_id,
                secondary_cluster_id=secondary_cluster_id,
            ),
            -original_index.get(cluster_id, 0),
        ),
    )


def _root_cluster_first_order(
    candidates_by_cluster: dict[str, list[Candidate]],
    *,
    relation_plan: dict[str, Any] | None,
    cluster_order: Sequence[str] | None = None,
) -> list[str]:
    ordered_ids = _ordered_cluster_ids(
        candidates_by_cluster, cluster_order=cluster_order
    )
    if not ordered_ids:
        return []
    root_cluster_id = _select_root_cluster_id(
        candidates_by_cluster,
        relation_plan=relation_plan,
        cluster_order=cluster_order,
    )
    if root_cluster_id is None:
        return ordered_ids

    profiles, adjacency = _build_cluster_macro_profiles(
        relation_plan,
        cluster_ids=ordered_ids,
    )
    primary_cluster_id = _extract_layout_primary_cluster_id(relation_plan)
    secondary_cluster_id = _extract_layout_secondary_cluster_id(relation_plan)
    original_index = {cluster_id: idx for idx, cluster_id in enumerate(ordered_ids)}
    root_scores = {
        cluster_id: _cluster_root_score(
            cluster_id=cluster_id,
            profile=profiles.get(cluster_id, {}),
            adjacency=adjacency,
            candidate_count=len(candidates_by_cluster.get(cluster_id, ())),
            original_index=original_index.get(cluster_id, 0),
            primary_cluster_id=primary_cluster_id,
            secondary_cluster_id=secondary_cluster_id,
        )
        for cluster_id in ordered_ids
    }

    placed = [root_cluster_id]
    placed_set = {root_cluster_id}
    remaining = {
        cluster_id for cluster_id in ordered_ids if cluster_id != root_cluster_id
    }
    while remaining:
        connected: list[tuple[float, float, int, str]] = []
        for cluster_id in remaining:
            edge_strength = max(
                (
                    float(adjacency.get(cluster_id, {}).get(parent_id, 0.0))
                    for parent_id in placed_set
                ),
                default=0.0,
            )
            if edge_strength <= 0.0:
                continue
            connected.append(
                (
                    edge_strength,
                    root_scores.get(cluster_id, 0.0),
                    -original_index.get(cluster_id, 0),
                    cluster_id,
                )
            )
        if connected:
            next_cluster_id = max(connected)[3]
        else:
            next_cluster_id = max(
                remaining,
                key=lambda cluster_id: (
                    root_scores.get(cluster_id, 0.0),
                    -original_index.get(cluster_id, 0),
                ),
            )
        placed.append(next_cluster_id)
        placed_set.add(next_cluster_id)
        remaining.remove(next_cluster_id)
    return placed


def _candidate_macro_bonus(
    candidate: Candidate,
    *,
    profile: Mapping[str, set[str]],
    is_root_cluster: bool,
) -> int:
    scale = 2 if is_root_cluster else 1
    bonus = 0
    anchor_kind = str(candidate.anchor_kind or "")
    if _cluster_prefers_wall(profile):
        wall_weight = 1600 * scale
        if _anchor_kind_is_wall_pinned(anchor_kind):
            bonus += wall_weight
            if "long_wall" in set(profile.get("prefer") or set()):
                bonus += 400 * scale
        else:
            bonus -= wall_weight // 2
    if _cluster_avoids_center(profile) and _anchor_kind_is_center(anchor_kind):
        bonus -= 800 * scale
    if _cluster_prefers_center(profile) and _anchor_kind_is_center(anchor_kind):
        bonus += 500 * scale
    if _cluster_avoids_window(profile) and _anchor_kind_is_window(anchor_kind):
        bonus -= 700 * scale
    if _cluster_prefers_window(profile) and _anchor_kind_is_window(anchor_kind):
        bonus += 500 * scale
    if _cluster_avoids_entry(profile) and _anchor_kind_is_entry(anchor_kind):
        bonus -= 750 * scale
    if _cluster_prefers_entry(profile) and _anchor_kind_is_entry(anchor_kind):
        bonus += 450 * scale
    return bonus


def _candidate_variant_family_bonus(
    candidate: Candidate,
    *,
    relation_plan: Mapping[str, Any] | None,
) -> int:
    compatibility = _variant_family_compatibility_score(
        concept_family=_concept_family_from_relation_plan(relation_plan),
        role_kind=_cluster_role_kind(relation_plan, candidate.cluster_id),
        variant_family=candidate.variant_family,
    )
    scale = (
        2400
        if _cluster_is_core_or_protected(relation_plan, candidate.cluster_id)
        else 1300
    )
    return int(round(compatibility * scale))


def _has_orientation_sensitive_wall_contract(profile: Mapping[str, set[str]]) -> bool:
    intents = set(profile.get("intents") or set())
    return bool(
        intents
        & {"back_to_wall", "face_cluster", "access_to_open_space", "inward_to_room"}
    )


def _has_hard_root_wall_contract(
    *,
    relation_plan: dict[str, Any] | None,
    root_cluster_id: str | None,
    profile: Mapping[str, set[str]],
) -> bool:
    if not root_cluster_id or not _cluster_prefers_wall(profile):
        return False
    intents = set(profile.get("intents") or set())
    target_cluster_id = _cluster_face_target_id(relation_plan, root_cluster_id)
    if "back_to_wall" not in intents or not target_cluster_id:
        return False
    if "face_cluster" not in intents:
        return False
    protected_cluster_ids = _viewing_contract_cluster_ids(relation_plan)
    return (
        root_cluster_id in protected_cluster_ids
        or target_cluster_id in protected_cluster_ids
        or target_cluster_id == _extract_layout_secondary_cluster_id(relation_plan)
    )


def _orientation_focus_candidates(
    candidates: Sequence[Candidate],
) -> list[Candidate]:
    if not candidates:
        return []
    best_critical = min(
        int(candidate.critical_orientation_penalty_mm) for candidate in candidates
    )
    best_focal = min(
        int(candidate.focal_orientation_penalty_mm) for candidate in candidates
    )
    best_orientation = min(
        int(candidate.orientation_penalty_mm) for candidate in candidates
    )
    focused = [
        candidate
        for candidate in candidates
        if candidate.acceptable_valid
        or (
            int(candidate.critical_orientation_penalty_mm) <= best_critical + 120
            and int(candidate.focal_orientation_penalty_mm) <= best_focal + 160
            and int(candidate.orientation_penalty_mm) <= best_orientation + 240
        )
    ]
    return focused or list(candidates)


def _hard_contract_focus_candidates(
    candidates: Sequence[Candidate],
) -> list[Candidate]:
    if not candidates:
        return []
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            0 if candidate.acceptable_valid else 1,
            int(candidate.critical_orientation_penalty_mm),
            int(candidate.focal_orientation_penalty_mm),
            int(candidate.orientation_penalty_mm),
            -int(candidate.rough_score),
            candidate.variant_id,
            candidate.rot,
            candidate.x,
            candidate.y,
        ),
    )
    best_critical = int(ranked[0].critical_orientation_penalty_mm)
    best_focal = int(ranked[0].focal_orientation_penalty_mm)
    best_orientation = int(ranked[0].orientation_penalty_mm)
    acceptable = [
        candidate
        for candidate in ranked
        if candidate.acceptable_valid
        and int(candidate.critical_orientation_penalty_mm) <= best_critical + 40
        and int(candidate.focal_orientation_penalty_mm) <= best_focal + 80
        and int(candidate.orientation_penalty_mm) <= best_orientation + 120
    ]
    if acceptable:
        return acceptable[:12]
    focused = [
        candidate
        for candidate in ranked
        if int(candidate.critical_orientation_penalty_mm) <= best_critical + 20
        and int(candidate.focal_orientation_penalty_mm) <= best_focal + 40
        and int(candidate.orientation_penalty_mm) <= best_orientation + 60
    ]
    return (focused or ranked)[:8]


def _wall_pin_focus_next_action(
    *,
    round_strategy: Mapping[str, Any] | None,
) -> str:
    if not isinstance(round_strategy, Mapping):
        return "none"
    if not bool(round_strategy.get("used_wall_pin_focus")):
        return "none"
    return "refine_with_focus"


def _prepare_round_candidates(
    candidates_by_cluster: dict[str, list[Candidate]],
    *,
    relation_plan: dict[str, Any] | None,
    cluster_order: Sequence[str] | None = None,
    wall_pin_focus: bool,
) -> tuple[
    dict[str, list[Candidate]],
    list[str],
    dict[tuple[str, str, int, int, int], int],
    dict[str, Any],
]:
    ordered_cluster_ids = _root_cluster_first_order(
        candidates_by_cluster,
        relation_plan=relation_plan,
        cluster_order=cluster_order,
    )
    root_cluster_id = _select_root_cluster_id(
        candidates_by_cluster,
        relation_plan=relation_plan,
        cluster_order=ordered_cluster_ids,
    )
    profiles, _adjacency = _build_cluster_macro_profiles(
        relation_plan,
        cluster_ids=ordered_cluster_ids,
    )
    candidate_bonus_by_key: dict[tuple[str, str, int, int, int], int] = {}
    for cluster_id in ordered_cluster_ids:
        profile = profiles.get(cluster_id, {})
        is_root_cluster = cluster_id == root_cluster_id
        for candidate in candidates_by_cluster.get(cluster_id, ()):
            candidate_bonus_by_key[candidate.key] = _candidate_macro_bonus(
                candidate,
                profile=profile,
                is_root_cluster=is_root_cluster,
            ) + _candidate_variant_family_bonus(
                candidate,
                relation_plan=relation_plan,
            )

    prioritized: dict[str, list[Candidate]] = {}
    used_wall_pin_focus = False
    used_hard_wall_contract = False
    for cluster_id in ordered_cluster_ids:
        sorted_candidates = sorted(
            list(candidates_by_cluster.get(cluster_id, ())),
            key=lambda candidate: (
                -_candidate_effective_score(candidate, candidate_bonus_by_key),
                candidate.variant_id,
                candidate.rot,
                candidate.x,
                candidate.y,
            ),
        )
        profile = profiles.get(cluster_id, {})
        if (
            wall_pin_focus
            and cluster_id == root_cluster_id
            and _cluster_prefers_wall(profile)
        ):
            wall_candidates = [
                candidate
                for candidate in sorted_candidates
                if _anchor_kind_is_wall_pinned(candidate.anchor_kind)
            ]
            if len(wall_candidates) >= min(4, len(sorted_candidates)):
                if _has_hard_root_wall_contract(
                    relation_plan=relation_plan,
                    root_cluster_id=root_cluster_id,
                    profile=profile,
                ):
                    wall_candidates = _hard_contract_focus_candidates(wall_candidates)
                    used_hard_wall_contract = True
                elif _has_orientation_sensitive_wall_contract(profile):
                    wall_candidates = sorted(
                        _orientation_focus_candidates(wall_candidates),
                        key=lambda candidate: (
                            0 if candidate.acceptable_valid else 1,
                            int(candidate.critical_orientation_penalty_mm),
                            int(candidate.focal_orientation_penalty_mm),
                            int(candidate.orientation_penalty_mm),
                            -_candidate_effective_score(
                                candidate, candidate_bonus_by_key
                            ),
                            candidate.variant_id,
                            candidate.rot,
                            candidate.x,
                            candidate.y,
                        ),
                    )
                sorted_candidates = wall_candidates[: min(len(wall_candidates), 20)]
                used_wall_pin_focus = True
        prioritized[cluster_id] = sorted_candidates

    debug = {
        "root_cluster_id": root_cluster_id,
        "cluster_order": ordered_cluster_ids,
        "used_wall_pin_focus": used_wall_pin_focus,
        "used_hard_wall_contract": used_hard_wall_contract,
        "wall_pin_focus_requested": bool(wall_pin_focus),
    }
    return prioritized, ordered_cluster_ids, candidate_bonus_by_key, debug


def _canonicalize_inputs(
    tools: Any, room_model: Any, clusters_outlines: Any
) -> tuple[Any, Any]:
    unwrap = _require_attr(tools, "_unwrap_any")
    canon = _require_attr(tools, "_canonicalize_clusters_local_origin")
    room_u = unwrap(room_model)
    clusters_u = unwrap(clusters_outlines)
    clusters_c, _offsets = canon(clusters_u)
    return room_u, clusters_c


def _normalize_cluster_constraints(
    cluster_constraints: Any,
) -> dict[str, Any] | None:
    if not isinstance(cluster_constraints, dict):
        return None
    clusters = cluster_constraints.get("clusters")
    if not isinstance(clusters, list):
        return None
    out_clusters: list[dict[str, Any]] = []
    for row in clusters:
        if not isinstance(row, dict):
            continue
        cid = row.get("cluster_id")
        if not isinstance(cid, str) or not cid.strip():
            continue
        out_clusters.append(row)
    if not out_clusters:
        return None
    return {"status": cluster_constraints.get("status"), "clusters": out_clusters}


def _cluster_constraints_summary(
    cluster_constraints: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(cluster_constraints, dict):
        return {"cluster_count": 0, "access_object_count": 0}
    cluster_count = 0
    access_object_ids: set[tuple[str, str]] = set()
    for row in cluster_constraints.get("clusters") or []:
        if not isinstance(row, dict):
            continue
        cid = row.get("cluster_id")
        if not isinstance(cid, str) or not cid:
            continue
        cluster_count += 1
        for item in row.get("hard_constraints") or []:
            if not isinstance(item, dict):
                continue
            if str(item.get("type") or "").lower() == "requires_access":
                oid = item.get("id")
                if isinstance(oid, str) and oid:
                    access_object_ids.add((cid, oid))
        rules = row.get("cluster_rules") or {}
        for item in rules.get("access_requirements") or []:
            if not isinstance(item, dict):
                continue
            if not bool(item.get("required")):
                continue
            oid = item.get("id")
            if isinstance(oid, str) and oid:
                access_object_ids.add((cid, oid))
    return {
        "cluster_count": cluster_count,
        "access_object_count": len(access_object_ids),
    }


def _augment_relation_plan_from_cluster_constraints(
    relation_plan: Any,
    cluster_constraints: dict[str, Any] | None,
) -> dict[str, Any]:
    import copy

    base = copy.deepcopy(relation_plan) if isinstance(relation_plan, dict) else {}
    if not isinstance(base.get("object_orientations"), list):
        base["object_orientations"] = []
    if not isinstance(base.get("notes"), list):
        base["notes"] = []

    existing = set()
    for item in base.get("object_orientations") or []:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("cluster_id") or "")
        oid = str(item.get("object_id") or "")
        intents = tuple(
            sorted(
                str(x).lower()
                for x in (item.get("intents") or [])
                if isinstance(x, str)
            )
        )
        existing.add((cid, oid, intents))

    if not isinstance(cluster_constraints, dict):
        return base

    added = 0
    for row in cluster_constraints.get("clusters") or []:
        if not isinstance(row, dict):
            continue
        cid = row.get("cluster_id")
        if not isinstance(cid, str) or not cid:
            continue

        access_ids: set[str] = set()

        for item in row.get("hard_constraints") or []:
            if not isinstance(item, dict):
                continue
            if str(item.get("type") or "").lower() == "requires_access":
                oid = item.get("id")
                mode = str(item.get("mode") or "").lower()
                if isinstance(oid, str) and oid and mode in {"front_clearance", ""}:
                    access_ids.add(oid)

        rules = row.get("cluster_rules") or {}
        for item in rules.get("access_requirements") or []:
            if not isinstance(item, dict):
                continue
            if not bool(item.get("required")):
                continue
            oid = item.get("id")
            typ = str(item.get("type") or "").lower()
            if isinstance(oid, str) and oid and typ in {"front_clearance", ""}:
                access_ids.add(oid)

        # Only explicit front-access requirements should become global object orientation intents.
        # A forge-facing declaration describes the asset's local front; it is not, by itself,
        # a requirement that the object front must point to room open space.
        for oid in sorted(access_ids):
            intents = tuple(sorted({"preserve_front_access", "front_to_open_space"}))
            sig = (cid, oid, intents)
            if sig in existing:
                continue
            base["object_orientations"].append(
                {
                    "cluster_id": cid,
                    "object_id": oid,
                    "intents": ["preserve_front_access", "front_to_open_space"],
                    "priority": "high",
                    "reason": "augmented from cluster_constraints",
                }
            )
            existing.add(sig)
            added += 1

    if added > 0:
        base["notes"].append(
            f"augmented_object_orientations_from_cluster_constraints={added}"
        )
    return base


def _cluster_allowed_rotations(
    cluster_constraints: dict[str, Any] | None,
) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    if not isinstance(cluster_constraints, dict):
        return out
    for row in cluster_constraints.get("clusters") or []:
        if not isinstance(row, dict):
            continue
        cid = row.get("cluster_id")
        if not isinstance(cid, str) or not cid:
            continue
        rules = row.get("cluster_rules") or {}
        allowed = rules.get("allowed_rotations")
        if not isinstance(allowed, dict):
            continue
        sets: list[set[int]] = []
        for _, vals in allowed.items():
            if not isinstance(vals, list):
                continue
            s = {int(v) % 360 for v in vals if int(v) % 90 == 0}
            if s:
                sets.append(s)
        if not sets:
            continue
        inter = set.intersection(*sets)
        if inter:
            out[cid] = inter
        else:
            uni = set.union(*sets)
            if uni:
                out[cid] = uni
    return out


def _filter_candidates_by_allowed_rotations(
    candidates_by_cluster: dict[str, list[Candidate]],
    cluster_allowed_rotations: dict[str, set[int]],
) -> dict[str, list[Candidate]]:
    out: dict[str, list[Candidate]] = {}
    for cid, cands in candidates_by_cluster.items():
        allowed = cluster_allowed_rotations.get(cid)
        if not allowed:
            out[cid] = list(cands)
            continue
        out[cid] = [c for c in cands if int(c.rot) in allowed]
    return out


def _build_variant_payload_lookup(
    tools: Any, clusters_outlines: Any, *, max_variants_per_cluster: int
) -> dict[str, dict[str, dict[str, Any]]]:
    build_variants = _require_attr(tools, "BuildGenericClusterVariants")
    info = build_variants(
        clusters_outlines=clusters_outlines,
        max_variants_per_cluster=max_variants_per_cluster,
        include_variant_payloads=True,
    )
    if info.get("result") != "OK":
        raise RuntimeError(f"BuildGenericClusterVariants failed: {info}")
    lookup: dict[str, dict[str, dict[str, Any]]] = {}
    for row in info.get("clusters") or []:
        cid = str(row.get("cluster_id") or "")
        vmap: dict[str, dict[str, Any]] = {}
        for v in row.get("variants") or []:
            payload = v.get("cluster_payload")
            vid = str(v.get("variant_id") or "")
            if vid and isinstance(payload, dict):
                vmap[vid] = payload
        if cid and vmap:
            lookup[cid] = vmap
    return lookup


def _enumerate_initial_candidates(
    tools: Any,
    *,
    room_model: Any,
    clusters_outlines: Any,
    relation_plan: Any,
    grid_mm: int,
    max_candidates_per_cluster: int,
    max_variants_per_cluster: int,
) -> tuple[dict[str, list[Candidate]], list[str]]:
    enum = _require_attr(tools, "EnumerateClusterCandidates")
    res = enum(
        room_model=room_model,
        clusters_outlines=clusters_outlines,
        grid_mm=grid_mm,
        relation_plan=relation_plan,
        max_candidates_per_cluster=max_candidates_per_cluster,
        max_variants_per_cluster=max_variants_per_cluster,
    )
    if res.get("result") != "OK":
        raise RuntimeError(f"EnumerateClusterCandidates failed: {res}")
    out: dict[str, list[Candidate]] = {}
    cluster_order: list[str] = []
    for row in res.get("clusters") or []:
        cid = str(row.get("cluster_id") or "")
        if not cid:
            continue
        out[cid] = [_candidate_from_dict(x) for x in (row.get("candidates") or [])]
        cluster_order.append(cid)
    return out, cluster_order


def _merge_cluster_order(*orders: Sequence[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for order in orders:
        for cluster_id in order:
            clean_cluster_id = str(cluster_id or "").strip()
            if not clean_cluster_id or clean_cluster_id in seen:
                continue
            merged.append(clean_cluster_id)
            seen.add(clean_cluster_id)
    return merged


def _ordered_cluster_ids(
    candidates_by_cluster: dict[str, list[Candidate]],
    *,
    cluster_order: Sequence[str] | None = None,
) -> list[str]:
    preferred_order = _merge_cluster_order(cluster_order or (), candidates_by_cluster)
    return [
        cluster_id
        for cluster_id in preferred_order
        if cluster_id in candidates_by_cluster
    ]


def _relax_relation_plan_for_candidate_search(
    relation_plan: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(relation_plan, dict):
        return relation_plan

    out = dict(relation_plan)
    protected_pairs = _viewing_contract_pairs(relation_plan)
    protected_cluster_ids = _viewing_contract_cluster_ids(relation_plan)
    out["cluster_relations"] = [
        dict(row)
        for row in relation_plan.get("cluster_relations") or []
        if isinstance(row, dict)
        and _normalized_cluster_pair(
            str(row.get("a") or ""),
            str(row.get("b") or ""),
        )
        in protected_pairs
    ]
    out["cluster_directional_relations"] = [
        dict(row)
        for row in relation_plan.get("cluster_directional_relations") or []
        if isinstance(row, dict)
        and _normalized_cluster_pair(
            str(row.get("a") or ""),
            str(row.get("b") or ""),
        )
        in protected_pairs
    ]

    cluster_orientations: list[dict[str, Any]] = []
    for entry in relation_plan.get("cluster_orientations") or []:
        if not isinstance(entry, dict):
            continue
        cluster_id = str(entry.get("cluster_id") or "").strip()
        if cluster_id in protected_cluster_ids:
            cluster_orientations.append(dict(entry))
            continue
        new_entry = dict(entry)
        intents = [
            item
            for item in (entry.get("intents") or [])
            if isinstance(item, str) and item != "face_cluster"
        ]
        if not intents:
            continue
        new_entry["intents"] = intents
        if "face_cluster" not in intents:
            new_entry["target_cluster_id"] = None
        cluster_orientations.append(new_entry)
    out["cluster_orientations"] = cluster_orientations

    object_orientations: list[dict[str, Any]] = []
    for entry in relation_plan.get("object_orientations") or []:
        if not isinstance(entry, dict):
            continue
        cluster_id = str(entry.get("cluster_id") or "").strip()
        target_cluster_id = str(entry.get("target_object_cluster_id") or "").strip()
        if (
            cluster_id in protected_cluster_ids
            and target_cluster_id in protected_cluster_ids
        ):
            object_orientations.append(dict(entry))
            continue
        new_entry = dict(entry)
        intents = [
            item
            for item in (entry.get("intents") or [])
            if isinstance(item, str)
            and item not in {"face_object", "face_away_from_object"}
        ]
        if not intents:
            continue
        new_entry["intents"] = intents
        if "face_object" not in intents and "face_away_from_object" not in intents:
            new_entry["target_object_id"] = None
        object_orientations.append(new_entry)
    out["object_orientations"] = object_orientations
    return out


def _augment_with_relaxed_candidates(
    tools: Any,
    *,
    room_model: Any,
    clusters_outlines: Any,
    relation_plan: dict[str, Any] | None,
    grid_mm: int,
    current_candidates: dict[str, list[Candidate]],
    cluster_allowed_rotations: dict[str, set[int]],
    max_variants_per_cluster: int,
    base_candidate_limit: int,
    cluster_order: Sequence[str] | None = None,
) -> dict[str, list[Candidate]]:
    relaxed_plan = _relax_relation_plan_for_candidate_search(relation_plan)
    extra_candidates, extra_cluster_order = _enumerate_initial_candidates(
        tools,
        room_model=room_model,
        clusters_outlines=clusters_outlines,
        relation_plan=relaxed_plan,
        grid_mm=grid_mm,
        max_candidates_per_cluster=max(base_candidate_limit + 16, 56),
        max_variants_per_cluster=max_variants_per_cluster,
    )
    extra_candidates = _filter_candidates_by_allowed_rotations(
        extra_candidates,
        cluster_allowed_rotations,
    )
    extra_candidates = _filter_candidates_by_concept_variant_policy(
        extra_candidates,
        relation_plan,
    )
    extra_candidates = _filter_protected_topology_candidates(
        extra_candidates,
        relation_plan,
    )

    merged: dict[str, list[Candidate]] = {}
    merged_order = _merge_cluster_order(
        cluster_order or tuple(current_candidates.keys()),
        extra_cluster_order,
        tuple(extra_candidates.keys()),
    )
    for cid in merged_order:
        merged[cid] = _dedupe_candidates(
            list(current_candidates.get(cid, [])) + list(extra_candidates.get(cid, []))
        )[: max(base_candidate_limit + 32, 64)]
    return merged


def _dedupe_candidates(cands: Iterable[Candidate]) -> list[Candidate]:
    best: dict[tuple[str, str, int, int, int], Candidate] = {}
    for c in cands:
        prev = best.get(c.key)
        if prev is None or _candidate_score(c) > _candidate_score(prev):
            best[c.key] = c
    ordered = list(best.values())
    ordered.sort(key=lambda x: (-_candidate_score(x), x.variant_id, x.rot, x.x, x.y))
    return ordered


def _cap_candidate_search_space(
    candidates_by_cluster: dict[str, list[Candidate]],
    *,
    max_per_cluster: int = DEFAULT_MAX_POSE_CANDIDATES_PER_CLUSTER,
    max_total: int = DEFAULT_MAX_TOTAL_BINARY_CANDIDATES,
) -> dict[str, list[Candidate]]:
    capped = {
        cluster_id: _dedupe_candidates(candidates)[: max(1, int(max_per_cluster))]
        for cluster_id, candidates in candidates_by_cluster.items()
    }
    while sum(len(candidates) for candidates in capped.values()) > max(
        1, int(max_total)
    ):
        largest_cluster_id = max(
            capped,
            key=lambda cluster_id: (
                len(capped[cluster_id]),
                cluster_id,
            ),
        )
        if len(capped[largest_cluster_id]) <= 1:
            break
        capped[largest_cluster_id] = capped[largest_cluster_id][:-1]
    return capped


def _build_candidate_geometries(
    tools: Any,
    *,
    variant_payload_lookup: dict[str, dict[str, dict[str, Any]]],
    candidates_by_cluster: dict[str, list[Candidate]],
) -> dict[tuple[str, int], Any]:
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    build_cluster_polys = _require_attr(tools, "_build_cluster_polys")
    fix_geom = _require_attr(tools, "_fix_geom")

    out: dict[tuple[str, int], Any] = {}
    for cid, cands in candidates_by_cluster.items():
        vmap = variant_payload_lookup.get(cid, {})
        for idx, cand in enumerate(cands):
            payload = vmap.get(cand.variant_id)
            if payload is None:
                raise RuntimeError(f"No payload for {cid} / {cand.variant_id}")
            polys = build_cluster_polys(Polygon, payload, cand.x, cand.y, cand.rot)
            if not polys:
                raise RuntimeError(f"No geometry for candidate {cand}")
            out[(cid, idx)] = fix_geom(unary_union(polys))
    return out


def _materialize_variantized_clusters(
    tools: Any,
    *,
    clusters_outlines: Any,
    selected_variants: dict[str, str],
) -> Any:
    materialize = _require_attr(tools, "MaterializeVariantizedClusters")
    result = materialize(
        clusters_outlines=clusters_outlines,
        selected_variants=selected_variants,
    )
    if result.get("result") != "OK":
        raise RuntimeError(f"MaterializeVariantizedClusters failed: {result}")
    return result.get("clusters_outlines")


def _pair_eval(
    tools: Any,
    *,
    room_model: Any,
    clusters_outlines: Any,
    relation_plan: Any,
    grid_mm: int,
    a: Candidate,
    b: Candidate,
) -> dict[str, Any]:
    verify = _require_attr(tools, "GlobalClusterVerifier")
    selected_variants = {
        a.cluster_id: a.variant_id,
        b.cluster_id: b.variant_id,
    }
    mat = _materialize_variantized_clusters(
        tools,
        clusters_outlines=clusters_outlines,
        selected_variants=selected_variants,
    )
    return verify(
        room_model=room_model,
        clusters_outlines=mat,
        cluster_transforms=[
            {"cluster_id": a.cluster_id, "x": a.x, "y": a.y, "rot": a.rot},
            {"cluster_id": b.cluster_id, "x": b.x, "y": b.y, "rot": b.rot},
        ],
        grid_mm=grid_mm,
        mode="partial",
        relation_plan=relation_plan,
        return_debug=False,
    )


def _build_pair_terms(
    tools: Any,
    *,
    room_model: Any,
    clusters_outlines: Any,
    relation_plan: Any,
    grid_mm: int,
    candidates_by_cluster: dict[str, list[Candidate]],
    candidate_geoms: dict[tuple[str, int], Any],
    cluster_order: Sequence[str] | None = None,
    max_pair_bonus_abs: int = 12000,
) -> tuple[
    set[tuple[tuple[str, int], tuple[str, int]]],
    dict[tuple[tuple[str, int], tuple[str, int]], int],
]:
    incompatible: set[tuple[tuple[str, int], tuple[str, int]]] = set()
    pair_bonus: dict[tuple[tuple[str, int], tuple[str, int]], int] = {}
    cluster_ids = _ordered_cluster_ids(
        candidates_by_cluster,
        cluster_order=cluster_order,
    )
    cache: dict[tuple[Any, ...], dict[str, Any]] = {}

    for i, a_id in enumerate(cluster_ids):
        for j in range(i + 1, len(cluster_ids)):
            b_id = cluster_ids[j]
            a_cands = candidates_by_cluster[a_id]
            b_cands = candidates_by_cluster[b_id]
            for ai, ca in enumerate(a_cands):
                ga = candidate_geoms[(a_id, ai)]
                for bi, cb in enumerate(b_cands):
                    gb = candidate_geoms[(b_id, bi)]
                    key = ((a_id, ai), (b_id, bi))
                    try:
                        inter_area = float(ga.intersection(gb).area)
                    except Exception:
                        inter_area = 1.0
                    if inter_area > 1e-6:
                        incompatible.add(key)
                        continue

                    cache_key = (
                        a_id,
                        ai,
                        ca.variant_id,
                        ca.x,
                        ca.y,
                        ca.rot,
                        b_id,
                        bi,
                        cb.variant_id,
                        cb.x,
                        cb.y,
                        cb.rot,
                    )
                    pair_res = cache.get(cache_key)
                    if pair_res is None:
                        pair_res = _pair_eval(
                            tools,
                            room_model=room_model,
                            clusters_outlines=clusters_outlines,
                            relation_plan=relation_plan,
                            grid_mm=grid_mm,
                            a=ca,
                            b=cb,
                        )
                        cache[cache_key] = pair_res
                    if not bool(pair_res.get("hard_valid")):
                        incompatible.add(key)
                        continue
                    pair_layout_score = _verify_search_score(pair_res)
                    bonus = (
                        pair_layout_score - _candidate_score(ca) - _candidate_score(cb)
                    )
                    bonus = int(
                        max(-max_pair_bonus_abs, min(max_pair_bonus_abs, bonus))
                    )
                    if bonus != 0:
                        pair_bonus[key] = bonus
    return incompatible, pair_bonus


def _selected_from_solver(
    candidates_by_cluster: dict[str, list[Candidate]],
    xvars: dict[tuple[str, int], cp_model.IntVar],
    solver: cp_model.CpSolver,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Candidate]]:
    transforms: list[dict[str, Any]] = []
    variants: list[dict[str, str]] = []
    chosen: dict[str, Candidate] = {}
    for cid, cands in candidates_by_cluster.items():
        found = None
        for i, cand in enumerate(cands):
            if solver.Value(xvars[(cid, i)]) == 1:
                found = cand
                break
        if found is None:
            raise RuntimeError(f"No selected candidate for {cid}")
        chosen[cid] = found
        transforms.append(
            {
                "cluster_id": cid,
                "x": found.x,
                "y": found.y,
                "rot": found.rot,
            }
        )
        variants.append({"cluster_id": cid, "variant_id": found.variant_id})
    transforms.sort(key=lambda x: x["cluster_id"])
    variants.sort(key=lambda x: x["cluster_id"])
    return transforms, variants, chosen


def _verify_complete(
    tools: Any,
    *,
    room_model: Any,
    clusters_outlines: Any,
    relation_plan: Any,
    grid_mm: int,
    cluster_transforms: list[dict[str, Any]],
    selected_variants: list[dict[str, str]],
) -> dict[str, Any]:
    verify = _require_attr(tools, "GlobalClusterVerifier")
    mat = _materialize_variantized_clusters(
        tools,
        clusters_outlines=clusters_outlines,
        selected_variants={x["cluster_id"]: x["variant_id"] for x in selected_variants},
    )
    return verify(
        room_model=room_model,
        clusters_outlines=mat,
        cluster_transforms=cluster_transforms,
        grid_mm=grid_mm,
        mode="complete",
        relation_plan=relation_plan,
        return_debug=False,
    )


def _neighborhood_offsets(grid_mm: int) -> list[tuple[int, int]]:
    g = int(grid_mm)
    vals = [0, -g, g, -2 * g, 2 * g]
    out: list[tuple[int, int]] = []
    for dx in vals:
        for dy in vals:
            out.append((dx, dy))
    return out


def _expand_problem_clusters(
    tools: Any,
    *,
    room_model: Any,
    clusters_outlines: Any,
    relation_plan: Any,
    grid_mm: int,
    candidates_by_cluster: dict[str, list[Candidate]],
    chosen: dict[str, Candidate],
    problematic_cluster_ids: Sequence[str],
    max_variants_per_cluster: int,
    cluster_allowed_rotations: dict[str, set[int]] | None = None,
    add_limit_per_cluster: int = 36,
) -> dict[str, list[Candidate]]:
    verify = _require_attr(tools, "GlobalClusterVerifier")
    build_variants = _require_attr(tools, "BuildGenericClusterVariants")

    new_map = {cid: list(cands) for cid, cands in candidates_by_cluster.items()}
    variants_info = build_variants(
        clusters_outlines=clusters_outlines,
        cluster_ids=list(problematic_cluster_ids),
        max_variants_per_cluster=max_variants_per_cluster,
        include_variant_payloads=False,
    )
    variant_ids_by_cluster: dict[str, list[str]] = {}
    if variants_info.get("result") == "OK":
        for row in variants_info.get("clusters") or []:
            cid = str(row.get("cluster_id") or "")
            raw_variants = [
                v for v in (row.get("variants") or []) if isinstance(v, Mapping)
            ]
            semantic_families_available = any(
                normalize_variant_family(v.get("family")) in SEMANTIC_VARIANT_FAMILIES
                for v in raw_variants
            )
            vids = [
                str(v.get("variant_id") or "")
                for v in raw_variants
                if _variant_family_allowed_for_cluster(
                    relation_plan=relation_plan,
                    cluster_id=cid,
                    variant_family=str(v.get("family") or ""),
                    semantic_families_available=semantic_families_available,
                )
            ]
            variant_ids_by_cluster[cid] = [v for v in vids if v]

    for cid in problematic_cluster_ids:
        current = chosen.get(cid)
        if current is None:
            continue
        existing = {c.key for c in new_map.get(cid, [])}
        proposals: list[Candidate] = []
        if cid in variant_ids_by_cluster:
            variant_ids = variant_ids_by_cluster[cid]
        else:
            variant_ids = [current.variant_id]
        if not variant_ids:
            continue
        allowed_rots = (cluster_allowed_rotations or {}).get(cid) or {0, 90, 180, 270}
        for variant_id in variant_ids[: min(4, len(variant_ids))]:
            for rot in tuple(r for r in (0, 90, 180, 270) if r in allowed_rots):
                for dx, dy in _neighborhood_offsets(grid_mm):
                    x = current.x + dx
                    y = current.y + dy
                    tmp = Candidate(
                        cluster_id=cid,
                        variant_id=variant_id,
                        variant_family=current.variant_family,
                        variant_priority=current.variant_priority,
                        x=x,
                        y=y,
                        rot=rot,
                        anchor_kind="local_refine",
                        anchor_priority=max(0.8, current.anchor_priority),
                        stage="refine_neighborhood",
                        hard_valid=False,
                        acceptable_valid=False,
                        rough_score=-(10**9),
                        macro_penalty_mm=0,
                        micro_penalty_mm=0,
                        orientation_penalty_mm=0,
                        critical_orientation_penalty_mm=0,
                        focal_orientation_penalty_mm=0,
                        quality_gate_reasons=(),
                        hard_error_codes=(),
                        state_signature="",
                    )
                    if tmp.key in existing:
                        continue
                    mat = _materialize_variantized_clusters(
                        tools,
                        clusters_outlines=clusters_outlines,
                        selected_variants={cid: variant_id},
                    )
                    res = verify(
                        room_model=room_model,
                        clusters_outlines=mat,
                        cluster_transforms=[
                            {"cluster_id": cid, "x": x, "y": y, "rot": rot}
                        ],
                        grid_mm=grid_mm,
                        mode="partial",
                        relation_plan=relation_plan,
                        return_debug=False,
                    )
                    if not bool(res.get("hard_valid")):
                        continue
                    q = res.get("quality") or {}
                    vrec = (res.get("violations_by_cluster") or {}).get(cid) or {}
                    candidate = Candidate(
                        cluster_id=cid,
                        variant_id=variant_id,
                        variant_family=current.variant_family,
                        variant_priority=current.variant_priority,
                        x=x,
                        y=y,
                        rot=rot,
                        anchor_kind="local_refine",
                        anchor_priority=max(0.8, current.anchor_priority),
                        stage="refine_neighborhood",
                        hard_valid=True,
                        acceptable_valid=True,
                        rough_score=int(q.get("layout_score") or 0),
                        macro_penalty_mm=int(q.get("macro_penalty_mm") or 0),
                        micro_penalty_mm=int(q.get("micro_penalty_mm") or 0),
                        orientation_penalty_mm=int(
                            vrec.get("orientation_penalty_mm") or 0
                        ),
                        critical_orientation_penalty_mm=int(
                            vrec.get("critical_orientation_penalty_mm") or 0
                        ),
                        focal_orientation_penalty_mm=int(
                            vrec.get("focal_orientation_penalty_mm") or 0
                        ),
                        quality_gate_reasons=tuple(
                            (res.get("quality_gate") or {}).get("reasons") or []
                        ),
                        hard_error_codes=tuple(),
                        state_signature=str(res.get("state_signature") or ""),
                    )
                    proposals.append(candidate)
                    existing.add(candidate.key)
                    if len(proposals) >= add_limit_per_cluster:
                        break
                if len(proposals) >= add_limit_per_cluster:
                    break
            if len(proposals) >= add_limit_per_cluster:
                break
        merged = _dedupe_candidates(list(new_map.get(cid, [])) + proposals)
        new_map[cid] = merged
    return new_map


def _solve_one_round(
    *,
    tools: Any,
    room_model: Any,
    clusters_outlines: Any,
    relation_plan: Any,
    grid_mm: int,
    candidates_by_cluster: dict[str, list[Candidate]],
    cluster_order: Sequence[str],
    variant_payload_lookup: dict[str, dict[str, dict[str, Any]]],
    time_limit_s: float,
    num_workers: int,
    candidate_bonus_by_key: Mapping[tuple[str, str, int, int, int], int] | None = None,
) -> tuple[
    list[dict[str, Any]], list[dict[str, str]], dict[str, Candidate], dict[str, Any]
]:
    candidate_geoms = _build_candidate_geometries(
        tools,
        variant_payload_lookup=variant_payload_lookup,
        candidates_by_cluster=candidates_by_cluster,
    )
    incompatible, pair_bonus = _build_pair_terms(
        tools,
        room_model=room_model,
        clusters_outlines=clusters_outlines,
        relation_plan=relation_plan,
        grid_mm=grid_mm,
        candidates_by_cluster=candidates_by_cluster,
        candidate_geoms=candidate_geoms,
        cluster_order=cluster_order,
    )

    model = cp_model.CpModel()
    xvars: dict[tuple[str, int], cp_model.IntVar] = {}
    ordered_pick_vars: list[cp_model.IntVar] = []
    for cid in _ordered_cluster_ids(candidates_by_cluster, cluster_order=cluster_order):
        cands = candidates_by_cluster[cid]
        vars_for_cluster = []
        for i, cand in enumerate(cands):
            var = model.NewBoolVar(f"pick[{cid},{i}]")
            xvars[(cid, i)] = var
            vars_for_cluster.append(var)
        model.AddExactlyOne(vars_for_cluster)
        ordered_pick_vars.extend(vars_for_cluster)

    for a_key, b_key in sorted(incompatible):
        model.Add(xvars[a_key] + xvars[b_key] <= 1)

    objective_terms: list[Any] = []
    for cid in _ordered_cluster_ids(candidates_by_cluster, cluster_order=cluster_order):
        cands = candidates_by_cluster[cid]
        for i, cand in enumerate(cands):
            objective_terms.append(
                _candidate_effective_score(cand, candidate_bonus_by_key)
                * xvars[(cid, i)]
            )

    for (a_key, b_key), bonus in pair_bonus.items():
        za = xvars[a_key]
        zb = xvars[b_key]
        z = model.NewBoolVar(f"pair[{a_key[0]},{a_key[1]}|{b_key[0]},{b_key[1]}]")
        model.Add(z <= za)
        model.Add(z <= zb)
        model.Add(z >= za + zb - 1)
        objective_terms.append(int(bonus) * z)

    model.Maximize(sum(objective_terms))

    add_decision_strategy = getattr(model, "AddDecisionStrategy", None)
    choose_first = getattr(cp_model, "CHOOSE_FIRST", None)
    select_max_value = getattr(cp_model, "SELECT_MAX_VALUE", None)
    if (
        callable(add_decision_strategy)
        and choose_first is not None
        and select_max_value is not None
        and ordered_pick_vars
    ):
        add_decision_strategy(ordered_pick_vars, choose_first, select_max_value)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = int(max(1, num_workers))
    solver.parameters.random_seed = 42

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("CP-SAT did not find a feasible assignment over the set")

    transforms, selected_variants, chosen = _selected_from_solver(
        candidates_by_cluster,
        xvars,
        solver,
    )
    verify = _verify_complete(
        tools,
        room_model=room_model,
        clusters_outlines=clusters_outlines,
        relation_plan=relation_plan,
        grid_mm=grid_mm,
        cluster_transforms=transforms,
        selected_variants=selected_variants,
    )
    return transforms, selected_variants, chosen, verify


def _solve_solution_pool_one_round(
    *,
    tools: Any,
    room_model: Any,
    clusters_outlines: Any,
    relation_plan: Any,
    grid_mm: int,
    candidates_by_cluster: dict[str, list[Candidate]],
    cluster_order: Sequence[str],
    variant_payload_lookup: dict[str, dict[str, dict[str, Any]]],
    time_limit_s: float,
    num_workers: int,
    candidate_bonus_by_key: Mapping[tuple[str, str, int, int, int], int] | None = None,
    max_solutions: int = DEFAULT_MAX_FEASIBLE_SOLUTIONS_PER_CONCEPT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_geoms = _build_candidate_geometries(
        tools,
        variant_payload_lookup=variant_payload_lookup,
        candidates_by_cluster=candidates_by_cluster,
    )
    incompatible, pair_bonus = _build_pair_terms(
        tools,
        room_model=room_model,
        clusters_outlines=clusters_outlines,
        relation_plan=relation_plan,
        grid_mm=grid_mm,
        candidates_by_cluster=candidates_by_cluster,
        candidate_geoms=candidate_geoms,
        cluster_order=cluster_order,
    )

    model = cp_model.CpModel()
    xvars: dict[tuple[str, int], cp_model.IntVar] = {}
    ordered_pick_vars: list[cp_model.IntVar] = []
    ordered_cluster_ids = _ordered_cluster_ids(
        candidates_by_cluster,
        cluster_order=cluster_order,
    )
    for cid in ordered_cluster_ids:
        cands = candidates_by_cluster[cid]
        vars_for_cluster = []
        for index, _cand in enumerate(cands):
            var = model.NewBoolVar(f"pick[{cid},{index}]")
            xvars[(cid, index)] = var
            vars_for_cluster.append(var)
        model.AddExactlyOne(vars_for_cluster)
        ordered_pick_vars.extend(vars_for_cluster)

    for a_key, b_key in sorted(incompatible):
        model.Add(xvars[a_key] + xvars[b_key] <= 1)

    objective_terms: list[Any] = []
    for cid in ordered_cluster_ids:
        for index, cand in enumerate(candidates_by_cluster[cid]):
            objective_terms.append(
                _candidate_effective_score(cand, candidate_bonus_by_key)
                * xvars[(cid, index)]
            )

    for (a_key, b_key), bonus in pair_bonus.items():
        za = xvars[a_key]
        zb = xvars[b_key]
        z = model.NewBoolVar(f"pair[{a_key[0]},{a_key[1]}|{b_key[0]},{b_key[1]}]")
        model.Add(z <= za)
        model.Add(z <= zb)
        model.Add(z >= za + zb - 1)
        objective_terms.append(int(bonus) * z)

    model.Maximize(sum(objective_terms))

    add_decision_strategy = getattr(model, "AddDecisionStrategy", None)
    choose_first = getattr(cp_model, "CHOOSE_FIRST", None)
    select_max_value = getattr(cp_model, "SELECT_MAX_VALUE", None)
    if (
        callable(add_decision_strategy)
        and choose_first is not None
        and select_max_value is not None
        and ordered_pick_vars
    ):
        add_decision_strategy(ordered_pick_vars, choose_first, select_max_value)

    pool: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()
    statuses: list[str] = []
    solve_limit = max(1, int(max_solutions))
    per_solve_time = max(0.5, float(time_limit_s) / max(1, min(solve_limit, 4)))

    for _idx in range(solve_limit):
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = per_solve_time
        solver.parameters.num_search_workers = int(max(1, num_workers))
        solver.parameters.random_seed = 42

        status = solver.Solve(model)
        status_name = (
            solver.StatusName(status)
            if callable(getattr(solver, "StatusName", None))
            else str(status)
        )
        statuses.append(status_name)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break

        selected_keys: list[tuple[str, int]] = []
        for cid in ordered_cluster_ids:
            for index, _cand in enumerate(candidates_by_cluster[cid]):
                key = (cid, index)
                if solver.Value(xvars[key]) == 1:
                    selected_keys.append(key)
                    break

        if len(selected_keys) != len(ordered_cluster_ids):
            break

        transforms, selected_variants, chosen = _selected_from_solver(
            candidates_by_cluster,
            xvars,
            solver,
        )
        signature = _solution_signature(transforms, selected_variants)
        if signature not in seen_signatures:
            verify = _verify_complete(
                tools,
                room_model=room_model,
                clusters_outlines=clusters_outlines,
                relation_plan=relation_plan,
                grid_mm=grid_mm,
                cluster_transforms=transforms,
                selected_variants=selected_variants,
            )
            if bool(verify.get("hard_valid")) and bool(verify.get("complete")):
                pool.append(
                    {
                        "cluster_transforms": transforms,
                        "selected_variants": selected_variants,
                        "chosen": chosen,
                        "verify": verify,
                        "objective_value": int(round(solver.ObjectiveValue())),
                        "signature": signature,
                    }
                )
                seen_signatures.add(signature)

        model.Add(sum(xvars[key] for key in selected_keys) <= len(selected_keys) - 1)

    debug = {
        "incompatible_pair_count": len(incompatible),
        "pair_bonus_count": len(pair_bonus),
        "solver_statuses": statuses,
        "pool_size": len(pool),
    }
    return pool, debug


def _filter_unsat_input_clusters(
    *,
    clusters_outlines: dict[str, Any] | list[Any],
    relation_plan: dict[str, Any] | None,
    cluster_constraints: dict[str, Any] | None,
) -> tuple[dict[str, Any] | list[Any], dict[str, Any] | None, dict[str, Any] | None]:
    filtered_outlines, removed_ids = _filter_unsat_clusters_outlines(clusters_outlines)
    if removed_ids:
        logger.info(
            "Solver filtered UNSAT clusters: %s",
            sorted(removed_ids),
        )
    if not removed_ids:
        return clusters_outlines, relation_plan, cluster_constraints
    filtered_relation_plan = _filter_relation_plan_clusters(relation_plan, removed_ids)
    filtered_constraints = _filter_cluster_constraints(cluster_constraints, removed_ids)
    return filtered_outlines, filtered_relation_plan, filtered_constraints


def _filter_unsat_clusters_outlines(
    clusters_outlines: dict[str, Any] | list[Any],
) -> tuple[dict[str, Any] | list[Any], set[str]]:
    removed_ids: set[str] = set()

    def is_unsat_status(value: Any) -> bool:
        return isinstance(value, str) and value.strip().upper() == "UNSAT"

    if isinstance(clusters_outlines, dict):
        clusters = clusters_outlines.get("clusters")
        if isinstance(clusters, list):
            kept = []
            for item in clusters:
                if isinstance(item, dict) and is_unsat_status(item.get("status")):
                    cid = item.get("cluster_id")
                    if isinstance(cid, str) and cid:
                        removed_ids.add(cid)
                    continue
                kept.append(item)
            out = dict(clusters_outlines)
            out["clusters"] = kept
            return out, removed_ids

        out: dict[str, Any] = {}
        for cid, payload in clusters_outlines.items():
            if isinstance(payload, dict) and is_unsat_status(payload.get("status")):
                if isinstance(cid, str) and cid:
                    removed_ids.add(cid)
                continue
            out[cid] = payload
        return out, removed_ids

    if isinstance(clusters_outlines, list):
        kept_list: list[Any] = []
        for item in clusters_outlines:
            if isinstance(item, dict) and is_unsat_status(item.get("status")):
                cid = item.get("cluster_id")
                if isinstance(cid, str) and cid:
                    removed_ids.add(cid)
                continue
            kept_list.append(item)
        return kept_list, removed_ids

    return clusters_outlines, removed_ids


def _filter_relation_plan_clusters(
    relation_plan: dict[str, Any] | None, removed_ids: set[str]
) -> dict[str, Any] | None:
    if not removed_ids or not isinstance(relation_plan, dict):
        return relation_plan

    def _keep_cluster_id(value: Any) -> bool:
        return not (isinstance(value, str) and value in removed_ids)

    out = dict(relation_plan)

    def _filter_list(key: str, predicate) -> None:
        items = out.get(key)
        if not isinstance(items, list):
            return
        out[key] = [
            item for item in items if isinstance(item, dict) and predicate(item)
        ]

    _filter_list("cluster_affinities", lambda i: _keep_cluster_id(i.get("cluster_id")))
    _filter_list(
        "cluster_orientations",
        lambda i: _keep_cluster_id(i.get("cluster_id"))
        and _keep_cluster_id(i.get("target_cluster_id")),
    )
    _filter_list(
        "object_orientations",
        lambda i: _keep_cluster_id(i.get("cluster_id"))
        and _keep_cluster_id(i.get("target_cluster_id"))
        and _keep_cluster_id(i.get("target_object_cluster_id")),
    )
    _filter_list(
        "cluster_relations",
        lambda i: _keep_cluster_id(i.get("a")) and _keep_cluster_id(i.get("b")),
    )
    _filter_list(
        "cluster_directional_relations",
        lambda i: _keep_cluster_id(i.get("a")) and _keep_cluster_id(i.get("b")),
    )

    circ = out.get("circulation_plan")
    if isinstance(circ, dict):
        circ_out = dict(circ)
        main_paths = circ.get("main_paths")
        if isinstance(main_paths, list):
            circ_out["main_paths"] = [
                item
                for item in main_paths
                if isinstance(item, dict) and _keep_cluster_id(item.get("to_cluster"))
            ]
        keep_open = circ.get("keep_open_regions")
        if isinstance(keep_open, list):
            circ_out["keep_open_regions"] = [
                item
                for item in keep_open
                if isinstance(item, dict) and _keep_cluster_id(item.get("near"))
            ]
        out["circulation_plan"] = circ_out

    return out


def _filter_cluster_constraints(
    cluster_constraints: dict[str, Any] | None, removed_ids: set[str]
) -> dict[str, Any] | None:
    if not removed_ids or not isinstance(cluster_constraints, dict):
        return cluster_constraints
    clusters = cluster_constraints.get("clusters")
    if not isinstance(clusters, list):
        return cluster_constraints
    kept = [
        item
        for item in clusters
        if not (
            isinstance(item, dict)
            and isinstance(item.get("cluster_id"), str)
            and item.get("cluster_id") in removed_ids
        )
    ]
    out = dict(cluster_constraints)
    out["clusters"] = kept
    return out


def _filter_cluster_ids_from_outlines(
    clusters_outlines: dict[str, Any] | list[Any],
    removed_ids: set[str],
) -> dict[str, Any] | list[Any]:
    if not removed_ids:
        return clusters_outlines

    if isinstance(clusters_outlines, dict):
        clusters = clusters_outlines.get("clusters")
        if isinstance(clusters, list):
            out = dict(clusters_outlines)
            out["clusters"] = [
                item
                for item in clusters
                if not (
                    isinstance(item, dict)
                    and isinstance(item.get("cluster_id"), str)
                    and item.get("cluster_id") in removed_ids
                )
            ]
            return out
        return {
            cid: payload
            for cid, payload in clusters_outlines.items()
            if not (isinstance(cid, str) and cid in removed_ids)
        }

    if isinstance(clusters_outlines, list):
        return [
            item
            for item in clusters_outlines
            if not (
                isinstance(item, dict)
                and isinstance(item.get("cluster_id"), str)
                and item.get("cluster_id") in removed_ids
            )
        ]

    return clusters_outlines


def _cluster_can_be_skipped_without_candidates(
    *,
    relation_plan: Mapping[str, Any] | None,
    cluster_id: str,
) -> bool:
    if not isinstance(relation_plan, Mapping):
        return True
    if _cluster_is_core_or_protected(relation_plan, cluster_id):
        return False
    return _cluster_priority_kind(relation_plan, cluster_id) in {"support", "optional"}


def _skip_clusters_without_candidates(
    *,
    clusters_outlines: dict[str, Any] | list[Any],
    relation_plan: dict[str, Any] | None,
    cluster_constraints: dict[str, Any] | None,
    candidates_by_cluster: dict[str, list[Candidate]],
    variant_payload_lookup: dict[str, dict[str, dict[str, Any]]],
    cluster_allowed_rotations: dict[str, set[int]],
) -> tuple[
    dict[str, Any] | list[Any],
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, list[Candidate]],
    dict[str, dict[str, dict[str, Any]]],
    dict[str, set[int]],
    list[str],
]:
    skipped_ids = sorted(
        cid
        for cid, candidates in candidates_by_cluster.items()
        if not candidates
        and _cluster_can_be_skipped_without_candidates(
            relation_plan=relation_plan,
            cluster_id=cid,
        )
    )
    if not skipped_ids:
        return (
            clusters_outlines,
            relation_plan,
            cluster_constraints,
            candidates_by_cluster,
            variant_payload_lookup,
            cluster_allowed_rotations,
            [],
        )

    removed_ids = set(skipped_ids)
    filtered_outlines = _filter_cluster_ids_from_outlines(
        clusters_outlines, removed_ids
    )
    filtered_relation_plan = _filter_relation_plan_clusters(relation_plan, removed_ids)
    filtered_constraints = _filter_cluster_constraints(cluster_constraints, removed_ids)
    filtered_candidates = {
        cid: candidates
        for cid, candidates in candidates_by_cluster.items()
        if cid not in removed_ids
    }
    filtered_variants = {
        cid: variants
        for cid, variants in variant_payload_lookup.items()
        if cid not in removed_ids
    }
    filtered_rotations = {
        cid: rotations
        for cid, rotations in cluster_allowed_rotations.items()
        if cid not in removed_ids
    }
    return (
        filtered_outlines,
        filtered_relation_plan,
        filtered_constraints,
        filtered_candidates,
        filtered_variants,
        filtered_rotations,
        skipped_ids,
    )


def _selected_variants_map(
    selected_variants: Sequence[dict[str, str]],
) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in selected_variants:
        if not isinstance(item, dict):
            continue
        cid = item.get("cluster_id")
        vid = item.get("variant_id")
        if isinstance(cid, str) and cid and isinstance(vid, str) and vid:
            out[cid] = vid
    return out


def _build_placer_seed(
    tools: Any,
    *,
    clusters_outlines: Any,
    relation_plan_used: dict[str, Any] | None,
    cluster_constraints_used: dict[str, Any] | None,
    grid_mm: int,
    cluster_transforms: Sequence[dict[str, Any]],
    selected_variants: Sequence[dict[str, str]],
    verify: dict[str, Any] | None,
    solver_status: str,
    candidate_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    if not cluster_transforms or not selected_variants or not isinstance(verify, dict):
        return {
            "phase": "micro_repair",
            "ready": False,
            "needed": False,
            "reason": "no_complete_hard_valid_seed",
        }

    selected_variants_list = [
        {"cluster_id": str(x["cluster_id"]), "variant_id": str(x["variant_id"])}
        for x in selected_variants
        if isinstance(x, dict)
        and isinstance(x.get("cluster_id"), str)
        and isinstance(x.get("variant_id"), str)
    ]
    cluster_transforms_list = [
        {
            "cluster_id": str(x["cluster_id"]),
            "x": int(x["x"]),
            "y": int(x["y"]),
            "rot": int(x["rot"]),
        }
        for x in cluster_transforms
        if isinstance(x, dict) and isinstance(x.get("cluster_id"), str)
    ]

    materialized = _materialize_variantized_clusters(
        tools,
        clusters_outlines=clusters_outlines,
        selected_variants=_selected_variants_map(selected_variants_list),
    )
    quality = verify.get("quality") or {}
    quality_gate = verify.get("quality_gate") or {}
    repair_guidance = verify.get("repair_guidance") or {}
    acceptable = bool(verify.get("acceptable_valid"))
    macro_ready = _verify_macro_ready(verify)
    micro_ready = acceptable

    return {
        "phase": "micro_repair" if macro_ready else "macro_blocked",
        "ready": macro_ready,
        "needed": bool(macro_ready and not acceptable),
        "seed_kind": (
            "acceptable_valid"
            if acceptable
            else "macro_ready_complete"
            if macro_ready
            else "hard_valid_complete"
        ),
        "solver_status": solver_status,
        "grid_mm": int(grid_mm),
        "seed_layout": {
            "cluster_transforms": cluster_transforms_list,
            "selected_variants": selected_variants_list,
        },
        "seed_metrics": {
            "layout_score": int(quality.get("layout_score") or 0),
            "state_signature": str(verify.get("state_signature") or ""),
            "hard_valid": bool(verify.get("hard_valid")),
            "acceptable_valid": acceptable,
            "macro_ready": macro_ready,
            "micro_ready": micro_ready,
            "macro_penalty_mm": int(quality.get("macro_penalty_mm") or 0),
            "micro_penalty_mm": int(quality.get("micro_penalty_mm") or 0),
            "quality_gate_reasons": list(quality_gate.get("reasons") or []),
        },
        "repair_targets": {
            "prioritized_clusters": list(
                repair_guidance.get("prioritized_clusters") or []
            ),
            "conflict_sets": list(repair_guidance.get("conflict_sets") or []),
            "top_critical_orientation_issues": list(
                (quality.get("critical_orientation_debug") or [])
            )[:16],
            "top_orientation_issues": list((quality.get("orientation_debug") or []))[
                :24
            ],
        },
        "edit_contract": {
            "must_preserve_hard_valid": True,
            "hard_constraints_must_hold": [
                "inside_room",
                "no_cluster_overlap",
                "no_hard_obstacle_intersection",
                "door_swing_clear",
            ],
            "primary_soft_targets": [
                "front_to_open_space",
                "preserve_front_access",
                "face_window",
                "critical_orientation_penalty",
                "focal_pair_penalty",
            ],
            "allowed_edit_levels": [
                "cluster_variant",
                "cluster_pose",
                "object_pose",
                "object_swap",
                "object_nudge",
            ],
        },
        "materialized_clusters_outlines": materialized,
        "relation_plan_used": relation_plan_used,
        "cluster_constraints_used": cluster_constraints_used,
        "seed_verify": verify,
        "candidate_counts": dict(candidate_counts or {}),
    }


def _solution_record_from_pool_entry(
    *,
    pool_entry: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
    solution_index: int,
) -> dict[str, Any]:
    chosen_raw = pool_entry.get("chosen")
    chosen = chosen_raw if isinstance(chosen_raw, Mapping) else {}
    verify_raw = pool_entry.get("verify")
    verify = verify_raw if isinstance(verify_raw, Mapping) else {}
    cluster_transforms = [
        dict(item)
        for item in pool_entry.get("cluster_transforms", [])
        if isinstance(item, dict)
    ]
    selected_variants = [
        dict(item)
        for item in pool_entry.get("selected_variants", [])
        if isinstance(item, dict)
    ]
    quality = _global_quality_scores(
        verify=verify,
        relation_plan=relation_plan,
        chosen=chosen,
    )
    diagnostics = _solution_diagnostics(
        verify=verify,
        relation_plan=relation_plan,
        chosen=chosen,
    )
    record = {
        "solution_id": f"sol_{solution_index:02d}",
        "quality_rank": solution_index,
        "hard_valid": bool(verify.get("hard_valid")),
        "selected_variants": selected_variants,
        "cluster_transforms": cluster_transforms,
        "global_quality": quality,
        "diagnostics": diagnostics,
        "verify_summary": {
            "state_signature": str(verify.get("state_signature") or ""),
            "layout_score": _as_int(_quality_dict(verify).get("layout_score")),
            "macro_penalty_mm": _as_int(_quality_dict(verify).get("macro_penalty_mm")),
            "micro_penalty_mm": _as_int(_quality_dict(verify).get("micro_penalty_mm")),
            "quality_gate_reasons": list(
                (verify.get("quality_gate") or {}).get("reasons") or []
            )
            if isinstance(verify.get("quality_gate"), Mapping)
            else [],
        },
        "_rank_key": (
            quality["total_score"],
            quality["functionality_score"],
            quality["semantic_coherence_score"],
            _as_int(pool_entry.get("objective_value")),
        ),
        "_verify": dict(verify),
        "_chosen": dict(chosen),
        "_signature": str(pool_entry.get("signature") or ""),
    }
    violations = _solution_publishability_violations(
        record,
        relation_plan=relation_plan,
    )
    record["publishable"] = not violations
    record["publishability_reasons"] = violations
    return record


def _rank_solution_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    def rank_tuple(item: Mapping[str, Any]) -> tuple[Any, ...]:
        explicit = item.get("_rank_key")
        if explicit:
            return tuple(explicit)
        quality = item.get("global_quality")
        quality_map = quality if isinstance(quality, Mapping) else {}
        diagnostics = item.get("diagnostics")
        diagnostics_map = diagnostics if isinstance(diagnostics, Mapping) else {}
        return (
            _as_float(quality_map.get("total_score")),
            _as_float(quality_map.get("functionality_score")),
            _as_float(quality_map.get("semantic_coherence_score")),
            1 if diagnostics_map.get("dominant_anchor_correct") else 0,
        )

    ranked = sorted(
        records,
        key=lambda item: (
            bool(item.get("hard_valid")),
            rank_tuple(item),
            -_as_int((item.get("verify_summary") or {}).get("macro_penalty_mm")),
            -_as_int((item.get("verify_summary") or {}).get("micro_penalty_mm")),
        ),
        reverse=True,
    )
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in ranked:
        signature = str(item.get("_signature") or "")
        if signature and signature in seen:
            continue
        if signature:
            seen.add(signature)
        clean = {
            key: value
            for key, value in item.items()
            if key not in {"_rank_key", "_verify", "_chosen", "_signature"}
        }
        clean["quality_rank"] = len(out) + 1
        clean["solution_id"] = f"sol_{len(out) + 1:02d}"
        out.append(clean)
    return out


def _best_solution_debug_record(
    records: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Candidate]]:
    if not records:
        return None, {}
    ranked = sorted(
        records,
        key=lambda item: tuple(item.get("_rank_key") or ()),
        reverse=True,
    )
    best = ranked[0]
    verify_raw = best.get("_verify")
    chosen_raw = best.get("_chosen")
    verify = verify_raw if isinstance(verify_raw, dict) else None
    chosen = chosen_raw if isinstance(chosen_raw, dict) else {}
    return verify, chosen


def solve_layout(
    *,
    room_model: Any,
    clusters_outlines: Any,
    relation_plan: Any,
    cluster_constraints: Any | None = None,
    grid_mm: int,
    tools_path: str | os.PathLike[str],
    max_variants_per_cluster: int = DEFAULT_MAX_VARIANTS_PER_CLUSTER,
    initial_candidates_per_cluster: int = DEFAULT_MAX_POSE_CANDIDATES_PER_VARIANT,
    max_rounds: int = DEFAULT_MAX_DEGRADATION_ROUNDS,
    time_limit_s: float = DEFAULT_SOLVER_TIME_LIMIT_SEC_PER_CONCEPT,
    num_workers: int = 8,
    max_feasible_solutions_per_concept: int = DEFAULT_MAX_FEASIBLE_SOLUTIONS_PER_CONCEPT,
) -> dict[str, Any]:
    (
        clusters_outlines,
        relation_plan,
        cluster_constraints,
    ) = _filter_unsat_input_clusters(
        clusters_outlines=clusters_outlines,
        relation_plan=relation_plan if isinstance(relation_plan, dict) else None,
        cluster_constraints=(
            cluster_constraints if isinstance(cluster_constraints, dict) else None
        ),
    )

    tools = _load_tools_module(tools_path)

    room_model_c, clusters_c = _canonicalize_inputs(
        tools, room_model, clusters_outlines
    )
    cluster_constraints_n = _normalize_cluster_constraints(cluster_constraints)
    relation_plan_aug = _augment_relation_plan_from_cluster_constraints(
        relation_plan,
        cluster_constraints_n,
    )
    cluster_allowed_rots = _cluster_allowed_rotations(cluster_constraints_n)
    clusters_c = _filter_clusters_outlines_by_concept_variant_policy(
        clusters_c,
        relation_plan_aug,
    )

    variant_payload_lookup = _build_variant_payload_lookup(
        tools,
        clusters_c,
        max_variants_per_cluster=max_variants_per_cluster,
    )
    candidates_by_cluster, cluster_order = _enumerate_initial_candidates(
        tools,
        room_model=room_model_c,
        clusters_outlines=clusters_c,
        relation_plan=relation_plan_aug,
        grid_mm=grid_mm,
        max_candidates_per_cluster=initial_candidates_per_cluster,
        max_variants_per_cluster=max_variants_per_cluster,
    )
    candidates_by_cluster = _filter_candidates_by_allowed_rotations(
        candidates_by_cluster,
        cluster_allowed_rots,
    )
    candidates_by_cluster = _filter_candidates_by_concept_variant_policy(
        candidates_by_cluster,
        relation_plan_aug,
    )
    candidates_by_cluster = _filter_protected_topology_candidates(
        candidates_by_cluster,
        relation_plan_aug,
    )
    (
        clusters_c,
        relation_plan_aug,
        cluster_constraints_n,
        candidates_by_cluster,
        variant_payload_lookup,
        cluster_allowed_rots,
        skipped_candidate_clusters,
    ) = _skip_clusters_without_candidates(
        clusters_outlines=clusters_c,
        relation_plan=relation_plan_aug,
        cluster_constraints=cluster_constraints_n,
        candidates_by_cluster=candidates_by_cluster,
        variant_payload_lookup=variant_payload_lookup,
        cluster_allowed_rotations=cluster_allowed_rots,
    )
    if skipped_candidate_clusters:
        logger.info(
            "Solver skipped clusters without hard-valid candidates: %s",
            skipped_candidate_clusters,
        )
    cluster_ids = _ordered_cluster_ids(
        candidates_by_cluster,
        cluster_order=cluster_order,
    )
    required_clusters_without_candidates = [
        cluster_id
        for cluster_id in cluster_ids
        if not candidates_by_cluster.get(cluster_id)
    ]
    if required_clusters_without_candidates:
        return {
            "status": "UNSAT",
            "selected_concept_id": _concept_id_from_relation_plan(relation_plan_aug),
            "solutions": [],
            "best_solution_id": None,
            "grid_mm": grid_mm,
            "cluster_transforms": [],
            "selected_variants": [],
            "global_quality": {},
            "diagnostics": {},
            "degradation_applied": [],
            "missing": [],
            "conflicts": [
                "Required clusters have no concept-compatible hard-valid candidates."
            ],
            "notes": [
                "Candidate generation rejected all variants for required clusters: "
                f"{required_clusters_without_candidates}",
            ],
            "skipped_clusters": skipped_candidate_clusters,
            "placer_seed": {
                "phase": "macro_blocked",
                "ready": False,
                "needed": False,
                "reason": "required_clusters_without_candidates",
                "required_clusters_without_candidates": (
                    required_clusters_without_candidates
                ),
            },
            "solver_debug": {
                "candidate_counts": {
                    cid: len(vals) for cid, vals in candidates_by_cluster.items()
                },
                "cluster_constraints_summary": _cluster_constraints_summary(
                    cluster_constraints_n
                ),
            },
        }
    if not cluster_ids:
        notes = ["No candidates were generated."]
        if skipped_candidate_clusters:
            notes.insert(
                0,
                f"Skipped clusters without hard-valid candidates: {skipped_candidate_clusters}",
            )
        return {
            "status": "UNSAT",
            "grid_mm": grid_mm,
            "cluster_transforms": [],
            "selected_variants": [],
            "notes": notes,
            "skipped_clusters": skipped_candidate_clusters,
            "placer_seed": {
                "phase": "micro_repair",
                "ready": False,
                "needed": False,
                "reason": "no_candidates_generated",
                "skipped_clusters": skipped_candidate_clusters,
            },
        }

    solution_records: list[dict[str, Any]] = []
    best_verify: dict[str, Any] | None = None
    best_transforms: list[dict[str, Any]] = []
    best_variants: list[dict[str, str]] = []
    best_chosen: dict[str, Candidate] = {}
    used_relaxed_candidate_expansion = False
    infeasible_candidate_set = False
    wall_pin_focus = True
    solver_pool_debug: list[dict[str, Any]] = []
    root_strategy_debug: dict[str, Any] = {
        "root_cluster_id": _select_root_cluster_id(
            candidates_by_cluster,
            relation_plan=relation_plan_aug,
            cluster_order=cluster_ids,
        ),
        "cluster_order": list(cluster_ids),
        "used_wall_pin_focus": True,
        "wall_pin_focus_requested": True,
    }

    current_candidates = {
        cid: list(vals) for cid, vals in candidates_by_cluster.items()
    }
    current_candidates = _cap_candidate_search_space(current_candidates)

    for round_idx in range(max_rounds):
        logger.info(
            "Solver round %s/%s: candidate_counts=%s",
            round_idx + 1,
            max_rounds,
            {cid: len(vals) for cid, vals in current_candidates.items()},
        )
        round_candidates, round_cluster_ids, candidate_bonus_by_key, round_strategy = (
            _prepare_round_candidates(
                current_candidates,
                relation_plan=relation_plan_aug,
                cluster_order=cluster_ids,
                wall_pin_focus=wall_pin_focus,
            )
        )
        root_strategy_debug = round_strategy
        try:
            pool, pool_debug = _solve_solution_pool_one_round(
                tools=tools,
                room_model=room_model_c,
                clusters_outlines=clusters_c,
                relation_plan=relation_plan_aug,
                grid_mm=grid_mm,
                candidates_by_cluster=round_candidates,
                cluster_order=round_cluster_ids,
                variant_payload_lookup=variant_payload_lookup,
                time_limit_s=time_limit_s,
                num_workers=num_workers,
                candidate_bonus_by_key=candidate_bonus_by_key,
                max_solutions=max_feasible_solutions_per_concept,
            )
            solver_pool_debug.append(
                {
                    "round": round_idx + 1,
                    **pool_debug,
                    "candidate_counts": {
                        cid: len(vals) for cid, vals in round_candidates.items()
                    },
                }
            )
        except RuntimeError as exc:
            if bool(round_strategy.get("used_wall_pin_focus")):
                if not used_relaxed_candidate_expansion:
                    logger.info(
                        "Solver root wall-pin focus was infeasible at round %s/%s; "
                        "expanding candidates while preserving planner wall focus.",
                        round_idx + 1,
                        max_rounds,
                    )
                    current_candidates = _augment_with_relaxed_candidates(
                        tools,
                        room_model=room_model_c,
                        clusters_outlines=clusters_c,
                        relation_plan=relation_plan_aug,
                        grid_mm=grid_mm,
                        current_candidates=current_candidates,
                        cluster_allowed_rotations=cluster_allowed_rots,
                        max_variants_per_cluster=max_variants_per_cluster,
                        base_candidate_limit=initial_candidates_per_cluster,
                        cluster_order=cluster_ids,
                    )
                    current_candidates = _cap_candidate_search_space(current_candidates)
                    used_relaxed_candidate_expansion = True
                    continue
                logger.warning(
                    "Solver exact solve remained infeasible under mandatory wall-pin focus: %s",
                    exc,
                )
                infeasible_candidate_set = True
                break
            if not used_relaxed_candidate_expansion:
                logger.info(
                    "Solver exact solve was infeasible at round %s/%s; "
                    "expanding candidates with relaxed macro pairwise hints.",
                    round_idx + 1,
                    max_rounds,
                )
                current_candidates = _augment_with_relaxed_candidates(
                    tools,
                    room_model=room_model_c,
                    clusters_outlines=clusters_c,
                    relation_plan=relation_plan_aug,
                    grid_mm=grid_mm,
                    current_candidates=current_candidates,
                    cluster_allowed_rotations=cluster_allowed_rots,
                    max_variants_per_cluster=max_variants_per_cluster,
                    base_candidate_limit=initial_candidates_per_cluster,
                    cluster_order=cluster_ids,
                )
                current_candidates = _cap_candidate_search_space(current_candidates)
                used_relaxed_candidate_expansion = True
                continue
            logger.warning("Solver exact solve remained infeasible: %s", exc)
            infeasible_candidate_set = True
            break

        if not pool:
            if not used_relaxed_candidate_expansion:
                logger.info(
                    "Solver found no complete hard-valid solution at round %s/%s; "
                    "expanding candidates with relaxed macro pairwise hints.",
                    round_idx + 1,
                    max_rounds,
                )
                current_candidates = _augment_with_relaxed_candidates(
                    tools,
                    room_model=room_model_c,
                    clusters_outlines=clusters_c,
                    relation_plan=relation_plan_aug,
                    grid_mm=grid_mm,
                    current_candidates=current_candidates,
                    cluster_allowed_rotations=cluster_allowed_rots,
                    max_variants_per_cluster=max_variants_per_cluster,
                    base_candidate_limit=initial_candidates_per_cluster,
                    cluster_order=cluster_ids,
                )
                current_candidates = _cap_candidate_search_space(current_candidates)
                used_relaxed_candidate_expansion = True
                continue
            infeasible_candidate_set = True
            break

        for entry in pool:
            solution_records.append(
                _solution_record_from_pool_entry(
                    pool_entry=entry,
                    relation_plan=relation_plan_aug,
                    solution_index=len(solution_records) + 1,
                )
            )

        best_round_records = sorted(
            solution_records,
            key=lambda item: tuple(item.get("_rank_key") or ()),
            reverse=True,
        )
        if best_round_records:
            best_record = best_round_records[0]
            best_verify_raw = best_record.get("_verify")
            best_chosen_raw = best_record.get("_chosen")
            best_verify = best_verify_raw if isinstance(best_verify_raw, dict) else None
            best_chosen = best_chosen_raw if isinstance(best_chosen_raw, dict) else {}
            best_transforms = [
                dict(item)
                for item in best_record.get("cluster_transforms", [])
                if isinstance(item, dict)
            ]
            best_variants = [
                dict(item)
                for item in best_record.get("selected_variants", [])
                if isinstance(item, dict)
            ]

        publishable_records = [
            item
            for item in solution_records
            if _is_publishable_solution(item, relation_plan=relation_plan_aug)
        ]
        if (
            publishable_records
            and len(solution_records) >= max_feasible_solutions_per_concept
        ):
            logger.info(
                "Solver collected %s feasible solutions including %s publishable "
                "solutions at round %s/%s.",
                len(solution_records),
                len(publishable_records),
                round_idx + 1,
                max_rounds,
            )
            break

        repair = (
            best_verify.get("repair_guidance") if isinstance(best_verify, dict) else {}
        )
        if not isinstance(repair, Mapping):
            repair = {}
        problem_clusters = [
            x.get("cluster_id")
            for x in (repair.get("prioritized_clusters") or [])
            if isinstance(x, dict) and isinstance(x.get("cluster_id"), str)
        ]
        problem_clusters = _primary_seed_problem_clusters(
            problem_clusters,
            cluster_ids=cluster_ids,
            max_clusters=2,
        )

        logger.info(
            "Solver refine clusters at round %s/%s: %s",
            round_idx + 1,
            max_rounds,
            problem_clusters,
        )
        current_candidates = _expand_problem_clusters(
            tools,
            room_model=room_model_c,
            clusters_outlines=clusters_c,
            relation_plan=relation_plan_aug,
            grid_mm=grid_mm,
            candidates_by_cluster=current_candidates,
            chosen=best_chosen,
            problematic_cluster_ids=problem_clusters,
            max_variants_per_cluster=max_variants_per_cluster,
            cluster_allowed_rotations=cluster_allowed_rots,
            add_limit_per_cluster=_round_candidate_growth_limit(
                initial_candidates_per_cluster
            ),
        )
        # Keep the search exact but bounded.
        for cid, vals in current_candidates.items():
            current_candidates[cid] = _dedupe_candidates(vals)[
                : _expanded_candidate_limit(initial_candidates_per_cluster)
            ]
        current_candidates = _cap_candidate_search_space(current_candidates)

    publishable_solution_records = [
        item
        for item in solution_records
        if _is_publishable_solution(item, relation_plan=relation_plan_aug)
    ]
    ranked_solutions = _rank_solution_records(publishable_solution_records)[
        :DEFAULT_MAX_CROSS_CONCEPT_FINALISTS
    ]
    all_ranked_solutions = _rank_solution_records(solution_records)[
        :DEFAULT_MAX_CROSS_CONCEPT_FINALISTS
    ]
    if ranked_solutions:
        best_solution = ranked_solutions[0]
        best_solution_id = str(best_solution.get("solution_id") or "sol_01")
        best_transforms = [
            dict(item)
            for item in best_solution.get("cluster_transforms", [])
            if isinstance(item, dict)
        ]
        best_variants = [
            dict(item)
            for item in best_solution.get("selected_variants", [])
            if isinstance(item, dict)
        ]
        best_signature = _solution_signature(best_transforms, best_variants)
        for record in solution_records:
            if str(record.get("_signature") or "") != best_signature:
                continue
            best_verify_raw = record.get("_verify")
            best_chosen_raw = record.get("_chosen")
            best_verify = best_verify_raw if isinstance(best_verify_raw, dict) else None
            best_chosen = best_chosen_raw if isinstance(best_chosen_raw, dict) else {}
            break
        candidate_counts = {cid: len(vals) for cid, vals in current_candidates.items()}
        status = "OK"
        degradation_applied = (
            [
                {
                    "type": "skip_clusters_without_candidates",
                    "cluster_ids": skipped_candidate_clusters,
                }
            ]
            if skipped_candidate_clusters
            else []
        )
        return {
            "status": status,
            "selected_concept_id": _concept_id_from_relation_plan(relation_plan_aug),
            "solutions": ranked_solutions,
            "best_solution_id": best_solution_id,
            "grid_mm": grid_mm,
            "cluster_transforms": best_transforms,
            "selected_variants": best_variants,
            "global_quality": best_solution.get("global_quality") or {},
            "diagnostics": best_solution.get("diagnostics") or {},
            "degradation_applied": degradation_applied,
            "missing": [],
            "conflicts": [],
            "notes": [
                *(
                    [
                        "Skipped clusters without hard-valid candidates: "
                        f"{skipped_candidate_clusters}"
                    ]
                    if skipped_candidate_clusters
                    else []
                ),
                "Solved with quality-aware finite-pose CP-SAT over enumerated candidates.",
                "Filtered hard-valid solutions through the publishable gate before reranking.",
                f"macro_root_cluster={root_strategy_debug.get('root_cluster_id')}",
                f"root_wall_pin_focus_used={bool(root_strategy_debug.get('used_wall_pin_focus'))}",
                f"solution_pool_size={len(ranked_solutions)}",
            ],
            "skipped_clusters": skipped_candidate_clusters,
            "placer_seed": _build_placer_seed(
                tools,
                clusters_outlines=clusters_c,
                relation_plan_used=relation_plan_aug,
                cluster_constraints_used=cluster_constraints_n,
                grid_mm=grid_mm,
                cluster_transforms=best_transforms,
                selected_variants=best_variants,
                verify=best_verify,
                solver_status=status,
                candidate_counts=candidate_counts,
            ),
            "solver_debug": {
                "best_verify": best_verify,
                "candidate_counts": candidate_counts,
                "macro_strategy": root_strategy_debug,
                "solution_pool_debug": solver_pool_debug,
                "unpublishable_solution_count": max(
                    0,
                    len(solution_records) - len(publishable_solution_records),
                ),
                "cluster_constraints_summary": _cluster_constraints_summary(
                    cluster_constraints_n
                ),
            },
        }

    if all_ranked_solutions:
        candidate_counts = {cid: len(vals) for cid, vals in current_candidates.items()}
        reject_reasons = sorted(
            {
                reason
                for item in solution_records
                for reason in _solution_publishability_violations(
                    item,
                    relation_plan=relation_plan_aug,
                )
            }
        )
        notes = [
            "Found complete hard-valid layouts, but none passed the publishable gate.",
            "Returning HARD_VALID_BUT_UNACCEPTABLE instead of publishing a macro-bad layout.",
            f"publishability_reasons={json.dumps(reject_reasons, ensure_ascii=False)}",
            f"macro_root_cluster={root_strategy_debug.get('root_cluster_id')}",
            f"root_wall_pin_focus_used={bool(root_strategy_debug.get('used_wall_pin_focus'))}",
            f"hard_valid_solution_pool_size={len(all_ranked_solutions)}",
        ]
        return {
            "status": "HARD_VALID_BUT_UNACCEPTABLE",
            "selected_concept_id": _concept_id_from_relation_plan(relation_plan_aug),
            "solutions": [],
            "best_solution_id": None,
            "grid_mm": grid_mm,
            "cluster_transforms": [],
            "selected_variants": [],
            "global_quality": {},
            "diagnostics": {},
            "degradation_applied": [],
            "missing": [],
            "conflicts": notes,
            "notes": notes,
            "skipped_clusters": skipped_candidate_clusters,
            "placer_seed": _build_placer_seed(
                tools,
                clusters_outlines=clusters_c,
                relation_plan_used=relation_plan_aug,
                cluster_constraints_used=cluster_constraints_n,
                grid_mm=grid_mm,
                cluster_transforms=best_transforms,
                selected_variants=best_variants,
                verify=best_verify,
                solver_status="HARD_VALID_BUT_UNACCEPTABLE",
                candidate_counts=candidate_counts,
            ),
            "solver_debug": {
                "best_verify": best_verify,
                "best_transforms": best_transforms,
                "best_selected_variants": best_variants,
                "candidate_counts": candidate_counts,
                "macro_strategy": root_strategy_debug,
                "solution_pool_debug": solver_pool_debug,
                "unpublishable_solutions": all_ranked_solutions,
                "publishability_reasons": reject_reasons,
                "cluster_constraints_summary": _cluster_constraints_summary(
                    cluster_constraints_n
                ),
            },
        }

    notes = [
        "No complete hard-valid solution found within the finite candidate space.",
    ]
    if infeasible_candidate_set:
        notes.insert(
            0,
            "No feasible exact assignment existed over the available macro candidate set.",
        )
    if used_relaxed_candidate_expansion:
        notes.append(
            "Expanded the candidate pool once with relaxed macro pairwise hints."
        )
    notes.append(f"macro_root_cluster={root_strategy_debug.get('root_cluster_id')}")
    notes.append(
        f"root_wall_pin_focus_used={bool(root_strategy_debug.get('used_wall_pin_focus'))}"
    )
    if best_verify is not None:
        reasons = (best_verify.get("quality_gate") or {}).get("reasons") or []
        notes.append(f"quality_gate_reasons={json.dumps(reasons, ensure_ascii=False)}")
        notes.append(f"state_signature={best_verify.get('state_signature')}")
        notes.append(
            f"best_layout_score={_as_int(_quality_dict(best_verify).get('layout_score'))}"
        )
    candidate_counts = {cid: len(vals) for cid, vals in current_candidates.items()}
    return {
        "status": "UNSAT",
        "selected_concept_id": _concept_id_from_relation_plan(relation_plan_aug),
        "solutions": [],
        "best_solution_id": None,
        "grid_mm": grid_mm,
        "cluster_transforms": [],
        "selected_variants": [],
        "global_quality": {},
        "diagnostics": {},
        "degradation_applied": [],
        "missing": [],
        "conflicts": notes,
        "notes": [
            *(
                [
                    "Skipped clusters without hard-valid candidates: "
                    f"{skipped_candidate_clusters}"
                ]
                if skipped_candidate_clusters
                else []
            ),
            *notes,
        ],
        "skipped_clusters": skipped_candidate_clusters,
        "placer_seed": _build_placer_seed(
            tools,
            clusters_outlines=clusters_c,
            relation_plan_used=relation_plan_aug,
            cluster_constraints_used=cluster_constraints_n,
            grid_mm=grid_mm,
            cluster_transforms=best_transforms,
            selected_variants=best_variants,
            verify=best_verify,
            solver_status="UNSAT",
            candidate_counts=candidate_counts,
        ),
        "solver_debug": {
            "best_verify": best_verify,
            "best_transforms": best_transforms,
            "best_selected_variants": best_variants,
            "candidate_counts": candidate_counts,
            "macro_strategy": root_strategy_debug,
            "solution_pool_debug": solver_pool_debug,
            "cluster_constraints_summary": _cluster_constraints_summary(
                cluster_constraints_n
            ),
        },
    }


def _normalize_point_list(points: Any) -> list[dict[str, int]]:
    out: list[dict[str, int]] = []
    if not isinstance(points, list):
        return out
    for point in points:
        try:
            if isinstance(point, Mapping):
                x = point.get("x")
                y = point.get("y")
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                x = point[0]
                y = point[1]
            else:
                continue
            out.append({"x": int(round(float(x))), "y": int(round(float(y)))})
        except (TypeError, ValueError):
            continue
    return out


def _normalize_polygon_list(polygons: Any) -> list[list[dict[str, int]]]:
    if not isinstance(polygons, list):
        return []
    out: list[list[dict[str, int]]] = []
    for polygon in polygons:
        normalized = _normalize_point_list(polygon)
        if normalized:
            out.append(normalized)
    return out


def _normalize_room_model_for_tools(room_model: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(room_model)
    room = out.get("room")
    if isinstance(room, Mapping) and isinstance(room.get("polygon_ccw"), list):
        return out

    polygon = _normalize_point_list(
        room_model.get("polygon_mm") or room_model.get("polygon_ccw")
    )
    if polygon:
        out["room"] = {
            **(dict(room) if isinstance(room, Mapping) else {}),
            "polygon_ccw": polygon,
        }
    return out


def _cluster_payload_from_variant_bundle(
    row: Mapping[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    cluster_id = str(row.get("cluster_id") or "").strip()
    variants = row.get("variant_bundle")
    if not cluster_id or not isinstance(variants, list) or not variants:
        return None
    first_variant = next((item for item in variants if isinstance(item, dict)), None)
    if first_variant is None:
        return None

    outline_polygons = _normalize_polygon_list(
        first_variant.get("interaction_hull_polygons_mm")
    ) or _normalize_polygon_list(first_variant.get("tight_hull_polygons_mm"))
    outline = _normalize_point_list(
        first_variant.get("interaction_hull_polygon_mm")
        or first_variant.get("tight_hull_polygon_mm")
    )
    if outline and not outline_polygons:
        outline_polygons = [outline]
    rects = [
        dict(item)
        for item in first_variant.get("interaction_placements")
        or first_variant.get("local_placements")
        or []
        if isinstance(item, Mapping)
    ]
    if not outline_polygons and not rects:
        return None
    xs = [
        point["x"]
        for polygon in outline_polygons
        for point in polygon
        if isinstance(point, Mapping)
    ]
    ys = [
        point["y"]
        for polygon in outline_polygons
        for point in polygon
        if isinstance(point, Mapping)
    ]
    if not xs or not ys:
        xs = []
        ys = []
        for rect in rects:
            try:
                x = int(round(float(rect.get("x", 0))))
                y = int(round(float(rect.get("y", 0))))
                w = int(round(float(rect.get("w", 0))))
                h = int(round(float(rect.get("h", 0))))
            except (TypeError, ValueError):
                continue
            xs.extend([x, x + max(w, 0)])
            ys.extend([y, y + max(h, 0)])
    if not xs or not ys:
        return None
    payload = {
        "status": "OK",
        "cluster_id": cluster_id,
        "local_frame": {
            "unit": "mm",
            "grid_mm": DEFAULT_GRID_MM,
            "origin_note": "(0,0) is an arbitrary local origin for this cluster",
        },
        "local_placements": deepcopy(first_variant.get("local_placements") or []),
        "cluster_footprint": {
            "type": "union_of_rects",
            "rects": rects,
            "local_bbox": {
                "min_x": min(xs),
                "min_y": min(ys),
                "max_x": max(xs),
                "max_y": max(ys),
            },
            "outline_polygons_ccw": outline_polygons,
        },
        "variant_bundle": [dict(item) for item in variants if isinstance(item, dict)],
        "notes": [],
        "missing": [],
        "conflicts": [],
    }
    return cluster_id, payload


def _clusters_outlines_from_variant_bundles(
    cluster_variant_bundles: Any,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not isinstance(cluster_variant_bundles, list):
        return out
    for row in cluster_variant_bundles:
        if not isinstance(row, Mapping):
            continue
        converted = _cluster_payload_from_variant_bundle(row)
        if converted is None:
            continue
        cluster_id, payload = converted
        out[cluster_id] = payload
    return out


def _relation_plan_for_macro_concept(
    *,
    concept: Mapping[str, Any],
    room_model: Mapping[str, Any],
    semantic_layout_program: Mapping[str, Any],
) -> dict[str, Any]:
    explicit = concept.get("solver_relation_plan")
    if isinstance(explicit, dict):
        out = dict(explicit)
        out["macro_concept"] = {
            key: value
            for key, value in concept.items()
            if key != "solver_relation_plan"
        }
        return out

    relation_keys = {
        "cluster_affinities",
        "cluster_orientations",
        "object_orientations",
        "cluster_relations",
        "cluster_directional_relations",
        "circulation_plan",
        "layout_intent_profile",
    }
    if any(key in concept for key in relation_keys):
        out = {
            key: deepcopy(concept.get(key)) for key in relation_keys if key in concept
        }
        out["status"] = "OK"
        out["macro_concept"] = {
            key: value for key, value in concept.items() if key not in relation_keys
        }
        return out

    from agent.seed_concept_generator import solver_plan_from_concept

    return solver_plan_from_concept(
        concept=dict(concept),
        room_model_json=dict(room_model),
        room_type=str(semantic_layout_program.get("room_type") or ""),
    )


def _macro_concepts(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    bundle = payload.get("macro_concept_bundle")
    if not isinstance(bundle, Mapping):
        return []
    concepts = bundle.get("concepts")
    if not isinstance(concepts, list):
        return []
    out = [dict(item) for item in concepts if isinstance(item, Mapping)]
    return out[:DEFAULT_MAX_CONCEPTS]


def _inventory_cluster_constraints(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    inventory = payload.get("inventory_decision_program")
    if isinstance(inventory, Mapping) and isinstance(inventory.get("clusters"), list):
        return dict(inventory)
    semantic = payload.get("semantic_layout_program")
    if isinstance(semantic, Mapping) and isinstance(semantic.get("clusters"), list):
        return dict(semantic)
    return None


def _primary_cluster_ids_from_concepts(
    concepts: Sequence[Mapping[str, Any]],
) -> set[str]:
    protected: set[str] = set()
    for concept in concepts:
        for row in concept.get("cluster_zone_plan") or []:
            if not isinstance(row, Mapping):
                continue
            role = str(row.get("role") or row.get("cluster_role") or "").lower()
            claim = str(row.get("anchor_role") or row.get("wall_claim") or "").lower()
            cluster_id = str(row.get("cluster_id") or "").strip()
            if cluster_id and (
                "primary" in role
                or "dominant" in role
                or "anchor" in role
                or claim == "strong"
            ):
                protected.add(cluster_id)
    return protected


def _controlled_degradation_candidates(
    *,
    payload: Mapping[str, Any],
    concepts: Sequence[Mapping[str, Any]],
    available_cluster_ids: Sequence[str],
) -> list[str]:
    protected = _primary_cluster_ids_from_concepts(concepts)
    scores: dict[str, tuple[int, str]] = {}

    def consider(cluster_id: str, marker: str, priority: int) -> None:
        clean_id = str(cluster_id or "").strip()
        if (
            not clean_id
            or clean_id in protected
            or clean_id not in available_cluster_ids
        ):
            return
        current = scores.get(clean_id)
        item = (priority, marker)
        if current is None or item > current:
            scores[clean_id] = item

    semantic = payload.get("semantic_layout_program")
    if isinstance(semantic, Mapping):
        for row in semantic.get("active_clusters") or []:
            if not isinstance(row, Mapping):
                continue
            cluster_id = str(row.get("cluster_id") or "").strip()
            text = json.dumps(row, sort_keys=True).lower()
            if "optional" in text:
                consider(cluster_id, "semantic_optional", 30)
            elif "support" in text:
                consider(cluster_id, "semantic_support", 20)
            elif "low" in text:
                consider(cluster_id, "semantic_low_priority", 10)

    inventory = payload.get("inventory_decision_program")
    if isinstance(inventory, Mapping):
        for row in inventory.get("cluster_decisions") or []:
            if not isinstance(row, Mapping):
                continue
            cluster_id = str(row.get("cluster_id") or "").strip()
            text = json.dumps(row, sort_keys=True).lower()
            if "optional" in text:
                consider(cluster_id, "inventory_optional", 30)
            elif "support" in text:
                consider(cluster_id, "inventory_support", 20)
            elif "low" in text:
                consider(cluster_id, "inventory_low_priority", 10)

    return [
        cluster_id
        for cluster_id, _score in sorted(
            scores.items(),
            key=lambda item: (-item[1][0], item[0]),
        )
    ]


def solve_global_layout_bundle(
    *,
    payload: dict[str, Any],
    grid_mm: int,
    tools_path: str | os.PathLike[str],
    max_variants_per_cluster: int = DEFAULT_MAX_VARIANTS_PER_CLUSTER,
    initial_candidates_per_cluster: int = DEFAULT_MAX_POSE_CANDIDATES_PER_VARIANT,
    max_rounds: int = DEFAULT_MAX_DEGRADATION_ROUNDS,
    time_limit_s: float = DEFAULT_SOLVER_TIME_LIMIT_SEC_PER_CONCEPT,
    num_workers: int = 8,
    max_feasible_solutions_per_concept: int = DEFAULT_MAX_FEASIBLE_SOLUTIONS_PER_CONCEPT,
) -> dict[str, Any]:
    required = [
        "room_model",
        "semantic_layout_program",
        "cluster_variant_bundles",
        "macro_concept_bundle",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        return {
            "status": "UNSAT",
            "selected_concept_id": None,
            "solutions": [],
            "best_solution_id": None,
            "degradation_applied": [],
            "missing": missing,
            "conflicts": [],
            "notes": ["Missing required solver input."],
        }

    room_model_raw = payload.get("room_model")
    semantic_raw = payload.get("semantic_layout_program")
    if not isinstance(room_model_raw, Mapping) or not isinstance(semantic_raw, Mapping):
        return {
            "status": "UNSAT",
            "selected_concept_id": None,
            "solutions": [],
            "best_solution_id": None,
            "degradation_applied": [],
            "missing": [],
            "conflicts": ["room_model and semantic_layout_program must be objects"],
            "notes": [],
        }

    room_model = _normalize_room_model_for_tools(room_model_raw)
    concepts = _macro_concepts(payload)
    if not concepts:
        return {
            "status": "UNSAT",
            "selected_concept_id": None,
            "solutions": [],
            "best_solution_id": None,
            "degradation_applied": [],
            "missing": ["macro_concept_bundle.concepts"],
            "conflicts": [],
            "notes": ["No macro concepts were provided."],
        }

    base_clusters = _clusters_outlines_from_variant_bundles(
        payload.get("cluster_variant_bundles")
    )
    if not base_clusters:
        return {
            "status": "UNSAT",
            "selected_concept_id": None,
            "solutions": [],
            "best_solution_id": None,
            "degradation_applied": [],
            "missing": ["cluster_variant_bundles"],
            "conflicts": [],
            "notes": ["No hard-valid cluster variant bundles were usable."],
        }

    available_cluster_ids = list(base_clusters.keys())
    degradation_order = _controlled_degradation_candidates(
        payload=payload,
        concepts=concepts,
        available_cluster_ids=available_cluster_ids,
    )
    applied_degradations: list[dict[str, Any]] = []
    dropped_ids: set[str] = set()
    concept_outputs: list[dict[str, Any]] = []
    all_solutions: list[dict[str, Any]] = []
    hard_valid_unacceptable_count = 0

    for degradation_round in range(max(1, int(max_rounds)) + 1):
        clusters_for_round = _filter_cluster_ids_from_outlines(
            base_clusters, dropped_ids
        )
        for concept in concepts:
            relation_plan = _relation_plan_for_macro_concept(
                concept=concept,
                room_model=room_model,
                semantic_layout_program=semantic_raw,
            )
            relation_plan = _filter_relation_plan_clusters(relation_plan, dropped_ids)
            if relation_plan is None:
                continue
            concept_result = solve_layout(
                room_model=room_model,
                clusters_outlines=clusters_for_round,
                relation_plan=relation_plan,
                cluster_constraints=_inventory_cluster_constraints(payload),
                grid_mm=int(grid_mm or DEFAULT_GRID_MM),
                tools_path=tools_path,
                max_variants_per_cluster=max_variants_per_cluster,
                initial_candidates_per_cluster=initial_candidates_per_cluster,
                max_rounds=max_rounds,
                time_limit_s=time_limit_s,
                num_workers=num_workers,
                max_feasible_solutions_per_concept=max_feasible_solutions_per_concept,
            )
            concept_outputs.append(concept_result)
            if str(concept_result.get("status") or "") == (
                "HARD_VALID_BUT_UNACCEPTABLE"
            ):
                debug = concept_result.get("solver_debug")
                debug_map = debug if isinstance(debug, Mapping) else {}
                hard_valid_unacceptable_count += max(
                    1,
                    len(debug_map.get("unpublishable_solutions") or []),
                )
            concept_id = str(concept_result.get("selected_concept_id") or "")
            for solution in concept_result.get("solutions") or []:
                if not isinstance(solution, dict):
                    continue
                if not bool(solution.get("publishable", True)):
                    continue
                enriched = dict(solution)
                enriched["concept_id"] = concept_id
                enriched["degradation_applied"] = list(applied_degradations)
                all_solutions.append(enriched)

        if all_solutions:
            break
        if degradation_round >= max(0, int(max_rounds)):
            break
        next_drop = next(
            (
                cluster_id
                for cluster_id in degradation_order
                if cluster_id not in dropped_ids
            ),
            None,
        )
        if next_drop is None:
            break
        dropped_ids.add(next_drop)
        applied_degradations.append(
            {
                "round": degradation_round + 1,
                "type": "drop_optional_or_support_cluster",
                "cluster_ids": [next_drop],
            }
        )

    ranked = _rank_solution_records(all_solutions)[:DEFAULT_MAX_CROSS_CONCEPT_FINALISTS]
    if ranked:
        best_solution = ranked[0]
        selected_concept_id = str(best_solution.get("concept_id") or "")
        return {
            "status": "OK",
            "selected_concept_id": selected_concept_id,
            "solutions": ranked,
            "best_solution_id": best_solution.get("solution_id"),
            "cluster_transforms": best_solution.get("cluster_transforms") or [],
            "selected_variants": best_solution.get("selected_variants") or [],
            "global_quality": best_solution.get("global_quality") or {},
            "diagnostics": best_solution.get("diagnostics") or {},
            "degradation_applied": applied_degradations,
            "missing": [],
            "conflicts": [],
            "notes": [
                "Compared publishable solution pools across macro concepts.",
                f"concepts_evaluated={len(concepts)}",
                f"solution_pool_size={len(ranked)}",
            ],
            "solver_debug": {
                "concept_statuses": [
                    {
                        "concept_id": item.get("selected_concept_id"),
                        "status": item.get("status"),
                        "solution_count": len(item.get("solutions") or []),
                    }
                    for item in concept_outputs
                ],
            },
        }

    if hard_valid_unacceptable_count > 0:
        return {
            "status": "HARD_VALID_BUT_UNACCEPTABLE",
            "selected_concept_id": None,
            "solutions": [],
            "best_solution_id": None,
            "cluster_transforms": [],
            "selected_variants": [],
            "global_quality": {},
            "diagnostics": {},
            "degradation_applied": applied_degradations,
            "missing": [],
            "conflicts": [
                "Hard-valid solution pools existed, but none passed the publishable gate."
            ],
            "notes": [
                "Degradation rounds were attempted before returning non-publishable status.",
                f"concepts_evaluated={len(concepts)}",
                f"hard_valid_unacceptable_count={hard_valid_unacceptable_count}",
            ],
            "solver_debug": {
                "concept_statuses": [
                    {
                        "concept_id": item.get("selected_concept_id"),
                        "status": item.get("status"),
                        "solution_count": len(item.get("solutions") or []),
                    }
                    for item in concept_outputs
                ],
            },
        }

    return {
        "status": "UNSAT",
        "selected_concept_id": None,
        "solutions": [],
        "best_solution_id": None,
        "cluster_transforms": [],
        "selected_variants": [],
        "global_quality": {},
        "diagnostics": {},
        "degradation_applied": applied_degradations,
        "missing": [],
        "conflicts": [
            "No hard-valid solution found across the provided macro concepts."
        ],
        "notes": [
            "Exact finite-pose search exhausted the available concept candidate pools."
        ],
        "solver_debug": {
            "concept_statuses": [
                {
                    "concept_id": item.get("selected_concept_id"),
                    "status": item.get("status"),
                    "solution_count": len(item.get("solutions") or []),
                }
                for item in concept_outputs
            ],
        },
    }


# ---------------------------------------------------------------------------
# Object-level anchor-first solver
# ---------------------------------------------------------------------------


def solve_object_level_layout(
    *,
    room_model: dict[str, Any],
    merged_clusters: dict[str, Any],
    relation_plan: dict[str, Any] | None,
    cluster_constraints: dict[str, Any] | None = None,
    grid_mm: int = GLOBAL_LAYOUT_GRID_MM,
    max_rounds: int = 3,
) -> dict[str, Any]:
    room_bbox = _object_solver_room_bbox(room_model)
    if room_bbox[2] <= room_bbox[0] or room_bbox[3] <= room_bbox[1]:
        return {
            "status": "UNSAT",
            "solver_kind": "object_level_anchor_first",
            "notes": [
                "Room bounding box could not be resolved for object-level solving."
            ],
        }

    world = _build_object_solver_world(
        room_model=room_model,
        merged_clusters=merged_clusters,
        relation_plan=relation_plan,
        cluster_constraints=cluster_constraints,
        grid_mm=grid_mm,
    )
    anchor_order = world["anchor_cluster_order"]
    anchor_candidates_by_cluster = {
        cluster_id: _generate_anchor_pose_candidates(
            cluster_program=world["clusters_by_id"][cluster_id],
            room_model=room_model,
            relation_plan=relation_plan,
            world=world,
            grid_mm=grid_mm,
        )
        for cluster_id in anchor_order
    }
    if any(
        not rows
        for cluster_id, rows in anchor_candidates_by_cluster.items()
        if not _cluster_is_solver_trial_optional(world["clusters_by_id"][cluster_id])
    ):
        offending = [
            cluster_id
            for cluster_id, rows in anchor_candidates_by_cluster.items()
            if not rows
            and not _cluster_is_solver_trial_optional(
                world["clusters_by_id"][cluster_id]
            )
        ]
        return {
            "status": "UNSAT",
            "solver_kind": "object_level_anchor_first",
            "offending_clusters": offending,
            "notes": [
                "No anchor pose candidates could be generated for one or more core clusters."
            ],
            "solver_debug": {
                "anchor_candidate_counts": {
                    k: len(v) for k, v in anchor_candidates_by_cluster.items()
                }
            },
        }

    anchor_solutions = list(
        _search_anchor_solutions(
            anchor_order=anchor_order,
            anchor_candidates_by_cluster=anchor_candidates_by_cluster,
            world=world,
            room_model=room_model,
            relation_plan=relation_plan,
            max_solutions=max(8, int(max_rounds) * 4),
        )
    )
    if not anchor_solutions:
        return {
            "status": "UNSAT",
            "solver_kind": "object_level_anchor_first",
            "offending_clusters": list(anchor_order),
            "notes": [
                "Anchor candidates were generated, but no compatible non-overlapping anchor set could be assembled."
            ],
            "solver_debug": {
                "anchor_candidate_counts": {
                    k: len(v) for k, v in anchor_candidates_by_cluster.items()
                },
                "anchor_order": list(anchor_order),
            },
        }

    solution_pool: list[dict[str, Any]] = []
    for solution in anchor_solutions:
        support_results = _place_support_objects_for_solution(
            solution=solution,
            world=world,
            room_model=room_model,
            relation_plan=relation_plan,
            grid_mm=grid_mm,
            max_solutions=OBJECT_LEVEL_MAX_SUPPORT_SOLUTIONS_PER_ANCHOR,
        )
        if not support_results:
            continue
        for support_result in support_results:
            candidate_solution = {**solution, **support_result}
            repair_summary = _repair_object_level_solution_geometry(
                solution=candidate_solution,
                world=world,
                grid_mm=grid_mm,
            )
            if repair_summary is None:
                continue
            candidate_solution["placed_objects"] = repair_summary["placed_objects"]
            candidate_solution["geometry_repair"] = repair_summary["summary"]
            verify = _verify_object_level_solution(
                solution=candidate_solution,
                world=world,
                room_model=room_model,
                relation_plan=relation_plan,
            )
            _apply_object_level_geometry_repair_penalty(
                verify, candidate_solution["geometry_repair"]
            )
            if not bool(verify.get("geometry_valid")):
                continue
            candidate_solution["verify"] = verify
            candidate_solution["score"] = _object_solution_score(
                candidate_solution, relation_plan
            )
            candidate_solution["signature"] = _object_level_solution_signature(
                candidate_solution
            )
            solution_pool.append(candidate_solution)

    ranked_pool = _rank_object_level_solution_pool(solution_pool)[
        :OBJECT_LEVEL_MAX_OBJECT_SOLUTIONS
    ]
    best_solution = ranked_pool[0] if ranked_pool else None
    if best_solution is None:
        return {
            "status": "UNSAT",
            "solver_kind": "object_level_anchor_first",
            "offending_clusters": list(anchor_order),
            "notes": [
                "Anchor placement succeeded, but no geometry-valid support-object arrangement could be assembled."
            ],
            "solver_debug": {
                "anchor_candidate_counts": {
                    k: len(v) for k, v in anchor_candidates_by_cluster.items()
                },
                "candidate_solution_count": len(solution_pool),
            },
        }

    absolute_layout = _build_absolute_layout_from_object_solution(
        solution=best_solution,
        world=world,
        room_model=room_model,
        relation_plan=relation_plan,
    )
    ranked_solutions = [
        _build_object_level_solution_payload(
            solution=item,
            world=world,
            room_model=room_model,
            relation_plan=relation_plan,
            solution_index=index,
        )
        for index, item in enumerate(ranked_pool, start=1)
    ]
    return {
        "status": "OK" if best_solution["verify"].get("hard_valid") else "PARTIAL",
        "solver_kind": "object_level_anchor_first",
        "selected_concept_id": _concept_id_from_relation_plan(relation_plan),
        "absolute_layout": absolute_layout,
        "solutions": ranked_solutions,
        "hard_valid": bool(best_solution["verify"].get("hard_valid")),
        "geometry_valid": bool(best_solution["verify"].get("geometry_valid")),
        "acceptable_valid": bool(best_solution["verify"].get("gallery_eligible")),
        "complete": bool(best_solution["verify"].get("complete")),
        "gallery_eligible": bool(best_solution["verify"].get("gallery_eligible")),
        "coverage_ratio": float(best_solution["verify"].get("coverage_ratio") or 0.0),
        "offending_clusters": list(
            best_solution["verify"].get("offending_clusters") or []
        ),
        "dropped_inventory_by_cluster": deepcopy(
            best_solution.get("dropped_inventory_by_cluster") or {}
        ),
        "cluster_transforms": [],
        "selected_variants": [],
        "verify_summary": deepcopy(best_solution["verify"]),
        "solver_debug": {
            "anchor_candidate_counts": {
                k: len(v) for k, v in anchor_candidates_by_cluster.items()
            },
            "anchor_order": anchor_order,
            "object_count": len(best_solution.get("placed_objects") or []),
            "candidate_solution_count": len(solution_pool),
            "ranked_solution_count": len(ranked_solutions),
        },
        "notes": [
            "Solved directly at object level with anchor-first placement.",
            "Local support placement now enumerates multiple non-overlapping arrangements before ranking them.",
        ],
    }


def _build_object_solver_world(
    *,
    room_model: Mapping[str, Any],
    merged_clusters: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
    cluster_constraints: Mapping[str, Any] | None,
    grid_mm: int,
) -> dict[str, Any]:
    clusters = merged_clusters.get("clusters")
    cluster_rows = (
        [row for row in clusters if isinstance(row, Mapping)]
        if isinstance(clusters, list)
        else []
    )
    clusters_by_id: dict[str, dict[str, Any]] = {}
    for row in cluster_rows:
        cluster_id = str(row.get("cluster_id") or "").strip()
        if not cluster_id:
            continue
        cluster_rules = (
            row.get("cluster_rules")
            if isinstance(row.get("cluster_rules"), Mapping)
            else {}
        )
        object_program = (
            row.get("object_program")
            if isinstance(row.get("object_program"), Mapping)
            else {}
        )
        if not object_program:
            object_program = {
                "cluster_id": cluster_id,
                "members": [
                    m for m in (row.get("members") or []) if isinstance(m, str)
                ],
                "anchors": [
                    m for m in (row.get("anchors") or []) if isinstance(m, str)
                ],
                "dominant_anchor_id": next(iter(row.get("anchors") or []), None),
                "dominant_anchor_candidates": list(
                    cluster_rules.get("dominant_anchor_candidates") or []
                ),
                "placement_order": [
                    m for m in (row.get("members") or []) if isinstance(m, str)
                ],
                "support_edges": [
                    dict(item)
                    for item in (cluster_rules.get("semantic_placements") or [])
                    if isinstance(item, Mapping)
                ],
                "protected_ids": list(
                    (cluster_rules.get("anchor_first_policy") or {}).get(
                        "protected_ids"
                    )
                    or []
                ),
                "droppable_ids": list(
                    (cluster_rules.get("anchor_first_policy") or {}).get(
                        "droppable_ids"
                    )
                    or []
                ),
                "degradation_ladder": list(
                    cluster_rules.get("degradation_ladder") or []
                ),
                "zone_claims": dict(cluster_rules.get("zone_claims") or {}),
                "required_object_ids": [
                    m for m in (row.get("anchors") or []) if isinstance(m, str)
                ],
                "optional_object_ids": [
                    m
                    for m in (row.get("members") or [])
                    if isinstance(m, str) and m not in set(row.get("anchors") or [])
                ],
                "object_specs_by_id": {},
            }
        object_specs = (
            object_program.get("object_specs_by_id")
            if isinstance(object_program.get("object_specs_by_id"), Mapping)
            else {}
        )
        if not object_specs:
            for decision in row.get("decisions") or []:
                if not isinstance(decision, Mapping):
                    continue
                object_id = str(
                    decision.get("object_type") or decision.get("category") or ""
                ).strip()
                rep_dims = (
                    decision.get("rep_dims_m")
                    if isinstance(decision.get("rep_dims_m"), Mapping)
                    else {}
                )
                object_specs[object_id] = {
                    "object_id": object_id,
                    "cluster_id": cluster_id,
                    "category": str(decision.get("category") or object_id),
                    "role": str(decision.get("role") or ""),
                    "priority": str(decision.get("priority") or ""),
                    "preserve_level": str(decision.get("preserve_level") or ""),
                    "rep_dims_mm": {
                        "L": int(round(float(rep_dims.get("L") or 0.0) * 1000.0)),
                        "W": int(round(float(rep_dims.get("W") or 0.0) * 1000.0)),
                        "H": int(round(float(rep_dims.get("H") or 0.0) * 1000.0)),
                    },
                    "allowed_rotations": list(
                        (
                            (cluster_rules.get("allowed_rotations") or {}).get(
                                object_id
                            )
                            or [0, 90, 180, 270]
                        )
                    ),
                    "front": (
                        (cluster_rules.get("facing") or {}).get(object_id) or {}
                    ).get("front"),
                }
            object_program = dict(object_program)
            object_program["object_specs_by_id"] = object_specs
        clusters_by_id[cluster_id] = {
            "cluster_id": cluster_id,
            "tag": str(row.get("tag") or ""),
            "anchors": [
                item
                for item in object_program.get("anchors") or []
                if isinstance(item, str)
            ],
            "object_program": deepcopy(object_program),
            "cluster_rules": dict(cluster_rules),
            "members": [
                item for item in row.get("members") or [] if isinstance(item, str)
            ],
        }

    anchor_order = _object_level_anchor_cluster_order(clusters_by_id, relation_plan)
    region_index = _index_room_regions(room_model)
    protected_regions = _collect_protected_regions(
        room_model, relation_plan, region_index
    )
    return {
        "clusters_by_id": clusters_by_id,
        "anchor_cluster_order": anchor_order,
        "grid_mm": max(25, int(grid_mm)),
        "region_index": region_index,
        "room_bbox": _object_solver_room_bbox(room_model),
        "room_polygon": _object_solver_room_polygon(room_model),
        "protected_regions": protected_regions,
        "cluster_forbidden_regions": _collect_cluster_forbidden_regions(
            clusters_by_id, room_model, relation_plan, region_index
        ),
    }


def _object_level_anchor_cluster_order(
    clusters_by_id: Mapping[str, Mapping[str, Any]],
    relation_plan: Mapping[str, Any] | None,
) -> list[str]:
    cluster_ids = list(clusters_by_id.keys())
    primary = _extract_layout_primary_cluster_id(
        dict(relation_plan) if isinstance(relation_plan, dict) else None
    )
    secondary = _extract_layout_secondary_cluster_id(
        dict(relation_plan) if isinstance(relation_plan, dict) else None
    )
    ordered: list[str] = []
    for candidate in (primary, secondary):
        if candidate and candidate in clusters_by_id and candidate not in ordered:
            ordered.append(candidate)
    for cluster_id, row in sorted(clusters_by_id.items()):
        if cluster_id in ordered:
            continue
        object_program = (
            row.get("object_program")
            if isinstance(row.get("object_program"), Mapping)
            else {}
        )
        if object_program.get("dominant_anchor_id"):
            ordered.append(cluster_id)
    for cluster_id in cluster_ids:
        if cluster_id not in ordered:
            ordered.append(cluster_id)
    return ordered


def _index_room_regions(
    room_model: Mapping[str, Any],
) -> dict[str, tuple[int, int, int, int]]:
    room_bbox = _object_solver_room_bbox(room_model)
    out: dict[str, tuple[int, int, int, int]] = {}

    def add_region(region_id: str, bbox: tuple[int, int, int, int] | None) -> None:
        if region_id and bbox is not None and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
            out.setdefault(region_id, bbox)

    def bbox_from_mapping(row: Mapping[str, Any]) -> tuple[int, int, int, int] | None:
        bbox = row.get("bbox") or row.get("bbox_mm")
        if isinstance(bbox, Mapping):
            return _bbox_tuple(
                bbox.get("min_x"),
                bbox.get("min_y"),
                bbox.get("max_x"),
                bbox.get("max_y"),
            )
        for key in (
            "near_polygon_ccw",
            "anchor_polygon_ccw",
            "polygon",
            "polygon_ccw",
            "points",
            "mid_polygon_ccw",
        ):
            points = row.get(key)
            if isinstance(points, Sequence) and not isinstance(points, str):
                parsed = []
                for item in points:
                    if not isinstance(item, Mapping):
                        continue
                    try:
                        parsed.append((float(item.get("x")), float(item.get("y"))))
                    except (TypeError, ValueError):
                        continue
                if len(parsed) >= 2:
                    xs = [p[0] for p in parsed]
                    ys = [p[1] for p in parsed]
                    return (
                        int(round(min(xs))),
                        int(round(min(ys))),
                        int(round(max(xs))),
                        int(round(max(ys))),
                    )
        polyline = row.get("polyline_mm")
        if isinstance(polyline, Sequence) and not isinstance(polyline, str):
            parsed = []
            for item in polyline:
                if not isinstance(item, Mapping):
                    continue
                try:
                    parsed.append((float(item.get("x")), float(item.get("y"))))
                except (TypeError, ValueError):
                    continue
            if len(parsed) >= 2:
                try:
                    width_mm = max(0, int(row.get("width_mm") or 0))
                except (TypeError, ValueError):
                    width_mm = 0
                pad = max(1, width_mm // 2)
                xs = [p[0] for p in parsed]
                ys = [p[1] for p in parsed]
                return (
                    int(round(min(xs))) - pad,
                    int(round(min(ys))) - pad,
                    int(round(max(xs))) + pad,
                    int(round(max(ys))) + pad,
                )
        return None

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            region_id = str(value.get("region_id") or value.get("id") or "").strip()
            if region_id:
                add_region(region_id, bbox_from_mapping(value))
            for child in value.values():
                walk(child)
        elif isinstance(value, Sequence) and not isinstance(value, str):
            for item in value:
                walk(item)

    walk(room_model)
    _add_primary_corridor_aliases(room_model, out, bbox_from_mapping)

    # Fallback synthetic regions
    min_x, min_y, max_x, max_y = room_bbox
    width = max_x - min_x
    height = max_y - min_y
    center_box = (
        min_x + int(round(width * 0.25)),
        min_y + int(round(height * 0.25)),
        max_x - int(round(width * 0.25)),
        max_y - int(round(height * 0.25)),
    )
    out.setdefault("center_openness_core", center_box)
    out.setdefault("floating_center_zone", center_box)
    out.setdefault(
        "wall_1_usable_200_5800_focal",
        (min_x, min_y, max_x, min_y + max(600, int(round(height * 0.18)))),
    )
    out.setdefault("wall_1_usable_200_5800_anchor", out["wall_1_usable_200_5800_focal"])
    out.setdefault(
        "wall_2_usable_200_2800_anchor",
        (max_x - max(600, int(round(width * 0.18))), min_y, max_x, max_y),
    )
    out.setdefault(
        "wall_3_usable_200_2300_anchor",
        (min_x, max_y - max(600, int(round(height * 0.18))), max_x, max_y),
    )
    out.setdefault(
        "wall_6_usable_2500_3800_focal",
        (min_x, min_y, min_x + max(600, int(round(width * 0.18))), max_y),
    )
    out.setdefault(
        "wall_6_usable_2500_3800_anchor", out["wall_6_usable_2500_3800_focal"]
    )
    out.setdefault(
        "door_1_entry_clearance",
        (
            min_x,
            max_y - max(850, int(round(height * 0.22))),
            min_x + max(1000, int(round(width * 0.22))),
            max_y,
        ),
    )
    out.setdefault(
        "door_1_to_room_center_corridor",
        (
            min_x + int(round(width * 0.15)),
            min_y + int(round(height * 0.35)),
            min_x + int(round(width * 0.55)),
            max_y - int(round(height * 0.15)),
        ),
    )
    out.setdefault(
        "window_1_daylight",
        (min_x, min_y, max_x, min_y + max(850, int(round(height * 0.24)))),
    )
    out.setdefault(
        "privacy_back_zone",
        (
            min_x + int(round(width * 0.45)),
            min_y + int(round(height * 0.45)),
            max_x,
            max_y,
        ),
    )
    return out


def _add_primary_corridor_aliases(
    room_model: Mapping[str, Any],
    region_index: dict[str, tuple[int, int, int, int]],
    bbox_from_mapping: Callable[[Mapping[str, Any]], tuple[int, int, int, int] | None],
) -> None:
    affordance = (
        room_model.get("affordance_map")
        if isinstance(room_model.get("affordance_map"), Mapping)
        else {}
    )
    corridors = affordance.get("circulation_corridors") if affordance else []
    if not isinstance(corridors, Sequence) or isinstance(corridors, str):
        return
    primary_bbox: tuple[int, int, int, int] | None = None
    for row in corridors:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("from") or "") != "entry" or str(row.get("to") or "") not in {
            "room_center",
            "center",
        }:
            continue
        primary_bbox = bbox_from_mapping(row)
        if primary_bbox is not None:
            break
    if primary_bbox is None:
        return
    region_index.setdefault("entry_to_center_corridor", primary_bbox)
    openings = (
        room_model.get("openings")
        if isinstance(room_model.get("openings"), Mapping)
        else {}
    )
    doors = openings.get("doors") if isinstance(openings, Mapping) else []
    if not isinstance(doors, Sequence) or isinstance(doors, str):
        return
    for index, door in enumerate(doors, start=1):
        door_id = (
            str(door.get("id") or "").strip()
            if isinstance(door, Mapping)
            else f"door_{index}"
        )
        if not door_id:
            door_id = f"door_{index}"
        region_index.setdefault(f"{door_id}_to_room_center_corridor", primary_bbox)


def _collect_protected_regions(
    room_model: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
    region_index: Mapping[str, tuple[int, int, int, int]],
) -> list[dict[str, Any]]:
    protected_rows: list[dict[str, Any]] = []
    concept = _concept_from_relation_plan(relation_plan)
    macro_constraints = (
        concept.get("macro_constraints") if isinstance(concept, Mapping) else {}
    )
    protected_topology = (
        macro_constraints.get("protected_topology")
        if isinstance(macro_constraints, Mapping)
        else []
    )
    if not isinstance(protected_topology, Sequence) or isinstance(
        protected_topology, str
    ):
        protected_topology = []
    for row in protected_topology:
        if not isinstance(row, Mapping):
            continue
        region_id = str(row.get("region") or "").strip()
        bbox = _region_bbox_from_ref(region_id, region_index, relation_plan, room_model)
        if bbox is None:
            continue
        applies_to = row.get("applies_to")
        if not isinstance(applies_to, Sequence) or isinstance(applies_to, str):
            applies_to = []
        protected_rows.append(
            {
                "region_id": region_id,
                "bbox": bbox,
                "max_overlap_ratio": float(row.get("max_overlap_ratio") or 0.0),
                "priority": str(row.get("priority") or "medium"),
                "enforcement": str(row.get("enforcement") or "soft"),
                "violation_severity": str(row.get("violation_severity") or "advisory"),
                "zone_type": str(row.get("zone_type") or ""),
                "applies_to": [str(item) for item in applies_to if str(item).strip()],
            }
        )
    return protected_rows


def _protected_region_policy_indexes(
    room_model: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
    region_index: Mapping[str, tuple[int, int, int, int]],
) -> tuple[
    dict[tuple[str, tuple[int, int, int, int]], dict[str, Any]],
    dict[tuple[int, int, int, int], dict[str, Any]],
]:
    policy_by_ref: dict[tuple[str, tuple[int, int, int, int]], dict[str, Any]] = {}
    policy_by_bbox: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for row in _collect_protected_regions(room_model, relation_plan, region_index):
        region_id = str(row.get("region_id") or "").strip()
        bbox = row.get("bbox")
        if not isinstance(bbox, tuple) or len(bbox) != 4:
            continue
        policy = _cluster_forbidden_region_policy_from_protected_row(row)
        if region_id:
            policy_by_ref[(region_id, bbox)] = policy
        policy_by_bbox.setdefault(bbox, policy)
    return policy_by_ref, policy_by_bbox


def _cluster_forbidden_region_policy_from_protected_row(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "max_overlap_ratio": float(row.get("max_overlap_ratio") or 0.0),
        "priority": str(row.get("priority") or "medium"),
        "enforcement": str(row.get("enforcement") or "soft"),
        "violation_severity": str(row.get("violation_severity") or "advisory"),
        "zone_type": str(row.get("zone_type") or "cluster_forbidden_region"),
    }


def _default_cluster_forbidden_region_policy() -> dict[str, Any]:
    return {
        "max_overlap_ratio": 0.0,
        "priority": "high",
        "enforcement": "hard",
        "violation_severity": "blocking",
        "zone_type": "cluster_forbidden_region",
    }


def _default_cluster_forbidden_region_policy_for_cluster(
    *,
    region_id: str,
    cluster_program: Mapping[str, Any],
) -> dict[str, Any]:
    if region_id.strip().lower().replace("-", "_") == "center_access_lane":
        return {
            "max_overlap_ratio": 0.35,
            "priority": "medium",
            "enforcement": "hard_soft",
            "violation_severity": "advisory",
            "zone_type": "conceptual_center_access_lane",
        }
    if (
        _region_id_is_daylight_clearance(region_id)
        and _cluster_max_height_mm(cluster_program)
        <= OBJECT_LEVEL_LOW_HEIGHT_DAYLIGHT_SOFT_MAX_MM
    ):
        return {
            "max_overlap_ratio": OBJECT_LEVEL_LOW_HEIGHT_DAYLIGHT_MAX_OVERLAP_RATIO,
            "priority": "medium",
            "enforcement": "soft",
            "violation_severity": "advisory",
            "zone_type": "daylight_clearance_soft",
        }
    return _default_cluster_forbidden_region_policy()


def _region_id_is_daylight_clearance(region_id: str) -> bool:
    token = region_id.strip().lower()
    return "window" in token or "daylight" in token


def _cluster_max_height_mm(cluster_program: Mapping[str, Any]) -> int:
    object_program = (
        cluster_program.get("object_program")
        if isinstance(cluster_program.get("object_program"), Mapping)
        else {}
    )
    specs = (
        object_program.get("object_specs_by_id")
        if isinstance(object_program.get("object_specs_by_id"), Mapping)
        else {}
    )
    heights: list[int] = []
    for spec in specs.values():
        if not isinstance(spec, Mapping):
            continue
        dims = (
            spec.get("rep_dims_mm")
            if isinstance(spec.get("rep_dims_mm"), Mapping)
            else {}
        )
        try:
            heights.append(int(round(float(dims.get("H") or 0))))
        except (TypeError, ValueError):
            continue
    return max(heights, default=0)


def _auto_window_blocking_region_ids_for_cluster(
    *,
    cluster_program: Mapping[str, Any],
    region_index: Mapping[str, tuple[int, int, int, int]],
) -> list[str]:
    if (
        _cluster_max_height_mm(cluster_program)
        <= OBJECT_LEVEL_LOW_HEIGHT_DAYLIGHT_SOFT_MAX_MM
    ):
        return []
    if _cluster_is_window_treatment_only(cluster_program):
        return []

    clearance_ids: list[str] = []
    daylight_ids: list[str] = []
    for raw_region_id in region_index:
        region_id = str(raw_region_id or "").strip()
        if not region_id:
            continue
        token = region_id.lower()
        if "window" in token and "clearance" in token:
            clearance_ids.append(region_id)
        elif _region_id_is_daylight_clearance(region_id):
            daylight_ids.append(region_id)
    return sorted({*clearance_ids, *daylight_ids})


def _cluster_is_window_treatment_only(cluster_program: Mapping[str, Any]) -> bool:
    object_program = (
        cluster_program.get("object_program")
        if isinstance(cluster_program.get("object_program"), Mapping)
        else {}
    )
    specs = (
        object_program.get("object_specs_by_id")
        if isinstance(object_program.get("object_specs_by_id"), Mapping)
        else {}
    )
    object_tokens: list[list[str]] = []
    for object_id, spec in specs.items():
        if not isinstance(spec, Mapping):
            continue
        object_tokens.append(
            [
                str(object_id or ""),
                str(spec.get("object_id") or ""),
                str(spec.get("object_type") or ""),
                str(spec.get("category") or ""),
            ]
        )
    if not object_tokens:
        members = object_program.get("members")
        if isinstance(members, Sequence) and not isinstance(members, str):
            object_tokens = [[str(member or "")] for member in members]
    return bool(object_tokens) and all(
        any(_is_window_treatment_token(token) for token in tokens)
        for tokens in object_tokens
    )


def _is_window_treatment_token(value: str) -> bool:
    token = value.strip().lower()
    if not token:
        return False
    normalized = token.replace("-", "_").replace(" ", "_")
    return any(part in normalized for part in _WINDOW_TREATMENT_OBJECT_TOKENS)


def _cluster_forbidden_region_policy(
    *,
    region_id: str,
    bbox: tuple[int, int, int, int],
    cluster_program: Mapping[str, Any],
    protected_policy_by_ref: Mapping[
        tuple[str, tuple[int, int, int, int]], Mapping[str, Any]
    ],
    protected_policy_by_bbox: Mapping[tuple[int, int, int, int], Mapping[str, Any]],
) -> dict[str, Any]:
    protected_policy = protected_policy_by_ref.get(
        (region_id, bbox)
    ) or protected_policy_by_bbox.get(bbox)
    if protected_policy is None:
        return _default_cluster_forbidden_region_policy_for_cluster(
            region_id=region_id,
            cluster_program=cluster_program,
        )
    return {
        "max_overlap_ratio": float(protected_policy.get("max_overlap_ratio") or 0.0),
        "priority": str(protected_policy.get("priority") or "medium"),
        "enforcement": str(protected_policy.get("enforcement") or "soft"),
        "violation_severity": str(
            protected_policy.get("violation_severity") or "advisory"
        ),
        "zone_type": str(
            protected_policy.get("zone_type") or "cluster_forbidden_region"
        ),
    }


def _collect_cluster_forbidden_regions(
    clusters_by_id: Mapping[str, Mapping[str, Any]],
    room_model: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
    region_index: Mapping[str, tuple[int, int, int, int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    hint_rows = _cluster_forbidden_hint_rows(relation_plan)
    protected_policy_by_ref, protected_policy_by_bbox = (
        _protected_region_policy_indexes(room_model, relation_plan, region_index)
    )
    seen: set[tuple[str, str, tuple[int, int, int, int]]] = set()
    for cluster_id, cluster_program in clusters_by_id.items():
        object_program = (
            cluster_program.get("object_program")
            if isinstance(cluster_program.get("object_program"), Mapping)
            else {}
        )
        cluster_rules = (
            cluster_program.get("cluster_rules")
            if isinstance(cluster_program.get("cluster_rules"), Mapping)
            else {}
        )
        object_zone_claims = (
            object_program.get("zone_claims")
            if isinstance(object_program.get("zone_claims"), Mapping)
            else {}
        )
        rule_zone_claims = (
            cluster_rules.get("zone_claims")
            if isinstance(cluster_rules.get("zone_claims"), Mapping)
            else {}
        )
        hint = hint_rows.get(cluster_id, {})
        region_ids = _dedupe_string_sequence(
            [
                *_string_sequence(object_zone_claims.get("avoid_regions")),
                *_string_sequence(rule_zone_claims.get("avoid_regions")),
                *_string_sequence(
                    hint.get("forbidden_region_ids")
                    if isinstance(hint, Mapping)
                    else []
                ),
                *_auto_window_blocking_region_ids_for_cluster(
                    cluster_program=cluster_program,
                    region_index=region_index,
                ),
            ]
        )
        for region_id in region_ids:
            for bbox in _region_bboxes_from_ref(
                region_id, region_index, relation_plan, room_model
            ):
                key = (cluster_id, region_id, bbox)
                if key in seen:
                    continue
                seen.add(key)
                policy = _cluster_forbidden_region_policy(
                    region_id=region_id,
                    bbox=bbox,
                    cluster_program=cluster_program,
                    protected_policy_by_ref=protected_policy_by_ref,
                    protected_policy_by_bbox=protected_policy_by_bbox,
                )
                rows.append(
                    {
                        "cluster_id": cluster_id,
                        "region_id": region_id,
                        "bbox": bbox,
                        **policy,
                    }
                )
    return rows


def _cluster_forbidden_hint_rows(
    relation_plan: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if not isinstance(relation_plan, Mapping):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for section_name in (
        "anchor_layout_hints_by_cluster",
        "anchor_region_preferences_by_cluster",
    ):
        section = relation_plan.get(section_name)
        if not isinstance(section, Mapping):
            continue
        for cluster_id, row in section.items():
            if not isinstance(row, Mapping):
                continue
            key = str(cluster_id or "").strip()
            if not key:
                continue
            target = out.setdefault(key, {})
            target["forbidden_region_ids"] = _dedupe_string_sequence(
                [
                    *_string_sequence(target.get("forbidden_region_ids")),
                    *_string_sequence(row.get("forbidden_region_ids")),
                ]
            )
    concept = _concept_from_relation_plan(relation_plan)
    macro_constraints = (
        concept.get("macro_constraints") if isinstance(concept, Mapping) else {}
    )
    anchor_constraints = (
        macro_constraints.get("anchor_region_constraints")
        if isinstance(macro_constraints, Mapping)
        else []
    )
    if isinstance(anchor_constraints, Sequence) and not isinstance(
        anchor_constraints, str
    ):
        for row in anchor_constraints:
            if not isinstance(row, Mapping):
                continue
            cluster_id = str(row.get("cluster_id") or "").strip()
            if not cluster_id:
                continue
            target = out.setdefault(cluster_id, {})
            target["forbidden_region_ids"] = _dedupe_string_sequence(
                [
                    *_string_sequence(target.get("forbidden_region_ids")),
                    *_string_sequence(row.get("forbidden_region_ids")),
                ]
            )
    return out


def _object_solver_room_bbox(
    room_model: Mapping[str, Any],
) -> tuple[int, int, int, int]:
    room = room_model.get("room") if isinstance(room_model.get("room"), Mapping) else {}
    polygon = room.get("polygon_ccw") if isinstance(room, Mapping) else None
    points = []
    if isinstance(polygon, Sequence) and not isinstance(polygon, str):
        for row in polygon:
            if not isinstance(row, Mapping):
                continue
            try:
                points.append((float(row.get("x")), float(row.get("y"))))
            except (TypeError, ValueError):
                continue
    if len(points) >= 2:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return (
            int(round(min(xs))),
            int(round(min(ys))),
            int(round(max(xs))),
            int(round(max(ys))),
        )
    bbox = room.get("bbox") if isinstance(room, Mapping) else None
    if isinstance(bbox, Mapping):
        parsed = _bbox_tuple(
            bbox.get("min_x"), bbox.get("min_y"), bbox.get("max_x"), bbox.get("max_y")
        )
        if parsed is not None:
            return parsed
    return (0, 0, 6000, 4000)


def _object_solver_room_polygon(
    room_model: Mapping[str, Any],
) -> tuple[tuple[float, float], ...]:
    room = room_model.get("room") if isinstance(room_model.get("room"), Mapping) else {}
    polygon = room.get("polygon_ccw") if isinstance(room, Mapping) else None
    points: list[tuple[float, float]] = []
    if isinstance(polygon, Sequence) and not isinstance(polygon, str):
        for row in polygon:
            if not isinstance(row, Mapping):
                continue
            try:
                points.append((float(row.get("x")), float(row.get("y"))))
            except (TypeError, ValueError):
                continue
    if len(points) < 3:
        return ()
    if points[0] == points[-1]:
        points = points[:-1]
    return tuple(points)


def _room_id(room_model: Mapping[str, Any]) -> str:
    room = room_model.get("room") if isinstance(room_model.get("room"), Mapping) else {}
    value = room.get("room_id") if isinstance(room, Mapping) else None
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "room_1"


def _room_type(room_model: Mapping[str, Any]) -> str:
    meta = room_model.get("meta") if isinstance(room_model.get("meta"), Mapping) else {}
    value = meta.get("room_type") if isinstance(meta, Mapping) else None
    if isinstance(value, str) and value.strip():
        return value.strip()
    value = room_model.get("room_type")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "room"


def _bbox_tuple(
    min_x: Any, min_y: Any, max_x: Any, max_y: Any
) -> tuple[int, int, int, int] | None:
    try:
        x1 = int(round(float(min_x)))
        y1 = int(round(float(min_y)))
        x2 = int(round(float(max_x)))
        y2 = int(round(float(max_y)))
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _region_bbox_from_ref(
    region_ref: str,
    region_index: Mapping[str, tuple[int, int, int, int]],
    relation_plan: Mapping[str, Any] | None,
    room_model: Mapping[str, Any],
) -> tuple[int, int, int, int] | None:
    if not region_ref:
        return None
    if region_ref in region_index:
        return region_index[region_ref]
    macro_region_map = (
        relation_plan.get("macro_region_map")
        if isinstance(relation_plan, Mapping)
        else {}
    )
    if isinstance(macro_region_map, Mapping):
        for row in macro_region_map.get("regions") or []:
            if not isinstance(row, Mapping):
                continue
            if str(row.get("region_id") or "") != region_ref:
                continue
            bboxes = [
                _region_index_bbox(str(source_id), region_index)
                or _region_bbox_from_ref(
                    str(source_id),
                    region_index,
                    _without_macro_region_map(relation_plan),
                    room_model,
                )
                for source_id in (row.get("source_ids") or [])
            ]
            if bboxes:
                return _union_bboxes([bbox for bbox in bboxes if bbox is not None])
    # Fallback heuristics from name
    room_bbox = _object_solver_room_bbox(room_model)
    min_x, min_y, max_x, max_y = room_bbox
    width = max_x - min_x
    height = max_y - min_y
    token = str(region_ref).lower()
    if "bottom" in token and "wall" in token:
        return (min_x, max_y - max(700, int(round(height * 0.18))), max_x, max_y)
    if "left" in token and "wall" in token:
        return (min_x, min_y, min_x + max(700, int(round(width * 0.18))), max_y)
    if "right" in token and "wall" in token:
        return (max_x - max(700, int(round(width * 0.18))), min_y, max_x, max_y)
    if "focal" in token or ("wall" in token and "top" in token):
        return (min_x, min_y, max_x, min_y + max(700, int(round(height * 0.18))))
    if "daylight" in token or "window" in token:
        return (min_x, min_y, max_x, min_y + max(900, int(round(height * 0.24))))
    if "entry" in token or "door" in token:
        return (
            min_x,
            max_y - max(850, int(round(height * 0.22))),
            min_x + max(1000, int(round(width * 0.22))),
            max_y,
        )
    if "center" in token or "floating" in token:
        return (
            min_x + int(round(width * 0.25)),
            min_y + int(round(height * 0.25)),
            max_x - int(round(width * 0.25)),
            max_y - int(round(height * 0.25)),
        )
    if "privacy" in token or "deep" in token:
        return (
            min_x + int(round(width * 0.45)),
            min_y + int(round(height * 0.45)),
            max_x,
            max_y,
        )
    if "edge" in token or "storage" in token:
        return (min_x, min_y, max_x, max_y)
    return None


def _region_index_bbox(
    region_ref: str,
    region_index: Mapping[str, tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    if region_ref in region_index:
        return region_index[region_ref]
    aliases = []
    if "_focal" in region_ref:
        aliases.append(region_ref.replace("_focal", "_anchor"))
    if "_anchor" in region_ref:
        aliases.append(region_ref.replace("_anchor", "_focal"))
    for alias in aliases:
        if alias in region_index:
            return region_index[alias]
    return None


def _without_macro_region_map(
    relation_plan: Mapping[str, Any] | None,
) -> dict[str, Any]:
    fallback_plan = dict(relation_plan) if isinstance(relation_plan, Mapping) else {}
    fallback_plan.pop("macro_region_map", None)
    return fallback_plan


def _dedupe_region_bboxes(
    bboxes: Sequence[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    out: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for bbox in bboxes:
        if bbox in seen:
            continue
        seen.add(bbox)
        out.append(bbox)
    return out


def _region_bboxes_from_ref(
    region_ref: str,
    region_index: Mapping[str, tuple[int, int, int, int]],
    relation_plan: Mapping[str, Any] | None,
    room_model: Mapping[str, Any],
) -> list[tuple[int, int, int, int]]:
    if not region_ref:
        return []
    direct_bbox = _region_index_bbox(region_ref, region_index)
    if direct_bbox is not None:
        return [direct_bbox]

    macro_region_map = (
        relation_plan.get("macro_region_map")
        if isinstance(relation_plan, Mapping)
        else {}
    )
    if isinstance(macro_region_map, Mapping):
        for row in macro_region_map.get("regions") or []:
            if not isinstance(row, Mapping):
                continue
            if str(row.get("region_id") or "") != region_ref:
                continue
            source_bboxes = [
                bbox
                for source_id in (row.get("source_ids") or [])
                if (
                    bbox := (
                        _region_index_bbox(str(source_id), region_index)
                        or _region_bbox_from_ref(
                            str(source_id),
                            region_index,
                            _without_macro_region_map(relation_plan),
                            room_model,
                        )
                    )
                )
                is not None
            ]
            if source_bboxes:
                return _dedupe_region_bboxes(source_bboxes)

    fallback_bbox = _region_bbox_from_ref(
        region_ref, region_index, _without_macro_region_map(relation_plan), room_model
    )
    return [fallback_bbox] if fallback_bbox is not None else []


def _union_bboxes(
    bboxes: Sequence[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    rows = [bbox for bbox in bboxes if bbox is not None]
    if not rows:
        return None
    return (
        min(b[0] for b in rows),
        min(b[1] for b in rows),
        max(b[2] for b in rows),
        max(b[3] for b in rows),
    )


def _cluster_program_dominant_anchor(cluster_program: Mapping[str, Any]) -> str | None:
    object_program = (
        cluster_program.get("object_program")
        if isinstance(cluster_program.get("object_program"), Mapping)
        else {}
    )
    for value in (
        object_program.get("dominant_anchor_id"),
        next(iter(object_program.get("anchors") or []), None),
        next(iter(object_program.get("dominant_anchor_candidates") or []), None),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _cluster_is_solver_trial_optional(cluster_program: Mapping[str, Any]) -> bool:
    anchor_id = _cluster_program_dominant_anchor(cluster_program)
    if anchor_id is None:
        return False
    spec = _object_spec(cluster_program, anchor_id)
    if not isinstance(spec, Mapping):
        return False
    return any(
        bool(spec.get(key))
        for key in ("trial_optional", "solver_trial", "budget_trial")
    )


def _object_spec(
    cluster_program: Mapping[str, Any], object_id: str
) -> Mapping[str, Any] | None:
    object_program = (
        cluster_program.get("object_program")
        if isinstance(cluster_program.get("object_program"), Mapping)
        else {}
    )
    specs = (
        object_program.get("object_specs_by_id")
        if isinstance(object_program.get("object_specs_by_id"), Mapping)
        else {}
    )
    row = specs.get(object_id)
    return row if isinstance(row, Mapping) else None


def _rotate_dims_for_rot(length_mm: int, width_mm: int, rot: int) -> tuple[int, int]:
    return (length_mm, width_mm) if int(rot) % 180 == 0 else (width_mm, length_mm)


def _generate_anchor_pose_candidates(
    *,
    cluster_program: Mapping[str, Any],
    room_model: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
    world: Mapping[str, Any],
    grid_mm: int,
) -> list[dict[str, Any]]:
    cluster_id = str(cluster_program.get("cluster_id") or "")
    anchor_id = _cluster_program_dominant_anchor(cluster_program)
    if not anchor_id:
        return []
    spec = _object_spec(cluster_program, anchor_id)
    if spec is None:
        return []
    dims = (
        spec.get("rep_dims_mm") if isinstance(spec.get("rep_dims_mm"), Mapping) else {}
    )
    length_mm = max(200, int(dims.get("L") or 0))
    width_mm = max(200, int(dims.get("W") or 0))
    allowed_rotations = (
        spec.get("allowed_rotations")
        if isinstance(spec.get("allowed_rotations"), Sequence)
        else [0, 90, 180, 270]
    )
    relation_plan = relation_plan or {}
    anchor_hints = (
        relation_plan.get("anchor_layout_hints_by_cluster")
        if isinstance(relation_plan.get("anchor_layout_hints_by_cluster"), Mapping)
        else {}
    )
    anchor_region_preferences = (
        relation_plan.get("anchor_region_preferences_by_cluster")
        if isinstance(
            relation_plan.get("anchor_region_preferences_by_cluster"), Mapping
        )
        else {}
    )
    hint = (
        anchor_hints.get(cluster_id)
        if isinstance(anchor_hints, Mapping)
        and isinstance(anchor_hints.get(cluster_id), Mapping)
        else {}
    )
    region_preference = (
        anchor_region_preferences.get(cluster_id)
        if isinstance(anchor_region_preferences, Mapping)
        and isinstance(anchor_region_preferences.get(cluster_id), Mapping)
        else {}
    )
    if not hint and region_preference:
        hint = region_preference
    placement_behavior = _placement_behavior_from_rows(hint, region_preference)
    zone_assignment = str(hint.get("zone_assignment") or "")
    required_region_ids = _string_sequence(hint.get("required_region_ids"))
    forbidden_region_ids = _dedupe_string_sequence(
        [
            *_string_sequence(hint.get("forbidden_region_ids")),
            *_auto_window_blocking_region_ids_for_cluster(
                cluster_program=cluster_program,
                region_index=world["region_index"],
            ),
        ]
    )
    preferred_region_ids = _string_sequence(hint.get("preferred_region_ids"))
    preferred_region_ids = _dedupe_string_sequence(
        [
            *required_region_ids,
            zone_assignment,
            *preferred_region_ids,
            *[
                item
                for item in _string_sequence(
                    region_preference.get("preferred_region_ids")
                )
                if item != zone_assignment
            ],
        ]
    )
    preferred_wall_side = str(hint.get("preferred_wall_side") or "").strip()
    anchor_strength = str(hint.get("anchor_strength") or "medium").strip().lower()
    region_bboxes = [
        bbox
        for region_id in preferred_region_ids
        if region_id
        for bbox in _region_bboxes_from_ref(
            str(region_id), world["region_index"], relation_plan, room_model
        )
    ]
    region_bboxes = _dedupe_region_bboxes(region_bboxes)
    if not region_bboxes:
        region_bboxes = [world["room_bbox"]]
    forbidden_region_bboxes = [
        bbox
        for region_id in forbidden_region_ids
        for bbox in _region_bboxes_from_ref(
            str(region_id), world["region_index"], relation_plan, room_model
        )
    ]
    forbidden_region_bboxes = _dedupe_region_bboxes(forbidden_region_bboxes)
    forbidden_regions = _anchor_forbidden_regions_for_cluster(
        cluster_id=cluster_id,
        world=world,
        resolved_region_bboxes=forbidden_region_bboxes,
    )
    primary_cluster = _extract_layout_primary_cluster_id(
        dict(relation_plan) if isinstance(relation_plan, dict) else None
    )
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()

    def add_candidates_for_regions(
        candidate_region_bboxes: Sequence[tuple[int, int, int, int]],
        *,
        search_stage: str,
        score_adjustment: float,
    ) -> None:
        for region_bbox in candidate_region_bboxes[:12]:
            min_x, min_y, max_x, max_y = region_bbox
            region_width = max_x - min_x
            region_height = max_y - min_y
            for rot in allowed_rotations:
                w_mm, h_mm = _rotate_dims_for_rot(length_mm, width_mm, int(rot))
                if (
                    w_mm >= region_width
                    and h_mm >= region_height
                    and region_bbox != world["room_bbox"]
                ):
                    continue
                anchor_kind = _anchor_kind_for_candidate(
                    zone_assignment=zone_assignment,
                    room_bbox=world["room_bbox"],
                    region_bbox=region_bbox,
                )
                origin_positions = _wall_flush_origin_candidates(
                    region_bbox=region_bbox,
                    room_bbox=world["room_bbox"],
                    zone_assignment=zone_assignment,
                    placement_behavior=placement_behavior,
                    w_mm=w_mm,
                    h_mm=h_mm,
                    grid_mm=grid_mm,
                )
                centers = _candidate_centers_for_region(
                    region_bbox=region_bbox,
                    room_bbox=world["room_bbox"],
                    zone_assignment=zone_assignment,
                    role_kind=str(
                        hint.get("role_kind") or cluster_program.get("tag") or ""
                    ),
                    primary=isinstance(primary_cluster, str)
                    and cluster_id == primary_cluster,
                )
                origin_positions.extend(
                    _snap_origin_inside_room(
                        x=int(round(cx - w_mm / 2.0)),
                        y=int(round(cy - h_mm / 2.0)),
                        w_mm=w_mm,
                        h_mm=h_mm,
                        room_bbox=world["room_bbox"],
                        grid_mm=grid_mm,
                    )
                    for cx, cy in centers
                )
                for x, y in origin_positions:
                    rect = (x, y, x + w_mm, y + h_mm)
                    if not _rect_inside_room(rect, world["room_bbox"]):
                        continue
                    if _anchor_forbidden_overlap_exceeds_limit(
                        rect=rect,
                        forbidden_regions=forbidden_regions,
                    ):
                        continue
                    key = (x, y, int(rot) % 360)
                    if key in seen:
                        continue
                    seen.add(key)
                    front_token = (
                        spec.get("front")
                        if isinstance(spec.get("front"), str)
                        else "top"
                    )
                    front_world = _front_vector_from_rotation(front_token, int(rot))
                    desired_front = _desired_anchor_front_vector(
                        cluster_id=cluster_id,
                        object_id=anchor_id,
                        rect=rect,
                        room_bbox=world["room_bbox"],
                        relation_plan=relation_plan,
                        anchor_kind=anchor_kind,
                        placement_behavior=placement_behavior,
                    )
                    orientation_score = _front_alignment_score(
                        front_world=front_world,
                        desired_front=desired_front,
                    )
                    candidates.append(
                        {
                            "cluster_id": cluster_id,
                            "object_id": anchor_id,
                            "x": x,
                            "y": y,
                            "rot": int(rot) % 360,
                            "w": w_mm,
                            "h": h_mm,
                            "rect": rect,
                            "anchor_kind": anchor_kind,
                            "front_world": {
                                "dx": front_world[0],
                                "dy": front_world[1],
                            },
                            "desired_front_world": None
                            if desired_front is None
                            else {"dx": desired_front[0], "dy": desired_front[1]},
                            "orientation_score": orientation_score,
                            "search_stage": search_stage,
                            "candidate_score": _anchor_candidate_score(
                                rect=rect,
                                room_bbox=world["room_bbox"],
                                region_bbox=region_bbox,
                                zone_assignment=zone_assignment,
                                anchor_kind=anchor_kind,
                                preferred_wall_side=preferred_wall_side,
                                anchor_strength=anchor_strength,
                                placement_behavior=placement_behavior,
                                forbidden_regions=forbidden_regions,
                                cluster_id=cluster_id,
                                relation_plan=relation_plan,
                            )
                            + orientation_score
                            + score_adjustment,
                        }
                    )

    add_candidates_for_regions(
        region_bboxes,
        search_stage="planned_regions",
        score_adjustment=0.0,
    )
    if not candidates:
        add_candidates_for_regions(
            _anchor_fallback_region_bboxes(
                world=world,
                existing_region_bboxes=region_bboxes,
            ),
            search_stage="expanded_wall_fallback",
            score_adjustment=-900.0,
        )
    candidates.sort(
        key=lambda item: (
            -float(item.get("candidate_score") or 0.0),
            item["y"],
            item["x"],
            item["rot"],
        )
    )
    return candidates[:OBJECT_LEVEL_MAX_ANCHOR_CANDIDATES_PER_CLUSTER]


def _anchor_forbidden_regions_for_cluster(
    *,
    cluster_id: str,
    world: Mapping[str, Any],
    resolved_region_bboxes: Sequence[tuple[int, int, int, int]],
) -> list[dict[str, Any]]:
    if not resolved_region_bboxes:
        return []
    resolved_bbox_set = set(resolved_region_bboxes)
    raw_rows = (
        world.get("cluster_forbidden_regions")
        if isinstance(world.get("cluster_forbidden_regions"), Sequence)
        else []
    )
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[int, int, int, int]]] = set()
    for raw_row in raw_rows:
        if not isinstance(raw_row, Mapping):
            continue
        if str(raw_row.get("cluster_id") or "") != cluster_id:
            continue
        bbox = _rect_tuple(raw_row.get("bbox"))
        if bbox is None or bbox not in resolved_bbox_set:
            continue
        region_id = str(raw_row.get("region_id") or "")
        key = (region_id, bbox)
        if key in seen:
            continue
        seen.add(key)
        row = dict(raw_row)
        row["bbox"] = bbox
        rows.append(row)
    if rows:
        return rows
    return [
        {
            "cluster_id": cluster_id,
            "region_id": "",
            "bbox": bbox,
            **_default_cluster_forbidden_region_policy(),
        }
        for bbox in resolved_region_bboxes
    ]


def _anchor_fallback_region_bboxes(
    *,
    world: Mapping[str, Any],
    existing_region_bboxes: Sequence[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    region_index = (
        world.get("region_index")
        if isinstance(world.get("region_index"), Mapping)
        else {}
    )
    existing = set(existing_region_bboxes)
    scored: list[tuple[int, str, tuple[int, int, int, int]]] = []
    for raw_region_id, bbox in region_index.items():
        region_id = str(raw_region_id or "")
        rect = _rect_tuple(bbox)
        if rect is None or rect in existing:
            continue
        if not _region_id_is_anchor_fallback_region(region_id):
            continue
        scored.append((_anchor_fallback_region_priority(region_id), region_id, rect))
    scored.sort(key=lambda item: (item[0], item[1]))
    fallback_bboxes = _dedupe_region_bboxes([item[2] for item in scored])
    if fallback_bboxes:
        return fallback_bboxes
    room_bbox = _rect_tuple(world.get("room_bbox"))
    return [] if room_bbox is None else [room_bbox]


def _region_id_is_anchor_fallback_region(region_id: str) -> bool:
    token = region_id.strip().lower()
    if not token:
        return False
    if any(part in token for part in ("door", "entry", "corridor", "clearance")):
        return False
    if any(part in token for part in ("center", "floating")):
        return False
    return any(
        part in token
        for part in (
            "_usable_",
            "anchor",
            "edge",
            "perimeter",
            "privacy",
            "private",
            "wall",
            "window_side",
        )
    )


def _anchor_fallback_region_priority(region_id: str) -> int:
    token = region_id.strip().lower()
    if "_usable_" in token and "anchor" in token:
        return 0
    if token in {
        "top_wall_zone",
        "right_wall_zone",
        "bottom_wall_zone",
        "left_wall_zone",
    }:
        return 1
    if "edge" in token or "perimeter" in token:
        return 2
    if "privacy" in token or "private" in token:
        return 3
    if "window_side" in token:
        return 4
    return 5


def _wall_flush_origin_candidates(
    *,
    region_bbox: tuple[int, int, int, int],
    room_bbox: tuple[int, int, int, int],
    zone_assignment: str,
    placement_behavior: Mapping[str, Any] | None,
    w_mm: int,
    h_mm: int,
    grid_mm: int,
) -> list[tuple[int, int]]:
    token = str(zone_assignment).lower()
    wall_backing = _placement_behavior_value(placement_behavior, "wall_backing")
    wall_backed_tokens = (
        "daylight",
        "edge",
        "focal",
        "private",
        "privacy",
        "quiet",
        "storage",
        "wall",
        "window",
    )
    if wall_backing not in {"preferred", "required"} and not any(
        part in token for part in wall_backed_tokens
    ):
        return []

    wall = _nearest_wall_name_for_region(region_bbox=region_bbox, room_bbox=room_bbox)
    min_x, min_y, max_x, max_y = region_bbox
    room_min_x, room_min_y, room_max_x, room_max_y = room_bbox
    room_cx = int(round((room_min_x + room_max_x) / 2.0))
    room_cy = int(round((room_min_y + room_max_y) / 2.0))
    origins: list[tuple[int, int]] = []

    def append_origin(x: int, y: int) -> None:
        origins.append(
            _snap_origin_inside_room(
                x=x,
                y=y,
                w_mm=w_mm,
                h_mm=h_mm,
                room_bbox=room_bbox,
                grid_mm=grid_mm,
            )
        )

    if wall in {"top_wall", "bottom_wall"}:
        y = room_min_y if wall == "top_wall" else room_max_y - h_mm
        span_min = max(room_min_x, min_x)
        span_max = min(room_max_x - w_mm, max_x - w_mm)
        if span_max < span_min:
            span_min = room_min_x
            span_max = room_max_x - w_mm
        for x in (span_min, (span_min + span_max) // 2, span_max, room_cx - w_mm // 2):
            append_origin(int(x), int(y))
    else:
        x = room_min_x if wall == "left_wall" else room_max_x - w_mm
        span_min = max(room_min_y, min_y)
        span_max = min(room_max_y - h_mm, max_y - h_mm)
        if span_max < span_min:
            span_min = room_min_y
            span_max = room_max_y - h_mm
        for y in (span_min, (span_min + span_max) // 2, span_max, room_cy - h_mm // 2):
            append_origin(int(x), int(y))

    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for origin in origins:
        if origin in seen:
            continue
        seen.add(origin)
        out.append(origin)
    return out


def _candidate_centers_for_region(
    *,
    region_bbox: tuple[int, int, int, int],
    room_bbox: tuple[int, int, int, int],
    zone_assignment: str,
    role_kind: str,
    primary: bool,
) -> list[tuple[int, int]]:
    min_x, min_y, max_x, max_y = region_bbox
    cx = int(round((min_x + max_x) / 2.0))
    cy = int(round((min_y + max_y) / 2.0))
    room_cx = int(round((room_bbox[0] + room_bbox[2]) / 2.0))
    room_cy = int(round((room_bbox[1] + room_bbox[3]) / 2.0))
    centers = [(cx, cy)]
    token = str(zone_assignment).lower()
    if "focal" in token or "wall" in token or "edge" in token or "storage" in token:
        centers.extend(
            [
                (cx, cy),
                (room_cx, cy),
                (cx, room_cy),
                (int(round(min_x + (max_x - min_x) * 0.25)), cy),
                (int(round(min_x + (max_x - min_x) * 0.75)), cy),
            ]
        )
    elif "daylight" in token or "window" in token:
        centers.extend(
            [
                (cx, cy),
                (room_cx, cy),
                (cx, min(cy + 250, room_bbox[3] - 250)),
                (int(round(min_x + (max_x - min_x) * 0.25)), cy),
                (int(round(min_x + (max_x - min_x) * 0.75)), cy),
            ]
        )
    elif "center" in token or "floating" in token:
        centers.extend(
            [
                (room_cx, room_cy),
                (cx, cy),
                (room_cx, cy),
                (int(round(min_x + (max_x - min_x) * 0.33)), room_cy),
                (int(round(min_x + (max_x - min_x) * 0.67)), room_cy),
            ]
        )
    if primary:
        centers.insert(0, (cx, cy))
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for item in centers:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out[:8]


def _anchor_kind_for_candidate(
    *,
    zone_assignment: str,
    room_bbox: tuple[int, int, int, int],
    region_bbox: tuple[int, int, int, int],
) -> str:
    token = str(zone_assignment).lower()
    if "center" in token or "floating" in token:
        return "center"
    if "daylight" in token or "window" in token:
        return "window_side"
    if "entry" in token or "door" in token:
        return "entry_side"
    # infer closest wall from region
    return _nearest_wall_name_for_region(region_bbox=region_bbox, room_bbox=room_bbox)


def _nearest_wall_name_for_region(
    *,
    region_bbox: tuple[int, int, int, int],
    room_bbox: tuple[int, int, int, int],
) -> str:
    region_width = max(0, region_bbox[2] - region_bbox[0])
    region_height = max(0, region_bbox[3] - region_bbox[1])
    distances = {
        "top_wall": abs(region_bbox[1] - room_bbox[1]),
        "bottom_wall": abs(room_bbox[3] - region_bbox[3]),
        "left_wall": abs(region_bbox[0] - room_bbox[0]),
        "right_wall": abs(room_bbox[2] - region_bbox[2]),
    }
    horizontal_wall = (
        "top_wall"
        if distances["top_wall"] <= distances["bottom_wall"]
        else "bottom_wall"
    )
    vertical_wall = (
        "left_wall"
        if distances["left_wall"] <= distances["right_wall"]
        else "right_wall"
    )
    if region_width <= max(700, int(round(region_height * 0.45))) and distances[
        vertical_wall
    ] <= max(900, distances[horizontal_wall] + 900):
        return vertical_wall
    if region_height <= max(700, int(round(region_width * 0.45))) and distances[
        horizontal_wall
    ] <= max(900, distances[vertical_wall] + 900):
        return horizontal_wall
    return min(distances.items(), key=lambda item: item[1])[0]


def _anchor_candidate_score(
    *,
    rect: tuple[int, int, int, int],
    room_bbox: tuple[int, int, int, int],
    region_bbox: tuple[int, int, int, int],
    zone_assignment: str,
    anchor_kind: str,
    preferred_wall_side: str,
    anchor_strength: str,
    placement_behavior: Mapping[str, Any] | None,
    forbidden_regions: Sequence[Mapping[str, Any]],
    cluster_id: str,
    relation_plan: Mapping[str, Any] | None,
) -> float:
    cx = (rect[0] + rect[2]) / 2.0
    cy = (rect[1] + rect[3]) / 2.0
    rcx = (region_bbox[0] + region_bbox[2]) / 2.0
    rcy = (region_bbox[1] + region_bbox[3]) / 2.0
    room_cx = (room_bbox[0] + room_bbox[2]) / 2.0
    room_cy = (room_bbox[1] + room_bbox[3]) / 2.0
    dist_region = math.hypot(cx - rcx, cy - rcy)
    dist_center = math.hypot(cx - room_cx, cy - room_cy)
    token = str(zone_assignment).lower()
    score = 10000.0 - dist_region
    if "center" in token or "floating" in token:
        score += 1200.0 - dist_center
    elif any(part in token for part in ("focal", "wall", "edge", "storage")):
        score += dist_center * 0.25
    wall_backing = _placement_behavior_value(placement_behavior, "wall_backing")
    front_space = _placement_behavior_value(placement_behavior, "front_space")
    pose_flexibility = _placement_behavior_value(
        placement_behavior,
        "pose_flexibility",
    )
    if wall_backing in {"preferred", "required"}:
        if anchor_kind in {
            "bottom_wall",
            "left_wall",
            "right_wall",
            "top_wall",
            "window_side",
        }:
            score += 700.0 if wall_backing == "required" else 350.0
        else:
            score -= 700.0 if wall_backing == "required" else 250.0
    if front_space == "required":
        score += 180.0
    elif front_space == "preferred":
        score += 80.0
    if pose_flexibility == "low":
        score += 120.0
    if cluster_id == _extract_layout_primary_cluster_id(
        dict(relation_plan) if isinstance(relation_plan, dict) else None
    ):
        score += 300.0
    preferred = preferred_wall_side.strip().lower()
    if preferred:
        strength = anchor_strength.strip().lower()
        match_bonus = {"hard": 3600.0, "strong": 2600.0}.get(strength, 1200.0)
        mismatch_penalty = {"hard": 5200.0, "strong": 3200.0}.get(strength, 1200.0)
        if _anchor_kind_matches_preferred_side(anchor_kind, preferred):
            score += match_bonus
        else:
            score -= mismatch_penalty
    if forbidden_regions:
        excess_overlap = _forbidden_region_excess_overlap(
            rect=rect,
            forbidden_regions=forbidden_regions,
        )
        if excess_overlap > 0.0:
            penalty = {"hard": 4200.0, "strong": 2600.0}.get(
                anchor_strength.strip().lower(),
                1200.0,
            )
            score -= min(1.5, excess_overlap) * penalty
    return score


def _forbidden_region_excess_overlap(
    *,
    rect: tuple[int, int, int, int],
    forbidden_regions: Sequence[Mapping[str, Any]],
) -> float:
    excess_overlap = 0.0
    for forbidden in forbidden_regions:
        bbox = _rect_tuple(forbidden.get("bbox"))
        if bbox is None:
            continue
        max_overlap = float(forbidden.get("max_overlap_ratio") or 0.0)
        excess_overlap += max(0.0, _rect_overlap_ratio(rect, bbox) - max_overlap)
    return excess_overlap


def _anchor_forbidden_overlap_exceeds_limit(
    *,
    rect: tuple[int, int, int, int],
    forbidden_regions: Sequence[Mapping[str, Any]],
) -> bool:
    if not forbidden_regions:
        return False
    return any(
        _forbidden_region_rejects_anchor_candidate(rect=rect, forbidden=forbidden)
        for forbidden in forbidden_regions
    )


def _forbidden_region_rejects_anchor_candidate(
    *,
    rect: tuple[int, int, int, int],
    forbidden: Mapping[str, Any],
) -> bool:
    bbox = _rect_tuple(forbidden.get("bbox"))
    if bbox is None:
        return False
    max_overlap = float(forbidden.get("max_overlap_ratio") or 0.0)
    ratio = _rect_overlap_ratio(rect, bbox)
    if ratio <= max_overlap + 1e-9:
        return False
    enforcement = str(forbidden.get("enforcement") or "hard").strip().lower()
    if enforcement == "hard":
        return True
    if enforcement == "hard_soft":
        return ratio > _hard_soft_anchor_reject_ratio(forbidden) + 1e-9
    return False


def _hard_soft_anchor_reject_ratio(forbidden: Mapping[str, Any]) -> float:
    max_overlap = float(forbidden.get("max_overlap_ratio") or 0.0)
    return max(OBJECT_LEVEL_HARD_SOFT_ANCHOR_REJECT_RATIO, max_overlap)


def _anchor_kind_matches_preferred_side(anchor_kind: str, preferred: str) -> bool:
    actual = anchor_kind.strip().lower()
    target = preferred.strip().lower()
    if not target:
        return True
    if target == "center":
        return actual == "center"
    if target == "window_side":
        return actual == "window_side"
    return actual == target


def _desired_anchor_front_vector(
    *,
    cluster_id: str,
    object_id: str,
    rect: tuple[int, int, int, int],
    room_bbox: tuple[int, int, int, int],
    relation_plan: Mapping[str, Any] | None,
    anchor_kind: str,
    placement_behavior: Mapping[str, Any] | None = None,
) -> tuple[float, float] | None:
    intents = _object_level_orientation_intents(
        cluster_id=cluster_id,
        object_id=object_id,
        relation_plan=relation_plan,
        include_cluster=True,
    )
    contact_front = _wall_contact_inward_front(rect=rect, room_bbox=room_bbox)
    if contact_front is not None:
        return contact_front
    if anchor_kind in {
        "bottom_wall",
        "entry_side",
        "left_wall",
        "right_wall",
        "top_wall",
        "window_side",
    }:
        wall_backing = _placement_behavior_value(placement_behavior, "wall_backing")
        front_space = _placement_behavior_value(placement_behavior, "front_space")
        if (
            {"back_to_wall", "front_to_open_space"} & intents
            or wall_backing in {"preferred", "required"}
            or front_space in {"preferred", "required"}
        ):
            return _nearest_wall_inward_front(rect=rect, room_bbox=room_bbox)
    return None


def _object_level_orientation_intents(
    *,
    cluster_id: str,
    object_id: str,
    relation_plan: Mapping[str, Any] | None,
    include_cluster: bool,
) -> set[str]:
    if not isinstance(relation_plan, Mapping):
        return set()
    intents: set[str] = set()
    for row in relation_plan.get("object_orientations") or []:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("cluster_id") or "").strip() != cluster_id:
            continue
        if str(row.get("object_id") or "").strip() != object_id:
            continue
        intents.update(_normalized_text_set(row.get("intents")))
    if include_cluster:
        for row in relation_plan.get("cluster_orientations") or []:
            if not isinstance(row, Mapping):
                continue
            if str(row.get("cluster_id") or "").strip() == cluster_id:
                intents.update(_normalized_text_set(row.get("intents")))
        intents.update(
            _placement_behavior_orientation_intents(
                _placement_behavior_for_cluster(relation_plan, cluster_id)
            )
        )
    return intents


def _object_level_target_cluster_id(
    *,
    cluster_id: str,
    object_id: str,
    relation_plan: Mapping[str, Any] | None,
) -> str | None:
    if not isinstance(relation_plan, Mapping):
        return None
    for row in relation_plan.get("object_orientations") or []:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("cluster_id") or "").strip() != cluster_id:
            continue
        if str(row.get("object_id") or "").strip() != object_id:
            continue
        target = str(row.get("target_cluster_id") or "").strip()
        if target:
            return target
    return _cluster_face_target_id(dict(relation_plan), cluster_id)


def _nearest_wall_inward_front(
    *,
    rect: tuple[int, int, int, int],
    room_bbox: tuple[int, int, int, int],
) -> tuple[float, float]:
    distances = [
        (abs(rect[0] - room_bbox[0]), (1.0, 0.0)),
        (abs(room_bbox[2] - rect[2]), (-1.0, 0.0)),
        (abs(rect[1] - room_bbox[1]), (0.0, 1.0)),
        (abs(room_bbox[3] - rect[3]), (0.0, -1.0)),
    ]
    return min(distances, key=lambda item: item[0])[1]


def _wall_contact_inward_front(
    *,
    rect: tuple[int, int, int, int],
    room_bbox: tuple[int, int, int, int],
    tolerance_mm: int = OBJECT_LEVEL_WALL_CONTACT_TOLERANCE_MM,
) -> tuple[float, float] | None:
    contacts = [
        (abs(rect[0] - room_bbox[0]), (1.0, 0.0)),
        (abs(room_bbox[2] - rect[2]), (-1.0, 0.0)),
        (abs(rect[1] - room_bbox[1]), (0.0, 1.0)),
        (abs(room_bbox[3] - rect[3]), (0.0, -1.0)),
    ]
    flush_contacts = [item for item in contacts if item[0] <= max(0, int(tolerance_mm))]
    if not flush_contacts:
        return None
    return min(flush_contacts, key=lambda item: item[0])[1]


def _front_alignment_score(
    *,
    front_world: tuple[float, float],
    desired_front: tuple[float, float] | None,
) -> float:
    if desired_front is None:
        return 0.0
    dot = _dot_normalized(front_world, desired_front)
    return dot * 1400.0


def _front_matches_required_direction(
    *,
    front_world: tuple[float, float] | None,
    desired_front: tuple[float, float] | None,
    min_dot: float = OBJECT_LEVEL_FRONT_ALIGNMENT_MIN_DOT,
) -> bool:
    if front_world is None or desired_front is None:
        return False
    return _dot_normalized(front_world, desired_front) >= float(min_dot)


def _dot_normalized(
    left: tuple[float, float],
    right: tuple[float, float],
) -> float:
    left_norm = math.hypot(left[0], left[1])
    right_norm = math.hypot(right[0], right[1])
    if left_norm <= 1e-9 or right_norm <= 1e-9:
        return 0.0
    return (left[0] / left_norm) * (right[0] / right_norm) + (left[1] / left_norm) * (
        right[1] / right_norm
    )


def _search_anchor_solutions(
    *,
    anchor_order: Sequence[str],
    anchor_candidates_by_cluster: Mapping[str, Sequence[dict[str, Any]]],
    world: Mapping[str, Any],
    room_model: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
    max_solutions: int,
) -> Sequence[dict[str, Any]]:
    solutions: list[dict[str, Any]] = []
    visited_leaf_count = 0
    max_leaf_count = max(12000, int(max_solutions) * 512)
    protected_regions = (
        world.get("protected_regions")
        if isinstance(world.get("protected_regions"), Sequence)
        else []
    )
    forbidden_regions = (
        world.get("cluster_forbidden_regions")
        if isinstance(world.get("cluster_forbidden_regions"), Sequence)
        else []
    )

    def rec(
        index: int,
        chosen: list[dict[str, Any]],
        dropped_inventory_by_cluster: dict[str, list[dict[str, Any]]],
    ) -> None:
        nonlocal visited_leaf_count
        if visited_leaf_count >= max_leaf_count:
            return
        if index >= len(anchor_order):
            visited_leaf_count += 1
            placed = {row["cluster_id"]: row for row in chosen}
            score = _anchor_pair_solution_score(placed, relation_plan) + sum(
                float(row.get("candidate_score") or 0.0) for row in chosen
            )
            score -= _object_level_protected_region_penalty(
                _object_level_protected_region_issues(
                    placed_objects=chosen,
                    protected_regions=protected_regions,
                )
            )
            score -= _object_level_protected_region_penalty(
                _object_level_cluster_forbidden_region_issues(
                    placed_objects=chosen,
                    forbidden_regions=forbidden_regions,
                )
            )
            solutions.append(
                {
                    "anchor_solution": placed,
                    "anchor_score": score,
                    "dropped_inventory_by_cluster": deepcopy(
                        {
                            cluster_id: rows
                            for cluster_id, rows in dropped_inventory_by_cluster.items()
                            if rows
                        }
                    ),
                }
            )
            _trim_anchor_solution_pool(solutions, max_solutions=max_solutions)
            return
        cluster_id = anchor_order[index]
        cluster_program = world["clusters_by_id"][cluster_id]
        if _cluster_is_solver_trial_optional(cluster_program):
            drop_records = _optional_trial_cluster_drop_records(cluster_program)
            dropped_inventory_by_cluster.setdefault(cluster_id, []).extend(drop_records)
            rec(index + 1, chosen, dropped_inventory_by_cluster)
            if drop_records:
                del dropped_inventory_by_cluster[cluster_id][-len(drop_records) :]
        for candidate in anchor_candidates_by_cluster.get(cluster_id, []):
            rect = candidate["rect"]
            if any(_rects_overlap(rect, row["rect"]) for row in chosen):
                continue
            chosen.append(candidate)
            rec(index + 1, chosen, dropped_inventory_by_cluster)
            chosen.pop()

    rec(0, [], {})
    return solutions


def _trim_anchor_solution_pool(
    solutions: list[dict[str, Any]],
    *,
    max_solutions: int,
) -> None:
    if len(solutions) <= max_solutions:
        solutions.sort(
            key=lambda item: float(item.get("anchor_score") or 0.0),
            reverse=True,
        )
        return

    ranked = sorted(
        solutions,
        key=lambda item: float(item.get("anchor_score") or 0.0),
        reverse=True,
    )
    kept = ranked[:max_solutions]
    trial_candidates = [
        item for item in ranked if _anchor_solution_has_dropped_trial(item)
    ]
    if trial_candidates and not any(
        _anchor_solution_has_dropped_trial(item) for item in kept
    ):
        kept[-1] = trial_candidates[0]
    solutions[:] = kept


def _anchor_solution_has_dropped_trial(solution: Mapping[str, Any]) -> bool:
    dropped = solution.get("dropped_inventory_by_cluster")
    if not isinstance(dropped, Mapping):
        return False
    return any(
        isinstance(rows, Sequence) and not isinstance(rows, str) and bool(rows)
        for rows in dropped.values()
    )


def _optional_trial_cluster_drop_records(
    cluster_program: Mapping[str, Any],
) -> list[dict[str, str]]:
    object_program = (
        cluster_program.get("object_program")
        if isinstance(cluster_program.get("object_program"), Mapping)
        else {}
    )
    members = [
        item
        for item in object_program.get("members")
        or cluster_program.get("members")
        or []
        if isinstance(item, str) and item.strip()
    ]
    if not members:
        anchor_id = _cluster_program_dominant_anchor(cluster_program)
        if anchor_id is not None:
            members = [anchor_id]
    return [
        {
            "object_id": object_id,
            "reason": "optional_trial_anchor_cluster_not_placed",
        }
        for object_id in members
    ]


def _anchor_pair_solution_score(
    anchor_solution: Mapping[str, dict[str, Any]],
    relation_plan: Mapping[str, Any] | None,
) -> float:
    primary = _extract_layout_primary_cluster_id(
        dict(relation_plan) if isinstance(relation_plan, dict) else None
    )
    secondary = _extract_layout_secondary_cluster_id(
        dict(relation_plan) if isinstance(relation_plan, dict) else None
    )
    if (
        not primary
        or not secondary
        or primary not in anchor_solution
        or secondary not in anchor_solution
    ):
        return 0.0
    a = anchor_solution[primary]["rect"]
    b = anchor_solution[secondary]["rect"]
    ac = ((a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0)
    bc = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
    distance = math.hypot(ac[0] - bc[0], ac[1] - bc[1])
    desired = 2400.0
    pair_score = 1800.0 - abs(distance - desired)
    if a[1] < b[1]:
        pair_score += 120.0
    pair_score += _scenario_anchor_pair_score(
        primary_anchor=anchor_solution[primary],
        secondary_anchor=anchor_solution[secondary],
        relation_plan=relation_plan,
    )
    pair_score += _anchor_pair_orientation_score(
        left_cluster_id=primary,
        right_cluster_id=secondary,
        left_anchor=anchor_solution[primary],
        right_anchor=anchor_solution[secondary],
        relation_plan=relation_plan,
    )
    return pair_score


def _scenario_anchor_pair_score(
    *,
    primary_anchor: Mapping[str, Any],
    secondary_anchor: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
) -> float:
    contract = _primary_anchor_pair_contract(relation_plan)
    pair_type = str(contract.get("pair_type") or "").strip().lower()
    if not pair_type:
        return 0.0
    primary_kind = str(primary_anchor.get("anchor_kind") or "").strip().lower()
    secondary_kind = str(secondary_anchor.get("anchor_kind") or "").strip().lower()
    if pair_type in {"opposite_walls", "daylight_opposite"}:
        if _anchor_kinds_are_opposite(primary_kind, secondary_kind):
            return 2200.0
        if primary_kind == secondary_kind:
            return -2600.0
        return -650.0
    if pair_type == "floating_to_wall":
        score = 0.0
        score += 2400.0 if primary_kind == "center" else -2800.0
        score += 1200.0 if secondary_kind.endswith("_wall") else -900.0
        return score
    if pair_type in {"adjacent_walls", "offset_axis"}:
        if _anchor_kinds_are_adjacent(primary_kind, secondary_kind):
            return 1500.0
        if primary_kind == secondary_kind:
            return -1600.0
    return 0.0


def _primary_anchor_pair_contract(
    relation_plan: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if not isinstance(relation_plan, Mapping):
        return {}
    contracts = relation_plan.get("anchor_pair_contracts")
    if isinstance(contracts, Sequence) and not isinstance(contracts, str):
        for row in contracts:
            if isinstance(row, Mapping):
                return row
    return {}


def _anchor_kinds_are_opposite(left: str, right: str) -> bool:
    pairs = {
        frozenset(("top_wall", "bottom_wall")),
        frozenset(("left_wall", "right_wall")),
        frozenset(("window_side", "bottom_wall")),
    }
    return frozenset((left, right)) in pairs


def _anchor_kinds_are_adjacent(left: str, right: str) -> bool:
    if not left.endswith("_wall") or not right.endswith("_wall"):
        return False
    return not _anchor_kinds_are_opposite(left, right) and left != right


def _anchor_pair_orientation_score(
    *,
    left_cluster_id: str,
    right_cluster_id: str,
    left_anchor: Mapping[str, Any],
    right_anchor: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
) -> float:
    if not isinstance(relation_plan, Mapping):
        return 0.0
    score = 0.0
    pairs = (
        (left_cluster_id, left_anchor, right_cluster_id, right_anchor),
        (right_cluster_id, right_anchor, left_cluster_id, left_anchor),
    )
    for cluster_id, anchor, target_cluster_id, target_anchor in pairs:
        object_id = str(anchor.get("object_id") or "")
        intents = _object_level_orientation_intents(
            cluster_id=cluster_id,
            object_id=object_id,
            relation_plan=relation_plan,
            include_cluster=True,
        )
        if "face_cluster" not in intents:
            continue
        configured_target = _object_level_target_cluster_id(
            cluster_id=cluster_id,
            object_id=object_id,
            relation_plan=relation_plan,
        )
        if configured_target and configured_target != target_cluster_id:
            continue
        front = _front_tuple(anchor.get("front_world"))
        desired = _vector_between_rect_centers(
            tuple(anchor["rect"]),
            tuple(target_anchor["rect"]),
        )
        if front is None or desired is None:
            continue
        score += _dot_normalized(front, desired) * 1600.0
    return score


def _place_support_objects_for_solution(
    *,
    solution: Mapping[str, Any],
    world: Mapping[str, Any],
    room_model: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
    grid_mm: int,
    max_solutions: int,
) -> list[dict[str, Any]]:
    placed_objects: list[dict[str, Any]] = []
    dropped_inventory_by_cluster: dict[str, list[dict[str, Any]]] = {
        str(cluster_id): [dict(row) for row in rows if isinstance(row, Mapping)]
        for cluster_id, rows in (
            solution.get("dropped_inventory_by_cluster") or {}
        ).items()
        if isinstance(rows, Sequence) and not isinstance(rows, str)
    }
    anchor_solution = (
        solution.get("anchor_solution")
        if isinstance(solution.get("anchor_solution"), Mapping)
        else {}
    )
    occupied = [row["rect"] for row in anchor_solution.values()]
    # materialize anchors first
    for cluster_id, anchor_row in anchor_solution.items():
        cluster_program = world["clusters_by_id"][cluster_id]
        anchor_id = anchor_row["object_id"]
        placed_objects.append(
            _materialize_object_row(
                cluster_program,
                anchor_id,
                anchor_row["x"],
                anchor_row["y"],
                anchor_row["rot"],
            )
        )
    placed_by_id = {
        (row["cluster_id"], row["object_id"]): row for row in placed_objects
    }
    protected_regions = (
        world.get("protected_regions")
        if isinstance(world.get("protected_regions"), Sequence)
        else []
    )
    results: list[dict[str, Any]] = []
    tasks: list[tuple[str, str, Mapping[str, Any], str]] = []

    for cluster_id in world["anchor_cluster_order"]:
        cluster_program = world["clusters_by_id"][cluster_id]
        if cluster_id not in anchor_solution and _cluster_is_solver_trial_optional(
            cluster_program
        ):
            continue
        object_program = cluster_program["object_program"]
        anchor_id = _cluster_program_dominant_anchor(cluster_program)
        placement_order = [
            object_id
            for object_id in object_program.get("placement_order") or []
            if isinstance(object_id, str) and object_id != anchor_id
        ]
        support_edges = {
            str(row.get("object_id")): row
            for row in object_program.get("support_edges") or []
            if isinstance(row, Mapping) and isinstance(row.get("object_id"), str)
        }
        for object_id in placement_order:
            spec = _object_spec(cluster_program, object_id)
            if spec is None:
                continue
            edge = support_edges.get(object_id, {})
            base_id = str(edge.get("relative_to") or anchor_id or "")
            tasks.append((cluster_id, object_id, edge, base_id))

    def search(index: int, support_score: float) -> None:
        if len(results) >= max(1, int(max_solutions)):
            return
        if index >= len(tasks):
            results.append(
                {
                    "placed_objects": deepcopy(placed_objects),
                    "dropped_inventory_by_cluster": deepcopy(
                        {
                            cluster_id: rows
                            for cluster_id, rows in dropped_inventory_by_cluster.items()
                            if rows
                        }
                    ),
                    "support_score": round(float(support_score), 3),
                    "protected_region_issue_count": _object_level_protected_region_issue_count(
                        placed_objects=placed_objects,
                        protected_regions=protected_regions,
                    ),
                }
            )
            return
        cluster_id, object_id, edge, base_id = tasks[index]
        cluster_program = world["clusters_by_id"][cluster_id]
        object_program = cluster_program["object_program"]
        anchor_id = _cluster_program_dominant_anchor(cluster_program)
        droppable = _droppable_object_ids(object_program)
        protected = set(object_program.get("protected_ids") or [])

        base_row = placed_by_id.get((cluster_id, base_id))
        if base_row is None and anchor_id is not None:
            base_row = placed_by_id.get((cluster_id, anchor_id))
        if base_row is None:
            if object_id in droppable and object_id not in protected:
                dropped_inventory_by_cluster.setdefault(cluster_id, []).append(
                    {"object_id": object_id, "reason": "base_missing"}
                )
                search(index + 1, support_score - 120.0)
                dropped_inventory_by_cluster[cluster_id].pop()
                return
            return

        slots = _support_slot_candidates(
            cluster_program=cluster_program,
            object_id=object_id,
            base_row=base_row,
            edge=edge,
            grid_mm=grid_mm,
        )
        viable_slots: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
        for slot in slots:
            rect = slot["rect"]
            if not _rect_inside_room(rect, world["room_bbox"]):
                continue
            if any(_rects_overlap(rect, occ) for occ in occupied):
                continue
            materialized = _materialize_object_row(
                cluster_program,
                object_id,
                slot["x"],
                slot["y"],
                slot["rot"],
                relative_to=base_id,
            )
            relax_penalty = _support_slot_relaxation_penalty(
                candidate=materialized,
                placed_objects=placed_objects,
                protected_regions=protected_regions,
                world=world,
                relation_plan=relation_plan,
            )
            viable_slots.append((relax_penalty, slot, materialized))
        viable_slots.sort(
            key=lambda item: (
                item[0],
                -float(item[1].get("orientation_score") or 0.0),
                item[1]["y"],
                item[1]["x"],
                item[1]["rot"],
            )
        )

        for relax_penalty, slot, materialized in viable_slots:
            rect = slot["rect"]
            placed_objects.append(materialized)
            placed_by_id[(cluster_id, object_id)] = materialized
            occupied.append(rect)
            search(
                index + 1,
                support_score
                + float(slot.get("orientation_score") or 0.0)
                - relax_penalty,
            )
            occupied.pop()
            placed_by_id.pop((cluster_id, object_id), None)
            placed_objects.pop()
            if len(results) >= max(1, int(max_solutions)):
                break

        if object_id in droppable and object_id not in protected:
            dropped_inventory_by_cluster.setdefault(cluster_id, []).append(
                {"object_id": object_id, "reason": "support_slot_failed"}
            )
            search(index + 1, support_score - 90.0)
            dropped_inventory_by_cluster[cluster_id].pop()
            return
        return

    search(0, 0.0)
    return results


def _support_slot_relaxation_penalty(
    *,
    candidate: Mapping[str, Any],
    placed_objects: Sequence[Mapping[str, Any]],
    protected_regions: Sequence[Mapping[str, Any]],
    world: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
) -> float:
    protected_issues = _object_level_protected_region_issues(
        placed_objects=[candidate],
        protected_regions=protected_regions,
    )
    forbidden_regions = (
        world.get("cluster_forbidden_regions")
        if isinstance(world.get("cluster_forbidden_regions"), Sequence)
        else []
    )
    forbidden_issues = _object_level_cluster_forbidden_region_issues(
        placed_objects=[candidate],
        forbidden_regions=forbidden_regions,
    )
    functional_issues = _object_level_functional_geometry_issues(
        placed_objects=[*placed_objects, candidate],
        world=world,
        relation_plan=relation_plan,
    )
    blocking_functional = [
        issue
        for issue in functional_issues
        if (
            (
                str(issue.get("cluster_id") or "")
                == str(candidate.get("cluster_id") or "")
                and str(issue.get("object_id") or "")
                == str(candidate.get("object_id") or "")
            )
            or (
                str(issue.get("blocked_cluster_id") or "")
                == str(candidate.get("cluster_id") or "")
                and str(issue.get("blocked_object_id") or "")
                == str(candidate.get("object_id") or "")
            )
        )
        and str(issue.get("violation_severity") or "").strip().lower() == "blocking"
    ]
    return (
        _object_level_protected_region_penalty(protected_issues)
        + _object_level_protected_region_penalty(forbidden_issues)
        + 1400.0 * len(blocking_functional)
        + 180.0 * max(0, len(functional_issues) - len(blocking_functional))
    )


def _repair_object_level_solution_geometry(
    *,
    solution: Mapping[str, Any],
    world: Mapping[str, Any],
    grid_mm: int,
) -> dict[str, Any] | None:
    rows = [
        deepcopy(row)
        for row in (solution.get("placed_objects") or [])
        if isinstance(row, Mapping)
    ]
    if not rows:
        return {
            "placed_objects": [],
            "summary": {
                "status": "clean",
                "moved_objects": [],
                "total_shift_mm": 0.0,
                "penalty": 0,
            },
        }

    original_rows_by_key = {
        _object_repair_key(row): row
        for row in rows
        if _object_repair_key(row) is not None
    }
    repaired_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    repaired_by_index: dict[int, dict[str, Any]] = {}
    occupied_rects: list[tuple[int, int, int, int]] = []
    moved_objects: list[dict[str, Any]] = []

    for index, row in sorted(
        enumerate(rows), key=lambda item: _object_repair_order_key(item[1])
    ):
        rect = _rect_tuple(row.get("rect"))
        if rect is None:
            return None
        candidate_rect = _best_object_geometry_repair_rect(
            row=row,
            original_rect=rect,
            original_rows_by_key=original_rows_by_key,
            repaired_by_key=repaired_by_key,
            occupied_rects=occupied_rects,
            front_access_rows=tuple(repaired_by_key.values()),
            world=world,
            grid_mm=grid_mm,
        )
        if candidate_rect is None:
            return None
        repaired_row = _object_row_with_rect(row, candidate_rect)
        repaired_by_index[index] = repaired_row
        key = _object_repair_key(repaired_row)
        if key is not None:
            repaired_by_key[key] = repaired_row
        occupied_rects.append(candidate_rect)
        if candidate_rect != rect:
            old_center = _rect_center(rect)
            new_center = _rect_center(candidate_rect)
            shift_mm = math.hypot(
                new_center[0] - old_center[0], new_center[1] - old_center[1]
            )
            moved_objects.append(
                {
                    "cluster_id": str(row.get("cluster_id") or ""),
                    "object_id": str(row.get("object_id") or ""),
                    "from_rect": list(rect),
                    "to_rect": list(candidate_rect),
                    "shift_mm": round(shift_mm, 3),
                    "reason": "polygon_oob_or_overlap_guardrail",
                }
            )

    repaired_rows = [repaired_by_index[index] for index in range(len(rows))]
    total_shift = sum(float(item["shift_mm"]) for item in moved_objects)
    penalty = int(round(total_shift * 1.8 + len(moved_objects) * 180.0))
    return {
        "placed_objects": repaired_rows,
        "summary": {
            "status": "repaired" if moved_objects else "clean",
            "moved_objects": moved_objects,
            "total_shift_mm": round(total_shift, 3),
            "penalty": penalty,
        },
    }


def _best_object_geometry_repair_rect(
    *,
    row: Mapping[str, Any],
    original_rect: tuple[int, int, int, int],
    original_rows_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    repaired_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    occupied_rects: Sequence[tuple[int, int, int, int]],
    front_access_rows: Sequence[Mapping[str, Any]],
    world: Mapping[str, Any],
    grid_mm: int,
) -> tuple[int, int, int, int] | None:
    if _object_rect_is_usable(
        original_rect,
        occupied_rects,
        world,
        row=row,
        front_access_rows=front_access_rows,
    ):
        return original_rect

    room_bbox = world["room_bbox"]
    width = original_rect[2] - original_rect[0]
    height = original_rect[3] - original_rect[1]
    if width <= 0 or height <= 0:
        return None

    grid = max(25, int(grid_mm))
    original_center = _rect_center(original_rect)
    target_center = _object_repair_target_center(
        row=row,
        original_center=original_center,
        original_rows_by_key=original_rows_by_key,
        repaired_by_key=repaired_by_key,
    )
    x_min = int(room_bbox[0])
    y_min = int(room_bbox[1])
    x_max = int(room_bbox[2]) - width
    y_max = int(room_bbox[3]) - height
    if x_max < x_min or y_max < y_min:
        return None

    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    seen: set[tuple[int, int]] = set()

    def add_origin(x: int, y: int) -> None:
        sx = _snap_to_grid(x, grid)
        sy = _snap_to_grid(y, grid)
        sx = max(x_min, min(x_max, sx))
        sy = max(y_min, min(y_max, sy))
        key = (sx, sy)
        if key in seen:
            return
        seen.add(key)
        rect = (sx, sy, sx + width, sy + height)
        if not _object_rect_is_usable(
            rect,
            occupied_rects,
            world,
            row=row,
            front_access_rows=front_access_rows,
        ):
            return
        candidates.append(
            (
                _object_geometry_repair_score(
                    row=row,
                    rect=rect,
                    original_rect=original_rect,
                    target_center=target_center,
                    original_rows_by_key=original_rows_by_key,
                    repaired_by_key=repaired_by_key,
                ),
                rect,
            )
        )

    original_origin = (original_rect[0], original_rect[1])
    target_origin = (
        int(round(target_center[0] - width / 2.0)),
        int(round(target_center[1] - height / 2.0)),
    )
    add_origin(original_origin[0], original_origin[1])
    add_origin(target_origin[0], target_origin[1])
    local_radius = max(grid, OBJECT_LEVEL_GEOMETRY_REPAIR_LOCAL_RADIUS_MM)
    for center_x, center_y in (original_origin, target_origin):
        for radius in range(grid, local_radius + 1, grid):
            for dx in range(-radius, radius + 1, grid):
                add_origin(center_x + dx, center_y - radius)
                add_origin(center_x + dx, center_y + radius)
            for dy in range(-radius + grid, radius, grid):
                add_origin(center_x - radius, center_y + dy)
                add_origin(center_x + radius, center_y + dy)
            if len(candidates) >= OBJECT_LEVEL_GEOMETRY_REPAIR_MAX_CANDIDATES:
                candidates.sort(key=lambda item: item[0])
                return candidates[0][1]
    if candidates:
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    for x in range(x_min, x_max + 1, grid):
        for y in range(y_min, y_max + 1, grid):
            add_origin(x, y)

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _object_rect_is_usable(
    rect: tuple[int, int, int, int],
    occupied_rects: Sequence[tuple[int, int, int, int]],
    world: Mapping[str, Any],
    *,
    row: Mapping[str, Any] | None = None,
    front_access_rows: Sequence[Mapping[str, Any]] = (),
) -> bool:
    if not _rect_inside_room_footprint(rect, world):
        return False
    if any(_rects_overlap(rect, occupied) for occupied in occupied_rects):
        return False
    if row is None:
        return True
    if _rect_blocks_front_access_rows(
        rect=rect,
        row=row,
        front_access_rows=front_access_rows,
        room_bbox=world["room_bbox"],
    ):
        return False
    forbidden_regions = (
        world.get("cluster_forbidden_regions")
        if isinstance(world.get("cluster_forbidden_regions"), Sequence)
        else []
    )
    if not forbidden_regions:
        return True
    candidate = dict(row)
    candidate["rect"] = rect
    return not any(
        _forbidden_region_issue_blocks_geometry(issue)
        for issue in _object_level_cluster_forbidden_region_issues(
            placed_objects=[candidate],
            forbidden_regions=forbidden_regions,
        )
    )


def _rect_blocks_front_access_rows(
    *,
    rect: tuple[int, int, int, int],
    row: Mapping[str, Any],
    front_access_rows: Sequence[Mapping[str, Any]],
    room_bbox: tuple[int, int, int, int],
) -> bool:
    candidate = dict(row)
    candidate["rect"] = rect
    if _object_allowed_in_front_access_zone(candidate):
        return False
    candidate_key = _object_repair_key(candidate)
    for front_access_row in front_access_rows:
        front_access_key = _object_repair_key(front_access_row)
        if candidate_key is not None and front_access_key == candidate_key:
            continue
        access_zone = _front_access_zone_for_row(front_access_row, room_bbox)
        if access_zone is None:
            continue
        if _rect_overlap_ratio(rect, access_zone) > 0.02:
            return True
    return False


def _forbidden_region_issue_blocks_geometry(issue: Mapping[str, Any]) -> bool:
    enforcement = str(issue.get("enforcement") or "").strip().lower()
    return enforcement == "hard" and _protected_region_issue_is_blocking(issue)


def _object_repair_target_center(
    *,
    row: Mapping[str, Any],
    original_center: tuple[float, float],
    original_rows_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    repaired_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[float, float]:
    cluster_id = str(row.get("cluster_id") or "")
    relative_to = str(row.get("relative_to") or "")
    if not cluster_id or not relative_to:
        return original_center
    base_key = (cluster_id, relative_to)
    original_base = original_rows_by_key.get(base_key)
    repaired_base = repaired_by_key.get(base_key)
    if original_base is None or repaired_base is None:
        return original_center
    original_base_rect = _rect_tuple(original_base.get("rect"))
    repaired_base_rect = _rect_tuple(repaired_base.get("rect"))
    if original_base_rect is None or repaired_base_rect is None:
        return original_center
    original_base_center = _rect_center(original_base_rect)
    repaired_base_center = _rect_center(repaired_base_rect)
    return (
        repaired_base_center[0] + original_center[0] - original_base_center[0],
        repaired_base_center[1] + original_center[1] - original_base_center[1],
    )


def _object_geometry_repair_score(
    *,
    row: Mapping[str, Any],
    rect: tuple[int, int, int, int],
    original_rect: tuple[int, int, int, int],
    target_center: tuple[float, float],
    original_rows_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    repaired_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
) -> float:
    center = _rect_center(rect)
    original_center = _rect_center(original_rect)
    score = math.hypot(center[0] - original_center[0], center[1] - original_center[1])
    score += 0.55 * math.hypot(
        center[0] - target_center[0], center[1] - target_center[1]
    )
    score += _object_relative_side_penalty(
        row=row,
        candidate_center=center,
        original_center=original_center,
        original_rows_by_key=original_rows_by_key,
        repaired_by_key=repaired_by_key,
    )
    return score


def _object_relative_side_penalty(
    *,
    row: Mapping[str, Any],
    candidate_center: tuple[float, float],
    original_center: tuple[float, float],
    original_rows_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    repaired_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
) -> float:
    cluster_id = str(row.get("cluster_id") or "")
    relative_to = str(row.get("relative_to") or "")
    if not cluster_id or not relative_to:
        return 0.0
    base_key = (cluster_id, relative_to)
    original_base_rect = _rect_tuple(
        (original_rows_by_key.get(base_key) or {}).get("rect")
    )
    repaired_base_rect = _rect_tuple((repaired_by_key.get(base_key) or {}).get("rect"))
    if original_base_rect is None or repaired_base_rect is None:
        return 0.0
    original_base_center = _rect_center(original_base_rect)
    repaired_base_center = _rect_center(repaired_base_rect)
    original_vec = (
        original_center[0] - original_base_center[0],
        original_center[1] - original_base_center[1],
    )
    candidate_vec = (
        candidate_center[0] - repaired_base_center[0],
        candidate_center[1] - repaired_base_center[1],
    )
    penalty = 0.0
    if abs(original_vec[0]) >= abs(original_vec[1]) and abs(original_vec[0]) > 1e-6:
        if original_vec[0] * candidate_vec[0] <= 0:
            penalty += 2400.0
    elif abs(original_vec[1]) > 1e-6 and original_vec[1] * candidate_vec[1] <= 0:
        penalty += 2400.0
    dot = _dot_normalized(original_vec, candidate_vec)
    if dot < 0.25:
        penalty += (0.25 - dot) * 1800.0
    return penalty


def _object_repair_order_key(row: Mapping[str, Any]) -> tuple[int, str, str]:
    priority = str(row.get("priority") or "").strip().lower()
    role = str(row.get("role") or "").strip().lower()
    has_relative = bool(str(row.get("relative_to") or "").strip())
    if priority in {"anchor", "dominant_anchor"} or role == "dominant_anchor":
        rank = 0
    elif has_relative:
        rank = 2
    else:
        rank = 1
    return rank, str(row.get("cluster_id") or ""), str(row.get("object_id") or "")


def _object_repair_key(row: Mapping[str, Any]) -> tuple[str, str] | None:
    cluster_id = str(row.get("cluster_id") or "").strip()
    object_id = str(row.get("object_id") or "").strip()
    if not cluster_id or not object_id:
        return None
    return cluster_id, object_id


def _object_row_with_rect(
    row: Mapping[str, Any], rect: tuple[int, int, int, int]
) -> dict[str, Any]:
    next_row = dict(row)
    next_row["x"] = rect[0]
    next_row["y"] = rect[1]
    next_row["w"] = rect[2] - rect[0]
    next_row["h"] = rect[3] - rect[1]
    next_row["rect"] = rect
    next_row["bbox"] = {
        "min_x": rect[0],
        "min_y": rect[1],
        "max_x": rect[2],
        "max_y": rect[3],
    }
    next_row["center"] = {
        "x": int(round((rect[0] + rect[2]) / 2.0)),
        "y": int(round((rect[1] + rect[3]) / 2.0)),
    }
    return next_row


def _apply_object_level_geometry_repair_penalty(
    verify: dict[str, Any], repair_summary: Mapping[str, Any] | None
) -> None:
    if not isinstance(repair_summary, Mapping):
        return
    moved_objects = repair_summary.get("moved_objects")
    if not isinstance(moved_objects, Sequence) or isinstance(moved_objects, str):
        return
    moved_count = len(moved_objects)
    if moved_count <= 0:
        verify["geometry_repair"] = deepcopy(dict(repair_summary))
        verify["geometry_repair_penalty"] = 0
        return
    penalty = int(repair_summary.get("penalty") or 0)
    verify["geometry_repair"] = deepcopy(dict(repair_summary))
    verify["geometry_repair_penalty"] = penalty
    verify["soft_issue_count"] = int(verify.get("soft_issue_count") or 0) + moved_count
    verify["layout_score"] = int(verify.get("layout_score") or 0) - penalty


def _droppable_object_ids(object_program: Mapping[str, Any]) -> set[str]:
    members = {str(item) for item in (object_program.get("members") or []) if str(item)}
    protected = {str(item) for item in (object_program.get("protected_ids") or [])}
    optional = {str(item) for item in (object_program.get("optional_object_ids") or [])}
    support_edge_ids = {
        str(row.get("object_id"))
        for row in (object_program.get("support_edges") or [])
        if isinstance(row, Mapping) and str(row.get("object_id") or "")
    }
    object_specs = (
        object_program.get("object_specs_by_id")
        if isinstance(object_program.get("object_specs_by_id"), Mapping)
        else {}
    )
    droppable = {
        str(item)
        for item in (object_program.get("droppable_ids") or [])
        if str(item) in members
    }
    droppable.update(item for item in optional if item in members)
    for action in object_program.get("degradation_ladder") or []:
        token = str(action or "").strip()
        if token == "drop_secondary_support":
            for object_id in members:
                spec = (
                    object_specs.get(object_id)
                    if isinstance(object_specs, Mapping)
                    else None
                )
                role = (
                    str(spec.get("role") or "").strip().lower()
                    if isinstance(spec, Mapping)
                    else ""
                )
                priority = (
                    str(spec.get("priority") or "").strip().lower()
                    if isinstance(spec, Mapping)
                    else ""
                )
                if (
                    object_id not in protected
                    and object_id in support_edge_ids
                    and (
                        priority in {"secondary", "optional", "support"}
                        or role
                        in {
                            "support",
                            "secondary",
                            "secondary_support",
                            "decor_light",
                        }
                    )
                ):
                    droppable.add(object_id)
            continue
        if token.startswith("drop_"):
            object_id = token.removeprefix("drop_")
            if object_id and object_id in members:
                droppable.add(object_id)
    return droppable


def _normalized_support_side_options(edge: Mapping[str, Any]) -> list[str]:
    raw_options = [
        str(item).strip()
        for item in (edge.get("side_options") or ["front"])
        if str(item).strip()
    ]
    support_role = str(edge.get("support_role") or "").strip().lower()
    band_intent = str(edge.get("band_intent") or "").strip().lower()
    object_id = str(edge.get("object_id") or "").strip().lower()
    tokens = {support_role, band_intent, object_id}
    front_band_tokens = {
        "front_band",
        "frontal_support",
        "front_support",
        "coffee_table",
    }
    side_band_tokens = {
        "beside_base",
        "side_support",
        "side_table",
    }
    wall_band_tokens = {
        "wall_band",
        "wall_support",
    }
    flank_band_tokens = {"flank_band", "secondary_seat", "armchair", "chair"}
    if tokens & front_band_tokens:
        filtered = [
            option
            for option in raw_options
            if option
            in {
                "front",
                "head",
                "front_center",
                "head_center",
                "front_left",
                "front_right",
                "head_left",
                "head_right",
            }
        ]
        return filtered or ["front"]
    if tokens & side_band_tokens:
        rich_side_options = {
            "head_left",
            "head_right",
            "front_left",
            "front_right",
        }
        filtered = [
            option
            for option in raw_options
            if option in {"left", "right", *rich_side_options}
        ]
        if any(option in rich_side_options for option in filtered):
            return filtered
        return filtered or ["left", "right"]
    if tokens & wall_band_tokens:
        filtered = [
            option
            for option in raw_options
            if option
            in {
                "left",
                "right",
                "head",
                "front",
                "head_center",
                "front_center",
                "head_left",
                "head_right",
                "front_left",
                "front_right",
            }
        ]
        return filtered or ["left", "right"]
    if tokens & flank_band_tokens:
        filtered = [option for option in raw_options if option in {"left", "right"}]
        return filtered or ["left", "right"]
    return raw_options or ["front"]


def _support_slot_candidates(
    *,
    cluster_program: Mapping[str, Any],
    object_id: str,
    base_row: Mapping[str, Any],
    edge: Mapping[str, Any],
    grid_mm: int,
) -> list[dict[str, Any]]:
    spec = _object_spec(cluster_program, object_id)
    if spec is None:
        return []
    dims = (
        spec.get("rep_dims_mm") if isinstance(spec.get("rep_dims_mm"), Mapping) else {}
    )
    length_mm = max(160, int(dims.get("L") or 0))
    width_mm = max(160, int(dims.get("W") or 0))
    allowed_rotations = (
        spec.get("allowed_rotations")
        if isinstance(spec.get("allowed_rotations"), Sequence)
        else [0, 90, 180, 270]
    )
    side_options = _normalized_support_side_options(edge)
    gap_min = int(edge.get("gap_min_mm") or edge.get("gap_min") or 60)
    gap_max = int(
        edge.get("gap_max_mm") or edge.get("gap_max") or max(gap_min + 50, 180)
    )
    base_rect = base_row["rect"]
    base_center = (
        (base_rect[0] + base_rect[2]) / 2.0,
        (base_rect[1] + base_rect[3]) / 2.0,
    )
    base_w = base_rect[2] - base_rect[0]
    base_h = base_rect[3] - base_rect[1]
    base_front = _front_vector_from_rotation(
        base_row.get("front_token"), int(base_row.get("rot") or 0)
    )
    side_vec = (-base_front[1], base_front[0])
    orientation = str(edge.get("orientation") or "").strip().lower()
    out: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    for side_option in side_options:
        slot_side_option = _bedside_nightstand_slot_option(
            side_option=side_option,
            object_id=object_id,
            base_row=base_row,
        )
        for gap in _support_gap_samples(gap_min=gap_min, gap_max=gap_max):
            for rot in allowed_rotations:
                final_rot = int(rot) % 360
                if orientation == "same_direction":
                    final_rot = int(base_row.get("rot") or 0) % 360
                w_mm, h_mm = _rotate_dims_for_rot(length_mm, width_mm, final_rot)
                base_offset = _slot_offset(
                    side_option=slot_side_option,
                    gap=gap,
                    base_w=base_w,
                    base_h=base_h,
                    obj_w=w_mm,
                    obj_h=h_mm,
                    front=base_front,
                    side=side_vec,
                )
                for slide in _support_slot_slide_offsets(
                    side_option=slot_side_option,
                    base_w=base_w,
                    base_h=base_h,
                    front=base_front,
                    side=side_vec,
                ):
                    x = _snap_to_grid(
                        int(
                            round(
                                base_center[0] + base_offset[0] + slide[0] - w_mm / 2.0
                            )
                        ),
                        grid_mm,
                    )
                    y = _snap_to_grid(
                        int(
                            round(
                                base_center[1] + base_offset[1] + slide[1] - h_mm / 2.0
                            )
                        ),
                        grid_mm,
                    )
                    key = (x, y, final_rot)
                    if key in seen:
                        continue
                    seen.add(key)
                    rect = (x, y, x + w_mm, y + h_mm)
                    desired_front = _desired_support_front_vector(
                        orientation=orientation,
                        rect=rect,
                        base_row=base_row,
                    )
                    front_world = _front_vector_from_rotation(
                        spec.get("front")
                        if isinstance(spec.get("front"), str)
                        else "top",
                        final_rot,
                    )
                    out.append(
                        {
                            "x": x,
                            "y": y,
                            "rot": final_rot,
                            "w": w_mm,
                            "h": h_mm,
                            "rect": rect,
                            "orientation_score": _front_alignment_score(
                                front_world=front_world,
                                desired_front=desired_front,
                            ),
                        }
                    )
    out.sort(
        key=lambda item: (
            -float(item.get("orientation_score") or 0.0),
            item["y"],
            item["x"],
            item["rot"],
        )
    )
    return out[:OBJECT_LEVEL_MAX_SUPPORT_SLOT_CANDIDATES]


def _bedside_nightstand_slot_option(
    *,
    side_option: str,
    object_id: str,
    base_row: Mapping[str, Any],
) -> str:
    token = str(side_option).strip().lower()
    if token not in {"head_left", "head_right"}:
        return side_option

    object_key = object_id.strip().lower()
    base_key = str(base_row.get("category") or base_row.get("object_id") or "").lower()
    if "nightstand" not in object_key or "bed" not in base_key:
        return side_option

    return f"bedside_{token}"


def _support_gap_samples(*, gap_min: int, gap_max: int) -> list[int]:
    lower = max(0, int(gap_min))
    upper = max(lower, int(gap_max))
    if lower == upper:
        return [lower]
    midpoint = int(round((lower + upper) / 2.0))
    return sorted({lower, midpoint, upper})


def _support_slot_slide_offsets(
    *,
    side_option: str,
    base_w: int,
    base_h: int,
    front: tuple[int, int],
    side: tuple[int, int],
) -> list[tuple[float, float]]:
    token = str(side_option).lower()
    if token in {"bedside_head_left", "bedside_head_right"}:
        span = float(base_h if abs(front[1]) == 1 else base_w)
        return [
            (front[0] * span * factor, front[1] * span * factor)
            for factor in (0.35, 0.2, 0.0)
        ]
    if token in {"left", "right"}:
        span = float(base_h if abs(front[1]) == 1 else base_w)
        return [
            (front[0] * span * factor, front[1] * span * factor)
            for factor in OBJECT_LEVEL_SUPPORT_ALIGNMENT_FACTORS
        ]
    if token in {
        "head",
        "front",
        "head_center",
        "head_left",
        "front_left",
        "head_right",
        "front_right",
    }:
        span = float(base_w if abs(side[0]) == 1 else base_h)
        return [
            (side[0] * span * factor, side[1] * span * factor)
            for factor in OBJECT_LEVEL_SUPPORT_ALIGNMENT_FACTORS
        ]
    return [(0.0, 0.0)]


def _desired_support_front_vector(
    *,
    orientation: str,
    rect: tuple[int, int, int, int],
    base_row: Mapping[str, Any],
) -> tuple[float, float] | None:
    token = orientation.strip().lower()
    if token == "same_direction":
        return _front_tuple(base_row.get("front_world"))
    if token in {"face_base", "face_cluster"}:
        base_rect = base_row.get("rect")
        if not isinstance(base_rect, tuple) or len(base_rect) != 4:
            return None
        return _vector_between_rect_centers(rect, base_rect)
    return None


def _slot_offset(
    *,
    side_option: str,
    gap: int,
    base_w: int,
    base_h: int,
    obj_w: int,
    obj_h: int,
    front: tuple[int, int],
    side: tuple[int, int],
) -> tuple[float, float]:
    front_clear = (
        (base_h / 2.0) + (obj_h / 2.0) + gap
        if abs(front[1]) == 1
        else (base_w / 2.0) + (obj_w / 2.0) + gap
    )
    side_clear = (
        (base_w / 2.0) + (obj_w / 2.0) + gap
        if abs(side[0]) == 1
        else (base_h / 2.0) + (obj_h / 2.0) + gap
    )
    token = str(side_option).lower()
    fx, fy = front
    sx, sy = side
    if token in {"head", "front", "head_center"}:
        return (fx * front_clear, fy * front_clear)
    if token in {"left", "bedside_head_left"}:
        return (sx * -side_clear, sy * -side_clear)
    if token in {"right", "bedside_head_right"}:
        return (sx * side_clear, sy * side_clear)
    if token in {"head_left", "front_left"}:
        return (
            fx * front_clear - sx * side_clear * 0.85,
            fy * front_clear - sy * side_clear * 0.85,
        )
    if token in {"head_right", "front_right"}:
        return (
            fx * front_clear + sx * side_clear * 0.85,
            fy * front_clear + sy * side_clear * 0.85,
        )
    return (fx * front_clear, fy * front_clear)


def _front_vector_from_rotation(front_token: Any, rot: int) -> tuple[int, int]:
    token = str(front_token or "top").strip().lower()
    base = {"top": (0, 1), "bottom": (0, -1), "left": (-1, 0), "right": (1, 0)}.get(
        token, (0, 1)
    )
    turns = (int(rot) % 360) // 90
    x, y = base
    for _ in range(turns):
        x, y = y, -x
    return (int(x), int(y))


def _materialize_object_row(
    cluster_program: Mapping[str, Any],
    object_id: str,
    x: int,
    y: int,
    rot: int,
    *,
    relative_to: str | None = None,
) -> dict[str, Any]:
    spec = _object_spec(cluster_program, object_id) or {}
    dims = (
        spec.get("rep_dims_mm") if isinstance(spec.get("rep_dims_mm"), Mapping) else {}
    )
    length_mm = max(160, int(dims.get("L") or 0))
    width_mm = max(160, int(dims.get("W") or 0))
    w_mm, h_mm = _rotate_dims_for_rot(length_mm, width_mm, rot)
    rect = (int(x), int(y), int(x) + w_mm, int(y) + h_mm)
    front_token = spec.get("front") if isinstance(spec.get("front"), str) else "top"
    front_world = _front_vector_from_rotation(front_token, rot)
    front_side_world = {
        (0, 1): "top",
        (0, -1): "bottom",
        (-1, 0): "left",
        (1, 0): "right",
    }.get(front_world)
    return {
        "cluster_id": str(cluster_program.get("cluster_id") or ""),
        "object_id": object_id,
        "category": str(spec.get("category") or object_id),
        "x": int(x),
        "y": int(y),
        "rot": int(rot) % 360,
        "rotation_ccw": int(rot) % 360,
        "w": w_mm,
        "h": h_mm,
        "rect": rect,
        "bbox": {
            "min_x": rect[0],
            "min_y": rect[1],
            "max_x": rect[2],
            "max_y": rect[3],
        },
        "center": {
            "x": int(round((rect[0] + rect[2]) / 2.0)),
            "y": int(round((rect[1] + rect[3]) / 2.0)),
        },
        "front_token": front_token,
        "front_world": {"dx": front_world[0], "dy": front_world[1]},
        "front_side_world": front_side_world,
        "relative_to": relative_to,
        "role": str(spec.get("role") or ""),
        "priority": str(spec.get("priority") or ""),
        "requires_front_access": _object_requires_front_access(
            cluster_program, object_id
        ),
    }


def _object_requires_front_access(
    cluster_program: Mapping[str, Any],
    object_id: str,
) -> bool:
    object_program = (
        cluster_program.get("object_program")
        if isinstance(cluster_program.get("object_program"), Mapping)
        else {}
    )
    cluster_rules = (
        cluster_program.get("cluster_rules")
        if isinstance(cluster_program.get("cluster_rules"), Mapping)
        else {}
    )
    rows: list[Any] = []
    for source in (object_program, cluster_rules):
        raw_rows = (
            source.get("access_requirements") if isinstance(source, Mapping) else []
        )
        if isinstance(raw_rows, Sequence) and not isinstance(raw_rows, str):
            rows.extend(raw_rows)
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        candidate_id = str(
            row.get("id") or row.get("object_id") or row.get("target_id") or ""
        ).strip()
        if candidate_id != object_id:
            continue
        requirement_type = str(row.get("type") or row.get("mode") or "").strip().lower()
        if requirement_type not in {
            "",
            "access",
            "requires_access",
            "front_access",
            "front_clearance",
        }:
            continue
        if row.get("required") is False:
            continue
        return True
    return False


def _verify_object_level_solution(
    *,
    solution: Mapping[str, Any],
    world: Mapping[str, Any],
    room_model: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
) -> dict[str, Any]:
    placed_objects = [
        row for row in solution.get("placed_objects") or [] if isinstance(row, Mapping)
    ]
    geometry_valid = True
    hard_valid = True
    offending_clusters: list[str] = []
    protected_regions = (
        world.get("protected_regions")
        if isinstance(world.get("protected_regions"), Sequence)
        else []
    )
    forbidden_regions = (
        world.get("cluster_forbidden_regions")
        if isinstance(world.get("cluster_forbidden_regions"), Sequence)
        else []
    )
    protected_region_issues = _object_level_protected_region_issues(
        placed_objects=placed_objects,
        protected_regions=protected_regions,
    )
    forbidden_region_issues = _object_level_cluster_forbidden_region_issues(
        placed_objects=placed_objects,
        forbidden_regions=forbidden_regions,
    )
    blocking_protected_region_issues = [
        issue
        for issue in protected_region_issues
        if _protected_region_issue_is_blocking(issue)
    ]
    blocking_forbidden_region_issues = [
        issue
        for issue in forbidden_region_issues
        if _protected_region_issue_is_blocking(issue)
    ]
    protected_region_penalty = _object_level_protected_region_penalty(
        protected_region_issues
    )
    forbidden_region_penalty = _object_level_protected_region_penalty(
        forbidden_region_issues
    )
    if blocking_protected_region_issues:
        hard_valid = False
        offending_clusters.extend(
            str(issue.get("cluster_id") or "")
            for issue in blocking_protected_region_issues
        )
    if blocking_forbidden_region_issues:
        hard_valid = False
        offending_clusters.extend(
            str(issue.get("cluster_id") or "")
            for issue in blocking_forbidden_region_issues
        )
    for row in placed_objects:
        rect = row.get("rect")
        if not isinstance(rect, tuple) or len(rect) != 4:
            continue
        if not _rect_inside_room_footprint(rect, world):
            geometry_valid = False
            hard_valid = False
            offending_clusters.append(str(row.get("cluster_id") or ""))
    for index, left in enumerate(placed_objects):
        for right in placed_objects[index + 1 :]:
            if _rects_overlap(left["rect"], right["rect"]):
                geometry_valid = False
                hard_valid = False
                offending_clusters.extend(
                    [
                        str(left.get("cluster_id") or ""),
                        str(right.get("cluster_id") or ""),
                    ]
                )
    orientation_issues = _object_level_orientation_issues(
        placed_objects=placed_objects,
        world=world,
        relation_plan=relation_plan,
    )
    blocking_orientation_issues = _object_level_blocking_orientation_issues(
        orientation_issues=orientation_issues,
        placed_objects=placed_objects,
    )
    if blocking_orientation_issues:
        hard_valid = False
        offending_clusters.extend(
            str(issue.get("cluster_id") or "") for issue in blocking_orientation_issues
        )
    face_pair_issues = _object_level_face_pair_issues(
        placed_objects=placed_objects,
        relation_plan=relation_plan,
    )
    if face_pair_issues:
        hard_valid = False
        for item in face_pair_issues:
            offending_clusters.extend(
                [
                    str(item.get("a") or ""),
                    str(item.get("b") or ""),
                ]
            )
    functional_geometry_issues = _object_level_functional_geometry_issues(
        placed_objects=placed_objects,
        world=world,
        relation_plan=relation_plan,
    )
    blocking_functional_geometry_issues = [
        issue
        for issue in functional_geometry_issues
        if str(issue.get("violation_severity") or "").strip().lower() == "blocking"
    ]
    if blocking_functional_geometry_issues:
        hard_valid = False
        offending_clusters.extend(
            str(issue.get("cluster_id") or "")
            for issue in blocking_functional_geometry_issues
        )
    expected_clusters = list(world["clusters_by_id"].keys())
    present_clusters = sorted(
        {
            str(row.get("cluster_id") or "")
            for row in placed_objects
            if str(row.get("cluster_id") or "").strip()
        }
    )
    missing_clusters = [
        cluster_id
        for cluster_id in expected_clusters
        if cluster_id not in present_clusters
        and not _solution_dropped_solver_trial_cluster(
            solution=solution,
            world=world,
            cluster_id=cluster_id,
        )
    ]
    dropped_trial_clusters = [
        cluster_id
        for cluster_id in expected_clusters
        if _solution_dropped_solver_trial_cluster(
            solution=solution,
            world=world,
            cluster_id=cluster_id,
        )
    ]
    coverage_denominator = max(
        1,
        len(
            [
                cluster_id
                for cluster_id in expected_clusters
                if cluster_id not in set(dropped_trial_clusters)
            ]
        ),
    )
    complete = geometry_valid and not missing_clusters
    pair_ok = _object_level_primary_pair_ok(placed_objects, relation_plan) and not (
        face_pair_issues
    )
    critical_issue_count = sum(
        1
        for issue in protected_region_issues
        if str(issue.get("priority") or "").strip().lower() == "critical"
    )
    blocking_issue_count = (
        len(blocking_protected_region_issues)
        + len(blocking_forbidden_region_issues)
        + len(blocking_orientation_issues)
        + len(face_pair_issues)
        + len(blocking_functional_geometry_issues)
    )
    view_corridor_ok = not any(
        str(issue.get("reason") or "") == "view_corridor_blocked"
        for issue in blocking_functional_geometry_issues
    )
    front_access_ok = not any(
        str(issue.get("reason") or "") == "front_access_zone_blocked"
        for issue in blocking_functional_geometry_issues
    )
    gallery_eligible = (
        complete
        and hard_valid
        and critical_issue_count == 0
        and blocking_issue_count == 0
        and view_corridor_ok
        and front_access_ok
    )
    soft_issue_count = (
        len(orientation_issues)
        - len(blocking_orientation_issues)
        + len(protected_region_issues)
        - len(blocking_protected_region_issues)
        + len(forbidden_region_issues)
        - len(blocking_forbidden_region_issues)
        + len(functional_geometry_issues)
        - len(blocking_functional_geometry_issues)
    )
    soft_constraint_score = max(0, len(placed_objects) * 2 - soft_issue_count)
    quality_gate_reasons = _object_level_quality_gate_reasons(
        gallery_eligible=gallery_eligible,
        complete=complete,
        face_pair_issues=face_pair_issues,
        blocking_protected_region_issues=blocking_protected_region_issues,
        blocking_forbidden_region_issues=blocking_forbidden_region_issues,
        blocking_orientation_issues=blocking_orientation_issues,
        blocking_functional_geometry_issues=blocking_functional_geometry_issues,
    )
    return {
        "geometry_valid": geometry_valid,
        "hard_valid": hard_valid,
        "complete": complete,
        "gallery_eligible": gallery_eligible,
        "coverage_ratio": float(len(present_clusters) / coverage_denominator),
        "missing_cluster_ids": missing_clusters,
        "dropped_trial_cluster_ids": dropped_trial_clusters,
        "present_cluster_ids": present_clusters,
        "offending_clusters": sorted(
            {cluster_id for cluster_id in offending_clusters if cluster_id}
        ),
        "primary_pair_ok": pair_ok,
        "orientation_issues": orientation_issues,
        "blocking_orientation_issues": blocking_orientation_issues,
        "protected_region_issues": protected_region_issues,
        "blocking_protected_region_issues": blocking_protected_region_issues,
        "forbidden_region_issues": forbidden_region_issues,
        "blocking_forbidden_region_issues": blocking_forbidden_region_issues,
        "functional_geometry_issues": functional_geometry_issues,
        "blocking_functional_geometry_issues": blocking_functional_geometry_issues,
        "face_pair_issues": face_pair_issues,
        "critical_issue_count": critical_issue_count,
        "blocking_issue_count": blocking_issue_count,
        "view_corridor_ok": view_corridor_ok,
        "front_access_ok": front_access_ok,
        "protected_region_penalty": round(protected_region_penalty, 3),
        "forbidden_region_penalty": round(forbidden_region_penalty, 3),
        "soft_issue_count": soft_issue_count,
        "soft_constraint_score": soft_constraint_score,
        "quality_gate_reasons": quality_gate_reasons,
        "layout_score": int(
            1000
            + len(placed_objects) * 40
            + (200 if pair_ok else -120)
            + (600 if geometry_valid else -2000)
            + (200 if hard_valid else -500)
            + soft_constraint_score * 30
            - protected_region_penalty
            - forbidden_region_penalty
            - len(orientation_issues) * 30
            - len(functional_geometry_issues) * 90
            - blocking_issue_count * 1200
        ),
    }


def _solution_dropped_solver_trial_cluster(
    *,
    solution: Mapping[str, Any],
    world: Mapping[str, Any],
    cluster_id: str,
) -> bool:
    clusters_by_id = (
        world.get("clusters_by_id")
        if isinstance(world.get("clusters_by_id"), Mapping)
        else {}
    )
    cluster_program = clusters_by_id.get(cluster_id)
    if not isinstance(cluster_program, Mapping):
        return False
    if not _cluster_is_solver_trial_optional(cluster_program):
        return False
    dropped = solution.get("dropped_inventory_by_cluster")
    if not isinstance(dropped, Mapping):
        return False
    rows = dropped.get(cluster_id)
    return isinstance(rows, Sequence) and not isinstance(rows, str) and bool(rows)


def _object_level_protected_region_issues(
    *,
    placed_objects: Sequence[Mapping[str, Any]],
    protected_regions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for row in placed_objects:
        rect = row.get("rect")
        if not isinstance(rect, tuple) or len(rect) != 4:
            continue
        for protected in protected_regions:
            bbox = protected.get("bbox")
            if not isinstance(bbox, tuple) or len(bbox) != 4:
                continue
            applies_to = _string_sequence(protected.get("applies_to"))
            if (
                "support_clusters" in applies_to
                and "core_clusters" not in applies_to
                and str(row.get("anchor_kind") or "").strip().lower() == "center"
            ):
                continue
            ratio = _rect_overlap_ratio(rect, bbox)
            max_overlap = float(protected.get("max_overlap_ratio") or 0.0)
            if ratio <= max_overlap + 1e-9:
                continue
            overlap_area = _rect_overlap_area(rect, bbox)
            excess_overlap_ratio = max(0.0, ratio - max_overlap)
            priority = str(protected.get("priority") or "medium").strip().lower()
            enforcement = str(protected.get("enforcement") or "soft").strip().lower()
            violation_severity = (
                str(protected.get("violation_severity") or "advisory").strip().lower()
            )
            issues.append(
                {
                    "cluster_id": str(row.get("cluster_id") or ""),
                    "object_id": str(row.get("object_id") or ""),
                    "region_id": str(protected.get("region_id") or ""),
                    "zone_type": str(protected.get("zone_type") or ""),
                    "priority": priority,
                    "enforcement": enforcement,
                    "violation_severity": violation_severity,
                    "overlap_ratio": round(ratio, 3),
                    "max_overlap_ratio": round(max_overlap, 3),
                    "excess_overlap_ratio": round(excess_overlap_ratio, 3),
                    "overlap_area_mm2": int(round(overlap_area)),
                    "blocking": _protected_region_policy_is_blocking(
                        priority=priority,
                        enforcement=enforcement,
                        violation_severity=violation_severity,
                    ),
                }
            )
    return issues


def _object_level_cluster_forbidden_region_issues(
    *,
    placed_objects: Sequence[Mapping[str, Any]],
    forbidden_regions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for row in placed_objects:
        rect = row.get("rect")
        if not isinstance(rect, tuple) or len(rect) != 4:
            continue
        cluster_id = str(row.get("cluster_id") or "")
        for forbidden in forbidden_regions:
            if str(forbidden.get("cluster_id") or "") != cluster_id:
                continue
            bbox = forbidden.get("bbox")
            if not isinstance(bbox, tuple) or len(bbox) != 4:
                continue
            ratio = _rect_overlap_ratio(rect, bbox)
            max_overlap = float(forbidden.get("max_overlap_ratio") or 0.0)
            if ratio <= max_overlap + 1e-9:
                continue
            overlap_area = _rect_overlap_area(rect, bbox)
            excess_overlap_ratio = max(0.0, ratio - max_overlap)
            priority = str(forbidden.get("priority") or "high").strip().lower()
            enforcement = str(forbidden.get("enforcement") or "hard").strip().lower()
            violation_severity = (
                str(forbidden.get("violation_severity") or "blocking").strip().lower()
            )
            issues.append(
                {
                    "cluster_id": cluster_id,
                    "object_id": str(row.get("object_id") or ""),
                    "region_id": str(forbidden.get("region_id") or ""),
                    "zone_type": str(
                        forbidden.get("zone_type") or "cluster_forbidden_region"
                    ),
                    "priority": priority,
                    "enforcement": enforcement,
                    "violation_severity": violation_severity,
                    "overlap_ratio": round(ratio, 3),
                    "max_overlap_ratio": round(max_overlap, 3),
                    "excess_overlap_ratio": round(excess_overlap_ratio, 3),
                    "overlap_area_mm2": int(round(overlap_area)),
                    "blocking": _protected_region_policy_is_blocking(
                        priority=priority,
                        enforcement=enforcement,
                        violation_severity=violation_severity,
                    ),
                }
            )
    return issues


def _object_level_protected_region_issue_count(
    *,
    placed_objects: Sequence[Mapping[str, Any]],
    protected_regions: Sequence[Mapping[str, Any]],
) -> int:
    return len(
        _object_level_protected_region_issues(
            placed_objects=placed_objects,
            protected_regions=protected_regions,
        )
    )


def _protected_region_policy_is_blocking(
    *,
    priority: str,
    enforcement: str,
    violation_severity: str,
) -> bool:
    return (
        priority in OBJECT_LEVEL_BLOCKING_PROTECTED_PRIORITIES
        and enforcement in OBJECT_LEVEL_BLOCKING_PROTECTED_ENFORCEMENT
        and violation_severity in OBJECT_LEVEL_BLOCKING_PROTECTED_SEVERITIES
    )


def _protected_region_issue_is_blocking(issue: Mapping[str, Any]) -> bool:
    if bool(issue.get("blocking")):
        return True
    return _protected_region_policy_is_blocking(
        priority=str(issue.get("priority") or "").strip().lower(),
        enforcement=str(issue.get("enforcement") or "").strip().lower(),
        violation_severity=str(issue.get("violation_severity") or "").strip().lower(),
    )


def _protected_region_priority_weight(priority: Any) -> float:
    token = str(priority or "medium").strip().lower()
    return {
        "critical": 8.0,
        "high": 5.0,
        "medium": 2.5,
        "low": 1.0,
    }.get(token, 2.0)


def _object_level_protected_region_penalty(
    protected_region_issues: Sequence[Mapping[str, Any]],
) -> float:
    penalty = 0.0
    for issue in protected_region_issues:
        excess = float(issue.get("excess_overlap_ratio") or 0.0)
        affected_area = float(issue.get("overlap_area_mm2") or 0.0)
        priority_weight = _protected_region_priority_weight(issue.get("priority"))
        penalty += priority_weight * excess * (affected_area / 800.0)
        if _protected_region_issue_is_blocking(issue):
            penalty += 900.0
    return penalty


def _object_level_blocking_orientation_issues(
    *,
    orientation_issues: Sequence[Mapping[str, Any]],
    placed_objects: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_key = {
        (str(row.get("cluster_id") or ""), str(row.get("object_id") or "")): row
        for row in placed_objects
        if isinstance(row, Mapping)
    }
    blocking: list[dict[str, Any]] = []
    for issue in orientation_issues:
        if str(issue.get("reason") or "").strip().lower() != "wall_contact_inward":
            continue
        row = rows_by_key.get(
            (str(issue.get("cluster_id") or ""), str(issue.get("object_id") or ""))
        )
        if row is not None and _row_requires_front_access_zone(row):
            blocking.append(dict(issue))
    return blocking


def _object_level_quality_gate_reasons(
    *,
    gallery_eligible: bool,
    complete: bool,
    face_pair_issues: Sequence[Mapping[str, Any]],
    blocking_protected_region_issues: Sequence[Mapping[str, Any]],
    blocking_forbidden_region_issues: Sequence[Mapping[str, Any]],
    blocking_orientation_issues: Sequence[Mapping[str, Any]],
    blocking_functional_geometry_issues: Sequence[Mapping[str, Any]],
) -> list[str]:
    if gallery_eligible:
        return []
    reasons: list[str] = []
    if face_pair_issues:
        reasons.append("required_face_contract_failed")
    if blocking_protected_region_issues:
        reasons.append("blocking_protected_region_overlap")
    if blocking_forbidden_region_issues:
        reasons.append("blocking_forbidden_region_overlap")
    if blocking_orientation_issues:
        reasons.append("front_inward_contract_failed")
    for issue in blocking_functional_geometry_issues:
        reason = str(issue.get("reason") or "").strip()
        if reason:
            reasons.append(reason)
    if not complete:
        reasons.append("layout_incomplete")
    if not reasons:
        reasons.append("soft_constraints_relaxed_for_ranking")
    return sorted(set(reasons))


def _object_level_functional_geometry_issues(
    *,
    placed_objects: Sequence[Mapping[str, Any]],
    world: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    room_bbox = world["room_bbox"]
    rows = [row for row in placed_objects if isinstance(row, Mapping)]
    issues: list[dict[str, Any]] = []

    for base in rows:
        access_zone = _front_access_zone_for_row(base, room_bbox)
        if access_zone is None:
            continue
        for blocker in rows:
            if blocker is base:
                continue
            blocker_rect = blocker.get("rect")
            if not isinstance(blocker_rect, tuple) or len(blocker_rect) != 4:
                continue
            ratio = _rect_overlap_ratio(blocker_rect, access_zone)
            if ratio <= 0.02:
                continue
            if _object_allowed_in_front_access_zone(blocker):
                continue
            issues.append(
                {
                    "cluster_id": str(blocker.get("cluster_id") or ""),
                    "object_id": str(blocker.get("object_id") or ""),
                    "blocked_cluster_id": str(base.get("cluster_id") or ""),
                    "blocked_object_id": str(base.get("object_id") or ""),
                    "reason": "front_access_zone_blocked",
                    "violation_severity": "blocking",
                    "overlap_ratio": round(ratio, 3),
                    "overlap_area_mm2": int(
                        round(_rect_overlap_area(blocker_rect, access_zone))
                    ),
                }
            )

    corridor = _primary_view_corridor_rect(
        placed_objects=rows,
        room_bbox=room_bbox,
        relation_plan=relation_plan,
    )
    if corridor is not None:
        corridor_rect, allowed_clusters = corridor
        for row in rows:
            cluster_id = str(row.get("cluster_id") or "")
            if cluster_id in allowed_clusters and _is_cluster_anchor_row(row):
                continue
            if _object_allowed_in_view_corridor(row):
                continue
            rect = row.get("rect")
            if not isinstance(rect, tuple) or len(rect) != 4:
                continue
            ratio = _rect_overlap_ratio(rect, corridor_rect)
            if ratio <= 0.04:
                continue
            issues.append(
                {
                    "cluster_id": cluster_id,
                    "object_id": str(row.get("object_id") or ""),
                    "reason": "view_corridor_blocked",
                    "violation_severity": "blocking",
                    "overlap_ratio": round(ratio, 3),
                    "overlap_area_mm2": int(
                        round(_rect_overlap_area(rect, corridor_rect))
                    ),
                }
            )
    return issues


def _front_access_zone_for_row(
    row: Mapping[str, Any],
    room_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    if not _row_requires_front_access_zone(row):
        return None
    rect = row.get("rect")
    if not isinstance(rect, tuple) or len(rect) != 4:
        return None
    front = _dominant_cardinal_front(row.get("front_world"))
    if front is None:
        return None
    depth = OBJECT_LEVEL_FRONT_ACCESS_DEPTH_MM
    if front == (1, 0):
        zone = (rect[2], rect[1], rect[2] + depth, rect[3])
    elif front == (-1, 0):
        zone = (rect[0] - depth, rect[1], rect[0], rect[3])
    elif front == (0, 1):
        zone = (rect[0], rect[3], rect[2], rect[3] + depth)
    else:
        zone = (rect[0], rect[1] - depth, rect[2], rect[1])
    return _clip_rect_to_room(zone, room_bbox)


def _row_requires_front_access_zone(row: Mapping[str, Any]) -> bool:
    if bool(row.get("requires_front_access")):
        return True
    if not _is_cluster_anchor_row(row):
        return False
    tokens = _object_semantic_tokens(row)
    return bool(
        tokens
        & {
            "sofa",
            "loveseat",
            "settee",
            "sectional",
            "armchair",
            "chair",
            "recliner",
            "tv",
            "media",
            "console",
            "bookshelf",
            "bookcase",
            "shelving",
            "shelf",
            "cabinet",
        }
    )


def _object_allowed_in_front_access_zone(row: Mapping[str, Any]) -> bool:
    tokens = _object_semantic_tokens(row)
    return bool(tokens & {"coffee", "coffee_table", "ottoman", "rug"})


def _object_allowed_in_view_corridor(row: Mapping[str, Any]) -> bool:
    tokens = _object_semantic_tokens(row)
    return bool(tokens & {"coffee", "coffee_table", "ottoman", "rug"})


def _object_semantic_tokens(row: Mapping[str, Any]) -> set[str]:
    raw_values = [
        row.get("object_id"),
        row.get("category"),
        row.get("object_type"),
        row.get("role"),
    ]
    tokens: set[str] = set()
    for value in raw_values:
        normalized = str(value or "").strip().lower().replace("-", "_")
        if not normalized:
            continue
        tokens.add(normalized)
        tokens.update(part for part in normalized.split("_") if part)
    return tokens


def _dominant_cardinal_front(value: Any) -> tuple[int, int] | None:
    front = _front_tuple(value)
    if front is None:
        return None
    fx, fy = front
    if abs(fx) >= abs(fy) and abs(fx) > 1e-6:
        return (1 if fx > 0 else -1, 0)
    if abs(fy) > 1e-6:
        return (0, 1 if fy > 0 else -1)
    return None


def _primary_view_corridor_rect(
    *,
    placed_objects: Sequence[Mapping[str, Any]],
    room_bbox: tuple[int, int, int, int],
    relation_plan: Mapping[str, Any] | None,
) -> tuple[tuple[int, int, int, int], set[str]] | None:
    primary = _extract_layout_primary_cluster_id(
        dict(relation_plan) if isinstance(relation_plan, dict) else None
    )
    secondary = _extract_layout_secondary_cluster_id(
        dict(relation_plan) if isinstance(relation_plan, dict) else None
    )
    if not primary or not secondary:
        return None
    by_cluster: dict[str, list[Mapping[str, Any]]] = {}
    for row in placed_objects:
        by_cluster.setdefault(str(row.get("cluster_id") or ""), []).append(row)
    left = _dominant_row_for_cluster(by_cluster.get(primary, []))
    right = _dominant_row_for_cluster(by_cluster.get(secondary, []))
    if left is None or right is None:
        return None
    left_rect = left.get("rect")
    right_rect = right.get("rect")
    if not (
        isinstance(left_rect, tuple)
        and len(left_rect) == 4
        and isinstance(right_rect, tuple)
        and len(right_rect) == 4
    ):
        return None
    lx = (left_rect[0] + left_rect[2]) / 2.0
    ly = (left_rect[1] + left_rect[3]) / 2.0
    rx = (right_rect[0] + right_rect[2]) / 2.0
    ry = (right_rect[1] + right_rect[3]) / 2.0
    half_width = OBJECT_LEVEL_VIEW_CORRIDOR_WIDTH_MM // 2
    if abs(rx - lx) >= abs(ry - ly):
        cy = int(round((ly + ry) / 2.0))
        rect = (
            int(round(min(lx, rx))),
            cy - half_width,
            int(round(max(lx, rx))),
            cy + half_width,
        )
    else:
        cx = int(round((lx + rx) / 2.0))
        rect = (
            cx - half_width,
            int(round(min(ly, ry))),
            cx + half_width,
            int(round(max(ly, ry))),
        )
    clipped = _clip_rect_to_room(rect, room_bbox)
    if clipped is None:
        return None
    return clipped, {primary, secondary}


def _object_level_orientation_issues(
    *,
    placed_objects: Sequence[Mapping[str, Any]],
    world: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    by_cluster: dict[str, list[Mapping[str, Any]]] = {}
    for row in placed_objects:
        by_cluster.setdefault(str(row.get("cluster_id") or ""), []).append(row)
    support_edges = _support_edges_by_cluster(world)
    room_bbox = world["room_bbox"]
    issues: list[dict[str, Any]] = []
    for row in placed_objects:
        cluster_id = str(row.get("cluster_id") or "")
        object_id = str(row.get("object_id") or "")
        rect = row.get("rect")
        if not isinstance(rect, tuple) or len(rect) != 4:
            continue
        required_checks: list[tuple[str, tuple[float, float]]] = []
        edge = support_edges.get((cluster_id, object_id))
        orientation = str(edge.get("orientation") or "").strip().lower() if edge else ""
        if orientation in {"same_direction", "face_base", "face_cluster"}:
            base_id = str(edge.get("relative_to") or "")
            base_row = next(
                (
                    candidate
                    for candidate in by_cluster.get(cluster_id, [])
                    if str(candidate.get("object_id") or "") == base_id
                ),
                None,
            )
            if base_row is not None:
                desired_support_front = _desired_support_front_vector(
                    orientation=orientation,
                    rect=rect,
                    base_row=base_row,
                )
                if desired_support_front is not None:
                    required_checks.append((orientation, desired_support_front))
        intents = _object_level_orientation_intents(
            cluster_id=cluster_id,
            object_id=object_id,
            relation_plan=relation_plan,
            include_cluster=_is_cluster_anchor_row(row),
        )
        wall_contact_front = _wall_contact_inward_front(
            rect=rect,
            room_bbox=room_bbox,
        )
        if wall_contact_front is not None:
            required_checks.append(("wall_contact_inward", wall_contact_front))
        if "face_cluster" in intents:
            target_cluster_id = _object_level_target_cluster_id(
                cluster_id=cluster_id,
                object_id=object_id,
                relation_plan=relation_plan,
            )
            target_row = _dominant_row_for_cluster(
                by_cluster.get(target_cluster_id or "", [])
            )
            if target_row is not None:
                target_rect = target_row.get("rect")
                if isinstance(target_rect, tuple) and len(target_rect) == 4:
                    desired_cluster_front = _vector_between_rect_centers(
                        rect, target_rect
                    )
                    if desired_cluster_front is not None:
                        required_checks.append(("face_cluster", desired_cluster_front))
        if not required_checks:
            continue
        front = _front_tuple(row.get("front_world"))
        if front is None:
            continue
        for reason, desired_front in required_checks:
            dot = _dot_normalized(front, desired_front)
            if dot < OBJECT_LEVEL_FRONT_ALIGNMENT_MIN_DOT:
                issues.append(
                    {
                        "cluster_id": cluster_id,
                        "object_id": object_id,
                        "reason": reason,
                        "dot": round(dot, 3),
                        "front_world": {"dx": front[0], "dy": front[1]},
                        "desired_front_world": {
                            "dx": round(desired_front[0], 3),
                            "dy": round(desired_front[1], 3),
                        },
                    }
                )
    return issues


def _object_level_required_face_pairs(
    relation_plan: Mapping[str, Any] | None,
) -> set[tuple[str, str]]:
    if not isinstance(relation_plan, Mapping):
        return set()

    relations_by_pair: dict[tuple[str, str], set[str]] = {}
    for row in relation_plan.get("cluster_directional_relations") or []:
        if not isinstance(row, Mapping):
            continue
        pair = _normalized_cluster_pair(
            str(row.get("a") or ""),
            str(row.get("b") or ""),
        )
        if pair is None:
            continue
        relation = str(row.get("relation") or "").strip().lower()
        if relation:
            relations_by_pair.setdefault(pair, set()).add(relation)

    required = {
        pair
        for pair, relations in relations_by_pair.items()
        if "face_each_other" in relations and len(relations) == 1
    }

    layout_intent = relation_plan.get("layout_intent_profile")
    if isinstance(layout_intent, Mapping):
        focus_mode = str(layout_intent.get("focus_mode") or "").strip().lower()
        primary = str(layout_intent.get("primary_cluster_id") or "").strip()
        secondary = str(layout_intent.get("secondary_cluster_id") or "").strip()
        pair = _normalized_cluster_pair(primary, secondary)
        if focus_mode == "viewing" and pair is not None:
            alternatives = relations_by_pair.get(pair, {"face_each_other"})
            if alternatives <= {"face_each_other"}:
                required.add(pair)
        if (
            focus_mode in {"viewing", "mixed"}
            and pair is not None
            and _relation_plan_requires_legible_primary_pair(
                relation_plan=relation_plan,
                primary=primary,
                secondary=secondary,
            )
        ):
            required.add(pair)

    requirements = relation_plan.get("concept_readiness_requirements")
    if isinstance(requirements, Mapping):
        required_contracts = _normalized_text_set(
            requirements.get("required_pair_contracts")
        )
        if required_contracts & {
            "face_each_other",
            "buffered_support",
            "legible_primary_pair",
            "focal_face_axis",
        }:
            primary = _extract_layout_primary_cluster_id(
                dict(relation_plan) if isinstance(relation_plan, dict) else None
            )
            secondary = _extract_layout_secondary_cluster_id(
                dict(relation_plan) if isinstance(relation_plan, dict) else None
            )
            pair = _normalized_cluster_pair(primary or "", secondary or "")
            if pair is not None:
                alternatives = relations_by_pair.get(pair, {"face_each_other"})
                if alternatives <= {"face_each_other"}:
                    required.add(pair)
    return required


def _relation_plan_requires_legible_primary_pair(
    *,
    relation_plan: Mapping[str, Any],
    primary: str,
    secondary: str,
) -> bool:
    if not primary or not secondary:
        return False
    if _pair_has_mutual_face_cluster_orientation(
        relation_plan=relation_plan,
        primary=primary,
        secondary=secondary,
    ):
        return True
    if _pair_has_high_priority_alignment_preference(
        relation_plan=relation_plan,
        primary=primary,
        secondary=secondary,
    ):
        return True
    requirements = relation_plan.get("concept_readiness_requirements")
    if isinstance(requirements, Mapping):
        required_contracts = _normalized_text_set(
            requirements.get("required_pair_contracts")
        )
        if required_contracts & {
            "buffered_support",
            "legible_primary_pair",
            "focal_face_axis",
        }:
            return True
    return _pair_tokens_include_media(primary, secondary)


def _pair_has_mutual_face_cluster_orientation(
    *,
    relation_plan: Mapping[str, Any],
    primary: str,
    secondary: str,
) -> bool:
    forward = False
    backward = False
    for row in relation_plan.get("cluster_orientations") or []:
        if not isinstance(row, Mapping):
            continue
        cluster_id = str(row.get("cluster_id") or "").strip()
        target_cluster_id = str(row.get("target_cluster_id") or "").strip()
        intents = _normalized_text_set(row.get("intents"))
        priority = str(row.get("priority") or "").strip().lower()
        if "face_cluster" not in intents or priority not in {"high", "critical"}:
            continue
        if cluster_id == primary and target_cluster_id == secondary:
            forward = True
        if cluster_id == secondary and target_cluster_id == primary:
            backward = True
    return forward and backward


def _pair_has_high_priority_alignment_preference(
    *,
    relation_plan: Mapping[str, Any],
    primary: str,
    secondary: str,
) -> bool:
    concept = _concept_from_relation_plan(relation_plan)
    macro_constraints = (
        concept.get("macro_constraints") if isinstance(concept, Mapping) else {}
    )
    alignment_rows = []
    if isinstance(macro_constraints, Mapping):
        rows = macro_constraints.get("cluster_alignment_preferences")
        if isinstance(rows, Sequence) and not isinstance(rows, str):
            alignment_rows.extend(rows)
    rows = relation_plan.get("cluster_alignment_preferences")
    if isinstance(rows, Sequence) and not isinstance(rows, str):
        alignment_rows.extend(rows)
    pair = _normalized_cluster_pair(primary, secondary)
    if pair is None:
        return False
    for row in alignment_rows:
        if not isinstance(row, Mapping):
            continue
        candidate_pair = _normalized_cluster_pair(
            str(row.get("a") or ""),
            str(row.get("b") or ""),
        )
        if candidate_pair != pair:
            continue
        preference = str(row.get("preference") or "").strip().lower()
        priority = str(row.get("priority") or "").strip().lower()
        if priority in {"high", "critical"} and preference in {
            "legible_primary_pair",
            "focal_face_axis",
        }:
            return True
    return False


def _pair_tokens_include_media(primary: str, secondary: str) -> bool:
    tokens = {
        part
        for value in (primary, secondary)
        for part in str(value or "").strip().lower().replace("-", "_").split("_")
        if part
    }
    return bool(tokens & {"media", "tv", "television", "console"})


def _object_level_face_pair_issues(
    placed_objects: Sequence[Mapping[str, Any]],
    relation_plan: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    required_pairs = _object_level_required_face_pairs(relation_plan)
    if not required_pairs:
        return []
    rows_by_cluster: dict[str, list[Mapping[str, Any]]] = {}
    for row in placed_objects:
        rows_by_cluster.setdefault(str(row.get("cluster_id") or ""), []).append(row)

    issues: list[dict[str, Any]] = []
    for left_cluster_id, right_cluster_id in sorted(required_pairs):
        left = _dominant_row_for_cluster(rows_by_cluster.get(left_cluster_id, []))
        right = _dominant_row_for_cluster(rows_by_cluster.get(right_cluster_id, []))
        if left is None or right is None:
            issues.append(
                {
                    "a": left_cluster_id,
                    "b": right_cluster_id,
                    "reason": "required_face_pair_missing_anchor",
                }
            )
            continue
        left_rect = left.get("rect")
        right_rect = right.get("rect")
        if not (
            isinstance(left_rect, tuple)
            and len(left_rect) == 4
            and isinstance(right_rect, tuple)
            and len(right_rect) == 4
        ):
            continue
        left_desired = _vector_between_rect_centers(left_rect, right_rect)
        right_desired = _vector_between_rect_centers(right_rect, left_rect)
        left_front = _front_tuple(left.get("front_world"))
        right_front = _front_tuple(right.get("front_world"))
        left_ok = _front_matches_required_direction(
            front_world=left_front,
            desired_front=left_desired,
            min_dot=OBJECT_LEVEL_REQUIRED_FACE_PAIR_MIN_DOT,
        )
        right_ok = _front_matches_required_direction(
            front_world=right_front,
            desired_front=right_desired,
            min_dot=OBJECT_LEVEL_REQUIRED_FACE_PAIR_MIN_DOT,
        )
        if left_ok and right_ok:
            continue
        issues.append(
            {
                "a": left_cluster_id,
                "b": right_cluster_id,
                "reason": "required_face_each_other_failed",
                "left_dot": None
                if left_front is None or left_desired is None
                else round(_dot_normalized(left_front, left_desired), 3),
                "right_dot": None
                if right_front is None or right_desired is None
                else round(_dot_normalized(right_front, right_desired), 3),
            }
        )
    return issues


def _support_edges_by_cluster(
    world: Mapping[str, Any],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    out: dict[tuple[str, str], Mapping[str, Any]] = {}
    clusters_by_id = (
        world.get("clusters_by_id")
        if isinstance(world.get("clusters_by_id"), Mapping)
        else {}
    )
    for cluster_id, cluster_program in clusters_by_id.items():
        if not isinstance(cluster_program, Mapping):
            continue
        object_program = (
            cluster_program.get("object_program")
            if isinstance(cluster_program.get("object_program"), Mapping)
            else {}
        )
        for edge in object_program.get("support_edges") or []:
            if not isinstance(edge, Mapping):
                continue
            object_id = str(edge.get("object_id") or "").strip()
            if object_id:
                out[(str(cluster_id), object_id)] = edge
    return out


def _is_cluster_anchor_row(row: Mapping[str, Any]) -> bool:
    return str(row.get("priority") or "") in {
        "anchor",
        "dominant_anchor",
    } or not row.get("relative_to")


def _dominant_row_for_cluster(
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    return next(
        (
            row
            for row in rows
            if str(row.get("priority") or "") in {"anchor", "dominant_anchor"}
        ),
        rows[0] if rows else None,
    )


def _front_tuple(value: Any) -> tuple[float, float] | None:
    if isinstance(value, Mapping):
        try:
            return (float(value.get("dx")), float(value.get("dy")))
        except (TypeError, ValueError):
            return None
    if isinstance(value, Sequence) and not isinstance(value, str) and len(value) == 2:
        try:
            return (float(value[0]), float(value[1]))
        except (TypeError, ValueError):
            return None
    return None


def _vector_between_rect_centers(
    source: tuple[int, int, int, int],
    target: tuple[int, int, int, int],
) -> tuple[float, float] | None:
    sx = (source[0] + source[2]) / 2.0
    sy = (source[1] + source[3]) / 2.0
    tx = (target[0] + target[2]) / 2.0
    ty = (target[1] + target[3]) / 2.0
    dx, dy = tx - sx, ty - sy
    norm = math.hypot(dx, dy)
    if norm <= 1e-9:
        return None
    return (dx / norm, dy / norm)


def _object_level_primary_pair_ok(
    placed_objects: Sequence[Mapping[str, Any]], relation_plan: Mapping[str, Any] | None
) -> bool:
    primary = _extract_layout_primary_cluster_id(
        dict(relation_plan) if isinstance(relation_plan, dict) else None
    )
    secondary = _extract_layout_secondary_cluster_id(
        dict(relation_plan) if isinstance(relation_plan, dict) else None
    )
    if not primary or not secondary:
        return True
    left = next(
        (
            row
            for row in placed_objects
            if row.get("cluster_id") == primary
            and str(row.get("priority") or "") in {"anchor", "dominant_anchor"}
        ),
        None,
    )
    if left is None:
        left = next(
            (row for row in placed_objects if row.get("cluster_id") == primary), None
        )
    right = next(
        (
            row
            for row in placed_objects
            if row.get("cluster_id") == secondary
            and str(row.get("priority") or "") in {"anchor", "dominant_anchor"}
        ),
        None,
    )
    if right is None:
        right = next(
            (row for row in placed_objects if row.get("cluster_id") == secondary), None
        )
    if left is None or right is None:
        return False
    lc = left.get("center") if isinstance(left.get("center"), Mapping) else {}
    rc = right.get("center") if isinstance(right.get("center"), Mapping) else {}
    try:
        lx, ly = float(lc.get("x")), float(lc.get("y"))
        rx, ry = float(rc.get("x")), float(rc.get("y"))
    except (TypeError, ValueError):
        return False
    dx, dy = rx - lx, ry - ly
    norm = math.hypot(dx, dy)
    if norm <= 1e-6:
        return False
    direction = (dx / norm, dy / norm)
    left_front = (
        left.get("front_world") if isinstance(left.get("front_world"), Mapping) else {}
    )
    right_front = (
        right.get("front_world")
        if isinstance(right.get("front_world"), Mapping)
        else {}
    )
    try:
        left_vec = (float(left_front.get("dx")), float(left_front.get("dy")))
        right_vec = (float(right_front.get("dx")), float(right_front.get("dy")))
    except (TypeError, ValueError):
        return False
    pair = _normalized_cluster_pair(primary, secondary)
    min_dot = (
        OBJECT_LEVEL_REQUIRED_FACE_PAIR_MIN_DOT
        if pair is not None and pair in _object_level_required_face_pairs(relation_plan)
        else OBJECT_LEVEL_FRONT_ALIGNMENT_MIN_DOT
    )
    return _front_matches_required_direction(
        front_world=left_vec,
        desired_front=direction,
        min_dot=min_dot,
    ) and _front_matches_required_direction(
        front_world=right_vec,
        desired_front=(-direction[0], -direction[1]),
        min_dot=min_dot,
    )


def _object_solution_score(
    solution: Mapping[str, Any], relation_plan: Mapping[str, Any] | None
) -> float:
    verify = (
        solution.get("verify") if isinstance(solution.get("verify"), Mapping) else {}
    )
    score = float(solution.get("anchor_score") or 0.0)
    score += float(verify.get("layout_score") or 0.0)
    if verify.get("gallery_eligible"):
        score += 5000.0
    elif verify.get("hard_valid"):
        score += 1000.0
    score -= 250.0 * len(
        [
            item
            for rows in (solution.get("dropped_inventory_by_cluster") or {}).values()
            for item in rows
        ]
    )
    score += float(solution.get("support_score") or 0.0)
    score -= 80.0 * float(solution.get("protected_region_issue_count") or 0.0)
    score -= float(verify.get("protected_region_penalty") or 0.0)
    score -= float(verify.get("forbidden_region_penalty") or 0.0)
    score -= 1200.0 * float(verify.get("blocking_issue_count") or 0.0)
    return score


def _rank_object_level_solution_pool(
    solutions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    ranked = sorted(
        (dict(item) for item in solutions if isinstance(item, Mapping)),
        key=lambda item: (
            bool((item.get("verify") or {}).get("hard_valid")),
            int((item.get("verify") or {}).get("critical_issue_count") or 0) == 0,
            int((item.get("verify") or {}).get("blocking_issue_count") or 0) == 0,
            bool((item.get("verify") or {}).get("view_corridor_ok")),
            bool((item.get("verify") or {}).get("front_access_ok")),
            bool((item.get("verify") or {}).get("complete")),
            bool((item.get("verify") or {}).get("primary_pair_ok")),
            int((item.get("verify") or {}).get("soft_constraint_score") or 0),
            -int((item.get("verify") or {}).get("soft_issue_count") or 0),
            float(item.get("score") or 0.0),
        ),
        reverse=True,
    )
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in ranked:
        signature = str(item.get("signature") or "")
        if signature and signature in seen:
            continue
        if signature:
            seen.add(signature)
        out.append(item)
    return out


def _object_level_solution_signature(solution: Mapping[str, Any]) -> str:
    rows = [
        row
        for row in (solution.get("placed_objects") or [])
        if isinstance(row, Mapping)
    ]
    tokens = []
    for row in sorted(
        rows,
        key=lambda item: (
            str(item.get("cluster_id") or ""),
            str(item.get("object_id") or ""),
        ),
    ):
        bbox = row.get("bbox") if isinstance(row.get("bbox"), Mapping) else {}
        tokens.append(
            "|".join(
                [
                    str(row.get("cluster_id") or ""),
                    str(row.get("object_id") or ""),
                    str(int(row.get("rotation_ccw") or 0)),
                    str(int(bbox.get("min_x") or 0)),
                    str(int(bbox.get("min_y") or 0)),
                    str(int(bbox.get("max_x") or 0)),
                    str(int(bbox.get("max_y") or 0)),
                ]
            )
        )
    return "\n".join(tokens)


def _build_object_level_solution_payload(
    *,
    solution: Mapping[str, Any],
    world: Mapping[str, Any],
    room_model: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
    solution_index: int,
) -> dict[str, Any]:
    verify = (
        solution.get("verify") if isinstance(solution.get("verify"), Mapping) else {}
    )
    absolute_layout = _build_absolute_layout_from_object_solution(
        solution=solution,
        world=world,
        room_model=room_model,
        relation_plan=relation_plan,
    )
    return {
        "solution_id": f"object_sol_{solution_index:02d}",
        "absolute_layout": absolute_layout,
        "layout_score": int(verify.get("layout_score") or 0),
        "hard_valid": bool(verify.get("hard_valid")),
        "geometry_valid": bool(verify.get("geometry_valid")),
        "complete": bool(verify.get("complete")),
        "gallery_eligible": bool(verify.get("gallery_eligible")),
        "coverage_ratio": float(verify.get("coverage_ratio") or 0.0),
        "missing_cluster_ids": list(verify.get("missing_cluster_ids") or []),
        "verify_summary": deepcopy(dict(verify)),
        "dropped_inventory_by_cluster": deepcopy(
            solution.get("dropped_inventory_by_cluster") or {}
        ),
        "state_signature": str(solution.get("signature") or ""),
        "notes": [
            "Object-level solver kept this geometry-valid arrangement for ranked constraint search.",
            f"soft_issue_count={int(verify.get('soft_issue_count') or 0)}",
            f"blocking_issue_count={int(verify.get('blocking_issue_count') or 0)}",
        ],
    }


def _build_absolute_layout_from_object_solution(
    *,
    solution: Mapping[str, Any],
    world: Mapping[str, Any],
    room_model: Mapping[str, Any],
    relation_plan: Mapping[str, Any] | None,
) -> dict[str, Any]:
    verify = (
        solution.get("verify") if isinstance(solution.get("verify"), Mapping) else {}
    )
    placed_objects = [
        deepcopy(row)
        for row in (solution.get("placed_objects") or [])
        if isinstance(row, Mapping)
    ]
    clusters: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in placed_objects:
        grouped.setdefault(str(row.get("cluster_id") or ""), []).append(row)
    for cluster_id, rows in grouped.items():
        anchor_row = next(
            (
                row
                for row in rows
                if str(row.get("priority") or "") in {"anchor", "dominant_anchor"}
            ),
            rows[0] if rows else {},
        )
        bbox = _union_bboxes(
            [tuple(row["rect"]) for row in rows if isinstance(row.get("rect"), tuple)]
        )
        clusters.append(
            {
                "cluster_id": cluster_id,
                "object_ids": [str(row.get("object_id") or "") for row in rows],
                "dominant_anchor_id": str(anchor_row.get("object_id") or ""),
                "bbox": {
                    "min_x": bbox[0],
                    "min_y": bbox[1],
                    "max_x": bbox[2],
                    "max_y": bbox[3],
                }
                if bbox is not None
                else {},
            }
        )
    coverage = {
        "expected_cluster_ids": list(world["clusters_by_id"].keys()),
        "present_cluster_ids": list(verify.get("present_cluster_ids") or []),
        "missing_cluster_ids": list(verify.get("missing_cluster_ids") or []),
    }
    notes = [
        "Absolute layout was emitted directly by the object-level solver.",
        "Cluster local composition and phase-2 repair were bypassed in this flow.",
    ]
    geometry_repair = (
        solution.get("geometry_repair")
        if isinstance(solution.get("geometry_repair"), Mapping)
        else {}
    )
    if geometry_repair and geometry_repair.get("status") == "repaired":
        notes.append(
            "Backend geometry guardrail repaired polygon bounds and overlap before styling."
        )
    return {
        "status": "OK" if verify.get("hard_valid") else "PARTIAL",
        "layout_kind": "object_level_anchor_first",
        "room_id": _room_id(room_model),
        "room_type": _room_type(room_model),
        "room": deepcopy(
            room_model.get("room")
            if isinstance(room_model.get("room"), Mapping)
            else {}
        ),
        "openings": deepcopy(
            room_model.get("openings")
            if isinstance(room_model.get("openings"), Mapping)
            else {}
        ),
        "affordance_map": deepcopy(
            room_model.get("affordance_map")
            if isinstance(room_model.get("affordance_map"), Mapping)
            else {}
        ),
        "topology": deepcopy(
            room_model.get("topology")
            if isinstance(room_model.get("topology"), Mapping)
            else {}
        ),
        "objects": deepcopy(placed_objects),
        "placements": deepcopy(placed_objects),
        "object_placements": deepcopy(placed_objects),
        "clusters": clusters,
        "cluster_instances": deepcopy(clusters),
        "coverage": coverage,
        "complete": bool(verify.get("complete")),
        "gallery_eligible": bool(verify.get("gallery_eligible")),
        "coverage_ratio": float(verify.get("coverage_ratio") or 0.0),
        "missing_cluster_ids": list(verify.get("missing_cluster_ids") or []),
        "notes": notes,
        "solver_debug": {
            "selected_concept_id": _concept_id_from_relation_plan(relation_plan),
            "dropped_inventory_by_cluster": deepcopy(
                solution.get("dropped_inventory_by_cluster") or {}
            ),
            "verify_summary": deepcopy(verify),
            "geometry_repair": deepcopy(solution.get("geometry_repair") or {}),
        },
    }


def _rect_tuple(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 4:
        return None
    try:
        rect = tuple(int(round(float(item))) for item in value)
    except (TypeError, ValueError):
        return None
    if rect[2] <= rect[0] or rect[3] <= rect[1]:
        return None
    return (rect[0], rect[1], rect[2], rect[3])


def _rect_center(rect: tuple[int, int, int, int]) -> tuple[float, float]:
    return ((rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0)


def _rect_inside_room_footprint(
    rect: tuple[int, int, int, int], world: Mapping[str, Any]
) -> bool:
    if not _rect_inside_room(rect, world["room_bbox"]):
        return False
    polygon = world.get("room_polygon")
    if (
        not isinstance(polygon, Sequence)
        or isinstance(polygon, str)
        or len(polygon) < 3
    ):
        return True
    points: list[tuple[float, float]] = []
    for item in polygon:
        if not isinstance(item, Sequence) or isinstance(item, str) or len(item) != 2:
            return True
        try:
            points.append((float(item[0]), float(item[1])))
        except (TypeError, ValueError):
            return True
    return _rect_inside_polygon(rect, points)


def _rect_inside_polygon(
    rect: tuple[int, int, int, int], polygon: Sequence[tuple[float, float]]
) -> bool:
    if len(polygon) < 3:
        return True
    probe_points = _rect_probe_points(rect)
    if any(not _point_in_or_on_polygon(point, polygon) for point in probe_points):
        return False
    rect_edges = _rect_edges(rect)
    polygon_edges = [
        (polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    ]
    return not any(
        _segments_strictly_cross(rect_edge[0], rect_edge[1], poly_edge[0], poly_edge[1])
        for rect_edge in rect_edges
        for poly_edge in polygon_edges
    )


def _rect_probe_points(
    rect: tuple[int, int, int, int],
) -> tuple[tuple[float, float], ...]:
    x1, y1, x2, y2 = rect
    cx, cy = _rect_center(rect)
    return (
        (x1, y1),
        (x2, y1),
        (x2, y2),
        (x1, y2),
        (cx, cy),
        (cx, y1),
        (x2, cy),
        (cx, y2),
        (x1, cy),
    )


def _rect_edges(
    rect: tuple[int, int, int, int],
) -> tuple[
    tuple[tuple[float, float], tuple[float, float]],
    tuple[tuple[float, float], tuple[float, float]],
    tuple[tuple[float, float], tuple[float, float]],
    tuple[tuple[float, float], tuple[float, float]],
]:
    x1, y1, x2, y2 = rect
    return (
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    )


def _point_in_or_on_polygon(
    point: tuple[float, float], polygon: Sequence[tuple[float, float]]
) -> bool:
    if any(
        _point_on_segment(point, polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    ):
        return True
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, current in enumerate(polygon):
        previous = polygon[j]
        if (current[1] > y) != (previous[1] > y):
            denom = previous[1] - current[1]
            if abs(denom) > 1e-12:
                x_intersect = (previous[0] - current[0]) * (
                    y - current[1]
                ) / denom + current[0]
                if x < x_intersect:
                    inside = not inside
        j = i
    return inside


def _point_on_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    eps: float = 1e-6,
) -> bool:
    cross = (point[1] - start[1]) * (end[0] - start[0]) - (point[0] - start[0]) * (
        end[1] - start[1]
    )
    if abs(cross) > eps:
        return False
    dot_product = (point[0] - start[0]) * (end[0] - start[0]) + (
        point[1] - start[1]
    ) * (end[1] - start[1])
    if dot_product < -eps:
        return False
    squared_len = (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
    return dot_product <= squared_len + eps


def _segments_strictly_cross(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
    *,
    eps: float = 1e-6,
) -> bool:
    if (
        _point_on_segment(a, c, d, eps=eps)
        or _point_on_segment(b, c, d, eps=eps)
        or _point_on_segment(c, a, b, eps=eps)
        or _point_on_segment(d, a, b, eps=eps)
    ):
        return False
    ab_c = _orient(a, b, c)
    ab_d = _orient(a, b, d)
    cd_a = _orient(c, d, a)
    cd_b = _orient(c, d, b)
    return (ab_c * ab_d < -eps) and (cd_a * cd_b < -eps)


def _orient(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]
) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _snap_to_grid(value: int, grid_mm: int) -> int:
    grid = max(1, int(grid_mm))
    return int(round(float(value) / float(grid))) * grid


def _snap_origin_inside_room(
    *,
    x: int,
    y: int,
    w_mm: int,
    h_mm: int,
    room_bbox: tuple[int, int, int, int],
    grid_mm: int,
) -> tuple[int, int]:
    return (
        _snap_value_inside_span(
            value=x,
            lower=room_bbox[0],
            upper=room_bbox[2] - w_mm,
            grid_mm=grid_mm,
        ),
        _snap_value_inside_span(
            value=y,
            lower=room_bbox[1],
            upper=room_bbox[3] - h_mm,
            grid_mm=grid_mm,
        ),
    )


def _snap_value_inside_span(
    *,
    value: int,
    lower: int,
    upper: int,
    grid_mm: int,
) -> int:
    grid = max(1, int(grid_mm))
    if upper < lower:
        return _snap_to_grid(value, grid)
    snapped = _snap_to_grid(value, grid)
    if lower <= snapped <= upper:
        return snapped
    lower_grid = int(math.ceil(float(lower) / float(grid))) * grid
    upper_grid = int(math.floor(float(upper) / float(grid))) * grid
    candidates = [
        candidate
        for candidate in (
            int(math.floor(float(value) / float(grid))) * grid,
            int(math.ceil(float(value) / float(grid))) * grid,
            lower_grid,
            upper_grid,
        )
        if lower <= candidate <= upper
    ]
    if candidates:
        return min(candidates, key=lambda candidate: abs(candidate - value))
    return max(lower, min(upper, value))


def _rect_inside_room(
    rect: tuple[int, int, int, int], room_bbox: tuple[int, int, int, int]
) -> bool:
    return (
        rect[0] >= room_bbox[0]
        and rect[1] >= room_bbox[1]
        and rect[2] <= room_bbox[2]
        and rect[3] <= room_bbox[3]
    )


def _rects_overlap(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> bool:
    return (
        left[0] < right[2]
        and left[2] > right[0]
        and left[1] < right[3]
        and left[3] > right[1]
    )


def _rect_overlap_ratio(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> float:
    ix1 = max(left[0], right[0])
    iy1 = max(left[1], right[1])
    ix2 = min(left[2], right[2])
    iy2 = min(left[3], right[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = float((ix2 - ix1) * (iy2 - iy1))
    area = float(max(1, (left[2] - left[0]) * (left[3] - left[1])))
    return intersection / area


def _rect_overlap_area(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> float:
    ix1 = max(left[0], right[0])
    iy1 = max(left[1], right[1])
    ix2 = min(left[2], right[2])
    iy2 = min(left[3], right[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return float((ix2 - ix1) * (iy2 - iy1))


def _clip_rect_to_room(
    rect: tuple[int, int, int, int],
    room_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    clipped = (
        max(rect[0], room_bbox[0]),
        max(rect[1], room_bbox[1]),
        min(rect[2], room_bbox[2]),
        min(rect[3], room_bbox[3]),
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return clipped


def _default_tools_path() -> str:
    here = Path(__file__).resolve()
    return str(here.parents[2] / "cluster_placer" / "tools.py")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Finite-pose room layout CP-SAT solver"
    )
    parser.add_argument("--room", required=True, help="Path to room_model JSON")
    parser.add_argument(
        "--clusters", required=True, help="Path to clusters_outlines JSON"
    )
    parser.add_argument("--relations", required=True, help="Path to relation_plan JSON")
    parser.add_argument(
        "--cluster-constraints",
        default=None,
        help="Optional path to cluster-level constraints JSON",
    )
    parser.add_argument(
        "--grid-mm",
        type=int,
        default=GLOBAL_LAYOUT_GRID_MM,
        help="Placement grid in mm",
    )
    parser.add_argument(
        "--tools", default=_default_tools_path(), help="Path to tools.py"
    )
    parser.add_argument("--max-variants", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=40)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--time-limit-s", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default="solver_output.json")
    args = parser.parse_args(argv)

    room_model = _load_json(args.room)
    clusters = _load_json(args.clusters)
    relations = _load_json(args.relations)
    cluster_constraints = (
        _load_json(args.cluster_constraints) if args.cluster_constraints else None
    )

    result = solve_layout(
        room_model=room_model,
        clusters_outlines=clusters,
        relation_plan=relations,
        cluster_constraints=cluster_constraints,
        grid_mm=args.grid_mm,
        tools_path=args.tools,
        max_variants_per_cluster=args.max_variants,
        initial_candidates_per_cluster=args.max_candidates,
        max_rounds=args.rounds,
        time_limit_s=args.time_limit_s,
        num_workers=args.workers,
    )
    _dump_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
