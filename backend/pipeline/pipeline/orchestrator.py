from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.cluster_composer import (
    ClusterComposer,
    _build_cluster_output_from_placements,
)
from agent.cluster_forge import ClusterForge
from agent.cluster_placer import build_phase2_payload
from agent.phase2_controller import Phase2Controller, build_phase2_preview_candidate
from agent.cluster_relation_planner import ClusterRelationPlanner
from agent.room_interpreter import RoomInterpreter
from agent.seed_concept_generator import solver_plan_from_concept
from agent.solver.solver import MacroClusterSolver
from agent.stylist import Stylist, stylist_model_name_for_variant
from agent.tier_count_director import TierCountDirector
from agent.request_contract import (
    canonical_object_type,
    missing_functional_contract_types,
    missing_non_functional_contract_items,
    request_contract_from_payload,
)
from cluster_composer.merge import merge_cluster_outputs
from cluster_composer.outline import compute_cluster_outline
from layout.grid_policy import GLOBAL_LAYOUT_GRID_MM
from layout.room_profiles.registry import all_profile_non_functional_layout_specs
from pipeline.ablation_modes import (
    AblationMode,
    ablation_metadata,
    apply_tier_output_for_mode,
    resolve_ablation_mode,
    skip_accessory_refill,
    style_policy_for_mode,
    target_final_count_for_mode,
    uses_neutral_style_policy,
)
from pipeline.compose_feedback import (
    apply_compose_backoff,
    apply_variant_diversification,
    evaluate_composed_cluster_feasibility,
    evaluate_solver_cluster_feasibility,
)
from pipeline.dropped_inventory import (
    collect_dropped_inventory,
    collect_seed_omitted_inventory,
    controlled_accessory_refill,
    dropped_inventory_payload,
    merge_dropped_inventory,
)
from pipeline.layout_eligibility import annotate_layout_coverage
from pipeline.layout_variants import (
    build_final_gallery_selection_summary,
    select_distinct_final_gallery_candidates,
)
from pipeline.manual_placements import (
    append_manual_placements_guidance,
    apply_manual_placements_to_tier_output,
    augment_cluster_forge_with_manual_placements,
    merge_manual_placements_into_absolute_layout,
    merge_manual_placements_into_styled_output,
    prepare_input_payload_for_manual_placements,
)
from stylist.style_policy import decor_refill_policy, extract_style_policy

logger = logging.getLogger(__name__)
_PRIMARY_SEED_CACHE_VERSION = "v1"
_PRIMARY_SEED_LAYOUT_CACHE: dict[str, dict[str, Any]] = {}
_LAYOUT_SPEED_MODE_ENV = "TKNT_LAYOUT_SPEED_MODE"
_REQUEST_NON_FUNCTIONAL_LAYOUT_SPECS: dict[str, dict[str, Any]] = {
    "rug": {
        "width_ratio": 0.56,
        "height_ratio": 0.44,
        "min_w": 900,
        "min_h": 700,
        "max_w": 2400,
        "max_h": 2800,
        "collision_layer": "floor_underlay",
        "place_on": {"target_instance_id": "floor", "method": "floor"},
    },
    "ceiling_lamp": {
        "width": 450,
        "height": 450,
        "collision_layer": "ceiling",
        "place_on": {"target_instance_id": "ceiling", "method": "hang_on"},
    },
}
_REQUEST_NON_FUNCTIONAL_LAYOUT_SPECS.update(all_profile_non_functional_layout_specs())

try:
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
except Exception:  # pragma: no cover - import guard only
    Polygon = None  # type: ignore[assignment]
    unary_union = None  # type: ignore[assignment]


def _layout_speed_mode() -> str:
    return str(os.getenv(_LAYOUT_SPEED_MODE_ENV) or "").strip().lower()


def _is_fast_layout_mode() -> bool:
    return _layout_speed_mode() in {"fast", "speed", "speed-first", "speed_first"}


def _compose_attempt_limit(default_attempts: int) -> int:
    if _is_fast_layout_mode():
        return max(1, min(int(default_attempts), 2))
    return max(1, int(default_attempts))


def _solver_attempt_limit(default_attempts: int) -> int:
    if _is_fast_layout_mode():
        return max(1, min(int(default_attempts), 3))
    return max(1, int(default_attempts))


def _seed_diversification_attempt_limit(default_attempts: int) -> int:
    if _is_fast_layout_mode():
        return max(1, min(int(default_attempts), 8))
    return max(1, int(default_attempts))


def _build_runtime_solver() -> MacroClusterSolver:
    base_solver = MacroClusterSolver()
    if not _is_fast_layout_mode():
        return base_solver
    return MacroClusterSolver(
        tools_path=base_solver.tools_path,
        max_variants_per_cluster=max(
            3, min(int(base_solver.max_variants_per_cluster), 5)
        ),
        initial_candidates_per_cluster=max(
            16, min(int(base_solver.initial_candidates_per_cluster), 24)
        ),
        max_rounds=max(1, min(int(base_solver.max_rounds), 2)),
        time_limit_s=max(6.0, min(float(base_solver.time_limit_s), 10.0)),
        num_workers=max(1, int(base_solver.num_workers)),
    )


@dataclass(frozen=True)
class CasePaths:
    case_id: str
    root: Path

    @property
    def room_interpreter(self) -> Path:
        return self.root / "01_room_interpreter.json"

    @property
    def initial_intents(self) -> Path:
        return self.root / "01b_initial_intents.json"

    @property
    def cluster_forge(self) -> Path:
        return self.root / "02_cluster_forge.json"

    @property
    def tier_count(self) -> Path:
        return self.root / "03_tier_count.json"

    @property
    def cluster_merged(self) -> Path:
        return self.root / "04_cluster_merged.json"

    @property
    def clusters_dir(self) -> Path:
        return self.root / "clusters"

    @property
    def module_outputs_dir(self) -> Path:
        return self.root / "module_outputs"

    @property
    def module_io_manifest(self) -> Path:
        return self.root / "module_io_manifest.json"

    def module_output(self, module_name: str) -> Path:
        return self.module_outputs_dir / f"{module_name}.json"

    def module_cluster_composer(self, cluster_id: str) -> Path:
        return self.module_outputs_dir / "cluster_composer" / f"{cluster_id}.json"

    def module_cluster_outline(self, cluster_id: str) -> Path:
        return self.module_outputs_dir / "cluster_outline" / f"{cluster_id}.json"

    def cluster_composer(self, cluster_id: str) -> Path:
        return self.clusters_dir / f"cluster_composer_{cluster_id}.json"

    def cluster_outline(self, cluster_id: str) -> Path:
        return self.clusters_dir / f"cluster_outline_{cluster_id}.json"

    @property
    def cluster_outlines_all(self) -> Path:
        return self.root / "05_cluster_outlines.json"

    @property
    def cluster_relation_plan(self) -> Path:
        return self.root / "05b_cluster_relation_plan.json"

    @property
    def seed_relation_plans(self) -> Path:
        return self.root / "05c_seed_relation_plans.json"

    @property
    def cluster_placer(self) -> Path:
        return self.root / "06_cluster_placer.json"

    @property
    def cluster_solver(self) -> Path:
        return self.root / "06a_cluster_solver.json"

    @property
    def solver_dropped_inventory(self) -> Path:
        return self.root / "06b_solver_dropped_inventory.json"

    @property
    def absolute_layout(self) -> Path:
        return self.root / "07_absolute_layout.json"

    @property
    def accessory_refill(self) -> Path:
        return self.root / "07b_accessory_refill.json"

    @property
    def stylist(self) -> Path:
        return self.root / "08_stylist.json"

    @property
    def layout_variants(self) -> Path:
        return self.root / "09_layout_variants.json"

    @property
    def status(self) -> Path:
        return self.root / "status.json"

    @property
    def meta(self) -> Path:
        return self.root / "case_meta.json"


@dataclass(frozen=True)
class OrchestratorModuleSpec:
    name: str
    implementation: str
    inputs: tuple[str, ...]
    output: str
    output_artifact: str
    legacy_artifact: str | None = None


ORCHESTRATOR_MODULE_SPECS: tuple[OrchestratorModuleSpec, ...] = (
    OrchestratorModuleSpec(
        name="room_interpreter",
        implementation="agent.room_interpreter.RoomInterpreter.generate",
        inputs=("prepared_input_payload", "description", "special_notes"),
        output="room model with normalized geometry, openings, room metadata, notes, and conflicts",
        output_artifact="module_outputs/room_interpreter.json",
        legacy_artifact="01_room_interpreter.json",
    ),
    OrchestratorModuleSpec(
        name="stylist_style_policy",
        implementation="agent.stylist.Stylist.compile_style_policy",
        inputs=("room_type", "planning_guidance_text", "room_output"),
        output="style policy used by semantic planning and final styling",
        output_artifact="module_outputs/stylist_style_policy.json",
    ),
    OrchestratorModuleSpec(
        name="cluster_forge",
        implementation="agent.cluster_forge.ClusterForge.generate",
        inputs=(
            "room_type",
            "planning_guidance_text",
            "room_output",
            "inventory_catalog",
            "style_policy",
        ),
        output="semantic layout program and cluster definitions with anchor-first policies",
        output_artifact="module_outputs/cluster_forge.json",
        legacy_artifact="02_cluster_forge.json",
    ),
    OrchestratorModuleSpec(
        name="request_contract",
        implementation="agent.request_contract.request_contract_from_payload",
        inputs=("planning_guidance_text", "cluster_output"),
        output="request-aware object contract inferred before tier count",
        output_artifact="module_outputs/request_contract.json",
    ),
    OrchestratorModuleSpec(
        name="tier_count_director",
        implementation="agent.tier_count_director.TierCountDirector.generate",
        inputs=(
            "planning_guidance_text",
            "room_output",
            "prepared_input_payload",
            "cluster_output",
        ),
        output="inventory tier/count decisions per cluster",
        output_artifact="module_outputs/tier_count_director.json",
        legacy_artifact="03_tier_count.json",
    ),
    OrchestratorModuleSpec(
        name="cluster_output_merger",
        implementation="cluster_composer.merge.merge_cluster_outputs",
        inputs=("cluster_output", "tier_output"),
        output="solver-ready merged object program with filtered anchor/support graph",
        output_artifact="module_outputs/cluster_output_merger.json",
        legacy_artifact="04_cluster_merged.json",
    ),
    OrchestratorModuleSpec(
        name="cluster_relation_planner_bundle",
        implementation="agent.cluster_relation_planner.ClusterRelationPlanner.generate_bundle",
        inputs=("room_output", "merged_object_program", "target_count"),
        output="bundle of deterministic anchor-first macro concepts",
        output_artifact="module_outputs/cluster_relation_planner_bundle.json",
        legacy_artifact="05c_seed_relation_plans.json",
    ),
    OrchestratorModuleSpec(
        name="cluster_relation_plan",
        implementation="agent.seed_concept_generator.solver_plan_from_concept",
        inputs=("macro_concept", "room_output", "room_type"),
        output="canonical relation plan consumed by the object-level solver",
        output_artifact="module_outputs/cluster_relation_plan.json",
        legacy_artifact="05b_cluster_relation_plan.json",
    ),
    OrchestratorModuleSpec(
        name="object_level_solver",
        implementation="agent.solver.solver.MacroClusterSolver.generate_object_layout",
        inputs=(
            "room_output",
            "merged_object_program",
            "relation_plan",
            "cluster_output",
        ),
        output="absolute object-level layout solved directly from anchors and support constraints",
        output_artifact="module_outputs/object_level_solver.json",
        legacy_artifact="06a_cluster_solver.json",
    ),
    OrchestratorModuleSpec(
        name="controlled_accessory_refill",
        implementation="pipeline.dropped_inventory.controlled_accessory_refill",
        inputs=(
            "room_output",
            "absolute_layout",
            "cluster_output",
            "dropped_inventory_by_cluster",
            "refill_policy",
        ),
        output="optional accessory refill summary and updated absolute layout when eligible",
        output_artifact="module_outputs/controlled_accessory_refill.json",
        legacy_artifact="07b_accessory_refill.json",
    ),
    OrchestratorModuleSpec(
        name="absolute_layout",
        implementation="agent.solver.solver.MacroClusterSolver.generate_object_layout:absolute_layout",
        inputs=("object_level_solver output", "manual_placements"),
        output="canonical absolute layout consumed by Stylist",
        output_artifact="module_outputs/absolute_layout.json",
        legacy_artifact="07_absolute_layout.json",
    ),
    OrchestratorModuleSpec(
        name="stylist",
        implementation="agent.stylist.Stylist.generate_style_plan/apply_style_plan",
        inputs=("absolute_layout", "stylist_user_context", "tenant_id"),
        output="styled result with selected inventory/assets/material/style metadata",
        output_artifact="module_outputs/stylist.json",
        legacy_artifact="08_stylist.json",
    ),
    OrchestratorModuleSpec(
        name="layout_variants",
        implementation="pipeline.layout_variants.select_distinct_final_gallery_candidates",
        inputs=("final styled candidates", "target_final_count"),
        output="final selectable gallery variants",
        output_artifact="module_outputs/layout_variants.json",
        legacy_artifact="09_layout_variants.json",
    ),
)


def _build_primary_guidance_text(
    *,
    room_output: dict[str, Any],
    input_payload: dict[str, Any],
    description: str | None,
    special_notes: str | None,
) -> str:
    notes = room_output.get("notes")
    if isinstance(notes, list):
        cleaned_notes = [
            note.strip() for note in notes if isinstance(note, str) and note.strip()
        ]
        if cleaned_notes:
            return "\n".join(f"- {note}" for note in cleaned_notes)

    legacy_parts: list[str] = []
    for value in (description, special_notes):
        if isinstance(value, str) and value.strip():
            legacy_parts.append(value.strip())

    user_input = input_payload.get("user_input")
    if isinstance(user_input, dict):
        for key in (
            "description",
            "special_description",
            "special_notes",
            "notes",
            "feng_shui",
            "style",
        ):
            value = user_input.get(key)
            if not isinstance(value, str):
                continue
            text = value.strip()
            if text and text not in legacy_parts:
                legacy_parts.append(text)

    return "\n".join(legacy_parts)


_SPEED_SPLIT_TAG_BY_FAMILY = {
    "seating": "living",
    "media": "media",
    "storage": "storage",
    "work": "work",
    "sleep": "sleep",
    "dining": "dining",
    "decor": "misc",
    "misc": "misc",
}


def _speed_split_cluster_output(
    *,
    room_output: dict[str, Any],
    cluster_output: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    _ = room_output
    return cluster_output, []


def _log_cluster_forge_output(
    cluster_output: dict[str, Any],
    *,
    context: str,
) -> None:
    clusters = cluster_output.get("clusters")
    if not isinstance(clusters, list):
        logger.info("Cluster forge %s produced no clusters.", context)
        return

    member_to_clusters: dict[str, list[str]] = {}
    logger.info("Cluster forge %s produced %s cluster(s).", context, len(clusters))
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        cluster_id = str(cluster.get("cluster_id") or "").strip() or "unknown_cluster"
        tag = str(cluster.get("tag") or "").strip() or "unknown"
        members = [
            member.strip()
            for member in (cluster.get("members") or [])
            if isinstance(member, str) and member.strip()
        ]
        anchors = [
            anchor.strip()
            for anchor in (cluster.get("anchors") or [])
            if isinstance(anchor, str) and anchor.strip()
        ]
        logger.info(
            "Cluster forge %s cluster %s [%s]: members=%s anchors=%s",
            context,
            cluster_id,
            tag,
            members,
            anchors,
        )
        for member in members:
            member_to_clusters.setdefault(member, []).append(cluster_id)

    duplicates = {
        member: cluster_ids
        for member, cluster_ids in sorted(member_to_clusters.items())
        if len(cluster_ids) > 1
    }
    if duplicates:
        logger.warning(
            "Cluster forge %s duplicate members across clusters: %s",
            context,
            duplicates,
        )


def _speed_split_single_cluster(
    *,
    cluster: dict[str, Any],
    room_fill_ratio: float,
) -> tuple[list[dict[str, Any]], str] | None:
    cluster_id = str(cluster.get("cluster_id") or "").strip()
    if not cluster_id:
        return None

    members = [
        member
        for member in (cluster.get("members") or [])
        if isinstance(member, str) and member.strip()
    ]
    anchors = [
        anchor
        for anchor in (cluster.get("anchors") or [])
        if isinstance(anchor, str) and anchor.strip() and anchor in set(members)
    ]
    if len(members) < 6 and len(anchors) < 2:
        return None

    if not _speed_cluster_needs_split(
        cluster=cluster,
        members=members,
        anchors=anchors,
        room_fill_ratio=room_fill_ratio,
    ):
        return None

    core_members, support_members = _speed_split_member_groups(
        cluster=cluster,
        members=members,
        anchors=anchors,
    )
    if len(core_members) < 1 or len(support_members) < 1:
        return None

    core_cluster = _build_speed_split_cluster_payload(
        cluster=cluster,
        cluster_id=f"{cluster_id}__core",
        tag=_speed_split_tag_for_members(core_members),
        members=core_members,
    )
    support_cluster = _build_speed_split_cluster_payload(
        cluster=cluster,
        cluster_id=f"{cluster_id}__support",
        tag=_speed_split_tag_for_members(support_members),
        members=support_members,
    )
    note = (
        f"Fast mode split overloaded cluster {cluster_id} into "
        f"{core_cluster['cluster_id']} and {support_cluster['cluster_id']}."
    )
    return [core_cluster, support_cluster], note


def _speed_cluster_needs_split(
    *,
    cluster: dict[str, Any],
    members: list[str],
    anchors: list[str],
    room_fill_ratio: float,
) -> bool:
    families = {
        _speed_member_family(member, tag=str(cluster.get("tag") or ""))
        for member in members
    }
    pressure_score = sum(_speed_member_rank(member) for member in members)
    if len(anchors) >= 2:
        return True
    if len(families) >= 3 and len(members) >= 5:
        return True
    if room_fill_ratio < 0.94 and len(members) >= 5:
        return True
    return len(members) >= 7 or pressure_score >= 11


def _speed_split_member_groups(
    *,
    cluster: dict[str, Any],
    members: list[str],
    anchors: list[str],
) -> tuple[list[str], list[str]]:
    cluster_tag = str(cluster.get("tag") or "").strip().lower()
    by_family: dict[str, list[str]] = {}
    for member in members:
        family = _speed_member_family(member, tag=cluster_tag)
        by_family.setdefault(family, []).append(member)

    primary_family = _speed_primary_family_for_cluster(
        cluster_tag=cluster_tag,
        members=members,
        anchors=anchors,
    )
    core_members = list(by_family.get(primary_family) or [])
    if not core_members:
        ranked = sorted(members, key=_speed_member_sort_key)
        core_members = ranked[: min(3, len(ranked))]

    max_core_size = 4 if len(members) >= 7 else 3
    if len(core_members) > max_core_size:
        ranked_core = sorted(core_members, key=_speed_member_sort_key)
        kept = ranked_core[:max_core_size]
        overflow = [member for member in core_members if member not in set(kept)]
        core_members = kept
        support_members = [
            member for member in members if member not in set(core_members)
        ]
        support_members = overflow + [
            member for member in support_members if member not in overflow
        ]
    else:
        support_members = [
            member for member in members if member not in set(core_members)
        ]

    if len(support_members) < 2:
        ranked = sorted(members, key=_speed_member_sort_key)
        core_members = ranked[: min(max_core_size, max(1, len(members) - 2))]
        support_members = [
            member for member in ranked if member not in set(core_members)
        ]

    core_members = _preserve_member_order(members, core_members)
    support_members = _preserve_member_order(members, support_members)
    return core_members, support_members


def _build_speed_split_cluster_payload(
    *,
    cluster: dict[str, Any],
    cluster_id: str,
    tag: str,
    members: list[str],
) -> dict[str, Any]:
    kept = set(members)
    payload = deepcopy(cluster)
    payload["cluster_id"] = cluster_id
    payload["tag"] = tag
    payload["members"] = list(members)
    payload["anchors"] = _speed_split_anchor_members(cluster=cluster, members=members)
    payload["split_parent_cluster_id"] = str(cluster.get("cluster_id") or "")

    hard_constraints = cluster.get("hard_constraints")
    if isinstance(hard_constraints, list):
        payload["hard_constraints"] = [
            item
            for item in hard_constraints
            if isinstance(item, dict)
            and _constraint_subjects_local(item).issubset(kept)
        ]

    soft_constraints = cluster.get("soft_constraints")
    if isinstance(soft_constraints, list):
        payload["soft_constraints"] = [
            item
            for item in soft_constraints
            if isinstance(item, dict)
            and _constraint_subjects_local(item).issubset(kept)
        ]

    rules = cluster.get("cluster_rules")
    if isinstance(rules, dict):
        payload["cluster_rules"] = _filter_cluster_rules_local(rules, kept)
    return payload


def _speed_split_anchor_members(
    *,
    cluster: dict[str, Any],
    members: list[str],
) -> list[str]:
    kept = set(members)
    anchors = [
        anchor
        for anchor in (cluster.get("anchors") or [])
        if isinstance(anchor, str) and anchor in kept
    ]
    if anchors:
        return anchors[:1]
    ranked = sorted(members, key=_speed_member_sort_key)
    return ranked[:1]


def _speed_primary_family_for_cluster(
    *,
    cluster_tag: str,
    members: list[str],
    anchors: list[str],
) -> str:
    if cluster_tag in {"living", "media", "storage", "work", "sleep", "dining"}:
        preferred = {
            "living": "seating",
            "media": "media",
            "storage": "storage",
            "work": "work",
            "sleep": "sleep",
            "dining": "dining",
        }[cluster_tag]
        if any(
            _speed_member_family(member, tag=cluster_tag) == preferred
            for member in members
        ):
            return preferred

    for anchor in anchors:
        return _speed_member_family(anchor, tag=cluster_tag)
    ranked = sorted(members, key=_speed_member_sort_key)
    if ranked:
        return _speed_member_family(ranked[0], tag=cluster_tag)
    return "misc"


def _speed_split_tag_for_members(members: list[str]) -> str:
    if not members:
        return "misc"
    families: dict[str, int] = {}
    for member in members:
        family = _speed_member_family(member, tag="")
        families[family] = families.get(family, 0) + 1
    dominant_family = max(
        families.items(),
        key=lambda item: (item[1], _speed_member_rank(item[0])),
    )[0]
    return _SPEED_SPLIT_TAG_BY_FAMILY.get(dominant_family, "misc")


def _speed_member_family(member: str, *, tag: str) -> str:
    key = str(member or "").strip().lower()
    if any(
        token in key for token in ("tv_console", "media_shelf", "monitor", "projector")
    ):
        return "media"
    if any(
        token in key
        for token in (
            "sofa",
            "sectional",
            "armchair",
            "recliner",
            "ottoman",
            "coffee_table",
            "side_table",
            "bean_bag",
            "floor_lamp",
        )
    ):
        return "seating"
    if any(
        token in key
        for token in ("bookshelf", "storage_cabinet", "console_table", "cabinet")
    ):
        return "storage"
    if any(token in key for token in ("desk", "desk_chair", "office")):
        return "work"
    if any(token in key for token in ("bed", "nightstand", "dresser", "wardrobe")):
        return "sleep"
    if any(token in key for token in ("dining", "bar", "buffet")):
        return "dining"
    if any(token in key for token in ("art", "vase", "lamp", "plant")):
        return "decor"
    return _SPEED_SPLIT_TAG_BY_FAMILY.get(tag, "misc")


def _speed_member_rank(member: str) -> int:
    family = _speed_member_family(member, tag="")
    key = str(member or "").strip().lower()
    if family in {"seating", "media", "sleep"}:
        if any(token in key for token in ("sectional", "sofa", "tv_console", "bed")):
            return 4
        return 3
    if family in {"storage", "work", "dining"}:
        return 2
    if family == "decor":
        return 0
    return 1


def _speed_member_sort_key(member: str) -> tuple[int, str]:
    return (-_speed_member_rank(member), str(member))


def _preserve_member_order(
    original_members: list[str], scoped_members: list[str]
) -> list[str]:
    kept = set(scoped_members)
    return [member for member in original_members if member in kept]


def _constraint_subjects_local(constraint: dict[str, Any]) -> set[str]:
    subjects: set[str] = set()
    for key in ("a", "b", "id"):
        value = constraint.get(key)
        if isinstance(value, str) and value.strip():
            subjects.add(value)
    return subjects


def _filter_cluster_rules_local(
    rules: dict[str, Any],
    kept_ids: set[str],
) -> dict[str, Any]:
    out = deepcopy(rules)
    allowed_rotations = out.get("allowed_rotations")
    if isinstance(allowed_rotations, dict):
        out["allowed_rotations"] = {
            key: value for key, value in allowed_rotations.items() if key in kept_ids
        }

    facing = out.get("facing")
    if isinstance(facing, dict):
        out["facing"] = {key: value for key, value in facing.items() if key in kept_ids}

    access_requirements = out.get("access_requirements")
    if isinstance(access_requirements, list):
        out["access_requirements"] = [
            item
            for item in access_requirements
            if isinstance(item, dict)
            and isinstance(item.get("id"), str)
            and item.get("id") in kept_ids
        ]

    semantic_placements = out.get("semantic_placements")
    if isinstance(semantic_placements, list):
        out["semantic_placements"] = [
            item
            for item in semantic_placements
            if isinstance(item, dict)
            and isinstance(item.get("id"), str)
            and item.get("id") in kept_ids
            and (
                not isinstance(item.get("relative_to"), str)
                or item.get("relative_to") in kept_ids
            )
        ]
    return out


def _room_fill_ratio(room_output: dict[str, Any]) -> float:
    room = room_output.get("room")
    polygon = room.get("polygon_ccw") if isinstance(room, dict) else None
    if not isinstance(polygon, list) or len(polygon) < 3:
        return 1.0
    xs = [float(point.get("x", 0.0)) for point in polygon if isinstance(point, dict)]
    ys = [float(point.get("y", 0.0)) for point in polygon if isinstance(point, dict)]
    if not xs or not ys:
        return 1.0
    bbox_area = max(1.0, (max(xs) - min(xs)) * (max(ys) - min(ys)))
    area2 = 0.0
    points = [
        (float(point.get("x", 0.0)), float(point.get("y", 0.0)))
        for point in polygon
        if isinstance(point, dict)
    ]
    for idx, (x1, y1) in enumerate(points):
        x2, y2 = points[(idx + 1) % len(points)]
        area2 += (x1 * y2) - (x2 * y1)
    polygon_area = abs(area2) / 2.0
    return max(0.0, min(1.0, polygon_area / bbox_area))


def _append_initial_intent_guidance(
    base_text: str,
    intent: dict[str, Any] | None,
) -> str:
    if not isinstance(intent, dict):
        return base_text

    intent_payload = {
        "intent_id": intent.get("intent_id"),
        "label": intent.get("label"),
        "summary": intent.get("summary"),
        "focus_mode": intent.get("focus_mode"),
        "primary_tag": intent.get("primary_tag"),
        "secondary_tag": intent.get("secondary_tag"),
        "circulation_priority": intent.get("circulation_priority"),
        "center_open_preference": intent.get("center_open_preference"),
        "support_cluster_behavior": intent.get("support_cluster_behavior"),
        "distribution_mode": intent.get("distribution_mode"),
        "forge_guidance": intent.get("forge_guidance"),
        "composer_guidance": intent.get("composer_guidance"),
        "notes": intent.get("notes"),
    }
    parts = [base_text.strip()] if base_text.strip() else []
    parts.append(
        "INITIAL_LAYOUT_INTENT_JSON:\n"
        + json.dumps(intent_payload, ensure_ascii=True, indent=2)
    )
    return "\n\n".join(parts)


def _extract_primary_intent(
    initial_intents: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(initial_intents, dict):
        return None
    intents = initial_intents.get("intents")
    if not isinstance(intents, list):
        return None
    for item in intents:
        if isinstance(item, dict):
            return item
    return None


def run_case(
    *,
    input_payload: dict[str, Any],
    user_id: str,
    description: str | None = None,
    special_notes: str | None = None,
    cases_root: str | Path = "cases",
    case_id: str | None = None,
    ablation_mode: str | None = None,
) -> dict[str, Any]:
    resolved_ablation_mode = resolve_ablation_mode(ablation_mode)
    prepared_input_payload, manual_placements = (
        prepare_input_payload_for_manual_placements(input_payload)
    )
    case_id = case_id or _make_case_id(user_id)
    case_root = Path(cases_root) / case_id
    case_root.mkdir(parents=True, exist_ok=True)
    paths = CasePaths(case_id=case_id, root=case_root)
    paths.clusters_dir.mkdir(parents=True, exist_ok=True)
    paths.module_outputs_dir.mkdir(parents=True, exist_ok=True)
    (paths.module_outputs_dir / "cluster_composer").mkdir(parents=True, exist_ok=True)
    (paths.module_outputs_dir / "cluster_outline").mkdir(parents=True, exist_ok=True)

    _write_json(
        paths.meta,
        {
            "case_id": case_id,
            "user_id": user_id,
            "created_at_utc": _now_utc_iso(),
            **ablation_metadata(resolved_ablation_mode),
        },
    )
    _write_module_io_manifest(paths)
    _update_status(paths, "room_interpreter")

    room_interpreter = RoomInterpreter()
    room_output = room_interpreter.generate(
        prepared_input_payload,
        description=description,
        special_notes=special_notes,
    ).model_dump()
    room_output = _strip_raw_text(room_output)
    _write_module_json(
        paths, "room_interpreter", room_output, legacy_path=paths.room_interpreter
    )

    guidance_text = _build_primary_guidance_text(
        room_output=room_output,
        input_payload=prepared_input_payload,
        description=description,
        special_notes=special_notes,
    )
    planning_guidance_text = append_manual_placements_guidance(
        guidance_text, manual_placements
    )
    room_type = _extract_room_type(prepared_input_payload, room_output)

    cluster_forge = ClusterForge()
    tier_count = TierCountDirector()
    relation_planner = ClusterRelationPlanner()
    solver = _build_runtime_solver()
    stylist = Stylist()

    room_meta = room_output.get("meta") if isinstance(room_output, dict) else None
    stylist_tenant_id = _extract_tenant_id_from_payload(prepared_input_payload)
    stylist_user_context = {
        "user_id": user_id,
        "tenant_id": stylist_tenant_id,
        "style": room_meta.get("style") if isinstance(room_meta, dict) else "",
        "room_notes": room_output.get("notes", [])
        if isinstance(room_output, dict)
        else [],
        "guidance_text": guidance_text,
        "user_input": prepared_input_payload.get("user_input", {})
        if isinstance(prepared_input_payload, dict)
        else {},
    }

    _update_status(paths, "stylist_style_policy")
    compiled_style_policy = (
        {}
        if uses_neutral_style_policy(resolved_ablation_mode)
        else stylist.compile_style_policy(
            room_type=room_type,
            brief_text=planning_guidance_text,
            room_model_json=room_output,
        )
    )
    style_policy = style_policy_for_mode(
        mode=resolved_ablation_mode,
        room_type=room_type,
        current_policy=compiled_style_policy,
    )
    stylist_user_context["style_policy"] = style_policy
    _write_module_json(paths, "stylist_style_policy", style_policy)

    _update_status(paths, "cluster_forge")
    cluster_output = cluster_forge.generate(
        room_type,
        description=planning_guidance_text or None,
        special_notes=None,
        room_model_json=room_output,
        inventory_catalog=_extract_inventory_catalog(prepared_input_payload),
        style_policy_json=style_policy,
    ).model_dump()
    cluster_output = _strip_raw_text(cluster_output)
    cluster_output = augment_cluster_forge_with_manual_placements(
        cluster_output, manual_placements
    )
    cluster_output, speed_split_notes = _speed_split_cluster_output(
        room_output=room_output, cluster_output=cluster_output
    )
    if speed_split_notes:
        logger.info("Fast mode split clusters before tier count: %s", speed_split_notes)
    _log_cluster_forge_output(cluster_output, context="primary")
    _write_module_json(
        paths, "cluster_forge", cluster_output, legacy_path=paths.cluster_forge
    )
    request_contract = request_contract_from_payload(cluster_output)
    if request_contract:
        _write_module_json(paths, "request_contract", request_contract)

    _update_status(paths, "tier_count")
    tier_output = tier_count.generate(
        description=planning_guidance_text,
        special_notes="",
        room_model_json=room_output,
        user_intent_json=prepared_input_payload,
        clusters_json=cluster_output,
    )
    tier_output = _strip_raw_text(tier_output)
    tier_output = apply_manual_placements_to_tier_output(
        tier_output, manual_placements, clusters_json=cluster_output
    )
    tier_output = apply_tier_output_for_mode(
        mode=resolved_ablation_mode,
        tier_output=tier_output,
    )
    _write_module_json(
        paths, "tier_count_director", tier_output, legacy_path=paths.tier_count
    )

    _update_status(paths, "cluster_output_merger")
    merged_output = merge_cluster_outputs(cluster_output, tier_output)
    merged_output = _strip_raw_text(merged_output)
    _write_module_json(
        paths, "cluster_output_merger", merged_output, legacy_path=paths.cluster_merged
    )

    target_final_count = target_final_count_for_mode(resolved_ablation_mode, 5)
    _update_status(
        paths,
        "cluster_relation_planner",
        message=_variant_progress_message(
            "Planning layout concepts", 0, target_final_count
        ),
        progress_current=0,
        progress_total=target_final_count,
    )
    seed_relation_plans = relation_planner.generate_bundle(
        room_model_json=room_output,
        clusters_json=merged_output,
        target_count=target_final_count,
        description=planning_guidance_text or None,
    )
    seed_relation_plans = _strip_raw_text(seed_relation_plans)
    _write_module_json(
        paths,
        "cluster_relation_planner_bundle",
        seed_relation_plans,
        legacy_path=paths.seed_relation_plans,
    )
    _write_module_json(paths, "seed_concept_generator", seed_relation_plans)
    concepts = [
        item
        for item in (seed_relation_plans.get("concepts") or [])
        if isinstance(item, dict)
    ]
    if not concepts:
        _update_status(
            paths,
            "error",
            error="layout concept planner could not produce any viable intent plans",
        )
        return {
            "case_id": case_id,
            "case_dir": str(case_root),
            "final_output": None,
            "error": "layout concept planner could not produce any viable intent plans",
        }

    solved_bundles: list[dict[str, Any]] = []
    for concept_index, concept in enumerate(concepts[:target_final_count], start=1):
        relation_plan = solver_plan_from_concept(
            concept=concept,
            room_model_json=room_output,
            room_type=room_type,
        )
        relation_plan = _strip_raw_text(relation_plan)
        if concept_index == 1:
            _write_module_json(
                paths,
                "cluster_relation_plan",
                relation_plan,
                legacy_path=paths.cluster_relation_plan,
            )
            _write_module_json(paths, "seed_concept_relation_plan", relation_plan)
        _update_status(
            paths,
            "solver",
            message=f"Concept {concept_index}/{target_final_count}: solving object-level anchor-first layout.",
            progress_current=len(solved_bundles),
            progress_total=target_final_count,
        )
        concept_bundles = _solve_object_level_variant_bundle(
            concept=concept,
            relation_plan=relation_plan,
            room_output=room_output,
            cluster_output=cluster_output,
            tier_output=tier_output,
            merged_output=merged_output,
            solver=solver,
            manual_placements=manual_placements,
            variant_index=concept_index,
            ablation_mode=resolved_ablation_mode,
        )
        if not concept_bundles:
            continue
        for solution_offset, solved_bundle in enumerate(concept_bundles, start=1):
            solved_bundle = _refill_judged_variant_accessories(
                bundle=solved_bundle,
                room_output=room_output,
                variant_index=len(solved_bundles) + 1,
                ablation_mode=resolved_ablation_mode,
            )
            solved_bundles.append(solved_bundle)
            if concept_index == 1 and solution_offset == 1:
                _write_bundle_stage_artifacts(paths, solved_bundle)
        _update_status(
            paths,
            "solver",
            message=_variant_progress_message(
                "Solving layout concepts", len(solved_bundles), target_final_count
            ),
            progress_current=len(solved_bundles),
            progress_total=target_final_count,
        )

    if not solved_bundles:
        _update_status(
            paths,
            "error",
            error="object-level solver could not produce any feasible anchor-first layouts",
        )
        return {
            "case_id": case_id,
            "case_dir": str(case_root),
            "final_output": None,
            "error": "object-level solver could not produce any feasible anchor-first layouts",
        }

    styling_candidates = _select_final_styling_candidates(
        solved_bundles,
        max_variants=target_final_count,
    )
    styling_target_count = max(1, len(styling_candidates))

    _update_status(
        paths,
        "stylist",
        message=_variant_progress_message(
            "Styling final layouts", 0, styling_target_count
        ),
        progress_current=0,
        progress_total=styling_target_count,
    )
    final_candidates: list[dict[str, Any]] = []
    for variant_index, solved_bundle in enumerate(styling_candidates, start=1):
        finalized_bundle = _style_judged_variant_bundle(
            bundle=solved_bundle,
            stylist=stylist,
            stylist_user_context_json=_style_variant_user_context(
                stylist_user_context, variant_index=variant_index
            ),
            stylist_tenant_id=stylist_tenant_id,
            manual_placements=manual_placements,
            variant_index=variant_index,
        )
        if finalized_bundle is None:
            continue
        final_candidates.append(finalized_bundle)
        _update_status(
            paths,
            "stylist",
            message=_variant_progress_message(
                "Styling final layouts",
                len(final_candidates),
                styling_target_count,
            ),
            progress_current=len(final_candidates),
            progress_total=styling_target_count,
        )

    if not final_candidates:
        _update_status(
            paths, "error", error="stylist could not finalize any anchor-first layouts"
        )
        return {
            "case_id": case_id,
            "case_dir": str(case_root),
            "final_output": None,
            "error": "stylist could not finalize any anchor-first layouts",
        }

    _update_status(
        paths,
        "layout_variants",
        message=_variant_progress_message("Assembling final gallery", 0, 1),
        progress_current=0,
        progress_total=1,
    )
    final_gallery_candidates = select_distinct_final_gallery_candidates(
        candidates=final_candidates, max_variants=target_final_count
    )
    layout_variants = _build_layout_variants_payload_from_final_candidates(
        final_gallery_candidates, status="OK"
    )
    layout_variants["selection_summary"] = build_final_gallery_selection_summary(
        candidates=final_candidates,
        selected=final_gallery_candidates,
        requested_count=target_final_count,
    )
    _write_module_json(
        paths, "layout_variants", layout_variants, legacy_path=paths.layout_variants
    )
    payload_variants = layout_variants.get("variants")
    if not isinstance(payload_variants, list) or not payload_variants:
        _update_status(
            paths, "error", error="final gallery assembly produced no layout options"
        )
        return {
            "case_id": case_id,
            "case_dir": str(case_root),
            "final_output": None,
            "error": "final gallery assembly produced no layout options",
        }

    canonical_variant = payload_variants[0]
    canonical_absolute_layout = canonical_variant.get("absolute_layout")
    canonical_styled_output = canonical_variant.get("styled_result")
    if not isinstance(canonical_absolute_layout, dict) or not isinstance(
        canonical_styled_output, dict
    ):
        _update_status(
            paths,
            "error",
            error="canonical final option missing styled or layout payload",
        )
        return {
            "case_id": case_id,
            "case_dir": str(case_root),
            "final_output": None,
            "error": "canonical final option missing styled or layout payload",
        }

    canonical_bundle = deepcopy(final_gallery_candidates[0])
    canonical_bundle["source"] = str(
        canonical_variant.get("source")
        or final_gallery_candidates[0].get("source")
        or "object_level_solver"
    )
    canonical_bundle["reason"] = str(
        canonical_variant.get("reason")
        or final_gallery_candidates[0].get("reason")
        or "Primary canonical variant selected from the final anchor-first layout set."
    )
    _write_module_json(
        paths, "stylist", canonical_styled_output, legacy_path=paths.stylist
    )
    _write_canonical_branch_artifacts(
        paths=paths,
        bundle=canonical_bundle,
        absolute_layout=canonical_absolute_layout,
        styled_output=canonical_styled_output,
    )

    _update_status(
        paths,
        "done",
        message=f"{len(payload_variants)}/{target_final_count} final layout options ready.",
        progress_current=len(payload_variants),
        progress_total=target_final_count,
    )
    return {
        "case_id": case_id,
        "case_dir": str(case_root),
        "final_output": canonical_styled_output,
    }


def _solve_object_level_variant_bundle(
    *,
    concept: dict[str, Any],
    relation_plan: dict[str, Any],
    room_output: dict[str, Any],
    cluster_output: dict[str, Any],
    tier_output: dict[str, Any],
    merged_output: dict[str, Any],
    solver: MacroClusterSolver,
    manual_placements: list[dict[str, Any]] | None,
    variant_index: int,
    ablation_mode: AblationMode,
) -> list[dict[str, Any]]:
    try:
        solver_output = solver.generate_object_layout(
            room_model_json=room_output,
            merged_clusters_json=merged_output,
            relation_plan_json=relation_plan,
            cluster_constraints_json=cluster_output,
            grid_mm=GLOBAL_LAYOUT_GRID_MM,
        )
        solver_output = _strip_raw_text(solver_output)
        solver_candidates = [
            item
            for item in (solver_output.get("solutions") or [])
            if isinstance(item, dict)
        ]
        if not solver_candidates:
            solver_candidates = [solver_output]

        bundles: list[dict[str, Any]] = []
        for solution_index, solver_candidate in enumerate(solver_candidates, start=1):
            absolute_layout = _strip_raw_text(
                solver_candidate.get("absolute_layout")
                or solver_output.get("absolute_layout")
                or {}
            )
            if not isinstance(absolute_layout, dict) or str(
                absolute_layout.get("status") or ""
            ).upper() not in {"OK", "PARTIAL"}:
                continue
            absolute_layout = merge_manual_placements_into_absolute_layout(
                absolute_layout,
                manual_placements,
            )
            absolute_layout = _annotate_object_level_layout_coverage(
                absolute_layout=absolute_layout,
                merged_output=merged_output,
                solver_output=solver_candidate,
            )
            notes = _collect_unique_notes(
                [
                    f"Anchor-first concept {concept.get('concept_id') or variant_index} solved directly at object level."
                ],
                absolute_layout.get("notes")
                if isinstance(absolute_layout, dict)
                else [],
                solver_candidate.get("notes")
                if isinstance(solver_candidate, dict)
                else [],
                solver_output.get("notes") if isinstance(solver_output, dict) else [],
            )
            bundles.append(
                {
                    "concept": deepcopy(concept),
                    "relation_plan": relation_plan,
                    "room_output": room_output,
                    "cluster_output": cluster_output,
                    "tier_output": tier_output,
                    "merged_output": merged_output,
                    "solver_output": solver_candidate,
                    "absolute_layout": absolute_layout,
                    "layout_score": int(
                        (solver_candidate.get("verify_summary") or {}).get(
                            "layout_score"
                        )
                        or (solver_output.get("verify_summary") or {}).get(
                            "layout_score"
                        )
                        or 0
                    ),
                    "hard_valid": bool(
                        solver_candidate.get(
                            "hard_valid", solver_output.get("hard_valid", False)
                        )
                    ),
                    "geometry_valid": bool(
                        solver_candidate.get(
                            "geometry_valid",
                            solver_output.get(
                                "geometry_valid",
                                solver_candidate.get(
                                    "hard_valid", solver_output.get("hard_valid", False)
                                ),
                            ),
                        )
                    ),
                    "complete": bool(
                        solver_candidate.get(
                            "complete", solver_output.get("complete", False)
                        )
                    ),
                    "gallery_eligible": bool(
                        solver_candidate.get(
                            "gallery_eligible",
                            solver_output.get("gallery_eligible", False),
                        )
                    ),
                    "coverage_ratio": float(
                        solver_candidate.get("coverage_ratio")
                        or solver_output.get("coverage_ratio")
                        or 0.0
                    ),
                    "missing_cluster_ids": deepcopy(
                        (solver_candidate.get("verify_summary") or {}).get(
                            "missing_cluster_ids"
                        )
                        or (solver_output.get("verify_summary") or {}).get(
                            "missing_cluster_ids"
                        )
                        or []
                    ),
                    "notes": notes,
                    "source": f"concept:{concept.get('concept_id') or variant_index}",
                    "reason": f"Solved from macro concept {concept.get('concept_family') or concept.get('concept_id') or variant_index} using object-level anchor-first search.",
                    "dropped_inventory_by_cluster": deepcopy(
                        solver_candidate.get("dropped_inventory_by_cluster")
                        or solver_output.get("dropped_inventory_by_cluster")
                        or {}
                    ),
                    "state_signature": str(
                        solver_candidate.get("state_signature")
                        or concept.get("diversity_signature")
                        or concept.get("concept_id")
                        or f"{variant_index}_{solution_index}"
                    ),
                    "macro_layout_signature": str(
                        concept.get("diversity_signature")
                        or concept.get("concept_family")
                        or variant_index
                    ),
                    "ablation_mode": ablation_mode,
                }
            )
        return bundles
    except Exception as exc:
        logger.warning(
            "Skipping concept %s because object-level solving failed: %s",
            concept.get("concept_id") or variant_index,
            exc,
        )
        return []


def _annotate_object_level_layout_coverage(
    *,
    absolute_layout: dict[str, Any],
    merged_output: dict[str, Any],
    solver_output: dict[str, Any],
) -> dict[str, Any]:
    expected_cluster_ids = [
        str(cluster.get("cluster_id") or "")
        for cluster in (merged_output.get("clusters") or [])
        if isinstance(cluster, dict) and str(cluster.get("cluster_id") or "").strip()
    ]
    coverage = (
        absolute_layout.get("coverage")
        if isinstance(absolute_layout.get("coverage"), dict)
        else {}
    )
    present_cluster_ids = (
        coverage.get("present_cluster_ids")
        if isinstance(coverage.get("present_cluster_ids"), list)
        else []
    )
    if not present_cluster_ids:
        present_cluster_ids = sorted(
            {
                str(row.get("cluster_id") or "")
                for row in (absolute_layout.get("clusters") or [])
                if isinstance(row, dict) and str(row.get("cluster_id") or "").strip()
            }
        )
    missing_cluster_ids = [
        cluster_id
        for cluster_id in expected_cluster_ids
        if cluster_id not in set(present_cluster_ids)
    ]
    hard_valid = bool(solver_output.get("hard_valid", False))
    geometry_valid = bool(solver_output.get("geometry_valid", hard_valid))
    complete = geometry_valid and not missing_cluster_ids
    gallery_eligible = complete and bool(solver_output.get("gallery_eligible", False))
    absolute_layout["coverage"] = {
        "expected_cluster_ids": expected_cluster_ids,
        "present_cluster_ids": present_cluster_ids,
        "missing_cluster_ids": missing_cluster_ids,
    }
    absolute_layout["geometry_valid"] = geometry_valid
    absolute_layout["complete"] = complete
    absolute_layout["gallery_eligible"] = gallery_eligible
    absolute_layout["coverage_ratio"] = float(
        len(present_cluster_ids) / max(1, len(expected_cluster_ids))
    )
    absolute_layout["missing_cluster_ids"] = missing_cluster_ids
    return absolute_layout


def case_paths(case_id: str, cases_root: str | Path = "cases") -> CasePaths:
    return CasePaths(case_id=case_id, root=Path(cases_root) / case_id)


def _make_case_id(user_id: str) -> str:
    safe_user = re.sub(r"[^A-Za-z0-9_-]", "_", user_id or "user")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{safe_user}_{ts}"


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_room_type(
    input_payload: dict[str, Any], room_output: dict[str, Any]
) -> str:
    user_input = (
        input_payload.get("user_input") if isinstance(input_payload, dict) else None
    )
    if isinstance(user_input, dict):
        rt = user_input.get("room_type")
        if isinstance(rt, str) and rt:
            return rt
    meta = room_output.get("meta") if isinstance(room_output, dict) else None
    if isinstance(meta, dict):
        rt = meta.get("room_type")
        if isinstance(rt, str) and rt:
            return rt
    return "unknown"


def _extract_tenant_id_from_payload(input_payload: dict[str, Any]) -> str | None:
    tenant_id = input_payload.get("tenant_id")
    if isinstance(tenant_id, str) and tenant_id.strip():
        return tenant_id.strip()
    user_input = input_payload.get("user_input")
    if isinstance(user_input, dict):
        tenant_id = user_input.get("tenant_id")
        if isinstance(tenant_id, str) and tenant_id.strip():
            return tenant_id.strip()
    return None


def _extract_inventory_catalog(
    input_payload: dict[str, Any],
) -> list[dict[str, Any]] | None:
    for key in ("inventory_catalog", "inventory", "inventory_items"):
        value = input_payload.get(key)
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, dict)]

    user_input = input_payload.get("user_input")
    if isinstance(user_input, dict):
        for key in ("inventory_catalog", "inventory", "inventory_items"):
            value = user_input.get(key)
            if isinstance(value, list):
                return [dict(item) for item in value if isinstance(item, dict)]

    constraints = input_payload.get("constraints")
    if isinstance(constraints, dict):
        value = constraints.get("inventory_catalog")
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, dict)]
    return None


def _collect_unique_notes(*sources: Any) -> list[str]:
    notes: list[str] = []
    for source in sources:
        if not isinstance(source, list):
            continue
        for item in source:
            text = str(item).strip()
            if text and text not in notes:
                notes.append(text)
    return notes[:8]


def _tool_candidate_score(tool_evaluation: Any) -> int:
    if not isinstance(tool_evaluation, dict):
        return 0
    comparison = tool_evaluation.get("baseline_comparison")
    if not isinstance(comparison, dict):
        return 0
    return int(comparison.get("candidate_score") or 0)


def _tool_hard_valid(tool_evaluation: Any) -> bool:
    if not isinstance(tool_evaluation, dict):
        return True
    return bool(tool_evaluation.get("hard_valid", True))


def _tool_acceptable_valid(tool_evaluation: Any) -> bool:
    if not isinstance(tool_evaluation, dict):
        return False
    if "acceptable_valid" in tool_evaluation:
        return bool(tool_evaluation.get("acceptable_valid"))
    return bool(tool_evaluation.get("hard_valid", False))


def _layout_bundle_complete(bundle: dict[str, Any]) -> bool:
    return bool(bundle.get("complete", False))


def _layout_bundle_gallery_eligible(bundle: dict[str, Any]) -> bool:
    return bool(bundle.get("gallery_eligible", False))


def _make_layout_variant_entry(
    *,
    variant_index: int,
    source: str,
    reason: str,
    absolute_layout: dict[str, Any],
    styled_result: dict[str, Any],
    layout_score: int,
    hard_valid: bool,
    notes: list[str],
) -> dict[str, Any]:
    return {
        "variant_id": f"variant_{variant_index}",
        "label": f"Option {variant_index}",
        "source": source,
        "reason": reason,
        "layout_score": layout_score,
        "hard_valid": hard_valid,
        "notes": list(notes),
        "absolute_layout": deepcopy(absolute_layout),
        "styled_result": deepcopy(styled_result),
    }


def _build_external_intent_candidates(
    *,
    initial_intents: dict[str, Any] | None,
    primary_intent: dict[str, Any] | None,
    room_output: dict[str, Any],
    prepared_input_payload: dict[str, Any],
    room_type: str,
    base_guidance_text: str,
    manual_placements: list[dict[str, Any]] | None,
    cluster_forge: ClusterForge,
    tier_count: TierCountDirector,
    composer: ClusterComposer,
    relation_planner: ClusterRelationPlanner,
    solver: MacroClusterSolver,
    stylist: Stylist,
    stylist_user_context_json: dict[str, Any],
    stylist_tenant_id: str | None,
    style_plan: dict[str, Any],
    max_candidates: int = 4,
    on_candidate: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    intents = (
        initial_intents.get("intents") if isinstance(initial_intents, dict) else None
    )
    if not isinstance(intents, list):
        return []

    primary_id = (
        str(primary_intent.get("intent_id") or "")
        if isinstance(primary_intent, dict)
        else ""
    )
    candidates: list[dict[str, Any]] = []
    for intent in intents:
        if len(candidates) >= max(0, int(max_candidates)):
            break
        if not isinstance(intent, dict):
            continue
        intent_id = str(intent.get("intent_id") or "").strip()
        if not intent_id or intent_id == primary_id:
            continue
        candidate = _run_intent_branch_candidate(
            intent=intent,
            room_output=room_output,
            prepared_input_payload=prepared_input_payload,
            room_type=room_type,
            base_guidance_text=base_guidance_text,
            manual_placements=manual_placements,
            cluster_forge=cluster_forge,
            tier_count=tier_count,
            composer=composer,
            relation_planner=relation_planner,
            solver=solver,
            stylist=stylist,
            stylist_user_context_json=stylist_user_context_json,
            stylist_tenant_id=stylist_tenant_id,
            style_plan=style_plan,
        )
        if candidate is not None:
            candidates.append(candidate)
            if on_candidate is not None:
                on_candidate(candidates)
    return candidates


def _run_intent_branch_candidate(
    *,
    intent: dict[str, Any],
    room_output: dict[str, Any],
    prepared_input_payload: dict[str, Any],
    room_type: str,
    base_guidance_text: str,
    manual_placements: list[dict[str, Any]] | None,
    cluster_forge: ClusterForge,
    tier_count: TierCountDirector,
    composer: ClusterComposer,
    relation_planner: ClusterRelationPlanner,
    solver: MacroClusterSolver,
    stylist: Stylist,
    stylist_user_context_json: dict[str, Any],
    stylist_tenant_id: str | None,
    style_plan: dict[str, Any],
) -> dict[str, Any] | None:
    intent_id = str(intent.get("intent_id") or "").strip()
    if not intent_id:
        return None

    try:
        branch_guidance = _append_initial_intent_guidance(base_guidance_text, intent)
        cluster_output = cluster_forge.generate(
            room_type,
            description=branch_guidance or None,
            special_notes=None,
            room_model_json=room_output,
            inventory_catalog=_extract_inventory_catalog(prepared_input_payload),
            style_policy_json=extract_style_policy(stylist_user_context_json),
        ).model_dump()
        cluster_output = _strip_raw_text(cluster_output)
        cluster_output = augment_cluster_forge_with_manual_placements(
            cluster_output,
            manual_placements,
        )
        _log_cluster_forge_output(cluster_output, context=f"intent:{intent_id}")

        tier_output = tier_count.generate(
            description=branch_guidance,
            special_notes="",
            room_model_json=room_output,
            user_intent_json=prepared_input_payload,
            clusters_json=cluster_output,
        )
        tier_output = _strip_raw_text(tier_output)
        tier_output = apply_manual_placements_to_tier_output(
            tier_output,
            manual_placements,
            clusters_json=cluster_output,
        )

        merged_output = merge_cluster_outputs(cluster_output, tier_output)
        merged_output = _strip_raw_text(merged_output)

        cluster_outlines: dict[str, Any] = {}
        for cluster in merged_output.get("clusters", []):
            if not isinstance(cluster, dict):
                continue
            cluster_id = str(cluster.get("cluster_id") or "")
            if not cluster_id:
                continue
            cluster_result = composer.generate(
                merged_clusters=merged_output,
                cluster_id=cluster_id,
                description=branch_guidance,
                special_notes="",
            )
            cluster_result = _strip_raw_text(cluster_result)
            cluster_outlines[cluster_id] = _strip_raw_text(
                compute_cluster_outline(cluster_result)
            )

        relation_plan = relation_planner.generate(
            room_model_json=room_output,
            clusters_json=cluster_outlines,
            description=branch_guidance,
            special_notes="",
        )
        relation_plan = _strip_raw_text(relation_plan)

        filtered_cluster_outlines = _filter_unsat_cluster_outlines(cluster_outlines)
        preview_solver = MacroClusterSolver(
            tools_path=solver.tools_path,
            max_variants_per_cluster=min(int(solver.max_variants_per_cluster), 6),
            initial_candidates_per_cluster=min(
                int(solver.initial_candidates_per_cluster),
                24,
            ),
            max_rounds=min(int(solver.max_rounds), 2),
            time_limit_s=min(float(solver.time_limit_s), 8.0),
            num_workers=int(solver.num_workers),
        )
        solver_output = preview_solver.generate(
            room_model_json=room_output,
            clusters_outlines_json=filtered_cluster_outlines,
            relation_plan_json=relation_plan,
            cluster_constraints_json=cluster_output,
            grid_mm=GLOBAL_LAYOUT_GRID_MM,
        )
        solver_output = _strip_raw_text(solver_output)

        preview_payload = build_phase2_payload(
            room_output,
            merged_output,
            cluster_outlines,
            relation_plan,
            solver_output,
            cluster_output,
        )
        preview_result = build_phase2_preview_candidate(
            payload=preview_payload,
            rounds=1,
            move_limit=24,
            note=f"Fast preview candidate for intent branch {intent_id}.",
        )
        absolute_layout = _strip_raw_text(preview_result.get("absolute_layout") or {})
        absolute_layout = merge_manual_placements_into_absolute_layout(
            absolute_layout,
            manual_placements,
        )
        if str(absolute_layout.get("status") or "").upper() != "OK":
            return None

        notes: list[str] = []
        label = str(intent.get("label") or intent_id).strip() or intent_id
        summary = str(intent.get("summary") or "").strip()
        notes.append(f"Intent branch: {label}.")
        if summary:
            notes.append(summary)
        proposal = preview_result.get("proposal")
        if isinstance(proposal, dict):
            for note in proposal.get("notes") or []:
                if isinstance(note, str) and note.strip() and note.strip() not in notes:
                    notes.append(note.strip())
        for note in absolute_layout.get("notes") or []:
            if isinstance(note, str) and note.strip() and note.strip() not in notes:
                notes.append(note.strip())
        styled_result = stylist.apply_style_plan(
            layout_json=absolute_layout,
            user_context_json=stylist_user_context_json,
            tenant_id=stylist_tenant_id,
            style_plan=style_plan,
        )
        styled_result = _strip_raw_text(styled_result)
        styled_result = merge_manual_placements_into_styled_output(
            styled_result,
            manual_placements,
        )
        evaluation = preview_result.get("evaluation")

        return {
            "absolute_layout": absolute_layout,
            "styled_result": styled_result,
            "source": f"intent_branch:{intent_id}",
            "reason": summary or f"Generated from initial intent branch {label}.",
            "layout_score": _tool_candidate_score(evaluation),
            "hard_valid": _tool_hard_valid(evaluation),
            "notes": notes,
            "family": f"intent_branch:{intent_id}",
            "promoted_payload": preview_result.get("payload") or {},
        }
    except Exception as exc:
        logger.warning(
            "Skipping intent branch %s because branch generation failed: %s",
            intent_id,
            exc,
        )
        return None


def _variant_progress_message(label: str, current: int, total: int) -> str:
    safe_total = max(1, int(total))
    safe_current = min(max(0, int(current)), safe_total)
    return f"{label} {safe_current}/{safe_total}"


def _rotate_point_ccw_90s(x: float, y: float, rot: int) -> tuple[float, float]:
    normalized_rotation = int(rot) % 360
    if normalized_rotation == 0:
        return x, y
    if normalized_rotation == 90:
        return -y, x
    if normalized_rotation == 180:
        return -x, -y
    if normalized_rotation == 270:
        return y, -x
    raise ValueError(f"Unsupported rot={rot}")


def _normalize_vec2(vec: tuple[float, float] | None) -> tuple[float, float] | None:
    if vec is None:
        return None
    dx, dy = float(vec[0]), float(vec[1])
    norm = (dx * dx + dy * dy) ** 0.5
    if norm <= 1e-9:
        return None
    return (dx / norm, dy / norm)


def _rotate_vec_ccw_90s(
    vec: tuple[float, float] | None, rot: int
) -> tuple[float, float] | None:
    if vec is None:
        return None
    return _normalize_vec2(_rotate_point_ccw_90s(float(vec[0]), float(vec[1]), rot))


def _transform_local_points_to_world(
    points: list[dict[str, Any]],
    *,
    tx: int,
    ty: int,
    rot: int,
) -> list[tuple[float, float]]:
    world_points: list[tuple[float, float]] = []
    for row in points:
        if not isinstance(row, dict):
            continue
        local_x = float(row.get("x") or 0.0)
        local_y = float(row.get("y") or 0.0)
        world_x, world_y = _rotate_point_ccw_90s(local_x, local_y, rot)
        world_points.append((world_x + tx, world_y + ty))
    return world_points


def _bbox_from_world_points(points: list[tuple[float, float]]) -> dict[str, int]:
    if not points:
        return {"min_x": 0, "min_y": 0, "max_x": 0, "max_y": 0}
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return {
        "min_x": int(round(min(xs))),
        "min_y": int(round(min(ys))),
        "max_x": int(round(max(xs))),
        "max_y": int(round(max(ys))),
    }


def _solver_transform_map(solver_output: dict[str, Any]) -> dict[str, dict[str, Any]]:
    transform_map: dict[str, dict[str, Any]] = {}
    for row in solver_output.get("cluster_transforms") or []:
        if isinstance(row, dict) and isinstance(row.get("cluster_id"), str):
            transform_map[str(row["cluster_id"])] = deepcopy(row)
    return transform_map


def _solver_variant_map(solver_output: dict[str, Any]) -> dict[str, str]:
    variant_map: dict[str, str] = {}
    for row in solver_output.get("selected_variants") or []:
        if not isinstance(row, dict):
            continue
        cluster_id = row.get("cluster_id")
        variant_id = row.get("variant_id")
        if isinstance(cluster_id, str) and isinstance(variant_id, str):
            variant_map[cluster_id] = variant_id
    return variant_map


def _room_polygon_from_room_output(room_output: dict[str, Any]) -> Any | None:
    if Polygon is None:
        return None
    room = room_output.get("room") if isinstance(room_output.get("room"), dict) else {}
    polygon_rows = room.get("polygon_ccw")
    if not isinstance(polygon_rows, list):
        return None
    points = [
        (float(row.get("x") or 0.0), float(row.get("y") or 0.0))
        for row in polygon_rows
        if isinstance(row, dict)
    ]
    if len(points) < 3:
        return None
    try:
        return Polygon(points)
    except Exception:
        return None


def _seed_free_space_regions(
    *,
    room_output: dict[str, Any],
    cluster_outlines: dict[str, Any],
    solver_output: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    room_polygon = _room_polygon_from_room_output(room_output)
    if room_polygon is None or unary_union is None:
        return []

    transform_map = (
        _solver_transform_map(solver_output) if isinstance(solver_output, dict) else {}
    )
    if not transform_map:
        free_geometry = room_polygon
        geometries = [free_geometry]
        regions: list[dict[str, Any]] = []
        sorted_geometries = sorted(
            geometries,
            key=lambda item: float(getattr(item, "area", 0.0)),
            reverse=True,
        )
        for index, geometry in enumerate(sorted_geometries, start=1):
            area_mm2 = float(getattr(geometry, "area", 0.0))
            if area_mm2 < 250_000.0:
                continue
            min_x, min_y, max_x, max_y = geometry.bounds
            centroid = geometry.centroid
            regions.append(
                {
                    "label": f"open_zone_{index}",
                    "area_mm2": int(round(area_mm2)),
                    "bbox": {
                        "min_x": int(round(min_x)),
                        "min_y": int(round(min_y)),
                        "max_x": int(round(max_x)),
                        "max_y": int(round(max_y)),
                    },
                    "center": {
                        "x": int(round(float(centroid.x))),
                        "y": int(round(float(centroid.y))),
                    },
                }
            )
        return regions[:8]

    occupied_polygons: list[Any] = []
    for cluster_id, outline in cluster_outlines.items():
        if not isinstance(cluster_id, str) or not isinstance(outline, dict):
            continue
        cluster_transform = transform_map.get(
            cluster_id, {"cluster_id": cluster_id, "x": 0, "y": 0, "rot": 0}
        )
        tx = int(cluster_transform.get("x") or 0)
        ty = int(cluster_transform.get("y") or 0)
        rot = int(cluster_transform.get("rot") or 0)
        cluster_footprint = (
            outline.get("cluster_footprint")
            if isinstance(outline.get("cluster_footprint"), dict)
            else {}
        )
        outline_polygons = cluster_footprint.get("outline_polygons_ccw")
        if not isinstance(outline_polygons, list):
            continue
        for polygon_rows in outline_polygons:
            if not isinstance(polygon_rows, list):
                continue
            world_points = _transform_local_points_to_world(
                polygon_rows,
                tx=tx,
                ty=ty,
                rot=rot,
            )
            if len(world_points) < 3:
                continue
            try:
                occupied_polygons.append(Polygon(world_points))
            except Exception:
                continue

    free_geometry = (
        room_polygon
        if not occupied_polygons
        else room_polygon.difference(unary_union(occupied_polygons))
    )
    geometries = [free_geometry]
    if hasattr(free_geometry, "geoms"):
        geometries = [
            item
            for item in free_geometry.geoms
            if float(getattr(item, "area", 0.0)) > 0.0
        ]

    regions: list[dict[str, Any]] = []
    sorted_geometries = sorted(
        geometries,
        key=lambda item: float(getattr(item, "area", 0.0)),
        reverse=True,
    )
    for index, geometry in enumerate(sorted_geometries, start=1):
        area_mm2 = float(getattr(geometry, "area", 0.0))
        if area_mm2 < 250_000.0:
            continue
        min_x, min_y, max_x, max_y = geometry.bounds
        centroid = geometry.centroid
        regions.append(
            {
                "label": f"open_zone_{index}",
                "area_mm2": int(round(area_mm2)),
                "bbox": {
                    "min_x": int(round(min_x)),
                    "min_y": int(round(min_y)),
                    "max_x": int(round(max_x)),
                    "max_y": int(round(max_y)),
                },
                "center": {
                    "x": int(round(float(centroid.x))),
                    "y": int(round(float(centroid.y))),
                },
            }
        )
    return regions[:8]


def _build_seed_aware_planner_clusters_json(
    *,
    room_output: dict[str, Any],
    bundle: dict[str, Any],
    solver_output: dict[str, Any] | None,
) -> dict[str, Any]:
    cluster_outlines = bundle.get("cluster_outlines")
    if not isinstance(cluster_outlines, dict):
        return {}

    transform_map = (
        _solver_transform_map(solver_output) if isinstance(solver_output, dict) else {}
    )
    variant_map = (
        _solver_variant_map(solver_output) if isinstance(solver_output, dict) else {}
    )
    has_seed_layout = bool(transform_map)
    room_polygon = _room_polygon_from_room_output(room_output)
    room_center = room_polygon.centroid if room_polygon is not None else None
    openings = (
        room_output.get("openings")
        if isinstance(room_output.get("openings"), dict)
        else {}
    )
    door_rows = openings.get("doors") if isinstance(openings.get("doors"), list) else []
    window_rows = (
        openings.get("windows") if isinstance(openings.get("windows"), list) else []
    )

    def _nearest_opening_distance(
        opening_rows: list[dict[str, Any]], x: float, y: float
    ) -> int | None:
        best_distance: float | None = None
        for row in opening_rows:
            if not isinstance(row, dict):
                continue
            start = row.get("start") if isinstance(row.get("start"), dict) else {}
            end = row.get("end") if isinstance(row.get("end"), dict) else {}
            midpoint_x = (
                float(start.get("x") or 0.0) + float(end.get("x") or 0.0)
            ) / 2.0
            midpoint_y = (
                float(start.get("y") or 0.0) + float(end.get("y") or 0.0)
            ) / 2.0
            distance = ((midpoint_x - x) ** 2 + (midpoint_y - y) ** 2) ** 0.5
            if best_distance is None or distance < best_distance:
                best_distance = distance
        if best_distance is None:
            return None
        return int(round(best_distance))

    seed_clusters: list[dict[str, Any]] = []
    planner_clusters: dict[str, Any] = {}
    for cluster_id, outline in cluster_outlines.items():
        if not isinstance(cluster_id, str) or not isinstance(outline, dict):
            continue
        cluster_transform = transform_map.get(
            cluster_id, {"cluster_id": cluster_id, "x": 0, "y": 0, "rot": 0}
        )
        tx = int(cluster_transform.get("x") or 0)
        ty = int(cluster_transform.get("y") or 0)
        rot = int(cluster_transform.get("rot") or 0)
        cluster_footprint = (
            outline.get("cluster_footprint")
            if isinstance(outline.get("cluster_footprint"), dict)
            else {}
        )
        local_bbox = (
            cluster_footprint.get("local_bbox")
            if isinstance(cluster_footprint.get("local_bbox"), dict)
            else {}
        )
        local_bbox_points = [
            {"x": local_bbox.get("min_x") or 0, "y": local_bbox.get("min_y") or 0},
            {"x": local_bbox.get("max_x") or 0, "y": local_bbox.get("min_y") or 0},
            {"x": local_bbox.get("max_x") or 0, "y": local_bbox.get("max_y") or 0},
            {"x": local_bbox.get("min_x") or 0, "y": local_bbox.get("max_y") or 0},
        ]
        world_points = _transform_local_points_to_world(
            local_bbox_points,
            tx=tx,
            ty=ty,
            rot=rot,
        )
        world_bbox = _bbox_from_world_points(world_points)
        center_local_x = (
            float(local_bbox.get("min_x") or 0.0)
            + float(local_bbox.get("max_x") or 0.0)
        ) / 2.0
        center_local_y = (
            float(local_bbox.get("min_y") or 0.0)
            + float(local_bbox.get("max_y") or 0.0)
        ) / 2.0
        orientation_meta = (
            outline.get("orientation_meta")
            if isinstance(outline.get("orientation_meta"), dict)
            else {}
        )
        front_local = orientation_meta.get("cluster_front_local")
        axis_local = orientation_meta.get("cluster_axis_local")
        default_front_world = _rotate_vec_ccw_90s(
            (
                float(front_local.get("dx") or 0.0),
                float(front_local.get("dy") or 0.0),
            )
            if isinstance(front_local, dict)
            else None,
            rot,
        )
        axis_world = _rotate_vec_ccw_90s(
            (
                float(axis_local.get("dx") or 0.0),
                float(axis_local.get("dy") or 0.0),
            )
            if isinstance(axis_local, dict)
            else None,
            rot,
        )
        if has_seed_layout:
            center_world_x, center_world_y = _rotate_point_ccw_90s(
                center_local_x,
                center_local_y,
                rot,
            )
            center_world_x += tx
            center_world_y += ty
            front_world = default_front_world
        else:
            center_world_x = (
                float(room_center.x) if room_center is not None else center_local_x
            )
            center_world_y = (
                float(room_center.y) if room_center is not None else center_local_y
            )
            front_world = default_front_world
        region_tags: list[str] = []
        if has_seed_layout and room_center is not None:
            distance_to_center = (
                (float(room_center.x) - center_world_x) ** 2
                + (float(room_center.y) - center_world_y) ** 2
            ) ** 0.5
            region_tags.append(
                "near_center" if distance_to_center < 1200.0 else "edge_or_perimeter"
            )
        if (
            has_seed_layout
            and (
                door_distance := _nearest_opening_distance(
                    door_rows, center_world_x, center_world_y
                )
            )
            is not None
        ):
            if door_distance < 1400:
                region_tags.append("entry_side")
            elif door_distance > 2600:
                region_tags.append("far_from_entry")
        if (
            has_seed_layout
            and (
                window_distance := _nearest_opening_distance(
                    window_rows, center_world_x, center_world_y
                )
            )
            is not None
        ):
            if window_distance < 1400:
                region_tags.append("window_side")

        seed_state = {
            "variant_id": variant_map.get(cluster_id),
            "world_center": {
                "x": int(round(center_world_x)),
                "y": int(round(center_world_y)),
            },
            "world_bbox": world_bbox,
            "rotation_ccw": rot,
            "front_world": (
                {"dx": round(front_world[0], 3), "dy": round(front_world[1], 3)}
                if front_world is not None
                else None
            ),
            "axis_world": (
                {"dx": round(axis_world[0], 3), "dy": round(axis_world[1], 3)}
                if axis_world is not None
                else None
            ),
            "region_tags": region_tags,
        }
        planner_clusters[cluster_id] = {**deepcopy(outline), "seed_state": seed_state}
        seed_clusters.append({"cluster_id": cluster_id, **seed_state})

    payload = {
        "clusters": planner_clusters,
        "seed_layout_state": {
            "notes": [
                (
                    "The planner is reading a rough cluster seed layout before precise object-level repair."
                    if has_seed_layout
                    else "No macro seed layout has been solved yet; invent a distinct macro concept from room geometry, openings, and cluster footprints."
                ),
                (
                    "Use the current cluster positions, facing directions, and remaining free-space pockets as the starting reality."
                    if has_seed_layout
                    else "Use the room geometry, wall opportunities, and circulation clues to create a materially distinct concept."
                ),
            ],
            "clusters": seed_clusters,
        },
        "free_space_regions": _seed_free_space_regions(
            room_output=room_output,
            cluster_outlines=cluster_outlines,
            solver_output=solver_output if isinstance(solver_output, dict) else {},
        ),
    }
    cluster_output = bundle.get("cluster_output")
    if isinstance(cluster_output, dict) and isinstance(
        cluster_output.get("semantic_layout_program"), dict
    ):
        payload["semantic_layout_program"] = deepcopy(
            cluster_output["semantic_layout_program"]
        )
    return payload


def _planner_seed_relation_plan(room_output: dict[str, Any]) -> dict[str, Any]:
    room = room_output.get("room") if isinstance(room_output.get("room"), dict) else {}
    openings = (
        room_output.get("openings")
        if isinstance(room_output.get("openings"), dict)
        else {}
    )
    keep_open_regions: list[dict[str, Any]] = []
    if openings.get("doors"):
        keep_open_regions.append(
            {
                "type": "entry_buffer",
                "priority": "high",
                "reason": "Seed solve should keep the entry usable before semantic planning begins.",
            }
        )
    if openings.get("windows"):
        keep_open_regions.append(
            {
                "type": "window_buffer",
                "priority": "medium",
                "reason": "Seed solve should avoid obvious window blockage before post-seed planning.",
            }
        )
    return {
        "status": "OK",
        "room_id": str(room.get("room_id") or "room_1"),
        "notes": [
            "Primary solver seed uses a planner-free macro relation scaffold.",
            "Detailed orientation and cluster semantics are deferred until post-seed planning.",
        ],
        "layout_intent_profile": {},
        "cluster_affinities": [],
        "cluster_relations": [],
        "cluster_directional_relations": [],
        "cluster_orientations": [],
        "object_orientations": [],
        "circulation_plan": {"keep_open_regions": keep_open_regions, "main_paths": []},
        "placement_guidelines": [],
        "missing": [],
    }


def _plan_seed_variant_bundles(
    *,
    bundle: dict[str, Any],
    room_output: dict[str, Any],
    guidance_text: str,
    relation_planner: ClusterRelationPlanner,
    target_count: int,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    solver_output = bundle.get("solver_output")
    planner_clusters_json = _build_seed_aware_planner_clusters_json(
        room_output=room_output,
        bundle=bundle,
        solver_output=solver_output if isinstance(solver_output, dict) else None,
    )
    macro_concept_bundle = relation_planner.generate_bundle(
        room_model_json=room_output,
        clusters_json=planner_clusters_json,
        target_count=target_count,
        description=guidance_text or None,
    )
    planned_bundles: list[dict[str, Any]] = []
    payload_variants: list[dict[str, Any]] = []
    concepts = [
        concept
        for concept in macro_concept_bundle.get("concepts", [])
        if isinstance(concept, dict)
    ]
    total = len(concepts)
    for index, concept in enumerate(concepts, start=1):
        concept_family = str(concept.get("concept_family") or f"concept_{index}")
        if on_progress is not None:
            on_progress(
                index - 1,
                total,
                f"Planning layout concept {index}/{total}: {concept_family}.",
            )
        relation_plan = solver_plan_from_concept(
            concept=concept,
            room_model_json=room_output,
            room_type=str(macro_concept_bundle.get("room_type") or ""),
        )
        relation_plan = _strip_raw_text(relation_plan)
        planned_bundle = deepcopy(bundle)
        planned_bundle["relation_plan"] = relation_plan
        planned_bundle["macro_concept"] = deepcopy(concept)
        planned_bundle["source"] = concept_family
        planned_bundle["reason"] = str(
            concept.get("spatial_character")
            or "Deterministic topology-aware macro concept."
        )
        planned_bundle["planner_model_name"] = "deterministic"
        planned_bundle["seed_index"] = index
        planned_bundles.append(planned_bundle)
        payload_variants.append(
            {
                "variant_id": f"plan_{index}",
                "label": f"Intent {index}",
                "source": concept_family,
                "planner_model_name": "deterministic",
                "reason": planned_bundle["reason"],
                "macro_concept": deepcopy(concept),
                "relation_plan": deepcopy(relation_plan),
            }
        )
        if on_progress is not None:
            on_progress(
                index,
                total,
                f"Planning layout concept {index}/{total}: {concept_family} ready.",
            )
    macro_concept_bundle["seed_context"] = deepcopy(planner_clusters_json)
    macro_concept_bundle["variants"] = payload_variants
    macro_concept_bundle["guidance_summary"] = guidance_text[:1200]
    return planned_bundles, dict(macro_concept_bundle)


def _style_variant_user_context(
    user_context_json: dict[str, Any],
    *,
    variant_index: int,
) -> dict[str, Any]:
    palette_notes = [
        "Palette direction: keep the requested style, but lean warmer and slightly richer in contrast.",
        "Palette direction: keep the requested style, but lean cooler and calmer with balanced neutrals.",
        "Palette direction: keep the requested style, but lean earthier with grounded depth and darker anchor tones.",
        "Palette direction: keep the requested style, but lean lighter, softer, and more airy.",
        "Palette direction: keep the requested style, but use a bolder accent contrast while staying cohesive.",
    ]
    context = deepcopy(user_context_json)
    palette_note = palette_notes[(variant_index - 1) % len(palette_notes)]
    notes = context.get("notes")
    merged_notes = (
        [str(item).strip() for item in notes if str(item).strip()]
        if isinstance(notes, list)
        else []
    )
    if palette_note not in merged_notes:
        merged_notes.append(palette_note)
    context["notes"] = merged_notes[:8]
    guidance_text = str(context.get("guidance_text") or "").strip()
    if palette_note not in guidance_text:
        context["guidance_text"] = (
            f"{guidance_text}\n{palette_note}".strip()
            if guidance_text
            else palette_note
        )
    return context


def _solver_feasibility_allows_backoff(feasibility: dict[str, Any]) -> bool:
    offenders = feasibility.get("offenders")
    if not isinstance(offenders, list) or not offenders:
        return False
    if _solver_feasibility_requires_semantic_recompose(feasibility):
        return False
    reasons = {
        str(item.get("reason") or "").strip()
        for item in offenders
        if isinstance(item, dict)
    }
    return bool(reasons) and reasons <= {
        "SOLVER_EXACT_ASSIGNMENT_UNSAT",
        "SOLVER_HAS_NO_HARD_VALID_CANDIDATES",
    }


def _solver_feasibility_requires_semantic_recompose(
    feasibility: dict[str, Any],
) -> bool:
    offenders = feasibility.get("offenders")
    if not isinstance(offenders, list):
        return False
    for offender in offenders:
        if not isinstance(offender, dict):
            continue
        reason = str(offender.get("reason") or "").strip()
        placer_seed_reason = str(offender.get("placer_seed_reason") or "").strip()
        if reason == "SOLVER_CONCEPT_POLICY_EMPTY_POOL":
            return True
        before_count = offender.get("candidate_count_before_policy")
        after_count = offender.get("candidate_count_after_policy")
        if (
            placer_seed_reason == "required_clusters_without_candidates"
            and isinstance(before_count, int)
            and before_count > 0
            and after_count == 0
        ):
            return True
    return False


def _solver_feasibility_offender_ids(feasibility: dict[str, Any]) -> list[str]:
    offenders = feasibility.get("offenders")
    if not isinstance(offenders, list):
        return []
    cluster_ids: list[str] = []
    seen: set[str] = set()
    for offender in offenders:
        if not isinstance(offender, dict):
            continue
        cluster_id = str(offender.get("cluster_id") or "").strip()
        if not cluster_id or cluster_id in seen:
            continue
        cluster_ids.append(cluster_id)
        seen.add(cluster_id)
    return cluster_ids


def _solver_parallel_limit(target_final_count: int) -> int:
    return max(1, min(int(target_final_count), 2))


def _solver_workers_per_task(
    solver: MacroClusterSolver,
    *,
    parallel_tasks: int,
) -> int:
    base_workers = max(1, int(solver.num_workers))
    if parallel_tasks <= 1:
        return base_workers
    return max(1, min(4, base_workers // parallel_tasks))


def _solver_with_workers(
    solver: MacroClusterSolver, *, num_workers: int
) -> MacroClusterSolver:
    return MacroClusterSolver(
        tools_path=solver.tools_path,
        max_variants_per_cluster=int(solver.max_variants_per_cluster),
        initial_candidates_per_cluster=int(solver.initial_candidates_per_cluster),
        max_rounds=int(solver.max_rounds),
        time_limit_s=float(solver.time_limit_s),
        num_workers=max(1, int(num_workers)),
    )


def _build_primary_seed_solver(solver: MacroClusterSolver) -> MacroClusterSolver:
    if _is_fast_layout_mode():
        return MacroClusterSolver(
            tools_path=solver.tools_path,
            max_variants_per_cluster=max(
                3, min(int(solver.max_variants_per_cluster), 4)
            ),
            initial_candidates_per_cluster=max(
                12, min(int(solver.initial_candidates_per_cluster), 20)
            ),
            max_rounds=max(1, min(int(solver.max_rounds), 2)),
            time_limit_s=max(6.0, min(float(solver.time_limit_s), 8.0)),
            num_workers=1,
        )
    return MacroClusterSolver(
        tools_path=solver.tools_path,
        max_variants_per_cluster=max(3, min(int(solver.max_variants_per_cluster), 5)),
        initial_candidates_per_cluster=max(
            16, min(int(solver.initial_candidates_per_cluster), 28)
        ),
        max_rounds=max(2, min(int(solver.max_rounds), 3)),
        time_limit_s=max(12.0, min(float(solver.time_limit_s), 14.0)),
        num_workers=1,
    )


def _build_primary_seed_solver_profiles(
    solver: MacroClusterSolver,
) -> list[MacroClusterSolver]:
    base_solver = _build_primary_seed_solver(solver)
    if _is_fast_layout_mode():
        stable_solver = MacroClusterSolver(
            tools_path=base_solver.tools_path,
            max_variants_per_cluster=max(
                3, min(int(base_solver.max_variants_per_cluster) + 1, 5)
            ),
            initial_candidates_per_cluster=max(
                16, min(int(base_solver.initial_candidates_per_cluster) + 4, 24)
            ),
            max_rounds=max(1, min(int(base_solver.max_rounds), 2)),
            time_limit_s=max(8.0, min(float(base_solver.time_limit_s) + 2.0, 10.0)),
            num_workers=1,
        )
        return [base_solver, stable_solver]
    stable_solver = MacroClusterSolver(
        tools_path=base_solver.tools_path,
        max_variants_per_cluster=max(
            3, min(int(base_solver.max_variants_per_cluster) + 1, 6)
        ),
        initial_candidates_per_cluster=max(
            20, min(int(base_solver.initial_candidates_per_cluster) + 4, 32)
        ),
        max_rounds=max(2, min(int(base_solver.max_rounds) + 1, 4)),
        time_limit_s=max(14.0, min(float(base_solver.time_limit_s) + 3.0, 18.0)),
        num_workers=1,
    )
    return [base_solver, stable_solver]


def _primary_seed_cache_key(
    *,
    room_output: dict[str, Any],
    bundle: dict[str, Any],
) -> str:
    room_payload = {
        "room": room_output.get("room"),
        "openings": room_output.get("openings"),
        "fixed_obstacles": room_output.get("fixed_obstacles"),
    }
    cache_payload = {
        "version": _PRIMARY_SEED_CACHE_VERSION,
        "room": room_payload,
        "bundle_signature": _bundle_signature(bundle),
        "cluster_outlines": bundle.get("cluster_outlines"),
    }
    raw = json.dumps(cache_payload, ensure_ascii=True, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _build_solver_seed_relation_plan(
    relation_plan: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(relation_plan, dict):
        return relation_plan

    out = deepcopy(relation_plan)
    out["cluster_directional_relations"] = []
    out["cluster_orientations"] = []
    out["object_orientations"] = []
    out["placement_guidelines"] = []

    circulation_plan = out.get("circulation_plan")
    if isinstance(circulation_plan, dict):
        keep_open_regions = circulation_plan.get("keep_open_regions")
        filtered_regions = []
        if isinstance(keep_open_regions, list):
            filtered_regions = [
                item
                for item in keep_open_regions
                if isinstance(item, dict)
                and item.get("type") in {"entry_buffer", "window_buffer"}
            ]
        out["circulation_plan"] = {
            "main_paths": [],
            "keep_open_regions": filtered_regions,
        }

    notes = list(out.get("notes") or [])
    notes.append(
        "Primary solver uses a relaxed macro relation plan to find a fast temporary layout seed."
    )
    out["notes"] = notes
    return out


def _write_bundle_stage_artifacts(paths: CasePaths, bundle: dict[str, Any]) -> None:
    cluster_output = bundle.get("cluster_output")
    if isinstance(cluster_output, dict):
        _write_module_json(
            paths,
            "cluster_forge",
            cluster_output,
            legacy_path=paths.cluster_forge,
        )

    tier_output = bundle.get("tier_output")
    if isinstance(tier_output, dict):
        _write_module_json(
            paths,
            "tier_count_director",
            tier_output,
            legacy_path=paths.tier_count,
        )

    merged_output = bundle.get("merged_output")
    if isinstance(merged_output, dict):
        _write_module_json(
            paths,
            "cluster_output_merger",
            merged_output,
            legacy_path=paths.cluster_merged,
        )

    cluster_results = bundle.get("cluster_results")
    if isinstance(cluster_results, dict):
        for cluster_id, cluster_result in cluster_results.items():
            if isinstance(cluster_id, str) and isinstance(cluster_result, dict):
                _write_json(paths.cluster_composer(cluster_id), cluster_result)
                _write_json(paths.module_cluster_composer(cluster_id), cluster_result)

    cluster_outlines = bundle.get("cluster_outlines")
    if isinstance(cluster_outlines, dict):
        for cluster_id, outline in cluster_outlines.items():
            if isinstance(cluster_id, str) and isinstance(outline, dict):
                _write_json(paths.cluster_outline(cluster_id), outline)
                _write_json(paths.module_cluster_outline(cluster_id), outline)
        _write_module_json(
            paths,
            "cluster_outline_bundle",
            cluster_outlines,
            legacy_path=paths.cluster_outlines_all,
        )

    relation_plan = bundle.get("relation_plan")
    if isinstance(relation_plan, dict):
        _write_module_json(
            paths,
            "cluster_relation_plan",
            relation_plan,
            legacy_path=paths.cluster_relation_plan,
        )
        _write_module_json(paths, "seed_concept_relation_plan", relation_plan)

    solver_output = bundle.get("solver_output")
    if isinstance(solver_output, dict):
        _write_module_json(
            paths,
            "macro_cluster_solver",
            solver_output,
            legacy_path=paths.cluster_solver,
        )

    dropped_inventory = bundle.get("dropped_inventory_by_cluster")
    if isinstance(dropped_inventory, dict):
        _write_module_json(
            paths,
            "macro_cluster_solver_dropped_inventory",
            dropped_inventory_payload(dropped_inventory),
            legacy_path=paths.solver_dropped_inventory,
        )


def _remap_cluster_geometry_bundle(
    *,
    room_output: dict[str, Any],
    cluster_output: dict[str, Any],
    tier_output: dict[str, Any],
    previous_bundle: dict[str, Any],
) -> dict[str, Any] | None:
    merged_output = merge_cluster_outputs(cluster_output, tier_output)
    merged_output = _strip_raw_text(merged_output)

    previous_cluster_results = previous_bundle.get("cluster_results")
    if not isinstance(previous_cluster_results, dict):
        return None

    cluster_results: dict[str, Any] = {}
    cluster_outlines: dict[str, Any] = {}
    for cluster in merged_output.get("clusters", []):
        if not isinstance(cluster, dict):
            continue
        cluster_id = str(cluster.get("cluster_id") or "").strip()
        if not cluster_id:
            continue

        previous_cluster_result = previous_cluster_results.get(cluster_id)
        if not isinstance(previous_cluster_result, dict):
            return None

        placements = previous_cluster_result.get("local_placements")
        if not isinstance(placements, list):
            return None

        active_members = {
            str(member).strip()
            for member in (cluster.get("members") or [])
            if isinstance(member, str) and member.strip()
        }
        filtered_placements = [
            deepcopy(row)
            for row in placements
            if isinstance(row, dict)
            and str(row.get("id") or "").strip()
            and str(row.get("id") or "").strip() in active_members
        ]
        if active_members and not filtered_placements:
            return None

        cluster_result = _strip_raw_text(
            _build_cluster_output_from_placements(
                cluster=cluster,
                placements=filtered_placements,
                notes=_collect_unique_notes(
                    previous_cluster_result.get("notes"),
                    [
                        "Cluster geometry was remapped from the existing placements after solver backoff."
                    ],
                ),
            )
        )
        cluster_results[cluster_id] = cluster_result
        cluster_outlines[cluster_id] = _strip_raw_text(
            compute_cluster_outline(cluster_result)
        )

    remapped_bundle = deepcopy(previous_bundle)
    remapped_bundle["tier_output"] = deepcopy(tier_output)
    remapped_bundle["merged_output"] = merged_output
    remapped_bundle["cluster_results"] = cluster_results
    remapped_bundle["cluster_outlines"] = cluster_outlines
    remapped_bundle["compose_feasibility"] = evaluate_composed_cluster_feasibility(
        room_output=room_output,
        merged_output=merged_output,
        cluster_results=cluster_results,
        cluster_outlines=cluster_outlines,
    )
    return remapped_bundle


def _semantic_recompose_cluster_bundle(
    *,
    room_output: dict[str, Any],
    guidance_text: str,
    composer: ClusterComposer,
    feasibility: dict[str, Any],
    previous_bundle: dict[str, Any],
) -> dict[str, Any] | None:
    cluster_output = previous_bundle.get("cluster_output")
    tier_output = previous_bundle.get("tier_output")
    previous_cluster_results = previous_bundle.get("cluster_results")
    previous_cluster_outlines = previous_bundle.get("cluster_outlines")
    if not isinstance(cluster_output, dict):
        return None
    if not isinstance(tier_output, dict):
        return None
    if not isinstance(previous_cluster_results, dict):
        return None
    if not isinstance(previous_cluster_outlines, dict):
        return None

    offender_ids = set(_solver_feasibility_offender_ids(feasibility))
    if not offender_ids:
        return None

    merged_output = merge_cluster_outputs(cluster_output, tier_output)
    merged_output = _strip_raw_text(merged_output)
    special_notes = _semantic_recompose_notes(feasibility)

    cluster_results: dict[str, Any] = {}
    cluster_outlines: dict[str, Any] = {}
    for cluster in merged_output.get("clusters", []):
        if not isinstance(cluster, dict):
            continue
        cluster_id = str(cluster.get("cluster_id") or "").strip()
        if not cluster_id:
            continue

        previous_result = previous_cluster_results.get(cluster_id)
        previous_outline = previous_cluster_outlines.get(cluster_id)
        should_recompose = cluster_id in offender_ids or not isinstance(
            previous_result,
            dict,
        )
        if should_recompose:
            cluster_result = composer.generate(
                merged_clusters=merged_output,
                cluster_id=cluster_id,
                description=guidance_text,
                special_notes=special_notes,
            )
            cluster_result = _strip_raw_text(cluster_result)
        else:
            cluster_result = deepcopy(previous_result)

        if not isinstance(cluster_result, dict):
            return None

        cluster_results[cluster_id] = cluster_result
        if should_recompose or not isinstance(previous_outline, dict):
            cluster_outlines[cluster_id] = _strip_raw_text(
                compute_cluster_outline(cluster_result)
            )
        else:
            cluster_outlines[cluster_id] = deepcopy(previous_outline)

    if not cluster_results:
        return None

    recomposed_bundle = deepcopy(previous_bundle)
    recomposed_bundle["merged_output"] = merged_output
    recomposed_bundle["cluster_results"] = cluster_results
    recomposed_bundle["cluster_outlines"] = cluster_outlines
    recomposed_bundle["compose_feasibility"] = evaluate_composed_cluster_feasibility(
        room_output=room_output,
        merged_output=merged_output,
        cluster_results=cluster_results,
        cluster_outlines=cluster_outlines,
    )
    recomposed_bundle["notes"] = _collect_unique_notes(
        previous_bundle.get("notes"),
        [
            "Recomposed semantic cluster variants after concept policy emptied the solver candidate pool."
        ],
    )
    return recomposed_bundle


def _semantic_recompose_notes(feasibility: dict[str, Any]) -> str:
    lines = [
        "Solver feedback: regenerate semantic variants for these clusters.",
        "Do not shrink/drop furniture for this feedback; preserve the requested inventory unless the cluster itself is invalid.",
        "The macro solver had hard-valid candidates before concept-family policy, but policy removed the compatible pool or left required clusters without candidates.",
    ]
    for offender in feasibility.get("offenders") or []:
        if not isinstance(offender, dict):
            continue
        cluster_id = str(offender.get("cluster_id") or "").strip()
        reason = str(offender.get("reason") or "").strip()
        before_count = offender.get("candidate_count_before_policy")
        after_count = offender.get("candidate_count_after_policy")
        if not cluster_id:
            continue
        lines.append(
            f"{cluster_id}: {reason}; before_policy={before_count}; after_policy={after_count}."
        )
    return "\n".join(lines)


def _build_layout_variants_payload_from_final_candidates(
    candidates: list[dict[str, Any]],
    *,
    status: str,
) -> dict[str, Any]:
    payload_variants: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        absolute_layout = candidate.get("absolute_layout")
        styled_result = candidate.get("styled_result")
        if not isinstance(absolute_layout, dict) or not isinstance(styled_result, dict):
            continue
        payload_variants.append(
            {
                "variant_id": f"variant_{index}",
                "label": f"Option {index}",
                "source": str(candidate.get("source") or f"loop_variant:{index}"),
                "state_signature": str(
                    candidate.get("state_signature")
                    or candidate.get("layout_signature")
                    or ""
                ),
                "macro_layout_signature": str(
                    candidate.get("macro_layout_signature") or ""
                ),
                "gallery_selection_mode": str(
                    candidate.get("gallery_selection_mode") or ""
                ),
                "reason": str(
                    candidate.get("reason")
                    or "Feasible loop variant finalized through planner and phase-2."
                ),
                "layout_score": int(candidate.get("layout_score") or 0),
                "hard_valid": bool(candidate.get("hard_valid", True)),
                "complete": bool(candidate.get("complete", False)),
                "gallery_eligible": bool(candidate.get("gallery_eligible", False)),
                "coverage_ratio": float(candidate.get("coverage_ratio") or 0.0),
                "missing_cluster_ids": deepcopy(
                    candidate.get("missing_cluster_ids") or []
                ),
                "notes": deepcopy(candidate.get("notes") or []),
                "absolute_layout": deepcopy(absolute_layout),
                "styled_result": deepcopy(styled_result),
            }
        )

    return {
        "status": status,
        "selected_variant_id": payload_variants[0]["variant_id"]
        if payload_variants
        else None,
        "variants": payload_variants,
    }


def _select_final_styling_candidates(
    candidates: list[dict[str, Any]],
    *,
    max_variants: int,
) -> list[dict[str, Any]]:
    selected = select_distinct_final_gallery_candidates(
        candidates=candidates,
        max_variants=max_variants,
    )
    if selected:
        return selected
    return list(candidates[: max(0, int(max_variants))])


def _bundle_signature(bundle: dict[str, Any]) -> str:
    merged_output = bundle.get("merged_output")
    if not isinstance(merged_output, dict):
        return ""
    clusters = merged_output.get("clusters")
    if not isinstance(clusters, list):
        return ""

    rows: list[str] = []
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        cluster_id = str(cluster.get("cluster_id") or "").strip()
        decisions = cluster.get("decisions")
        if not cluster_id or not isinstance(decisions, list):
            continue
        active_rows = [
            row
            for row in decisions
            if isinstance(row, dict) and int(row.get("quantity") or 0) > 0
        ]
        for row in active_rows:
            rows.append(
                "|".join(
                    [
                        cluster_id,
                        str(row.get("object_type") or row.get("category") or ""),
                        str(int(row.get("quantity") or 0)),
                        str(row.get("size_tier") or ""),
                    ]
                )
            )
    rows.sort()
    return "\n".join(rows)


def _ensure_requested_non_functional_contract_objects(
    *,
    absolute_layout: dict[str, Any],
    room_output: dict[str, Any],
    cluster_output: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    objects = [
        row for row in absolute_layout.get("objects") or [] if isinstance(row, dict)
    ]
    missing_items = missing_non_functional_contract_items(
        contract=request_contract_from_payload(cluster_output),
        objects=objects,
    )
    if not missing_items:
        return absolute_layout, {"added_count": 0, "added_object_types": []}

    room_bbox = _room_bbox_for_non_functional_refill(
        room_output=room_output,
        absolute_layout=absolute_layout,
    )
    if room_bbox is None:
        return absolute_layout, {
            "added_count": 0,
            "added_object_types": [],
            "skipped_reason": "missing_room_polygon",
        }

    updated_layout = deepcopy(absolute_layout)
    updated_objects = [
        dict(row)
        for row in updated_layout.get("objects") or []
        if isinstance(row, dict)
    ]
    added_types: list[str] = []
    for item in missing_items:
        object_type = canonical_object_type(str(item.get("object_type") or ""))
        spec = _REQUEST_NON_FUNCTIONAL_LAYOUT_SPECS.get(object_type)
        if not spec:
            continue
        layout_object = _requested_non_functional_layout_object(
            object_type=object_type,
            spec=spec,
            room_bbox=room_bbox,
            existing_objects=updated_objects,
        )
        if layout_object is None:
            continue
        updated_objects.append(layout_object)
        added_types.append(object_type)

    if not added_types:
        return absolute_layout, {"added_count": 0, "added_object_types": []}

    updated_layout["objects"] = updated_objects
    return updated_layout, {
        "added_count": len(added_types),
        "added_object_types": added_types,
        "source": "request_contract_non_functional",
    }


def _room_bbox_for_non_functional_refill(
    *,
    room_output: dict[str, Any],
    absolute_layout: dict[str, Any],
) -> dict[str, int] | None:
    room = room_output.get("room") if isinstance(room_output.get("room"), dict) else {}
    layout_room = (
        absolute_layout.get("room")
        if isinstance(absolute_layout.get("room"), dict)
        else {}
    )
    for candidate in (
        room.get("polygon_ccw"),
        layout_room.get("polygon_ccw"),
        absolute_layout.get("polygon_ccw"),
    ):
        points = _non_functional_refill_points(candidate)
        if len(points) >= 3:
            return _bbox_from_world_points(points)
    return None


def _non_functional_refill_points(value: Any) -> list[tuple[float, float]]:
    if not isinstance(value, list):
        return []
    points: list[tuple[float, float]] = []
    for row in value:
        if not isinstance(row, dict):
            continue
        try:
            points.append((float(row.get("x") or 0.0), float(row.get("y") or 0.0)))
        except (TypeError, ValueError):
            continue
    return points


def _requested_non_functional_layout_object(
    *,
    object_type: str,
    spec: dict[str, Any],
    room_bbox: dict[str, int],
    existing_objects: list[dict[str, Any]],
) -> dict[str, Any] | None:
    room_width = max(0, int(room_bbox["max_x"]) - int(room_bbox["min_x"]))
    room_height = max(0, int(room_bbox["max_y"]) - int(room_bbox["min_y"]))
    if room_width <= 0 or room_height <= 0:
        return None

    width = _request_refill_dimension(
        spec=spec,
        fixed_key="width",
        ratio_key="width_ratio",
        min_key="min_w",
        max_key="max_w",
        room_span=room_width,
    )
    height = _request_refill_dimension(
        spec=spec,
        fixed_key="height",
        ratio_key="height_ratio",
        min_key="min_h",
        max_key="max_h",
        room_span=room_height,
    )
    if width <= 0 or height <= 0:
        return None

    target_object = _target_existing_layout_object(
        existing_objects,
        spec.get("target_object_types"),
    )
    target_center = _layout_object_center(target_object) if target_object else None
    if target_center is not None:
        center_x, center_y = target_center
    else:
        center_x = int(round((int(room_bbox["min_x"]) + int(room_bbox["max_x"])) / 2.0))
        center_y = int(round((int(room_bbox["min_y"]) + int(room_bbox["max_y"])) / 2.0))
    min_x = _clamp_int(
        center_x - width // 2,
        int(room_bbox["min_x"]),
        int(room_bbox["max_x"]) - width,
    )
    min_y = _clamp_int(
        center_y - height // 2,
        int(room_bbox["min_y"]),
        int(room_bbox["max_y"]) - height,
    )
    max_x = min_x + width
    max_y = min_y + height
    instance_id = _unique_refill_instance_id(object_type, existing_objects)
    place_on = deepcopy(
        spec.get("place_on") if isinstance(spec.get("place_on"), dict) else {}
    )
    if place_on.get("target_instance_id") == "floor":
        target_id = _largest_existing_layout_object_id(existing_objects)
        if target_id:
            place_on["target_instance_id"] = target_id
    elif target_object is not None and not place_on.get("target_instance_id"):
        target_id = target_object.get("instance_id") or target_object.get("object_id")
        if isinstance(target_id, str) and target_id.strip():
            place_on["target_instance_id"] = target_id.strip()

    return {
        "cluster_id": _preferred_existing_cluster_id(existing_objects),
        "object_id": instance_id,
        "instance_id": instance_id,
        "object_type": object_type,
        "category": object_type,
        "source": "request_contract",
        "x": min_x,
        "y": min_y,
        "rot": 0,
        "rotation_ccw": 0,
        "w": width,
        "h": height,
        "rect": [min_x, min_y, width, height],
        "bbox": {
            "min_x": min_x,
            "min_y": min_y,
            "max_x": max_x,
            "max_y": max_y,
        },
        "center": {
            "x": int(round((min_x + max_x) / 2.0)),
            "y": int(round((min_y + max_y) / 2.0)),
        },
        "polygon_ccw": [
            {"x": min_x, "y": min_y},
            {"x": max_x, "y": min_y},
            {"x": max_x, "y": max_y},
            {"x": min_x, "y": max_y},
        ],
        "place_on": place_on or None,
        "collision_layer": str(spec.get("collision_layer") or "floor_underlay"),
        "priority": "request_non_functional",
        "role": "decor_light",
    }


def _request_refill_dimension(
    *,
    spec: dict[str, Any],
    fixed_key: str,
    ratio_key: str,
    min_key: str,
    max_key: str,
    room_span: int,
) -> int:
    fixed_value = spec.get(fixed_key)
    if isinstance(fixed_value, (int, float)) and fixed_value > 0:
        return min(int(round(float(fixed_value))), max(1, room_span))
    ratio = float(spec.get(ratio_key) or 0.35)
    value = int(round(float(room_span) * ratio))
    lower = min(int(spec.get(min_key) or 1), max(1, room_span))
    upper = min(int(spec.get(max_key) or room_span), max(1, room_span))
    if upper < lower:
        lower = upper
    return _clamp_int(value, lower, upper)


def _target_existing_layout_object(
    objects: list[dict[str, Any]],
    target_object_types: Any,
) -> dict[str, Any] | None:
    if not isinstance(target_object_types, list):
        return None
    target_types = {
        canonical_object_type(str(item or ""))
        for item in target_object_types
        if str(item or "").strip()
    }
    if not target_types:
        return None

    candidates: list[tuple[float, dict[str, Any]]] = []
    for row in objects:
        if not isinstance(row, dict):
            continue
        object_type = canonical_object_type(
            str(row.get("object_type") or row.get("category") or "")
        )
        if object_type not in target_types:
            continue
        area = _layout_object_area(row)
        candidates.append((area, row))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _layout_object_center(row: dict[str, Any] | None) -> tuple[int, int] | None:
    if not isinstance(row, dict):
        return None
    center = row.get("center")
    if isinstance(center, dict):
        try:
            return (
                int(round(float(center.get("x") or 0.0))),
                int(round(float(center.get("y") or 0.0))),
            )
        except (TypeError, ValueError):
            pass
    bbox = row.get("bbox")
    if isinstance(bbox, dict):
        try:
            return (
                int(
                    round(
                        (
                            float(bbox.get("min_x") or 0.0)
                            + float(bbox.get("max_x") or 0.0)
                        )
                        / 2.0
                    )
                ),
                int(
                    round(
                        (
                            float(bbox.get("min_y") or 0.0)
                            + float(bbox.get("max_y") or 0.0)
                        )
                        / 2.0
                    )
                ),
            )
        except (TypeError, ValueError):
            pass
    try:
        x = float(row.get("x") or 0.0)
        y = float(row.get("y") or 0.0)
        width = float(row.get("w") or 0.0)
        height = float(row.get("h") or 0.0)
    except (TypeError, ValueError):
        return None
    return (int(round(x + width / 2.0)), int(round(y + height / 2.0)))


def _layout_object_area(row: dict[str, Any]) -> float:
    bbox = row.get("bbox")
    if isinstance(bbox, dict):
        try:
            width = float(bbox.get("max_x") or 0.0) - float(bbox.get("min_x") or 0.0)
            height = float(bbox.get("max_y") or 0.0) - float(bbox.get("min_y") or 0.0)
            return max(0.0, width * height)
        except (TypeError, ValueError):
            return 0.0
    try:
        return max(0.0, float(row.get("w") or 0.0) * float(row.get("h") or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _unique_refill_instance_id(
    object_type: str,
    existing_objects: list[dict[str, Any]],
) -> str:
    existing = {
        str(row.get("instance_id") or row.get("object_id") or "")
        for row in existing_objects
        if isinstance(row, dict)
    }
    for index in range(1, 1000):
        candidate = f"{object_type}_{index:03d}"
        if candidate not in existing:
            return candidate
    return f"{object_type}_999"


def _largest_existing_layout_object_id(objects: list[dict[str, Any]]) -> str | None:
    candidates: list[tuple[float, str]] = []
    for row in objects:
        bbox = row.get("bbox")
        if not isinstance(bbox, dict):
            continue
        object_id = row.get("instance_id") or row.get("object_id")
        if not isinstance(object_id, str) or not object_id.strip():
            continue
        try:
            width = float(bbox.get("max_x") or 0.0) - float(bbox.get("min_x") or 0.0)
            height = float(bbox.get("max_y") or 0.0) - float(bbox.get("min_y") or 0.0)
        except (TypeError, ValueError):
            continue
        candidates.append((max(0.0, width * height), object_id.strip()))
    if not candidates:
        return None
    return max(candidates, key=lambda row: row[0])[1]


def _preferred_existing_cluster_id(objects: list[dict[str, Any]]) -> str | None:
    for row in objects:
        cluster_id = row.get("cluster_id")
        if isinstance(cluster_id, str) and cluster_id.strip():
            return cluster_id.strip()
    return None


def _clamp_int(value: int, lower: int, upper: int) -> int:
    if upper < lower:
        return lower
    return max(lower, min(upper, value))


def _refill_judged_variant_accessories(
    *,
    bundle: dict[str, Any],
    room_output: dict[str, Any],
    variant_index: int,
    ablation_mode: AblationMode,
) -> dict[str, Any]:
    if skip_accessory_refill(ablation_mode):
        return bundle
    absolute_layout = bundle.get("absolute_layout")
    cluster_output = bundle.get("cluster_output")
    dropped_inventory = bundle.get("dropped_inventory_by_cluster")
    if not isinstance(absolute_layout, dict):
        return bundle
    if not isinstance(cluster_output, dict):
        return bundle

    updated_layout, request_refill_summary = (
        _ensure_requested_non_functional_contract_objects(
            absolute_layout=absolute_layout,
            room_output=room_output,
            cluster_output=cluster_output,
        )
    )
    working_bundle = bundle
    request_refill_count = int(request_refill_summary.get("added_count") or 0)
    if request_refill_count > 0:
        working_bundle = deepcopy(bundle)
        working_bundle["absolute_layout"] = updated_layout
        working_bundle["request_non_functional_refill_summary"] = request_refill_summary
        working_bundle["notes"] = _collect_unique_notes(
            working_bundle.get("notes"),
            [
                "Request Contract Refill added "
                f"{request_refill_count} non-blocking requested object(s) to final layout {variant_index}."
            ],
        )
        absolute_layout = updated_layout

    if not isinstance(dropped_inventory, dict) or not dropped_inventory:
        return working_bundle

    phase2_result = working_bundle.get("phase2_result")
    refined_layout_solution = (
        phase2_result.get("refined_layout_solution")
        if isinstance(phase2_result, dict)
        else None
    )
    refill_policy = decor_refill_policy(extract_style_policy(cluster_output))
    missing_contract_types = missing_functional_contract_types(
        contract=request_contract_from_payload(cluster_output),
        objects=[
            row for row in absolute_layout.get("objects") or [] if isinstance(row, dict)
        ],
    )
    if missing_contract_types:
        refill_policy["max_refills_total"] = 0
        refill_policy["disabled_reason"] = "functional_request_contract_unmet"
        refill_policy["missing_request_object_types"] = missing_contract_types

    updated_layout, refill_summary = controlled_accessory_refill(
        room_output=room_output,
        absolute_layout=absolute_layout,
        cluster_output=cluster_output,
        dropped_inventory_by_cluster=dropped_inventory,
        refined_layout_solution=(
            refined_layout_solution
            if isinstance(refined_layout_solution, dict)
            else None
        ),
        refill_policy=refill_policy,
        grid_mm=25,
    )
    updated_bundle = deepcopy(working_bundle)
    updated_bundle["absolute_layout"] = updated_layout
    updated_bundle["accessory_refill_summary"] = refill_summary
    refill_count = int(refill_summary.get("refill_count") or 0)
    if refill_count > 0:
        updated_bundle["notes"] = _collect_unique_notes(
            updated_bundle.get("notes"),
            [
                "Controlled Accessory Refill added "
                f"{refill_count} decor/accessory object(s) to final layout {variant_index}."
            ],
        )
    return updated_bundle


def _compose_status_only_feedback(
    compose_feasibility: dict[str, Any],
) -> dict[str, Any]:
    offenders = [
        deepcopy(item)
        for item in (compose_feasibility.get("offenders") or [])
        if isinstance(item, dict)
        and str(item.get("reason") or "")
        in {"COMPOSER_STATUS_NOT_OK", "CLUSTER_ENVELOPE_EXCEEDS_ROOM"}
    ]
    feedback = deepcopy(compose_feasibility)
    feedback["offenders"] = offenders
    feedback["feasible"] = not offenders
    feedback["stage"] = "composer"
    return feedback


def _compose_variant_bundle_from_tier_output(
    *,
    room_output: dict[str, Any],
    guidance_text: str,
    cluster_output: dict[str, Any],
    tier_output: dict[str, Any],
    composer: ClusterComposer,
    seed_notes: list[str] | None = None,
    max_attempts: int = 4,
) -> dict[str, Any] | None:
    max_attempts = _compose_attempt_limit(max_attempts)
    current_tier_output = deepcopy(tier_output)
    feedback_notes: list[str] = list(seed_notes or [])
    last_bundle: dict[str, Any] | None = None

    for _attempt in range(1, max(1, int(max_attempts)) + 1):
        merged_output = merge_cluster_outputs(cluster_output, current_tier_output)
        merged_output = _strip_raw_text(merged_output)

        cluster_results: dict[str, Any] = {}
        cluster_outlines: dict[str, Any] = {}
        for cluster in merged_output.get("clusters", []):
            if not isinstance(cluster, dict):
                continue
            cluster_id = str(cluster.get("cluster_id") or "")
            if not cluster_id:
                continue
            cluster_result = composer.generate(
                merged_clusters=merged_output,
                cluster_id=cluster_id,
                description=guidance_text,
                special_notes="",
            )
            cluster_result = _strip_raw_text(cluster_result)
            cluster_results[cluster_id] = cluster_result
            cluster_outlines[cluster_id] = _strip_raw_text(
                compute_cluster_outline(cluster_result)
            )

        compose_feasibility = evaluate_composed_cluster_feasibility(
            room_output=room_output,
            merged_output=merged_output,
            cluster_results=cluster_results,
            cluster_outlines=cluster_outlines,
        )
        status_feedback = _compose_status_only_feedback(compose_feasibility)
        last_bundle = {
            "cluster_output": deepcopy(cluster_output),
            "tier_output": deepcopy(current_tier_output),
            "merged_output": merged_output,
            "cluster_results": cluster_results,
            "cluster_outlines": cluster_outlines,
            "compose_feasibility": status_feedback,
            "notes": list(feedback_notes),
        }
        if status_feedback.get("feasible"):
            return last_bundle

        current_tier_output, backoff_notes, changed = apply_compose_backoff(
            tier_output=current_tier_output,
            merged_output=merged_output,
            feasibility=status_feedback,
        )
        feedback_notes = _collect_unique_notes(feedback_notes, backoff_notes)
        if not changed:
            break

    return (
        last_bundle
        if last_bundle and last_bundle.get("compose_feasibility", {}).get("feasible")
        else None
    )


def _build_composed_variant_bundles(
    *,
    room_output: dict[str, Any],
    guidance_text: str,
    cluster_output: dict[str, Any],
    tier_output: dict[str, Any],
    composer: ClusterComposer,
    target_variants: int,
    on_progress: Callable[[int, int, str], None] | None = None,
    max_seed_attempts: int = 20,
    max_compose_attempts: int = 4,
) -> list[dict[str, Any]]:
    max_seed_attempts = _seed_diversification_attempt_limit(max_seed_attempts)
    max_compose_attempts = _compose_attempt_limit(max_compose_attempts)
    target_count = max(1, int(target_variants))
    bundles: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()

    base_bundle = _compose_variant_bundle_from_tier_output(
        room_output=room_output,
        guidance_text=guidance_text,
        cluster_output=cluster_output,
        tier_output=tier_output,
        composer=composer,
        max_attempts=max_compose_attempts,
    )
    if base_bundle is None:
        return []

    base_bundle["source"] = "composer_bundle:1"
    base_bundle["reason"] = "Primary composer bundle built from the tier-count layout."
    base_bundle["family"] = "composer_bundle"
    base_bundle["seed_index"] = 1
    base_signature = _bundle_signature(base_bundle)
    if base_signature:
        bundles.append(base_bundle)
        seen_signatures.add(base_signature)
        if on_progress is not None:
            on_progress(
                len(bundles),
                target_count,
                _variant_progress_message(
                    "Composing cluster bundles",
                    len(bundles),
                    target_count,
                ),
            )

    current_bundle = deepcopy(base_bundle)
    attempts = 0
    while len(bundles) < target_count and attempts < max_seed_attempts:
        attempts += 1
        diversified_tier_output, diversify_notes, changed = (
            apply_variant_diversification(
                tier_output=current_bundle.get("tier_output") or {},
                merged_output=current_bundle.get("merged_output") or {},
                variant_index=len(bundles) + 1,
            )
        )
        if not changed:
            break

        seed_notes = _collect_unique_notes(current_bundle.get("notes"), diversify_notes)
        next_bundle = _compose_variant_bundle_from_tier_output(
            room_output=room_output,
            guidance_text=guidance_text,
            cluster_output=cluster_output,
            tier_output=diversified_tier_output,
            composer=composer,
            seed_notes=seed_notes,
            max_attempts=max_compose_attempts,
        )
        if next_bundle is None:
            continue

        signature = _bundle_signature(next_bundle)
        current_bundle = deepcopy(next_bundle)
        if not signature or signature in seen_signatures:
            continue

        next_index = len(bundles) + 1
        next_bundle["source"] = f"composer_bundle:{next_index}"
        next_bundle["reason"] = str(
            diversify_notes[0]
            if diversify_notes
            else f"Composer diversification seed {next_index}."
        )
        next_bundle["family"] = "composer_bundle"
        next_bundle["seed_index"] = next_index
        bundles.append(next_bundle)
        seen_signatures.add(signature)
        if on_progress is not None:
            on_progress(
                len(bundles),
                target_count,
                _variant_progress_message(
                    "Composing cluster bundles",
                    len(bundles),
                    target_count,
                ),
            )

    return bundles[:target_count]


def _plan_variant_bundle(
    *,
    bundle: dict[str, Any],
    room_output: dict[str, Any],
    guidance_text: str,
    relation_planner: ClusterRelationPlanner,
    variant_index: int,
) -> dict[str, Any] | None:
    try:
        cluster_outlines = bundle.get("cluster_outlines")
        if not isinstance(cluster_outlines, dict):
            return None
        relation_plan = relation_planner.generate(
            room_model_json=room_output,
            clusters_json=cluster_outlines,
            description=guidance_text,
            special_notes="",
        )
        relation_plan = _strip_raw_text(relation_plan)
        planned_bundle = deepcopy(bundle)
        planned_bundle["relation_plan"] = relation_plan
        planned_bundle["source"] = str(
            bundle.get("source") or f"composer_bundle:{variant_index}"
        )
        planned_bundle["reason"] = str(
            bundle.get("reason") or f"Composed variant bundle {variant_index}."
        )
        planned_bundle["seed_index"] = int(bundle.get("seed_index") or variant_index)
        return planned_bundle
    except Exception as exc:
        logger.warning(
            "Skipping composed variant bundle %s because relation planning failed: %s",
            int(variant_index),
            exc,
        )
        return None


def _solve_planned_variant_bundle(
    *,
    bundle: dict[str, Any],
    room_output: dict[str, Any],
    guidance_text: str,
    composer: ClusterComposer,
    solver: MacroClusterSolver,
    max_attempts: int = 8,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any] | None:
    max_attempts = _solver_attempt_limit(max_attempts)
    try:
        current_bundle = deepcopy(bundle)
        feedback_notes = list(current_bundle.get("notes") or [])
        semantic_recompose_used = False
        dropped_inventory_by_cluster = merge_dropped_inventory(
            deepcopy(current_bundle.get("dropped_inventory_by_cluster") or {}),
            collect_seed_omitted_inventory(
                tier_output=current_bundle.get("tier_output")
                if isinstance(current_bundle.get("tier_output"), dict)
                else {}
            ),
        )
        solver_relation_plan = current_bundle.get("relation_plan")
        if not isinstance(solver_relation_plan, dict):
            return None
        for _attempt in range(1, max(1, int(max_attempts)) + 1):
            cluster_outlines = current_bundle.get("cluster_outlines")
            cluster_output = current_bundle.get("cluster_output")
            merged_output = current_bundle.get("merged_output")
            tier_output = current_bundle.get("tier_output")
            if not isinstance(cluster_outlines, dict):
                return None
            if not isinstance(cluster_output, dict):
                return None
            if not isinstance(merged_output, dict):
                return None
            if not isinstance(tier_output, dict):
                return None

            current_bundle["solver_relation_plan"] = solver_relation_plan

            filtered_cluster_outlines = _filter_unsat_cluster_outlines(cluster_outlines)
            solver_output: dict[str, Any] | None = None
            solver_feasibility: dict[str, Any] | None = None
            if on_progress is not None:
                on_progress(
                    _attempt,
                    max_attempts,
                    f"solver attempt {_attempt}/{max_attempts}: evaluating this concept-guided macro layout.",
                )

            solver_output = solver.generate(
                room_model_json=room_output,
                clusters_outlines_json=filtered_cluster_outlines,
                relation_plan_json=solver_relation_plan,
                cluster_constraints_json=cluster_output,
                grid_mm=GLOBAL_LAYOUT_GRID_MM,
            )
            solver_output = _strip_raw_text(solver_output)
            solver_feasibility = evaluate_solver_cluster_feasibility(
                merged_output=merged_output,
                solver_output=solver_output,
            )
            current_bundle["solver_output"] = solver_output
            current_bundle["solver_feasibility"] = solver_feasibility

            if not isinstance(solver_output, dict) or not isinstance(
                solver_feasibility, dict
            ):
                return None

            current_bundle["solver_output"] = solver_output
            current_bundle["solver_feasibility"] = solver_feasibility
            if solver_feasibility.get("feasible"):
                current_bundle["notes"] = _collect_unique_notes(
                    feedback_notes,
                    solver_output.get("notes")
                    if isinstance(solver_output, dict)
                    else [],
                )
                current_bundle["dropped_inventory_by_cluster"] = deepcopy(
                    dropped_inventory_by_cluster
                )
                if on_progress is not None:
                    on_progress(
                        _attempt,
                        max_attempts,
                        f"solver attempt {_attempt}/{max_attempts}: concept-guided seed solved successfully.",
                    )
                return current_bundle

            offender_ids = [
                str(item.get("cluster_id") or "")
                for item in (solver_feasibility.get("offenders") or [])
                if isinstance(item, dict)
            ]
            if on_progress is not None:
                on_progress(
                    _attempt,
                    max_attempts,
                    "solver attempt "
                    f"{_attempt}/{max_attempts}: offending clusters "
                    f"{', '.join(offender_ids) if offender_ids else 'unknown'}.",
                )

            if _solver_feasibility_requires_semantic_recompose(solver_feasibility):
                if semantic_recompose_used:
                    if on_progress is not None:
                        on_progress(
                            _attempt,
                            max_attempts,
                            f"solver attempt {_attempt}/{max_attempts}: semantic candidate pool stayed empty after recompose; stopping this concept.",
                        )
                    break

                recomposed_bundle = _semantic_recompose_cluster_bundle(
                    room_output=room_output,
                    guidance_text=guidance_text,
                    composer=composer,
                    feasibility=solver_feasibility,
                    previous_bundle=current_bundle,
                )
                if recomposed_bundle is None:
                    if on_progress is not None:
                        on_progress(
                            _attempt,
                            max_attempts,
                            f"solver attempt {_attempt}/{max_attempts}: semantic recompose could not rebuild the offending clusters.",
                        )
                    break

                semantic_recompose_used = True
                current_bundle["merged_output"] = recomposed_bundle.get("merged_output")
                current_bundle["cluster_results"] = recomposed_bundle.get(
                    "cluster_results"
                )
                current_bundle["cluster_outlines"] = recomposed_bundle.get(
                    "cluster_outlines"
                )
                current_bundle["compose_feasibility"] = recomposed_bundle.get(
                    "compose_feasibility"
                )
                feedback_notes = _collect_unique_notes(
                    feedback_notes,
                    recomposed_bundle.get("notes"),
                    solver_output.get("notes")
                    if isinstance(solver_output, dict)
                    else [],
                )
                current_bundle["notes"] = list(feedback_notes)
                current_bundle["dropped_inventory_by_cluster"] = deepcopy(
                    dropped_inventory_by_cluster
                )
                if on_progress is not None:
                    on_progress(
                        _attempt,
                        max_attempts,
                        f"solver attempt {_attempt}/{max_attempts}: recomposed semantic variants for concept-policy empty candidate pools.",
                    )
                continue

            if not _solver_feasibility_allows_backoff(solver_feasibility):
                if on_progress is not None:
                    on_progress(
                        _attempt,
                        max_attempts,
                        f"solver attempt {_attempt}/{max_attempts}: stopping without shrink/drop because the exact assignment search has not reached concept-constrained UNSAT.",
                    )
                break

            next_tier_output, backoff_notes, changed = apply_compose_backoff(
                tier_output=tier_output,
                merged_output=merged_output,
                feasibility=solver_feasibility,
            )
            dropped_inventory = collect_dropped_inventory(
                previous_tier_output=tier_output,
                next_tier_output=next_tier_output,
                attempt=_attempt,
            )
            dropped_inventory_by_cluster = merge_dropped_inventory(
                dropped_inventory_by_cluster,
                dropped_inventory,
            )
            feedback_notes = _collect_unique_notes(
                feedback_notes,
                backoff_notes,
                solver_output.get("notes") if isinstance(solver_output, dict) else [],
            )
            if not changed:
                break

            remapped_bundle = _remap_cluster_geometry_bundle(
                room_output=room_output,
                cluster_output=cluster_output,
                tier_output=next_tier_output,
                previous_bundle=current_bundle,
            )
            if remapped_bundle is None:
                break

            current_bundle["tier_output"] = remapped_bundle.get("tier_output")
            current_bundle["merged_output"] = remapped_bundle.get("merged_output")
            current_bundle["cluster_results"] = remapped_bundle.get("cluster_results")
            current_bundle["cluster_outlines"] = remapped_bundle.get("cluster_outlines")
            current_bundle["compose_feasibility"] = remapped_bundle.get(
                "compose_feasibility"
            )
            current_bundle["notes"] = _collect_unique_notes(
                feedback_notes,
                remapped_bundle.get("notes"),
            )
            current_bundle["dropped_inventory_by_cluster"] = deepcopy(
                dropped_inventory_by_cluster
            )
            if on_progress is not None:
                on_progress(
                    _attempt,
                    max_attempts,
                    f"solver attempt {_attempt}/{max_attempts}: resized or reduced furniture, then remapped cluster outlines.",
                )

        current_bundle["dropped_inventory_by_cluster"] = deepcopy(
            dropped_inventory_by_cluster
        )
        return None
    except Exception as exc:
        logger.warning(
            "Skipping planned variant %s because solver loop failed: %s",
            int(bundle.get("seed_index") or 0),
            exc,
        )
        return None


def _seek_feasible_bundle_from_tier_output(
    *,
    room_output: dict[str, Any],
    guidance_text: str,
    cluster_output: dict[str, Any],
    tier_output: dict[str, Any],
    composer: ClusterComposer,
    relation_planner: ClusterRelationPlanner,
    solver: MacroClusterSolver,
    seed_notes: list[str] | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
    max_attempts: int = 8,
) -> dict[str, Any] | None:
    max_attempts = _solver_attempt_limit(max_attempts)
    current_tier_output = deepcopy(tier_output)
    feedback_notes: list[str] = list(seed_notes or [])
    last_bundle: dict[str, Any] | None = None
    seed_solver = _build_primary_seed_solver(solver)

    for attempt in range(1, max(1, int(max_attempts)) + 1):
        merged_output = merge_cluster_outputs(cluster_output, current_tier_output)
        merged_output = _strip_raw_text(merged_output)

        cluster_results: dict[str, Any] = {}
        cluster_outlines: dict[str, Any] = {}
        for cluster in merged_output.get("clusters", []):
            if not isinstance(cluster, dict):
                continue
            cluster_id = str(cluster.get("cluster_id") or "")
            if not cluster_id:
                continue
            cluster_result = composer.generate(
                merged_clusters=merged_output,
                cluster_id=cluster_id,
                description=guidance_text,
                special_notes="",
            )
            cluster_result = _strip_raw_text(cluster_result)
            cluster_results[cluster_id] = cluster_result
            cluster_outlines[cluster_id] = _strip_raw_text(
                compute_cluster_outline(cluster_result)
            )

        compose_feasibility = evaluate_composed_cluster_feasibility(
            room_output=room_output,
            merged_output=merged_output,
            cluster_results=cluster_results,
            cluster_outlines=cluster_outlines,
        )
        last_bundle = {
            "cluster_output": deepcopy(cluster_output),
            "tier_output": deepcopy(current_tier_output),
            "merged_output": merged_output,
            "cluster_results": cluster_results,
            "cluster_outlines": cluster_outlines,
            "compose_feasibility": compose_feasibility,
            "notes": list(feedback_notes),
        }

        active_feedback = compose_feasibility
        if compose_feasibility.get("feasible"):
            canonical_relation_plan = relation_planner.generate(
                room_model_json=room_output,
                clusters_json=cluster_outlines,
                description=guidance_text,
                special_notes="",
            )
            canonical_relation_plan = _strip_raw_text(canonical_relation_plan)
            solver_relation_plan = _build_solver_seed_relation_plan(
                canonical_relation_plan
            )
            filtered_cluster_outlines = _filter_unsat_cluster_outlines(cluster_outlines)
            canonical_solver_probe = seed_solver.generate(
                room_model_json=room_output,
                clusters_outlines_json=filtered_cluster_outlines,
                relation_plan_json=solver_relation_plan,
                cluster_constraints_json=cluster_output,
                grid_mm=GLOBAL_LAYOUT_GRID_MM,
            )
            canonical_solver_probe = _strip_raw_text(canonical_solver_probe)
            solver_feasibility = evaluate_solver_cluster_feasibility(
                merged_output=merged_output,
                solver_output=canonical_solver_probe,
            )
            last_bundle["canonical_relation_plan"] = canonical_relation_plan
            last_bundle["solver_relation_plan"] = solver_relation_plan
            last_bundle["canonical_solver_probe"] = canonical_solver_probe
            last_bundle["solver_feasibility"] = solver_feasibility
            if solver_feasibility.get("feasible"):
                if on_progress is not None:
                    on_progress(
                        attempt,
                        max_attempts,
                        _variant_progress_message(
                            "Composer feedback converged",
                            attempt,
                            max_attempts,
                        ),
                    )
                return last_bundle
            active_feedback = solver_feasibility

        offender_ids = [
            str(item.get("cluster_id") or "")
            for item in active_feedback.get("offenders") or []
            if isinstance(item, dict)
        ]
        feedback_stage = str(active_feedback.get("stage") or "compose_feedback")
        message = (
            f"{feedback_stage} attempt {attempt}/{max_attempts}: "
            f"offending clusters {', '.join(offender_ids) if offender_ids else 'unknown'}."
        )
        if on_progress is not None:
            on_progress(attempt, max_attempts, message)

        if feedback_stage == "solver_probe" and not _solver_feasibility_allows_backoff(
            active_feedback
        ):
            if on_progress is not None:
                on_progress(
                    attempt,
                    max_attempts,
                    f"{feedback_stage} attempt {attempt}/{max_attempts}: stopping without shrink/drop because this solver failure needs semantic recompose.",
                )
            break

        current_tier_output, backoff_notes, changed = apply_compose_backoff(
            tier_output=current_tier_output,
            merged_output=merged_output,
            feasibility=active_feedback,
        )
        feedback_notes.extend(backoff_notes)
        if not changed:
            break

    return (
        last_bundle
        if last_bundle
        and last_bundle.get("compose_feasibility", {}).get("feasible")
        and last_bundle.get("solver_feasibility", {}).get("feasible", True)
        else None
    )


def _build_feasible_variant_seed_bundles(
    *,
    feasible_base_bundle: dict[str, Any],
    room_output: dict[str, Any],
    guidance_text: str,
    manual_placements: list[dict[str, Any]] | None,
    cluster_output: dict[str, Any],
    tier_count: TierCountDirector,
    composer: ClusterComposer,
    relation_planner: ClusterRelationPlanner,
    solver: MacroClusterSolver,
    target_variants: int,
    on_progress: Callable[[int, int, str], None] | None = None,
    max_seed_attempts: int = 16,
) -> list[dict[str, Any]]:
    max_seed_attempts = _seed_diversification_attempt_limit(max_seed_attempts)
    _ = manual_placements
    bundles: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()

    seed_signature = _bundle_signature(feasible_base_bundle)
    if seed_signature:
        bundles.append(deepcopy(feasible_base_bundle))
        seen_signatures.add(seed_signature)

    current_bundle = deepcopy(feasible_base_bundle)
    attempts = 0
    while len(bundles) < max(1, int(target_variants)) and attempts < max_seed_attempts:
        attempts += 1
        diversified_tier_output, diversify_notes, changed = (
            apply_variant_diversification(
                tier_output=current_bundle.get("tier_output") or {},
                merged_output=current_bundle.get("merged_output") or {},
                variant_index=len(bundles) + 1,
            )
        )
        if not changed:
            break

        seed_notes = _collect_unique_notes(current_bundle.get("notes"), diversify_notes)
        next_bundle = _seek_feasible_bundle_from_tier_output(
            room_output=room_output,
            guidance_text=guidance_text,
            cluster_output=cluster_output,
            tier_output=diversified_tier_output,
            composer=composer,
            relation_planner=relation_planner,
            solver=solver,
            seed_notes=seed_notes,
            max_attempts=_solver_attempt_limit(8),
        )
        if next_bundle is None:
            if on_progress is not None:
                on_progress(
                    len(bundles),
                    target_variants,
                    f"Generating feasible variants {len(bundles)}/{target_variants}: diversification attempt {attempts} did not converge.",
                )
            continue

        signature = _bundle_signature(next_bundle)
        current_bundle = deepcopy(next_bundle)
        if not signature or signature in seen_signatures:
            if on_progress is not None:
                on_progress(
                    len(bundles),
                    target_variants,
                    f"Generating feasible variants {len(bundles)}/{target_variants}: diversification attempt {attempts} duplicated an existing variant seed.",
                )
            continue

        bundles.append(next_bundle)
        seen_signatures.add(signature)
        if on_progress is not None:
            on_progress(
                len(bundles),
                target_variants,
                _variant_progress_message(
                    "Generating feasible variants",
                    len(bundles),
                    target_variants,
                ),
            )

    return bundles[: max(1, int(target_variants))]


def _build_feasible_base_bundle(
    *,
    room_output: dict[str, Any],
    prepared_input_payload: dict[str, Any],
    guidance_text: str,
    manual_placements: list[dict[str, Any]] | None,
    cluster_output: dict[str, Any],
    tier_count: TierCountDirector,
    composer: ClusterComposer,
    relation_planner: ClusterRelationPlanner,
    solver: MacroClusterSolver,
    on_progress: Callable[[int, int, str], None] | None = None,
    max_attempts: int = 8,
) -> dict[str, Any] | None:
    tier_output = tier_count.generate(
        description=guidance_text,
        special_notes="",
        room_model_json=room_output,
        user_intent_json=prepared_input_payload,
        clusters_json=cluster_output,
    )
    tier_output = _strip_raw_text(tier_output)
    tier_output = apply_manual_placements_to_tier_output(
        tier_output,
        manual_placements,
        clusters_json=cluster_output,
    )
    return _seek_feasible_bundle_from_tier_output(
        room_output=room_output,
        guidance_text=guidance_text,
        cluster_output=cluster_output,
        tier_output=tier_output,
        composer=composer,
        relation_planner=relation_planner,
        solver=solver,
        on_progress=on_progress,
        max_attempts=max_attempts,
    )


def _run_canonical_screening_bundle(
    *,
    feasible_base_bundle: dict[str, Any],
    room_output: dict[str, Any],
    manual_placements: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    relation_plan = feasible_base_bundle.get("canonical_relation_plan")
    solver_output = feasible_base_bundle.get("canonical_solver_probe")
    merged_output = feasible_base_bundle.get("merged_output")
    cluster_outlines = feasible_base_bundle.get("cluster_outlines")
    cluster_output = feasible_base_bundle.get("cluster_output")
    if not isinstance(relation_plan, dict):
        return None
    if not isinstance(solver_output, dict):
        return None
    if not isinstance(merged_output, dict):
        return None
    if not isinstance(cluster_outlines, dict):
        return None
    if not isinstance(cluster_output, dict):
        return None

    preview_payload = build_phase2_payload(
        room_output,
        merged_output,
        cluster_outlines,
        relation_plan,
        solver_output,
        cluster_output,
    )
    preview_result = build_phase2_preview_candidate(
        payload=preview_payload,
        rounds=1,
        move_limit=24,
        note="Fast preview candidate for canonical macro branch.",
    )
    absolute_layout = _strip_raw_text(preview_result.get("absolute_layout") or {})
    absolute_layout = merge_manual_placements_into_absolute_layout(
        absolute_layout,
        manual_placements,
    )
    if str(absolute_layout.get("status") or "").upper() != "OK":
        return None

    notes = _collect_unique_notes(
        feasible_base_bundle.get("notes"),
        ["Canonical branch reused the feasible base package without extra bias."],
        (preview_result.get("proposal") or {}).get("notes")
        if isinstance(preview_result.get("proposal"), dict)
        else [],
        absolute_layout.get("notes") if isinstance(absolute_layout, dict) else [],
    )
    evaluation = preview_result.get("evaluation")
    hard_valid = _tool_hard_valid(evaluation)
    acceptable_valid = _tool_acceptable_valid(evaluation)
    absolute_layout = annotate_layout_coverage(
        absolute_layout=absolute_layout,
        merged_output=merged_output,
        relation_plan=relation_plan,
        cluster_outlines=cluster_outlines,
        hard_valid=hard_valid,
        acceptable_valid=acceptable_valid,
    )
    if absolute_layout.get("missing_cluster_ids"):
        notes.append(
            "Canonical screening bundle is incomplete: missing clusters "
            + ", ".join(absolute_layout["missing_cluster_ids"])
        )
    coverage = absolute_layout.get("coverage")
    return {
        "intent": None,
        "cluster_output": deepcopy(cluster_output),
        "tier_output": deepcopy(feasible_base_bundle.get("tier_output") or {}),
        "merged_output": deepcopy(merged_output),
        "cluster_results": deepcopy(feasible_base_bundle.get("cluster_results") or {}),
        "cluster_outlines": deepcopy(cluster_outlines),
        "relation_plan": deepcopy(relation_plan),
        "solver_output": deepcopy(solver_output),
        "phase2_result": None,
        "absolute_layout": absolute_layout,
        "source": "canonical_base",
        "reason": "Canonical branch built from the shared feasible base package.",
        "layout_score": _tool_candidate_score(evaluation),
        "hard_valid": hard_valid,
        "notes": notes,
        "family": "canonical_base",
        "promoted_payload": preview_result.get("payload") or {},
        "complete": bool(absolute_layout.get("complete", False)),
        "gallery_eligible": bool(absolute_layout.get("gallery_eligible", False)),
        "coverage_ratio": float(absolute_layout.get("coverage_ratio") or 0.0),
        "missing_cluster_ids": deepcopy(
            coverage.get("missing_cluster_ids") if isinstance(coverage, dict) else []
        ),
    }


def _run_intent_branch_screening_bundle(
    *,
    intent: dict[str, Any],
    feasible_base_bundle: dict[str, Any],
    room_output: dict[str, Any],
    base_guidance_text: str,
    manual_placements: list[dict[str, Any]] | None,
    relation_planner: ClusterRelationPlanner,
    solver: MacroClusterSolver,
) -> dict[str, Any] | None:
    intent_id = str(intent.get("intent_id") or "").strip()
    if not intent_id:
        return None

    try:
        branch_guidance = _append_initial_intent_guidance(base_guidance_text, intent)
        cluster_output = deepcopy(feasible_base_bundle.get("cluster_output") or {})
        tier_output = deepcopy(feasible_base_bundle.get("tier_output") or {})
        merged_output = deepcopy(feasible_base_bundle.get("merged_output") or {})
        cluster_results = deepcopy(feasible_base_bundle.get("cluster_results") or {})
        cluster_outlines = deepcopy(feasible_base_bundle.get("cluster_outlines") or {})
        base_notes = feasible_base_bundle.get("notes")
        relation_plan = relation_planner.generate(
            room_model_json=room_output,
            clusters_json=cluster_outlines,
            description=branch_guidance,
            special_notes="",
        )
        relation_plan = _strip_raw_text(relation_plan)

        filtered_cluster_outlines = _filter_unsat_cluster_outlines(cluster_outlines)
        preview_solver = MacroClusterSolver(
            tools_path=solver.tools_path,
            max_variants_per_cluster=min(int(solver.max_variants_per_cluster), 6),
            initial_candidates_per_cluster=min(
                int(solver.initial_candidates_per_cluster),
                24,
            ),
            max_rounds=min(int(solver.max_rounds), 2),
            time_limit_s=min(float(solver.time_limit_s), 8.0),
            num_workers=int(solver.num_workers),
        )
        solver_output = preview_solver.generate(
            room_model_json=room_output,
            clusters_outlines_json=filtered_cluster_outlines,
            relation_plan_json=relation_plan,
            cluster_constraints_json=cluster_output,
            grid_mm=GLOBAL_LAYOUT_GRID_MM,
        )
        solver_output = _strip_raw_text(solver_output)

        preview_payload = build_phase2_payload(
            room_output,
            merged_output,
            cluster_outlines,
            relation_plan,
            solver_output,
            cluster_output,
        )
        preview_result = build_phase2_preview_candidate(
            payload=preview_payload,
            rounds=1,
            move_limit=24,
            note=f"Fast preview candidate for intent branch {intent_id}.",
        )
        absolute_layout = _strip_raw_text(preview_result.get("absolute_layout") or {})
        absolute_layout = merge_manual_placements_into_absolute_layout(
            absolute_layout,
            manual_placements,
        )
        if str(absolute_layout.get("status") or "").upper() != "OK":
            return None

        label = str(intent.get("label") or intent_id).strip() or intent_id
        summary = str(intent.get("summary") or "").strip()
        notes: list[str] = [f"Intent branch: {label}."]
        if summary:
            notes.append(summary)
        notes = _collect_unique_notes(
            base_notes,
            notes,
            (preview_result.get("proposal") or {}).get("notes")
            if isinstance(preview_result.get("proposal"), dict)
            else [],
            absolute_layout.get("notes") if isinstance(absolute_layout, dict) else [],
        )
        evaluation = preview_result.get("evaluation")
        hard_valid = _tool_hard_valid(evaluation)
        acceptable_valid = _tool_acceptable_valid(evaluation)
        absolute_layout = annotate_layout_coverage(
            absolute_layout=absolute_layout,
            merged_output=merged_output,
            relation_plan=relation_plan,
            cluster_outlines=cluster_outlines,
            hard_valid=hard_valid,
            acceptable_valid=acceptable_valid,
        )
        if absolute_layout.get("missing_cluster_ids"):
            notes.append(
                "Screening bundle is incomplete: missing clusters "
                + ", ".join(absolute_layout["missing_cluster_ids"])
            )
        coverage = (
            absolute_layout.get("coverage") if isinstance(absolute_layout, dict) else {}
        )

        return {
            "intent": deepcopy(intent),
            "cluster_output": cluster_output,
            "tier_output": tier_output,
            "merged_output": merged_output,
            "cluster_results": cluster_results,
            "cluster_outlines": cluster_outlines,
            "relation_plan": relation_plan,
            "solver_output": solver_output,
            "phase2_result": None,
            "absolute_layout": absolute_layout,
            "source": f"intent_branch:{intent_id}",
            "reason": summary or f"Generated from initial intent branch {label}.",
            "layout_score": _tool_candidate_score(evaluation),
            "hard_valid": hard_valid,
            "notes": notes,
            "family": f"intent_bias:{intent_id}",
            "promoted_payload": preview_result.get("payload") or {},
            "complete": bool(absolute_layout.get("complete", False)),
            "gallery_eligible": bool(absolute_layout.get("gallery_eligible", False)),
            "coverage_ratio": float(absolute_layout.get("coverage_ratio") or 0.0),
            "missing_cluster_ids": deepcopy(
                coverage.get("missing_cluster_ids")
                if isinstance(coverage, dict)
                else []
            ),
        }
    except Exception as exc:
        logger.warning(
            "Skipping intent branch %s because preview screening failed: %s",
            intent_id,
            exc,
        )
        return None


def _judge_solved_variant_bundle(
    *,
    bundle: dict[str, Any],
    room_output: dict[str, Any],
    controller: Phase2Controller,
    manual_placements: list[dict[str, Any]] | None,
    variant_index: int,
) -> dict[str, Any] | None:
    try:
        merged_output = bundle.get("merged_output")
        cluster_outlines = bundle.get("cluster_outlines")
        cluster_output = bundle.get("cluster_output")
        relation_plan = bundle.get("relation_plan")
        solver_output = bundle.get("solver_output")
        if not isinstance(merged_output, dict):
            return None
        if not isinstance(cluster_outlines, dict):
            return None
        if not isinstance(cluster_output, dict):
            return None
        if not isinstance(relation_plan, dict):
            return None
        if not isinstance(solver_output, dict):
            return None

        phase2_result = controller.generate(
            room_interpreter_json=room_output,
            cluster_merged_json=merged_output,
            cluster_outlines_json=cluster_outlines,
            relation_plan_json=relation_plan,
            solver_output_json=solver_output,
            cluster_constraints_json=cluster_output,
        )
        phase2_result = _strip_raw_text(phase2_result)
        absolute_layout = _strip_raw_text(phase2_result.get("absolute_layout") or {})
        absolute_layout = merge_manual_placements_into_absolute_layout(
            absolute_layout,
            manual_placements,
        )
        if str(absolute_layout.get("status") or "").upper() != "OK":
            return None

        evaluation = phase2_result.get("tool_evaluation")
        hard_valid = _tool_hard_valid(evaluation)
        acceptable_valid = _tool_acceptable_valid(evaluation)
        absolute_layout = annotate_layout_coverage(
            absolute_layout=absolute_layout,
            merged_output=merged_output,
            relation_plan=relation_plan,
            cluster_outlines=cluster_outlines,
            hard_valid=hard_valid,
            acceptable_valid=acceptable_valid,
        )
        final_notes = _collect_unique_notes(
            bundle.get("notes"),
            (phase2_result.get("proposal") or {}).get("notes")
            if isinstance(phase2_result.get("proposal"), dict)
            else [],
            absolute_layout.get("notes") if isinstance(absolute_layout, dict) else [],
        )
        if absolute_layout.get("missing_cluster_ids"):
            final_notes = _collect_unique_notes(
                final_notes,
                [
                    "Final layout is incomplete: missing clusters "
                    + ", ".join(absolute_layout["missing_cluster_ids"])
                ],
            )

        coverage = (
            absolute_layout.get("coverage") if isinstance(absolute_layout, dict) else {}
        )
        judged_bundle = deepcopy(bundle)
        judged_bundle["phase2_result"] = phase2_result
        judged_bundle["absolute_layout"] = absolute_layout
        judged_bundle["layout_score"] = _tool_candidate_score(evaluation)
        judged_bundle["hard_valid"] = hard_valid
        judged_bundle["notes"] = final_notes
        judged_bundle["complete"] = bool(absolute_layout.get("complete", False))
        judged_bundle["gallery_eligible"] = bool(
            absolute_layout.get("gallery_eligible", False)
        )
        judged_bundle["coverage_ratio"] = float(
            absolute_layout.get("coverage_ratio") or 0.0
        )
        judged_bundle["missing_cluster_ids"] = deepcopy(
            coverage.get("missing_cluster_ids") if isinstance(coverage, dict) else []
        )
        judged_bundle["source"] = str(
            bundle.get("source") or f"loop_variant:{variant_index}"
        )
        judged_bundle["reason"] = str(
            bundle.get("reason")
            or f"Solver-complete variant {variant_index} passed the phase-2 judge."
        )
        return judged_bundle
    except Exception as exc:
        logger.warning(
            "Skipping solved variant %s because judge phase failed: %s",
            int(variant_index),
            exc,
        )
        return None


def _style_judged_variant_bundle(
    *,
    bundle: dict[str, Any],
    stylist: Stylist,
    stylist_user_context_json: dict[str, Any],
    stylist_tenant_id: str | None,
    manual_placements: list[dict[str, Any]] | None,
    variant_index: int,
) -> dict[str, Any] | None:
    try:
        absolute_layout = bundle.get("absolute_layout")
        if not isinstance(absolute_layout, dict):
            return None

        style_model_name = stylist_model_name_for_variant(variant_index)
        style_plan = stylist.generate_style_plan(
            layout_json=absolute_layout,
            user_context_json=stylist_user_context_json,
            tenant_id=stylist_tenant_id,
            model_name=style_model_name,
        )
        styled_result = stylist.apply_style_plan(
            layout_json=absolute_layout,
            user_context_json=stylist_user_context_json,
            tenant_id=stylist_tenant_id,
            style_plan=style_plan,
        )
        styled_result = _strip_raw_text(styled_result)
        styled_result = merge_manual_placements_into_styled_output(
            styled_result,
            manual_placements,
        )
        finalized_bundle = deepcopy(bundle)
        finalized_bundle["styled_result"] = styled_result
        finalized_bundle["source"] = str(
            bundle.get("source") or f"loop_variant:{variant_index}"
        )
        finalized_bundle["reason"] = str(
            bundle.get("reason")
            or f"Variant {variant_index} finalized through judge and stylist."
        )
        finalized_bundle["stylist_model_name"] = style_model_name
        return finalized_bundle
    except Exception as exc:
        logger.warning(
            "Skipping judged variant %s because styling failed: %s",
            int(variant_index),
            exc,
        )
        return None


def _finalize_feasible_variant_bundle(
    *,
    bundle: dict[str, Any],
    room_output: dict[str, Any],
    guidance_text: str,
    relation_planner: ClusterRelationPlanner,
    solver: MacroClusterSolver,
    controller: Phase2Controller,
    stylist: Stylist,
    stylist_user_context_json: dict[str, Any],
    stylist_tenant_id: str | None,
    manual_placements: list[dict[str, Any]] | None,
    variant_index: int,
) -> dict[str, Any] | None:
    try:
        merged_output = bundle.get("merged_output")
        cluster_outlines = bundle.get("cluster_outlines")
        cluster_output = bundle.get("cluster_output")
        if not isinstance(merged_output, dict):
            return None
        if not isinstance(cluster_outlines, dict):
            return None
        if not isinstance(cluster_output, dict):
            return None

        variant_guidance = guidance_text
        bundle_notes = bundle.get("notes")
        if isinstance(bundle_notes, list) and bundle_notes:
            variant_guidance = "\n\n".join(
                [
                    guidance_text.strip(),
                    "VARIANT_SEED_NOTES:\n"
                    + "\n".join(
                        f"- {str(note).strip()}"
                        for note in bundle_notes
                        if str(note).strip()
                    ),
                ]
            ).strip()

        relation_plan = relation_planner.generate(
            room_model_json=room_output,
            clusters_json=cluster_outlines,
            description=variant_guidance,
            special_notes="",
        )
        relation_plan = _strip_raw_text(relation_plan)

        filtered_cluster_outlines = _filter_unsat_cluster_outlines(cluster_outlines)
        solver_output = solver.generate(
            room_model_json=room_output,
            clusters_outlines_json=filtered_cluster_outlines,
            relation_plan_json=relation_plan,
            cluster_constraints_json=cluster_output,
            grid_mm=GLOBAL_LAYOUT_GRID_MM,
        )
        solver_output = _strip_raw_text(solver_output)

        phase2_result = controller.generate(
            room_interpreter_json=room_output,
            cluster_merged_json=merged_output,
            cluster_outlines_json=cluster_outlines,
            relation_plan_json=relation_plan,
            solver_output_json=solver_output,
            cluster_constraints_json=cluster_output,
        )
        phase2_result = _strip_raw_text(phase2_result)
        absolute_layout = _strip_raw_text(phase2_result.get("absolute_layout") or {})
        absolute_layout = merge_manual_placements_into_absolute_layout(
            absolute_layout,
            manual_placements,
        )
        if str(absolute_layout.get("status") or "").upper() != "OK":
            return None

        evaluation = phase2_result.get("tool_evaluation")
        hard_valid = _tool_hard_valid(evaluation)
        acceptable_valid = _tool_acceptable_valid(evaluation)
        absolute_layout = annotate_layout_coverage(
            absolute_layout=absolute_layout,
            merged_output=merged_output,
            relation_plan=relation_plan,
            cluster_outlines=cluster_outlines,
            hard_valid=hard_valid,
            acceptable_valid=acceptable_valid,
        )
        final_notes = _collect_unique_notes(
            bundle.get("notes"),
            (phase2_result.get("proposal") or {}).get("notes")
            if isinstance(phase2_result.get("proposal"), dict)
            else [],
            absolute_layout.get("notes") if isinstance(absolute_layout, dict) else [],
        )
        if absolute_layout.get("missing_cluster_ids"):
            final_notes = _collect_unique_notes(
                final_notes,
                [
                    "Final layout is incomplete: missing clusters "
                    + ", ".join(absolute_layout["missing_cluster_ids"])
                ],
            )
        coverage = (
            absolute_layout.get("coverage") if isinstance(absolute_layout, dict) else {}
        )
        finalized_bundle = deepcopy(bundle)
        finalized_bundle["solver_output"] = solver_output
        finalized_bundle["phase2_result"] = phase2_result
        finalized_bundle["absolute_layout"] = absolute_layout
        finalized_bundle["layout_score"] = _tool_candidate_score(evaluation)
        finalized_bundle["hard_valid"] = hard_valid
        finalized_bundle["notes"] = final_notes
        finalized_bundle["complete"] = bool(absolute_layout.get("complete", False))
        finalized_bundle["gallery_eligible"] = bool(
            absolute_layout.get("gallery_eligible", False)
        )
        finalized_bundle["coverage_ratio"] = float(
            absolute_layout.get("coverage_ratio") or 0.0
        )
        finalized_bundle["missing_cluster_ids"] = deepcopy(
            coverage.get("missing_cluster_ids") if isinstance(coverage, dict) else []
        )
        style_model_name = stylist_model_name_for_variant(variant_index)
        style_plan = stylist.generate_style_plan(
            layout_json=absolute_layout,
            user_context_json=stylist_user_context_json,
            tenant_id=stylist_tenant_id,
            model_name=style_model_name,
        )
        styled_result = stylist.apply_style_plan(
            layout_json=absolute_layout,
            user_context_json=stylist_user_context_json,
            tenant_id=stylist_tenant_id,
            style_plan=style_plan,
        )
        styled_result = _strip_raw_text(styled_result)
        styled_result = merge_manual_placements_into_styled_output(
            styled_result,
            manual_placements,
        )
        finalized_bundle["styled_result"] = styled_result
        finalized_bundle["source"] = str(
            bundle.get("source") or f"loop_variant:{variant_index}"
        )
        finalized_bundle["reason"] = str(
            bundle.get("reason")
            or f"Feasible loop variant {variant_index} finalized through planner and phase-2."
        )
        finalized_bundle["stylist_model_name"] = style_model_name
        return finalized_bundle
    except Exception as exc:
        logger.warning(
            "Skipping loop variant %s because finalization failed: %s",
            int(variant_index),
            exc,
        )
        return None


def _write_canonical_branch_artifacts(
    *,
    paths: CasePaths,
    bundle: dict[str, Any],
    absolute_layout: dict[str, Any],
    styled_output: dict[str, Any],
) -> None:
    cluster_output = bundle.get("cluster_output")
    if isinstance(cluster_output, dict):
        _write_module_json(
            paths,
            "cluster_forge",
            cluster_output,
            legacy_path=paths.cluster_forge,
        )

    tier_output = bundle.get("tier_output")
    if isinstance(tier_output, dict):
        _write_module_json(
            paths,
            "tier_count_director",
            tier_output,
            legacy_path=paths.tier_count,
        )

    merged_output = bundle.get("merged_output")
    if isinstance(merged_output, dict):
        _write_module_json(
            paths,
            "cluster_output_merger",
            merged_output,
            legacy_path=paths.cluster_merged,
        )

    cluster_results = bundle.get("cluster_results")
    if isinstance(cluster_results, dict):
        for cluster_id, cluster_result in cluster_results.items():
            if isinstance(cluster_id, str) and isinstance(cluster_result, dict):
                _write_json(paths.cluster_composer(cluster_id), cluster_result)
                _write_json(paths.module_cluster_composer(cluster_id), cluster_result)

    cluster_outlines = bundle.get("cluster_outlines")
    if isinstance(cluster_outlines, dict):
        for cluster_id, outline in cluster_outlines.items():
            if isinstance(cluster_id, str) and isinstance(outline, dict):
                _write_json(paths.cluster_outline(cluster_id), outline)
                _write_json(paths.module_cluster_outline(cluster_id), outline)
        _write_module_json(
            paths,
            "cluster_outline_bundle",
            cluster_outlines,
            legacy_path=paths.cluster_outlines_all,
        )

    relation_plan = bundle.get("relation_plan")
    if isinstance(relation_plan, dict):
        _write_module_json(
            paths,
            "cluster_relation_plan",
            relation_plan,
            legacy_path=paths.cluster_relation_plan,
        )
        _write_module_json(paths, "seed_concept_relation_plan", relation_plan)

    solver_output = bundle.get("solver_output")
    if isinstance(solver_output, dict):
        _write_module_json(
            paths,
            "macro_cluster_solver",
            solver_output,
            legacy_path=paths.cluster_solver,
        )

    dropped_inventory = bundle.get("dropped_inventory_by_cluster")
    if isinstance(dropped_inventory, dict):
        _write_module_json(
            paths,
            "macro_cluster_solver_dropped_inventory",
            dropped_inventory_payload(dropped_inventory),
            legacy_path=paths.solver_dropped_inventory,
        )

    phase2_result = bundle.get("phase2_result")
    if isinstance(phase2_result, dict):
        placer_output = _strip_raw_text(phase2_result.get("proposal") or {})
        _write_module_json(paths, "phase2_controller", phase2_result)
        _write_json(paths.cluster_placer, placer_output)

    _write_module_json(
        paths,
        "absolute_layout",
        absolute_layout,
        legacy_path=paths.absolute_layout,
    )
    accessory_refill_summary = bundle.get("accessory_refill_summary")
    if isinstance(accessory_refill_summary, dict):
        _write_module_json(
            paths,
            "controlled_accessory_refill",
            accessory_refill_summary,
            legacy_path=paths.accessory_refill,
        )
    _write_module_json(paths, "stylist", styled_output, legacy_path=paths.stylist)


def _strip_raw_text(payload: Any) -> Any:
    if isinstance(payload, dict):
        payload.pop("raw_text", None)
        for key, value in list(payload.items()):
            payload[key] = _strip_raw_text(value)
        return payload
    if isinstance(payload, list):
        return [_strip_raw_text(item) for item in payload]
    return payload


def _filter_unsat_cluster_outlines(
    clusters_outlines: dict[str, Any] | list[Any],
) -> dict[str, Any] | list[Any]:
    removed_ids: list[str] = []

    def is_unsat_status(value: Any) -> bool:
        return isinstance(value, str) and value.strip().upper() == "UNSAT"

    if isinstance(clusters_outlines, dict):
        clusters = clusters_outlines.get("clusters")
        if isinstance(clusters, list):
            kept = [
                item
                for item in clusters
                if not (isinstance(item, dict) and is_unsat_status(item.get("status")))
            ]
            for item in clusters:
                if isinstance(item, dict) and is_unsat_status(item.get("status")):
                    cid = item.get("cluster_id")
                    if isinstance(cid, str) and cid:
                        removed_ids.append(cid)
            out = dict(clusters_outlines)
            out["clusters"] = kept
            if removed_ids:
                logger.info(
                    "Orchestrator filtered UNSAT clusters: %s",
                    sorted(set(removed_ids)),
                )
            return out

        out: dict[str, Any] = {}
        for cid, payload in clusters_outlines.items():
            if isinstance(payload, dict) and is_unsat_status(payload.get("status")):
                if isinstance(cid, str) and cid:
                    removed_ids.append(cid)
                continue
            out[cid] = payload
        if removed_ids:
            logger.info(
                "Orchestrator filtered UNSAT clusters: %s",
                sorted(set(removed_ids)),
            )
        return out

    if isinstance(clusters_outlines, list):
        kept_list = [
            item
            for item in clusters_outlines
            if not (isinstance(item, dict) and is_unsat_status(item.get("status")))
        ]
        for item in clusters_outlines:
            if isinstance(item, dict) and is_unsat_status(item.get("status")):
                cid = item.get("cluster_id")
                if isinstance(cid, str) and cid:
                    removed_ids.append(cid)
        if removed_ids:
            logger.info(
                "Orchestrator filtered UNSAT clusters: %s",
                sorted(set(removed_ids)),
            )
        return kept_list

    return clusters_outlines


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    temp_path.replace(path)


def _write_module_json(
    paths: CasePaths,
    module_name: str,
    payload: Any,
    *,
    legacy_path: Path | None = None,
) -> None:
    _write_json(paths.module_output(module_name), payload)
    if legacy_path is not None:
        _write_json(legacy_path, payload)


def _write_module_io_manifest(paths: CasePaths) -> None:
    _write_json(paths.module_io_manifest, _module_io_manifest_payload(paths))


def _module_io_manifest_payload(paths: CasePaths) -> dict[str, Any]:
    return {
        "case_id": paths.case_id,
        "scope": "pipeline.orchestrator.run_case",
        "generated_at_utc": _now_utc_iso(),
        "modules": [
            {
                "module": spec.name,
                "implementation": spec.implementation,
                "inputs": list(spec.inputs),
                "output": spec.output,
                "output_artifact": spec.output_artifact,
                "legacy_artifact": spec.legacy_artifact,
            }
            for spec in ORCHESTRATOR_MODULE_SPECS
        ],
    }


def _update_status(
    paths: CasePaths,
    stage: str,
    *,
    error: str | None = None,
    message: str | None = None,
    progress_current: int | None = None,
    progress_total: int | None = None,
) -> None:
    existing_payload: dict[str, Any] = {}
    if paths.status.exists():
        try:
            existing_payload = json.loads(paths.status.read_text())
        except Exception:
            existing_payload = {}

    payload: dict[str, Any] = {
        "case_id": paths.case_id,
        "stage": stage,
        "updated_at_utc": _now_utc_iso(),
    }
    if error is not None:
        payload["error"] = str(error)
    if isinstance(message, str) and message.strip():
        payload["message"] = message.strip()
    if progress_current is not None:
        payload["progress_current"] = int(progress_current)
    if progress_total is not None:
        payload["progress_total"] = int(progress_total)
    actions = existing_payload.get("actions")
    action_history = (
        [item for item in actions if isinstance(item, dict)]
        if isinstance(actions, list)
        else []
    )
    action_message = (
        str(error).strip() if error is not None else str(message or stage).strip()
    )
    last_action = action_history[-1] if action_history else None
    next_action = {
        "stage": stage,
        "message": action_message,
        "updated_at_utc": payload["updated_at_utc"],
        "progress_current": payload.get("progress_current"),
        "progress_total": payload.get("progress_total"),
        "error": str(error).strip() if error is not None else None,
    }
    if not isinstance(last_action, dict) or any(
        last_action.get(key) != next_action.get(key)
        for key in (
            "stage",
            "message",
            "progress_current",
            "progress_total",
            "error",
        )
    ):
        action_history.append(next_action)
    payload["actions"] = action_history
    _write_json(paths.status, payload)
