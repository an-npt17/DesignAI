from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agent.tool_call_parser import extract_tool_calls as parse_tool_calls
from agent_schema.cluster_composer_schema import ClusterComposerOutput
from cluster_composer.outline import _outline_polygons_union_grid
from cluster_composer.tools import TOOL_REGISTRY, TOOL_SCHEMAS
from layout.grid_policy import normalize_layout_grid_mm
from layout.room_profiles.registry import (
    is_profile_floating_object,
    is_profile_storage_object,
    is_profile_workflow_object,
)
from layout.orientation_contract import (
    rotate_side_ccw_90s as _rotate_side_ccw_90s_contract,
)
from layout.orientation_contract import (
    side_to_vec as _side_to_vec_contract,
)
from layout.orientation_contract import (
    vec_to_side as _vec_to_side_contract,
)
from layout.variant_family import canonical_semantic_variant_family
from prompt.cluster_composer import CLUSTER_COMPOSER_PROMPT
from prompt.system import SYSTEM_PROMPT

logger = logging.getLogger(__name__)
FIXED_ACCESS_CLEARANCE_RATIO = 0.25
DEFAULT_TARGET_VALID_VARIANTS_PER_CLUSTER = 6
HARD_CAP_VARIANTS_PER_CLUSTER = 12
MAX_VARIANT_FAMILIES_PER_CLUSTER = 4
MIN_SEMANTIC_DISTANCE_BETWEEN_KEPT_VARIANTS = 0.25
MIN_POSE_DISTANCE_BETWEEN_KEPT_VARIANTS = 0.15
CORE_CLUSTER_FAMILY_FIDELITY_THRESHOLD = 0.60
SUPPORT_CLUSTER_FAMILY_FIDELITY_THRESHOLD = 0.50

ARCHETYPE_LIBRARY: dict[str, tuple[str, ...]] = {
    "sleep_core": (
        "headboard_wall_balanced",
        "headboard_wall_single_side",
        "bed_plus_storage_buffer",
        "bed_plus_window_side_bench",
    ),
    "sleep": (
        "headboard_wall_balanced",
        "headboard_wall_single_side",
        "bed_plus_storage_buffer",
        "bed_plus_window_side_bench",
    ),
    "living_media": (
        "media_facing",
        "wall_backed_focal",
        "focal_media",
    ),
    "living": (
        "media_facing",
        "conversation_facing",
        "window_oriented",
        "open_center",
    ),
    "seating": (
        "media_facing",
        "conversation_facing",
        "window_oriented",
        "open_center",
    ),
    "media": (
        "media_facing",
        "wall_backed_focal",
        "focal_media",
    ),
    "work_study": (
        "wall_desk",
        "daylight_desk",
        "compact_corner_work",
    ),
    "work": (
        "wall_desk",
        "daylight_desk",
        "compact_corner_work",
    ),
    "dining_core": (
        "centered_dining",
        "wall_shifted_dining",
        "hospitality_open_side",
    ),
    "dining": (
        "centered_dining",
        "wall_shifted_dining",
        "hospitality_open_side",
    ),
    "storage": (
        "wall_storage_linear",
        "storage_buffered_entry",
        "compact_storage_bank",
    ),
    "kitchen": (
        "wall_storage_linear",
        "compact_storage_bank",
        "edge_storage",
        "perimeter_storage",
    ),
}


@dataclass(frozen=True)
class SemanticSeedCandidate:
    family: str
    source_type: str
    placements: list[dict[str, Any]]
    family_fidelity: float
    semantic_confidence: float
    notes: list[str]


@dataclass(frozen=True)
class SemanticContext:
    cluster_id: str
    cluster_type: str
    semantic_role: str | None
    members: list[str]
    specs: dict[str, dict[str, Any]]
    roles: dict[str, str]
    role_ids: dict[str, list[str]]
    dominant_anchor_id: str | None
    required_object_ids: set[str]
    optional_object_ids: set[str]
    bundle_graph: dict[str, list[str]]
    grid_mm: int


@dataclass(frozen=True)
class LocalQualityBreakdown:
    functional_score: float
    naturalness_score: float
    semantic_coherence_score: float
    compactness_score: float
    family_fidelity_score: float
    awkwardness_penalty: float
    solver_friendliness_score: float
    split_cluster_penalty: float = 0.0
    awkward_grouping_penalty: float = 0.0
    fake_support_penalty: float = 0.0
    compaction_semantic_penalty: float = 0.0


@dataclass(frozen=True)
class FamilyContractResult:
    passed: bool
    family_fidelity: float
    semantic_confidence: float
    contract_reasons: list[str]


@dataclass(frozen=True)
class ClusterComposer:
    system_prompt: str = SYSTEM_PROMPT
    prompt_template: str = CLUSTER_COMPOSER_PROMPT

    def generate_raw(
        self,
        *,
        merged_clusters: dict[str, Any],
        cluster_id: str,
        description: str = "",
        special_notes: str = "",
        verifier_feedback_json: dict[str, Any] | None = None,
        max_steps: int = 30,
        access_clearance_ratio: float = FIXED_ACCESS_CLEARANCE_RATIO,
    ) -> str:
        access_clearance_ratio = FIXED_ACCESS_CLEARANCE_RATIO
        cluster = _find_cluster(merged_clusters, cluster_id)
        return _run_deterministic_controller(
            cluster=cluster,
            max_steps=max_steps,
            access_clearance_ratio=access_clearance_ratio,
            description=description,
            special_notes=special_notes,
            verifier_feedback_json=verifier_feedback_json,
        )

    def generate(
        self,
        *,
        merged_clusters: dict[str, Any],
        cluster_id: str,
        description: str = "",
        special_notes: str = "",
        verifier_feedback_json: dict[str, Any] | None = None,
        max_steps: int = 30,
        access_clearance_ratio: float = FIXED_ACCESS_CLEARANCE_RATIO,
    ) -> dict[str, Any]:
        access_clearance_ratio = FIXED_ACCESS_CLEARANCE_RATIO
        cluster = _find_cluster(merged_clusters, cluster_id)
        raw = self.generate_raw(
            merged_clusters=merged_clusters,
            cluster_id=cluster_id,
            description=description,
            special_notes=special_notes,
            verifier_feedback_json=verifier_feedback_json,
            max_steps=max_steps,
            access_clearance_ratio=access_clearance_ratio,
        )
        payload = _parse_json(raw)
        payload = _sanitize_cluster_output_payload(payload, cluster)
        return ClusterComposerOutput.model_validate(payload).model_dump()


# ---------------------------
# Prompt helpers
# ---------------------------


def _find_cluster(merged_clusters: dict[str, Any], cluster_id: str) -> dict[str, Any]:
    if not isinstance(merged_clusters, dict):
        raise ValueError("merged_clusters missing or invalid (expected dict).")
    if _cluster_id(merged_clusters) == cluster_id and (
        isinstance(merged_clusters.get("inventory_decision"), dict)
        or isinstance(merged_clusters.get("decisions"), list)
    ):
        return merged_clusters
    clusters = merged_clusters.get("clusters", [])
    if isinstance(clusters, list):
        for cluster in clusters:
            if isinstance(cluster, dict) and cluster.get("cluster_id") == cluster_id:
                return cluster
    available = [
        c.get("cluster_id")
        for c in clusters
        if isinstance(c, dict) and c.get("cluster_id")
    ]
    raise ValueError(f"cluster_id not found: {cluster_id}. available={available}")


def _build_prompt(
    *,
    template: str,
    description: str,
    special_notes: str,
    input_json: dict[str, Any],
    verifier_feedback_json: dict[str, Any] | None,
) -> str:
    mapping = {
        "DESCRIPTION": description or "",
        "SPECIAL_NOTES": special_notes or "",
        "INPUT_JSON": _json_block(input_json),
        "VERIFIER_FEEDBACK_JSON": _json_block(verifier_feedback_json),
    }
    out = template
    for k, v in mapping.items():
        out = out.replace("{" + k + "}", v)
    return out


def _json_block(obj: Any) -> str:
    if obj is None:
        return "null"
    return json.dumps(obj, ensure_ascii=True, indent=2)


# ---------------------------
# Core tool loop
# ---------------------------


def _run_deterministic_controller(
    *,
    cluster: dict[str, Any],
    max_steps: int,
    access_clearance_ratio: float,
    description: str,
    special_notes: str,
    verifier_feedback_json: dict[str, Any] | None,
) -> str:
    del description
    del special_notes
    del verifier_feedback_json

    payload = _generate_cluster_variant_bundle_payload(
        cluster=cluster,
        max_steps=max_steps,
        access_clearance_ratio=FIXED_ACCESS_CLEARANCE_RATIO,
    )
    return json.dumps(payload, ensure_ascii=True)


def _generate_cluster_variant_bundle_payload(
    *,
    cluster: dict[str, Any],
    max_steps: int,
    access_clearance_ratio: float,
) -> dict[str, Any]:
    member_ids = _member_ids(cluster)
    cluster_id = _cluster_id(cluster)
    if not member_ids:
        return {
            "status": "NEED_INFO",
            "cluster_id": cluster_id,
            "local_placements": [],
            "variant_bundle": [],
            "notes": ["Cluster has no members to compose."],
            "missing": ["members"],
            "conflicts": [],
        }

    ctx = _semantic_context(cluster)
    families = _select_variant_families(cluster, ctx)
    target_count = _target_variant_count(cluster)
    step_budget = max(1, int(max_steps))
    candidates = _collect_semantic_candidate_states(
        cluster=cluster,
        families=families,
        ctx=ctx,
    )

    valid_variants: list[dict[str, Any]] = []
    seen_placements: set[str] = set()
    conflicts: list[str] = []

    for candidate in candidates:
        canonical = _canonicalize_local_placements(candidate.placements)
        signature = _placements_signature(canonical)
        if not canonical or signature in seen_placements:
            continue
        seen_placements.add(signature)

        ok, detail = _validate_local_placements_full(canonical, member_ids)
        if not ok:
            conflicts.append(f"{candidate.family}: {detail}")
            continue

        eval_out = _run_verifier_once(
            cluster=cluster,
            placements=canonical,
            access_clearance_ratio=access_clearance_ratio,
        )
        if eval_out.get("result") != "VALID":
            repaired, repaired_eval, _ = _repair_invalid_seed_layout(
                cluster=cluster,
                access_clearance_ratio=access_clearance_ratio,
                placements=canonical,
                verifier_eval=eval_out,
                max_rounds=min(12, step_budget),
                family=candidate.family,
                ctx=ctx,
            )
            if repaired_eval.get("result") != "VALID":
                conflicts.extend(
                    _conflict_notes_for_family(candidate.family, repaired_eval)
                )
                continue
            canonical = repaired
            eval_out = repaired_eval

        improved, improved_eval, _ = _greedy_improve_valid_layout(
            cluster=cluster,
            access_clearance_ratio=access_clearance_ratio,
            placements=canonical,
            verifier_eval=eval_out,
            max_rounds=min(20, step_budget),
            family=candidate.family,
            ctx=ctx,
        )
        contract = _family_contract_validator(
            family=candidate.family,
            placements=improved,
            ctx=ctx,
            verifier_eval=improved_eval,
        )
        if not contract.passed:
            conflicts.append(
                f"{candidate.family}: family contract failed: "
                f"{'; '.join(contract.contract_reasons[:4])}"
            )
            continue

        fidelity_threshold = _family_fidelity_threshold(ctx)
        if (
            candidate.source_type == "family_native"
            and contract.family_fidelity < fidelity_threshold
        ):
            conflicts.append(
                f"{candidate.family}: family_fidelity "
                f"{contract.family_fidelity:.2f} below "
                f"{fidelity_threshold:.2f} threshold"
            )
            continue
        if candidate.source_type == "fallback_generic" and _is_core_cluster(ctx):
            conflicts.append(
                "fallback_generic: core cluster fallback reserved for debug/backoff, "
                "not solver output"
            )
            continue

        variant = _build_cluster_variant_payload(
            cluster=cluster,
            candidate=SemanticSeedCandidate(
                family=candidate.family,
                source_type=candidate.source_type,
                placements=improved,
                family_fidelity=contract.family_fidelity,
                semantic_confidence=contract.semantic_confidence,
                notes=[*candidate.notes, *contract.contract_reasons],
            ),
            verifier_eval=improved_eval,
            ctx=ctx,
            contract=contract,
        )
        valid_variants.append(variant)

    native_variants = [
        variant
        for variant in valid_variants
        if variant.get("source_type") == "family_native"
        and float(variant.get("family_fidelity") or 0.0)
        >= _family_fidelity_threshold(ctx)
    ]
    if native_variants:
        valid_variants = native_variants
    elif _is_core_cluster(ctx):
        return _semantic_fail_payload(
            cluster=cluster,
            conflicts=[
                *_dedupe_strings(conflicts)[:12],
                (
                    "Core cluster has no family_native variant at or above "
                    f"{CORE_CLUSTER_FAMILY_FIDELITY_THRESHOLD:.2f} fidelity; "
                    "upstream should back off inventory or regenerate a local family."
                ),
            ],
        )

    kept_variants = _select_diverse_variants(
        valid_variants,
        target_count=target_count,
    )
    for variant in kept_variants:
        variant["variant_id"] = f"{_safe_slug(cluster_id)}__{variant['variant_id']}"

    if not kept_variants:
        return {
            "status": "UNSAT",
            "cluster_id": cluster_id,
            "local_frame": {
                "unit": "mm",
                "grid_mm": _extract_grid_mm(cluster),
                "origin_note": "(0,0) is an arbitrary local origin for this cluster",
            },
            "local_placements": [],
            "cluster_footprint": {
                "type": "union_of_rects",
                "rects": [],
                "local_bbox": {"min_x": 0, "min_y": 0, "max_x": 0, "max_y": 0},
            },
            "variant_bundle": [],
            "notes": [
                "ClusterComposer deterministic semantic variant generation found no hard-valid layout."
            ],
            "missing": [],
            "conflicts": _dedupe_strings(conflicts)[:12],
        }

    primary = _select_canonical_variant(kept_variants)
    primary_output = _build_cluster_output_from_placements(
        cluster=cluster,
        placements=[
            item.model_dump() if hasattr(item, "model_dump") else item
            for item in primary["local_placements"]
        ],
        verifier_eval=None,
        notes=[
            "ClusterComposer used deterministic semantic archetypes instead of an LLM.",
            f"Kept {len(kept_variants)} diverse local variants for macro solving.",
        ],
        variant_family=str(primary.get("variant_family") or ""),
        family_fidelity=float(primary.get("family_fidelity") or 0.0),
    )
    primary_output["variant_bundle"] = kept_variants
    primary_output["canonical_variant_id"] = primary.get("variant_id")
    primary_output["canonical_variant_family"] = primary.get(
        "canonical_variant_family"
    ) or canonical_semantic_variant_family(primary.get("variant_family"))
    primary_output["variant_summary"] = _build_variant_summary(kept_variants)
    primary_output["family_coverage"] = _family_coverage(kept_variants, families)
    primary_output["source_type"] = primary.get("source_type")
    primary_output["family_fidelity"] = primary.get("family_fidelity")
    primary_output["semantic_confidence"] = primary.get("semantic_confidence")
    primary_output["fallback_heavy"] = primary.get("fallback_heavy")
    primary_output["solver_friendliness"] = primary.get("solver_friendliness")
    primary_output["family_contract_reasons"] = (
        primary.get("family_contract_reasons") or []
    )
    primary_output["conflicts"] = _dedupe_strings(conflicts)[:12]
    return primary_output


def _collect_semantic_candidate_states(
    *,
    cluster: dict[str, Any],
    families: list[str],
    ctx: SemanticContext,
) -> list[SemanticSeedCandidate]:
    native = _collect_family_native_candidates(
        cluster=cluster,
        families=families[:MAX_VARIANT_FAMILIES_PER_CLUSTER],
        ctx=ctx,
    )
    out = list(native)
    native_good = [
        candidate
        for candidate in native
        if candidate.family_fidelity >= _family_fidelity_threshold(ctx)
    ]
    fallback_threshold = max(1, min(len(families), 2))
    if (
        not native_good
        and len(native) < fallback_threshold
        and not _is_core_cluster(ctx)
    ):
        out.extend(_collect_fallback_generic_candidates(cluster=cluster, ctx=ctx))

    seen: set[tuple[str, str, str]] = set()
    deduped: list[SemanticSeedCandidate] = []
    for candidate in out:
        canonical = _canonicalize_local_placements(candidate.placements)
        key = (
            candidate.family,
            candidate.source_type,
            _placements_signature(canonical),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(
            SemanticSeedCandidate(
                family=candidate.family,
                source_type=candidate.source_type,
                placements=canonical,
                family_fidelity=_family_fidelity_score(candidate, ctx),
                semantic_confidence=candidate.semantic_confidence,
                notes=candidate.notes,
            )
        )
    return deduped[:96]


def _collect_family_native_candidates(
    *,
    cluster: dict[str, Any],
    families: list[str],
    ctx: SemanticContext,
) -> list[SemanticSeedCandidate]:
    out: list[SemanticSeedCandidate] = []
    for family in families:
        out.extend(_generate_family_seed_states(cluster, family, ctx))
    return out


def _collect_fallback_generic_candidates(
    *,
    cluster: dict[str, Any],
    ctx: SemanticContext,
) -> list[SemanticSeedCandidate]:
    out: list[SemanticSeedCandidate] = []
    for strategy in ("row", "compact_wrap", "balanced_wrap", "two_column"):
        placements = _build_fallback_pack_layout(cluster, strategy=strategy)
        if not placements:
            continue
        out.append(
            _seed_candidate(
                "fallback_generic",
                placements,
                "fallback_generic",
                ctx,
                notes=[f"Generic fallback pack layout using {strategy}."],
            )
        )
    return out


def _generate_family_seed_states(
    cluster: dict[str, Any],
    family: str,
    ctx: SemanticContext,
) -> list[SemanticSeedCandidate]:
    family_key = family.strip().lower()
    seeds: list[SemanticSeedCandidate] = []

    if family_key in {
        "media_facing",
        "wall_backed_focal",
        "focal_media",
        "conversation_facing",
        "window_oriented",
        "open_center",
    }:
        seeds.extend(_generate_living_family_states(cluster, family_key, ctx))
    elif family_key.startswith("headboard") or family_key.startswith("bed_plus"):
        seeds.extend(_generate_sleep_family_states(cluster, family_key, ctx))
    elif "desk" in family_key or "work" in family_key:
        seeds.extend(_generate_work_family_states(cluster, family_key, ctx))
    elif "dining" in family_key or "hospitality" in family_key:
        seeds.extend(_generate_dining_family_states(cluster, family_key, ctx))
    elif "storage" in family_key:
        seeds.extend(_generate_storage_family_states(cluster, family_key, ctx))

    out: list[SemanticSeedCandidate] = []
    seen: set[str] = set()
    for candidate in seeds:
        canonical = _seed_normalize_to_origin(candidate.placements)
        signature = _placements_signature(canonical)
        if not canonical or signature in seen:
            continue
        contract = _family_contract_validator(
            family=family_key,
            placements=canonical,
            ctx=ctx,
            verifier_eval=None,
        )
        if (
            not _validate_family_fidelity(family_key, canonical, ctx)
            or not contract.passed
            or contract.family_fidelity < _family_fidelity_threshold(ctx)
        ):
            continue
        seen.add(signature)
        out.append(
            SemanticSeedCandidate(
                family=family_key,
                source_type="family_native",
                placements=canonical,
                family_fidelity=contract.family_fidelity,
                semantic_confidence=contract.semantic_confidence,
                notes=[*candidate.notes, *contract.contract_reasons],
            )
        )
    return out


def _generate_living_family_states(
    cluster: dict[str, Any],
    family: str,
    ctx: SemanticContext,
) -> list[SemanticSeedCandidate]:
    composer_by_family = {
        "media_facing": _compose_media_facing_seed,
        "wall_backed_focal": _compose_wall_backed_focal_seed,
        "focal_media": _compose_focal_media_seed,
        "conversation_facing": _compose_conversation_seed,
        "window_oriented": _compose_window_oriented_seed,
        "open_center": _compose_open_center_seed,
    }
    composer = composer_by_family.get(family)
    if composer is None:
        return []
    seeds = [composer(cluster, ctx)]
    if family in {"conversation_facing", "open_center"} and seeds[0]:
        seeds.append(_mirror_seed_layout(seeds[0], cluster, axis="x"))
    if family == "window_oriented" and seeds[0]:
        seeds.append(_rotate_seed_layout(seeds[0], cluster, degrees=90))
    return [
        _seed_candidate(
            family,
            seed,
            "family_native",
            ctx,
            notes=[f"Native living-room semantic seed for {family}."],
        )
        for seed in seeds
        if seed
    ]


def _generate_sleep_family_states(
    cluster: dict[str, Any],
    family: str,
    ctx: SemanticContext,
) -> list[SemanticSeedCandidate]:
    composer_by_family = {
        "headboard_wall_balanced": _compose_headboard_balanced_seed,
        "headboard_wall_single_side": _compose_headboard_single_side_seed,
        "bed_plus_storage_buffer": _compose_bed_storage_buffer_seed,
        "bed_plus_window_side_bench": _compose_bed_window_bench_seed,
    }
    composer = composer_by_family.get(family)
    if composer is None:
        return []
    seed = composer(cluster, ctx)
    seeds = [seed]
    if seed and family in {"headboard_wall_single_side", "bed_plus_window_side_bench"}:
        seeds.append(_mirror_seed_layout(seed, cluster, axis="x"))
    return [
        _seed_candidate(
            family,
            item,
            "family_native",
            ctx,
            notes=[f"Native sleep semantic seed for {family}."],
        )
        for item in seeds
        if item and _bedside_access_ok(item, ctx)
    ]


def _generate_work_family_states(
    cluster: dict[str, Any],
    family: str,
    ctx: SemanticContext,
) -> list[SemanticSeedCandidate]:
    composer_by_family = {
        "wall_desk": _compose_wall_desk_seed,
        "daylight_desk": _compose_daylight_desk_seed,
        "compact_corner_work": _compose_compact_corner_work_seed,
    }
    composer = composer_by_family.get(family)
    if composer is None:
        return []
    seed = composer(cluster, ctx)
    seeds = [seed]
    if seed and family == "compact_corner_work":
        seeds.append(_rotate_seed_layout(seed, cluster, degrees=90))
    return [
        _seed_candidate(
            family,
            item,
            "family_native",
            ctx,
            notes=[f"Native work semantic seed for {family}."],
        )
        for item in seeds
        if item and _desk_chair_access_ok(item, ctx)
    ]


def _generate_dining_family_states(
    cluster: dict[str, Any],
    family: str,
    ctx: SemanticContext,
) -> list[SemanticSeedCandidate]:
    composer_by_family = {
        "centered_dining": _compose_centered_dining_seed,
        "wall_shifted_dining": _compose_wall_shifted_dining_seed,
        "hospitality_open_side": _compose_hospitality_open_side_seed,
    }
    composer = composer_by_family.get(family)
    if composer is None:
        return []
    seed = composer(cluster, ctx)
    seeds = [seed]
    if seed and family != "centered_dining":
        seeds.append(_mirror_seed_layout(seed, cluster, axis="x"))
    return [
        _seed_candidate(
            family,
            item,
            "family_native",
            ctx,
            notes=[f"Native dining semantic seed for {family}."],
        )
        for item in seeds
        if item and _dining_seat_ring_ok(item, ctx)
    ]


def _generate_storage_family_states(
    cluster: dict[str, Any],
    family: str,
    ctx: SemanticContext,
) -> list[SemanticSeedCandidate]:
    composer_by_family = {
        "wall_storage_linear": _compose_wall_storage_linear_seed,
        "storage_buffered_entry": _compose_storage_buffered_entry_seed,
        "compact_storage_bank": _compose_compact_storage_bank_seed,
    }
    composer = composer_by_family.get(family)
    if composer is None:
        return []
    seed = composer(cluster, ctx)
    seeds = [seed]
    if seed and family == "storage_buffered_entry":
        seeds.append(_mirror_seed_layout(seed, cluster, axis="x"))
    return [
        _seed_candidate(
            family,
            item,
            "family_native",
            ctx,
            notes=[f"Native storage semantic seed for {family}."],
        )
        for item in seeds
        if item and _storage_front_access_ok(item, ctx)
    ]


def _compose_media_facing_seed(
    cluster: dict[str, Any],
    ctx: SemanticContext,
) -> list[dict[str, Any]]:
    anchor_id = _living_anchor_id(ctx)
    if anchor_id is None:
        return []
    placements = _place_anchor_first(ctx, anchor_id, prefer_rot=0)
    anchor = placements.get(anchor_id)
    if anchor is None:
        return []
    front = _global_front_side(cluster, ctx, anchor_id, anchor)
    central_ids = _role_ids(ctx, "central_support")
    if central_ids:
        _place_object_on_side(
            cluster=cluster,
            ctx=ctx,
            placements=placements,
            object_id=central_ids[0],
            base_id=anchor_id,
            side=front,
            gap=max(300, ctx.grid_mm * 4),
            align="center",
            face_base=False,
        )
    for idx, oid in enumerate(_role_ids(ctx, "media_support", "media_anchor")):
        _place_object_on_side(
            cluster=cluster,
            ctx=ctx,
            placements=placements,
            object_id=oid,
            base_id=anchor_id,
            side=front,
            gap=max(1200, ctx.grid_mm * 16) + idx * ctx.grid_mm * 2,
            align="center",
            face_base=True,
        )
    _place_flanking_seats(cluster, ctx, placements, anchor_id, front)
    return _complete_semantic_seed(cluster, ctx, "media_facing", placements)


def _compose_wall_backed_focal_seed(
    cluster: dict[str, Any],
    ctx: SemanticContext,
) -> list[dict[str, Any]]:
    media_anchor_ids = _role_ids(ctx, "media_anchor", "media_support")
    anchor_id = media_anchor_ids[0] if media_anchor_ids else ctx.dominant_anchor_id
    if anchor_id is None:
        return []

    placements = _place_anchor_first(ctx, anchor_id, prefer_rot=0)
    anchor = placements.get(anchor_id)
    if anchor is None:
        return []

    front = _global_front_side(cluster, ctx, anchor_id, anchor)
    left = _rotate_cardinal_side(front, -1)
    right = _rotate_cardinal_side(front, 1)

    side_support_ids = [
        oid
        for oid in _role_ids(
            ctx, "side_support", "storage_support", "accessory_support"
        )
        if oid != anchor_id
    ]
    for idx, oid in enumerate(side_support_ids):
        _place_object_on_side(
            cluster=cluster,
            ctx=ctx,
            placements=placements,
            object_id=oid,
            base_id=anchor_id,
            side=left if idx % 2 == 0 else right,
            gap=max(50, ctx.grid_mm * 2),
            align="center",
            face_base=False,
        )

    return _complete_semantic_seed(cluster, ctx, "wall_backed_focal", placements)


def _compose_focal_media_seed(
    cluster: dict[str, Any],
    ctx: SemanticContext,
) -> list[dict[str, Any]]:
    media_anchor_ids = _role_ids(ctx, "media_anchor", "media_support")
    anchor_id = media_anchor_ids[0] if media_anchor_ids else ctx.dominant_anchor_id
    if anchor_id is None:
        return []

    placements = _place_anchor_first(ctx, anchor_id, prefer_rot=0)
    anchor = placements.get(anchor_id)
    if anchor is None:
        return []

    front = _global_front_side(cluster, ctx, anchor_id, anchor)
    left = _rotate_cardinal_side(front, -1)
    right = _rotate_cardinal_side(front, 1)

    support_ids = [
        oid
        for oid in _role_ids(
            ctx,
            "side_support",
            "storage_support",
            "accessory_support",
            "secondary_support",
        )
        if oid != anchor_id
    ]

    for idx, oid in enumerate(support_ids):
        _place_object_on_side(
            cluster=cluster,
            ctx=ctx,
            placements=placements,
            object_id=oid,
            base_id=anchor_id,
            side=left if idx % 2 == 0 else right,
            gap=max(100, ctx.grid_mm * 3),
            align="center",
            face_base=False,
        )

    return _complete_semantic_seed(cluster, ctx, "focal_media", placements)


def _compose_conversation_seed(
    cluster: dict[str, Any],
    ctx: SemanticContext,
) -> list[dict[str, Any]]:
    anchor_id = _living_anchor_id(ctx)
    if anchor_id is None:
        return []
    placements = _place_anchor_first(ctx, anchor_id, prefer_rot=0)
    anchor = placements.get(anchor_id)
    if anchor is None:
        return []
    front = _global_front_side(cluster, ctx, anchor_id, anchor)
    central_ids = _role_ids(ctx, "central_support")
    if central_ids:
        _place_object_on_side(
            cluster=cluster,
            ctx=ctx,
            placements=placements,
            object_id=central_ids[0],
            base_id=anchor_id,
            side=front,
            gap=max(250, ctx.grid_mm * 4),
            align="center",
            face_base=False,
        )
    opposite = _opposite_side(front)
    seat_ids = _role_ids(ctx, "secondary_anchor", "secondary_support")
    for idx, oid in enumerate(seat_ids):
        side = [
            opposite,
            _rotate_cardinal_side(front, -1),
            _rotate_cardinal_side(front, 1),
        ][idx % 3]
        _place_object_on_side(
            cluster=cluster,
            ctx=ctx,
            placements=placements,
            object_id=oid,
            base_id=anchor_id,
            side=side,
            gap=max(450, ctx.grid_mm * 6),
            align="center",
            face_base=True,
        )
    for oid in _role_ids(ctx, "media_support", "media_anchor"):
        _place_object_on_side(
            cluster=cluster,
            ctx=ctx,
            placements=placements,
            object_id=oid,
            base_id=anchor_id,
            side=_rotate_cardinal_side(front, 1),
            gap=max(900, ctx.grid_mm * 12),
            align="end",
            face_base=True,
        )
    return _complete_semantic_seed(cluster, ctx, "conversation_facing", placements)


def _compose_window_oriented_seed(
    cluster: dict[str, Any],
    ctx: SemanticContext,
) -> list[dict[str, Any]]:
    anchor_id = _living_anchor_id(ctx)
    if anchor_id is None:
        return []
    placements = _place_anchor_first(ctx, anchor_id, prefer_rot=90)
    anchor = placements.get(anchor_id)
    if anchor is None:
        return []
    front = _global_front_side(cluster, ctx, anchor_id, anchor)
    secondary = _role_ids(ctx, "secondary_anchor", "secondary_support")
    window_side = _rotate_cardinal_side(front, 1)
    for idx, oid in enumerate(secondary):
        _place_object_on_side(
            cluster=cluster,
            ctx=ctx,
            placements=placements,
            object_id=oid,
            base_id=anchor_id,
            side=window_side if idx == 0 else _opposite_side(window_side),
            gap=max(350, ctx.grid_mm * 5),
            align="center",
            face_base=False,
        )
    for oid in _role_ids(ctx, "central_support"):
        _place_object_on_side(
            cluster=cluster,
            ctx=ctx,
            placements=placements,
            object_id=oid,
            base_id=anchor_id,
            side=front,
            gap=max(250, ctx.grid_mm * 4),
            align="center",
            face_base=False,
        )
    return _complete_semantic_seed(cluster, ctx, "window_oriented", placements)


def _compose_open_center_seed(
    cluster: dict[str, Any],
    ctx: SemanticContext,
) -> list[dict[str, Any]]:
    anchor_id = _living_anchor_id(ctx)
    if anchor_id is None:
        return []
    placements = _place_anchor_first(ctx, anchor_id, prefer_rot=0)
    anchor = placements.get(anchor_id)
    if anchor is None:
        return []
    front = _global_front_side(cluster, ctx, anchor_id, anchor)
    edge_sides = [
        _rotate_cardinal_side(front, -1),
        _rotate_cardinal_side(front, 1),
        _opposite_side(front),
    ]
    edge_ids = [
        *_role_ids(ctx, "secondary_anchor", "secondary_support"),
        *_role_ids(ctx, "media_support", "media_anchor"),
        *_role_ids(ctx, "side_support", "accessory_support", "storage_support"),
    ]
    for idx, oid in enumerate(edge_ids):
        _place_object_on_side(
            cluster=cluster,
            ctx=ctx,
            placements=placements,
            object_id=oid,
            base_id=anchor_id,
            side=edge_sides[idx % len(edge_sides)],
            gap=max(600, ctx.grid_mm * 8),
            align="center",
            face_base=ctx.roles.get(oid) in {"secondary_anchor", "secondary_support"},
        )
    return _complete_semantic_seed(cluster, ctx, "open_center", placements)


def _compose_headboard_balanced_seed(
    cluster: dict[str, Any],
    ctx: SemanticContext,
) -> list[dict[str, Any]]:
    return _compose_sleep_seed(
        cluster=cluster,
        ctx=ctx,
        family="headboard_wall_balanced",
        side_sequence=["left", "right"],
        storage_sides=["right", "left"],
    )


def _compose_headboard_single_side_seed(
    cluster: dict[str, Any],
    ctx: SemanticContext,
) -> list[dict[str, Any]]:
    return _compose_sleep_seed(
        cluster=cluster,
        ctx=ctx,
        family="headboard_wall_single_side",
        side_sequence=["left"],
        storage_sides=["right", "bottom"],
    )


def _compose_bed_storage_buffer_seed(
    cluster: dict[str, Any],
    ctx: SemanticContext,
) -> list[dict[str, Any]]:
    return _compose_sleep_seed(
        cluster=cluster,
        ctx=ctx,
        family="bed_plus_storage_buffer",
        side_sequence=["left", "right"],
        storage_sides=["bottom", "right", "left"],
    )


def _compose_bed_window_bench_seed(
    cluster: dict[str, Any],
    ctx: SemanticContext,
) -> list[dict[str, Any]]:
    return _compose_sleep_seed(
        cluster=cluster,
        ctx=ctx,
        family="bed_plus_window_side_bench",
        side_sequence=["left", "right"],
        storage_sides=["top", "bottom"],
        bench_side="top",
    )


def _compose_sleep_seed(
    *,
    cluster: dict[str, Any],
    ctx: SemanticContext,
    family: str,
    side_sequence: list[str],
    storage_sides: list[str],
    bench_side: str | None = None,
) -> list[dict[str, Any]]:
    anchor_id = _first_matching_id(ctx.members, ("bed",)) or ctx.dominant_anchor_id
    if anchor_id is None:
        return []
    placements = _place_anchor_first(ctx, anchor_id, prefer_rot=0)
    side_ids = [
        oid
        for oid in ctx.members
        if oid != anchor_id and _object_kind(oid) in {"nightstand", "side_table"}
    ]
    for idx, oid in enumerate(side_ids):
        _place_object_on_side(
            cluster=cluster,
            ctx=ctx,
            placements=placements,
            object_id=oid,
            base_id=anchor_id,
            side=side_sequence[min(idx, len(side_sequence) - 1)],
            gap=ctx.grid_mm,
            align="center",
            face_base=False,
        )
    storage_ids = [
        oid
        for oid in ctx.members
        if oid not in placements
        and (
            ctx.roles.get(oid) == "storage_support"
            or any(
                token in _object_kind(oid)
                for token in ("bench", "dresser", "wardrobe", "closet")
            )
        )
    ]
    for idx, oid in enumerate(storage_ids):
        side = (
            bench_side
            if bench_side and "bench" in _object_kind(oid)
            else storage_sides[idx % len(storage_sides)]
        )
        _place_object_on_side(
            cluster=cluster,
            ctx=ctx,
            placements=placements,
            object_id=oid,
            base_id=anchor_id,
            side=side,
            gap=max(450, ctx.grid_mm * 6),
            align="center",
            face_base=False,
        )
    return _complete_semantic_seed(cluster, ctx, family, placements)


def _compose_wall_desk_seed(
    cluster: dict[str, Any],
    ctx: SemanticContext,
) -> list[dict[str, Any]]:
    return _compose_work_seed(cluster, ctx, "wall_desk", prefer_rot=0, chair_side=None)


def _compose_daylight_desk_seed(
    cluster: dict[str, Any],
    ctx: SemanticContext,
) -> list[dict[str, Any]]:
    return _compose_work_seed(
        cluster,
        ctx,
        "daylight_desk",
        prefer_rot=90,
        chair_side=None,
    )


def _compose_compact_corner_work_seed(
    cluster: dict[str, Any],
    ctx: SemanticContext,
) -> list[dict[str, Any]]:
    return _compose_work_seed(
        cluster,
        ctx,
        "compact_corner_work",
        prefer_rot=0,
        chair_side="top",
    )


def _compose_work_seed(
    cluster: dict[str, Any],
    ctx: SemanticContext,
    family: str,
    *,
    prefer_rot: int,
    chair_side: str | None,
) -> list[dict[str, Any]]:
    anchor_id = (
        _first_matching_id(ctx.members, ("desk", "work")) or ctx.dominant_anchor_id
    )
    if anchor_id is None:
        return []
    placements = _place_anchor_first(ctx, anchor_id, prefer_rot=prefer_rot)
    anchor = placements.get(anchor_id)
    if anchor is None:
        return []
    front = chair_side or _global_front_side(cluster, ctx, anchor_id, anchor)
    for oid in [
        mid for mid in ctx.members if mid != anchor_id and "chair" in _object_kind(mid)
    ]:
        _place_object_on_side(
            cluster=cluster,
            ctx=ctx,
            placements=placements,
            object_id=oid,
            base_id=anchor_id,
            side=front,
            gap=0,
            align="center",
            face_base=True,
        )
    return _complete_semantic_seed(cluster, ctx, family, placements)


def _compose_centered_dining_seed(
    cluster: dict[str, Any],
    ctx: SemanticContext,
) -> list[dict[str, Any]]:
    return _compose_dining_seed(
        cluster, ctx, "centered_dining", ["bottom", "top", "left", "right"]
    )


def _compose_wall_shifted_dining_seed(
    cluster: dict[str, Any],
    ctx: SemanticContext,
) -> list[dict[str, Any]]:
    return _compose_dining_seed(
        cluster, ctx, "wall_shifted_dining", ["top", "left", "right"]
    )


def _compose_hospitality_open_side_seed(
    cluster: dict[str, Any],
    ctx: SemanticContext,
) -> list[dict[str, Any]]:
    return _compose_dining_seed(
        cluster, ctx, "hospitality_open_side", ["bottom", "left", "right"]
    )


def _compose_dining_seed(
    cluster: dict[str, Any],
    ctx: SemanticContext,
    family: str,
    sides: list[str],
) -> list[dict[str, Any]]:
    anchor_id = (
        _first_matching_id(ctx.members, ("dining", "table")) or ctx.dominant_anchor_id
    )
    if anchor_id is None:
        return []
    placements = _place_anchor_first(ctx, anchor_id, prefer_rot=0)
    seats = [
        oid
        for oid in ctx.members
        if oid != anchor_id
        and any(token in _object_kind(oid) for token in ("chair", "bench", "stool"))
    ]
    for idx, oid in enumerate(seats):
        _place_object_on_side(
            cluster=cluster,
            ctx=ctx,
            placements=placements,
            object_id=oid,
            base_id=anchor_id,
            side=sides[idx % len(sides)],
            gap=max(ctx.grid_mm * 2, 150),
            align="center",
            face_base=True,
        )
    service_ids = [
        oid
        for oid in ctx.members
        if oid not in placements
        and any(
            token in _object_kind(oid)
            for token in ("sideboard", "buffet", "cart", "bar")
        )
    ]
    for oid in service_ids:
        _place_object_on_side(
            cluster=cluster,
            ctx=ctx,
            placements=placements,
            object_id=oid,
            base_id=anchor_id,
            side="top" if family != "wall_shifted_dining" else "bottom",
            gap=max(600, ctx.grid_mm * 8),
            align="center",
            face_base=False,
        )
    return _complete_semantic_seed(cluster, ctx, family, placements)


def _compose_wall_storage_linear_seed(
    cluster: dict[str, Any],
    ctx: SemanticContext,
) -> list[dict[str, Any]]:
    return _compose_storage_seed(cluster, ctx, "wall_storage_linear", "row")


def _compose_storage_buffered_entry_seed(
    cluster: dict[str, Any],
    ctx: SemanticContext,
) -> list[dict[str, Any]]:
    return _compose_storage_seed(cluster, ctx, "storage_buffered_entry", "buffer")


def _compose_compact_storage_bank_seed(
    cluster: dict[str, Any],
    ctx: SemanticContext,
) -> list[dict[str, Any]]:
    return _compose_storage_seed(cluster, ctx, "compact_storage_bank", "bank")


def _compose_storage_seed(
    cluster: dict[str, Any],
    ctx: SemanticContext,
    family: str,
    mode: str,
) -> list[dict[str, Any]]:
    anchor_id = _preferred_storage_anchor(ctx) or ctx.dominant_anchor_id
    if anchor_id is None:
        return []
    placements = _place_anchor_first(ctx, anchor_id, prefer_rot=0)
    ordered = [oid for oid in ctx.members if oid != anchor_id]
    if mode == "bank":
        sides = ["right", "top", "right", "top"]
        gap = max(ctx.grid_mm, 100)
    elif mode == "buffer":
        sides = ["bottom", "right", "left"]
        gap = max(350, ctx.grid_mm * 5)
    else:
        sides = ["right", "right", "right", "right"]
        gap = max(ctx.grid_mm, 150)
    base_id = anchor_id
    for idx, oid in enumerate(ordered):
        before = set(placements)
        _place_object_on_side(
            cluster=cluster,
            ctx=ctx,
            placements=placements,
            object_id=oid,
            base_id=base_id if mode == "row" else anchor_id,
            side=sides[idx % len(sides)],
            gap=gap,
            align="center",
            face_base=False,
        )
        if mode == "row" and set(placements) != before:
            base_id = oid
    return _complete_semantic_seed(cluster, ctx, family, placements)


def _role_ids(ctx: SemanticContext, *roles: str) -> list[str]:
    out: list[str] = []
    for role in roles:
        out.extend(ctx.role_ids.get(role, []))
    return [oid for oid in out if oid in ctx.members]


def _living_anchor_id(ctx: SemanticContext) -> str | None:
    candidates = [
        oid
        for oid in ctx.members
        if any(
            token in _object_kind(oid)
            for token in ("sofa", "sectional", "loveseat", "chair", "recliner")
        )
    ]
    return candidates[0] if candidates else ctx.dominant_anchor_id


def _preferred_storage_anchor(ctx: SemanticContext) -> str | None:
    return _first_matching_id(
        ctx.members,
        ("wardrobe", "closet", "shelf", "cabinet", "dresser", "storage"),
    )


def _place_flanking_seats(
    cluster: dict[str, Any],
    ctx: SemanticContext,
    placements: dict[str, dict[str, Any]],
    anchor_id: str,
    front: str,
) -> None:
    side_sequence = [
        _rotate_cardinal_side(front, -1),
        _rotate_cardinal_side(front, 1),
    ]
    for idx, oid in enumerate(_role_ids(ctx, "secondary_anchor", "secondary_support")):
        _place_object_on_side(
            cluster=cluster,
            ctx=ctx,
            placements=placements,
            object_id=oid,
            base_id=anchor_id,
            side=side_sequence[idx % len(side_sequence)],
            gap=max(300, ctx.grid_mm * 4),
            align="center",
            face_base=True,
        )


def _complete_semantic_seed(
    cluster: dict[str, Any],
    ctx: SemanticContext,
    family: str,
    placements: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    remaining = [oid for oid in ctx.members if oid not in placements]
    updated, unplaced = _place_leftover_objects(
        cluster=cluster,
        ctx=ctx,
        placements=placements,
        object_ids=remaining,
        family=family,
    )
    if unplaced:
        return []
    return _seed_normalize_to_origin(list(updated.values()))


def _seed_candidate(
    family: str,
    placements: list[dict[str, Any]],
    source_type: str,
    ctx: SemanticContext,
    notes: list[str] | None = None,
) -> SemanticSeedCandidate:
    canonical = _seed_normalize_to_origin(placements)
    candidate = SemanticSeedCandidate(
        family=family,
        source_type=source_type,
        placements=canonical,
        family_fidelity=0.0,
        semantic_confidence=0.92 if source_type == "family_native" else 0.45,
        notes=list(notes or []),
    )
    return SemanticSeedCandidate(
        family=family,
        source_type=source_type,
        placements=canonical,
        family_fidelity=_family_fidelity_score(candidate, ctx),
        semantic_confidence=candidate.semantic_confidence,
        notes=candidate.notes,
    )


def _validate_family_fidelity(
    family: str,
    placements: list[dict[str, Any]],
    ctx: SemanticContext,
) -> bool:
    if not placements:
        return False
    placed_ids = {p.get("id") for p in placements if isinstance(p, dict)}
    if not set(ctx.required_object_ids).issubset(placed_ids):
        return False
    return _score_family_fidelity(family, placements, ctx, {}) >= 0.45


def _family_contract_validator(
    *,
    family: str,
    placements: list[dict[str, Any]],
    ctx: SemanticContext,
    verifier_eval: dict[str, Any] | None,
) -> FamilyContractResult:
    canonical = _canonicalize_local_placements(placements)
    placed = {p.get("id") for p in canonical if isinstance(p, dict)}
    reasons: list[str] = []
    penalties = 0.0
    bonuses = 0.0

    if not canonical:
        return FamilyContractResult(False, 0.0, 0.0, ["no placements"])

    missing_required = sorted(ctx.required_object_ids - placed)
    if missing_required:
        reasons.append(f"missing required objects: {missing_required}")
        penalties += 0.35

    placement_by_id = {
        str(p["id"]): p
        for p in canonical
        if isinstance(p, dict) and isinstance(p.get("id"), str)
    }
    anchor_id = ctx.dominant_anchor_id
    anchor = placement_by_id.get(anchor_id or "")
    if anchor_id and anchor is None:
        reasons.append(f"dominant anchor not placed: {anchor_id}")
        penalties += 0.30
    elif anchor_id:
        bonuses += 0.08
    family_token = _normalize_token(family)
    if (
        _is_media_like_cluster(
            cluster_type=ctx.cluster_type,
            semantic_role=ctx.semantic_role,
            cluster_id=ctx.cluster_id,
            members=ctx.members,
        )
        and family_token == "window_oriented"
    ):
        reasons.append("media cluster cannot use window_oriented family")
        penalties += 0.50

    if anchor_id and anchor is not None:
        front = _global_front_side({}, ctx, anchor_id, anchor)
        support_penalty, support_reasons, support_bonus = _family_support_contract(
            family=family,
            ctx=ctx,
            placement_by_id=placement_by_id,
            anchor_id=anchor_id,
            anchor_front=front,
        )
        penalties += support_penalty
        bonuses += support_bonus
        reasons.extend(support_reasons)

    rects = _rects_for_context_placements(ctx, canonical)
    component_count = _cluster_component_count(
        rects,
        gap_tolerance=max(900, ctx.grid_mm * 16),
    )
    if component_count > 1:
        reasons.append(f"split cluster into {component_count} islands")
        penalties += min(0.35, 0.15 * float(component_count - 1))
    else:
        bonuses += 0.07

    if _family_breaks_center_reserve(family=family, rects=rects):
        reasons.append("center reserve occupied by family layout")
        penalties += 0.25

    if _verifier_has_error(verifier_eval, {"ACCESS_BLOCKED"}):
        reasons.append("important access zone blocked")
        penalties += 0.30
    elif isinstance(verifier_eval, dict) and verifier_eval.get("result") == "VALID":
        bonuses += 0.08

    base = _score_family_fidelity(family, canonical, ctx, verifier_eval or {})
    if _normalize_token(family) == "fallback_generic":
        reasons.append("fallback generic reserve candidate")
        base = min(base, 0.35)

    fidelity = _clamp01(base + bonuses - penalties)
    threshold = _family_fidelity_threshold(ctx)
    passed = not missing_required and fidelity >= min(threshold, 0.50)
    if _normalize_token(family) == "fallback_generic":
        passed = not missing_required and component_count <= 2

    if not reasons:
        reasons.append("family contract passed")
    confidence = _clamp01(0.35 + 0.65 * fidelity - min(0.25, penalties * 0.35))
    return FamilyContractResult(
        passed=passed,
        family_fidelity=fidelity,
        semantic_confidence=confidence,
        contract_reasons=_dedupe_strings(reasons),
    )


def _family_support_contract(
    *,
    family: str,
    ctx: SemanticContext,
    placement_by_id: dict[str, dict[str, Any]],
    anchor_id: str,
    anchor_front: str,
) -> tuple[float, list[str], float]:
    family_token = _normalize_token(family)
    reasons: list[str] = []
    penalty = 0.0
    bonus = 0.0

    central_ids = _placed_role_ids(ctx, placement_by_id, "central_support")
    if central_ids:
        for object_id in central_ids:
            side = _relative_side_between(ctx, anchor_id, object_id, placement_by_id)
            if side != anchor_front:
                reasons.append(f"central support {object_id} outside anchor use band")
                penalty += 0.16
            else:
                bonus += 0.08

    media_ids = _placed_role_ids(ctx, placement_by_id, "media_support", "media_anchor")
    if "media" in family_token and media_ids:
        for object_id in media_ids:
            side = _relative_side_between(ctx, anchor_id, object_id, placement_by_id)
            if side != anchor_front:
                reasons.append(f"media support {object_id} not in viewing band")
                penalty += 0.18
            else:
                bonus += 0.08

    secondary_ids = _placed_role_ids(
        ctx,
        placement_by_id,
        "secondary_anchor",
        "secondary_support",
        "side_support",
    )
    allowed_secondary = _family_allowed_secondary_sides(
        family_token=family_token,
        anchor_front=anchor_front,
    )
    for object_id in secondary_ids:
        side = _relative_side_between(ctx, anchor_id, object_id, placement_by_id)
        if side not in allowed_secondary:
            reasons.append(f"secondary support {object_id} drifted to {side}")
            penalty += 0.12
        else:
            bonus += 0.04

    workflow_ids = _placed_role_ids(ctx, placement_by_id, "workflow_anchor")
    if ("desk" in family_token or "work" in family_token) and workflow_ids:
        for object_id in workflow_ids:
            if object_id == anchor_id:
                continue
            side = _relative_side_between(ctx, anchor_id, object_id, placement_by_id)
            if side != anchor_front:
                reasons.append(f"workflow support {object_id} outside use band")
                penalty += 0.14
            else:
                bonus += 0.06

    return min(0.45, penalty), reasons, min(0.25, bonus)


def _placed_role_ids(
    ctx: SemanticContext,
    placement_by_id: dict[str, dict[str, Any]],
    *roles: str,
) -> list[str]:
    return [
        object_id
        for role in roles
        for object_id in ctx.role_ids.get(role, [])
        if object_id in placement_by_id
    ]


def _family_allowed_secondary_sides(
    *,
    family_token: str,
    anchor_front: str,
) -> set[str]:
    left = _rotate_cardinal_side(anchor_front, -1)
    right = _rotate_cardinal_side(anchor_front, 1)
    back = _opposite_side(anchor_front)
    if "media" in family_token:
        return {left, right, back}
    if "conversation" in family_token:
        return {left, right, back}
    if "open_center" in family_token:
        return {left, right, back}
    if "headboard" in family_token or "bed_plus" in family_token:
        return {left, right, back}
    if "dining" in family_token or "hospitality" in family_token:
        return {anchor_front, left, right, back}
    return {anchor_front, left, right, back}


def _relative_side_between(
    ctx: SemanticContext,
    base_id: str,
    object_id: str,
    placement_by_id: dict[str, dict[str, Any]],
) -> str:
    base_rect = _rect_for_placement(ctx=ctx, placement=placement_by_id[base_id])
    object_rect = _rect_for_placement(ctx=ctx, placement=placement_by_id[object_id])
    if base_rect is None or object_rect is None:
        return "unknown"
    base_cx = (base_rect[0] + base_rect[2]) / 2.0
    base_cy = (base_rect[1] + base_rect[3]) / 2.0
    object_cx = (object_rect[0] + object_rect[2]) / 2.0
    object_cy = (object_rect[1] + object_rect[3]) / 2.0
    dx = object_cx - base_cx
    dy = object_cy - base_cy
    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    return "top" if dy > 0 else "bottom"


def _cluster_component_count(
    rects: list[dict[str, Any]],
    *,
    gap_tolerance: int,
) -> int:
    if not rects:
        return 0
    remaining = set(range(len(rects)))
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            idx = stack.pop()
            connected = [
                other
                for other in list(remaining)
                if _rect_gap_distance(rects[idx], rects[other]) <= gap_tolerance
            ]
            for other in connected:
                remaining.remove(other)
                stack.append(other)
    return components


def _rect_gap_distance(first: dict[str, Any], second: dict[str, Any]) -> int:
    ax1 = int(first.get("x", 0))
    ay1 = int(first.get("y", 0))
    ax2 = ax1 + int(first.get("w", 0))
    ay2 = ay1 + int(first.get("h", 0))
    bx1 = int(second.get("x", 0))
    by1 = int(second.get("y", 0))
    bx2 = bx1 + int(second.get("w", 0))
    by2 = by1 + int(second.get("h", 0))
    gap_x = max(0, max(ax1, bx1) - min(ax2, bx2))
    gap_y = max(0, max(ay1, by1) - min(ay2, by2))
    return int(math.hypot(gap_x, gap_y))


def _family_breaks_center_reserve(
    *,
    family: str,
    rects: list[dict[str, Any]],
) -> bool:
    family_token = _normalize_token(family)
    if "open_center" not in family_token or not rects:
        return False
    bbox = _local_bbox_from_rects(rects)
    return _center_usage_signature(rects, bbox) == "occupied"


def _verifier_has_error(
    verifier_eval: dict[str, Any] | None,
    codes: set[str],
) -> bool:
    if not isinstance(verifier_eval, dict):
        return False
    errors = verifier_eval.get("errors")
    if not isinstance(errors, list):
        return False
    for error in errors:
        if isinstance(error, dict) and error.get("code") in codes:
            return True
    return False


def _is_core_cluster(ctx: SemanticContext) -> bool:
    tokens = [
        _normalize_token(ctx.cluster_id),
        _normalize_token(ctx.cluster_type),
        _normalize_token(ctx.semantic_role or ""),
    ]
    return any("core" in token or token in {"primary", "main"} for token in tokens)


def _family_fidelity_threshold(ctx: SemanticContext) -> float:
    if _is_core_cluster(ctx):
        return CORE_CLUSTER_FAMILY_FIDELITY_THRESHOLD
    return SUPPORT_CLUSTER_FAMILY_FIDELITY_THRESHOLD


def _semantic_fail_payload(
    *,
    cluster: dict[str, Any],
    conflicts: list[str],
) -> dict[str, Any]:
    return {
        "status": "SEMANTIC_FAIL",
        "cluster_id": _cluster_id(cluster),
        "local_frame": {
            "unit": "mm",
            "grid_mm": _extract_grid_mm(cluster),
            "origin_note": "(0,0) is an arbitrary local origin for this cluster",
        },
        "local_placements": [],
        "cluster_footprint": {
            "type": "union_of_rects",
            "rects": [],
            "local_bbox": {"min_x": 0, "min_y": 0, "max_x": 0, "max_y": 0},
            "tight_hull_polygon_mm": [],
            "tight_hull_polygons_mm": [],
            "interaction_hull_polygon_mm": [],
            "interaction_hull_polygons_mm": [],
        },
        "variant_bundle": [],
        "notes": [
            "ClusterComposer rejected the core candidate set by semantic family contract."
        ],
        "missing": [],
        "conflicts": _dedupe_strings(conflicts)[:12],
    }


def _family_fidelity_score(
    candidate: SemanticSeedCandidate,
    ctx: SemanticContext,
) -> float:
    if candidate.source_type == "fallback_generic":
        return min(
            0.35,
            _score_family_fidelity(candidate.family, candidate.placements, ctx, {}),
        )
    return _score_family_fidelity(candidate.family, candidate.placements, ctx, {})


def _bedside_access_ok(placements: list[dict[str, Any]], ctx: SemanticContext) -> bool:
    del ctx
    return bool(placements)


def _desk_chair_access_ok(
    placements: list[dict[str, Any]], ctx: SemanticContext
) -> bool:
    del ctx
    return bool(placements)


def _dining_seat_ring_ok(
    placements: list[dict[str, Any]], ctx: SemanticContext
) -> bool:
    del ctx
    return bool(placements)


def _storage_front_access_ok(
    placements: list[dict[str, Any]], ctx: SemanticContext
) -> bool:
    del ctx
    return bool(placements)


def _score_living_grouping_naturalness(
    family: str,
    placements: list[dict[str, Any]],
    ctx: SemanticContext,
) -> float:
    return _score_group_naturalness(family=family, placements=placements, ctx=ctx)


def _cluster_id(cluster: dict[str, Any]) -> str:
    direct = cluster.get("cluster_id") if isinstance(cluster, dict) else None
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    program = cluster.get("cluster_program") if isinstance(cluster, dict) else None
    if isinstance(program, dict):
        value = program.get("cluster_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown_cluster"


def _select_variant_families(
    cluster: dict[str, Any],
    ctx: SemanticContext | None = None,
) -> list[str]:
    policy = cluster.get("composer_policy") if isinstance(cluster, dict) else None
    cluster_type = (
        ctx.cluster_type if ctx is not None else _resolve_cluster_type(cluster)
    )
    semantic_role = (
        ctx.semantic_role if ctx is not None else _semantic_role_from_cluster(cluster)
    )

    if isinstance(policy, dict):
        raw = policy.get("variant_families")
        if isinstance(raw, list):
            families = [
                item.strip() for item in raw if isinstance(item, str) and item.strip()
            ]
            if families:
                families = _normalize_variant_families_for_cluster(
                    families,
                    cluster_type=cluster_type,
                    semantic_role=semantic_role,
                    cluster_id=_cluster_id(cluster),
                    members=(ctx.members if ctx is not None else _member_ids(cluster)),
                )
                return families[:MAX_VARIANT_FAMILIES_PER_CLUSTER]

    library = (
        cluster.get("cluster_archetype_library")
        if isinstance(cluster.get("cluster_archetype_library"), dict)
        else None
    )
    if isinstance(library, dict):
        raw = library.get("variant_families")
        if isinstance(raw, list):
            families = [
                item.strip() for item in raw if isinstance(item, str) and item.strip()
            ]
            if families:
                families = _normalize_variant_families_for_cluster(
                    families,
                    cluster_type=cluster_type,
                    semantic_role=semantic_role,
                    cluster_id=_cluster_id(cluster),
                    members=(ctx.members if ctx is not None else _member_ids(cluster)),
                )
                return families[:MAX_VARIANT_FAMILIES_PER_CLUSTER]

    role_token = _normalize_token(semantic_role or "")
    if role_token:
        if "kitchen" in role_token:
            cluster_type = "kitchen"
        elif "media" in role_token:
            cluster_type = "living_media"
        elif "social" in role_token or "conversation" in role_token:
            cluster_type = "living"
        elif "storage" in role_token or "entry" in role_token:
            cluster_type = "storage"
        elif "sleep" in role_token or "bed" in role_token:
            cluster_type = "sleep"
        elif "work" in role_token or "study" in role_token:
            cluster_type = "work"
        elif "dining" in role_token:
            cluster_type = "dining"

    families = list(
        ARCHETYPE_LIBRARY.get(cluster_type, ARCHETYPE_LIBRARY.get("living", ()))
    )
    families = _normalize_variant_families_for_cluster(
        families,
        cluster_type=cluster_type,
        semantic_role=semantic_role,
        cluster_id=_cluster_id(cluster),
        members=(ctx.members if ctx is not None else _member_ids(cluster)),
    )
    return families[:MAX_VARIANT_FAMILIES_PER_CLUSTER]


def _resolve_cluster_type(cluster: dict[str, Any]) -> str:
    program = cluster.get("cluster_program") if isinstance(cluster, dict) else None
    if isinstance(program, dict):
        value = program.get("cluster_type")
        if isinstance(value, str) and value.strip():
            token = _normalize_token(value)
            resolved = _cluster_type_from_token(token)
            if resolved is not None:
                return resolved

    value = cluster.get("cluster_type")
    if isinstance(value, str) and value.strip():
        token = _normalize_token(value)
        resolved = _cluster_type_from_token(token)
        if resolved is not None:
            return resolved

    inferred_type = _infer_cluster_type_from_structure(cluster)
    if inferred_type is not None:
        return inferred_type

    for key in ("semantic_role", "tag"):
        value = cluster.get(key)
        if isinstance(value, str) and value.strip():
            token = _normalize_token(value)
            resolved = _cluster_type_from_token(token)
            if resolved is not None:
                return resolved

    return "living"


def _infer_cluster_type(cluster: dict[str, Any]) -> str:
    return _resolve_cluster_type(cluster)


def _cluster_type_from_token(token: str) -> str | None:
    if token in ARCHETYPE_LIBRARY:
        return token
    if "kitchen" in token or any(
        marker in token
        for marker in ("fridge", "stove", "sink", "cooktop", "dishwasher")
    ):
        return "kitchen"
    if "media" in token:
        return "living_media"
    if "living" in token or "seating" in token or "social" in token:
        return "living"
    if "sleep" in token or "bed" in token:
        return "sleep"
    if "work" in token or "study" in token or "desk" in token:
        return "work"
    if "dining" in token or "meal" in token:
        return "dining"
    if "storage" in token or "closet" in token or "wardrobe" in token:
        return "storage"
    return None


def _semantic_role_from_cluster(cluster: dict[str, Any]) -> str | None:
    program = cluster.get("cluster_program") if isinstance(cluster, dict) else None
    if isinstance(program, dict):
        value = program.get("semantic_role") or program.get("role")
        if isinstance(value, str) and value.strip():
            return value.strip()
    value = cluster.get("semantic_role")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _is_media_like_cluster(
    *,
    cluster_type: str | None,
    semantic_role: str | None,
    cluster_id: str | None,
    members: Sequence[str] | None,
) -> bool:
    tokens = " ".join(
        [
            _normalize_token(cluster_type or ""),
            _normalize_token(semantic_role or ""),
            _normalize_token(cluster_id or ""),
            *[_normalize_token(item) for item in (members or [])],
        ]
    )
    if "media" in tokens or "tv" in tokens or "screen" in tokens:
        return True
    return False


def _normalize_variant_families_for_cluster(
    families: Sequence[str],
    *,
    cluster_type: str | None,
    semantic_role: str | None,
    cluster_id: str | None,
    members: Sequence[str] | None,
) -> list[str]:
    deduped: list[str] = []
    for family in families:
        if isinstance(family, str) and family.strip() and family.strip() not in deduped:
            deduped.append(family.strip())

    if not _is_media_like_cluster(
        cluster_type=cluster_type,
        semantic_role=semantic_role,
        cluster_id=cluster_id,
        members=members,
    ):
        return deduped

    allowed = ["media_facing", "wall_backed_focal", "focal_media"]
    filtered = [family for family in deduped if family in allowed]
    for family in allowed:
        if family not in filtered:
            filtered.append(family)
    return filtered


def _infer_cluster_type_from_structure(cluster: dict[str, Any]) -> str | None:
    scores = {
        "living": 0,
        "living_media": 0,
        "sleep": 0,
        "work": 0,
        "dining": 0,
        "storage": 0,
        "kitchen": 0,
    }
    for token in _cluster_structure_tokens(cluster):
        inferred = _infer_type_from_token(token)
        if inferred is None:
            continue
        score_type, weight = inferred
        scores[score_type] += weight

    ranked = sorted(
        ((score, cluster_type) for cluster_type, score in scores.items() if score > 0),
        key=lambda item: (-item[0], item[1]),
    )
    if not ranked:
        return None
    return ranked[0][1]


def _cluster_structure_tokens(cluster: dict[str, Any]) -> list[str]:
    tokens = [_normalize_token(_cluster_id(cluster))]
    for key in ("members", "anchors"):
        values = cluster.get(key)
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, str) and value.strip():
                tokens.append(_normalize_token(value))

    decisions = cluster.get("decisions")
    if isinstance(decisions, list):
        for row in decisions:
            if not isinstance(row, dict):
                continue
            for key in ("object_type", "category"):
                value = row.get(key)
                if isinstance(value, str) and value.strip():
                    tokens.append(_normalize_token(value))
    return [token for token in tokens if token]


def _infer_type_from_token(token: str) -> tuple[str, int] | None:
    if not token:
        return None
    if "kitchen" in token or any(
        marker in token
        for marker in (
            "fridge",
            "stove",
            "sink",
            "cooktop",
            "range_hood",
            "dishwasher",
        )
    ):
        return ("kitchen", 4)
    if "media" in token or token.startswith("tv_") or token == "tv":
        return ("living_media", 4)
    if "seat" in token or "living" in token:
        return ("living", 3)
    if "sleep" in token or "bed" in token:
        return ("sleep", 3)
    if "work" in token or "study" in token or "desk" in token:
        return ("work", 3)
    if "dining" in token:
        return ("dining", 3)
    if "storage" in token:
        return ("storage", 3)
    if any(
        marker in token
        for marker in (
            "sofa",
            "sectional",
            "armchair",
            "recliner",
            "loveseat",
            "ottoman",
            "coffee_table",
            "floor_lamp",
        )
    ):
        return ("living", 2)
    if any(
        marker in token for marker in ("nightstand", "headboard", "mattress", "dresser")
    ):
        return ("sleep", 2)
    if any(marker in token for marker in ("office_chair", "task_chair", "workstation")):
        return ("work", 2)
    if any(
        marker in token
        for marker in ("dining_table", "bar_stool", "sideboard", "buffet")
    ):
        return ("dining", 2)
    if any(
        marker in token
        for marker in (
            "bookshelf",
            "shoe_rack",
            "wardrobe",
            "closet",
            "cabinet",
        )
    ):
        return ("storage", 2)
    return None


def _target_variant_count(cluster: dict[str, Any]) -> int:
    policy = cluster.get("composer_policy") if isinstance(cluster, dict) else None
    raw = policy.get("target_variant_count") if isinstance(policy, dict) else None
    if not isinstance(raw, int):
        rules = cluster.get("cluster_rules") if isinstance(cluster, dict) else None
        raw = rules.get("target_variant_count") if isinstance(rules, dict) else None
    if not isinstance(raw, int):
        raw = DEFAULT_TARGET_VALID_VARIANTS_PER_CLUSTER
    return max(1, min(int(raw), HARD_CAP_VARIANTS_PER_CLUSTER))


def _build_cluster_variant_payload(
    *,
    cluster: dict[str, Any],
    candidate: SemanticSeedCandidate,
    verifier_eval: dict[str, Any],
    ctx: SemanticContext,
    contract: FamilyContractResult,
) -> dict[str, Any]:
    base = _build_cluster_output_from_placements(
        cluster=cluster,
        placements=candidate.placements,
        verifier_eval=verifier_eval,
        notes=[],
        variant_family=candidate.family,
        family_fidelity=candidate.family_fidelity,
    )
    rects = (base.get("cluster_footprint") or {}).get("rects")
    rects = rects if isinstance(rects, list) else []
    bbox = _local_bbox_from_rects(rects)
    body_polygons = _outline_polygons_union_grid(rects)
    body_polygon = body_polygons[0] if len(body_polygons) == 1 else []
    interaction_rects = _interaction_rects_from_verifier(verifier_eval)
    interaction_polygons = _outline_polygons_union_grid(interaction_rects)
    interaction_polygon = (
        interaction_polygons[0] if len(interaction_polygons) == 1 else []
    )
    quality = _local_quality_from_eval(
        family=candidate.family,
        verifier_eval=verifier_eval,
        rects=rects,
        placements=base.get("local_placements") or [],
        ctx=ctx,
    )
    fallback_heavy = candidate.source_type == "fallback_generic"

    return {
        "variant_id": "",
        "variant_family": candidate.family,
        "canonical_variant_family": canonical_semantic_variant_family(candidate.family),
        "source_type": candidate.source_type,
        "family_fidelity": _clamp01(contract.family_fidelity),
        "semantic_confidence": _clamp01(contract.semantic_confidence),
        "fallback_heavy": fallback_heavy,
        "solver_friendliness": quality.solver_friendliness_score,
        "semantic_signature": _semantic_signature_for_variant(
            cluster=cluster,
            family=candidate.family,
            placements=base.get("local_placements") or [],
            ctx=ctx,
            rects=rects,
            wall_contacts=_wall_contact_edges(rects, bbox),
            access_zones=_required_access_zones_from_verifier(verifier_eval),
        ),
        "local_placements": base.get("local_placements") or [],
        "interaction_placements": interaction_rects,
        "tight_hull_polygon_mm": body_polygon,
        "tight_hull_polygons_mm": body_polygons,
        "interaction_hull_polygon_mm": interaction_polygon,
        "interaction_hull_polygons_mm": interaction_polygons,
        "family_contract_reasons": contract.contract_reasons,
        "local_bbox_mm": {
            "min": (int(bbox["min_x"]), int(bbox["min_y"])),
            "max": (int(bbox["max_x"]), int(bbox["max_y"])),
        },
        "wall_contact_edges": _wall_contact_edges(rects, bbox),
        "required_access_zones": _required_access_zones_from_verifier(verifier_eval),
        "local_quality": quality.__dict__,
        "hard_valid": verifier_eval.get("result") == "VALID",
        "_rank": _variant_rank_key(
            quality.__dict__,
            verifier_eval,
            fallback_heavy=fallback_heavy,
        ),
        "_pose_signature": _normalized_pose_signature(
            base.get("local_placements") or [],
            bbox=bbox,
        ),
        "_hull_signature": _hull_shape_signature(rects, bbox),
        "_contact_signature": _contact_pattern_signature(
            _wall_contact_edges(rects, bbox)
        ),
        "_center_signature": _center_usage_signature(rects, bbox),
    }


def _select_diverse_variants(
    variants: list[dict[str, Any]],
    *,
    target_count: int,
) -> list[dict[str, Any]]:
    ranked = sorted(variants, key=lambda item: item.get("_rank", ()))
    selected = _select_best_per_family(ranked, target_count=target_count)
    if len(selected) < target_count:
        selected.extend(
            _select_cross_family_diverse(
                ranked,
                selected=selected,
                target_count=target_count,
            )
        )

    family_counts: dict[str, int] = {}
    cleaned: list[dict[str, Any]] = []
    for variant in selected[:target_count]:
        item = dict(variant)
        family = str(item.get("variant_family") or "variant")
        family_counts[family] = family_counts.get(family, 0) + 1
        item["variant_id"] = f"{_safe_slug(family)}__{family_counts[family]:02d}"
        item.pop("_rank", None)
        item.pop("_pose_signature", None)
        item.pop("_hull_signature", None)
        item.pop("_contact_signature", None)
        item.pop("_center_signature", None)
        cleaned.append(item)
    return cleaned


def _select_canonical_variant(variants: list[dict[str, Any]]) -> dict[str, Any]:
    if not variants:
        return {}
    return sorted(
        variants,
        key=lambda item: (
            bool(item.get("fallback_heavy")),
            -float(item.get("family_fidelity") or 0.0),
            -float(item.get("semantic_confidence") or 0.0),
            -float(
                (
                    (item.get("local_quality") or {})
                    if isinstance(item.get("local_quality"), dict)
                    else {}
                ).get(
                    "semantic_coherence_score",
                    0.0,
                )
            ),
            str(item.get("variant_id") or ""),
        ),
    )[0]


def _build_variant_summary(variants: list[dict[str, Any]]) -> dict[str, Any]:
    families: dict[str, int] = {}
    fallback_count = 0
    best_quality = 0.0
    for variant in variants:
        family = str(variant.get("variant_family") or "unknown")
        families[family] = families.get(family, 0) + 1
        if variant.get("fallback_heavy"):
            fallback_count += 1
        quality = variant.get("local_quality")
        if isinstance(quality, dict):
            best_quality = max(
                best_quality,
                float(quality.get("semantic_coherence_score") or 0.0),
            )
    return {
        "variant_count": len(variants),
        "families": families,
        "fallback_variant_count": fallback_count,
        "best_semantic_coherence_score": round(best_quality, 4),
    }


def _family_coverage(
    variants: list[dict[str, Any]],
    requested_families: list[str],
) -> dict[str, Any]:
    present = {
        str(variant.get("variant_family") or "")
        for variant in variants
        if variant.get("source_type") != "fallback_generic"
    }
    requested = [family for family in requested_families if family]
    return {
        "requested": requested,
        "covered": sorted(family for family in requested if family in present),
        "missing": sorted(family for family in requested if family not in present),
    }


def _select_best_per_family(
    variants: list[dict[str, Any]],
    *,
    target_count: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_families: set[str] = set()
    for variant in variants:
        if len(selected) >= target_count:
            break
        if variant.get("source_type") == "fallback_generic":
            continue
        family = str(variant.get("variant_family") or "")
        if family in seen_families:
            continue
        selected.append(variant)
        seen_families.add(family)
    return selected


def _select_cross_family_diverse(
    variants: list[dict[str, Any]],
    *,
    selected: list[dict[str, Any]],
    target_count: int,
) -> list[dict[str, Any]]:
    additions: list[dict[str, Any]] = []
    working = list(selected)
    native = [
        item for item in variants if item.get("source_type") != "fallback_generic"
    ]
    fallback = [
        item for item in variants if item.get("source_type") == "fallback_generic"
    ]
    for pool in (native, fallback):
        for variant in pool:
            if len(working) >= target_count:
                return additions
            if variant in working:
                continue
            if not _is_diverse_variant(variant, working):
                continue
            working.append(variant)
            additions.append(variant)
    if len(working) < target_count:
        for variant in variants:
            if len(working) >= target_count:
                break
            if variant in working:
                continue
            working.append(variant)
            additions.append(variant)
    return additions


def _is_diverse_variant(
    candidate: dict[str, Any],
    selected: list[dict[str, Any]],
) -> bool:
    candidate_signature = set(candidate.get("semantic_signature") or [])
    candidate_pose = candidate.get("_pose_signature")
    for existing in selected:
        existing_signature = set(existing.get("semantic_signature") or [])
        semantic_distance = _jaccard_distance(candidate_signature, existing_signature)
        pose_distance = _pose_distance(
            candidate_pose,
            existing.get("_pose_signature"),
        )
        hull_distance = _hull_shape_distance(
            candidate.get("_hull_signature"),
            existing.get("_hull_signature"),
        )
        contact_distance = _contact_pattern_distance(
            candidate.get("_contact_signature"),
            existing.get("_contact_signature"),
        )
        center_distance = _center_usage_distance(
            candidate.get("_center_signature"),
            existing.get("_center_signature"),
        )
        same_family = candidate.get("variant_family") == existing.get("variant_family")
        if same_family and pose_distance < MIN_POSE_DISTANCE_BETWEEN_KEPT_VARIANTS:
            return False
        if semantic_distance < MIN_SEMANTIC_DISTANCE_BETWEEN_KEPT_VARIANTS:
            pattern_distance = (
                hull_distance + contact_distance + center_distance
            ) / 3.0
            if (
                pose_distance < MIN_POSE_DISTANCE_BETWEEN_KEPT_VARIANTS
                and pattern_distance < 0.20
            ):
                return False
    return True


def _semantic_context(cluster: dict[str, Any]) -> SemanticContext:
    specs = {
        spec["id"]: spec
        for spec in _build_object_specs(cluster)
        if isinstance(spec, dict) and isinstance(spec.get("id"), str)
    }
    members = [member for member in _member_ids(cluster) if member in specs]
    upstream_role_map = _extract_upstream_role_map(cluster)
    roles = _assign_object_roles(
        cluster=cluster,
        specs=specs,
        members=members,
        upstream_role_map=upstream_role_map,
    )
    role_ids: dict[str, list[str]] = {}
    for object_id, role in roles.items():
        role_ids.setdefault(role, []).append(object_id)
    required_ids, optional_ids = _extract_required_optional_ids(cluster)
    required_ids = {oid for oid in required_ids if oid in members}
    optional_ids = {oid for oid in optional_ids if oid in members}
    if not required_ids:
        required_ids = set(members) - optional_ids
    upstream_anchor = _extract_upstream_dominant_anchor(cluster)
    return SemanticContext(
        cluster_id=_cluster_id(cluster),
        cluster_type=_resolve_cluster_type(cluster),
        semantic_role=_semantic_role_from_cluster(cluster),
        members=members,
        specs=specs,
        roles=roles,
        role_ids=role_ids,
        dominant_anchor_id=_pick_dominant_anchor(
            cluster,
            specs,
            members,
            roles,
            upstream_dominant_anchor=upstream_anchor,
        ),
        required_object_ids=required_ids,
        optional_object_ids=optional_ids,
        bundle_graph=_extract_bundle_graph(cluster),
        grid_mm=_extract_grid_mm(cluster),
    )


def _extract_upstream_role_map(cluster: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    program = cluster.get("cluster_program") if isinstance(cluster, dict) else None
    candidates: list[Any] = []
    if isinstance(program, dict):
        candidates.extend(
            [
                program.get("roles"),
                program.get("role_map"),
                program.get("object_roles"),
            ]
        )
    candidates.extend(
        [
            cluster.get("roles"),
            cluster.get("role_map"),
            cluster.get("object_roles"),
        ]
    )
    for raw in candidates:
        if isinstance(raw, dict):
            for object_id, role in raw.items():
                if (
                    isinstance(object_id, str)
                    and isinstance(role, str)
                    and role.strip()
                ):
                    out[object_id] = _normalize_role_name(role)
        elif isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                object_id = item.get("id") or item.get("object_id")
                role = item.get("role") or item.get("semantic_role")
                if (
                    isinstance(object_id, str)
                    and isinstance(role, str)
                    and role.strip()
                ):
                    out[object_id] = _normalize_role_name(role)
    return out


def _extract_bundle_graph(cluster: dict[str, Any]) -> dict[str, list[str]]:
    raw_values: list[Any] = []
    program = cluster.get("cluster_program") if isinstance(cluster, dict) else None
    if isinstance(program, dict):
        raw_values.extend([program.get("bundle_graph"), program.get("relations")])
    raw_values.extend([cluster.get("bundle_graph"), cluster.get("relations")])
    graph: dict[str, set[str]] = {}
    for raw in raw_values:
        if isinstance(raw, dict):
            for key, values in raw.items():
                if not isinstance(key, str):
                    continue
                if isinstance(values, list):
                    graph.setdefault(key, set()).update(
                        item for item in values if isinstance(item, str)
                    )
        elif isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                a = item.get("a") or item.get("source") or item.get("from")
                b = item.get("b") or item.get("target") or item.get("to")
                if isinstance(a, str) and isinstance(b, str):
                    graph.setdefault(a, set()).add(b)
                    graph.setdefault(b, set()).add(a)
    return {key: sorted(values) for key, values in graph.items()}


def _extract_required_optional_ids(
    cluster: dict[str, Any],
) -> tuple[set[str], set[str]]:
    required: set[str] = set()
    optional: set[str] = set()
    decisions = cluster.get("decisions")
    if isinstance(decisions, list):
        for item in decisions:
            if not isinstance(item, dict):
                continue
            object_id = (
                item.get("object_id") or item.get("object_type") or item.get("category")
            )
            if not isinstance(object_id, str) or not object_id:
                continue
            priority = str(item.get("priority") or "").lower()
            required_flag = item.get("required")
            if required_flag is True or priority in {"anchor", "primary", "required"}:
                required.add(object_id)
            elif required_flag is False or priority == "optional":
                optional.add(object_id)

    inventory = cluster.get("inventory_decision")
    objects = inventory.get("objects") if isinstance(inventory, dict) else None
    if isinstance(objects, list):
        for item in objects:
            if not isinstance(item, dict):
                continue
            object_id = (
                item.get("object_id") or item.get("object_type") or item.get("category")
            )
            if not isinstance(object_id, str) or not object_id:
                continue
            if item.get("required") is True:
                required.add(object_id)
            elif (
                item.get("required") is False
                or str(item.get("priority") or "").lower() == "optional"
            ):
                optional.add(object_id)

    required.update(_seed_collect_access_required(cluster))
    optional -= required
    return required, optional


def _extract_upstream_dominant_anchor(cluster: dict[str, Any]) -> str | None:
    program = cluster.get("cluster_program") if isinstance(cluster, dict) else None
    candidates: list[Any] = []
    if isinstance(program, dict):
        candidates.extend(
            [
                program.get("dominant_anchor_id"),
                program.get("anchor_id"),
                program.get("primary_anchor"),
            ]
        )
    candidates.extend(
        [
            cluster.get("dominant_anchor_id"),
            cluster.get("anchor_id"),
            cluster.get("primary_anchor"),
        ]
    )
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _normalize_role_name(role: str) -> str:
    token = _normalize_token(role)
    if any(marker in token for marker in ("dominant", "primary_anchor", "main")):
        return "dominant_anchor"
    if "workflow" in token:
        return "workflow_anchor"
    if any(marker in token for marker in ("media", "tv", "console")):
        return "media_support"
    if "storage" in token:
        return "storage_support"
    if any(marker in token for marker in ("side", "nightstand", "lamp")):
        return "side_support"
    if any(marker in token for marker in ("central", "table", "ottoman")):
        return "central_support"
    if "anchor" in token:
        return "secondary_anchor"
    if "support" in token:
        return "secondary_support"
    return token or "secondary_support"


def _assign_object_roles(
    *,
    cluster: dict[str, Any],
    specs: dict[str, dict[str, Any]],
    members: list[str],
    upstream_role_map: dict[str, str] | None = None,
) -> dict[str, str]:
    anchors = {
        item
        for item in cluster.get("anchors", [])
        if isinstance(item, str) and item in members
    }
    priority_by_id = _seed_build_priority_map(cluster)
    roles: dict[str, str] = {}
    sorted_members = sorted(
        members,
        key=lambda oid: (
            0 if oid in anchors else 1,
            int(priority_by_id.get(oid, 9)),
            -_spec_area(specs.get(oid)),
            oid,
        ),
    )

    dominant_assigned = False
    for oid in sorted_members:
        upstream_role = (upstream_role_map or {}).get(oid)
        if upstream_role:
            roles[oid] = upstream_role
            if upstream_role == "dominant_anchor":
                dominant_assigned = True
            continue

        kind = _object_kind(oid)
        priority = int(priority_by_id.get(oid, 9))
        if oid in anchors or priority == 0:
            if not dominant_assigned:
                roles[oid] = "dominant_anchor"
                dominant_assigned = True
            elif is_profile_workflow_object(kind):
                roles[oid] = "workflow_anchor"
            elif is_profile_storage_object(kind):
                roles[oid] = "storage_support"
            elif is_profile_floating_object(kind):
                roles[oid] = "central_support"
            elif "tv" in kind or "media" in kind or "console" in kind:
                roles[oid] = "media_support"
            elif any(
                token in kind for token in ("coffee_table", "ottoman", "center_table")
            ):
                roles[oid] = "central_support"
            elif "table" in kind and not any(
                token in kind for token in ("side", "night")
            ):
                roles[oid] = "central_support"
            elif any(token in kind for token in ("desk", "workstation", "vanity")):
                roles[oid] = "workflow_anchor"
            elif any(
                token in kind
                for token in ("wardrobe", "closet", "dresser", "shelf", "cabinet")
            ):
                roles[oid] = "storage_support"
            else:
                roles[oid] = "secondary_anchor"
            continue
        if is_profile_workflow_object(kind):
            roles[oid] = "workflow_anchor"
        elif is_profile_storage_object(kind):
            roles[oid] = "storage_support"
        elif is_profile_floating_object(kind):
            roles[oid] = "central_support"
        elif any(token in kind for token in ("tv", "media", "console")):
            roles[oid] = "media_support"
        elif any(
            token in kind for token in ("coffee_table", "ottoman", "center_table")
        ):
            roles[oid] = "central_support"
        elif "table" in kind and not any(token in kind for token in ("side", "night")):
            roles[oid] = "central_support"
        elif any(token in kind for token in ("desk", "workstation", "vanity")):
            roles[oid] = "workflow_anchor"
        elif any(
            token in kind
            for token in ("wardrobe", "closet", "dresser", "shelf", "cabinet")
        ):
            roles[oid] = "storage_support"
        elif any(token in kind for token in ("chair", "sofa", "bench", "recliner")):
            roles[oid] = "secondary_support"
        elif any(token in kind for token in ("lamp", "side", "nightstand")):
            roles[oid] = "side_support"
        else:
            roles[oid] = "secondary_support" if priority <= 2 else "accessory_support"
    return roles


def _pick_dominant_anchor(
    cluster: dict[str, Any],
    specs: dict[str, dict[str, Any]],
    members: list[str],
    roles: dict[str, str],
    upstream_dominant_anchor: str | None = None,
) -> str | None:
    if upstream_dominant_anchor in members:
        return upstream_dominant_anchor
    cluster_type = _resolve_cluster_type(cluster)
    priority_map = _seed_build_priority_map(cluster)
    priority_anchors = [oid for oid in members if int(priority_map.get(oid, 9)) == 0]
    role_anchors = [oid for oid in members if roles.get(oid) == "dominant_anchor"]
    workflow_anchors = [
        oid for oid in members if roles.get(oid) in {"workflow_anchor", "media_support"}
    ]
    candidates = priority_anchors or role_anchors or workflow_anchors or list(members)
    preferred = _preferred_anchor_candidates(
        candidates=candidates,
        cluster_type=cluster_type,
    )
    if preferred:
        candidates = preferred
    if candidates:
        return sorted(candidates, key=lambda oid: (-_spec_area(specs.get(oid)), oid))[0]
    return None


def _preferred_anchor_candidates(
    *,
    candidates: list[str],
    cluster_type: str,
) -> list[str]:
    preferred_markers_by_type = {
        "living": ("sofa", "sectional", "loveseat", "armchair", "recliner", "bench"),
        "living_media": ("tv", "media", "console", "projector"),
        "sleep": ("bed", "headboard"),
        "work": ("desk", "workstation"),
        "dining": ("dining_table", "table"),
        "storage": ("wardrobe", "closet", "bookshelf", "cabinet", "shelf"),
        "kitchen": (
            "kitchen_base_cabinet",
            "sink",
            "stove",
            "fridge",
            "counter",
        ),
    }
    markers = preferred_markers_by_type.get(cluster_type, ())
    if not markers:
        return []
    return [
        object_id
        for object_id in candidates
        if any(marker in _object_kind(object_id) for marker in markers)
    ]


def _place_anchor_first(
    ctx: SemanticContext,
    anchor_id: str,
    *,
    prefer_rot: int,
) -> dict[str, dict[str, Any]]:
    spec = ctx.specs.get(anchor_id)
    if not isinstance(spec, dict):
        return {}
    rot = _seed_first_allowed_rotation(spec, prefer=prefer_rot)
    return {anchor_id: {"id": anchor_id, "x": 0, "y": 0, "rot": int(rot)}}


def _place_object_on_side(
    *,
    cluster: dict[str, Any],
    ctx: SemanticContext,
    placements: dict[str, dict[str, Any]],
    object_id: str,
    base_id: str,
    side: str,
    gap: int,
    align: str,
    face_base: bool,
) -> None:
    if object_id in placements or base_id not in placements:
        return
    specs = ctx.specs
    spec = specs.get(object_id)
    base_spec = specs.get(base_id)
    if not isinstance(spec, dict) or not isinstance(base_spec, dict):
        return

    grid_mm = int(ctx.grid_mm)
    base = placements[base_id]
    bx = int(base["x"])
    by = int(base["y"])
    bw, bh = _seed_rotated_wh(base_spec, int(base["rot"]))
    desired_front = _opposite_side(side) if face_base else None
    rot = _rotation_for_front_side(cluster, object_id, spec, desired_front)
    ow, oh = _seed_rotated_wh(spec, rot)
    gap = _seed_snap_up_value(max(0, int(gap)), grid_mm)

    if side == "top":
        x = _aligned_axis_start(bx, bw, ow, align, grid_mm)
        y = _seed_snap_up_value(by + bh + gap, grid_mm)
    elif side == "bottom":
        x = _aligned_axis_start(bx, bw, ow, align, grid_mm)
        y = _seed_snap_down_value(by - oh - gap, grid_mm)
    elif side == "left":
        x = _seed_snap_down_value(bx - ow - gap, grid_mm)
        y = _aligned_axis_start(by, bh, oh, align, grid_mm)
    else:
        x = _seed_snap_up_value(bx + bw + gap, grid_mm)
        y = _aligned_axis_start(by, bh, oh, align, grid_mm)

    candidate = {"id": object_id, "x": int(x), "y": int(y), "rot": int(rot)}
    if _candidate_overlaps_existing(
        ctx=ctx, candidate=candidate, placements=placements
    ):
        return
    placements[object_id] = candidate


def _place_leftover_objects(
    *,
    cluster: dict[str, Any],
    ctx: SemanticContext,
    placements: dict[str, dict[str, Any]],
    object_ids: list[str],
    family: str,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    updated = dict(placements)
    unplaced: list[str] = []
    for oid in object_ids:
        if oid in updated:
            continue
        placed = _try_place_on_semantic_slot(
            cluster=cluster,
            ctx=ctx,
            placements=updated,
            object_id=oid,
            family=family,
        )
        if placed is None:
            unplaced.append(oid)
            continue
        updated[oid] = placed
    return updated, unplaced


def _semantic_slot_candidates_for_leftover(
    *,
    family: str,
    object_id: str,
    ctx: SemanticContext,
    placements: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    role = ctx.roles.get(object_id, "")
    anchor_id = ctx.dominant_anchor_id
    if anchor_id is None or anchor_id not in placements:
        anchor_id = next(iter(placements), None)
    if anchor_id is None:
        return []

    anchor_pl = placements[anchor_id]
    front = "top"
    if anchor_id in placements:
        front = _seed_rotate_side("top", int(anchor_pl.get("rot", 0) or 0))
    left = _rotate_cardinal_side(front, -1)
    right = _rotate_cardinal_side(front, 1)
    back = _opposite_side(front)
    gap = max(ctx.grid_mm * 2, 150)

    family_token = _normalize_token(family)
    if role in {"media_support", "media_anchor"} or "media" in _object_kind(object_id):
        return [
            {
                "base_id": anchor_id,
                "side": front,
                "gap": max(1200, ctx.grid_mm * 16),
                "align": "center",
                "face_base": True,
            }
        ]
    if role in {"side_support", "accessory_support"}:
        return [
            {
                "base_id": anchor_id,
                "side": left,
                "gap": gap,
                "align": "center",
                "face_base": False,
            },
            {
                "base_id": anchor_id,
                "side": right,
                "gap": gap,
                "align": "center",
                "face_base": False,
            },
        ]
    if role == "storage_support" or "storage" in family_token:
        side_order = (
            [back, right, left] if "entry" in family_token else [right, left, back]
        )
        return [
            {
                "base_id": anchor_id,
                "side": side,
                "gap": max(450, ctx.grid_mm * 6),
                "align": "center",
                "face_base": False,
            }
            for side in side_order
        ]
    if role in {"secondary_support", "secondary_anchor"}:
        return [
            {
                "base_id": anchor_id,
                "side": left,
                "gap": max(300, ctx.grid_mm * 4),
                "align": "center",
                "face_base": True,
            },
            {
                "base_id": anchor_id,
                "side": right,
                "gap": max(300, ctx.grid_mm * 4),
                "align": "center",
                "face_base": True,
            },
            {
                "base_id": anchor_id,
                "side": back,
                "gap": max(300, ctx.grid_mm * 4),
                "align": "center",
                "face_base": True,
            },
        ]
    if role == "central_support":
        return [
            {
                "base_id": anchor_id,
                "side": front,
                "gap": max(250, ctx.grid_mm * 4),
                "align": "center",
                "face_base": False,
            }
        ]
    return []


def _try_place_on_semantic_slot(
    *,
    cluster: dict[str, Any],
    ctx: SemanticContext,
    placements: dict[str, dict[str, Any]],
    object_id: str,
    family: str,
) -> dict[str, Any] | None:
    for slot in _semantic_slot_candidates_for_leftover(
        family=family,
        object_id=object_id,
        ctx=ctx,
        placements=placements,
    ):
        trial = dict(placements)
        _place_object_on_side(
            cluster=cluster,
            ctx=ctx,
            placements=trial,
            object_id=object_id,
            base_id=str(slot["base_id"]),
            side=str(slot["side"]),
            gap=int(slot["gap"]),
            align=str(slot["align"]),
            face_base=bool(slot["face_base"]),
        )
        placed = trial.get(object_id)
        if placed is not None:
            return placed
    return None


def _candidate_overlaps_existing(
    *,
    ctx: SemanticContext,
    candidate: dict[str, Any],
    placements: dict[str, dict[str, Any]],
) -> bool:
    candidate_rect = _rect_for_placement(ctx=ctx, placement=candidate)
    if candidate_rect is None:
        return True
    for placement in placements.values():
        other_rect = _rect_for_placement(ctx=ctx, placement=placement)
        if other_rect is None:
            continue
        if _rects_intersect(candidate_rect, other_rect):
            return True
    return False


def _rect_for_placement(
    *,
    ctx: SemanticContext,
    placement: dict[str, Any],
) -> tuple[int, int, int, int] | None:
    oid = placement.get("id")
    if not isinstance(oid, str):
        return None
    spec = ctx.specs.get(oid)
    if not isinstance(spec, dict):
        return None
    w, h = _seed_rotated_wh(spec, int(placement.get("rot", 0) or 0))
    x = int(placement.get("x", 0) or 0)
    y = int(placement.get("y", 0) or 0)
    return (x, y, x + w, y + h)


def _bbox_for_placements(
    *,
    ctx: SemanticContext,
    placements: dict[str, dict[str, Any]],
) -> dict[str, int]:
    rects = []
    for placement in placements.values():
        rect = _rect_for_placement(ctx=ctx, placement=placement)
        if rect is None:
            continue
        x1, y1, x2, y2 = rect
        rects.append(
            {"id": str(placement["id"]), "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}
        )
    return _local_bbox_from_rects(rects)


def _rects_intersect(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> bool:
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


def _global_front_side(
    cluster: dict[str, Any],
    ctx: SemanticContext,
    object_id: str,
    placement: dict[str, Any],
) -> str:
    spec = ctx.specs.get(object_id, {})
    front = _seed_get_front_base(cluster, object_id, spec)
    return _seed_rotate_side(front, int(placement.get("rot", 0) or 0))


def _living_side_sequence(front_side: str, family: str) -> list[str]:
    left = _rotate_cardinal_side(front_side, -1)
    right = _rotate_cardinal_side(front_side, 1)
    opposite = _opposite_side(front_side)
    if family == "conversation_facing":
        return [opposite, left, right, front_side]
    if family == "open_center":
        return [left, right, opposite, front_side]
    return [left, right, opposite, front_side]


def _rotation_for_front_side(
    cluster: dict[str, Any],
    object_id: str,
    spec: dict[str, Any],
    desired_front: str | None,
) -> int:
    if desired_front is None:
        return _seed_first_allowed_rotation(spec, prefer=0)
    allowed = spec.get("allowed_rotations")
    allowed_rots = (
        [int(value) % 360 for value in allowed if isinstance(value, int)]
        if isinstance(allowed, list)
        else [0, 90, 180, 270]
    )
    base_front = _seed_get_front_base(cluster, object_id, spec)
    for rot in allowed_rots:
        if _seed_rotate_side(base_front, rot) == desired_front:
            return int(rot)
    return _seed_first_allowed_rotation(spec, prefer=0)


def _aligned_axis_start(
    base_start: int,
    base_extent: int,
    object_extent: int,
    align: str,
    grid_mm: int,
) -> int:
    if align == "start":
        raw = base_start
    elif align == "end":
        raw = base_start + base_extent - object_extent
    else:
        raw = base_start + (base_extent - object_extent) // 2
    return _seed_snap_nearest_value(int(raw), grid_mm)


def _interaction_rects_from_verifier(
    verifier_eval: dict[str, Any],
) -> list[dict[str, Any]]:
    debug = verifier_eval.get("debug") if isinstance(verifier_eval, dict) else None
    if not isinstance(debug, dict):
        return []
    rects: list[dict[str, Any]] = []
    rects_clear = debug.get("rects_clear")
    if isinstance(rects_clear, dict):
        for object_id, rect in rects_clear.items():
            item = _debug_rect_to_rect(str(object_id), rect)
            if item is not None:
                rects.append(item)
    clearances = debug.get("front_clearance_rects")
    if isinstance(clearances, dict):
        for object_id, rect in clearances.items():
            item = _debug_rect_to_rect(f"access:{object_id}", rect)
            if item is not None:
                rects.append(item)
    return rects


def _required_access_zones_from_verifier(
    verifier_eval: dict[str, Any],
) -> list[dict[str, Any]]:
    debug = verifier_eval.get("debug") if isinstance(verifier_eval, dict) else None
    if not isinstance(debug, dict):
        return []
    clearances = debug.get("front_clearance_rects")
    if not isinstance(clearances, dict):
        return []
    zones: list[dict[str, Any]] = []
    for object_id, rect in clearances.items():
        item = _debug_rect_to_rect(str(object_id), rect)
        if item is None:
            continue
        item["kind"] = "front_clearance"
        zones.append(item)
    return zones


def _debug_rect_to_rect(object_id: str, rect: Any) -> dict[str, Any] | None:
    if not isinstance(rect, dict):
        return None
    if all(key in rect for key in ("x1", "y1", "x2", "y2")):
        x1 = int(rect.get("x1", 0) or 0)
        y1 = int(rect.get("y1", 0) or 0)
        x2 = int(rect.get("x2", 0) or 0)
        y2 = int(rect.get("y2", 0) or 0)
    else:
        x1 = int(rect.get("x", 0) or 0)
        y1 = int(rect.get("y", 0) or 0)
        x2 = x1 + int(rect.get("w", 0) or 0)
        y2 = y1 + int(rect.get("h", 0) or 0)
    if x2 <= x1 or y2 <= y1:
        return None
    return {"id": object_id, "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}


def _largest_outline_polygon(
    polygons: list[list[dict[str, int]]],
) -> list[dict[str, int]]:
    if not polygons:
        return []
    return max(
        polygons,
        key=lambda poly: abs(_polygon_area_from_dict_points(poly)),
    )


def _polygon_area_from_dict_points(points: list[dict[str, int]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for idx, point in enumerate(points):
        nxt = points[(idx + 1) % len(points)]
        area += float(point["x"] * nxt["y"] - nxt["x"] * point["y"])
    return area * 0.5


def _local_quality_from_eval(
    *,
    family: str,
    verifier_eval: dict[str, Any],
    rects: list[dict[str, Any]],
    placements: list[dict[str, Any]],
    ctx: SemanticContext,
) -> LocalQualityBreakdown:
    quality = verifier_eval.get("quality") if isinstance(verifier_eval, dict) else None
    quality = quality if isinstance(quality, dict) else {}
    hard_valid = verifier_eval.get("result") == "VALID"
    constraint_scores = (
        ((verifier_eval.get("debug") or {}).get("constraint_scores") or {})
        if isinstance(verifier_eval.get("debug"), dict)
        else {}
    )
    soft = (
        constraint_scores.get("soft_summary")
        if isinstance(constraint_scores, dict)
        else {}
    )
    soft_penalty = float((soft or {}).get("total_weighted_penalty") or 0.0)
    fill_bbox = float(quality.get("fill_ratio_bbox", 1.0) or 1.0)
    fill_hull = float(quality.get("fill_ratio_hull", 1.0) or 1.0)
    compact_raw = float(quality.get("compactness_perimeter2_over_4piA", 1.0) or 1.0)
    compactness = max(
        0.0, min(1.0, (fill_bbox + fill_hull) / 2.0 / max(1.0, compact_raw))
    )
    contract = _family_contract_validator(
        family=family,
        placements=placements,
        ctx=ctx,
        verifier_eval=verifier_eval,
    )
    family_fidelity = contract.family_fidelity
    functional = _score_functional_usefulness(
        verifier_eval=verifier_eval,
        placements=placements,
        ctx=ctx,
    )
    naturalness = _score_group_naturalness(
        family=family,
        placements=placements,
        ctx=ctx,
    )
    solver_friendliness = _score_solver_friendliness(
        verifier_eval=verifier_eval,
        rects=rects,
        placements=placements,
        ctx=ctx,
    )
    awkwardness = _score_awkwardness_penalty(
        family=family,
        placements=placements,
        ctx=ctx,
        verifier_eval=verifier_eval,
    )
    component_count = _cluster_component_count(
        rects,
        gap_tolerance=max(900, ctx.grid_mm * 16),
    )
    split_cluster_penalty = _clamp01(max(0, component_count - 1) * 0.22)
    fake_support_penalty = _fake_support_penalty(contract.contract_reasons)
    awkward_grouping_penalty = _clamp01(
        awkwardness + split_cluster_penalty + fake_support_penalty
    )
    if not contract.passed:
        functional *= 0.65
        naturalness *= 0.70
        solver_friendliness *= 0.75
    naturalness = _clamp01((naturalness + max(0.0, 1.0 - soft_penalty / 1000.0)) / 2.0)
    semantic = _clamp01(
        0.38 * family_fidelity
        + 0.24 * functional
        + 0.18 * naturalness
        + 0.20 * solver_friendliness
        - 0.32 * split_cluster_penalty
        - 0.24 * awkward_grouping_penalty
        - 0.28 * fake_support_penalty
    )
    return LocalQualityBreakdown(
        functional_score=_clamp01(functional if hard_valid else functional * 0.35),
        naturalness_score=_clamp01(naturalness),
        semantic_coherence_score=semantic,
        compactness_score=_clamp01(
            (compactness - split_cluster_penalty) if rects else 0.0
        ),
        family_fidelity_score=_clamp01(family_fidelity),
        awkwardness_penalty=_clamp01(awkwardness),
        solver_friendliness_score=_clamp01(solver_friendliness),
        split_cluster_penalty=split_cluster_penalty,
        awkward_grouping_penalty=awkward_grouping_penalty,
        fake_support_penalty=fake_support_penalty,
        compaction_semantic_penalty=0.0,
    )


def _variant_rank_key(
    quality: dict[str, float],
    verifier_eval: dict[str, Any],
    *,
    fallback_heavy: bool = False,
) -> tuple[float, ...]:
    tool_rank = _tool_rank(verifier_eval)
    fallback_penalty = 1.0 if fallback_heavy else 0.0
    return (
        fallback_penalty,
        *[float(value) for value in tool_rank],
        -float(quality.get("semantic_coherence_score", 0.0)),
        -float(quality.get("family_fidelity_score", 0.0)),
        float(quality.get("awkwardness_penalty", 1.0)),
        float(quality.get("split_cluster_penalty", 1.0)),
        float(quality.get("awkward_grouping_penalty", 1.0)),
        float(quality.get("fake_support_penalty", 1.0)),
        -float(quality.get("functional_score", 0.0)),
        -float(quality.get("naturalness_score", 0.0)),
        -float(quality.get("solver_friendliness_score", 0.0)),
        -float(quality.get("compactness_score", 0.0)),
    )


def _semantic_signature_for_variant(
    *,
    cluster: dict[str, Any],
    family: str,
    placements: list[dict[str, Any]],
    ctx: SemanticContext | None = None,
    rects: list[dict[str, Any]] | None = None,
    wall_contacts: list[dict[str, str]] | None = None,
    access_zones: list[dict[str, Any]] | None = None,
) -> list[str]:
    ctx = ctx or _semantic_context(cluster)
    placed_ids = {
        row.get("id")
        for row in placements
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    signature = [family]
    for role in (
        "dominant_anchor",
        "media_anchor",
        "media_support",
        "workflow_anchor",
        "central_support",
        "secondary_anchor",
        "secondary_support",
        "side_support",
        "storage_support",
        "accessory_support",
    ):
        if any(oid in placed_ids for oid in ctx.role_ids.get(role, [])):
            signature.append(role)
    for object_id, related in ctx.bundle_graph.items():
        if object_id in placed_ids and any(oid in placed_ids for oid in related):
            signature.append(f"bundle:{object_id}")
    if _seed_collect_access_required(cluster):
        signature.append("required_access_preserved")
    if rects:
        bbox = _local_bbox_from_rects(rects)
        signature.append(f"center:{_center_usage_signature(rects, bbox)}")
        signature.append(f"hull:{_hull_shape_signature(rects, bbox)}")
    for item in wall_contacts or []:
        signature.append(f"wall:{item.get('object_id')}:{item.get('edge')}")
    if access_zones:
        signature.append(f"access_zones:{len(access_zones)}")
    return _dedupe_strings(signature)


def _normalized_pose_signature(
    placements: list[dict[str, Any]],
    *,
    bbox: dict[str, int],
) -> dict[str, tuple[float, float, int]]:
    span_x = max(1, int(bbox.get("max_x", 0)) - int(bbox.get("min_x", 0)))
    span_y = max(1, int(bbox.get("max_y", 0)) - int(bbox.get("min_y", 0)))
    min_x = int(bbox.get("min_x", 0))
    min_y = int(bbox.get("min_y", 0))
    out: dict[str, tuple[float, float, int]] = {}
    for p in placements:
        if not isinstance(p, dict) or not isinstance(p.get("id"), str):
            continue
        out[p["id"]] = (
            round((int(p.get("x", 0) or 0) - min_x) / span_x, 4),
            round((int(p.get("y", 0) or 0) - min_y) / span_y, 4),
            int(p.get("rot", 0) or 0) % 360,
        )
    return out


def _pose_distance(first: Any, second: Any) -> float:
    if not isinstance(first, dict) or not isinstance(second, dict):
        return 1.0
    common = sorted(set(first.keys()) & set(second.keys()))
    if not common:
        return 1.0
    total = 0.0
    for oid in common:
        a = first[oid]
        b = second[oid]
        if not (
            isinstance(a, tuple)
            and isinstance(b, tuple)
            and len(a) == 3
            and len(b) == 3
        ):
            continue
        rot_delta = 0.0 if a[2] == b[2] else 0.25
        total += (
            abs(float(a[0]) - float(b[0])) + abs(float(a[1]) - float(b[1])) + rot_delta
        )
    return total / max(1, len(common))


def _hull_shape_signature(
    rects: list[dict[str, Any]],
    bbox: dict[str, int],
) -> tuple[float, float, int]:
    width = max(1, int(bbox.get("max_x", 0)) - int(bbox.get("min_x", 0)))
    height = max(1, int(bbox.get("max_y", 0)) - int(bbox.get("min_y", 0)))
    area = sum(int(r.get("w", 0) or 0) * int(r.get("h", 0) or 0) for r in rects)
    bbox_area = max(1, width * height)
    return (
        round(width / height, 3),
        round(area / bbox_area, 3),
        len(rects),
    )


def _contact_pattern_signature(
    wall_contacts: list[dict[str, str]],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            f"{item.get('object_id')}:{item.get('edge')}"
            for item in wall_contacts
            if isinstance(item, dict)
        )
    )


def _center_usage_signature(
    rects: list[dict[str, Any]],
    bbox: dict[str, int],
) -> str:
    width = max(1, int(bbox.get("max_x", 0)) - int(bbox.get("min_x", 0)))
    height = max(1, int(bbox.get("max_y", 0)) - int(bbox.get("min_y", 0)))
    cx1 = int(bbox.get("min_x", 0)) + width // 3
    cx2 = int(bbox.get("min_x", 0)) + (2 * width) // 3
    cy1 = int(bbox.get("min_y", 0)) + height // 3
    cy2 = int(bbox.get("min_y", 0)) + (2 * height) // 3
    occupied = 0
    for rect in rects:
        rx1 = int(rect.get("x", 0) or 0)
        ry1 = int(rect.get("y", 0) or 0)
        rx2 = rx1 + int(rect.get("w", 0) or 0)
        ry2 = ry1 + int(rect.get("h", 0) or 0)
        if max(rx1, cx1) < min(rx2, cx2) and max(ry1, cy1) < min(ry2, cy2):
            occupied += 1
    if occupied == 0:
        return "open"
    if occupied == 1:
        return "light"
    return "occupied"


def _hull_shape_distance(first: Any, second: Any) -> float:
    if not (
        isinstance(first, tuple)
        and isinstance(second, tuple)
        and len(first) == 3
        and len(second) == 3
    ):
        return 1.0
    return min(
        1.0,
        (
            abs(float(first[0]) - float(second[0]))
            + abs(float(first[1]) - float(second[1]))
            + min(1.0, abs(int(first[2]) - int(second[2])) / 4.0)
        )
        / 3.0,
    )


def _contact_pattern_distance(first: Any, second: Any) -> float:
    first_set = set(first) if isinstance(first, tuple) else set()
    second_set = set(second) if isinstance(second, tuple) else set()
    return _jaccard_distance(first_set, second_set)


def _center_usage_distance(first: Any, second: Any) -> float:
    if not isinstance(first, str) or not isinstance(second, str):
        return 1.0
    return 0.0 if first == second else 1.0


def _wall_contact_edges(
    rects: list[dict[str, Any]],
    bbox: dict[str, int],
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for rect in rects:
        if not isinstance(rect, dict) or not isinstance(rect.get("id"), str):
            continue
        object_id = str(rect["id"])
        x = int(rect.get("x", 0) or 0)
        y = int(rect.get("y", 0) or 0)
        w = int(rect.get("w", 0) or 0)
        h = int(rect.get("h", 0) or 0)
        if x == int(bbox.get("min_x", 0)):
            out.append({"object_id": object_id, "edge": "left"})
        if y == int(bbox.get("min_y", 0)):
            out.append({"object_id": object_id, "edge": "bottom"})
        if x + w == int(bbox.get("max_x", 0)):
            out.append({"object_id": object_id, "edge": "right"})
        if y + h == int(bbox.get("max_y", 0)):
            out.append({"object_id": object_id, "edge": "top"})
    return out


def _conflict_notes_for_family(family: str, verifier_eval: dict[str, Any]) -> list[str]:
    errors = verifier_eval.get("errors") if isinstance(verifier_eval, dict) else None
    if not isinstance(errors, list):
        return [f"{family}: verifier returned invalid."]
    notes: list[str] = []
    for err in errors[:3]:
        if not isinstance(err, dict):
            continue
        code = str(err.get("code") or "INVALID")
        detail = str(err.get("detail") or "").strip()
        notes.append(f"{family}: {code}{': ' + detail if detail else ''}")
    return notes


def _first_matching_id(object_ids: list[str], tokens: tuple[str, ...]) -> str | None:
    for oid in object_ids:
        kind = _object_kind(oid)
        if any(token in kind for token in tokens):
            return oid
    return None


def _object_kind(object_id: str) -> str:
    token = _normalize_token(object_id)
    token = re.sub(r"_[0-9]+$", "", token)
    return token


def _normalize_token(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip().lower()).strip("_")
    return slug or "variant"


def _spec_area(spec: dict[str, Any] | None) -> int:
    if not isinstance(spec, dict):
        return 0
    return int(spec.get("w", 0) or 0) * int(spec.get("h", 0) or 0)


def _opposite_side(side: str) -> str:
    return {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}.get(
        side,
        side,
    )


def _rotate_cardinal_side(side: str, quarter_turns: int) -> str:
    sides = ["top", "right", "bottom", "left"]
    if side not in sides:
        return side
    return sides[(sides.index(side) + int(quarter_turns)) % len(sides)]


def _jaccard_distance(first: set[str], second: set[str]) -> float:
    if not first and not second:
        return 0.0
    union = first | second
    if not union:
        return 0.0
    return 1.0 - (len(first & second) / len(union))


def _score_family_fidelity(
    family: str,
    placements: list[dict[str, Any]],
    ctx: SemanticContext,
    verifier_eval: dict[str, Any],
) -> float:
    del verifier_eval
    placed = {p.get("id") for p in placements if isinstance(p, dict)}
    if not placed:
        return 0.0
    required_ratio = len(set(ctx.required_object_ids) & placed) / max(
        1,
        len(ctx.required_object_ids),
    )
    family_token = _normalize_token(family)
    role_bonus = 0.0
    if "media" in family_token:
        role_bonus += 0.20 if _role_ids(ctx, "media_support", "media_anchor") else 0.10
        role_bonus += 0.15 if _role_ids(ctx, "central_support") else 0.0
    elif "conversation" in family_token:
        role_bonus += (
            0.20 if _role_ids(ctx, "secondary_anchor", "secondary_support") else 0.0
        )
        role_bonus += 0.15 if _role_ids(ctx, "central_support") else 0.0
    elif "open_center" in family_token:
        role_bonus += 0.25
    elif "headboard" in family_token or "bed_plus" in family_token:
        role_bonus += (
            0.30
            if any("bed" in _object_kind(oid) for oid in placed if isinstance(oid, str))
            else 0.0
        )
        role_bonus += (
            0.10
            if any(
                "nightstand" in _object_kind(oid)
                for oid in placed
                if isinstance(oid, str)
            )
            else 0.0
        )
    elif "desk" in family_token or "work" in family_token:
        role_bonus += (
            0.30
            if any(
                "desk" in _object_kind(oid) for oid in placed if isinstance(oid, str)
            )
            else 0.0
        )
        role_bonus += (
            0.10
            if any(
                "chair" in _object_kind(oid) for oid in placed if isinstance(oid, str)
            )
            else 0.0
        )
    elif "dining" in family_token or "hospitality" in family_token:
        role_bonus += (
            0.30
            if any(
                "table" in _object_kind(oid) for oid in placed if isinstance(oid, str)
            )
            else 0.0
        )
        role_bonus += (
            0.10
            if any(
                "chair" in _object_kind(oid) or "bench" in _object_kind(oid)
                for oid in placed
                if isinstance(oid, str)
            )
            else 0.0
        )
    elif "storage" in family_token:
        role_bonus += 0.35 if _role_ids(ctx, "storage_support") else 0.20
    return _clamp01(0.55 * required_ratio + role_bonus + 0.15)


def _score_functional_usefulness(
    *,
    verifier_eval: dict[str, Any],
    placements: list[dict[str, Any]],
    ctx: SemanticContext,
) -> float:
    placed = {p.get("id") for p in placements if isinstance(p, dict)}
    required_ratio = len(ctx.required_object_ids & placed) / max(
        1,
        len(ctx.required_object_ids),
    )
    hard = 1.0 if verifier_eval.get("result") == "VALID" else 0.25
    access = 0.35 if _verifier_has_error(verifier_eval, {"ACCESS_BLOCKED"}) else 1.0
    workflow = 1.0
    if _role_ids(ctx, "workflow_anchor"):
        workflow = 1.0 if any("chair" in _object_kind(oid) for oid in placed) else 0.65
    return _clamp01(
        0.42 * hard + 0.26 * required_ratio + 0.20 * access + 0.12 * workflow
    )


def _score_group_naturalness(
    *,
    family: str,
    placements: list[dict[str, Any]],
    ctx: SemanticContext,
) -> float:
    rects = _rects_for_context_placements(ctx, placements)
    if not rects:
        return 0.0
    bbox = _local_bbox_from_rects(rects)
    center_usage = _center_usage_signature(rects, bbox)
    family_token = _normalize_token(family)
    score = 0.72
    if "open_center" in family_token and center_usage == "open":
        score += 0.18
    if "open_center" not in family_token and center_usage != "occupied":
        score += 0.06
    if "storage" in family_token and _wall_contact_edges(rects, bbox):
        score += 0.12
    if _cluster_component_count(rects, gap_tolerance=max(900, ctx.grid_mm * 16)) > 1:
        score -= 0.24
    return _clamp01(score)


def _score_solver_friendliness(
    *,
    verifier_eval: dict[str, Any],
    rects: list[dict[str, Any]],
    placements: list[dict[str, Any]],
    ctx: SemanticContext,
) -> float:
    bbox = _local_bbox_from_rects(rects)
    width = max(1, int(bbox.get("max_x", 0)) - int(bbox.get("min_x", 0)))
    height = max(1, int(bbox.get("max_y", 0)) - int(bbox.get("min_y", 0)))
    aspect = max(width / height, height / width)
    aspect_score = _clamp01(1.2 / max(1.2, aspect))
    access_zones = _required_access_zones_from_verifier(verifier_eval)
    access_score = (
        1.0 if access_zones or verifier_eval.get("result") == "VALID" else 0.5
    )
    component_penalty = _clamp01(
        max(
            0,
            _cluster_component_count(rects, gap_tolerance=max(900, ctx.grid_mm * 16))
            - 1,
        )
        * 0.20
    )
    tail_penalty = _solver_tail_geometry_penalty(rects)
    return _clamp01(
        0.55 * aspect_score
        + 0.35 * access_score
        + 0.10 * _clamp01(len(placements) / max(1, len(ctx.members)))
        - component_penalty
        - tail_penalty
    )


def _score_awkwardness_penalty(
    *,
    family: str,
    placements: list[dict[str, Any]],
    ctx: SemanticContext,
    verifier_eval: dict[str, Any],
) -> float:
    penalty = 0.0
    if verifier_eval.get("result") != "VALID":
        penalty += 0.40
    rects = _rects_for_context_placements(ctx, placements)
    bbox = _local_bbox_from_rects(rects)
    center_usage = _center_usage_signature(rects, bbox)
    if family == "open_center" and center_usage == "occupied":
        penalty += 0.30
    if family == "conversation_facing" and not _role_ids(
        ctx, "secondary_anchor", "secondary_support"
    ):
        penalty += 0.15
    if family == "media_facing" and not _role_ids(ctx, "media_support", "media_anchor"):
        penalty += 0.15
    return _clamp01(penalty)


def _fake_support_penalty(reasons: list[str]) -> float:
    penalty = 0.0
    for reason in reasons:
        token = _normalize_token(reason)
        if any(
            marker in token
            for marker in ("outside", "drifted", "not_in_viewing", "not_in")
        ):
            penalty += 0.16
    return _clamp01(penalty)


def _solver_tail_geometry_penalty(rects: list[dict[str, Any]]) -> float:
    if len(rects) < 3:
        return 0.0
    bbox = _local_bbox_from_rects(rects)
    width = max(1, int(bbox.get("max_x", 0)) - int(bbox.get("min_x", 0)))
    height = max(1, int(bbox.get("max_y", 0)) - int(bbox.get("min_y", 0)))
    min_extent = max(1, min(width, height))
    tail_count = 0
    for rect in rects:
        cx = int(rect.get("x", 0)) + int(rect.get("w", 0)) / 2.0
        cy = int(rect.get("y", 0)) + int(rect.get("h", 0)) / 2.0
        near_edge = min(
            abs(cx - int(bbox["min_x"])),
            abs(cx - int(bbox["max_x"])),
            abs(cy - int(bbox["min_y"])),
            abs(cy - int(bbox["max_y"])),
        )
        if near_edge < min_extent * 0.08:
            tail_count += 1
    return _clamp01(max(0, tail_count - 1) * 0.08)


def _score_relative_candidate_semantic(
    *,
    candidate: dict[str, Any],
    rel: dict[str, Any],
    ctx: SemanticContext | None,
) -> float:
    if ctx is None:
        return 0.0
    object_id = candidate.get("id")
    if not isinstance(object_id, str):
        return 0.0
    role = ctx.roles.get(object_id, "")
    relation_type = str(rel.get("type") or rel.get("kind") or "")
    if role in {"side_support", "central_support"} and relation_type == "anchor_side":
        return 0.20
    if role in {"media_support", "storage_support"} and relation_type == "dock_to_edge":
        return 0.15
    return 0.0


def _rects_for_context_placements(
    ctx: SemanticContext,
    placements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rects: list[dict[str, Any]] = []
    for placement in placements:
        rect = _rect_for_placement(
            ctx=ctx,
            placement=placement,
        )
        if rect is None:
            continue
        x1, y1, x2, y2 = rect
        rects.append(
            {
                "id": str(placement.get("id") or ""),
                "x": x1,
                "y": y1,
                "w": x2 - x1,
                "h": y2 - y1,
            }
        )
    return rects


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _dedupe_strings(values: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    if not isinstance(values, list):
        return out
    for item in values:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _run_with_tools(
    *,
    messages: list[dict[str, Any]],
    cluster: dict[str, Any],
    max_steps: int,
    access_clearance_ratio: float,
) -> str:
    from clients.llm_client import get_llm_client

    client = get_llm_client()
    access_clearance_ratio = FIXED_ACCESS_CLEARANCE_RATIO

    expected_member_ids = _member_ids(cluster)

    last_draft: dict[str, Any] | None = None
    last_verifier_result: str | None = None  # VALID | INVALID | None

    best_valid_draft: dict[str, Any] | None = None
    best_valid_eval: dict[str, Any] | None = None
    best_valid_rank: tuple[Any, ...] | None = None

    verified_state_counts: dict[str, int] = {}
    tried_patch_keys_by_state: dict[str, set[str]] = {}

    # ---------------------------
    # Deterministic seed before first LLM turn
    # ---------------------------
    seeded_placements = _seed_local_layout(cluster)

    if seeded_placements:
        ok, detail = _validate_local_placements_full(
            seeded_placements, expected_member_ids
        )
        if ok:
            last_draft = {
                "cluster_id": cluster.get("cluster_id"),
                "local_placements": _canonicalize_local_placements(seeded_placements),
            }

            seed_eval = _run_verifier_once(
                cluster=cluster,
                placements=seeded_placements,
                access_clearance_ratio=access_clearance_ratio,
            )
            seed_preview = _verifier_preview(seed_eval)

            logger.info(
                "Deterministic seed eval: %s",
                json.dumps(seed_preview, ensure_ascii=True),
            )

            # Nếu seed đã VALID thì tối ưu host-side vài bước rồi đưa cho model tiếp tục
            if seed_eval.get("result") == "VALID":
                current_seed = _canonicalize_local_placements(seeded_placements)
                current_eval = seed_eval
                current_rank = _tool_rank(current_eval)

                best_valid_draft = {
                    "cluster_id": cluster.get("cluster_id"),
                    "local_placements": current_seed,
                }
                best_valid_eval = current_eval
                best_valid_rank = current_rank

                current_seed, current_eval, improve_rounds = (
                    _greedy_improve_valid_layout(
                        cluster=cluster,
                        access_clearance_ratio=access_clearance_ratio,
                        placements=current_seed,
                        verifier_eval=current_eval,
                        max_rounds=min(max(1, int(max_steps)), 12),
                    )
                )
                current_rank = _tool_rank(current_eval)

                best_valid_draft = {
                    "cluster_id": cluster.get("cluster_id"),
                    "local_placements": current_seed,
                }
                best_valid_eval = current_eval
                best_valid_rank = current_rank

                if improve_rounds > 0:
                    logger.info(
                        "Deterministic seed greedy refinement applied %s improving rounds.",
                        improve_rounds,
                    )

                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "A deterministic seeded VALID layout is available. "
                            "Use these verified local_placements as your current best state. "
                            "Only continue if you can find a strictly better VALID forge-faithful layout with a single-object change. "
                            "Treat compactness as secondary to preserving semantic placements and facing.\n\n"
                            f"verified_local_placements={json.dumps(current_seed, ensure_ascii=True)}\n"
                            f"seed_eval_preview={json.dumps(_verifier_preview(current_eval), ensure_ascii=True)}"
                        ),
                    }
                )

            # Nếu seed chưa VALID thì ép model dùng seed này làm first attempt
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Use the following deterministic seeded local_placements as your first attempt. "
                        "Call LocalClusterVerifier with EXACTLY these local_placements first, then repair only if needed. "
                        "Do not invent a brand-new layout before checking this seed.\n\n"
                        f"seeded_local_placements={json.dumps(seeded_placements, ensure_ascii=True)}\n"
                        f"seed_eval_preview={json.dumps(seed_preview, ensure_ascii=True)}"
                    ),
                }
            )
        else:
            logger.warning("Deterministic seed rejected: %s", detail)

    for step in range(1, max_steps + 1):
        logger.info("ClusterComposer step %s/%s", step, max_steps)

        try:
            response = client.chat_completion(
                messages,
                model_key="primary",
                temperature=0.0,
                tools=TOOL_SCHEMAS,
            )
        except Exception as exc:
            if _is_context_length_exceeded(exc):
                logger.warning(
                    "ClusterComposer stopped early due to context length exceeded.",
                    exc_info=exc,
                )
                if best_valid_draft is not None and isinstance(
                    best_valid_draft.get("local_placements"), list
                ):
                    placements = _canonicalize_local_placements(
                        best_valid_draft["local_placements"]
                    )
                    payload = _build_cluster_output_from_placements(
                        cluster=cluster,
                        placements=placements,
                        verifier_eval=best_valid_eval,
                        notes=[
                            "Stopped due to context length exceeded; returned best verified valid layout."
                        ],
                    )
                    return json.dumps(payload, ensure_ascii=True)

                fallback = {
                    "status": "UNSAT",
                    "cluster_id": str(cluster.get("cluster_id") or ""),
                    "local_frame": {
                        "unit": "mm",
                        "grid_mm": _extract_grid_mm(cluster),
                        "origin_note": "(0,0) is an arbitrary local origin for this cluster",
                    },
                    "local_placements": [],
                    "cluster_footprint": {
                        "type": "union_of_rects",
                        "rects": [],
                        "local_bbox": {"min_x": 0, "min_y": 0, "max_x": 0, "max_y": 0},
                    },
                    "notes": [
                        "Stopped due to context length exceeded before finding a valid layout."
                    ],
                    "missing": [],
                }
                return json.dumps(fallback, ensure_ascii=True)
            raise

        message = _extract_message(response)
        tool_calls = _extract_tool_calls(message)
        content = getattr(message, "content", "") or ""

        draft_in_content = _try_parse_json_object(content)
        if isinstance(draft_in_content, dict) and isinstance(
            draft_in_content.get("local_placements"), list
        ):
            last_draft = draft_in_content

        # ---------------------------
        # Tool calls branch
        # ---------------------------
        if tool_calls:
            logger.info(
                "Tool calls requested: %s",
                [(c.get("function", {}) or {}).get("name") for c in tool_calls],
            )

            messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                }
            )

            restart_outer = False
            handled_call_ids: set[str] = set()

            for idx, call in enumerate(tool_calls):
                call_id = call.get("id") or f"tool_{idx}"
                fn = call.get("function", {}) or {}
                name = fn.get("name")
                args_text = fn.get("arguments") or "{}"
                args = _safe_json_loads(args_text)

                if name != "LocalClusterVerifier":
                    tool_output = {
                        "error": "unsupported_tool",
                        "tool": name,
                        "message": "Only LocalClusterVerifier is supported in ClusterComposer.",
                    }
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": str(name),
                            "content": json.dumps(tool_output, ensure_ascii=True),
                        }
                    )
                    handled_call_ids.add(call_id)
                    continue

                if name == "LocalClusterVerifier":
                    args = _coerce_verifier_args(
                        args=args,
                        cluster=cluster,
                        last_draft=last_draft,
                        access_clearance_ratio=access_clearance_ratio,
                    )

                    lp = args.get("local_placements")
                    if not isinstance(lp, list) or not lp:
                        tool_output = {
                            "result": "INVALID",
                            "errors": [
                                {
                                    "code": "MISSING_LOCAL_PLACEMENTS",
                                    "detail": "local_placements is required in tool arguments",
                                }
                            ],
                        }
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call_id,
                                "name": name,
                                "content": json.dumps(tool_output, ensure_ascii=True),
                            }
                        )
                        handled_call_ids.add(call_id)
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "You attempted to call LocalClusterVerifier without local_placements. "
                                    "Re-call LocalClusterVerifier and INCLUDE local_placements inside the TOOL CALL arguments. "
                                    f"local_placements must contain every member id exactly once: {expected_member_ids}. "
                                    "Do NOT output a separate draft; put placements directly in tool arguments."
                                ),
                            }
                        )
                        restart_outer = True
                        continue

                    ok, detail = _validate_local_placements_full(
                        lp, expected_member_ids
                    )
                    if not ok:
                        tool_output = {
                            "result": "INVALID",
                            "errors": [
                                {
                                    "code": "INVALID_LOCAL_PLACEMENTS",
                                    "detail": detail,
                                }
                            ],
                        }
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call_id,
                                "name": name,
                                "content": json.dumps(tool_output, ensure_ascii=True),
                            }
                        )
                        handled_call_ids.add(call_id)
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Your local_placements is incomplete/invalid. "
                                    f"Requirement: every member id exactly once: {expected_member_ids}. "
                                    f"Issue: {detail}. "
                                    "Re-call LocalClusterVerifier with corrected local_placements in tool arguments."
                                ),
                            }
                        )
                        restart_outer = True
                        continue

                    last_draft = {
                        "cluster_id": cluster.get("cluster_id"),
                        "local_placements": _canonicalize_local_placements(lp),
                    }

                tool_output = _safe_run_tool(name, args)

                logger.info(
                    "Tool %s output: %s",
                    name,
                    json.dumps(tool_output, ensure_ascii=True),
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": json.dumps(tool_output, ensure_ascii=True),
                    }
                )
                handled_call_ids.add(call_id)

                if name != "LocalClusterVerifier":
                    continue

                result = tool_output.get("result")
                lp_verified = args.get("local_placements") or []
                lp_verified = _canonicalize_local_placements(lp_verified)

                state_sig = _placements_signature(lp_verified)
                verified_state_counts[state_sig] = (
                    verified_state_counts.get(state_sig, 0) + 1
                )
                state_repeat_count = verified_state_counts[state_sig]
                tried_patch_keys_by_state.setdefault(state_sig, set())

                if result == "VALID":
                    last_verifier_result = "VALID"
                    current_valid_placements, current_valid_eval, improve_rounds = (
                        _greedy_improve_valid_layout(
                            cluster=cluster,
                            access_clearance_ratio=access_clearance_ratio,
                            placements=lp_verified,
                            verifier_eval=tool_output,
                            max_rounds=min(max(1, int(max_steps)), 10),
                        )
                    )
                    current_valid_draft = {
                        "cluster_id": cluster.get("cluster_id"),
                        "local_placements": current_valid_placements,
                    }
                    last_draft = current_valid_draft

                    current_rank = _tool_rank(current_valid_eval)

                    if best_valid_rank is None or current_rank < best_valid_rank:
                        best_valid_rank = current_rank
                        best_valid_draft = current_valid_draft
                        best_valid_eval = current_valid_eval

                    if improve_rounds > 0:
                        logger.info(
                            "ClusterComposer greedy refinement applied %s improving rounds after VALID verification.",
                            improve_rounds,
                        )

                    # Host-side refinement exhausted -> finalize immediately from host
                    final_draft = best_valid_draft or current_valid_draft
                    final_eval = best_valid_eval or current_valid_eval
                    placements = _canonicalize_local_placements(
                        final_draft["local_placements"]
                    )

                    payload = _build_cluster_output_from_placements(
                        cluster=cluster,
                        placements=placements,
                        verifier_eval=final_eval,
                        notes=["Returned best verified valid layout from controller."],
                    )
                    return json.dumps(payload, ensure_ascii=True)

                last_verifier_result = "INVALID"

                try:
                    selected_move, patched, patched_eval = _choose_best_single_patch(
                        cluster=cluster,
                        access_clearance_ratio=access_clearance_ratio,
                        tool_output=tool_output,
                        placements=lp_verified,
                        tried_patch_keys=tried_patch_keys_by_state[state_sig],
                    )
                except Exception as exc:
                    logger.exception(
                        "Patch selection failed; falling back to manual fix prompt: %s",
                        exc,
                    )
                    selected_move, patched, patched_eval = None, None, None

                if selected_move is not None and patched is not None:
                    ok, detail = _validate_local_placements_full(
                        patched, expected_member_ids
                    )

                    if ok:
                        move_key = _move_key(selected_move)
                        tried_patch_keys_by_state[state_sig].add(move_key)

                        last_draft = {
                            "cluster_id": cluster.get("cluster_id"),
                            "local_placements": patched,
                        }

                        patched_eval_preview = _verifier_preview(patched_eval)

                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "LocalClusterVerifier returned INVALID. "
                                    "Use the following SINGLE patched candidate as your next attempt. "
                                    "Do NOT combine it with other suggested moves in the same step. "
                                    "Re-call LocalClusterVerifier with EXACTLY these local_placements unless you have one strictly better single-object patch.\n\n"
                                    f"selected_move={json.dumps(selected_move, ensure_ascii=True)}\n"
                                    f"patched_local_placements={json.dumps(patched, ensure_ascii=True)}\n"
                                    f"patched_eval_preview={json.dumps(patched_eval_preview, ensure_ascii=True)}"
                                ),
                            }
                        )
                    else:
                        errs = tool_output.get("errors", [])
                        preview = errs[:8] if isinstance(errs, list) else errs
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "LocalClusterVerifier returned INVALID and the deterministic patch could not be applied safely. "
                                    "Fix ONLY the reported issues and re-check. "
                                    "Change only one object if possible, and do not mix multiple suggested moves in the same iteration. "
                                    f"Errors (preview): {json.dumps(preview, ensure_ascii=True)}"
                                ),
                            }
                        )
                else:
                    errs = tool_output.get("errors", [])
                    preview = errs[:8] if isinstance(errs, list) else errs

                    if state_repeat_count >= 2:
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "You are repeating the same invalid placement state. "
                                    "Do NOT repeat previous repairs. "
                                    "Either propose a different single-object fix and re-call LocalClusterVerifier, "
                                    "or conclude UNSAT if no grid-valid repair exists with the allowed rotations.\n"
                                    f"Repeated state count={state_repeat_count}\n"
                                    f"Errors (preview): {json.dumps(preview, ensure_ascii=True)}"
                                ),
                            }
                        )
                    else:
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "LocalClusterVerifier returned INVALID. "
                                    "Fix ONLY the reported issues and re-check. "
                                    "Prefer changing one object at a time. "
                                    f"Errors (preview): {json.dumps(preview, ensure_ascii=True)}"
                                ),
                            }
                        )

            if len(handled_call_ids) < len(tool_calls):
                for idx, call in enumerate(tool_calls):
                    call_id = call.get("id") or f"tool_{idx}"
                    if call_id in handled_call_ids:
                        continue
                    fn = call.get("function", {}) or {}
                    name = fn.get("name")
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": name,
                            "content": json.dumps(
                                {
                                    "error": "tool_skipped",
                                    "tool": name,
                                    "message": "Skipped due to restart or earlier tool handling.",
                                },
                                ensure_ascii=True,
                            ),
                        }
                    )

            if restart_outer:
                continue

            continue

        # ---------------------------
        # No tool calls: final JSON
        # ---------------------------
        if not isinstance(content, str) or not content.strip():
            messages.append(
                {"role": "user", "content": "Return a JSON object (no markdown)."}
            )
            continue

        final_payload = _try_parse_json_object(content)
        if not isinstance(final_payload, dict):
            messages.append(
                {
                    "role": "user",
                    "content": "Your output must be valid JSON only (no markdown).",
                }
            )
            continue

        status = final_payload.get("status")
        cid = final_payload.get("cluster_id")
        if status not in {"OK", "UNSAT", "NEED_INFO"} or not isinstance(cid, str):
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your JSON must include top-level fields: status (OK|UNSAT|NEED_INFO) and cluster_id (string)."
                    ),
                }
            )
            continue

        if status in {"UNSAT", "NEED_INFO"}:
            if best_valid_draft is not None and isinstance(
                best_valid_draft.get("local_placements"), list
            ):
                logger.warning(
                    "ClusterComposer received status=%s but a verified valid layout already exists; returning best valid.",
                    status,
                )
                placements = _canonicalize_local_placements(
                    best_valid_draft["local_placements"]
                )
                payload = _build_cluster_output_from_placements(
                    cluster=cluster,
                    placements=placements,
                    verifier_eval=best_valid_eval,
                    notes=[
                        "Model returned non-OK status; returned best verified valid layout instead."
                    ],
                )
                return json.dumps(payload, ensure_ascii=True)

            if status == "UNSAT" and last_verifier_result is None:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Do NOT conclude status=UNSAT before attempting LocalClusterVerifier at least once. "
                            "Call LocalClusterVerifier now with local_placements in tool arguments, including every member exactly once: "
                            f"{expected_member_ids}."
                        ),
                    }
                )
                continue

            return content

        if status == "OK" and last_verifier_result != "VALID":
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You cannot output status=OK until LocalClusterVerifier returns VALID. "
                        "Call LocalClusterVerifier now with local_placements inside tool arguments."
                    ),
                }
            )
            continue

        if status == "OK":
            lp = final_payload.get("local_placements")
            ok, detail = _validate_local_placements_full(lp, expected_member_ids)
            if not ok:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your local_placements is incomplete/invalid. "
                            f"Requirement: every member id exactly once: {expected_member_ids}. "
                            f"Issue: {detail}"
                        ),
                    }
                )
                continue

            # Re-verify final output
            final_eval = _run_verifier_once(
                cluster=cluster,
                placements=lp,
                access_clearance_ratio=access_clearance_ratio,
            )

            if final_eval.get("result") != "VALID":
                preview = _verifier_preview(final_eval)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your final JSON does not verify as VALID. "
                            "Do NOT change the schema. "
                            "Re-call LocalClusterVerifier if needed, or output FINAL JSON using the verified placements.\n\n"
                            f"final_eval_preview={json.dumps(preview, ensure_ascii=True)}"
                        ),
                    }
                )
                continue

            final_rank = _tool_rank(final_eval)

            if (
                best_valid_rank is not None
                and best_valid_draft is not None
                and final_rank > best_valid_rank
            ):
                chosen_placements = _canonicalize_local_placements(
                    best_valid_draft["local_placements"]
                )
                payload = _build_cluster_output_from_placements(
                    cluster=cluster,
                    placements=chosen_placements,
                    verifier_eval=best_valid_eval,
                    notes=["Returned better previously verified valid layout."],
                )
                return json.dumps(payload, ensure_ascii=True)

            chosen_placements = _canonicalize_local_placements(lp)

            if best_valid_rank is None or final_rank < best_valid_rank:
                best_valid_rank = final_rank
                best_valid_draft = {
                    "cluster_id": cluster.get("cluster_id"),
                    "local_placements": chosen_placements,
                }
                best_valid_eval = final_eval

            payload = _build_cluster_output_from_placements(
                cluster=cluster,
                placements=chosen_placements,
                verifier_eval=final_eval,
                notes=["Returned verified valid layout."],
            )
            return json.dumps(payload, ensure_ascii=True)

    if best_valid_draft is not None and isinstance(
        best_valid_draft.get("local_placements"), list
    ):
        logger.warning(
            "ClusterComposer hit max_steps; returning best verified valid layout."
        )
        placements = _canonicalize_local_placements(
            best_valid_draft["local_placements"]
        )
        payload = _build_cluster_output_from_placements(
            cluster=cluster,
            placements=placements,
            verifier_eval=best_valid_eval,
            notes=["Stopped due to max_steps; returned best verified valid layout."],
        )
        return json.dumps(payload, ensure_ascii=True)

    logger.warning("ClusterComposer hit max_steps with no valid layout.")
    return json.dumps(
        {
            "status": "UNSAT",
            "cluster_id": str(cluster.get("cluster_id") or ""),
            "local_frame": {
                "unit": "mm",
                "grid_mm": _extract_grid_mm(cluster),
                "origin_note": "(0,0) is an arbitrary local origin for this cluster",
            },
            "local_placements": [],
            "cluster_footprint": {
                "type": "union_of_rects",
                "rects": [],
                "local_bbox": {"min_x": 0, "min_y": 0, "max_x": 0, "max_y": 0},
            },
            "notes": ["Stopped due to max_steps with no verified valid layout."],
            "missing": [],
        },
        ensure_ascii=True,
    )


# ---------------------------
# Tool / args helpers
# ---------------------------


def _safe_json_loads(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _safe_run_tool(name: str | None, args: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(name, str) or name not in TOOL_REGISTRY:
        return {"error": "unknown_tool", "tool": name, "args": args}
    try:
        result = TOOL_REGISTRY[name](**args)
        if result is None:
            return {"error": "tool_returned_none", "tool": name, "args": args}
        if isinstance(result, dict):
            return result
        return {"error": "tool_returned_non_dict", "tool": name, "result": str(result)}
    except Exception as exc:
        return {"error": "tool_failed", "tool": name, "message": str(exc), "args": args}


def _coerce_verifier_args(
    *,
    args: dict[str, Any],
    cluster: dict[str, Any],
    last_draft: dict[str, Any] | None,
    access_clearance_ratio: float,
) -> dict[str, Any]:
    output = dict(args)

    if "hard_constraints" not in output:
        output["hard_constraints"] = cluster.get("hard_constraints", [])
    if "soft_constraints" not in output:
        output["soft_constraints"] = cluster.get("soft_constraints", [])

    rules = cluster.get("cluster_rules", {}) if isinstance(cluster, dict) else {}
    if "grid_mm" not in output:
        output["grid_mm"] = _extract_grid_mm(cluster)

    if "objects" not in output:
        output["objects"] = _build_object_specs(cluster)

    if "cluster_rules" not in output:
        output["cluster_rules"] = rules if isinstance(rules, dict) else None

    output["access_clearance_ratio"] = float(FIXED_ACCESS_CLEARANCE_RATIO)

    if not isinstance(output.get("local_placements"), list) or not output.get(
        "local_placements"
    ):
        if isinstance(last_draft, dict) and isinstance(
            last_draft.get("local_placements"), list
        ):
            output["local_placements"] = last_draft["local_placements"]

    if "use_clearance" not in output:
        output["use_clearance"] = True

    placement_ids = _placement_ids(output.get("local_placements"))
    if not _objects_have_valid_dims(output.get("objects")) or (
        placement_ids and not _objects_cover_ids(output.get("objects"), placement_ids)
    ):
        output["objects"] = _build_object_specs(cluster)

    return output


def _build_object_specs(cluster: dict[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []

    decisions = cluster.get("decisions", [])
    inventory = (
        cluster.get("inventory_decision")
        if isinstance(cluster.get("inventory_decision"), dict)
        else None
    )
    rules = cluster.get("cluster_rules", {}) if isinstance(cluster, dict) else {}
    allowed_raw = rules.get("allowed_rotations") if isinstance(rules, dict) else None
    allowed = allowed_raw if isinstance(allowed_raw, dict) else {}
    facing_raw = rules.get("facing") if isinstance(rules, dict) else None
    facing = facing_raw if isinstance(facing_raw, dict) else {}

    if not isinstance(decisions, list):
        decisions = []

    if not decisions and isinstance(inventory, dict):
        objects = inventory.get("objects")
        decisions = objects if isinstance(objects, list) else []

    composer_policy = (
        cluster.get("composer_policy")
        if isinstance(cluster.get("composer_policy"), dict)
        else {}
    )
    policy_rotations = composer_policy.get("allowed_rotations")

    for d in decisions:
        if not isinstance(d, dict):
            continue

        obj_id = d.get("object_id") or d.get("object_type") or d.get("category")
        if not isinstance(obj_id, str) or not obj_id:
            continue

        dims_mm = d.get("dims_mm")
        if isinstance(dims_mm, list) and len(dims_mm) >= 2:
            L = float(dims_mm[0] or 0) / 1000.0
            W = float(dims_mm[1] or 0) / 1000.0
        else:
            rep = _decision_rep_dims_for_current_tier(d)
            if not isinstance(rep, dict):
                continue
            L = float(rep.get("L", 0) or 0)
            W = float(rep.get("W", 0) or 0)
        if L <= 0 or W <= 0:
            continue

        allowed_rots = allowed.get(obj_id)
        if not isinstance(allowed_rots, list):
            allowed_rots = (
                policy_rotations
                if isinstance(policy_rotations, list)
                else [0, 90, 180, 270]
            )

        w = int(round(L * 1000))
        h = int(round(W * 1000))
        clearances = d.get("clearances")
        clearance_mm = int(d.get("clearance_mm", 0) or 0)
        if isinstance(clearances, dict):
            clearance_mm = max(
                clearance_mm,
                int(clearances.get("access_front_mm", 0) or 0),
                int(clearances.get("access_side_mm", 0) or 0),
            )

        spec: dict[str, Any] = {
            "id": obj_id,
            "w": w,
            "h": h,
            "clearance_mm": clearance_mm,
            "allowed_rotations": allowed_rots,
            "collision": d.get("collision", "solid"),
        }

        f = facing.get(obj_id)
        if isinstance(f, dict) and f.get("front") in {"top", "bottom", "left", "right"}:
            spec["front"] = f["front"]

        specs.append(spec)

    return specs


def _decision_rep_dims_for_current_tier(
    decision: dict[str, Any],
) -> dict[str, Any] | None:
    tier = str(
        decision.get("recommended_size_tier") or decision.get("size_tier") or ""
    ).upper()
    rep_by_tier = decision.get("rep_dims_m_by_tier")
    if isinstance(rep_by_tier, dict) and tier:
        rep = rep_by_tier.get(tier)
        if isinstance(rep, dict):
            return rep

    size_profile = decision.get("size_profile")
    if isinstance(size_profile, dict):
        rep_dims = size_profile.get("rep_dims_m")
        if isinstance(rep_dims, dict) and tier:
            rep = rep_dims.get(tier)
            if isinstance(rep, dict):
                return rep

    rep = decision.get("rep_dims_m")
    return rep if isinstance(rep, dict) else None


def _objects_have_valid_dims(objects: Any) -> bool:
    if isinstance(objects, dict):
        if not objects:
            return False
        for obj_id, spec in objects.items():
            if not isinstance(obj_id, str) or not obj_id.strip():
                return False
            if not isinstance(spec, dict):
                return False
            w = int(spec.get("w", 0) or 0)
            h = int(spec.get("h", 0) or 0)
            if w <= 0 or h <= 0:
                return False
        return True

    if not isinstance(objects, list) or not objects:
        return False
    for spec in objects:
        if not isinstance(spec, dict):
            return False
        obj_id = spec.get("id")
        if not isinstance(obj_id, str) or not obj_id.strip():
            return False
        w = int(spec.get("w", 0) or 0)
        h = int(spec.get("h", 0) or 0)
        if w <= 0 or h <= 0:
            return False
    return True


def _objects_cover_ids(objects: Any, required_ids: set[str]) -> bool:
    if not required_ids:
        return True
    if isinstance(objects, dict):
        return required_ids.issubset({k for k in objects.keys() if isinstance(k, str)})
    if isinstance(objects, list):
        ids: set[str] = set()
        for spec in objects:
            if not isinstance(spec, dict):
                continue
            obj_id = spec.get("id")
            if isinstance(obj_id, str) and obj_id:
                ids.add(obj_id)
        return required_ids.issubset(ids)
    return False


def _placement_ids(local_placements: Any) -> set[str]:
    if not isinstance(local_placements, list):
        return set()
    out: set[str] = set()
    for p in local_placements:
        if not isinstance(p, dict):
            continue
        pid = p.get("id")
        if isinstance(pid, str) and pid:
            out.add(pid)
    return out


def _seed_resolve_anchor_side(side: str) -> tuple[str, str | None]:
    if side in {"left", "right", "top", "bottom"}:
        return side, None
    if side.startswith("head_"):
        return "head", side.split("_", 1)[1]
    if side.startswith("foot_"):
        return "foot", side.split("_", 1)[1]
    if side in {"head", "foot"}:
        return side, None
    return side, None


def _seed_anchor_zone_bounds(
    free_span: int,
    qualifier_world: str | None,
    span_mode: str = "any",
) -> tuple[int, int]:
    free_span = max(0, int(free_span))

    if span_mode == "center":
        return free_span // 3, (2 * free_span + 2) // 3
    if span_mode == "left":
        return 0, free_span // 3
    if span_mode == "right":
        return (2 * free_span + 2) // 3, free_span

    # outer-third semantics:
    # left/bottom = outer 1/3 đầu
    # right/top   = outer 1/3 cuối
    if qualifier_world in {"left", "bottom"}:
        return 0, free_span // 3
    if qualifier_world in {"right", "top"}:
        return (2 * free_span + 2) // 3, free_span

    return 0, free_span


def _seed_pick_anchor_axis_start(
    *,
    base_start: int,
    base_extent: int,
    a_extent: int,
    qualifier_world: str | None,
    span_mode: str,
    grid_mm: int,
) -> int:
    free_span = max(0, int(base_extent - a_extent))
    lo = int(base_start)

    z_lo, z_hi = _seed_anchor_zone_bounds(
        free_span,
        qualifier_world,
        span_mode=span_mode,
    )
    q_lo = lo + z_lo
    q_hi = lo + z_hi

    if qualifier_world in {"left", "bottom", "right", "top"}:
        current = q_lo + max(0, (q_hi - q_lo)) // 2
    else:
        current = lo + free_span // 2

    return _seed_pick_grid_in_interval(current, (q_lo, q_hi), grid_mm)


def _extract_semantic_placements(cluster: dict[str, Any]) -> list[dict[str, Any]]:
    rules = cluster.get("cluster_rules", {}) if isinstance(cluster, dict) else {}
    if not isinstance(rules, dict):
        return []
    raw = rules.get("semantic_placements")
    if not isinstance(raw, list):
        return []

    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        obj_id = item.get("id")
        relative_to = item.get("relative_to")
        kind = item.get("kind")
        if not (
            isinstance(obj_id, str)
            and obj_id
            and isinstance(relative_to, str)
            and relative_to
            and isinstance(kind, str)
            and kind in {"dock_to_edge", "anchor_side"}
        ):
            continue
        out.append(dict(item))
    return out


def _build_fallback_pack_layout(
    cluster: dict[str, Any], *, strategy: str = "row"
) -> list[dict[str, Any]]:
    members = _member_ids(cluster)
    if not members:
        return []

    grid_mm = _extract_grid_mm(cluster)
    spec_by_id = {
        s["id"]: s
        for s in _build_object_specs(cluster)
        if isinstance(s, dict) and isinstance(s.get("id"), str)
    }

    if any(mid not in spec_by_id for mid in members):
        return []

    hard_constraints = (
        cluster.get("hard_constraints", [])
        if isinstance(cluster.get("hard_constraints"), list)
        else []
    )
    anchors = [
        a
        for a in (
            cluster.get("anchors", [])
            if isinstance(cluster.get("anchors"), list)
            else []
        )
        if isinstance(a, str) and a in spec_by_id
    ]
    priority_map = _seed_build_priority_map(cluster)
    access_required = _seed_collect_access_required(cluster)
    semantic_placements = _extract_semantic_placements(cluster)

    dependent_ids: set[str] = set()
    relations: list[dict[str, Any]] = []
    seen_relation_keys: set[tuple[str, str, str]] = set()

    for c in hard_constraints:
        if not isinstance(c, dict):
            continue
        ctype = c.get("type")
        a = c.get("a")
        b = c.get("b")
        if ctype in {"dock_to_edge", "anchor_side", "contain_in"}:
            if isinstance(a, str) and isinstance(b, str):
                if a in spec_by_id and b in spec_by_id:
                    rel = dict(c)
                    for semantic in semantic_placements:
                        if (
                            semantic.get("id") == a
                            and semantic.get("relative_to") == b
                            and semantic.get("kind") == ctype
                        ):
                            rel.update(semantic)
                    relations.append(rel)
                    seen_relation_keys.add((a, b, str(ctype)))
                    dependent_ids.add(a)

    for semantic in semantic_placements:
        a = semantic.get("id")
        b = semantic.get("relative_to")
        ctype = semantic.get("kind")
        if not (
            isinstance(a, str)
            and isinstance(b, str)
            and isinstance(ctype, str)
            and a in spec_by_id
            and b in spec_by_id
        ):
            continue
        key = (a, b, ctype)
        if key in seen_relation_keys:
            continue
        synthetic = dict(semantic)
        synthetic["type"] = ctype
        relations.append(synthetic)
        dependent_ids.add(a)

    freestanding = [m for m in members if m in spec_by_id and m not in dependent_ids]
    if not freestanding:
        freestanding = [m for m in members if m in spec_by_id]

    def _main_sort_key(oid: str) -> tuple[Any, ...]:
        area = int(spec_by_id[oid].get("w", 0) or 0) * int(
            spec_by_id[oid].get("h", 0) or 0
        )
        return (
            0 if oid in anchors else 1,
            int(priority_map.get(oid, 9)),
            0 if oid in access_required else 1,
            -area,
            oid,
        )

    freestanding.sort(key=_main_sort_key)

    base_row_ids: list[str] = []
    back_row_ids: list[str] = []

    for oid in freestanding:
        pr = int(priority_map.get(oid, 9))
        if pr >= 3:
            back_row_ids.append(oid)
        else:
            base_row_ids.append(oid)

    if not base_row_ids and back_row_ids:
        base_row_ids = back_row_ids
        back_row_ids = []

    placements = _seed_place_freestanding(
        spec_by_id=spec_by_id,
        grid_mm=grid_mm,
        base_row_ids=base_row_ids,
        back_row_ids=back_row_ids,
        strategy=strategy,
    )

    # ---- place dependents relative to their base
    unresolved = relations[:]
    for _ in range(len(relations) + 3):
        progressed = False
        still_unresolved: list[dict[str, Any]] = []

        for rel in unresolved:
            a = rel.get("a")
            b = rel.get("b")
            if not isinstance(a, str) or not isinstance(b, str):
                continue

            if a in placements:
                continue
            if b not in placements:
                still_unresolved.append(rel)
                continue

            placed = _seed_place_relative(
                rel=rel,
                cluster=cluster,
                spec_by_id=spec_by_id,
                placements=placements,
                grid_mm=grid_mm,
                ctx=None,
            )
            if placed is None:
                still_unresolved.append(rel)
                continue

            placements[a] = placed
            progressed = True

        unresolved = still_unresolved
        if not progressed:
            break

    # ---- any leftover members: place on a far fallback row
    fallback_ids = [m for m in members if m in spec_by_id and m not in placements]
    fallback_y = _seed_fallback_y(
        spec_by_id=spec_by_id,
        placements=placements,
        grid_mm=grid_mm,
    )
    fallback_x = 0
    fallback_gap = max(grid_mm, 150)

    for oid in fallback_ids:
        spec = spec_by_id[oid]
        rot = _seed_first_allowed_rotation(spec, prefer=0)
        w, h = _seed_rotated_wh(spec, rot)

        x = _seed_snap_up_value(fallback_x, grid_mm)
        placements[oid] = {
            "id": oid,
            "x": int(x),
            "y": int(fallback_y),
            "rot": int(rot),
        }
        fallback_x = x + w + fallback_gap

    return _seed_normalize_to_origin(list(placements.values()))


def _seed_local_layout(
    cluster: dict[str, Any], *, strategy: str = "row"
) -> list[dict[str, Any]]:
    return _build_fallback_pack_layout(cluster, strategy=strategy)


def _seed_place_freestanding(
    *,
    spec_by_id: dict[str, dict[str, Any]],
    grid_mm: int,
    base_row_ids: list[str],
    back_row_ids: list[str],
    strategy: str,
) -> dict[str, dict[str, Any]]:
    if strategy == "compact_wrap":
        return _seed_place_wrapped_layout(
            object_ids=[*base_row_ids, *back_row_ids],
            spec_by_id=spec_by_id,
            grid_mm=grid_mm,
            compactness=1.20,
        )
    if strategy == "balanced_wrap":
        return _seed_place_wrapped_layout(
            object_ids=[*base_row_ids, *back_row_ids],
            spec_by_id=spec_by_id,
            grid_mm=grid_mm,
            compactness=1.45,
        )
    if strategy == "two_column":
        return _seed_place_two_column_layout(
            object_ids=[*base_row_ids, *back_row_ids],
            spec_by_id=spec_by_id,
            grid_mm=grid_mm,
        )
    return _seed_place_row_layout(
        spec_by_id=spec_by_id,
        grid_mm=grid_mm,
        base_row_ids=base_row_ids,
        back_row_ids=back_row_ids,
    )


def _seed_place_row_layout(
    *,
    spec_by_id: dict[str, dict[str, Any]],
    grid_mm: int,
    base_row_ids: list[str],
    back_row_ids: list[str],
) -> dict[str, dict[str, Any]]:
    placements: dict[str, dict[str, Any]] = {}
    row_gap = max(grid_mm, 150)
    back_row_gap = max(grid_mm, 250)

    x_cursor = 0
    for oid in base_row_ids:
        spec = spec_by_id[oid]
        rot = _seed_first_allowed_rotation(spec, prefer=0)
        w, _ = _seed_rotated_wh(spec, rot)
        x = _seed_snap_up_value(x_cursor, grid_mm)
        placements[oid] = {"id": oid, "x": int(x), "y": 0, "rot": int(rot)}
        x_cursor = x + w + row_gap

    max_main_h = 0
    for oid in base_row_ids:
        spec = spec_by_id[oid]
        rot = placements[oid]["rot"]
        _, h = _seed_rotated_wh(spec, rot)
        max_main_h = max(max_main_h, h)

    back_y = -_seed_snap_up_value(max_main_h + back_row_gap, grid_mm)
    back_x_cursor = 0
    for oid in back_row_ids:
        spec = spec_by_id[oid]
        rot = _seed_first_allowed_rotation(spec, prefer=0)
        w, _ = _seed_rotated_wh(spec, rot)
        x = _seed_snap_up_value(back_x_cursor, grid_mm)
        placements[oid] = {"id": oid, "x": int(x), "y": int(back_y), "rot": int(rot)}
        back_x_cursor = x + w + row_gap

    return placements


def _seed_place_wrapped_layout(
    *,
    object_ids: list[str],
    spec_by_id: dict[str, dict[str, Any]],
    grid_mm: int,
    compactness: float,
) -> dict[str, dict[str, Any]]:
    placements: dict[str, dict[str, Any]] = {}
    if not object_ids:
        return placements

    row_gap = max(grid_mm, 150)
    target_span = _seed_wrap_target_span(
        object_ids=object_ids,
        spec_by_id=spec_by_id,
        grid_mm=grid_mm,
        compactness=compactness,
    )
    x_cursor = 0
    y_cursor = 0
    row_height = 0

    for oid in object_ids:
        spec = spec_by_id[oid]
        rot = _seed_first_allowed_rotation(spec, prefer=0)
        w, h = _seed_rotated_wh(spec, rot)
        proposed_x = _seed_snap_up_value(x_cursor, grid_mm)
        if proposed_x > 0 and proposed_x + w > target_span:
            y_cursor += row_height + row_gap
            x_cursor = 0
            row_height = 0
            proposed_x = 0
        placements[oid] = {
            "id": oid,
            "x": int(proposed_x),
            "y": int(y_cursor),
            "rot": int(rot),
        }
        x_cursor = proposed_x + w + row_gap
        row_height = max(row_height, h)

    return placements


def _seed_place_two_column_layout(
    *,
    object_ids: list[str],
    spec_by_id: dict[str, dict[str, Any]],
    grid_mm: int,
) -> dict[str, dict[str, Any]]:
    placements: dict[str, dict[str, Any]] = {}
    if not object_ids:
        return placements

    row_gap = max(grid_mm, 150)
    left_ids: list[str] = []
    right_ids: list[str] = []
    left_height = 0
    right_height = 0
    left_width = 0

    for oid in object_ids:
        spec = spec_by_id[oid]
        rot = _seed_first_allowed_rotation(spec, prefer=0)
        w, h = _seed_rotated_wh(spec, rot)
        if left_height <= right_height:
            left_ids.append(oid)
            left_height += h + row_gap
            left_width = max(left_width, w)
        else:
            right_ids.append(oid)
            right_height += h + row_gap

    y_cursor = 0
    for oid in left_ids:
        spec = spec_by_id[oid]
        rot = _seed_first_allowed_rotation(spec, prefer=0)
        _, h = _seed_rotated_wh(spec, rot)
        placements[oid] = {"id": oid, "x": 0, "y": int(y_cursor), "rot": int(rot)}
        y_cursor += h + row_gap

    x_right = _seed_snap_up_value(left_width + row_gap, grid_mm)
    y_cursor = 0
    for oid in right_ids:
        spec = spec_by_id[oid]
        rot = _seed_first_allowed_rotation(spec, prefer=0)
        _, h = _seed_rotated_wh(spec, rot)
        placements[oid] = {
            "id": oid,
            "x": int(x_right),
            "y": int(y_cursor),
            "rot": int(rot),
        }
        y_cursor += h + row_gap

    return placements


def _seed_wrap_target_span(
    *,
    object_ids: list[str],
    spec_by_id: dict[str, dict[str, Any]],
    grid_mm: int,
    compactness: float,
) -> int:
    total_area = 0
    max_width = 0
    for oid in object_ids:
        spec = spec_by_id[oid]
        rot = _seed_first_allowed_rotation(spec, prefer=0)
        w, h = _seed_rotated_wh(spec, rot)
        total_area += w * h
        max_width = max(max_width, w)

    estimated_span = int(round(math.sqrt(max(total_area, 1)) * compactness))
    return max(max_width, _seed_snap_up_value(estimated_span, grid_mm))


def _seed_fallback_y(
    *,
    spec_by_id: dict[str, dict[str, Any]],
    placements: dict[str, dict[str, Any]],
    grid_mm: int,
) -> int:
    if not placements:
        return 0

    max_y = 0
    max_h = 0
    for placement in placements.values():
        if not isinstance(placement, dict):
            continue
        spec = spec_by_id.get(str(placement.get("id") or ""))
        if spec is None:
            continue
        rot = int(placement.get("rot", 0) or 0)
        _, h = _seed_rotated_wh(spec, rot)
        max_y = max(max_y, int(placement.get("y", 0)) + h)
        max_h = max(max_h, h)
    return max_y + _seed_snap_up_value(max(max_h, 600) + max(grid_mm, 250), grid_mm)


def _seed_build_priority_map(cluster: dict[str, Any]) -> dict[str, int]:
    rank_map = {"anchor": 0, "primary": 1, "secondary": 2, "optional": 3}
    out: dict[str, int] = {}

    decisions = cluster.get("decisions", [])
    if not isinstance(decisions, list):
        return out

    for d in decisions:
        if not isinstance(d, dict):
            continue
        oid = d.get("object_type") or d.get("category")
        if not isinstance(oid, str) or not oid:
            continue
        priority = str(d.get("priority", "secondary")).lower()
        out[oid] = rank_map.get(priority, 2)

    return out


def _seed_collect_access_required(cluster: dict[str, Any]) -> set[str]:
    out: set[str] = set()

    rules = cluster.get("cluster_rules", {})
    if isinstance(rules, dict):
        ar = rules.get("access_requirements")
        if isinstance(ar, list):
            for item in ar:
                if not isinstance(item, dict):
                    continue
                if (
                    item.get("type") == "front_clearance"
                    and item.get("required") is True
                ):
                    oid = item.get("id")
                    if isinstance(oid, str) and oid:
                        out.add(oid)

    hard = cluster.get("hard_constraints", [])
    if isinstance(hard, list):
        for c in hard:
            if not isinstance(c, dict):
                continue
            if (
                c.get("type") == "requires_access"
                and c.get("mode") == "front_clearance"
            ):
                oid = c.get("id")
                if isinstance(oid, str) and oid:
                    out.add(oid)

    return out


def _seed_first_allowed_rotation(spec: dict[str, Any], prefer: int = 0) -> int:
    allowed = spec.get("allowed_rotations")
    if isinstance(allowed, list) and allowed:
        allowed_norm = [int(x) % 360 for x in allowed if isinstance(x, int)]
        if prefer in allowed_norm:
            return int(prefer)
        if 0 in allowed_norm:
            return 0
        return int(allowed_norm[0])
    return int(prefer) % 360


def _seed_rotated_wh(spec: dict[str, Any], rot: int) -> tuple[int, int]:
    w = int(spec.get("w", 0) or 0)
    h = int(spec.get("h", 0) or 0)
    if rot % 360 in (90, 270):
        return h, w
    return w, h


def _seed_snap_up_value(v: int, grid_mm: int) -> int:
    if grid_mm <= 0:
        return int(v)
    return int(math.ceil(v / grid_mm) * grid_mm)


def _seed_snap_down_value(v: int, grid_mm: int) -> int:
    if grid_mm <= 0:
        return int(v)
    return int(math.floor(v / grid_mm) * grid_mm)


def _seed_snap_nearest_value(v: int, grid_mm: int) -> int:
    if grid_mm <= 0:
        return int(v)
    down = math.floor(v / grid_mm) * grid_mm
    up = down + grid_mm
    return int(down if abs(v - down) <= abs(up - v) else up)


def _seed_pick_grid_in_interval(
    current: int,
    interval: tuple[int, int],
    grid_mm: int,
) -> int:
    lo, hi = interval
    if lo > hi:
        lo, hi = hi, lo

    if grid_mm <= 0:
        return int(min(max(current, lo), hi))

    first = int(math.ceil(lo / grid_mm) * grid_mm)
    last = int(math.floor(hi / grid_mm) * grid_mm)

    if first <= last:
        cand = _seed_snap_nearest_value(current, grid_mm)
        if cand < first:
            return first
        if cand > last:
            return last
        return int(cand)

    mid = int((lo + hi) // 2)
    return _seed_snap_nearest_value(mid, grid_mm)


def _seed_rotate_side(side: str, rot: int) -> str:
    return _rotate_side_ccw_90s_contract(side, rot) or side


def _seed_get_front_base(
    cluster: dict[str, Any], obj_id: str, spec: dict[str, Any]
) -> str:
    rules = cluster.get("cluster_rules", {})
    if isinstance(rules, dict):
        facing = rules.get("facing", {})
        if isinstance(facing, dict):
            f = facing.get(obj_id)
            if isinstance(f, dict):
                front = f.get("front")
                if front in {"top", "bottom", "left", "right"}:
                    return str(front)

    spec_front = spec.get("front")
    if spec_front in {"top", "bottom", "left", "right"}:
        return str(spec_front)
    return "top"


def _seed_opposite_side(side: str) -> str:
    return {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}.get(
        side, side
    )


def _seed_resolve_edge_token_to_base_side(edge_token: str, front_base: str) -> str:
    if edge_token in {"top", "bottom", "left", "right"}:
        return edge_token
    if edge_token == "front":
        return front_base
    if edge_token == "back":
        return _seed_opposite_side(front_base)
    return "top"


def _seed_place_relative(
    *,
    rel: dict[str, Any],
    cluster: dict[str, Any],
    spec_by_id: dict[str, dict[str, Any]],
    placements: dict[str, dict[str, Any]],
    grid_mm: int,
    ctx: SemanticContext | None = None,
) -> dict[str, Any] | None:
    a = rel.get("a")
    b = rel.get("b")
    if not isinstance(a, str) or not isinstance(b, str):
        return None
    if a not in spec_by_id or b not in spec_by_id or b not in placements:
        return None

    a_spec = spec_by_id[a]
    b_spec = spec_by_id[b]
    b_pl = placements[b]

    b_rot = int(b_pl["rot"]) % 360
    bx = int(b_pl["x"])
    by = int(b_pl["y"])
    bw, bh = _seed_rotated_wh(b_spec, b_rot)

    a_rot = _seed_first_allowed_rotation(a_spec, prefer=0)
    aw, ah = _seed_rotated_wh(a_spec, a_rot)
    span_mode = str(rel.get("span") or "any")

    ctype = rel.get("type") or rel.get("kind")

    if ctype == "contain_in":
        x_lo = bx
        x_hi = bx + bw - aw
        y_lo = by
        y_hi = by + bh - ah
        if x_hi < x_lo or y_hi < y_lo:
            return None

        x_mid = bx + (bw - aw) // 2
        y_mid = by + (bh - ah) // 2

        x = _seed_pick_grid_in_interval(x_mid, (x_lo, x_hi), grid_mm)
        y = _seed_pick_grid_in_interval(y_mid, (y_lo, y_hi), grid_mm)
        return {"id": a, "x": int(x), "y": int(y), "rot": int(a_rot)}

    qualifier_local: str | None = None

    if ctype == "dock_to_edge":
        b_edge = rel.get("b_edge")
        if not isinstance(b_edge, str):
            return None
        front_base = _seed_get_front_base(cluster, b, b_spec)
        base_side = _seed_resolve_edge_token_to_base_side(b_edge, front_base)

    elif ctype == "anchor_side":
        side_options = _seed_semantic_side_options(rel)
        side = rel.get("side")
        if not isinstance(side, str) and not side_options:
            return None
        candidate_sides = side_options or ([side] if isinstance(side, str) else [])
        best_candidate: dict[str, Any] | None = None
        best_score: tuple[float, int, int, int, int] | None = None
        for candidate_side in candidate_sides:
            base_side, qualifier_local = _seed_resolve_anchor_side(candidate_side)
            if base_side == "head":
                base_side = "top"
            elif base_side == "foot":
                base_side = "bottom"
            candidate = _seed_place_relative_with_side(
                cluster=cluster,
                rel=rel,
                spec_by_id=spec_by_id,
                placements=placements,
                grid_mm=grid_mm,
                a=a,
                b=b,
                a_spec=a_spec,
                b_spec=b_spec,
                a_rot=a_rot,
                aw=aw,
                ah=ah,
                b_rot=b_rot,
                bx=bx,
                by=by,
                bw=bw,
                bh=bh,
                mapped_side=_seed_rotate_side(base_side, b_rot),
                qualifier_local=qualifier_local,
                span_mode=span_mode,
            )
            if candidate is None:
                continue
            score = _seed_relative_candidate_score(
                candidate=candidate,
                spec_by_id=spec_by_id,
                a_spec=a_spec,
                placements=placements,
            )
            semantic_score = _score_relative_candidate_semantic(
                candidate=candidate,
                rel=rel,
                ctx=ctx,
            )
            ranked_score = (-semantic_score, *score)
            if str(rel.get("selection") or "best_fit") == "first_fit" and score[0] == 0:
                return candidate
            if best_score is None or ranked_score < best_score:
                best_candidate = candidate
                best_score = ranked_score
        return best_candidate

    else:
        return None

    mapped_side = _seed_rotate_side(base_side, b_rot)
    return _seed_place_relative_with_side(
        cluster=cluster,
        rel=rel,
        spec_by_id=spec_by_id,
        placements=placements,
        grid_mm=grid_mm,
        a=a,
        b=b,
        a_spec=a_spec,
        b_spec=b_spec,
        a_rot=a_rot,
        aw=aw,
        ah=ah,
        b_rot=b_rot,
        bx=bx,
        by=by,
        bw=bw,
        bh=bh,
        mapped_side=mapped_side,
        qualifier_local=qualifier_local,
        span_mode=span_mode,
    )


def _seed_semantic_side_options(rel: dict[str, Any]) -> list[str]:
    raw = rel.get("side_options")
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        token = _seed_normalize_anchor_side_token(item)
        if token in {
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
        }:
            out.append(token)
    return out


def _seed_normalize_anchor_side_token(value: str) -> str:
    token = value.strip().lower()
    mapping = {
        "front": "head",
        "back": "foot",
        "front_left": "head_left",
        "front_right": "head_right",
        "back_left": "foot_left",
        "back_right": "foot_right",
        "front_left_up": "head_left",
        "front_right_up": "head_right",
        "upper_left": "head_left",
        "upper_right": "head_right",
        "left_up": "head_left",
        "right_up": "head_right",
        "left_upper": "head_left",
        "right_upper": "head_right",
        "left_side": "left",
        "right_side": "right",
        "beside_left": "left",
        "beside_right": "right",
        "in_front": "head",
        "behind": "foot",
    }
    return mapping.get(token, token)


def _seed_choose_relative_rotation(
    *,
    cluster: dict[str, Any],
    obj_id: str,
    spec: dict[str, Any],
    orientation: str | None,
    base_id: str,
    base_spec: dict[str, Any],
    base_rot: int,
    mapped_side: str,
) -> int:
    allowed = spec.get("allowed_rotations")
    allowed_rotations = (
        [int(value) % 360 for value in allowed if isinstance(value, int)]
        if isinstance(allowed, list)
        else [0, 90, 180, 270]
    )
    if not allowed_rotations:
        return 0

    if orientation == "face_base":
        desired_front = _seed_opposite_side(mapped_side)
        for rot in allowed_rotations:
            if (
                _seed_rotate_side(_seed_get_front_base(cluster, obj_id, spec), rot)
                == desired_front
            ):
                return rot

    if orientation == "same_direction":
        base_front = _seed_get_front_base(cluster, base_id, base_spec)
        desired_front = _seed_rotate_side(base_front, base_rot)
        for rot in allowed_rotations:
            if (
                _seed_rotate_side(_seed_get_front_base(cluster, obj_id, spec), rot)
                == desired_front
            ):
                return rot

    return _seed_first_allowed_rotation(spec, prefer=0)


def _seed_place_relative_with_side(
    *,
    cluster: dict[str, Any],
    rel: dict[str, Any],
    spec_by_id: dict[str, dict[str, Any]],
    placements: dict[str, dict[str, Any]],
    grid_mm: int,
    a: str,
    b: str,
    a_spec: dict[str, Any],
    b_spec: dict[str, Any],
    a_rot: int,
    aw: int,
    ah: int,
    b_rot: int,
    bx: int,
    by: int,
    bw: int,
    bh: int,
    mapped_side: str,
    qualifier_local: str | None,
    span_mode: str,
) -> dict[str, Any] | None:
    orientation = rel.get("orientation")
    if isinstance(orientation, str):
        a_rot = _seed_choose_relative_rotation(
            cluster=cluster,
            obj_id=a,
            spec=a_spec,
            orientation=orientation,
            base_id=b,
            base_spec=b_spec,
            base_rot=b_rot,
            mapped_side=mapped_side,
        )
        aw, ah = _seed_rotated_wh(a_spec, a_rot)

    gap_min = int(rel.get("gap_min", 0) or 0)
    gap_max = int(rel.get("gap_max", 0) or 0)

    qualifier_world: str | None = None
    if qualifier_local in {"left", "right"}:
        qualifier_world = _seed_rotate_side(qualifier_local, b_rot)

    if mapped_side == "top":
        y = _seed_pick_grid_in_interval(
            by + bh + gap_min,
            (by + bh + gap_min, by + bh + gap_max),
            grid_mm,
        )
        x = _seed_pick_anchor_axis_start(
            base_start=bx,
            base_extent=bw,
            a_extent=aw,
            qualifier_world=qualifier_world,
            span_mode=span_mode,
            grid_mm=grid_mm,
        )
        return {"id": a, "x": int(x), "y": int(y), "rot": int(a_rot)}

    if mapped_side == "bottom":
        y = _seed_pick_grid_in_interval(
            by - ah - gap_min,
            (by - ah - gap_max, by - ah - gap_min),
            grid_mm,
        )
        x = _seed_pick_anchor_axis_start(
            base_start=bx,
            base_extent=bw,
            a_extent=aw,
            qualifier_world=qualifier_world,
            span_mode=span_mode,
            grid_mm=grid_mm,
        )
        return {"id": a, "x": int(x), "y": int(y), "rot": int(a_rot)}

    if mapped_side == "left":
        x = _seed_pick_grid_in_interval(
            bx - aw - gap_min,
            (bx - aw - gap_max, bx - aw - gap_min),
            grid_mm,
        )
        y = _seed_pick_anchor_axis_start(
            base_start=by,
            base_extent=bh,
            a_extent=ah,
            qualifier_world=qualifier_world,
            span_mode=span_mode,
            grid_mm=grid_mm,
        )
        return {"id": a, "x": int(x), "y": int(y), "rot": int(a_rot)}

    if mapped_side == "right":
        x = _seed_pick_grid_in_interval(
            bx + bw + gap_min,
            (bx + bw + gap_min, bx + bw + gap_max),
            grid_mm,
        )
        y = _seed_pick_anchor_axis_start(
            base_start=by,
            base_extent=bh,
            a_extent=ah,
            qualifier_world=qualifier_world,
            span_mode=span_mode,
            grid_mm=grid_mm,
        )
        return {"id": a, "x": int(x), "y": int(y), "rot": int(a_rot)}

    return None


def _seed_relative_candidate_score(
    *,
    candidate: dict[str, Any],
    spec_by_id: dict[str, dict[str, Any]],
    a_spec: dict[str, Any],
    placements: dict[str, dict[str, Any]],
) -> tuple[int, int, int, int]:
    aw, ah = _seed_rotated_wh(a_spec, int(candidate.get("rot", 0) or 0))
    ax1 = int(candidate.get("x", 0) or 0)
    ay1 = int(candidate.get("y", 0) or 0)
    ax2 = ax1 + aw
    ay2 = ay1 + ah

    overlap_count = 0
    overlap_area = 0
    min_x = ax1
    min_y = ay1
    max_x = ax2
    max_y = ay2

    for placed in placements.values():
        if not isinstance(placed, dict):
            continue
        placed_id = placed.get("id")
        if not isinstance(placed_id, str):
            continue
        placed_spec = spec_by_id.get(placed_id)
        if not isinstance(placed_spec, dict):
            continue
        px1 = int(placed.get("x", 0) or 0)
        py1 = int(placed.get("y", 0) or 0)
        pw, ph = _seed_rotated_wh(placed_spec, int(placed.get("rot", 0) or 0))
        px2 = px1 + pw
        py2 = py1 + ph
        min_x = min(min_x, px1)
        min_y = min(min_y, py1)
        max_x = max(max_x, px2)
        max_y = max(max_y, py2)

        overlap_w = max(0, min(ax2, px2) - max(ax1, px1))
        overlap_h = max(0, min(ay2, py2) - max(ay1, py1))
        if overlap_w > 0 and overlap_h > 0:
            overlap_count += 1
            overlap_area += overlap_w * overlap_h

    bbox_area = max(0, max_x - min_x) * max(0, max_y - min_y)
    return (overlap_count, overlap_area, bbox_area, ax1 + ay1)


def _seed_normalize_to_origin(
    placements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    canonical = _canonicalize_local_placements(placements)
    if not canonical:
        return canonical

    min_x = min(int(p["x"]) for p in canonical)
    min_y = min(int(p["y"]) for p in canonical)

    out: list[dict[str, Any]] = []
    for p in canonical:
        out.append(
            {
                "id": p["id"],
                "x": int(p["x"] - min_x),
                "y": int(p["y"] - min_y),
                "rot": int(p["rot"]) % 360,
            }
        )
    out.sort(key=lambda item: item["id"])
    return out


def _validate_local_placements_full(
    placements: Any,
    expected_ids: list[str],
) -> tuple[bool, str]:
    if not isinstance(placements, list) or not placements:
        return False, "local_placements missing/empty"

    seen: list[str] = []
    for p in placements:
        if not isinstance(p, dict):
            continue
        pid = p.get("id")
        if isinstance(pid, str) and pid:
            seen.append(pid)

    if not seen:
        return False, "local_placements has no valid id entries"

    missing = [cid for cid in expected_ids if cid not in seen]
    extra = [cid for cid in seen if cid not in expected_ids]
    dups = sorted({cid for cid in seen if seen.count(cid) > 1})

    if missing:
        return False, f"missing ids: {missing}"
    if extra:
        return False, f"unexpected ids: {extra}"
    if dups:
        return False, f"duplicate ids: {dups}"
    if len(seen) != len(expected_ids):
        return False, "count mismatch"
    return True, "ok"


# ---------------------------
# Deterministic repair helpers
# ---------------------------


def _build_search_seed_states(cluster: dict[str, Any]) -> list[list[dict[str, Any]]]:
    seeds: list[list[dict[str, Any]]] = []
    seen: set[str] = set()

    base_layouts = [
        _seed_local_layout(cluster, strategy="row"),
        _seed_local_layout(cluster, strategy="compact_wrap"),
        _seed_local_layout(cluster, strategy="balanced_wrap"),
        _seed_local_layout(cluster, strategy="two_column"),
    ]

    for base in base_layouts:
        if not base:
            continue
        for candidate in (
            _canonicalize_local_placements(base),
            _mirror_seed_layout(base, cluster, axis="x"),
            _mirror_seed_layout(base, cluster, axis="y"),
            _rotate_seed_layout(base, cluster, degrees=90),
            _rotate_seed_layout(base, cluster, degrees=180),
            _rotate_seed_layout(base, cluster, degrees=270),
        ):
            if not candidate:
                continue
            signature = _placements_signature(candidate)
            if signature in seen:
                continue
            seen.add(signature)
            seeds.append(candidate)

    return seeds


def _rotate_seed_layout(
    placements: list[dict[str, Any]],
    cluster: dict[str, Any],
    *,
    degrees: int,
) -> list[dict[str, Any]]:
    canonical = _canonicalize_local_placements(placements)
    if not canonical:
        return []

    degrees = int(degrees) % 360
    if degrees not in {90, 180, 270}:
        return []

    spec_by_id = {
        s["id"]: s
        for s in _build_object_specs(cluster)
        if isinstance(s, dict) and isinstance(s.get("id"), str)
    }
    rects: list[dict[str, Any]] = []
    for placement in canonical:
        spec = spec_by_id.get(placement["id"])
        if spec is None:
            return []
        w = int(spec.get("w", 0) or 0)
        h = int(spec.get("h", 0) or 0)
        rot = int(placement["rot"]) % 360
        if rot in (90, 270):
            w, h = h, w
        rects.append(
            {
                "id": placement["id"],
                "x": int(placement["x"]),
                "y": int(placement["y"]),
                "w": w,
                "h": h,
                "rot": rot,
            }
        )

    bbox = _local_bbox_from_rects(rects)
    width = int(bbox["max_x"]) - int(bbox["min_x"])
    height = int(bbox["max_y"]) - int(bbox["min_y"])
    rotated: list[dict[str, Any]] = []
    for rect in rects:
        x = int(rect["x"])
        y = int(rect["y"])
        w = int(rect["w"])
        h = int(rect["h"])
        rot = int(rect["rot"]) % 360

        if degrees == 90:
            new_x = height - (y + h)
            new_y = x
        elif degrees == 180:
            new_x = width - (x + w)
            new_y = height - (y + h)
        else:
            new_x = y
            new_y = width - (x + w)

        rotated.append(
            {
                "id": rect["id"],
                "x": int(new_x),
                "y": int(new_y),
                "rot": int((rot + degrees) % 360),
            }
        )

    return _seed_normalize_to_origin(rotated)


def _mirror_seed_layout(
    placements: list[dict[str, Any]],
    cluster: dict[str, Any],
    *,
    axis: str,
) -> list[dict[str, Any]]:
    canonical = _canonicalize_local_placements(placements)
    if not canonical:
        return []

    spec_by_id = {
        s["id"]: s
        for s in _build_object_specs(cluster)
        if isinstance(s, dict) and isinstance(s.get("id"), str)
    }
    rects: list[dict[str, Any]] = []
    for placement in canonical:
        spec = spec_by_id.get(placement["id"])
        if spec is None:
            return []
        w = int(spec.get("w", 0) or 0)
        h = int(spec.get("h", 0) or 0)
        rot = int(placement["rot"]) % 360
        if rot in (90, 270):
            w, h = h, w
        rects.append(
            {
                "id": placement["id"],
                "x": int(placement["x"]),
                "y": int(placement["y"]),
                "w": w,
                "h": h,
                "rot": rot,
            }
        )

    bbox = _local_bbox_from_rects(rects)
    max_x = int(bbox["max_x"])
    max_y = int(bbox["max_y"])

    mirrored: list[dict[str, Any]] = []
    for rect in rects:
        x = int(rect["x"])
        y = int(rect["y"])
        w = int(rect["w"])
        h = int(rect["h"])
        if axis == "x":
            x = max_x - (x + w)
        elif axis == "y":
            y = max_y - (y + h)
        mirrored.append(
            {
                "id": rect["id"],
                "x": x,
                "y": y,
                "rot": int(rect["rot"]),
            }
        )

    return _seed_normalize_to_origin(mirrored)


def _canonicalize_local_placements(
    placements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in placements:
        if not isinstance(p, dict):
            continue
        pid = p.get("id")
        if not isinstance(pid, str) or not pid:
            continue
        out.append(
            {
                "id": pid,
                "x": int(p.get("x", 0)),
                "y": int(p.get("y", 0)),
                "rot": int(p.get("rot", 0)) % 360,
            }
        )
    out.sort(key=lambda item: item["id"])
    return out


def _placements_signature(placements: list[dict[str, Any]]) -> str:
    canonical = _canonicalize_local_placements(placements)
    return json.dumps(canonical, ensure_ascii=True, separators=(",", ":"))


def _move_key(move: dict[str, Any]) -> str:
    payload = {
        "reason": move.get("reason"),
        "move_object": move.get("move_object"),
        "dx": move.get("dx"),
        "dy": move.get("dy"),
        "new_x": move.get("new_x"),
        "new_y": move.get("new_y"),
        "new_rot": move.get("new_rot"),
    }
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _choose_best_single_patch(
    *,
    cluster: dict[str, Any],
    access_clearance_ratio: float,
    tool_output: dict[str, Any],
    placements: list[dict[str, Any]],
    tried_patch_keys: set[str],
    family: str | None = None,
    ctx: SemanticContext | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None, dict[str, Any] | None]:
    baseline_rank = _tool_rank(tool_output)
    grid_mm = _extract_grid_mm(cluster)

    candidates = _collect_patch_candidates(
        cluster=cluster,
        access_clearance_ratio=access_clearance_ratio,
        tool_output=tool_output,
        placements=placements,
        grid_mm=grid_mm,
        family=family,
        ctx=ctx,
    )

    best_move: dict[str, Any] | None = None
    best_patched: list[dict[str, Any]] | None = None
    best_eval: dict[str, Any] | None = None
    best_sort_key: tuple[Any, ...] | None = None
    best_rank: tuple[Any, ...] | None = None

    for move in candidates:
        key = _move_key(move)
        if key in tried_patch_keys:
            continue

        patched = _apply_single_move(placements, move)
        eval_out = _run_verifier_once(
            cluster=cluster,
            placements=patched,
            access_clearance_ratio=access_clearance_ratio,
        )

        candidate_rank = _tool_rank(eval_out)
        move_cost = _movement_cost(move)
        reason_rank = _reason_rank(move.get("reason"))

        sort_key = (
            *candidate_rank,
            move_cost,
            reason_rank,
            move.get("move_object") or "",
            key,
        )

        if best_sort_key is None or sort_key < best_sort_key:
            best_sort_key = sort_key
            best_rank = candidate_rank
            best_move = move
            best_patched = patched
            best_eval = eval_out

    if best_move is None or best_rank is None:
        return None, None, None

    if not _rank_better(best_rank, baseline_rank):
        return None, None, None

    return best_move, best_patched, best_eval


def _collect_ranked_patch_successors(
    *,
    cluster: dict[str, Any],
    access_clearance_ratio: float,
    tool_output: dict[str, Any],
    placements: list[dict[str, Any]],
    limit: int,
) -> list[tuple[tuple[Any, ...], list[dict[str, Any]], dict[str, Any]]]:
    baseline_rank = _tool_rank(tool_output)
    grid_mm = _extract_grid_mm(cluster)
    candidates = _collect_patch_candidates(
        cluster=cluster,
        access_clearance_ratio=access_clearance_ratio,
        tool_output=tool_output,
        placements=placements,
        grid_mm=grid_mm,
    )

    best_by_signature: dict[
        str,
        tuple[tuple[Any, ...], tuple[Any, ...], list[dict[str, Any]], dict[str, Any]],
    ] = {}

    for move in candidates:
        patched = _apply_single_move(placements, move)
        eval_out = _run_verifier_once(
            cluster=cluster,
            placements=patched,
            access_clearance_ratio=access_clearance_ratio,
        )
        candidate_rank = _tool_rank(eval_out)
        if not _rank_better(candidate_rank, baseline_rank):
            continue

        score_key = (
            *candidate_rank,
            _movement_cost(move),
            _reason_rank(move.get("reason")),
            move.get("move_object") or "",
            _move_key(move),
        )
        canonical = _canonicalize_local_placements(patched)
        signature = _placements_signature(canonical)
        current_best = best_by_signature.get(signature)
        if current_best is None or score_key < current_best[0]:
            best_by_signature[signature] = (
                score_key,
                candidate_rank,
                canonical,
                eval_out,
            )

    ranked = sorted(best_by_signature.values(), key=lambda item: item[0])
    return [(item[1], item[2], item[3]) for item in ranked[: max(0, int(limit))]]


def _greedy_improve_valid_layout(
    *,
    cluster: dict[str, Any],
    access_clearance_ratio: float,
    placements: list[dict[str, Any]],
    verifier_eval: dict[str, Any],
    max_rounds: int,
    family: str | None = None,
    ctx: SemanticContext | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    current = _canonicalize_local_placements(placements)
    current_eval = verifier_eval if isinstance(verifier_eval, dict) else {}
    if current_eval.get("result") != "VALID":
        current_eval = _run_verifier_once(
            cluster=cluster,
            placements=current,
            access_clearance_ratio=access_clearance_ratio,
        )
    if current_eval.get("result") != "VALID":
        return current, current_eval, 0

    rounds = 0
    current_rank = _tool_rank(current_eval)
    tried_patch_keys_by_state: dict[str, set[str]] = {}

    for _ in range(max(0, int(max_rounds))):
        state_sig = _placements_signature(current)
        tried_patch_keys = tried_patch_keys_by_state.setdefault(state_sig, set())
        selected_move, patched, patched_eval = _choose_best_single_patch(
            cluster=cluster,
            access_clearance_ratio=access_clearance_ratio,
            tool_output=current_eval,
            placements=current,
            tried_patch_keys=tried_patch_keys,
            family=family,
            ctx=ctx,
        )
        if (
            selected_move is None
            or patched is None
            or not isinstance(patched_eval, dict)
            or patched_eval.get("result") != "VALID"
        ):
            break

        patched_rank = _tool_rank(patched_eval)
        if not _rank_better(patched_rank, current_rank):
            break

        tried_patch_keys.add(_move_key(selected_move))
        current = _canonicalize_local_placements(patched)
        current_eval = patched_eval
        current_rank = patched_rank
        rounds += 1

    return current, current_eval, rounds


def _repair_invalid_seed_layout(
    *,
    cluster: dict[str, Any],
    access_clearance_ratio: float,
    placements: list[dict[str, Any]],
    verifier_eval: dict[str, Any],
    max_rounds: int,
    family: str | None = None,
    ctx: SemanticContext | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    current = _canonicalize_local_placements(placements)
    current_eval = verifier_eval if isinstance(verifier_eval, dict) else {}
    if current_eval.get("result") == "VALID":
        return current, current_eval, 0

    rounds = 0
    tried_patch_keys_by_state: dict[str, set[str]] = {}

    for _ in range(max(0, int(max_rounds))):
        state_sig = _placements_signature(current)
        tried_patch_keys = tried_patch_keys_by_state.setdefault(state_sig, set())
        selected_move, patched, patched_eval = _choose_best_single_patch(
            cluster=cluster,
            access_clearance_ratio=access_clearance_ratio,
            tool_output=current_eval,
            placements=current,
            tried_patch_keys=tried_patch_keys,
            family=family,
            ctx=ctx,
        )
        if (
            selected_move is None
            or patched is None
            or not isinstance(patched_eval, dict)
        ):
            break

        tried_patch_keys.add(_move_key(selected_move))
        current = _canonicalize_local_placements(patched)
        current_eval = patched_eval
        rounds += 1

        if current_eval.get("result") == "VALID":
            return current, current_eval, rounds

    return current, current_eval, rounds


def _collect_patch_candidates(
    *,
    cluster: dict[str, Any],
    access_clearance_ratio: float,
    tool_output: dict[str, Any],
    placements: list[dict[str, Any]],
    grid_mm: int,
    family: str | None = None,
    ctx: SemanticContext | None = None,
) -> list[dict[str, Any]]:
    del access_clearance_ratio  # tool ignores it; keep signature stable

    out: list[dict[str, Any]] = []

    suggested = tool_output.get("suggested_moves")
    if isinstance(suggested, list):
        for move in suggested:
            if isinstance(move, dict):
                out.append(move)

    out.extend(
        _generate_overlap_escape_moves(
            tool_output=tool_output,
            placements=placements,
            grid_mm=grid_mm,
        )
    )
    out.extend(
        _generate_access_block_escape_moves(
            tool_output=tool_output,
            placements=placements,
            grid_mm=grid_mm,
        )
    )
    out.extend(
        _generate_rotation_escape_moves(
            cluster=cluster,
            placements=placements,
        )
    )
    out.extend(
        _generate_compaction_moves(
            cluster=cluster,
            placements=placements,
            grid_mm=grid_mm,
            family=family,
            ctx=ctx,
        )
    )

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for move in out:
        key = _move_key(move)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(move)
    return deduped


def _generate_overlap_escape_moves(
    *,
    tool_output: dict[str, Any],
    placements: list[dict[str, Any]],
    grid_mm: int,
) -> list[dict[str, Any]]:
    debug = tool_output.get("debug") or {}
    rects = debug.get("rects") or {}
    pair_overlaps = debug.get("pair_overlaps") or []

    placement_by_id = {p["id"]: p for p in _canonicalize_local_placements(placements)}
    moves: list[dict[str, Any]] = []

    for ov in pair_overlaps:
        if not isinstance(ov, dict):
            continue
        a = ov.get("a")
        b = ov.get("b")
        if not isinstance(a, str) or not isinstance(b, str):
            continue
        if a not in rects or b not in rects:
            continue
        if a not in placement_by_id or b not in placement_by_id:
            continue

        ra = rects[a]
        rb = rects[b]
        if not isinstance(ra, dict) or not isinstance(rb, dict):
            continue

        ax1 = int(ra.get("x1", 0))
        ay1 = int(ra.get("y1", 0))
        ax2 = int(ra.get("x2", 0))
        ay2 = int(ra.get("y2", 0))

        bx1 = int(rb.get("x1", 0))
        by1 = int(rb.get("y1", 0))
        bx2 = int(rb.get("x2", 0))
        by2 = int(rb.get("y2", 0))

        margin = max(grid_mm, 0)

        candidate_specs = [
            (a, bx2 - ax1 + margin, 0, "HOST_OVERLAP_ESCAPE_RIGHT"),
            (a, -(ax2 - bx1 + margin), 0, "HOST_OVERLAP_ESCAPE_LEFT"),
            (a, 0, by2 - ay1 + margin, "HOST_OVERLAP_ESCAPE_UP"),
            (a, 0, -(ay2 - by1 + margin), "HOST_OVERLAP_ESCAPE_DOWN"),
            (b, ax2 - bx1 + margin, 0, "HOST_OVERLAP_ESCAPE_RIGHT"),
            (b, -(bx2 - ax1 + margin), 0, "HOST_OVERLAP_ESCAPE_LEFT"),
            (b, 0, ay2 - by1 + margin, "HOST_OVERLAP_ESCAPE_UP"),
            (b, 0, -(by2 - ay1 + margin), "HOST_OVERLAP_ESCAPE_DOWN"),
        ]

        for move_object, dx, dy, reason in candidate_specs:
            dx = _host_snap_delta(dx, grid_mm)
            dy = _host_snap_delta(dy, grid_mm)
            if dx == 0 and dy == 0:
                continue

            base = placement_by_id[move_object]
            moves.append(
                {
                    "reason": reason,
                    "a": a,
                    "b": b,
                    "move_object": move_object,
                    "dx": int(dx),
                    "dy": int(dy),
                    "new_x": int(base["x"] + dx),
                    "new_y": int(base["y"] + dy),
                    "note": "Host-generated overlap escape move.",
                }
            )

    return moves


def _generate_access_block_escape_moves(
    *,
    tool_output: dict[str, Any],
    placements: list[dict[str, Any]],
    grid_mm: int,
) -> list[dict[str, Any]]:
    debug = tool_output.get("debug") or {}
    rects = debug.get("rects") or {}
    clearance_rects = debug.get("front_clearance_rects") or {}
    blocks = debug.get("front_clearance_blocks") or []

    placement_by_id = {p["id"]: p for p in _canonicalize_local_placements(placements)}
    moves: list[dict[str, Any]] = []

    for block in blocks:
        if not isinstance(block, dict):
            continue

        owner = block.get("owner")
        blocker = block.get("blocker")
        if not isinstance(owner, str) or not isinstance(blocker, str):
            continue
        if owner not in clearance_rects or blocker not in rects:
            continue
        if blocker not in placement_by_id:
            continue

        cr = clearance_rects[owner]
        br = rects[blocker]
        if not isinstance(cr, dict) or not isinstance(br, dict):
            continue

        cx1 = int(cr.get("x1", 0))
        cy1 = int(cr.get("y1", 0))
        cx2 = int(cr.get("x2", 0))
        cy2 = int(cr.get("y2", 0))

        bx1 = int(br.get("x1", 0))
        by1 = int(br.get("y1", 0))
        bx2 = int(br.get("x2", 0))
        by2 = int(br.get("y2", 0))

        margin = max(grid_mm, 0)

        candidate_specs = [
            (cx2 - bx1 + margin, 0, "HOST_ACCESS_ESCAPE_RIGHT"),
            (-(bx2 - cx1 + margin), 0, "HOST_ACCESS_ESCAPE_LEFT"),
            (0, cy2 - by1 + margin, "HOST_ACCESS_ESCAPE_UP"),
            (0, -(by2 - cy1 + margin), "HOST_ACCESS_ESCAPE_DOWN"),
        ]

        base = placement_by_id[blocker]
        for dx, dy, reason in candidate_specs:
            dx = _host_snap_delta(dx, grid_mm)
            dy = _host_snap_delta(dy, grid_mm)
            if dx == 0 and dy == 0:
                continue

            moves.append(
                {
                    "reason": reason,
                    "owner": owner,
                    "blocker": blocker,
                    "move_object": blocker,
                    "dx": int(dx),
                    "dy": int(dy),
                    "new_x": int(base["x"] + dx),
                    "new_y": int(base["y"] + dy),
                    "note": "Host-generated access-clearance escape move.",
                }
            )

    return moves


def _generate_rotation_escape_moves(
    *,
    cluster: dict[str, Any],
    placements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    decisions = cluster.get("decisions", [])
    rules = cluster.get("cluster_rules", {}) if isinstance(cluster, dict) else {}
    allowed_raw = rules.get("allowed_rotations") if isinstance(rules, dict) else None
    allowed = allowed_raw if isinstance(allowed_raw, dict) else {}

    placement_by_id = {p["id"]: p for p in _canonicalize_local_placements(placements)}
    moves: list[dict[str, Any]] = []

    for d in decisions:
        if not isinstance(d, dict):
            continue
        obj_id = d.get("object_type") or d.get("category")
        if not isinstance(obj_id, str) or obj_id not in placement_by_id:
            continue

        current_rot = int(placement_by_id[obj_id]["rot"]) % 360
        allowed_rots = allowed.get(obj_id, [0, 90, 180, 270])
        if not isinstance(allowed_rots, list):
            continue

        for rot in allowed_rots:
            if not isinstance(rot, int):
                continue
            rot = rot % 360
            if rot == current_rot:
                continue

            moves.append(
                {
                    "reason": "HOST_ROTATION_ESCAPE",
                    "move_object": obj_id,
                    "new_rot": rot,
                    "note": "Host-generated rotation candidate.",
                }
            )

    return moves


def _generate_compaction_moves(
    *,
    cluster: dict[str, Any],
    placements: list[dict[str, Any]],
    grid_mm: int,
    family: str | None = None,
    ctx: SemanticContext | None = None,
) -> list[dict[str, Any]]:
    canonical = _canonicalize_local_placements(placements)
    if not canonical:
        return []

    spec_by_id = {
        s["id"]: s
        for s in _build_object_specs(cluster)
        if isinstance(s, dict) and isinstance(s.get("id"), str)
    }
    rects: list[dict[str, Any]] = []
    for placement in canonical:
        spec = spec_by_id.get(placement["id"])
        if spec is None:
            continue
        w = int(spec.get("w", 0) or 0)
        h = int(spec.get("h", 0) or 0)
        rot = int(placement.get("rot", 0) or 0) % 360
        if rot in (90, 270):
            w, h = h, w
        rects.append(
            {
                "id": placement["id"],
                "x": int(placement["x"]),
                "y": int(placement["y"]),
                "w": w,
                "h": h,
                "rot": rot,
            }
        )

    bbox = _local_bbox_from_rects(rects)
    rect_by_id = {rect["id"]: rect for rect in rects if isinstance(rect.get("id"), str)}
    moves: list[dict[str, Any]] = []

    for placement in canonical:
        object_id = placement["id"]
        rect = rect_by_id.get(object_id)
        if rect is None:
            continue

        if not _compaction_allowed_for_family(
            family=family,
            object_id=object_id,
            rect=rect,
            rects=rects,
            bbox=bbox,
            ctx=ctx,
        ):
            continue

        forbidden_x, forbidden_y = _compaction_forbidden_ranges(
            family=family,
            bbox=bbox,
        )

        left_target = _compaction_target_left(
            rect=rect,
            rects=rects,
            bbox=bbox,
            forbidden_x_ranges=forbidden_x,
        )
        if left_target < int(rect["x"]):
            dx = _host_snap_delta(left_target - int(rect["x"]), grid_mm)
            semantic_penalty = _compaction_semantic_penalty(family, dx, 0)
            if (
                dx != 0
                and semantic_penalty < 0.50
                and _compaction_move_preserves_semantics(
                    placements=canonical,
                    move_object=object_id,
                    dx=dx,
                    dy=0,
                    family=family,
                    ctx=ctx,
                )
            ):
                moves.append(
                    {
                        "reason": "COMPACT_LEFT_EDGE",
                        "move_object": object_id,
                        "dx": int(dx),
                        "dy": 0,
                        "new_x": int(placement["x"] + dx),
                        "new_y": int(placement["y"]),
                        "compaction_semantic_penalty": semantic_penalty,
                        "note": "Host-generated compaction move toward the left envelope.",
                    }
                )

        down_target = _compaction_target_down(
            rect=rect,
            rects=rects,
            bbox=bbox,
            forbidden_y_ranges=forbidden_y,
        )
        if down_target < int(rect["y"]):
            dy = _host_snap_delta(down_target - int(rect["y"]), grid_mm)
            semantic_penalty = _compaction_semantic_penalty(family, 0, dy)
            if (
                dy != 0
                and semantic_penalty < 0.50
                and _compaction_move_preserves_semantics(
                    placements=canonical,
                    move_object=object_id,
                    dx=0,
                    dy=dy,
                    family=family,
                    ctx=ctx,
                )
            ):
                moves.append(
                    {
                        "reason": "COMPACT_TOP_EDGE",
                        "move_object": object_id,
                        "dx": 0,
                        "dy": int(dy),
                        "new_x": int(placement["x"]),
                        "new_y": int(placement["y"] + dy),
                        "compaction_semantic_penalty": semantic_penalty,
                        "note": "Host-generated compaction move toward the top envelope.",
                    }
                )

    return moves


def _compaction_target_left(
    *,
    rect: dict[str, Any],
    rects: list[dict[str, Any]],
    bbox: dict[str, int],
    forbidden_x_ranges: list[tuple[int, int]] | None = None,
) -> int:
    best = int(bbox.get("min_x", 0))
    current_x = int(rect.get("x", 0))
    current_y = int(rect.get("y", 0))
    current_h = int(rect.get("h", 0))
    for other in rects:
        if other.get("id") == rect.get("id"):
            continue
        other_right = int(other.get("x", 0)) + int(other.get("w", 0))
        if other_right > current_x:
            continue
        if not _rects_overlap_on_axis(
            start_a=current_y,
            length_a=current_h,
            start_b=int(other.get("y", 0)),
            length_b=int(other.get("h", 0)),
        ):
            continue
        best = max(best, other_right)
    for start, end in forbidden_x_ranges or []:
        width = int(rect.get("w", 0))
        if best < end and best + width > start:
            best = max(best, end)
    return best


def _compaction_target_down(
    *,
    rect: dict[str, Any],
    rects: list[dict[str, Any]],
    bbox: dict[str, int],
    forbidden_y_ranges: list[tuple[int, int]] | None = None,
) -> int:
    best = int(bbox.get("min_y", 0))
    current_x = int(rect.get("x", 0))
    current_w = int(rect.get("w", 0))
    current_y = int(rect.get("y", 0))
    for other in rects:
        if other.get("id") == rect.get("id"):
            continue
        other_top = int(other.get("y", 0)) + int(other.get("h", 0))
        if other_top > current_y:
            continue
        if not _rects_overlap_on_axis(
            start_a=current_x,
            length_a=current_w,
            start_b=int(other.get("x", 0)),
            length_b=int(other.get("w", 0)),
        ):
            continue
        best = max(best, other_top)
    for start, end in forbidden_y_ranges or []:
        height = int(rect.get("h", 0))
        if best < end and best + height > start:
            best = max(best, end)
    return best


def _compaction_allowed_for_family(
    *,
    family: str | None,
    object_id: str,
    rect: dict[str, Any],
    rects: list[dict[str, Any]],
    bbox: dict[str, int],
    ctx: SemanticContext | None,
) -> bool:
    del rect
    del rects
    del bbox
    if ctx is not None and object_id == ctx.dominant_anchor_id:
        return False
    family_token = _normalize_token(family or "")
    if family_token in {"open_center", "conversation_facing", "media_facing"}:
        role = ctx.roles.get(object_id, "") if ctx is not None else ""
        return role in {"accessory_support", "side_support", "storage_support"}
    return True


def _compaction_semantic_penalty(
    family: str | None,
    dx: int,
    dy: int,
) -> float:
    family_token = _normalize_token(family or "")
    if family_token == "media_facing" and abs(dy) > abs(dx):
        return 0.60
    if family_token == "conversation_facing" and (dx != 0 or dy != 0):
        return 0.35
    if family_token == "open_center" and (dx != 0 or dy != 0):
        return 0.45
    return 0.0


def _compaction_move_preserves_semantics(
    *,
    placements: list[dict[str, Any]],
    move_object: str,
    dx: int,
    dy: int,
    family: str | None,
    ctx: SemanticContext | None,
) -> bool:
    if family is None or ctx is None:
        return True
    move = {
        "move_object": move_object,
        "dx": int(dx),
        "dy": int(dy),
    }
    before = _family_contract_validator(
        family=family,
        placements=placements,
        ctx=ctx,
        verifier_eval=None,
    )
    patched = _apply_single_move(placements, move)
    after = _family_contract_validator(
        family=family,
        placements=patched,
        ctx=ctx,
        verifier_eval=None,
    )
    threshold = _family_fidelity_threshold(ctx)
    if not after.passed or after.family_fidelity < threshold:
        return False
    if after.family_fidelity + 0.02 < before.family_fidelity:
        return False
    return True


def _compaction_forbidden_ranges(
    *,
    family: str | None,
    bbox: dict[str, int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    family_token = _normalize_token(family or "")
    if family_token != "open_center":
        return [], []
    width = max(1, int(bbox.get("max_x", 0)) - int(bbox.get("min_x", 0)))
    height = max(1, int(bbox.get("max_y", 0)) - int(bbox.get("min_y", 0)))
    x1 = int(bbox.get("min_x", 0)) + width // 3
    x2 = int(bbox.get("min_x", 0)) + (2 * width) // 3
    y1 = int(bbox.get("min_y", 0)) + height // 3
    y2 = int(bbox.get("min_y", 0)) + (2 * height) // 3
    return [(x1, x2)], [(y1, y2)]


def _rects_overlap_on_axis(
    *,
    start_a: int,
    length_a: int,
    start_b: int,
    length_b: int,
) -> bool:
    end_a = int(start_a) + int(length_a)
    end_b = int(start_b) + int(length_b)
    return max(int(start_a), int(start_b)) < min(end_a, end_b)


def _run_verifier_once(
    *,
    cluster: dict[str, Any],
    placements: list[dict[str, Any]],
    access_clearance_ratio: float,
) -> dict[str, Any]:
    access_clearance_ratio = FIXED_ACCESS_CLEARANCE_RATIO
    args = {
        "hard_constraints": cluster.get("hard_constraints", []),
        "soft_constraints": cluster.get("soft_constraints", []),
        "objects": _build_object_specs(cluster),
        "local_placements": _canonicalize_local_placements(placements),
        "grid_mm": _extract_grid_mm(cluster),
        "use_clearance": True,
        "cluster_rules": (
            cluster.get("cluster_rules")
            if isinstance(cluster.get("cluster_rules"), dict)
            else None
        ),
        "access_clearance_ratio": float(access_clearance_ratio),
    }
    return _safe_run_tool("LocalClusterVerifier", args)


def _extract_grid_mm(cluster: dict[str, Any]) -> int:
    rules = cluster.get("cluster_rules", {}) if isinstance(cluster, dict) else {}
    raw = (rules or {}).get("grid_mm") if isinstance(rules, dict) else None
    if raw is None:
        policy = (
            cluster.get("composer_policy")
            if isinstance(cluster.get("composer_policy"), dict)
            else {}
        )
        raw = policy.get("grid_mm") if isinstance(policy, dict) else None
    return normalize_layout_grid_mm(raw)


def _build_cluster_output_from_placements(
    *,
    cluster: dict[str, Any],
    placements: list[dict[str, Any]],
    verifier_eval: dict[str, Any] | None = None,
    notes: list[str] | None = None,
    variant_family: str | None = None,
    family_fidelity: float | None = None,
) -> dict[str, Any]:
    cluster_id = str(cluster.get("cluster_id") or "")
    grid_mm = _extract_grid_mm(cluster)
    canonical_variant_family = canonical_semantic_variant_family(variant_family)

    canonical_placements = _canonicalize_local_placements(placements)
    spec_by_id = {
        s["id"]: s
        for s in _build_object_specs(cluster)
        if isinstance(s, dict) and isinstance(s.get("id"), str)
    }

    rects: list[dict[str, Any]] = []
    for p in canonical_placements:
        pid = p.get("id")
        if not isinstance(pid, str) or not pid:
            continue
        spec = spec_by_id.get(pid)
        if spec is None:
            continue

        w = int(spec.get("w", 0) or 0)
        h = int(spec.get("h", 0) or 0)
        rot = int(p.get("rot", 0)) % 360
        if rot in (90, 270):
            w, h = h, w

        rects.append(
            {
                "id": pid,
                "x": int(p.get("x", 0)),
                "y": int(p.get("y", 0)),
                "w": int(w),
                "h": int(h),
            }
        )

    bbox = _local_bbox_from_rects(rects)
    tight_hull_polygons = _outline_polygons_union_grid(rects)
    tight_hull_polygon = tight_hull_polygons[0] if len(tight_hull_polygons) == 1 else []
    interaction_rects = (
        _interaction_rects_from_verifier(verifier_eval)
        if isinstance(verifier_eval, dict)
        else []
    )
    interaction_hull_polygons = _outline_polygons_union_grid(interaction_rects)
    interaction_hull_polygon = (
        interaction_hull_polygons[0] if len(interaction_hull_polygons) == 1 else []
    )

    orientation_meta = _infer_orientation_meta(
        cluster=cluster,
        placements=canonical_placements,
        rects=rects,
        spec_by_id=spec_by_id,
        verifier_eval=verifier_eval,
    )

    return {
        "status": "OK",
        "cluster_id": cluster_id,
        "local_frame": {
            "unit": "mm",
            "grid_mm": int(grid_mm),
            "origin_note": "(0,0) is an arbitrary local origin for this cluster",
        },
        "local_placements": canonical_placements,
        "cluster_footprint": {
            "type": "union_of_rects",
            "rects": rects,
            "local_bbox": bbox,
            "tight_hull_polygon_mm": tight_hull_polygon,
            "tight_hull_polygons_mm": tight_hull_polygons,
            "interaction_hull_polygon_mm": interaction_hull_polygon,
            "interaction_hull_polygons_mm": interaction_hull_polygons,
            "variant_family": variant_family,
            "family_fidelity": family_fidelity,
        },
        "tight_hull_polygon_mm": tight_hull_polygon,
        "tight_hull_polygons_mm": tight_hull_polygons,
        "interaction_hull_polygon_mm": interaction_hull_polygon,
        "interaction_hull_polygons_mm": interaction_hull_polygons,
        "variant_family": variant_family,
        "canonical_variant_family": canonical_variant_family,
        "family_fidelity": family_fidelity,
        "orientation_meta": orientation_meta,
        "notes": list(notes or []),
        "missing": [],
    }


def _local_bbox_from_rects(rects: list[dict[str, Any]]) -> dict[str, int]:
    if not rects:
        return {"min_x": 0, "min_y": 0, "max_x": 0, "max_y": 0}

    min_x = 10**18
    min_y = 10**18
    max_x = -(10**18)
    max_y = -(10**18)

    for r in rects:
        if not isinstance(r, dict):
            continue
        x = int(r.get("x", 0))
        y = int(r.get("y", 0))
        w = int(r.get("w", 0))
        h = int(r.get("h", 0))
        if w <= 0 or h <= 0:
            continue
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)

    if min_x == 10**18:
        return {"min_x": 0, "min_y": 0, "max_x": 0, "max_y": 0}

    return {
        "min_x": int(min_x),
        "min_y": int(min_y),
        "max_x": int(max_x),
        "max_y": int(max_y),
    }


# ---------------------------
# Orientation meta inference
# ---------------------------


def _infer_orientation_meta(
    *,
    cluster: dict[str, Any],
    placements: list[dict[str, Any]],
    rects: list[dict[str, Any]],
    spec_by_id: dict[str, dict[str, Any]],
    verifier_eval: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # 1) Prefer verifier-derived orientation evidence from actual VALID layout
    meta_from_verifier = _infer_orientation_meta_from_verifier_eval(
        cluster=cluster,
        verifier_eval=verifier_eval,
    )
    if meta_from_verifier is not None:
        return meta_from_verifier

    # 2) Fallback heuristic only if verifier did not provide orientation_inference
    rect_by_id = {
        r["id"]: r
        for r in rects
        if isinstance(r, dict) and isinstance(r.get("id"), str)
    }
    placement_by_id = {
        p["id"]: p
        for p in placements
        if isinstance(p, dict) and isinstance(p.get("id"), str)
    }

    cluster_bbox = _local_bbox_from_rects(rects)
    cluster_axis = _infer_cluster_axis_local(rects)

    important_ids = _select_orientation_important_object_ids(
        cluster=cluster,
        rect_by_id=rect_by_id,
    )

    front_votes: dict[tuple[int, int], int] = {}
    axis_votes: dict[tuple[int, int], int] = {}
    important_objects: dict[str, dict[str, dict[str, int]]] = {}

    access_required = _seed_collect_access_required(cluster)
    anchors = {
        a
        for a in (
            cluster.get("anchors", [])
            if isinstance(cluster.get("anchors"), list)
            else []
        )
        if isinstance(a, str) and a
    }

    for oid in important_ids:
        rect = rect_by_id.get(oid)
        placement = placement_by_id.get(oid)
        spec = spec_by_id.get(oid)

        if rect is None or placement is None or spec is None:
            continue

        front = _infer_object_front_local_fallback(
            oid=oid,
            rect=rect,
            placement=placement,
            spec=spec,
            cluster_bbox=cluster_bbox,
            access_required=access_required,
        )
        axis = _infer_object_axis_local_fallback(
            rect=rect,
            front=front,
        )

        if front is None and axis is None:
            continue
        if front is None and axis is not None:
            front = _default_front_from_axis(axis)
        if axis is None and front is not None:
            axis = _axis_from_front(front)

        if front is None or axis is None:
            continue

        important_objects[oid] = {
            "front_local": _vec_to_dict(front),
            "effective_front_side": _vec_to_side(front),
            "axis_local": _vec_to_dict(axis),
        }

        weight = _orientation_vote_weight(
            oid=oid,
            access_required=access_required,
            anchors=anchors,
        )
        front_votes[front] = front_votes.get(front, 0) + weight
        axis_votes[axis] = axis_votes.get(axis, 0) + weight

    if axis_votes:
        cluster_axis = _pick_best_axis_vote(axis_votes, fallback=cluster_axis)

    cluster_front = _infer_cluster_front_local_fallback(
        cluster_axis=cluster_axis,
        front_votes=front_votes,
    )

    return {
        "cluster_front_local": _vec_to_dict(cluster_front),
        "cluster_effective_front_side": _vec_to_side(cluster_front),
        "cluster_axis_local": _vec_to_dict(cluster_axis),
        "important_objects": important_objects,
    }


def _infer_orientation_meta_from_verifier_eval(
    *,
    cluster: dict[str, Any],
    verifier_eval: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(verifier_eval, dict):
        return None

    debug = verifier_eval.get("debug")
    if not isinstance(debug, dict):
        return None

    orientation = debug.get("orientation_inference")
    if not isinstance(orientation, dict):
        return None

    cluster_axis = _coerce_axis_vec(orientation.get("cluster_axis_local_candidate"))
    cluster_front = _coerce_axis_vec(orientation.get("cluster_front_local_candidate"))

    objects_raw = orientation.get("objects")
    objects_raw = objects_raw if isinstance(objects_raw, dict) else {}

    anchors = {
        a
        for a in (
            cluster.get("anchors", [])
            if isinstance(cluster.get("anchors"), list)
            else []
        )
        if isinstance(a, str) and a
    }
    access_required = _seed_collect_access_required(cluster)
    member_order = _member_ids(cluster)
    member_set = set(member_order)

    important_objects: dict[str, dict[str, dict[str, int]]] = {}
    front_votes: dict[tuple[int, int], int] = {}
    axis_votes: dict[tuple[int, int], int] = {}

    ordered_ids = [oid for oid in member_order if oid in objects_raw]
    for oid in sorted(objects_raw.keys()):
        if oid not in member_set and oid not in ordered_ids:
            ordered_ids.append(oid)

    for oid in ordered_ids:
        item = objects_raw.get(oid)
        if not isinstance(item, dict):
            continue

        front = _coerce_axis_vec(item.get("front_local"))
        axis = _coerce_axis_vec(item.get("axis_local"))

        if front is None and axis is None:
            continue
        if front is None and axis is not None:
            front = _default_front_from_axis(axis)
        if axis is None and front is not None:
            axis = _axis_from_front(front)

        if front is None or axis is None:
            continue

        include = bool(item.get("important_for_orientation"))
        include = (
            include
            or bool(item.get("has_front_access"))
            or bool(item.get("is_anchor_like"))
            or oid in anchors
            or oid in access_required
            or _looks_directional_object_id(oid)
        )

        if not include:
            continue

        important_objects[oid] = {
            "front_local": _vec_to_dict(front),
            "effective_front_side": _vec_to_side(front),
            "axis_local": _vec_to_dict(axis),
        }

        weight = _orientation_vote_weight(
            oid=oid,
            access_required=access_required,
            anchors=anchors,
        )
        front_votes[front] = front_votes.get(front, 0) + weight
        axis_votes[axis] = axis_votes.get(axis, 0) + weight

    if cluster_axis is None:
        if axis_votes:
            cluster_axis = _pick_best_axis_vote(axis_votes, fallback=(1, 0))
        else:
            cluster_axis = (1, 0)

    if cluster_front is None:
        if front_votes:
            cluster_front = _infer_cluster_front_local_fallback(
                cluster_axis=cluster_axis,
                front_votes=front_votes,
            )
        else:
            cluster_front = _default_front_from_axis(cluster_axis)

    return {
        "cluster_front_local": _vec_to_dict(cluster_front),
        "cluster_effective_front_side": _vec_to_side(cluster_front),
        "cluster_axis_local": _vec_to_dict(cluster_axis),
        "important_objects": important_objects,
    }


def _coerce_axis_vec(value: Any) -> tuple[int, int] | None:
    dx: int | None = None
    dy: int | None = None

    if isinstance(value, dict):
        try:
            dx = int(value.get("dx", 0))
            dy = int(value.get("dy", 0))
        except Exception:
            return None
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            dx = int(value[0])
            dy = int(value[1])
        except Exception:
            return None
    else:
        return None

    if dx is None or dy is None:
        return None
    if (dx, dy) not in {(1, 0), (-1, 0), (0, 1), (0, -1)}:
        return None
    return (dx, dy)


# ---------------------------
# Fallback orientation helpers
# ---------------------------


def _select_orientation_important_object_ids(
    *,
    cluster: dict[str, Any],
    rect_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    members = _member_ids(cluster)
    anchors = {
        a
        for a in (
            cluster.get("anchors", [])
            if isinstance(cluster.get("anchors"), list)
            else []
        )
        if isinstance(a, str) and a
    }
    access_required = _seed_collect_access_required(cluster)

    selected: list[str] = []
    for oid in members:
        if oid not in rect_by_id:
            continue
        if (
            oid in anchors
            or oid in access_required
            or _looks_directional_object_id(oid)
        ):
            selected.append(oid)

    out: list[str] = []
    seen: set[str] = set()
    for oid in selected:
        if oid in seen:
            continue
        seen.add(oid)
        out.append(oid)
    return out


def _looks_directional_object_id(oid: str) -> bool:
    s = oid.lower()
    keywords = (
        "sofa",
        "armchair",
        "chair",
        "bench",
        "bed",
        "desk",
        "table",
        "island",
        "fridge",
        "cabinet",
        "pantry",
        "wardrobe",
        "dresser",
        "bookshelf",
        "console",
        "tv",
        "monitor",
        "sink",
        "stove",
        "dishwasher",
        "washer",
        "dryer",
        "toilet",
        "vanity",
    )
    return any(k in s for k in keywords)


def _orientation_vote_weight(
    *,
    oid: str,
    access_required: set[str],
    anchors: set[str],
) -> int:
    if oid in access_required:
        return 4
    if oid in anchors:
        return 3
    if _looks_directional_object_id(oid):
        return 2
    return 1


def _infer_cluster_axis_local(rects: list[dict[str, Any]]) -> tuple[int, int]:
    if not rects:
        return (1, 0)

    bbox = _local_bbox_from_rects(rects)
    span_x = int(bbox.get("max_x", 0)) - int(bbox.get("min_x", 0))
    span_y = int(bbox.get("max_y", 0)) - int(bbox.get("min_y", 0))

    if span_x >= span_y:
        return (1, 0)
    return (0, 1)


def _pick_best_axis_vote(
    axis_votes: dict[tuple[int, int], int],
    *,
    fallback: tuple[int, int],
) -> tuple[int, int]:
    normalized_votes: dict[tuple[int, int], int] = {}
    for axis, weight in axis_votes.items():
        canonical = _canonical_axis(axis)
        normalized_votes[canonical] = normalized_votes.get(canonical, 0) + int(weight)

    best_axis = fallback
    best_score: tuple[int, int] | None = None

    for axis in ((1, 0), (0, 1)):
        score = (
            normalized_votes.get(axis, 0),
            1 if axis == fallback else 0,
        )
        if best_score is None or score > best_score:
            best_score = score
            best_axis = axis

    return best_axis


def _infer_cluster_front_local_fallback(
    *,
    cluster_axis: tuple[int, int],
    front_votes: dict[tuple[int, int], int],
) -> tuple[int, int]:
    if not front_votes:
        return _default_front_from_axis(cluster_axis)

    best_front: tuple[int, int] | None = None
    best_score: tuple[int, int, int] | None = None

    for front in ((0, 1), (1, 0), (-1, 0), (0, -1)):
        weight = front_votes.get(front, 0)
        perpendicular_bonus = 1 if _is_perpendicular(front, cluster_axis) else 0
        priority = _front_priority(front)
        score = (weight, perpendicular_bonus, -priority)
        if best_score is None or score > best_score:
            best_score = score
            best_front = front

    if best_front is None or front_votes.get(best_front, 0) <= 0:
        return _default_front_from_axis(cluster_axis)
    return best_front


def _infer_object_front_local_fallback(
    *,
    oid: str,
    rect: dict[str, Any],
    placement: dict[str, Any],
    spec: dict[str, Any],
    cluster_bbox: dict[str, int],
    access_required: set[str],
) -> tuple[int, int] | None:
    base_front = spec.get("front")
    rot = int(placement.get("rot", 0)) % 360

    if base_front in {"top", "bottom", "left", "right"}:
        final_side = _seed_rotate_side(str(base_front), rot)
        return _side_to_vec(final_side)

    if oid in access_required:
        return _outward_vec_from_bbox(rect, cluster_bbox)

    return None


def _infer_object_axis_local_fallback(
    *,
    rect: dict[str, Any],
    front: tuple[int, int] | None,
) -> tuple[int, int] | None:
    w = int(rect.get("w", 0) or 0)
    h = int(rect.get("h", 0) or 0)

    if w > h:
        return (1, 0)
    if h > w:
        return (0, 1)
    if front is not None:
        return _axis_from_front(front)
    return None


def _side_to_vec(side: str) -> tuple[int, int] | None:
    return _side_to_vec_contract(side)


def _vec_to_dict(vec: tuple[int, int]) -> dict[str, int]:
    return {"dx": int(vec[0]), "dy": int(vec[1])}


def _vec_to_side(vec: tuple[int, int]) -> str | None:
    return _vec_to_side_contract(vec)


def _canonical_axis(vec: tuple[int, int]) -> tuple[int, int]:
    dx = int(vec[0])
    if abs(dx) == 1:
        return (1, 0)
    return (0, 1)


def _axis_from_front(front: tuple[int, int]) -> tuple[int, int]:
    dx = int(front[0])
    if abs(dx) == 1:
        return (0, 1)
    return (1, 0)


def _default_front_from_axis(axis: tuple[int, int]) -> tuple[int, int]:
    if _canonical_axis(axis) == (1, 0):
        return (0, 1)
    return (1, 0)


def _is_perpendicular(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return (int(a[0]) * int(b[0]) + int(a[1]) * int(b[1])) == 0


def _front_priority(front: tuple[int, int]) -> int:
    order = {
        (0, 1): 0,
        (1, 0): 1,
        (-1, 0): 2,
        (0, -1): 3,
    }
    return order.get(front, 99)


def _outward_vec_from_bbox(
    rect: dict[str, Any],
    cluster_bbox: dict[str, int],
) -> tuple[int, int]:
    cx = float(rect.get("x", 0)) + float(rect.get("w", 0)) / 2.0
    cy = float(rect.get("y", 0)) + float(rect.get("h", 0)) / 2.0

    bx = (int(cluster_bbox.get("min_x", 0)) + int(cluster_bbox.get("max_x", 0))) / 2.0
    by = (int(cluster_bbox.get("min_y", 0)) + int(cluster_bbox.get("max_y", 0))) / 2.0

    dx = cx - bx
    dy = cy - by

    if abs(dx) >= abs(dy):
        return (1, 0) if dx >= 0 else (-1, 0)
    return (0, 1) if dy >= 0 else (0, -1)


def _is_context_length_exceeded(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    if code == "context_length_exceeded":
        return True
    text = str(exc)
    return ("context_length_exceeded" in text) or ("maximum context length" in text)


def _error_score(tool_output: dict[str, Any]) -> int:
    if not isinstance(tool_output, dict):
        return 10**9
    if tool_output.get("result") == "VALID":
        return 0

    errors = tool_output.get("errors")
    if not isinstance(errors, list):
        return 10**8

    weight = {
        "MISSING_OBJECT_SPECS": 100000,
        "INVALID_DIMS": 100000,
        "UNKNOWN_OBJECT": 50000,
        "GRID_VIOLATION": 2000,
        "ROTATION_NOT_ALLOWED": 1500,
        "CONTAIN_VIOLATION": 3500,
        "ANCHOR_VIOLATION": 3000,
        "DOCK_VIOLATION": 3000,
        "OVERLAP": 5000,
        "ACCESS_BLOCKED": 4500,
        "LOCAL_BBOX_BUDGET_EXCEEDED": 2500,
        "LOCAL_OUTLINE_BUDGET_EXCEEDED": 2500,
        "LOCAL_HULL_BUDGET_EXCEEDED": 2500,
        "LOCAL_FILL_RATIO_TOO_LOW": 2500,
    }

    score = 0
    for err in errors:
        if not isinstance(err, dict):
            score += 10000
            continue
        code = err.get("code")
        score += weight.get(code, 4000)
    return score


def _extract_layout_key(tool_output: dict[str, Any]) -> tuple[float, ...]:
    debug = tool_output.get("debug", {}) if isinstance(tool_output, dict) else {}
    constraint_scores = (
        debug.get("constraint_scores", {}) if isinstance(debug, dict) else {}
    )
    lexi = (
        constraint_scores.get("lexicographic_key", [])
        if isinstance(constraint_scores, dict)
        else []
    )

    quality = tool_output.get("quality", {}) if isinstance(tool_output, dict) else {}
    bbox = quality.get("bbox", {}) if isinstance(quality, dict) else {}

    compact_score = float(quality.get("compact_score", 10**12) or 0.0)
    bbox_area = float(bbox.get("area_mm2", 10**12) or 0.0)
    max_span = float(bbox.get("max_span_mm", 10**12) or 0.0)
    fill_ratio_bbox = float(quality.get("fill_ratio_bbox", 0.0) or 0.0)
    fill_ratio_hull = float(quality.get("fill_ratio_hull", 0.0) or 0.0)
    aspect_ratio = float(bbox.get("aspect_ratio", 10**6) or 0.0)

    def _at(i: int, default: float) -> float:
        if isinstance(lexi, list) and i < len(lexi):
            try:
                return float(lexi[i])
            except Exception:
                return default
        return default

    return (
        _at(0, 0.0),
        _at(1, 0.0),
        _at(2, 0.0),
        _at(3, compact_score),
        _at(4, 0.0),
        compact_score,
        bbox_area,
        max_span,
        -fill_ratio_bbox,
        -fill_ratio_hull,
        aspect_ratio,
    )


def _tool_rank(tool_output: dict[str, Any]) -> tuple[Any, ...]:
    if not isinstance(tool_output, dict):
        return (
            9,
            10**9,
            10**9,
            10**9,
            10**9,
            10**9,
            10**9,
            10**9,
            10**9,
            10**9,
            10**9,
            10**9,
        )

    if tool_output.get("result") == "VALID":
        return (0, *_extract_layout_key(tool_output))

    quality = tool_output.get("quality", {}) if isinstance(tool_output, dict) else {}
    bbox = quality.get("bbox", {}) if isinstance(quality, dict) else {}

    compact_score = float(quality.get("compact_score", 10**12) or 0.0)
    bbox_area = float(bbox.get("area_mm2", 10**12) or 0.0)
    max_span = float(bbox.get("max_span_mm", 10**12) or 0.0)

    return (
        1,
        float(_error_score(tool_output)),
        compact_score,
        bbox_area,
        max_span,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )


def _rank_better(
    candidate_rank: tuple[Any, ...], baseline_rank: tuple[Any, ...]
) -> bool:
    return candidate_rank < baseline_rank


def _reason_rank(reason: Any) -> int:
    rank = {
        "GRID_VIOLATION": 0,
        "ROTATION_NOT_ALLOWED": 1,
        "HOST_OVERLAP_ESCAPE_LEFT": 2,
        "HOST_OVERLAP_ESCAPE_RIGHT": 2,
        "HOST_OVERLAP_ESCAPE_UP": 3,
        "HOST_OVERLAP_ESCAPE_DOWN": 3,
        "HOST_ACCESS_ESCAPE_LEFT": 4,
        "HOST_ACCESS_ESCAPE_RIGHT": 4,
        "HOST_ACCESS_ESCAPE_UP": 5,
        "HOST_ACCESS_ESCAPE_DOWN": 5,
        "OVERLAP": 6,
        "ACCESS_BLOCKED": 7,
        "DOCK_VIOLATION": 8,
        "ANCHOR_VIOLATION": 9,
        "DOCK_REFINEMENT": 10,
        "ANCHOR_REFINEMENT": 11,
        "SOFT_ALIGN_EDGE": 12,
        "SOFT_PREFER_NEAR": 13,
        "CONTAIN_VIOLATION": 14,
        "HOST_ROTATION_ESCAPE": 15,
        "COMPACT_LEFT_EDGE": 16,
        "COMPACT_RIGHT_EDGE": 16,
        "COMPACT_BOTTOM_EDGE": 17,
        "COMPACT_TOP_EDGE": 17,
    }
    return rank.get(reason, 99)


def _movement_cost(move: dict[str, Any]) -> int:
    dx = int(move.get("dx", 0) or 0)
    dy = int(move.get("dy", 0) or 0)
    rot_penalty = 500 if "new_rot" in move and move.get("new_rot") is not None else 0
    return abs(dx) + abs(dy) + rot_penalty


def _verifier_preview(tool_output: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(tool_output, dict):
        return {"result": "INVALID", "error_count": 999999}

    errors = tool_output.get("errors")
    preview = errors[:6] if isinstance(errors, list) else []

    quality = tool_output.get("quality", {}) if isinstance(tool_output, dict) else {}
    bbox = quality.get("bbox", {}) if isinstance(quality, dict) else {}

    return {
        "result": tool_output.get("result"),
        "error_count": len(errors) if isinstance(errors, list) else None,
        "errors_preview": preview,
        "error_score": _error_score(tool_output),
        "tool_rank": list(_tool_rank(tool_output)),
        "compact_score": quality.get("compact_score"),
        "bbox_area_mm2": bbox.get("area_mm2"),
        "max_span_mm": bbox.get("max_span_mm"),
    }


def _apply_single_move(
    placements: list[dict[str, Any]],
    move: dict[str, Any],
) -> list[dict[str, Any]]:
    move_object = move.get("move_object")
    if not isinstance(move_object, str):
        return _canonicalize_local_placements(placements)

    patched = _canonicalize_local_placements(placements)
    out: list[dict[str, Any]] = []

    for p in patched:
        if p["id"] != move_object:
            out.append(dict(p))
            continue

        q = dict(p)

        if "new_x" in move and move.get("new_x") is not None:
            q["x"] = int(move["new_x"])
        else:
            q["x"] = int(q["x"]) + int(move.get("dx", 0) or 0)

        if "new_y" in move and move.get("new_y") is not None:
            q["y"] = int(move["new_y"])
        else:
            q["y"] = int(q["y"]) + int(move.get("dy", 0) or 0)

        if "new_rot" in move and move.get("new_rot") is not None:
            q["rot"] = int(move["new_rot"]) % 360
        else:
            q["rot"] = int(q.get("rot", 0)) % 360

        out.append(q)

    out.sort(key=lambda item: item["id"])
    return out


def _host_snap_delta(delta: int, grid_mm: int) -> int:
    if delta == 0 or grid_mm <= 0:
        return int(delta)
    sign = 1 if delta > 0 else -1
    mag = abs(int(delta))
    if mag % grid_mm == 0:
        return sign * mag
    mag = ((mag // grid_mm) + 1) * grid_mm
    return sign * mag


# ---------------------------
# Parsing helpers
# ---------------------------


def _extract_message(response: object) -> object:
    choices = getattr(response, "choices", None)
    if isinstance(choices, list) and choices:
        return getattr(choices[0], "message", None)
    raise ValueError("OpenAI response missing message")


def _extract_tool_calls(message: object) -> list[dict[str, Any]]:
    return parse_tool_calls(message)


def _try_parse_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        parsed = json.loads(_extract_json_object(_coerce_json_text(text)))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_json(raw: str) -> dict[str, Any]:
    text = _coerce_json_text(raw)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        text = _extract_json_object(text)
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("ClusterComposer response must be a JSON object")
    return payload


def _sanitize_cluster_output_payload(
    payload: dict[str, Any], cluster: dict[str, Any]
) -> dict[str, Any]:
    allowed_keys = {
        "status",
        "cluster_id",
        "local_frame",
        "local_placements",
        "cluster_footprint",
        "orientation_meta",
        "variant_bundle",
        "canonical_variant_id",
        "canonical_variant_family",
        "variant_summary",
        "family_coverage",
        "tight_hull_polygon_mm",
        "tight_hull_polygons_mm",
        "interaction_hull_polygon_mm",
        "interaction_hull_polygons_mm",
        "variant_family",
        "source_type",
        "family_fidelity",
        "semantic_confidence",
        "fallback_heavy",
        "solver_friendliness",
        "family_contract_reasons",
        "conflicts",
        "notes",
        "missing",
    }

    notes: list[str] = []
    if isinstance(payload.get("notes"), list):
        notes = [str(x).strip() for x in payload.get("notes") if str(x).strip()]

    reason = payload.get("reason")
    if isinstance(reason, str) and reason.strip():
        notes.append(reason.strip())

    cleaned: dict[str, Any] = {k: payload[k] for k in allowed_keys if k in payload}
    if notes:
        cleaned["notes"] = notes
    elif "notes" in cleaned and not isinstance(cleaned.get("notes"), list):
        cleaned["notes"] = []

    cleaned = _ensure_required_fields_for_ok(cleaned, cluster)
    if not _has_required_orientation_meta(cleaned.get("orientation_meta")):
        if cleaned.get("status") == "OK":
            cleaned = _downgrade_ok_payload(cleaned, ["orientation_meta"])
        else:
            cleaned.pop("orientation_meta", None)
    return cleaned


def _ensure_required_fields_for_ok(
    payload: dict[str, Any], cluster: dict[str, Any]
) -> dict[str, Any]:
    if payload.get("status") != "OK":
        return payload

    missing_fields: list[str] = []

    cluster_id = payload.get("cluster_id")
    if not isinstance(cluster_id, str) or not cluster_id.strip():
        inferred_id = cluster.get("cluster_id")
        if isinstance(inferred_id, str) and inferred_id.strip():
            payload["cluster_id"] = inferred_id.strip()

    placements_raw = payload.get("local_placements")
    if not isinstance(placements_raw, list):
        missing_fields.append("local_placements")
        return _downgrade_ok_payload(payload, missing_fields)

    placements = _canonicalize_local_placements(placements_raw)
    if not placements:
        missing_fields.append("local_placements")
        return _downgrade_ok_payload(payload, missing_fields)

    if not isinstance(payload.get("local_frame"), dict):
        payload["local_frame"] = {
            "unit": "mm",
            "grid_mm": int(_extract_grid_mm(cluster)),
            "origin_note": "(0,0) is an arbitrary local origin for this cluster",
        }

    spec_by_id = {
        s["id"]: s
        for s in _build_object_specs(cluster)
        if isinstance(s, dict) and isinstance(s.get("id"), str)
    }

    rects: list[dict[str, Any]] = []
    for p in placements:
        pid = p.get("id")
        if not isinstance(pid, str) or not pid:
            continue
        spec = spec_by_id.get(pid)
        if spec is None:
            continue

        w = int(spec.get("w", 0) or 0)
        h = int(spec.get("h", 0) or 0)
        rot = int(p.get("rot", 0)) % 360
        if rot in (90, 270):
            w, h = h, w

        rects.append(
            {
                "id": pid,
                "x": int(p.get("x", 0)),
                "y": int(p.get("y", 0)),
                "w": int(w),
                "h": int(h),
            }
        )

    placement_ids = {p["id"] for p in placements}
    rect_ids = {r["id"] for r in rects}
    if rect_ids != placement_ids:
        missing_fields.append("cluster_footprint.rects")
    elif not isinstance(payload.get("cluster_footprint"), dict):
        payload["cluster_footprint"] = {
            "type": "union_of_rects",
            "rects": rects,
            "local_bbox": _local_bbox_from_rects(rects),
        }

    if not _has_required_orientation_meta(payload.get("orientation_meta")):
        payload["orientation_meta"] = _infer_orientation_meta(
            cluster=cluster,
            placements=placements,
            rects=rects,
            spec_by_id=spec_by_id,
            verifier_eval=None,
        )

    if not _has_required_orientation_meta(payload.get("orientation_meta")):
        missing_fields.append("orientation_meta")

    if missing_fields:
        return _downgrade_ok_payload(payload, missing_fields)

    return payload


def _has_required_orientation_meta(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    front = _coerce_axis_vec(value.get("cluster_front_local"))
    axis = _coerce_axis_vec(value.get("cluster_axis_local"))
    if front is None or axis is None:
        return False
    return True


def _downgrade_ok_payload(
    payload: dict[str, Any], missing_fields: list[str]
) -> dict[str, Any]:
    payload["status"] = "NEED_INFO"
    existing = payload.get("missing")
    missing: list[str] = []
    if isinstance(existing, list):
        missing = [str(x).strip() for x in existing if str(x).strip()]
    for item in missing_fields:
        if item and item not in missing:
            missing.append(item)
    payload["missing"] = missing

    if not _has_required_orientation_meta(payload.get("orientation_meta")):
        payload.pop("orientation_meta", None)

    return payload


def _coerce_json_text(raw: str) -> str:
    text = raw.strip()
    if not text.startswith("```"):
        return text
    parts = text.split("```")
    for idx in range(1, len(parts), 2):
        candidate = parts[idx].strip()
        if candidate.startswith("json"):
            candidate = candidate[4:].lstrip("\n").strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            return candidate
    return text


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return text
    return text[start : end + 1]


def _member_ids(cluster: dict[str, Any]) -> list[str]:
    members = cluster.get("members") if isinstance(cluster, dict) else None
    if isinstance(members, list):
        return [m for m in members if isinstance(m, str)]

    inventory = (
        cluster.get("inventory_decision")
        if isinstance(cluster.get("inventory_decision"), dict)
        else None
    )
    objects = inventory.get("objects") if isinstance(inventory, dict) else None
    if not isinstance(objects, list):
        return []
    out: list[str] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        object_id = obj.get("object_id") or obj.get("object_type")
        if isinstance(object_id, str) and object_id:
            out.append(object_id)
    return out
