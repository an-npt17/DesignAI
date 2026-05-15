from __future__ import annotations

import json
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from agent.cluster_placer import (
    MacroClusterPlacer,
    build_phase2_payload,
    compile_phase2_to_final_output,
    make_no_improvement_repair,
)
from agent.micro_refiner import MicroRefiner
from agent.phase2_judge import Phase2Judge
from cluster_placer.tools_v2 import (
    EnumeratePhase2RepairMoves,
    EvaluatePhase2Proposal,
    PromotePhase2RepairToSeedPayload,
)
from layout.grid_policy import normalize_layout_grid_mm

logger = logging.getLogger(__name__)


def _normalize_solver_output_for_phase2(
    solver_output: dict[str, Any] | None,
) -> dict[str, Any]:
    normalized = deepcopy(solver_output or {})
    placer_seed = normalized.get("placer_seed") or {}
    seed_layout = placer_seed.get("seed_layout") or {}

    if isinstance(seed_layout, dict):
        if not normalized.get("cluster_transforms") and seed_layout.get(
            "cluster_transforms"
        ):
            normalized["cluster_transforms"] = deepcopy(
                seed_layout.get("cluster_transforms") or []
            )
        if not normalized.get("selected_variants") and seed_layout.get(
            "selected_variants"
        ):
            normalized["selected_variants"] = deepcopy(
                seed_layout.get("selected_variants") or []
            )

    return normalized


def _phase2_controller_result(
    *,
    proposal: dict[str, Any],
    evaluation: dict[str, Any],
    absolute_layout: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
    hard_fix: dict[str, Any] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    hard_valid = bool(evaluation.get("hard_valid"))
    refined_layout_solution = {
        "status": "OK"
        if str(absolute_layout.get("status") or "").upper() == "OK"
        else "PARTIAL_OK",
        "hard_valid": hard_valid,
        "quality_before": {},
        "quality_after": {},
        "applied_moves": [],
        "notes": list(notes or []),
    }
    return {
        "proposal": proposal,
        "tool_evaluation": evaluation,
        "judge_evaluation": {
            "verdict": "ACCEPT" if hard_valid else "REVISE",
            "reasonableness_score": int(
                max(
                    0,
                    (
                        (evaluation.get("baseline_comparison") or {}).get(
                            "candidate_score"
                        )
                        or 0
                    ),
                )
                // 100
            ),
            "next_step_mode": "stop" if hard_valid else "macro_layout",
            "top_issues": [],
            "repair_advice": list(notes or []),
            "priority_clusters": [],
        },
        "absolute_layout": absolute_layout,
        "refined_layout_solution": refined_layout_solution,
        "history": history or [],
        "hard_fix": hard_fix or {"result": "SKIPPED", "attempts": []},
    }


_MEANINGFUL_DELTA_BY_PHASE = {
    "macro_layout": 180,
    "object_refine": 120,
}
_MACRO_ISSUE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "orientation": ("orientation", "facing", "face", "align"),
    "focal": ("focal", "tv", "view", "viewing"),
    "circulation": ("circulation", "walk", "path", "entry", "door", "lane"),
    "zoning": ("zone", "zoning", "center", "central", "congestion"),
    "openings": ("window", "opening", "wall", "edge", "perimeter"),
}
_OBJECT_ISSUE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "front_access": ("front access", "front clearance", "clearance"),
    "object_pose": ("object", "chair", "nightstand", "lamp", "desk", "sofa"),
    "local_fidelity": ("internal", "local", "variant", "fidelity"),
}
_PLANNER_OBJECT_INTENTS = {
    "face_object",
    "face_away_from_object",
    "front_to_open_space",
    "preserve_front_access",
    "back_to_wall",
    "face_entry",
    "front_to_entry",
    "face_window",
    "front_to_window",
    "in_front_of_anchor",
    "align_with_anchor_axis",
    "same_view_side_as_primary_pair",
    "flank_anchor",
    "not_behind_anchor_view",
    "beside_secondary_seat",
    "same_direction_as_anchor",
}


def _judge_rank(verdict: str) -> int:
    if verdict == "ACCEPT":
        return 2
    if verdict == "REVISE":
        return 1
    return 0


def _evaluation_rank(
    evaluation: dict[str, Any], judge_output: dict[str, Any]
) -> tuple[int, int, int, int, int]:
    hard_valid = 1 if bool(evaluation.get("hard_valid")) else 0
    verdict_rank = _judge_rank(str(judge_output.get("verdict") or "REJECT"))
    tool_score = int(
        ((evaluation.get("baseline_comparison") or {}).get("candidate_score")) or 0
    )
    judge_score = int(judge_output.get("reasonableness_score") or 0)
    issue_penalty = -len(judge_output.get("top_issues") or [])
    return (hard_valid, verdict_rank, tool_score, judge_score, issue_penalty)


def _baseline_judge_stub(evaluation: dict[str, Any]) -> dict[str, Any]:
    if bool(evaluation.get("hard_valid")):
        return {
            "reasonableness_score": 50,
            "verdict": "REVISE",
            "next_step_mode": "macro_layout",
            "top_issues": [],
            "repair_advice": [],
            "priority_clusters": [],
        }
    return {
        "reasonableness_score": 0,
        "verdict": "REJECT",
        "next_step_mode": "macro_layout",
        "top_issues": ["Baseline seed is hard-invalid."],
        "repair_advice": ["Repair hard validity before soft reasoning."],
        "priority_clusters": [],
    }


def _candidate_score(evaluation: dict[str, Any]) -> int:
    return int(
        ((evaluation.get("baseline_comparison") or {}).get("candidate_score")) or 0
    )


def _delta_score(evaluation: dict[str, Any]) -> int:
    return int(((evaluation.get("baseline_comparison") or {}).get("delta_score")) or 0)


def _search_candidate_rank(
    evaluation: dict[str, Any],
) -> tuple[int, int, int, int, int]:
    delta = _delta_score(evaluation)
    return (
        1 if bool(evaluation.get("hard_valid")) else 0,
        1 if delta > 0 else 0,
        delta,
        _candidate_score(evaluation),
        -len(evaluation.get("errors") or []),
    )


def _primary_cluster_from_direction(direction_summary: dict[str, Any]) -> str:
    clusters = direction_summary.get("primary_clusters") or []
    if clusters and isinstance(clusters[0], str):
        return clusters[0]
    return ""


def _proposal_has_object_repairs(proposal: dict[str, Any]) -> bool:
    return bool((proposal.get("object_repairs") or []))


def _normalize_search_phase(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text == "object_refine":
        return "object_refine"
    return "macro_layout"


def _current_search_phase(payload: dict[str, Any]) -> str:
    phase_control = payload.get("phase_control") or {}
    return _normalize_search_phase(phase_control.get("repair_phase"))


def _set_search_phase(payload: dict[str, Any], search_phase: str) -> None:
    phase_control = deepcopy(payload.get("phase_control") or {})
    phase_control["repair_phase"] = _normalize_search_phase(search_phase)
    payload["phase_control"] = phase_control


def _move_matches_search_phase(move: dict[str, Any], search_phase: str) -> bool:
    kind = str(move.get("kind") or "")
    normalized_phase = _normalize_search_phase(search_phase)
    if normalized_phase == "object_refine":
        return kind == "object_pose"
    return kind in {"cluster_variant", "cluster_pose"}


def _should_replace_with_deterministic_candidate(
    shortlist_evaluation: dict[str, Any],
    placer_evaluation: dict[str, Any],
) -> bool:
    if not bool(shortlist_evaluation.get("hard_valid")):
        return False
    if _delta_score(shortlist_evaluation) < 0:
        return False
    return _search_candidate_rank(shortlist_evaluation) > _search_candidate_rank(
        placer_evaluation
    )


def _invalid_judge_stub(evaluation: dict[str, Any]) -> dict[str, Any]:
    errors = evaluation.get("errors") or []
    top_issues: list[str] = []
    for error in errors[:3]:
        if not isinstance(error, dict):
            continue
        code = str(error.get("code") or "UNKNOWN")
        if code == "OBJECT_OVERLAP":
            top_issues.append("Candidate is hard-invalid because objects overlap.")
        elif code == "OBJECT_OUT_OF_BOUNDS":
            top_issues.append(
                "Candidate is hard-invalid because at least one object leaves the room."
            )
        elif code == "OBJECT_HITS_OBSTACLE":
            top_issues.append(
                "Candidate is hard-invalid because it intersects a hard obstacle."
            )
        else:
            top_issues.append(f"Candidate is hard-invalid ({code}).")
    if not top_issues:
        top_issues = ["Candidate is hard-invalid."]
    return {
        "reasonableness_score": 10,
        "verdict": "REJECT",
        "next_step_mode": "macro_layout",
        "top_issues": top_issues,
        "repair_advice": ["Repair hard validity before spending more search."],
        "priority_clusters": [],
    }


def _seed_transform_map(seed_layout: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in seed_layout.get("cluster_transforms") or []:
        if isinstance(item, dict) and isinstance(item.get("cluster_id"), str):
            out[item["cluster_id"]] = deepcopy(item)
    return out


def _seed_variant_map(seed_layout: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in seed_layout.get("selected_variants") or []:
        if isinstance(item, dict) and isinstance(item.get("cluster_id"), str):
            out[item["cluster_id"]] = deepcopy(item)
    return out


def _canonical_repair(
    payload: dict[str, Any],
    repair: dict[str, Any] | None = None,
    *,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    seed_layout = payload.get("seed_layout") or {}
    tmap = _seed_transform_map(seed_layout)
    vmap = _seed_variant_map(seed_layout)

    for row in (repair or {}).get("cluster_transforms") or []:
        if isinstance(row, dict) and isinstance(row.get("cluster_id"), str):
            tmap[row["cluster_id"]] = deepcopy(row)

    for row in (repair or {}).get("selected_variants") or []:
        if isinstance(row, dict) and isinstance(row.get("cluster_id"), str):
            vmap[row["cluster_id"]] = deepcopy(row)

    merged_notes: list[str] = []
    for source in (((repair or {}).get("notes") or []), notes or []):
        for item in source:
            text = str(item).strip()
            if text and text not in merged_notes:
                merged_notes.append(text)

    cluster_ids = sorted(tmap.keys())
    return {
        "status": "REPAIRED",
        "cluster_transforms": [deepcopy(tmap[cid]) for cid in cluster_ids],
        "selected_variants": [
            deepcopy(vmap[cid]) for cid in cluster_ids if cid in vmap
        ],
        "object_repairs": deepcopy((repair or {}).get("object_repairs") or []),
        "notes": merged_notes,
    }


def _proposal_signature(proposal: dict[str, Any]) -> str:
    return json.dumps(
        {
            "cluster_transforms": proposal.get("cluster_transforms") or [],
            "selected_variants": proposal.get("selected_variants") or [],
            "object_repairs": proposal.get("object_repairs") or [],
        },
        ensure_ascii=True,
        sort_keys=True,
    )


def _issue_families_from_texts(texts: list[object]) -> list[str]:
    lowered_text = " ".join(
        str(item).strip().lower() for item in texts if str(item).strip()
    )
    if not lowered_text:
        return []

    families: list[str] = []
    for family, keywords in _MACRO_ISSUE_KEYWORDS.items():
        if any(keyword in lowered_text for keyword in keywords):
            families.append(family)
    for family, keywords in _OBJECT_ISSUE_KEYWORDS.items():
        if any(keyword in lowered_text for keyword in keywords):
            families.append(family)
    return families


def _judge_issue_families(judge_eval: dict[str, Any]) -> list[str]:
    texts: list[object] = []
    texts.extend(judge_eval.get("top_issues") or [])
    texts.extend(judge_eval.get("repair_advice") or [])
    return _issue_families_from_texts(texts)


def _has_macro_orientation_pressure(evaluation: dict[str, Any]) -> bool:
    metrics = evaluation.get("metrics") or {}
    critical_rows = [
        row
        for row in (metrics.get("orientation_debug") or [])
        if isinstance(row, dict) and int(row.get("penalty_mm") or 0) >= 300
    ]
    if not critical_rows:
        return False

    if any(
        str(row.get("kind") or "")
        in {"cluster_directional_relation", "cluster_orientation"}
        for row in critical_rows
    ):
        return True

    macro_object_intents = {
        "face_object",
        "face_away_from_object",
        "front_to_open_space",
        "preserve_front_access",
        "back_to_wall",
    }
    affected_clusters = {
        str(row.get("cluster_id") or "")
        for row in critical_rows
        if str(row.get("kind") or "") == "object_orientation"
        and str(row.get("intent") or "") in macro_object_intents
        and isinstance(row.get("cluster_id"), str)
    }
    return len({cluster_id for cluster_id in affected_clusters if cluster_id}) >= 2


def _planner_object_pressure_summary(evaluation: dict[str, Any]) -> dict[str, Any]:
    metrics = evaluation.get("metrics") or {}
    diagnosis = evaluation.get("diagnosis") or {}

    planner_object_penalty = 0
    macro_penalty = 0
    planner_clusters: set[str] = set()
    planner_objects: set[tuple[str, str]] = set()

    for row in metrics.get("orientation_debug") or []:
        if not isinstance(row, dict):
            continue
        penalty = int(row.get("penalty_mm") or 0)
        if penalty <= 0:
            continue
        kind = str(row.get("kind") or "").strip().lower()
        if (
            kind == "object_orientation"
            and str(row.get("intent") or "").strip().lower() in _PLANNER_OBJECT_INTENTS
        ):
            cluster_id = str(row.get("cluster_id") or "").strip()
            object_id = str(row.get("object_id") or "").strip()
            planner_object_penalty += penalty
            if cluster_id:
                planner_clusters.add(cluster_id)
            if cluster_id and object_id:
                planner_objects.add((cluster_id, object_id))
        elif kind in {"cluster_directional_relation", "cluster_orientation"}:
            macro_penalty += penalty

    prioritized_objects = [
        row
        for row in diagnosis.get("prioritized_objects") or []
        if isinstance(row, dict)
        and isinstance(row.get("cluster_id"), str)
        and isinstance(row.get("object_id"), str)
    ]
    dominant = bool(
        planner_object_penalty >= max(900, macro_penalty)
        and len(planner_clusters) <= 2
        and (planner_objects or prioritized_objects)
    )
    return {
        "dominant": dominant,
        "planner_object_penalty": planner_object_penalty,
        "macro_penalty": macro_penalty,
        "cluster_count": len(planner_clusters),
        "object_count": len(planner_objects) or len(prioritized_objects),
    }


def _summarize_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    metrics = evaluation.get("metrics") or {}
    diagnosis = evaluation.get("diagnosis") or {}
    layout = evaluation.get("materialized_layout") or {}

    cluster_summary = []
    zone_by_cluster = {
        str(row.get("cluster_id") or ""): str(row.get("zone") or "")
        for row in metrics.get("zone_usage_summary") or []
        if isinstance(row, dict)
    }
    entry_by_cluster = {
        str(row.get("cluster_id") or ""): row.get("distance_to_nearest_door_mm")
        for row in metrics.get("cluster_entry_proximity") or []
        if isinstance(row, dict)
    }
    window_by_cluster = {
        str(row.get("cluster_id") or ""): row.get("distance_to_nearest_window_mm")
        for row in metrics.get("cluster_window_alignment") or []
        if isinstance(row, dict)
    }
    priority_cluster_scores = {
        str(row.get("cluster_id") or ""): row.get("score")
        for row in diagnosis.get("prioritized_clusters") or []
        if isinstance(row, dict)
    }

    for cluster in (layout.get("clusters") or [])[:8]:
        cid = str(cluster.get("cluster_id") or "")
        cluster_summary.append(
            {
                "cluster_id": cid,
                "variant_id": cluster.get("variant_id"),
                "zone": zone_by_cluster.get(cid),
                "distance_to_nearest_door_mm": entry_by_cluster.get(cid),
                "distance_to_nearest_window_mm": window_by_cluster.get(cid),
                "soft_score": priority_cluster_scores.get(cid),
            }
        )

    object_summary = []
    for row in (metrics.get("object_front_clearance") or [])[:8]:
        if not isinstance(row, dict):
            continue
        object_summary.append(
            {
                "cluster_id": row.get("cluster_id"),
                "object_id": row.get("object_id"),
                "current_front_clear_mm": row.get("current_front_clear_mm"),
                "best_open_clear_mm": row.get("best_open_clear_mm"),
                "shortage_mm": row.get("shortage_mm"),
            }
        )

    return {
        "hard_valid": bool(evaluation.get("hard_valid")),
        "score_summary": deepcopy(metrics.get("score_summary") or {}),
        "goal_alignment_summary": deepcopy(metrics.get("goal_alignment_summary") or {}),
        "cluster_summary": cluster_summary,
        "object_summary": object_summary,
        "key_findings": deepcopy(diagnosis.get("key_findings") or []),
    }


def _summarize_proposal_direction(
    payload: dict[str, Any], proposal: dict[str, Any]
) -> dict[str, Any]:
    seed_layout = payload.get("seed_layout") or {}
    seed_tmap = _seed_transform_map(seed_layout)
    seed_vmap = _seed_variant_map(seed_layout)
    changed_clusters: list[str] = []
    move_families: list[str] = []

    for row in proposal.get("cluster_transforms") or []:
        if not isinstance(row, dict):
            continue
        cid = row.get("cluster_id")
        if not isinstance(cid, str):
            continue
        seed_row = seed_tmap.get(cid, {})
        if int(row.get("rot") or 0) != int(seed_row.get("rot") or 0):
            move_families.append("cluster_rotation")
            changed_clusters.append(cid)
        if int(row.get("x") or 0) != int(seed_row.get("x") or 0) or int(
            row.get("y") or 0
        ) != int(seed_row.get("y") or 0):
            move_families.append("cluster_translation")
            changed_clusters.append(cid)

    for row in proposal.get("selected_variants") or []:
        if not isinstance(row, dict):
            continue
        cid = row.get("cluster_id")
        if not isinstance(cid, str):
            continue
        seed_row = seed_vmap.get(cid, {})
        if str(row.get("variant_id") or "") != str(seed_row.get("variant_id") or ""):
            move_families.append("variant_switch")
            changed_clusters.append(cid)

    for row in proposal.get("object_repairs") or []:
        if not isinstance(row, dict):
            continue
        cluster_id = row.get("cluster_id")
        if isinstance(cluster_id, str):
            changed_clusters.append(cluster_id)
        move_families.append(str(row.get("op") or "object_repair"))

    if not move_families:
        move_families.append("seed_retain")

    dedup_clusters = list(dict.fromkeys(changed_clusters))[:4]
    dedup_families = list(dict.fromkeys(move_families))
    summary = ", ".join(dedup_families)
    if dedup_clusters:
        summary = f"{summary} on {', '.join(dedup_clusters)}"

    return {
        "primary_clusters": dedup_clusters,
        "move_families": dedup_families,
        "summary": summary,
    }


def _build_history_context(history: list[dict[str, Any]]) -> dict[str, Any]:
    recent_attempts: list[dict[str, Any]] = []
    cluster_counts: dict[str, int] = {}
    priority_cluster_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    issue_family_counts: dict[str, int] = {}

    for entry in history[-8:]:
        direction = entry.get("direction_summary") or {}
        tool_eval = entry.get("tool_evaluation") or {}
        judge_eval = entry.get("judge_evaluation") or {}
        priority_clusters = [
            str(cluster_id)
            for cluster_id in (judge_eval.get("priority_clusters") or [])
            if isinstance(cluster_id, str) and cluster_id
        ]
        issue_families = _judge_issue_families(judge_eval)
        attempt = {
            "iteration": entry.get("iteration"),
            "search_phase": _normalize_search_phase(entry.get("search_phase")),
            "summary": direction.get("summary"),
            "primary_clusters": deepcopy(direction.get("primary_clusters") or []),
            "move_families": deepcopy(direction.get("move_families") or []),
            "priority_clusters": priority_clusters,
            "issue_families": issue_families,
            "next_step_mode": str(judge_eval.get("next_step_mode") or ""),
            "hard_valid": bool(tool_eval.get("hard_valid")),
            "judge_verdict": judge_eval.get("verdict"),
            "judge_score": judge_eval.get("reasonableness_score"),
            "delta_score": (
                (tool_eval.get("baseline_comparison") or {}).get("delta_score")
            ),
        }
        recent_attempts.append(attempt)
        for cluster_id in attempt["primary_clusters"]:
            cluster_counts[str(cluster_id)] = cluster_counts.get(str(cluster_id), 0) + 1
        for cluster_id in priority_clusters:
            priority_cluster_counts[cluster_id] = (
                priority_cluster_counts.get(cluster_id, 0) + 1
            )
        for family in attempt["move_families"]:
            family_counts[str(family)] = family_counts.get(str(family), 0) + 1
        for family in issue_families:
            issue_family_counts[str(family)] = (
                issue_family_counts.get(str(family), 0) + 1
            )

    stuck_clusters = [
        cluster_id
        for cluster_id, count in sorted(
            cluster_counts.items(), key=lambda item: (-item[1], item[0])
        )
        if count >= 3
    ]
    stuck_priority_clusters = [
        cluster_id
        for cluster_id, count in sorted(
            priority_cluster_counts.items(), key=lambda item: (-item[1], item[0])
        )
        if count >= 3
    ]
    avoid_move_families = [
        family
        for family, count in sorted(
            family_counts.items(), key=lambda item: (-item[1], item[0])
        )
        if count >= 3
    ]
    repeated_issue_families = [
        family
        for family, count in sorted(
            issue_family_counts.items(), key=lambda item: (-item[1], item[0])
        )
        if count >= 3
    ]

    stuck_patterns: list[str] = []
    if stuck_clusters:
        stuck_patterns.append(
            "Repeated focus on the same clusters without acceptance: "
            + ", ".join(stuck_clusters[:4])
        )
    if stuck_priority_clusters:
        stuck_patterns.append(
            "Judge keeps prioritizing the same clusters: "
            + ", ".join(stuck_priority_clusters[:4])
        )
    if avoid_move_families:
        stuck_patterns.append(
            "Repeated move families without acceptance: "
            + ", ".join(avoid_move_families[:4])
        )
    if repeated_issue_families:
        stuck_patterns.append(
            "Repeated issue families without convergence: "
            + ", ".join(repeated_issue_families[:4])
        )

    return {
        "recent_attempts": recent_attempts,
        "stuck_clusters": stuck_clusters,
        "stuck_priority_clusters": stuck_priority_clusters,
        "avoid_move_families": avoid_move_families,
        "repeated_issue_families": repeated_issue_families,
        "stuck_patterns": stuck_patterns,
    }


def _select_shortlist_candidate(
    candidates: list[dict[str, Any]],
    history: list[dict[str, Any]],
    search_phase: str,
) -> dict[str, Any] | None:
    if not candidates:
        return None
    best = candidates[0]
    if _normalize_search_phase(search_phase) != "macro_layout" or len(candidates) == 1:
        return best

    best_delta = _delta_score(best.get("evaluation") or {})
    recent_macro_attempts = [
        entry
        for entry in history[-3:]
        if _normalize_search_phase(entry.get("search_phase")) == "macro_layout"
    ]
    recent_clusters = {
        _primary_cluster_from_direction(entry.get("direction_summary") or {})
        for entry in recent_macro_attempts
    }
    recent_clusters.discard("")
    recent_families = {
        str(family)
        for entry in recent_macro_attempts
        for family in (
            (entry.get("direction_summary") or {}).get("move_families") or []
        )
        if isinstance(family, str)
    }

    if best_delta >= 160 or not recent_macro_attempts:
        return best

    for candidate in candidates[1:]:
        evaluation = candidate.get("evaluation") or {}
        if not bool(evaluation.get("hard_valid")):
            continue
        if _delta_score(evaluation) < 0:
            continue
        direction = candidate.get("direction_summary") or {}
        primary_cluster = _primary_cluster_from_direction(direction)
        move_families = {
            str(family)
            for family in (direction.get("move_families") or [])
            if isinstance(family, str)
        }
        if primary_cluster and primary_cluster not in recent_clusters:
            return candidate
        if move_families and move_families.isdisjoint(recent_families):
            return candidate

    return best


def _meaningful_delta(search_phase: str) -> int:
    return _MEANINGFUL_DELTA_BY_PHASE.get(
        _normalize_search_phase(search_phase),
        _MEANINGFUL_DELTA_BY_PHASE["macro_layout"],
    )


def _recent_attempts_for_phase(
    history: list[dict[str, Any]], search_phase: str, limit: int
) -> list[dict[str, Any]]:
    normalized_phase = _normalize_search_phase(search_phase)
    return [
        entry
        for entry in history[-limit:]
        if _normalize_search_phase(entry.get("search_phase")) == normalized_phase
    ]


def _should_stop_macro_search(
    *,
    history: list[dict[str, Any]],
    deterministic_shortlist: list[dict[str, Any]],
    search_phase: str,
) -> bool:
    if _normalize_search_phase(search_phase) != "macro_layout":
        return False

    recent_macro_attempts = _recent_attempts_for_phase(history, search_phase, limit=4)
    if len(recent_macro_attempts) < 4:
        return False

    if any(
        bool((item.get("evaluation") or {}).get("hard_valid"))
        and _delta_score(item.get("evaluation") or {})
        >= _meaningful_delta(search_phase)
        for item in deterministic_shortlist
    ):
        return False

    history_context = _build_history_context(recent_macro_attempts)
    repeated_clusters = bool(history_context.get("stuck_priority_clusters"))
    repeated_issue_families = bool(history_context.get("repeated_issue_families"))
    if not repeated_clusters and not repeated_issue_families:
        return False

    for entry in recent_macro_attempts:
        tool_eval = entry.get("tool_evaluation") or {}
        judge_eval = entry.get("judge_evaluation") or {}
        if str(judge_eval.get("verdict") or "").upper() == "ACCEPT":
            return False
        if not bool(tool_eval.get("hard_valid")):
            return False
        if _delta_score(tool_eval) >= _meaningful_delta(search_phase):
            return False

    return True


def _should_stop_object_refine_search(
    *,
    history: list[dict[str, Any]],
    deterministic_shortlist: list[dict[str, Any]],
    search_phase: str,
) -> bool:
    if _normalize_search_phase(search_phase) != "object_refine":
        return False

    recent_object_attempts = [
        entry
        for entry in history[-3:]
        if _normalize_search_phase(entry.get("search_phase")) == "object_refine"
    ]
    if len(recent_object_attempts) < 3:
        return False

    if any(
        bool((item.get("evaluation") or {}).get("hard_valid"))
        and _delta_score(item.get("evaluation") or {})
        >= _meaningful_delta(search_phase)
        for item in deterministic_shortlist
    ):
        return False

    for entry in recent_object_attempts:
        tool_eval = entry.get("tool_evaluation") or {}
        judge_eval = entry.get("judge_evaluation") or {}
        if bool(tool_eval.get("hard_valid")) and _delta_score(
            tool_eval
        ) >= _meaningful_delta(search_phase):
            return False
        if str(judge_eval.get("verdict") or "").upper() == "ACCEPT":
            return False

    return True


def _should_enter_object_refine_phase(
    *,
    iteration: int,
    history: list[dict[str, Any]],
    macro_shortlist: list[dict[str, Any]],
    baseline_evaluation: dict[str, Any],
) -> bool:
    positive_macro = any(
        bool(item.get("evaluation", {}).get("hard_valid"))
        and _delta_score(item.get("evaluation") or {}) > 0
        for item in macro_shortlist
    )
    if positive_macro:
        return False

    recent_macro_attempts = [
        entry
        for entry in history[-3:]
        if _normalize_search_phase(entry.get("search_phase")) == "macro_layout"
    ]
    stalled_macro_attempts = [
        entry
        for entry in recent_macro_attempts
        if int(
            (
                (entry.get("tool_evaluation") or {})
                .get("baseline_comparison", {})
                .get("delta_score")
            )
            or 0
        )
        <= 0
    ]

    planner_object_pressure = _planner_object_pressure_summary(baseline_evaluation)
    if _has_macro_orientation_pressure(baseline_evaluation) and not bool(
        planner_object_pressure.get("dominant")
    ):
        return False

    if bool(planner_object_pressure.get("dominant")):
        return iteration >= 2 or len(stalled_macro_attempts) >= 1

    return iteration >= 4 or len(stalled_macro_attempts) >= 2


def _build_judge_payload(
    *,
    payload: dict[str, Any],
    baseline_evaluation: dict[str, Any],
    candidate_evaluation: dict[str, Any],
    iteration: int,
    max_iterations: int,
    history: list[dict[str, Any]],
    direction_summary: dict[str, Any],
) -> dict[str, Any]:
    room_model = (payload.get("room_context") or {}).get("room_model_used") or {}
    room_notes = deepcopy(room_model.get("notes") or [])
    candidate_metrics = candidate_evaluation.get("metrics") or {}
    return {
        "room_context": deepcopy(payload.get("room_context") or {}),
        "design_goals": deepcopy(payload.get("goals") or {}),
        "baseline_summary": _summarize_evaluation(baseline_evaluation),
        "candidate_summary": _summarize_evaluation(candidate_evaluation),
        "comparison_summary": deepcopy(
            candidate_evaluation.get("baseline_comparison") or {}
        ),
        "hard_check_summary": {
            "hard_valid": bool(candidate_evaluation.get("hard_valid")),
            "errors": deepcopy(candidate_evaluation.get("errors") or []),
            "violations_by_cluster": deepcopy(
                candidate_evaluation.get("violations_by_cluster") or {}
            ),
        },
        "metrics": deepcopy(candidate_metrics),
        "diagnosis": deepcopy(candidate_evaluation.get("diagnosis") or {}),
        "generic_signals": {
            "room_notes": room_notes,
            "goal_alignment_summary": deepcopy(
                candidate_metrics.get("goal_alignment_summary") or {}
            ),
            "path_obstruction_summary": deepcopy(
                candidate_metrics.get("path_obstruction_summary") or {}
            ),
            "zone_usage_summary": deepcopy(
                candidate_metrics.get("zone_usage_summary") or []
            ),
        },
        "controller_context": {
            "iteration": iteration,
            "max_iterations": max_iterations,
            "search_phase": _normalize_search_phase(
                (payload.get("phase_control") or {}).get("repair_phase")
            ),
            "current_direction": deepcopy(direction_summary),
            **_build_history_context(history),
        },
    }


def _recommended_search_phase(
    *,
    evaluation: dict[str, Any],
    judge_output: dict[str, Any] | None,
    history: list[dict[str, Any]],
) -> str:
    if not bool(evaluation.get("hard_valid")):
        return "macro_layout"
    if _has_macro_orientation_pressure(evaluation):
        return "macro_layout"

    history_context = _build_history_context(history)
    requested = _normalize_search_phase(
        ((judge_output or {}).get("next_step_mode") or "macro_layout")
    )
    repeated_macro_issues = bool(
        set(history_context.get("repeated_issue_families") or []).intersection(
            {"orientation", "focal", "circulation", "zoning", "openings"}
        )
    )
    repeated_priority_clusters = bool(history_context.get("stuck_priority_clusters"))
    current_delta = _delta_score(evaluation)

    if repeated_macro_issues and current_delta < _meaningful_delta("macro_layout"):
        return "macro_layout"
    if repeated_priority_clusters and current_delta < _meaningful_delta(requested):
        return "macro_layout"
    if str((judge_output or {}).get("verdict") or "").upper() == "ACCEPT":
        return "stop"
    return requested


def _build_controller_feedback(
    *,
    iteration: int,
    evaluation: dict[str, Any],
    judge_output: dict[str, Any] | None,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    diagnosis = evaluation.get("diagnosis") or {}
    errors = evaluation.get("errors") or []
    history_context = _build_history_context(history)
    priority_clusters = []
    if isinstance(judge_output, dict) and judge_output.get("priority_clusters"):
        priority_clusters = deepcopy(judge_output.get("priority_clusters") or [])
    else:
        priority_clusters = [
            row.get("cluster_id")
            for row in diagnosis.get("prioritized_clusters") or []
            if isinstance(row, dict) and isinstance(row.get("cluster_id"), str)
        ][:4]

    candidate_status = "hard_invalid"
    repair_direction = "Fix hard validity before making new soft edits."
    judge_top_issues: list[str] = []
    if bool(evaluation.get("hard_valid")):
        candidate_status = "soft_revision_needed"
        repair_direction = "Keep the useful direction, but switch tactic if the same move keeps repeating."
    if isinstance(judge_output, dict):
        verdict = str(judge_output.get("verdict") or "").upper()
        if verdict == "ACCEPT":
            candidate_status = "accepted"
            repair_direction = "This candidate is acceptable."
        elif verdict == "REJECT":
            candidate_status = "rejected"
            repair_direction = "Try a different cluster or move family."
        judge_top_issues = deepcopy(judge_output.get("top_issues") or [])
        if judge_output.get("repair_advice"):
            repair_direction = str(
                (judge_output.get("repair_advice") or [""])[0] or repair_direction
            )

    recommended_search_phase = _recommended_search_phase(
        evaluation=evaluation,
        judge_output=judge_output,
        history=history,
    )

    return {
        "iteration": iteration,
        "candidate_status": candidate_status,
        "hard_invalid_reasons": deepcopy(errors),
        "judge_top_issues": judge_top_issues,
        "priority_clusters": priority_clusters,
        "priority_objects": deepcopy((diagnosis.get("prioritized_objects") or [])[:6]),
        "repair_direction": repair_direction,
        "stuck_clusters": deepcopy(history_context.get("stuck_clusters") or []),
        "stuck_priority_clusters": deepcopy(
            history_context.get("stuck_priority_clusters") or []
        ),
        "avoid_move_families": deepcopy(
            history_context.get("avoid_move_families") or []
        ),
        "repeated_issue_families": deepcopy(
            history_context.get("repeated_issue_families") or []
        ),
        "recommended_search_phase": recommended_search_phase,
        "recent_attempt_summaries": deepcopy(
            history_context.get("recent_attempts") or []
        ),
    }


def _fallback_moves(
    payload: dict[str, Any],
    baseline_evaluation: dict[str, Any],
    history: list[dict[str, Any]],
    search_phase: str,
) -> list[dict[str, Any]]:
    history_context = _build_history_context(history)
    stuck_clusters = set(history_context.get("stuck_clusters") or [])
    avoid_move_families = set(history_context.get("avoid_move_families") or [])

    moves = deepcopy(
        (baseline_evaluation.get("diagnosis") or {}).get("enumerated_moves") or []
    )
    if not moves:
        moves = deepcopy(
            (EnumeratePhase2RepairMoves(payload=payload, limit=32).get("moves") or [])
        )
    moves = [
        move
        for move in moves
        if isinstance(move, dict) and _move_matches_search_phase(move, search_phase)
    ]

    if _needs_primary_pair_axis_repair(payload, baseline_evaluation):
        object_payload = deepcopy(payload)
        _set_search_phase(object_payload, "object_refine")
        extra_object_moves = deepcopy(
            (
                EnumeratePhase2RepairMoves(payload=object_payload, limit=24).get(
                    "moves"
                )
                or []
            )
        )
        seen_move_signatures = {
            json.dumps(move, sort_keys=True, ensure_ascii=True)
            for move in moves
            if isinstance(move, dict)
        }
        for move in extra_object_moves:
            if not isinstance(move, dict):
                continue
            signature = json.dumps(move, sort_keys=True, ensure_ascii=True)
            if signature in seen_move_signatures:
                continue
            moves.append(move)
            seen_move_signatures.add(signature)

    def _move_family(move: dict[str, Any]) -> str:
        kind = str(move.get("kind") or "")
        if kind == "cluster_variant":
            return "variant_switch"
        if kind == "cluster_pose":
            proposal = move.get("proposal") or {}
            transforms = proposal.get("cluster_transforms") or []
            transform = (transforms or [None])[0] or {}
            if len(transforms) > 1:
                if any(
                    int(row.get("rot") or 0)
                    != int(
                        _seed_transform_map(payload.get("seed_layout") or {})
                        .get(
                            str(row.get("cluster_id") or ""),
                            {},
                        )
                        .get("rot")
                        or 0
                    )
                    for row in transforms
                    if isinstance(row, dict)
                ):
                    return "cluster_pair_rotation"
                return "cluster_pair_translation"
            seed_row = _seed_transform_map(payload.get("seed_layout") or {}).get(
                str(transform.get("cluster_id") or ""),
                {},
            )
            if int(transform.get("rot") or 0) != int(seed_row.get("rot") or 0):
                return "cluster_rotation"
            return "cluster_translation"
        if kind == "object_pose":
            repair_row = ((move.get("proposal") or {}).get("object_repairs") or [None])[
                0
            ] or {}
            return str(repair_row.get("op") or "object_repair")
        return kind or "unknown"

    moves.sort(
        key=lambda move: (
            _viewing_pair_move_priority(move, payload),
            1 if str(move.get("cluster_id") or "") in stuck_clusters else 0,
            1 if _move_family(move) in avoid_move_families else 0,
        )
    )
    cluster_order: list[str] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for move in moves:
        cluster_id = str(move.get("cluster_id") or "")
        grouped.setdefault(cluster_id, []).append(move)
        if cluster_id not in cluster_order:
            cluster_order.append(cluster_id)

    interleaved: list[dict[str, Any]] = []
    appended = True
    while appended:
        appended = False
        for cluster_id in cluster_order:
            bucket = grouped.get(cluster_id) or []
            if not bucket:
                continue
            interleaved.append(bucket.pop(0))
            appended = True
    return interleaved


def _forced_macro_moves(
    payload: dict[str, Any], history: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    history_context = _build_history_context(history)
    stuck_clusters = set(history_context.get("stuck_clusters") or [])
    diagnosis = payload.get("repair_debug") or {}
    targets = (
        (diagnosis.get("repair_targets") or {}).get("prioritized_clusters")
        or (diagnosis.get("seed_verify") or {})
        .get("repair_guidance", {})
        .get("prioritized_clusters")
        or []
    )
    cluster_ids = [
        row.get("cluster_id")
        for row in targets
        if isinstance(row, dict) and isinstance(row.get("cluster_id"), str)
    ]
    if not cluster_ids:
        cluster_ids = sorted(
            _seed_transform_map(payload.get("seed_layout") or {}).keys()
        )
    cluster_ids = sorted(
        cluster_ids, key=lambda cluster_id: (cluster_id in stuck_clusters, cluster_id)
    )

    grid_mm = normalize_layout_grid_mm(
        (payload.get("room_context") or {}).get("grid_mm")
    )
    tmap = _seed_transform_map(payload.get("seed_layout") or {})
    forced: list[dict[str, Any]] = []
    for cluster_id in cluster_ids[:4]:
        seed_row = tmap.get(cluster_id)
        if not isinstance(seed_row, dict):
            continue
        for rot in (0, 90, 180, 270):
            if rot == int(seed_row.get("rot") or 0):
                continue
            forced.append(
                {
                    "cluster_transforms": [{**deepcopy(seed_row), "rot": rot}],
                    "selected_variants": [],
                    "object_repairs": [],
                    "notes": [f"Controller forced macro rotation for {cluster_id}."],
                }
            )
        for dx, dy in (
            (grid_mm, 0),
            (-grid_mm, 0),
            (0, grid_mm),
            (0, -grid_mm),
        ):
            forced.append(
                {
                    "cluster_transforms": [
                        {
                            **deepcopy(seed_row),
                            "x": int(seed_row.get("x") or 0) + dx,
                            "y": int(seed_row.get("y") or 0) + dy,
                        }
                    ],
                    "selected_variants": [],
                    "object_repairs": [],
                    "notes": [f"Controller forced macro translation for {cluster_id}."],
                }
            )
    return forced


def _cluster_cards_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for card in payload.get("cluster_cards") or []:
        if not isinstance(card, dict):
            continue
        cluster_id = str(card.get("cluster_id") or "").strip()
        if cluster_id:
            out[cluster_id] = card
    return out


def _objects_world_map(
    payload: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in payload.get("objects_world") or []:
        if not isinstance(row, dict):
            continue
        cluster_id = str(row.get("cluster_id") or "").strip()
        object_id = str(row.get("object_id") or "").strip()
        if cluster_id and object_id:
            out[(cluster_id, object_id)] = row
    return out


def _bbox_center(bbox: dict[str, Any]) -> tuple[float, float]:
    return (
        (float(bbox.get("min_x", 0)) + float(bbox.get("max_x", 0))) / 2.0,
        (float(bbox.get("min_y", 0)) + float(bbox.get("max_y", 0))) / 2.0,
    )


def _world_relative_side(base_bbox: dict[str, Any], object_bbox: dict[str, Any]) -> str:
    base_cx, base_cy = _bbox_center(base_bbox)
    obj_cx, obj_cy = _bbox_center(object_bbox)
    dx = obj_cx - base_cx
    dy = obj_cy - base_cy
    if abs(dx) > abs(dy):
        return "right" if dx >= 0 else "left"
    return "top" if dy >= 0 else "bottom"


def _semantic_side_targets_world(
    *,
    rule: dict[str, Any],
    base_row: dict[str, Any],
    object_row: dict[str, Any],
    grid_mm: int,
) -> list[dict[str, Any]]:
    base_bbox = base_row.get("bbox") or {}
    object_bbox = object_row.get("bbox") or {}
    if not isinstance(base_bbox, dict) or not isinstance(object_bbox, dict):
        return []
    base_cx, base_cy = _bbox_center(base_bbox)
    base_w = float(base_bbox.get("max_x", 0)) - float(base_bbox.get("min_x", 0))
    base_h = float(base_bbox.get("max_y", 0)) - float(base_bbox.get("min_y", 0))
    obj_w = float(object_bbox.get("max_x", 0)) - float(object_bbox.get("min_x", 0))
    obj_h = float(object_bbox.get("max_y", 0)) - float(object_bbox.get("min_y", 0))
    gap = max(
        int(rule.get("gap_min") or 0), min(int(rule.get("gap_max") or 0) or 0, 400)
    )
    if gap <= 0:
        gap = max(grid_mm * 2, 150)

    front_world = (
        base_row.get("front_world")
        if isinstance(base_row.get("front_world"), dict)
        else {}
    )
    fx = float(front_world.get("dx", 0.0))
    fy = float(front_world.get("dy", 0.0))
    if abs(fx) >= abs(fy):
        front_side = "right" if fx >= 0 else "left"
    else:
        front_side = "top" if fy >= 0 else "bottom"

    def _rotate_side(side: str, quarter_turns: int) -> str:
        sides = ["top", "right", "bottom", "left"]
        try:
            idx = sides.index(side)
        except ValueError:
            return side
        return sides[(idx + quarter_turns) % 4]

    left = _rotate_side(front_side, -1)
    right = _rotate_side(front_side, 1)
    back = _rotate_side(front_side, 2)
    option_to_side = {
        "head": front_side,
        "front": front_side,
        "left": left,
        "right": right,
        "back": back,
        "foot": back,
        "head_left": front_side,
        "head_right": front_side,
        "front_left": front_side,
        "front_right": front_side,
        "foot_left": back,
        "foot_right": back,
        "back_left": back,
        "back_right": back,
    }
    targets: list[dict[str, Any]] = []
    for option in rule.get("side_options") or []:
        token = str(option).strip().lower()
        world_side = option_to_side.get(token)
        if world_side is None:
            continue
        if world_side == "right":
            tx = base_cx + (base_w / 2.0 + obj_w / 2.0 + gap)
            ty = base_cy
        elif world_side == "left":
            tx = base_cx - (base_w / 2.0 + obj_w / 2.0 + gap)
            ty = base_cy
        elif world_side == "top":
            tx = base_cx
            ty = base_cy + (base_h / 2.0 + obj_h / 2.0 + gap)
        else:
            tx = base_cx
            ty = base_cy - (base_h / 2.0 + obj_h / 2.0 + gap)
        if token.endswith("_left"):
            tx -= max(base_w * 0.18, obj_w * 0.18)
        elif token.endswith("_right"):
            tx += max(base_w * 0.18, obj_w * 0.18)
        targets.append(
            {"option": str(option), "world_side": world_side, "cx": tx, "cy": ty}
        )
    return targets


def _semantic_object_move_proposals(
    *,
    payload: dict[str, Any],
    prioritized_objects: list[tuple[str, str]],
    grid_mm: int,
) -> list[dict[str, Any]]:
    cards = _cluster_cards_map(payload)
    objects_world = _objects_world_map(payload)
    proposals: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _opposite_side(side: str) -> str:
        return {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}.get(
            side, side
        )

    def _front_side_from_row(row: dict[str, Any]) -> str:
        front = (
            row.get("front_world") if isinstance(row.get("front_world"), dict) else {}
        )
        dx = float(front.get("dx", 0.0))
        dy = float(front.get("dy", 0.0))
        if abs(dx) >= abs(dy):
            return "right" if dx >= 0 else "left"
        return "top" if dy >= 0 else "bottom"

    for cluster_id, object_id in prioritized_objects[:8]:
        card = cards.get(cluster_id) or {}
        rules = (card.get("cluster_rules") or {}).get("semantic_placements") or []
        object_row = objects_world.get((cluster_id, object_id))
        if not isinstance(object_row, dict):
            continue
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            if str(rule.get("id") or "") != object_id:
                continue
            base_id = str(rule.get("relative_to") or "").strip()
            base_row = objects_world.get((cluster_id, base_id))
            if not isinstance(base_row, dict):
                continue
            targets = _semantic_side_targets_world(
                rule=rule,
                base_row=base_row,
                object_row=object_row,
                grid_mm=grid_mm,
            )
            if not targets:
                continue
            object_bbox = object_row.get("bbox") or {}
            base_bbox = base_row.get("bbox") or {}
            if not isinstance(object_bbox, dict) or not isinstance(base_bbox, dict):
                continue
            current_side = _world_relative_side(base_bbox, object_bbox)
            support_role = str(rule.get("support_role") or "").strip().lower()
            base_front_side = _front_side_from_row(base_row)
            obj_front_side = _front_side_from_row(object_row)
            target = min(
                targets,
                key=lambda t: (
                    0
                    if str(t.get("world_side") or "") == base_front_side
                    and support_role == "frontal_support"
                    else 1,
                    abs(float(t["cx"]) - _bbox_center(object_bbox)[0])
                    + abs(float(t["cy"]) - _bbox_center(object_bbox)[1]),
                ),
            )
            obj_cx, obj_cy = _bbox_center(object_bbox)
            dx = int(round((float(target["cx"]) - obj_cx) / max(grid_mm, 1))) * grid_mm
            dy = int(round((float(target["cy"]) - obj_cy) / max(grid_mm, 1))) * grid_mm
            desired_front_side = None
            orientation = str(rule.get("orientation") or "").strip().lower()
            if orientation == "same_direction" or support_role == "wall_support":
                desired_front_side = base_front_side
            elif support_role == "frontal_support":
                desired_front_side = base_front_side
            elif support_role == "secondary_seat":
                desired_front_side = base_front_side
            elif support_role == "side_support":
                desired_front_side = base_front_side
            elif orientation == "face_base" and current_side in {
                "left",
                "right",
                "top",
                "bottom",
            }:
                desired_front_side = _opposite_side(current_side)
            repairs = []
            if (
                dx != 0
                or dy != 0
                or current_side != str(target.get("world_side") or current_side)
            ):
                repairs.append(
                    {
                        "cluster_id": cluster_id,
                        "object_id": object_id,
                        "op": "nudge_object",
                        "params": {"dx": dx, "dy": dy},
                    }
                )
            if desired_front_side and desired_front_side != obj_front_side:
                side_to_vec = {
                    "left": {"dx": -1.0, "dy": 0.0},
                    "right": {"dx": 1.0, "dy": 0.0},
                    "top": {"dx": 0.0, "dy": 1.0},
                    "bottom": {"dx": 0.0, "dy": -1.0},
                }
                vec = side_to_vec.get(desired_front_side)
                if vec:
                    repairs.insert(
                        0,
                        {
                            "cluster_id": cluster_id,
                            "object_id": object_id,
                            "op": "set_front_override",
                            "params": vec,
                        },
                    )
            if not repairs:
                continue
            sig = json.dumps(repairs, sort_keys=True, ensure_ascii=True)
            if sig in seen:
                continue
            seen.add(sig)
            proposals.append(
                {
                    "cluster_transforms": [],
                    "selected_variants": [],
                    "object_repairs": repairs,
                    "notes": [
                        f"Controller snapped {cluster_id}.{object_id} toward semantic slot {target.get('option')} relative to {base_id}."
                    ],
                }
            )
            break
    return proposals


def _forced_object_moves(
    payload: dict[str, Any], history: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    history_context = _build_history_context(history)
    stuck_clusters = set(history_context.get("stuck_clusters") or [])
    grid_mm = normalize_layout_grid_mm(None)

    prioritized_objects = []
    diagnosis = payload.get("repair_debug") or {}
    meaningful_object_intents = {
        "face_object",
        "face_away_from_object",
        "front_to_open_space",
        "preserve_front_access",
        "back_to_wall",
        "face_entry",
        "front_to_entry",
        "face_window",
        "front_to_window",
        "in_front_of_anchor",
        "align_with_anchor_axis",
        "same_view_side_as_primary_pair",
        "not_behind_anchor_view",
        "preserve_access_around_anchor",
        "flank_anchor",
        "beside_secondary_seat",
        "same_direction_as_anchor",
        "face_cluster",
    }
    critical_orientation_targets = []
    for row in (
        (diagnosis.get("repair_targets") or {}).get("top_critical_orientation_issues")
    ) or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("kind") or "") != "object_orientation":
            continue
        intent = str(row.get("intent") or "").strip().lower()
        cid = row.get("cluster_id")
        oid = row.get("object_id")
        if (
            intent in meaningful_object_intents
            and isinstance(cid, str)
            and isinstance(oid, str)
        ):
            critical_orientation_targets.append((cid, oid))

    targets = (
        (diagnosis.get("repair_targets") or {}).get("prioritized_objects")
        or (diagnosis.get("seed_verify") or {})
        .get("repair_guidance", {})
        .get("prioritized_objects")
        or []
    )
    if critical_orientation_targets:
        prioritized_objects.extend(critical_orientation_targets)
    else:
        for row in targets:
            if not isinstance(row, dict):
                continue
            cid = row.get("cluster_id")
            oid = row.get("object_id")
            if isinstance(cid, str) and isinstance(oid, str):
                prioritized_objects.append((cid, oid))

    semantic_priority_seen: set[tuple[str, str]] = set(prioritized_objects)
    for row in payload.get("cluster_cards") or []:
        if not isinstance(row, dict):
            continue
        cluster_id = str(row.get("cluster_id") or "").strip()
        rules = (row.get("cluster_rules") or {}).get("semantic_placements") or []
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            object_id = str(rule.get("id") or "").strip()
            support_role = str(rule.get("support_role") or "").strip().lower()
            if not cluster_id or not object_id:
                continue
            if support_role in {"frontal_support", "secondary_seat", "side_support"}:
                key = (cluster_id, object_id)
                if key not in semantic_priority_seen:
                    prioritized_objects.append(key)
                    semantic_priority_seen.add(key)
    if not prioritized_objects:
        for row in payload.get("objects_world") or []:
            if not isinstance(row, dict):
                continue
            cid = row.get("cluster_id")
            oid = row.get("object_id")
            if isinstance(cid, str) and isinstance(oid, str):
                prioritized_objects.append((cid, oid))

    prioritized_objects = list(
        dict.fromkeys(
            sorted(
                prioritized_objects,
                key=lambda item: (item[0] in stuck_clusters, item[0], item[1]),
            )
        )
    )

    forced: list[dict[str, Any]] = []
    forced.extend(
        _semantic_object_move_proposals(
            payload=payload,
            prioritized_objects=prioritized_objects,
            grid_mm=grid_mm,
        )
    )
    single_repairs: list[dict[str, Any]] = []
    seen_single_repairs: set[tuple[str, str, str, str]] = set()
    seen_pair_repairs: set[str] = set()

    def _append_single_repair(
        *,
        cluster_id: str,
        object_id: str,
        op: str,
        params: dict[str, Any],
        note: str,
    ) -> None:
        repair_sig = (
            cluster_id,
            object_id,
            op,
            json.dumps(params, sort_keys=True, ensure_ascii=True),
        )
        if repair_sig in seen_single_repairs:
            return
        seen_single_repairs.add(repair_sig)
        repair = {
            "cluster_id": cluster_id,
            "object_id": object_id,
            "op": op,
            "params": deepcopy(params),
        }
        forced.append(
            {
                "cluster_transforms": [],
                "selected_variants": [],
                "object_repairs": [repair],
                "notes": [note],
            }
        )
        single_repairs.append(repair)

    for idx, (cluster_id, object_id) in enumerate(prioritized_objects[:6]):
        for rot in (90, 180, 270):
            _append_single_repair(
                cluster_id=cluster_id,
                object_id=object_id,
                op="rotate_object",
                params={"rot": rot},
                note=f"Controller forced object rotation for {cluster_id}.{object_id}.",
            )
        for axis in ("x", "y"):
            _append_single_repair(
                cluster_id=cluster_id,
                object_id=object_id,
                op="mirror_object",
                params={"axis": axis},
                note=f"Controller forced object mirror for {cluster_id}.{object_id}.",
            )
        if idx < 4:
            for dx, dy in ((0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)):
                _append_single_repair(
                    cluster_id=cluster_id,
                    object_id=object_id,
                    op="set_front_override",
                    params={"dx": dx, "dy": dy},
                    note=(
                        "Controller forced front override for "
                        f"{cluster_id}.{object_id}."
                    ),
                )
        for dx, dy in (
            (grid_mm, 0),
            (-grid_mm, 0),
            (0, grid_mm),
            (0, -grid_mm),
            (grid_mm, grid_mm),
            (grid_mm, -grid_mm),
            (-grid_mm, grid_mm),
            (-grid_mm, -grid_mm),
        ):
            _append_single_repair(
                cluster_id=cluster_id,
                object_id=object_id,
                op="nudge_object",
                params={"dx": dx, "dy": dy},
                note=f"Controller forced object nudge for {cluster_id}.{object_id}.",
            )

    pair_budget = 10
    for left_index, left in enumerate(single_repairs[:16]):
        for right in single_repairs[left_index + 1 : 16]:
            left_cluster_id = str(left.get("cluster_id") or "")
            right_cluster_id = str(right.get("cluster_id") or "")
            if not left_cluster_id or not right_cluster_id:
                continue
            if left_cluster_id == right_cluster_id:
                continue
            ordered_repairs = sorted(
                [deepcopy(left), deepcopy(right)],
                key=lambda row: (
                    str(row.get("cluster_id") or ""),
                    str(row.get("object_id") or ""),
                    str(row.get("op") or ""),
                    json.dumps(
                        row.get("params") or {}, sort_keys=True, ensure_ascii=True
                    ),
                ),
            )
            pair_sig = json.dumps(ordered_repairs, sort_keys=True, ensure_ascii=True)
            if pair_sig in seen_pair_repairs:
                continue
            seen_pair_repairs.add(pair_sig)
            forced.append(
                {
                    "cluster_transforms": [],
                    "selected_variants": [],
                    "object_repairs": ordered_repairs,
                    "notes": [
                        "Controller forced synchronized object repair for "
                        f"{left_cluster_id}.{left.get('object_id')} and "
                        f"{right_cluster_id}.{right.get('object_id')}."
                    ],
                }
            )
            pair_budget -= 1
            if pair_budget <= 0:
                return forced
    return forced


def _choose_fallback_proposal(
    *,
    payload: dict[str, Any],
    baseline_evaluation: dict[str, Any],
    attempted_signatures: set[str],
    history: list[dict[str, Any]],
    search_phase: str,
) -> dict[str, Any]:
    deterministic_shortlist = _shortlist_deterministic_candidates(
        payload=payload,
        baseline_evaluation=baseline_evaluation,
        attempted_signatures=attempted_signatures,
        history=history,
        search_phase=search_phase,
        max_candidates=16,
        shortlist_size=1,
    )
    if deterministic_shortlist:
        return deepcopy(deterministic_shortlist[0]["proposal"])

    for move in _fallback_moves(payload, baseline_evaluation, history, search_phase):
        reason = str(move.get("reason") or "Controller fallback move.")
        proposal = _canonical_repair(
            payload,
            move.get("proposal") or {},
            notes=[f"Controller fallback: {reason}"],
        )
        if _proposal_signature(proposal) not in attempted_signatures:
            return proposal

    forced_moves = (
        _forced_object_moves(payload, history)
        if _normalize_search_phase(search_phase) == "object_refine"
        else _forced_macro_moves(payload, history)
    )
    for partial in forced_moves:
        proposal = _canonical_repair(payload, partial)
        if _proposal_signature(proposal) not in attempted_signatures:
            return proposal

    return _canonical_repair(
        payload,
        notes=[
            "Controller retained the current baseline after exhausting deterministic move search."
        ],
    )


def _shortlist_deterministic_candidates(
    *,
    payload: dict[str, Any],
    baseline_evaluation: dict[str, Any],
    attempted_signatures: set[str],
    history: list[dict[str, Any]],
    search_phase: str,
    max_candidates: int = 12,
    shortlist_size: int = 3,
) -> list[dict[str, Any]]:
    shortlisted: list[dict[str, Any]] = []
    local_signatures: set[str] = set()
    deterministic_moves = _fallback_moves(
        payload, baseline_evaluation, history, search_phase
    )
    fallback_budget = max(4, int(max(1, int(max_candidates)) * 0.6))

    def _append_candidate(proposal: dict[str, Any], source: str) -> None:
        signature = _proposal_signature(proposal)
        if signature in attempted_signatures or signature in local_signatures:
            return
        evaluation = EvaluatePhase2Proposal(payload=payload, repair=proposal)
        direction_summary = _summarize_proposal_direction(payload, proposal)
        shortlisted.append(
            {
                "proposal": proposal,
                "evaluation": evaluation,
                "direction_summary": direction_summary,
                "source": source,
                "signature": signature,
            }
        )
        local_signatures.add(signature)

    for move in deterministic_moves:
        reason = str(move.get("reason") or "Controller shortlist move.")
        proposal = _canonical_repair(
            payload,
            move.get("proposal") or {},
            notes=[f"Deterministic shortlist: {reason}"],
        )
        _append_candidate(proposal, reason)
        if len(shortlisted) >= fallback_budget:
            break

    if len(shortlisted) < max(1, int(max_candidates)):
        forced_moves = (
            _forced_object_moves(payload, history)
            if _normalize_search_phase(search_phase) == "object_refine"
            else _forced_macro_moves(payload, history)
        )
        for partial in forced_moves:
            proposal = _canonical_repair(payload, partial)
            _append_candidate(proposal, "Controller forced move.")
            if len(shortlisted) >= max(1, int(max_candidates)):
                break

    if len(shortlisted) < max(1, int(max_candidates)):
        for move in deterministic_moves[fallback_budget:]:
            reason = str(move.get("reason") or "Controller shortlist move.")
            proposal = _canonical_repair(
                payload,
                move.get("proposal") or {},
                notes=[f"Deterministic shortlist: {reason}"],
            )
            _append_candidate(proposal, reason)
            if len(shortlisted) >= max(1, int(max_candidates)):
                break

    shortlisted.sort(
        key=lambda item: _search_candidate_rank(item["evaluation"]),
        reverse=True,
    )
    return shortlisted[: max(1, int(shortlist_size))]


def _should_use_deterministic_candidate(
    evaluation: dict[str, Any], history: list[dict[str, Any]]
) -> bool:
    if not bool(evaluation.get("hard_valid")):
        return False
    delta = _delta_score(evaluation)
    if delta > 0:
        return True
    history_context = _build_history_context(history)
    return bool(
        delta >= 0
        and (
            history_context.get("stuck_clusters")
            or history_context.get("avoid_move_families")
        )
    )


def _should_promote_candidate(
    evaluation: dict[str, Any],
    judge_output: dict[str, Any],
    *,
    search_phase: str,
) -> bool:
    if not bool(evaluation.get("hard_valid")):
        return False
    verdict = str(judge_output.get("verdict") or "").upper()
    if verdict == "REJECT":
        return False
    next_step_mode = _normalize_search_phase(judge_output.get("next_step_mode"))
    delta = _delta_score(evaluation)
    if verdict == "ACCEPT":
        if delta < 0:
            return False
        judge_score = int(judge_output.get("reasonableness_score") or 0)
        return judge_score >= 85
    if delta < _meaningful_delta(search_phase):
        return False
    if (
        _normalize_search_phase(search_phase) == "object_refine"
        and next_step_mode == "macro_layout"
    ):
        return False
    if delta > 0:
        return True
    if delta < 0:
        return False
    return False


def _is_orientation_snap_move(payload: dict[str, Any], move: dict[str, Any]) -> bool:
    if not isinstance(move, dict):
        return False
    kind = str(move.get("kind") or "")
    proposal = move.get("proposal") or {}
    if kind == "cluster_variant":
        return bool((proposal.get("selected_variants") or []))
    if kind != "cluster_pose":
        return False

    transforms = proposal.get("cluster_transforms") or []
    if len(transforms) != 1 or not isinstance(transforms[0], dict):
        return False
    transform = transforms[0]
    cluster_id = str(transform.get("cluster_id") or "")
    if not cluster_id:
        return False
    seed_map = _seed_transform_map(payload.get("seed_layout") or {})
    seed_row = seed_map.get(cluster_id) or {}
    return int(transform.get("rot") or 0) != int(seed_row.get("rot") or 0)


def _is_object_orientation_snap_move(move: dict[str, Any]) -> bool:
    if not isinstance(move, dict):
        return False
    if str(move.get("kind") or "") != "object_pose":
        return False
    repairs = (move.get("proposal") or {}).get("object_repairs") or []
    if len(repairs) != 1 or not isinstance(repairs[0], dict):
        return False
    return str(repairs[0].get("op") or "") in {
        "rotate_object",
        "mirror_object",
        "set_front_override",
    }


def _relation_plan_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    relation_plan = (payload.get("goals") or {}).get("relation_plan_used") or {}
    return relation_plan if isinstance(relation_plan, dict) else {}


def _normalized_cluster_pair(a: Any, b: Any) -> tuple[str, str] | None:
    left = str(a or "").strip()
    right = str(b or "").strip()
    if not left or not right or left == right:
        return None
    return (left, right)


def _anchor_layout_hints_by_cluster(
    payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    relation_plan = _relation_plan_from_payload(payload)
    candidates = []
    for container in (
        relation_plan,
        relation_plan.get("macro_concept")
        if isinstance(relation_plan.get("macro_concept"), dict)
        else None,
        (payload.get("goals") or {}),
    ):
        if isinstance(container, dict):
            candidates.append(container.get("anchor_layout_hints_by_cluster"))
    out: dict[str, dict[str, Any]] = {}
    for mapping in candidates:
        if not isinstance(mapping, dict):
            continue
        for cluster_id, row in mapping.items():
            cid = str(cluster_id or "").strip()
            if not cid or not isinstance(row, dict):
                continue
            out[cid] = deepcopy(row)
    return out


def _anchor_pairs_from_payload(payload: dict[str, Any]) -> list[tuple[str, str, str]]:
    relation_plan = _relation_plan_from_payload(payload)
    pair_rows: list[tuple[str, str, str]] = []
    raw_sources = [
        relation_plan.get("anchor_pairs"),
        (relation_plan.get("macro_concept") or {}).get("anchor_pairs")
        if isinstance(relation_plan.get("macro_concept"), dict)
        else None,
        (
            (payload.get("goals") or {}).get("anchor_pairs")
            if isinstance((payload.get("goals") or {}), dict)
            else None
        ),
    ]
    seen: set[tuple[str, str, str]] = set()
    for raw in raw_sources:
        if not isinstance(raw, list):
            continue
        for row in raw:
            if not isinstance(row, dict):
                continue
            pair = _normalized_cluster_pair(
                row.get("left_cluster_id") or row.get("a") or row.get("left"),
                row.get("right_cluster_id") or row.get("b") or row.get("right"),
            )
            if pair is None:
                continue
            mode = (
                str(
                    row.get("mode")
                    or row.get("relation")
                    or row.get("focus_mode")
                    or ""
                )
                .strip()
                .lower()
            )
            if mode in {"viewing", "face_each_other", "turn_toward"}:
                relation = "face_each_other"
            elif mode in {"access", "access_faces_other", "mixed"}:
                relation = "access_faces_other"
            else:
                relation = "face_each_other"
            key = (pair[0], pair[1], relation)
            if key in seen:
                continue
            seen.add(key)
            pair_rows.append(key)
    return pair_rows


def _viewing_cluster_pairs(payload: dict[str, Any]) -> list[tuple[str, str]]:
    relation_plan = (payload.get("goals") or {}).get("relation_plan_used") or {}
    layout_intent = relation_plan.get("layout_intent_profile") or {}
    primary_cluster_id = layout_intent.get("primary_cluster_id")
    secondary_cluster_id = layout_intent.get("secondary_cluster_id")
    focus_mode = str(layout_intent.get("focus_mode") or "").strip().lower()
    pairs: list[tuple[str, str]] = []

    if (
        focus_mode in {"viewing", "mixed"}
        and isinstance(primary_cluster_id, str)
        and primary_cluster_id
        and isinstance(secondary_cluster_id, str)
        and secondary_cluster_id
        and secondary_cluster_id != primary_cluster_id
    ):
        pairs.append((primary_cluster_id, secondary_cluster_id))

    for row in relation_plan.get("cluster_directional_relations") or []:
        if not isinstance(row, dict):
            continue
        relation = str(row.get("relation") or "").strip().lower()
        if relation not in {"face_each_other", "access_faces_other"}:
            continue
        a = row.get("a")
        b = row.get("b")
        if not isinstance(a, str) or not isinstance(b, str) or not a or not b:
            continue
        pair = (a, b)
        if pair not in pairs and (b, a) not in pairs:
            pairs.append(pair)

    return pairs


def _needs_primary_pair_axis_repair(
    payload: dict[str, Any],
    evaluation: dict[str, Any],
) -> bool:
    pair_clusters: set[str] = set()
    for a, b in _viewing_cluster_pairs(payload):
        pair_clusters.add(a)
        pair_clusters.add(b)
    if not pair_clusters:
        return False
    metrics = evaluation.get("metrics") or {}
    gate_reasons = {
        str(reason).strip()
        for reason in (metrics.get("score_summary") or {}).get(
            "quality_gate_reasons", []
        )
        if str(reason).strip()
    }
    if {
        "critical_orientation_penalty_too_high",
        "focal_pair_penalty_too_high",
    }.intersection(gate_reasons):
        return True
    for row in metrics.get("orientation_debug") or []:
        if not isinstance(row, dict):
            continue
        if int(row.get("penalty_mm") or 0) < 180:
            continue
        kind = str(row.get("kind") or "").strip()
        cluster_id = str(row.get("cluster_id") or "").strip()
        if kind in {
            "cluster_directional_relation",
            "cluster_orientation",
            "object_orientation",
        }:
            if not cluster_id or cluster_id in pair_clusters:
                return True
    diagnosis = evaluation.get("diagnosis") or {}
    prioritized_objects = {
        str(row.get("object_id") or row.get("id") or "").strip()
        for row in (diagnosis.get("prioritized_objects") or [])
        if isinstance(row, dict)
    }
    if prioritized_objects.intersection(
        {"sofa", "armchair", "coffee_table", "tv_console"}
    ):
        return True
    return False


def _viewing_pair_move_priority(
    move: dict[str, Any],
    payload: dict[str, Any],
) -> int:
    pair_clusters: set[str] = set()
    for a, b in _viewing_cluster_pairs(payload):
        pair_clusters.add(a)
        pair_clusters.add(b)
    if not pair_clusters:
        return 5
    kind = str(move.get("kind") or "")
    if kind == "cluster_pose":
        proposal = move.get("proposal") or {}
        transforms = [
            row
            for row in (proposal.get("cluster_transforms") or [])
            if isinstance(row, dict)
        ]
        seed_map = _seed_transform_map(payload.get("seed_layout") or {})
        if any(
            str(row.get("cluster_id") or "") in pair_clusters
            and int(row.get("rot") or 0)
            != int(
                (seed_map.get(str(row.get("cluster_id") or "")) or {}).get("rot") or 0
            )
            for row in transforms
        ):
            return 0
        if any(str(row.get("cluster_id") or "") in pair_clusters for row in transforms):
            return 1
    if kind == "object_pose":
        repairs = [
            row
            for row in ((move.get("proposal") or {}).get("object_repairs") or [])
            if isinstance(row, dict)
        ]
        for repair in repairs:
            cluster_id = str(repair.get("cluster_id") or "")
            object_id = str(repair.get("object_id") or "")
            op = str(repair.get("op") or "")
            if cluster_id in pair_clusters and op in {
                "rotate_object",
                "mirror_object",
                "set_front_override",
            }:
                return 2 if object_id in {"sofa", "tv_console"} else 3
            if cluster_id in pair_clusters and object_id in {
                "coffee_table",
                "armchair",
                "side_table",
            }:
                return 4
    return 5


def _anchor_contract_cluster_ids(payload: dict[str, Any]) -> set[str]:
    relation_plan = _relation_plan_from_payload(payload)
    protected_cluster_ids: set[str] = set()
    for left_cluster_id, right_cluster_id in _viewing_cluster_pairs(payload):
        protected_cluster_ids.add(left_cluster_id)
        protected_cluster_ids.add(right_cluster_id)
    for left_cluster_id, right_cluster_id, _relation in _anchor_pairs_from_payload(
        payload
    ):
        protected_cluster_ids.add(left_cluster_id)
        protected_cluster_ids.add(right_cluster_id)
    for cluster_id, hint in _anchor_layout_hints_by_cluster(payload).items():
        if not isinstance(hint, dict):
            continue
        if bool(hint.get("protect_anchor_orientation", True)) or bool(
            hint.get("protect_pair_axis", False)
        ):
            protected_cluster_ids.add(cluster_id)
    for row in relation_plan.get("cluster_orientations") or []:
        if not isinstance(row, dict):
            continue
        intents = {
            str(intent).strip().lower()
            for intent in (row.get("intents") or [])
            if str(intent).strip()
        }
        if "face_cluster" not in intents:
            continue
        cluster_id = str(row.get("cluster_id") or "").strip()
        target_cluster_id = str(row.get("target_cluster_id") or "").strip()
        if cluster_id:
            protected_cluster_ids.add(cluster_id)
        if target_cluster_id:
            protected_cluster_ids.add(target_cluster_id)
    return protected_cluster_ids


def _anchor_ids_by_cluster(payload: dict[str, Any]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    hints_by_cluster = _anchor_layout_hints_by_cluster(payload)
    for card in payload.get("cluster_cards") or []:
        if not isinstance(card, dict):
            continue
        cluster_id = str(card.get("cluster_id") or "").strip()
        if not cluster_id:
            continue
        anchors = {
            str(anchor).strip()
            for anchor in (card.get("anchors") or [])
            if isinstance(anchor, str) and str(anchor).strip()
        }
        anchor_override = str(card.get("anchor_override") or "").strip()
        if anchor_override:
            anchors.add(anchor_override)
        rules = card.get("cluster_rules") or {}
        if isinstance(rules, dict):
            anchor_first_policy = rules.get("anchor_first_policy") or {}
            if isinstance(anchor_first_policy, dict):
                dominant_anchor_id = str(
                    anchor_first_policy.get("dominant_anchor_id") or ""
                ).strip()
                if dominant_anchor_id:
                    anchors.add(dominant_anchor_id)
                for candidate in (
                    anchor_first_policy.get("dominant_anchor_candidates") or []
                ):
                    if isinstance(candidate, str) and candidate.strip():
                        anchors.add(candidate.strip())
        hint = hints_by_cluster.get(cluster_id) or {}
        if isinstance(hint, dict):
            for key in (
                "dominant_anchor_id",
                "anchor_object_id",
                "primary_anchor_object_id",
            ):
                value = str(hint.get(key) or "").strip()
                if value:
                    anchors.add(value)
        for row in card.get("decisions") or []:
            if not isinstance(row, dict):
                continue
            if str(row.get("priority") or "").strip().lower() != "anchor":
                continue
            object_id = str(row.get("object_type") or row.get("category") or "").strip()
            if object_id:
                anchors.add(object_id)
        out[cluster_id] = anchors
    return out


def _proposal_breaks_anchor_contract(
    payload: dict[str, Any],
    proposal: dict[str, Any],
) -> bool:
    protected_cluster_ids = _anchor_contract_cluster_ids(payload)
    if not protected_cluster_ids:
        return False

    seed_map = _seed_transform_map(payload.get("seed_layout") or {})
    for row in proposal.get("cluster_transforms") or []:
        if not isinstance(row, dict):
            continue
        cluster_id = str(row.get("cluster_id") or "").strip()
        if cluster_id not in protected_cluster_ids:
            continue
        seed_rot = int((seed_map.get(cluster_id) or {}).get("rot") or 0) % 360
        proposal_rot = int(row.get("rot") or 0) % 360
        if proposal_rot != seed_rot:
            return True

    anchor_ids_by_cluster = _anchor_ids_by_cluster(payload)
    for repair in proposal.get("object_repairs") or []:
        if not isinstance(repair, dict):
            continue
        cluster_id = str(repair.get("cluster_id") or "").strip()
        if cluster_id not in protected_cluster_ids:
            continue
        op = str(repair.get("op") or "").strip()
        object_id = str(repair.get("object_id") or "").strip()
        anchor_ids = anchor_ids_by_cluster.get(cluster_id, set())
        if op == "set_anchor":
            return True
        if op in {"rotate_object", "mirror_object", "set_front_override"}:
            if object_id and object_id in anchor_ids:
                return True
        if op == "swap_objects":
            other_object_id = str(
                (repair.get("params") or {}).get("other_object_id") or ""
            ).strip()
            if object_id in anchor_ids or other_object_id in anchor_ids:
                return True
    return False


def _postsolve_orientation_snap(
    *,
    payload: dict[str, Any],
    repair: dict[str, Any],
    rounds: int = 2,
    move_limit: int = 48,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    promoted_payload = PromotePhase2RepairToSeedPayload(payload=payload, repair=repair)
    baseline_repair = _canonical_repair(promoted_payload)
    baseline_evaluation = EvaluatePhase2Proposal(
        payload=promoted_payload,
        repair=baseline_repair,
    )

    if not bool(baseline_evaluation.get("hard_valid")):
        return payload, repair, baseline_evaluation

    current_payload = promoted_payload
    current_repair = baseline_repair
    current_evaluation = baseline_evaluation

    for _ in range(max(1, int(rounds))):
        macro_moves = (
            EnumeratePhase2RepairMoves(
                payload=current_payload,
                limit=move_limit,
            ).get("moves")
            or []
        )
        object_payload = deepcopy(current_payload)
        _set_search_phase(object_payload, "object_refine")
        object_moves = (
            EnumeratePhase2RepairMoves(
                payload=object_payload,
                limit=move_limit,
            ).get("moves")
            or []
        )

        best_macro_candidate: tuple[dict[str, Any], dict[str, Any]] | None = None
        best_macro_delta = 0
        best_object_candidate: tuple[dict[str, Any], dict[str, Any]] | None = None
        best_object_delta = 0
        seen_signatures: set[str] = set()
        orientation_moves_by_cluster: dict[str, list[dict[str, Any]]] = {}

        for move in object_moves:
            if not _is_object_orientation_snap_move(move):
                continue
            proposal = _canonical_repair(
                current_payload,
                move.get("proposal") or {},
                notes=[
                    "Post-solve object orientation snap: "
                    f"{move.get('reason') or 'object orientation move'}"
                ],
            )
            signature = _proposal_signature(proposal)
            if signature in seen_signatures:
                continue
            if _proposal_breaks_anchor_contract(current_payload, proposal):
                continue
            seen_signatures.add(signature)
            evaluation = EvaluatePhase2Proposal(
                payload=current_payload,
                repair=proposal,
            )
            if not bool(evaluation.get("hard_valid")):
                continue
            delta_score = _delta_score(evaluation)
            if delta_score > best_object_delta:
                best_object_delta = delta_score
                best_object_candidate = (proposal, evaluation)

        for move in macro_moves:
            if not _is_orientation_snap_move(current_payload, move):
                continue
            cluster_id = str(move.get("cluster_id") or "")
            if cluster_id:
                orientation_moves_by_cluster.setdefault(cluster_id, []).append(move)
            proposal = _canonical_repair(
                current_payload,
                move.get("proposal") or {},
                notes=[
                    f"Post-solve orientation snap: {move.get('reason') or 'orientation move'}"
                ],
            )
            signature = _proposal_signature(proposal)
            if signature in seen_signatures:
                continue
            if _proposal_breaks_anchor_contract(current_payload, proposal):
                continue
            seen_signatures.add(signature)
            evaluation = EvaluatePhase2Proposal(
                payload=current_payload, repair=proposal
            )
            if not bool(evaluation.get("hard_valid")):
                continue
            delta_score = _delta_score(evaluation)
            if delta_score > best_macro_delta:
                best_macro_delta = delta_score
                best_macro_candidate = (proposal, evaluation)

        for left_cluster_id, right_cluster_id in _viewing_cluster_pairs(
            current_payload
        ):
            left_moves = orientation_moves_by_cluster.get(left_cluster_id) or []
            right_moves = orientation_moves_by_cluster.get(right_cluster_id) or []
            for left_move in left_moves[:4]:
                for right_move in right_moves[:4]:
                    pair_partial = {
                        "cluster_transforms": deepcopy(
                            (
                                (left_move.get("proposal") or {}).get(
                                    "cluster_transforms"
                                )
                                or []
                            )
                        )
                        + deepcopy(
                            (
                                (right_move.get("proposal") or {}).get(
                                    "cluster_transforms"
                                )
                                or []
                            )
                        ),
                        "selected_variants": deepcopy(
                            (
                                (left_move.get("proposal") or {}).get(
                                    "selected_variants"
                                )
                                or []
                            )
                        )
                        + deepcopy(
                            (
                                (right_move.get("proposal") or {}).get(
                                    "selected_variants"
                                )
                                or []
                            )
                        ),
                        "object_repairs": [],
                    }
                    proposal = _canonical_repair(
                        current_payload,
                        pair_partial,
                        notes=[
                            "Post-solve orientation snap: paired focal reorientation for "
                            f"{left_cluster_id} and {right_cluster_id}"
                        ],
                    )
                    signature = _proposal_signature(proposal)
                    if signature in seen_signatures:
                        continue
                    if _proposal_breaks_anchor_contract(current_payload, proposal):
                        continue
                    seen_signatures.add(signature)
                    evaluation = EvaluatePhase2Proposal(
                        payload=current_payload,
                        repair=proposal,
                    )
                    if not bool(evaluation.get("hard_valid")):
                        continue
                    delta_score = _delta_score(evaluation)
                    if delta_score > best_macro_delta:
                        best_macro_delta = delta_score
                        best_macro_candidate = (proposal, evaluation)

        chosen_candidate: tuple[dict[str, Any], dict[str, Any]] | None = None
        chosen_delta = 0
        if best_object_candidate is not None and best_object_delta > 0:
            chosen_candidate = best_object_candidate
            chosen_delta = best_object_delta
        elif best_macro_candidate is not None and best_macro_delta > 0:
            chosen_candidate = best_macro_candidate
            chosen_delta = best_macro_delta

        if chosen_candidate is None or chosen_delta <= 0:
            break

        candidate_repair, candidate_evaluation = chosen_candidate
        current_payload = PromotePhase2RepairToSeedPayload(
            payload=current_payload,
            repair=candidate_repair,
        )
        current_repair = _canonical_repair(current_payload)
        current_evaluation = EvaluatePhase2Proposal(
            payload=current_payload,
            repair=current_repair,
        )
        if not bool(current_evaluation.get("hard_valid")):
            break

    return current_payload, current_repair, current_evaluation


def build_phase2_preview_candidate(
    *,
    payload: dict[str, Any],
    rounds: int = 1,
    move_limit: int = 24,
    note: str = "Fast preview from solver seed.",
) -> dict[str, Any]:
    preview_payload = deepcopy(payload)
    seed_verify = (preview_payload.get("repair_debug") or {}).get("seed_verify") or {}
    if isinstance(seed_verify, dict) and _needs_primary_pair_axis_repair(
        preview_payload, seed_verify
    ):
        _set_search_phase(preview_payload, "object_refine")
    baseline_repair = make_no_improvement_repair(preview_payload, note=note)
    snapped_payload, snapped_repair, snapped_eval = _postsolve_orientation_snap(
        payload=preview_payload,
        repair=baseline_repair,
        rounds=rounds,
        move_limit=move_limit,
    )
    compilable_payload = deepcopy(snapped_payload)
    compilable_payload["phase2_placer"] = {"phase2_repair": snapped_repair}
    absolute_layout = compile_phase2_to_final_output(compilable_payload)
    return {
        "payload": snapped_payload,
        "proposal": snapped_repair,
        "evaluation": snapped_eval,
        "absolute_layout": absolute_layout,
    }


@dataclass(frozen=True)
class Phase2Controller:
    placer: MacroClusterPlacer = field(default_factory=MacroClusterPlacer)
    judge: Phase2Judge = field(default_factory=Phase2Judge)
    max_iterations: int = 20

    def generate(
        self,
        *,
        room_interpreter_json: dict[str, Any],
        cluster_merged_json: dict[str, Any],
        cluster_outlines_json: dict[str, Any],
        relation_plan_json: dict[str, Any],
        solver_output_json: dict[str, Any],
        cluster_constraints_json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_solver_output = _normalize_solver_output_for_phase2(
            solver_output_json
        )
        current_payload = build_phase2_payload(
            room_01=room_interpreter_json,
            clusters_04=cluster_merged_json,
            outlines_05=cluster_outlines_json,
            relation_05b=relation_plan_json,
            placer_06=normalized_solver_output,
            cluster_constraints=cluster_constraints_json,
        )

        result = MicroRefiner().refine(current_payload).as_controller_result()
        absolute_layout = result.get("absolute_layout") or {}
        if str(absolute_layout.get("status") or "").upper() == "OK":
            return result

        # Fallback path: if the solver produced a hard-valid seed that is still
        # macro-blocked or locally invalid after materialization, run the
        # deterministic phase-2 placer and explicitly try an object-refine pass
        # when the seed diagnostics say the primary viewing pair is wrong.
        fallback_notes = [
            "Phase-2 fallback activated: deterministic repair from solver seed.",
        ]
        try:
            seed_verify = (
                (normalized_solver_output.get("placer_seed") or {}).get("seed_verify")
                if isinstance(normalized_solver_output, dict)
                else None
            )
            candidate_payloads: list[tuple[str, dict[str, Any], list[str]]] = [
                ("macro_layout", deepcopy(current_payload), list(fallback_notes))
            ]
            if isinstance(seed_verify, dict) and _needs_primary_pair_axis_repair(
                current_payload,
                seed_verify,
            ):
                object_payload = deepcopy(current_payload)
                _set_search_phase(object_payload, "object_refine")
                candidate_payloads.insert(
                    0,
                    (
                        "object_refine",
                        object_payload,
                        [
                            *fallback_notes,
                            "Primary viewing-pair/object-orientation pressure detected; object-refine fallback promoted before generic macro repair.",
                        ],
                    ),
                )

            best_candidate: (
                tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[str], str]
                | None
            ) = None
            best_rank: tuple[int, int, int, int, int] | None = None

            for mode, candidate_payload, candidate_notes in candidate_payloads:
                fallback_raw = self.placer.generate_from_payload(
                    payload=candidate_payload,
                    max_attempts=3,
                )
                fallback_proposal = _canonical_repair(
                    candidate_payload,
                    fallback_raw if isinstance(fallback_raw, dict) else {},
                    notes=candidate_notes,
                )
                fallback_eval = EvaluatePhase2Proposal(
                    payload=candidate_payload,
                    repair=fallback_proposal,
                )
                if not bool(fallback_eval.get("hard_valid")):
                    continue
                snapped_payload, snapped_repair, snapped_eval = (
                    _postsolve_orientation_snap(
                        payload=candidate_payload,
                        repair=fallback_proposal,
                        rounds=2,
                        move_limit=64 if mode == "object_refine" else 48,
                    )
                )
                final_payload = deepcopy(snapped_payload)
                final_payload["phase2_placer"] = {"phase2_repair": snapped_repair}
                final_absolute_layout = compile_phase2_to_final_output(final_payload)
                rank = _search_candidate_rank(snapped_eval)
                if str(final_absolute_layout.get("status") or "").upper() == "OK":
                    rank = (rank[0] + 1, *rank[1:])
                if best_candidate is None or rank > (best_rank or (0, 0, 0, 0, 0)):
                    best_candidate = (
                        snapped_repair,
                        snapped_eval,
                        final_absolute_layout,
                        candidate_notes,
                        mode,
                    )
                    best_rank = rank

            if (
                best_candidate is not None
                and str(best_candidate[2].get("status") or "").upper() == "OK"
            ):
                (
                    snapped_repair,
                    snapped_eval,
                    final_absolute_layout,
                    candidate_notes,
                    mode,
                ) = best_candidate
                return _phase2_controller_result(
                    proposal=snapped_repair,
                    evaluation=snapped_eval,
                    absolute_layout=final_absolute_layout,
                    history=[
                        {
                            "stage": "micro_refiner",
                            "status": "invalid_seed",
                            "absolute_layout_status": str(
                                absolute_layout.get("status") or "UNKNOWN"
                            ),
                        },
                        {
                            "stage": "deterministic_phase2_fallback",
                            "mode": mode,
                            "status": "accepted",
                            "proposal_status": str(
                                snapped_repair.get("status") or "REPAIRED"
                            ),
                        },
                    ],
                    hard_fix={
                        "result": "APPLIED",
                        "attempts": [
                            {
                                "mode": mode,
                                "hard_valid": True,
                                "notes": candidate_notes,
                            }
                        ],
                    },
                    notes=candidate_notes,
                )
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.exception("Phase-2 deterministic fallback failed: %s", exc)

        return result
