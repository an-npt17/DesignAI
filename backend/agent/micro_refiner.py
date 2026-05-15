from __future__ import annotations

import json
import math
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal

from cluster_placer.tools_v2 import (
    CompileAcceptedPhase2Proposal,
    EnumeratePhase2RepairMoves,
    EvaluatePhase2Proposal,
    PromotePhase2RepairToSeedPayload,
    materialize_phase2_state,
)

MoveType = Literal[
    "small_translate",
    "wall_slide",
    "axis_align",
    "spacing_equalize",
    "pair_distance_adjust",
    "symmetry_rebalance",
    "focal_align",
    "circulation_relief",
    "cluster_compactness_adjust",
    "opening_clearance_relief",
]
RefinerStatus = Literal["OK", "UNCHANGED", "PARTIAL_OK"]

_HEAVY_OBJECT_KEYWORDS = (
    "sofa",
    "sectional",
    "bed",
    "wardrobe",
    "closet",
    "cabinet",
    "dresser",
)
_FLEX_OBJECT_KEYWORDS = (
    "table",
    "coffee",
    "chair",
    "armchair",
    "stool",
    "ottoman",
    "nightstand",
    "side",
    "lamp",
)
_WALL_SLIDE_OBJECT_KEYWORDS = ("desk", "console", "tv", "media", "shelf")
_ANCHOR_OBJECT_KEYWORDS = ("sofa", "bed", "wardrobe", "dining_table", "desk")


@dataclass(frozen=True)
class MicroRefinePolicy:
    grid_mm: int = 25
    max_iterations: int = 40
    max_candidate_moves_per_iteration: int = 24
    max_objects_touched_per_pass: int = 6
    max_translation_step_mm: int = 100
    max_wall_slide_step_mm: int = 150
    max_pair_adjust_step_mm: int = 100
    max_rotation_options: int = 2
    tabu_length: int = 12
    min_score_improvement: float = 0.003

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> MicroRefinePolicy:
        raw = payload.get("micro_refine_policy")
        if not isinstance(raw, Mapping):
            return cls()

        def _int_value(key: str, fallback: int) -> int:
            value = raw.get(key)
            try:
                return int(value)
            except (TypeError, ValueError):
                return fallback

        def _float_value(key: str, fallback: float) -> float:
            value = raw.get(key)
            try:
                return float(value)
            except (TypeError, ValueError):
                return fallback

        return cls(
            grid_mm=_int_value("grid_mm", cls.grid_mm),
            max_iterations=_int_value("max_iterations", cls.max_iterations),
            max_candidate_moves_per_iteration=_int_value(
                "max_moves_per_iteration",
                _int_value(
                    "max_candidate_moves_per_iteration",
                    cls.max_candidate_moves_per_iteration,
                ),
            ),
            max_objects_touched_per_pass=_int_value(
                "max_objects_touched_per_pass",
                cls.max_objects_touched_per_pass,
            ),
            max_translation_step_mm=_int_value(
                "max_translation_step_mm",
                cls.max_translation_step_mm,
            ),
            max_wall_slide_step_mm=_int_value(
                "max_wall_slide_step_mm",
                cls.max_wall_slide_step_mm,
            ),
            max_pair_adjust_step_mm=_int_value(
                "max_pair_adjust_step_mm",
                cls.max_pair_adjust_step_mm,
            ),
            max_rotation_options=_int_value(
                "max_rotation_options",
                cls.max_rotation_options,
            ),
            tabu_length=_int_value("tabu_length", cls.tabu_length),
            min_score_improvement=_float_value(
                "min_score_improvement",
                cls.min_score_improvement,
            ),
        )


@dataclass(frozen=True)
class ObjectKey:
    cluster_id: str
    object_id: str

    def label(self) -> str:
        return f"{self.cluster_id}.{self.object_id}"


@dataclass(frozen=True)
class MicroMove:
    move_type: MoveType
    repairs: tuple[dict[str, Any], ...]
    reason: str
    touched_objects: frozenset[ObjectKey]
    priority: int = 0

    def target_label(self) -> str:
        if len(self.touched_objects) == 1:
            return next(iter(self.touched_objects)).object_id
        return ", ".join(sorted(key.object_id for key in self.touched_objects))

    def tabu_key(self) -> str:
        return json.dumps(
            {
                "type": self.move_type,
                "targets": sorted(key.label() for key in self.touched_objects),
                "repairs": self.repairs,
            },
            ensure_ascii=True,
            sort_keys=True,
        )


@dataclass(frozen=True)
class QualityBreakdown:
    functionality: float
    naturalness: float
    semantic: float
    spatial: float
    total: float

    def as_dict(self) -> dict[str, float]:
        return {
            "functionality": round(self.functionality, 3),
            "naturalness": round(self.naturalness, 3),
            "semantic": round(self.semantic, 3),
            "spatial": round(self.spatial, 3),
            "total": round(self.total, 3),
        }


@dataclass
class AppliedMove:
    move_type: MoveType
    target: str
    delta_mm: tuple[int, int] | None
    reason: str
    score_gain: float
    touched_objects: frozenset[ObjectKey]

    def as_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "type": self.move_type,
            "target": self.target,
            "reason": self.reason,
            "score_gain": round(self.score_gain, 4),
        }
        if self.delta_mm is not None:
            out["delta_mm"] = [self.delta_mm[0], self.delta_mm[1]]
        return out


@dataclass
class MicroRefineResult:
    status: RefinerStatus
    payload: dict[str, Any]
    repair: dict[str, Any]
    evaluation: dict[str, Any]
    absolute_layout: dict[str, Any]
    refined_layout_solution: dict[str, Any]
    history: list[dict[str, Any]] = field(default_factory=list)

    def as_controller_result(self) -> dict[str, Any]:
        return {
            "proposal": self.repair,
            "tool_evaluation": self.evaluation,
            "judge_evaluation": {
                "verdict": "ACCEPT" if self.status in {"OK", "UNCHANGED"} else "REVISE",
                "reasonableness_score": int(
                    round(
                        float(
                            (
                                self.refined_layout_solution.get("quality_after") or {}
                            ).get("total")
                            or 0.0
                        )
                        * 100.0
                    )
                ),
                "next_step_mode": "stop",
                "top_issues": [],
                "repair_advice": [],
                "priority_clusters": [],
            },
            "absolute_layout": self.absolute_layout,
            "refined_layout_solution": self.refined_layout_solution,
            "history": self.history,
            "hard_fix": {"result": "SKIPPED", "attempts": []},
        }


class MicroRefiner:
    def refine(self, payload: dict[str, Any]) -> MicroRefineResult:
        policy = MicroRefinePolicy.from_payload(payload)
        original_payload = deepcopy(payload)
        current_payload = deepcopy(payload)
        current_repair = _base_repair(current_payload)
        current_eval = EvaluatePhase2Proposal(
            payload=current_payload,
            repair=current_repair,
        )
        before_quality = _quality_from_evaluation(current_eval, movement_penalty=0.0)

        if not bool(current_eval.get("hard_valid")):
            return self._finish(
                status="PARTIAL_OK",
                original_payload=original_payload,
                current_payload=current_payload,
                current_repair=current_repair,
                current_eval=current_eval,
                quality_before=before_quality,
                quality_after=before_quality,
                applied_moves=[],
                history=[],
                notes=["Micro refiner skipped because the macro seed is hard-invalid."],
            )

        tabu: deque[str] = deque(maxlen=max(1, policy.tabu_length))
        applied_moves: list[AppliedMove] = []
        history: list[dict[str, Any]] = []
        touched: set[ObjectKey] = set()

        for iteration in range(1, max(1, policy.max_iterations) + 1):
            current_quality = _quality_from_evaluation(
                current_eval,
                movement_penalty=_movement_penalty(original_payload, current_payload),
            )
            candidates = _candidate_moves(
                payload=current_payload,
                evaluation=current_eval,
                policy=policy,
                already_touched=touched,
            )
            best: tuple[MicroMove, dict[str, Any], QualityBreakdown, float] | None = (
                None
            )

            for move in candidates:
                if move.tabu_key() in tabu:
                    continue
                if (
                    len(touched | set(move.touched_objects))
                    > policy.max_objects_touched_per_pass
                ):
                    continue
                proposal = _proposal_from_move(current_payload, move)
                evaluation = EvaluatePhase2Proposal(
                    payload=current_payload,
                    repair=proposal,
                )
                if not bool(evaluation.get("hard_valid")):
                    continue
                quality = _quality_from_evaluation(
                    evaluation,
                    movement_penalty=_movement_penalty_after_move(
                        original_payload,
                        current_payload,
                        proposal,
                    ),
                )
                gain = quality.total - current_quality.total
                if gain < policy.min_score_improvement:
                    continue
                if best is None or _candidate_rank(
                    move, quality, gain
                ) > _candidate_rank(
                    best[0],
                    best[2],
                    best[3],
                ):
                    best = (move, evaluation, quality, gain)

            if best is None:
                history.append(
                    {
                        "iteration": iteration,
                        "status": "converged",
                        "candidate_count": len(candidates),
                        "quality": current_quality.as_dict(),
                    }
                )
                break

            move, move_eval, next_quality, gain = best
            proposal = _proposal_from_move(current_payload, move)
            current_payload = PromotePhase2RepairToSeedPayload(
                payload=current_payload,
                repair=proposal,
            )
            current_repair = _base_repair(
                current_payload,
                notes=[f"Micro refiner accepted: {move.reason}"],
            )
            current_eval = EvaluatePhase2Proposal(
                payload=current_payload,
                repair=current_repair,
            )
            touched.update(move.touched_objects)
            tabu.append(_reverse_tabu_key(move))
            applied_moves.append(
                AppliedMove(
                    move_type=move.move_type,
                    target=move.target_label(),
                    delta_mm=_first_delta(move),
                    reason=move.reason,
                    score_gain=gain,
                    touched_objects=move.touched_objects,
                )
            )
            history.append(
                {
                    "iteration": iteration,
                    "status": "accepted",
                    "move_type": move.move_type,
                    "target": move.target_label(),
                    "hard_valid": bool(move_eval.get("hard_valid")),
                    "quality": next_quality.as_dict(),
                    "gain": round(gain, 4),
                }
            )

        after_quality = _quality_from_evaluation(
            current_eval,
            movement_penalty=_movement_penalty(original_payload, current_payload),
        )
        status: RefinerStatus = "OK" if applied_moves else "UNCHANGED"
        return self._finish(
            status=status,
            original_payload=original_payload,
            current_payload=current_payload,
            current_repair=current_repair,
            current_eval=current_eval,
            quality_before=before_quality,
            quality_after=after_quality,
            applied_moves=applied_moves,
            history=history,
            notes=[],
        )

    def _finish(
        self,
        *,
        status: RefinerStatus,
        original_payload: dict[str, Any],
        current_payload: dict[str, Any],
        current_repair: dict[str, Any],
        current_eval: dict[str, Any],
        quality_before: QualityBreakdown,
        quality_after: QualityBreakdown,
        applied_moves: Sequence[AppliedMove],
        history: list[dict[str, Any]],
        notes: list[str],
    ) -> MicroRefineResult:
        final_payload = deepcopy(current_payload)
        final_payload["phase2_placer"] = {"phase2_repair": deepcopy(current_repair)}
        absolute_layout = CompileAcceptedPhase2Proposal(
            payload=final_payload,
            repair=current_repair,
        )
        refined_layout_solution = _refined_layout_solution(
            status=status,
            original_payload=original_payload,
            current_payload=current_payload,
            repair=current_repair,
            evaluation=current_eval,
            quality_before=quality_before,
            quality_after=quality_after,
            applied_moves=applied_moves,
            notes=notes,
        )
        return MicroRefineResult(
            status=status,
            payload=final_payload,
            repair=current_repair,
            evaluation=current_eval,
            absolute_layout=absolute_layout,
            refined_layout_solution=refined_layout_solution,
            history=history,
        )


def _base_repair(
    payload: Mapping[str, object], notes: Sequence[str] | None = None
) -> dict[str, Any]:
    seed = payload.get("seed_layout") if isinstance(payload, Mapping) else {}
    if not isinstance(seed, Mapping):
        seed = {}
    return {
        "status": "NO_IMPROVEMENT",
        "cluster_transforms": deepcopy(seed.get("cluster_transforms") or []),
        "selected_variants": deepcopy(seed.get("selected_variants") or []),
        "object_repairs": [],
        "notes": [text for text in (notes or []) if text],
    }


def _proposal_from_move(
    payload: Mapping[str, object], move: MicroMove
) -> dict[str, Any]:
    repair = _base_repair(payload, notes=[move.reason])
    repair["status"] = "REPAIRED"
    repair["object_repairs"] = [deepcopy(item) for item in move.repairs]
    return repair


def _candidate_moves(
    *,
    payload: dict[str, Any],
    evaluation: dict[str, Any],
    policy: MicroRefinePolicy,
    already_touched: set[ObjectKey],
) -> list[MicroMove]:
    state = materialize_phase2_state(payload, repair=None)
    objects = [
        row
        for row in (state.get("objects") or [])
        if isinstance(row, Mapping)
        and isinstance(row.get("cluster_id"), str)
        and isinstance(row.get("object_id"), str)
    ]
    object_by_key = {
        ObjectKey(str(row["cluster_id"]), str(row["object_id"])): row for row in objects
    }
    moves: list[MicroMove] = []
    moves.extend(
        _existing_semantic_object_moves(
            payload=payload,
            limit=policy.max_candidate_moves_per_iteration,
        )
    )
    moves.extend(
        _clearance_moves(
            evaluation=evaluation,
            object_by_key=object_by_key,
            policy=policy,
        )
    )
    moves.extend(
        _path_relief_moves(
            payload=payload,
            evaluation=evaluation,
            object_by_key=object_by_key,
            policy=policy,
        )
    )
    moves.extend(
        _opening_relief_moves(
            payload=payload,
            evaluation=evaluation,
            object_by_key=object_by_key,
            policy=policy,
        )
    )
    moves.extend(
        _wall_slide_moves(
            object_by_key=object_by_key,
            policy=policy,
        )
    )
    moves.extend(
        _axis_align_moves(
            object_by_key=object_by_key,
            policy=policy,
        )
    )
    moves.extend(
        _pair_template_moves(
            object_by_key=object_by_key,
            policy=policy,
        )
    )
    moves.extend(
        _cluster_compactness_moves(
            object_by_key=object_by_key,
            policy=policy,
        )
    )

    deduped: list[MicroMove] = []
    seen: set[str] = set()
    for move in moves:
        if move.touched_objects and move.touched_objects.issubset(already_touched):
            continue
        key = move.tabu_key()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(move)
    deduped.sort(
        key=lambda move: (
            -move.priority,
            len(move.touched_objects),
            move.move_type,
            move.target_label(),
            move.tabu_key(),
        )
    )
    return deduped[: max(1, policy.max_candidate_moves_per_iteration)]


def _existing_semantic_object_moves(
    *, payload: dict[str, Any], limit: int
) -> list[MicroMove]:
    object_payload = deepcopy(payload)
    phase_control = deepcopy(object_payload.get("phase_control") or {})
    phase_control["repair_phase"] = "object_refine"
    object_payload["phase_control"] = phase_control
    raw_moves = (
        EnumeratePhase2RepairMoves(
            payload=object_payload,
            limit=max(limit, 32),
        ).get("moves")
        or []
    )
    moves: list[MicroMove] = []
    for raw in raw_moves:
        if not isinstance(raw, Mapping):
            continue
        repairs = [
            deepcopy(row)
            for row in ((raw.get("proposal") or {}).get("object_repairs") or [])
            if isinstance(row, Mapping)
        ]
        if not repairs:
            continue
        touched = _repair_touched_objects(repairs)
        move_type = _move_type_from_repairs(repairs)
        moves.append(
            MicroMove(
                move_type=move_type,
                repairs=tuple(repairs),
                reason=str(raw.get("reason") or f"{move_type} refinement"),
                touched_objects=frozenset(touched),
                priority=80,
            )
        )
    return moves


def _clearance_moves(
    *,
    evaluation: Mapping[str, object],
    object_by_key: Mapping[ObjectKey, Mapping[str, object]],
    policy: MicroRefinePolicy,
) -> list[MicroMove]:
    metrics = evaluation.get("metrics")
    if not isinstance(metrics, Mapping):
        return []
    moves: list[MicroMove] = []
    for row in metrics.get("object_front_clearance") or []:
        if not isinstance(row, Mapping):
            continue
        shortage = int(row.get("shortage_mm") or 0)
        if shortage <= 0:
            continue
        key = ObjectKey(
            str(row.get("cluster_id") or ""), str(row.get("object_id") or "")
        )
        obj = object_by_key.get(key)
        if not isinstance(obj, Mapping):
            continue
        front = _vec(obj.get("front_world"))
        if front is None:
            continue
        step = min(policy.max_translation_step_mm, max(policy.grid_mm, shortage))
        local_dx, local_dy = _world_to_local_offset(
            obj, -front[0] * step, -front[1] * step
        )
        moves.append(
            _nudge_move(
                move_type="small_translate",
                key=key,
                dx=local_dx,
                dy=local_dy,
                reason=f"Improve front clearance for {key.label()}",
                priority=95 + min(shortage // 50, 20),
            )
        )
    return moves


def _path_relief_moves(
    *,
    payload: Mapping[str, object],
    evaluation: Mapping[str, object],
    object_by_key: Mapping[ObjectKey, Mapping[str, object]],
    policy: MicroRefinePolicy,
) -> list[MicroMove]:
    metrics = evaluation.get("metrics")
    if not isinstance(metrics, Mapping):
        return []
    path = metrics.get("main_path_clearance")
    if not isinstance(path, Mapping):
        return []
    room_center = _room_center(payload)
    moves: list[MicroMove] = []
    for row in path.get("blocking_objects") or []:
        if not isinstance(row, Mapping):
            continue
        key = ObjectKey(
            str(row.get("cluster_id") or ""), str(row.get("object_id") or "")
        )
        obj = object_by_key.get(key)
        if not isinstance(obj, Mapping):
            continue
        center = _center(obj)
        direction = _unit(center[0] - room_center[0], center[1] - room_center[1]) or (
            1.0,
            0.0,
        )
        step = min(
            policy.max_translation_step_mm,
            max(policy.grid_mm, int(row.get("penalty_mm") or 0)),
        )
        local_dx, local_dy = _world_to_local_offset(
            obj, direction[0] * step, direction[1] * step
        )
        moves.append(
            _nudge_move(
                move_type="circulation_relief",
                key=key,
                dx=local_dx,
                dy=local_dy,
                reason=f"Relieve main-path squeeze near {key.label()}",
                priority=110,
            )
        )
    return moves


def _opening_relief_moves(
    *,
    payload: Mapping[str, object],
    evaluation: Mapping[str, object],
    object_by_key: Mapping[ObjectKey, Mapping[str, object]],
    policy: MicroRefinePolicy,
) -> list[MicroMove]:
    metrics = evaluation.get("metrics")
    if not isinstance(metrics, Mapping):
        return []
    room_center = _room_center(payload)
    moves: list[MicroMove] = []
    for row in metrics.get("opening_band_blocking") or []:
        if not isinstance(row, Mapping):
            continue
        cluster_id = str(row.get("cluster_id") or "")
        cluster_objects = [
            (key, obj)
            for key, obj in object_by_key.items()
            if key.cluster_id == cluster_id and _move_freedom(key.object_id) != "low"
        ]
        for key, obj in cluster_objects[:2]:
            center = _center(obj)
            direction = _unit(
                center[0] - room_center[0], center[1] - room_center[1]
            ) or (1.0, 0.0)
            step = min(
                policy.max_translation_step_mm,
                max(policy.grid_mm, int(row.get("penalty_mm") or 0)),
            )
            local_dx, local_dy = _world_to_local_offset(
                obj, direction[0] * step, direction[1] * step
            )
            moves.append(
                _nudge_move(
                    move_type="opening_clearance_relief",
                    key=key,
                    dx=local_dx,
                    dy=local_dy,
                    reason=f"Ease opening clearance around {cluster_id}",
                    priority=100,
                )
            )
    return moves


def _wall_slide_moves(
    *,
    object_by_key: Mapping[ObjectKey, Mapping[str, object]],
    policy: MicroRefinePolicy,
) -> list[MicroMove]:
    moves: list[MicroMove] = []
    for key, obj in object_by_key.items():
        if not _has_keyword(key.object_id, _WALL_SLIDE_OBJECT_KEYWORDS):
            continue
        rect = obj.get("local_rect")
        if not isinstance(rect, Mapping):
            continue
        width = int(rect.get("w") or 0)
        height = int(rect.get("h") or 0)
        step = min(policy.max_wall_slide_step_mm, max(policy.grid_mm, 50))
        offsets = (
            ((step, 0), (-step, 0)) if width >= height else ((0, step), (0, -step))
        )
        for dx, dy in offsets:
            moves.append(
                _nudge_move(
                    move_type="wall_slide",
                    key=key,
                    dx=dx,
                    dy=dy,
                    reason=f"Slide {key.label()} along its local wall axis",
                    priority=76,
                )
            )
    return moves


def _axis_align_moves(
    *,
    object_by_key: Mapping[ObjectKey, Mapping[str, object]],
    policy: MicroRefinePolicy,
) -> list[MicroMove]:
    moves: list[MicroMove] = []
    directions = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))
    for key, obj in object_by_key.items():
        front = _vec(obj.get("front_world"))
        if front is None:
            continue
        ranked = sorted(
            directions,
            key=lambda direction: -(front[0] * direction[0] + front[1] * direction[1]),
        )
        for direction in ranked[: max(1, policy.max_rotation_options)]:
            local_dx, local_dy = _world_to_local_offset(obj, direction[0], direction[1])
            if (local_dx, local_dy) == (0, 0):
                continue
            moves.append(
                MicroMove(
                    move_type="axis_align",
                    repairs=(
                        {
                            "cluster_id": key.cluster_id,
                            "object_id": key.object_id,
                            "op": "set_front_override",
                            "params": {"dx": local_dx, "dy": local_dy},
                        },
                    ),
                    reason=f"Snap {key.label()} front direction to a room axis",
                    touched_objects=frozenset({key}),
                    priority=72,
                )
            )
    return moves


def _pair_template_moves(
    *,
    object_by_key: Mapping[ObjectKey, Mapping[str, object]],
    policy: MicroRefinePolicy,
) -> list[MicroMove]:
    by_cluster: dict[str, list[tuple[ObjectKey, Mapping[str, object]]]] = {}
    for key, obj in object_by_key.items():
        by_cluster.setdefault(key.cluster_id, []).append((key, obj))

    moves: list[MicroMove] = []
    for rows in by_cluster.values():
        anchors = [
            (key, obj)
            for key, obj in rows
            if _has_keyword(key.object_id, _ANCHOR_OBJECT_KEYWORDS)
        ]
        flex = [
            (key, obj) for key, obj in rows if _move_freedom(key.object_id) == "high"
        ]
        for anchor_key, anchor in anchors[:2]:
            anchor_center = _center(anchor)
            for flex_key, flex_obj in flex[:4]:
                if flex_key == anchor_key:
                    continue
                vector = _unit(
                    _center(flex_obj)[0] - anchor_center[0],
                    _center(flex_obj)[1] - anchor_center[1],
                )
                if vector is None:
                    continue
                current_dist = _distance(_center(flex_obj), anchor_center)
                target_dist = _target_pair_distance(
                    anchor_key.object_id, flex_key.object_id
                )
                if abs(current_dist - target_dist) < policy.grid_mm:
                    continue
                sign = -1.0 if current_dist > target_dist else 1.0
                step = min(
                    policy.max_pair_adjust_step_mm, abs(current_dist - target_dist)
                )
                world_dx = vector[0] * step * sign
                world_dy = vector[1] * step * sign
                local_dx, local_dy = _world_to_local_offset(
                    flex_obj, world_dx, world_dy
                )
                moves.append(
                    _nudge_move(
                        move_type="pair_distance_adjust",
                        key=flex_key,
                        dx=local_dx,
                        dy=local_dy,
                        reason=f"Rebalance {flex_key.object_id} distance to {anchor_key.object_id}",
                        priority=90,
                    )
                )
        nightstands = [
            (key, obj)
            for key, obj in rows
            if "nightstand" in key.object_id.lower()
            or "bedside" in key.object_id.lower()
        ]
        beds = [(key, obj) for key, obj in rows if "bed" in key.object_id.lower()]
        if len(nightstands) >= 2 and beds:
            bed_center = _center(beds[0][1])
            left, right = nightstands[:2]
            left_dist = _distance(_center(left[1]), bed_center)
            right_dist = _distance(_center(right[1]), bed_center)
            if abs(left_dist - right_dist) >= policy.grid_mm:
                target, other = (
                    (left, right) if left_dist > right_dist else (right, left)
                )
                target_center = _center(target[1])
                other_center = _center(other[1])
                direction = _unit(
                    other_center[0] - target_center[0],
                    other_center[1] - target_center[1],
                )
                if direction is not None:
                    step = min(
                        policy.max_pair_adjust_step_mm,
                        abs(left_dist - right_dist) / 2.0,
                    )
                    local_dx, local_dy = _world_to_local_offset(
                        target[1], direction[0] * step, direction[1] * step
                    )
                    moves.append(
                        _nudge_move(
                            move_type="symmetry_rebalance",
                            key=target[0],
                            dx=local_dx,
                            dy=local_dy,
                            reason="Rebalance bedside pair symmetry",
                            priority=92,
                        )
                    )
    return moves


def _cluster_compactness_moves(
    *,
    object_by_key: Mapping[ObjectKey, Mapping[str, object]],
    policy: MicroRefinePolicy,
) -> list[MicroMove]:
    by_cluster: dict[str, list[tuple[ObjectKey, Mapping[str, object]]]] = {}
    for key, obj in object_by_key.items():
        by_cluster.setdefault(key.cluster_id, []).append((key, obj))
    moves: list[MicroMove] = []
    for rows in by_cluster.values():
        if len(rows) < 3:
            continue
        cx = sum(_center(obj)[0] for _, obj in rows) / len(rows)
        cy = sum(_center(obj)[1] for _, obj in rows) / len(rows)
        for key, obj in rows:
            if _move_freedom(key.object_id) == "low":
                continue
            center = _center(obj)
            dist = _distance(center, (cx, cy))
            if dist < 900:
                continue
            direction = _unit(cx - center[0], cy - center[1])
            if direction is None:
                continue
            step = min(
                policy.max_translation_step_mm, max(policy.grid_mm, int(dist * 0.08))
            )
            local_dx, local_dy = _world_to_local_offset(
                obj, direction[0] * step, direction[1] * step
            )
            moves.append(
                _nudge_move(
                    move_type="cluster_compactness_adjust",
                    key=key,
                    dx=local_dx,
                    dy=local_dy,
                    reason=f"Reduce local grouping drift for {key.label()}",
                    priority=70,
                )
            )
    return moves


def _nudge_move(
    *,
    move_type: MoveType,
    key: ObjectKey,
    dx: int,
    dy: int,
    reason: str,
    priority: int,
) -> MicroMove:
    return MicroMove(
        move_type=move_type,
        repairs=(
            {
                "cluster_id": key.cluster_id,
                "object_id": key.object_id,
                "op": "nudge_object",
                "params": {"dx": int(round(dx)), "dy": int(round(dy))},
            },
        ),
        reason=reason,
        touched_objects=frozenset({key}),
        priority=priority,
    )


def _repair_touched_objects(repairs: Iterable[Mapping[str, object]]) -> set[ObjectKey]:
    touched: set[ObjectKey] = set()
    for repair in repairs:
        cluster_id = str(repair.get("cluster_id") or "").strip()
        object_id = str(repair.get("object_id") or "").strip()
        if cluster_id and object_id:
            touched.add(ObjectKey(cluster_id, object_id))
        params = repair.get("params")
        other_object_id = (
            str(params.get("other_object_id") or "").strip()
            if isinstance(params, Mapping)
            else ""
        )
        if cluster_id and other_object_id:
            touched.add(ObjectKey(cluster_id, other_object_id))
    return touched


def _move_type_from_repairs(repairs: Sequence[Mapping[str, object]]) -> MoveType:
    ops = {str(repair.get("op") or "") for repair in repairs}
    if ops & {"rotate_object", "mirror_object", "set_front_override"}:
        return "focal_align"
    if "swap_objects" in ops:
        return "symmetry_rebalance"
    if len(repairs) > 1:
        return "pair_distance_adjust"
    return "spacing_equalize"


def _quality_from_evaluation(
    evaluation: Mapping[str, object],
    *,
    movement_penalty: float,
) -> QualityBreakdown:
    metrics = evaluation.get("metrics")
    if not isinstance(metrics, Mapping):
        return QualityBreakdown(0.0, 0.0, 0.0, 0.0, 0.0)
    score_summary = metrics.get("score_summary")
    penalties = (
        score_summary.get("penalties") if isinstance(score_summary, Mapping) else {}
    )
    if not isinstance(penalties, Mapping):
        penalties = {}
    micro_penalty = float(penalties.get("micro_penalty") or 0.0)
    macro_penalty = float(penalties.get("macro_penalty") or 0.0)

    clearance_penalty = 0.0
    for row in metrics.get("object_front_clearance") or []:
        if isinstance(row, Mapping):
            clearance_penalty += float(row.get("shortage_mm") or 0.0) * 2.4

    path_penalty = 0.0
    main_path = metrics.get("main_path_clearance")
    if isinstance(main_path, Mapping):
        for row in main_path.get("paths") or []:
            if isinstance(row, Mapping):
                path_penalty += float(row.get("clearance_shortage_mm") or 0.0) * 2.0
                path_penalty += float(row.get("blocked_samples") or 0.0) * 18.0
        for row in main_path.get("blocking_objects") or []:
            if isinstance(row, Mapping):
                path_penalty += float(row.get("penalty_mm") or 0.0)

    internal_penalty = 0.0
    for row in metrics.get("cluster_internal_constraint_fidelity") or []:
        if isinstance(row, Mapping):
            internal_penalty += float(row.get("penalty_mm") or 0.0)

    alignment_penalty = 0.0
    for row in metrics.get("orientation_debug") or []:
        if isinstance(row, Mapping):
            alignment_penalty += float(row.get("penalty_mm") or 0.0)

    spatial_penalty = path_penalty
    for key in (
        "opening_band_blocking",
        "central_congestion",
        "cluster_edge_vs_center_fit",
    ):
        for row in metrics.get(key) or []:
            if isinstance(row, Mapping):
                spatial_penalty += float(row.get("penalty_mm") or 0.0)

    functionality = _penalty_to_score(clearance_penalty + path_penalty, 1800.0)
    naturalness = _penalty_to_score(
        internal_penalty + alignment_penalty + micro_penalty * 0.25, 2400.0
    )
    spatial = _penalty_to_score(spatial_penalty + macro_penalty * 0.25, 2600.0)
    semantic = _penalty_to_score(internal_penalty * 0.4 + movement_penalty, 1800.0)
    total = 0.35 * functionality + 0.30 * naturalness + 0.20 * spatial + 0.15 * semantic
    if not bool(evaluation.get("hard_valid")):
        total = min(total, 0.05)
    return QualityBreakdown(
        functionality=functionality,
        naturalness=naturalness,
        semantic=semantic,
        spatial=spatial,
        total=total,
    )


def _penalty_to_score(penalty: float, scale: float) -> float:
    penalty = max(0.0, penalty)
    return max(0.0, min(1.0, 1.0 - (penalty / (penalty + scale))))


def _movement_penalty(
    original_payload: Mapping[str, object],
    current_payload: Mapping[str, object],
) -> float:
    original_state = materialize_phase2_state(dict(original_payload), repair=None)
    current_state = materialize_phase2_state(dict(current_payload), repair=None)
    return _movement_penalty_between_states(original_state, current_state)


def _movement_penalty_after_move(
    original_payload: Mapping[str, object],
    current_payload: Mapping[str, object],
    proposal: Mapping[str, object],
) -> float:
    candidate_state = materialize_phase2_state(
        dict(current_payload), repair=dict(proposal)
    )
    original_state = materialize_phase2_state(dict(original_payload), repair=None)
    return _movement_penalty_between_states(original_state, candidate_state)


def _movement_penalty_between_states(
    original_state: Mapping[str, object],
    current_state: Mapping[str, object],
) -> float:
    original = {
        (str(row.get("cluster_id") or ""), str(row.get("object_id") or "")): _center(
            row
        )
        for row in original_state.get("objects") or []
        if isinstance(row, Mapping)
    }
    penalty = 0.0
    for row in current_state.get("objects") or []:
        if not isinstance(row, Mapping):
            continue
        key = (str(row.get("cluster_id") or ""), str(row.get("object_id") or ""))
        before = original.get(key)
        if before is None:
            continue
        dist = _distance(before, _center(row))
        penalty += max(0.0, dist - 50.0) * (
            1.8 if _move_freedom(key[1]) == "low" else 1.0
        )
    return penalty


def _candidate_rank(
    move: MicroMove,
    quality: QualityBreakdown,
    gain: float,
) -> tuple[float, float, int, int, str]:
    return (
        round(gain, 6),
        round(quality.total, 6),
        move.priority,
        -len(move.touched_objects),
        move.tabu_key(),
    )


def _refined_layout_solution(
    *,
    status: RefinerStatus,
    original_payload: Mapping[str, object],
    current_payload: Mapping[str, object],
    repair: Mapping[str, object],
    evaluation: Mapping[str, object],
    quality_before: QualityBreakdown,
    quality_after: QualityBreakdown,
    applied_moves: Sequence[AppliedMove],
    notes: Sequence[str],
) -> dict[str, Any]:
    _ = original_payload
    state = materialize_phase2_state(dict(current_payload), repair=None)
    objects = []
    for row in state.get("objects") or []:
        if not isinstance(row, Mapping):
            continue
        center = (
            row.get("world_center")
            if isinstance(row.get("world_center"), Mapping)
            else {}
        )
        objects.append(
            {
                "object_id": str(row.get("object_id") or ""),
                "cluster_id": str(row.get("cluster_id") or ""),
                "x": int(center.get("x") or 0),
                "y": int(center.get("y") or 0),
                "rot": int(row.get("rotation_ccw") or 0),
            }
        )
    _ = repair
    touched_objects = sorted(
        {key.object_id for move in applied_moves for key in move.touched_objects}
    )
    clusters_touched = sorted(
        {key.cluster_id for move in applied_moves for key in move.touched_objects}
    )
    return {
        "status": status,
        "base_solution_id": str(
            ((current_payload.get("phase_control") or {}).get("solver_status"))
            or "selected_macro_solution"
        ),
        "refined_layout": objects,
        "refinement_summary": {
            "moves_applied": [move.as_dict() for move in applied_moves],
            "objects_touched": touched_objects,
            "clusters_touched": clusters_touched,
        },
        "quality_before": quality_before.as_dict(),
        "quality_after": quality_after.as_dict(),
        "hard_valid": bool(evaluation.get("hard_valid")),
        "notes": list(notes),
    }


def _first_delta(move: MicroMove) -> tuple[int, int] | None:
    for repair in move.repairs:
        params = repair.get("params") or {}
        if repair.get("op") == "nudge_object":
            return (
                int(round(float(params.get("dx", 0)))),
                int(round(float(params.get("dy", 0)))),
            )
    return None


def _reverse_tabu_key(move: MicroMove) -> str:
    reversed_repairs = []
    for repair in move.repairs:
        row = deepcopy(repair)
        params = deepcopy(row.get("params") or {})
        if row.get("op") == "nudge_object":
            params["dx"] = -int(round(float(params.get("dx", 0))))
            params["dy"] = -int(round(float(params.get("dy", 0))))
            row["params"] = params
        reversed_repairs.append(row)
    reverse_move = MicroMove(
        move_type=move.move_type,
        repairs=tuple(reversed_repairs),
        reason=move.reason,
        touched_objects=move.touched_objects,
        priority=move.priority,
    )
    return reverse_move.tabu_key()


def _center(row: Mapping[str, object]) -> tuple[float, float]:
    center = row.get("world_center")
    if isinstance(center, Mapping):
        return (float(center.get("x") or 0.0), float(center.get("y") or 0.0))
    return (0.0, 0.0)


def _room_center(payload: Mapping[str, object]) -> tuple[float, float]:
    state = materialize_phase2_state(dict(payload), repair=None)
    center = state.get("room_center")
    if isinstance(center, tuple) and len(center) == 2:
        return (float(center[0]), float(center[1]))
    return (0.0, 0.0)


def _vec(value: object) -> tuple[float, float] | None:
    if not isinstance(value, Mapping):
        return None
    return _unit(float(value.get("dx") or 0.0), float(value.get("dy") or 0.0))


def _unit(x: float, y: float) -> tuple[float, float] | None:
    norm = math.hypot(x, y)
    if norm <= 1e-9:
        return None
    return (x / norm, y / norm)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _world_to_local_offset(
    obj: Mapping[str, object],
    world_dx: float,
    world_dy: float,
) -> tuple[int, int]:
    rot = int(obj.get("rotation_ccw") or 0) % 360
    if rot == 0:
        dx, dy = world_dx, world_dy
    elif rot == 90:
        dx, dy = world_dy, -world_dx
    elif rot == 180:
        dx, dy = -world_dx, -world_dy
    elif rot == 270:
        dx, dy = -world_dy, world_dx
    else:
        dx, dy = world_dx, world_dy
    return (int(round(dx)), int(round(dy)))


def _move_freedom(object_id: str) -> Literal["low", "medium", "high"]:
    text = object_id.lower()
    if _has_keyword(text, _HEAVY_OBJECT_KEYWORDS):
        return "low"
    if _has_keyword(text, _FLEX_OBJECT_KEYWORDS):
        return "high"
    return "medium"


def _target_pair_distance(anchor_id: str, object_id: str) -> float:
    text = f"{anchor_id} {object_id}".lower()
    if "coffee" in text and "sofa" in text:
        return 450.0
    if "nightstand" in text or "bedside" in text:
        return 120.0
    if "chair" in text and "desk" in text:
        return 450.0
    if "side" in text and ("armchair" in text or "chair" in text):
        return 250.0
    return 600.0


def _has_keyword(text: str, keywords: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)
