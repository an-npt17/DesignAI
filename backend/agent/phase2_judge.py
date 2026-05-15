from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from agent_schema.phase2_judge_schema import Phase2JudgeOutput

try:
    from clients.base_client import ChatMessage
    from clients.llm_client import get_llm_client
    from config.gemini_config import GeminiConfig
except Exception:  # pragma: no cover
    ChatMessage = dict[str, str]  # type: ignore[misc,assignment]
    get_llm_client = None  # type: ignore[assignment]
    GeminiConfig = None  # type: ignore[assignment]

from prompt.phase2_judge import (
    PHASE2_JUDGE_PROMPT,
    PHASE2_JUDGE_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

_MACRO_ISSUE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "orientation": ("orientation", "facing", "face", "align"),
    "focal": ("focal", "tv", "viewing", "view"),
    "circulation": ("circulation", "walk", "path", "entry", "door", "lane"),
    "zoning": ("zone", "zoning", "center", "central", "congestion"),
    "openings": ("window", "opening", "wall", "edge", "perimeter"),
}
_OBJECT_ISSUE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "front_access": ("front access", "front clearance", "clearance"),
    "object_pose": ("object", "chair", "nightstand", "lamp", "desk", "sofa"),
    "local_fidelity": (
        "internal",
        "local",
        "variant",
        "fidelity",
        "spacing",
        "gap",
        "crowded",
        "too close",
        "too far",
    ),
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
}


class JudgeLLMCallable(Protocol):
    def __call__(
        self, *, system_prompt: str, user_payload_json: str
    ) -> str | dict[str, Any]: ...


@dataclass(frozen=True)
class Phase2Judge:
    system_prompt: str = PHASE2_JUDGE_SYSTEM_PROMPT
    prompt_template: str = PHASE2_JUDGE_PROMPT

    def build_messages(self, input_payload: Mapping[str, object]) -> list[ChatMessage]:
        payload = json.dumps(input_payload, ensure_ascii=True)
        user_prompt = f"{self.prompt_template}\n\nINPUT_JSON:\n{payload}"
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def generate_raw(
        self,
        input_payload: Mapping[str, object],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        _ = temperature
        _ = max_tokens
        logger.info(
            "Phase2Judge input: hard_valid=%s baseline_score=%s candidate_score=%s",
            ((input_payload.get("hard_check_summary") or {}).get("hard_valid")),
            (
                (
                    (input_payload.get("baseline_summary") or {}).get("score_summary")
                    or {}
                ).get("score")
            ),
            (
                (
                    (input_payload.get("candidate_summary") or {}).get("score_summary")
                    or {}
                ).get("score")
            ),
        )
        payload = _build_deterministic_judge_payload(input_payload)
        return json.dumps(payload, ensure_ascii=True)

    def generate(
        self,
        input_payload: Mapping[str, object],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Phase2JudgeOutput:
        raw = self.generate_raw(
            input_payload,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        payload = _parse_json(raw)
        payload = _calibrate_judge_payload(input_payload, payload)
        result = Phase2JudgeOutput.model_validate(payload)
        logger.info(
            "Phase2Judge output: verdict=%s score=%s priority_clusters=%s",
            result.verdict,
            result.reasonableness_score,
            result.priority_clusters,
        )
        return result


def run_phase2_judge(
    *,
    payload: Mapping[str, object],
    llm_call: JudgeLLMCallable,
    system_prompt: str = PHASE2_JUDGE_SYSTEM_PROMPT,
    prompt_template: str = PHASE2_JUDGE_PROMPT,
) -> dict[str, Any]:
    user_payload_json = (
        f"{prompt_template}\n\nINPUT_JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    raw = llm_call(system_prompt=system_prompt, user_payload_json=user_payload_json)
    parsed = _calibrate_judge_payload(payload, _parse_json(raw))
    return Phase2JudgeOutput.model_validate(parsed).model_dump()


def _build_deterministic_judge_payload(
    input_payload: Mapping[str, object],
) -> dict[str, Any]:
    hard_check = input_payload.get("hard_check_summary") or {}
    comparison = input_payload.get("comparison_summary") or {}
    diagnosis = input_payload.get("diagnosis") or {}
    metrics = input_payload.get("metrics") or {}

    hard_valid = bool(hard_check.get("hard_valid"))
    delta_score = int(comparison.get("delta_score") or 0)
    severe_metric_count = _count_severe_metric_rows(metrics)
    priority_clusters = _cluster_priority_fallback(input_payload)
    top_issues = _build_top_issues(
        hard_check=hard_check,
        diagnosis=diagnosis,
        metrics=metrics,
    )
    issue_families = _issue_families_from_texts(top_issues)
    planner_object_pressure = _planner_object_pressure_summary(input_payload)
    repair_advice = _build_repair_advice(
        hard_valid=hard_valid,
        delta_score=delta_score,
        issue_families=issue_families,
        severe_metric_count=severe_metric_count,
        planner_object_pressure=planner_object_pressure,
    )

    if not hard_valid:
        verdict = "REJECT"
        reasonableness_score = 10
    elif delta_score >= 1200 and severe_metric_count == 0:
        verdict = "ACCEPT"
        reasonableness_score = 90
    elif delta_score >= 600 and severe_metric_count <= 1:
        verdict = "ACCEPT"
        reasonableness_score = 86
    elif delta_score > 0:
        verdict = "REVISE"
        reasonableness_score = 72 if severe_metric_count <= 1 else 62
    elif severe_metric_count <= 1:
        verdict = "REVISE"
        reasonableness_score = 58
    else:
        verdict = "REJECT"
        reasonableness_score = 30

    return {
        "reasonableness_score": reasonableness_score,
        "verdict": verdict,
        "top_issues": top_issues[:4],
        "repair_advice": repair_advice[:3],
        "priority_clusters": priority_clusters[:4],
    }


def _build_top_issues(
    *,
    hard_check: Mapping[str, object],
    diagnosis: Mapping[str, object],
    metrics: Mapping[str, object],
) -> list[str]:
    issues: list[str] = []
    errors = hard_check.get("errors") or []
    if isinstance(errors, Sequence):
        for error in errors[:3]:
            if not isinstance(error, Mapping):
                continue
            code = str(error.get("code") or "hard-invalid")
            issues.append(f"Hard-validity failed due to {code.lower()}.")

    for finding in diagnosis.get("key_findings") or []:
        text = str(finding).strip()
        if text and text not in issues:
            issues.append(text)
        if len(issues) >= 4:
            break

    if issues:
        return issues

    main_paths = (metrics.get("main_path_clearance") or {}).get("paths") or []
    for row in main_paths:
        if not isinstance(row, Mapping):
            continue
        shortage = int(row.get("clearance_shortage_mm") or 0)
        target_cluster_id = str(row.get("target_cluster_id") or "").strip()
        if shortage >= 80 and target_cluster_id:
            issues.append(
                f"Circulation to {target_cluster_id} still has a {shortage}mm clearance shortage."
            )
            break

    if issues:
        return issues

    internal_rows = metrics.get("cluster_internal_constraint_fidelity") or []
    for row in internal_rows:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("constraint_type") or "") != "semantic_proximity":
            continue
        cluster_id = str(row.get("cluster_id") or "").strip()
        subjects = row.get("subjects") or {}
        object_id = str(subjects.get("a") or "").strip()
        base_id = str(subjects.get("b") or "").strip()
        proximity = str(row.get("proximity") or "balanced")
        if cluster_id and object_id and base_id:
            issues.append(
                f"{cluster_id} still has awkward {proximity} spacing between {object_id} and {base_id}."
            )
            break

    if issues:
        return issues

    prioritized_clusters = diagnosis.get("prioritized_clusters") or []
    for row in prioritized_clusters:
        if not isinstance(row, Mapping):
            continue
        cluster_id = str(row.get("cluster_id") or "").strip()
        orientation_penalty = int(row.get("orientation_penalty_mm") or 0)
        if cluster_id and orientation_penalty > 0:
            issues.append(
                f"{cluster_id} still has orientation pressure ({orientation_penalty}mm penalty)."
            )
            break

    return issues or ["Layout still needs another repair pass."]


def _build_repair_advice(
    *,
    hard_valid: bool,
    delta_score: int,
    issue_families: list[str],
    severe_metric_count: int,
    planner_object_pressure: Mapping[str, object] | None = None,
) -> list[str]:
    if not hard_valid:
        return ["Repair hard validity before spending search on softer layout goals."]
    if delta_score <= 0:
        return [
            "Switch macro direction instead of repeating the same local move family."
        ]
    planner_object_pressure = planner_object_pressure or {}
    if bool(planner_object_pressure.get("dominant")):
        return [
            "Use planner-first object refinement and allow a synchronized two-cluster object repair if needed."
        ]
    if {"orientation", "focal", "circulation", "zoning", "openings"}.intersection(
        issue_families
    ):
        return [
            "Use a macro repair: rotate clusters, switch variants, or rebalance zones."
        ]
    if {"front_access", "object_pose", "local_fidelity"}.intersection(issue_families):
        return [
            "Use object-level refinement for front access, object pose, or local fidelity."
        ]
    if severe_metric_count >= 2:
        return ["Prefer macro repair because multiple global metrics are still severe."]
    return ["Keep improving the current layout without breaking hard validity."]


def make_file_response_adapter(path: str | Path) -> JudgeLLMCallable:
    def _call(*, system_prompt: str, user_payload_json: str) -> str | dict[str, Any]:
        _ = system_prompt
        _ = user_payload_json
        return json.loads(Path(path).read_text(encoding="utf-8"))

    return _call


def _extract_content(response: object) -> str:
    choices = getattr(response, "choices", None)
    if isinstance(choices, Sequence) and choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
    raise ValueError("OpenAI response missing message content")


def _parse_json(raw: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    text = _coerce_json_text(str(raw))
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        text = _extract_json_object(text)
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Phase2Judge response must be a JSON object")
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


def _issue_families_from_texts(texts: Sequence[object]) -> list[str]:
    families: list[str] = []
    lowered_text = " ".join(
        str(item).strip().lower() for item in texts if str(item).strip()
    )
    if not lowered_text:
        return families

    for family, keywords in _MACRO_ISSUE_KEYWORDS.items():
        if any(keyword in lowered_text for keyword in keywords):
            families.append(family)
    for family, keywords in _OBJECT_ISSUE_KEYWORDS.items():
        if any(keyword in lowered_text for keyword in keywords):
            families.append(family)
    return families


def _infer_next_step_mode(
    input_payload: Mapping[str, object],
    calibrated_payload: Mapping[str, object],
    *,
    severe_metric_count: int,
) -> str:
    hard_check = input_payload.get("hard_check_summary") or {}
    comparison = input_payload.get("comparison_summary") or {}
    metrics = input_payload.get("metrics") or {}
    diagnosis = input_payload.get("diagnosis") or {}

    if not bool(hard_check.get("hard_valid")):
        return "macro_layout"

    verdict = str(calibrated_payload.get("verdict") or "REJECT").upper()
    if verdict == "ACCEPT":
        return "stop"
    if verdict == "REJECT":
        return "macro_layout"

    explicit_mode = str(calibrated_payload.get("next_step_mode") or "").strip().lower()
    if explicit_mode in {"macro_layout", "object_refine", "stop"}:
        return explicit_mode

    delta_score = int(comparison.get("delta_score") or 0)
    priority_clusters = [
        item
        for item in calibrated_payload.get("priority_clusters") or []
        if isinstance(item, str) and item
    ]
    prioritized_objects = [
        row
        for row in diagnosis.get("prioritized_objects") or []
        if isinstance(row, dict)
    ]
    issue_texts: list[object] = []
    issue_texts.extend(calibrated_payload.get("top_issues") or [])
    issue_texts.extend(calibrated_payload.get("repair_advice") or [])
    issue_texts.extend(diagnosis.get("key_findings") or [])
    issue_families = set(_issue_families_from_texts(issue_texts))
    planner_object_pressure = _planner_object_pressure_summary(input_payload)

    macro_metric_pressure = bool(
        severe_metric_count >= 2
        or len(priority_clusters) >= 2
        or int((metrics.get("central_congestion") or [{}])[0].get("penalty_mm") or 0)
        >= 750
    )
    object_metric_pressure = bool(
        prioritized_objects and len(priority_clusters) <= 1 and severe_metric_count == 0
    )

    if delta_score <= 0:
        return "macro_layout"
    if bool(planner_object_pressure.get("dominant")):
        return "object_refine"
    if issue_families.intersection(
        {"orientation", "focal", "circulation", "zoning", "openings"}
    ):
        return "macro_layout"
    if macro_metric_pressure:
        return "macro_layout"
    if issue_families.intersection({"front_access", "object_pose", "local_fidelity"}):
        return "object_refine"
    if object_metric_pressure and delta_score >= 180:
        return "object_refine"
    return "macro_layout"


def _cluster_priority_fallback(input_payload: Mapping[str, object]) -> list[str]:
    metrics = input_payload.get("metrics") or {}
    diagnosis = input_payload.get("diagnosis") or {}
    priorities: list[str] = []
    for row in diagnosis.get("prioritized_clusters") or []:
        if isinstance(row, dict) and isinstance(row.get("cluster_id"), str):
            priorities.append(row["cluster_id"])
    for row in metrics.get("cluster_affinity_to_preferred_zone") or []:
        if isinstance(row, dict) and isinstance(row.get("cluster_id"), str):
            priorities.append(row["cluster_id"])
    deduped: list[str] = []
    for cluster_id in priorities:
        if cluster_id not in deduped:
            deduped.append(cluster_id)
    return deduped[:4]


def _count_severe_metric_rows(metrics: Mapping[str, object]) -> int:
    severe = 0
    threshold_by_key = {
        "cluster_affinity_to_preferred_zone": 700,
        "opening_band_blocking": 450,
        "central_congestion": 750,
        "cluster_edge_vs_center_fit": 220,
        "cluster_internal_constraint_fidelity": 260,
    }
    for key, threshold in threshold_by_key.items():
        for row in metrics.get(key) or []:
            if not isinstance(row, dict):
                continue
            if int(row.get("penalty_mm") or 0) >= threshold:
                severe += 1
    main_path = (metrics.get("main_path_clearance") or {}).get("paths") or []
    for row in main_path:
        if not isinstance(row, dict):
            continue
        shortage = int(row.get("clearance_shortage_mm") or 0)
        blocked_samples = int(row.get("blocked_samples") or 0)
        sample_count = max(int(row.get("sample_count") or 0), 1)
        min_clearance = int(row.get("min_clearance_mm") or 0)
        if shortage >= 220 or min_clearance < 450:
            severe += 2
            continue
        if shortage >= 80 or blocked_samples / sample_count >= 0.25:
            severe += 1
            continue
        if float(row.get("blocked_ratio") or 0.0) >= 0.18:
            severe += 1
    return severe


def _planner_object_pressure_summary(
    input_payload: Mapping[str, object],
) -> dict[str, object]:
    metrics = input_payload.get("metrics") or {}
    diagnosis = input_payload.get("diagnosis") or {}

    planner_object_penalty = 0
    macro_penalty = 0
    planner_clusters: set[str] = set()
    planner_objects: set[tuple[str, str]] = set()

    for row in metrics.get("orientation_debug") or []:
        if not isinstance(row, Mapping):
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
        if isinstance(row, Mapping)
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


def _calibrate_judge_payload(
    input_payload: Mapping[str, object], output_payload: dict[str, Any]
) -> dict[str, Any]:
    calibrated = dict(output_payload)
    comparison = input_payload.get("comparison_summary") or {}
    hard_check = input_payload.get("hard_check_summary") or {}
    metrics = input_payload.get("metrics") or {}

    hard_valid = bool(hard_check.get("hard_valid"))
    delta_score = int(comparison.get("delta_score") or 0)
    severe_metric_count = _count_severe_metric_rows(metrics)
    verdict = str(calibrated.get("verdict") or "REJECT").upper()
    reasonableness_score = int(calibrated.get("reasonableness_score") or 0)

    if not hard_valid:
        calibrated["verdict"] = "REJECT"
        calibrated["reasonableness_score"] = min(reasonableness_score, 30)
    elif delta_score < 0:
        calibrated["verdict"] = "REJECT"
        calibrated["reasonableness_score"] = min(reasonableness_score, 35)
    elif delta_score == 0:
        if verdict == "ACCEPT" and severe_metric_count == 0:
            calibrated["verdict"] = "REVISE"
            calibrated["reasonableness_score"] = max(75, reasonableness_score)
        elif severe_metric_count <= 1:
            calibrated["verdict"] = "REVISE"
            calibrated["reasonableness_score"] = min(max(reasonableness_score, 58), 72)
        else:
            calibrated["verdict"] = "REJECT"
            calibrated["reasonableness_score"] = min(reasonableness_score, 45)
    elif delta_score >= 1200 and severe_metric_count == 0:
        calibrated["verdict"] = "ACCEPT"
        calibrated["reasonableness_score"] = max(reasonableness_score, 88)
    elif delta_score >= 600 and severe_metric_count <= 1:
        calibrated["verdict"] = "ACCEPT"
        calibrated["reasonableness_score"] = max(reasonableness_score, 85)
    elif delta_score >= 180:
        if verdict in {"REJECT", "ACCEPT"}:
            calibrated["verdict"] = "REVISE"
        calibrated["reasonableness_score"] = min(max(reasonableness_score, 72), 84)
    else:
        if verdict == "REJECT" and severe_metric_count <= 1:
            calibrated["verdict"] = "REVISE"
        upper_bound = 69 if calibrated.get("verdict") == "REVISE" else 45
        lower_bound = 55 if calibrated.get("verdict") == "REVISE" else 20
        calibrated["reasonableness_score"] = min(
            max(reasonableness_score, lower_bound), upper_bound
        )

    if calibrated.get("verdict") == "ACCEPT":
        calibrated["repair_advice"] = []

    fallback_priorities = _cluster_priority_fallback(input_payload)
    priority_clusters: list[str] = []
    for item in calibrated.get("priority_clusters") or []:
        if isinstance(item, str) and item and item not in priority_clusters:
            priority_clusters.append(item)
    for cluster_id in fallback_priorities:
        if cluster_id not in priority_clusters:
            priority_clusters.append(cluster_id)
    calibrated["priority_clusters"] = priority_clusters[:4]
    calibrated["next_step_mode"] = _infer_next_step_mode(
        input_payload,
        calibrated,
        severe_metric_count=severe_metric_count,
    )
    return calibrated
