from __future__ import annotations

from copy import deepcopy
from typing import Any


def summarize_layout_coverage(
    *,
    absolute_layout: dict[str, Any],
    merged_output: dict[str, Any],
    relation_plan: dict[str, Any] | None = None,
    cluster_outlines: dict[str, Any] | None = None,
    hard_valid: bool,
    acceptable_valid: bool | None = None,
) -> dict[str, Any]:
    expected_cluster_ids = _extract_expected_cluster_ids(
        merged_output=merged_output,
        cluster_outlines=cluster_outlines,
    )
    present_cluster_ids = _extract_present_cluster_ids(absolute_layout)
    primary_cluster_id = _extract_primary_cluster_id(relation_plan)
    missing_cluster_ids = sorted(set(expected_cluster_ids) - set(present_cluster_ids))
    unexpected_cluster_ids = sorted(
        set(present_cluster_ids) - set(expected_cluster_ids)
    )
    matched_count = len(set(expected_cluster_ids) & set(present_cluster_ids))
    coverage_ratio = 1.0
    if expected_cluster_ids:
        coverage_ratio = matched_count / float(len(expected_cluster_ids))
    primary_cluster_present = (
        primary_cluster_id in present_cluster_ids if primary_cluster_id else True
    )
    complete = len(missing_cluster_ids) == 0
    acceptable = bool(hard_valid if acceptable_valid is None else acceptable_valid)
    gallery_eligible = (
        bool(hard_valid)
        and acceptable
        and complete
        and primary_cluster_present
    )

    return {
        "expected_cluster_ids": expected_cluster_ids,
        "present_cluster_ids": present_cluster_ids,
        "missing_cluster_ids": missing_cluster_ids,
        "unexpected_cluster_ids": unexpected_cluster_ids,
        "primary_cluster_id": primary_cluster_id,
        "primary_cluster_present": primary_cluster_present,
        "acceptable_valid": acceptable,
        "coverage_ratio": round(float(coverage_ratio), 4),
        "complete": complete,
        "gallery_eligible": gallery_eligible,
    }


def annotate_layout_coverage(
    *,
    absolute_layout: dict[str, Any],
    merged_output: dict[str, Any],
    relation_plan: dict[str, Any] | None = None,
    cluster_outlines: dict[str, Any] | None = None,
    hard_valid: bool,
    acceptable_valid: bool | None = None,
) -> dict[str, Any]:
    coverage = summarize_layout_coverage(
        absolute_layout=absolute_layout,
        merged_output=merged_output,
        relation_plan=relation_plan,
        cluster_outlines=cluster_outlines,
        hard_valid=hard_valid,
        acceptable_valid=acceptable_valid,
    )
    payload = deepcopy(absolute_layout)
    payload["coverage"] = deepcopy(coverage)
    payload["acceptable_valid"] = bool(coverage["acceptable_valid"])
    payload["complete"] = bool(coverage["complete"])
    payload["gallery_eligible"] = bool(coverage["gallery_eligible"])
    payload["coverage_ratio"] = float(coverage["coverage_ratio"])
    payload["missing_cluster_ids"] = deepcopy(coverage["missing_cluster_ids"])
    payload["primary_cluster_id"] = coverage["primary_cluster_id"]
    payload["primary_cluster_present"] = bool(coverage["primary_cluster_present"])

    existing_missing = payload.get("missing")
    missing_rows = list(existing_missing) if isinstance(existing_missing, list) else []
    for cluster_id in coverage["missing_cluster_ids"]:
        missing_rows.append(
            {
                "code": "MISSING_CLUSTER",
                "cluster_id": cluster_id,
                "detail": f"Cluster '{cluster_id}' is missing from the final layout.",
            }
        )
    payload["missing"] = missing_rows
    return payload


def _extract_primary_cluster_id(relation_plan: dict[str, Any] | None) -> str | None:
    if not isinstance(relation_plan, dict):
        return None
    intent_profile = relation_plan.get("layout_intent_profile")
    if not isinstance(intent_profile, dict):
        return None
    cluster_id = str(intent_profile.get("primary_cluster_id") or "").strip()
    return cluster_id or None


def _extract_present_cluster_ids(absolute_layout: dict[str, Any]) -> list[str]:
    present: set[str] = set()
    for row in absolute_layout.get("objects") or []:
        if not isinstance(row, dict):
            continue
        cluster_id = str(row.get("cluster_id") or "").strip()
        if cluster_id:
            present.add(cluster_id)
    return sorted(present)


def _extract_expected_cluster_ids(
    *,
    merged_output: dict[str, Any],
    cluster_outlines: dict[str, Any] | None,
) -> list[str]:
    expected: list[str] = []
    for cluster in merged_output.get("clusters") or []:
        if not isinstance(cluster, dict):
            continue
        cluster_id = str(cluster.get("cluster_id") or "").strip()
        if not cluster_id:
            continue
        if _cluster_is_expected(
            cluster=cluster, outline=_get_outline(cluster_outlines, cluster_id)
        ):
            expected.append(cluster_id)
    return expected


def _get_outline(
    cluster_outlines: dict[str, Any] | None, cluster_id: str
) -> dict[str, Any] | None:
    if not isinstance(cluster_outlines, dict):
        return None
    row = cluster_outlines.get(cluster_id)
    return row if isinstance(row, dict) else None


def _cluster_is_expected(
    *,
    cluster: dict[str, Any],
    outline: dict[str, Any] | None,
) -> bool:
    decisions = cluster.get("decisions")
    if isinstance(decisions, list) and decisions:
        return any(
            isinstance(item, dict) and int(item.get("quantity") or 0) > 0
            for item in decisions
        )

    if isinstance(outline, dict):
        if str(outline.get("status") or "").upper() == "UNSAT":
            return False
        placements = outline.get("local_placements")
        if isinstance(placements, list) and placements:
            return True

    members = cluster.get("members")
    return isinstance(members, list) and len(members) > 0
