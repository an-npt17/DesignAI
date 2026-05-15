from __future__ import annotations

import os
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Literal, cast

from stylist.style_policy import build_neutral_style_policy

AblationMode = Literal[
    "full",
    "no_style_policy",
    "no_capacity_control",
    "single_concept",
]

ABLATION_ENV = "TKNT_ABLATION_MODE"

_VALID_MODES: set[str] = {
    "full",
    "no_style_policy",
    "no_capacity_control",
    "single_concept",
}

_SYSTEM_ID_BY_MODE: dict[str, str] = {
    "full": "TKNT-Full",
    "no_style_policy": "TKNT-A1-NoStylePolicy",
    "no_capacity_control": "TKNT-A2-NoCapacityControl",
    "single_concept": "TKNT-A3-SingleConcept",
}


def resolve_ablation_mode(value: str | None = None) -> AblationMode:
    raw = (value if value is not None else os.getenv(ABLATION_ENV, "full")).strip()
    normalized = raw.lower().replace("-", "_")
    if normalized in {"", "none", "baseline"}:
        normalized = "full"
    if normalized not in _VALID_MODES:
        raise ValueError(
            f"Unknown ablation mode {raw!r}. Expected one of: "
            + ", ".join(sorted(_VALID_MODES))
        )
    # Safe because normalized is checked against the complete mode set above.
    return cast(AblationMode, normalized)


def ablation_system_id(mode: AblationMode) -> str:
    return _SYSTEM_ID_BY_MODE[mode]


def ablation_metadata(mode: AblationMode) -> dict[str, str]:
    return {
        "ablation_mode": mode,
        "system_id": ablation_system_id(mode),
        "ablation_file": "pipeline/ablation_modes.py",
    }


def uses_neutral_style_policy(mode: AblationMode) -> bool:
    return mode == "no_style_policy"


def style_policy_for_mode(
    *,
    mode: AblationMode,
    room_type: str,
    current_policy: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if uses_neutral_style_policy(mode):
        return build_neutral_style_policy(
            room_type=room_type,
            reason="Ablation A1 disables style-derived layout policy.",
        )
    return deepcopy(dict(current_policy or {}))


def target_final_count_for_mode(mode: AblationMode, default_count: int) -> int:
    if mode == "single_concept":
        return 1
    return max(1, int(default_count))


def apply_tier_output_for_mode(
    *,
    mode: AblationMode,
    tier_output: Mapping[str, Any],
) -> dict[str, Any]:
    payload = deepcopy(dict(tier_output))
    if mode != "no_capacity_control":
        return payload

    changed = False
    decisions = payload.get("decisions")
    if isinstance(decisions, list):
        for row in decisions:
            if not isinstance(row, dict):
                continue
            changed = _force_keep_decision(row) or changed

    cluster_decisions = payload.get("cluster_decisions")
    if isinstance(cluster_decisions, list):
        for cluster in cluster_decisions:
            if not isinstance(cluster, dict):
                continue
            bundles = cluster.get("selected_bundles")
            if not isinstance(bundles, list):
                continue
            for bundle in bundles:
                if not isinstance(bundle, dict):
                    continue
                objects = bundle.get("objects")
                if not isinstance(objects, list):
                    continue
                for row in objects:
                    if isinstance(row, dict):
                        changed = _force_keep_decision(row) or changed

    notes = _string_list(payload.get("notes"))
    note = (
        "Ablation A2 disabled capacity pruning/backoff; zero-quantity tier "
        "decisions were forced active before layout solving."
    )
    if changed and note not in notes:
        notes.append(note)
    payload["notes"] = notes
    payload["capacity_control_disabled"] = True
    payload["budget_valid"] = False
    return payload


def skip_accessory_refill(mode: AblationMode) -> bool:
    return mode == "no_capacity_control"


def _force_keep_decision(row: dict[str, Any]) -> bool:
    object_type = str(row.get("object_type") or row.get("category") or "").strip()
    if not object_type:
        return False
    quantity = _coerce_int(row.get("quantity"))
    if quantity is not None and quantity > 0:
        return False
    row["quantity"] = 1
    if not row.get("size_tier"):
        row["size_tier"] = "S"
    row["ablation_forced_active"] = True
    return True


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
