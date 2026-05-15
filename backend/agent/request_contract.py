from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from layout.room_profiles.registry import (
    all_profile_non_functional_contract_types,
    all_profile_object_aliases,
)

_OBJECT_ALIASES: dict[str, tuple[str, ...]] = {
    "armchair": ("armchair", "arm chair", "lounge chair", "reading chair"),
    "bed": ("bed", "queen bed", "king bed", "single bed", "double bed"),
    "bookshelf": ("bookshelf", "book shelf", "bookcase", "shelf"),
    "ceiling_lamp": ("ceiling lamp", "ceiling light", "overhead light"),
    "chair": ("chair", "desk chair", "office chair", "vanity chair"),
    "coffee_table": ("coffee table", "tea table", "center table"),
    "desk": ("desk", "work desk", "study desk", "vanity"),
    "floor_lamp": ("floor lamp", "standing lamp", "reading lamp"),
    "nightstand": ("nightstand", "night stand", "bedside table"),
    "rug": ("rug", "carpet"),
    "side_table": ("side table", "end table"),
    "sofa": ("sofa", "couch", "loveseat", "sectional"),
    "tv_console": ("tv console", "media console", "tv stand"),
    "wardrobe": ("wardrobe", "closet", "armoire"),
}
_OBJECT_ALIASES.update(all_profile_object_aliases())

_OPTIONAL_ONLY_PATTERNS = (
    "only if",
    "only when",
    "if there is enough space",
    "if there is sufficient space",
    "if enough space",
    "only if there is enough space",
    "only if it fits",
    "only if they fit",
)
_PREFERRED_IF_FIT_PATTERNS = (
    "if it fits",
    "if they fit",
    "if possible",
    "where possible",
    "when possible",
)
_MUST_VERBS = (
    "include",
    "use",
    "with",
    "feature",
    "needs",
    "need",
    "should have",
    "must have",
)
_COUNT_WORDS = {
    "one": 1,
    "a": 1,
    "an": 1,
    "single": 1,
    "two": 2,
    "pair": 2,
    "three": 3,
    "four": 4,
}
_NON_FUNCTIONAL_CONTRACT_TYPES = {
    "ceiling_lamp",
    "rug",
} | all_profile_non_functional_contract_types()
_HARD_CONTRACT_INTENTS = {"must_keep", "must_try"}
_TARGET_CONTRACT_INTENTS = {"target_if_viable", "preferred_if_fit"}
_SOFT_CONTRACT_INTENTS = {*_TARGET_CONTRACT_INTENTS, "optional_if_surplus"}
_CONTRACT_INTENT_ALIASES = {
    "required": "must_keep",
    "must": "must_keep",
    "must_have": "must_keep",
    "must_try": "must_try",
    "must_keep": "must_keep",
    "preferred": "target_if_viable",
    "prefer": "target_if_viable",
    "preferred_if_fit": "target_if_viable",
    "target": "target_if_viable",
    "target_if_viable": "target_if_viable",
    "if_viable": "target_if_viable",
    "optional": "optional_if_surplus",
    "optional_if_surplus": "optional_if_surplus",
    "max0": "max0",
    "forbidden": "max0",
    "avoid": "max0",
}


def build_request_contract(
    *,
    brief_text: str,
    available_object_types: Sequence[str] = (),
) -> dict[str, Any]:
    text = _normalize_text(brief_text)
    available = {
        canonical_object_type(item)
        for item in available_object_types
        if str(item).strip()
    }
    candidates = sorted(set(_OBJECT_ALIASES) | available)
    contract_objects: list[dict[str, Any]] = []

    for object_type in candidates:
        mentions = _find_mentions(text, object_type)
        if not mentions:
            continue
        best = _best_mention_contract(text, object_type, mentions)
        best["available_in_program"] = object_type in available
        contract_objects.append(best)

    contract_objects.sort(
        key=lambda row: (
            _intent_rank(str(row.get("intent") or "")),
            str(row.get("object_type") or ""),
        )
    )
    return {
        "version": 1,
        "source": "brief_text_heuristic",
        "objects": contract_objects,
        "groups": [],
        "notes": [
            (
                "Request contract is inferred from explicit brief object mentions; "
                "tier count may degrade but should not silently drop protected items."
            )
        ],
    }


def sanitize_request_contract(
    contract: Mapping[str, Any] | None,
    *,
    brief_text: str,
    available_object_types: Sequence[str] = (),
    fallback_to_heuristic: bool = False,
) -> dict[str, Any]:
    if not isinstance(contract, Mapping):
        if fallback_to_heuristic:
            return build_request_contract(
                brief_text=brief_text,
                available_object_types=available_object_types,
            )
        return _empty_contract(source="invalid_contract")

    text = _normalize_text(brief_text)
    available = {
        canonical_object_type(item)
        for item in available_object_types
        if str(item).strip()
    }
    by_type: dict[str, dict[str, Any]] = {}
    raw_objects = contract.get("objects")
    if isinstance(raw_objects, Sequence) and not isinstance(raw_objects, str):
        for raw_item in raw_objects:
            item = _sanitize_contract_object(
                raw_item,
                text=text,
                available_object_types=available,
            )
            if item is None:
                continue
            object_type = str(item["object_type"])
            existing = by_type.get(object_type)
            if existing is None or _intent_rank(str(item["intent"])) < _intent_rank(
                str(existing.get("intent") or "")
            ):
                by_type[object_type] = item

    objects = sorted(
        by_type.values(),
        key=lambda row: (
            _intent_rank(str(row.get("intent") or "")),
            str(row.get("object_type") or ""),
        ),
    )
    if not objects and fallback_to_heuristic:
        return build_request_contract(
            brief_text=brief_text,
            available_object_types=available_object_types,
        )

    source = str(contract.get("source") or "llm_request_contract").strip()
    return {
        "version": 1,
        "source": source,
        "objects": objects,
        "groups": _sanitize_contract_groups(contract.get("groups"), objects),
        "notes": _dedupe_strings(
            [
                *_coerce_note_strings(contract.get("notes")),
                "Request contract was normalized before tier-count enforcement.",
            ]
        ),
    }


def attach_request_contract_to_semantic_program(
    semantic_program: Mapping[str, Any],
    *,
    brief_text: str,
) -> dict[str, Any]:
    out = deepcopy(dict(semantic_program))
    available_object_types = _semantic_program_object_types(out)
    existing_contract = out.get("request_contract")
    if isinstance(existing_contract, Mapping):
        contract = sanitize_request_contract(
            existing_contract,
            brief_text=brief_text,
            available_object_types=available_object_types,
            fallback_to_heuristic=True,
        )
    else:
        contract = build_request_contract(
            brief_text=brief_text,
            available_object_types=available_object_types,
        )
    out["request_contract"] = contract
    notes = [str(item) for item in out.get("notes") or [] if str(item).strip()]
    notes.append(
        (
            "Request contract attached before tier count so explicit functional objects "
            "are protected from silent pruning."
        )
    )
    out["notes"] = _dedupe_strings(notes)
    return out


def request_contract_from_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    contract = payload.get("request_contract")
    if isinstance(contract, Mapping):
        return deepcopy(dict(contract))
    semantic_program = payload.get("semantic_layout_program")
    if isinstance(semantic_program, Mapping):
        contract = semantic_program.get("request_contract")
        if isinstance(contract, Mapping):
            return deepcopy(dict(contract))
    return {}


def contract_item_for_object_type(
    contract: Mapping[str, Any] | None,
    object_type: str,
) -> dict[str, Any] | None:
    canonical = canonical_object_type(object_type)
    if not canonical:
        return None
    for item in _contract_objects(contract):
        if canonical_object_type(str(item.get("object_type") or "")) == canonical:
            return deepcopy(item)
    return None


def contract_min_keep(item: Mapping[str, Any] | None) -> int:
    if not isinstance(item, Mapping):
        return 0
    return max(0, _int_value(item.get("min_keep"), default=0))


def contract_target_count(item: Mapping[str, Any] | None) -> int:
    if not isinstance(item, Mapping):
        return 0
    target = _int_value(
        item.get("target_count", item.get("preferred_count")),
        default=0,
    )
    return max(contract_min_keep(item), target, 0)


def contract_intent(item: Mapping[str, Any] | None) -> str:
    if not isinstance(item, Mapping):
        return ""
    return str(item.get("intent") or "").strip()


def missing_functional_contract_types(
    *,
    contract: Mapping[str, Any] | None,
    objects: Sequence[Mapping[str, Any]],
) -> list[str]:
    present = {
        canonical_object_type(
            str(
                row.get("object_type")
                or row.get("type")
                or row.get("category")
                or row.get("name")
                or row.get("id")
                or ""
            )
        )
        for row in objects
        if isinstance(row, Mapping)
    }
    missing: list[str] = []
    for item in _contract_objects(contract):
        object_type = canonical_object_type(str(item.get("object_type") or ""))
        if not object_type or object_type in _NON_FUNCTIONAL_CONTRACT_TYPES:
            continue
        if contract_min_keep(item) <= 0:
            continue
        if object_type not in present:
            missing.append(object_type)
    return sorted(set(missing))


def missing_non_functional_contract_items(
    *,
    contract: Mapping[str, Any] | None,
    objects: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    present = {
        canonical_object_type(
            str(
                row.get("object_type")
                or row.get("type")
                or row.get("category")
                or row.get("name")
                or row.get("id")
                or ""
            )
        )
        for row in objects
        if isinstance(row, Mapping)
    }
    missing: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in _contract_objects(contract):
        object_type = canonical_object_type(str(item.get("object_type") or ""))
        if (
            not object_type
            or object_type in seen
            or object_type not in _NON_FUNCTIONAL_CONTRACT_TYPES
            or object_type in present
            or contract_intent(item) == "max0"
            or contract_target_count(item) <= 0
        ):
            continue
        missing.append(deepcopy(item))
        seen.add(object_type)
    return missing


def canonical_object_type(object_type: str) -> str:
    normalized = str(object_type).strip().lower().replace("-", "_").replace(" ", "_")
    normalized = normalized.rstrip("0123456789").rstrip("_")
    if not normalized:
        return ""
    for canonical, aliases in _OBJECT_ALIASES.items():
        alias_keys = {canonical, *(alias.replace(" ", "_") for alias in aliases)}
        if normalized in alias_keys:
            return canonical
    for canonical, aliases in _OBJECT_ALIASES.items():
        alias_keys = tuple(
            sorted(
                {canonical, *(alias.replace(" ", "_") for alias in aliases)},
                key=len,
                reverse=True,
            )
        )
        if any(_alias_matches(normalized, alias) for alias in alias_keys):
            return canonical
    return normalized


def _best_mention_contract(
    text: str,
    object_type: str,
    mentions: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    scored: list[tuple[int, dict[str, Any]]] = []
    for start, end in mentions:
        window = _clause_window(text, start, end)
        intent = _intent_for_window(window, object_type=object_type)
        count_hint = _count_hint(window)
        min_keep, preferred_count, max_keep = _counts_for_intent(intent, count_hint)
        scored.append(
            (
                _intent_rank(intent),
                {
                    "object_type": object_type,
                    "intent": intent,
                    "min_keep": min_keep,
                    "target_count": preferred_count,
                    "preferred_count": preferred_count,
                    "max_keep": max_keep,
                    "evidence": window.strip(),
                    "reason": _reason_for_intent(intent),
                },
            )
        )
    return sorted(scored, key=lambda row: row[0])[0][1]


def _intent_for_window(window: str, *, object_type: str) -> str:
    if _has_direct_negative_intent(window, object_type=object_type):
        return "max0"
    if any(pattern in window for pattern in _OPTIONAL_ONLY_PATTERNS):
        return "optional_if_surplus"
    if any(pattern in window for pattern in _PREFERRED_IF_FIT_PATTERNS):
        return "target_if_viable"
    if any(pattern in window for pattern in _MUST_VERBS):
        return "must_keep"
    return "must_keep"


def _counts_for_intent(
    intent: str,
    count_hint: tuple[int, int | None],
) -> tuple[int, int, int | None]:
    low, high = count_hint
    if intent == "max0":
        return 0, 0, 0
    if intent == "optional_if_surplus":
        preferred = high or low or 1
        return 0, preferred, high or preferred
    preferred = high or low or 1
    if intent in _TARGET_CONTRACT_INTENTS:
        return 0, preferred, high or preferred
    return max(1, low), preferred, high


def _count_hint(window: str) -> tuple[int, int | None]:
    if re.search(r"\bone\s+or\s+two\b|\b1\s+or\s+2\b", window):
        return (1, 2)
    if re.search(r"\btwo\s+or\s+one\b|\b2\s+or\s+1\b", window):
        return (1, 2)
    for word, count in sorted(
        _COUNT_WORDS.items(),
        key=lambda item: (item[1], len(item[0])),
        reverse=True,
    ):
        if re.search(rf"\b{re.escape(word)}\b", window):
            return (count, count)
    numeric = re.search(r"\b([1-4])\b", window)
    if numeric:
        count = int(numeric.group(1))
        return (count, count)
    return (1, None)


def _find_mentions(text: str, object_type: str) -> list[tuple[int, int]]:
    aliases = _OBJECT_ALIASES.get(object_type, (object_type.replace("_", " "),))
    spans: list[tuple[int, int]] = []
    for alias in aliases:
        pattern = re.compile(rf"\b{re.escape(alias.lower())}s?\b")
        spans.extend((match.start(), match.end()) for match in pattern.finditer(text))
    return spans


def _has_direct_negative_intent(window: str, *, object_type: str) -> bool:
    aliases = _OBJECT_ALIASES.get(object_type, (object_type.replace("_", " "),))
    for alias in aliases:
        alias_pattern = re.escape(alias.lower())
        direct_patterns = (
            rf"\b(?:do not include|don't include|avoid|without)\s+"
            rf"(?:a\s+|an\s+|the\s+|any\s+)?{alias_pattern}s?\b",
            rf"\bno\s+{alias_pattern}s?\b",
        )
        if any(re.search(pattern, window) for pattern in direct_patterns):
            return True
    return False


def _semantic_program_object_types(program: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    active_clusters = program.get("active_clusters")
    if not isinstance(active_clusters, Sequence) or isinstance(active_clusters, str):
        return out
    for cluster in active_clusters:
        if not isinstance(cluster, Mapping):
            continue
        for bundle_key in ("required_bundles", "optional_bundles"):
            bundles = cluster.get(bundle_key)
            if not isinstance(bundles, Sequence) or isinstance(bundles, str):
                continue
            for bundle in bundles:
                if not isinstance(bundle, Mapping):
                    continue
                objects = bundle.get("objects")
                if not isinstance(objects, Sequence) or isinstance(objects, str):
                    continue
                for row in objects:
                    if isinstance(row, Mapping):
                        object_type = str(row.get("object_type") or "").strip()
                        if object_type:
                            out.append(object_type)
        for member in cluster.get("members") or []:
            if isinstance(member, str) and member.strip():
                out.append(member)
    return out


def _sanitize_contract_object(
    item: object,
    *,
    text: str,
    available_object_types: set[str],
) -> dict[str, Any] | None:
    if not isinstance(item, Mapping):
        return None
    object_type = canonical_object_type(str(item.get("object_type") or ""))
    if not object_type:
        return None

    mentions = _find_mentions(text, object_type)
    raw_evidence = str(item.get("evidence") or "").strip()
    evidence = _sanitize_evidence(
        raw_evidence,
        text=text,
        mentions=mentions,
    )
    if not evidence:
        return None

    intent = _normalize_contract_intent(item.get("intent"))
    low, high = _count_hint(_normalize_text(evidence))
    target_count = _positive_int(
        item.get("target_count", item.get("preferred_count")),
        default=high or low or 1,
    )
    raw_max_keep = item.get("max_keep")
    max_keep = (
        max(0, _int_value(raw_max_keep, default=target_count))
        if raw_max_keep is not None
        else high or target_count
    )
    min_keep = max(0, _int_value(item.get("min_keep"), default=0))

    if intent == "max0":
        min_keep = 0
        target_count = 0
        max_keep = 0
    elif intent in _HARD_CONTRACT_INTENTS:
        min_keep = max(1, min_keep or low)
        target_count = max(target_count, min_keep)
    elif intent in _SOFT_CONTRACT_INTENTS:
        min_keep = 0
        target_count = max(1, target_count)

    if max_keep is not None:
        target_count = min(target_count, max_keep)
        min_keep = min(min_keep, max_keep)

    return {
        "object_type": object_type,
        "intent": intent,
        "min_keep": min_keep,
        "target_count": target_count,
        "preferred_count": target_count,
        "max_keep": max_keep,
        "evidence": evidence,
        "reason": str(item.get("reason") or _reason_for_intent(intent)).strip(),
        "available_in_program": object_type in available_object_types,
        "confidence": _confidence_value(item.get("confidence")),
    }


def _sanitize_evidence(
    evidence: str,
    *,
    text: str,
    mentions: Sequence[tuple[int, int]],
) -> str:
    normalized_evidence = _normalize_text(evidence)
    if normalized_evidence and normalized_evidence in text:
        return evidence
    if mentions:
        start, end = mentions[0]
        return _clause_window(text, start, end)
    return ""


def _normalize_contract_intent(value: object) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return _CONTRACT_INTENT_ALIASES.get(text, "target_if_viable")


def _sanitize_contract_groups(
    value: object,
    objects: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    allowed_object_types = {
        str(item.get("object_type") or "")
        for item in objects
        if str(item.get("object_type") or "")
    }
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    groups: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw_group in value:
        if not isinstance(raw_group, Mapping):
            continue
        raw_members = raw_group.get("members")
        if not isinstance(raw_members, Sequence) or isinstance(raw_members, str):
            continue
        members = [
            canonical
            for item in raw_members
            if (canonical := canonical_object_type(str(item or "")))
            in allowed_object_types
        ]
        members = _dedupe_strings(members)
        if len(members) < 2:
            continue
        group_id = str(raw_group.get("group_id") or "_".join(members)).strip()
        if not group_id or group_id in seen_ids:
            continue
        seen_ids.add(group_id)
        group: dict[str, Any] = {
            "group_id": group_id,
            "members": members,
            "intent": str(raw_group.get("intent") or "composition").strip(),
        }
        priority = str(raw_group.get("priority") or "").strip()
        if priority:
            group["priority"] = priority
        drop_policy = str(raw_group.get("drop_policy") or "").strip()
        if drop_policy:
            group["drop_policy"] = drop_policy
        drop_order_bias = str(raw_group.get("drop_order_bias") or "").strip()
        if drop_order_bias:
            group["drop_order_bias"] = drop_order_bias
        groups.append(group)
    return groups


def _contract_objects(contract: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    if not isinstance(contract, Mapping):
        return []
    objects = contract.get("objects")
    if not isinstance(objects, Sequence) or isinstance(objects, str):
        return []
    return [item for item in objects if isinstance(item, Mapping)]


def _clause_window(text: str, start: int, end: int) -> str:
    left = start
    while left > 0 and text[left - 1] not in ",.;:":
        left -= 1
    right = end
    while right < len(text) and text[right] not in ",.;:":
        right += 1
    return text[left:right].strip()


def _intent_rank(intent: str) -> int:
    return {
        "must_keep": 0,
        "must_try": 0,
        "target_if_viable": 1,
        "preferred_if_fit": 1,
        "optional_if_surplus": 2,
        "max0": 3,
    }.get(intent, 2)


def _reason_for_intent(intent: str) -> str:
    if intent in _HARD_CONTRACT_INTENTS:
        return "explicitly requested in the brief"
    if intent in _TARGET_CONTRACT_INTENTS:
        return "requested as a viable target, not a hard requirement"
    if intent == "optional_if_surplus":
        return "requested only when surplus space exists"
    if intent == "max0":
        return "brief wording discourages or forbids this object"
    return "brief mention"


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").lower().replace("-", " ").split())


def _alias_matches(normalized: str, alias: str) -> bool:
    if not alias:
        return False
    if normalized == alias:
        return True
    if "_" in alias:
        return (
            normalized.startswith(f"{alias}_")
            or normalized.endswith(f"_{alias}")
            or f"_{alias}_" in normalized
        )
    return alias in normalized.split("_")


def _int_value(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _positive_int(value: Any, *, default: int) -> int:
    return max(1, _int_value(value, default=default))


def _confidence_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def _empty_contract(*, source: str) -> dict[str, Any]:
    return {
        "version": 1,
        "source": source,
        "objects": [],
        "groups": [],
        "notes": [],
    }


def _coerce_note_strings(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _dedupe_strings(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out
