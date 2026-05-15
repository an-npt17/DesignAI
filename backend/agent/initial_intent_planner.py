from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from agent_schema.initial_intent_planner_schema import (
    InitialIntentPlannerOutput,
)
from clients.base_client import ChatMessage
from clients.llm_client import get_llm_client

try:
    from config.llm_config import TextLLMConfig
except Exception:  # pragma: no cover
    from config.gemini_config import GeminiConfig

    class TextLLMConfig:
        PROVIDER = "gemini"
        STRICT_SINGLE_TEXT_MODEL = str(
            getattr(GeminiConfig, "STRICT_SINGLE_TEXT_MODEL", "") or ""
        ).strip()
        AGENT_MODELS = dict(getattr(GeminiConfig, "AGENT_MODELS", {}) or {})

        @classmethod
        def primary_model_name(cls) -> str:
            return ""

        @classmethod
        def agent_model(cls, key: str) -> str | None:
            return cls.AGENT_MODELS.get(key)


from prompt.initial_intent_planner import INITIAL_INTENT_PLANNER_PROMPT
from prompt.system import SYSTEM_PROMPT

_INTENT_COUNT = 10
_DEFAULT_PRIORITY = "medium"
logger = logging.getLogger(__name__)


def _record_llm_retry(*, stage: str, model_name: str | None, reason: str) -> None:
    if getattr(TextLLMConfig, "PROVIDER", "") != "gemini":
        return
    try:
        from clients.gemini_client import GeminiClient

        GeminiClient.record_retry_event(
            stage=stage,
            model_name=model_name,
            reason=reason,
        )
    except Exception:
        logger.debug("Failed to record Gemini retry event.", exc_info=True)


@dataclass(frozen=True)
class InitialIntentPlanner:
    system_prompt: str = SYSTEM_PROMPT
    prompt_template: str = INITIAL_INTENT_PLANNER_PROMPT

    def build_messages(
        self,
        *,
        room_model_json: dict[str, Any],
        description: str | None = None,
        special_notes: str | None = None,
    ) -> list[ChatMessage]:
        user_prompt = (
            self.prompt_template.replace(
                "{ROOM_MODEL_JSON}",
                json.dumps(
                    _minify_room_model(room_model_json), ensure_ascii=True, indent=2
                ),
            )
            .replace("{DESCRIPTION}", (description or "").strip())
            .replace("{SPECIAL_NOTES}", (special_notes or "").strip())
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def generate_raw(
        self,
        *,
        room_model_json: dict[str, Any],
        description: str | None = None,
        special_notes: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        messages = self.build_messages(
            room_model_json=room_model_json,
            description=description,
            special_notes=special_notes,
        )
        client = get_llm_client()
        model_name = TextLLMConfig.agent_model("planner")
        last_raw = ""
        for attempt in range(2):
            response = client.chat_completion(
                messages,
                model_key="primary",
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            last_raw = _extract_content(response)
            try:
                _parse_json(last_raw)
                return last_raw
            except ValueError:
                if attempt == 1:
                    break
                _record_llm_retry(
                    stage="initial_intent_planner",
                    model_name=model_name,
                    reason="invalid_json",
                )
                messages.append({"role": "assistant", "content": last_raw})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Return valid JSON only with exactly 10 intents when possible. "
                            "Do not include markdown fences or prose outside the JSON object."
                        ),
                    }
                )
        return last_raw

    def generate(
        self,
        *,
        room_model_json: dict[str, Any],
        description: str | None = None,
        special_notes: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        fallback = _default_output(
            room_model_json=room_model_json,
            description=description,
            special_notes=special_notes,
        )
        try:
            raw = self.generate_raw(
                room_model_json=room_model_json,
                description=description,
                special_notes=special_notes,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            payload = _parse_json(raw)
            payload = _normalize_initial_intent_payload(
                payload,
                room_model_json=room_model_json,
                description=description,
                special_notes=special_notes,
            )
        except Exception:
            payload = fallback
        else:
            if not payload.get("intents"):
                payload = fallback
        return InitialIntentPlannerOutput.model_validate(payload).model_dump()


def _extract_content(response: object) -> str:
    choices = getattr(response, "choices", None)
    if isinstance(choices, list) and choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
    raise ValueError("InitialIntentPlanner response missing message content")


def _parse_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        for index in range(1, len(parts), 2):
            candidate = parts[index].strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].lstrip("\n").strip()
            if candidate.startswith("{") and candidate.endswith("}"):
                text = candidate
                break
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("InitialIntentPlanner returned invalid JSON") from None
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("InitialIntentPlanner response must be a JSON object")
    return payload


def _minify_room_model(room_model_json: dict[str, Any]) -> dict[str, Any]:
    room = room_model_json.get("room")
    meta = room_model_json.get("meta")
    openings = room_model_json.get("openings")
    if not isinstance(room, dict):
        room = {}
    if not isinstance(meta, dict):
        meta = {}
    if not isinstance(openings, dict):
        openings = {}
    return {
        "room": {
            "room_id": str(room.get("room_id") or "room_1"),
            "polygon_ccw": room.get("polygon_ccw")
            if isinstance(room.get("polygon_ccw"), list)
            else [],
        },
        "meta": {
            "room_type": meta.get("room_type"),
            "style": meta.get("style"),
            "window_direction": meta.get("window_direction"),
        },
        "openings": {
            "doors": openings.get("doors")
            if isinstance(openings.get("doors"), list)
            else [],
            "windows": openings.get("windows")
            if isinstance(openings.get("windows"), list)
            else [],
        },
    }


def _normalize_initial_intent_payload(
    payload: dict[str, Any],
    *,
    room_model_json: dict[str, Any],
    description: str | None,
    special_notes: str | None,
) -> dict[str, Any]:
    room_id = _extract_room_id(room_model_json)
    out: dict[str, Any] = {
        "status": str(payload.get("status") or "OK").strip().upper() or "OK",
        "room_id": room_id,
        "intents": [],
        "notes": _coerce_str_list(payload.get("notes")),
        "missing": _coerce_str_list(payload.get("missing")),
    }

    seen_signatures: set[tuple[Any, ...]] = set()
    seen_families: set[tuple[str, str, str, str]] = set()
    raw_intents = payload.get("intents")
    if isinstance(raw_intents, list):
        for index, item in enumerate(raw_intents, start=1):
            normalized = _normalize_intent_item(item, index=index)
            if normalized is None:
                continue
            signature = (
                normalized["focus_mode"],
                normalized["primary_tag"],
                normalized.get("secondary_tag"),
                normalized["circulation_priority"],
                normalized["center_open_preference"],
                normalized["support_cluster_behavior"],
                normalized["distribution_mode"],
            )
            family_signature = _intent_family_signature(normalized)
            if signature in seen_signatures:
                continue
            if family_signature in seen_families:
                continue
            seen_signatures.add(signature)
            seen_families.add(family_signature)
            out["intents"].append(normalized)

    fallback_intents = _default_intents(
        room_model_json=room_model_json,
        description=description,
        special_notes=special_notes,
    )
    existing_ids = {
        str(item.get("intent_id") or "")
        for item in out["intents"]
        if isinstance(item, dict)
    }
    for item in fallback_intents:
        signature = (
            item["focus_mode"],
            item["primary_tag"],
            item.get("secondary_tag"),
            item["circulation_priority"],
            item["center_open_preference"],
            item["support_cluster_behavior"],
            item["distribution_mode"],
        )
        family_signature = _intent_family_signature(item)
        if signature in seen_signatures:
            continue
        if family_signature in seen_families:
            continue
        if item["intent_id"] in existing_ids:
            continue
        seen_signatures.add(signature)
        seen_families.add(family_signature)
        out["intents"].append(item)
        existing_ids.add(item["intent_id"])
        if len(out["intents"]) >= _INTENT_COUNT:
            break

    out["intents"] = out["intents"][:_INTENT_COUNT]
    if len(out["intents"]) < _INTENT_COUNT:
        filler = _default_intents(
            room_model_json=room_model_json,
            description="",
            special_notes="",
        )
        for item in filler:
            if len(out["intents"]) >= _INTENT_COUNT:
                break
            if any(
                existing.get("intent_id") == item["intent_id"]
                for existing in out["intents"]
                if isinstance(existing, dict)
            ):
                continue
            out["intents"].append(item)
    return out


def _normalize_intent_item(item: Any, *, index: int) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    focus_mode = _coerce_choice(
        item.get("focus_mode"),
        {"viewing", "conversation", "rest", "work", "dining", "display", "mixed"},
        default="mixed",
    )
    primary_tag = _coerce_choice(
        item.get("primary_tag"),
        {"sleep", "work", "living", "dining", "storage", "misc"},
        default="living",
    )
    secondary_tag = _coerce_optional_choice(
        item.get("secondary_tag"),
        {"sleep", "work", "living", "dining", "storage", "misc"},
    )
    if secondary_tag == primary_tag:
        secondary_tag = None
    label = str(item.get("label") or f"Intent {index}").strip() or f"Intent {index}"
    summary = (
        str(item.get("summary") or "").strip()
        or f"{focus_mode.title()}-leaning layout around {primary_tag}."
    )
    return {
        "intent_id": str(item.get("intent_id") or f"intent_{index}").strip()
        or f"intent_{index}",
        "label": label,
        "summary": summary,
        "focus_mode": focus_mode,
        "primary_tag": primary_tag,
        "secondary_tag": secondary_tag,
        "circulation_priority": _coerce_choice(
            item.get("circulation_priority"),
            {"high", "medium", "low"},
            default=_DEFAULT_PRIORITY,
        ),
        "center_open_preference": _coerce_choice(
            item.get("center_open_preference"),
            {"high", "medium", "low"},
            default=_DEFAULT_PRIORITY,
        ),
        "support_cluster_behavior": _coerce_choice(
            item.get("support_cluster_behavior"),
            {"recede", "balanced", "integrate"},
            default="balanced",
        ),
        "distribution_mode": _coerce_choice(
            item.get("distribution_mode"),
            {"balanced", "edge_weighted", "focal_grouped", "zoned"},
            default="balanced",
        ),
        "forge_guidance": _coerce_str_list(item.get("forge_guidance")),
        "composer_guidance": _coerce_str_list(item.get("composer_guidance")),
        "notes": _coerce_str_list(item.get("notes")),
    }


def _intent_family_signature(item: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(item.get("focus_mode") or "mixed"),
        str(item.get("primary_tag") or "living"),
        str(item.get("distribution_mode") or "balanced"),
        str(item.get("center_open_preference") or _DEFAULT_PRIORITY),
    )


def _coerce_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            text = item.strip()
            if text and text not in out:
                out.append(text)
        return out
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return []


def _coerce_choice(value: Any, allowed: set[str], *, default: str) -> str:
    text = str(value or "").strip().lower()
    if text in allowed:
        return text
    return default


def _coerce_optional_choice(value: Any, allowed: set[str]) -> str | None:
    text = str(value or "").strip().lower()
    if text in allowed:
        return text
    return None


def _extract_room_id(room_model_json: dict[str, Any]) -> str:
    room = room_model_json.get("room")
    if isinstance(room, dict):
        room_id = room.get("room_id")
        if isinstance(room_id, str) and room_id.strip():
            return room_id.strip()
    return "room_1"


def _default_output(
    *,
    room_model_json: dict[str, Any],
    description: str | None,
    special_notes: str | None,
) -> dict[str, Any]:
    return {
        "status": "OK",
        "room_id": _extract_room_id(room_model_json),
        "intents": _default_intents(
            room_model_json=room_model_json,
            description=description,
            special_notes=special_notes,
        ),
        "notes": ["InitialIntentPlanner used deterministic fallback intents."],
        "missing": [],
    }


def _default_intents(
    *,
    room_model_json: dict[str, Any],
    description: str | None,
    special_notes: str | None,
) -> list[dict[str, Any]]:
    room_type = _extract_room_type(room_model_json)
    seed_text = f"{room_type}\n{description or ''}\n{special_notes or ''}".lower()
    primary_tag, secondary_tag, focus_mode = _infer_primary_roles(seed_text, room_type)

    support_tag = secondary_tag or "storage"
    alternate_tag = "misc" if support_tag != "misc" else "storage"
    conversation_mode = "conversation" if primary_tag == "living" else focus_mode
    display_mode = "display" if primary_tag == "living" else focus_mode
    work_mode = "work" if primary_tag in {"work", "living"} else focus_mode
    rest_mode = "rest" if primary_tag in {"sleep", "living"} else focus_mode

    templates = [
        {
            "intent_id": "intent_1",
            "label": "Balanced Anchor",
            "summary": f"Balanced layout centered on {primary_tag}.",
            "focus_mode": focus_mode,
            "primary_tag": primary_tag,
            "secondary_tag": secondary_tag,
            "circulation_priority": "medium",
            "center_open_preference": "medium",
            "support_cluster_behavior": "balanced",
            "distribution_mode": "balanced",
            "forge_guidance": [
                f"Treat {primary_tag} as the primary planning anchor.",
                "Keep support clusters coherent without collapsing the whole room into one edge.",
            ],
            "composer_guidance": [
                "Prefer practical, centered local relationships around anchors.",
            ],
            "notes": ["Balanced baseline intent."],
        },
        {
            "intent_id": "intent_2",
            "label": "Open Center",
            "summary": "Airier perimeter-biased layout with a cleaner center lane.",
            "focus_mode": focus_mode,
            "primary_tag": primary_tag,
            "secondary_tag": secondary_tag,
            "circulation_priority": "high",
            "center_open_preference": "high",
            "support_cluster_behavior": "recede",
            "distribution_mode": "edge_weighted",
            "forge_guidance": [
                "Favor perimeter-support behavior and leave the center visually calmer.",
                "Reduce bulky support clustering near the primary activity band.",
            ],
            "composer_guidance": [
                "Prefer compact local clusters that preserve external circulation.",
            ],
            "notes": ["Airier branch with stronger open-center bias."],
        },
        {
            "intent_id": "intent_3",
            "label": "Focal Group",
            "summary": "Tighter focal grouping around the primary use-case.",
            "focus_mode": focus_mode,
            "primary_tag": primary_tag,
            "secondary_tag": secondary_tag,
            "circulation_priority": "medium",
            "center_open_preference": "low",
            "support_cluster_behavior": "integrate",
            "distribution_mode": "focal_grouped",
            "forge_guidance": [
                "Group the primary and supporting functions more tightly around the focal activity.",
                "Allow support pieces to integrate if they strengthen the focal zone.",
            ],
            "composer_guidance": [
                "Favor inward-facing local relationships when functionally meaningful.",
            ],
            "notes": ["More grouped and focal intent."],
        },
        {
            "intent_id": "intent_4",
            "label": "Zoned Split",
            "summary": "Separate major functions into clearer zones.",
            "focus_mode": focus_mode,
            "primary_tag": primary_tag,
            "secondary_tag": secondary_tag,
            "circulation_priority": "high",
            "center_open_preference": "medium",
            "support_cluster_behavior": "recede",
            "distribution_mode": "zoned",
            "forge_guidance": [
                "Separate primary and support functions into clearer macro zones.",
                "Avoid mixing storage-like support directly into the main focal band.",
            ],
            "composer_guidance": [
                "Favor crisp anchor-support relationships over sprawling local layouts.",
            ],
            "notes": ["Zoned branch with stronger separation."],
        },
        {
            "intent_id": "intent_5",
            "label": "Support Weighted",
            "summary": "Keep the primary function clear while organizing support more intentionally.",
            "focus_mode": focus_mode,
            "primary_tag": primary_tag,
            "secondary_tag": secondary_tag or "storage",
            "circulation_priority": "medium",
            "center_open_preference": "medium",
            "support_cluster_behavior": "recede",
            "distribution_mode": "edge_weighted",
            "forge_guidance": [
                "Promote clearer secondary/support organization without competing with the main use-zone.",
                "Let support clusters take cleaner edge or recess positions.",
            ],
            "composer_guidance": [
                "Prefer tidy, support-like local layouts that do not overgrow their anchor.",
            ],
            "notes": ["Support-weighted branch."],
        },
        {
            "intent_id": "intent_6",
            "label": "Conversation Perimeter",
            "summary": "Perimeter seating with a stronger social-facing bias.",
            "focus_mode": conversation_mode,
            "primary_tag": primary_tag,
            "secondary_tag": support_tag,
            "circulation_priority": "high",
            "center_open_preference": "low",
            "support_cluster_behavior": "integrate",
            "distribution_mode": "edge_weighted",
            "forge_guidance": [
                "Favor social-facing anchors along the perimeter while preserving a clear arrival lane.",
                "Keep secondary support visible but subordinate to the primary social band.",
            ],
            "composer_guidance": [
                "Prefer inward-facing lounge relationships around the main seating anchor.",
            ],
            "notes": ["Conversation-led perimeter branch."],
        },
        {
            "intent_id": "intent_7",
            "label": "Display Spine",
            "summary": "Promote a cleaner focal spine with stronger display hierarchy.",
            "focus_mode": display_mode,
            "primary_tag": primary_tag,
            "secondary_tag": alternate_tag,
            "circulation_priority": "medium",
            "center_open_preference": "high",
            "support_cluster_behavior": "balanced",
            "distribution_mode": "zoned",
            "forge_guidance": [
                "Prioritize a legible focal spine and keep visual clutter away from it.",
                "Let display-like support align to the same macro direction as the primary focal anchor.",
            ],
            "composer_guidance": [
                "Prefer aligned anchor-support compositions over scattered accessory placement.",
            ],
            "notes": ["Display-oriented zoning branch."],
        },
        {
            "intent_id": "intent_8",
            "label": "Compact Utility",
            "summary": "Compress support functions to preserve more contiguous usable area.",
            "focus_mode": work_mode,
            "primary_tag": primary_tag,
            "secondary_tag": support_tag,
            "circulation_priority": "low",
            "center_open_preference": "high",
            "support_cluster_behavior": "recede",
            "distribution_mode": "balanced",
            "forge_guidance": [
                "Compress secondary utility clusters so the primary zone can breathe.",
                "Prefer cleaner edge packing over broad support sprawl.",
            ],
            "composer_guidance": [
                "Favor compact docking relationships that preserve front usability.",
            ],
            "notes": ["Compact utility-first branch."],
        },
        {
            "intent_id": "intent_9",
            "label": "Quiet Retreat",
            "summary": "Soften the layout into a calmer retreat with reduced visual competition.",
            "focus_mode": rest_mode,
            "primary_tag": primary_tag,
            "secondary_tag": support_tag,
            "circulation_priority": "medium",
            "center_open_preference": "low",
            "support_cluster_behavior": "recede",
            "distribution_mode": "focal_grouped",
            "forge_guidance": [
                "Reduce competition around the primary anchor and keep support visually quieter.",
                "Group related pieces more tightly so the room feels calmer and less scattered.",
            ],
            "composer_guidance": [
                "Favor softer inward-facing support around the primary anchor.",
            ],
            "notes": ["Calmer retreat-style branch."],
        },
        {
            "intent_id": "intent_10",
            "label": "Dual Zone Contrast",
            "summary": "Create a stronger contrast between primary and secondary activity zones.",
            "focus_mode": focus_mode,
            "primary_tag": primary_tag,
            "secondary_tag": alternate_tag,
            "circulation_priority": "low",
            "center_open_preference": "medium",
            "support_cluster_behavior": "integrate",
            "distribution_mode": "zoned",
            "forge_guidance": [
                "Push the primary and secondary functions into clearly contrasting macro zones.",
                "Allow the support zone to have its own identity as long as the primary zone stays legible.",
            ],
            "composer_guidance": [
                "Prefer distinct local poses that reinforce a two-zone reading.",
            ],
            "notes": ["Dual-zone contrast branch."],
        },
    ]
    return templates


def _infer_primary_roles(seed_text: str, room_type: str) -> tuple[str, str | None, str]:
    room_type_lower = room_type.lower()
    if "bed" in room_type_lower or "sleep" in seed_text:
        secondary = (
            "storage" if "storage" in seed_text or "wardrobe" in seed_text else "living"
        )
        return "sleep", secondary, "rest"
    if "office" in room_type_lower or "work" in seed_text or "desk" in seed_text:
        return "work", "storage", "work"
    if "dining" in room_type_lower:
        return "dining", "storage", "dining"
    if any(token in seed_text for token in ("tv", "screen", "media", "viewing")):
        return "living", "storage", "viewing"
    if (
        "living" in room_type_lower
        or "sofa" in seed_text
        or "conversation" in seed_text
    ):
        return "living", "storage", "conversation"
    return "living", "storage", "mixed"


def _extract_room_type(room_model_json: dict[str, Any]) -> str:
    meta = room_model_json.get("meta")
    if isinstance(meta, dict):
        room_type = meta.get("room_type")
        if isinstance(room_type, str) and room_type.strip():
            return room_type.strip()
    return "room"
