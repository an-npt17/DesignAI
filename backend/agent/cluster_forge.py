from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass

from agent.semantic_layout_planner import (
    SemanticLayoutPlanner,
    _normalize_profile_semantic_program,
    semantic_program_to_cluster_forge_payload,
    summarize_room_affordance,
)
from agent.request_contract import attach_request_contract_to_semantic_program
from agent_schema.clusterF_schema import ClusterForgeOutput
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


from layout.grid_policy import GLOBAL_LAYOUT_GRID_MM, normalize_cluster_rules_grid
from layout.semantic_roles import (
    is_bed_like,
    is_bedside_support_like,
    is_bench_like,
    is_lounge_anchor_like,
    is_seat_like,
    is_surface_side_accessory_like,
    is_work_surface_like,
)
from prompt.cluster_forge import CLUSTER_FORGE_PROMPT
from prompt.system import SYSTEM_PROMPT
from stylist.room_selection_rules import get_room_selection_rule
from stylist.style_policy import (
    apply_style_policy_to_semantic_program,
    compile_style_policy,
)

logger = logging.getLogger(__name__)
_FORGE_RESPONSE_MIME_TYPE = "application/json"


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
class ClusterForge:
    system_prompt: str = SYSTEM_PROMPT
    prompt_template: str = CLUSTER_FORGE_PROMPT

    def build_input_json(
        self,
        room_type: str,
        room_rules: Mapping[str, Mapping[str, object]] | None = None,
    ) -> list[dict[str, object]]:
        room_rule = (
            get_room_selection_rule(room_type)
            if room_rules is None
            else dict(room_rules.get(room_type, {}))
        )
        if not room_rule:
            return [{"room_type": room_type}]
        room_rule["room_type"] = room_type
        return [room_rule]

    def build_messages(
        self,
        room_type: str,
        room_rules: Mapping[str, Mapping[str, object]] | None = None,
        description: str | None = None,
        special_notes: str | None = None,
    ) -> list[ChatMessage]:
        input_json = self.build_input_json(room_type, room_rules)
        payload = json.dumps(input_json, ensure_ascii=True)
        description_block = ""
        if description:
            description_block = (
                f"USER DESCRIPTION (free text):\n{description.strip()}\n\n"
            )
        special_notes_block = ""
        if special_notes:
            special_notes_block = (
                f"USER SPECIAL NOTES (free text):\n{special_notes.strip()}\n\n"
            )
        user_prompt = (
            self.prompt_template.replace("{INPUT_JSON}", payload)
            .replace("{DESCRIPTION_BLOCK}", description_block)
            .replace("{SPECIAL_NOTES_BLOCK}", special_notes_block)
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def generate_raw(
        self,
        room_type: str,
        room_rules: Mapping[str, Mapping[str, object]] | None = None,
        description: str | None = None,
        special_notes: str | None = None,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        messages = self.build_messages(
            room_type,
            room_rules,
            description=description,
            special_notes=special_notes,
        )
        response = get_llm_client().chat_completion(
            messages,
            model_key="primary",
            model_name=TextLLMConfig.agent_model("forge"),
            temperature=temperature,
            max_tokens=max_tokens,
            response_mime_type=_FORGE_RESPONSE_MIME_TYPE,
        )
        return _extract_content(response)

    def generate(
        self,
        room_type: str,
        room_rules: Mapping[str, Mapping[str, object]] | None = None,
        description: str | None = None,
        special_notes: str | None = None,
        *,
        room_model_json: Mapping[str, object] | None = None,
        inventory_catalog: Sequence[Mapping[str, object]] | None = None,
        semantic_program_rules: Mapping[str, object] | None = None,
        style_policy_json: Mapping[str, object] | None = None,
        use_semantic_planner: bool = True,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ClusterForgeOutput:
        if use_semantic_planner and room_model_json is not None:
            brief_text = "\n".join(
                item
                for item in (description or "", special_notes or "")
                if item.strip()
            )
            style_policy = (
                dict(style_policy_json)
                if isinstance(style_policy_json, Mapping)
                else compile_style_policy(
                    room_type=room_type,
                    brief_text=brief_text,
                    room_model_json=room_model_json,
                    semantic_program_rules=semantic_program_rules,
                )
            )
            semantic_program = SemanticLayoutPlanner().generate(
                room_model_json=room_model_json,
                room_type=room_type,
                brief_text=brief_text,
                inventory_catalog=inventory_catalog,
                semantic_program_rules=semantic_program_rules,
                style_policy=style_policy,
                use_llm=True,
                temperature=0.2,
                top_p=0.9,
                max_tokens=max_tokens,
            )
            semantic_payload = apply_style_policy_to_semantic_program(
                semantic_program.model_dump(),
                style_policy,
            )
            semantic_payload = attach_request_contract_to_semantic_program(
                semantic_payload,
                brief_text=brief_text,
            )
            semantic_payload = _normalize_profile_semantic_program(
                semantic_payload,
                affordance_summary=summarize_room_affordance(room_model_json),
            )
            payload = semantic_program_to_cluster_forge_payload(semantic_payload)
            payload["style_policy"] = style_policy
            payload = _normalize_cluster_forge_payload(payload)
            return ClusterForgeOutput.model_validate(payload)

        client = get_llm_client()
        model_name = TextLLMConfig.agent_model("forge")
        messages = self.build_messages(
            room_type,
            room_rules,
            description=description,
            special_notes=special_notes,
        )
        last_raw = ""
        for attempt in range(2):
            response = client.chat_completion(
                messages,
                model_key="primary",
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                response_mime_type=_FORGE_RESPONSE_MIME_TYPE,
            )
            last_raw = _extract_content(response)
            try:
                payload = _parse_json(
                    last_raw,
                    finish_reason=_extract_finish_reason(response),
                )
                payload = _attach_request_contract_to_payload(
                    payload,
                    brief_text="\n".join(
                        item
                        for item in (description or "", special_notes or "")
                        if item.strip()
                    ),
                )
                payload = _normalize_cluster_forge_payload(payload)
                return ClusterForgeOutput.model_validate(payload)
            except Exception:
                if attempt == 1:
                    raise
                _record_llm_retry(
                    stage="cluster_forge",
                    model_name=model_name,
                    reason="invalid_json_or_schema",
                )
                messages.append({"role": "assistant", "content": last_raw})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Return exactly one valid JSON object that satisfies "
                            "the ClusterForge schema. No markdown. No prose."
                        ),
                    }
                )
        raise ValueError("ClusterForge returned invalid JSON")


def _extract_content(response: object) -> str:
    choices = getattr(response, "choices", None)
    if isinstance(choices, Sequence) and choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
    raise ValueError("OpenAI response missing message content")


def _parse_json(raw: str, *, finish_reason: str | None = None) -> dict[str, object]:
    text = _coerce_json_text(raw)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        logger.warning(
            "ClusterForge JSON parse failed on initial payload finish_reason=%s preview=%s",
            finish_reason,
            _truncate_text(text),
        )
        text = _extract_json_object(text)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning(
                "ClusterForge JSON parse failed after object extraction finish_reason=%s preview=%s",
                finish_reason,
                _truncate_text(text),
            )
            raise ValueError("ClusterForge returned invalid JSON") from exc
    if not isinstance(payload, dict):
        logger.warning(
            "ClusterForge response parsed but was not an object: %s",
            type(payload).__name__,
        )
        raise ValueError("ClusterForge response must be a JSON object")
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


def _extract_finish_reason(response: object) -> str | None:
    choices = getattr(response, "choices", None)
    if not isinstance(choices, Sequence) or not choices:
        return None
    finish_reason = getattr(choices[0], "finish_reason", None)
    return finish_reason if isinstance(finish_reason, str) else None


def _attach_request_contract_to_payload(
    payload: dict[str, object],
    *,
    brief_text: str,
) -> dict[str, object]:
    semantic_program = payload.get("semantic_layout_program")
    if not isinstance(semantic_program, Mapping):
        return payload
    out = dict(payload)
    out["semantic_layout_program"] = attach_request_contract_to_semantic_program(
        semantic_program,
        brief_text=brief_text,
    )
    return out


def _truncate_text(text: str, limit: int = 600) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3]}..."


def _normalize_cluster_forge_payload(payload: dict[str, object]) -> dict[str, object]:
    if not isinstance(payload, dict):
        return payload
    payload["notes"] = _coerce_str_list(payload.get("notes"))
    payload["missing"] = _coerce_str_list(payload.get("missing"))

    clusters = payload.get("clusters")
    if not isinstance(clusters, list):
        payload["clusters"] = []
        return payload

    semantic_anchor_candidates = _dominant_anchor_candidates_by_cluster(
        payload.get("semantic_layout_program")
    )
    room_type = _cluster_forge_room_type(payload)
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        cluster_id = cluster.get("cluster_id")
        cluster["cluster_rules"] = normalize_cluster_rules_grid(
            cluster.get("cluster_rules")
        )
        cluster["notes"] = _coerce_str_list(cluster.get("notes"))
        cluster["members"] = _coerce_str_list(cluster.get("members"))
        cluster["anchors"] = _coerce_str_list(cluster.get("anchors"))
        hard = cluster.get("hard_constraints")
        if not isinstance(hard, list):
            cluster["hard_constraints"] = []
        else:
            normalized_hard: list[dict[str, object]] = []
            for constraint in hard:
                normalized_constraint = _normalize_hard_constraint(constraint)
                if normalized_constraint is None:
                    _append_payload_note(
                        payload,
                        f"Dropped unsupported hard constraint: {constraint!r}",
                    )
                    continue
                normalized_hard.append(normalized_constraint)
            cluster["hard_constraints"] = normalized_hard
        soft = cluster.get("soft_constraints")
        if not isinstance(soft, list):
            cluster["soft_constraints"] = []
        else:
            normalized_soft: list[dict[str, object]] = []
            for constraint in soft:
                normalized_constraint = _normalize_soft_constraint(constraint)
                if normalized_constraint is None:
                    _append_payload_note(
                        payload,
                        f"Dropped unsupported soft constraint: {constraint!r}",
                    )
                    continue
                normalized_soft.append(normalized_constraint)
            cluster["soft_constraints"] = normalized_soft
        _sanitize_cluster_constraints(
            payload=payload,
            cluster_id=cluster_id if isinstance(cluster_id, str) else "",
            cluster=cluster,
            semantic_anchor_candidates=semantic_anchor_candidates,
        )
        _relax_core_social_wall_attachment(
            payload=payload,
            room_type=room_type,
            cluster_id=cluster_id if isinstance(cluster_id, str) else "",
            cluster=cluster,
        )
        _finalize_object_level_solver_contract(
            payload=payload,
            room_type=room_type,
            cluster_id=cluster_id if isinstance(cluster_id, str) else "",
            cluster=cluster,
        )
    payload.setdefault("planner_kind", "semantic_cluster_program")
    payload.setdefault("layout_flow", "object_level_anchor_first")
    return payload


def _cluster_forge_room_type(payload: Mapping[str, object] | None) -> str:
    if not isinstance(payload, Mapping):
        return ""
    semantic_program = payload.get("semantic_layout_program")
    if isinstance(semantic_program, Mapping):
        value = str(semantic_program.get("room_type") or "").strip()
        if value:
            return value
    return str(payload.get("room_type") or "").strip()


def _relax_core_social_wall_attachment(
    *,
    payload: dict[str, object],
    room_type: str,
    cluster_id: str,
    cluster: dict[str, object],
) -> None:
    if room_type != "living_room":
        return
    members = _coerce_str_list(cluster.get("members"))
    if not members:
        return
    cluster_rules = cluster.get("cluster_rules")
    if not isinstance(cluster_rules, dict):
        return
    has_lounge_anchor = any(is_lounge_anchor_like(member) for member in members)
    if not has_lounge_anchor:
        return
    zone_claims = cluster_rules.get("zone_claims")
    if not isinstance(zone_claims, dict):
        zone_claims = {}
        cluster_rules["zone_claims"] = zone_claims

    wall_affinity = str(zone_claims.get("wall_affinity") or "").strip().lower()
    if _cluster_targets_media(payload=payload, cluster_id=cluster_id):
        zone_claims["wall_affinity"] = "low"
    elif wall_affinity == "high":
        zone_claims["wall_affinity"] = "medium"
    elif not wall_affinity:
        zone_claims["wall_affinity"] = "medium"

    zone_claims["daylight_affinity"] = "high"

    if not bool(zone_claims.get("floating_allowed")):
        zone_claims["floating_allowed"] = True

    preferred_regions = _coerce_str_list(zone_claims.get("preferred_regions"))
    preferred_regions = [
        region for region in preferred_regions if region != "floating_support_zone"
    ]
    preferred_regions.insert(0, "floating_support_zone")
    zone_claims["preferred_regions"] = preferred_regions

    avoid_regions = [
        region
        for region in _coerce_str_list(zone_claims.get("avoid_regions"))
        if region != "center_openness_core"
    ]
    zone_claims["avoid_regions"] = avoid_regions
    _append_payload_note(
        payload,
        f"Cluster {cluster_id}: relaxed wall attachment for core social cluster to allow floating exploration.",
    )


def _cluster_targets_media(*, payload: Mapping[str, object], cluster_id: str) -> bool:
    if not cluster_id:
        return False
    semantic_program = payload.get("semantic_layout_program")
    if not isinstance(semantic_program, Mapping):
        return False
    active_clusters = semantic_program.get("active_clusters")
    if not isinstance(active_clusters, Sequence) or isinstance(active_clusters, str):
        active_clusters = ()
    for semantic_cluster in active_clusters:
        if not isinstance(semantic_cluster, Mapping):
            continue
        if str(semantic_cluster.get("cluster_id") or "").strip() != cluster_id:
            continue
        relation_intents = semantic_cluster.get("relation_intents")
        if not isinstance(relation_intents, Sequence) or isinstance(
            relation_intents, str
        ):
            continue
        for intent in relation_intents:
            if not isinstance(intent, Mapping):
                continue
            target = str(
                intent.get("target_cluster")
                or intent.get("target_cluster_id")
                or intent.get("target")
                or ""
            ).strip()
            if target == "media":
                return True
    for key in ("orientation_preferences", "adjacency_preferences"):
        preferences = semantic_program.get(key)
        if not isinstance(preferences, Sequence) or isinstance(preferences, str):
            continue
        for preference in preferences:
            if not isinstance(preference, Mapping):
                continue
            left = str(
                preference.get("a")
                or preference.get("cluster_a")
                or preference.get("source_cluster")
                or ""
            ).strip()
            right = str(
                preference.get("b")
                or preference.get("cluster_b")
                or preference.get("target_cluster")
                or ""
            ).strip()
            if (left, right) in {(cluster_id, "media"), ("media", cluster_id)}:
                return True
    return False


def _coerce_str_list(value: object) -> list[str]:
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            text = item.strip()
            if text:
                out.append(text)
        return out
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return []


def _normalize_soft_constraint(constraint: object) -> dict[str, object] | None:
    if not isinstance(constraint, dict):
        return None

    normalized_constraint = dict(constraint)
    ctype = normalized_constraint.get("type")
    if not isinstance(ctype, str):
        return None

    normalized_type = ctype.strip().lower()
    normalized_constraint["type"] = normalized_type
    if normalized_type == "prefer_facing":
        mode = _normalize_prefer_facing_mode(normalized_constraint.get("mode"))
        if mode is None:
            return None
        normalized_constraint["mode"] = mode
        normalized_constraint["weight"] = _normalize_constraint_weight(
            normalized_constraint.get("weight")
        )
        return normalized_constraint

    if normalized_type == "prefer_near":
        normalized_constraint["weight"] = _normalize_constraint_weight(
            normalized_constraint.get("weight")
        )
        return normalized_constraint

    if normalized_type == "prefer_align_edge":
        edge = _normalize_align_edge_token(normalized_constraint.get("edge"))
        if edge is None:
            return None
        normalized_constraint["edge"] = edge
        normalized_constraint["weight"] = _normalize_constraint_weight(
            normalized_constraint.get("weight")
        )
        return normalized_constraint

    return None


def _normalize_hard_constraint(constraint: object) -> dict[str, object] | None:
    if not isinstance(constraint, dict):
        return None

    normalized_constraint = dict(constraint)
    ctype = normalized_constraint.get("type")
    if not isinstance(ctype, str):
        return None

    normalized_type = ctype.strip().lower()
    normalized_constraint["type"] = normalized_type

    if normalized_type in {"no_overlap", "contain_in"}:
        return normalized_constraint

    if normalized_type == "anchor_side":
        side = _normalize_anchor_side_token(normalized_constraint.get("side"))
        if side is None:
            return None
        gap_min, gap_max = _normalize_gap_range(
            normalized_constraint.get("gap_min"),
            normalized_constraint.get("gap_max"),
        )
        normalized_constraint["side"] = side
        normalized_constraint["gap_min"] = gap_min
        normalized_constraint["gap_max"] = gap_max
        return normalized_constraint

    if normalized_type == "dock_to_edge":
        b_edge = _normalize_dock_edge_token(normalized_constraint.get("b_edge"))
        if b_edge is None:
            return None
        span = _normalize_dock_span_token(normalized_constraint.get("span"))
        gap_min, gap_max = _normalize_gap_range(
            normalized_constraint.get("gap_min"),
            normalized_constraint.get("gap_max"),
        )
        normalized_constraint["b_edge"] = b_edge
        normalized_constraint["span"] = span
        normalized_constraint["gap_min"] = gap_min
        normalized_constraint["gap_max"] = gap_max
        return normalized_constraint

    if normalized_type == "requires_access":
        obj_id = normalized_constraint.get("id")
        if not isinstance(obj_id, str) or not obj_id.strip():
            fallback_id = normalized_constraint.get("a")
            if isinstance(fallback_id, str) and fallback_id.strip():
                normalized_constraint["id"] = fallback_id.strip()
        mode = normalized_constraint.get("mode")
        if not isinstance(mode, str) or not mode.strip():
            normalized_constraint["mode"] = "front_clearance"
        return normalized_constraint

    return None


def _normalize_gap_range(gap_min: object, gap_max: object) -> tuple[int, int]:
    min_value = _coerce_int(gap_min, default=0)
    max_value = _coerce_int(gap_max, default=min_value)
    if max_value < min_value:
        min_value, max_value = max_value, min_value
    return min_value, max_value


def _normalize_constraint_weight(value: object) -> int:
    weight = _coerce_int(value, default=5)
    return max(1, weight)


def _normalize_anchor_side_token(value: object) -> str | None:
    if not isinstance(value, str):
        return None
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
    token = mapping.get(token, token)
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
        return token
    return None


def _normalize_dock_edge_token(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    token = value.strip().lower()
    mapping = {
        "head": "front",
        "foot": "back",
        "top": "top",
        "bottom": "bottom",
        "front_left": "left",
        "front_right": "right",
        "back_left": "left",
        "back_right": "right",
        "head_left": "left",
        "head_right": "right",
        "foot_left": "left",
        "foot_right": "right",
    }
    token = mapping.get(token, token)
    if token in {"front", "back", "left", "right", "top", "bottom"}:
        return token
    return None


def _normalize_dock_span_token(value: object) -> str:
    if not isinstance(value, str):
        return "any"
    token = value.strip().lower()
    if token in {"any", "center", "left", "right", "short_edge", "long_edge"}:
        return token
    if token in {"middle", "mid"}:
        return "center"
    return "any"


def _normalize_align_edge_token(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    token = value.strip().lower()
    mapping = {
        "front_left": "left",
        "front_right": "right",
        "back_left": "left",
        "back_right": "right",
        "head_left": "left",
        "head_right": "right",
        "foot_left": "left",
        "foot_right": "right",
    }
    token = mapping.get(token, token)
    if token in {"left", "right", "top", "bottom", "front", "back", "head", "foot"}:
        return token
    return None


def _coerce_int(value: object, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    if isinstance(value, str):
        try:
            return int(round(float(value.strip())))
        except ValueError:
            return default
    return default


def _normalize_prefer_facing_mode(value: object) -> str | None:
    if value is None:
        return "face_same_direction"
    if not isinstance(value, str):
        return None

    normalized_value = value.strip().lower()
    if normalized_value in {
        "face_each_other",
        "face_eachother",
        "each_other",
        "towards",
        "face_each",
    }:
        return "face_each_other"
    if normalized_value in {
        "face_same_direction",
        "same_direction",
        "same_dir",
        "aligned",
        "same",
    }:
        return "face_same_direction"
    return None


def _normalize_facing_front_token(value: object) -> str | None:
    if not isinstance(value, str):
        return None

    token = value.strip().lower()
    mapping = {
        "front": "top",
        "back": "bottom",
        "head": "top",
        "foot": "bottom",
        "up": "top",
        "down": "bottom",
        "north": "top",
        "south": "bottom",
        "west": "left",
        "east": "right",
    }
    token = mapping.get(token, token)
    if token in {"top", "bottom", "left", "right"}:
        return token
    return None


def _finalize_object_level_solver_contract(
    *,
    payload: dict[str, object],
    room_type: str,
    cluster_id: str,
    cluster: dict[str, object],
) -> None:
    members = _coerce_str_list(cluster.get("members"))
    if not members:
        return
    rules = cluster.get("cluster_rules")
    if not isinstance(rules, dict):
        rules = {}
        cluster["cluster_rules"] = rules

    dominant_anchor_candidates = _coerce_str_list(
        rules.get("dominant_anchor_candidates")
    )
    anchors = [
        anchor
        for anchor in _coerce_str_list(cluster.get("anchors"))
        if anchor in members
    ]
    if not anchors and dominant_anchor_candidates:
        anchors = [dominant_anchor_candidates[0]]
        cluster["anchors"] = anchors

    semantic_placements = rules.get("semantic_placements")
    if not isinstance(semantic_placements, list):
        semantic_placements = []
        rules["semantic_placements"] = semantic_placements

    anchor_policy = rules.get("anchor_first_policy")
    if not isinstance(anchor_policy, dict):
        anchor_policy = {}
        rules["anchor_first_policy"] = anchor_policy

    dominant_anchor_id = str(
        anchor_policy.get("dominant_anchor_id")
        or (anchors[0] if anchors else "")
        or (dominant_anchor_candidates[0] if dominant_anchor_candidates else "")
    ).strip()
    if dominant_anchor_id and dominant_anchor_id in members:
        anchor_policy["dominant_anchor_id"] = dominant_anchor_id

    filtered_candidates = [
        candidate for candidate in dominant_anchor_candidates if candidate in members
    ]
    if not filtered_candidates and dominant_anchor_id:
        filtered_candidates = [dominant_anchor_id]
    rules["dominant_anchor_candidates"] = filtered_candidates
    anchor_policy["dominant_anchor_candidates"] = filtered_candidates

    placement_order = [
        item
        for item in _coerce_str_list(anchor_policy.get("placement_order"))
        if item in members
    ]
    if dominant_anchor_id:
        placement_order = [
            item for item in placement_order if item != dominant_anchor_id
        ]
        placement_order.insert(0, dominant_anchor_id)
    for member in members:
        if member not in placement_order:
            placement_order.append(member)
    anchor_policy["placement_order"] = placement_order

    support_chain: list[dict[str, object]] = []
    for row in semantic_placements:
        if not isinstance(row, Mapping):
            continue
        object_id = row.get("id")
        relative_to = row.get("relative_to")
        if not isinstance(object_id, str) or object_id not in members:
            continue
        if not isinstance(relative_to, str) or relative_to not in members:
            continue
        record = {
            "object_id": object_id,
            "relative_to": relative_to,
            "support_role": str(row.get("support_role") or "").strip().lower(),
            "band_intent": str(row.get("band_intent") or "").strip().lower(),
            "orientation": str(row.get("orientation") or "").strip().lower(),
            "kind": str(row.get("kind") or "").strip().lower(),
        }
        support_chain.append(record)
    anchor_policy["support_chain"] = support_chain

    protected_ids = [
        item
        for item in _coerce_str_list(anchor_policy.get("protected_ids"))
        if item in members
    ]
    if dominant_anchor_id and dominant_anchor_id not in protected_ids:
        protected_ids.insert(0, dominant_anchor_id)
    for row in support_chain:
        support_role = str(row.get("support_role") or "")
        band_intent = str(row.get("band_intent") or "")
        object_id = str(row.get("object_id") or "")
        if not object_id:
            continue
        if support_role == "frontal_support" or band_intent == "front_band":
            if object_id not in protected_ids:
                protected_ids.append(object_id)
    anchor_policy["protected_ids"] = _unique_str_list(protected_ids)

    droppable_ids = [
        item
        for item in _coerce_str_list(anchor_policy.get("droppable_ids"))
        if item in members and item not in protected_ids
    ]
    anchor_policy["droppable_ids"] = _unique_str_list(droppable_ids)

    object_solver_policy = rules.get("object_solver_policy")
    if not isinstance(object_solver_policy, dict):
        object_solver_policy = {}
    object_solver_policy.update(
        {
            "enabled": True,
            "layout_flow": "object_level_anchor_first",
            "solve_level": "object",
            "anchor_first": True,
            "support_strategy": "relative_to_graph",
            "local_cluster_composer_enabled": False,
            "cluster_outline_enabled": False,
            "phase2_controller_enabled": False,
            "degrade_inside_solver": True,
            "room_type": room_type,
            "cluster_id": cluster_id,
        }
    )
    rules["object_solver_policy"] = object_solver_policy

    cluster["object_level_contract"] = {
        "dominant_anchor_id": anchor_policy.get("dominant_anchor_id"),
        "dominant_anchor_candidates": deepcopy(
            anchor_policy.get("dominant_anchor_candidates") or []
        ),
        "placement_order": deepcopy(anchor_policy.get("placement_order") or []),
        "protected_ids": deepcopy(anchor_policy.get("protected_ids") or []),
        "droppable_ids": deepcopy(anchor_policy.get("droppable_ids") or []),
        "support_edge_count": len(support_chain),
        "solver_mode": "object_level_anchor_first",
    }


def _object_prefers_explicit_facing(object_id: str) -> bool:
    key = object_id.lower()
    if (
        is_seat_like(object_id)
        or is_work_surface_like(object_id)
        or is_lounge_anchor_like(object_id)
    ):
        return True
    return any(
        pattern in key
        for pattern in (
            "wardrobe",
            "closet",
            "dresser",
            "cabinet",
            "console",
            "tv",
            "vanity",
            "storage",
        )
    )


def _default_facing_front(object_id: str) -> str:
    _ = object_id
    return "top"


def _normalize_facing_rules(
    *,
    payload: dict[str, object],
    cluster_id: str,
    members: set[str],
    facing: object,
) -> dict[str, dict[str, str]]:
    if not isinstance(facing, Mapping):
        return {}

    normalized: dict[str, dict[str, str]] = {}
    for obj_id, raw_rule in facing.items():
        if not isinstance(obj_id, str) or obj_id not in members:
            _append_payload_note(
                payload,
                f"Cluster {cluster_id}: dropped facing rule for unknown object {obj_id!r}.",
            )
            continue

        rule = raw_rule if isinstance(raw_rule, Mapping) else {}
        front = _normalize_facing_front_token(rule.get("front"))
        if front is None:
            if _object_prefers_explicit_facing(obj_id):
                front = _default_facing_front(obj_id)
                _append_payload_note(
                    payload,
                    f"Cluster {cluster_id}: defaulted ambiguous facing.front for {obj_id} to {front}.",
                )
            else:
                _append_payload_note(
                    payload,
                    f"Cluster {cluster_id}: dropped facing rule for {obj_id} because it has no stable front side.",
                )
                continue

        normalized_rule: dict[str, str] = {"front": front}
        notes = rule.get("notes")
        if isinstance(notes, str):
            note_text = notes.strip()
            if note_text:
                normalized_rule["notes"] = note_text
        normalized[obj_id] = normalized_rule
    return normalized


def _append_payload_note(payload: dict[str, object], note: str) -> None:
    notes = payload.get("notes")
    if not isinstance(notes, list):
        notes = []
        payload["notes"] = notes
    if note not in notes:
        notes.append(note)


def _dominant_anchor_candidates_by_cluster(
    semantic_layout_program: object,
) -> dict[str, list[str]]:
    if not isinstance(semantic_layout_program, Mapping):
        return {}
    out: dict[str, list[str]] = {}
    clusters = semantic_layout_program.get("active_clusters")
    if not isinstance(clusters, Sequence):
        return out
    for cluster in clusters:
        if not isinstance(cluster, Mapping):
            continue
        cluster_id = cluster.get("cluster_id")
        if not isinstance(cluster_id, str) or not cluster_id.strip():
            continue
        candidates = _coerce_str_list(cluster.get("dominant_anchor_candidates"))
        if candidates:
            out[cluster_id.strip()] = _unique_str_list(candidates)
    return out


def _unique_str_list(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _dominant_anchor_candidates_for_cluster(
    *,
    cluster_id: str,
    cluster: Mapping[str, object],
    semantic_anchor_candidates: Mapping[str, Sequence[str]],
) -> list[str]:
    cluster_rules = cluster.get("cluster_rules")
    rule_candidates = (
        _coerce_str_list(cluster_rules.get("dominant_anchor_candidates"))
        if isinstance(cluster_rules, Mapping)
        else []
    )
    semantic_candidates = list(semantic_anchor_candidates.get(cluster_id, []))
    return _unique_str_list(rule_candidates + semantic_candidates)


def _sanitize_cluster_constraints(
    *,
    payload: dict[str, object],
    cluster_id: str,
    cluster: dict[str, object],
    semantic_anchor_candidates: Mapping[str, Sequence[str]],
) -> None:
    member_list = [
        item
        for item in _coerce_str_list(cluster.get("members"))
        if isinstance(item, str) and item
    ]
    cluster["members"] = member_list
    members = set(member_list)
    cluster_rules = cluster.get("cluster_rules")
    anchor_candidates = _dominant_anchor_candidates_for_cluster(
        cluster_id=cluster_id,
        cluster=cluster,
        semantic_anchor_candidates=semantic_anchor_candidates,
    )
    if anchor_candidates and isinstance(cluster_rules, dict):
        cluster_rules["dominant_anchor_candidates"] = anchor_candidates
    cluster["anchors"] = _sanitize_cluster_anchors(
        payload=payload,
        cluster_id=cluster_id,
        cluster_tag=str(cluster.get("tag") or "").lower(),
        members=member_list,
        anchors=cluster.get("anchors"),
        anchor_candidates=anchor_candidates,
    )
    grid_mm = GLOBAL_LAYOUT_GRID_MM
    if isinstance(cluster_rules, Mapping):
        grid_mm = _coerce_int(
            cluster_rules.get("grid_mm"),
            default=GLOBAL_LAYOUT_GRID_MM,
        )
    if isinstance(cluster_rules, dict):
        allowed_rotations = cluster_rules.get("allowed_rotations")
        if isinstance(allowed_rotations, Mapping):
            cluster_rules["allowed_rotations"] = {
                obj_id: rotations
                for obj_id, rotations in allowed_rotations.items()
                if isinstance(obj_id, str) and obj_id in members
            }
        cluster_rules["facing"] = _normalize_facing_rules(
            payload=payload,
            cluster_id=cluster_id,
            members=members,
            facing=cluster_rules.get("facing"),
        )
        access_requirements = cluster_rules.get("access_requirements")
        if isinstance(access_requirements, list):
            sanitized_access_requirements: list[dict[str, object]] = []
            for requirement in access_requirements:
                if not isinstance(requirement, Mapping):
                    continue
                obj_id = requirement.get("id")
                if not isinstance(obj_id, str) or obj_id not in members:
                    _append_payload_note(
                        payload,
                        f"Cluster {cluster_id}: dropped access requirement for unknown object {obj_id!r}.",
                    )
                    continue
                sanitized_access_requirements.append(dict(requirement))
            cluster_rules["access_requirements"] = sanitized_access_requirements
        else:
            cluster_rules["access_requirements"] = []

    hard_constraints = cluster.get("hard_constraints")
    if isinstance(hard_constraints, list):
        sanitized_hard: list[dict[str, object]] = []
        for constraint in hard_constraints:
            if not isinstance(constraint, dict):
                continue
            if not _constraint_subjects_are_known(constraint, members):
                _append_payload_note(
                    payload,
                    f"Cluster {cluster_id}: dropped hard constraint with unknown subjects: {constraint!r}",
                )
                continue
            if _should_drop_object_containment(constraint):
                _append_payload_note(
                    payload,
                    f"Cluster {cluster_id}: dropped contain_in between floor objects: {constraint!r}",
                )
                continue
            normalized = dict(constraint)
            if str(normalized.get("type") or "").lower() in {
                "anchor_side",
                "dock_to_edge",
            }:
                gap_min, gap_max = _normalize_gap_range_for_grid(
                    normalized.get("gap_min"),
                    normalized.get("gap_max"),
                    grid_mm=grid_mm,
                )
                if gap_min != normalized.get("gap_min") or gap_max != normalized.get(
                    "gap_max"
                ):
                    _append_payload_note(
                        payload,
                        "Cluster "
                        f"{cluster_id}: widened {normalized.get('type')} gap window "
                        f"to fit grid {grid_mm}mm.",
                    )
                normalized["gap_min"] = gap_min
                normalized["gap_max"] = gap_max
            sanitized_hard.append(normalized)
        cluster["hard_constraints"] = sanitized_hard

    soft_constraints = cluster.get("soft_constraints")
    if isinstance(soft_constraints, list):
        sanitized_soft: list[dict[str, object]] = []
        for constraint in soft_constraints:
            if not isinstance(constraint, dict):
                continue
            if not _constraint_subjects_are_known(constraint, members):
                _append_payload_note(
                    payload,
                    f"Cluster {cluster_id}: dropped soft constraint with unknown subjects: {constraint!r}",
                )
                continue
            sanitized_soft.append(dict(constraint))
        cluster["soft_constraints"] = sanitized_soft
    _apply_semantic_layout_intents(
        payload=payload,
        cluster_id=cluster_id,
        cluster=cluster,
        members=members,
    )


def _constraint_subjects_are_known(
    constraint: Mapping[str, object],
    members: set[str],
) -> bool:
    ctype = str(constraint.get("type") or "").lower()
    if ctype == "requires_access":
        obj_id = constraint.get("id")
        return isinstance(obj_id, str) and obj_id in members

    a = constraint.get("a")
    b = constraint.get("b")
    if not isinstance(a, str) or a not in members:
        return False
    if ctype == "contain_in" and b == "room":
        return True
    return isinstance(b, str) and b in members


def _sanitize_cluster_anchors(
    *,
    payload: dict[str, object],
    cluster_id: str,
    cluster_tag: str,
    members: Sequence[str],
    anchors: object,
    anchor_candidates: Sequence[str] | None = None,
) -> list[str]:
    member_set = {member for member in members if isinstance(member, str) and member}
    raw_anchors = _coerce_str_list(anchors)
    kept = [anchor for anchor in raw_anchors if anchor in member_set]
    dropped = [anchor for anchor in raw_anchors if anchor not in member_set]
    valid_anchor_candidates = [
        candidate
        for candidate in _coerce_str_list(list(anchor_candidates or []))
        if candidate in member_set
    ]

    if dropped:
        _append_payload_note(
            payload,
            f"Cluster {cluster_id}: dropped anchors not present in members: {sorted(set(dropped))}.",
        )

    if valid_anchor_candidates:
        candidate_set = set(valid_anchor_candidates)
        invalid_semantic_anchors = [
            anchor for anchor in kept if anchor not in candidate_set
        ]
        kept = [anchor for anchor in kept if anchor in candidate_set]
        if invalid_semantic_anchors:
            _append_payload_note(
                payload,
                f"Cluster {cluster_id}: dropped anchors outside dominant anchor candidates: {sorted(set(invalid_semantic_anchors))}.",
            )

    kept = _cohere_primary_anchor_members(
        payload=payload,
        cluster_id=cluster_id,
        cluster_tag=cluster_tag,
        anchors=kept,
    )

    if kept:
        return kept

    fallback_members = valid_anchor_candidates or list(members)
    fallback = _choose_fallback_anchor(
        cluster_tag=cluster_tag,
        members=fallback_members,
    )
    if fallback is not None:
        if raw_anchors:
            note = (
                f"Cluster {cluster_id}: defaulted anchors to {fallback!r} because the original anchors "
                "were invalid for this cluster."
            )
        else:
            note = f"Cluster {cluster_id}: defaulted anchors to {fallback!r} because none were provided."
        _append_payload_note(payload, note)
        return [fallback]

    return []


def _choose_fallback_anchor(
    *,
    cluster_tag: str,
    members: Sequence[str],
) -> str | None:
    if not members:
        return None

    best_member: str | None = None
    best_score = -1
    for member in members:
        if not isinstance(member, str) or not member:
            continue
        score = _anchor_priority_score(cluster_tag=cluster_tag, member=member)
        if score > best_score:
            best_member = member
            best_score = score

    return best_member


def _anchor_priority_score(*, cluster_tag: str, member: str) -> int:
    key = member.lower()
    score = 0

    if cluster_tag == "sleep":
        if is_bed_like(member):
            score += 1_000
        elif is_bedside_support_like(member):
            score += 700
        elif is_bench_like(member):
            score += 500
    elif cluster_tag == "work":
        if is_work_surface_like(member):
            score += 1_000
        elif is_seat_like(member):
            score += 700
    elif cluster_tag == "living":
        if "sectional" in key:
            score += 1_050
        elif "sofa" in key or "loveseat" in key:
            score += 1_000
        elif is_lounge_anchor_like(member):
            score += 850
        elif is_seat_like(member):
            score += 700
    elif cluster_tag == "dining":
        if "dining_table" in key or "table" in key:
            score += 1_000
        elif is_seat_like(member):
            score += 700
    elif cluster_tag == "storage":
        if "wardrobe" in key or "closet" in key:
            score += 1_120
        elif "storage_cabinet" in key:
            score += 1_100
        elif "dresser" in key or "sideboard" in key or "buffet" in key:
            score += 1_080
        elif "media_shelf" in key or "bookshelf" in key:
            score += 1_060
        elif "tv_console" in key:
            score += 1_040
        elif "console_table" in key:
            score += 1_000
        elif "coffee_table" in key or "ottoman" in key:
            score += 300
    else:
        if is_bed_like(member):
            score += 950
        if is_work_surface_like(member):
            score += 900
        if "sofa" in key or "sectional" in key:
            score += 900
        if is_lounge_anchor_like(member):
            score += 850
        if any(token in key for token in ("wardrobe", "storage_cabinet", "dresser")):
            score += 800

    return score


def _cluster_prefers_single_anchor(cluster_tag: str) -> bool:
    return cluster_tag in {"sleep", "work", "living", "dining", "storage", "misc"}


def _cohere_primary_anchor_members(
    *,
    payload: dict[str, object],
    cluster_id: str,
    cluster_tag: str,
    anchors: Sequence[str],
) -> list[str]:
    unique_anchors: list[str] = []
    seen: set[str] = set()
    for anchor in anchors:
        if anchor in seen:
            continue
        seen.add(anchor)
        unique_anchors.append(anchor)

    if len(unique_anchors) <= 1 or not _cluster_prefers_single_anchor(cluster_tag):
        return unique_anchors

    best_anchor = max(
        unique_anchors,
        key=lambda member: _anchor_priority_score(
            cluster_tag=cluster_tag, member=member
        ),
    )
    removed = [anchor for anchor in unique_anchors if anchor != best_anchor]
    if removed:
        _append_payload_note(
            payload,
            f"Cluster {cluster_id}: kept {best_anchor!r} as the primary anchor and removed secondary anchors {removed!r} to keep downstream composition coherent.",
        )
    return [best_anchor]


def _is_media_display_like(member: str) -> bool:
    key = member.lower()
    return any(
        token in key
        for token in ("tv_console", "television", "monitor", "projector", "screen")
    )


def _is_media_side_support_like(member: str) -> bool:
    key = member.lower()
    return any(token in key for token in ("media_shelf", "bookshelf", "speaker"))


def _is_low_center_surface_like(member: str) -> bool:
    key = member.lower()
    return any(token in key for token in ("coffee_table", "ottoman", "bench"))


def _is_floor_lamp_like(member: str) -> bool:
    return "floor_lamp" in member.lower()


def _is_anchor_side_accessory_like(member: str) -> bool:
    key = member.lower()
    return (
        is_surface_side_accessory_like(member)
        or _is_floor_lamp_like(member)
        or "side_table" in key
    )


def _is_storage_anchor_like(member: str) -> bool:
    key = member.lower()
    return any(
        token in key
        for token in (
            "wardrobe",
            "closet",
            "storage_cabinet",
            "dresser",
            "sideboard",
            "buffet",
            "bookshelf",
            "media_shelf",
            "console_table",
            "tv_console",
            "cabinet",
        )
    )


def _is_storage_support_like(member: str) -> bool:
    if _is_storage_anchor_like(member):
        return True
    key = member.lower()
    return any(
        token in key
        for token in (
            "laundry_basket",
            "hamper",
            "shoe_rack",
            "basket",
            "bin",
            "organizer",
            "crate",
        )
    )


def _has_relative_layout_intent(
    *,
    cluster: Mapping[str, object],
    semantic_placements: Sequence[Mapping[str, object]],
    object_id: str,
) -> bool:
    for record in semantic_placements:
        if record.get("id") == object_id:
            return True

    hard_constraints = cluster.get("hard_constraints")
    if not isinstance(hard_constraints, Sequence):
        return False
    for constraint in hard_constraints:
        if not isinstance(constraint, Mapping):
            continue
        if constraint.get("a") != object_id:
            continue
        if str(constraint.get("type") or "").lower() in {
            "anchor_side",
            "dock_to_edge",
            "contain_in",
        }:
            return True
    return False


def _pick_related_seat_base(
    *,
    members: Sequence[str],
    primary_anchor: str,
) -> str | None:
    best_member: str | None = None
    best_score = -1
    for member in members:
        if member == primary_anchor:
            continue
        if not (is_seat_like(member) or is_lounge_anchor_like(member)):
            continue
        score = 0
        key = member.lower()
        if "sectional" in key:
            score += 1_000
        elif "sofa" in key or "loveseat" in key:
            score += 920
        elif is_lounge_anchor_like(member):
            score += 820
        elif is_seat_like(member):
            score += 700
        if score > best_score:
            best_score = score
            best_member = member
    return best_member


def _semantic_orientation_for_member(member: str) -> str | None:
    if _object_prefers_explicit_facing(member):
        return "same_direction"
    return None


def _normalize_semantic_proximity(value: object) -> str:
    if not isinstance(value, str):
        return "balanced"
    token = value.strip().lower()
    if token in {"compact", "balanced", "loose"}:
        return token
    return "balanced"


def _anchor_uses_lounge_semantics(member: str) -> bool:
    key = member.lower()
    return is_lounge_anchor_like(member) or "sofa" in key or "loveseat" in key


def _upsert_anchor_side_semantic(
    semantic_placements: list[dict[str, object]],
    *,
    object_id: str,
    relative_to: str,
    side_options: list[str],
    gap_min: int,
    gap_max: int,
    proximity: str = "balanced",
    orientation: str | None = None,
) -> None:
    record: dict[str, object] = {
        "id": object_id,
        "relative_to": relative_to,
        "kind": "anchor_side",
        "side_options": side_options,
        "gap_min": gap_min,
        "gap_max": gap_max,
        "proximity": _normalize_semantic_proximity(proximity),
        "selection": "best_fit",
    }
    support_role = _infer_semantic_support_role(
        object_id=object_id,
        relative_to=relative_to,
        kind="anchor_side",
        side_options=side_options,
    )
    if support_role is not None:
        record["support_role"] = support_role
        band_intent = _semantic_band_intent_from_role(support_role)
        if band_intent is not None:
            record["band_intent"] = band_intent
    if orientation is not None:
        record["orientation"] = orientation
    _upsert_semantic_placement(semantic_placements, record)


def _upsert_dock_semantic(
    semantic_placements: list[dict[str, object]],
    *,
    object_id: str,
    relative_to: str,
    b_edge: str,
    span: str,
    gap_min: int,
    gap_max: int,
    proximity: str = "balanced",
    orientation: str | None = None,
) -> None:
    record: dict[str, object] = {
        "id": object_id,
        "relative_to": relative_to,
        "kind": "dock_to_edge",
        "b_edge": b_edge,
        "span": span,
        "gap_min": gap_min,
        "gap_max": gap_max,
        "proximity": _normalize_semantic_proximity(proximity),
        "selection": "best_fit",
    }
    support_role = _infer_semantic_support_role(
        object_id=object_id,
        relative_to=relative_to,
        kind="dock_to_edge",
        b_edge=b_edge,
    )
    if support_role is not None:
        record["support_role"] = support_role
        band_intent = _semantic_band_intent_from_role(support_role)
        if band_intent is not None:
            record["band_intent"] = band_intent
    if orientation is not None:
        record["orientation"] = orientation
    _upsert_semantic_placement(semantic_placements, record)


def _should_drop_object_containment(constraint: Mapping[str, object]) -> bool:
    return (
        str(constraint.get("type") or "").lower() == "contain_in"
        and constraint.get("b") != "room"
    )


def _normalize_gap_range_for_grid(
    gap_min: object,
    gap_max: object,
    *,
    grid_mm: int,
) -> tuple[int, int]:
    min_value, max_value = _normalize_gap_range(gap_min, gap_max)
    snapped_grid = max(1, _coerce_int(grid_mm, default=GLOBAL_LAYOUT_GRID_MM))
    if max_value <= 0:
        return min_value, max_value
    if max_value <= snapped_grid:
        return 0, snapped_grid
    return min_value, max_value


def _apply_semantic_layout_intents(
    *,
    payload: dict[str, object],
    cluster_id: str,
    cluster: dict[str, object],
    members: set[str],
) -> None:
    rules = cluster.get("cluster_rules")
    if not isinstance(rules, dict):
        rules = {}
        cluster["cluster_rules"] = rules

    semantic_placements = _normalize_semantic_placements(
        rules.get("semantic_placements"),
        members=members,
    )

    hard_constraints = cluster.get("hard_constraints")
    if not isinstance(hard_constraints, list):
        hard_constraints = []
        cluster["hard_constraints"] = hard_constraints

    # Strengthen common serving-surface relationships so the composer can place
    # dependents in a semantically meaningful region, not merely any valid edge.
    for constraint in hard_constraints:
        if not isinstance(constraint, dict):
            continue
        if str(constraint.get("type") or "").lower() != "dock_to_edge":
            continue

        a = constraint.get("a")
        b = constraint.get("b")
        if not isinstance(a, str) or not isinstance(b, str):
            continue

        if is_seat_like(a) and is_work_surface_like(b):
            if constraint.get("span") in {"any", "short_edge", "long_edge"}:
                constraint["span"] = "center"
                _append_payload_note(
                    payload,
                    f"Cluster {cluster_id}: refined {a}->{b} dock span to center for a serving-surface seating relationship.",
                )
            _upsert_semantic_placement(
                semantic_placements,
                {
                    "id": a,
                    "relative_to": b,
                    "kind": "dock_to_edge",
                    "b_edge": str(constraint.get("b_edge") or "front"),
                    "span": str(constraint.get("span") or "center"),
                    "gap_min": _coerce_int(constraint.get("gap_min"), default=0),
                    "gap_max": _coerce_int(constraint.get("gap_max"), default=0),
                    "proximity": "compact",
                    "selection": "best_fit",
                    "orientation": "face_base",
                },
            )
            _ensure_soft_constraint(
                cluster,
                {
                    "type": "prefer_facing",
                    "a": a,
                    "b": b,
                    "mode": "face_each_other",
                    "weight": 8,
                },
            )
            continue

        if is_bench_like(a) and is_bed_like(b):
            if constraint.get("span") in {"any", "short_edge", "long_edge"}:
                constraint["span"] = "center"
                _append_payload_note(
                    payload,
                    f"Cluster {cluster_id}: refined {a}->{b} dock span to center for a foot-of-bed seating relationship.",
                )
            _upsert_semantic_placement(
                semantic_placements,
                {
                    "id": a,
                    "relative_to": b,
                    "kind": "dock_to_edge",
                    "b_edge": str(constraint.get("b_edge") or "back"),
                    "span": str(constraint.get("span") or "center"),
                    "gap_min": _coerce_int(constraint.get("gap_min"), default=0),
                    "gap_max": _coerce_int(constraint.get("gap_max"), default=0),
                    "proximity": "balanced",
                    "selection": "best_fit",
                    "orientation": "face_base",
                },
            )

    bedside_relations: list[dict[str, object]] = []
    for constraint in hard_constraints:
        if not isinstance(constraint, dict):
            continue
        if str(constraint.get("type") or "").lower() != "anchor_side":
            continue
        a = constraint.get("a")
        b = constraint.get("b")
        side = constraint.get("side")
        if (
            isinstance(a, str)
            and isinstance(b, str)
            and isinstance(side, str)
            and is_bedside_support_like(a)
            and is_bed_like(b)
            and side in {"head", "foot"}
        ):
            bedside_relations.append(constraint)

    if len(bedside_relations) >= 2:
        ordered = sorted(
            bedside_relations,
            key=lambda item: str(item.get("a") or ""),
        )
        side_tokens = ["head_left", "head_right"]
        if str(ordered[0].get("side") or "") == "foot":
            side_tokens = ["foot_left", "foot_right"]
        for constraint, side_token in zip(ordered[:2], side_tokens, strict=False):
            if constraint.get("side") != side_token:
                constraint["side"] = side_token
                _append_payload_note(
                    payload,
                    f"Cluster {cluster_id}: assigned {constraint.get('a')} to {side_token} for clearer bedside semantics.",
                )

    for constraint in bedside_relations:
        a = str(constraint.get("a") or "")
        b = str(constraint.get("b") or "")
        side = str(constraint.get("side") or "")
        if side in {"head", "foot"}:
            option_prefix = "head" if side == "head" else "foot"
            _upsert_semantic_placement(
                semantic_placements,
                {
                    "id": a,
                    "relative_to": b,
                    "kind": "anchor_side",
                    "side_options": [
                        f"{option_prefix}_left",
                        f"{option_prefix}_right",
                    ],
                    "gap_min": _coerce_int(constraint.get("gap_min"), default=0),
                    "gap_max": _coerce_int(constraint.get("gap_max"), default=0),
                    "proximity": "compact",
                    "selection": "best_fit",
                    "orientation": "same_direction",
                },
            )

    soft_constraints = cluster.get("soft_constraints")
    if isinstance(soft_constraints, list):
        for constraint in soft_constraints:
            if not isinstance(constraint, dict):
                continue
            if str(constraint.get("type") or "").lower() != "prefer_near":
                continue
            a = constraint.get("a")
            b = constraint.get("b")
            if not isinstance(a, str) or not isinstance(b, str):
                continue

            if is_surface_side_accessory_like(a) and (
                is_lounge_anchor_like(b) or is_seat_like(b)
            ):
                _upsert_semantic_placement(
                    semantic_placements,
                    {
                        "id": a,
                        "relative_to": b,
                        "kind": "anchor_side",
                        "side_options": ["left", "right"],
                        "gap_min": 50,
                        "gap_max": 200,
                        "proximity": "balanced",
                        "selection": "best_fit",
                        "orientation": "same_direction",
                    },
                )

    anchors = [
        anchor
        for anchor in _coerce_str_list(cluster.get("anchors"))
        if anchor in members
    ]
    primary_anchor = anchors[0] if anchors else None

    if primary_anchor is not None:
        support_seat_base = _pick_related_seat_base(
            members=sorted(members),
            primary_anchor=primary_anchor,
        )

        for member in sorted(members):
            if member == primary_anchor:
                continue
            if _has_relative_layout_intent(
                cluster=cluster,
                semantic_placements=semantic_placements,
                object_id=member,
            ):
                continue

            if is_bed_like(primary_anchor):
                if is_bedside_support_like(member):
                    _upsert_anchor_side_semantic(
                        semantic_placements,
                        object_id=member,
                        relative_to=primary_anchor,
                        side_options=["head_left", "head_right"],
                        gap_min=0,
                        gap_max=100,
                        proximity="compact",
                        orientation="same_direction",
                    )
                    continue
                if is_bench_like(member) or "ottoman" in member.lower():
                    _upsert_dock_semantic(
                        semantic_placements,
                        object_id=member,
                        relative_to=primary_anchor,
                        b_edge="back",
                        span="center",
                        gap_min=0,
                        gap_max=150,
                        proximity="balanced",
                        orientation="face_base",
                    )
                    continue

            if is_work_surface_like(primary_anchor):
                if is_seat_like(member):
                    _upsert_dock_semantic(
                        semantic_placements,
                        object_id=member,
                        relative_to=primary_anchor,
                        b_edge="front",
                        span="center",
                        gap_min=0,
                        gap_max=100,
                        proximity="compact",
                        orientation="face_base",
                    )
                    continue
                if _is_anchor_side_accessory_like(member):
                    _upsert_anchor_side_semantic(
                        semantic_placements,
                        object_id=member,
                        relative_to=primary_anchor,
                        side_options=["left", "right"],
                        gap_min=50,
                        gap_max=200,
                        proximity="balanced",
                        orientation=_semantic_orientation_for_member(member),
                    )
                    continue

            if _is_media_display_like(primary_anchor) and _is_media_side_support_like(
                member
            ):
                _upsert_anchor_side_semantic(
                    semantic_placements,
                    object_id=member,
                    relative_to=primary_anchor,
                    side_options=["left", "right"],
                    gap_min=0,
                    gap_max=200,
                    proximity="balanced",
                    orientation="same_direction",
                )
                continue

            if _anchor_uses_lounge_semantics(primary_anchor):
                if "coffee_table" in member.lower():
                    _upsert_anchor_side_semantic(
                        semantic_placements,
                        object_id=member,
                        relative_to=primary_anchor,
                        side_options=["head", "left", "right"],
                        gap_min=100,
                        gap_max=350,
                        proximity="compact",
                    )
                    continue
                if "ottoman" in member.lower():
                    _upsert_anchor_side_semantic(
                        semantic_placements,
                        object_id=member,
                        relative_to=primary_anchor,
                        side_options=["foot", "left", "right"],
                        gap_min=50,
                        gap_max=250,
                        proximity="balanced",
                    )
                    continue
                if is_seat_like(member):
                    _upsert_anchor_side_semantic(
                        semantic_placements,
                        object_id=member,
                        relative_to=primary_anchor,
                        side_options=["left", "right", "head_left", "head_right"],
                        gap_min=100,
                        gap_max=350,
                        proximity="loose",
                        orientation="face_base",
                    )
                    continue

            if _is_storage_anchor_like(primary_anchor) and _is_storage_support_like(
                member
            ):
                _upsert_anchor_side_semantic(
                    semantic_placements,
                    object_id=member,
                    relative_to=primary_anchor,
                    side_options=["left", "right", "head_left", "head_right"],
                    gap_min=100,
                    gap_max=450,
                    proximity="loose",
                    orientation=_semantic_orientation_for_member(member),
                )
                continue

            if _is_anchor_side_accessory_like(member):
                base_id = support_seat_base or primary_anchor
                if base_id == member:
                    base_id = primary_anchor
                _upsert_anchor_side_semantic(
                    semantic_placements,
                    object_id=member,
                    relative_to=base_id,
                    side_options=["left", "right", "head_left", "head_right"],
                    gap_min=50,
                    gap_max=200,
                    proximity="balanced",
                    orientation=_semantic_orientation_for_member(member),
                )
                continue

    rules["semantic_placements"] = semantic_placements


def _normalize_semantic_placements(
    value: object,
    *,
    members: set[str],
) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []

    out: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        obj_id = item.get("id")
        relative_to = item.get("relative_to")
        kind = item.get("kind")
        if not (
            isinstance(obj_id, str)
            and obj_id in members
            and isinstance(relative_to, str)
            and relative_to in members
            and isinstance(kind, str)
        ):
            continue
        record: dict[str, object] = {
            "id": obj_id,
            "relative_to": relative_to,
            "kind": kind.strip().lower(),
            "proximity": _normalize_semantic_proximity(item.get("proximity")),
            "selection": _normalize_semantic_selection(item.get("selection")),
        }
        orientation = _normalize_semantic_orientation(item.get("orientation"))
        if orientation is not None:
            record["orientation"] = orientation
        support_role = _normalize_semantic_support_role(item.get("support_role"))
        if support_role is None:
            support_role = _infer_semantic_support_role(
                object_id=obj_id,
                relative_to=relative_to,
                kind=record["kind"],
                side_options=item.get("side_options")
                if isinstance(item.get("side_options"), list)
                else None,
                b_edge=str(item.get("b_edge") or "") or None,
            )
        if support_role is not None:
            record["support_role"] = support_role
            band_intent = _semantic_band_intent_from_role(support_role)
            if band_intent is not None:
                record["band_intent"] = band_intent
        if "gap_min" in item or "gap_max" in item:
            gap_min, gap_max = _normalize_gap_range(
                item.get("gap_min"),
                item.get("gap_max"),
            )
            record["gap_min"] = gap_min
            record["gap_max"] = gap_max

        if record["kind"] == "dock_to_edge":
            b_edge = _normalize_dock_edge_token(item.get("b_edge"))
            if b_edge is not None:
                record["b_edge"] = b_edge
            record["span"] = _normalize_dock_span_token(item.get("span"))
        elif record["kind"] == "anchor_side":
            side_options = item.get("side_options")
            if isinstance(side_options, list):
                normalized_options = [
                    side
                    for raw_side in side_options
                    if (side := _normalize_anchor_side_token(raw_side)) is not None
                ]
                if normalized_options:
                    record["side_options"] = normalized_options
        out.append(record)
    return out


def _normalize_semantic_selection(value: object) -> str:
    if not isinstance(value, str):
        return "best_fit"
    token = value.strip().lower()
    if token in {"best_fit", "first_fit"}:
        return token
    return "best_fit"


def _normalize_semantic_orientation(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    token = value.strip().lower()
    if token in {
        "back_to_wall",
        "face_base",
        "face_cluster",
        "front_to_open_space",
        "same_direction",
    }:
        return token
    return None


def _normalize_semantic_support_role(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    token = value.strip().lower()
    if token in {
        "frontal_support",
        "side_support",
        "secondary_seat",
        "peripheral_support",
        "wall_support",
    }:
        return token
    return None


def _semantic_band_intent_from_role(value: object) -> str | None:
    token = _normalize_semantic_support_role(value)
    if token == "frontal_support":
        return "front_band"
    if token == "secondary_seat":
        return "flank_band"
    if token == "side_support":
        return "beside_base"
    if token == "wall_support":
        return "wall_band"
    if token == "peripheral_support":
        return "peripheral_band"
    return None


def _infer_semantic_support_role(
    *,
    object_id: str,
    relative_to: str,
    kind: str,
    side_options: Sequence[str] | None = None,
    b_edge: str | None = None,
) -> str | None:
    key = object_id.lower()
    base = relative_to.lower()
    options = {
        str(item).strip().lower() for item in (side_options or []) if str(item).strip()
    }
    edge = str(b_edge or "").strip().lower()

    if any(
        token in key for token in ("coffee_table", "bench", "ottoman", "console_table")
    ):
        return "frontal_support"
    if any(token in key for token in ("armchair", "chair", "stool", "seat")):
        return "secondary_seat"
    if any(
        token in key for token in ("side_table", "nightstand", "bedside", "end_table")
    ):
        return "side_support"
    if any(token in key for token in ("lamp",)):
        return "peripheral_support"
    if any(
        token in key
        for token in (
            "shelf",
            "bookshelf",
            "cabinet",
            "dresser",
            "wardrobe",
            "buffet",
            "sideboard",
        )
    ):
        return "wall_support"
    if kind == "dock_to_edge" and edge in {"front", "back"}:
        return "frontal_support"
    if kind == "anchor_side" and options & {
        "head",
        "front",
        "head_left",
        "head_right",
        "front_left",
        "front_right",
    }:
        return "frontal_support"
    if kind == "anchor_side" and options & {"left", "right"}:
        return "side_support"
    if is_seat_like(object_id) and is_lounge_anchor_like(relative_to):
        return "secondary_seat"
    if is_lounge_anchor_like(base) and _is_storage_anchor_like(object_id):
        return "wall_support"
    return None


def _upsert_semantic_placement(
    semantic_placements: list[dict[str, object]],
    record: dict[str, object],
) -> None:
    obj_id = record.get("id")
    relative_to = record.get("relative_to")
    kind = record.get("kind")
    for existing in semantic_placements:
        if not isinstance(existing, dict):
            continue
        if (
            existing.get("id") == obj_id
            and existing.get("relative_to") == relative_to
            and existing.get("kind") == kind
        ):
            existing.update(record)
            return
    semantic_placements.append(record)


def _ensure_soft_constraint(
    cluster: dict[str, object], record: dict[str, object]
) -> None:
    soft_constraints = cluster.get("soft_constraints")
    if not isinstance(soft_constraints, list):
        soft_constraints = []
        cluster["soft_constraints"] = soft_constraints

    for existing in soft_constraints:
        if not isinstance(existing, dict):
            continue
        if (
            existing.get("type") == record.get("type")
            and existing.get("a") == record.get("a")
            and existing.get("b") == record.get("b")
        ):
            return
    soft_constraints.append(record)
