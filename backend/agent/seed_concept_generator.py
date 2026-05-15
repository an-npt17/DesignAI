from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

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
        def agent_model_chain(
            cls,
            model_config_keys: Sequence[str],
            default_model_chain: Sequence[str] = (),
        ) -> list[str]:
            if cls.STRICT_SINGLE_TEXT_MODEL:
                return [cls.STRICT_SINGLE_TEXT_MODEL]
            configured = [
                cls.AGENT_MODELS.get(config_key) for config_key in model_config_keys
            ]
            return [
                model_name
                for model_name in (*configured, *default_model_chain)
                if isinstance(model_name, str) and model_name.strip()
            ]


from layout.room_profiles.registry import profile_room_type_for_objects
from layout.variant_family import SLEEP_VARIANT_FAMILIES

logger = logging.getLogger(__name__)

ConceptFamily = Literal[
    "focal_axis",
    "open_center",
    "edge_weighted",
    "zoned",
    "daylight_oriented",
]
Priority = Literal["core", "support", "optional"]
ZoneComplementRole = Literal["focal_claim", "topology_complement", "neutral"]
AnchorStrength = Literal["hard", "strong", "medium"]

CONCEPT_FAMILIES: tuple[ConceptFamily, ...] = (
    "focal_axis",
    "open_center",
    "edge_weighted",
    "zoned",
    "daylight_oriented",
)
MACRO_REGION_CAP = 16
MAX_REGION_CANDIDATES_PER_CLUSTER = 4
MAX_ASSIGNMENT_CANDIDATES_PER_CONCEPT = 32
DEDUPE_THRESHOLD = 0.75
MIN_CONCEPT_DISTANCE = 0.25
MAX_VARIANT_FAMILIES_PER_CLUSTER = 4
MIN_VARIANT_FAMILIES_PER_CLUSTER = 2
_RESPONSE_MIME_TYPE = "application/json"
_GUIDANCE_LLM_TEMPERATURE = 0.2
_GUIDANCE_LLM_TOP_P = 0.9
_SEED_CONCEPT_GUIDED_LLM_ENV = "TKNT_SEED_CONCEPT_GUIDED_LLM"
_GUIDED_STAGE_NOTE = (
    "LLM-guided SeedConceptGenerator refined macro concept blueprints while "
    "deterministic validation and solver projection remained active."
)
_GUIDED_STAGE_FALLBACK_NOTE = (
    "LLM-guided SeedConceptGenerator fell back to deterministic macro concept "
    "generation after guidance validation failed."
)
_CENTER_USAGE_VALUES = {"none", "partial", "primary", "open_reserved"}
_GUIDED_WALL_SIDES = {
    "",
    "center",
    "window_side",
    "top_wall",
    "right_wall",
    "bottom_wall",
    "left_wall",
}
_GUIDED_TOPOLOGY_POLICY_KEYS = {
    "reserve_center_degree",
    "wall_loading_bias",
    "daylight_bias",
    "entry_avoidance_strength",
    "secondary_zone_placement_bias",
}
_FIT_CHECKED_REQUIRED_REGION_IDS: frozenset[str] = frozenset()


@dataclass(frozen=True)
class _MacroConceptGuidanceSpec:
    model_config_keys: tuple[str, str, str]
    default_model_chain: tuple[str, str, str]
    system_prompt: str
    task_prompt: str
    output_contract: str
    repair_prompt: str


_MACRO_CONCEPT_GUIDANCE_SPEC = _MacroConceptGuidanceSpec(
    model_config_keys=(
        "relation_planner_macro_concept_guidance_primary",
        "relation_planner_macro_concept_guidance_fallback_primary",
        "relation_planner_macro_concept_guidance_fallback_secondary",
    ),
    default_model_chain=(
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite-preview",
        "gemini-2.5-flash-lite",
    ),
    system_prompt=(
        "You refine macro layout concept blueprints for a deterministic interior "
        "layout planner. Return exactly one JSON object. No markdown. No prose "
        "outside JSON. Use only the provided concept families, cluster ids, and "
        "macro region ids."
    ),
    task_prompt=(
        "Increase macro-concept diversity without breaking solver stability. "
        "Adjust concept family order, topology bias, and cluster-to-zone steering "
        "for this exact room, brief, and semantic cluster set. Favor legible, "
        "plausible alternatives over novelty for its own sake."
    ),
    output_contract=(
        "Top-level keys: `family_order`, `concept_blueprints`, `notes`. "
        "`family_order` must contain only provided concept families. "
        "`concept_blueprints` must be a list of guidance entries. Each entry must "
        "contain `concept_family`, `topology_policy`, and `cluster_zone_overrides`. "
        "`topology_policy` may only set: `reserve_center_degree`, "
        "`wall_loading_bias`, `daylight_bias`, `entry_avoidance_strength`, "
        "`secondary_zone_placement_bias`. Each cluster override must contain "
        "`cluster_id`, `zone_assignment`, `preferred_wall_side`, and "
        "`center_usage`. Use only provided cluster ids and macro region ids. "
        "Never assign any cluster to `keep_open_center`."
    ),
    repair_prompt=(
        "Return exactly one valid JSON object with top-level keys "
        "`family_order`, `concept_blueprints`, and `notes`. No markdown."
    ),
)


@dataclass(frozen=True)
class ClusterProgram:
    cluster_id: str
    semantic_role: str
    layout_role: str
    role_kind: str
    priority: Priority
    zone_claims: Mapping[str, object]
    relation_intents: tuple[Mapping[str, object], ...]
    seed_region_tags: tuple[str, ...]
    object_ids: tuple[str, ...]
    required_object_ids: tuple[str, ...]
    optional_object_ids: tuple[str, ...]
    droppable_object_ids: tuple[str, ...]

    @property
    def dominant_anchor_object_id(self) -> str | None:
        return self.object_ids[0] if self.object_ids else None


@dataclass(frozen=True)
class MacroRegion:
    region_id: str
    region_type: str
    label: str
    source_ids: tuple[str, ...]
    tags: tuple[str, ...]


@dataclass(frozen=True)
class MacroScenario:
    scenario_id: str
    label: str
    primary_zone_id: str
    secondary_zone_id: str
    storage_zone_id: str
    support_zone_id: str
    primary_wall_side: str
    secondary_wall_side: str
    storage_wall_side: str
    pair_type: str
    center_policy: str
    primary_anchor_strength: AnchorStrength
    secondary_anchor_strength: AnchorStrength
    support_anchor_strength: AnchorStrength
    allow_primary_center_overlap: bool = False


@dataclass(frozen=True)
class RegionCandidate:
    cluster_id: str
    region_id: str
    region_type: str
    score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class SeedConceptGenerator:
    """Deterministic topology-aware macro concept generator.

    This replaces the old LLM relation planner. It works at zone-claim and room
    topology level, then emits solver-compatible projections as an adapter layer.
    """

    def generate_bundle(
        self,
        *,
        room_model_json: Mapping[str, object],
        clusters_json: Mapping[str, object],
        target_count: int = 5,
        description: str | None = None,
        special_notes: str | None = None,
        use_llm: bool = True,
        temperature: float = _GUIDANCE_LLM_TEMPERATURE,
        top_p: float = _GUIDANCE_LLM_TOP_P,
        max_tokens: int | None = None,
    ) -> dict[str, object]:
        room_model = _unwrap_room_model(room_model_json)
        room_type = _room_type(room_model)
        cluster_programs = _build_cluster_programs(clusters_json)
        style_policy = _extract_style_policy(clusters_json)
        macro_region_map = _build_macro_region_map(room_model)
        region_candidates = _match_clusters_to_macro_regions(
            cluster_programs=cluster_programs,
            macro_regions=_macro_regions_from_map(macro_region_map),
        )
        guidance_notes: list[str] = []
        guidance_bundle: dict[str, object] | None = None
        controller = "deterministic_seed_concept_generator"
        if (
            use_llm
            and _guided_seed_concept_llm_enabled()
            and cluster_programs
            and region_candidates
        ):
            try:
                guidance_bundle = self._generate_macro_concept_guidance(
                    room_model=room_model,
                    room_type=room_type,
                    clusters_json=clusters_json,
                    cluster_programs=cluster_programs,
                    macro_region_map=macro_region_map,
                    region_candidates=region_candidates,
                    style_policy=style_policy,
                    description=description,
                    special_notes=special_notes,
                    target_count=target_count,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                guidance_notes.extend(_string_list(guidance_bundle.get("notes")))
                controller = "llm_guided_seed_concept_generator"
            except Exception as exc:
                logger.warning(
                    "SeedConceptGenerator LLM guidance failed; using deterministic macro concepts: %s",
                    exc,
                )
                guidance_notes.append(_GUIDED_STAGE_FALLBACK_NOTE)
        concepts = _instantiate_concept_families(
            room_model=room_model,
            room_type=room_type,
            cluster_programs=cluster_programs,
            macro_region_map=macro_region_map,
            region_candidates=region_candidates,
            style_policy=style_policy,
            target_count=target_count,
            guidance_bundle=guidance_bundle,
        )

        status = "OK" if concepts and cluster_programs else "UNSAT"
        notes = [
            "Relation planning is absorbed into semantic zone claims and deterministic macro concept generation.",
            "Concepts do not place exact coordinates; MacroClusterSolver owns pose search.",
        ]
        if guidance_bundle is not None:
            notes.insert(0, _GUIDED_STAGE_NOTE)
        notes.extend(note for note in guidance_notes if note not in notes)
        return {
            "status": status,
            "room_id": _room_id(room_model),
            "room_type": room_type,
            "controller": controller,
            "settings": {
                "macro_region_cap": MACRO_REGION_CAP,
                "concept_family_count": len(CONCEPT_FAMILIES),
                "max_region_candidates_per_cluster": MAX_REGION_CANDIDATES_PER_CLUSTER,
                "max_assignment_candidates_per_concept": MAX_ASSIGNMENT_CANDIDATES_PER_CONCEPT,
                "dedupe_threshold": DEDUPE_THRESHOLD,
                "min_concept_distance": MIN_CONCEPT_DISTANCE,
                "llm_guided": guidance_bundle is not None,
            },
            "style_policy": style_policy,
            "macro_region_map": macro_region_map,
            "region_assignment_candidates": [
                _region_candidate_to_dict(candidate)
                for candidate in region_candidates[
                    :MAX_ASSIGNMENT_CANDIDATES_PER_CONCEPT
                ]
            ],
            "concepts": concepts,
            "notes": notes,
            "llm_guidance": guidance_bundle or {},
        }

    def generate(
        self,
        *,
        room_model_json: Mapping[str, object],
        clusters_json: Mapping[str, object],
        description: str | None = None,
        special_notes: str | None = None,
        model_name: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> dict[str, object]:
        _ = model_name
        family = _family_from_text(special_notes)
        bundle = self.generate_bundle(
            room_model_json=room_model_json,
            clusters_json=clusters_json,
            target_count=len(CONCEPT_FAMILIES),
            description=description,
            special_notes=special_notes,
            use_llm=True,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        concepts = [
            concept
            for concept in bundle.get("concepts", [])
            if isinstance(concept, dict)
        ]
        concept = next(
            (item for item in concepts if item.get("concept_family") == family),
            concepts[0] if concepts else None,
        )
        room_model = _unwrap_room_model(room_model_json)
        if not isinstance(concept, dict):
            return _empty_solver_plan(room_model)
        return _concept_to_solver_plan(
            concept=concept,
            room_model=room_model,
            room_type=str(bundle.get("room_type") or _room_type(room_model_json)),
        )

    def _generate_macro_concept_guidance(
        self,
        *,
        room_model: Mapping[str, object],
        room_type: str,
        clusters_json: Mapping[str, object],
        cluster_programs: Sequence[ClusterProgram],
        macro_region_map: Mapping[str, object],
        region_candidates: Sequence[RegionCandidate],
        style_policy: Mapping[str, object],
        description: str | None,
        special_notes: str | None,
        target_count: int,
        temperature: float,
        top_p: float,
        max_tokens: int | None,
    ) -> dict[str, object]:
        payload = {
            "room_type": room_type,
            "description": (description or "").strip(),
            "special_notes": (special_notes or "").strip(),
            "target_count": max(1, min(int(target_count), len(CONCEPT_FAMILIES))),
            "style_policy": style_policy,
            "cluster_programs": [
                {
                    "cluster_id": cluster.cluster_id,
                    "semantic_role": cluster.semantic_role,
                    "layout_role": cluster.layout_role,
                    "role_kind": cluster.role_kind,
                    "priority": cluster.priority,
                    "zone_claims": dict(cluster.zone_claims),
                    "object_ids": list(cluster.object_ids),
                }
                for cluster in cluster_programs
            ],
            "macro_regions": [
                {
                    "region_id": region.get("region_id"),
                    "region_type": region.get("region_type"),
                    "tags": region.get("tags"),
                }
                for region in _sequence_or_empty(macro_region_map.get("regions"))
                if isinstance(region, Mapping)
            ],
            "top_region_candidates_by_cluster": _top_region_candidates_by_cluster(
                region_candidates
            ),
            "available_concept_families": list(CONCEPT_FAMILIES),
            "deterministic_family_order": list(
                _style_ranked_concept_families(style_policy)
            ),
            "clusters_json_summary": _guided_clusters_summary(clusters_json),
            "room_topology_summary": _guided_room_summary(room_model, macro_region_map),
        }
        guidance = self._call_macro_concept_guidance_stage(
            prompt_payload=payload,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            response_validator=lambda response: _sanitize_macro_concept_guidance(
                response,
                target_count=max(1, min(int(target_count), len(CONCEPT_FAMILIES))),
                available_families=CONCEPT_FAMILIES,
                cluster_ids={cluster.cluster_id for cluster in cluster_programs},
                region_ids=_available_macro_region_ids(macro_region_map),
            ),
        )
        return _sanitize_macro_concept_guidance(
            guidance,
            target_count=max(1, min(int(target_count), len(CONCEPT_FAMILIES))),
            available_families=CONCEPT_FAMILIES,
            cluster_ids={cluster.cluster_id for cluster in cluster_programs},
            region_ids=_available_macro_region_ids(macro_region_map),
        )

    def _call_macro_concept_guidance_stage(
        self,
        *,
        prompt_payload: Mapping[str, object],
        temperature: float,
        top_p: float,
        max_tokens: int | None,
        response_validator: Callable[[dict[str, object]], object] | None = None,
    ) -> dict[str, object]:
        stage_spec = _MACRO_CONCEPT_GUIDANCE_SPEC
        user_prompt = (
            "Task:\n"
            f"{stage_spec.task_prompt}\n\n"
            "Output Contract:\n"
            f"{stage_spec.output_contract}\n\n"
            "Rules:\n"
            "- Return exactly one JSON object.\n"
            "- No markdown.\n"
            "- No commentary outside JSON.\n"
            "- Do not omit required keys.\n\n"
            "Payload:\n"
            f"{json.dumps(prompt_payload, ensure_ascii=True, indent=2)}"
        )
        base_messages: list[ChatMessage] = [
            {"role": "system", "content": stage_spec.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        client = get_llm_client()
        model_chain = _seed_concept_guidance_model_chain()
        last_error: Exception | None = None
        for model_index, model_name in enumerate(model_chain):
            messages = list(base_messages)
            fallback_model_names = model_chain[model_index + 1 :]
            try:
                last_raw = ""
                for attempt in range(2):
                    response = client.chat_completion(
                        messages,
                        model_key="primary",
                        model_name=model_name,
                        fallback_model_names=fallback_model_names,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        response_mime_type=_RESPONSE_MIME_TYPE,
                    )
                    last_raw = _extract_llm_content(response)
                    try:
                        parsed = _parse_guidance_json(last_raw)
                        if response_validator is not None:
                            response_validator(parsed)
                        return parsed
                    except ValueError as exc:
                        if attempt == 1:
                            raise ValueError(
                                "macro concept guidance validation failed on "
                                f"{model_name}: {exc}"
                            ) from exc
                        _record_llm_retry(
                            stage="seed_concept_generator.macro_concept_guidance",
                            model_name=model_name,
                            reason="invalid_json_or_schema",
                        )
                        messages.append({"role": "assistant", "content": last_raw})
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"{stage_spec.repair_prompt}\n"
                                    f"Validation error: {exc}"
                                ),
                            }
                        )
            except Exception as exc:
                last_error = exc
                if model_index == len(model_chain) - 1:
                    break
                logger.warning(
                    "SeedConceptGenerator guidance failed on model %s; retrying with %s: %s",
                    model_name,
                    model_chain[model_index + 1],
                    exc,
                )
        if last_error is not None:
            raise last_error
        raise ValueError("macro concept guidance produced no response")


def solver_plan_from_concept(
    *,
    concept: Mapping[str, object],
    room_model_json: Mapping[str, object],
    room_type: str | None = None,
) -> dict[str, object]:
    return _concept_to_solver_plan(
        concept=concept,
        room_model=_unwrap_room_model(room_model_json),
        room_type=room_type or _room_type(room_model_json),
    )


def _extract_style_policy(clusters_json: Mapping[str, object]) -> dict[str, object]:
    policy = clusters_json.get("style_policy")
    if isinstance(policy, Mapping):
        return dict(policy)
    semantic = clusters_json.get("semantic_layout_program")
    if isinstance(semantic, Mapping):
        policy = semantic.get("style_policy")
        if isinstance(policy, Mapping):
            return dict(policy)
    return {}


def _style_layout_policy(style_policy: Mapping[str, object]) -> Mapping[str, object]:
    layout_policy = style_policy.get("layout_policy")
    return layout_policy if isinstance(layout_policy, Mapping) else {}


def _style_ranked_concept_families(
    style_policy: Mapping[str, object],
) -> tuple[ConceptFamily, ...]:
    layout_policy = _style_layout_policy(style_policy)
    ranked: list[ConceptFamily] = list(CONCEPT_FAMILIES)
    if _bias_level(layout_policy.get("center_openness_bias")) >= 3:
        ranked = _promote_family(ranked, "open_center")
    if _bias_level(layout_policy.get("daylight_bias")) >= 3:
        ranked = _promote_family(ranked, "daylight_oriented")
    if str(layout_policy.get("wall_loading_bias") or "") in {
        "medium_high",
        "perimeter_heavy",
    }:
        ranked = _promote_family(ranked, "edge_weighted")
    if _bias_level(layout_policy.get("symmetry_bias")) >= 3:
        ranked = _promote_family(ranked, "focal_axis")
    return tuple(ranked)


def _promote_family(
    families: Sequence[ConceptFamily],
    family: ConceptFamily,
) -> list[ConceptFamily]:
    return [family, *[item for item in families if item != family]]


def _bias_level(value: object) -> int:
    text = str(value or "").lower()
    if "very_high" in text or text == "high":
        return 4
    if "medium_high" in text:
        return 3
    if "low_to_medium" in text or "low_to_balanced" in text:
        return 1
    if "medium" in text or "balanced" in text:
        return 2
    if "low" in text:
        return 0
    return 2


def _unwrap_room_model(room_model_json: Mapping[str, object]) -> Mapping[str, object]:
    room = room_model_json.get("room")
    if isinstance(room, Mapping):
        return room_model_json
    for key in ("parsed", "raw"):
        nested = room_model_json.get(key)
        if isinstance(nested, Mapping) and isinstance(nested.get("room"), Mapping):
            return nested
    return room_model_json


def _room_id(room_model: Mapping[str, object]) -> str:
    room = room_model.get("room")
    if isinstance(room, Mapping):
        value = room.get("room_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "room_1"


def _room_type(room_model: Mapping[str, object]) -> str:
    meta = room_model.get("meta")
    if isinstance(meta, Mapping):
        value = meta.get("room_type")
        if isinstance(value, str) and value.strip():
            return value.strip()
    value = room_model.get("room_type")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "room"


def _extract_clusters_map(
    clusters_json: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    clusters = clusters_json.get("clusters")
    if isinstance(clusters, Mapping):
        return {
            str(cluster_id): cluster
            for cluster_id, cluster in clusters.items()
            if isinstance(cluster_id, str) and isinstance(cluster, Mapping)
        }
    if isinstance(clusters, Sequence) and not isinstance(clusters, str):
        out: dict[str, Mapping[str, object]] = {}
        for cluster in clusters:
            if not isinstance(cluster, Mapping):
                continue
            cluster_id = _clean_str(cluster.get("cluster_id"))
            if cluster_id is not None:
                out[cluster_id] = cluster
        return out

    out = {}
    for cluster_id, cluster in clusters_json.items():
        if not isinstance(cluster_id, str) or not isinstance(cluster, Mapping):
            continue
        if {"cluster_id", "cluster_footprint", "local_placements"} & set(cluster):
            out[cluster_id] = cluster
    return out


def _build_cluster_programs(
    clusters_json: Mapping[str, object],
) -> list[ClusterProgram]:
    clusters = _extract_clusters_map(clusters_json)
    semantic_by_id = _semantic_clusters_by_id(clusters_json)
    programs: list[ClusterProgram] = []
    for cluster_id in sorted(clusters):
        cluster = clusters[cluster_id]
        semantic = semantic_by_id.get(cluster_id, {})
        zone_claims = _mapping_or_empty(
            semantic.get("zone_claims") or _cluster_rules(cluster).get("zone_claims")
        )
        object_ids = tuple(_cluster_object_ids(cluster))
        object_program = _mapping_or_empty(cluster.get("object_program"))
        required_object_ids = tuple(
            item
            for item in _string_list(object_program.get("required_object_ids"))
            if item in object_ids
        )
        optional_object_ids = tuple(
            item
            for item in _string_list(object_program.get("optional_object_ids"))
            if item in object_ids
        )
        droppable_object_ids = tuple(
            item
            for item in _string_list(object_program.get("droppable_ids"))
            if item in object_ids
        )
        semantic_role = _clean_str(
            semantic.get("semantic_role")
        ) or _semantic_role_from_cluster(cluster_id, object_ids)
        role_kind = _role_kind(cluster_id, object_ids, semantic_role)
        priority = _priority(semantic.get("priority"), role_kind)
        relation_intents = tuple(
            item
            for item in _sequence_or_empty(semantic.get("relation_intents"))
            if isinstance(item, Mapping)
        )
        programs.append(
            ClusterProgram(
                cluster_id=cluster_id,
                semantic_role=semantic_role,
                layout_role=_layout_role(semantic),
                role_kind=role_kind,
                priority=priority,
                zone_claims=zone_claims,
                relation_intents=relation_intents,
                seed_region_tags=tuple(_seed_region_tags(cluster)),
                object_ids=object_ids,
                required_object_ids=required_object_ids,
                optional_object_ids=optional_object_ids,
                droppable_object_ids=droppable_object_ids,
            )
        )
    return programs


def _semantic_clusters_by_id(
    clusters_json: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    program = clusters_json.get("semantic_layout_program")
    if not isinstance(program, Mapping):
        return {}
    active = program.get("active_clusters")
    if not isinstance(active, Sequence) or isinstance(active, str):
        return {}
    out: dict[str, Mapping[str, object]] = {}
    for cluster in active:
        if not isinstance(cluster, Mapping):
            continue
        cluster_id = _clean_str(cluster.get("cluster_id"))
        if cluster_id is not None:
            out[cluster_id] = cluster
    return out


def _build_macro_region_map(room_model: Mapping[str, object]) -> dict[str, object]:
    affordance = _mapping_or_empty(room_model.get("affordance_map"))
    topology = _mapping_or_empty(room_model.get("topology"))
    openings = _mapping_or_empty(room_model.get("openings"))
    room = _mapping_or_empty(room_model.get("room"))

    door_ids = _opening_ids(openings.get("doors"), "door")
    window_ids = _opening_ids(openings.get("windows"), "window")
    wall_ids = _wall_region_ids(room, affordance)
    center_ids = _region_refs(affordance.get("center_openness_regions")) or [
        "room_center"
    ]
    corridor_ids = _region_refs(affordance.get("primary_circulation_corridors"))
    if not corridor_ids and door_ids:
        corridor_ids = [f"{door_id}_to_room_center_corridor" for door_id in door_ids]
    entry_ids = _region_refs(affordance.get("entry_landing_zones")) or door_ids
    daylight_ids = _region_refs(affordance.get("daylight_regions")) or window_ids
    private_ids = _region_refs(affordance.get("privacy_regions")) or [
        "deep_zone_far_from_entry"
    ]
    side_wall_ids = (
        "top_wall_zone",
        "right_wall_zone",
        "bottom_wall_zone",
        "left_wall_zone",
    )

    regions = [
        MacroRegion(
            region_id="primary_focal_wall_zone",
            region_type="focal_wall_zone",
            label="primary focal wall zone",
            source_ids=tuple(
                _region_refs(affordance.get("focal_surfaces")) or wall_ids[:2]
            ),
            tags=("wall", "focal", "anchor"),
        ),
        MacroRegion(
            region_id="quiet_private_deep_zone",
            region_type="private_deep_zone",
            label="quiet private deep zone",
            source_ids=tuple(private_ids),
            tags=("privacy", "far_from_entry"),
        ),
        MacroRegion(
            region_id="daylight_biased_zone",
            region_type="daylight_zone",
            label="daylight-biased zone",
            source_ids=tuple(daylight_ids),
            tags=("daylight", "window_side"),
        ),
        MacroRegion(
            region_id="entry_adjacent_active_zone",
            region_type="entry_active_zone",
            label="entry-adjacent active zone",
            source_ids=tuple(entry_ids),
            tags=("entry_side", "active"),
        ),
        MacroRegion(
            region_id="keep_open_center",
            region_type="keep_open_center",
            label="keep-open center",
            source_ids=tuple(center_ids),
            tags=("center", "keep_open"),
        ),
        MacroRegion(
            region_id="floating_support_zone",
            region_type="floating_support_zone",
            label="floating support zone",
            source_ids=tuple(
                _region_refs(affordance.get("floating_zone_candidates"))
                or ["floating_center_zone"]
            ),
            tags=("floating", "support", "center"),
        ),
        MacroRegion(
            region_id="storage_service_edge_zone",
            region_type="storage_service_edge_zone",
            label="storage/service edge zone",
            source_ids=side_wall_ids or tuple(wall_ids),
            tags=("wall", "edge", "storage"),
        ),
        MacroRegion(
            region_id="edge_loading_zone",
            region_type="edge_loading_zone",
            label="edge loading zone",
            source_ids=tuple(
                _region_refs(affordance.get("wall_anchor_candidates"))
                or side_wall_ids
                or wall_ids
            ),
            tags=("wall", "edge", "perimeter"),
        ),
        MacroRegion(
            region_id="top_wall_zone",
            region_type="side_wall_zone",
            label="top wall zone",
            source_ids=("top_wall_zone",),
            tags=("wall", "edge", "top_wall"),
        ),
        MacroRegion(
            region_id="right_wall_zone",
            region_type="side_wall_zone",
            label="right wall zone",
            source_ids=("right_wall_zone",),
            tags=("wall", "edge", "right_wall"),
        ),
        MacroRegion(
            region_id="bottom_wall_zone",
            region_type="side_wall_zone",
            label="bottom wall zone",
            source_ids=("bottom_wall_zone",),
            tags=("wall", "edge", "bottom_wall"),
        ),
        MacroRegion(
            region_id="left_wall_zone",
            region_type="side_wall_zone",
            label="left wall zone",
            source_ids=("left_wall_zone",),
            tags=("wall", "edge", "left_wall"),
        ),
        MacroRegion(
            region_id="window_side_zone",
            region_type="window_side_zone",
            label="window side zone",
            source_ids=tuple(daylight_ids or ["window_side_zone"]),
            tags=("wall", "daylight", "window_side"),
        ),
        MacroRegion(
            region_id="floating_center_zone",
            region_type="floating_center_zone",
            label="floating center zone",
            source_ids=("floating_center_zone",),
            tags=("floating", "center"),
        ),
    ][:MACRO_REGION_CAP]
    return {
        "regions": [_macro_region_to_dict(region) for region in regions],
        "protected_topology": {
            "entry_landing_zones": entry_ids,
            "primary_circulation_corridors": corridor_ids,
            "center_openness_regions": center_ids,
            "operational_zones": _operational_protected_topology_zones(
                entry_refs=entry_ids,
                corridor_refs=corridor_ids,
                center_refs=center_ids,
            ),
        },
        "topology_summary": {
            "entry_nodes": _region_refs(topology.get("entry_nodes")) or door_ids,
            "window_nodes": _region_refs(topology.get("window_nodes")) or window_ids,
            "subzones": _region_refs(topology.get("subzones")),
        },
    }


def _macro_regions_from_map(
    macro_region_map: Mapping[str, object],
) -> list[MacroRegion]:
    out: list[MacroRegion] = []
    regions = macro_region_map.get("regions")
    if not isinstance(regions, Sequence) or isinstance(regions, str):
        return out
    for region in regions:
        if not isinstance(region, Mapping):
            continue
        region_id = _clean_str(region.get("region_id"))
        region_type = _clean_str(region.get("region_type"))
        label = _clean_str(region.get("label")) or region_id
        if region_id is None or region_type is None or label is None:
            continue
        out.append(
            MacroRegion(
                region_id=region_id,
                region_type=region_type,
                label=label,
                source_ids=tuple(_string_list(region.get("source_ids"))),
                tags=tuple(_string_list(region.get("tags"))),
            )
        )
    return out


def _apply_zone_forbidden_guardrails(
    *,
    room_model: Mapping[str, object],
    macro_region_map: Mapping[str, object],
    cluster_programs: Sequence[ClusterProgram],
    family: ConceptFamily,
    cluster_zone_plan: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    region_index = _guardrail_region_index(room_model)
    clusters_by_id = {cluster.cluster_id: cluster for cluster in cluster_programs}
    rows = [dict(row) for row in cluster_zone_plan]
    for row in rows:
        cluster_id = _clean_str(row.get("cluster_id"))
        if cluster_id is None:
            continue
        cluster = clusters_by_id.get(cluster_id)
        if cluster is None:
            continue
        current_zone = _clean_str(row.get("zone_assignment"))
        if current_zone is None:
            continue
        forbidden_region_ids = _string_list(row.get("forbidden_region_ids"))
        if not forbidden_region_ids:
            continue
        if not _zone_overlaps_forbidden_regions(
            zone_id=current_zone,
            forbidden_region_ids=forbidden_region_ids,
            region_index=region_index,
            macro_region_map=macro_region_map,
            room_model=room_model,
        ):
            continue
        replacement = _non_conflicting_zone_assignment(
            row=row,
            forbidden_region_ids=forbidden_region_ids,
            region_index=region_index,
            macro_region_map=macro_region_map,
            room_model=room_model,
        )
        if replacement is None or replacement == current_zone:
            row["zone_guardrail_notes"] = [
                "No non-conflicting macro zone candidate was available."
            ]
            continue
        _apply_zone_assignment_to_row(
            row=row,
            cluster=cluster,
            family=family,
            zone_assignment=replacement,
        )
        row["zone_guardrail_notes"] = [
            f"Reassigned from {current_zone} because it overlaps a forbidden region."
        ]
    return rows


def _non_conflicting_zone_assignment(
    *,
    row: Mapping[str, object],
    forbidden_region_ids: Sequence[str],
    region_index: Mapping[str, tuple[int, int, int, int]],
    macro_region_map: Mapping[str, object],
    room_model: Mapping[str, object],
) -> str | None:
    current_zone = _clean_str(row.get("zone_assignment"))
    candidate_ids = [
        _clean_str(candidate.get("region_id"))
        for candidate in _sequence_or_empty(row.get("region_candidates"))
        if isinstance(candidate, Mapping)
    ]
    candidate_ids.extend(
        [
            "edge_loading_zone",
            "storage_service_edge_zone",
            "top_wall_zone",
            "right_wall_zone",
            "bottom_wall_zone",
            "left_wall_zone",
            "primary_focal_wall_zone",
            "quiet_private_deep_zone",
            "daylight_biased_zone",
            "window_side_zone",
        ]
    )
    seen: set[str] = set()
    best_zone = current_zone
    best_score = float("inf")
    for candidate_id in candidate_ids:
        if candidate_id is None or candidate_id in seen:
            continue
        seen.add(candidate_id)
        if candidate_id == "keep_open_center":
            continue
        score = _zone_forbidden_overlap_score(
            zone_id=candidate_id,
            forbidden_region_ids=forbidden_region_ids,
            region_index=region_index,
            macro_region_map=macro_region_map,
            room_model=room_model,
        )
        if score <= 1e-9:
            return candidate_id
        if score < best_score:
            best_score = score
            best_zone = candidate_id
    return best_zone


def _zone_overlaps_forbidden_regions(
    *,
    zone_id: str,
    forbidden_region_ids: Sequence[str],
    region_index: Mapping[str, tuple[int, int, int, int]],
    macro_region_map: Mapping[str, object],
    room_model: Mapping[str, object],
) -> bool:
    return (
        _zone_forbidden_overlap_score(
            zone_id=zone_id,
            forbidden_region_ids=forbidden_region_ids,
            region_index=region_index,
            macro_region_map=macro_region_map,
            room_model=room_model,
        )
        > 0.25
    )


def _zone_forbidden_overlap_score(
    *,
    zone_id: str,
    forbidden_region_ids: Sequence[str],
    region_index: Mapping[str, tuple[int, int, int, int]],
    macro_region_map: Mapping[str, object],
    room_model: Mapping[str, object],
) -> float:
    zone_bboxes = _guardrail_region_bboxes_from_ref(
        zone_id, region_index, macro_region_map, room_model
    )
    forbidden_bboxes = [
        bbox
        for region_id in forbidden_region_ids
        for bbox in _guardrail_region_bboxes_from_ref(
            region_id, region_index, macro_region_map, room_model
        )
    ]
    score = 0.0
    for zone_bbox in zone_bboxes:
        for forbidden_bbox in forbidden_bboxes:
            score = max(
                score,
                _guardrail_rect_overlap_ratio(zone_bbox, forbidden_bbox),
                _guardrail_rect_overlap_ratio(forbidden_bbox, zone_bbox),
            )
    return score


def _guardrail_region_index(
    room_model: Mapping[str, object],
) -> dict[str, tuple[int, int, int, int]]:
    out: dict[str, tuple[int, int, int, int]] = {}

    def walk(value: object) -> None:
        if isinstance(value, Mapping):
            region_id = _clean_str(value.get("region_id")) or _clean_str(
                value.get("id")
            )
            bbox = _guardrail_bbox_from_mapping(value)
            if region_id is not None and bbox is not None:
                out.setdefault(region_id, bbox)
            for child in value.values():
                walk(child)
        elif isinstance(value, Sequence) and not isinstance(value, str):
            for item in value:
                walk(item)

    walk(room_model)
    _add_guardrail_primary_corridor_aliases(room_model, out)
    return out


def _guardrail_region_bboxes_from_ref(
    region_ref: str,
    region_index: Mapping[str, tuple[int, int, int, int]],
    macro_region_map: Mapping[str, object],
    room_model: Mapping[str, object],
) -> list[tuple[int, int, int, int]]:
    if region_ref in region_index:
        return [region_index[region_ref]]
    for region in _sequence_or_empty(macro_region_map.get("regions")):
        if not isinstance(region, Mapping):
            continue
        if _clean_str(region.get("region_id")) != region_ref:
            continue
        source_bboxes = [
            bbox
            for source_id in _string_list(region.get("source_ids"))
            for bbox in _guardrail_region_bboxes_from_ref(
                source_id, region_index, {}, room_model
            )
        ]
        if source_bboxes:
            return _dedupe_guardrail_bboxes(source_bboxes)
    fallback_bbox = _guardrail_fallback_region_bbox(region_ref, room_model)
    return [fallback_bbox] if fallback_bbox is not None else []


def _dedupe_guardrail_bboxes(
    bboxes: Sequence[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    out: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for bbox in bboxes:
        if bbox in seen:
            continue
        seen.add(bbox)
        out.append(bbox)
    return out


def _guardrail_bbox_from_mapping(
    value: Mapping[str, object],
) -> tuple[int, int, int, int] | None:
    bbox = value.get("bbox_mm") or value.get("bbox")
    if isinstance(bbox, Mapping):
        parsed = _guardrail_bbox_tuple(
            bbox.get("min_x"),
            bbox.get("min_y"),
            bbox.get("max_x"),
            bbox.get("max_y"),
        )
        if parsed is not None:
            return parsed
    for key in (
        "near_polygon_ccw",
        "anchor_polygon_ccw",
        "polygon_ccw",
        "polygon",
        "points",
        "mid_polygon_ccw",
    ):
        parsed = _guardrail_bbox_from_points(value.get(key))
        if parsed is not None:
            return parsed
    polyline_bbox = _guardrail_bbox_from_polyline(value)
    if polyline_bbox is not None:
        return polyline_bbox
    return None


def _guardrail_bbox_from_points(
    value: object,
) -> tuple[int, int, int, int] | None:
    points: list[tuple[float, float]] = []
    for item in _sequence_or_empty(value):
        if not isinstance(item, Mapping):
            continue
        try:
            points.append((float(item.get("x")), float(item.get("y"))))
        except (TypeError, ValueError):
            continue
    if len(points) < 2:
        return None
    return _guardrail_bbox_tuple(
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    )


def _guardrail_bbox_from_polyline(
    value: Mapping[str, object],
) -> tuple[int, int, int, int] | None:
    points_bbox = _guardrail_bbox_from_points(value.get("polyline_mm"))
    if points_bbox is None:
        return None
    try:
        width_mm = max(0, int(value.get("width_mm") or 0))
    except (TypeError, ValueError):
        width_mm = 0
    pad = max(1, width_mm // 2)
    return (
        points_bbox[0] - pad,
        points_bbox[1] - pad,
        points_bbox[2] + pad,
        points_bbox[3] + pad,
    )


def _add_guardrail_primary_corridor_aliases(
    room_model: Mapping[str, object],
    region_index: dict[str, tuple[int, int, int, int]],
) -> None:
    affordance = _mapping_or_empty(room_model.get("affordance_map"))
    primary_bbox: tuple[int, int, int, int] | None = None
    for row in _sequence_or_empty(affordance.get("circulation_corridors")):
        if not isinstance(row, Mapping):
            continue
        if str(row.get("from") or "") != "entry" or str(row.get("to") or "") not in {
            "room_center",
            "center",
        }:
            continue
        primary_bbox = _guardrail_bbox_from_mapping(row)
        if primary_bbox is not None:
            break
    if primary_bbox is None:
        return
    region_index.setdefault("entry_to_center_corridor", primary_bbox)
    openings = _mapping_or_empty(room_model.get("openings"))
    for index, door in enumerate(_sequence_or_empty(openings.get("doors")), start=1):
        door_id = (
            _clean_str(door.get("id")) if isinstance(door, Mapping) else f"door_{index}"
        )
        alias_id = door_id or f"door_{index}"
        region_index.setdefault(f"{alias_id}_to_room_center_corridor", primary_bbox)


def _guardrail_bbox_tuple(
    min_x: object,
    min_y: object,
    max_x: object,
    max_y: object,
) -> tuple[int, int, int, int] | None:
    try:
        x1 = int(round(float(min_x)))
        y1 = int(round(float(min_y)))
        x2 = int(round(float(max_x)))
        y2 = int(round(float(max_y)))
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _guardrail_fallback_region_bbox(
    region_ref: str,
    room_model: Mapping[str, object],
) -> tuple[int, int, int, int] | None:
    room_bbox = _guardrail_room_bbox(room_model)
    min_x, min_y, max_x, max_y = room_bbox
    width = max_x - min_x
    height = max_y - min_y
    token = region_ref.lower()
    if "bottom" in token and "wall" in token:
        return (min_x, max_y - max(700, int(round(height * 0.18))), max_x, max_y)
    if "left" in token and "wall" in token:
        return (min_x, min_y, min_x + max(700, int(round(width * 0.18))), max_y)
    if "right" in token and "wall" in token:
        return (max_x - max(700, int(round(width * 0.18))), min_y, max_x, max_y)
    if "focal" in token or ("wall" in token and "top" in token):
        return (min_x, min_y, max_x, min_y + max(700, int(round(height * 0.18))))
    if "daylight" in token or "window" in token:
        return (min_x, min_y, max_x, min_y + max(900, int(round(height * 0.24))))
    if "entry" in token or "door" in token:
        return (
            min_x,
            max_y - max(850, int(round(height * 0.22))),
            min_x + max(1000, int(round(width * 0.22))),
            max_y,
        )
    if "center" in token or "floating" in token:
        return (
            min_x + int(round(width * 0.25)),
            min_y + int(round(height * 0.25)),
            max_x - int(round(width * 0.25)),
            max_y - int(round(height * 0.25)),
        )
    if "privacy" in token or "deep" in token:
        return (
            min_x + int(round(width * 0.45)),
            min_y + int(round(height * 0.45)),
            max_x,
            max_y,
        )
    if "edge" in token or "storage" in token:
        return room_bbox
    return None


def _guardrail_room_bbox(room_model: Mapping[str, object]) -> tuple[int, int, int, int]:
    room = _mapping_or_empty(room_model.get("room"))
    bbox = _mapping_or_empty(room.get("bbox_mm") or room.get("bbox"))
    parsed = _guardrail_bbox_tuple(
        bbox.get("min_x"),
        bbox.get("min_y"),
        bbox.get("max_x"),
        bbox.get("max_y"),
    )
    if parsed is not None:
        return parsed
    polygon_bbox = _guardrail_bbox_from_points(room.get("polygon_ccw"))
    if polygon_bbox is not None:
        return polygon_bbox
    return 0, 0, 6000, 4000


def _guardrail_rect_overlap_ratio(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    ix1 = max(left[0], right[0])
    iy1 = max(left[1], right[1])
    ix2 = min(left[2], right[2])
    iy2 = min(left[3], right[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = float((ix2 - ix1) * (iy2 - iy1))
    area = float(max(1, (left[2] - left[0]) * (left[3] - left[1])))
    return intersection / area


def _operational_protected_topology_zones(
    *,
    entry_refs: Sequence[str],
    corridor_refs: Sequence[str],
    center_refs: Sequence[str],
) -> list[dict[str, object]]:
    zones: list[dict[str, object]] = []
    for ref in entry_refs:
        zones.append(
            {
                "region": ref,
                "zone_type": "entry_landing",
                "priority": "high",
                "enforcement": "hard",
                "max_overlap_ratio": 0.0,
                "applies_to": [
                    "core_clusters",
                    "support_clusters",
                    "interaction_hulls",
                ],
                "violation_severity": "blocking",
            }
        )
    for ref in corridor_refs:
        zones.append(
            {
                "region": ref,
                "zone_type": "primary_circulation_corridor",
                "priority": "high",
                "enforcement": "hard_soft",
                "max_overlap_ratio": 0.05,
                "applies_to": [
                    "core_clusters",
                    "support_clusters",
                    "interaction_hulls",
                ],
                "violation_severity": "blocking",
            }
        )
    for ref in center_refs:
        zones.append(
            {
                "region": ref,
                "zone_type": "center_openness_core",
                "priority": "high",
                "enforcement": "hard_soft",
                "max_overlap_ratio": 0.10,
                "applies_to": ["core_clusters", "interaction_hulls"],
                "violation_severity": "blocking",
            }
        )
    return zones


def _match_clusters_to_macro_regions(
    *,
    cluster_programs: Sequence[ClusterProgram],
    macro_regions: Sequence[MacroRegion],
) -> list[RegionCandidate]:
    candidates: list[RegionCandidate] = []
    for cluster in cluster_programs:
        scored = [
            _score_cluster_region(cluster=cluster, region=region)
            for region in macro_regions
        ]
        scored.sort(key=lambda item: (-item.score, item.region_id))
        candidates.extend(scored[:MAX_REGION_CANDIDATES_PER_CLUSTER])
    return candidates


def _score_cluster_region(
    *,
    cluster: ClusterProgram,
    region: MacroRegion,
) -> RegionCandidate:
    claims = cluster.zone_claims
    tags = set(region.tags)
    role = cluster.role_kind
    score = 0.0
    reasons: list[str] = []

    if cluster.priority == "core":
        score += 1.2
    elif cluster.priority == "support":
        score += 0.5

    wall_affinity = _affinity_level(claims.get("wall_affinity"))
    daylight_affinity = _affinity_level(claims.get("daylight_affinity"))
    privacy_affinity = _affinity_level(claims.get("privacy_affinity"))
    floating_allowed = bool(claims.get("floating_allowed"))

    if "wall" in tags:
        score += {"high": 3.0, "medium": 1.8, "low": 0.8, "none": -0.4}[wall_affinity]
        reasons.append(f"wall_affinity={wall_affinity}")
    if "daylight" in tags:
        score += {"high": 3.2, "medium": 1.8, "low": 0.8, "none": -0.5}[
            daylight_affinity
        ]
        reasons.append(f"daylight_affinity={daylight_affinity}")
    if "privacy" in tags or "far_from_entry" in tags:
        score += {"high": 2.8, "medium": 1.5, "low": 0.5, "none": 0.0}[privacy_affinity]
        reasons.append(f"privacy_affinity={privacy_affinity}")
    if "floating" in tags or "center" in tags:
        score += 1.1 if floating_allowed else -0.8
        reasons.append("floating_allowed" if floating_allowed else "floating_limited")

    if role in {"social_anchor", "lounge"} and region.region_id in {
        "primary_focal_wall_zone",
        "floating_support_zone",
    }:
        score += 1.6
        reasons.append("social_anchor_compatible")
    if role in {"media", "focal"} and "focal" in tags:
        score += 2.2
        reasons.append("focal_compatible")
    if role in {"storage", "service"} and ("storage" in tags or "edge" in tags):
        score += 2.0
        reasons.append("storage_edge_compatible")
    if role == "work" and "daylight" in tags:
        score += 2.0
        reasons.append("work_daylight_compatible")
    if role == "sleep" and "privacy" in tags:
        score += 2.2
        reasons.append("sleep_privacy_compatible")
    if "entry_side" in tags and _cluster_avoids_entry(cluster):
        score -= 2.5
        reasons.append("entry_conflict_penalty")
    if "keep_open" in tags:
        score -= 3.5
        reasons.append("keep_open_not_assignment")
    for seed_tag in cluster.seed_region_tags:
        if seed_tag in tags:
            score += 0.35
            reasons.append(f"seed_tag={seed_tag}")

    return RegionCandidate(
        cluster_id=cluster.cluster_id,
        region_id=region.region_id,
        region_type=region.region_type,
        score=round(score, 3),
        reasons=tuple(reasons[:5]),
    )


def _macro_scenario_for_family(family: ConceptFamily) -> MacroScenario:
    scenarios: dict[ConceptFamily, MacroScenario] = {
        "focal_axis": MacroScenario(
            scenario_id="focal_axis_top_bottom",
            label="primary seating top wall, focal media opposite",
            primary_zone_id="top_wall_zone",
            secondary_zone_id="bottom_wall_zone",
            storage_zone_id="right_wall_zone",
            support_zone_id="left_wall_zone",
            primary_wall_side="top_wall",
            secondary_wall_side="bottom_wall",
            storage_wall_side="right_wall",
            pair_type="opposite_walls",
            center_policy="balanced_open",
            primary_anchor_strength="strong",
            secondary_anchor_strength="strong",
            support_anchor_strength="medium",
        ),
        "open_center": MacroScenario(
            scenario_id="open_center_left_right",
            label="primary and media on opposing side walls with open center",
            primary_zone_id="left_wall_zone",
            secondary_zone_id="right_wall_zone",
            storage_zone_id="bottom_wall_zone",
            support_zone_id="top_wall_zone",
            primary_wall_side="left_wall",
            secondary_wall_side="right_wall",
            storage_wall_side="bottom_wall",
            pair_type="opposite_walls",
            center_policy="open_reserved",
            primary_anchor_strength="strong",
            secondary_anchor_strength="strong",
            support_anchor_strength="medium",
        ),
        "edge_weighted": MacroScenario(
            scenario_id="edge_weighted_bottom_top",
            label="perimeter loaded layout with primary bottom and media top",
            primary_zone_id="bottom_wall_zone",
            secondary_zone_id="top_wall_zone",
            storage_zone_id="left_wall_zone",
            support_zone_id="right_wall_zone",
            primary_wall_side="bottom_wall",
            secondary_wall_side="top_wall",
            storage_wall_side="left_wall",
            pair_type="opposite_walls",
            center_policy="open_reserved",
            primary_anchor_strength="strong",
            secondary_anchor_strength="strong",
            support_anchor_strength="strong",
        ),
        "zoned": MacroScenario(
            scenario_id="floating_primary_right_media",
            label="floating primary seating with wall-backed media zone",
            primary_zone_id="floating_center_zone",
            secondary_zone_id="right_wall_zone",
            storage_zone_id="left_wall_zone",
            support_zone_id="bottom_wall_zone",
            primary_wall_side="center",
            secondary_wall_side="right_wall",
            storage_wall_side="left_wall",
            pair_type="floating_to_wall",
            center_policy="floating_primary",
            primary_anchor_strength="strong",
            secondary_anchor_strength="strong",
            support_anchor_strength="medium",
            allow_primary_center_overlap=True,
        ),
        "daylight_oriented": MacroScenario(
            scenario_id="daylight_primary_bottom_media",
            label="primary seating claims daylight with media opposite",
            primary_zone_id="window_side_zone",
            secondary_zone_id="bottom_wall_zone",
            storage_zone_id="right_wall_zone",
            support_zone_id="left_wall_zone",
            primary_wall_side="window_side",
            secondary_wall_side="bottom_wall",
            storage_wall_side="right_wall",
            pair_type="daylight_opposite",
            center_policy="balanced_open",
            primary_anchor_strength="strong",
            secondary_anchor_strength="strong",
            support_anchor_strength="medium",
        ),
    }
    return scenarios[family]


def _macro_scenario_to_dict(scenario: MacroScenario) -> dict[str, object]:
    return {
        "scenario_id": scenario.scenario_id,
        "label": scenario.label,
        "primary_zone_id": scenario.primary_zone_id,
        "secondary_zone_id": scenario.secondary_zone_id,
        "storage_zone_id": scenario.storage_zone_id,
        "support_zone_id": scenario.support_zone_id,
        "primary_wall_side": scenario.primary_wall_side,
        "secondary_wall_side": scenario.secondary_wall_side,
        "storage_wall_side": scenario.storage_wall_side,
        "pair_type": scenario.pair_type,
        "center_policy": scenario.center_policy,
        "primary_anchor_strength": scenario.primary_anchor_strength,
        "secondary_anchor_strength": scenario.secondary_anchor_strength,
        "support_anchor_strength": scenario.support_anchor_strength,
        "allow_primary_center_overlap": scenario.allow_primary_center_overlap,
    }


def _instantiate_concept_families(
    *,
    room_model: Mapping[str, object],
    room_type: str,
    cluster_programs: Sequence[ClusterProgram],
    macro_region_map: Mapping[str, object],
    region_candidates: Sequence[RegionCandidate],
    style_policy: Mapping[str, object],
    target_count: int,
    guidance_bundle: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    family_order = _concept_family_order(
        style_policy=style_policy,
        guidance_bundle=guidance_bundle,
    )
    concept_count = max(1, min(int(target_count), len(family_order)))
    primary_cluster_id = _primary_cluster_id(cluster_programs)
    secondary_cluster_id = _secondary_cluster_id(cluster_programs, primary_cluster_id)
    guidance_by_family = _guidance_by_family(guidance_bundle)
    concepts: list[dict[str, object]] = []
    for index, family in enumerate(family_order[:concept_count], start=1):
        scenario = _macro_scenario_for_family(family)
        family_guidance = guidance_by_family.get(family, {})
        topology_policy = _merge_topology_policy_guidance(
            _topology_policy_for_family(family, style_policy=style_policy),
            family_guidance=family_guidance,
        )
        cluster_zone_plan = [
            _cluster_zone_assignment(
                family=family,
                scenario=scenario,
                cluster=cluster,
                primary_cluster_id=primary_cluster_id,
                secondary_cluster_id=secondary_cluster_id,
                region_candidates=region_candidates,
                topology_policy=topology_policy,
            )
            for cluster in cluster_programs
        ]
        baseline_cluster_zone_plan = _resolve_complementary_zone_assignments(
            family=family,
            scenario=scenario,
            cluster_programs=cluster_programs,
            cluster_zone_plan=cluster_zone_plan,
            primary_cluster_id=primary_cluster_id,
            secondary_cluster_id=secondary_cluster_id,
        )
        cluster_zone_plan = _apply_family_guidance_to_cluster_zone_plan(
            family=family,
            cluster_programs=cluster_programs,
            baseline_cluster_zone_plan=baseline_cluster_zone_plan,
            family_guidance=family_guidance,
            primary_cluster_id=primary_cluster_id,
            secondary_cluster_id=secondary_cluster_id,
        )
        cluster_zone_plan = _apply_zone_forbidden_guardrails(
            room_model=room_model,
            macro_region_map=macro_region_map,
            cluster_programs=cluster_programs,
            family=family,
            cluster_zone_plan=cluster_zone_plan,
        )
        cluster_zone_plan = _apply_placement_behavior_contracts(cluster_zone_plan)
        macro_constraints = _macro_constraints_for_concept(
            family=family,
            scenario=scenario,
            room_model=room_model,
            macro_region_map=macro_region_map,
            cluster_zone_plan=cluster_zone_plan,
            primary_cluster_id=primary_cluster_id,
            secondary_cluster_id=secondary_cluster_id,
        )
        primary_pair_contracts = _primary_pair_contracts_for_concept(
            family=family,
            scenario=scenario,
            cluster_zone_plan=cluster_zone_plan,
            primary_cluster_id=primary_cluster_id,
            secondary_cluster_id=secondary_cluster_id,
        )
        concept = {
            "concept_id": f"concept_{index:02d}",
            "concept_family": family,
            "macro_scenario": _macro_scenario_to_dict(scenario),
            "spatial_character": _spatial_character(family, room_type),
            "macro_region_map_snapshot": dict(macro_region_map),
            "cluster_zone_plan": cluster_zone_plan,
            "allowed_variant_families_by_cluster": (
                _allowed_variant_families_by_cluster(
                    family=family,
                    cluster_zone_plan=cluster_zone_plan,
                )
            ),
            "critical_object_orientations": _critical_object_orientations_for_concept(
                family=family,
                cluster_programs=cluster_programs,
                cluster_zone_plan=cluster_zone_plan,
                primary_cluster_id=primary_cluster_id,
                secondary_cluster_id=secondary_cluster_id,
            ),
            "primary_pair_contracts": primary_pair_contracts,
            "anchor_region_preferences_by_cluster": (
                _anchor_region_preferences_by_cluster(
                    cluster_zone_plan=cluster_zone_plan,
                )
            ),
            "anchor_pair_contracts": _anchor_pair_contracts_for_concept(
                primary_pair_contracts=primary_pair_contracts,
                primary_cluster_id=primary_cluster_id,
                secondary_cluster_id=secondary_cluster_id,
                cluster_programs=cluster_programs,
            ),
            "object_solver_policy": _object_solver_policy_for_concept(
                family=family,
                cluster_programs=cluster_programs,
            ),
            "topology_policy": topology_policy,
            "macro_constraints": macro_constraints,
            "concept_readiness_requirements": _concept_readiness_requirements(
                family=family,
                primary_pair_contracts=primary_pair_contracts,
            ),
            "variant_bias_weights": _variant_bias_weights_for_concept(
                family,
                style_policy=style_policy,
            ),
            "concept_score_prior": _concept_score_prior(
                family,
                style_policy=style_policy,
            ),
            "llm_guided": bool(family_guidance),
            "guidance_notes": list(_string_list(family_guidance.get("notes"))),
            "diversity_signature": _diversity_signature(
                family=family,
                cluster_zone_plan=cluster_zone_plan,
                topology_policy=topology_policy,
            ),
        }
        concepts.append(concept)
    return concepts


def _concept_family_order(
    *,
    style_policy: Mapping[str, object],
    guidance_bundle: Mapping[str, object] | None,
) -> tuple[ConceptFamily, ...]:
    default_order = list(_style_ranked_concept_families(style_policy))
    if not isinstance(guidance_bundle, Mapping):
        return tuple(default_order)
    raw_order = _string_list(guidance_bundle.get("family_order"))
    ordered = [
        family
        for family in raw_order
        if family in CONCEPT_FAMILIES and family in default_order
    ]
    for family in default_order:
        if family not in ordered:
            ordered.append(family)
    return tuple(ordered[: len(CONCEPT_FAMILIES)])  # type: ignore[return-value]


def _guidance_by_family(
    guidance_bundle: Mapping[str, object] | None,
) -> dict[ConceptFamily, Mapping[str, object]]:
    if not isinstance(guidance_bundle, Mapping):
        return {}
    out: dict[ConceptFamily, Mapping[str, object]] = {}
    for item in _sequence_or_empty(guidance_bundle.get("concept_blueprints")):
        if not isinstance(item, Mapping):
            continue
        family = _clean_str(item.get("concept_family"))
        if family not in CONCEPT_FAMILIES:
            continue
        out[family] = item  # type: ignore[assignment]
    return out


def _merge_topology_policy_guidance(
    base_policy: Mapping[str, object],
    *,
    family_guidance: Mapping[str, object],
) -> dict[str, object]:
    merged = dict(base_policy)
    guided_policy = _mapping_or_empty(family_guidance.get("topology_policy"))
    for key in _GUIDED_TOPOLOGY_POLICY_KEYS:
        value = _clean_str(guided_policy.get(key))
        if value is not None:
            merged[key] = value
    return merged


def _apply_family_guidance_to_cluster_zone_plan(
    *,
    family: ConceptFamily,
    cluster_programs: Sequence[ClusterProgram],
    baseline_cluster_zone_plan: Sequence[Mapping[str, object]],
    family_guidance: Mapping[str, object],
    primary_cluster_id: str | None,
    secondary_cluster_id: str | None,
) -> list[dict[str, object]]:
    rows = [dict(row) for row in baseline_cluster_zone_plan]
    if not family_guidance:
        return rows

    clusters_by_id = {cluster.cluster_id: cluster for cluster in cluster_programs}
    baseline_by_cluster = {
        str(row.get("cluster_id")): dict(row)
        for row in baseline_cluster_zone_plan
        if _clean_str(row.get("cluster_id")) is not None
    }
    for override in _sequence_or_empty(family_guidance.get("cluster_zone_overrides")):
        if not isinstance(override, Mapping):
            continue
        cluster_id = _clean_str(override.get("cluster_id"))
        zone_assignment = _clean_str(override.get("zone_assignment"))
        if cluster_id is None or zone_assignment is None:
            continue
        row = _zone_row_by_cluster(rows, cluster_id)
        cluster = clusters_by_id.get(cluster_id)
        baseline_row = baseline_by_cluster.get(cluster_id)
        if row is None or cluster is None or baseline_row is None:
            continue
        if zone_assignment == "keep_open_center":
            continue
        if zone_assignment in _string_list(row.get("forbidden_region_ids")):
            continue
        _apply_zone_assignment_to_row(
            row=row,
            cluster=cluster,
            family=family,
            zone_assignment=zone_assignment,
        )
        center_usage = _guided_center_usage(
            raw_value=override.get("center_usage"),
            cluster_id=cluster_id,
            primary_cluster_id=primary_cluster_id,
        )
        row["center_usage"] = center_usage
        preferred_wall_side = _guided_wall_side(
            raw_value=override.get("preferred_wall_side"),
            fallback=str(baseline_row.get("preferred_wall_side") or ""),
        )
        row["preferred_wall_side"] = preferred_wall_side
        if bool(row.get("scenario_locked")):
            row["required_region_ids"] = _required_region_ids_for_assignment(
                zone_assignment=zone_assignment,
                scenario_role=str(row.get("scenario_role") or ""),
            )
        row["placement_bias"] = _placement_bias(
            zone_assignment,
            str(row.get("wall_claim") or "none"),
            center_usage,
        )
        row["guidance_applied"] = True
    return _stabilize_guided_cluster_zone_plan(
        rows=rows,
        baseline_by_cluster=baseline_by_cluster,
        primary_cluster_id=primary_cluster_id,
        secondary_cluster_id=secondary_cluster_id,
    )


def _guided_center_usage(
    *,
    raw_value: object,
    cluster_id: str,
    primary_cluster_id: str | None,
) -> str:
    value = str(raw_value or "").strip()
    if value not in _CENTER_USAGE_VALUES:
        return "primary" if cluster_id == primary_cluster_id else "none"
    if cluster_id != primary_cluster_id and value in {"primary", "open_reserved"}:
        return "partial"
    return value


def _guided_wall_side(*, raw_value: object, fallback: str) -> str:
    value = str(raw_value or "").strip()
    if value in _GUIDED_WALL_SIDES:
        return value
    return fallback


def _stabilize_guided_cluster_zone_plan(
    *,
    rows: Sequence[dict[str, object]],
    baseline_by_cluster: Mapping[str, Mapping[str, object]],
    primary_cluster_id: str | None,
    secondary_cluster_id: str | None,
) -> list[dict[str, object]]:
    stabilized = [dict(row) for row in rows]
    primary_center_owner = False
    for row in stabilized:
        cluster_id = _clean_str(row.get("cluster_id"))
        if cluster_id is None:
            continue
        if str(row.get("zone_assignment") or "") == "keep_open_center":
            baseline = baseline_by_cluster.get(cluster_id)
            if baseline is not None:
                row.update(dict(baseline))
            continue
        center_usage = str(row.get("center_usage") or "none")
        if cluster_id == primary_cluster_id:
            if center_usage == "primary":
                primary_center_owner = True
            elif center_usage not in {"partial", "none", "open_reserved"}:
                row["center_usage"] = "primary"
                primary_center_owner = True
        elif center_usage in {"primary", "open_reserved"}:
            row["center_usage"] = "partial"
    if not primary_center_owner and primary_cluster_id is not None:
        row = _zone_row_by_cluster(stabilized, primary_cluster_id)
        if row is not None:
            row["center_usage"] = "primary"

    if primary_cluster_id and secondary_cluster_id:
        primary_row = _zone_row_by_cluster(stabilized, primary_cluster_id)
        secondary_row = _zone_row_by_cluster(stabilized, secondary_cluster_id)
        baseline_secondary = baseline_by_cluster.get(secondary_cluster_id)
        if (
            primary_row is not None
            and secondary_row is not None
            and primary_row.get("zone_assignment")
            == secondary_row.get("zone_assignment")
            and baseline_secondary is not None
        ):
            secondary_row.update(dict(baseline_secondary))
    return stabilized


def _cluster_zone_assignment(
    *,
    family: ConceptFamily,
    scenario: MacroScenario,
    cluster: ClusterProgram,
    primary_cluster_id: str | None,
    secondary_cluster_id: str | None,
    region_candidates: Sequence[RegionCandidate],
    topology_policy: Mapping[str, object],
) -> dict[str, object]:
    preferred_region_id = _preferred_region_for_family(
        family=family,
        scenario=scenario,
        cluster=cluster,
        primary_cluster_id=primary_cluster_id,
        secondary_cluster_id=secondary_cluster_id,
    )
    candidates = [
        candidate
        for candidate in region_candidates
        if candidate.cluster_id == cluster.cluster_id
    ]
    candidate = next(
        (item for item in candidates if item.region_id == preferred_region_id),
        candidates[0] if candidates else None,
    )
    zone_assignment = preferred_region_id
    if candidate is not None and family not in {"open_center", "edge_weighted"}:
        zone_assignment = candidate.region_id
    scenario_role = _scenario_role_for_cluster(
        cluster=cluster,
        primary_cluster_id=primary_cluster_id,
        secondary_cluster_id=secondary_cluster_id,
    )
    if scenario_role in {"primary", "secondary"}:
        zone_assignment = preferred_region_id
    wall_claim = _wall_claim_for(cluster, zone_assignment, family)
    center_usage = _center_usage_for(
        cluster=cluster,
        family=family,
        primary_cluster_id=primary_cluster_id,
        scenario=scenario,
        scenario_role=scenario_role,
    )
    anchor_strength = _anchor_strength_for_scenario_role(
        scenario=scenario,
        scenario_role=scenario_role,
    )
    preferred_wall_side = _preferred_wall_side_for_scenario_role(
        scenario=scenario,
        scenario_role=scenario_role,
    )
    forbidden_region_ids = _uniq(
        [
            *_forbidden_region_ids_for_scenario_role(
                scenario=scenario,
                scenario_role=scenario_role,
            ),
            *_cluster_avoid_region_ids(cluster),
        ]
    )
    return {
        "cluster_id": cluster.cluster_id,
        "semantic_role": cluster.semantic_role,
        "layout_role": cluster.layout_role,
        "role_kind": cluster.role_kind,
        "priority": cluster.priority,
        "macro_scenario_id": scenario.scenario_id,
        "scenario_role": scenario_role,
        "scenario_locked": scenario_role in {"primary", "secondary"},
        "zone_assignment": zone_assignment,
        "required_region_ids": _required_region_ids_for_assignment(
            zone_assignment=zone_assignment,
            scenario_role=scenario_role,
        ),
        "preferred_region_ids": _preferred_region_ids_for_assignment(
            zone_assignment=zone_assignment,
            candidates=candidates,
        ),
        "forbidden_region_ids": forbidden_region_ids,
        "preferred_wall_side": preferred_wall_side,
        "anchor_strength": anchor_strength,
        "region_candidates": [
            _region_candidate_to_dict(item) for item in candidates[:3]
        ],
        "wall_claim": wall_claim,
        "center_usage": center_usage,
        "entry_relation": _entry_relation_for(cluster, topology_policy),
        "daylight_relation": _daylight_relation_for(cluster, zone_assignment, family),
        "privacy_relation": _privacy_relation_for(cluster, zone_assignment),
        "placement_bias": _placement_bias(zone_assignment, wall_claim, center_usage),
    }


def _resolve_complementary_zone_assignments(
    *,
    family: ConceptFamily,
    scenario: MacroScenario,
    cluster_programs: Sequence[ClusterProgram],
    cluster_zone_plan: Sequence[Mapping[str, object]],
    primary_cluster_id: str | None,
    secondary_cluster_id: str | None,
) -> list[dict[str, object]]:
    rows = [dict(row) for row in cluster_zone_plan]
    if primary_cluster_id is None or secondary_cluster_id is None:
        return rows

    clusters_by_id = {cluster.cluster_id: cluster for cluster in cluster_programs}
    primary = clusters_by_id.get(primary_cluster_id)
    secondary = clusters_by_id.get(secondary_cluster_id)
    if primary is None or secondary is None:
        return rows

    primary_row = _zone_row_by_cluster(rows, primary_cluster_id)
    secondary_row = _zone_row_by_cluster(rows, secondary_cluster_id)
    if primary_row is None or secondary_row is None:
        return rows
    if primary.priority != "core" or secondary.priority != "core":
        return rows
    if bool(primary_row.get("scenario_locked")) or bool(
        secondary_row.get("scenario_locked")
    ):
        _mark_zone_complement_role(
            rows,
            focal_cluster_id=_focal_claim_cluster_id(
                primary=primary,
                secondary=secondary,
                primary_cluster_id=primary_cluster_id,
            ),
            primary_cluster_id=primary_cluster_id,
            secondary_cluster_id=secondary_cluster_id,
        )
        return rows
    if not _zone_claims_same_logic_class(primary_row, secondary_row):
        _mark_zone_complement_role(
            rows,
            focal_cluster_id=_focal_claim_cluster_id(
                primary=primary,
                secondary=secondary,
                primary_cluster_id=primary_cluster_id,
            ),
            primary_cluster_id=primary_cluster_id,
            secondary_cluster_id=secondary_cluster_id,
        )
        return rows

    focal_cluster_id = _focal_claim_cluster_id(
        primary=primary,
        secondary=secondary,
        primary_cluster_id=primary_cluster_id,
    )
    focal = clusters_by_id[focal_cluster_id]
    focal_row = _zone_row_by_cluster(rows, focal_cluster_id)
    if focal_row is not None:
        _apply_zone_assignment_to_row(
            row=focal_row,
            cluster=focal,
            family=family,
            zone_assignment="primary_focal_wall_zone",
        )
    complement_cluster_id = (
        secondary_cluster_id
        if focal_cluster_id == primary_cluster_id
        else primary_cluster_id
    )
    complement = clusters_by_id[complement_cluster_id]
    complement_row = _zone_row_by_cluster(rows, complement_cluster_id)
    if complement_row is not None:
        zone_assignment = _complementary_zone_for_cluster(
            family=family,
            scenario=scenario,
            cluster=complement,
        )
        _apply_zone_assignment_to_row(
            row=complement_row,
            cluster=complement,
            family=family,
            zone_assignment=zone_assignment,
        )
    _mark_zone_complement_role(
        rows,
        focal_cluster_id=focal_cluster_id,
        primary_cluster_id=primary_cluster_id,
        secondary_cluster_id=secondary_cluster_id,
    )
    return rows


def _zone_row_by_cluster(
    rows: Sequence[dict[str, object]],
    cluster_id: str,
) -> dict[str, object] | None:
    for row in rows:
        if row.get("cluster_id") == cluster_id:
            return row
    return None


def _zone_claims_same_logic_class(
    first: Mapping[str, object],
    second: Mapping[str, object],
) -> bool:
    return _zone_logic_class(first.get("zone_assignment")) == _zone_logic_class(
        second.get("zone_assignment")
    )


def _zone_logic_class(value: object) -> str:
    zone = str(value or "").strip()
    if "focal" in zone:
        return "focal_wall"
    if "daylight" in zone:
        return "daylight"
    if "edge" in zone or "storage" in zone:
        return "perimeter"
    if "private" in zone:
        return "private"
    if "floating" in zone or "center" in zone:
        return "center"
    if "entry" in zone:
        return "entry"
    return zone or "unknown"


def _focal_claim_cluster_id(
    *,
    primary: ClusterProgram,
    secondary: ClusterProgram,
    primary_cluster_id: str,
) -> str:
    for cluster in (primary, secondary):
        if cluster.role_kind in {"media", "focal"}:
            return cluster.cluster_id
    return primary_cluster_id


def _scenario_role_for_cluster(
    *,
    cluster: ClusterProgram,
    primary_cluster_id: str | None,
    secondary_cluster_id: str | None,
) -> str:
    if cluster.cluster_id == primary_cluster_id:
        return "primary"
    if cluster.cluster_id == secondary_cluster_id:
        return "secondary"
    if cluster.layout_role == "optional":
        return "optional"
    return "support"


def _anchor_strength_for_scenario_role(
    *,
    scenario: MacroScenario,
    scenario_role: str,
) -> AnchorStrength:
    if scenario_role == "primary":
        return scenario.primary_anchor_strength
    if scenario_role == "secondary":
        return scenario.secondary_anchor_strength
    return scenario.support_anchor_strength


def _preferred_wall_side_for_scenario_role(
    *,
    scenario: MacroScenario,
    scenario_role: str,
) -> str:
    if scenario_role == "primary":
        return scenario.primary_wall_side
    if scenario_role == "secondary":
        return scenario.secondary_wall_side
    return ""


def _forbidden_region_ids_for_scenario_role(
    *,
    scenario: MacroScenario,
    scenario_role: str,
) -> list[str]:
    if scenario.center_policy == "open_reserved" and scenario_role != "primary":
        return ["floating_center_zone", "keep_open_center"]
    if scenario.center_policy == "open_reserved" and scenario_role == "primary":
        return ["floating_center_zone"]
    if scenario.center_policy == "floating_primary" and scenario_role != "primary":
        return ["floating_center_zone", "keep_open_center"]
    return []


def _cluster_avoid_region_ids(cluster: ClusterProgram) -> list[str]:
    return _string_list(cluster.zone_claims.get("avoid_regions"))


def _complementary_zone_for_cluster(
    *,
    family: ConceptFamily,
    scenario: MacroScenario,
    cluster: ClusterProgram,
) -> str:
    if cluster.role_kind == "work":
        return "daylight_biased_zone"
    if cluster.role_kind == "sleep":
        return "quiet_private_deep_zone"
    if family == "daylight_oriented" and cluster.role_kind in {
        "social_anchor",
        "lounge",
    }:
        return "daylight_biased_zone"
    if family == "zoned":
        return scenario.support_zone_id
    return scenario.support_zone_id


def _apply_zone_assignment_to_row(
    *,
    row: dict[str, object],
    cluster: ClusterProgram,
    family: ConceptFamily,
    zone_assignment: str,
) -> None:
    wall_claim = _wall_claim_for(cluster, zone_assignment, family)
    center_usage = str(row.get("center_usage") or "none")
    row["zone_assignment"] = zone_assignment
    if row.get("scenario_role") in {"primary", "secondary"}:
        row["required_region_ids"] = _required_region_ids_for_assignment(
            zone_assignment=zone_assignment,
            scenario_role=str(row.get("scenario_role") or ""),
        )
        row["preferred_region_ids"] = _preferred_region_ids_for_row(row)
        row["preferred_wall_side"] = _preferred_wall_side_for_zone(zone_assignment)
    row["wall_claim"] = wall_claim
    row["daylight_relation"] = _daylight_relation_for(
        cluster,
        zone_assignment,
        family,
    )
    row["privacy_relation"] = _privacy_relation_for(cluster, zone_assignment)
    row["placement_bias"] = _placement_bias(
        zone_assignment,
        wall_claim,
        center_usage,
    )


def _preferred_wall_side_for_zone(zone_assignment: str) -> str:
    token = zone_assignment.strip().lower()
    if "window" in token or "daylight" in token:
        return "window_side"
    for side in ("top_wall", "right_wall", "bottom_wall", "left_wall"):
        if side in token:
            return side
    if "center" in token or "floating" in token:
        return "center"
    return ""


def _required_region_ids_for_assignment(
    *,
    zone_assignment: str,
    scenario_role: str,
) -> list[str]:
    if scenario_role not in {"primary", "secondary"}:
        return []
    if zone_assignment in _FIT_CHECKED_REQUIRED_REGION_IDS:
        return [zone_assignment]
    return []


def _preferred_region_ids_for_assignment(
    *,
    zone_assignment: str,
    candidates: Sequence[RegionCandidate],
) -> list[str]:
    return _uniq([zone_assignment, *(candidate.region_id for candidate in candidates)])


def _preferred_region_ids_for_row(row: Mapping[str, object]) -> list[str]:
    region_ids: list[str] = []
    zone_assignment = _clean_str(row.get("zone_assignment"))
    if zone_assignment is not None:
        region_ids.append(zone_assignment)
    region_ids.extend(_string_list(row.get("preferred_region_ids")))
    for candidate in _sequence_or_empty(row.get("region_candidates")):
        if not isinstance(candidate, Mapping):
            continue
        candidate_id = _clean_str(candidate.get("region_id"))
        if candidate_id is not None:
            region_ids.append(candidate_id)
    return _uniq(region_ids)


def _mark_zone_complement_role(
    rows: Sequence[dict[str, object]],
    *,
    focal_cluster_id: str,
    primary_cluster_id: str,
    secondary_cluster_id: str,
) -> None:
    for row in rows:
        cluster_id = _clean_str(row.get("cluster_id"))
        if cluster_id == focal_cluster_id:
            role: ZoneComplementRole = "focal_claim"
        elif cluster_id in {primary_cluster_id, secondary_cluster_id}:
            role = "topology_complement"
        else:
            role = "neutral"
        row["zone_complement_role"] = role


def _preferred_region_for_family(
    *,
    family: ConceptFamily,
    scenario: MacroScenario,
    cluster: ClusterProgram,
    primary_cluster_id: str | None,
    secondary_cluster_id: str | None,
) -> str:
    scenario_role = _scenario_role_for_cluster(
        cluster=cluster,
        primary_cluster_id=primary_cluster_id,
        secondary_cluster_id=secondary_cluster_id,
    )
    if scenario_role == "primary":
        return scenario.primary_zone_id
    if scenario_role == "secondary":
        return scenario.secondary_zone_id
    if scenario_role in {"support", "optional"} and cluster.role_kind not in {
        "work",
        "sleep",
    }:
        return scenario.support_zone_id
    if family == "focal_axis":
        if cluster.cluster_id in {primary_cluster_id, secondary_cluster_id}:
            return "primary_focal_wall_zone"
        return "storage_service_edge_zone"
    if family == "open_center":
        if cluster.cluster_id in {primary_cluster_id, secondary_cluster_id}:
            return "primary_focal_wall_zone"
        return "edge_loading_zone"
    if family == "edge_weighted":
        return "edge_loading_zone"
    if family == "zoned":
        if cluster.cluster_id == primary_cluster_id:
            return "floating_support_zone"
        if cluster.cluster_id == secondary_cluster_id:
            return "primary_focal_wall_zone"
        if cluster.role_kind == "work":
            return "daylight_biased_zone"
        if cluster.role_kind == "sleep":
            return "quiet_private_deep_zone"
        return "storage_service_edge_zone"
    if family == "daylight_oriented":
        if cluster.role_kind in {"work", "lounge", "social_anchor"}:
            return "daylight_biased_zone"
        if cluster.cluster_id == secondary_cluster_id:
            return "primary_focal_wall_zone"
        return "edge_loading_zone"
    return "primary_focal_wall_zone"


def _topology_policy_for_family(
    family: ConceptFamily,
    *,
    style_policy: Mapping[str, object],
) -> dict[str, object]:
    policies: dict[ConceptFamily, dict[str, object]] = {
        "focal_axis": {
            "preserve_entry_landing": True,
            "preserve_primary_corridor": True,
            "reserve_center_degree": "medium",
            "wall_loading_bias": "focal_wall_first",
            "daylight_bias": "secondary_cluster_preferred",
            "entry_avoidance_strength": "high",
            "secondary_zone_placement_bias": "focal_edge",
        },
        "open_center": {
            "preserve_entry_landing": True,
            "preserve_primary_corridor": True,
            "reserve_center_degree": "high",
            "wall_loading_bias": "balanced_edge",
            "daylight_bias": "support_allowed",
            "entry_avoidance_strength": "high",
            "secondary_zone_placement_bias": "edge_recede",
        },
        "edge_weighted": {
            "preserve_entry_landing": True,
            "preserve_primary_corridor": True,
            "reserve_center_degree": "very_high",
            "wall_loading_bias": "perimeter_heavy",
            "daylight_bias": "avoid_blocking",
            "entry_avoidance_strength": "high",
            "secondary_zone_placement_bias": "perimeter",
        },
        "zoned": {
            "preserve_entry_landing": True,
            "preserve_primary_corridor": True,
            "reserve_center_degree": "medium",
            "wall_loading_bias": "zone_specific",
            "daylight_bias": "role_based",
            "entry_avoidance_strength": "medium",
            "secondary_zone_placement_bias": "separate_support_zone",
        },
        "daylight_oriented": {
            "preserve_entry_landing": True,
            "preserve_primary_corridor": True,
            "reserve_center_degree": "medium",
            "wall_loading_bias": "balanced",
            "daylight_bias": "primary_support_preferred",
            "entry_avoidance_strength": "high",
            "secondary_zone_placement_bias": "daylight_if_not_focal",
        },
    }
    policy = dict(policies[family])
    layout_policy = _style_layout_policy(style_policy)
    if (
        _bias_level(layout_policy.get("center_openness_bias")) >= 3
        and family != "zoned"
    ):
        policy["reserve_center_degree"] = "very_high"
        policy["secondary_zone_placement_bias"] = "edge_recede"
    elif _bias_level(layout_policy.get("center_openness_bias")) <= 1:
        policy["reserve_center_degree"] = "medium"
    wall_loading_bias = str(layout_policy.get("wall_loading_bias") or "")
    if wall_loading_bias in {"medium_high", "perimeter_heavy"}:
        policy["wall_loading_bias"] = "perimeter_heavy"
        policy["secondary_zone_placement_bias"] = "perimeter"
    elif wall_loading_bias == "focal_balanced":
        policy["wall_loading_bias"] = "focal_wall_first"
    if _bias_level(layout_policy.get("daylight_bias")) >= 3:
        policy["daylight_bias"] = "primary_support_preferred"
    policy["style_name"] = str(style_policy.get("style_name") or "")
    policy["style_density_target"] = str(layout_policy.get("target_density") or "")
    policy["style_visual_balance"] = str(layout_policy.get("visual_balance_bias") or "")
    return policy


def _macro_constraints_for_concept(
    *,
    family: ConceptFamily,
    scenario: MacroScenario,
    room_model: Mapping[str, object],
    macro_region_map: Mapping[str, object],
    cluster_zone_plan: Sequence[Mapping[str, object]],
    primary_cluster_id: str | None,
    secondary_cluster_id: str | None,
) -> dict[str, object]:
    protected = _mapping_or_empty(macro_region_map.get("protected_topology"))
    entry_refs = _string_list(protected.get("entry_landing_zones"))
    corridor_refs = _string_list(protected.get("primary_circulation_corridors"))
    center_refs = _string_list(protected.get("center_openness_regions")) or [
        "room_center"
    ]
    window_refs = _opening_ids(
        _mapping_or_empty(room_model.get("openings")).get("windows"), "window"
    )

    keep_open_regions = [
        {"type": "entry_buffer", "near": ref, "priority": "high"} for ref in entry_refs
    ]
    if family in {"open_center", "edge_weighted"}:
        keep_open_regions.extend(
            {"type": "center_lane", "near": ref, "priority": "high"}
            for ref in center_refs
        )
    elif family == "daylight_oriented":
        keep_open_regions.extend(
            {"type": "window_buffer", "near": ref, "priority": "medium"}
            for ref in window_refs
        )

    reserved_regions = [
        {"region": ref, "reason": "entry landing must stay usable"}
        for ref in entry_refs
    ]
    reserved_regions.extend(
        {"region": ref, "reason": "primary circulation corridor must stay usable"}
        for ref in corridor_refs
    )
    if family in {"open_center", "edge_weighted"}:
        reserved_regions.extend(
            {"region": ref, "reason": "center openness is the concept driver"}
            for ref in center_refs
        )

    separation = []
    if family in {"open_center", "edge_weighted", "zoned"} and primary_cluster_id:
        for row in cluster_zone_plan:
            cluster_id = _clean_str(row.get("cluster_id"))
            if cluster_id and cluster_id not in {
                primary_cluster_id,
                secondary_cluster_id,
            }:
                separation.append(
                    {
                        "a": cluster_id,
                        "b": primary_cluster_id,
                        "preference": "separate_macro_fields",
                        "priority": "high" if family != "open_center" else "medium",
                    }
                )

    alignment = []
    if primary_cluster_id and secondary_cluster_id:
        alignment.append(
            {
                "a": primary_cluster_id,
                "b": secondary_cluster_id,
                "preference": "focal_face_axis"
                if family == "focal_axis"
                else scenario.pair_type,
                "priority": "high",
            }
        )
    anchor_region_constraints = [
        {
            "cluster_id": str(row.get("cluster_id")),
            "required_region_ids": list(
                _sequence_or_empty(row.get("required_region_ids"))
            ),
            "preferred_region_ids": list(
                _sequence_or_empty(row.get("preferred_region_ids"))
            ),
            "forbidden_region_ids": list(
                _sequence_or_empty(row.get("forbidden_region_ids"))
            ),
            "preferred_wall_side": str(row.get("preferred_wall_side") or ""),
            "anchor_strength": str(row.get("anchor_strength") or "medium"),
            "scenario_role": str(row.get("scenario_role") or "support"),
        }
        for row in cluster_zone_plan
        if isinstance(row, Mapping) and _clean_str(row.get("cluster_id")) is not None
    ]
    return {
        "keep_open_regions": keep_open_regions,
        "reserved_regions": reserved_regions,
        "protected_topology": _concept_protected_topology(
            protected,
            family=family,
            scenario=scenario,
        ),
        "cluster_separation_preferences": separation,
        "cluster_alignment_preferences": alignment,
        "anchor_region_constraints": anchor_region_constraints,
    }


def _concept_protected_topology(
    protected: Mapping[str, object],
    *,
    family: ConceptFamily,
    scenario: MacroScenario,
) -> list[dict[str, object]]:
    zones = [
        dict(zone)
        for zone in _sequence_or_empty(protected.get("operational_zones"))
        if isinstance(zone, Mapping)
    ]
    for zone in zones:
        if (
            family in {"open_center", "edge_weighted"}
            and zone.get("zone_type") == "center_openness_core"
        ):
            zone["priority"] = "critical"
            zone["max_overlap_ratio"] = 0.05
        elif (
            family == "daylight_oriented"
            and zone.get("zone_type") == "center_openness_core"
        ):
            zone["max_overlap_ratio"] = 0.12
        elif (
            scenario.allow_primary_center_overlap
            and zone.get("zone_type") == "center_openness_core"
        ):
            zone["priority"] = "medium"
            zone["enforcement"] = "soft"
            zone["max_overlap_ratio"] = 0.35
            zone["applies_to"] = ["support_clusters", "interaction_hulls"]
            zone["violation_severity"] = "advisory"
    return zones


def _allowed_variant_families_by_cluster(
    *,
    family: ConceptFamily,
    cluster_zone_plan: Sequence[Mapping[str, object]],
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for row in cluster_zone_plan:
        cluster_id = _clean_str(row.get("cluster_id"))
        if cluster_id is None:
            continue
        families = _allowed_variant_families_for_row(family=family, row=row)
        out[cluster_id] = families
    return out


def _allowed_variant_families_for_row(
    *,
    family: ConceptFamily,
    row: Mapping[str, object],
) -> list[str]:
    role_kind = str(row.get("role_kind") or "support")
    zone_assignment = str(row.get("zone_assignment") or "")
    wall_claim = str(row.get("wall_claim") or "")
    center_usage = str(row.get("center_usage") or "")
    families: list[str] = []

    if role_kind in {"social_anchor", "lounge"}:
        if center_usage in {"partial", "primary", "open_reserved"}:
            families.append("open_center")
        if _zone_logic_class(zone_assignment) in {"perimeter", "focal_wall"}:
            families.append("perimeter_facing")
        families.extend(["conversation_facing", "social_anchor"])
    elif role_kind in {"media", "focal"}:
        families.extend(["media_facing", "wall_backed_focal"])
        families.append("focal_axis" if family == "focal_axis" else "focal_media")
        if wall_claim == "strong":
            families.append("focal_wall")
    elif role_kind == "work":
        if "daylight" in zone_assignment:
            families.extend(["daylight_work", "window_oriented"])
        families.extend(["work_core", "workflow"])
    elif role_kind == "kitchen":
        if wall_claim in {"strong", "medium"}:
            families.append("storage_wall")
        families.extend(["edge_storage", "perimeter_storage", "workflow"])
    elif role_kind == "storage":
        if wall_claim in {"strong", "medium"}:
            families.append("storage_wall")
        families.extend(["edge_storage", "perimeter_storage", "edge_weighted"])
    elif role_kind == "sleep":
        families.extend(
            _sleep_variant_families_for_row(
                family=family,
                zone_assignment=zone_assignment,
                wall_claim=wall_claim,
            )
        )
    else:
        if "daylight" in zone_assignment:
            families.append("window_oriented")
        if _zone_logic_class(zone_assignment) == "perimeter":
            families.append("support_edge")
        families.extend(["base", "perimeter_facing"])

    if family == "open_center" and role_kind not in {
        "media",
        "focal",
        "storage",
        "sleep",
    }:
        families.insert(0, "open_center")
    elif family == "edge_weighted" and role_kind in {"storage", "support"}:
        families.insert(0, "edge_weighted")
    elif (
        family == "daylight_oriented"
        and role_kind != "sleep"
        and "daylight" in zone_assignment
    ):
        families.insert(0, "window_oriented")

    return _bounded_variant_family_list(_uniq(families))


def _sleep_variant_families_for_row(
    *,
    family: ConceptFamily,
    zone_assignment: str,
    wall_claim: str,
) -> list[str]:
    families = list(SLEEP_VARIANT_FAMILIES)
    if wall_claim == "strong":
        families = _promote_variant_family(
            families,
            target_family="headboard_wall_balanced",
        )
    elif wall_claim == "medium":
        families = _promote_variant_family(
            families,
            target_family="headboard_wall_single_side",
        )

    if "daylight" in zone_assignment or family == "daylight_oriented":
        families = _promote_variant_family(
            families,
            target_family="bed_plus_window_side_bench",
        )
    elif family == "edge_weighted":
        families = _promote_variant_family(
            families,
            target_family="bed_plus_storage_buffer",
        )
    elif family in {"focal_axis", "zoned"}:
        families = _promote_variant_family(
            families,
            target_family="headboard_wall_balanced",
        )

    return _uniq(families)


def _promote_variant_family(
    families: Sequence[str],
    *,
    target_family: str,
) -> list[str]:
    promoted = [family for family in families if family != target_family]
    promoted.insert(0, target_family)
    return promoted


def _bounded_variant_family_list(families: Sequence[str]) -> list[str]:
    out = list(families[:MAX_VARIANT_FAMILIES_PER_CLUSTER])
    for fallback in ("base", "support_edge", "perimeter_facing"):
        if len(out) >= MIN_VARIANT_FAMILIES_PER_CLUSTER:
            break
        if fallback not in out:
            out.append(fallback)
    return out[:MAX_VARIANT_FAMILIES_PER_CLUSTER]


def _anchor_region_preferences_by_cluster(
    *,
    cluster_zone_plan: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for row in cluster_zone_plan:
        cluster_id = _clean_str(row.get("cluster_id"))
        if cluster_id is None:
            continue
        out[cluster_id] = {
            "zone_assignment": str(row.get("zone_assignment") or ""),
            "required_region_ids": list(_string_list(row.get("required_region_ids"))),
            "forbidden_region_ids": list(_string_list(row.get("forbidden_region_ids"))),
            "preferred_region_ids": _preferred_region_ids_for_row(row),
            "wall_claim": str(row.get("wall_claim") or "none"),
            "center_usage": str(row.get("center_usage") or "none"),
            "preferred_wall_side": str(row.get("preferred_wall_side") or ""),
            "anchor_strength": str(row.get("anchor_strength") or "medium"),
            "scenario_role": str(row.get("scenario_role") or "support"),
            "entry_relation": str(row.get("entry_relation") or ""),
            "daylight_relation": str(row.get("daylight_relation") or ""),
            "placement_bias": str(row.get("placement_bias") or "balanced"),
            "placement_behavior": _placement_behavior_contract(row),
        }
    return out


def _ensure_placement_behavior_by_cluster(
    preferences_by_cluster: Mapping[str, object],
    *,
    cluster_zone_plan: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {
        str(cluster_id): dict(row)
        for cluster_id, row in preferences_by_cluster.items()
        if isinstance(row, Mapping)
    }
    for row in cluster_zone_plan:
        cluster_id = _clean_str(row.get("cluster_id"))
        if cluster_id is None:
            continue
        target = out.setdefault(cluster_id, {})
        target["placement_behavior"] = _placement_behavior_contract(row)
    return out


def _anchor_layout_hints_by_cluster(
    *,
    cluster_zone_plan: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for row in cluster_zone_plan:
        cluster_id = _clean_str(row.get("cluster_id"))
        if cluster_id is None:
            continue
        out[cluster_id] = {
            "zone_assignment": str(row.get("zone_assignment") or ""),
            "required_region_ids": list(_string_list(row.get("required_region_ids"))),
            "forbidden_region_ids": list(_string_list(row.get("forbidden_region_ids"))),
            "preferred_region_ids": _preferred_region_ids_for_row(row),
            "role_kind": str(row.get("role_kind") or ""),
            "wall_claim": str(row.get("wall_claim") or "none"),
            "center_usage": str(row.get("center_usage") or "none"),
            "preferred_wall_side": str(row.get("preferred_wall_side") or ""),
            "anchor_strength": str(row.get("anchor_strength") or "medium"),
            "scenario_role": str(row.get("scenario_role") or "support"),
            "placement_bias": str(row.get("placement_bias") or "balanced"),
            "placement_behavior": _placement_behavior_contract(row),
        }
    return out


def _anchor_pair_contracts_for_concept(
    *,
    primary_pair_contracts: Sequence[Mapping[str, object]],
    primary_cluster_id: str | None,
    secondary_cluster_id: str | None,
    cluster_programs: Sequence[ClusterProgram],
) -> list[dict[str, object]]:
    required_cluster_ids = {
        cluster.cluster_id
        for cluster in cluster_programs
        if _cluster_has_required_solver_objects(cluster)
    }
    out: list[dict[str, object]] = []
    for row in primary_pair_contracts:
        if not isinstance(row, Mapping):
            continue
        next_row = dict(row)
        row_clusters = {
            str(next_row.get("cluster_a") or ""),
            str(next_row.get("cluster_b") or ""),
        }
        if not row_clusters.issubset(required_cluster_ids):
            next_row["required"] = False
            next_row["strength"] = str(next_row.get("strength") or "medium")
        out.append(next_row)
    if (
        not out
        and primary_cluster_id
        and secondary_cluster_id
        and primary_cluster_id in required_cluster_ids
        and secondary_cluster_id in required_cluster_ids
    ):
        out.append(
            {
                "pair_type": "face_each_other",
                "cluster_a": primary_cluster_id,
                "cluster_b": secondary_cluster_id,
                "strength": "high",
                "required": True,
            }
        )
    return out


def _cluster_has_required_solver_objects(cluster: ClusterProgram) -> bool:
    has_explicit_object_policy = bool(
        cluster.required_object_ids
        or cluster.optional_object_ids
        or cluster.droppable_object_ids
    )
    if has_explicit_object_policy:
        return bool(cluster.required_object_ids)
    return bool(cluster.object_ids)


def _object_solver_policy_for_concept(
    *,
    family: ConceptFamily,
    cluster_programs: Sequence[ClusterProgram],
) -> dict[str, object]:
    protected_by_cluster: dict[str, list[str]] = {}
    droppable_by_cluster: dict[str, list[str]] = {}
    placement_order_by_cluster: dict[str, list[str]] = {}
    for cluster in cluster_programs:
        placement_order_by_cluster[cluster.cluster_id] = list(cluster.object_ids)
        required_ids = list(cluster.required_object_ids)
        has_explicit_object_policy = bool(
            cluster.required_object_ids
            or cluster.optional_object_ids
            or cluster.droppable_object_ids
        )
        if (
            not required_ids
            and not has_explicit_object_policy
            and cluster.dominant_anchor_object_id
        ):
            required_ids = [cluster.dominant_anchor_object_id]
        protected_by_cluster[cluster.cluster_id] = required_ids
        optional_ids = [
            object_id
            for object_id in cluster.object_ids
            if object_id not in protected_by_cluster.get(cluster.cluster_id, [])
        ]
        droppable_by_cluster[cluster.cluster_id] = _uniq(
            [*list(cluster.droppable_object_ids), *optional_ids]
        )
    return {
        "placement_mode": "anchor_first_object_level",
        "anchor_stage_first": True,
        "family": family,
        "support_stage_enabled": True,
        "micro_refine_inside_solver": True,
        "placement_order_by_cluster": placement_order_by_cluster,
        "protected_ids_by_cluster": protected_by_cluster,
        "droppable_ids_by_cluster": droppable_by_cluster,
    }


def _critical_object_orientations_for_concept(
    *,
    family: ConceptFamily,
    cluster_programs: Sequence[ClusterProgram],
    cluster_zone_plan: Sequence[Mapping[str, object]],
    primary_cluster_id: str | None,
    secondary_cluster_id: str | None,
) -> list[dict[str, object]]:
    rows_by_cluster = {
        str(row.get("cluster_id")): row
        for row in cluster_zone_plan
        if _clean_str(row.get("cluster_id")) is not None
    }
    out: list[dict[str, object]] = []
    for cluster in cluster_programs:
        row = rows_by_cluster.get(cluster.cluster_id)
        if row is None:
            continue
        object_id = _critical_object_id(cluster)
        if object_id is None or not _object_orientation_is_critical(cluster, row):
            continue
        target_cluster_id = _orientation_target_cluster_id(
            cluster_id=cluster.cluster_id,
            primary_cluster_id=primary_cluster_id,
            secondary_cluster_id=secondary_cluster_id,
        )
        intents = _critical_object_orientation_intents(
            family=family,
            cluster=cluster,
            row=row,
            target_cluster_id=target_cluster_id,
        )
        if not intents:
            continue
        item: dict[str, object] = {
            "cluster_id": cluster.cluster_id,
            "object_id": object_id,
            "intents": intents,
            "priority": "high"
            if cluster.cluster_id in {primary_cluster_id, secondary_cluster_id}
            else "medium",
            "reason": "Critical object orientation emitted by macro concept.",
        }
        if target_cluster_id is not None and "face_cluster" in intents:
            item["target_cluster_id"] = target_cluster_id
        else:
            item["target_region"] = str(row.get("zone_assignment") or "")
        out.append(item)
    return out


def _critical_object_id(cluster: ClusterProgram) -> str | None:
    return cluster.object_ids[0] if cluster.object_ids else None


def _object_orientation_is_critical(
    cluster: ClusterProgram,
    row: Mapping[str, object],
) -> bool:
    if cluster.priority == "core":
        return True
    if cluster.role_kind in {"work", "media", "focal"}:
        return True
    return False


def _orientation_target_cluster_id(
    *,
    cluster_id: str,
    primary_cluster_id: str | None,
    secondary_cluster_id: str | None,
) -> str | None:
    if cluster_id == primary_cluster_id:
        return secondary_cluster_id
    if cluster_id == secondary_cluster_id:
        return primary_cluster_id
    return None


def _critical_object_orientation_intents(
    *,
    family: ConceptFamily,
    cluster: ClusterProgram,
    row: Mapping[str, object],
    target_cluster_id: str | None,
) -> list[str]:
    intents: list[str] = []
    if row.get("wall_claim") in {"strong", "medium"}:
        intents.append("back_to_wall")
    if target_cluster_id is not None:
        intents.extend(["face_cluster", "align_with_focal_axis"])
    if cluster.role_kind in {"social_anchor", "lounge", "work"}:
        intents.append("front_to_open_space")
    if cluster.role_kind in {"media", "focal"}:
        intents.extend(["front_to_open_space", "avoid_entry_axis"])
    if family in {"open_center", "edge_weighted"}:
        intents.append("front_to_open_space")
    if not intents and cluster.priority == "core":
        intents.append("front_to_open_space")
    return _uniq(intents)


def _primary_pair_contracts_for_concept(
    *,
    family: ConceptFamily,
    scenario: MacroScenario,
    cluster_zone_plan: Sequence[Mapping[str, object]],
    primary_cluster_id: str | None,
    secondary_cluster_id: str | None,
) -> list[dict[str, object]]:
    if primary_cluster_id is None or secondary_cluster_id is None:
        return []
    primary_row = next(
        (
            row
            for row in cluster_zone_plan
            if row.get("cluster_id") == primary_cluster_id
        ),
        {},
    )
    secondary_row = next(
        (
            row
            for row in cluster_zone_plan
            if row.get("cluster_id") == secondary_cluster_id
        ),
        {},
    )
    pair_type = _primary_pair_type_for_family(
        family=family,
        scenario=scenario,
        primary_row=primary_row,
        secondary_row=secondary_row,
    )
    contracts = [
        {
            "pair_type": pair_type,
            "cluster_a": primary_cluster_id,
            "cluster_b": secondary_cluster_id,
            "strength": "high",
            "required": True,
        }
    ]
    if family in {"focal_axis", "daylight_oriented"}:
        contracts.append(
            {
                "pair_type": "supports_use_axis",
                "cluster_a": primary_cluster_id,
                "cluster_b": secondary_cluster_id,
                "strength": "medium",
                "required": family == "focal_axis",
            }
        )
    return contracts


def _primary_pair_type_for_family(
    *,
    family: ConceptFamily,
    scenario: MacroScenario,
    primary_row: Mapping[str, object],
    secondary_row: Mapping[str, object],
) -> str:
    if scenario.pair_type:
        return scenario.pair_type
    if family == "zoned":
        return "separate_but_legible"
    if family == "edge_weighted":
        return "buffered_support"
    if family == "open_center":
        return "ring_around_anchor"
    if (
        primary_row.get("zone_complement_role") == "focal_claim"
        or secondary_row.get("zone_complement_role") == "focal_claim"
    ):
        return "face_each_other"
    return "supports_use_axis"


def _concept_readiness_requirements(
    *,
    family: ConceptFamily,
    primary_pair_contracts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    required_pair_contracts = [
        str(contract.get("pair_type"))
        for contract in primary_pair_contracts
        if bool(contract.get("required")) and contract.get("pair_type")
    ]
    diagnostics = [
        "circulation_clear",
        "entry_preserved",
        "dominant_anchor_correct",
        "workflow_preserved",
    ]
    if family in {"open_center", "edge_weighted"}:
        diagnostics.append("center_openness_preserved")
    return {
        "required_publishable_diagnostics": diagnostics,
        "required_pair_contracts": required_pair_contracts,
        "required_center_preservation": "high"
        if family in {"open_center", "edge_weighted"}
        else "medium",
        "required_entry_preservation": True,
    }


def _variant_bias_weights_for_concept(
    family: ConceptFamily,
    *,
    style_policy: Mapping[str, object],
) -> dict[str, float]:
    weights: dict[ConceptFamily, dict[str, float]] = {
        "focal_axis": {
            "family_match_weight": 1.45,
            "center_preserve_weight": 1.10,
            "focal_axis_weight": 1.55,
            "entry_clear_weight": 1.45,
            "wall_loading_weight": 1.35,
            "daylight_alignment_weight": 1.05,
            "edge_loading_weight": 1.15,
            "symmetry_weight": 1.35,
        },
        "open_center": {
            "family_match_weight": 1.40,
            "center_preserve_weight": 1.55,
            "focal_axis_weight": 1.25,
            "entry_clear_weight": 1.50,
            "wall_loading_weight": 1.15,
            "daylight_alignment_weight": 1.10,
            "edge_loading_weight": 1.25,
            "symmetry_weight": 1.10,
        },
        "edge_weighted": {
            "family_match_weight": 1.35,
            "center_preserve_weight": 1.50,
            "focal_axis_weight": 1.10,
            "entry_clear_weight": 1.45,
            "wall_loading_weight": 1.50,
            "daylight_alignment_weight": 1.00,
            "edge_loading_weight": 1.55,
            "symmetry_weight": 1.05,
        },
        "zoned": {
            "family_match_weight": 1.35,
            "center_preserve_weight": 1.20,
            "focal_axis_weight": 1.20,
            "entry_clear_weight": 1.35,
            "wall_loading_weight": 1.20,
            "daylight_alignment_weight": 1.15,
            "edge_loading_weight": 1.25,
            "symmetry_weight": 1.05,
        },
        "daylight_oriented": {
            "family_match_weight": 1.40,
            "center_preserve_weight": 1.20,
            "focal_axis_weight": 1.30,
            "entry_clear_weight": 1.45,
            "wall_loading_weight": 1.15,
            "daylight_alignment_weight": 1.55,
            "edge_loading_weight": 1.10,
            "symmetry_weight": 1.05,
        },
    }
    out = dict(weights[family])
    layout_policy = _style_layout_policy(style_policy)
    if _bias_level(layout_policy.get("center_openness_bias")) >= 3:
        out["center_preserve_weight"] = min(out["center_preserve_weight"] + 0.15, 1.7)
    if _bias_level(layout_policy.get("daylight_bias")) >= 3:
        out["daylight_alignment_weight"] = min(
            out["daylight_alignment_weight"] + 0.15,
            1.7,
        )
    if str(layout_policy.get("wall_loading_bias") or "") in {
        "medium_high",
        "perimeter_heavy",
    }:
        out["wall_loading_weight"] = min(out["wall_loading_weight"] + 0.15, 1.7)
        out["edge_loading_weight"] = min(out["edge_loading_weight"] + 0.10, 1.7)
    return out


def _concept_to_solver_plan(
    *,
    concept: Mapping[str, object],
    room_model: Mapping[str, object],
    room_type: str,
) -> dict[str, object]:
    cluster_zone_plan = [
        row
        for row in _sequence_or_empty(concept.get("cluster_zone_plan"))
        if isinstance(row, Mapping)
    ]
    cluster_zone_plan = _apply_placement_behavior_contracts(cluster_zone_plan)
    primary_cluster_id = _primary_from_concept(cluster_zone_plan)
    secondary_cluster_id = _secondary_from_concept(
        cluster_zone_plan, primary_cluster_id
    )
    family = str(concept.get("concept_family") or "focal_axis")
    policy = _mapping_or_empty(concept.get("topology_policy"))
    constraints = _mapping_or_empty(concept.get("macro_constraints"))
    pair_contracts = [
        row
        for row in _sequence_or_empty(concept.get("primary_pair_contracts"))
        if isinstance(row, Mapping)
    ]
    if not pair_contracts and primary_cluster_id and secondary_cluster_id:
        pair_contracts = [
            {
                "pair_type": "face_each_other",
                "cluster_a": primary_cluster_id,
                "cluster_b": secondary_cluster_id,
                "strength": "high",
                "required": True,
            }
        ]

    affinities = [
        _solver_affinity_for_assignment(row, policy) for row in cluster_zone_plan
    ]
    orientations = [
        item
        for row in cluster_zone_plan
        if (
            item := _solver_orientation_for_assignment(
                row, primary_cluster_id, secondary_cluster_id
            )
        )
        is not None
    ]
    relations = []
    directional_relations = []
    relations.extend(_solver_relations_from_pair_contracts(pair_contracts))
    directional_relations.extend(
        _solver_directional_relations_from_pair_contracts(pair_contracts)
    )
    for row in _sequence_or_empty(constraints.get("cluster_separation_preferences")):
        if not isinstance(row, Mapping):
            continue
        a = _clean_str(row.get("a"))
        b = _clean_str(row.get("b"))
        if a and b and a != b:
            relations.append(
                {
                    "a": a,
                    "b": b,
                    "relation": "far_if_possible",
                    "priority": _priority_text(row.get("priority")),
                    "reason": "Concept-specific topology keeps these macro fields distinct.",
                }
            )

    keep_open_regions = [
        {
            "type": str(row.get("type")),
            "near": str(row.get("near")),
            "priority": _priority_text(row.get("priority")),
            "reason": "Protected by deterministic macro concept policy.",
        }
        for row in _sequence_or_empty(constraints.get("keep_open_regions"))
        if isinstance(row, Mapping) and row.get("type") and row.get("near")
    ]
    macro_concept_payload = {
        key: value for key, value in concept.items() if key != "solver_relation_plan"
    }
    macro_concept_payload["cluster_zone_plan"] = cluster_zone_plan
    anchor_region_preferences = _ensure_placement_behavior_by_cluster(
        dict(_mapping_or_empty(concept.get("anchor_region_preferences_by_cluster"))),
        cluster_zone_plan=cluster_zone_plan,
    )
    return {
        "status": "OK",
        "room_id": _room_id(room_model),
        "room_type": room_type,
        "planner_kind": "seed_concept_generator",
        "macro_concept": macro_concept_payload,
        "cluster_affinities": _dedupe_by_cluster(affinities),
        "cluster_orientations": _dedupe_by_cluster(orientations),
        "object_orientations": _concept_object_orientations_for_solver(concept),
        "anchor_region_preferences_by_cluster": anchor_region_preferences,
        "anchor_layout_hints_by_cluster": _anchor_layout_hints_by_cluster(
            cluster_zone_plan=cluster_zone_plan
        ),
        "anchor_pair_contracts": list(
            _sequence_or_empty(concept.get("anchor_pair_contracts"))
        ),
        "object_solver_policy": dict(
            _mapping_or_empty(concept.get("object_solver_policy"))
        ),
        "macro_region_map": dict(
            _mapping_or_empty(concept.get("macro_region_map_snapshot"))
        ),
        "allowed_variant_families_by_cluster": dict(
            _mapping_or_empty(concept.get("allowed_variant_families_by_cluster"))
        ),
        "variant_bias_weights": dict(
            _mapping_or_empty(concept.get("variant_bias_weights"))
        ),
        "concept_readiness_requirements": dict(
            _mapping_or_empty(concept.get("concept_readiness_requirements"))
        ),
        "cluster_relations": _dedupe_relations(relations),
        "cluster_directional_relations": _dedupe_relations(directional_relations),
        "circulation_plan": {
            "main_paths": _main_paths(room_model, primary_cluster_id),
            "keep_open_regions": _dedupe_keep_open(keep_open_regions),
        },
        "layout_intent_profile": {
            "focus_mode": "viewing" if family == "focal_axis" else "mixed",
            "primary_cluster_id": primary_cluster_id,
            "secondary_cluster_id": secondary_cluster_id,
            "circulation_priority": "high"
            if policy.get("reserve_center_degree") in {"high", "very_high"}
            else "medium",
            "center_open_preference": "high"
            if policy.get("reserve_center_degree") in {"high", "very_high"}
            else "medium",
            "support_cluster_behavior": "recede"
            if family in {"open_center", "edge_weighted", "zoned"}
            else "balanced",
            "distribution_mode": "edge_weighted"
            if family == "edge_weighted"
            else "focal_grouped"
            if family == "focal_axis"
            else "zoned"
            if family == "zoned"
            else "balanced",
        },
        "placement_guidelines": [
            "Follow macro_concept.cluster_zone_plan before applying local repair.",
            "Preserve entry landing, primary corridor, and declared center openness before support clusters compete for space.",
        ],
        "notes": [
            (
                f"Generated by {'LLM-guided' if bool(concept.get('llm_guided')) else 'deterministic'} "
                f"SeedConceptGenerator family={family}."
            ),
            "Legacy relation fields are a solver projection of the macro concept, not the planning source of truth.",
            *_string_list(concept.get("guidance_notes")),
        ],
        "missing": [],
    }


def _solver_affinity_for_assignment(
    row: Mapping[str, object],
    policy: Mapping[str, object],
) -> dict[str, object]:
    cluster_id = str(row.get("cluster_id") or "")
    zone = str(row.get("zone_assignment") or "")
    wall_claim = str(row.get("wall_claim") or "")
    center_usage = str(row.get("center_usage") or "")
    prefer: list[str] = []
    avoid = ["entry_blocking", "main_path"]
    if wall_claim in {"strong", "medium"} or "wall" in zone or "edge" in zone:
        prefer.extend(["wall", "recess_or_edge"])
    if zone == "daylight_biased_zone":
        prefer.append("window_side")
    if (
        zone == "quiet_private_deep_zone"
        or row.get("entry_relation") == "avoid_direct_entry_conflict"
    ):
        prefer.append("far_from_entry")
    if zone == "entry_adjacent_active_zone":
        prefer.append("entry_side")
    reserve_center = policy.get("reserve_center_degree") in {"high", "very_high"}
    if center_usage in {"none", "open_reserved"} or reserve_center:
        avoid.append("center")
    elif center_usage in {"partial", "primary"}:
        prefer.append("center")
    if str(row.get("daylight_relation") or "") == "avoid_window_blocking":
        avoid.append("window_blocking")
    return {
        "cluster_id": cluster_id,
        "prefer": _uniq(prefer),
        "avoid": _uniq(avoid),
        "priority": "high" if wall_claim == "strong" else "medium",
        "reason": "Projection from concept zone assignment.",
    }


def _solver_orientation_for_assignment(
    row: Mapping[str, object],
    primary_cluster_id: str | None,
    secondary_cluster_id: str | None,
) -> dict[str, object] | None:
    cluster_id = _clean_str(row.get("cluster_id"))
    if cluster_id is None:
        return None
    intents: list[str] = []
    target_cluster_id: str | None = None
    if row.get("wall_claim") in {"strong", "medium"}:
        intents.extend(["back_to_wall", "axis_parallel_wall"])
    if cluster_id == primary_cluster_id and secondary_cluster_id:
        intents.extend(["face_cluster", "inward_to_room"])
        target_cluster_id = secondary_cluster_id
    elif cluster_id == secondary_cluster_id and primary_cluster_id:
        intents.extend(["face_cluster", "access_to_open_space"])
        target_cluster_id = primary_cluster_id
    elif row.get("placement_bias") != "wall_backed":
        intents.append("access_to_open_space")
    intents = _uniq(intents)
    if not intents:
        return None
    if "face_cluster" not in intents:
        target_cluster_id = None
    return {
        "cluster_id": cluster_id,
        "intents": intents,
        "target_cluster_id": target_cluster_id,
        "priority": "high"
        if cluster_id in {primary_cluster_id, secondary_cluster_id}
        else "medium",
        "reason": "Projection from concept wall/center/focal policy.",
    }


def _solver_relations_from_pair_contracts(
    pair_contracts: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    relations: list[dict[str, object]] = []
    for contract in pair_contracts:
        cluster_a = _clean_str(contract.get("cluster_a"))
        cluster_b = _clean_str(contract.get("cluster_b"))
        if cluster_a is None or cluster_b is None or cluster_a == cluster_b:
            continue
        pair_type = str(contract.get("pair_type") or "")
        relation = "near"
        if pair_type in {"buffered_support", "separate_but_legible"}:
            relation = "far_if_possible"
        relations.append(
            {
                "a": cluster_a,
                "b": cluster_b,
                "relation": relation,
                "priority": _contract_priority(contract),
                "reason": f"Projection from primary_pair_contract={pair_type}.",
            }
        )
    return relations


def _solver_directional_relations_from_pair_contracts(
    pair_contracts: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    relations: list[dict[str, object]] = []
    for contract in pair_contracts:
        cluster_a = _clean_str(contract.get("cluster_a"))
        cluster_b = _clean_str(contract.get("cluster_b"))
        if cluster_a is None or cluster_b is None or cluster_a == cluster_b:
            continue
        pair_type = str(contract.get("pair_type") or "")
        relation = _directional_relation_for_pair_type(pair_type)
        if relation is None:
            continue
        relations.append(
            {
                "a": cluster_a,
                "b": cluster_b,
                "relation": relation,
                "priority": _contract_priority(contract),
                "reason": f"Projection from primary_pair_contract={pair_type}.",
            }
        )
    return relations


def _directional_relation_for_pair_type(pair_type: str) -> str | None:
    if pair_type in {"face_each_other", "ring_around_anchor"}:
        return "face_each_other"
    if pair_type in {"supports_use_axis", "workflow_chain"}:
        return "access_faces_other"
    return None


def _contract_priority(contract: Mapping[str, object]) -> str:
    if bool(contract.get("required")):
        return "high"
    strength = str(contract.get("strength") or "medium").strip().lower()
    return strength if strength in {"high", "medium", "low"} else "medium"


def _concept_object_orientations_for_solver(
    concept: Mapping[str, object],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for item in _sequence_or_empty(concept.get("critical_object_orientations")):
        if not isinstance(item, Mapping):
            continue
        object_id = _clean_str(item.get("object_id"))
        cluster_id = _clean_str(item.get("cluster_id"))
        if object_id is None or cluster_id is None:
            continue
        out.append(dict(item))
    return out


def _empty_solver_plan(room_model: Mapping[str, object]) -> dict[str, object]:
    return {
        "status": "UNSAT",
        "room_id": _room_id(room_model),
        "cluster_affinities": [],
        "cluster_orientations": [],
        "object_orientations": [],
        "cluster_relations": [],
        "cluster_directional_relations": [],
        "circulation_plan": {"main_paths": [], "keep_open_regions": []},
        "layout_intent_profile": None,
        "placement_guidelines": [],
        "notes": ["No macro concept could be generated."],
        "missing": [],
    }


def _macro_region_to_dict(region: MacroRegion) -> dict[str, object]:
    return {
        "region_id": region.region_id,
        "region_type": region.region_type,
        "label": region.label,
        "source_ids": list(region.source_ids),
        "tags": list(region.tags),
    }


def _region_candidate_to_dict(candidate: RegionCandidate) -> dict[str, object]:
    return {
        "cluster_id": candidate.cluster_id,
        "region_id": candidate.region_id,
        "region_type": candidate.region_type,
        "score": candidate.score,
        "reasons": list(candidate.reasons),
    }


def _clean_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _mapping_or_empty(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _sequence_or_empty(value: object) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    return ()


def _string_list(value: object) -> list[str]:
    out: list[str] = []
    for item in _sequence_or_empty(value):
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
        elif isinstance(item, Mapping):
            ref = _clean_str(
                item.get("id")
                or item.get("region")
                or item.get("region_id")
                or item.get("node_id")
                or item.get("label")
            )
            if ref is not None:
                out.append(ref)
    return _uniq(out)


def _region_refs(value: object) -> list[str]:
    return _string_list(value)


def _opening_ids(value: object, prefix: str) -> list[str]:
    out: list[str] = []
    for index, item in enumerate(_sequence_or_empty(value), start=1):
        if not isinstance(item, Mapping):
            continue
        out.append(_clean_str(item.get("id")) or f"{prefix}_{index}")
    return out


def _wall_region_ids(
    room: Mapping[str, object],
    affordance: Mapping[str, object],
) -> tuple[str, ...]:
    explicit = _region_refs(affordance.get("usable_wall_segments"))
    if explicit:
        return tuple(explicit)
    polygon = _sequence_or_empty(room.get("polygon_ccw"))
    count = max(1, min(6, len(polygon)))
    return tuple(f"wall_segment_{index}" for index in range(1, count + 1))


def _cluster_rules(cluster: Mapping[str, object]) -> Mapping[str, object]:
    return _mapping_or_empty(cluster.get("cluster_rules"))


def _cluster_object_ids(cluster: Mapping[str, object]) -> list[str]:
    out: list[str] = []
    for key in ("members", "local_placements"):
        for item in _sequence_or_empty(cluster.get(key)):
            value = item.get("id") if isinstance(item, Mapping) else item
            text = _clean_str(value)
            if text is not None and text not in out:
                out.append(text)
    footprint = _mapping_or_empty(cluster.get("cluster_footprint"))
    for rect in _sequence_or_empty(footprint.get("rects")):
        if not isinstance(rect, Mapping):
            continue
        text = _clean_str(rect.get("id"))
        if text is not None and text not in out:
            out.append(text)
    return out


def _semantic_role_from_cluster(cluster_id: str, object_ids: Sequence[str]) -> str:
    role = _role_kind(cluster_id, object_ids, "")
    mapping = {
        "social_anchor": "social_anchor",
        "lounge": "social_anchor",
        "media": "focal_anchor",
        "focal": "focal_anchor",
        "storage": "secondary_storage",
        "kitchen": "kitchen_workflow",
        "work": "work_support",
        "sleep": "rest_anchor",
        "service": "service_support",
    }
    return mapping.get(role, "secondary_zone")


def _role_kind(cluster_id: str, object_ids: Sequence[str], semantic_role: str) -> str:
    tokens = " ".join([cluster_id, semantic_role, *object_ids]).lower()
    if "kitchen" in tokens or profile_room_type_for_objects(object_ids) == "kitchen":
        return "kitchen"
    if any(token in tokens for token in ("sofa", "sectional", "lounge", "seating")):
        return "social_anchor"
    if any(token in tokens for token in ("tv", "media", "screen", "fireplace")):
        return "media"
    if any(
        token in tokens
        for token in ("storage", "cabinet", "console", "shelf", "wardrobe")
    ):
        return "storage"
    if any(token in tokens for token in ("desk", "work", "office", "study")):
        return "work"
    if any(token in tokens for token in ("bed", "sleep")):
        return "sleep"
    return "support"


def _priority(value: object, role_kind: str) -> Priority:
    text = str(value or "").strip().lower()
    if text in {"core", "support", "optional"}:
        return text  # type: ignore[return-value]
    if role_kind in {"social_anchor", "media", "sleep", "kitchen"}:
        return "core"
    if role_kind in {"storage", "work", "support"}:
        return "support"
    return "optional"


def _layout_role(semantic: Mapping[str, object]) -> str:
    text = str(semantic.get("layout_role") or "").strip().lower()
    if text in {"primary", "secondary", "support", "optional"}:
        return text
    return ""


def _affinity_level(value: object) -> str:
    text = str(value or "none").strip().lower()
    if text in {"none", "low", "medium", "high"}:
        return text
    return "none"


def _cluster_avoids_entry(cluster: ClusterProgram) -> bool:
    claims = cluster.zone_claims
    avoid_regions = " ".join(_string_list(claims.get("avoid_regions"))).lower()
    relation_types = {
        str(intent.get("type") or "").strip().lower()
        for intent in cluster.relation_intents
    }
    return (
        "entry" in avoid_regions
        or "avoid_entry" in relation_types
        or cluster.role_kind in {"sleep", "media"}
    )


def _seed_region_tags(cluster: Mapping[str, object]) -> list[str]:
    seed_state = _mapping_or_empty(cluster.get("seed_state"))
    return _string_list(seed_state.get("region_tags"))


def _primary_cluster_id(clusters: Sequence[ClusterProgram]) -> str | None:
    explicit = sorted(
        (cluster for cluster in clusters if cluster.layout_role == "primary"),
        key=lambda cluster: cluster.cluster_id,
    )
    if explicit:
        return explicit[0].cluster_id
    if any(cluster.layout_role for cluster in clusters):
        return None
    ranked = sorted(
        clusters,
        key=lambda cluster: (
            0
            if cluster.role_kind == "social_anchor"
            else 1
            if cluster.priority == "core"
            else 2,
            cluster.cluster_id,
        ),
    )
    return ranked[0].cluster_id if ranked else None


def _secondary_cluster_id(
    clusters: Sequence[ClusterProgram],
    primary_cluster_id: str | None,
) -> str | None:
    explicit = sorted(
        (
            cluster
            for cluster in clusters
            if cluster.cluster_id != primary_cluster_id
            and cluster.layout_role == "secondary"
        ),
        key=lambda cluster: cluster.cluster_id,
    )
    if explicit:
        return explicit[0].cluster_id
    if any(cluster.layout_role for cluster in clusters):
        return None
    ranked = sorted(
        (cluster for cluster in clusters if cluster.cluster_id != primary_cluster_id),
        key=lambda cluster: (
            0
            if cluster.role_kind == "media"
            else 1
            if cluster.priority == "core"
            else 2,
            cluster.cluster_id,
        ),
    )
    return ranked[0].cluster_id if ranked else None


def _wall_claim_for(
    cluster: ClusterProgram,
    zone_assignment: str,
    family: ConceptFamily,
) -> str:
    if family in {"focal_axis", "edge_weighted"} and (
        cluster.role_kind == "media" or "wall" in zone_assignment
    ):
        return "strong"
    if "wall" in zone_assignment or "edge" in zone_assignment:
        return "medium"
    if _affinity_level(cluster.zone_claims.get("wall_affinity")) == "high":
        return "strong"
    return "none"


def _center_usage_for(
    *,
    cluster: ClusterProgram,
    family: ConceptFamily,
    primary_cluster_id: str | None,
    scenario: MacroScenario,
    scenario_role: str,
) -> str:
    if scenario.center_policy == "floating_primary" and scenario_role == "primary":
        return "primary"
    if family in {"open_center", "edge_weighted"}:
        return "open_reserved"
    if family == "zoned" and cluster.cluster_id == primary_cluster_id:
        return "primary"
    if (
        family in {"focal_axis", "daylight_oriented"}
        and cluster.cluster_id == primary_cluster_id
    ):
        return "partial"
    return "none"


def _entry_relation_for(
    cluster: ClusterProgram,
    topology_policy: Mapping[str, object],
) -> str:
    if _cluster_avoids_entry(cluster):
        return "avoid_direct_entry_conflict"
    if topology_policy.get("entry_avoidance_strength") == "medium":
        return "buffer_entry_if_possible"
    return "neutral"


def _daylight_relation_for(
    cluster: ClusterProgram,
    zone_assignment: str,
    family: ConceptFamily,
) -> str:
    if zone_assignment == "daylight_biased_zone":
        return "claim_daylight"
    if family == "daylight_oriented" and cluster.role_kind in {"work", "social_anchor"}:
        return "daylight_preferred"
    avoid_regions = " ".join(_string_list(cluster.zone_claims.get("avoid_regions")))
    if cluster.role_kind == "media" or any(
        token in avoid_regions.lower() for token in ("window", "daylight")
    ):
        return "avoid_window_blocking"
    return "neutral"


def _privacy_relation_for(cluster: ClusterProgram, zone_assignment: str) -> str:
    if zone_assignment == "quiet_private_deep_zone":
        return "claim_privacy"
    if _affinity_level(cluster.zone_claims.get("privacy_affinity")) in {
        "medium",
        "high",
    }:
        return "privacy_preferred"
    return "neutral"


def _apply_placement_behavior_contracts(
    cluster_zone_plan: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in cluster_zone_plan:
        out = dict(row)
        out["placement_behavior"] = _placement_behavior_contract(out)
        rows.append(out)
    return rows


def _placement_behavior_contract(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "wall_backing": _wall_backing_contract(row),
        "front_space": _front_space_contract(row),
        "daylight_blocking": _daylight_blocking_contract(row),
        "pose_flexibility": _pose_flexibility_contract(row),
    }


def _wall_backing_contract(row: Mapping[str, object]) -> str:
    wall_claim = str(row.get("wall_claim") or "none").strip().lower()
    placement_bias = str(row.get("placement_bias") or "balanced").strip().lower()
    if wall_claim == "strong" or placement_bias == "wall_backed":
        return "required"
    if wall_claim == "medium" or placement_bias in {"daylight_edge", "edge_loaded"}:
        return "preferred"
    return "optional"


def _front_space_contract(row: Mapping[str, object]) -> str:
    scenario_role = str(row.get("scenario_role") or "support").strip().lower()
    layout_role = str(row.get("layout_role") or "").strip().lower()
    anchor_strength = str(row.get("anchor_strength") or "medium").strip().lower()
    placement_bias = str(row.get("placement_bias") or "balanced").strip().lower()
    if layout_role == "optional" or scenario_role == "optional":
        return "optional"
    if scenario_role in {"primary", "secondary"} and anchor_strength in {
        "hard",
        "strong",
    }:
        return "required"
    if placement_bias in {"daylight_edge", "edge_loaded", "wall_backed"}:
        return "preferred"
    return "optional"


def _daylight_blocking_contract(row: Mapping[str, object]) -> str:
    daylight_relation = str(row.get("daylight_relation") or "neutral").strip().lower()
    if daylight_relation == "avoid_window_blocking":
        return "avoid"
    if daylight_relation in {"claim_daylight", "daylight_preferred"}:
        return "prefer"
    return "neutral"


def _pose_flexibility_contract(row: Mapping[str, object]) -> str:
    scenario_role = str(row.get("scenario_role") or "support").strip().lower()
    layout_role = str(row.get("layout_role") or "").strip().lower()
    anchor_strength = str(row.get("anchor_strength") or "medium").strip().lower()
    if scenario_role in {"primary", "secondary"} or anchor_strength in {
        "hard",
        "strong",
    }:
        return "low"
    if layout_role == "optional" or scenario_role == "optional":
        return "high"
    return "medium"


def _placement_bias(zone_assignment: str, wall_claim: str, center_usage: str) -> str:
    if wall_claim == "strong":
        return "wall_backed"
    if zone_assignment == "daylight_biased_zone":
        return "daylight_edge"
    if center_usage in {"partial", "primary"}:
        return "float_allowed"
    if "edge" in zone_assignment or "storage" in zone_assignment:
        return "edge_loaded"
    return "balanced"


def _spatial_character(family: ConceptFamily, room_type: str) -> str:
    by_family = {
        "focal_axis": f"{room_type}_focal_axis_balanced",
        "open_center": f"{room_type}_open_center_clear_circulation",
        "edge_weighted": f"{room_type}_perimeter_loaded_open_field",
        "zoned": f"{room_type}_primary_secondary_zoned",
        "daylight_oriented": f"{room_type}_daylight_oriented_support",
    }
    return by_family[family]


def _concept_score_prior(
    family: ConceptFamily,
    *,
    style_policy: Mapping[str, object],
) -> dict[str, float]:
    priors = {
        "focal_axis": (0.95, 0.90, 0.96, 0.89),
        "open_center": (0.92, 0.93, 0.90, 0.94),
        "edge_weighted": (0.90, 0.91, 0.88, 0.93),
        "zoned": (0.93, 0.89, 0.94, 0.88),
        "daylight_oriented": (0.91, 0.90, 0.92, 0.91),
    }
    functionality, naturalness, semantic, spatial = priors[family]
    layout_policy = _style_layout_policy(style_policy)
    if (
        family == "open_center"
        and _bias_level(layout_policy.get("center_openness_bias")) >= 3
    ):
        naturalness += 0.04
        spatial += 0.04
    if (
        family == "daylight_oriented"
        and _bias_level(layout_policy.get("daylight_bias")) >= 3
    ):
        naturalness += 0.04
        semantic += 0.03
    if family == "edge_weighted" and str(layout_policy.get("wall_loading_bias")) in {
        "medium_high",
        "perimeter_heavy",
    }:
        spatial += 0.04
    if family == "focal_axis" and _bias_level(layout_policy.get("symmetry_bias")) >= 3:
        semantic += 0.04
        spatial += 0.03
    return {
        "functionality": min(functionality, 1.0),
        "naturalness": min(naturalness, 1.0),
        "semantic_coherence": min(semantic, 1.0),
        "spatial_quality": min(spatial, 1.0),
    }


def _diversity_signature(
    *,
    family: ConceptFamily,
    cluster_zone_plan: Sequence[Mapping[str, object]],
    topology_policy: Mapping[str, object],
) -> str:
    scenario_id = next(
        (
            str(row.get("macro_scenario_id") or "")
            for row in cluster_zone_plan
            if str(row.get("macro_scenario_id") or "")
        ),
        "",
    )
    assignments = ",".join(
        f"{row.get('cluster_id')}:{row.get('zone_assignment')}:"
        f"{row.get('preferred_wall_side')}:{row.get('center_usage')}:"
        f"{row.get('wall_claim')}"
        for row in cluster_zone_plan
    )
    return (
        f"{family}|scenario={scenario_id}|"
        f"center={topology_policy.get('reserve_center_degree')}|"
        f"wall={topology_policy.get('wall_loading_bias')}|{assignments}"
    )


def _guided_seed_concept_llm_enabled() -> bool:
    raw = os.getenv(_SEED_CONCEPT_GUIDED_LLM_ENV, "0")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _seed_concept_guidance_model_chain() -> tuple[str, ...]:
    return tuple(
        TextLLMConfig.agent_model_chain(
            _MACRO_CONCEPT_GUIDANCE_SPEC.model_config_keys,
            _MACRO_CONCEPT_GUIDANCE_SPEC.default_model_chain,
        )
    )


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


def _available_macro_region_ids(macro_region_map: Mapping[str, object]) -> set[str]:
    region_ids: set[str] = set()
    for item in _sequence_or_empty(macro_region_map.get("regions")):
        if not isinstance(item, Mapping):
            continue
        region_id = _clean_str(item.get("region_id"))
        if region_id is not None:
            region_ids.add(region_id)
    return region_ids


def _top_region_candidates_by_cluster(
    region_candidates: Sequence[RegionCandidate],
) -> dict[str, list[dict[str, object]]]:
    out: dict[str, list[dict[str, object]]] = {}
    for candidate in region_candidates:
        rows = out.setdefault(candidate.cluster_id, [])
        if len(rows) >= 3:
            continue
        rows.append(_region_candidate_to_dict(candidate))
    return out


def _guided_clusters_summary(clusters_json: Mapping[str, object]) -> dict[str, object]:
    clusters = _extract_clusters_map(clusters_json)
    return {
        "cluster_ids": sorted(clusters),
        "semantic_cluster_ids": sorted(_semantic_clusters_by_id(clusters_json)),
        "style_name": _clean_str(_extract_style_policy(clusters_json).get("style_name"))
        or "",
    }


def _guided_room_summary(
    room_model: Mapping[str, object],
    macro_region_map: Mapping[str, object],
) -> dict[str, object]:
    room = _mapping_or_empty(room_model.get("room"))
    protected = _mapping_or_empty(macro_region_map.get("protected_topology"))
    return {
        "room_id": _room_id(room_model),
        "polygon_vertex_count": len(_sequence_or_empty(room.get("polygon_ccw"))),
        "protected_regions": {
            "entry_landing_zones": _string_list(protected.get("entry_landing_zones")),
            "primary_circulation_corridors": _string_list(
                protected.get("primary_circulation_corridors")
            ),
            "center_openness_regions": _string_list(
                protected.get("center_openness_regions")
            ),
        },
    }


def _extract_llm_content(response: object) -> str:
    choices = getattr(response, "choices", None)
    if isinstance(choices, Sequence) and choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
    raise ValueError("SeedConceptGenerator guidance response missing message content")


def _parse_guidance_json(raw: str) -> dict[str, object]:
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        for idx in range(1, len(parts), 2):
            candidate = parts[idx].strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].lstrip()
            if candidate.startswith("{") and candidate.endswith("}"):
                text = candidate
                break
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("SeedConceptGenerator guidance returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("SeedConceptGenerator guidance JSON must be an object")
    return payload


def _sanitize_macro_concept_guidance(
    payload: Mapping[str, object],
    *,
    target_count: int,
    available_families: Sequence[ConceptFamily],
    cluster_ids: set[str],
    region_ids: set[str],
) -> dict[str, object]:
    family_order = [
        family
        for family in _string_list(payload.get("family_order"))
        if family in available_families
    ]
    for family in available_families:
        if family not in family_order:
            family_order.append(family)
    family_order = family_order[: len(available_families)]

    concept_blueprints: list[dict[str, object]] = []
    seen_families: set[str] = set()
    for item in _sequence_or_empty(payload.get("concept_blueprints")):
        if not isinstance(item, Mapping):
            continue
        family = _clean_str(item.get("concept_family"))
        if family not in available_families or family in seen_families:
            continue
        concept_blueprints.append(
            {
                "concept_family": family,
                "topology_policy": _sanitize_guided_topology_policy(
                    item.get("topology_policy")
                ),
                "cluster_zone_overrides": _sanitize_cluster_zone_overrides(
                    item.get("cluster_zone_overrides"),
                    cluster_ids=cluster_ids,
                    region_ids=region_ids,
                ),
                "notes": _string_list(item.get("notes")),
            }
        )
        seen_families.add(family)
        if len(concept_blueprints) >= max(1, target_count):
            break

    return {
        "family_order": family_order,
        "concept_blueprints": concept_blueprints,
        "notes": _string_list(payload.get("notes")),
    }


def _sanitize_guided_topology_policy(value: object) -> dict[str, object]:
    policy = _mapping_or_empty(value)
    out: dict[str, object] = {}
    for key in _GUIDED_TOPOLOGY_POLICY_KEYS:
        text = _clean_str(policy.get(key))
        if text is not None:
            out[key] = text
    return out


def _sanitize_cluster_zone_overrides(
    value: object,
    *,
    cluster_ids: set[str],
    region_ids: set[str],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    seen_clusters: set[str] = set()
    for item in _sequence_or_empty(value):
        if not isinstance(item, Mapping):
            continue
        cluster_id = _clean_str(item.get("cluster_id"))
        zone_assignment = _clean_str(item.get("zone_assignment"))
        if (
            cluster_id is None
            or cluster_id not in cluster_ids
            or cluster_id in seen_clusters
            or zone_assignment is None
            or zone_assignment not in region_ids
            or zone_assignment == "keep_open_center"
        ):
            continue
        center_usage = str(item.get("center_usage") or "").strip()
        if center_usage not in _CENTER_USAGE_VALUES:
            center_usage = "none"
        preferred_wall_side = str(item.get("preferred_wall_side") or "").strip()
        if preferred_wall_side not in _GUIDED_WALL_SIDES:
            preferred_wall_side = ""
        out.append(
            {
                "cluster_id": cluster_id,
                "zone_assignment": zone_assignment,
                "preferred_wall_side": preferred_wall_side,
                "center_usage": center_usage,
            }
        )
        seen_clusters.add(cluster_id)
    return out


def _primary_from_concept(
    cluster_zone_plan: Sequence[Mapping[str, object]],
) -> str | None:
    for row in cluster_zone_plan:
        if row.get("priority") == "core" and row.get("center_usage") in {
            "partial",
            "primary",
        }:
            return _clean_str(row.get("cluster_id"))
    for row in cluster_zone_plan:
        if row.get("priority") == "core":
            return _clean_str(row.get("cluster_id"))
    return (
        _clean_str(cluster_zone_plan[0].get("cluster_id"))
        if cluster_zone_plan
        else None
    )


def _secondary_from_concept(
    cluster_zone_plan: Sequence[Mapping[str, object]],
    primary_cluster_id: str | None,
) -> str | None:
    for row in cluster_zone_plan:
        cluster_id = _clean_str(row.get("cluster_id"))
        if (
            cluster_id
            and cluster_id != primary_cluster_id
            and row.get("wall_claim") == "strong"
        ):
            return cluster_id
    for row in cluster_zone_plan:
        cluster_id = _clean_str(row.get("cluster_id"))
        if cluster_id and cluster_id != primary_cluster_id:
            return cluster_id
    return None


def _main_paths(
    room_model: Mapping[str, object],
    primary_cluster_id: str | None,
) -> list[dict[str, object]]:
    if primary_cluster_id is None:
        return []
    openings = _mapping_or_empty(room_model.get("openings"))
    return [
        {
            "from": door_id,
            "to_cluster": primary_cluster_id,
            "priority": "high",
            "reason": "Primary circulation path from entry to core cluster.",
        }
        for door_id in _opening_ids(openings.get("doors"), "door")[:2]
    ]


def _dedupe_by_cluster(
    rows: Sequence[Mapping[str, object] | None],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        cluster_id = _clean_str(row.get("cluster_id"))
        if cluster_id is None or cluster_id in seen:
            continue
        out.append(dict(row))
        seen.add(cluster_id)
    return out


def _dedupe_relations(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        a = _clean_str(row.get("a"))
        b = _clean_str(row.get("b"))
        if a is None or b is None or a == b:
            continue
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        out.append(dict(row))
        seen.add(key)
    return out


def _dedupe_keep_open(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        region_type = _clean_str(row.get("type"))
        near = _clean_str(row.get("near"))
        if region_type is None or near is None:
            continue
        key = (region_type, near)
        if key in seen:
            continue
        out.append(dict(row))
        seen.add(key)
    return out


def _priority_text(value: object) -> str:
    text = str(value or "medium").strip().lower()
    return text if text in {"high", "medium", "low"} else "medium"


def _family_from_text(text: str | None) -> ConceptFamily:
    lowered = str(text or "").lower()
    for family in CONCEPT_FAMILIES:
        if family in lowered or family.replace("_", "-") in lowered:
            return family
    if "open center" in lowered:
        return "open_center"
    if "edge" in lowered or "perimeter" in lowered:
        return "edge_weighted"
    if "daylight" in lowered or "window" in lowered:
        return "daylight_oriented"
    if "zoned" in lowered or "separate support" in lowered:
        return "zoned"
    return "focal_axis"


def _uniq(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = value.strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out
