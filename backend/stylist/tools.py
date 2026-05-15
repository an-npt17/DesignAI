from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from config.demo_inventory import (
    is_demo_inventory_tenant,
    is_enabled_demo_inventory_tenant,
)
from config.semantic_search_config import SemanticSearchConfig
from db.models import AssetFilter, DesignKnowledgeFilter, TenantId
from db.pg_repository import PostgresAssetRepository, PostgresDesignKnowledgeRepository
from stylist.room_essentials_seed import ROOM_SURFACE_GROUPS

logger = logging.getLogger(__name__)


def _stable_unique_strs(values: List[str] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        if not isinstance(value, str):
            continue
        v = value.strip()
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _normalize_lookup_set(values: List[str] | None) -> set[str]:
    return {
        v.strip().lower() for v in (values or []) if isinstance(v, str) and v.strip()
    }


def _serialize_dimensions(asset: Any) -> dict[str, Any] | None:
    dims = getattr(asset, "dimensions", None)
    if dims is None:
        return None

    length_mm = getattr(dims, "length_mm", None)
    width_mm = getattr(dims, "width_mm", None)
    height_mm = getattr(dims, "height_mm", None)

    return {
        "length_mm": int(length_mm) if isinstance(length_mm, (int, float)) else None,
        "width_mm": int(width_mm) if isinstance(width_mm, (int, float)) else None,
        "height_mm": int(height_mm) if isinstance(height_mm, (int, float)) else None,
    }


def GetRoomSurfaceGroups(*, room_type: str) -> Dict[str, Any]:
    groups = ROOM_SURFACE_GROUPS.get(room_type)
    if groups is None:
        return {
            "error": "room_type_not_found",
            "room_type": room_type,
            "groups": {},
        }
    return {
        "room_type": room_type,
        "groups": groups,
    }


def ListInventoryByTypes(
    *,
    tenant_id: str | None,
    types: List[str],
    limit: int | None = 200,
) -> Dict[str, Any]:
    resolved_tenant = tenant_id or "demo_tenant"
    if is_demo_inventory_tenant(
        resolved_tenant
    ) and not is_enabled_demo_inventory_tenant(resolved_tenant):
        return {
            "tenant_id": resolved_tenant,
            "requested_types": _stable_unique_strs(types),
            "count": 0,
            "items": [],
            "disabled": True,
        }
    repo = PostgresAssetRepository()
    assets = list(repo.list_assets(AssetFilter(tenant_id=TenantId(resolved_tenant))))

    normalized = _normalize_lookup_set(types)
    results: list[dict[str, Any]] = []

    for asset in assets:
        asset_type = str(getattr(asset, "type", "") or "").strip()
        asset_name = str(getattr(asset, "name", "") or "").strip()
        attrs = dict(getattr(asset, "attributes", {}) or {})
        category = str(attrs.get("category") or asset_type).strip()

        if normalized:
            candidates = {
                asset_type.lower(),
                category.lower(),
                asset_name.lower(),
            }
            if candidates.isdisjoint(normalized):
                continue

        dims = _serialize_dimensions(asset)

        results.append(
            {
                "id": str(asset.id),
                "inventory_id": str(asset.id),
                "name": asset_name,
                "type": category,
                "asset_type": asset_type,
                "style_tags": list(getattr(asset, "style_tags", []) or []),
                "material": getattr(asset, "material", None),
                "brand": getattr(asset, "brand", None),
                "length_mm": dims["length_mm"] if isinstance(dims, dict) else None,
                "width_mm": dims["width_mm"] if isinstance(dims, dict) else None,
                "height_mm": dims["height_mm"] if isinstance(dims, dict) else None,
                "dimensions": dims,
                "attributes": attrs,
            }
        )

    results.sort(
        key=lambda x: (
            str(x.get("type") or ""),
            str(x.get("name") or ""),
            str(x.get("inventory_id") or ""),
        )
    )

    if isinstance(limit, int) and limit > 0:
        results = results[:limit]

    return {
        "tenant_id": resolved_tenant,
        "requested_types": _stable_unique_strs(types),
        "count": len(results),
        "items": results,
    }


def ListDesignKnowledge(
    *,
    tenant_id: str | None,
    tags: List[str] | None = None,
    category: str | None = None,
    limit: int | None = 20,
) -> Dict[str, Any]:
    repo = PostgresDesignKnowledgeRepository()
    normalized_tags = _stable_unique_strs(tags)

    if not SemanticSearchConfig.ENABLED:
        return _list_design_knowledge_lexically(
            repo=repo,
            tenant_id=tenant_id,
            tags=normalized_tags,
            category=category,
            limit=limit,
        )

    # Ưu tiên tenant-specific trước, sau đó fallback global.
    filters: list[DesignKnowledgeFilter] = []
    if tenant_id:
        filters.append(
            DesignKnowledgeFilter(
                tenant_id=TenantId(tenant_id),
                tags=normalized_tags,
                category=category,
            )
        )

    filters.append(
        DesignKnowledgeFilter(
            tenant_id=None,
            tags=normalized_tags,
            category=category,
        )
    )

    results: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for filter_obj in filters:
        items = list(repo.list_knowledge(filter_obj))
        items.sort(
            key=lambda item: (
                str(getattr(item, "category", "") or ""),
                str(getattr(item, "title", "") or ""),
                str(getattr(item, "id", "") or ""),
            )
        )

        for item in items:
            item_id = str(item.id)
            if item_id in seen_ids:
                continue
            seen_ids.add(item_id)

            results.append(
                {
                    "id": item_id,
                    "title": item.title,
                    "content": item.content,
                    "category": item.category,
                    "tags": list(item.tags or []),
                    "source": item.source,
                    "meta": dict(item.meta or {}),
                }
            )

            if isinstance(limit, int) and limit > 0 and len(results) >= limit:
                return {
                    "tenant_id": tenant_id,
                    "tags": normalized_tags,
                    "category": category,
                    "search_mode": "metadata_filter",
                    "count": len(results),
                    "items": results,
                }

    return {
        "tenant_id": tenant_id,
        "tags": normalized_tags,
        "category": category,
        "search_mode": "metadata_filter",
        "count": len(results),
        "items": results,
    }


def _list_design_knowledge_lexically(
    *,
    repo: PostgresDesignKnowledgeRepository,
    tenant_id: str | None,
    tags: list[str],
    category: str | None,
    limit: int | None,
) -> Dict[str, Any]:
    desired_limit = limit if isinstance(limit, int) and limit > 0 else 20
    scopes: list[TenantId | None] = []
    if tenant_id:
        scopes.append(TenantId(tenant_id))
    scopes.append(None)

    results: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    query_terms = _build_knowledge_query_terms(tags=tags, category=category)

    if not query_terms:
        for scope in scopes:
            items = list(repo.list_knowledge(DesignKnowledgeFilter(tenant_id=scope)))
            items.sort(
                key=lambda item: (
                    str(getattr(item, "category", "") or ""),
                    str(getattr(item, "title", "") or ""),
                    str(getattr(item, "id", "") or ""),
                )
            )
            for item in items:
                item_id = str(item.id)
                if item_id in seen_ids:
                    continue
                seen_ids.add(item_id)
                results.append(_serialize_design_knowledge_item(item))
                if len(results) >= desired_limit:
                    return {
                        "tenant_id": tenant_id,
                        "tags": tags,
                        "category": category,
                        "search_mode": "lexical_fallback",
                        "count": len(results),
                        "items": results,
                    }
        return {
            "tenant_id": tenant_id,
            "tags": tags,
            "category": category,
            "search_mode": "lexical_fallback",
            "count": len(results),
            "items": results,
        }

    for scope_index, scope in enumerate(scopes):
        items = list(repo.list_knowledge(DesignKnowledgeFilter(tenant_id=scope)))
        ranked: list[tuple[float, str, Any]] = []

        for item in items:
            score = _score_design_knowledge_item(
                item=item,
                tags=tags,
                category=category,
                query_terms=query_terms,
                tenant_scope_index=scope_index,
            )
            if score <= 0:
                continue
            ranked.append((score, str(getattr(item, "id", "") or ""), item))

        ranked.sort(key=lambda row: (-row[0], row[1]))

        for _, _, item in ranked:
            item_id = str(item.id)
            if item_id in seen_ids:
                continue
            seen_ids.add(item_id)
            results.append(_serialize_design_knowledge_item(item))
            if len(results) >= desired_limit:
                logger.info(
                    "ListDesignKnowledge lexical fallback returned %s items for category=%s tags=%s",
                    len(results),
                    category,
                    tags,
                )
                return {
                    "tenant_id": tenant_id,
                    "tags": tags,
                    "category": category,
                    "search_mode": "lexical_fallback",
                    "count": len(results),
                    "items": results,
                }

    logger.info(
        "ListDesignKnowledge lexical fallback returned %s items for category=%s tags=%s",
        len(results),
        category,
        tags,
    )
    return {
        "tenant_id": tenant_id,
        "tags": tags,
        "category": category,
        "search_mode": "lexical_fallback",
        "count": len(results),
        "items": results,
    }


def _serialize_design_knowledge_item(item: Any) -> dict[str, Any]:
    return {
        "id": str(item.id),
        "title": item.title,
        "content": item.content,
        "category": item.category,
        "tags": list(item.tags or []),
        "source": item.source,
        "meta": dict(item.meta or {}),
    }


def _build_knowledge_query_terms(*, tags: list[str], category: str | None) -> set[str]:
    terms: set[str] = set()
    for tag in tags:
        terms.update(_tokenize_text(tag))
        terms.add(tag.strip().lower())
    if category:
        terms.update(_tokenize_text(category))
        terms.add(category.strip().lower())
    return {term for term in terms if term}


def _score_design_knowledge_item(
    *,
    item: Any,
    tags: list[str],
    category: str | None,
    query_terms: set[str],
    tenant_scope_index: int,
) -> float:
    haystack = _knowledge_haystack(item)
    haystack_tokens = _tokenize_text(haystack)
    item_tags = _normalize_lookup_set(list(getattr(item, "tags", []) or []))
    item_category = str(getattr(item, "category", "") or "").strip().lower()

    score = 0.0

    if tenant_scope_index == 0 and getattr(item, "tenant_id", None) is not None:
        score += 2.0

    if category:
        normalized_category = category.strip().lower()
        if normalized_category == item_category:
            score += 14.0
        elif normalized_category and normalized_category in haystack:
            score += 5.0

    for tag in tags:
        normalized_tag = tag.strip().lower()
        if not normalized_tag:
            continue
        if normalized_tag in item_tags:
            score += 10.0
        elif normalized_tag in haystack:
            score += 4.0

    overlap_count = len(query_terms & haystack_tokens)
    score += float(overlap_count)

    return score


def _knowledge_haystack(item: Any) -> str:
    text_parts = [
        str(getattr(item, "title", "") or ""),
        str(getattr(item, "content", "") or ""),
        str(getattr(item, "category", "") or ""),
        " ".join(str(tag) for tag in list(getattr(item, "tags", []) or [])),
        _flatten_meta_value(getattr(item, "meta", {}) or {}),
    ]
    return " ".join(part.strip().lower() for part in text_parts if part).strip()


def _flatten_meta_value(value: Any) -> str:
    if isinstance(value, dict):
        parts: list[str] = []
        for key, item in value.items():
            parts.append(str(key))
            parts.append(_flatten_meta_value(item))
        return " ".join(part for part in parts if part)
    if isinstance(value, list):
        return " ".join(_flatten_meta_value(item) for item in value)
    if value is None:
        return ""
    return str(value)


def _tokenize_text(value: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9_]+", value.lower()) if len(token) >= 2
    }


TOOL_REGISTRY: Dict[str, Any] = {
    "GetRoomSurfaceGroups": GetRoomSurfaceGroups,
    "ListInventoryByTypes": ListInventoryByTypes,
    "ListDesignKnowledge": ListDesignKnowledge,
}

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "GetRoomSurfaceGroups",
            "description": "Return surface grouping rules for a room type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "room_type": {"type": "string"},
                },
                "required": ["room_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ListInventoryByTypes",
            "description": (
                "List inventory items filtered by eligible object types or categories for a tenant. "
                "Returns flattened dimensions (length_mm, width_mm, height_mm) and inventory_id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tenant_id": {"type": ["string", "null"]},
                    "types": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": ["integer", "null"]},
                },
                "required": ["types"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ListDesignKnowledge",
            "description": (
                "List design knowledge entries for a tenant with tenant-specific priority and global fallback. "
                "Supports tags/category filtering and a result limit. "
                "When semantic search is disabled, this tool falls back to lexical ranking over title/content/tags/meta."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tenant_id": {"type": ["string", "null"]},
                    "tags": {"type": ["array", "null"], "items": {"type": "string"}},
                    "category": {"type": ["string", "null"]},
                    "limit": {"type": ["integer", "null"]},
                },
            },
        },
    },
]
