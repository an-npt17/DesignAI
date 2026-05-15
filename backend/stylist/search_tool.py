from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Any
from collections.abc import Sequence
from urllib.parse import parse_qs, urlparse

import httpx
from bs4 import BeautifulSoup

from clients.base_client import LLMClientProtocol
from clients.llm_client import get_llm_client
from config.semantic_search_config import SemanticSearchConfig
from db.models import (
    DesignKnowledge,
    DesignKnowledgeEmbedding,
    DesignKnowledgeFilter,
    TenantId,
)
from db.repositories import DesignKnowledgeRepository
from stylist.rules import get_soft_rule_templates

logger = logging.getLogger(__name__)
search_logger = logging.getLogger("search")

_USERAGENT_LIST = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36",
]

_CACHE_PATH = os.getenv("SEARCH_CACHE_PATH", os.path.join("logs", "search_cache.json"))
_CACHE_TTL_SEC = int(os.getenv("SEARCH_CACHE_TTL_SEC", "86400"))
_CACHE_MAX = int(os.getenv("SEARCH_CACHE_MAX", "200"))
_CONTENT_CACHE_PATH = os.getenv(
    "SEARCH_CONTENT_CACHE_PATH", os.path.join("logs", "search_content_cache.json")
)
_CONTENT_CACHE_MAX = int(os.getenv("SEARCH_CONTENT_CACHE_MAX", "400"))


@dataclass(frozen=True)
class SearchResult:
    title: str
    snippet: str
    url: str
    content: str = ""


class SearchProvider:
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        raise NotImplementedError


class GoogleSearchProvider(SearchProvider):
    def __init__(self, timeout: int = 6) -> None:
        self._timeout = timeout
        self._lang = os.getenv("GOOGLE_LANG", "en")
        self._region = os.getenv("GOOGLE_REGION", "")
        self._safe = os.getenv("GOOGLE_SAFE", "active")
        self._basic = os.getenv("GOOGLE_BASIC", "1").lower() != "0"
        self._retries = max(1, int(os.getenv("SEARCH_RETRIES", "3")))
        self._retry_base = float(os.getenv("SEARCH_RETRY_BASE", "0.6"))
        self._retry_max = float(os.getenv("SEARCH_RETRY_MAX", "4.0"))

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        return asyncio.run(self._search_async(query, top_k))

    async def _search_async(self, query: str, top_k: int) -> list[SearchResult]:
        cache_key = _cache_key(
            provider="google",
            query=query,
            params={
                "top_k": top_k,
                "lang": self._lang,
                "region": self._region,
                "safe": self._safe,
                "basic": self._basic,
            },
        )
        cached = _cache_get(cache_key)
        if cached:
            search_logger.info("Search cache hit (google).")
            return cached

        headers = {
            "User-Agent": random.choice(_USERAGENT_LIST),
            "Accept-Language": f"{self._lang},en;q=0.8",
        }
        params = {
            "q": query,
            "num": top_k + 2,
            "hl": self._lang,
            "safe": self._safe,
        }
        if self._basic:
            params["gbv"] = "1"
        if self._region:
            params["gl"] = self._region

        async with httpx.AsyncClient(
            timeout=self._timeout, follow_redirects=True
        ) as client:
            last_error: Exception | None = None
            for attempt in range(1, self._retries + 1):
                try:
                    if attempt > 1:
                        delay = min(
                            self._retry_base * (2 ** (attempt - 2)), self._retry_max
                        )
                        jitter = random.uniform(0, delay * 0.2)
                        await asyncio.sleep(delay + jitter)
                    search_logger.info(
                        "Google search attempt %d/%d", attempt, self._retries
                    )
                    resp = await client.get(
                        "https://www.google.com/search",
                        headers=headers,
                        params=params,
                    )
                    resp.raise_for_status()
                    html = resp.text
                    if _is_google_blocked(html):
                        search_logger.warning(
                            "Google blocked or consent page detected."
                        )
                        _dump_search_html(html, query)
                        last_error = RuntimeError("google_blocked")
                        continue

                    soup = BeautifulSoup(html, "html.parser")
                    results = _parse_google_results(soup, top_k)
                    if not results:
                        _dump_search_html(html, query)
                    else:
                        _log_search_results("google", query, results)
                        _cache_set(cache_key, results)
                    return results
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    search_logger.warning(
                        "Google search failed (attempt %d): %s", attempt, exc
                    )
                    continue
            if last_error:
                logger.warning("Google search failed after retries: %s", last_error)
            return []


class DuckDuckGoSearchProvider(SearchProvider):
    def __init__(self, timeout: int = 6) -> None:
        self._timeout = timeout
        self._lang = os.getenv("DDG_LANG", "en")
        self._region = os.getenv("DDG_REGION", "")
        self._safe = os.getenv("DDG_SAFE", "1")
        self._retries = max(1, int(os.getenv("SEARCH_RETRIES", "3")))
        self._retry_base = float(os.getenv("SEARCH_RETRY_BASE", "0.6"))
        self._retry_max = float(os.getenv("SEARCH_RETRY_MAX", "4.0"))

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        return asyncio.run(self._search_async(query, top_k))

    async def _search_async(self, query: str, top_k: int) -> list[SearchResult]:
        cache_key = _cache_key(
            provider="duckduckgo",
            query=query,
            params={
                "top_k": top_k,
                "lang": self._lang,
                "region": self._region,
                "safe": self._safe,
            },
        )
        cached = _cache_get(cache_key)
        if cached:
            search_logger.info("Search cache hit (duckduckgo).")
            return cached

        headers = {
            "User-Agent": random.choice(_USERAGENT_LIST),
            "Accept-Language": f"{self._lang},en;q=0.8",
        }
        params = {
            "q": query,
            "kl": self._region or "us-en",
            "kp": self._safe,
        }
        async with httpx.AsyncClient(
            timeout=self._timeout, follow_redirects=True
        ) as client:
            last_error: Exception | None = None
            for attempt in range(1, self._retries + 1):
                try:
                    if attempt > 1:
                        delay = min(
                            self._retry_base * (2 ** (attempt - 2)), self._retry_max
                        )
                        jitter = random.uniform(0, delay * 0.2)
                        await asyncio.sleep(delay + jitter)
                    search_logger.info(
                        "DuckDuckGo search attempt %d/%d", attempt, self._retries
                    )
                    resp = await client.get(
                        "https://html.duckduckgo.com/html/",
                        headers=headers,
                        params=params,
                    )
                    resp.raise_for_status()
                    soup = BeautifulSoup(resp.text, "html.parser")
                    results = _parse_duckduckgo_results(soup, top_k)
                    if results:
                        _log_search_results("duckduckgo", query, results)
                        _cache_set(cache_key, results)
                    return results
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    search_logger.warning(
                        "DuckDuckGo search failed (attempt %d): %s", attempt, exc
                    )
                    continue
            if last_error:
                logger.warning("DuckDuckGo search failed after retries: %s", last_error)
            return []


def build_search_provider() -> SearchProvider | None:
    provider = os.getenv("SEARCH_PROVIDER", "google").lower()
    if provider == "google":
        return GoogleSearchProvider()
    if provider == "duckduckgo":
        return DuckDuckGoSearchProvider()
    return None


class StyleKnowledgeHarvester:
    def __init__(
        self,
        knowledge_repo: DesignKnowledgeRepository,
        client: LLMClientProtocol | None = None,
        search_provider: SearchProvider | None = None,
    ) -> None:
        self._repo = knowledge_repo
        self._client = client or get_llm_client()
        self._search = search_provider or build_search_provider()
        fallback = os.getenv("SEARCH_FALLBACK", "duckduckgo").lower()
        if fallback == "duckduckgo":
            self._fallback: SearchProvider | None = DuckDuckGoSearchProvider()
        elif fallback == "google":
            self._fallback = GoogleSearchProvider()
        else:
            self._fallback = None

    def ensure_knowledge(
        self,
        *,
        user_preferences: dict[str, Any],
        tenant_id: str | None,
        template_ids: Sequence[str] | None = None,
    ) -> None:
        if self._search is None:
            logger.info(
                "Search provider not configured; skipping soft-rule enrichment."
            )
            return

        style = (
            str(user_preferences.get("style") or "general").strip().lower() or "general"
        )
        room_type = str(user_preferences.get("room_type") or "room").strip().lower()
        if _style_has_soft_rules(repo=self._repo, style=style, tenant_id=tenant_id):
            search_logger.info(
                "Soft-rule style '%s' already exists in DB; skipping search.",
                style,
            )
            return
        search_logger.info(
            "Style harvester start. style=%s room_type=%s", style, room_type
        )

        templates = get_soft_rule_templates()
        selected_templates = (
            {
                template_id: templates[template_id]
                for template_id in template_ids or []
                if template_id in templates
            }
            if template_ids
            else templates
        )
        for template_id, template in selected_templates.items():
            if self._has_existing_knowledge(template_id, style, tenant_id):
                search_logger.info(
                    "Soft-rule knowledge exists: %s/%s", template_id, style
                )
                continue
            queries = _build_queries(
                template_id, template, style, room_type, user_preferences
            )
            results: list[SearchResult] = []
            query_used = ""
            for query in queries:
                search_logger.info("Search query: %s", query)
                results = self._search.search(query, top_k=5)
                if results:
                    query_used = query
                    break
                if self._fallback:
                    search_logger.info(
                        "Primary search empty; fallback to %s",
                        type(self._fallback).__name__,
                    )
                    logger.info(
                        "Primary search empty; fallback to %s",
                        type(self._fallback).__name__,
                    )
                    results = self._fallback.search(query, top_k=5)
                    if results:
                        query_used = query
                        break
            if not results:
                search_logger.warning("No search results for %s", template_id)
                logger.warning(
                    "Search yielded no results for %s (style=%s, room=%s).",
                    template_id,
                    style,
                    room_type,
                )
                continue
            existing_links = _extract_existing_links(
                repo=self._repo,
                template_id=template_id,
                style=style,
                tenant_id=tenant_id,
            )
            if existing_links:
                filtered = []
                for item in results:
                    url = _normalize_url(item.url)
                    if url and url not in existing_links:
                        filtered.append(item)
                if not filtered:
                    search_logger.info(
                        "All search results already stored for %s/%s. Skipping crawl.",
                        template_id,
                        style,
                    )
                    continue
                results = filtered
            results = _crawl_results(results)
            summaries = _summarize_results(
                client=self._client,
                template_id=template_id,
                template=template,
                style=style,
                user_preferences=user_preferences,
                results=results,
            )
            search_logger.info(
                "Summaries generated: %d for %s", len(summaries), template_id
            )
            _store_summaries(
                repo=self._repo,
                client=self._client,
                tenant_id=tenant_id,
                template_id=template_id,
                style=style,
                query=query_used,
                summaries=summaries,
            )

    def _has_existing_knowledge(
        self, template_id: str, style: str, tenant_id: str | None
    ) -> bool:
        filter_tags = ["soft_rule", template_id, style]
        knowledge_filter = DesignKnowledgeFilter(
            tenant_id=TenantId(tenant_id) if tenant_id else None,
            tags=filter_tags,
        )
        existing = self._repo.list_knowledge(knowledge_filter)
        if len(existing) > 0:
            return True
        fallback_tags = ["soft_rule", template_id]
        fallback_filter = DesignKnowledgeFilter(
            tenant_id=TenantId(tenant_id) if tenant_id else None,
            tags=fallback_tags,
        )
        fallback = self._repo.list_knowledge(fallback_filter)
        return len(fallback) > 0


def _build_queries(
    template_id: str,
    template: dict[str, Any],
    style: str,
    room_type: str,
    user_preferences: dict[str, Any],
) -> list[str]:
    style_term = style or "interior design"
    room_term = room_type or "room"
    feng_raw = user_preferences.get("feng_shui") or {}
    if isinstance(feng_raw, dict):
        menh = str(feng_raw.get("menh") or "").strip()
        year = str(feng_raw.get("nam_sinh") or "").strip()
    else:
        menh = ""
        year = ""

    if template_id == "color_composition":
        return [
            f"{style_term} {room_term} interior color palette",
            f"{style_term} {room_term} color scheme 60-30-10",
            f"{style_term} {room_term} neutral palette natural materials",
        ]
    if template_id == "style_specific_aesthetics":
        return [
            f"{style_term} interior design characteristics materials texture",
            f"{style_term} {room_term} furniture material palette",
        ]
    if template_id == "room_specific_feng_shui":
        return [
            f"feng shui {room_term} layout circulation path",
            f"feng shui {room_term} furniture placement guidelines",
        ]
    if template_id == "personal_feng_shui":
        base = "feng shui personal element colors interior"
        extras = " ".join(val for val in [menh, year] if val)
        return [f"{base} {extras}".strip()]
    if template_id == "lighting_color_interaction":
        return [
            f"{style_term} {room_term} lighting color temperature palette",
            f"{room_term} natural light color palette interior design",
        ]

    fields = ", ".join(template.get("context_fields_to_find", []))
    return [f"{style_term} {room_term} interior design {fields}".strip()]


def _summarize_results(
    *,
    client: LLMClientProtocol,
    template_id: str,
    template: dict[str, Any],
    style: str,
    user_preferences: dict[str, Any],
    results: list[SearchResult],
) -> list[dict[str, Any]]:
    context_fields = template.get("context_fields_to_find", [])
    rule_items = _extract_rule_candidates(
        client=client,
        template_id=template_id,
        style=style,
        results=results,
    )
    summaries = _map_rules_to_fields(
        client=client,
        template_id=template_id,
        template=template,
        style=style,
        user_preferences=user_preferences,
        rules=rule_items,
        context_fields=context_fields,
    )
    output: list[dict[str, Any]] = []
    for item in summaries:
        if not isinstance(item, dict):
            continue
        field = item.get("field")
        summary = item.get("summary")
        sources = item.get("sources")
        if not field:
            continue
        if summary:
            url = ""
            if isinstance(sources, list) and sources:
                url = str(sources[0])
            output.append(
                {
                    "field": str(field),
                    "summary": str(summary),
                    "url": url,
                    "sources": [str(src) for src in sources]
                    if isinstance(sources, list)
                    else [],
                }
            )
        else:
            search_logger.info("No info for soft-rule field: %s", field)
            logger.info("No info for soft-rule field: %s", field)
    return output


def _extract_rule_candidates(
    *,
    client: LLMClientProtocol,
    template_id: str,
    style: str,
    results: list[SearchResult],
) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for result in results:
        text = (result.content or result.snippet or "").strip()
        if not text:
            continue
        chunks = _chunk_text(text)
        for chunk in chunks[: int(os.getenv("SEARCH_CHUNK_MAX", "5"))]:
            items = _llm_extract_rules(
                client=client,
                template_id=template_id,
                style=style,
                chunk=chunk,
                source_url=result.url,
            )
            if items:
                rules.extend(items)
        if rules:
            search_logger.info(
                "Extracted %d rule items from %s", len(rules), result.url
            )
            for item in rules[:10]:
                search_logger.info(
                    "Rule item: kind=%s text=%s source=%s",
                    item.get("kind"),
                    str(item.get("text", ""))[:180],
                    item.get("source_url"),
                )
                logger.info(
                    "Rule item: kind=%s text=%s source=%s",
                    item.get("kind"),
                    str(item.get("text", ""))[:180],
                    item.get("source_url"),
                )
    deduped = _dedupe_rules(rules)
    return deduped


def _llm_extract_rules(
    *,
    client: LLMClientProtocol,
    template_id: str,
    style: str,
    chunk: str,
    source_url: str,
) -> list[dict[str, Any]]:
    system = (
        "You extract actionable design rules, principles, or recommendations from text. "
        "Return JSON only."
    )
    payload = {
        "template_id": template_id,
        "style": style,
        "source_url": source_url,
        "text": chunk[:2000],
    }
    user = (
        "Return JSON with shape: {rules: [{kind, text, source_url}]}. "
        "kind must be one of: rule, principle, recommendation. text must be short.\n\n"
        f"INPUT: {json.dumps(payload, ensure_ascii=True)}"
    )
    try:
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            model_key="helper",
            temperature=0.1,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Rule extraction failed: %s", exc)
        return []
    choices = getattr(response, "choices", None)
    if not choices:
        return []
    content = getattr(choices[0].message, "content", "")
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return []
    items = data.get("rules")
    if not isinstance(items, list):
        return []
    output: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        kind = item.get("kind")
        if not text:
            continue
        output.append(
            {
                "kind": str(kind or "rule"),
                "text": str(text),
                "source_url": str(item.get("source_url") or source_url),
            }
        )
    return output


def _map_rules_to_fields(
    *,
    client: LLMClientProtocol,
    template_id: str,
    template: dict[str, Any],
    style: str,
    user_preferences: dict[str, Any],
    rules: list[dict[str, Any]],
    context_fields: list[str],
) -> list[dict[str, Any]]:
    payload = {
        "template_id": template_id,
        "style": style,
        "context_fields": context_fields,
        "user_preferences": user_preferences,
        "rules": rules[: int(os.getenv("SEARCH_RULE_LIMIT", "50"))],
    }
    system = (
        "You are a stylist research assistant. Map extracted rules to the given "
        "context_fields. Output JSON only."
    )
    user = (
        "Return JSON with shape: {items: [{field, summary, sources}]}. "
        "You MUST include every field from context_fields. "
        'If no info, set summary="" and sources=[].\n\n'
        f"INPUT: {json.dumps(payload, ensure_ascii=True)}"
    )
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model_key="helper",
        temperature=0.1,
    )
    choices = getattr(response, "choices", None)
    if not choices:
        return []
    content = getattr(choices[0].message, "content", "")
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return []
    items = data.get("items")
    if not isinstance(items, list):
        return []
    output: list[dict[str, Any]] = []
    for field in context_fields:
        matched = next((item for item in items if item.get("field") == field), None)
        if not isinstance(matched, dict):
            output.append({"field": field, "summary": "", "sources": []})
            continue
        summary = matched.get("summary") or ""
        sources = matched.get("sources") or []
        output.append({"field": field, "summary": summary, "sources": sources})
    return output


def _chunk_text(text: str) -> list[str]:
    max_chars = int(os.getenv("SEARCH_CHUNK_CHARS", "1200"))
    overlap = int(os.getenv("SEARCH_CHUNK_OVERLAP", "120"))
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def _dedupe_rules(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for item in rules:
        text = str(item.get("text", "")).strip().lower()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(item)
    return output


def _store_summaries(
    *,
    repo: DesignKnowledgeRepository,
    client: LLMClientProtocol,
    tenant_id: str | None,
    template_id: str,
    style: str,
    query: str,
    summaries: list[dict[str, Any]],
) -> None:
    for item in summaries:
        field = item["field"]
        summary = item["summary"]
        url = item.get("url", "")
        sources = item.get("sources") or []
        links = _unique_links([url, *sources])
        search_logger.info(
            "Store soft-rule: field=%s summary=%s source=%s",
            field,
            summary[:160] + ("..." if len(summary) > 160 else ""),
            url,
        )
        logger.info(
            "Store soft-rule: field=%s summary=%s source=%s",
            field,
            summary[:160] + ("..." if len(summary) > 160 else ""),
            url,
        )
        doc_id = _hash_id(style, template_id, field, url)
        tags = ["soft_rule", template_id, style, field]
        knowledge = DesignKnowledge(
            id=doc_id,
            tenant_id=TenantId(tenant_id) if tenant_id else None,
            title=f"{style}:{template_id}:{field}",
            content=summary,
            category="soft_rule",
            tags=tags,
            source=url,
            meta={
                "query": query,
                "field": field,
                "style": style,
                "url": url,
                "links": links,
            },
        )
        repo.create_knowledge(knowledge)
        embedding = _embed_text(client, summary)
        if embedding:
            repo.upsert_knowledge_embedding(
                DesignKnowledgeEmbedding(
                    knowledge_id=knowledge.id,
                    content=summary,
                    vector=embedding,
                    model=client.get_model_name("embedding"),
                    meta={
                        "source": url,
                        "field": field,
                        "style": style,
                        "links": links,
                    },
                )
            )


def _hash_id(style: str, template_id: str, field: str, url: str) -> str:
    raw = f"{style}:{template_id}:{field}:{url}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def _embed_text(client: LLMClientProtocol, text: str) -> list[float]:
    if not SemanticSearchConfig.ENABLED:
        return []
    response = client.embeddings(inputs=text, model_key="embedding")
    data = getattr(response, "data", None)
    if not data:
        return []
    embedding = getattr(data[0], "embedding", None)
    if not embedding:
        return []
    return [float(val) for val in embedding]


def _extract_existing_links(
    *,
    repo: DesignKnowledgeRepository,
    template_id: str,
    style: str,
    tenant_id: str | None,
) -> set[str]:
    filter_tags = ["soft_rule", template_id, style]
    knowledge_filter = DesignKnowledgeFilter(
        tenant_id=TenantId(tenant_id) if tenant_id else None,
        tags=filter_tags,
    )
    existing = repo.list_knowledge(knowledge_filter)
    links: set[str] = set()
    for item in existing:
        if item.source:
            url = _normalize_url(str(item.source))
            if url:
                links.add(url)
        if isinstance(item.meta, dict):
            meta_links = item.meta.get("links")
            if isinstance(meta_links, list):
                for link in meta_links:
                    if isinstance(link, str):
                        normalized = _normalize_url(link)
                        if normalized:
                            links.add(normalized)
    return links


def _style_has_soft_rules(
    *,
    repo: DesignKnowledgeRepository,
    style: str,
    tenant_id: str | None,
) -> bool:
    if not style:
        return False
    filter_tags = ["soft_rule", style]
    if tenant_id:
        knowledge_filter = DesignKnowledgeFilter(
            tenant_id=TenantId(tenant_id),
            tags=filter_tags,
        )
        if list(repo.list_knowledge(knowledge_filter)):
            return True
    global_filter = DesignKnowledgeFilter(
        tenant_id=None,
        tags=filter_tags,
    )
    return bool(list(repo.list_knowledge(global_filter)))


def _unique_links(raw_links: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for link in raw_links:
        if not isinstance(link, str):
            continue
        normalized = _normalize_url(link)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


def _crawl_results(results: list[SearchResult]) -> list[SearchResult]:
    if os.getenv("CRAWL4AI_ENABLED", "1").lower() in {"0", "false", "no", "off"}:
        search_logger.info("crawl4ai disabled via CRAWL4AI_ENABLED.")
        return results
    try:
        return asyncio.run(_crawl_async(results))
    except RuntimeError:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_crawl_async(results))
    except Exception as exc:  # noqa: BLE001
        search_logger.warning("crawl4ai failed to run: %s", exc)
        return results


async def _crawl_async(results: list[SearchResult]) -> list[SearchResult]:
    try:
        from crawl4ai import AsyncWebCrawler
    except ImportError:
        logger.warning("crawl4ai not installed; returning snippets only.")
        return results

    try:
        async with AsyncWebCrawler() as crawler:
            enriched: list[SearchResult] = []
            for result in results:
                url = _normalize_url(result.url)
                if not url:
                    search_logger.warning(
                        "crawl4ai skipped invalid url: %s", result.url
                    )
                    enriched.append(result)
                    continue
                try:
                    crawl = await crawler.arun(url)
                    content = _extract_crawl_text(crawl)
                    if content:
                        _content_cache_set(url, content)
                except Exception:
                    content = ""
                    search_logger.warning("crawl4ai failed for %s", url)
                enriched.append(
                    SearchResult(
                        title=result.title,
                        snippet=result.snippet,
                        url=url,
                        content=content,
                    )
                )
            return enriched
    except Exception as exc:  # noqa: BLE001
        search_logger.warning("crawl4ai init failed: %s", exc)
        return results


def _extract_crawl_text(crawl: object) -> str:
    for attr in ("markdown", "text", "content", "cleaned_html"):
        value = getattr(crawl, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _parse_google_block(block: BeautifulSoup) -> SearchResult | None:
    link = block.find("a", href=True)
    title = block.find("h3")
    snippet = block.find("div", {"style": "-webkit-line-clamp:2"}) or block.find(
        "span", attrs={"class": "aCOpRe"}
    )
    if not link or not title:
        return None
    return SearchResult(
        url=str(link["href"]),
        title=title.text,
        snippet=snippet.text if snippet else "",
    )


def _is_google_blocked(html: str) -> bool:
    lowered = html.lower()
    return (
        "unusual traffic" in lowered
        or "consent" in lowered
        or "captcha" in lowered
        or "enablejs" in lowered
        or "having trouble accessing google search" in lowered
    )


def _parse_google_results(soup: BeautifulSoup, top_k: int) -> list[SearchResult]:
    results: list[SearchResult] = []

    for block in soup.select("div.g"):
        result = _parse_google_block(block)
        if result:
            results.append(result)
        if len(results) >= top_k:
            return results

    for block in soup.select("div.tF2Cxc"):
        result = _parse_google_block(block)
        if result:
            results.append(result)
        if len(results) >= top_k:
            return results

    # Basic HTML layout (gbv=1)
    for block in soup.select("div.kCrYT"):
        link = block.find("a", href=True)
        if not link:
            continue
        url = _clean_google_url(str(link["href"]))
        if not url:
            continue
        title_text = link.get_text(" ", strip=True)
        if not title_text:
            continue
        snippet_node = block.find("div", class_="BNeawe s3v9rd AP7Wnd")
        snippet = snippet_node.get_text(" ", strip=True) if snippet_node else ""
        results.append(SearchResult(title=title_text, snippet=snippet, url=url))
        if len(results) >= top_k:
            return results

    return results


def _parse_duckduckgo_results(soup: BeautifulSoup, top_k: int) -> list[SearchResult]:
    results: list[SearchResult] = []
    for block in soup.select("div.result"):
        link = block.find("a", class_="result__a", href=True)
        if not link:
            continue
        title = link.get_text(" ", strip=True)
        snippet_node = block.find("a", class_="result__snippet") or block.find(
            "div", class_="result__snippet"
        )
        snippet = snippet_node.get_text(" ", strip=True) if snippet_node else ""
        url = _normalize_url(str(link["href"]))
        if not url:
            continue
        results.append(SearchResult(title=title, snippet=snippet, url=url))
        if len(results) >= top_k:
            break
    return results


def _clean_google_url(raw: str) -> str:
    if raw.startswith("/url?"):
        parsed = urlparse(raw)
        query = parse_qs(parsed.query)
        url = query.get("q", [""])[0]
        return url
    if raw.startswith("http"):
        return raw
    return ""


def _normalize_url(raw: str) -> str:
    if not raw:
        return ""
    if raw.startswith("//"):
        raw = f"https:{raw}"
    if "duckduckgo.com/l/?" in raw:
        parsed = urlparse(raw)
        query = parse_qs(parsed.query)
        url = query.get("uddg", [""])[0]
        if url and url.startswith("http"):
            return url
    if raw.startswith("/url?"):
        return _clean_google_url(raw)
    if raw.startswith("http"):
        return raw
    return ""


def _dump_search_html(html: str, query: str) -> None:
    try:
        safe = hashlib.sha256(query.encode("utf-8")).hexdigest()[:8]
        path = os.path.join("logs", f"search_debug_{safe}.html")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(html)
        search_logger.info("Saved search debug HTML to %s", path)
    except Exception:
        return


def _cache_key(*, provider: str, query: str, params: dict[str, Any]) -> str:
    payload = {"provider": provider, "query": query, "params": params}
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _cache_get(key: str) -> list[SearchResult]:
    cache = _load_cache()
    entry = cache.get(key)
    if not entry:
        return []
    ts = entry.get("ts", 0)
    if _CACHE_TTL_SEC > 0 and time.time() - ts > _CACHE_TTL_SEC:
        return []
    raw_results = entry.get("results", [])
    results: list[SearchResult] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url", ""))
        content = str(item.get("content", "")) if item.get("content") else ""
        if not content:
            content = _content_cache_get(url) or ""
        results.append(
            SearchResult(
                title=str(item.get("title", "")),
                snippet=str(item.get("snippet", "")),
                url=url,
                content=content,
            )
        )
    return results


def _cache_set(key: str, results: list[SearchResult]) -> None:
    cache = _load_cache()
    cache[key] = {
        "ts": time.time(),
        "results": [
            {
                "title": r.title,
                "snippet": r.snippet,
                "url": r.url,
                "content": r.content,
            }
            for r in results
        ],
    }
    _save_cache(cache)


def _load_cache() -> dict[str, Any]:
    try:
        if not os.path.exists(_CACHE_PATH):
            return {}
        with open(_CACHE_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return data
    except Exception:  # noqa: BLE001
        return {}
    return {}


def _save_cache(cache: dict[str, Any]) -> None:
    try:
        if _CACHE_MAX > 0 and len(cache) > _CACHE_MAX:
            items = sorted(cache.items(), key=lambda item: item[1].get("ts", 0))
            cache = dict(items[-_CACHE_MAX:])
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        with open(_CACHE_PATH, "w", encoding="utf-8") as handle:
            json.dump(cache, handle, ensure_ascii=True, indent=2)
    except Exception:  # noqa: BLE001
        return


def _log_search_results(provider: str, query: str, results: list[SearchResult]) -> None:
    if not results:
        return
    preview = results[:5]
    for idx, item in enumerate(preview, start=1):
        snippet = (item.snippet or "").strip().replace("\n", " ")
        if len(snippet) > 180:
            snippet = snippet[:177] + "..."
        message = f"{provider} result {idx}: {item.title} | {snippet} | {item.url}"
        search_logger.info(message)
        logger.info(message)


def _content_cache_get(url: str) -> str:
    if not url:
        return ""
    cache = _load_content_cache()
    entry = cache.get(url)
    if not entry:
        return ""
    return str(entry.get("content", "")) if entry.get("content") else ""


def _content_cache_set(url: str, content: str) -> None:
    if not url or not content:
        return
    cache = _load_content_cache()
    cache[url] = {"ts": time.time(), "content": content}
    _save_content_cache(cache)


def _load_content_cache() -> dict[str, Any]:
    try:
        if not os.path.exists(_CONTENT_CACHE_PATH):
            return {}
        with open(_CONTENT_CACHE_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return data
    except Exception:  # noqa: BLE001
        return {}
    return {}


def _save_content_cache(cache: dict[str, Any]) -> None:
    try:
        if _CONTENT_CACHE_MAX > 0 and len(cache) > _CONTENT_CACHE_MAX:
            items = sorted(cache.items(), key=lambda item: item[1].get("ts", 0))
            cache = dict(items[-_CONTENT_CACHE_MAX:])
        os.makedirs(os.path.dirname(_CONTENT_CACHE_PATH), exist_ok=True)
        with open(_CONTENT_CACHE_PATH, "w", encoding="utf-8") as handle:
            json.dump(cache, handle, ensure_ascii=True, indent=2)
    except Exception:  # noqa: BLE001
        return
