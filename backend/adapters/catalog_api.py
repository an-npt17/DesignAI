from __future__ import annotations

import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Literal
from urllib.parse import urljoin

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)

DEFAULT_API_BASE_URL = "https://auto-furniture-api2.a-star.group"
DEFAULT_ASSET_BASE_URL = "https://storage.mazig.io"
DEFAULT_PAGE_LIMIT = 500
DEFAULT_MAX_PAGES = 20
DEFAULT_TIMEOUT_SECONDS = 15.0

CatalogRotationPresence = Literal["null", "present"]

_SEMANTIC_OBJECT_TYPE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "nightstand",
        (
            "nightstand",
            "bedside_table",
            "sidebed",
            "side_bed",
            "side_table_bed",
            "tu_canh_giuong",
            "tu_dau_giuong",
        ),
    ),
    (
        "wardrobe",
        ("wardrobe", "closet", "armoire", "tu_quan_ao", "tu_ao"),
    ),
    (
        "bookshelf",
        ("bookshelf", "bookcase", "book_shelf", "shelf", "ke_sach", "tu_sach"),
    ),
    (
        "desk",
        ("desk", "work_desk", "study_desk", "ban_lam_viec", "ban_go_lam_viec"),
    ),
    (
        "dining_table",
        ("dining_table", "ban_an"),
    ),
    (
        "coffee_table",
        ("coffee_table", "ban_tra", "ban_cafe"),
    ),
    (
        "bed",
        ("bed", "giuong", "giuong_ngu"),
    ),
    (
        "chair",
        ("chair", "desk_chair", "dining_chair", "ghe", "ghe_tua"),
    ),
    (
        "sofa",
        ("sofa", "couch", "ghe_sofa"),
    ),
    (
        "tv_console",
        ("tv_console", "media_console", "tv_stand", "ke_tivi", "tu_tivi"),
    ),
    (
        "dresser",
        ("dresser", "chest_of_drawers", "tu_ngan_keo"),
    ),
)


class CatalogApiError(RuntimeError):
    pass


@dataclass(frozen=True)
class CatalogApiSettings:
    api_base_url: str = DEFAULT_API_BASE_URL
    asset_base_url: str = DEFAULT_ASSET_BASE_URL
    page_limit: int = DEFAULT_PAGE_LIMIT
    max_pages: int = DEFAULT_MAX_PAGES
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS


class CatalogCategory(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str
    parent_id: str | None = Field(default=None, alias="parentId")
    slug: str
    name: str
    name_vn: str | None = Field(default=None, alias="nameVn")
    icon_url: str | None = Field(default=None, alias="iconUrl")
    sort_order: float | None = Field(default=None, alias="sortOrder")


class CatalogItem(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str
    slug: str | None = None
    sku_slug: str | None = Field(default=None, alias="skuSlug")
    category_id: str | None = Field(default=None, alias="categoryId")
    name: str
    name_vn: str | None = Field(default=None, alias="nameVn")
    description: str | None = None
    description_vn: str | None = Field(default=None, alias="descriptionVn")
    model_url: str | None = Field(default=None, alias="modelUrl")
    thumbnail_url: str | None = Field(default=None, alias="thumbnailUrl")
    shape_type: str | None = Field(default=None, alias="shapeType")
    placement_type: str | None = Field(default=None, alias="placementType")
    size: tuple[float, float, float] | None = None
    color_default: str | None = Field(default=None, alias="colorDefault")
    brand: str | None = None
    price_cents: float | None = Field(default=None, alias="priceCents")
    currency: str | None = None
    default_variant_sku: str | None = Field(default=None, alias="defaultVariantSku")
    object_role: str | None = Field(default=None, alias="objectRole")
    default_rotation: tuple[float, float, float, float] | None = Field(
        default=None,
        alias="defaultRotation",
    )

    @field_validator("size", mode="before")
    @classmethod
    def _clean_size(cls, value: object) -> tuple[float, float, float] | None:
        if not isinstance(value, list | tuple) or len(value) != 3:
            return None
        try:
            size = tuple(float(item) for item in value)
        except (TypeError, ValueError):
            return None
        if any(item <= 0 for item in size):
            return None
        return size

    @field_validator("default_rotation", mode="before")
    @classmethod
    def _clean_rotation(
        cls,
        value: object,
    ) -> tuple[float, float, float, float] | None:
        if value is None:
            return None
        if not isinstance(value, list | tuple) or len(value) != 4:
            return None
        try:
            return tuple(float(item) for item in value)
        except (TypeError, ValueError):
            return None

    def dimensions_mm(self) -> dict[str, float | None]:
        if self.size is None:
            return {"length_mm": None, "width_mm": None, "height_mm": None}
        x_m, y_m, z_m = self.size
        return {
            "length_mm": x_m * 1000.0,
            "width_mm": z_m * 1000.0,
            "height_mm": y_m * 1000.0,
        }

    def inventory_type(self, *, category_slug: str | None) -> str:
        if isinstance(self.object_role, str) and self.object_role.strip():
            return _norm_key(self.object_role)
        inferred_type = infer_catalog_object_type(
            name=self.name,
            name_vn=self.name_vn,
            slug=self.slug,
            sku_slug=self.sku_slug,
            model_url=self.model_url,
            category_slug=category_slug,
        )
        if inferred_type is not None:
            return inferred_type
        for value in (category_slug, self.category_id, self.name):
            if isinstance(value, str) and value.strip():
                return _norm_key(value)
        return "unknown"

    def matches_types(
        self,
        requested_types: set[str],
        *,
        category_slug: str | None,
    ) -> bool:
        if not requested_types:
            return True
        candidates = {
            _norm_key(value)
            for value in (
                self.object_role,
                category_slug,
                self.category_id,
                self.slug,
                self.sku_slug,
                self.name,
                self.name_vn,
                self.model_url,
                self.inventory_type(category_slug=category_slug),
            )
            if isinstance(value, str) and value.strip()
        }
        return not candidates.isdisjoint(requested_types)

    def matches_search(self, query: str, *, category_slug: str | None) -> bool:
        text = query.strip().lower()
        if not text:
            return True
        haystack = " ".join(
            value
            for value in (
                self.object_role,
                category_slug,
                self.category_id,
                self.slug,
                self.sku_slug,
                self.name,
                self.name_vn,
                self.description,
                self.description_vn,
                self.model_url,
            )
            if isinstance(value, str) and value.strip()
        ).lower()
        return text in haystack

    def to_inventory_payload(
        self,
        *,
        category_slug: str | None,
        asset_base_url: str,
    ) -> dict[str, object]:
        dimensions = self.dimensions_mm()
        inventory_type = self.inventory_type(category_slug=category_slug)
        catalog_category_slug = _norm_key(category_slug) if category_slug else None
        model_url = _resolve_asset_url(asset_base_url, self.model_url)
        thumbnail_url = _resolve_asset_url(asset_base_url, self.thumbnail_url)
        default_rotation = (
            list(self.default_rotation) if self.default_rotation is not None else None
        )
        size_m = list(self.size) if self.size is not None else None

        files: list[dict[str, object]] = []
        if model_url is not None:
            files.append(
                {
                    "id": f"{self.id}:model",
                    "file_kind": "MODEL",
                    "provider": "remote",
                    "storage_key": model_url,
                    "mime": "model/gltf-binary",
                    "role": "model",
                    "meta": {"source": "catalog_api"},
                    "url": model_url,
                }
            )
        if thumbnail_url is not None:
            files.append(
                {
                    "id": f"{self.id}:thumbnail",
                    "file_kind": "PREVIEW",
                    "provider": "remote",
                    "storage_key": thumbnail_url,
                    "mime": None,
                    "role": "thumbnail",
                    "meta": {"source": "catalog_api"},
                    "url": thumbnail_url,
                }
            )

        attributes: dict[str, object] = {
            "source": "catalog_api",
            "ownership_scope": "shared",
            "catalog_id": self.id,
            "inventory_id": self.id,
            "inventory_name": self.name_vn or self.name,
            "category": inventory_type,
            "semantic_object_type": inventory_type,
            "category_id": self.category_id,
            "category_slug": category_slug,
            "catalog_category_slug": catalog_category_slug,
            "catalog_name": self.name,
            "catalog_name_vn": self.name_vn,
            "slug": self.slug,
            "sku_slug": self.sku_slug,
            "shape_type": self.shape_type,
            "placement_type": self.placement_type,
            "color_hex": self.color_default,
            "model_url": model_url,
            "modelUrl": model_url,
            "thumbnail_url": thumbnail_url,
            "thumbnailUrl": thumbnail_url,
            "size_m": size_m,
            "size_mm_xyz": [value * 1000.0 for value in self.size]
            if self.size is not None
            else None,
            "default_rotation": default_rotation,
            "defaultRotation": default_rotation,
            "object_role": self.object_role,
            "objectRole": self.object_role,
            "price_cents": self.price_cents,
            "currency": self.currency,
            "default_variant_sku": self.default_variant_sku,
            "files": files,
        }
        for key, value in dimensions.items():
            attributes[key] = value

        return {
            "id": self.id,
            "inventory_id": self.id,
            "catalog_id": self.id,
            "name": self.name_vn or self.name,
            "inventory_name": self.name_vn or self.name,
            "object_type": inventory_type,
            "type": inventory_type,
            "asset_type": "FURNITURE",
            "style_tags": _stable_tags(category_slug, inventory_type),
            "material": None,
            "brand": self.brand,
            "length_mm": dimensions["length_mm"],
            "width_mm": dimensions["width_mm"],
            "height_mm": dimensions["height_mm"],
            "dimensions": dimensions,
            "attributes": attributes,
        }


class CatalogItemsPage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    limit: int
    offset: int
    total: int
    count: int
    category: CatalogCategory | None = None
    items: list[CatalogItem] = Field(default_factory=list)


class CatalogCategoriesResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    categories: list[CatalogCategory] = Field(default_factory=list)


class CatalogApiClient:
    def __init__(
        self,
        settings: CatalogApiSettings | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._settings = settings or load_catalog_api_settings()
        self._http_client = http_client or httpx.Client(
            timeout=self._settings.timeout_seconds
        )
        self._owns_http_client = http_client is None

    def close(self) -> None:
        if self._owns_http_client:
            self._http_client.close()

    def list_categories(self) -> list[CatalogCategory]:
        payload = self._get_json("/api/catalog/categories", params={})
        return CatalogCategoriesResponse.model_validate(payload).categories

    def list_items_page(
        self,
        *,
        item_id: str | None = None,
        category_id: str | None = None,
        search: str | None = None,
        limit: int | None = None,
        offset: int = 0,
        default_rotation_presence: CatalogRotationPresence | None = "present",
    ) -> CatalogItemsPage:
        params: dict[str, str] = {
            "limit": str(_clamp_page_limit(limit or self._settings.page_limit)),
            "offset": str(max(0, offset)),
        }
        if item_id:
            params["id"] = item_id
        elif category_id:
            params["categoryId"] = category_id
        elif search:
            params["search"] = search
        if default_rotation_presence is not None:
            params["defaultRotationPresence"] = default_rotation_presence

        payload = self._get_json("/api/catalog/items", params=params)
        page = CatalogItemsPage.model_validate(payload)
        if default_rotation_presence == "present":
            page.items = [
                item for item in page.items if item.default_rotation is not None
            ]
            page.count = len(page.items)
        return page

    def list_all_items(
        self,
        *,
        search: str | None = None,
        default_rotation_presence: CatalogRotationPresence | None = "present",
    ) -> list[CatalogItem]:
        limit = _clamp_page_limit(self._settings.page_limit)
        offset = 0
        items: list[CatalogItem] = []

        for _ in range(max(1, self._settings.max_pages)):
            page = self.list_items_page(
                search=search,
                limit=limit,
                offset=offset,
                default_rotation_presence=default_rotation_presence,
            )
            items.extend(page.items)
            if page.offset + page.limit >= page.total:
                break
            offset += page.limit

        return items

    def list_inventory_payloads(
        self,
        *,
        item_ids: list[str] | None = None,
        types: list[str] | None = None,
        search: str | None = None,
        limit: int | None = None,
        default_rotation_presence: CatalogRotationPresence | None = "present",
    ) -> list[dict[str, object]]:
        categories = {category.id: category.slug for category in self.list_categories()}
        requested_types = {_norm_key(value) for value in types or [] if value.strip()}
        payloads: list[dict[str, object]] = []
        seen_ids: set[str] = set()

        def append_item(item: CatalogItem) -> None:
            if item.id in seen_ids:
                return
            category_slug = categories.get(item.category_id or "")
            payloads.append(
                item.to_inventory_payload(
                    category_slug=category_slug,
                    asset_base_url=self._settings.asset_base_url,
                )
            )
            seen_ids.add(item.id)

        for item_id in item_ids or []:
            page = self.list_items_page(
                item_id=item_id,
                limit=1,
                offset=0,
                default_rotation_presence=default_rotation_presence,
            )
            for item in page.items:
                append_item(item)

        should_list_items = bool(requested_types or search or not item_ids)
        items = (
            self.list_all_items(
                search=search,
                default_rotation_presence=default_rotation_presence,
            )
            if should_list_items
            else []
        )
        for item in items:
            if item.id in seen_ids:
                continue
            category_slug = categories.get(item.category_id or "")
            if not item.matches_types(requested_types, category_slug=category_slug):
                continue
            if search and not item.matches_search(search, category_slug=category_slug):
                continue
            append_item(item)
            if isinstance(limit, int) and limit > 0 and len(payloads) >= limit:
                break
        return payloads

    def _get_json(self, path: str, *, params: dict[str, str]) -> object:
        url = urljoin(self._settings.api_base_url.rstrip("/") + "/", path.lstrip("/"))
        try:
            response = self._http_client.get(url, params=params)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise CatalogApiError(f"Catalog API request failed: {url}") from exc
        return response.json()


def load_catalog_api_settings() -> CatalogApiSettings:
    return CatalogApiSettings(
        api_base_url=_env_str("TKNT_CATALOG_API_BASE_URL", DEFAULT_API_BASE_URL),
        asset_base_url=_env_str("TKNT_CATALOG_ASSET_BASE_URL", DEFAULT_ASSET_BASE_URL),
        page_limit=_env_int("TKNT_CATALOG_API_PAGE_LIMIT", DEFAULT_PAGE_LIMIT),
        max_pages=_env_int("TKNT_CATALOG_API_MAX_PAGES", DEFAULT_MAX_PAGES),
        timeout_seconds=_env_float(
            "TKNT_CATALOG_API_TIMEOUT_SECONDS",
            DEFAULT_TIMEOUT_SECONDS,
        ),
    )


def load_catalog_inventory_payloads(
    *,
    item_ids: list[str] | None = None,
    types: list[str] | None = None,
    search: str | None = None,
    limit: int | None = None,
    default_rotation_presence: CatalogRotationPresence | None = "present",
) -> list[dict[str, object]]:
    client = CatalogApiClient()
    try:
        return client.list_inventory_payloads(
            item_ids=item_ids,
            types=types,
            search=search,
            limit=limit,
            default_rotation_presence=default_rotation_presence,
        )
    finally:
        client.close()


def _resolve_asset_url(base_url: str, value: str | None) -> str | None:
    if not value:
        return None
    if value.startswith(("http://", "https://")):
        return value
    return urljoin(base_url.rstrip("/") + "/", value.lstrip("/"))


def _clamp_page_limit(value: int) -> int:
    return min(max(value, 1), DEFAULT_PAGE_LIMIT)


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid integer for %s: %s", name, raw)
        return default
    return value if value > 0 else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Invalid float for %s: %s", name, raw)
        return default
    return value if value > 0 else default


def _norm_key(value: str | None) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _stable_tags(*values: str | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        tag = value.strip()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        out.append(tag)
    return out


def infer_catalog_object_type(
    *,
    name: str | None = None,
    name_vn: str | None = None,
    slug: str | None = None,
    sku_slug: str | None = None,
    model_url: str | None = None,
    category_slug: str | None = None,
) -> str | None:
    normalized_values = [
        _ascii_norm_key(value)
        for value in (name, name_vn, slug, sku_slug, model_url, category_slug)
        if isinstance(value, str) and value.strip()
    ]
    if not normalized_values:
        return None
    haystack = " ".join(normalized_values)
    for object_type, tokens in _SEMANTIC_OBJECT_TYPE_PATTERNS:
        if any(_token_in_haystack(token, haystack) for token in tokens):
            return object_type
    return None


def infer_catalog_object_type_from_payload(
    payload: Mapping[str, object],
) -> str | None:
    attributes_raw = payload.get("attributes")
    attributes = attributes_raw if isinstance(attributes_raw, Mapping) else {}
    for key in ("object_type", "semantic_object_type", "object_role", "objectRole"):
        value = payload.get(key) if key in payload else attributes.get(key)
        if isinstance(value, str) and value.strip():
            return _norm_key(value)
    return infer_catalog_object_type(
        name=_payload_str(payload, "name") or _payload_str(attributes, "catalog_name"),
        name_vn=_payload_str(attributes, "catalog_name_vn"),
        slug=_payload_str(attributes, "slug"),
        sku_slug=_payload_str(attributes, "sku_slug"),
        model_url=_payload_str(attributes, "model_url")
        or _payload_str(attributes, "modelUrl"),
        category_slug=_payload_str(attributes, "category_slug")
        or _payload_str(payload, "type"),
    )


def _payload_str(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _ascii_norm_key(value: str | None) -> str:
    stripped = "".join(
        char
        for char in unicodedata.normalize("NFKD", str(value or ""))
        if not unicodedata.combining(char)
    )
    lowered = stripped.lower()
    return re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")


def _token_in_haystack(token: str, haystack: str) -> bool:
    normalized = _ascii_norm_key(token)
    if not normalized:
        return False
    return re.search(rf"(?:^|_){re.escape(normalized)}(?:_|$)", haystack) is not None
