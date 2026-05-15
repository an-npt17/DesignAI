from __future__ import annotations

import logging
import mimetypes
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from adapters.catalog_api import CatalogApiError, load_catalog_inventory_payloads
from api.deps import get_optional_current_user
from config.demo_inventory import (
    is_demo_inventory_tenant,
    is_enabled_demo_inventory_tenant,
)
from db.models import UserAccount
from db.pg_assets import PostgresAssetRepository
from services.auth_service import get_shared_inventory_tenant_id
from services.user_content_service import UserContentService

router = APIRouter(prefix="/inventory", tags=["inventory"])
logger = logging.getLogger(__name__)


class InventoryItem(BaseModel):
    id: str = Field(..., description="Unique item identifier.")
    inventory_id: str | None = Field(default=None, description="Internal inventory record ID.")
    catalog_id: str | None = Field(default=None, description="ID in the external catalog system.")
    name: str = Field(..., description="Display name of the item.")
    inventory_name: str | None = Field(default=None, description="Name as it appears in the inventory source.")
    object_type: str | None = Field(default=None, description="Semantic object type (e.g. 'sofa', 'table').")
    type: str = Field(..., description="Top-level category used for filtering (e.g. 'seating', 'storage').")
    style_tags: list[str] = Field(default_factory=list, description="Style keywords such as 'modern', 'scandinavian', 'minimalist'.")
    material: str | None = Field(default=None, description="Primary material (e.g. 'wood', 'fabric', 'metal').")
    brand: str | None = Field(default=None, description="Manufacturer or brand name.")
    dimensions: dict[str, float | None] | None = Field(default=None, description="Physical dimensions in metres: keys are 'width', 'depth', 'height'.")
    attributes: dict[str, object] = Field(default_factory=dict, description="Additional item-specific attributes from the catalog.")


class InventoryListResponse(BaseModel):
    tenant_id: str
    items: list[InventoryItem]


class InventoryTypesResponse(BaseModel):
    tenant_id: str
    types: list[str]


class InventorySearchResponse(BaseModel):
    tenant_id: str
    query: str
    items: list[InventoryItem]


def get_user_content_service() -> UserContentService:
    return UserContentService()


def get_asset_repository() -> PostgresAssetRepository:
    return PostgresAssetRepository()


@router.get("/items", response_model=InventoryListResponse, summary="List inventory items")
def list_items(
    tenant_id: str | None = Query(default=None, description="Tenant whose catalog to query. Defaults to the shared demo tenant."),
    types: list[str] | None = Query(default=None, description="Filter by one or more item types (e.g. `types=seating&types=storage`)."),
    style_tags: list[str] | None = Query(default=None, description="Filter to items that have at least one of the given style tags."),
) -> InventoryListResponse:
    """
    Return all inventory items for the given tenant, with optional filters.

    Filters are applied server-side after loading from the catalog. If both `types` and
    `style_tags` are provided, both conditions must be satisfied.
    """
    shared_tenant = tenant_id or get_shared_inventory_tenant_id()
    items = _load_inventory_items(
        shared_tenant_id=shared_tenant,
        types=types,
        style_tags=style_tags,
    )
    return InventoryListResponse(tenant_id=shared_tenant, items=items)


@router.get("/types", response_model=InventoryTypesResponse, summary="List all available item types")
def list_types(
    tenant_id: str | None = Query(default=None, description="Tenant whose catalog to query. Defaults to the shared demo tenant."),
) -> InventoryTypesResponse:
    """Return a sorted list of unique `type` values present in the inventory. Useful for populating filter dropdowns."""
    shared_tenant = tenant_id or get_shared_inventory_tenant_id()
    items = _load_inventory_items(
        shared_tenant_id=shared_tenant,
    )
    types_set = {item.type for item in items}
    return InventoryTypesResponse(tenant_id=shared_tenant, types=sorted(types_set))


@router.get("/search", response_model=InventorySearchResponse, summary="Search inventory by keyword")
def search_items(
    q: str = Query(..., min_length=1, description="Search query string. Matched against item name and attributes."),
    tenant_id: str | None = Query(default=None, description="Tenant whose catalog to query. Defaults to the shared demo tenant."),
    limit: int = Query(default=20, ge=1, le=200, description="Maximum number of results to return (1–200)."),
) -> InventorySearchResponse:
    """Search the inventory by keyword. Returns up to `limit` matching items."""
    shared_tenant = tenant_id or get_shared_inventory_tenant_id()
    items = _load_inventory_items(
        shared_tenant_id=shared_tenant,
        search=q,
        limit=limit,
    )
    return InventorySearchResponse(
        tenant_id=shared_tenant,
        query=q,
        items=items,
    )


@router.get("/files/{asset_file_id}", summary="Download an asset file (3D model, texture, image)")
def get_inventory_file(
    asset_file_id: str,
    current_user: UserAccount | None = Depends(get_optional_current_user),
    service: UserContentService = Depends(get_user_content_service),
    asset_repository: PostgresAssetRepository = Depends(get_asset_repository),
) -> FileResponse:
    """
    Stream an asset file by its `asset_file_id`.

    Supported file types include GLB (3D models), PNG/JPEG (textures), and others.
    `.glb` files are served with `Content-Type: model/gltf-binary`.

    - Raises `403` if the authenticated user does not own the asset.
    - Raises `404` if the file record or the file on disk cannot be found.
    - Public demo inventory files can be accessed without authentication.
    """
    asset_file = asset_repository.get_asset_file(asset_file_id)
    if asset_file is None:
        raise HTTPException(status_code=404, detail="Asset file not found.")
    asset = asset_repository.get_asset(asset_file.asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail="Asset not found.")
    if is_demo_inventory_tenant(
        str(asset.tenant_id)
    ) and not is_enabled_demo_inventory_tenant(str(asset.tenant_id)):
        raise HTTPException(status_code=404, detail="Asset file not found.")
    if not service.can_access_asset(user=current_user, asset=asset):
        raise HTTPException(status_code=403, detail="Access denied.")

    file_path = Path(asset_file.storage_key).expanduser().resolve()
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Asset file is missing on disk.")

    media_type, _ = mimetypes.guess_type(file_path.name)
    if file_path.suffix.lower() == ".glb":
        media_type = "model/gltf-binary"
    return FileResponse(
        file_path,
        media_type=media_type or "application/octet-stream",
        filename=file_path.name,
    )


def _load_inventory_items(
    *,
    shared_tenant_id: str,
    types: list[str] | None = None,
    style_tags: list[str] | None = None,
    search: str | None = None,
    limit: int | None = None,
) -> list[InventoryItem]:
    if is_demo_inventory_tenant(
        shared_tenant_id
    ) and not is_enabled_demo_inventory_tenant(shared_tenant_id):
        return []
    try:
        payloads = load_catalog_inventory_payloads(
            types=types,
            search=search,
            limit=limit,
        )
    except CatalogApiError as exc:
        logger.exception("Catalog inventory API request failed.")
        raise HTTPException(
            status_code=502,
            detail="Catalog inventory API is unavailable.",
        ) from exc

    style_set = {
        value.strip()
        for value in (style_tags or [])
        if isinstance(value, str) and value.strip()
    }
    results: list[InventoryItem] = []
    for payload in payloads:
        item = InventoryItem(**payload)
        if style_set and style_set.isdisjoint(set(item.style_tags)):
            continue
        results.append(item)
        if isinstance(limit, int) and limit > 0 and len(results) >= limit:
            break
    return results
