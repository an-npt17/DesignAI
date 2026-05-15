from __future__ import annotations

from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from api.deps import get_current_user
from db.models import JsonValue, UserAccount
from pipeline.image_flow_logging import log_image_flow_event, summarize_image_data_url
from services.user_content_service import UserContentService

router = APIRouter(prefix="/account", tags=["account"])


class SavedLayoutRecord(BaseModel):
    id: str
    name: str
    floorplan_json: dict[str, Any]
    design_json: dict[str, Any] | None = None
    styled_result_json: dict[str, Any] | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None


class SavedLayoutCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=160, description="Human-readable name for the layout.")
    floorplan_json: dict[str, Any] = Field(..., description="Raw floorplan geometry from the frontend.")
    design_json: dict[str, Any] | None = Field(default=None, description="Optional design configuration (style, palette, etc.).")
    styled_result_json: dict[str, Any] | None = Field(default=None, description="Optional pipeline output (placed objects with positions).")
    meta: dict[str, Any] = Field(default_factory=dict, description="Arbitrary key/value metadata (e.g. thumbnail URL, tags).")


class GeneratedRenderRecord(BaseModel):
    id: str
    source: str
    model_name: str
    prompt: str
    negative_prompt: str | None = None
    mime_type: str
    meta: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    file_url: str


class GeneratedRenderDetailRecord(BaseModel):
    render: GeneratedRenderRecord
    image_data_url: str


class GeneratedRenderMetaUpdateRequest(BaseModel):
    meta: dict[str, Any] = Field(default_factory=dict)


def get_user_content_service() -> UserContentService:
    return UserContentService()


@router.get("/layouts", response_model=list[SavedLayoutRecord], summary="List saved layouts")
def list_layouts(
    user: UserAccount = Depends(get_current_user),
    service: UserContentService = Depends(get_user_content_service),
) -> list[SavedLayoutRecord]:
    """Return all layouts saved by the authenticated user, ordered by creation date (newest first)."""
    return [
        SavedLayoutRecord(**service.serialize_saved_layout(layout))
        for layout in service.list_layouts(user=user)
    ]


@router.post("/layouts", response_model=SavedLayoutRecord, summary="Save a new layout")
def create_layout(
    request: SavedLayoutCreateRequest,
    user: UserAccount = Depends(get_current_user),
    service: UserContentService = Depends(get_user_content_service),
) -> SavedLayoutRecord:
    """
    Persist a floorplan layout for the authenticated user.

    Optionally include `design_json` (style preferences) and `styled_result_json`
    (the pipeline output) to save the full design state in a single call.
    """
    layout = service.save_layout(
        user=user,
        name=request.name,
        floorplan_json=_coerce_json_dict(request.floorplan_json),
        design_json=_coerce_json_dict(request.design_json)
        if request.design_json is not None
        else None,
        styled_result_json=_coerce_json_dict(request.styled_result_json)
        if request.styled_result_json is not None
        else None,
        meta=_coerce_json_dict(request.meta),
    )
    return SavedLayoutRecord(**service.serialize_saved_layout(layout))


@router.get("/layouts/{layout_id}", response_model=SavedLayoutRecord, summary="Get a saved layout by ID")
def get_layout(
    layout_id: str,
    user: UserAccount = Depends(get_current_user),
    service: UserContentService = Depends(get_user_content_service),
) -> SavedLayoutRecord:
    """
    Fetch a single saved layout. Raises `404` if it doesn't exist or belongs to a different user.
    """
    layout = service.get_layout(user=user, layout_id=layout_id)
    if layout is None:
        raise HTTPException(status_code=404, detail="Layout not found.")
    return SavedLayoutRecord(**service.serialize_saved_layout(layout))


@router.get("/renders", response_model=list[GeneratedRenderRecord], summary="List generated render images")
def list_renders(
    user: UserAccount = Depends(get_current_user),
    service: UserContentService = Depends(get_user_content_service),
) -> list[GeneratedRenderRecord]:
    """Return all AI-generated room renders saved for the authenticated user. Does not include the image data — use `GET /renders/{id}` for that."""
    return [
        GeneratedRenderRecord(
            **service.serialize_generated_render(
                render,
                file_url=f"/account/renders/{render.id}/file",
            )
        )
        for render in service.list_generated_renders(user=user)
    ]


@router.get("/renders/{render_id}", response_model=GeneratedRenderDetailRecord, summary="Get a render with its image data URL")
def get_render(
    render_id: str,
    user: UserAccount = Depends(get_current_user),
    service: UserContentService = Depends(get_user_content_service),
) -> GeneratedRenderDetailRecord:
    """
    Fetch a single render record including the full base64 `image_data_url` (e.g. `data:image/png;base64,...`).

    Use `GET /renders/{id}/file` instead when you only need to display or download the raw image bytes.
    Raises `404` if the render doesn't exist or belongs to a different user.
    """
    trace_id = uuid4().hex
    log_image_flow_event(
        "snapshot_image.saved_render.load_input",
        {
            "trace_id": trace_id,
            "route": "/account/renders/{render_id}",
            "render_id": render_id,
            "user_id": str(user.id),
        },
    )
    render = service.get_generated_render(user=user, render_id=render_id)
    if render is None:
        log_image_flow_event(
            "snapshot_image.saved_render.load_error",
            {
                "trace_id": trace_id,
                "render_id": render_id,
                "status_code": 404,
                "error_message": "Render not found.",
            },
        )
        raise HTTPException(status_code=404, detail="Render not found.")
    try:
        image_data_url = service.build_generated_render_data_url(render)
    except Exception as exc:
        log_image_flow_event(
            "snapshot_image.saved_render.load_error",
            {
                "trace_id": trace_id,
                "render_id": render_id,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise
    serialized = service.serialize_generated_render(
        render,
        file_url=f"/account/renders/{render.id}/file",
    )
    log_image_flow_event(
        "snapshot_image.saved_render.load_output",
        {
            "trace_id": trace_id,
            "render_id": str(render.id),
            "model_name": render.model_name,
            "mime_type": render.mime_type,
            "storage_path": render.storage_path,
            "meta": dict(render.meta or {}),
            "image": summarize_image_data_url(image_data_url),
        },
    )
    return GeneratedRenderDetailRecord(
        render=GeneratedRenderRecord(**serialized),
        image_data_url=image_data_url,
    )


@router.patch("/renders/{render_id}/meta", response_model=GeneratedRenderRecord, summary="Update render metadata")
def update_render_meta(
    render_id: str,
    request: GeneratedRenderMetaUpdateRequest,
    user: UserAccount = Depends(get_current_user),
    service: UserContentService = Depends(get_user_content_service),
) -> GeneratedRenderRecord:
    """
    Merge the provided `meta` dict into the render's existing metadata (shallow patch).

    Keys present in the request overwrite existing values; omitted keys are preserved.
    Raises `404` if the render doesn't exist or belongs to a different user.
    """
    render = service.update_generated_render_meta(
        user=user,
        render_id=render_id,
        meta_patch=_coerce_json_dict(request.meta),
    )
    if render is None:
        raise HTTPException(status_code=404, detail="Render not found.")
    return GeneratedRenderRecord(
        **service.serialize_generated_render(
            render,
            file_url=f"/account/renders/{render.id}/file",
        )
    )


@router.get("/renders/{render_id}/file", summary="Download render image as a binary file")
def get_render_file(
    render_id: str,
    user: UserAccount = Depends(get_current_user),
    service: UserContentService = Depends(get_user_content_service),
) -> Response:
    """
    Stream the render image as raw bytes with the correct `Content-Type` (e.g. `image/png`).

    Prefer this over `GET /renders/{id}` when embedding or downloading the image directly,
    since it avoids the overhead of base64 encoding.
    """
    render = service.get_generated_render(user=user, render_id=render_id)
    if render is None:
        raise HTTPException(status_code=404, detail="Render not found.")
    if render.image_bytes is not None:
        return Response(
            content=service.read_generated_render_bytes(render),
            media_type=render.mime_type,
            headers={
                "Content-Disposition": f'inline; filename="{render.id}"',
            },
        )
    if not render.storage_path:
        raise HTTPException(status_code=404, detail="Render file not found.")
    return FileResponse(
        render.storage_path,
        media_type=render.mime_type,
        filename=f"{render.id}",
    )


def _coerce_json_dict(value: dict[str, Any]) -> dict[str, JsonValue]:
    return {str(key): item for key, item in value.items()}
