from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.coordinate_normalization_service import CoordinateNormalizationService

router = APIRouter(prefix="/coordinates", tags=["coordinates"])


class CoordinateNormalizeRequest(BaseModel):
    payload: dict[str, Any] = Field(..., description="Raw frontend floorplan payload (walls, rooms, objects in frontend units).")
    source_unit: Literal[
        "auto",
        "m",
        "meter",
        "meters",
        "mm",
        "millimeter",
        "millimeters",
    ] = Field(
        default="auto",
        description="Unit of the coordinates in the payload. Use 'auto' to let the system infer from the data.",
    )
    tenant_id: str | None = Field(default="demo_tenant", description="Tenant context used for room configuration lookups.")
    user_id: str | None = Field(default="coordinate_normalizer_preview", description="User ID attached to the normalized payload for tracing.")
    description: str | None = Field(default=None, description="Optional design description forwarded to the pipeline.")
    special_notes: str | None = Field(default=None, description="Optional extra instructions forwarded to the pipeline.")
    style: str | None = Field(default="modern", description="Design style hint (e.g. 'modern', 'classic') used during normalization.")
    split_largest_room: bool = Field(default=True, description="If true, the largest room is split into sub-zones for better layout coverage.")


class CoordinateNormalizeResponse(BaseModel):
    normalized_payload: dict[str, Any]
    transform: dict[str, Any]
    apartment: dict[str, Any]
    rooms: list[dict[str, Any]]
    system_inputs: list[dict[str, Any]] = Field(default_factory=list)
    room_split: dict[str, Any] = Field(default_factory=dict)


class CoordinateRestoreRequest(BaseModel):
    output_payload: Any = Field(
        ...,
        description="Pipeline output payload whose coordinates should be mapped back to the original frontend space.",
    )
    transform: dict[str, Any] = Field(
        ...,
        description="The `transform` object returned by `POST /coordinates/normalize-input`. Must match the normalization call.",
    )
    coordinate_space: Literal["apartment_normalized", "room_local"] = Field(
        default="room_local",
        description=(
            "Coordinate space the pipeline output is expressed in:\n"
            "- `room_local` — coordinates are relative to a single room origin\n"
            "- `apartment_normalized` — coordinates are in the whole-apartment normalized frame"
        ),
    )
    room_id: str | None = Field(default=None, description="Required when coordinate_space is 'room_local'. ID of the room the output belongs to.")
    rotation_input: Literal["auto", "degrees", "radians", "quaternion"] = Field(
        default="auto",
        description="Format of rotation values in the output payload. Use 'auto' to infer automatically.",
    )


class CoordinateRestoreResponse(BaseModel):
    restored_payload: Any
    transform_applied: dict[str, Any]


def get_coordinate_service() -> CoordinateNormalizationService:
    return CoordinateNormalizationService()


@router.post("/normalize-input", response_model=CoordinateNormalizeResponse, summary="Normalize frontend coordinates for pipeline input")
def normalize_input(request: CoordinateNormalizeRequest) -> dict[str, Any]:
    """
    Convert a raw frontend floorplan payload into the normalized coordinate system expected by the pipeline.

    Returns:
    - `normalized_payload`: ready-to-use pipeline input
    - `transform`: opaque object needed to reverse the normalization — **save this** and pass it to `POST /coordinates/restore-output`
    - `apartment` / `rooms`: parsed room geometry for reference
    - `system_inputs`: per-room pipeline input blocks
    - `room_split`: details of how the largest room was subdivided (if `split_largest_room=true`)

    Raises `422` if the payload cannot be parsed or normalized.
    """
    try:
        return get_coordinate_service().normalize_input(
            request.payload,
            source_unit=request.source_unit,
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            description=request.description,
            special_notes=request.special_notes,
            style=request.style,
            split_largest_room=request.split_largest_room,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/restore-output", response_model=CoordinateRestoreResponse, summary="Restore pipeline output coordinates back to frontend space")
def restore_output(request: CoordinateRestoreRequest) -> dict[str, Any]:
    """
    Apply the inverse of the normalization transform to map pipeline output coordinates
    back into the original frontend coordinate space.

    Use this after `GET /pipeline/{case_id}/result` to get object positions that can be
    directly rendered in the frontend viewer.

    Raises `422` if the transform is incompatible with the output payload.
    """
    try:
        return get_coordinate_service().restore_output(
            request.output_payload,
            request.transform,
            coordinate_space=request.coordinate_space,
            room_id=request.room_id,
            rotation_input=request.rotation_input,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
