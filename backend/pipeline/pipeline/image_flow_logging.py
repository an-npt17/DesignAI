from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Final


logger = logging.getLogger("image_flow")

_IMAGE_DATA_KEYS: Final[set[str]] = {
    "data",
    "data_url",
    "dataUrl",
    "image_base64",
    "imageBase64",
    "image_data_url",
    "imageDataUrl",
    "snapshot_image_data_url",
    "snapshotImageDataUrl",
    "layout_reference_image_data_url",
    "layoutReferenceImageDataUrl",
    "annotated_reference_image_data_url",
    "annotatedReferenceImageDataUrl",
    "scene_reference_image_data_url",
    "sceneReferenceImageDataUrl",
    "edit_source_image_data_url",
    "editSourceImageDataUrl",
    "replacement_image_data_url",
    "replacementImageDataUrl",
}
_MAX_TEXT_LOG_CHARS: Final[int] = 20_000


def log_image_flow_event(event: str, payload: dict[str, object]) -> None:
    entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **payload,
    }
    logger.info(
        json.dumps(
            redact_for_image_log(entry),
            ensure_ascii=False,
            default=str,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def redact_for_image_log(value: object, *, key: str | None = None) -> object:
    if isinstance(value, str):
        return _redact_string(value, key=key)
    if isinstance(value, bytes):
        return _summarize_bytes(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(item_key): redact_for_image_log(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [redact_for_image_log(item) for item in value]
    if isinstance(value, tuple):
        return [redact_for_image_log(item) for item in value]
    return value


def summarize_image_data_url(data_url: str | None) -> dict[str, object] | None:
    if data_url is None:
        return None
    normalized = data_url.strip()
    if not normalized.startswith("data:image/") or "," not in normalized:
        return {
            "kind": "invalid_or_non_image_data_url",
            "length": len(normalized),
            "sha256_16": _sha256_16(normalized.encode("utf-8", errors="replace")),
        }
    prefix, _, image_base64 = normalized.partition(",")
    mime_section = prefix.removeprefix("data:")
    mime_type, _, _ = mime_section.partition(";")
    return {
        "kind": "image_data_url",
        "mime_type": mime_type or "application/octet-stream",
        "base64_length": len(image_base64),
        "approx_bytes": _approx_base64_bytes(image_base64),
        "sha256_16": _sha256_16(image_base64.encode("ascii", errors="ignore")),
    }


def summarize_gemini_payload(payload: object) -> object:
    return redact_for_image_log(payload)


def summarize_gemini_response(payload: object) -> object:
    return redact_for_image_log(payload)


def _redact_string(value: str, *, key: str | None) -> object:
    normalized_key = key or ""
    if value.startswith("data:image/") and "," in value:
        return summarize_image_data_url(value)
    if normalized_key in _IMAGE_DATA_KEYS and _looks_like_base64(value):
        return _summarize_base64_string(value)
    if len(value) > _MAX_TEXT_LOG_CHARS:
        return {
            "kind": "long_text",
            "length": len(value),
            "preview": value[:_MAX_TEXT_LOG_CHARS],
            "truncated": True,
            "sha256_16": _sha256_16(value.encode("utf-8", errors="replace")),
        }
    return value


def _summarize_base64_string(value: str) -> dict[str, object]:
    normalized = value.strip()
    return {
        "kind": "base64_string",
        "base64_length": len(normalized),
        "approx_bytes": _approx_base64_bytes(normalized),
        "sha256_16": _sha256_16(normalized.encode("ascii", errors="ignore")),
    }


def _summarize_bytes(value: bytes) -> dict[str, object]:
    return {
        "kind": "bytes",
        "byte_length": len(value),
        "sha256_16": _sha256_16(value),
    }


def _looks_like_base64(value: str) -> bool:
    normalized = value.strip()
    if len(normalized) < 128:
        return False
    allowed = set(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n\r"
    )
    return all(character in allowed for character in normalized)


def _approx_base64_bytes(value: str) -> int:
    normalized = "".join(value.split())
    padding = len(normalized) - len(normalized.rstrip("="))
    return max(0, (len(normalized) * 3) // 4 - padding)


def _sha256_16(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()[:16]
