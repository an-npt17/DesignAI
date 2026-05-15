from __future__ import annotations

import json
from collections.abc import Sequence

from db.models import DesignKnowledge, TenantId
from db.repositories import DesignKnowledgeRepository
from stylist.room_essentials_seed import ROOM_ESSENTIALS_SEED


def load_room_essentials(
    repo: DesignKnowledgeRepository,
    *,
    tenant_id: TenantId | None = None,
) -> int:
    count = 0
    for room_type, items in ROOM_ESSENTIALS_SEED.items():
        knowledge_id = f"room_essentials:{room_type}"
        content = _build_content(room_type, items)
        knowledge = DesignKnowledge(
            id=knowledge_id,
            tenant_id=tenant_id,
            title=f"Room essentials: {room_type}",
            content=content,
            category="room_essentials",
            tags=["room_essentials", room_type],
            source="room_essentials_seed",
            meta={"room_type": room_type, "items": items},
        )
        repo.upsert_knowledge(knowledge)
        count += 1
    return count


def _build_content(room_type: str, items: Sequence[dict[str, object]]) -> str:
    parts: list[str] = []
    for item in items:
        name = str(item.get("item") or "").strip()
        if not name:
            continue
        recommended = item.get("recommended")
        if isinstance(recommended, (int, float)):
            parts.append(f"{name} x{int(recommended)}")
        else:
            parts.append(name)
    joined = ", ".join(parts)
    return json.dumps(
        {"room_type": room_type, "items": items, "summary": joined},
        ensure_ascii=True,
    )
