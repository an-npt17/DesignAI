from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from db import PostgresAssetRepository, PostgresDesignKnowledgeRepository
from db.models import AssetFilter, DesignKnowledgeFilter, TenantId
from db.postgres import create_connection

# ============================================================
# Size profile builder (S/M/L) with robust stats
# ============================================================


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        raise ValueError("percentile() on empty list")
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ys[int(k)]
    return ys[f] + (ys[c] - ys[f]) * (k - f)


def _median(xs: list[float]) -> float:
    if not xs:
        raise ValueError("median() on empty list")
    ys = sorted(xs)
    mid = len(ys) // 2
    if len(ys) % 2 == 1:
        return ys[mid]
    return (ys[mid - 1] + ys[mid]) / 2


def _mad(xs: list[float]) -> float:
    if not xs:
        raise ValueError("mad() on empty list")
    m = _median(xs)
    return _median([abs(x - m) for x in xs])


def _safe_log(x: float, eps: float = 1e-12) -> float:
    return math.log(max(x, eps))


@dataclass(frozen=True)
class _NormItem:
    inventory_id: str
    category: str
    L: float
    W: float
    H: float
    A: float
    R: float
    raw: dict[str, Any]


def _get_category(item: dict[str, Any]) -> str:
    attrs = item.get("attributes") or {}
    return str(attrs.get("category") or item.get("type") or "unknown")


def _normalize_item(item: dict[str, Any]) -> _NormItem | None:
    try:
        inv_id = str(item.get("id", "")).strip()
        if not inv_id:
            return None

        cat = _get_category(item)

        length_mm = float(item["length_mm"])
        width_mm = float(item["width_mm"])
        height_mm = float(item["height_mm"])
        if length_mm <= 0 or width_mm <= 0 or height_mm <= 0:
            return None

        L = length_mm / 1000.0
        W = width_mm / 1000.0
        H = height_mm / 1000.0
        if W > L:
            L, W = W, L

        A = L * W
        R = L / W if W > 0 else float("inf")

        return _NormItem(
            inventory_id=inv_id,
            category=cat,
            L=L,
            W=W,
            H=H,
            A=A,
            R=R,
            raw=item,
        )
    except Exception:
        return None


def _filter_outliers_iqr_logA(
    items: list[_NormItem], k: float = 1.5
) -> list[_NormItem]:
    if len(items) < 8:
        return items[:]
    log_as = [_safe_log(it.A) for it in items]
    q1 = _percentile(log_as, 25)
    q3 = _percentile(log_as, 75)
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    kept = [it for it in items if lo <= _safe_log(it.A) <= hi]
    return kept if kept else items[:]


def _compute_thresholds(
    items: list[_NormItem], p_low: float, p_high: float
) -> tuple[float, float]:
    areas = [it.A for it in items]
    a_low = _percentile(areas, p_low)
    a_high = _percentile(areas, p_high)
    if a_high < a_low:
        a_low, a_high = a_high, a_low
    return a_low, a_high


def _tier_of_area(area: float, a_low: float, a_high: float) -> str:
    if area <= a_low:
        return "S"
    if area <= a_high:
        return "M"
    return "L"


def _medoid(items: list[_NormItem]) -> _NormItem | None:
    if not items:
        return None
    if len(items) == 1:
        return items[0]

    ls = [it.L for it in items]
    ws = [it.W for it in items]
    hs = [it.H for it in items]
    m_l, m_w, m_h = _median(ls), _median(ws), _median(hs)
    mad_l, mad_w, mad_h = _mad(ls), _mad(ws), _mad(hs)

    eps = 1e-9
    mad_l = mad_l if mad_l > 0 else eps
    mad_w = mad_w if mad_w > 0 else eps
    mad_h = mad_h if mad_h > 0 else eps

    best: _NormItem | None = None
    best_score = float("inf")
    for it in items:
        z_l = (it.L - m_l) / mad_l
        z_w = (it.W - m_w) / mad_w
        z_h = (it.H - m_h) / mad_h
        score = z_l * z_l + z_w * z_w + z_h * z_h
        if score < best_score:
            best_score = score
            best = it
    return best


def _build_size_profiles(
    inventory: list[dict[str, Any]],
    *,
    p_low: float = 30,
    p_high: float = 70,
    outlier_k: float = 1.5,
    min_n_for_cat_thresholds: int = 30,
) -> dict[str, Any]:
    norm = [n for item in inventory if (n := _normalize_item(item)) is not None]
    if not norm:
        return {}

    by_cat: dict[str, list[_NormItem]] = {}
    for it in norm:
        by_cat.setdefault(it.category, []).append(it)

    filtered_global = _filter_outliers_iqr_logA(norm, k=outlier_k)
    filtered_cat = {
        c: _filter_outliers_iqr_logA(v, k=outlier_k) for c, v in by_cat.items()
    }

    profiles: dict[str, Any] = {}
    for cat, items_cat in by_cat.items():
        base = filtered_cat[cat]
        source = "category"
        if len(base) < min_n_for_cat_thresholds:
            base = filtered_global
            source = "global"

        a_low, a_high = _compute_thresholds(base, p_low, p_high)

        tiers: dict[str, list[_NormItem]] = {"S": [], "M": [], "L": []}
        for it in items_cat:
            tiers[_tier_of_area(it.A, a_low, a_high)].append(it)

        rep_dims: dict[str, Any] = {}
        for t in ("S", "M", "L"):
            rep = _medoid(tiers[t])
            rep_dims[t] = (
                None
                if rep is None
                else {
                    "L": rep.L,
                    "W": rep.W,
                    "H": rep.H,
                    "A": rep.A,
                    "R": rep.R,
                    "source_id": rep.inventory_id,
                }
            )

        profiles[cat] = {
            "metric": "footprint_area_m2",
            "percentiles": {"p_low": p_low, "p_high": p_high},
            "thresholds": {"A_low": a_low, "A_high": a_high},
            "rep_dims_m": rep_dims,
            "counts": {
                "n_category": len(items_cat),
                "n_threshold_base": len(base),
                "threshold_source": source,
                "n_by_tier": {k: len(v) for k, v in tiers.items()},
            },
        }

    return profiles


# ============================================================
# Tool: get_size_profiles
# ============================================================


def _load_inventory_from_db(tenant_id: str) -> list[dict[str, Any]]:
    repo = PostgresAssetRepository(connection_factory=create_connection)
    assets = list(
        repo.list_assets(AssetFilter(tenant_id=TenantId(tenant_id), type="FURNITURE"))
    )

    items: list[dict[str, Any]] = []
    for asset in assets:
        dims = asset.dimensions
        if dims is None:
            continue
        if dims.length_mm is None or dims.width_mm is None or dims.height_mm is None:
            continue
        items.append(
            {
                "id": str(asset.id),
                "type": str(asset.type),
                "attributes": dict(asset.attributes),
                "length_mm": float(dims.length_mm),
                "width_mm": float(dims.width_mm),
                "height_mm": float(dims.height_mm),
            }
        )
    return items


def _load_profiles_from_db(tenant_id: str | None) -> dict[str, Any] | None:
    repo = PostgresDesignKnowledgeRepository(connection_factory=create_connection)
    if tenant_id:
        knowledge = repo.get_knowledge(f"size_rules:{tenant_id}")
        if knowledge is not None:
            return _parse_profiles_payload(knowledge.content)

    knowledge_global = repo.get_knowledge("size_rules:global")
    if knowledge_global is not None:
        return _parse_profiles_payload(knowledge_global.content)

    candidates = list(repo.list_knowledge(DesignKnowledgeFilter(category="size_rules")))
    if candidates:
        return _parse_profiles_payload(candidates[0].content)
    return None


def _parse_profiles_payload(content: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def get_size_profiles(
    *,
    categories: list[str],
    tenant_id: str | None = None,
) -> dict[str, Any]:
    profiles_payload = _load_profiles_from_db(tenant_id)
    if profiles_payload is None:
        inventory = _load_inventory_from_db(tenant_id or "demo_tenant")
        profiles_payload = _build_size_profiles(inventory)

    profiles_by_category = (
        profiles_payload if isinstance(profiles_payload, dict) else {}
    )
    requested: dict[str, Any] = {}
    missing: list[str] = []
    for category in categories:
        if category in profiles_by_category:
            requested[category] = profiles_by_category[category]
        else:
            missing.append(category)

    return {
        "size_profiles_by_category": requested,
        "missing_categories": missing,
    }


# ============================================================
# Tool: estimate_budget
# ============================================================


def _polygon_area_m2(points_ccw_mm: list[dict[str, int]]) -> float:
    if len(points_ccw_mm) < 3:
        return 0.0
    s = 0
    n = len(points_ccw_mm)
    for i in range(n):
        x1, y1 = points_ccw_mm[i]["x"], points_ccw_mm[i]["y"]
        x2, y2 = points_ccw_mm[(i + 1) % n]["x"], points_ccw_mm[(i + 1) % n]["y"]
        s += x1 * y2 - x2 * y1
    area_mm2 = abs(s) / 2.0
    return area_mm2 / 1_000_000.0


def _polygon_perimeter_m(points_mm: list[dict[str, int]]) -> float:
    if len(points_mm) < 2:
        return 0.0
    per_mm = 0.0
    n = len(points_mm)
    for i in range(n):
        x1, y1 = points_mm[i]["x"], points_mm[i]["y"]
        x2, y2 = points_mm[(i + 1) % n]["x"], points_mm[(i + 1) % n]["y"]
        dx, dy = x2 - x1, y2 - y1
        per_mm += (dx * dx + dy * dy) ** 0.5
    return per_mm / 1000.0


def _compactness(room_pts_mm: list[dict[str, int]]) -> float:
    area = _polygon_area_m2(room_pts_mm)
    perimeter = _polygon_perimeter_m(room_pts_mm)
    if area <= 0 or perimeter <= 0:
        return 0.0
    return (4.0 * math.pi * area) / (perimeter * perimeter)


def _available_area_m2(room_model: dict[str, Any]) -> float:
    room_pts = (room_model.get("room") or {}).get("polygon_ccw") or []
    room_area = _polygon_area_m2(room_pts)
    obstacles = room_model.get("obstacles", []) or []

    try:
        from shapely.geometry import Polygon  # type: ignore
        from shapely.ops import unary_union  # type: ignore

        room_poly = Polygon([(p["x"], p["y"]) for p in room_pts])
        obs_polys = []
        for ob in obstacles:
            if not ob.get("hard", True):
                continue
            pts = ob.get("polygon_ccw") or []
            if len(pts) >= 3:
                obs_polys.append(Polygon([(p["x"], p["y"]) for p in pts]))
        if not obs_polys:
            return room_poly.area / 1_000_000.0
        free = room_poly.difference(unary_union(obs_polys))
        return float(free.area) / 1_000_000.0
    except Exception:
        obs_area = 0.0
        for ob in obstacles:
            if not ob.get("hard", True):
                continue
            pts = ob.get("polygon_ccw") or []
            obs_area += _polygon_area_m2(pts) if pts else 0.0
        return max(0.0, room_area - obs_area)


def _footprint_m2_from_profile(size_profile: dict[str, Any], tier: str) -> float:
    rep = size_profile.get("rep_dims_m", {}).get(tier.upper())
    if not rep:
        raise ValueError(f"Missing rep_dims_m for tier={tier}")
    return float(rep["L"]) * float(rep["W"])


def _estimate_decision_footprint_m2(
    decision: dict[str, Any],
    size_profiles_by_category: dict[str, dict[str, Any]],
) -> float:
    category = str(decision["category"])
    tier = str(decision["size_tier"])
    qty = int(decision.get("quantity", 1))
    profile = size_profiles_by_category.get(category)
    if profile is None:
        raise ValueError(f"Missing size profile for category={category}")
    return _footprint_m2_from_profile(profile, tier) * qty


def _sum_footprint_by_cluster_m2(
    decisions: list[dict[str, Any]],
    size_profiles_by_category: dict[str, dict[str, Any]],
) -> dict[str, float]:
    totals: dict[str, float] = {}
    for decision in decisions:
        cluster_id = str(decision.get("cluster_id", "unknown_cluster"))
        totals.setdefault(cluster_id, 0.0)
        totals[cluster_id] += _estimate_decision_footprint_m2(
            decision, size_profiles_by_category
        )
    return totals


def _sum_footprint_breakdown(
    decisions: list[dict[str, Any]],
    size_profiles_by_category: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for decision in decisions:
        cluster_id = str(decision.get("cluster_id", "unknown_cluster"))
        priority = str(decision.get("priority", "secondary"))
        out.setdefault(
            cluster_id,
            {"anchor": 0.0, "primary": 0.0, "secondary": 0.0, "optional": 0.0},
        )
        fp = _estimate_decision_footprint_m2(decision, size_profiles_by_category)
        if priority not in out[cluster_id]:
            priority = "secondary"
        out[cluster_id][priority] += fp
    return out


def _choose_budget_ratio(style: str, user_notes: str) -> float:
    style_l = (style or "").lower()
    notes_l = (user_notes or "").lower()

    ratio = 0.50
    if "japandi" in style_l or "minimal" in style_l or "scandi" in style_l:
        ratio = 0.45
    if "cozy" in notes_l or "ấm cúng" in notes_l:
        ratio = 0.60
    if (
        "airy" in notes_l
        or "spacious" in notes_l
        or "thoáng" in notes_l
        or "tối giản" in notes_l
    ):
        ratio = min(ratio, 0.45)

    return ratio


def _check_cluster_budgets(
    cluster_areas_m2: dict[str, float],
    cluster_footprints_m2: dict[str, float],
    ratio: float,
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for cid, fp in cluster_footprints_m2.items():
        area = cluster_areas_m2.get(cid)
        if area is None or area <= 0:
            continue
        limit = area * ratio
        if fp > limit:
            violations.append(
                {
                    "cluster_id": cid,
                    "area_m2": area,
                    "ratio": ratio,
                    "limit_m2": limit,
                    "footprint_m2": fp,
                    "over_by_m2": fp - limit,
                }
            )
    return violations


def _extract_zone_areas_m2(room_model: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    zones = room_model.get("zones", []) or []
    for zone in zones:
        zone_id = str(zone.get("id", "unknown_zone"))
        poly = zone.get("polygon_ccw", [])
        out[zone_id] = _polygon_area_m2(poly) if poly else 0.0
    return out


def estimate_budget(
    *,
    room_model: dict[str, Any],
    decisions: list[dict[str, Any]],
    size_profiles_by_category: dict[str, dict[str, Any]],
    style: str,
    user_notes: str,
    ratio_override: float | None = None,
    cluster_area_mode: str = "room",
) -> dict[str, Any]:
    base_ratio = (
        float(ratio_override)
        if ratio_override is not None
        else _choose_budget_ratio(style, user_notes)
    )

    cluster_footprints = _sum_footprint_by_cluster_m2(
        decisions, size_profiles_by_category
    )
    footprint_breakdown = _sum_footprint_breakdown(decisions, size_profiles_by_category)

    room_poly = (room_model.get("room") or {}).get("polygon_ccw") or []
    room_area = _available_area_m2(room_model) if room_poly else 0.0
    compactness = _compactness(room_poly) if room_poly else 0.0
    shape_factor = max(0.7, min(1.0, compactness / 0.6)) if compactness > 0 else 1.0
    ratio = base_ratio * shape_factor

    if cluster_area_mode == "zones":
        zone_areas = _extract_zone_areas_m2(room_model)
        cluster_areas = {
            cid: zone_areas.get(cid, 0.0) for cid in cluster_footprints.keys()
        }
        for cid in list(cluster_areas.keys()):
            if cluster_areas[cid] <= 0 and room_area > 0:
                cluster_areas[cid] = room_area
    else:
        cluster_areas = {cid: room_area for cid in cluster_footprints.keys()}

    violations = _check_cluster_budgets(cluster_areas, cluster_footprints, ratio)

    return {
        "ratio": ratio,
        "base_ratio": base_ratio,
        "shape_compactness": compactness,
        "shape_factor": shape_factor,
        "cluster_area_mode": cluster_area_mode,
        "room_area_m2": room_area,
        "cluster_areas_m2": cluster_areas,
        "cluster_footprints_m2": cluster_footprints,
        "cluster_footprints_by_priority_m2": footprint_breakdown,
        "violations": violations,
    }


TOOL_REGISTRY: dict[str, Any] = {
    "get_size_profiles": get_size_profiles,
    "estimate_budget": estimate_budget,
}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_size_profiles",
            "description": "Return representative dimensions (S/M/L) and thresholds for each category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "categories": {"type": "array", "items": {"type": "string"}},
                    "tenant_id": {"type": ["string", "null"]},
                },
                "required": ["categories"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_budget",
            "description": "Estimate cluster footprints (m^2) from tiers and check area budget violations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "room_model": {"type": "object"},
                    "decisions": {"type": "array", "items": {"type": "object"}},
                    "size_profiles_by_category": {"type": "object"},
                    "style": {"type": "string"},
                    "user_notes": {"type": "string"},
                    "ratio_override": {"type": ["number", "null"]},
                    "cluster_area_mode": {"type": "string", "enum": ["room", "zones"]},
                },
                "required": [
                    "room_model",
                    "decisions",
                    "size_profiles_by_category",
                    "style",
                    "user_notes",
                ],
            },
        },
    },
]
