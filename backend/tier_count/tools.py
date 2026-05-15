from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

from db import PostgresAssetRepository, PostgresDesignKnowledgeRepository
from db.models import AssetFilter, DesignKnowledgeFilter, TenantId
from db.postgres import create_connection

# ============================================================
# Generic helpers
# ============================================================


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _norm_key(value: str | None) -> str:
    s = str(value or "").strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    return s


def _safe_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


_HARD_REQUEST_CONTRACT_INTENTS = {"must_keep", "must_try"}
_SOFT_REQUEST_CONTRACT_INTENTS = {
    "target_if_viable",
    "preferred_if_fit",
    "optional_if_surplus",
}


# ============================================================
# Size profile builder (S/M/L) with robust stats + backfill
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


def _rep_from_items_for_tier(items_cat: list[_NormItem], tier: str) -> _NormItem | None:
    if not items_cat:
        return None
    if tier == "S":
        return min(items_cat, key=lambda it: it.A)
    if tier == "L":
        return max(items_cat, key=lambda it: it.A)
    return _medoid(items_cat)


def _rep_dict(rep: _NormItem | None) -> dict[str, Any] | None:
    if rep is None:
        return None
    return {
        "L": rep.L,
        "W": rep.W,
        "H": rep.H,
        "A": rep.A,
        "R": rep.R,
        "source_id": rep.inventory_id,
    }


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
        backfilled: dict[str, bool] = {"S": False, "M": False, "L": False}

        for t in ("S", "M", "L"):
            rep = _medoid(tiers[t])
            if rep is None:
                rep = _rep_from_items_for_tier(items_cat, t)
                backfilled[t] = True
            rep_dims[t] = _rep_dict(rep)

        profiles[cat] = {
            "metric": "footprint_area_m2",
            "percentiles": {"p_low": p_low, "p_high": p_high},
            "thresholds": {"A_low": a_low, "A_high": a_high},
            "rep_dims_m": rep_dims,
            "rep_dims_backfilled": backfilled,
            "counts": {
                "n_category": len(items_cat),
                "n_threshold_base": len(base),
                "threshold_source": source,
                "n_by_tier": {k: len(v) for k, v in tiers.items()},
            },
        }

    generic_s = _medoid([it for items_cat in by_cat.values() for it in items_cat[:]])
    generic_base = filtered_global if filtered_global else norm
    a_low_g, a_high_g = _compute_thresholds(generic_base, p_low, p_high)

    tiers_g: dict[str, list[_NormItem]] = {"S": [], "M": [], "L": []}
    for it in norm:
        tiers_g[_tier_of_area(it.A, a_low_g, a_high_g)].append(it)

    generic_rep_dims: dict[str, Any] = {}
    generic_backfilled = {"S": False, "M": False, "L": False}
    for t in ("S", "M", "L"):
        rep = _medoid(tiers_g[t])
        if rep is None:
            rep = generic_s
            generic_backfilled[t] = True
        generic_rep_dims[t] = _rep_dict(rep)

    profiles["__generic__"] = {
        "metric": "footprint_area_m2",
        "percentiles": {"p_low": p_low, "p_high": p_high},
        "thresholds": {"A_low": a_low_g, "A_high": a_high_g},
        "rep_dims_m": generic_rep_dims,
        "rep_dims_backfilled": generic_backfilled,
        "counts": {
            "n_category": len(norm),
            "n_threshold_base": len(generic_base),
            "threshold_source": "global",
            "n_by_tier": {k: len(v) for k, v in tiers_g.items()},
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


def _parse_profiles_payload(content: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None

    if isinstance(parsed.get("size_profiles_by_category"), dict):
        return parsed["size_profiles_by_category"]

    return parsed


def _load_profiles_from_db(tenant_id: str | None) -> dict[str, Any] | None:
    repo = PostgresDesignKnowledgeRepository(connection_factory=create_connection)

    if tenant_id:
        knowledge = repo.get_knowledge(f"size_rules:{tenant_id}")
        if knowledge is not None:
            payload = _parse_profiles_payload(knowledge.content)
            if payload is not None:
                return payload

    knowledge_global = repo.get_knowledge("size_rules:global")
    if knowledge_global is not None:
        payload = _parse_profiles_payload(knowledge_global.content)
        if payload is not None:
            return payload

    candidates = list(repo.list_knowledge(DesignKnowledgeFilter(category="size_rules")))
    if candidates:
        payload = _parse_profiles_payload(candidates[0].content)
        if payload is not None:
            return payload

    return None


def _rep_has_positive_dims(rep: Any) -> bool:
    if not isinstance(rep, dict):
        return False
    try:
        return (
            float(rep.get("L", 0)) > 0
            and float(rep.get("W", 0)) > 0
            and float(rep.get("H", 0)) > 0
        )
    except Exception:
        return False


def _sanitize_profiles_payload(payload: dict[str, Any]) -> dict[str, Any]:
    out = payload if isinstance(payload, dict) else {}
    for _, prof in list(out.items()):
        if not isinstance(prof, dict):
            continue
        rep_dims = prof.get("rep_dims_m")
        if not isinstance(rep_dims, dict):
            continue

        fallbacks = []
        for t in ("M", "L", "S"):
            rep = rep_dims.get(t)
            if _rep_has_positive_dims(rep):
                fallbacks.append(rep)

        for t in ("S", "M", "L"):
            rep = rep_dims.get(t)
            if _rep_has_positive_dims(rep):
                continue
            if fallbacks:
                rep_dims[t] = fallbacks[0]
    return out


def _build_generic_profile_from_profiles(
    profiles_by_category: dict[str, Any],
) -> dict[str, Any] | None:
    reps_by_tier: dict[str, list[dict[str, float]]] = {"S": [], "M": [], "L": []}

    for cat, prof in profiles_by_category.items():
        if str(cat).startswith("__"):
            continue
        if not isinstance(prof, dict):
            continue
        rep_dims = prof.get("rep_dims_m") or {}
        if not isinstance(rep_dims, dict):
            continue
        for tier in ("S", "M", "L"):
            rep = rep_dims.get(tier)
            if _rep_has_positive_dims(rep):
                reps_by_tier[tier].append(
                    {
                        "L": float(rep["L"]),
                        "W": float(rep["W"]),
                        "H": float(rep["H"]),
                        "A": float(rep.get("A", float(rep["L"]) * float(rep["W"]))),
                        "R": float(
                            rep.get("R", float(rep["L"]) / max(float(rep["W"]), 1e-9))
                        ),
                        "source_id": str(rep.get("source_id", "__generic__")),
                    }
                )

    if not any(reps_by_tier.values()):
        return None

    def _median_rep(reps: list[dict[str, float]]) -> dict[str, Any] | None:
        if not reps:
            return None
        return {
            "L": _median([r["L"] for r in reps]),
            "W": _median([r["W"] for r in reps]),
            "H": _median([r["H"] for r in reps]),
            "A": _median([r["A"] for r in reps]),
            "R": _median([r["R"] for r in reps]),
            "source_id": "__generic__",
        }

    rep_s = (
        _median_rep(reps_by_tier["S"])
        or _median_rep(reps_by_tier["M"])
        or _median_rep(reps_by_tier["L"])
    )
    rep_m = (
        _median_rep(reps_by_tier["M"])
        or _median_rep(reps_by_tier["S"])
        or _median_rep(reps_by_tier["L"])
    )
    rep_l = (
        _median_rep(reps_by_tier["L"])
        or _median_rep(reps_by_tier["M"])
        or _median_rep(reps_by_tier["S"])
    )

    if not rep_s or not rep_m or not rep_l:
        return None

    return {
        "metric": "footprint_area_m2",
        "percentiles": {"p_low": 30, "p_high": 70},
        "thresholds": {"A_low": float(rep_s["A"]), "A_high": float(rep_l["A"])},
        "rep_dims_m": {"S": rep_s, "M": rep_m, "L": rep_l},
        "rep_dims_backfilled": {"S": False, "M": False, "L": False},
        "counts": {
            "n_category": sum(len(v) for v in reps_by_tier.values()),
            "n_threshold_base": sum(len(v) for v in reps_by_tier.values()),
            "threshold_source": "derived_from_existing_profiles",
            "n_by_tier": {k: len(v) for k, v in reps_by_tier.items()},
        },
    }


def get_size_profiles(
    *, categories: list[str], tenant_id: str | None = None
) -> dict[str, Any]:
    profiles_payload = _load_profiles_from_db(tenant_id)
    if isinstance(profiles_payload, dict):
        profiles_payload = _sanitize_profiles_payload(profiles_payload)

    if not isinstance(profiles_payload, dict) or not profiles_payload:
        inventory = _load_inventory_from_db(tenant_id or "demo_tenant")
        profiles_payload = _build_size_profiles(inventory)

    profiles_by_category = (
        profiles_payload if isinstance(profiles_payload, dict) else {}
    )

    if "__generic__" not in profiles_by_category:
        generic = _build_generic_profile_from_profiles(profiles_by_category)
        if generic is not None:
            profiles_by_category["__generic__"] = generic

    requested: dict[str, Any] = {}
    missing: list[str] = []
    normalized_lookup = {_norm_key(k): v for k, v in profiles_by_category.items()}

    for category in categories:
        if category in profiles_by_category:
            requested[category] = profiles_by_category[category]
            continue

        nk = _norm_key(category)
        if nk in normalized_lookup:
            requested[category] = normalized_lookup[nk]
        else:
            missing.append(category)

    if "__generic__" in profiles_by_category:
        requested["__generic__"] = profiles_by_category["__generic__"]

    return {"size_profiles_by_category": requested, "missing_categories": missing}


# ============================================================
# Tool: estimate_budget
# ============================================================

FIXED_ACCESS_CLEARANCE_RATIO = 0.25


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
            return float(room_poly.area) / 1_000_000.0
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


@dataclass(frozen=True)
class _AccessRequirements:
    object_ids: frozenset[str]
    categories: frozenset[str]


def _is_access_required(entry: dict[str, Any]) -> bool:
    return entry.get("required", True) is not False


def _extract_access_requirements(clusters_json: Any) -> _AccessRequirements:
    object_ids: set[str] = set()
    categories: set[str] = set()

    if not isinstance(clusters_json, dict):
        return _AccessRequirements(frozenset(), frozenset())

    clusters = clusters_json.get("clusters")
    if not isinstance(clusters, list):
        return _AccessRequirements(frozenset(), frozenset())

    def _consume_entry(entry: dict[str, Any]) -> None:
        if not isinstance(entry, dict):
            return
        if not _is_access_required(entry):
            return

        oid = entry.get("id") or entry.get("object_id") or entry.get("name")
        if isinstance(oid, str) and oid.strip():
            object_ids.add(_norm_key(oid))

        cat = entry.get("category")
        if isinstance(cat, str) and cat.strip():
            categories.add(_norm_key(cat))

        cats = entry.get("categories")
        if isinstance(cats, list):
            for c in cats:
                if isinstance(c, str) and c.strip():
                    categories.add(_norm_key(c))

    for c in clusters:
        if not isinstance(c, dict):
            continue

        rules = c.get("cluster_rules") or {}
        if isinstance(rules, dict):
            access_reqs = rules.get("access_requirements") or []
            if isinstance(access_reqs, list):
                for ar in access_reqs:
                    if not isinstance(ar, dict):
                        continue
                    if ar.get("type") != "front_clearance":
                        continue
                    _consume_entry(ar)

        hard = c.get("hard_constraints") or []
        if isinstance(hard, list):
            for hc in hard:
                if not isinstance(hc, dict):
                    continue
                if hc.get("type") != "requires_access":
                    continue
                if hc.get("mode") != "front_clearance":
                    continue
                _consume_entry(hc)

    return _AccessRequirements(frozenset(object_ids), frozenset(categories))


def _clearance_policy_for_category(category: str) -> tuple[float, float, float]:
    k = _norm_key(category)

    seating_tokens = ("sofa", "sectional", "armchair", "recliner", "chair", "bench")
    bed_tokens = ("bed",)
    work_tokens = (
        "desk",
        "dining_table",
        "table",
        "kitchen_island",
        "island",
        "worktop",
        "counter",
    )
    storage_tokens = (
        "wardrobe",
        "closet",
        "cabinet",
        "storage",
        "bookshelf",
        "shelf",
        "dresser",
    )
    media_tokens = ("tv_console", "tv_stand", "media_console", "media", "console")

    if any(t in k for t in seating_tokens):
        return (0.45, 0.90, 1.00)
    if any(t in k for t in bed_tokens):
        return (0.45, 0.80, 0.90)
    if any(t in k for t in work_tokens):
        return (0.40, 0.80, 0.95)
    if any(t in k for t in storage_tokens):
        return (0.35, 0.65, 0.65)
    if any(t in k for t in media_tokens):
        return (0.35, 0.60, 0.55)

    return (0.30, 0.75, 0.80)


def _effective_footprint_m2_from_profile(
    size_profile: dict[str, Any],
    tier: str,
    *,
    category: str,
    needs_front_clearance: bool,
) -> tuple[float, float, float]:
    rep = (size_profile.get("rep_dims_m") or {}).get(tier.upper())
    if not isinstance(rep, dict):
        raise ValueError(f"Missing rep_dims_m for tier={tier}")

    L = float(rep.get("L", 0) or 0)
    W = float(rep.get("W", 0) or 0)
    if L <= 0 or W <= 0:
        raise ValueError(f"Invalid rep_dims_m: L/W non-positive for tier={tier}")

    base = L * W
    if not needs_front_clearance:
        return base, base, 0.0

    min_depth, max_depth, occupancy_multiplier = _clearance_policy_for_category(
        category
    )
    clearance_depth = _clamp(W * FIXED_ACCESS_CLEARANCE_RATIO, min_depth, max_depth)
    extra = L * clearance_depth * occupancy_multiplier
    effective = base + extra
    return base, effective, clearance_depth


def _prepare_profile_lookup(
    size_profiles_by_category: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    normalized: dict[str, dict[str, Any]] = {}
    for k, v in size_profiles_by_category.items():
        if isinstance(v, dict):
            normalized[_norm_key(k)] = v

    generic = size_profiles_by_category.get("__generic__")
    if not isinstance(generic, dict):
        generic = _build_generic_profile_from_profiles(size_profiles_by_category)

    return normalized, generic


def _resolve_profile_for_category(
    category: str,
    normalized_profiles: dict[str, dict[str, Any]],
    generic_profile: dict[str, Any] | None,
) -> tuple[dict[str, Any], str]:
    nk = _norm_key(category)

    if nk in normalized_profiles:
        return normalized_profiles[nk], "exact_or_normalized"

    if generic_profile is not None:
        return generic_profile, "generic_fallback"

    raise ValueError(f"Missing size profile for category={category}")


def _decision_identity_keys(decision: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for field in ("id", "object_id", "furniture_id", "asset_id", "name"):
        value = decision.get(field)
        if isinstance(value, str) and value.strip():
            keys.add(_norm_key(value))
    return keys


def _decision_needs_access(
    decision: dict[str, Any],
    access_requirements: _AccessRequirements,
) -> bool:
    category = _norm_key(str(decision.get("category", "")))
    decision_keys = _decision_identity_keys(decision)

    if category and category in access_requirements.categories:
        return True
    if decision_keys and any(
        k in access_requirements.object_ids for k in decision_keys
    ):
        return True
    if category and category in access_requirements.object_ids:
        return True

    return False


def _estimate_decision_footprint_detail(
    decision: dict[str, Any],
    normalized_profiles: dict[str, dict[str, Any]],
    generic_profile: dict[str, Any] | None,
    *,
    access_requirements: _AccessRequirements,
) -> dict[str, Any]:
    category = str(decision["category"])
    tier = str(decision["size_tier"]).upper()
    qty = int(decision.get("quantity", 1))
    qty = max(qty, 0)

    profile, profile_source = _resolve_profile_for_category(
        category,
        normalized_profiles,
        generic_profile,
    )

    needs_access = _decision_needs_access(decision, access_requirements)

    footprint_options_by_tier: dict[str, dict[str, float]] = {}
    for cand_tier in ("S", "M", "L"):
        try:
            base_fp_per_item, effective_fp_per_item, clearance_depth_m = (
                _effective_footprint_m2_from_profile(
                    profile,
                    cand_tier,
                    category=category,
                    needs_front_clearance=needs_access,
                )
            )
            footprint_options_by_tier[cand_tier] = {
                "base_footprint_m2_per_item": base_fp_per_item,
                "effective_footprint_m2_per_item": effective_fp_per_item,
                "clearance_depth_m": clearance_depth_m,
            }
        except ValueError:
            continue

    if tier not in footprint_options_by_tier:
        raise ValueError(f"Missing rep_dims_m for category={category}, tier={tier}")

    chosen = footprint_options_by_tier[tier]

    return {
        "decision_id": next(
            (
                str(decision.get(k))
                for k in ("id", "object_id", "furniture_id", "asset_id", "name")
                if decision.get(k) is not None
            ),
            "",
        ),
        "cluster_id": str(decision.get("cluster_id", "unknown_cluster")),
        "category": category,
        "size_tier": tier,
        "quantity": qty,
        "min_quantity": max(0, _safe_int(decision.get("min_keep"), default=0)),
        "priority": str(decision.get("priority", "secondary")),
        "role": str(decision.get("role", "")),
        "bundle_id": str(decision.get("bundle_id", "")),
        "protected": bool(decision.get("protected", False)),
        "droppable": bool(decision.get("droppable", False)),
        "drop_order_bias": str(decision.get("drop_order_bias", "")),
        "preserve_level": str(decision.get("preserve_level", "")),
        "request_contract_intent": str(decision.get("request_contract_intent", "")),
        "request_contract_evidence": str(decision.get("request_contract_evidence", "")),
        "needs_front_clearance": needs_access,
        "profile_source": profile_source,
        "clearance_depth_m": float(chosen["clearance_depth_m"]),
        "base_footprint_m2_per_item": float(chosen["base_footprint_m2_per_item"]),
        "effective_footprint_m2_per_item": float(
            chosen["effective_footprint_m2_per_item"]
        ),
        "base_footprint_m2_total": float(chosen["base_footprint_m2_per_item"]) * qty,
        "effective_footprint_m2_total": float(chosen["effective_footprint_m2_per_item"])
        * qty,
        "footprint_options_by_tier": footprint_options_by_tier,
    }


def _sum_footprint_by_cluster_m2(
    decision_details: list[dict[str, Any]],
    *,
    field: str = "effective_footprint_m2_total",
) -> dict[str, float]:
    totals: dict[str, float] = {}
    for d in decision_details:
        cluster_id = str(d["cluster_id"])
        totals.setdefault(cluster_id, 0.0)
        totals[cluster_id] += float(d[field])
    return totals


def _sum_footprint_breakdown(
    decision_details: list[dict[str, Any]],
    *,
    field: str = "effective_footprint_m2_total",
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for d in decision_details:
        cluster_id = str(d["cluster_id"])
        priority = str(d.get("priority", "secondary"))
        out.setdefault(
            cluster_id,
            {"anchor": 0.0, "primary": 0.0, "secondary": 0.0, "optional": 0.0},
        )
        if priority not in out[cluster_id]:
            priority = "secondary"
        out[cluster_id][priority] += float(d[field])
    return out


def _choose_budget_ratio_style_prior(style: str, user_notes: str) -> float:
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


def _smallest_available_tier(options: dict[str, Any]) -> str | None:
    for t in ("S", "M", "L"):
        if isinstance(options.get(t), dict):
            return t
    return None


def _is_mandatory_priority(priority: str) -> bool:
    return _norm_key(priority) in {"anchor", "primary"}


def _summarize_constraint_pressure(
    *,
    decision_details: list[dict[str, Any]],
    available_area_m2: float,
    room_area_m2: float,
    compactness: float,
) -> dict[str, float]:
    mandatory_base_m2 = 0.0
    mandatory_effective_m2 = 0.0
    mandatory_item_count = 0

    for row in decision_details:
        if not isinstance(row, dict):
            continue
        if not _is_mandatory_priority(str(row.get("priority", ""))):
            continue

        qty = max(0, int(row.get("quantity", 0)))
        if qty <= 0:
            continue

        options = row.get("footprint_options_by_tier") or {}
        if not isinstance(options, dict):
            continue

        tier = _smallest_available_tier(options)
        if tier is None:
            continue

        opt = options.get(tier)
        if not isinstance(opt, dict):
            continue

        base_per_item = float(opt.get("base_footprint_m2_per_item", 0.0))
        effective_per_item = float(opt.get("effective_footprint_m2_per_item", 0.0))
        mandatory_base_m2 += base_per_item * qty
        mandatory_effective_m2 += effective_per_item * qty
        mandatory_item_count += qty

    obstacle_ratio = 0.0
    if room_area_m2 > 1e-9:
        obstacle_ratio = _clamp(
            max(0.0, room_area_m2 - available_area_m2) / room_area_m2,
            0.0,
            0.95,
        )

    mandatory_ratio = 0.0
    if available_area_m2 > 1e-9:
        mandatory_ratio = mandatory_effective_m2 / available_area_m2

    access_extra_m2 = max(0.0, mandatory_effective_m2 - mandatory_base_m2)
    access_pressure = 0.0
    if mandatory_effective_m2 > 1e-9:
        access_pressure = access_extra_m2 / mandatory_effective_m2

    compactness_bonus = 0.05 * max(0.0, compactness - 0.60)

    constraint_pressure_ratio = _clamp(
        mandatory_ratio
        + 0.06
        + compactness_bonus
        - 0.08 * obstacle_ratio
        - 0.05 * access_pressure,
        0.40,
        0.68,
    )

    return {
        "mandatory_base_footprint_m2": mandatory_base_m2,
        "mandatory_effective_footprint_m2": mandatory_effective_m2,
        "mandatory_item_count": float(mandatory_item_count),
        "mandatory_ratio": mandatory_ratio,
        "obstacle_ratio": obstacle_ratio,
        "access_pressure": access_pressure,
        "constraint_pressure_ratio": constraint_pressure_ratio,
    }


def _choose_budget_ratio_from_constraints(
    *,
    decision_details: list[dict[str, Any]],
    room_model: dict[str, Any],
    style: str,
    user_notes: str,
    rescue_mode: bool,
) -> tuple[float, dict[str, float | bool | str]]:
    room_pts = (room_model.get("room") or {}).get("polygon_ccw") or []
    room_area_m2 = _polygon_area_m2(room_pts) if room_pts else 0.0
    available_area_m2 = _available_area_m2(room_model) if room_pts else 0.0
    compactness = _compactness(room_pts) if room_pts else 0.0

    pressure = _summarize_constraint_pressure(
        decision_details=decision_details,
        available_area_m2=available_area_m2,
        room_area_m2=room_area_m2,
        compactness=compactness,
    )

    style_prior_ratio = _choose_budget_ratio_style_prior(style, user_notes)
    style_bias = 0.5 * (style_prior_ratio - 0.50)

    base_ratio = _clamp(
        max(
            float(pressure["constraint_pressure_ratio"]),
            float(pressure["mandatory_ratio"]) + 0.04,
        )
        + style_bias,
        0.40,
        0.68,
    )

    base_ratio_source = "constraint_pressure"

    if rescue_mode:
        rescue_floor = _clamp(float(pressure["mandatory_ratio"]) + 0.10, 0.50, 0.78)
        base_ratio = _clamp(max(base_ratio + 0.08, rescue_floor), 0.45, 0.78)
        base_ratio_source = "constraint_pressure_rescue"

    diagnostics: dict[str, float | bool | str] = {
        "style_prior_ratio": style_prior_ratio,
        "constraint_pressure_ratio": float(pressure["constraint_pressure_ratio"]),
        "mandatory_ratio": float(pressure["mandatory_ratio"]),
        "mandatory_base_footprint_m2": float(pressure["mandatory_base_footprint_m2"]),
        "mandatory_effective_footprint_m2": float(
            pressure["mandatory_effective_footprint_m2"]
        ),
        "mandatory_item_count": float(pressure["mandatory_item_count"]),
        "obstacle_ratio": float(pressure["obstacle_ratio"]),
        "access_pressure": float(pressure["access_pressure"]),
        "available_area_m2": available_area_m2,
        "gross_room_area_m2": room_area_m2,
        "base_ratio_source": base_ratio_source,
        "rescue_mode": rescue_mode,
    }
    return base_ratio, diagnostics


def _cluster_ratio_multiplier(cluster_id: str) -> float:
    k = _norm_key(cluster_id)

    if any(t in k for t in ("misc", "decor", "accent", "plant", "accessory")):
        return 0.75
    if any(t in k for t in ("storage", "wardrobe", "closet", "tv", "media", "console")):
        return 0.85
    if any(
        t in k
        for t in (
            "seating",
            "living",
            "lounge",
            "bed",
            "sleep",
            "kitchen",
            "cook",
            "prep",
            "dining",
            "study",
            "work",
        )
    ):
        return 1.10
    return 1.00


def _extract_zone_areas_m2(room_model: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    zones = room_model.get("zones", []) or []
    for zone in zones:
        zone_id = str(zone.get("id", "unknown_zone"))
        poly = zone.get("polygon_ccw", [])
        out[zone_id] = _polygon_area_m2(poly) if poly else 0.0
    return out


def _check_budget_limits(
    *,
    limits_m2: dict[str, float],
    footprints_m2: dict[str, float],
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for cid, fp in footprints_m2.items():
        limit = float(limits_m2.get(cid, 0.0))
        if fp > limit:
            violations.append(
                {
                    "cluster_id": cid,
                    "limit_m2": limit,
                    "footprint_m2": fp,
                    "over_by_m2": fp - limit,
                }
            )
    return violations


def _cluster_is_droppable(
    cluster_id: str, clusters_json: dict[str, Any] | None
) -> bool:
    if not isinstance(clusters_json, dict):
        return False

    clusters = clusters_json.get("clusters")
    if not isinstance(clusters, list):
        return False

    for c in clusters:
        if not isinstance(c, dict):
            continue
        if str(c.get("cluster_id", "")) != str(cluster_id):
            continue

        tag = _norm_key(str(c.get("tag", "")))
        rules = c.get("cluster_rules") or {}

        if tag in {"misc", "decor", "accent", "accessory"}:
            return True
        if isinstance(rules, dict) and rules.get("allow_empty_cluster") is True:
            return True
        return False

    return False


def _repair_priority(row: dict[str, Any], clusters_json: dict[str, Any] | None) -> str:
    priority = _norm_key(str(row.get("priority", "secondary")))
    intent = _norm_key(str(row.get("request_contract_intent", "")))

    if intent in _HARD_REQUEST_CONTRACT_INTENTS:
        return priority if priority in {"anchor", "primary"} else "primary"

    if intent in _SOFT_REQUEST_CONTRACT_INTENTS:
        if priority == "anchor" and _cluster_is_droppable(
            str(row.get("cluster_id", "")),
            clusters_json,
        ):
            return "primary"
        return "primary" if priority in {"optional", "secondary"} else priority

    if priority == "anchor" and _cluster_is_droppable(
        str(row.get("cluster_id", "")),
        clusters_json,
    ):
        return "optional"

    return priority


def _minimal_required_cluster_footprint_m2(
    decision_details: list[dict[str, Any]],
    *,
    clusters_json: dict[str, Any] | None,
) -> dict[str, float]:
    """
    Minimum footprint that should survive per cluster after reasonable trimming:
    - optional / secondary can go to 0
    - primary / anchor keep quantity 1 if originally > 0
    - use smallest available tier
    - droppable clusters may be reduced fully to 0
    """
    out: dict[str, float] = {}

    for row in decision_details:
        if not isinstance(row, dict):
            continue

        cid = str(row.get("cluster_id", ""))
        if not cid:
            continue

        if _cluster_is_droppable(cid, clusters_json):
            out.setdefault(cid, 0.0)
            continue

        priority = _norm_key(str(row.get("priority", "")))
        if priority not in {"anchor", "primary"}:
            out.setdefault(cid, 0.0)
            continue

        qty = max(0, int(row.get("quantity", 0)))
        if qty <= 0:
            out.setdefault(cid, 0.0)
            continue

        options = row.get("footprint_options_by_tier") or {}
        if not isinstance(options, dict):
            out.setdefault(cid, 0.0)
            continue

        smallest = _smallest_available_tier(options)
        if smallest is None:
            out.setdefault(cid, 0.0)
            continue

        opt = options.get(smallest)
        if not isinstance(opt, dict):
            out.setdefault(cid, 0.0)
            continue

        eff_per_item = float(opt.get("effective_footprint_m2_per_item", 0.0))
        out[cid] = out.get(cid, 0.0) + eff_per_item

    return out


def _rebalance_cluster_budget_limits(
    *,
    cluster_budget_limits_m2: dict[str, float],
    min_required_cluster_limits_m2: dict[str, float],
    raw_cluster_footprints_m2: dict[str, float],
    global_budget_m2: float,
) -> dict[str, float]:
    cluster_ids = sorted(
        set(cluster_budget_limits_m2.keys())
        | set(min_required_cluster_limits_m2.keys())
    )
    mins = {
        cid: max(0.0, float(min_required_cluster_limits_m2.get(cid, 0.0)))
        for cid in cluster_ids
    }

    total_min = sum(mins.values())
    if total_min > global_budget_m2 + 1e-9:
        return {cid: mins[cid] for cid in cluster_ids}

    limits = {cid: mins[cid] for cid in cluster_ids}
    remaining = max(0.0, global_budget_m2 - total_min)

    if remaining <= 1e-9:
        return limits

    demands: dict[str, float] = {}
    for cid in cluster_ids:
        raw = float(raw_cluster_footprints_m2.get(cid, 0.0))
        extra_need = max(0.0, raw - mins[cid])
        demands[cid] = extra_need

    demand_sum = sum(demands.values())
    if demand_sum <= 1e-9:
        return limits

    for cid in cluster_ids:
        limits[cid] += remaining * (demands[cid] / demand_sum)

    return limits


def _derive_cluster_budget_limits(
    *,
    cluster_ids: list[str],
    raw_cluster_footprints_m2: dict[str, float],
    min_required_cluster_limits_m2: dict[str, float],
    room_area_m2: float,
    base_ratio: float,
    shape_factor: float,
    cluster_area_mode: str,
    room_model: dict[str, Any],
    adaptive_cluster_ratios: bool,
    cluster_ratio_overrides: dict[str, float] | None,
) -> tuple[float, dict[str, float], dict[str, float]]:
    global_ratio = _clamp(base_ratio * shape_factor, 0.20, 0.75)
    global_budget_m2 = room_area_m2 * global_ratio

    cluster_ratios: dict[str, float] = {}
    for cid in cluster_ids:
        if cluster_ratio_overrides and cid in cluster_ratio_overrides:
            r = float(cluster_ratio_overrides[cid])
        else:
            mult = _cluster_ratio_multiplier(cid) if adaptive_cluster_ratios else 1.0
            r = _clamp(base_ratio * shape_factor * mult, 0.20, 0.75)
        cluster_ratios[cid] = r

    limits: dict[str, float] = {cid: 0.0 for cid in cluster_ids}

    zone_areas = (
        _extract_zone_areas_m2(room_model) if cluster_area_mode == "zones" else {}
    )
    zone_based_total = 0.0
    zone_backed_ids: set[str] = set()

    for cid in cluster_ids:
        zone_area = float(zone_areas.get(cid, 0.0))
        if zone_area > 0:
            limits[cid] = zone_area * cluster_ratios[cid]
            zone_based_total += limits[cid]
            zone_backed_ids.add(cid)

    remaining_ids = [cid for cid in cluster_ids if cid not in zone_backed_ids]
    remaining_budget = max(0.0, global_budget_m2 - zone_based_total)

    if remaining_ids:
        weights: dict[str, float] = {}
        for cid in remaining_ids:
            base_fp = float(raw_cluster_footprints_m2.get(cid, 0.0))
            base_fp = max(base_fp, 0.25)
            weights[cid] = base_fp * _cluster_ratio_multiplier(cid)

        weight_sum = sum(weights.values())
        if weight_sum <= 0:
            share = remaining_budget / max(len(remaining_ids), 1)
            for cid in remaining_ids:
                limits[cid] = share
        else:
            for cid in remaining_ids:
                limits[cid] = remaining_budget * (weights[cid] / weight_sum)

    total_limits = sum(limits.values())
    if total_limits > global_budget_m2 and total_limits > 0:
        scale = global_budget_m2 / total_limits
        for cid in list(limits.keys()):
            limits[cid] *= scale

    limits = _rebalance_cluster_budget_limits(
        cluster_budget_limits_m2=limits,
        min_required_cluster_limits_m2=min_required_cluster_limits_m2,
        raw_cluster_footprints_m2=raw_cluster_footprints_m2,
        global_budget_m2=global_budget_m2,
    )

    return global_budget_m2, cluster_ratios, limits


def _next_smaller_tier(tier: str) -> str | None:
    t = str(tier).upper()
    if t == "L":
        return "M"
    if t == "M":
        return "S"
    return None


def _recompute_recommended_row_totals(row: dict[str, Any]) -> None:
    tier = str(row["recommended_size_tier"]).upper()
    qty = max(0, int(row.get("recommended_quantity", 0)))

    options = row.get("footprint_options_by_tier") or {}
    opt = options.get(tier)
    if not isinstance(opt, dict):
        raise ValueError(
            f"Missing footprint option for tier={tier}, row={row.get('decision_id', '')}"
        )

    base_per_item = float(opt["base_footprint_m2_per_item"])
    eff_per_item = float(opt["effective_footprint_m2_per_item"])
    clearance_depth = float(opt["clearance_depth_m"])

    row["recommended_clearance_depth_m"] = clearance_depth
    row["recommended_base_footprint_m2_per_item"] = base_per_item
    row["recommended_effective_footprint_m2_per_item"] = eff_per_item
    row["recommended_base_footprint_m2_total"] = base_per_item * qty
    row["recommended_effective_footprint_m2_total"] = eff_per_item * qty


def _fit_decisions_to_budget(
    decision_details: list[dict[str, Any]],
    *,
    cluster_budget_limits_m2: dict[str, float],
    global_budget_m2: float,
    clusters_json: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float], float]:
    """
    Repair order:
    1. optional quantity
    2. secondary quantity
    3. optional size
    4. secondary size
    5. primary quantity
    6. primary size
    7. anchor quantity
    8. anchor size
    """

    fitted: list[dict[str, Any]] = []
    for d in decision_details:
        q = max(0, int(d.get("quantity", 0)))
        original_tier = str(d.get("size_tier", "M")).upper()

        row = dict(d)
        row["original_quantity"] = q
        row["recommended_quantity"] = q
        row["dropped_quantity"] = 0
        row["original_size_tier"] = original_tier
        row["recommended_size_tier"] = original_tier
        row["size_reduction_steps"] = 0

        _recompute_recommended_row_totals(row)
        fitted.append(row)

    def _cluster_totals() -> dict[str, float]:
        out: dict[str, float] = {}
        for row in fitted:
            cid = str(row["cluster_id"])
            out.setdefault(cid, 0.0)
            out[cid] += float(row["recommended_effective_footprint_m2_total"])
        return out

    def _global_total(cluster_totals: dict[str, float]) -> float:
        return sum(cluster_totals.values())

    def _cluster_anchor_total_qty(cluster_id: str) -> int:
        total = 0
        for row in fitted:
            if str(row["cluster_id"]) != cluster_id:
                continue
            if _norm_key(str(row.get("priority", ""))) != "anchor":
                continue
            total += int(row.get("recommended_quantity", 0))
        return total

    def _can_reduce_anchor_quantity(row: dict[str, Any]) -> bool:
        if _norm_key(str(row.get("priority", ""))) != "anchor":
            return True

        cid = str(row["cluster_id"])
        if _cluster_anchor_total_qty(cid) > 1:
            return True

        return _cluster_is_droppable(cid, clusters_json)

    def _quantity_reduction_candidate_score(
        row: dict[str, Any],
        cluster_overruns: dict[str, float],
    ) -> tuple[float, float, str]:
        cid = str(row["cluster_id"])
        over = float(cluster_overruns.get(cid, 0.0))
        per_item = float(row["recommended_effective_footprint_m2_per_item"])
        cat = str(row.get("category", ""))
        return (1.0 if over > 0 else 0.0, per_item, cat)

    def _size_saving_per_item(row: dict[str, Any]) -> float:
        current_tier = str(row["recommended_size_tier"]).upper()
        next_tier = _next_smaller_tier(current_tier)
        if next_tier is None:
            return 0.0

        options = row.get("footprint_options_by_tier") or {}
        cur = options.get(current_tier)
        nxt = options.get(next_tier)
        if not isinstance(cur, dict) or not isinstance(nxt, dict):
            return 0.0

        cur_eff = float(cur["effective_footprint_m2_per_item"])
        nxt_eff = float(nxt["effective_footprint_m2_per_item"])
        return max(0.0, cur_eff - nxt_eff)

    def _size_reduction_candidate_score(
        row: dict[str, Any],
        cluster_overruns: dict[str, float],
    ) -> tuple[float, float, int, str]:
        cid = str(row["cluster_id"])
        over = float(cluster_overruns.get(cid, 0.0))
        saving_per_item = _size_saving_per_item(row)
        qty = int(row.get("recommended_quantity", 0))
        cat = str(row.get("category", ""))
        return (1.0 if over > 0 else 0.0, saving_per_item, qty, cat)

    def _reduce_quantity_once(
        priority_names: set[str],
        *,
        cluster_id: str | None,
        cluster_overruns: dict[str, float],
    ) -> bool:
        candidates: list[dict[str, Any]] = []

        for row in fitted:
            if cluster_id is not None and str(row["cluster_id"]) != cluster_id:
                continue

            pri = _repair_priority(row, clusters_json)
            if pri not in priority_names:
                continue

            recommended_quantity = int(row.get("recommended_quantity", 0))
            min_quantity = max(0, _safe_int(row.get("min_quantity"), default=0))
            if recommended_quantity <= min_quantity:
                continue

            if pri == "anchor" and not _can_reduce_anchor_quantity(row):
                continue

            candidates.append(row)

        if not candidates:
            return False

        candidates.sort(
            key=lambda row: (
                -_quantity_reduction_candidate_score(row, cluster_overruns)[0],
                -_quantity_reduction_candidate_score(row, cluster_overruns)[1],
                _quantity_reduction_candidate_score(row, cluster_overruns)[2],
            )
        )

        row = candidates[0]
        min_quantity = max(0, _safe_int(row.get("min_quantity"), default=0))
        row["recommended_quantity"] = max(
            min_quantity,
            int(row["recommended_quantity"]) - 1,
        )
        row["dropped_quantity"] = int(row["original_quantity"]) - int(
            row["recommended_quantity"]
        )
        _recompute_recommended_row_totals(row)
        return True

    def _reduce_size_once(
        priority_names: set[str],
        *,
        cluster_id: str | None,
        cluster_overruns: dict[str, float],
    ) -> bool:
        candidates: list[dict[str, Any]] = []

        for row in fitted:
            if cluster_id is not None and str(row["cluster_id"]) != cluster_id:
                continue

            pri = _repair_priority(row, clusters_json)
            if pri not in priority_names:
                continue

            if int(row.get("recommended_quantity", 0)) <= 0:
                continue

            next_tier = _next_smaller_tier(str(row["recommended_size_tier"]).upper())
            if next_tier is None:
                continue

            saving_per_item = _size_saving_per_item(row)
            if saving_per_item <= 1e-9:
                continue

            candidates.append(row)

        if not candidates:
            return False

        candidates.sort(
            key=lambda row: (
                -_size_reduction_candidate_score(row, cluster_overruns)[0],
                -_size_reduction_candidate_score(row, cluster_overruns)[1],
                -_size_reduction_candidate_score(row, cluster_overruns)[2],
                _size_reduction_candidate_score(row, cluster_overruns)[3],
            )
        )

        row = candidates[0]
        next_tier = _next_smaller_tier(str(row["recommended_size_tier"]).upper())
        if next_tier is None:
            return False

        row["recommended_size_tier"] = next_tier
        row["size_reduction_steps"] = int(row.get("size_reduction_steps", 0)) + 1
        _recompute_recommended_row_totals(row)
        return True

    repair_steps: list[tuple[str, set[str]]] = [
        ("quantity", {"optional"}),
        ("quantity", {"secondary"}),
        ("size", {"optional"}),
        ("size", {"secondary"}),
        ("quantity", {"primary"}),
        ("size", {"primary"}),
        ("quantity", {"anchor"}),
        ("size", {"anchor"}),
    ]

    for action, priorities in repair_steps:
        while True:
            cluster_totals = _cluster_totals()
            total = _global_total(cluster_totals)

            cluster_overruns = {
                cid: cluster_totals.get(cid, 0.0)
                - float(cluster_budget_limits_m2.get(cid, 0.0))
                for cid in cluster_budget_limits_m2.keys()
            }

            if (
                not any(over > 1e-9 for over in cluster_overruns.values())
                and total <= global_budget_m2 + 1e-9
            ):
                return fitted, cluster_totals, total

            changed = False

            active_clusters = sorted(
                [cid for cid, over in cluster_overruns.items() if over > 1e-9],
                key=lambda cid: cluster_overruns[cid],
                reverse=True,
            )

            for cid in active_clusters:
                if action == "quantity":
                    changed = _reduce_quantity_once(
                        priorities,
                        cluster_id=cid,
                        cluster_overruns=cluster_overruns,
                    )
                else:
                    changed = _reduce_size_once(
                        priorities,
                        cluster_id=cid,
                        cluster_overruns=cluster_overruns,
                    )
                if changed:
                    break

            if changed:
                continue

            if total > global_budget_m2 + 1e-9:
                if action == "quantity":
                    changed = _reduce_quantity_once(
                        priorities,
                        cluster_id=None,
                        cluster_overruns=cluster_overruns,
                    )
                else:
                    changed = _reduce_size_once(
                        priorities,
                        cluster_id=None,
                        cluster_overruns=cluster_overruns,
                    )

            if not changed:
                break

    final_cluster_totals = _cluster_totals()
    final_global_total = _global_total(final_cluster_totals)
    return fitted, final_cluster_totals, final_global_total


def _build_recommended_decisions_from_details(
    original_decisions: list[dict[str, Any]],
    fitted_details: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}

    for row in fitted_details:
        if not isinstance(row, dict):
            continue
        key = (
            str(row.get("cluster_id", "")),
            str(row.get("category", "")),
        )
        by_key[key] = row

    out: list[dict[str, Any]] = []
    for d in original_decisions:
        if not isinstance(d, dict):
            continue

        cluster_id = str(d.get("cluster_id", ""))
        category = str(d.get("category") or d.get("object_type") or "")
        key = (cluster_id, category)

        row = by_key.get(key)
        if row is None:
            out.append(dict(d))
            continue

        nd = dict(d)
        if isinstance(row.get("recommended_quantity"), int):
            nd["quantity"] = int(row["recommended_quantity"])
        if isinstance(row.get("recommended_size_tier"), str):
            nd["size_tier"] = str(row["recommended_size_tier"]).upper()
        out.append(nd)

    return out


def _smallest_available_tier_in_row(row: dict[str, Any]) -> str:
    options = row.get("footprint_options_by_tier") or {}
    if isinstance(options, dict):
        for t in ("S", "M", "L"):
            if isinstance(options.get(t), dict):
                return t
    return str(row.get("recommended_size_tier") or row.get("size_tier") or "S").upper()


def _row_can_reduce_further(
    row: dict[str, Any],
    *,
    total_anchor_qty_in_cluster: int,
    cluster_is_droppable: bool,
    clusters_json: dict[str, Any] | None,
) -> bool:
    pri = _repair_priority(row, clusters_json)
    qty = max(0, int(row.get("recommended_quantity", 0)))
    cur_tier = str(
        row.get("recommended_size_tier") or row.get("size_tier") or "S"
    ).upper()
    min_tier = _smallest_available_tier_in_row(row)
    has_smaller_tier = cur_tier != min_tier and _next_smaller_tier(cur_tier) is not None

    if pri in {"optional", "secondary"}:
        return qty > max(0, _safe_int(row.get("min_quantity"), default=0))

    if pri == "primary":
        min_quantity = max(0, _safe_int(row.get("min_quantity"), default=0))
        return qty > min_quantity or has_smaller_tier

    if pri == "anchor":
        min_quantity = max(0, _safe_int(row.get("min_quantity"), default=0))
        if qty <= min_quantity:
            return False
        if cluster_is_droppable:
            return True
        if total_anchor_qty_in_cluster > 1:
            return True
        return has_smaller_tier

    return False


def _group_rows_by_cluster(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        cid = str(row.get("cluster_id", "unknown_cluster"))
        out.setdefault(cid, []).append(row)
    return out


def _cluster_has_live_anchor(
    rows: list[dict[str, Any]],
    *,
    qty_field: str,
) -> bool:
    for row in rows:
        if _norm_key(str(row.get("priority", ""))) != "anchor":
            continue
        try:
            if int(row.get(qty_field, 0)) > 0:
                return True
        except Exception:
            continue
    return False


def _remove_clusters_without_live_anchor(
    fitted_details: list[dict[str, Any]],
) -> list[str]:
    """
    If a cluster no longer has any surviving anchor after fit,
    zero out all remaining items in that cluster and treat it as removed.
    """
    removed: list[str] = []
    by_cluster = _group_rows_by_cluster(fitted_details)

    for cid, rows in by_cluster.items():
        if _cluster_has_live_anchor(rows, qty_field="recommended_quantity"):
            continue

        removed.append(cid)
        for row in rows:
            row["recommended_quantity"] = 0
            row["dropped_quantity"] = int(row.get("original_quantity", 0))
            _recompute_recommended_row_totals(row)

    return sorted(set(removed))


def _filter_float_map_keys(
    data: dict[str, float],
    keep_keys: set[str],
) -> dict[str, float]:
    return {k: float(v) for k, v in data.items() if k in keep_keys}


def _filter_nested_float_map_keys(
    data: dict[str, dict[str, float]],
    keep_keys: set[str],
) -> dict[str, dict[str, float]]:
    return {k: dict(v) for k, v in data.items() if k in keep_keys}


def _filter_violations_by_cluster_ids(
    violations: list[dict[str, Any]],
    keep_keys: set[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for v in violations:
        cid = str(v.get("cluster_id", ""))
        if cid in keep_keys:
            out.append(v)
    return out


def _summarize_post_fit_feasibility(
    *,
    fitted_details: list[dict[str, Any]],
    cluster_budget_limits_m2: dict[str, float],
    global_budget_m2: float,
    fitted_global_footprint: float,
    clusters_json: dict[str, Any] | None,
) -> dict[str, Any]:
    by_cluster = _group_rows_by_cluster(fitted_details)

    cluster_repair_status: dict[str, Any] = {}
    repair_blocked_clusters: list[str] = []
    core_minimum_infeasible_clusters: list[str] = []

    any_further_capacity_anywhere = False

    for cid, rows in by_cluster.items():
        cluster_is_droppable = _cluster_is_droppable(cid, clusters_json)
        limit = float(cluster_budget_limits_m2.get(cid, 0.0))
        footprint = sum(
            float(r.get("recommended_effective_footprint_m2_total", 0.0)) for r in rows
        )
        over_by = max(0.0, footprint - limit)

        total_anchor_qty = sum(
            max(0, int(r.get("recommended_quantity", 0)))
            for r in rows
            if _norm_key(str(r.get("priority", ""))) == "anchor"
        )

        positive_anchor_categories = [
            str(r.get("category", ""))
            for r in rows
            if _norm_key(str(r.get("priority", ""))) == "anchor"
            and int(r.get("recommended_quantity", 0)) > 0
        ]
        positive_primary_categories = [
            str(r.get("category", ""))
            for r in rows
            if _norm_key(str(r.get("priority", ""))) == "primary"
            and int(r.get("recommended_quantity", 0)) > 0
        ]
        positive_secondary_optional_categories = [
            str(r.get("category", ""))
            for r in rows
            if _norm_key(str(r.get("priority", ""))) in {"secondary", "optional"}
            and int(r.get("recommended_quantity", 0)) > 0
        ]

        further_capacity = any(
            _row_can_reduce_further(
                r,
                total_anchor_qty_in_cluster=total_anchor_qty,
                cluster_is_droppable=cluster_is_droppable,
                clusters_json=clusters_json,
            )
            for r in rows
        )

        if further_capacity:
            any_further_capacity_anywhere = True

        repair_blocked = over_by > 1e-9 and not further_capacity
        if repair_blocked:
            repair_blocked_clusters.append(cid)
            if not cluster_is_droppable:
                core_minimum_infeasible_clusters.append(cid)

        if repair_blocked:
            if cluster_is_droppable:
                reason = (
                    "droppable_cluster_still_over_limit_after_all_allowed_reductions"
                )
            else:
                reason = (
                    "core_cluster_minimum_surviving_set_still_exceeds_cluster_budget"
                )
        else:
            reason = "repair_possible_or_cluster_within_budget"

        cluster_repair_status[cid] = {
            "cluster_id": cid,
            "droppable": cluster_is_droppable,
            "limit_m2": limit,
            "footprint_m2": footprint,
            "over_by_m2": over_by,
            "total_anchor_quantity": total_anchor_qty,
            "positive_anchor_categories": positive_anchor_categories,
            "positive_primary_categories": positive_primary_categories,
            "positive_secondary_optional_categories": positive_secondary_optional_categories,
            "further_repair_capacity": further_capacity,
            "repair_blocked": repair_blocked,
            "reason": reason,
        }

    violating_cluster_ids = {
        cid
        for cid, info in cluster_repair_status.items()
        if float(info.get("over_by_m2", 0.0)) > 1e-9
    }
    blocked_cluster_ids = {
        cid
        for cid, info in cluster_repair_status.items()
        if bool(info.get("repair_blocked", False))
    }

    local_repairs_exhausted = bool(
        violating_cluster_ids
    ) and violating_cluster_ids.issubset(blocked_cluster_ids)

    global_over_by = max(0.0, fitted_global_footprint - global_budget_m2)
    global_repair_blocked = global_over_by > 1e-9 and not any_further_capacity_anywhere

    repair_exhausted = local_repairs_exhausted or global_repair_blocked

    return {
        "cluster_repair_status": cluster_repair_status,
        "repair_blocked_clusters": sorted(repair_blocked_clusters),
        "core_minimum_infeasible": len(core_minimum_infeasible_clusters) > 0,
        "core_minimum_infeasible_clusters": sorted(core_minimum_infeasible_clusters),
        "local_repairs_exhausted": local_repairs_exhausted,
        "global_repair_blocked": global_repair_blocked,
        "global_over_by_m2": global_over_by,
        "repair_exhausted": repair_exhausted,
    }


def estimate_budget(
    *,
    room_model: dict[str, Any],
    decisions: list[dict[str, Any]],
    size_profiles_by_category: dict[str, dict[str, Any]],
    style: str,
    user_notes: str,
    ratio_override: float | None = None,
    cluster_area_mode: str = "room",
    clusters_json: dict[str, Any] | None = None,
    adaptive_cluster_ratios: bool = True,
    cluster_ratio_overrides: dict[str, float] | None = None,
    frozen_cluster_budget_limits_m2: dict[str, float] | None = None,
    rescue_mode: bool = True,
) -> dict[str, Any]:
    """
    Budget estimator + repair recommender.

    Main changes:
    - access_clearance_ratio is FIXED to 0.25
    - base_ratio is chosen from constraint pressure
    - rescue_mode can temporarily relax ratio selection and unfreeze limits
    - per-cluster limits are rebalanced so minimum surviving CORE sets are not starved
    - if a cluster has no surviving anchor after fit, the whole cluster is zeroed and
      removed from cluster-level output maps
    """

    access_requirements = (
        _extract_access_requirements(clusters_json)
        if clusters_json
        else _AccessRequirements(frozenset(), frozenset())
    )

    normalized_profiles, generic_profile = _prepare_profile_lookup(
        size_profiles_by_category
    )

    decision_details: list[dict[str, Any]] = []
    missing_profile_categories: set[str] = set()

    for decision in decisions:
        try:
            detail = _estimate_decision_footprint_detail(
                decision,
                normalized_profiles,
                generic_profile,
                access_requirements=access_requirements,
            )
            if detail["profile_source"] == "generic_fallback":
                missing_profile_categories.add(str(decision.get("category", "unknown")))
            decision_details.append(detail)
        except ValueError as e:
            missing_profile_categories.add(str(decision.get("category", "unknown")))
            raise ValueError(f"{e}. decision={decision}") from e

    if ratio_override is not None:
        base_ratio = float(ratio_override)
        ratio_diagnostics: dict[str, float | bool | str] = {
            "style_prior_ratio": _choose_budget_ratio_style_prior(style, user_notes),
            "constraint_pressure_ratio": float("nan"),
            "mandatory_ratio": float("nan"),
            "mandatory_base_footprint_m2": float("nan"),
            "mandatory_effective_footprint_m2": float("nan"),
            "mandatory_item_count": float("nan"),
            "obstacle_ratio": float("nan"),
            "access_pressure": float("nan"),
            "available_area_m2": _available_area_m2(room_model),
            "gross_room_area_m2": _polygon_area_m2(
                (room_model.get("room") or {}).get("polygon_ccw") or []
            ),
            "base_ratio_source": "ratio_override",
            "rescue_mode": rescue_mode,
        }
    else:
        base_ratio, ratio_diagnostics = _choose_budget_ratio_from_constraints(
            decision_details=decision_details,
            room_model=room_model,
            style=style,
            user_notes=user_notes,
            rescue_mode=rescue_mode,
        )

    room_poly = (room_model.get("room") or {}).get("polygon_ccw") or []
    room_area = _available_area_m2(room_model) if room_poly else 0.0
    compactness = _compactness(room_poly) if room_poly else 0.0
    shape_factor = max(0.7, min(1.0, compactness / 0.6)) if compactness > 0 else 1.0
    ratio = _clamp(base_ratio * shape_factor, 0.20, 0.75)

    raw_cluster_footprints = _sum_footprint_by_cluster_m2(
        decision_details,
        field="effective_footprint_m2_total",
    )
    raw_breakdown = _sum_footprint_breakdown(
        decision_details,
        field="effective_footprint_m2_total",
    )

    cluster_ids = sorted(raw_cluster_footprints.keys())
    min_required_cluster_limits_m2 = _minimal_required_cluster_footprint_m2(
        decision_details,
        clusters_json=clusters_json,
    )

    use_frozen_limits = frozen_cluster_budget_limits_m2 is not None and not rescue_mode
    if use_frozen_limits:
        global_budget_m2 = room_area * ratio
        cluster_ratios = {cid: 0.0 for cid in cluster_ids}
        cluster_budget_limits_m2 = {
            cid: float(frozen_cluster_budget_limits_m2.get(cid, 0.0))
            for cid in cluster_ids
        }
    else:
        global_budget_m2, cluster_ratios, cluster_budget_limits_m2 = (
            _derive_cluster_budget_limits(
                cluster_ids=cluster_ids,
                raw_cluster_footprints_m2=raw_cluster_footprints,
                min_required_cluster_limits_m2=min_required_cluster_limits_m2,
                room_area_m2=room_area,
                base_ratio=base_ratio,
                shape_factor=shape_factor,
                cluster_area_mode=cluster_area_mode,
                room_model=room_model,
                adaptive_cluster_ratios=adaptive_cluster_ratios,
                cluster_ratio_overrides=cluster_ratio_overrides,
            )
        )

    violations_before_fit = _check_budget_limits(
        limits_m2=cluster_budget_limits_m2,
        footprints_m2=raw_cluster_footprints,
    )
    global_violation_before_fit = max(
        0.0, sum(raw_cluster_footprints.values()) - global_budget_m2
    )

    fitted_details, fitted_cluster_footprints, fitted_global_footprint = (
        _fit_decisions_to_budget(
            decision_details,
            cluster_budget_limits_m2=cluster_budget_limits_m2,
            global_budget_m2=global_budget_m2,
            clusters_json=clusters_json,
        )
    )

    removed_clusters_after_fit = _remove_clusters_without_live_anchor(fitted_details)
    if removed_clusters_after_fit:
        keep_cluster_ids_for_refit = sorted(
            set(cluster_ids) - set(removed_clusters_after_fit)
        )
        if keep_cluster_ids_for_refit:
            refit_details = [
                dict(row)
                for row in decision_details
                if str(row.get("cluster_id", "")) in keep_cluster_ids_for_refit
            ]
            removed_details = [
                dict(row)
                for row in fitted_details
                if str(row.get("cluster_id", "")) in set(removed_clusters_after_fit)
            ]
            for row in removed_details:
                row["recommended_quantity"] = 0
                row["dropped_quantity"] = int(row.get("original_quantity", 0))
                _recompute_recommended_row_totals(row)

            keep_raw_cluster_footprints = _sum_footprint_by_cluster_m2(
                refit_details,
                field="effective_footprint_m2_total",
            )
            keep_min_required_limits = _minimal_required_cluster_footprint_m2(
                refit_details,
                clusters_json=clusters_json,
            )
            _, keep_cluster_ratios, keep_cluster_budget_limits = (
                _derive_cluster_budget_limits(
                    cluster_ids=keep_cluster_ids_for_refit,
                    raw_cluster_footprints_m2=keep_raw_cluster_footprints,
                    min_required_cluster_limits_m2=keep_min_required_limits,
                    room_area_m2=room_area,
                    base_ratio=base_ratio,
                    shape_factor=shape_factor,
                    cluster_area_mode=cluster_area_mode,
                    room_model=room_model,
                    adaptive_cluster_ratios=adaptive_cluster_ratios,
                    cluster_ratio_overrides=cluster_ratio_overrides,
                )
            )
            refitted_keep_details, _, _ = _fit_decisions_to_budget(
                refit_details,
                cluster_budget_limits_m2=keep_cluster_budget_limits,
                global_budget_m2=global_budget_m2,
                clusters_json=clusters_json,
            )
            fitted_details = refitted_keep_details + removed_details
            for cid in keep_cluster_ids_for_refit:
                cluster_ratios[cid] = keep_cluster_ratios.get(cid, 0.0)
                cluster_budget_limits_m2[cid] = keep_cluster_budget_limits.get(
                    cid,
                    0.0,
                )
                min_required_cluster_limits_m2[cid] = keep_min_required_limits.get(
                    cid,
                    0.0,
                )

    fitted_cluster_footprints = _sum_footprint_by_cluster_m2(
        fitted_details,
        field="recommended_effective_footprint_m2_total",
    )
    fitted_global_footprint = sum(fitted_cluster_footprints.values())
    fitted_breakdown = _sum_footprint_breakdown(
        fitted_details,
        field="recommended_effective_footprint_m2_total",
    )

    violations_after_fit = _check_budget_limits(
        limits_m2=cluster_budget_limits_m2,
        footprints_m2=fitted_cluster_footprints,
    )
    global_violation_after_fit = max(0.0, fitted_global_footprint - global_budget_m2)

    recommended_decisions = _build_recommended_decisions_from_details(
        decisions,
        fitted_details,
    )

    post_fit = _summarize_post_fit_feasibility(
        fitted_details=fitted_details,
        cluster_budget_limits_m2=cluster_budget_limits_m2,
        global_budget_m2=global_budget_m2,
        fitted_global_footprint=fitted_global_footprint,
        clusters_json=clusters_json,
    )

    keep_cluster_ids = set(cluster_ids) - set(removed_clusters_after_fit)

    raw_cluster_footprints = _filter_float_map_keys(
        raw_cluster_footprints, keep_cluster_ids
    )
    raw_breakdown = _filter_nested_float_map_keys(raw_breakdown, keep_cluster_ids)
    fitted_cluster_footprints = _filter_float_map_keys(
        fitted_cluster_footprints, keep_cluster_ids
    )
    fitted_breakdown = _filter_nested_float_map_keys(fitted_breakdown, keep_cluster_ids)
    cluster_ratios = _filter_float_map_keys(cluster_ratios, keep_cluster_ids)
    cluster_budget_limits_m2 = _filter_float_map_keys(
        cluster_budget_limits_m2, keep_cluster_ids
    )
    min_required_cluster_limits_m2 = _filter_float_map_keys(
        min_required_cluster_limits_m2, keep_cluster_ids
    )
    violations_before_fit = _filter_violations_by_cluster_ids(
        violations_before_fit, keep_cluster_ids
    )
    violations_after_fit = _filter_violations_by_cluster_ids(
        violations_after_fit, keep_cluster_ids
    )
    post_fit["cluster_repair_status"] = {
        cid: info
        for cid, info in post_fit["cluster_repair_status"].items()
        if cid in keep_cluster_ids
    }
    post_fit["repair_blocked_clusters"] = [
        cid for cid in post_fit["repair_blocked_clusters"] if cid in keep_cluster_ids
    ]
    post_fit["core_minimum_infeasible_clusters"] = [
        cid
        for cid in post_fit["core_minimum_infeasible_clusters"]
        if cid in keep_cluster_ids
    ]

    return {
        "ratio": ratio,
        "base_ratio": base_ratio,
        "style_prior_ratio": ratio_diagnostics["style_prior_ratio"],
        "constraint_pressure_ratio": ratio_diagnostics["constraint_pressure_ratio"],
        "mandatory_ratio": ratio_diagnostics["mandatory_ratio"],
        "mandatory_base_footprint_m2": ratio_diagnostics["mandatory_base_footprint_m2"],
        "mandatory_effective_footprint_m2": ratio_diagnostics[
            "mandatory_effective_footprint_m2"
        ],
        "mandatory_item_count": ratio_diagnostics["mandatory_item_count"],
        "obstacle_ratio": ratio_diagnostics["obstacle_ratio"],
        "access_pressure": ratio_diagnostics["access_pressure"],
        "gross_room_area_m2": ratio_diagnostics["gross_room_area_m2"],
        "available_area_m2": ratio_diagnostics["available_area_m2"],
        "base_ratio_source": ratio_diagnostics["base_ratio_source"],
        "rescue_mode": ratio_diagnostics["rescue_mode"],
        "shape_compactness": compactness,
        "shape_factor": shape_factor,
        "cluster_area_mode": cluster_area_mode,
        "room_area_m2": room_area,
        "global_budget_m2": global_budget_m2,
        "global_requested_footprint_m2": sum(raw_cluster_footprints.values()),
        "global_recommended_footprint_m2": sum(fitted_cluster_footprints.values()),
        "cluster_ratios": cluster_ratios,
        "cluster_budget_limits_m2": cluster_budget_limits_m2,
        "min_required_cluster_limits_m2": min_required_cluster_limits_m2,
        "cluster_requested_footprints_m2": raw_cluster_footprints,
        "cluster_recommended_footprints_m2": fitted_cluster_footprints,
        "cluster_requested_footprints_by_priority_m2": raw_breakdown,
        "cluster_recommended_footprints_by_priority_m2": fitted_breakdown,
        "violations_before_fit": violations_before_fit,
        "violations_after_fit": violations_after_fit,
        "global_violation_before_fit_m2": global_violation_before_fit,
        "global_violation_after_fit_m2": global_violation_after_fit,
        "decision_footprint_details": fitted_details,
        "missing_profile_categories": sorted(missing_profile_categories),
        "access_required_object_ids": sorted(access_requirements.object_ids),
        "access_required_categories": sorted(access_requirements.categories),
        "access_clearance_ratio_fixed": FIXED_ACCESS_CLEARANCE_RATIO,
        "access_clearance_rule": (
            "effective = L*W + L*clamp(W*(0.25), min_depth, max_depth)*occupancy_multiplier "
            "for decisions requiring front_clearance"
        ),
        "adaptive_cluster_ratios": adaptive_cluster_ratios,
        "cluster_ratio_overrides": cluster_ratio_overrides or {},
        "input_decisions_fit": (
            len(violations_before_fit) == 0 and global_violation_before_fit <= 1e-9
        ),
        "recommended_decisions_fit": (
            len(violations_after_fit) == 0 and global_violation_after_fit <= 1e-9
        ),
        "repair_policy_applied": [
            "reduce_quantity_optional",
            "reduce_quantity_secondary",
            "reduce_size_optional",
            "reduce_size_secondary",
            "reduce_quantity_primary",
            "reduce_size_primary",
            "reduce_quantity_anchor",
            "reduce_size_anchor",
            "drop_cluster_if_no_anchor_survives",
        ],
        "cluster_budget_limits_source": (
            "frozen_override" if use_frozen_limits else "derived"
        ),
        "recommended_decisions": recommended_decisions,
        "removed_clusters_after_fit": removed_clusters_after_fit,
        "cluster_repair_status": post_fit["cluster_repair_status"],
        "repair_blocked_clusters": post_fit["repair_blocked_clusters"],
        "core_minimum_infeasible": post_fit["core_minimum_infeasible"],
        "core_minimum_infeasible_clusters": post_fit[
            "core_minimum_infeasible_clusters"
        ],
        "local_repairs_exhausted": post_fit["local_repairs_exhausted"],
        "global_repair_blocked": post_fit["global_repair_blocked"],
        "global_over_by_m2": post_fit["global_over_by_m2"],
        "repair_exhausted": post_fit["repair_exhausted"],
    }


# ============================================================
# Tool registry + schema
# ============================================================

TOOL_REGISTRY: dict[str, Any] = {
    "get_size_profiles": get_size_profiles,
    "estimate_budget": estimate_budget,
}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_size_profiles",
            "description": "Return representative dimensions (S/M/L) and thresholds for each category. Guarantees rep_dims_m[S/M/L] are non-null when category has inventory, and includes __generic__ fallback when possible.",
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
            "description": (
                "Estimate cluster footprints, choose budget ratio from constraint pressure, "
                "derive one global budget, rebalance per-cluster limits so mandatory surviving sets are not starved, "
                "and recommend repairs. Repair order: optional qty, secondary qty, optional size, secondary size, "
                "primary qty, primary size, anchor qty, anchor size. Uses fixed front-clearance ratio 0.25. "
                "If rescue_mode=true, frozen limits are ignored and the ratio can be relaxed."
            ),
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
                    "clusters_json": {"type": ["object", "null"]},
                    "adaptive_cluster_ratios": {"type": "boolean"},
                    "cluster_ratio_overrides": {
                        "type": ["object", "null"],
                        "additionalProperties": {"type": "number"},
                    },
                    "frozen_cluster_budget_limits_m2": {
                        "type": ["object", "null"],
                        "additionalProperties": {"type": "number"},
                    },
                    "rescue_mode": {"type": "boolean"},
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
