"""Versioned SKU x channel forecast engine.

Each version adds exactly one logic change on top of the previous one, so a back-test
isolates the effect of that change:

  v1.0  flat 3-month moving average of actual units
  v1.1  + cap bulk one-off orders in the base
  v1.2  + category x channel seasonality from last year
  v1.3  + lifecycle: launch curve for SKUs < 12 months, continued fade for declining SKUs
  v1.4  + 50/50 blend with YoY-scaled seasonal naive for mature SKUs
  v1.5  + partial (50%) reconciliation to a top-down category x channel plan
  v1.6a + seasonal-returner rule, all categories            (rejected)
  v1.6  + seasonal-returner rule, Christmas & Seasonal only (live)

``t`` is always the index of the last month of actuals (the forecast origin).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import PlanningBase
from .lifecycle import launch_curve

VERSIONS = ["v1.0", "v1.1", "v1.2", "v1.3", "v1.4", "v1.5", "v1.6a", "v1.6"]
_ORDER = {v: i for i, v in enumerate(VERSIONS)}


def _at_least(ver: str, ref: str) -> bool:
    """True if ``ver`` includes the logic introduced in ``ref``.
    v1.6a and v1.6 are siblings: both include v1.5, each adds its own returner rule."""
    if ref in ("v1.6", "v1.6a"):
        return ver == ref
    return _ORDER[ver] >= _ORDER[ref]


def _safe_div(a, b, fill=1.0):
    b = np.asarray(b, dtype=float)
    return np.where(b > 0, np.asarray(a, dtype=float) / np.where(b > 0, b, 1.0), fill)


@dataclass
class Forecast:
    values: np.ndarray                  # (n_units, H)
    version: str
    origin: int
    parts: dict = field(default_factory=dict)   # intermediate factors, for explanations


def forecast(base: PlanningBase, t: int, horizon: int, version: str = "v1.6") -> Forecast:
    if version not in VERSIONS:
        raise ValueError(f"unknown version {version}; choose from {VERSIONS}")
    X = base.units if version == "v1.0" else base.capped
    C = base.capped
    H = horizon
    age = t - base.first_idx
    parts: dict = {"age": age}

    base_rate = X[:, t - 2:t + 1].mean(1)
    F = np.repeat(base_rate[:, None], H, 1)
    parts["base"] = base_rate
    if not _at_least(version, "v1.2"):
        return Forecast(F, version, t, parts)

    # v1.2 seasonality: last year's category x channel ratio of each target month to the
    # same 3 base months, i.e. "how much did this group lift from Jun-Aug to Sep last year"
    G = base.group_capped.loc[base.group].to_numpy()
    ly_base = G[:, t - 14:t - 11].mean(1)
    season = np.stack([_safe_div(G[:, t + h - 12], ly_base) for h in range(1, H + 1)], 1)
    season = np.clip(season, 0.4, 3.0)
    F = F * season
    parts["season"] = season
    if not _at_least(version, "v1.3"):
        return Forecast(F, version, t, parts)

    # v1.3 lifecycle
    curve = launch_curve(base, t)
    life = np.ones_like(F)
    F[age < 0] = 0.0                                   # not launched yet at origin
    young = (age >= 0) & (age < 12)
    for i in np.where(young)[0]:
        a = age[i]
        k = min(3, a + 1)
        level = curve[a - k + 1:a + 1].mean()
        obs = C[i, t - k + 1:t + 1].mean()
        for h in range(1, H + 1):
            life[i, h - 1] = curve[min(a + h, 11)] / max(level, 1e-6)
            F[i, h - 1] = obs * life[i, h - 1] * season[i, h - 1]
    # deseasonalised trend: SKU last3/prior3 relative to its group's last3/prior3
    prior, last = C[:, t - 5:t - 2].sum(1), C[:, t - 2:t + 1].sum(1)
    g_prior, g_last = G[:, t - 5:t - 2].sum(1), G[:, t - 2:t + 1].sum(1)
    ok = (prior > 0) & (g_prior > 0)
    trend = np.where(ok, _safe_div(last, prior) / _safe_div(g_last, g_prior), 1.0)
    trend = np.clip(trend, 0.3, 1.5)
    mature = age >= 12
    for h in range(1, H + 1):
        fade = np.where(trend < 0.85, trend ** (h / 3), 1.0)
        F[mature, h - 1] *= fade[mature]
        life[mature, h - 1] = fade[mature]
    parts.update(curve=curve, life=life, trend=trend)
    if not _at_least(version, "v1.4"):
        return Forecast(F, version, t, parts)

    # v1.4 blend with YoY-scaled seasonal naive (mature SKUs with last-year history)
    ly = C[:, t + 1 - 12:t + 1 - 12 + H]
    ly_b3 = C[:, t - 14:t - 11].sum(1)
    yoy = np.clip(_safe_div(last, ly_b3, np.nan), 0.5, 1.5)
    has = mature & (ly_b3 > 0) & (ly.sum(1) > 0)
    F[has] = 0.5 * F[has] + 0.5 * (ly[has] * yoy[has, None])
    parts.update(blend=has, yoy=yoy)
    if not _at_least(version, "v1.5"):
        return Forecast(F, version, t, parts)

    # v1.6 / v1.6a seasonal returner (applied before top-down reconciliation)
    if version in ("v1.6", "v1.6a"):
        g_yoy = np.clip(_safe_div(G[:, t - 2:t + 1].sum(1), G[:, t - 14:t - 11].sum(1)), 0.6, 1.6)
        ly_h = C[:, t + 1 - 12:t + 1 - 12 + H]
        ly_b = C[:, t - 14:t - 11].mean(1)
        live = C[:, t - 11:t + 1].sum(1) > 0
        peak_ly = ly_h[:, :min(H, 3)].mean(1)
        ret = live & (age >= 12) & (peak_ly > 0) & (ly_b < 0.25 * peak_ly)
        if version == "v1.6":
            ret &= np.char.startswith(base.group.astype(str), "Christmas")
        F[ret] = np.maximum(F[ret], (ly_h * g_yoy[:, None])[ret])
        parts["returner"] = ret
    else:
        parts["returner"] = np.zeros(len(F), bool)

    # v1.5 partial reconciliation to top-down (group LY same month x group YoY)
    g_yoy = np.clip(_safe_div(G[:, t - 2:t + 1].sum(1), G[:, t - 14:t - 11].sum(1)), 0.6, 1.6)
    top_down = G[:, t + 1 - 12:t + 1 - 12 + H] * g_yoy[:, None]
    exists = base.first_idx <= t
    group_sum = pd.DataFrame(F * exists[:, None]).groupby(base.group).transform("sum").to_numpy()
    scale = np.clip(_safe_div(top_down, group_sum), 0.6, 1.6)
    F = F * (0.5 + 0.5 * scale)
    parts["td_scale"] = scale
    return Forecast(F, version, t, parts)


# ----------------------------------------------------------------------------- back-test
def score(base: PlanningBase, fc: Forecast, mask: np.ndarray | None = None) -> dict:
    """SKU WMAPE, bias, and aggregate WMAPE at category x channel x month, over planning
    units that existed at the origin (new launches in the horizon can't be forecast)."""
    t, H = fc.origin, fc.values.shape[1]
    A = base.capped[:, t + 1:t + 1 + H]
    m = base.first_idx <= t
    if mask is not None:
        m &= mask
    F, A, g = fc.values[m], A[m], base.group[m]
    agg_f = pd.DataFrame(F).groupby(g).sum().to_numpy()
    agg_a = pd.DataFrame(A).groupby(g).sum().to_numpy()
    return {
        "sku_wmape": np.abs(F - A).sum() / A.sum(),
        "bias": (F.sum() - A.sum()) / A.sum(),
        "agg_wmape": np.abs(agg_f - agg_a).sum() / agg_a.sum(),
    }


def backtest(base: PlanningBase, origins: dict[str, int], horizon: int = 3,
             versions: list[str] | None = None) -> pd.DataFrame:
    rows = []
    for label, t in origins.items():
        for v in versions or VERSIONS:
            rows.append({"origin": label, "version": v, **score(base, forecast(base, t, horizon, v))})
        naive = Forecast(base.capped[:, t + 1 - 12:t + 1 - 12 + horizon], "Seasonal naive", t)
        rows.append({"origin": label, "version": "Seasonal naive", **score(base, naive)})
    return pd.DataFrame(rows)
