"""Launch-and-fade curves.

Two uses:
  * ``archetypes`` - descriptive: classify new SKUs by curve shape and derive rebuy rules.
  * ``launch_curve`` - predictive: the pooled new-SKU curve the forecast uses, estimated
    only from data available at the forecast origin (no look-ahead).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import PlanningBase

ARCHETYPES = ["Spike & fade", "Gradual fade", "Slow build", "Evergreen"]

REBUY_RULES = {
    "Spike & fade": "Buy launch qty for ~3 months; no rebuy after M2 unless sell-through beats plan",
    "Gradual fade": "Rebuy through M3-M4 with shrinking cover; stop when trailing 2m < 50% of peak",
    "Slow build": "Smaller initial buy; rebuy on evidence; full safety stock only after M4",
    "Evergreen": "Treat as core from M6; standard ABC service level",
}


def _sku_matrix(base: PlanningBase) -> pd.DataFrame:
    """Capped units by SKU (channels summed) x month."""
    return pd.DataFrame(base.capped, index=base.keys.get_level_values(0)).groupby(level=0).sum()


def classify(time_to_peak: int, tail_ratio: float) -> str:
    """Rule-based archetype from months-to-peak and months 5-6 level vs peak."""
    if time_to_peak >= 4:
        return "Slow build"
    if tail_ratio >= 0.6:
        return "Evergreen"
    if tail_ratio < 0.3:
        return "Spike & fade"
    return "Gradual fade"


def archetypes(base: PlanningBase, season_total: np.ndarray, launch_from="2010-03-01",
               launch_to="2011-05-01", min_units=300) -> dict:
    """Classify SKUs launched in [launch_from, launch_to] with >= min_units lifetime units.

    Curves are deseasonalised with the core-SKU index, smoothed (2-month mean), and
    normalised to each SKU's own peak.
    """
    cs = _sku_matrix(base)
    sku = base.sku
    new = sku.index[(sku["first"] >= launch_from) & (sku["first"] <= launch_to) & (sku["U"] >= min_units)]
    first = cs.gt(0).to_numpy().argmax(1)
    pos = {c: i for i, c in enumerate(cs.index)}
    rows = {}
    for code in new:
        k = first[pos[code]]
        units = cs.iloc[pos[code], k:k + 12].to_numpy() / season_total[k:k + 12]
        rows[code] = np.pad(units, (0, 12 - len(units)), constant_values=np.nan)
    lc = pd.DataFrame(rows, index=[f"m{i}" for i in range(12)]).T
    obs = lc.iloc[:, :7].dropna()
    smooth = obs.T.rolling(2, min_periods=1).mean().T
    peak = smooth.max(axis=1)
    norm = smooth.div(peak, axis=0)
    ttp = smooth.to_numpy().argmax(1)
    tail = norm[["m5", "m6"]].mean(axis=1).to_numpy()
    label = pd.Series([classify(t, r) for t, r in zip(ttp, tail)], index=obs.index, name="Archetype")

    full = lc.loc[obs.index].T.rolling(2, min_periods=1).mean().T
    stats = {}
    for a in ARCHETYPES:
        ids = label.index[label == a]
        below = []
        for code in ids:
            r = full.loc[code].to_numpy()
            pi = int(np.nanargmax(r))
            hit = np.where(r[pi:] < 0.3 * r[pi])[0]
            below.append(pi + hit[0] if len(hit) else np.nan)
        below = pd.Series(below, dtype=float)
        first3 = lc.loc[ids].iloc[:, :3].sum(axis=1) / lc.loc[ids].sum(axis=1)
        stats[a] = {
            "skus": len(ids),
            "median_peak_month": float(np.median(ttp[[obs.index.get_loc(i) for i in ids]])),
            "share_units_first3": float(first3.median()),
            "median_month_below_30pct": float(below.median()),
            "share_never_below_30pct": float(below.isna().mean()),
            "rebuy_rule": REBUY_RULES[a],
        }
    cat_mix = pd.crosstab(sku.loc[label.index, "Category"], label, normalize="index")[ARCHETYPES]
    return {
        "labels": label,
        "curves_peak": pd.DataFrame({a: norm.loc[label == a].median() for a in ARCHETYPES}),
        "stats": pd.DataFrame(stats).T,
        "category_mix": cat_mix,
    }


def launch_curve(base: PlanningBase, t: int, min_units: int = 300) -> np.ndarray:
    """Median 12-month new-SKU curve, deseasonalised and indexed to the first-3-month
    average, using only months <= t. SKUs need >= 7 observed months before t."""
    cs = _sku_matrix(base).to_numpy()
    first = (cs > 0).argmax(1)
    tot = cs.sum(0)
    si = tot / tot.mean()
    ok = (first >= 3) & (first + 6 <= t) & (cs[:, :t + 1].sum(1) >= min_units)
    rows = []
    for i in np.where(ok)[0]:
        k = first[i]
        n = min(12, t + 1 - k)
        r = np.full(12, np.nan)
        r[:n] = cs[i, k:k + n] / si[k:k + n]
        rows.append(r / np.nanmean(r[:3]))
    curve = np.nanmedian(np.array(rows), 0)
    for j in range(1, 12):                       # carry forward any unobserved tail
        if np.isnan(curve[j]):
            curve[j] = curve[j - 1]
    return curve
