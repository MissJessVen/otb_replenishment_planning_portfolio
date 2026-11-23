"""One call runs a full monthly cycle: clean -> curves -> back-test -> forecast ->
buy -> OTB reconciliation -> risks -> what changed -> post-mortem."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import data as D
from .config import V16, V17, PlannerConfig
from .evaluate import actual_demand, replay
from .forecast import backtest, forecast
from .lifecycle import archetypes
from .otb import bridge, top_down_otb
from .replenishment import EXIT_STAGES, plan_cycle, recompute_buy, safety_stock, time_phased_receipts

# v1.6a's back-test is kept in the log even though it was rolled back
DECISIONS = {
    "v1.0": ("Flat 3-month moving average", "Reference"),
    "v1.1": ("Cap bulk one-off orders in the forecast base", "Accepted"),
    "v1.2": ("Category x channel seasonality from last year", "Accepted"),
    "v1.3": ("Launch curve for SKUs < 12 months; continued fade for declining SKUs", "Accepted"),
    "v1.4": ("50/50 blend with YoY-scaled same-month-last-year (mature SKUs)", "Accepted"),
    "v1.5": ("Pull SKU forecasts halfway to the top-down category plan", "Accepted"),
    "v1.6a": ("Seasonal-returner rule, all categories", "Rejected - rolled back"),
    "v1.6": ("Seasonal-returner rule, Christmas & Seasonal only", "Live"),
    "Seasonal naive": ("Same month last year (benchmark)", "Benchmark"),
}


@dataclass
class CycleResult:
    cfg: PlannerConfig
    sales: pd.DataFrame
    clean_log: pd.DataFrame
    base: D.PlanningBase
    seasonality: pd.DataFrame
    lifecycle: dict
    backtest: pd.DataFrame
    plan: pd.DataFrame
    otb: pd.DataFrame
    bridge: pd.DataFrame
    bridge_v17: pd.DataFrame
    postmortem: pd.DataFrame
    what_changed: pd.DataFrame
    origin: int


def _why(prior_m0: float, actual_m0: float, delta: float, stage: str) -> str:
    if prior_m0 == 0 and actual_m0 > 0:
        return "First sales last month: new item picked up"
    if delta > 0 and actual_m0 > 1.15 * prior_m0:
        return f"Sold {actual_m0:,.0f} vs {prior_m0:,.0f} expected: base rate raised"
    if delta < 0 and actual_m0 < 0.85 * prior_m0:
        return (f"Sold {actual_m0:,.0f} vs {prior_m0:,.0f} expected: base rate cut"
                + (", fade continued" if stage in ("Fading", "Exit") else ""))
    return f"Seasonal and trend factors re-estimated (sold {actual_m0:,.0f} vs {prior_m0:,.0f} expected)"


def what_changed(base: D.PlanningBase, plan: pd.DataFrame, t: int, version: str) -> pd.DataFrame:
    """Compare last month's run (origin t-1) with this run (origin t) on the two months
    both runs cover (t+1, t+2)."""
    prev = forecast(base, t - 1, 5, version).values
    now = forecast(base, t, 4, version).values
    idx = base.keys.get_indexer(pd.MultiIndex.from_arrays([plan["StockCode"], plan["Channel"]]))
    out = plan[["StockCode", "Channel", "Description", "Category", "Stage", "ASP"]].copy()
    out["Fcst_prior"] = prev[idx, 1:3].sum(1)
    out["Fcst_now"] = now[idx, 0:2].sum(1)
    out["LastMonth_expected"] = prev[idx, 0]
    out["LastMonth_actual"] = base.capped[idx, t]
    out["Delta_units"] = out["Fcst_now"] - out["Fcst_prior"]
    out["Delta_rev"] = out["Delta_units"] * out["ASP"]
    out["Why"] = [_why(a, b, d, s) for a, b, d, s in zip(out["LastMonth_expected"], out["LastMonth_actual"],
                                                         out["Delta_rev"], out["Stage"])]
    return out


def post_mortem(base: D.PlanningBase, plan: pd.DataFrame, t: int, cfg: PlannerConfig) -> pd.DataFrame:
    mn = plan.attrs["months"]
    demand = actual_demand(base, plan, t)
    idx = base.keys.get_indexer(pd.MultiIndex.from_arrays([plan["StockCode"], plan["Channel"]]))
    lt, cp, boh = plan["LeadTime"].to_numpy(), plan["CasePack"].to_numpy(float), plan["BOH"].to_numpy()
    exit_mask = plan["Stage"].isin(EXIT_STAGES).to_numpy()
    zabc = plan["ABC"].map(cfg.z_by_abc).to_numpy(float)
    sig = lambda f: cfg.mad_to_sigma * plan["ABC"].map(plan.attrs["mad"]).to_numpy() * f[:, :3].mean(1)

    # v1.0: flat average, no lifecycle stages -> no exit rule
    f10 = forecast(base, t, 4, "v1.0").values[idx]
    r10, _ = time_phased_receipts(f10, boh, safety_stock(zabc, sig(f10), lt, cfg.review_months),
                                  np.zeros(len(plan), bool), cp, cfg.cycle_stock_cover)
    # current plan
    r16 = plan[[f"Rcpt_{m}" for m in mn[:3]]].to_numpy()
    # perfect foresight: actual demand as the forecast (ceiling for this setup)
    fp = np.concatenate([demand, plan[[f"Fcst_{mn[3]}"]].to_numpy()], 1)
    rp, _ = time_phased_receipts(fp, boh, safety_stock(zabc, sig(fp), lt, cfg.review_months),
                                 exit_mask, cp, cfg.cycle_stock_cover)
    # proposed v1.7 drivers on the same forecast
    r17 = recompute_buy(plan, V17)
    rows = {
        "Planner v1.0 (flat 3-mo avg)": replay(plan, r10, demand),
        "Planner v1.6 (live)": replay(plan, r16, demand),
        "Proposed v1.7 (safety-stock cap)": replay(plan, r17, demand),
        "Perfect foresight (reference)": replay(plan, rp, demand),
    }
    return pd.DataFrame(rows).T


def run_cycle(csv_path: str, origin: str = "2011-08-01", version: str = "v1.6",
              cfg: PlannerConfig = V16, run_backtest: bool = True, verbose: bool = True) -> CycleResult:
    say = print if verbose else (lambda *a, **k: None)
    say("1/7 loading and cleaning ...")
    raw = D.load_raw(csv_path)
    sales, log = D.clean(raw, cfg)
    base = D.build_base(sales, cfg)
    t = base.month_index(origin)
    si = D.seasonality_index(base)

    say("2/7 launch-and-fade curves ...")
    life = archetypes(base, si["Total"].to_numpy())

    bt = pd.DataFrame()
    if run_backtest:
        say("3/7 back-testing every model version ...")
        origins = {f"{base.months[t - 3]:%b-%y} -> next 3m": t - 3, f"{base.months[t]:%b-%y} -> next 3m": t}
        bt = backtest(base, origins)
        bt["change"] = bt["version"].map(lambda v: DECISIONS[v][0])
        bt["decision"] = bt["version"].map(lambda v: DECISIONS[v][1])

    say(f"4/7 forecasting and building the buy ({version}) ...")
    plan = plan_cycle(base, sales, cfg, t, version)

    say("5/7 top-down OTB and reconciliation ...")
    otb = top_down_otb(plan, sales, cfg, base.months, t)
    br = bridge(plan, otb, cfg)
    r17 = recompute_buy(plan, V17)
    br17 = pd.DataFrame({"step": ["Top-down OTB", "Bottom-up SKU receipts (v1.7)"],
                         "value": [otb["OTB_Cost"].sum(), float((r17 * plan["UnitCost"].to_numpy()[:, None]).sum())]})

    say("6/7 what changed since last month ...")
    wc = what_changed(base, plan, t, version)

    say("7/7 post-mortem against actual sales ...")
    pm = post_mortem(base, plan, t, cfg)
    say("done.")
    return CycleResult(cfg, sales, log, base, si, life, bt, plan, otb, br, br17, pm, wc, t)
