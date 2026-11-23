"""From forecast to purchase order: lifecycle stage, ABC, safety stock, time-phased
receipts (DRP), this cycle's PO and risk flags."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PlannerConfig
from .data import PlanningBase
from .forecast import Forecast, forecast

EXIT_STAGES = ("Exit", "Dormant")
FADING_STAGES = ("Fading", "Post-launch fade")


# ------------------------------------------------------------------ building blocks
def safety_stock(z, sigma, lead_time, review, cap_units=None):
    """SS = z * sigma * sqrt(LT + R), optionally capped, rounded up to whole units."""
    ss = np.asarray(z) * np.asarray(sigma) * np.sqrt(np.asarray(lead_time) + review)
    if cap_units is not None:
        ss = np.minimum(ss, cap_units)
    return np.ceil(ss)


def time_phased_receipts(fcst: np.ndarray, on_hand: np.ndarray, ss: np.ndarray,
                         exit_mask: np.ndarray, case_pack: np.ndarray,
                         cycle_cover: float, months: int = 3):
    """Classic DRP netting. ``fcst`` needs months+1 columns (next month sets the target).

    receipt_m = roundup_to_case(max(0, F_m + target_EOM_m - projected_on_hand_m))
    target_EOM_m = SS + cycle_cover * F_{m+1};   exit SKUs: no target, no receipts.
    """
    n = len(fcst)
    rec, proj = np.zeros((n, months)), np.zeros((n, months))
    oh = on_hand.astype(float).copy()
    for m in range(months):
        target = np.where(exit_mask, 0.0, ss + cycle_cover * fcst[:, m + 1])
        need = np.where(exit_mask, 0.0, np.maximum(0.0, fcst[:, m] + target - oh))
        rec[:, m] = np.ceil(need / case_pack) * case_pack
        oh = oh + rec[:, m] - fcst[:, m]
        proj[:, m] = oh
    return rec, proj


def assign_stage(age, base_rate, trend, life_next3, returner) -> np.ndarray:
    out = []
    for a, b, tr, lf, r in zip(age, base_rate, trend, life_next3, returner):
        if r:
            s = "Seasonal return"
        elif b == 0:
            s = "Dormant"
        elif a < 3:
            s = "New"
        elif a < 12:
            s = "Exit" if lf < 0.5 else "Post-launch fade"
        else:
            s = "Core" if tr >= 0.85 else ("Fading" if tr >= 0.5 else "Exit")
        out.append(s)
    return np.array(out)


def assign_abc(df: pd.DataFrame, cuts=(0.7, 0.9)) -> pd.Series:
    abc = pd.Series("C", index=df.index)
    for _, g in df.groupby("Channel"):
        g = g.sort_values("Rev_L12M", ascending=False)
        cum = g["Rev_L12M"].cumsum() / g["Rev_L12M"].sum()
        abc[g.index[cum <= cuts[0]]] = "A"
        abc[g.index[(cum > cuts[0]) & (cum <= cuts[1])]] = "B"
    return abc


def simulate_opening_stock(base_rate, avg12, stage, cfg: PlannerConfig) -> np.ndarray:
    """The dataset has no inventory, so opening stock is simulated: lognormal months of
    cover around the recent run rate, a thin-stock tail, and extra stock on exit SKUs so
    the markdown logic has something to act on."""
    rng = np.random.default_rng(cfg.sim_seed)
    n = len(base_rate)
    cover = np.exp(rng.normal(np.log(cfg.sim_cover_median), cfg.sim_cover_sigma, n))
    thin = rng.random(n) < cfg.sim_thin_share
    cover[thin] *= 0.15
    stale = avg12 * np.exp(rng.normal(np.log(2.5), 0.5, n))
    boh = np.where(base_rate > 0, base_rate * cover, stale)
    boh = np.where(stage == "Exit", np.maximum(boh, base_rate * 2.5), boh)
    return np.round(boh)


def mad_by_abc(base: PlanningBase, abc: pd.Series, universe: np.ndarray, t_bt: int,
               version: str) -> dict:
    """Forecast error (MAD as % of actual) by ABC class from an earlier origin."""
    fc = forecast(base, t_bt, 3, version).values
    act = base.capped[:, t_bt + 1:t_bt + 4]
    out = {}
    for c in "ABC":
        m = (abc.to_numpy() == c) & universe & (base.first_idx <= t_bt)
        out[c] = float(np.abs(fc[m] - act[m]).sum() / max(act[m].sum(), 1))
    return out


# ------------------------------------------------------------------ the cycle
def plan_cycle(base: PlanningBase, sales: pd.DataFrame, cfg: PlannerConfig, t: int,
               version: str = "v1.6", fc: Forecast | None = None,
               mad: dict | None = None, opening_stock: np.ndarray | None = None) -> pd.DataFrame:
    """Run one replenishment cycle at origin ``t`` (last month of actuals).

    Returns one row per active SKU x channel with every intermediate number, so any line
    can be explained ("why is the model saying this?").
    """
    H = 4
    fc = fc or forecast(base, t, H, version)
    F, parts = fc.values, fc.parts
    mn = [base.months[t + 1 + h].strftime("%b") if t + 1 + h < len(base.months)
          else (base.months[-1] + pd.DateOffset(months=t + 1 + h - len(base.months) + 1)).strftime("%b")
          for h in range(H)]

    df = base.keys.to_frame(index=False)
    df["Description"] = base.sku.loc[df["StockCode"], "Description"].to_numpy()
    df["Category"] = base.category
    l12 = slice(t - 11, t + 1)
    u12, r12 = base.units[:, l12].sum(1), base.revenue[:, l12].sum(1)
    df["Units_L12M"], df["Rev_L12M"] = u12, r12
    df["ASP"] = np.where(u12 > 0, r12 / np.where(u12 > 0, u12, 1), base.sku.loc[df["StockCode"], "ASP"].to_numpy())
    df["UnitCost"] = df["ASP"] * cfg.cost_pct
    age = parts["age"]
    df["FirstSale"] = base.sku.loc[df["StockCode"], "first"].dt.strftime("%b-%y").to_numpy()
    df["AgeMonths"] = age
    for k in range(3):
        df[f"Act_{base.months[t - 2 + k]:%b}"] = base.capped[:, t - 2 + k]
    df["Base_MA3"] = base.capped[:, t - 2:t + 1].mean(1)
    df["Season_m1"] = parts["season"][:, 0]
    df["Season_m3"] = parts["season"][:, 2]
    df["Trend"] = parts["trend"]
    df["LifeFactor"] = parts["life"][:, :3].mean(1)
    df["Stage"] = assign_stage(age, df["Base_MA3"].to_numpy(), parts["trend"],
                               parts["life"][:, :3].mean(1), parts["returner"])
    for h in range(H):
        df[f"Fcst_{mn[h]}"] = F[:, h]

    universe = (u12 > 0) & (age >= 0)
    df["ABC"] = "C"
    df.loc[universe, "ABC"] = assign_abc(df[universe], cfg.abc_cuts)
    mad = mad or mad_by_abc(base, df["ABC"], universe, t - 3, version)

    lt = df["Channel"].map(cfg.lead_time_months).to_numpy()
    z = df["ABC"].map(cfg.z_by_abc).to_numpy(float)
    z = np.where(df["Stage"].isin(FADING_STAGES), np.minimum(z, cfg.z_cap_fading), z)
    z = np.where(df["Stage"].isin(EXIT_STAGES), 0.0, z)
    f_avg = F[:, :3].mean(1)
    sigma = cfg.mad_to_sigma * df["ABC"].map(mad).to_numpy() * f_avg
    ss = safety_stock(z, sigma, lt, cfg.review_months, cfg.ss_cap_months * f_avg)
    df["LeadTime"], df["z"], df["Sigma"], df["SafetyStock"] = lt, z, sigma, ss

    case = sales.groupby("StockCode")["Quantity"].agg(lambda s: s.mode().iat[0]).clip(1, 48)
    df["CasePack"] = df["StockCode"].map(case).fillna(1).astype(int)

    if opening_stock is None:
        opening_stock = simulate_opening_stock(df["Base_MA3"].to_numpy(), u12 / 12, df["Stage"].to_numpy(), cfg)
    df["BOH"] = np.where(universe, opening_stock, 0)

    exit_mask = df["Stage"].isin(EXIT_STAGES).to_numpy()
    rec, proj = time_phased_receipts(F, df["BOH"].to_numpy(), ss, exit_mask,
                                     df["CasePack"].to_numpy(float), cfg.cycle_stock_cover)
    for m in range(3):
        df[f"Rcpt_{mn[m]}"] = rec[:, m]
        df[f"EOH_{mn[m]}"] = proj[:, m]
    # A PO placed this cycle lands one lead time out: month 2 for Domestic, month 3 for International.
    df["PO_units"] = np.where(lt == 1, rec[:, 1], rec[:, 2])
    df["PO_receipt"] = np.where(lt == 1, mn[1], mn[2])
    df["PO_cost"] = df["PO_units"] * df["UnitCost"]

    wos = np.where(f_avg > 0, df["BOH"] / (f_avg / cfg.weeks_per_month), np.where(df["BOH"] > 0, 999, 0))
    df["WOS"] = wos
    df["RiskFlag"] = risk_flags(df, F[:, 0], f_avg, cfg)
    out = df[universe].reset_index(drop=True)
    out.attrs.update(months=mn, mad=mad, version=version, origin=str(base.months[t].date()))
    return out


def risk_flags(df: pd.DataFrame, f_m1: np.ndarray, f_avg: np.ndarray, cfg: PlannerConfig) -> list[str]:
    boh, cost, stage, b = df["BOH"].to_numpy(), df["UnitCost"].to_numpy(), df["Stage"].to_numpy(), df["Base_MA3"].to_numpy()
    flags = []
    for i in range(len(df)):
        f = []
        live = stage[i] not in EXIT_STAGES
        if f_avg[i] > 0 and boh[i] < cfg.stockout_share_of_month1 * f_m1[i] and live:
            f.append("Stockout risk")
        if df["WOS"].iat[i] > cfg.overstock_wos and boh[i] * cost[i] > 50:
            f.append("Overstock")
        if not live and boh[i] > 0:
            f.append("EOL / markdown")
        if b[i] == 0 and boh[i] > 0 and stage[i] != "Seasonal return":
            f.append("Slow mover")
        flags.append(", ".join(f))
    return flags


def recompute_buy(plan: pd.DataFrame, cfg: PlannerConfig) -> np.ndarray:
    """Re-run safety stock + DRP on an existing plan under different drivers (scenario
    analysis, e.g. v1.7) without touching the forecast. Returns receipts (n, 3)."""
    mn = plan.attrs["months"]
    F = plan[[f"Fcst_{m}" for m in mn]].to_numpy()
    f_avg = F[:, :3].mean(1)
    z = plan["ABC"].map(cfg.z_by_abc).to_numpy(float)
    z = np.where(plan["Stage"].isin(FADING_STAGES), np.minimum(z, cfg.z_cap_fading), z)
    z = np.where(plan["Stage"].isin(EXIT_STAGES), 0.0, z)
    sigma = cfg.mad_to_sigma * plan["ABC"].map(plan.attrs["mad"]).to_numpy() * f_avg
    ss = safety_stock(z, sigma, plan["LeadTime"].to_numpy(), cfg.review_months, cfg.ss_cap_months * f_avg)
    rec, _ = time_phased_receipts(F, plan["BOH"].to_numpy(), ss, plan["Stage"].isin(EXIT_STAGES).to_numpy(),
                                  plan["CasePack"].to_numpy(float), cfg.cycle_stock_cover)
    return rec
