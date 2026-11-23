"""Top-down open-to-buy and its reconciliation to the bottom-up SKU buy.

OTB_m (at cost) = planned sales_m + planned EOM stock_m - BOM stock_m
  planned sales_m  = last year same month x YoY trend (last 3 months TY vs LY)
  planned EOM_m    = planned sales_{m+1} x months-of-supply target (by channel)
  BOM_{m+1}        = planned EOM_m
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PlannerConfig
from .replenishment import EXIT_STAGES


def top_down_otb(plan: pd.DataFrame, sales: pd.DataFrame, cfg: PlannerConfig,
                 months: pd.DatetimeIndex, t: int) -> pd.DataFrame:
    mn = plan.attrs["months"]
    rev = sales.groupby(["Category", "Channel", "Month"])["Rev"].sum()
    get = lambda c, h, m: float(rev.get((c, h, m), 0.0))
    ly_month = lambda h: months[t + 1 + h - 12]
    rows = []
    for (cat, ch), g in plan.groupby(["Category", "Channel"]):
        ty3 = sum(get(cat, ch, months[t - k]) for k in range(3))
        ly3 = sum(get(cat, ch, months[t - 12 - k]) for k in range(3))
        yoy = float(np.clip(ty3 / ly3 if ly3 > 0 else 1.0, *cfg.yoy_clip))
        planned = [get(cat, ch, ly_month(h)) * yoy for h in range(4)]
        bom = float((g["BOH"] * g["UnitCost"]).sum())
        for h in range(3):
            sales_cost = planned[h] * cfg.cost_pct
            eom = planned[h + 1] * cfg.cost_pct * cfg.mos_target[ch]
            otb = sales_cost + eom - bom
            bottom_up = float((g[f"Rcpt_{mn[h]}"] * g["UnitCost"]).sum())
            rows.append({
                "Category": cat, "Channel": ch, "Month": mn[h], "YoY": yoy,
                "PlanSales_Retail": planned[h],
                "SKU_Fcst_Retail": float((g[f"Fcst_{mn[h]}"] * g["ASP"]).sum()),
                "PlanSales_Cost": sales_cost, "BOM_Cost": bom, "EOM_Target_Cost": eom,
                "OTB_Cost": otb, "BottomUp_Cost": bottom_up, "Variance": bottom_up - otb,
            })
            bom = eom
    return pd.DataFrame(rows)


def bridge(plan: pd.DataFrame, otb: pd.DataFrame, cfg: PlannerConfig, receipts: np.ndarray | None = None) -> pd.DataFrame:
    """Explain bottom-up minus top-down (3 months, at cost) in four additive steps.

    Because OTB telescopes to  sum(sales) + EOM_last - BOM_first, the gap splits into:
    forecast difference + end-stock-target difference + netting/rounding residual.
    """
    mn = plan.attrs["months"]
    cost = plan["UnitCost"].to_numpy()
    if receipts is None:
        receipts = plan[[f"Rcpt_{m}" for m in mn[:3]]].to_numpy()
    bu = float((receipts * cost[:, None]).sum())
    total_otb = otb["OTB_Cost"].sum()
    fc_cost = float((plan[[f"Fcst_{m}" for m in mn[:3]]].to_numpy() * cost[:, None]).sum())
    live = ~plan["Stage"].isin(EXIT_STAGES).to_numpy()
    eom_bu = float(((plan["SafetyStock"] + cfg.cycle_stock_cover * plan[f"Fcst_{mn[3]}"]) * plan["UnitCost"])[live].sum())
    eom_td = otb.loc[otb["Month"] == mn[2], "EOM_Target_Cost"].sum()
    d_fc = fc_cost - otb["PlanSales_Cost"].sum()
    d_eom = eom_bu - eom_td
    steps = [
        ("Top-down OTB", total_otb),
        ("SKU forecast vs plan sales", d_fc),
        ("SKU end-stock target vs plan EOM", d_eom),
        ("Unnettable stock & case-pack rounding", bu - (total_otb + d_fc + d_eom)),
        ("Bottom-up SKU receipts", bu),
    ]
    return pd.DataFrame(steps, columns=["step", "value"])
