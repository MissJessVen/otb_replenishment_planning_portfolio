"""Post-mortem: replay a buy against what actually sold."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import PlanningBase


def actual_demand(base: PlanningBase, plan: pd.DataFrame, t: int, months: int = 3) -> np.ndarray:
    idx = base.keys.get_indexer(pd.MultiIndex.from_arrays([plan["StockCode"], plan["Channel"]]))
    return base.capped[idx, t + 1:t + 1 + months]


def replay(plan: pd.DataFrame, receipts: np.ndarray, demand: np.ndarray) -> dict:
    """Sell from opening stock + planned receipts, month by month, no re-plan."""
    oh = plan["BOH"].to_numpy(float).copy()
    asp, cost = plan["ASP"].to_numpy(), plan["UnitCost"].to_numpy()
    sold = np.zeros_like(demand, dtype=float)
    stocked_out = np.zeros(len(oh), bool)
    for m in range(demand.shape[1]):
        avail = oh + receipts[:, m]
        sold[:, m] = np.minimum(avail, demand[:, m])
        stocked_out |= (demand[:, m] > avail) & (demand[:, m] > 0)
        oh = avail - sold[:, m]
    dem_rev = (demand * asp[:, None]).sum()
    sold_rev = (sold * asp[:, None]).sum()
    return {
        "fill_units": sold.sum() / demand.sum(),
        "fill_rev": sold_rev / dem_rev,
        "lost_rev": dem_rev - sold_rev,
        "sku_stockouts": int(stocked_out.sum()),
        "receipts_cost": float((receipts * cost[:, None]).sum()),
        "end_inv_cost": float((oh * cost).sum()),
    }
