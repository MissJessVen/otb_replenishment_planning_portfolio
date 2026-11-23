"""Planner drivers.

Every number the model uses that is a *choice* rather than a *fact from the data*
lives here, so a logic change is a one-line, reviewable diff.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PlannerConfig:
    # --- data window -------------------------------------------------------
    first_month: str = "2009-12-01"
    last_full_month: str = "2011-11-01"     # Dec-2011 has only 9 days -> excluded

    # --- channel proxy -----------------------------------------------------
    domestic_country: str = "United Kingdom"  # UK = DTC proxy, rest = marketplace/FBA proxy

    # --- cleaning ------------------------------------------------------------
    bulk_min_units: int = 500        # a line is a "bulk one-off" if qty >= max(500, 20x SKU median)
    bulk_median_multiple: int = 20

    # --- economics -----------------------------------------------------------
    cost_pct: float = 0.45           # unit cost as % of selling price (no cost field in data)

    # --- replenishment -----------------------------------------------------
    lead_time_months: dict = field(default_factory=lambda: {"Domestic": 1, "International": 2})
    review_months: int = 1
    cycle_stock_cover: float = 0.5   # target EOM = safety stock + 0.5 x next month forecast
    mad_to_sigma: float = 1.25       # sigma ~= 1.25 x MAD for normal errors
    z_by_abc: dict = field(default_factory=lambda: {"A": 1.65, "B": 1.28, "C": 0.84})
    z_cap_fading: float = 1.28       # lower service once past peak
    ss_cap_months: float = 99.0      # 99 = off (v1.6); 1.0 = proposed v1.7
    abc_cuts: tuple = (0.70, 0.90)   # cumulative revenue share for A / B

    # --- top-down OTB --------------------------------------------------------
    mos_target: dict = field(default_factory=lambda: {"Domestic": 1.0, "International": 1.5})
    yoy_clip: tuple = (0.7, 1.4)

    # --- risk rules ------------------------------------------------------------
    overstock_wos: float = 26.0
    stockout_share_of_month1: float = 0.5
    weeks_per_month: float = 4.33

    # --- simulation of opening stock (dataset has no inventory) -----------------
    sim_seed: int = 7
    sim_cover_median: float = 1.4
    sim_cover_sigma: float = 0.6
    sim_thin_share: float = 0.07


V16 = PlannerConfig()
# Proposed v1.7: cap safety stock at 1 month of forecast, C-class service 80% -> 69%
V17 = PlannerConfig(ss_cap_months=1.0, z_by_abc={"A": 1.65, "B": 1.28, "C": 0.50})
