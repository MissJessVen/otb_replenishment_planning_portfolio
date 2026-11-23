"""Unit tests for the planning maths. Run: pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from otb_planner.data import PlanningBase, categorize                       # noqa: E402
from otb_planner.evaluate import replay                                     # noqa: E402
from otb_planner.forecast import forecast                                   # noqa: E402
from otb_planner.lifecycle import classify                                  # noqa: E402
from otb_planner.replenishment import assign_stage, safety_stock, time_phased_receipts  # noqa: E402


# ---------------------------------------------------------------- safety stock
def test_safety_stock_formula():
    # 1.65 * 100 * sqrt(1 + 1) = 233.3 -> rounds up
    assert safety_stock(1.65, 100, 1, 1)[()] == 234


def test_safety_stock_cap():
    assert safety_stock(1.65, 100, 1, 1, cap_units=150)[()] == 150


def test_exit_sku_has_zero_safety_stock():
    assert safety_stock(0.0, 500, 2, 1)[()] == 0


# ---------------------------------------------------------------- DRP netting
def test_receipts_net_against_on_hand_and_target():
    f = np.array([[100, 100, 100, 100]], float)
    rec, proj = time_phased_receipts(f, np.array([50.0]), np.array([20.0]), np.array([False]),
                                     np.array([1.0]), cycle_cover=0.5)
    # month 1: need = 100 + (20 + 50) - 50 = 120; then stock settles at 70 so each month needs 100
    assert rec[0].tolist() == [120, 100, 100]
    assert proj[0].tolist() == [70, 70, 70]


def test_case_pack_rounds_up():
    f = np.array([[10, 10, 10, 10]], float)
    rec, _ = time_phased_receipts(f, np.array([0.0]), np.array([0.0]), np.array([False]),
                                  np.array([12.0]), cycle_cover=0.0)
    assert rec[0, 0] == 12 and rec[0, 1] == 12  # 12 on hand -> 2 left -> need 8 -> 12


def test_exit_sku_gets_no_receipts():
    f = np.array([[100, 100, 100, 100]], float)
    rec, _ = time_phased_receipts(f, np.array([0.0]), np.array([50.0]), np.array([True]),
                                  np.array([1.0]), cycle_cover=0.5)
    assert rec.sum() == 0


# ---------------------------------------------------------------- lifecycle & stage
@pytest.mark.parametrize("ttp,tail,expected", [
    (5, 0.9, "Slow build"), (1, 0.7, "Evergreen"), (1, 0.1, "Spike & fade"), (2, 0.45, "Gradual fade")])
def test_archetype_rules(ttp, tail, expected):
    assert classify(ttp, tail) == expected


def test_stage_rules():
    s = assign_stage(age=[1, 20, 20, 20, 6, 20], base_rate=[5, 5, 5, 0, 5, 0],
                     trend=[1, 1.0, 0.4, 1, 1, 1], life_next3=[1, 1, 1, 1, 0.3, 1],
                     returner=[False, False, False, False, False, True])
    assert s.tolist() == ["New", "Core", "Exit", "Dormant", "Exit", "Seasonal return"]


def test_categorize():
    assert categorize("DIAMANTE NECKLACE BLACK") == "Jewelry & Accessories"
    assert categorize("PAPER CHAIN KIT 50'S CHRISTMAS") == "Christmas & Seasonal"
    assert categorize("WHITE HANGING HEART T-LIGHT HOLDER") == "Lighting & Candles"
    assert categorize("SOMETHING ELSE") == "Home Décor"


# ---------------------------------------------------------------- forecast v1.0
def _toy_base(units: np.ndarray) -> PlanningBase:
    n, T = units.shape
    months = pd.date_range("2010-01-01", periods=T, freq="MS")
    keys = pd.MultiIndex.from_tuples([(f"S{i}", "Domestic") for i in range(n)])
    return PlanningBase(months=months, keys=keys, units=units, capped=units, revenue=units * 2.0,
                        group=np.array(["Home Décor|Domestic"] * n), category=np.array(["Home Décor"] * n),
                        channel=np.array(["Domestic"] * n), first_idx=np.zeros(n, int),
                        sku=pd.DataFrame(index=[f"S{i}" for i in range(n)]))


def test_v10_is_three_month_average():
    u = np.array([[0, 0, 0, 30, 60, 90, 0, 0]], float)
    fc = forecast(_toy_base(u), t=5, horizon=2, version="v1.0")
    assert fc.values.tolist() == [[60.0, 60.0]]


# ---------------------------------------------------------------- post-mortem replay
def test_replay_fill_and_stockout():
    plan = pd.DataFrame({"BOH": [10.0], "ASP": [2.0], "UnitCost": [1.0]})
    r = replay(plan, receipts=np.array([[0, 5, 0]], float), demand=np.array([[8, 8, 8]], float))
    # sells 8, then 2+5=7, then 0 -> 15 of 24 units
    assert r["fill_units"] == pytest.approx(15 / 24)
    assert r["sku_stockouts"] == 1 and r["end_inv_cost"] == 0


# ---------------------------------------------------------------- integration (needs the CSV)
DATA = Path(__file__).resolve().parents[1] / "data" / "online_retail_II_combined.csv"


@pytest.mark.skipif(not DATA.exists(), reason="dataset not present")
def test_full_cycle_reconciles():
    from otb_planner import run_cycle
    res = run_cycle(str(DATA), run_backtest=False, verbose=False)
    br = res.bridge.set_index("step")["value"]
    # bridge steps add up exactly
    assert br.iloc[0] + br.iloc[1:4].sum() == pytest.approx(br.iloc[-1])
    # OTB bottom-up equals the SKU receipts
    assert res.otb["BottomUp_Cost"].sum() == pytest.approx(br.iloc[-1])
    # exit and dormant SKUs never get a PO
    assert res.plan.loc[res.plan["Stage"].isin(["Exit", "Dormant"]), "PO_units"].sum() == 0
