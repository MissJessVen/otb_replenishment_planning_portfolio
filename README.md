# OTB & Replenishment Planning Portfolio (Python)

A SKU-level forecasting, replenishment and open-to-buy (OTB) planner for a high-SKU, short-lifecycle assortment. It runs one full monthly buy cycle end to end on ~1 million order lines and scores that buy against what actually sold.

Built as a work sample for a **Planning & Forecast Analyst** role (jewelry & accessories, DTC + Amazon).

| | |
|---|---|
| **Start here** | `notebooks/OTB_Replenishment_Portfolio.ipynb` (executed, with charts) or the `.html` export next to it |
| **Run a cycle** | `python run_cycle.py --origin 2011-08` |
| **Buy file** | `outputs/buy_recommendation_2011-08.xlsx` |
| **Tests** | `pytest -q` (15 tests: planning maths + an end-to-end reconciliation check) |

## Results for the 31-Aug-2011 cycle (Sep–Nov gifting peak)

| | |
|---|---|
| PO to approve this cycle | **$434k** at cost, 3,877 SKU × channel lines |
| Bottom-up SKU receipts vs top-down OTB (3 months) | $1.69M vs $1.40M (**+21%**) |
| Main driver of the gap | Safety stock + cycle stock: **+$452k** over the OTB months-of-supply target |
| Peak-season forecast bias | **+4.7%** (v1.6) vs **−30.8%** (v1.0 flat average) |
| Peak aggregate error (category × channel × month) | **18.3%** vs 32.8% |
| Post-mortem unit fill rate vs actual demand | **81.0%** (v1.6) vs 70.2% (v1.0) |
| Proposed v1.7 (safety-stock cap) | Receipts −$247k, Nov-end stock −$217k, fill −2.1 pts. Needs sign-off |

## What's in the package

```
src/otb_planner/
  config.py         every driver (lead times, service levels, cost %, risk thresholds); V16 live, V17 proposed
  data.py           load, clean (logged), categorise, SKU × channel × month matrices, seasonality index
  lifecycle.py      launch-and-fade archetypes + rebuy rules; pooled launch curve (no look-ahead)
  forecast.py       versioned forecast engine v1.0 → v1.6 and the back-test harness
  replenishment.py  lifecycle stage, ABC, safety stock, time-phased receipts (DRP), PO, risk flags
  otb.py            top-down OTB by category × channel × month, and the reconciliation bridge
  evaluate.py       post-mortem replay of a buy against actual demand
  pipeline.py       run_cycle(): one call for the whole monthly cycle, plus "what changed"
  reporting.py      charts, the Excel buy file, markdown summary
run_cycle.py        command-line entry point
tests/test_core.py  unit + integration tests
```

## Model logic (v1.6)

**Forecast**, per SKU × channel. Each step was added as its own version and back-tested at two origins.

| Ver. | Logic change | Decision |
|---|---|---|
| v1.0 | Flat 3-month moving average | Reference |
| v1.1 | Cap bulk one-off B2B orders in the base (kept in actuals) | Accepted |
| v1.2 | Category × channel seasonality from last year | Accepted |
| v1.3 | Launch curve for SKUs < 12 months; continued fade for declining SKUs | Accepted |
| v1.4 | 50/50 blend with YoY-scaled same-month-last-year (mature SKUs) | Accepted |
| v1.5 | Pull SKU forecasts halfway to the top-down category plan | Accepted |
| v1.6a | Seasonal-returner rule, all categories | **Rejected**: bias +8%, revived dead SKUs |
| v1.6 | Seasonal-returner rule, Christmas & Seasonal only | **Live** |

**Replenishment**
- Safety stock: `SS = z × σ × √(LT + R)`, with `σ = 1.25 × MAD(ABC) × avg forecast`.
  - z is 1.65 / 1.28 / 0.84 for A / B / C, capped at 1.28 for fading SKUs, and 0 for exit SKUs.
- Receipts: `receipt = roundup_to_case(max(0, F + SS + 0.5·F_next − projected on hand))`.
- A PO placed this cycle lands one lead time out: 1 month for Domestic, 2 months for International.

**OTB**
- `OTB = planned sales + planned EOM − BOM`, at cost.
  - Planned sales = last year's same month × YoY trend.
  - Planned EOM = next month's sales × months of supply (1.0 Domestic, 1.5 International).
- The bridge to the bottom-up buy splits the gap into: forecast difference, end-stock target difference, and netting/rounding.

## Data & assumptions

- **Source:** UCI Machine Learning Repository, *Online Retail II*: a UK online giftware seller, 1-Dec-2009 to 9-Dec-2011. Download it and save as `data/online_retail_II_combined.csv` (both sheets stacked, original columns). It is not included because of its size (~90 MB).
- **Channels.** The dataset has no channel, cost or inventory fields:
  - **Domestic** = UK orders, as the DTC proxy.
  - **International** = all other countries, as the marketplace / FBA proxy, with a longer lead time.
- **Cost and stock:**
  - Unit cost is 45% of selling price.
  - Opening stock on 1-Sep-2011 is **simulated**: lognormal cover around 1.4 months, with a fixed seed. Risk-flag counts depend on it; the logic does not.
- **Categories** are derived from description keywords (7 groups).
- **Currency:** values are in the source currency (GBP), shown with `$` formatting for readability.

## Setup

```bash
pip install -r requirements.txt
python run_cycle.py --data data/online_retail_II_combined.csv --origin 2011-08 --out outputs
pytest -q
```

To try the proposed v1.7 policy, use `otb_planner.V17`. It's a one-line driver change in `config.py`, and rolling back means reverting that line.
