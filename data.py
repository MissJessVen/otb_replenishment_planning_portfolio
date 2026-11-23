"""Load, clean and shape the Online Retail II order lines into a planning base.

Output of this module is everything downstream needs:
  * ``sales``  - cleaned order lines with Channel, Category, Month and a capped quantity
  * ``PlanningBase`` - SKU x channel x month matrices (units, capped units, revenue)
  * ``clean_log`` - an auditable record of every exclusion
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import PlannerConfig

# Ordered keyword rules: the first match wins. Built from the highest-revenue words in
# the product descriptions; anything unmatched falls to Home Decor.
CATEGORY_RULES: list[tuple[str, str]] = [
    ("Jewelry & Accessories", r"NECKLACE|EARRING|BRACELET|\bRING\b|CHARM|DIAMANTE|BEAD|BROOCH|HAIR|PURSE|SCARF|UMBRELLA|PARASOL|HANDBAG|KEYRING|KEY RING|BANGLE|PENDANT|JEWEL|CRYSTAL|SLIDE|COMB|MIRROR"),
    ("Christmas & Seasonal", r"CHRISTMAS|XMAS|ADVENT|SANTA|REINDEER|SNOW|EASTER|HALLOWEEN|BAUBLE|NATIVITY|ANGEL|STAR TREE|TREE DECORATION|STOCKING"),
    ("Lighting & Candles", r"LIGHT|CANDLE|LANTERN|LAMP|T-LIGHT|TLIGHT|NIGHTLIGHT"),
    ("Bags & Storage", r"\bBAG|SHOPPER|STORAGE|\bBOX|BOXES|\bTINS?\b|BASKET|\bCASES?\b|DRAWER|CABINET|HAMPER"),
    ("Kitchen & Dining", r"CAKE|MUG|\bCUPS?\b|\bTEA|PLATE|BOWL|CUTLERY|NAPKIN|\bJAR|BAKING|APRON|LUNCH|SNACK|COFFEE|SAUCER|TEACUP|KITCHEN|PANTRY|SPOON|JUG|TRAY|DINER|PICNIC|BAKE|EGG|MOULD|TOWEL"),
    ("Party, Stationery & Gifts", r"CARD|PAPER|WRAP|PARTY|BUNTING|PENCIL|NOTEBOOK|STICKER|GIFT|BIRTHDAY|RIBBON|CHAIN KIT|CRAYON|ERASER|BALLOON|CONFETTI|SPACEBOY|GAME|PLAYING|DOLLY|DOLL|CRAFT|KIT"),
]
DEFAULT_CATEGORY = "Home Décor"
MERCH_CODE = r"^\d{5}[A-Za-z]{0,2}$"   # 5 digits + optional colour/size suffix


def categorize(description: str) -> str:
    for name, pattern in CATEGORY_RULES:
        if re.search(pattern, description):
            return name
    return DEFAULT_CATEGORY


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"StockCode": str, "Invoice": str})
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


def clean(raw: pd.DataFrame, cfg: PlannerConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (sales, clean_log)."""
    log: list[dict] = []

    def step(name: str, df: pd.DataFrame, note: str = "") -> None:
        log.append({"step": name, "rows": len(df), "note": note})

    step("Raw rows", raw)
    d = raw.drop_duplicates()
    step("Drop exact duplicate lines", d, f"{len(raw) - len(d):,} duplicates")
    d = d[d["StockCode"].str.match(MERCH_CODE)]
    step("Keep merchandise SKUs only", d, "removed postage, fees, manual adjustments, gift vouchers, samples")
    d = d[d["Price"] > 0]
    step("Drop zero/negative price lines", d, "write-offs, damages, free samples")

    is_cancel = d["Invoice"].str.startswith("C") | (d["Quantity"] < 0)
    cancels = d[is_cancel]
    sales = d[~is_cancel & (d["Quantity"] > 0)].copy()

    # Net out sales fully reversed by a cancellation (same customer, SKU, |qty|).
    key = ["Customer ID", "StockCode"]
    c = cancels.dropna(subset=["Customer ID"]).assign(q=lambda x: -x["Quantity"])
    n_cancel = c.groupby(key + ["q"]).size().rename("ncanc").reset_index()
    sales["q"] = sales["Quantity"]
    sales = sales.merge(n_cancel, on=key + ["q"], how="left")
    sales["ncanc"] = sales["ncanc"].fillna(0)
    sales = sales.sort_values("InvoiceDate")
    sales["rk"] = sales.groupby(key + ["q"]).cumcount(ascending=False)  # latest sale first
    reversed_ = sales["rk"] < sales["ncanc"]
    note = f"{int(reversed_.sum()):,} lines / {int(sales.loc[reversed_, 'Quantity'].sum()):,} units netted out"
    sales = sales[~reversed_].drop(columns=["q", "ncanc", "rk"])
    step("Remove sales fully reversed by a cancellation", sales, note + " (incl. the 80,995 and 74,215-unit test orders)")

    # Bulk one-off flag: kept in actuals, capped in the forecast base.
    med = sales.groupby("StockCode")["Quantity"].transform("median")
    cap = np.maximum(cfg.bulk_min_units, cfg.bulk_median_multiple * med)
    sales["bulk"] = sales["Quantity"] >= cap
    sales["Qcap"] = np.where(sales["bulk"], np.minimum(cap, sales["Quantity"]), sales["Quantity"])
    step("Flag bulk one-off lines (kept in actuals, capped in forecast base)", sales,
         f"{int(sales['bulk'].sum()):,} lines / {int(sales.loc[sales['bulk'], 'Quantity'].sum()):,} units flagged")

    end = pd.Timestamp(cfg.last_full_month) + pd.offsets.MonthBegin(1)
    sales = sales[sales["InvoiceDate"] < end]
    step("Trim partial month (1-9 Dec 2011)", sales, "24 complete months: Dec 2009 - Nov 2011")

    sales["Rev"] = sales["Quantity"] * sales["Price"]
    sales["Month"] = sales["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    sales["Channel"] = np.where(sales["Country"] == cfg.domestic_country, "Domestic", "International")
    desc = (sales.dropna(subset=["Description"]).groupby("StockCode")["Description"]
            .agg(lambda s: s.value_counts().index[0]).str.strip())
    sales["Description"] = sales["StockCode"].map(desc)
    sales["Category"] = sales["StockCode"].map(desc.apply(categorize)).fillna(DEFAULT_CATEGORY)
    return sales.reset_index(drop=True), pd.DataFrame(log)


@dataclass
class PlanningBase:
    """SKU x channel planning units as aligned numpy matrices (rows = units, cols = months)."""
    months: pd.DatetimeIndex
    keys: pd.MultiIndex           # (StockCode, Channel)
    units: np.ndarray             # actual units
    capped: np.ndarray            # units with bulk one-offs capped (forecast base)
    revenue: np.ndarray
    group: np.ndarray             # "Category|Channel" per row
    category: np.ndarray
    channel: np.ndarray
    first_idx: np.ndarray         # month index of the SKU's first sale (any channel)
    sku: pd.DataFrame             # SKU attributes

    @property
    def group_capped(self) -> pd.DataFrame:
        """Category x channel aggregate of capped units (rows = group, cols = month index)."""
        return pd.DataFrame(self.capped).groupby(self.group).sum()

    def month_index(self, month: str) -> int:
        return self.months.get_loc(pd.Timestamp(month))


def build_base(sales: pd.DataFrame, cfg: PlannerConfig) -> PlanningBase:
    months = pd.date_range(cfg.first_month, cfg.last_full_month, freq="MS")
    cube = (sales.groupby(["StockCode", "Channel", "Month"])
            .agg(U=("Quantity", "sum"), Ucap=("Qcap", "sum"), Rev=("Rev", "sum")).reset_index())

    def wide(col: str) -> pd.DataFrame:
        return (cube.pivot_table(index=["StockCode", "Channel"], columns="Month", values=col, aggfunc="sum")
                .reindex(columns=months).fillna(0))

    U, C, R = wide("U"), wide("Ucap"), wide("Rev")
    sku = (sales.groupby("StockCode")
           .agg(Description=("Description", "first"), Category=("Category", "first"),
                first=("Month", "min"), last=("Month", "max"), U=("Quantity", "sum"),
                Rev=("Rev", "sum"), nmonths=("Month", "nunique")))
    sku["ASP"] = sku["Rev"] / sku["U"]
    codes = C.index.get_level_values(0)
    cat = sku.loc[codes, "Category"].to_numpy()
    ch = C.index.get_level_values(1).to_numpy()
    first_idx = sku.loc[codes, "first"].map(months.get_loc).to_numpy()
    return PlanningBase(months=months, keys=C.index, units=U.to_numpy(), capped=C.to_numpy(),
                        revenue=R.to_numpy(), group=np.array([f"{a}|{b}" for a, b in zip(cat, ch)]),
                        category=cat, channel=ch, first_idx=first_idx, sku=sku)


def seasonality_index(base: PlanningBase, min_months: int = 22) -> pd.DataFrame:
    """Seasonal index from 'core' SKUs sold in >= min_months of 24, so assortment growth
    doesn't masquerade as seasonality."""
    core = base.sku.index[base.sku["nmonths"] >= min_months]
    rows = np.isin(base.keys.get_level_values(0), core)
    out = {}
    for ch in ("Domestic", "International"):
        m = base.capped[rows & (base.channel == ch)].sum(0)
        out[ch] = m / m.mean()
    tot = base.capped[rows].sum(0)
    out["Total"] = tot / tot.mean()
    return pd.DataFrame(out, index=base.months).round(3)
