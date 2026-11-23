"""Charts, the Excel buy file and a one-page markdown summary."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

from .lifecycle import ARCHETYPES

INK, MUTED, GRID = "#16231F", "#5B6B66", "#DCE3E0"
TEAL, GOLD, BLUE, ROSE, RED, GREEN = "#0E6B5C", "#C08A2B", "#5E7FB8", "#B3577A", "#B3392F", "#2E7D4F"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.titlecolor": INK, "axes.titlelocation": "left", "figure.dpi": 110,
})


def money(v, _=None):
    a = abs(v)
    s = f"${a/1e6:.1f}M" if a >= 1e6 else f"${a/1e3:.0f}k" if a >= 1e3 else f"${a:.0f}"
    return ("-" if v < 0 else "") + s


# ------------------------------------------------------------------ charts
def chart_monthly_revenue(sales: pd.DataFrame):
    m = sales.groupby(["Month", "Channel"])["Rev"].sum().unstack()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(m))
    ax.bar(x, m["Domestic"], color=TEAL, label="Domestic (UK)")
    ax.bar(x, m["International"], bottom=m["Domestic"], color=GOLD, label="International")
    ax.set_xticks(x[::3], [d.strftime("%b %y") for d in m.index[::3]])
    ax.yaxis.set_major_formatter(FuncFormatter(money))
    ax.set_title("Monthly revenue by channel: Q4 gifting peak in both years")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def chart_q4_lift(sales: pd.DataFrame):
    s = sales.assign(mo=sales["Month"].dt.month, yr=sales["Month"].dt.year)
    s = s[s["yr"] == 2010]
    peak = s[s["mo"].isin([9, 10, 11])].groupby(["Category", "Channel"])["Quantity"].sum() / 3
    summer = s[s["mo"].isin([6, 7, 8])].groupby(["Category", "Channel"])["Quantity"].sum() / 3
    lift = (peak / summer).unstack().sort_values("Domestic")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    y = np.arange(len(lift))
    ax.hlines(y, lift["Domestic"], lift["International"], color=GRID, lw=3)
    ax.scatter(lift["Domestic"], y, color=TEAL, s=55, label="Domestic", zorder=3)
    ax.scatter(lift["International"], y, color=GOLD, s=55, label="International", zorder=3)
    ax.set_yticks(y, lift.index)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}x"))
    ax.set_title("Q4 lift (Sep-Nov vs Jun-Aug units, 2010) differs by channel")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return fig, lift


def chart_lifecycle(curves: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for a, c in zip(ARCHETYPES, [ROSE, GOLD, BLUE, TEAL]):
        ax.plot(range(len(curves)), curves[a], marker="o", color=c, lw=2.2, label=a)
    ax.set_xticks(range(len(curves)), [f"M{i}" for i in range(len(curves))])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Units, index to own peak")
    ax.set_title("Launch-and-fade curves by archetype (deseasonalised)")
    ax.legend(frameon=False, ncol=2, loc="lower left")
    fig.tight_layout()
    return fig


def chart_backtest(bt: pd.DataFrame, origin_label: str):
    d = bt[(bt["origin"] == origin_label) & ~bt["version"].isin(["Seasonal naive", "v1.6a"])]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    x = np.arange(len(d))
    ax.bar(x - 0.18, d["agg_wmape"], 0.34, color=TEAL, label="Aggregate WMAPE (cat x channel x month)")
    ax.bar(x + 0.18, d["bias"], 0.34, color=GOLD, label="Bias")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(x, d["version"])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title(f"Peak-season back-test, origin {origin_label}")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def chart_bridge(br: pd.DataFrame):
    labels = ["Top-down\nOTB", "SKU fcst\nvs plan", "SS + cycle\nstock", "Unnettable\n& rounding", "Bottom-up\nSKU buy"]
    v = br["value"].to_numpy()
    lo, hi, col = [], [], []
    run = 0.0
    for i, x in enumerate(v):
        if i in (0, len(v) - 1):
            lo.append(0); hi.append(x); col.append(TEAL); run = x
        else:
            lo.append(min(run, run + x)); hi.append(max(run, run + x)); col.append(RED if x > 0 else GREEN); run += x
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(range(len(v)), np.array(hi) - np.array(lo), bottom=lo, color=col, width=0.6)
    for i, x in enumerate(v):
        txt = money(x) if i in (0, len(v) - 1) else ("+" if x > 0 else "") + money(x)
        ax.text(i, hi[i], txt, ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(range(len(v)), labels)
    ax.yaxis.set_major_formatter(FuncFormatter(money))
    ax.set_title("Bridge: top-down OTB to bottom-up SKU receipts (3 months, at cost)")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def chart_otb_by_month(otb: pd.DataFrame, months: list[str]):
    g = otb.groupby(["Channel", "Month"])[["OTB_Cost", "BottomUp_Cost"]].sum()
    order = [(c, m) for c in ("Domestic", "International") for m in months[:3]]
    g = g.loc[order]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(g))
    ax.bar(x - 0.18, g["OTB_Cost"], 0.34, color=BLUE, label="Top-down OTB")
    ax.bar(x + 0.18, g["BottomUp_Cost"], 0.34, color=TEAL, label="Bottom-up SKU receipts")
    ax.set_xticks(x, [f"{m}\n{c}" for c, m in g.index])
    ax.yaxis.set_major_formatter(FuncFormatter(money))
    ax.set_title("OTB vs SKU receipts by channel and month")
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def save_all_charts(res, out_dir: str | Path) -> list[Path]:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    peak_label = res.backtest["origin"].unique()[-1] if len(res.backtest) else None
    figs = {
        "01_monthly_revenue.png": chart_monthly_revenue(res.sales),
        "02_q4_lift_by_channel.png": chart_q4_lift(res.sales)[0],
        "03_launch_fade_curves.png": chart_lifecycle(res.lifecycle["curves_peak"]),
        "05_otb_bridge.png": chart_bridge(res.bridge),
        "06_otb_by_month.png": chart_otb_by_month(res.otb, res.plan.attrs["months"]),
    }
    if peak_label:
        figs["04_backtest_peak.png"] = chart_backtest(res.backtest, peak_label)
    paths = []
    for name, fig in figs.items():
        p = out / name
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    return paths


# ------------------------------------------------------------------ Excel
def write_excel(res, path: str | Path) -> Path:
    """The buy file merchants review: one sheet per question they ask."""
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    plan, mn = res.plan, res.plan.attrs["months"]
    cols = (["StockCode", "Channel", "Description", "Category", "Stage", "ABC", "AgeMonths",
             *[c for c in plan.columns if c.startswith("Act_")], "Base_MA3", "Trend",
             *[f"Fcst_{m}" for m in mn], "ASP", "UnitCost", "BOH", "CasePack", "LeadTime", "z", "Sigma",
             "SafetyStock", *[f"Rcpt_{m}" for m in mn[:3]], "PO_units", "PO_receipt", "PO_cost", "WOS", "RiskFlag"])
    sheets = {
        "Summary": summary_table(res),
        "PO_this_cycle": plan.loc[plan["PO_units"] > 0, cols].sort_values("PO_cost", ascending=False),
        "SKU_Plan": plan.sort_values("Rev_L12M", ascending=False)[cols],
        "OTB_Plan": res.otb,
        "OTB_Bridge": res.bridge,
        "Risk_Stockout": plan[plan["RiskFlag"].str.contains("Stockout")].assign(
            Rev_at_risk=lambda d: np.maximum(0, d[f"Fcst_{mn[0]}"] - d["BOH"]) * d["ASP"]
        ).sort_values("Rev_at_risk", ascending=False)[cols + ["Rev_at_risk"]],
        "Risk_EOL_Overstock": plan[plan["RiskFlag"].str.contains("EOL|Overstock")].assign(
            Stock_cost=lambda d: d["BOH"] * d["UnitCost"]).sort_values("Stock_cost", ascending=False)[cols + ["Stock_cost"]],
        "What_Changed": res.what_changed.reindex(res.what_changed["Delta_rev"].abs().sort_values(ascending=False).index).head(200),
        "Backtest_ChangeLog": res.backtest,
        "Post_Mortem": res.postmortem.reset_index(names="Plan"),
        "Lifecycle": res.lifecycle["stats"].reset_index(names="Archetype"),
        "Data_Quality": res.clean_log,
    }
    path = Path(path)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            for j, c in enumerate(df.columns, 1):
                cell = ws.cell(row=1, column=j)
                cell.font = Font(name="Arial", bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="1F3A5F")
                cell.alignment = Alignment(wrap_text=True, vertical="center")
                width = min(48, max(10, len(str(c)) + 2, *(len(str(v)) + 2 for v in df[c].head(50))))
                ws.column_dimensions[get_column_letter(j)].width = width
                fmt = _fmt(c)
                if fmt:
                    for r in range(2, len(df) + 2):
                        ws.cell(row=r, column=j).number_format = fmt
            ws.freeze_panes = "B2"
            ws.auto_filter.ref = ws.dimensions
    return path


def _fmt(col: str) -> str | None:
    c = col.lower()
    if any(k in c for k in ("wmape", "bias", "fill", "share", "yoy")):
        return "0.0%"
    if any(k in c for k in ("cost", "rev", "retail", "asp", "value", "variance", "otb", "bom", "eom", "sales")):
        return "$#,##0;($#,##0);-"
    if any(k in c for k in ("fcst", "rcpt", "boh", "units", "safety", "sigma", "base", "act_", "delta")):
        return "#,##0;(#,##0);-"
    return None


# ------------------------------------------------------------------ summary
def summary_table(res) -> pd.DataFrame:
    plan, pm, br = res.plan, res.postmortem, res.bridge
    otb_total = br["value"].iloc[0]
    bu_total = br["value"].iloc[-1]
    rows = [
        ("Forecast origin (last month of actuals)", plan.attrs["origin"][:7]),
        ("Model version", plan.attrs["version"]),
        ("Active SKU x channel planning units", f"{len(plan):,}"),
        ("PO to approve this cycle (cost)", money(plan["PO_cost"].sum())),
        ("PO lines", f"{int((plan['PO_units'] > 0).sum()):,}"),
        ("Bottom-up receipts, next 3 months (cost)", money(bu_total)),
        ("Top-down OTB, next 3 months (cost)", money(otb_total)),
        ("Bottom-up vs OTB", f"{bu_total / otb_total - 1:+.1%}"),
        ("Proposed v1.7 receipts (cost)", money(res.bridge_v17["value"].iloc[-1])),
        ("Stockout-risk SKUs", f"{plan['RiskFlag'].str.contains('Stockout').sum():,}"),
        ("Exit / markdown SKUs", f"{plan['RiskFlag'].str.contains('EOL').sum():,}"),
        ("Post-mortem unit fill: v1.0 / v1.6 / v1.7",
         " / ".join(f"{pm.iloc[i]['fill_units']:.1%}" for i in range(3))),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def write_markdown(res, path: str | Path) -> Path:
    s = summary_table(res)
    lines = ["# Monthly buy recommendation", ""]
    lines += [f"- **{m}:** {v}" for m, v in s.itertuples(index=False)]
    lines += ["", "## OTB bridge (cost)", ""]
    lines += [f"- {r.step}: {money(r.value)}" for r in res.bridge.itertuples()]
    lines += ["", "## Post-mortem", "", res.postmortem.to_markdown(floatfmt=".3f") if _has_tabulate() else res.postmortem.to_string()]
    Path(path).write_text("\n".join(lines) + "\n")
    return Path(path)


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False
