"""Run one monthly OTB & replenishment cycle end to end.

    python run_cycle.py --data data/online_retail_II_combined.csv --origin 2011-08 --out outputs
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
warnings.filterwarnings("ignore", category=RuntimeWarning)

from otb_planner import V16, run_cycle                      # noqa: E402
from otb_planner import reporting as R                     # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/online_retail_II_combined.csv")
    ap.add_argument("--origin", default="2011-08", help="last month of actuals, YYYY-MM")
    ap.add_argument("--version", default="v1.6", help="forecast version (v1.0 ... v1.6)")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--skip-backtest", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    res = run_cycle(a.data, origin=f"{a.origin}-01", version=a.version, cfg=V16, run_backtest=not a.skip_backtest)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    xlsx = R.write_excel(res, out / f"buy_recommendation_{a.origin}.xlsx")
    md = R.write_markdown(res, out / f"buy_summary_{a.origin}.md")
    charts = R.save_all_charts(res, out / "charts")
    print()
    print(R.summary_table(res).to_string(index=False))
    print(f"\nwrote {xlsx}, {md} and {len(charts)} charts in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
