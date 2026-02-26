"""
Re-run the full LLM trading experiment with all code fixes applied.

Fixes active in this run vs. the original:
  1. EMA-based RSI-14 (Wilder's method) instead of SMA-based
  2. Frequency-aware annualization in compute_metrics()
  3. Regex fallback for JSON parsing in trading_agent.py

Results are saved to improvements/fixed_experiment/results/ so the
original results/ directory is preserved for comparison.
"""

import sys
import json
import logging
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

import backtester
from data_loader import EXPERIMENT_TICKERS

OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

backtester.RESULTS_DIR = OUTPUT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

if __name__ == "__main__":
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Stocks: {EXPERIMENT_TICKERS}")
    print(f"Frequencies: daily, weekly, monthly")
    print(f"This will make ~1,570 API calls to gpt-4.1-mini.\n")

    results = backtester.run_full_experiment(
        tickers=EXPERIMENT_TICKERS,
        frequencies=["daily", "weekly", "monthly"],
        model="gpt-4.1-mini",
        temperature=0.0,
    )

    print("\n\nSUMMARY")
    print("=" * 80)
    for key, res in results.items():
        if "error" in res:
            print(f"{key}: ERROR - {res['error']}")
        else:
            llm = res["llm_metrics"]
            bh = res["bh_metrics"]
            print(
                f"{key:20s} | LLM CR={llm['cumulative_return']:+7.2%} SR={llm['sharpe_ratio']:+6.2f} "
                f"MDD={llm['max_drawdown']:+7.2%} | B&H CR={bh['cumulative_return']:+7.2%} "
                f"SR={bh['sharpe_ratio']:+6.2f}"
            )
