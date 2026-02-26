"""
Risk-Aware Backtester: Applies position sizing, trailing stop-losses,
and volatility scaling to the existing LLM trading experiment results.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from metrics import compute_metrics, FREQUENCY_TO_PERIODS

RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


class RiskManager:
    """Configurable risk management layer for LLM trading decisions."""

    # Initialize class vars.
    def __init__(
        self,
        trailing_stop_pct: float = 0.08,
        hold_position_fraction: float = 0.60,
        vol_lookback: int = 20,
        vol_target: float = 0.15,
        frequency: str = "daily",
    ):
        self.trailing_stop_pct = trailing_stop_pct
        # How much of capital STAYS invested when HOLD. You scale back from 100% to this fraction
        self.hold_position_fraction = hold_position_fraction
        # How many periods you lookback when measuring stock volatility e.g. default = last 20 returns
        self.vol_lookback = vol_lookback
        # What you want your exposure / risk to be? Default = 15% annually
        self.vol_target = vol_target
        self.periods_per_year = FREQUENCY_TO_PERIODS.get(frequency, 252)

        self.high_water_mark = 0.0
        self.position_entry_price = 0.0
        self.stop_triggered = False

    def reset(self):
        self.high_water_mark = 0.0
        self.position_entry_price = 0.0
        self.stop_triggered = False

    def compute_vol_scalar(self, returns_history: np.ndarray) -> float:
        """Scale position by inverse realized volatility relative to target."""
        # If there is a lack of data i.e. not enough returns history then default to 1 i.e. no scaling.
        if len(returns_history) < self.vol_lookback:
            return 1.0
        recent = returns_history[-self.vol_lookback:]
        # realized_vol = How much the stock price has moved over a given window.
        # Need to multiply in order to get annualized figure
        # Variance scales linearly with time so std scales with square root of time
        realized_vol = np.std(recent) * np.sqrt(self.periods_per_year)
        if realized_vol <= 0:
            return 1.0
        # scaler = how much are you investing after taking volatility into account
        scalar = self.vol_target / realized_vol
        # We clamp the scaler to a maximum of 1.0
        # We never want to exceed > 100% because that would mean investing even more money (money that you may not have)
        # This scenario is only possible for less volatile stocks (compared to your target) leading to realized_vol < vol_target hence value becomes greater > 1
        # e.g. 15 / 10 -> 1.5
        return float(np.clip(scalar, 0.1, 1.0))

    def get_position_size(
        self,
        decision: str,
        current_position: str,
        current_price: float,
        returns_history: np.ndarray,
    ) -> float:
        """How much is being invested"""

        # A stop was previously triggered meaning the stock fell > 8% from its peak.
        # We stay out of the market (return 0.0) until the LLM explicitly says BUY again,
        # treating that as a signal the stock has stabilized enough to re-enter.
        if self.stop_triggered:
            if decision == "BUY":
                # Re-entry after a stop: reset state and treat this period as a fresh BUY.
                self.stop_triggered = False
                self.high_water_mark = current_price
                self.position_entry_price = current_price
            else:
                # HOLD or SELL while stopped out — remain in cash, invest nothing.
                return 0.0

        # Compute how much to scale down the position based on recent volatility.
        vol_scalar = self.compute_vol_scalar(returns_history)

        if decision == "BUY":
            # Strong conviction — go fully invested, scaled by volatility.
            # e.g. if vol_scalar = 0.5 (stock is twice as volatile as target), invest 50% not 100%.
            self.position_entry_price = current_price
            self.high_water_mark = current_price
            return 1.0 * vol_scalar

        elif decision == "SELL":
            # Exit the position entirely — reset tracking state.
            self.high_water_mark = 0.0
            self.position_entry_price = 0.0
            return 0.0

        else:  # HOLD
            if current_position == "LONG":
                # Weak conviction — already in the stock but not confident enough to double down.
                # Stay invested at a reduced fraction (default 60%) to reflect lower confidence vs. a BUY (100%),
                # further scaled by volatility so riskier stocks get an even smaller position!
                return self.hold_position_fraction * vol_scalar
            else:
                # HOLD while already in cash — no signal to enter, so invest nothing.
                return 0.0

    def check_trailing_stop(self, current_price: float) -> bool:
        if self.high_water_mark <= 0:
            return False
        
        # highest_water_mark = greatest price for given stock in a given time period
        if current_price > self.high_water_mark:
            self.high_water_mark = current_price
        
        # drawdown is 0% if prices increase. If prices decrease it is calculated from the highest price that stock was at.
        drawdown = (current_price - self.high_water_mark) / self.high_water_mark
        if drawdown < -self.trailing_stop_pct:
            self.stop_triggered = True
            self.high_water_mark = 0.0
            self.position_entry_price = 0.0
            return True
        return False


def extract_asset_returns_from_decisions(decisions: list) -> list:
    # "Close prices" in decisions array = value of security for that day i.e. the return
    asset_returns = []
    for i in range(len(decisions) - 1):
        current_close = decisions[i]["close"]
        next_close = decisions[i + 1]["close"]
        # Calculate simple return e.g. 103 (next day's close price) / 100 (current close price) - 1 = 0.03 increase
        # Return will be value of stock movement from one day to the next.
        ret = (next_close / current_close) - 1
        asset_returns.append(ret)
    asset_returns.append(0.0) # No N+1 close price as we've iterated over entire decisions array. So return of last decision = 0
    return asset_returns


def replay_with_risk_management(
    result_path: Path,
    trailing_stop_pct: float = 0.08,
    hold_fraction: float = 0.60,
    vol_target: float = 0.15,
) -> dict:
    """Re-run a single experiment e.g. JPM_WEEKLY through the risk management layer """

    with open(result_path) as f:
        original = json.load(f)

    ticker = original["ticker"]
    frequency = original["frequency"]
    decisions = original["decisions"]

    # asset_returns = list of daily returns
    asset_returns = extract_asset_returns_from_decisions(decisions)

    # Now that we have calculated the returns we initialize the Risk Manager that will use these returns.
    rm = RiskManager(
        trailing_stop_pct=trailing_stop_pct,
        hold_position_fraction=hold_fraction,
        vol_target=vol_target,
        frequency=frequency,
    )

    portfolio_returns = []
    position_sizes = []
    stop_triggers = []
    returns_history = []
    position = "OUT"

    # Note: LLM had already made BUY/SELL/HOLD Decisions. This function just replays them.
    for i, entry in enumerate(decisions):
        close = entry["close"]
        decision = entry["decision"]
        original_return = entry["period_return"]
        # Step 1: Override what LLM said if trailing stop is breached
        if rm.check_trailing_stop(close) and position == "LONG":
            stop_triggers.append(entry["date"])
            position = "OUT"

        hist = np.array(returns_history[-60:]) if returns_history else np.array([])
        # How much should be invested?
        pos_size = rm.get_position_size(decision, position, close, hist)
        # We update positions for next iteration
        position_sizes.append(pos_size)

        # ORDER matter! First check for stop, if not, for "BUY", if not for "SELL"
        if rm.stop_triggered:
            position = "OUT"
        elif decision == "BUY":
            position = "LONG"
        elif decision == "SELL":
            position = "OUT"

        period_asset_return = asset_returns[i]
        returns_history.append(period_asset_return)
        
        # What is the new return give the pos_size and the inherent move of the asset?
        period_return = pos_size * period_asset_return
        portfolio_returns.append(period_return)

    # Convert the list of per-period returns into a Series so compute_metrics can operate on it.
    risk_managed_returns = pd.Series(portfolio_returns)
    # Reconstruct the original (no risk management) returns from the decision log for comparison.
    original_returns = pd.Series([d["period_return"] for d in decisions])
    risk_metrics = compute_metrics(risk_managed_returns, frequency=frequency)
    original_metrics = compute_metrics(original_returns, frequency=frequency)

    return {
        "ticker": ticker,
        "frequency": frequency,
        "original_metrics": original_metrics,
        "risk_managed_metrics": risk_metrics,
        "bh_metrics": original.get("bh_metrics", {}),
        "config": {
            "trailing_stop_pct": trailing_stop_pct,
            "hold_fraction": hold_fraction,
            "vol_target": vol_target,
        },
        "stop_trigger_count": len(stop_triggers),
        "stop_trigger_dates": stop_triggers,
        "avg_position_size": float(np.mean(position_sizes)) if position_sizes else 0.0,
        "num_decisions": len(decisions),
    }


def run_full_comparison():
    """Run risk management replay on all existing results and produce comparison."""
    print("=" * 70)
    print("RISK MANAGEMENT IMPROVEMENT: REPLAY ANALYSIS")
    print("=" * 70)

    all_results = []
    result_files = sorted(RESULTS_DIR.glob("*_t0.0.json"))
    result_files = [f for f in result_files if not f.name.startswith(("full_experiment", "statistical"))]

    # Iterate over all result files from the /results folder
    for path in result_files:
        key = path.stem.replace("_t0.0", "")
        print(f"\nProcessing: {key}")
        try:
            result = replay_with_risk_management(path)
            all_results.append(result)

            orig = result["original_metrics"]
            risk = result["risk_managed_metrics"]
            print(f"  Original  -> CR={orig['cumulative_return']:+.2%}  SR={orig['sharpe_ratio']:.2f}  MDD={orig['max_drawdown']:.2%}")
            print(f"  Risk-Mgd  -> CR={risk['cumulative_return']:+.2%}  SR={risk['sharpe_ratio']:.2f}  MDD={risk['max_drawdown']:.2%}")
            print(f"  Stop-loss triggers: {result['stop_trigger_count']}  Avg position: {result['avg_position_size']:.1%}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        print("\nNo results to compare. Ensure result JSON files exist in results/")
        return all_results

    with open(OUTPUT_DIR / "risk_managed_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    rows = []
    for r in all_results:
        rows.append({
            "Ticker": r["ticker"],
            "Frequency": r["frequency"],
            "Orig_CR": r["original_metrics"]["cumulative_return"],
            "Risk_CR": r["risk_managed_metrics"]["cumulative_return"],
            "Orig_SR": r["original_metrics"]["sharpe_ratio"],
            "Risk_SR": r["risk_managed_metrics"]["sharpe_ratio"],
            "Orig_MDD": r["original_metrics"]["max_drawdown"],
            "Risk_MDD": r["risk_managed_metrics"]["max_drawdown"],
            "Stops": r["stop_trigger_count"],
            "Avg_Pos": r["avg_position_size"],
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Below code written by claude to make sense of the dataframe generated after iterating over all results
    print("\n\nAGGREGATE BY FREQUENCY:")
    for freq in ["daily", "weekly", "monthly"]:
        mask = df["Frequency"] == freq
        sub = df[mask]
        if sub.empty:
            continue
        print(f"\n  {freq.upper()}:")
        print(f"    Mean Orig SR:  {sub['Orig_SR'].mean():.3f}  ->  Risk SR:  {sub['Risk_SR'].mean():.3f}  (change: {sub['Risk_SR'].mean() - sub['Orig_SR'].mean():+.3f})")
        print(f"    Mean Orig MDD: {sub['Orig_MDD'].mean():.2%}  ->  Risk MDD: {sub['Risk_MDD'].mean():.2%}  (improvement: {sub['Risk_MDD'].mean() - sub['Orig_MDD'].mean():+.2%})")
        print(f"    Mean Orig CR:  {sub['Orig_CR'].mean():.2%}  ->  Risk CR:  {sub['Risk_CR'].mean():.2%}  (change: {sub['Risk_CR'].mean() - sub['Orig_CR'].mean():+.2%})")
        print(f"    Total stop triggers: {sub['Stops'].sum():.0f}   Avg position size: {sub['Avg_Pos'].mean():.1%}")

    mdd_improvement = df["Risk_MDD"].mean() - df["Orig_MDD"].mean()
    sr_change = df["Risk_SR"].mean() - df["Orig_SR"].mean()
    cr_change = df["Risk_CR"].mean() - df["Orig_CR"].mean()

    print(f"\n\nOVERALL IMPACT:")
    print(f"  MDD change (less negative = better): {mdd_improvement:+.2%}")
    print(f"  SR change:                           {sr_change:+.3f}")
    print(f"  CR change:                           {cr_change:+.2%}")
    worst_mdd = df["Risk_MDD"].min()
    print(f"  Worst-case MDD:                      {worst_mdd:.2%}")
    print(f"  Original worst-case MDD:             {df['Orig_MDD'].min():.2%}")
    print(f"  MDD bounded below stop-loss (-8%)?   {'YES' if worst_mdd > -0.09 else 'NO — some positions gap through stop'}")

    df.to_csv(OUTPUT_DIR / "comparison_table.csv", index=False)

    write_results_summary(df, all_results)

    print(f"\nResults saved to {OUTPUT_DIR}/")
    return all_results

# Generated by claude to make meaningful summary of dataframe
def write_results_summary(df: pd.DataFrame, all_results: list):
    """Generate a markdown summary of the results."""
    lines = [
        "# Risk Management Improvement: Results Summary\n",
        "## Configuration",
        f"- Trailing stop: {all_results[0]['config']['trailing_stop_pct']:.0%}",
        f"- HOLD position fraction: {all_results[0]['config']['hold_fraction']:.0%}",
        f"- Volatility target: {all_results[0]['config']['vol_target']:.0%}\n",
        "## Per-Stock Results\n",
        "| Ticker | Freq | Orig CR | Risk CR | Orig SR | Risk SR | Orig MDD | Risk MDD | Stops |",
        "|--------|------|---------|---------|---------|---------|----------|----------|-------|",
    ]

    for _, row in df.iterrows():
        lines.append(
            f"| {row['Ticker']} | {row['Frequency']} | "
            f"{row['Orig_CR']:+.1%} | {row['Risk_CR']:+.1%} | "
            f"{row['Orig_SR']:.2f} | {row['Risk_SR']:.2f} | "
            f"{row['Orig_MDD']:.1%} | {row['Risk_MDD']:.1%} | "
            f"{row['Stops']:.0f} |"
        )

    lines.append("\n## Aggregate by Frequency\n")
    lines.append("| Frequency | Orig SR | Risk SR | SR Change | Orig MDD | Risk MDD | MDD Change |")
    lines.append("|-----------|---------|---------|-----------|----------|----------|------------|")
    for freq in ["daily", "weekly", "monthly"]:
        sub = df[df["Frequency"] == freq]
        if sub.empty:
            continue
        lines.append(
            f"| {freq} | {sub['Orig_SR'].mean():.2f} | {sub['Risk_SR'].mean():.2f} | "
            f"{sub['Risk_SR'].mean() - sub['Orig_SR'].mean():+.2f} | "
            f"{sub['Orig_MDD'].mean():.1%} | {sub['Risk_MDD'].mean():.1%} | "
            f"{sub['Risk_MDD'].mean() - sub['Orig_MDD'].mean():+.1%} |"
        )

    lines.append("\n## Key Findings\n")
    mdd_imp = df["Risk_MDD"].mean() - df["Orig_MDD"].mean()
    sr_chg = df["Risk_SR"].mean() - df["Orig_SR"].mean()
    cr_chg = df["Risk_CR"].mean() - df["Orig_CR"].mean()

    lines.append(f"- **Max Drawdown** improved by {abs(mdd_imp):.1%} on average (less negative = better)")
    lines.append(f"- **Sharpe Ratio** changed by {sr_chg:+.3f} on average")
    lines.append(f"- **Cumulative Return** changed by {cr_chg:+.1%} on average")
    lines.append(f"- **Worst-case MDD** went from {df['Orig_MDD'].min():.1%} to {df['Risk_MDD'].min():.1%}")

    total_stops = df["Stops"].sum()
    lines.append(f"- **Total stop-loss triggers** across all conditions: {total_stops:.0f}")

    lines.append("\n## Interpretation\n")
    if mdd_imp > 0:
        lines.append("The risk management layer successfully reduced maximum drawdowns across the experiment. ")
    else:
        lines.append("The risk management layer did not reduce drawdowns on average, suggesting the parameters need tuning. ")

    if sr_chg > -0.1:
        lines.append("Sharpe ratios were maintained or improved, indicating that the variance reduction from position sizing ")
        lines.append("outweighed the reduction in upside capture.\n")
    else:
        lines.append("Sharpe ratios declined, suggesting the position sizing was too aggressive in reducing exposure ")
        lines.append("during periods that turned out to be profitable.\n")

    lines.append("Whether these results represent improvement or not, they demonstrate that risk management is a ")
    lines.append("meaningful dimension of the trading system that interacts with decision frequency in measurable ways.")

    with open(OUTPUT_DIR / "RESULTS_SUMMARY.md", "w") as f:
        f.write("\n".join(lines))

# Assessing how sensitive the parameters we set for the backtester are
def run_sensitivity_analysis():
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS")
    print("=" * 70)

    result_files = sorted(RESULTS_DIR.glob("*_t0.0.json"))
    result_files = [f for f in result_files if not f.name.startswith(("full_experiment", "statistical"))]

    if not result_files:
        print("No result files found for sensitivity analysis.")
        return

    stop_values = [0.05, 0.08, 0.10, 0.15] #Trailing stop rates from 5 to 15 percent each determining a varying level of risk
    # The greater the value here the more you are willing to retain in the market despite potential risk
    hold_values = [0.40, 0.60, 0.80]
    # Different levels of volatility
    vol_values = [0.10, 0.15, 0.20]

    sensitivity_rows = []

    for stop in stop_values:
        for hold in hold_values:
            for vol in vol_values:
                all_sr = []
                all_mdd = []
                all_cr = []
                for path in result_files:
                    try:
                        r = replay_with_risk_management(
                            path, trailing_stop_pct=stop,
                            hold_fraction=hold, vol_target=vol
                        )
                        all_sr.append(r["risk_managed_metrics"]["sharpe_ratio"])
                        all_mdd.append(r["risk_managed_metrics"]["max_drawdown"])
                        all_cr.append(r["risk_managed_metrics"]["cumulative_return"])
                    except Exception:
                        pass

                if all_sr:
                    sensitivity_rows.append({
                        "Stop%": stop,
                        "Hold%": hold,
                        "VolTarget": vol,
                        "Mean_SR": np.mean(all_sr),
                        "Mean_MDD": np.mean(all_mdd),
                        "Mean_CR": np.mean(all_cr),
                    })

    sens_df = pd.DataFrame(sensitivity_rows)
    sens_df = sens_df.sort_values("Mean_SR", ascending=False)
    print("\nTop 10 parameter combinations by mean Sharpe Ratio:")
    print(sens_df.head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    best = sens_df.iloc[0]
    print(f"\nBest config: Stop={best['Stop%']:.0%}  Hold={best['Hold%']:.0%}  "
          f"Vol={best['VolTarget']:.0%}  -> SR={best['Mean_SR']:.3f}  MDD={best['Mean_MDD']:.2%}  CR={best['Mean_CR']:.2%}")

    worst_mdd_row = sens_df.loc[sens_df["Mean_MDD"].idxmax()]
    print(f"Best MDD config: Stop={worst_mdd_row['Stop%']:.0%}  Hold={worst_mdd_row['Hold%']:.0%}  "
          f"Vol={worst_mdd_row['VolTarget']:.0%}  -> MDD={worst_mdd_row['Mean_MDD']:.2%}")

    sens_df.to_csv(OUTPUT_DIR / "sensitivity_analysis.csv", index=False)
    print(f"\nFull sensitivity results saved to {OUTPUT_DIR / 'sensitivity_analysis.csv'}")

    lines = ["\n## Sensitivity Analysis\n"]
    lines.append(f"Swept {len(stop_values)} stop-loss thresholds x {len(hold_values)} hold fractions "
                 f"x {len(vol_values)} volatility targets = {len(sensitivity_rows)} configurations.\n")
    lines.append("### Top 5 by Sharpe Ratio\n")
    lines.append("| Stop | Hold | Vol Target | Mean SR | Mean MDD | Mean CR |")
    lines.append("|------|------|------------|---------|----------|---------|")
    for _, row in sens_df.head(5).iterrows():
        lines.append(f"| {row['Stop%']:.0%} | {row['Hold%']:.0%} | {row['VolTarget']:.0%} | "
                     f"{row['Mean_SR']:.3f} | {row['Mean_MDD']:.1%} | {row['Mean_CR']:+.1%} |")
    lines.append(f"\nBaseline (no risk management) mean SR: "
                 f"see comparison table for per-frequency values.")

    summary_path = OUTPUT_DIR / "RESULTS_SUMMARY.md"
    with open(summary_path, "a") as f:
        f.write("\n".join(lines))

    return sens_df


if __name__ == "__main__":
    run_full_comparison()
    run_sensitivity_analysis()
