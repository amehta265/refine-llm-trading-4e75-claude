"""
Backtesting engine for multi-frequency LLM trading experiments.
Runs the LLM agent through historical data and computes performance metrics.
"""

import json
import logging
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from data_loader import (
    prepare_experiment_data, format_price_history, load_price_data,
    compute_indicators, EXPERIMENT_TICKERS
)
from trading_agent import get_llm_decision
from metrics import compute_metrics

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"

# Moving compute_metrics to a separate metrics.py file so it can be used in multiple places

def run_llm_backtest(ticker: str, frequency: str,
                     start_date: str = "2024-01-01",
                     end_date: str = "2024-12-31",
                     model: str = "gpt-4.1-mini",
                     temperature: float = 0.0,
                     lookback: int = 15) -> dict:
    """Run LLM trading backtest for a single ticker and frequency.

    Args:
        ticker: Stock ticker
        frequency: 'daily', 'weekly', or 'monthly'
        start_date: Evaluation start date
        end_date: Evaluation end date
        model: LLM model to use
        temperature: Sampling temperature
        lookback: Number of historical periods to show the LLM

    Returns:
        Dict with decisions, returns, metrics, and metadata
    """
    # Load data with extra history for indicators
    full_df = load_price_data(ticker, frequency)
    full_df = compute_indicators(full_df)
    full_df = full_df.dropna()

    # Get evaluation period indices
    eval_mask = (full_df.index >= start_date) & (full_df.index <= end_date)
    eval_indices = np.where(eval_mask)[0]

    if len(eval_indices) == 0:
        raise ValueError(f"No data in evaluation period for {ticker} {frequency}")

    decisions = []
    position = "OUT"  # Start with no position
    portfolio_returns = []
    total_tokens = 0
    decision_log = []
    """ 
        REVIEW (Note): 
            Most trading agents and platforms have a stop loss calculator where if a price falls below a threshold the system automatically
            sells the stock to prevent further losses. This kind of a "force-exit" strategy could increase gains for all frequencies
    """
    logger.info(f"Starting backtest: {ticker} {frequency} ({len(eval_indices)} periods)")

    for i, idx in enumerate(eval_indices):
        # Format price history for this decision point
        price_history = format_price_history(full_df, idx, lookback=lookback)

        """
        REVIEW (Improvement): 
            get_llm_decision function is called to make a BUY/HOLD/SELL decisions based on price data and technical indicators.
            However, this function is stateless. It does not receive any information about its prior decisions and their outcomes. 
            This is a weakness as it limits the model's ability to learn from its own decisions and improve over time.
                - This improvement is grounded in the FinMem paper "A Performance-Enhanced LLM Trading Agent With Layered Memory and Character Design (2023) 
                by Yu et al. Here, a memory-module framework stores reflections, observations, and summaries of prior decisions over different time horizons.
                The decay rates (14-day, 90-day, 365-day) used in the paper maintain context across decisions, improving model performance, something we should strive to achieve.

            Solution: Add a DecisionMemory module that injects recent decision history and outcomes thus enabling self-correction and regime adaptation.
        """
        # Get LLM decision
        result = get_llm_decision(
            ticker=ticker,
            frequency=frequency,
            price_history=price_history,
            current_position="LONG" if position == "LONG" else "OUT",
            model=model,
            temperature=temperature,
        )

        decision = result["decision"]
        total_tokens += result["tokens_used"]

        # Compute return for this period
        if idx + 1 < len(full_df):
            next_return = (full_df.iloc[idx + 1]["Close"] / full_df.iloc[idx]["Close"]) - 1
        else:
            next_return = 0.0

        """
            REVIEW (Note):
                Most LLM trading literature (FinMem, InvestorBench, StockBench) share this binary implementation of all in or all out.
                So it makes sense the AI agent followed in the same footsteps. But this prevents the model from expressing confidence. I expand on this
                further on Line 18 in `trading_agent.py`
        """
        if decision == "BUY":
            position = "LONG"
        elif decision == "SELL":
            position = "OUT"
        # HOLD: maintain current position

        # Portfolio return depends on position
        if position == "LONG":
            period_return = next_return
        else:
            period_return = 0.0
            """
                REVIEW (Note):
                    This condition is met when position === "OUT" i.e. agent has cash and is not holding stock.
                    Technically this isn't 0.0. It is the risk free rate (hard-coded to 5%) because if this cash sat in the market or a treasury account
                    it would make 5% annualized risk free.
                    This isn't a big issue unless this condition is met numerously. It would undermine how much money a trader makes if he is out of the market 
                    e.g. a monthly trader that is out for 6 months during the year in reality would have earned 2.5% not 0.
            """ 

        portfolio_returns.append(period_return)

        """
            REVIEW (Note):
                It is common to decompose returns into alpha (trader's skill) and beta (market exposure).
                This would answer whether the LLM adds value with proper timing or simply rides market beta
                with varying exposure. (https://www.mdpi.com/2227-9091/6/4/124)
        """

        decision_entry = {
            "date": full_df.index[idx].strftime("%Y-%m-%d"),
            "close": float(full_df.iloc[idx]["Close"]),
            "decision": decision,
            "reasoning": result["reasoning"],
            "position": position,
            "period_return": float(period_return),
            "tokens": result["tokens_used"],
        }
        decision_log.append(decision_entry)
        decisions.append(decision)

        # Progress logging every 50 decisions
        if (i + 1) % 50 == 0:
            logger.info(f"  {ticker} {frequency}: {i+1}/{len(eval_indices)} decisions complete")

        # Small delay to avoid rate limiting
        if i % 20 == 0 and i > 0:
            time.sleep(0.5)

    # Compute metrics
    returns_series = pd.Series(portfolio_returns)
    metrics = compute_metrics(returns_series, frequency=frequency)

    # Compute turnover (position changes)
    position_changes = sum(1 for j in range(1, len(decisions))
                          if decisions[j] != decisions[j-1])
    turnover_rate = position_changes / max(len(decisions) - 1, 1)

    """
        REVIEW (Improvement): No transaction costs are applied to any of the trades above.
            Transaction costs are the fees you pay every time you buy or sell stocks, the bid-ask spread (the gap between what
            buyers offer and sellers accept), and slippage (price moving against you while your order executes). These compound quickly.
            This is a bias towards daily trading that incurs more of this variable cost due to a higher volume of transactions.
            This alone would likely make the monthly advantage statistically significant. The omission actually understates the paper's core finding.
    """

    """
        REVIEW (Note):
            I like the use of two baselines.  It lets us interpret the  LLM's performance from two angles: 
            vs. doing nothing, and vs. a simple rule-based strategy.
    """

    # Buy-and-hold baseline for comparison
    bh_returns = []
    for idx in eval_indices:
        if idx + 1 < len(full_df):
            bh_returns.append((full_df.iloc[idx + 1]["Close"] / full_df.iloc[idx]["Close"]) - 1)
        else:
            bh_returns.append(0.0)
    bh_metrics = compute_metrics(pd.Series(bh_returns), frequency=frequency)

    # SMA crossover baseline
    sma_returns = []
    sma_position = "OUT"
    for idx in eval_indices:
        row = full_df.iloc[idx]
        if row["SMA_10"] > row["SMA_20"]:
            sma_position = "LONG"
        else:
            sma_position = "OUT"

        if idx + 1 < len(full_df):
            next_ret = (full_df.iloc[idx + 1]["Close"] / full_df.iloc[idx]["Close"]) - 1
        else:
            next_ret = 0.0

        if sma_position == "LONG":
            sma_returns.append(next_ret)
        else:
            sma_returns.append(0.0)
    sma_metrics = compute_metrics(pd.Series(sma_returns), frequency=frequency)

    result = {
        "ticker": ticker,
        "frequency": frequency,
        "model": model,
        "temperature": temperature,
        "start_date": start_date,
        "end_date": end_date,
        "llm_metrics": metrics,
        "bh_metrics": bh_metrics,
        "sma_metrics": sma_metrics,
        "decisions": decision_log,
        "decision_counts": {
            "BUY": decisions.count("BUY"),
            "HOLD": decisions.count("HOLD"),
            "SELL": decisions.count("SELL"),
        },
        "turnover_rate": turnover_rate,
        "total_tokens": total_tokens,
        "num_decisions": len(decisions),
        "timestamp": datetime.now().isoformat(),
    }

    return result


def run_full_experiment(tickers: list = None,
                        frequencies: list = None,
                        model: str = "gpt-4.1-mini",
                        temperature: float = 0.0,
                        start_date: str = "2024-01-01",
                        end_date: str = "2024-12-31") -> dict:
    """Run the full multi-frequency trading experiment.

    Args:
        tickers: List of stock tickers (default: EXPERIMENT_TICKERS)
        frequencies: List of frequencies (default: daily/weekly/monthly)
        model: LLM model
        temperature: Sampling temperature
        start_date: Evaluation start
        end_date: Evaluation end

    Returns:
        Dict mapping (ticker, frequency) to results
    """
    if tickers is None:
        tickers = EXPERIMENT_TICKERS
    if frequencies is None:
        frequencies = ["daily", "weekly", "monthly"]

    all_results = {}
    total_start = time.time()

    for ticker in tickers:
        for freq in frequencies:
            key = f"{ticker}_{freq}"
            logger.info(f"\n{'='*60}")
            logger.info(f"Running: {ticker} @ {freq} frequency")
            logger.info(f"{'='*60}")

            try:
                result = run_llm_backtest(
                    ticker=ticker,
                    frequency=freq,
                    start_date=start_date,
                    end_date=end_date,
                    model=model,
                    temperature=temperature,
                )
                all_results[key] = result

                # Save intermediate results
                result_path = RESULTS_DIR / f"{key}_t{temperature}.json"
                with open(result_path, "w") as f:
                    json.dump(result, f, indent=2, default=str)

                cr = result["llm_metrics"]["cumulative_return"]
                sr = result["llm_metrics"]["sharpe_ratio"]
                bh_cr = result["bh_metrics"]["cumulative_return"]
                logger.info(
                    f"  LLM CR={cr:.2%} SR={sr:.2f} | "
                    f"B&H CR={bh_cr:.2%} | "
                    f"Decisions: {result['decision_counts']} | "
                    f"Tokens: {result['total_tokens']}"
                )

            except Exception as e:
                logger.error(f"Failed: {ticker} {freq}: {e}")
                all_results[key] = {"error": str(e)}

    elapsed = time.time() - total_start
    logger.info(f"\nTotal experiment time: {elapsed:.1f}s")

    # Save full results
    summary_path = RESULTS_DIR / f"full_experiment_t{temperature}.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    return all_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Run experiment with default settings
    results = run_full_experiment(
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
