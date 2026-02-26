"""
Extracted the compute_metrics function into this reusable module so it can be used by both the 
backtester and new risk_backtester without repeating computational logic
"""

import numpy as np
import pandas as pd

FREQUENCY_TO_PERIODS = {"daily": 252, "weekly": 52, "monthly": 12}

def compute_metrics(returns: pd.Series, frequency: str = None, risk_free_rate: float = 0.05) -> dict:
    """Compute trading performance metrics from a return series.

    Args:
        returns: Series of period returns (as fractions, not percentages)
        frequency: Trading frequency ('daily', 'weekly', 'monthly'). When provided,
                   directly maps to the correct annualization factor instead of guessing.
        risk_free_rate: Annual risk-free rate (default 5% for 2024)

    Returns:
        Dict with CR, SR, MDD, Sortino, Win Rate, etc.
    """
    if len(returns) == 0 or returns.std() == 0:
        return {
            "cumulative_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "sortino_ratio": 0.0,
            "win_rate": 0.0,
            "num_trades": 0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
        }

    # Cumulative return
    cum_return = (1 + returns).prod() - 1

    """
        REVIEW (Note):
            Whats happening: The n_periods heuristic determines the frequency (daily/weekly/monthly) based on 
            training data points in order to calculate annualized return and volatility.

            Current setup: The model was run with 1 year of data. An year has 252 trading days so > 200 would correctly classify daily,
            52 would correctly classify weekly and so on.

            Issue: This isn't scalable. Consider you had 6 months of daily data. Thats around 126 data points. The conditions below would classify this as WEEKLY which is incorrect producing
            wildly incorrect annualized metrics. A similar situation when you consider 2 years of weekly data which is around 104 data points.

            Solution: Pass the frequency to the compute_metrics() function directly e.g.: def compute_metrics(returns, frequency="daily", risk_free_rate=0.05)
    """
    n_periods = len(returns)
    if frequency and frequency in FREQUENCY_TO_PERIODS:
        periods_per_year = FREQUENCY_TO_PERIODS[frequency]
    else:
        # Fallback heuristic when frequency is not provided
        if n_periods > 200:  # daily
            periods_per_year = 252
        elif n_periods > 40:  # weekly
            periods_per_year = 52
        else:  # monthly
            periods_per_year = 12

    # Annualized return
    years = n_periods / periods_per_year
    """
        REVIEW (Improvement):
            The ann_return calculation below disproportionately amplifies the noise for shorter periods.
            To my understanding, ann_return converts a cumulative return over some period into "what it would be over a full year" - 
            When `years` is 1, ann_return = cum_return. But when years < 1 e.g. weekly or daily, the exponent becomes larger and amplifies small returns.
            For example if you made 3% over 2 months -> ann_return = (1 + 0.03) ** (1 / (2/12)) - 1 which is 19.4%. So 3% gets projected to 19.4% which is wildly wrong.
            Then this gets fed into the sharpe ratio which is used for model evaluation making it unreliable.
    """
    ann_return = (1 + cum_return) ** (1 / max(years, 0.01)) - 1

    # Annualized volatility
    ann_vol = returns.std() * np.sqrt(periods_per_year) # How much returns fluctuate / risk

    # Sharpe ratio (annualized)
    """
        REVIEW (Note):
            Why is the risk free rate held at a constant 5%? This rate changes over time. We should consider pulling real data
            from the US treasury (https://www.wallstreetprep.com/knowledge/risk-free-rate/)
    """
    excess_return = ann_return - risk_free_rate # Risk-free rate = return you'd get with zero risk (e.g. buying government bonds)
    sharpe = excess_return / ann_vol if ann_vol > 0 else 0.0

    # Maximum drawdown
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.cummax()
    drawdowns = cum_returns / rolling_max - 1
    max_drawdown = drawdowns.min()

    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(periods_per_year) if len(downside_returns) > 0 else ann_vol
    sortino = excess_return / downside_std if downside_std > 0 else 0.0

    # Win rate
    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0.0

    return {
        "cumulative_return": float(cum_return),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "sortino_ratio": float(sortino),
        "win_rate": float(win_rate),
        "num_periods": int(n_periods),
        "annualized_return": float(ann_return),
        "annualized_volatility": float(ann_vol),
    }