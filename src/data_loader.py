"""
Data loading and preprocessing for multi-frequency LLM trading experiment.
Loads daily/weekly/monthly stock price data and computes technical indicators.
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "datasets"

# Stocks available at all 3 frequencies
TICKERS = ["AAPL", "AMZN", "BAC", "GOOGL", "JPM", "META", "MSFT", "NFLX", "NVDA", "TSLA"]

# Subset for main experiment (diverse sectors + volatility)
EXPERIMENT_TICKERS = ["AAPL", "JPM", "NVDA", "TSLA", "MSFT"]


def load_price_data(ticker: str, frequency: str) -> pd.DataFrame:
    """Load price data for a ticker at a given frequency.

    Args:
        ticker: Stock ticker symbol
        frequency: 'daily', 'weekly', or 'monthly'

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume, indexed by Date
    """
    if frequency == "daily":
        path = DATA_DIR / "stock_prices" / f"{ticker}_daily.csv"
    elif frequency == "weekly":
        path = DATA_DIR / "stock_prices_weekly" / f"{ticker}_weekly.csv"
    elif frequency == "monthly":
        path = DATA_DIR / "stock_prices_monthly" / f"{ticker}_monthly.csv"
    else:
        raise ValueError(f"Unknown frequency: {frequency}")

    # Read CSV with multi-level headers (Price/Ticker rows)
    df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)

    # Flatten multi-level columns — take the first level (Price type)
    df.columns = df.columns.get_level_values(0)

    # Rename and select columns
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.astype(float)
    df.index.name = "Date"
    df = df.sort_index()

    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators: SMA-10, SMA-20, RSI-14.

    Args:
        df: Price DataFrame with Close column

    Returns:
        DataFrame with added indicator columns
    """
    df = df.copy()

    # Simple Moving Averages
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()

    # RSI-14
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # Price change (%) over last period
    df["Return_1"] = df["Close"].pct_change() * 100

    # 5-period return
    df["Return_5"] = df["Close"].pct_change(5) * 100

    return df


def prepare_experiment_data(ticker: str, frequency: str,
                            start_date: str = "2024-01-01",
                            end_date: str = "2024-12-31") -> pd.DataFrame:
    """Load and prepare data for a single experiment condition.

    Args:
        ticker: Stock ticker
        frequency: 'daily', 'weekly', or 'monthly'
        start_date: Start of evaluation period
        end_date: End of evaluation period

    Returns:
        DataFrame with price data and indicators, filtered to evaluation period
    """
    df = load_price_data(ticker, frequency)
    df = compute_indicators(df)

    # Need extra history for indicators — load from before start_date
    # but only return rows within the evaluation period
    eval_mask = (df.index >= start_date) & (df.index <= end_date)
    df_eval = df[eval_mask].dropna()

    return df_eval


def format_price_history(df: pd.DataFrame, current_idx: int, lookback: int = 15) -> str:
    """Format recent price history as a string for the LLM prompt.

    Args:
        df: Full DataFrame with indicators
        current_idx: Integer position of current decision point
        lookback: Number of prior periods to include

    Returns:
        Formatted string of recent price data
    """
    start = max(0, current_idx - lookback)
    window = df.iloc[start:current_idx + 1]

    lines = []
    lines.append("Date       | Close    | Volume      | SMA10    | SMA20    | RSI14  | Ret1%  | Ret5%")
    lines.append("-" * 95)
    for date, row in window.iterrows():
        lines.append(
            f"{date.strftime('%Y-%m-%d')} | {row['Close']:8.2f} | {row['Volume']:11.0f} | "
            f"{row['SMA_10']:8.2f} | {row['SMA_20']:8.2f} | {row['RSI_14']:6.1f} | "
            f"{row['Return_1']:+6.2f} | {row['Return_5']:+6.2f}"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    # Quick test
    for freq in ["daily", "weekly", "monthly"]:
        df = prepare_experiment_data("AAPL", freq)
        print(f"\nAAPL {freq}: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
        print(f"  Close range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
        print(f"  Columns: {list(df.columns)}")
