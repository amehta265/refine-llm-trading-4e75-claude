# Datasets

Datasets for the research project: "Refining LLM Trading."

Data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset 1: Stock Price Data (Daily, Weekly, Monthly)

### Overview
- **Source:** Yahoo Finance via yfinance Python package
- **Size:** 22 tickers × 3 frequencies = ~66 CSV files
- **Format:** CSV with Date, Open, High, Low, Close, Adj Close, Volume
- **Task:** Multi-frequency trading evaluation
- **Tickers:** AAPL, MSFT, AMZN, GOOGL, TSLA, NFLX, NVDA, META, JPM, BAC, COIN, JNJ, KO, CVX, UNH, WMT, V, PG, HD, DIS, SPY, QQQ
- **Time period:** 2016-02-22 to 2026-02-20 (10 years daily), same range weekly/monthly
- **License:** Yahoo ToS (research use)

### Download Instructions

```python
import yfinance as yf
import os

tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "NFLX", "NVDA", "META",
           "JPM", "BAC", "COIN", "JNJ", "KO", "CVX", "UNH", "WMT", "V", "PG",
           "HD", "DIS", "SPY", "QQQ"]

# Daily
os.makedirs("datasets/stock_prices", exist_ok=True)
for t in tickers:
    data = yf.download(t, period="10y", interval="1d")
    data.to_csv(f"datasets/stock_prices/{t}_daily.csv")

# Weekly (top 10)
os.makedirs("datasets/stock_prices_weekly", exist_ok=True)
for t in tickers[:10]:
    data = yf.download(t, period="10y", interval="1wk")
    data.to_csv(f"datasets/stock_prices_weekly/{t}_weekly.csv")

# Monthly (top 10)
os.makedirs("datasets/stock_prices_monthly", exist_ok=True)
for t in tickers[:10]:
    data = yf.download(t, period="10y", interval="1mo")
    data.to_csv(f"datasets/stock_prices_monthly/{t}_monthly.csv")
```

### Loading
```python
import pandas as pd
df = pd.read_csv("datasets/stock_prices/AAPL_daily.csv", parse_dates=["Date"], index_col="Date")
```

---

## Dataset 2: Financial Classification (Sentiment)

### Overview
- **Source:** HuggingFace `nickmuchi/financial-classification`
- **Size:** 4,551 train + 506 test examples
- **Format:** HuggingFace Dataset (Arrow)
- **Task:** Financial news sentiment classification
- **License:** Research use

### Download Instructions
```python
from datasets import load_dataset
ds = load_dataset("nickmuchi/financial-classification")
ds.save_to_disk("datasets/financial_classification")
```

### Loading
```python
from datasets import load_from_disk
ds = load_from_disk("datasets/financial_classification")
```

---

## Dataset 3: FINSABER Integrated Dataset (CRITICAL — Download Separately)

### Overview
- **Source:** https://github.com/waylonli/FINSABER (Google Drive links)
- **Size:** 10.23 GB (S&P 500 Full), 253 MB (Price Only), 48.1 MB (Selected Symbols)
- **Format:** Python dictionary serialized by date
- **Contents:** Daily prices (7,000+ US equities, 2000-2024), FNSPID news (15.7M records), SEC 10-K/10-Q filings
- **License:** CC BY 4.0

### Download Instructions
1. Visit https://github.com/waylonli/FINSABER
2. Follow Google Drive download links in the README
3. Choose appropriate size:
   - **Selected Symbols** (48.1 MB): TSLA, AMZN, MSFT, NFLX, COIN — for quick prototyping
   - **Price Only** (253 MB): All 7,000+ US equities price data only
   - **S&P 500 Full** (10.23 GB): Complete price + news + filings — for full experiments

### Loading
```python
# See FINSABER repo for data loader examples
# Reference: backtest/data_util/finmem_dataset.py
```

---

## Dataset 4: FNSPID (Financial News + Stock Prices)

### Overview
- **Source:** https://huggingface.co/datasets/Zihan1004/FNSPID
- **Size:** 29.7M price records + 15.7M news articles
- **Format:** Parquet
- **Time period:** 1999-2023
- **Coverage:** 4,775 S&P 500 companies
- **License:** CC BY-NC 4.0

### Download Instructions
```python
from datasets import load_dataset
ds = load_dataset("Zihan1004/FNSPID")
ds.save_to_disk("datasets/fnspid")
```

---

## Notes
- Stock price data is already downloaded locally (daily/weekly/monthly)
- Financial classification dataset is already downloaded locally
- FINSABER and FNSPID should be downloaded when running experiments
- See `dataset_search_results.md` for comprehensive search results and additional sources
