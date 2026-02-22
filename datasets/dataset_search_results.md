# Dataset Search Results: LLM Agents for Financial Trading

**Research focus:** Comparing short-term vs long-term trading horizons for LLM-based trading agents
**Date compiled:** 2026-02-22

---

## Table of Contents

1. [Summary and Recommendations](#1-summary-and-recommendations)
2. [Stock Price Data (OHLCV)](#2-stock-price-data-ohlcv)
3. [Financial News Datasets](#3-financial-news-datasets)
4. [SEC Filings and Fundamental Data](#4-sec-filings-and-fundamental-data)
5. [Integrated Datasets (Multi-Modal)](#5-integrated-datasets-multi-modal)
6. [Benchmarks from Reviewed Papers](#6-benchmarks-from-reviewed-papers)
7. [Sentiment Analysis Datasets](#7-sentiment-analysis-datasets)
8. [Data Access Summary Table](#8-data-access-summary-table)

---

## 1. Summary and Recommendations

### Recommended Primary Data Stack

For our research on comparing short-term vs long-term trading horizons, the following combination provides the most comprehensive coverage:

| Data Need | Recommended Source | Backup Source |
|-----------|-------------------|---------------|
| **Daily OHLCV prices** | yfinance (Yahoo Finance) | Alpha Vantage API |
| **Financial news** | FNSPID (HuggingFace) + Alpaca News API | financial-news-multisource (HuggingFace) |
| **SEC filings (10-K, 10-Q)** | SEC EDGAR via edgartools | sec-api Python package |
| **Sentiment labels** | financial_phrasebank + FinBERT | GPT-generated labels |
| **Integrated benchmark** | FINSABER (GitHub) | InvestorBench |

### Why This Stack

1. **yfinance** provides 20+ years of daily OHLCV for any US-traded stock, free and unrestricted for research. It supports daily, weekly, and monthly aggregation natively, which is essential for our multi-frequency analysis.

2. **FNSPID** offers 15.7 million time-aligned news records for 4,775 S&P 500 companies (1999-2023), already paired with stock prices. This is the same dataset used by FINSABER.

3. **SEC EDGAR** via edgartools provides free programmatic access to all 10-K and 10-Q filings, which map directly to the intermediate and deep memory layers in FinMem-style architectures.

4. **FINSABER** provides the most comprehensive pre-built integrated dataset (price + news + filings) covering 2004-2024 with 7,000+ US equities and built-in bias mitigation.

---

## 2. Stock Price Data (OHLCV)

### 2.1 Yahoo Finance via yfinance

- **Source URL:** https://github.com/ranaroussi/yfinance
- **PyPI:** https://pypi.org/project/yfinance/
- **Size:** On-demand download; no fixed dataset size
- **Format:** pandas DataFrame (OHLCV columns)
- **Contents:** Daily Open, High, Low, Close, Adjusted Close, Volume, Dividends, Stock Splits
- **Time period:** Varies by ticker; typically 20-30+ years for major US equities
- **Coverage:** Global equities, ETFs, mutual funds, currencies, cryptocurrencies
- **How to access:**
  ```python
  pip install yfinance
  import yfinance as yf
  # Download 5+ years of daily data
  data = yf.download("AAPL", start="2019-01-01", end="2024-12-31")
  # Weekly data
  data_weekly = yf.download("AAPL", start="2019-01-01", end="2024-12-31", interval="1wk")
  # Monthly data
  data_monthly = yf.download("AAPL", start="2019-01-01", end="2024-12-31", interval="1mo")
  ```
- **Rate limits:** ~2,000 requests/hour (unofficial); no API key required
- **License:** Apache 2.0 (library); data intended for personal/research use per Yahoo ToS
- **Relevance:** PRIMARY source for price data. Supports daily/weekly/monthly frequencies natively, which is critical for our multi-horizon comparison. Used by FinMem, TradingAgents, InvestorBench, and most reviewed papers.

### 2.2 Alpha Vantage API

- **Source URL:** https://www.alphavantage.co/
- **Documentation:** https://www.alphavantage.co/documentation/
- **Size:** On-demand via API
- **Format:** JSON or CSV
- **Contents:** Daily, weekly, monthly, and intraday OHLCV; technical indicators; fundamental data
- **Time period:** 20+ years of historical data for most tickers
- **Coverage:** 200,000+ stock tickers across 20+ global exchanges
- **How to access:**
  ```python
  pip install alpha_vantage
  from alpha_vantage.timeseries import TimeSeries
  ts = TimeSeries(key='YOUR_API_KEY', output_format='pandas')
  data, meta = ts.get_daily(symbol='AAPL', outputsize='full')
  ```
- **Free tier limits:** 25 API requests/day, 5 requests/minute
- **Paid plans:** Start at $49.99/month for higher limits
- **License:** Free tier available; commercial use requires premium
- **Relevance:** BACKUP source for price data. Free tier is very limited (25 calls/day), which makes bulk data collection slow. However, it offers built-in technical indicator endpoints and fundamental data that yfinance lacks.

### 2.3 OHLCV-1m (HuggingFace)

- **Source URL:** https://huggingface.co/datasets/mito0o852/OHLCV-1m
- **Size:** Large (minute-level data)
- **Format:** Parquet
- **Contents:** 1-minute OHLCV candlestick data for thousands of US stocks
- **Time period:** 1992-2025
- **How to access:**
  ```python
  from datasets import load_dataset
  ds = load_dataset("mito0o852/OHLCV-1m")
  ```
- **Source of data:** Originally from Finnhub.io
- **Relevance:** Useful if we need intraday resolution (e.g., for testing intraday vs daily vs weekly). The minute-level granularity allows us to test trading at virtually any frequency. However, the dataset is very large and may require significant storage.

---

## 3. Financial News Datasets

### 3.1 FNSPID (Financial News and Stock Price Integration Dataset)

- **Source URL:** https://huggingface.co/datasets/Zihan1004/FNSPID
- **GitHub:** https://github.com/Zdong104/FNSPID_Financial_News_Dataset
- **Paper:** arXiv:2402.06698 (KDD 2024)
- **Size:** 29.7 million stock price records + 15.7 million news articles
- **Format:** Parquet files
- **Contents:**
  - Daily stock prices for 4,775 S&P 500 companies
  - Time-stamped financial news articles with stock ticker alignment
  - Fields: Date, Article_title, Stock_symbol, Url, Publisher
  - News sources: NASDAQ, Bloomberg, Reuters, Benzinga, Lenta
- **Time period:** January 1, 1999 to December 31, 2023 (~24 years)
- **How to access:**
  ```python
  from datasets import load_dataset
  dataset = load_dataset("Zihan1004/FNSPID")
  ```
- **License:** CC BY-NC 4.0 (non-commercial use only)
- **Relevance:** HIGHLY RELEVANT. This is the primary news dataset used by FINSABER for its 20-year backtesting. The time-alignment with stock prices and broad coverage (4,775 companies, 24 years) makes it ideal for our multi-horizon research. Already proven to work with LLM trading agent frameworks.

### 3.2 financial-news-multisource

- **Source URL:** https://huggingface.co/datasets/Brianferrell787/financial-news-multisource
- **Size:** 57.1 million rows across 24 subsets
- **Format:** Parquet shards (streamable)
- **Contents:**
  - Unified corpus of 24 public news datasets
  - Standardized schema: date (UTC ISO 8601), text (title + body), extra_fields (JSON with URLs, publishers, tickers, timing metadata)
  - Sources include: Bloomberg (2006-2013), Reuters (2007-2013), Benzinga, CNBC (2017-2020), Yahoo Finance (2017-2025), NYT (1990-2025), Reddit finance (2008-2025), S&P 500 headlines (2008-2024), HuffPost (2012-2018), and more
  - Trading-date safe: includes `date_trading` field (NYSE next-session open in UTC) to prevent look-ahead bias
- **Time period:** 1990-2025 (~35 years)
- **How to access:**
  ```python
  from datasets import load_dataset
  # Stream all data
  ds = load_dataset("Brianferrell787/financial-news-multisource",
                     data_files="data/*/*.parquet", split="train", streaming=True)
  # Load specific subset
  bloomberg = load_dataset("Brianferrell787/financial-news-multisource",
                           data_files="data/bloomberg_reuters/*.parquet",
                           split="train", streaming=True)
  ```
- **License:** Non-commercial research/education only
- **Relevance:** HIGHLY RELEVANT. The largest and most comprehensive financial news corpus available on HuggingFace. The built-in look-ahead bias prevention (date_trading field) is particularly valuable for backtesting. Covers a much longer time span than FNSPID and includes social media data.

### 3.3 Alpaca News API

- **Source URL:** https://docs.alpaca.markets/docs/streaming-real-time-news
- **Documentation:** https://docs.alpaca.markets/reference/news-3
- **Size:** On-demand via API; historical news dating back to 2015
- **Format:** JSON (REST API) or WebSocket (streaming)
- **Contents:**
  - Real-time and historical news articles
  - Fields: author, created_at, updated_at, headline, summary, content, images, URL, symbols, source
  - News provided by Benzinga
- **Time period:** 2015-present
- **How to access:**
  ```python
  pip install alpaca-trade-api
  from alpaca_trade_api.rest import REST
  api = REST('API_KEY', 'SECRET_KEY')
  news = api.get_news(symbol='AAPL', start='2023-01-01', end='2023-12-31')
  ```
- **Free tier:** 200 calls/minute; requires free Alpaca account
- **Relevance:** Used by FinMem as its primary news source. Good for real-time or recent historical data, but the 2015 start date limits long-term backtesting. Best used as a complement to FNSPID for more recent data.

### 3.4 ashraq/financial-news-articles

- **Source URL:** https://huggingface.co/datasets/ashraq/financial-news-articles
- **Size:** ~300K+ articles
- **Format:** HuggingFace Dataset
- **Contents:** Financial news articles with full text
- **Relevance:** Smaller corpus, useful for prototyping and testing pipelines before scaling to FNSPID or financial-news-multisource.

### 3.5 Reuters Financial News

- **Source URL:** https://huggingface.co/datasets/danidanou/Reuters_Financial_News
- **Size:** 105,359 articles
- **Format:** HuggingFace Dataset
- **Contents:** Financial news articles originally sourced from Reuters
- **Time period:** 2006-2013
- **Relevance:** A clean, focused financial news corpus from a single trusted source. Limited time period but high quality. Already included in the financial-news-multisource compilation.

---

## 4. SEC Filings and Fundamental Data

### 4.1 SEC EDGAR (Official)

- **Source URL:** https://www.sec.gov/search-filings/edgar-application-programming-interfaces
- **Bulk data:** https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data
- **Size:** Millions of filings (all public US company filings since 1993)
- **Format:** HTML, XBRL, XML, JSON (via APIs)
- **Contents:**
  - 10-K (annual reports), 10-Q (quarterly reports), 8-K (current reports)
  - XBRL financial statements
  - All 150+ filing types
  - Company facts (standardized financial data via XBRL)
- **Time period:** 1993-present
- **How to access:**
  - Direct download (bulk): companyfacts.zip, submissions.zip
  - REST API: 10 requests/second (requires User-Agent header with email)
  - EDGAR Full-Text Search: https://efts.sec.gov/LATEST/search-index
- **Rate limits:** 10 requests/second, no API key required (just User-Agent identification)
- **License:** Public domain (US government data)
- **Relevance:** ESSENTIAL for fundamental data. 10-K and 10-Q filings are used by FinMem (deep and intermediate memory layers), FINSABER, and InvestorBench. Free, comprehensive, and authoritative.

### 4.2 edgartools (Python Library)

- **Source URL:** https://github.com/dgunning/edgartools
- **PyPI:** https://pypi.org/project/edgartools/
- **Documentation:** https://edgartools.readthedocs.io/
- **Format:** Python objects, pandas DataFrames
- **Contents:** Programmatic access to all SEC EDGAR filings, XBRL financial statements, insider trading data
- **How to access:**
  ```python
  pip install edgartools
  from edgar import *
  set_identity("researcher@university.edu")

  # Get 10-K filings for Apple
  company = Company("AAPL")
  filings = company.get_filings(form="10-K")

  # Get financial statements
  filing = filings[0]
  financials = filing.xbrl().financials
  ```
- **License:** MIT (open source)
- **Relevance:** RECOMMENDED tool for accessing SEC filings. Designed specifically for AI/LLM workflows. Extracts structured financial data from XBRL in a few lines of code. Supports all filing types (10-K, 10-Q, 8-K, 13F, Form 4).

### 4.3 sec-api (Python Package)

- **Source URL:** https://github.com/janlukasschroeder/sec-api-python
- **PyPI:** https://pypi.org/project/sec-api/
- **Documentation:** https://sec-api.io/
- **Contents:** 18+ million filings, all 150 filing types, XBRL-to-JSON conversion
- **How to access:**
  ```python
  pip install sec-api
  from sec_api import QueryApi
  queryApi = QueryApi(api_key="YOUR_API_KEY")
  query = {"query": {"query_string": {"query": "formType:\"10-K\" AND ticker:AAPL"}}}
  filings = queryApi.get_filings(query)
  ```
- **Free tier:** 20 filings/second download (no API key needed for downloads); search API requires key
- **Relevance:** Alternative to edgartools with a more API-oriented approach. The XBRL-to-JSON converter is particularly useful for extracting standardized financial statements.

---

## 5. Integrated Datasets (Multi-Modal)

### 5.1 FINSABER Dataset

- **Source URL:** https://github.com/waylonli/FINSABER
- **Paper:** arXiv:2505.07078 (KDD 2026)
- **Size:** Three download options via Google Drive:
  - S&P 500 Full (Price + News + Filings): **10.23 GB**
  - Price Only (CSV): **253 MB**
  - Selected Symbols (TSLA, AMZN, MSFT, NFLX, COIN): **48.1 MB**
- **Format:** Python dictionary serialized format, keyed by date
- **Contents:**
  - Daily stock prices for 7,000+ US equities (including delisted S&P 500 symbols)
  - Financial news: 15.7 million records for 4,775 S&P 500 companies (from FNSPID)
  - SEC filings: 10-K and 10-Q for Russell 3000 companies
  - Data structure per date:
    ```python
    {
      datetime.date(2024,1,1): {
        "price": {...},
        "news": {...},
        "filing_k": {...},   # 10-K annual filings
        "filing_q": {...}    # 10-Q quarterly filings
      }
    }
    ```
- **Time period:** 2000-2024 (prices), 1999-2023 (news), varies for filings
- **How to access:**
  1. Download from Google Drive links provided in the GitHub repo
  2. Implement custom data loader by subclassing `BacktestDataset`
  3. Reference example: `backtest/data_util/finmem_dataset.py`
- **License:** CC BY 4.0 (Creative Commons Attribution)
- **Bias mitigation features:**
  - Includes delisted symbols (prevents survivorship bias)
  - Historical S&P 500 constituent lists (prevents look-ahead bias)
  - Rolling-window evaluation with dynamically changing asset selections
- **Relevance:** THE MOST RELEVANT integrated dataset for our research. FINSABER was specifically designed for evaluating LLM trading strategies across long time horizons (20 years) with multi-source data. It includes the exact data types we need (prices, news, filings) and has built-in bias mitigation. The 20-year coverage enables testing daily, weekly, and monthly trading frequencies across multiple market regimes (bull, bear, sideways).

### 5.2 FinRL Data Layer

- **Source URL:** https://github.com/AI4Finance-Foundation/FinRL
- **Documentation:** https://finrl.readthedocs.io/
- **Size:** On-demand via integrated APIs
- **Format:** pandas DataFrames with standardized columns
- **Contents:**
  - OHLCV data with calculated technical indicators (MACD, Bollinger Bands, RSI, DX, SMA)
  - Integrated data processors for multiple exchanges
  - Train-test-trade pipeline
- **Supported data sources and time periods:**
  | Provider | Data | Resolution | Start |
  |----------|------|-----------|-------|
  | Yahoo Finance | US stocks, ETFs | Daily+ | Varies |
  | Alpaca | US stocks, ETFs | 1-minute | 2015 |
  | IEXCloud | US stocks | Daily | 1970 |
  | Binance/CCXT | Crypto | Second/minute | Varies |
  | WRDS | US stocks | Millisecond | 2003 |
  | Akshare/Baostock | Chinese stocks | Daily/minute | 2005 |
- **Markets covered:** NASDAQ-100, DJIA, S&P 500, HSI, SSE 50, CSI 300
- **How to access:**
  ```python
  pip install finrl
  from finrl.meta.data_processors.processor_yahoofinance import YahooFinanceProcessor
  dp = YahooFinanceProcessor()
  data = dp.download_data(start_date="2019-01-01", end_date="2024-01-01",
                          ticker_list=["AAPL","MSFT","AMZN"], time_interval="1D")
  ```
- **License:** MIT
- **Relevance:** Useful as a data processing layer rather than a standalone dataset. FinRL's data processors handle cleaning, feature engineering (technical indicators), and train-test splitting. FINSABER uses FinRL for its RL baselines. Good for standardizing data preparation across experiments.

### 5.3 FinRL HuggingFace Datasets

- **NASDAQ 2013-2023:** https://huggingface.co/datasets/benstaf/nasdaq_2013_2023/
- **NASDAQ News Sentiment:** https://huggingface.co/datasets/benstaf/nasdaq_news_sentiment
- **Format:** Parquet/CSV on HuggingFace
- **Contents:** NASDAQ stock data with optional news sentiment scores
- **Relevance:** Pre-processed datasets ready for use with FinRL. Useful for quick prototyping.

---

## 6. Benchmarks from Reviewed Papers

### 6.1 StockBench

- **Source URL:** https://github.com/ChenYXxxx/stockbench
- **Project page:** https://stockbench.github.io/
- **Paper:** arXiv:2510.02209
- **Contents:**
  - Back-trading environment for 20 DJIA stocks
  - Daily OHLCV prices, fundamental indicators (market cap, P/E, dividend yield), news articles
  - Data sources: Polygon API (market data), Finnhub API (news)
  - Agent workflow: Portfolio Overview -> In-Depth Analysis -> Decision -> Execution
- **Time period:** March 3 to June 30, 2025 (82 trading days)
- **Key feature:** Contamination-free evaluation (uses post-LLM-training-cutoff data)
- **How to access:**
  ```bash
  git clone https://github.com/ChenYXxxx/stockbench
  # Requires: Polygon API key, Finnhub API key, LLM provider API key
  ```
- **License:** Apache 2.0
- **Relevance:** Provides a clean, contamination-free benchmark framework. However, its evaluation period (4 months) is too short for our long-term analysis. The agent workflow design (4-stage decision process) is a useful reference architecture. Polygon and Finnhub have free tiers.

### 6.2 InvestorBench

- **Paper:** arXiv:2412.18174 (ACL 2025)
- **Contents:**
  - Stock trading: 7 stocks with OHLCV (Yahoo Finance) + news + SEC filings + sentiment labels
  - Cryptocurrency trading: BTC, ETH with OHLCV (CoinMarketCap) + crypto news
  - ETF trading: NIFTY dataset with curated daily news headlines
  - Multi-modal data warehouse: Yahoo Finance + SEC EDGAR + third-party APIs
- **Time periods:**
  - Stock trading: 2020-07-01 to 2021-05-06
  - Crypto trading: 2023-02-11 to 2023-11-05
  - ETF trading: 2019-07-29 to 2020-09-21
- **License:** MIT
- **Relevance:** Provides a standardized framework for evaluating LLM agents across asset classes (stocks, crypto, ETFs). The layered memory architecture (shallow=14d, intermediate=90d, deep=365d) from FinMem is built into the benchmark. Limited time periods (< 1 year each) but well-designed evaluation methodology.

### 6.3 AI-Trader

- **Source URL:** https://github.com/HKUDS/AI-Trader
- **Paper:** arXiv:2512.10971
- **Contents:**
  - Live, data-uncontaminated benchmark
  - Three markets: US stocks (NASDAQ 100), A-shares (SSE 50), cryptocurrencies
  - Multiple trading granularities (daily and hourly)
  - LLM inputs: trading rules, market data, account status, news
- **Key feature:** First fully-automated live evaluation benchmark
- **Relevance:** Covers multiple markets and trading granularities (daily vs hourly), which is relevant to our multi-frequency analysis. The cross-market evaluation (US, China, crypto) provides robustness.

### 6.4 StockNet Dataset

- **Source URL:** https://github.com/yumoxu/stocknet-dataset
- **Paper:** ACL 2018 - "Stock Movement Prediction from Tweets and Historical Prices"
- **Size:** 2 years of data for 88 stocks
- **Format:** CSV (prices), JSON (tweets)
- **Contents:**
  - Daily OHLCV prices: date, open, high, low, close, adjusted close, volume
  - Twitter data: tweets related to the 88 stocks
  - 88 stocks: top 10 by market cap from each of 8 S&P 500 sectors + all 8 Conglomerates stocks
- **Time period:** January 1, 2014 to January 1, 2016
- **How to access:**
  ```bash
  git clone https://github.com/yumoxu/stocknet-dataset
  ```
- **License:** Available for research
- **Relevance:** MODERATE. Classic dataset for stock movement prediction with social media. The 2-year period is short but the sector-diverse stock selection (88 stocks across 9 sectors) is well-designed. The Twitter data provides social sentiment signals. However, the data is now dated (2014-2016).

---

## 7. Sentiment Analysis Datasets

### 7.1 Financial PhraseBank

- **Source URL:** https://huggingface.co/datasets/takala/financial_phrasebank
- **Size:** 4,840 sentences
- **Format:** HuggingFace Dataset
- **Contents:**
  - English financial news sentences annotated with sentiment (positive, negative, neutral)
  - Annotated by 16 domain experts
  - Four configurations by annotator agreement level:
    - sentences_50agree: 4,846 instances
    - sentences_66agree: 4,217 instances
    - sentences_75agree: 3,453 instances
    - sentences_allagree: 2,264 instances
- **How to access:**
  ```python
  from datasets import load_dataset
  ds = load_dataset("takala/financial_phrasebank", "sentences_allagree")
  ```
- **License:** CC BY-NC-SA 3.0
- **Relevance:** Useful for training/fine-tuning sentiment classifiers that can be applied to our news data. The expert-annotated labels provide a gold standard for financial sentiment. Small size limits direct use but excellent for calibration and evaluation of sentiment models.

### 7.2 FinBERT (Pre-trained Sentiment Model)

- **Source URL:** https://github.com/ProsusAI/finBERT
- **HuggingFace:** https://huggingface.co/ProsusAI/finbert
- **Contents:** BERT model fine-tuned on financial text for sentiment analysis
- **How to access:**
  ```python
  from transformers import BertTokenizer, BertForSequenceClassification
  tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
  model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')
  ```
- **Relevance:** Pre-trained model that can be applied to generate sentiment scores for any financial text. Can be used to enrich our news data with sentiment labels for LLM agent consumption.

### 7.3 FinBen / PIXIU Benchmark

- **Source URL:** https://github.com/The-FinAI/PIXIU
- **HuggingFace:** Multiple datasets under TheFinAI organization
- **Paper:** NeurIPS 2024
- **Size:** 36 datasets spanning 24 financial tasks
- **Contents:**
  - Information extraction, textual analysis, question answering, text generation
  - Risk management, forecasting, and decision-making tasks
  - Includes stock trading evaluation dataset
  - RAG (Retrieval-Augmented Generation) evaluation tasks
- **Relevance:** Comprehensive financial NLP benchmark. The stock trading subset and forecasting tasks are directly relevant. Useful for evaluating our LLM's financial reasoning capabilities separate from trading performance.

---

## 8. Data Access Summary Table

| Dataset | Type | Size | Period | Free? | API Key? | Multi-Frequency? | Our Priority |
|---------|------|------|--------|-------|----------|------------------|-------------|
| **yfinance** | OHLCV prices | On-demand | 20+ years | Yes | No | Daily/Weekly/Monthly | **HIGH** |
| **Alpha Vantage** | OHLCV + fundamentals | On-demand | 20+ years | Free tier (25/day) | Yes | Daily/Weekly/Monthly/Intraday | MEDIUM |
| **FNSPID** | News + prices | 29.7M prices + 15.7M news | 1999-2023 | Yes (HF) | No | Daily (can aggregate) | **HIGH** |
| **financial-news-multisource** | News corpus | 57.1M rows | 1990-2025 | Yes (HF) | No | Daily | **HIGH** |
| **SEC EDGAR** | 10-K, 10-Q filings | Millions of filings | 1993-present | Yes | No | Quarterly/Annual | **HIGH** |
| **edgartools** | SEC filing access | N/A (tool) | 1993-present | Yes | No | N/A | **HIGH** |
| **FINSABER** | Integrated (price+news+filings) | 10.23 GB (full) | 2000-2024 | Yes (GDrive) | No | Daily (can aggregate) | **CRITICAL** |
| **FinRL** | Data processing framework | On-demand | Varies | Yes | Varies | All frequencies | MEDIUM |
| **Alpaca News API** | Financial news | On-demand | 2015-present | Free tier (200/min) | Yes | N/A | MEDIUM |
| **Financial PhraseBank** | Sentiment labels | 4,840 sentences | N/A | Yes (HF) | No | N/A | LOW |
| **StockBench** | Trading benchmark | 82 days | Mar-Jun 2025 | Yes (GitHub) | Yes (Polygon, Finnhub) | Daily only | LOW |
| **InvestorBench** | Trading benchmark | < 1 year per task | 2019-2023 | Yes | No | Daily only | MEDIUM |
| **AI-Trader** | Trading benchmark | Multi-market | Varies | Yes (GitHub) | Varies | Daily + Hourly | MEDIUM |
| **StockNet** | Prices + tweets | 88 stocks | 2014-2016 | Yes (GitHub) | No | Daily | LOW |
| **OHLCV-1m** | Minute-level prices | Large | 1992-2025 | Yes (HF) | No | All frequencies | LOW |

---

## 9. Dataset Combinations for Our Experiments

### Experiment 1: Short-term vs Long-term on Selected Stocks (Replication + Extension)

Replicate FINSABER's evaluation on the "Selected 4" (TSLA, NFLX, AMZN, MSFT) at different trading frequencies:

| Component | Source | Notes |
|-----------|--------|-------|
| Daily OHLCV | FINSABER dataset or yfinance | 2004-2024 |
| Financial news | FINSABER (FNSPID subset) | Pre-aligned with trading dates |
| 10-K filings | FINSABER or edgartools | Annual reports |
| 10-Q filings | FINSABER or edgartools | Quarterly reports |

### Experiment 2: Broad Universe with Bias Mitigation

Use FINSABER's composite setup with 100+ symbols and rolling windows:

| Component | Source | Notes |
|-----------|--------|-------|
| Price data | FINSABER S&P 500 Full (10.23 GB) | Includes delisted symbols |
| News data | FINSABER (FNSPID integrated) | 15.7M records |
| SEC filings | FINSABER (included) | Russell 3000 coverage |
| S&P 500 constituents | FINSABER (historical lists) | For survivorship bias mitigation |

### Experiment 3: Extended News Coverage

If we need more recent or broader news data:

| Component | Source | Notes |
|-----------|--------|-------|
| Historical news (pre-2023) | FNSPID via HuggingFace | 1999-2023, 4,775 companies |
| Extended news (1990-2025) | financial-news-multisource | 57.1M rows, 24 sources |
| Recent news (2023+) | Alpaca News API | Real-time + historical from 2015 |
| Sentiment scores | FinBERT model | Apply to any text corpus |

---

## 10. Key Observations and Caveats

### Data Quality Considerations

1. **Survivorship bias:** Most readily available stock lists (current S&P 500, DJIA) suffer from survivorship bias. FINSABER explicitly addresses this with historical constituent lists including delisted stocks. For any custom data collection, use historical index membership data.

2. **Look-ahead bias:** When pairing news with prices, ensure news timestamps precede trading decisions. The financial-news-multisource dataset includes a `date_trading` field for this purpose. FINSABER aligns all data to prevent look-ahead bias.

3. **Data leakage from LLM training:** LLMs like GPT-4 may have seen historical financial data during pre-training. FINSABER found that even with potential data leakage favoring LLMs, they still failed to outperform traditional strategies. StockBench addresses this by using only post-cutoff data.

4. **News coverage gaps:** FNSPID coverage ends in 2023. For testing on more recent periods, supplement with Alpaca News API or financial-news-multisource (extends to 2025).

5. **SEC filing processing:** Raw SEC filings are HTML/XBRL and require significant parsing. edgartools handles this automatically. FINSABER provides pre-processed filing data.

### Licensing and Usage Restrictions

| Dataset | License | Commercial Use |
|---------|---------|----------------|
| yfinance | Apache 2.0 (library) / Yahoo ToS (data) | Personal/research only (data) |
| FNSPID | CC BY-NC 4.0 | Non-commercial only |
| financial-news-multisource | Non-commercial research | Non-commercial only |
| FINSABER | CC BY 4.0 | Allowed with attribution |
| SEC EDGAR | Public domain | Unrestricted |
| Financial PhraseBank | CC BY-NC-SA 3.0 | Non-commercial only |
| StockBench | Apache 2.0 | Allowed |
| FinRL | MIT | Allowed |

### Data Collection Priority and Next Steps

1. **Immediate:** Download FINSABER S&P 500 Full dataset (10.23 GB) from Google Drive -- this provides the most comprehensive ready-to-use integrated dataset.
2. **Short-term:** Set up yfinance data pipeline for supplementary/recent price data and to support custom stock universe selection.
3. **Short-term:** Set up edgartools for on-demand SEC filing access beyond what FINSABER provides.
4. **Medium-term:** Download FNSPID and/or financial-news-multisource from HuggingFace for extended news coverage.
5. **Medium-term:** Register for Alpaca (free) and Alpha Vantage (free tier) API keys for supplementary data access.
6. **Optional:** Download StockBench code and set up Polygon/Finnhub API access for contamination-free evaluation on recent data.

---

## 11. Datasets Referenced in Our Reviewed Papers

Cross-reference of which datasets each reviewed paper uses:

| Paper | Price Source | News Source | Filings | Other |
|-------|-------------|-------------|---------|-------|
| **FINSABER** | 7,000+ US equities (2000-2024) | FNSPID (15.7M records) | 10-K, 10-Q (Russell 3000) | Historical S&P 500 constituents |
| **FinMem** | Yahoo Finance (yfinance) | Alpaca News API (Benzinga) | SEC 10-K, 10-Q | FAISS vector DB |
| **InvestorBench** | Yahoo Finance | Zhou et al. (2021) + Refinitiv | SEC 10-K, 10-Q | CoinMarketCap (crypto), NIFTY (ETF) |
| **TradingAgents** | Yahoo Finance | EODHD, Finnhub, Reddit, X/Twitter | Finnhub financials | Social media sentiment |
| **StockBench** | Polygon API | Finnhub News API | Fundamental indicators (P/E, market cap) | -- |
| **FinCon** | Yahoo Finance | Alpaca News API | SEC filings | Multi-agent communication |
| **AI-Trader** | Multiple (US, China, crypto) | Multiple | -- | Hourly + daily granularity |
| **DeepFund** | Yahoo Finance | -- | -- | Technical indicators |
| **FLAG-Trader** | Yahoo Finance | Financial news (unspecified) | -- | -- |

**Common thread:** Yahoo Finance (yfinance) is the de facto standard for price data. Alpaca News API and Finnhub are the most common news sources. SEC EDGAR is the standard for fundamental data. FINSABER provides the most comprehensive integrated package.
