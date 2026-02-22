# Resources Catalog

Research project: **Refining LLM Trading**
Hypothesis: LLM agents may perform better in finance when making longer-term trading decisions rather than optimizing for short-term, day-to-day profit.

---

## Summary

| Resource Type | Count | Location |
|---------------|-------|----------|
| Papers downloaded | 23 | `papers/` |
| Papers deep-read | 10 | `papers/notes/` |
| Datasets downloaded | 3 | `datasets/` |
| Datasets documented | 8+ | `datasets/README.md` |
| Repositories cloned | 6 | `code/` |

---

## Papers

Total papers downloaded: 23

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| FINSABER | Li et al. | 2026 | `finsaber_2505_07078.pdf` | 20-year backtesting; LLM daily trading fails long-term |
| FinMem | Yu et al. | 2023 | `finmem_2311_13743.pdf` | Layered memory architecture; 148 citations |
| FinAgent | Li et al. | 2024 | `finagent_2402_18485.pdf` | Multimodal foundation agent; 126 citations |
| LLM Finance Survey | Li et al. | 2024 | `llm_finance_survey_2406_11903.pdf` | Comprehensive survey; 126 citations |
| FinCon | Yu et al. | 2024 | `fincon_2407_06567.pdf` | Multi-agent with verbal reinforcement; NeurIPS 2024 |
| TradingAgents | Xiao et al. | 2025 | `tradingagents_2412_20138.pdf` | Trading firm multi-agent; 92 citations |
| TradingGPT | Li et al. | 2023 | `tradinggpt_2309_03736.pdf` | Multi-agent with debate; 86 citations |
| LLM Trading Survey | Ding et al. | 2024 | `llm_trading_survey_2408_06361.pdf` | First trading agent survey; 53 citations |
| Sim Financial Market | Wang et al. | 2024 | `sim_financial_market_2406_19966.pdf` | LLM market simulation; 29 citations |
| InvestorBench | Li et al. | 2024 | `investorbench_2412_18174.pdf` | Multi-asset benchmark; ACL 2025 |
| Auto Strategy | Kang et al. | 2024 | `auto_strategy_2409_06289.pdf` | LLM quant strategy finding |
| StockAgent | Zhang et al. | 2024 | `stockagent_2407_18957.pdf` | Simulated real-world trading |
| FinVision | Wang et al. | 2024 | `finvision_2411_08899.pdf` | Multi-agent stock prediction |
| FLAG-Trader | Li et al. | 2025 | `flag_trader_2502_11433.pdf` | RL fine-tuned LLM beats GPT-4 |
| DL to LLM Survey | Multiple | 2025 | `dl_to_llm_survey_2503_21422.pdf` | AI in quant investment survey |
| MarketSenseAI 2.0 | Fatouros et al. | 2025 | `marketsenseai2_2502_00415.pdf` | LLM stock analysis agents |
| TradExpert | Shan et al. | 2024 | `tradexpert_2411_00782.pdf` | Mixture of expert LLMs |
| Can LLMs Trade | Lopez-Lira | 2025 | `can_llms_trade_2504_10789.pdf` | Financial theory testing |
| Trading-R1 | Zhao et al. | 2025 | `trading_r1_2504_17727.pdf` | LLM reasoning via RL |
| DeepFund | HKUST | 2025 | `deepfund_2505_11065.pdf` | Live trading benchmark; NeurIPS 2025 |
| MASS | Multiple | 2025 | `mass_2505_10278.pdf` | Multi-agent portfolio scaling |
| StockBench | Chen et al. | 2025 | `stockbench_2510_02209.pdf` | Contamination-free benchmark |
| FinArena | Multiple | 2025 | `finarena_2503_02692.pdf` | Human-agent collaboration |

See `papers/README.md` for detailed descriptions.

---

## Datasets

Total datasets downloaded: 3

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| Stock Prices (Daily) | Yahoo Finance | 22 tickers, 2515 rows each | Multi-frequency trading | `datasets/stock_prices/` | 10 years daily OHLCV |
| Stock Prices (Weekly) | Yahoo Finance | 10 tickers, 522 rows each | Weekly trading | `datasets/stock_prices_weekly/` | 10 years weekly |
| Stock Prices (Monthly) | Yahoo Finance | 10 tickers, 120 rows each | Monthly trading | `datasets/stock_prices_monthly/` | 10 years monthly |
| Financial Classification | HuggingFace | 5,057 examples | Sentiment | `datasets/financial_classification/` | Train/test split |

### Critical Datasets (Download for Experiments)

| Name | Source | Size | Priority |
|------|--------|------|----------|
| FINSABER Full | Google Drive via GitHub | 10.23 GB | **CRITICAL** |
| FNSPID | HuggingFace | ~30M records | HIGH |
| financial-news-multisource | HuggingFace | 57.1M rows | MEDIUM |

See `datasets/README.md` for download instructions.

---

## Code Repositories

Total repositories cloned: 6

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| FINSABER | github.com/waylonli/FINSABER | 20-year backtesting framework | `code/FINSABER/` | Primary framework |
| InvestorBench | github.com/felis33/INVESTOR-BENCH | Multi-asset LLM benchmark | `code/InvestorBench/` | Agent architecture |
| TradingAgents | github.com/TauricResearch/TradingAgents | Multi-agent trading | `code/TradingAgents/` | Multi-agent design |
| FinMem | github.com/pipiku915/FinMem-LLM-StockTrading | Layered memory agent | `code/FinMem/` | Memory architecture |
| DeepFund | github.com/HKUSTDial/DeepFund | Live trading benchmark | `code/DeepFund/` | Live evaluation |
| FinRL | github.com/AI4Finance-Foundation/FinRL | RL for finance library | `code/FinRL/` | DRL baselines |

See `code/README.md` for detailed descriptions.

---

## Resource Gathering Notes

### Search Strategy
1. Used paper-finder with diligent mode on two queries: "LLM agents financial trading stock market" and "long-term investment horizon LLM decision making finance"
2. Found 367 unique papers, 73 with relevance score ≥ 2
3. Downloaded top 23 papers based on relevance and citation count
4. Deep-read 10 most relevant papers with full chunk-by-chunk analysis
5. Identified datasets and code from paper methods sections
6. Searched HuggingFace, GitHub, and Papers with Code for additional resources

### Selection Criteria
- Papers: Prioritized work directly testing LLM trading agents, especially those addressing trading frequency, long-term evaluation, or market regime effects
- Datasets: Focused on multi-year coverage enabling multi-frequency comparison; bias-aware datasets preferred
- Code: Selected frameworks that could serve as baselines or be extended for our experiments

### Challenges Encountered
- Several arXiv IDs from paper-finder were incorrect; resolved by web search for correct IDs
- Financial PhraseBank dataset has a deprecated loading script on HuggingFace; documented alternative access
- FINSABER's 10.23 GB dataset too large for immediate download; documented download instructions
- InvestorBench GitHub organization changed; found correct URL via web search

### Gaps and Workarounds
- No paper systematically compares different trading frequencies — this is our core contribution
- Real-time news data requires API keys (Alpaca, Finnhub) — documented free tiers
- Financial PhraseBank loading script deprecated — alternatives documented in dataset_search_results.md

---

## Recommendations for Experiment Design

### 1. Primary Dataset
**FINSABER S&P 500 Full** (10.23 GB): The most comprehensive dataset with 20-year coverage, integrated price+news+filings, and built-in bias mitigation. Provides the foundation for testing our hypothesis across multiple market regimes.

### 2. Baseline Methods
| Category | Methods | Source |
|----------|---------|--------|
| Passive | Buy-and-Hold | Standard |
| Traditional | SMA Crossover, ARIMA | FINSABER |
| ML | XGBoost | FINSABER |
| DRL | PPO, A2C, SAC | FinRL |
| LLM (daily) | FinMem agent | FINSABER/FinMem |

### 3. Evaluation Metrics
- Cumulative Return (CR)
- Sharpe Ratio (SR)
- Maximum Drawdown (MDD)
- Sortino Ratio
- CAPM Alpha
- Paired t-tests between frequency conditions

### 4. Code to Adapt/Reuse
1. **FINSABER** backtesting pipeline → extend for weekly/monthly decisions
2. **FinMem** layered memory → adapt decay rates for different frequencies
3. **FinRL** DRL agents → standard baselines at all frequencies
4. **DeepFund** multi-agent architecture → reference for agent design

### 5. Experimental Design
- **Independent variable:** Decision frequency (daily, weekly, monthly)
- **Dependent variables:** CR, SR, MDD, Alpha
- **Control:** Same LLM backbone, same stocks, same time period, same information set
- **Evaluation period:** Minimum 5 years, covering at least 1 bull and 1 bear market
- **Statistical testing:** Paired t-tests or Wilcoxon between frequency conditions
