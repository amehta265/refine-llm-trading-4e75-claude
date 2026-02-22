# Literature Review: Refining LLM Trading

**Research Hypothesis:** LLM agents may perform better in finance when making longer-term trading decisions rather than optimizing for short-term, day-to-day profit.

**Date:** 2026-02-22

---

## 1. Research Area Overview

The application of Large Language Models (LLMs) to financial trading has exploded since 2023, with over 70 relevant papers published in the past two years. These systems use LLMs as autonomous agents that process financial news, price data, SEC filings, and social media to generate buy/sell/hold trading decisions. The field has progressed from simple news-driven sentiment trading to sophisticated multi-agent systems with layered memory, reinforcement learning, and market simulation.

A critical gap in this literature is that **virtually all existing systems operate at daily trading frequency**, and **evaluation periods are typically very short** (median 1.3 years). No published work systematically compares LLM trading performance across different decision horizons (daily vs. weekly vs. monthly). This gap is the core motivation for our research.

---

## 2. Key Papers

### 2.1 FINSABER (Li et al., 2026) — *Most Directly Relevant*
- **Title:** Can LLM-based Financial Investing Strategies Outperform the Market in Long Run?
- **Source:** arXiv:2505.07078, KDD 2026
- **Key Contribution:** Introduces a bias-mitigated backtesting framework that evaluates LLM trading strategies over 20 years with 100+ stocks, addressing survivorship bias, look-ahead bias, and data-snooping bias.
- **Central Finding:** LLM strategies that appeared profitable in short-term evaluations (6 months, 4-8 hand-picked stocks) **consistently underperform Buy-and-Hold** when tested over 20 years. Even a 2-month extension caused FinMem's MSFT Sharpe to collapse from 1.44 to -1.25.
- **Regime Miscalibration:** LLMs are overly conservative in bull markets (FinMem Sharpe -0.19 vs B&H 0.61) and overly aggressive in bear markets (FinMem Sharpe -0.97, worst tested).
- **Datasets:** 7,000+ US equities (2000-2024), FNSPID news (15.7M records), SEC filings.
- **Code:** https://github.com/waylonli/FINSABER
- **Relevance:** Directly challenges daily LLM trading over long evaluation periods. However, it only tests daily *decision frequency*, not weekly/monthly. The diagnosed failure mode (regime miscalibration) actually supports testing longer decision horizons where LLMs could reason about macro trends.

### 2.2 FinMem (Yu et al., 2023)
- **Title:** A Performance-Enhanced LLM Trading Agent With Layered Memory and Character Design
- **Source:** arXiv:2311.13743, 148 citations
- **Key Contribution:** Layered memory architecture mapping to different time horizons: shallow (daily news, 14-day decay), intermediate (10-Q quarterly, 90-day decay), deep (10-K annual + reflections, 365-day decay).
- **Results:** TSLA 61.78% CR, 2.68 Sharpe over ~6 months. GPT-4-Turbo was best backbone.
- **Baselines:** Buy-and-Hold, PPO, DQN, A2C, FinGPT.
- **Limitations:** Only 6-month evaluation on 5 stocks. FINSABER later showed these results collapse over longer periods.
- **Code:** https://github.com/pipiku915/FinMem-LLM-StockTrading
- **Relevance:** The layered memory architecture explicitly encodes multi-horizon information, but the system still makes daily decisions. Key design pattern for our work.

### 2.3 FinCon (Yu et al., 2024)
- **Title:** A Synthesized LLM Multi-Agent System with Conceptual Verbal Reinforcement for Enhanced Financial Decision Making
- **Source:** arXiv:2407.06567, NeurIPS 2024, 97 citations
- **Key Contribution:** Manager-analyst hierarchy with Conceptual Verbal Reinforcement (CVRF) — prompt optimization via "textual gradient descent" that updates investment beliefs across episodes.
- **Results:** TSLA 82.87% CR, 1.972 SR. Outperformed in bearish markets (NIO: +17.46% vs B&H -77.21%).
- **Key Design:** 7 specialized analyst agents (textual, audio, data), dual-level risk control (CVaR + CVRF). Only 4 training episodes needed.
- **Code:** https://github.com/The-FinAI/FinCon
- **Relevance:** CVRF belief updates represent longer-term learning. The multi-agent structure could be adapted for different decision frequencies.

### 2.4 TradingAgents (Xiao et al., 2025)
- **Title:** Multi-Agents LLM Financial Trading Framework
- **Source:** arXiv:2412.20138, 92 citations
- **Key Contribution:** Mirrors real trading firm organization with analyst team, researcher debate (bull vs bear), trader, risk management team, and fund manager.
- **Results:** 23-27% CR with Sharpe 5.6-8.21 over 3 months on AAPL, GOOGL, AMZN.
- **Limitation:** Very short evaluation (3 months). 11 LLM calls + 20+ tool calls per daily decision make longer backtesting prohibitively expensive.
- **Code:** https://github.com/TauricResearch/TradingAgents
- **Relevance:** Computational cost highlights a key argument for less frequent decisions — weekly/monthly trading would reduce API costs by 5-20x while allowing more deliberate reasoning.

### 2.5 LLM Agent in Financial Trading: A Survey (Ding et al., 2024)
- **Title:** Large Language Model Agent in Financial Trading: A Survey
- **Source:** arXiv:2408.06361, 53 citations
- **Key Findings:**
  - Classifies approaches: news-driven, reflection-driven, debate-driven, RL-driven, and alpha mining.
  - Nearly all agents operate at daily frequency. HFT is ruled out by inference latency.
  - Median backtesting period is only 1.3 years; only 4/14 papers test >5 years.
  - LLM agents achieve 15-30% annualized excess returns in backtesting.
  - Trading costs almost universally ignored.
- **Relevance:** Identifies the systematic gap our research addresses: no paper compares different trading frequencies. Recommends longer, more diverse evaluation as a critical future direction.

### 2.6 FLAG-Trader (Li et al., 2025)
- **Title:** Fusion LLM-Agent with Gradient-based Reinforcement Learning for Financial Trading
- **Source:** arXiv:2502.11433, 17 citations
- **Key Contribution:** Shows a 135M-parameter LLM fine-tuned with PPO outperforms GPT-4 and GPT-o1 in agentic frameworks. Uses Sharpe ratio change as RL reward.
- **Results:** JNJ SR 3.344 (vs 1.343 B&H), BTC SR 1.734 (vs 0.683 B&H).
- **Relevance:** Demonstrates RL-driven optimization outperforms pure prompting. Suggests that the failure of LLM daily trading (FINSABER) might be addressable through RL fine-tuning, not just frequency changes.

### 2.7 DeepFund (2025)
- **Title:** Time Travel is Cheating: Going Live with DeepFund for Real-Time Fund Investment Benchmarking
- **Source:** arXiv:2505.11065, NeurIPS 2025
- **Key Contribution:** Live (real-time) fund investment benchmark eliminating backtesting data leakage. Tests 9 LLMs with $100K initial capital.
- **Central Finding:** Only 1/9 LLMs (Grok 3) achieved positive returns (+1.1%) in 24 trading days. Grok 3 succeeded via conservative cash management (~60% reserves), not superior prediction.
- **Extended Results:** Over Q2 2025, some models (GPT-4.1, Claude 3.7, DeepSeek-V3) eventually turned profitable.
- **Code:** https://github.com/HKUSTDial/DeepFund
- **Relevance:** The dramatic gap between backtested and live performance confirms data leakage concerns. Conservative, low-frequency strategies outperforming aggressive daily trading directly supports our hypothesis.

### 2.8 StockBench (Chen et al., 2025)
- **Title:** Can LLM Agents Trade Stocks Profitably In Real-world Markets?
- **Source:** arXiv:2510.02209
- **Key Contribution:** Contamination-free benchmark on 20 DJIA stocks using post-training-cutoff data (Mar-Jun 2025). Tests 14 LLMs.
- **Results:** Most LLMs marginally beat buy-and-hold (1-3% vs 0.4%) while managing risk better. Rankings shift dramatically between market regimes.
- **Code:** https://github.com/ChenYXxxx/stockbench
- **Relevance:** Regime-dependent performance (struggling in downturns, excelling in upturns) reinforces the need for regime-aware strategies, potentially enabled by longer decision horizons.

### 2.9 InvestorBench (Li et al., 2024)
- **Title:** A Benchmark for Financial Decision-Making Tasks with LLM-based Agent
- **Source:** arXiv:2412.18174, ACL 2025
- **Key Contribution:** First benchmark testing 13 LLMs across stocks, crypto, and ETFs.
- **Key Finding:** ETF trading (most strategic/long-term) was the hardest for most LLMs — only proprietary models succeeded.
- **Relevance:** Suggests longer-term decisions require stronger reasoning capabilities; sufficiently capable LLMs might excel at strategic decisions.

### 2.10 Can Large Language Models Trade? (Lopez-Lira, 2025)
- **Source:** arXiv:2504.10789
- **Key Contribution:** Simulated stock market with heterogeneous LLM agents testing financial theories.
- **Key Finding:** Asymmetric price discovery — undervaluation corrects but overvaluation persists. LLMs excel at value identification, which aligns with longer-term value investing.
- **Relevance:** Supports the notion that LLMs' strengths (fundamental analysis, value identification) map better to longer holding periods.

---

## 3. Common Methodologies

| Approach | Papers | Description |
|----------|--------|-------------|
| **News-driven prompting** | FinMem, MarketSenseAI | LLM processes news → sentiment → trading signal |
| **Reflection-driven** | FinMem, FinAgent, FinCon | Memory + self-improvement loops |
| **Multi-agent debate** | TradingAgents, TradingGPT, FinCon | Multiple agents with different roles deliberate |
| **RL-augmented** | FLAG-Trader, Trading-R1 | Fine-tune LLM policy via reinforcement learning |
| **Simulation-based** | Can LLMs Trade?, StockAgent | LLMs as agents in simulated markets |

---

## 4. Standard Baselines

- **Passive:** Buy-and-Hold (used in all papers)
- **Traditional quant:** SMA Crossover, MACD, Bollinger Bands, ATR Band, RSI+KDJ
- **ML models:** ARIMA, XGBoost, Random Forest, LSTM
- **DRL agents:** PPO, A2C, DQN, SAC, TD3, DDPG (via FinRL)
- **LLM baselines:** FinGPT, Generative Agents, FinMem (for later papers)

---

## 5. Evaluation Metrics

| Metric | Used By | Formula/Notes |
|--------|---------|---------------|
| Cumulative Return (CR) | All papers | Total percentage gain/loss |
| Sharpe Ratio (SR) | All papers | Risk-adjusted return |
| Maximum Drawdown (MDD) | Most papers | Worst peak-to-trough decline |
| Annualized Volatility (AV) | Several | Standard deviation of returns × √252 |
| Sortino Ratio | StockBench | Downside-risk-adjusted return |
| CAPM Alpha/Beta | FINSABER | Market-relative performance decomposition |
| Win Rate | DeepFund | Fraction of profitable trading days |

---

## 6. Datasets in the Literature

| Dataset/Source | Used By | Coverage |
|----------------|---------|----------|
| Yahoo Finance (yfinance) | FinMem, InvestorBench, TradingAgents, FinCon, DeepFund | Daily OHLCV, 20+ years |
| FNSPID | FINSABER | 15.7M news records, 1999-2023 |
| SEC EDGAR (10-K, 10-Q) | FinMem, InvestorBench, FinCon, FINSABER | All public US filings since 1993 |
| Alpaca News API | FinMem, FinCon | Benzinga news, 2015-present |
| Polygon API | StockBench | Real-time market data |
| Finnhub | TradingAgents, StockBench | News + financials |

---

## 7. Gaps and Opportunities

### Gap 1: No Systematic Frequency Comparison
**Every paper uses daily trading frequency.** No published work compares daily vs. weekly vs. monthly decision-making. This is our core research opportunity.

### Gap 2: Short Evaluation Windows
Median backtesting period is 1.3 years. When extended to 20 years (FINSABER), LLM advantages vanish. This suggests results are fragile and environment-specific.

### Gap 3: Regime Miscalibration
LLMs are consistently shown to be poor at detecting bull/bear market transitions (FINSABER, StockBench, DeepFund). Longer decision horizons could provide more context for regime detection.

### Gap 4: Computational Cost of Daily Trading
Multi-agent systems like TradingAgents require 11+ LLM calls per daily decision. Weekly or monthly decisions would reduce costs by 5-20x while allowing more thorough analysis per decision.

### Gap 5: Backtesting vs. Live Performance
DeepFund demonstrated a dramatic gap between backtested and live performance. The only successful live strategy used conservative, low-frequency trading.

---

## 8. Recommendations for Our Experiment

### Recommended Approach
1. **Primary dataset:** FINSABER (10.23 GB integrated dataset with 20-year coverage, bias mitigation)
2. **Supplementary prices:** yfinance for daily/weekly/monthly data at all three frequencies
3. **Experimental design:** Compare LLM trading agents at daily, weekly, and monthly decision frequencies on the same stocks and time periods
4. **Baselines:** Buy-and-Hold, SMA Crossover, ARIMA, PPO (via FinRL)
5. **Metrics:** CR, SR, MDD, CAPM Alpha, Sortino Ratio

### Key Design Decisions
- **LLM backbone:** GPT-4o-mini (cost-effective) and GPT-4o (capability) following FINSABER's approach
- **Agent architecture:** Adapt FinMem's layered memory for multi-frequency decisions
- **Evaluation period:** Minimum 5 years, ideally 10+ years across multiple market regimes
- **Bias mitigation:** Follow FINSABER's rolling-window approach with historical constituent lists
- **Statistical testing:** Paired t-tests or Wilcoxon signed-rank tests between frequency conditions

### Hypothesis Operationalization
- **H1:** LLM agents making weekly decisions achieve higher risk-adjusted returns (Sharpe ratio) than daily decisions over multi-year evaluation periods
- **H2:** LLM agents making monthly decisions outperform daily decisions in bear market regimes
- **H3:** Longer decision horizons reduce regime miscalibration (measured by bull/bear market performance differential)
