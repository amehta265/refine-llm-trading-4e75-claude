# FINSABER: Can LLM-based Financial Investing Strategies Outperform the Market in Long Run?

**Paper:** Li, Kim, Cucuringu, Ma (2026). arXiv:2505.07078v5. Published at KDD '26.
**Institutions:** University of Edinburgh, Sungkyunkwan University, UCLA, University of Oxford.
**GitHub:** https://github.com/waylonli/FINSABER

---

## 1. What is FINSABER?

FINSABER stands for **Financial INvesting Strategy Assessment with Bias mitigation, Expanded time, and Range of symbols**. It is a comprehensive backtesting framework specifically designed to evaluate LLM timing-based investing strategies with explicit bias mitigation.

### Core Problem Addressed
Most existing evaluations of LLM-based investing strategies are conducted on:
- **Narrow timeframes** (often under 1 year)
- **Limited stock universes** (fewer than 10 stocks, often hand-picked winners like TSLA, AMZN, NFLX, MSFT)
- **Without code release**, limiting reproducibility

This leads to three critical biases:
1. **Survivorship bias** -- only currently active/successful stocks are included; delisted/bankrupt stocks are omitted, systematically overstating returns
2. **Look-ahead bias** -- future information (e.g., using today's S&P 500 constituents as historical universe) inadvertently influences past decisions
3. **Data-snooping bias** (multiple testing bias) -- repeated experimentation on the same narrow dataset leads to overfitting and inflated false positive rates

### Framework Architecture (Three Core Modules)

**Module 1: Multi-source Data Module**
- Daily stock prices for 7,000+ US equities from 2000-2024
- Financial news: 15.7 million records for 4,775 S&P 500 companies (1999-2023)
- 10-K and 10-Q SEC filings for Russell 3000 companies
- Includes delisted symbols from historical S&P 500 constituent lists
- All data aligned with backtest windows using only information available prior to the start date (preventing look-ahead bias)

**Module 2: Modular Strategies Base**
- Covers timing-based and selection-based strategies across multiple paradigms
- Easily extensible for custom implementations

**Module 3: Bias-Aware Two-Step Backtesting Pipeline**
- Step 1: Selection-based strategies operate on historically accurate constituent lists (including delisted symbols) at each window
- Step 2: Timing-based strategies execute daily trading decisions on selected stocks
- Rolling-window evaluations over diverse, dynamically changing asset selections
- Window size and step are customizable

---

## 2. What LLM Strategies Are Tested?

### LLM-Based Timing Strategies (Primary Evaluation Targets)
- **FinMem** (Yu et al., 2023) -- Performance-enhanced LLM trading agent with layered memory and character design. Uses GPT-4o-mini (also tested with GPT-4o and originally reported results).
- **FinAgent** (Zhang et al., 2024) -- Multimodal foundation agent for financial trading; tool-augmented, diversified, generalist. Uses GPT-4o-mini.
- **FinGPT** -- Open-source financial LLM (tested in supplementary/FinCon comparison tables)
- **FinCon** (Yu et al., 2024) -- Synthesized LLM multi-agent system with conceptual verbal reinforcement (tested in FinCon selection comparison, Table 7)

### Summary of Existing LLM Strategies (Table 1 -- Showing Inadequate Evaluations)
| Method | Eval Period | Eval Symbols | Code Available |
|---|---|---|---|
| MarketSenseAI | 1 year 3 months | 100 | No |
| TradingGPT | N/A | N/A | No |
| FinMem | 6 months | 5 | Yes |
| FinAgent | 6 months | 6 | Yes |
| FinRobot | N/A | N/A | Yes |
| TradExpert | 1 year | 30 | No |
| FinCon | 8 months | 8 | No |
| TradingAgents | 3 months | 3 | No |
| MarketSenseAI 2.0 | 2 years | 100 | No |

### Traditional Baselines Tested
**Rule-Based:**
- Buy and Hold
- Simple Moving Average (SMA) Crossover
- Weighted Moving Average (WMA) Crossover
- ATR Band
- Bollinger Bands
- Trend Following
- Turn of the Month

**ML/DL Predictor-Based:**
- ARIMA (order 5,1,0)
- XGBoost

**RL-Based (via FinRL framework):**
- A2C (Advantage Actor-Critic)
- PPO (Proximal Policy Optimization)
- SAC (Soft Actor-Critic)
- TD3 (Twin Delayed DDPG)
- DDPG

### Selection-Based Strategies (for Composite Setup)
- **Random Five** -- random selection from S&P 500 constituents (91 unique symbols)
- **Momentum Factor** -- top-k by recent price momentum (84 unique symbols)
- **Volatility Effect** -- top-k lowest volatility stocks (63 unique symbols)
- **FinCon Selection Agent** -- diversified selection based on Sharpe + low correlation (80 unique symbols)

---

## 3. Time Horizons Evaluated

### SHORT-TERM vs. LONG-TERM -- THIS IS THE CENTRAL FINDING

**Short-Term (Originally Reported Period):**
- FinMem's original evaluation: 2022-10-06 to 2023-04-10 (approximately 6 months)
- FinCon's original evaluation: 2022-10-05 to 2023-06-10 (approximately 8 months)
- Only 4-8 stocks tested

**Extended Period (FINSABER):**
- 2004-01-01 to 2024-01-01 (20 years) for the same 4 symbols (TSLA, NFLX, AMZN, MSFT)
- Rolling windows: 2-year windows with 1-year step for Selected 4; 1-year windows with 1-year step for Composite setup
- Training data: up to 3 years prior (Selected 4) or 2 years prior (Composite)

### Critical Finding on Time Horizon Effects

**Short-term results (6 months, 4 stocks):** LLM strategies appear to outperform -- FinMem shows high Sharpe ratios on selected symbols (e.g., 2.679 SPR on TSLA reported, 2.017 on NFLX reported).

**Extended to 20 years, same 4 stocks:** LLM advantages **deteriorate significantly**:
- TSLA: FinMem still leads in AR (42.153%) but Buy and Hold has comparable SPR (0.630 vs 0.641)
- NFLX: Buy and Hold dominates with SPR 0.622 and AR 23.919%; FinMem drops to SPR 0.293
- AMZN: Buy and Hold wins (SPR 0.551); FinMem only SPR 0.188
- MSFT: Buy and Hold wins (SPR 0.461); FinMem only SPR 0.203

**Key quote:** "previously reported LLM advantages are likely short-lived, potentially hand-picked, and highly sensitive to the evaluation period"

**Even marginal 2-month extension causes instability:** Extending FinMem's evaluation by just 2 months beyond the original reported period caused:
- MSFT cumulative return: from +23.261% to -22.036%
- MSFT Sharpe ratio: from 1.440 to -1.247
- NFLX Sharpe ratio: from 2.017 to -0.478

**Composite setup (100+ symbols, 20 years, bias-mitigated):** LLM strategies consistently underperform:
- Random 5 selection: FinMem SPR -0.253, FinAgent SPR 0.094 vs Buy and Hold SPR 0.315
- Volatility Effect selection: FinMem SPR -0.228, FinAgent SPR 0.241 vs Buy and Hold SPR 0.703
- Momentum Factor: FinMem SPR 0.025, FinAgent SPR 0.104 vs Buy and Hold SPR 0.384

---

## 4. Market Regime Analysis

### Regime Classification
Each calendar year classified based on S&P 500 annual return (Ry):
- **Bull:** Ry >= +20%
- **Bear:** Ry <= -20%
- **Sideways:** -20% < Ry < +20%

### Regime-Specific Average Sharpe Ratios (Figure 2 -- CRITICAL RESULTS)

| Strategy | Bull | Sideways | Bear |
|---|---|---|---|
| Buy and Hold | **0.61** | **0.48** | -0.28 |
| ARIMA | **0.58** | 0.34 | 0.19 |
| ATR Band | 0.06 | 0.16 | 0.04 |
| Turn of the Month | 0.07 | -0.06 | 0.12 |
| RL-PPO | 0.30 | 0.33 | -0.25 |
| RL-SAC | 0.22 | 0.25 | -0.04 |
| **FinMem** | **-0.19** | **-0.10** | **-0.97** |
| **FinAgent** | **0.12** | **0.19** | **-0.38** |

### The Central Behavioral Finding: Regime Miscalibration

**LLM strategies are OVERLY CONSERVATIVE in bull markets:**
- FinMem: -0.19 Sharpe in bull markets (negative! -- actually loses money when the market is going up)
- FinAgent: 0.12 Sharpe in bull markets (far below Buy and Hold's 0.61)
- No active strategy surpasses Buy and Hold's passive SPR in the bull regime
- This means LLM agents are "too cautious when risk is rewarded"
- Agents fail to capitalize on strong uptrends, missing the equity beta

**LLM strategies are OVERLY AGGRESSIVE in bear markets:**
- FinMem: -0.97 Sharpe in bear markets (catastrophic -- worst of ANY strategy tested)
- FinAgent: -0.38 Sharpe in bear markets (worse than Buy and Hold's -0.28)
- Both are "too aggressive when it is penalized"
- Agents make pro-cyclical decisions that accelerate losses during downturns
- During 2008 GFC, FinMem's drawdown on DE approached -75% vs SPX benchmark's -50%

**This behavioral flaw contradicts the Adaptive Markets Hypothesis** -- the agents are pathologically miscalibrated, with decision-making policies fundamentally misaligned with market regimes.

### Drawdown Analysis (Appendix F -- Underwater Plots)
- **Bull markets:** FinAgent shows overly conservative posture (shallow drawdowns but missed upside). FinMem consistently fails to manage single-stock volatility with deeper, more prolonged drawdowns.
- **Bear markets (2008 GFC):** LLM strategies, especially FinMem, catastrophically amplify downside risk. FinMem drawdown on DE ~75% vs SPX ~50%.
- **Sideways markets:** FinAgent generally shows shallower drawdowns than FinMem but can experience complete inactivity (no trades triggered).

---

## 5. Key Results Summary

### 5.1 LLM Alpha is a Methodological Artefact
- Performance advantages reported in short-term, selective studies **vanish** under bias-mitigated backtests
- Neither FinMem nor FinAgent generates **statistically significant alpha** (all alpha p-values > 0.34)
- Consistent with the Efficient Market Hypothesis -- prior gains stemmed from survivorship and look-ahead biases, not genuine market inefficiency

### 5.2 Statistical Validation (Paired t-tests, Table 5)

**Selective period (4 stocks):** Statistical significance is inconsistent and limited to individual stocks.

**Composite setup (bias-mitigated):**
| Setup | B&H vs FinMem | B&H vs FinAgent | FinMem vs FinAgent |
|---|---|---|---|
| Random 5 | **3.0e-6** | **7.7e-4** | 4.0e-3 |
| Momentum | **4.0e-5** | **0.0117** | 0.2001 |
| Volatility Effect | **4.0e-6** | **5.9e-4** | 3.8e-3 |

Buy and Hold **significantly outperforms** both LLM strategies across all robust setups (p < 0.05 in all cases).

### 5.3 Model Complexity Does Not Equal Market Competence
- Scaling laws of NLP do not translate to financial markets
- Larger models (GPT-4o) do not reliably outperform smaller ones (GPT-4o-mini)
- Both are consistently beaten by simpler models like ARIMA on risk-adjusted metrics
- "Without encoded financial logic, architectural complexity appears to add noise rather than value"

### 5.4 Behavioral Hierarchy: FinAgent > FinMem (but both fail)
- FinMem has a **pathological trading profile**: excessive turnover, commission ratio 5-9x higher than FinAgent
- FinMem has longer drawdown durations and negative alpha in all scenarios
- FinAgent is more restrained but still unskilled
- FinAgent's only positive alpha (+6.57%) is in the Momentum selection context (p=0.35, not significant)
- **LLMs' primary strength may be in exploiting strong, pre-existing market trends rather than discovering novel signals**

### 5.5 Alpha and Beta Decomposition (Table 6)

| Strategy | Selection | Avg Max DD (Days) | Avg Regular DD (Days) | Alpha (%) | Beta | Alpha p-value |
|---|---|---|---|---|---|---|
| FinMem | Momentum | 210 | 80 | -1.343 | 0.518 | 0.477 |
| FinAgent | Momentum | 150 | 59 | +6.571 | 0.758 | 0.345 |
| FinMem | Volatility Effect | 177 | 71 | -1.036 | 0.199 | 0.430 |
| FinAgent | Volatility Effect | 123 | 39 | -0.196 | 0.354 | 0.368 |

### 5.6 Selection Strategy Quality Matters
- Volatility Effect selection produces best results (Buy and Hold SPR 0.703)
- RL methods are **most dependent** on selection quality (best under Volatility, worst otherwise)
- FinAgent shows greater dependency on selection quality than FinMem
- Selection ranking: Volatility Effect > FinCon Selection Agent > Momentum Factor > Random Five

### 5.7 Potential Data Leakage Doesn't Help
- GPT-4o may have seen parts of the evaluation data during pretraining
- Despite this potential advantage, LLM strategies still fail to outperform traditional strategies
- "Any such leakage would bias results in favour of LLMs and therefore does not alter our central findings"

---

## 6. Datasets and Time Periods

### Data Sources
- **Stock Prices:** 7,000+ US equities, 2000-2024, including delisted S&P 500 symbols
- **Financial News:** FNSPID dataset (Dong et al., 2024) -- 15.7 million records, 4,775 S&P 500 companies, 1999-2023
- **SEC Filings:** 10-K and 10-Q filings for Russell 3000 companies from SEC EDGAR
- **Extensible:** Supports integration of proprietary data (Alpaca Markets, Refinitiv, earning transcripts, etc.)

### Evaluation Periods
- **Short-term replication:** 2022-10-06 to 2023-04-10 (FinMem) and 2022-10-05 to 2023-06-10 (FinCon)
- **Extended evaluation:** 2004-01-01 to 2024-01-01 (20 years)
- **Rolling windows:** 2-year windows (1-year step) for Selected 4; 1-year windows (1-year step) for Composite

### Symbols
- **Selected 4 (replication):** TSLA, NFLX, AMZN, MSFT
- **FinCon replication:** TSLA, AMZN, NIO, MSFT, AAPL, GOOG, NFLX, COIN
- **Composite setup (bias-mitigated):**
  - Random Five: 91 unique symbols across rolling windows
  - Momentum Factor: 84 unique symbols
  - Volatility Effect: 63 unique symbols
  - FinCon Selection Agent: 80 unique symbols
  - All drawn from historical S&P 500 constituents (including delisted stocks)

---

## 7. Baselines Compared

### Passive Benchmarks
- **Buy and Hold** -- primary passive benchmark; holds position throughout
- **S&P 500 (SPX)** -- used as market return benchmark for CAPM analysis

### Active Traditional Strategies
- SMA Crossover, WMA Crossover, ATR Band, Bollinger Bands, Trend Following, Turn of the Month
- ARIMA, XGBoost (predictor-based)
- A2C, PPO, SAC, TD3, DDPG (RL-based via FinRL)

### Key Comparison Result
**Traditional strategies consistently outperform LLMs:**
- ATR Band, Turn of the Month, and ARIMA deliver positive Sharpe in EVERY regime (bull, sideways, bear)
- Buy and Hold posts 0.61 in bulls, 0.48 in sideways, -0.28 in bears
- ARIMA is often the strongest single strategy across setups

---

## 8. Evaluation Metrics

### Return Metrics
- **Annualised Return (AR):** Geometric average yearly return; R_annual = (1+C)^(252/T) - 1
- **Cumulative Return (CR):** Total return over the full test period

### Risk Metrics
- **Annualised Volatility (AV):** Standard deviation of returns scaled to yearly; sigma_daily * sqrt(252)
- **Maximum Drawdown (MDD):** Largest peak-to-trough decline in portfolio value

### Risk-Adjusted Performance Metrics
- **Sharpe Ratio (SPR):** Excess return per unit of total volatility; (R_bar - R_f) / sigma_daily * sqrt(252)
- **Sortino Ratio (STR):** Excess return per unit of downside risk (only penalizes negative returns)

### Statistical Tests
- **Paired t-tests** comparing Buy and Hold vs FinMem, Buy and Hold vs FinAgent, FinMem vs FinAgent
- **CAPM Alpha/Beta decomposition:** Rs - Rf = alpha + beta(Rm - Rf) + epsilon

### Transaction Costs
- Risk-free rate: 0.03 (historical average)
- Commission: Moomoo standard US commission ($0.0049/share, minimum $0.99/order)

---

## 9. "Overly Conservative in Bull Markets" and "Overly Aggressive in Bear Markets" -- Detailed Analysis

This is the paper's most distinctive contribution -- a diagnosis of HOW LLM agents fail.

### The Paradox
LLM agents exhibit behavior that is precisely the **opposite** of what would be optimal:
- When markets are rising strongly (bull), agents should be aggressive -- but they are conservative
- When markets are falling sharply (bear), agents should be defensive -- but they are aggressive

### Evidence for Conservative Behavior in Bull Markets
1. FinMem has **negative Sharpe (-0.19)** during bull markets -- it actually loses money when the market goes up
2. FinAgent captures only **0.12 Sharpe** vs Buy and Hold's **0.61** in bulls
3. Low beta values confirm underexposure: FinMem beta = 0.199 in Volatility Effect setup
4. Underwater plots show FinAgent sometimes triggers NO trades during bull periods (e.g., KO 2019-2020, APA 2023-2024)
5. This conservatism means missed opportunity to ride the market upward

### Evidence for Aggressive Behavior in Bear Markets
1. FinMem has **Sharpe -0.97** during bear markets -- the WORST of any strategy tested
2. FinAgent -0.38 in bears, worse than even passive Buy and Hold (-0.28)
3. During 2008 GFC, FinMem's drawdown on DE stock approached **-75%**, far worse than the SPX benchmark's -50%
4. Rather than reducing exposure during downturns, agents make **pro-cyclical decisions** that accelerate losses
5. Agents "catastrophically amplify downside risk" rather than providing risk mitigation

### Root Cause
The agents lack:
- **Trend detection capability** -- they cannot identify whether the market is in a bullish or bearish regime
- **Regime-aware risk controls** -- they have no mechanism to dynamically adjust their aggression level
- **Encoded financial logic** -- the underlying LLMs do not have domain-specific financial reasoning about market dynamics

This is described as "a more profound failure in the agents' decision-making policies" -- not just a lack of profitability but a fundamental flaw in how they make decisions.

---

## 10. Recommendations for Improving LLM Strategies

### Two Priority Directions for Future LLM-Based Investors

**Priority 1: Enhance Uptrend Detection**
- Strategies must be able to at least **match passive equity beta** during upward market phases
- Current agents miss bull market returns due to excessive caution
- Need trend-detection capabilities to recognize and participate in strong uptrends

**Priority 2: Incorporate Regime-Aware Risk Controls**
- Dynamically adjust aggression based on detected market regime
- Reduce exposure as volatility or drawdown risk increases
- Explicit risk management mechanisms that adapt to changing conditions

### Additional Design Guidance
- **Do NOT just scale model size** -- "model complexity does not equate to market competence"
- Scaling laws from NLP do not transfer to financial markets (intrinsic limits on extractable signals)
- The primary barrier is **lack of domain-aware financial logic**, not model scale
- "Designing smarter, more adaptive agents" is more important than increasing architectural complexity
- Balance risk-taking and risk management rather than simply increasing framework complexity

### Practical Considerations
- Factor API costs into performance evaluation for real-world deployment
- FinAgent backtesting cost: ~$198 vs FinMem: ~$32 (GPT-4o-mini)
- Total Composite experiments: ~$700 in LLM API costs
- Recommend open-source LLMs (LLaMA, Qwen, Mistral) for cost-effective benchmarking
- Incorporate API usage cost into risk-adjusted metrics (cost-adjusted Sharpe/Sortino)

---

## 11. Code and Data Availability

- **GitHub repository:** https://github.com/waylonli/FINSABER
- **License:** Creative Commons Attribution 4.0 International
- **Implementation:** BackTrader and Papers With Backtest for rule-based strategies; FinRL for RL strategies
- **Two operational modes:** "LLM" mode (multi-modal inputs including news and filings) and "BT" mode (traditional BackTrader-based)
- Open-source data equivalents provided for reproducibility
- Framework designed to be modular and extensible for proprietary data integration

---

## 12. Limitations and Implications for Future Research

### Acknowledged Limitations

1. **No individual tuning of traditional strategies:** Rule-based strategies were not optimized for each rolling window. The authors argue this actually strengthens their conclusions -- tuning traditional strategies would likely elevate their performance further, widening the gap with LLM strategies.

2. **Look-ahead bias not fully eliminated:** Pre-trained LLMs may contain stock-related information from historical periods overlapping test sets. However, observed underperformance of LLM strategies despite this potential advantage strengthens the critical assessment.

3. **Only publicly available data:** Excluded proprietary sources (private newsfeeds, earning transcripts, expert analyses). Framework is designed to be extensible for researchers with access to private data.

4. **Only go-long positions considered:** Aligning with current LLM strategies, which typically only support long positions.

5. **Cost of large-scale backtesting:** LLM backtesting is financially intensive (~$700 for Composite experiments alone). Future work should pursue cost-efficient model designs.

### Key Implications for Future Research

1. **Evaluation standards must improve:** The field needs longer backtesting periods (minimum 3 years for daily trading, 10-20 years for weekly/monthly), broader symbol universes, and explicit bias mitigation.

2. **Regime-awareness is essential:** Future LLM investors must incorporate market regime detection and adaptive behavior.

3. **EMH remains largely valid:** Current LLMs do not overcome the Efficient Market Hypothesis in practice.

4. **Domain-specific financial logic needed:** The path forward requires encoding financial reasoning into agents, not just scaling model parameters.

5. **Cost-performance tradeoff:** API costs should be factored into performance evaluation for practical deployment scenarios.

---

## 13. Relevance to Our Research Hypothesis

### Direct Relevance: LLMs and Longer-Term Trading Decisions

This paper directly CHALLENGES the hypothesis that LLM agents perform better with longer-term trading decisions:

**Against our hypothesis:**
- When evaluation is extended from short-term (6 months) to long-term (20 years), LLM advantages **disappear**
- LLM strategies are fundamentally miscalibrated across market regimes, which becomes more apparent over longer horizons
- Neither FinMem nor FinAgent generates statistically significant alpha over any extended period

**However, important nuances supporting our research direction:**
1. The paper evaluates **daily timing-based** strategies (Buy/Hold/Sell each day), NOT weekly/monthly decision frequencies. Our hypothesis about longer-term DECISION HORIZONS (e.g., weekly rebalancing vs daily trading) is a different question from longer-term EVALUATION HORIZONS.

2. The paper explicitly notes that LLMs' "primary strength may be in exploiting strong, pre-existing market trends rather than discovering novel signals" -- this suggests LLMs might be better suited to LONGER decision horizons where trend-following is more relevant.

3. The regime miscalibration finding suggests that LLMs need **time to observe and reason about market regimes** -- which aligns with longer decision horizons where more context is available.

4. The paper's recommendation for "regime-aware risk controls" and "trend detection" is directly aligned with our exploration of whether longer decision horizons allow LLMs to better incorporate macro context.

5. **ARIMA consistently outperforms LLMs** even in their preferred setups -- this statistical method captures temporal patterns that LLMs miss at daily frequency. At longer horizons, LLMs might have advantage over ARIMA due to ability to process qualitative information.

### Key Takeaway for Our Research
FINSABER shows that LLM daily trading strategies fail over long evaluation periods. But the failure mode (regime miscalibration, inability to detect trends) suggests that the problem may be with the DAILY decision frequency, not with LLMs' fundamental capabilities. Our hypothesis about longer-term decision horizons (weekly/monthly) remains viable and is actually supported by the paper's diagnosis of why daily LLM trading fails.
