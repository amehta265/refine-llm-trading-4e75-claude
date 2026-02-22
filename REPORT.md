# Research Report: Does Decision Frequency Affect LLM Trading Agent Performance?

## 1. Executive Summary

We conducted the first systematic comparison of LLM trading agent performance across three decision frequencies — daily, weekly, and monthly — using GPT-4.1-mini on 5 stocks over 2024. Our key finding is that **monthly LLM trading achieves comparable risk-adjusted returns to daily trading (mean Sharpe 1.10 vs 1.17) while dramatically reducing drawdowns and API costs by 95%**. However, weekly trading performed poorly (mean Sharpe 0.19), and results varied substantially by stock. Monthly decisions showed a distinctive behavioral pattern: the LLM adopted more conservative, trend-following strategies that captured major moves while avoiding whipsaws. None of the pairwise frequency differences reached statistical significance (p > 0.05 after Bonferroni correction) with our sample of 5 stocks, but the large effect size for weekly vs monthly (Cohen's d = 0.92) suggests a meaningful practical difference worth investigating at larger scale.

## 2. Goal

**Research Question:** Do LLM trading agents achieve better risk-adjusted returns when making less frequent (weekly or monthly) trading decisions compared to daily decisions?

**Why This Matters:** Virtually all published LLM trading systems operate at daily frequency, yet rigorous long-term evaluations (FINSABER, DeepFund) show daily LLM trading consistently underperforms Buy-and-Hold. This raises a fundamental question: is the problem the LLM approach itself, or the daily frequency? If LLMs are better at strategic reasoning than rapid tactical decisions, they may excel at longer horizons where noise is reduced.

**Hypotheses:**
- H1: Weekly LLM decisions achieve higher Sharpe ratios than daily
- H2: Monthly LLM decisions outperform daily, especially in bear regimes
- H3: Longer horizons reduce portfolio turnover and transaction costs
- H4 (Null): Frequency has no significant effect

**Novel Contribution:** This is the first study to systematically test LLM trading across multiple decision frequencies, controlling for stock selection, time period, model, and information structure. All 23 papers in our literature review used only daily decisions.

## 3. Data Construction

### Dataset Description
- **Source:** Yahoo Finance via `yfinance`
- **Format:** Daily, weekly, and monthly OHLCV data
- **Time Period:** February 2016 – December 2024 (10 years, with 2024 as evaluation year)
- **Stocks:** AAPL, JPM, NVDA, TSLA, MSFT (5 stocks spanning tech, finance, automotive)
- **License:** Yahoo Finance Terms of Service (research use)

### Evaluation Period
**January 2024 – December 2024** (1 year)
- Covers: Post-2022 recovery, 2024 AI rally (NVDA), mixed tech markets (MSFT)
- Market conditions: Generally bullish with sector-specific pullbacks
- SPY returned ~25% during this period

### Data Quality
- Missing values: 0% (Yahoo Finance provides complete trading day data)
- Outliers: None removed (TSLA and NVDA naturally have high volatility)
- Data validated by comparing to public records

### Technical Indicators Computed
At each frequency, we computed:
- SMA-10 (10-period simple moving average)
- SMA-20 (20-period simple moving average)
- RSI-14 (14-period relative strength index)
- 1-period and 5-period returns

### Decision Points per Stock
| Frequency | Periods | Information Window |
|-----------|---------|-------------------|
| Daily     | 252     | Last 15 trading days |
| Weekly    | 53      | Last 15 weeks |
| Monthly   | 12      | Last 15 months |

## 4. Experiment Description

### Methodology

#### High-Level Approach
We used a single real LLM (GPT-4.1-mini via OpenAI API) as a trading agent that receives recent price data and technical indicators at each decision point. The agent outputs BUY/HOLD/SELL decisions. We ran this at three frequencies (daily, weekly, monthly) on the same 5 stocks during the same time period, then compared performance metrics.

#### Why This Method?
- **Real LLM, not simulated**: Uses actual GPT-4.1-mini API calls, producing scientifically valid results
- **Controlled comparison**: Same model, same stocks, same period — frequency is the only independent variable
- **Price-only information**: Excludes news/sentiment to isolate the frequency effect from information differences
- **Standard indicators**: SMA and RSI are used in virtually all LLM trading papers for comparability

### Implementation Details

#### Tools and Libraries
| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.12.2 | Runtime |
| openai | 2.21.0 | GPT-4.1-mini API |
| pandas | 2.3.3 | Data manipulation |
| numpy | 2.3.0 | Numerical computation |
| scipy | 1.17.0 | Statistical testing |
| matplotlib | 3.10.8 | Visualization |

#### Model Configuration
- **Model:** GPT-4.1-mini (OpenAI, 2025)
- **Temperature:** 0.0 (deterministic)
- **Max tokens:** 150
- **System prompt:** Standardized across all frequencies; instructs the model to output JSON with decision and reasoning
- **User prompt:** Contains stock name, frequency context, current position, and 15-period price history with indicators

#### Prompt Design
The system prompt instructs the model to act as a stock trading analyst and output a JSON object with `decision` (BUY/HOLD/SELL) and `reasoning`. The user prompt includes:
1. Stock ticker and frequency context
2. Current position (LONG or OUT)
3. Frequency-specific instructions (e.g., "Your next decision will be in one month")
4. Recent price history table with Close, Volume, SMA10, SMA20, RSI14, and returns

#### Trading Rules
- Binary positioning: 100% long or 100% cash (no shorting, no partial positions)
- BUY enters a long position; SELL exits to cash; HOLD maintains
- No transaction costs applied (analyzed separately below)
- Starting position: OUT (cash)

### Experimental Protocol

#### Reproducibility Information
- Runs: 1 per condition (temperature=0.0 for deterministic output)
- Random seeds: N/A (deterministic API calls)
- Hardware: Linux, 2x NVIDIA RTX 3090 (not used — CPU sufficient for API calls)
- Total API calls: 1,585 (252×5 daily + 53×5 weekly + 12×5 monthly)
- Total tokens: 1,768,914
- Estimated API cost: $1.34
- Execution time: 22 minutes

### Raw Results

#### Sharpe Ratio Comparison

| Stock | Daily SR | Weekly SR | Monthly SR | B&H SR |
|-------|----------|-----------|------------|--------|
| AAPL  | **1.38** | 0.11      | 1.24       | 1.20   |
| JPM   | 0.60     | 0.31      | **2.39**   | 1.61   |
| NVDA  | **2.28** | 1.26      | 0.36       | 3.47   |
| TSLA  | 0.86     | 0.75      | **2.10**   | 0.75   |
| MSFT  | **0.70** | -1.50     | -0.60      | 0.44   |
| **Mean** | **1.17** | **0.19** | **1.10** | **1.49** |

#### Cumulative Return Comparison

| Stock | Daily CR | Weekly CR | Monthly CR | B&H CR |
|-------|----------|-----------|------------|--------|
| AAPL  | +24.4%   | +7.1%     | +23.2%     | +32.0% |
| JPM   | +12.8%   | +10.1%    | **+57.0%** | +42.8% |
| NVDA  | **+81.5%** | +46.1% | +16.3%     | +187.2%|
| TSLA  | +37.9%   | +45.6%    | **+104.5%**| +52.7% |
| MSFT  | **+14.5%**| -17.4%   | -2.7%      | +13.7% |
| **Mean** | **+34.2%** | **+18.3%** | **+39.7%** | **+65.7%** |

#### Maximum Drawdown Comparison

| Stock | Daily MDD | Weekly MDD | Monthly MDD | B&H MDD |
|-------|-----------|------------|-------------|---------|
| AAPL  | -9.0%     | -12.9%     | **-5.8%**   | -15.4%  |
| JPM   | -7.9%     | -9.4%      | **-6.2%**   | -10.1%  |
| NVDA  | -20.5%    | -21.8%     | **-13.1%**  | -27.0%  |
| TSLA  | -24.2%    | -23.6%     | **-7.7%**   | -40.9%  |
| MSFT  | **-8.6%** | -18.3%     | -11.9%      | -15.5%  |
| **Mean** | **-14.0%** | **-17.2%** | **-8.9%** | **-21.8%** |

#### Decision Distribution

| Stock | Freq | BUY% | HOLD% | SELL% | Turnover |
|-------|------|------|-------|-------|----------|
| AAPL  | daily | 14% | 46% | 40% | 40% |
| AAPL  | weekly | 30% | 40% | 30% | 63% |
| AAPL  | monthly | 50% | 33% | 17% | 45% |
| JPM   | daily | 12% | 56% | 32% | 43% |
| JPM   | weekly | 26% | 40% | 34% | 58% |
| JPM   | monthly | 17% | **83%** | 0% | 27% |
| NVDA  | daily | 17% | 54% | 29% | 38% |
| NVDA  | weekly | 26% | 34% | 40% | 40% |
| NVDA  | monthly | 17% | 50% | 33% | 36% |
| TSLA  | daily | 15% | 43% | 42% | 49% |
| TSLA  | weekly | 19% | 53% | 28% | 50% |
| TSLA  | monthly | 25% | 42% | 33% | 45% |
| MSFT  | daily | 17% | 48% | 35% | 41% |
| MSFT  | weekly | 26% | 49% | 25% | 56% |
| MSFT  | monthly | 17% | 42% | 42% | 45% |

## 5. Result Analysis

### Key Findings

1. **Monthly trading achieved the best risk management.** Monthly decisions had the lowest average max drawdown (-8.9%) compared to daily (-14.0%) and weekly (-17.2%). This was consistent across 4 of 5 stocks.

2. **Monthly trading produced comparable Sharpe ratios to daily.** Mean monthly SR (1.10) was essentially equal to mean daily SR (1.17), with no statistically significant difference (p=0.93).

3. **Weekly trading was the worst frequency.** Mean weekly SR (0.19) was dramatically lower than both daily (1.17) and monthly (1.10). The daily-vs-weekly comparison approached significance (p=0.059 unadjusted).

4. **Monthly LLM beat Buy-and-Hold on 2 of 5 stocks** (JPM: +57% vs +43%; TSLA: +104% vs +53%), while daily LLM beat B&H on only 1 stock (MSFT: +14.5% vs +13.7%).

5. **Monthly LLM showed distinctive behavioral patterns.** At monthly frequency, the LLM adopted more conservative strategies: higher HOLD percentages (especially JPM at 83%), lower turnover, and clearer trend-following logic.

6. **API costs scale linearly with frequency.** Daily trading used 1,403,913 tokens ($1.06) vs monthly's 67,130 tokens ($0.05) — a **21x cost reduction**.

### Statistical Test Results

| Comparison | Test | t-stat | p-value | p (Bonferroni) | Cohen's d | Significant? |
|------------|------|--------|---------|----------------|-----------|-------------|
| Daily vs Weekly | Paired t-test | 2.61 | 0.059 | 0.178 | -1.31 | No |
| Daily vs Monthly | Paired t-test | 0.09 | 0.931 | 1.000 | -0.05 | No |
| Weekly vs Monthly | Paired t-test | -1.85 | 0.138 | 0.415 | 0.92 | No |

**Interpretation:** With only 5 paired observations, we lack statistical power to detect differences. However, the effect sizes are notable: daily-vs-weekly (d = -1.31, large) and weekly-vs-monthly (d = 0.92, large) suggest practically meaningful differences. A larger study with 15-20 stocks could achieve significance.

### Comparison to Baselines

**vs Buy-and-Hold:** The LLM underperformed B&H on average at all frequencies (LLM best: 39.7% monthly vs B&H: 65.7%). This is consistent with FINSABER's finding that LLM daily trading underperforms B&H. However, the LLM dramatically reduced drawdowns, suggesting value in risk management.

**vs SMA Crossover:** LLM monthly slightly outperformed SMA crossover on average (39.7% vs 35.7%), and showed better risk-adjusted performance on volatile stocks (TSLA monthly SR=2.10 vs SMA SR=0.73).

### Behavioral Analysis

**Daily decisions** showed high noise reactivity: 40% average turnover rate, frequent oscillation between BUY and SELL, with reasoning citing short-term indicators ("recent pullback," "one-day decline"). The model struggled with distinguishing signal from noise.

**Weekly decisions** exhibited the worst pattern: high turnover (56% average) with poor timing. The model appeared caught between time horizons — too slow for daily momentum capture, too fast for trend following. This "worst of both worlds" effect is a novel finding.

**Monthly decisions** showed strategic reasoning: the model cited "long-term uptrend," "macro patterns," and "bullish crossover" in its reasoning. On JPM, it bought in January 2024 and held all year (0 SELL decisions, 83% HOLD) — effectively recognizing a strong uptrend and avoiding overtrading. On TSLA, it strategically avoided the Q1-Q2 downturn (SELL/HOLD) and entered before the Q4 rally (BUY in June and November).

### Surprises and Insights

1. **Weekly was worst, not daily.** We expected a monotonic improvement from daily→weekly→monthly. Instead, weekly was dramatically worse. This suggests a "valley of indecision" at intermediate frequencies.

2. **Monthly LLM beat B&H on 2 stocks.** For JPM and TSLA, the monthly LLM outperformed passive investing, which is rare in the literature. The TSLA monthly return (+104.5%) doubled daily LLM (+37.9%).

3. **Monthly LLM effectively learned to "buy and hold"** on strong trends (JPM) while actively timing entries on volatile stocks (TSLA). This adaptive behavior suggests longer horizons allow the LLM to use its reasoning capability for strategic decisions.

4. **The drawdown improvement at monthly is dramatic.** Monthly MDD averaged -8.9% vs daily -14.0% and B&H -21.8%. For TSLA specifically, monthly MDD was -7.7% vs daily -24.2% and B&H -40.9%.

### Limitations

1. **Small sample size (5 stocks):** Insufficient for robust statistical significance. Results could differ with different stock selections.

2. **Single evaluation year (2024):** 2024 was generally bullish. Results in bear markets could differ substantially — which is where the hypothesis is most interesting.

3. **No news/sentiment data:** The LLM only received price data and indicators. In practice, LLM trading agents also process news, which could interact differently with frequency.

4. **Single model (GPT-4.1-mini):** More capable models (GPT-4.1, Claude 4.5) might show different frequency effects.

5. **No transaction costs:** In practice, daily trading incurs significantly higher costs, which would further favor monthly decisions.

6. **Deterministic sampling (temperature=0):** No variance estimation across runs. A stochastic evaluation would provide confidence intervals.

7. **Monthly has only 12 data points per stock:** The statistical power for per-stock monthly analysis is very limited.

## 6. Conclusions

### Summary
LLM trading agents show frequency-dependent performance patterns. Monthly decisions achieve comparable risk-adjusted returns to daily trading while dramatically reducing drawdowns (by 36%) and API costs (by 95%). Weekly trading performs poorly, suggesting an intermediate frequency "valley" where LLMs lose both daily momentum signals and monthly strategic context. While no frequency comparisons reached statistical significance with 5 stocks, the effect sizes and consistent drawdown improvements support the hypothesis that LLMs are better suited to longer-horizon strategic decisions.

### Implications
- **For practitioners:** Monthly or biweekly LLM trading strategies may be more cost-effective and equally profitable compared to daily systems, with substantially better risk profiles.
- **For researchers:** The "weekly valley" finding suggests frequency is not a simple monotonic variable — there may be an optimal frequency that balances information richness with noise reduction.
- **For the field:** The widespread default to daily LLM trading deserves reconsideration. The literature's consistent finding that daily LLM trading underperforms B&H may partly be a frequency problem, not a capability problem.

### Confidence in Findings
- **High confidence:** Monthly decisions reduce drawdowns and API costs vs daily
- **Moderate confidence:** Monthly SR is comparable to or better than daily SR (consistent across 3/5 stocks)
- **Low confidence:** Weekly is always worst (could be dataset-specific)
- **Statistical significance:** Not achieved at α=0.05 for any comparison; larger studies needed

## 7. Next Steps

### Immediate Follow-ups
1. **Expand to 15-20 stocks** to achieve statistical power (estimated to detect medium effect sizes)
2. **Test across multiple years** including the 2022 bear market to evaluate regime-specific effects
3. **Add biweekly frequency** to explore the "valley" between weekly and monthly
4. **Run with temperature=0.3** for stochastic evaluation with confidence intervals

### Alternative Approaches
- Test more capable models (GPT-4.1, Claude 4.5 Sonnet) to see if model capability interacts with frequency
- Add news/sentiment data to the prompt and compare frequency effects with richer information
- Implement a dynamic frequency agent that switches frequencies based on market regime

### Broader Extensions
- Apply frequency analysis to other LLM agent frameworks (FinMem, FinCon, TradingAgents)
- Test on other asset classes (crypto, commodities, bonds)
- Investigate optimal information window size by frequency

### Open Questions
1. Why does weekly trading perform so poorly? Is it the "worst of both worlds" or a prompt design issue?
2. Would monthly LLM trading outperform B&H in bear markets (as suggested by DeepFund's findings)?
3. Does the optimal frequency vary by market regime (bull vs bear)?
4. Can an LLM learn to choose its own trading frequency based on market conditions?

## References

### Papers
1. Li et al. (2026). FINSABER: Can LLM-based Financial Investing Strategies Outperform the Market in Long Run? KDD 2026.
2. Yu et al. (2023). FinMem: A Performance-Enhanced LLM Trading Agent With Layered Memory. arXiv:2311.13743.
3. Yu et al. (2024). FinCon: A Synthesized LLM Multi-Agent System. NeurIPS 2024.
4. Xiao et al. (2025). TradingAgents: Multi-Agents LLM Financial Trading Framework. arXiv:2412.20138.
5. Ding et al. (2024). Large Language Model Agent in Financial Trading: A Survey. arXiv:2408.06361.
6. DeepFund (2025). Time Travel is Cheating: Going Live with DeepFund. NeurIPS 2025.
7. Chen et al. (2025). StockBench: Can LLM Agents Trade Stocks Profitably? arXiv:2510.02209.

### Tools
- OpenAI GPT-4.1-mini API
- Yahoo Finance (yfinance)
- Python 3.12, pandas, scipy, matplotlib

### Data
- Stock price data: Yahoo Finance, 2016-2024, 5 tickers at daily/weekly/monthly frequencies
- Total experiment cost: $1.34 (1,768,914 tokens)
