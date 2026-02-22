# Research Plan: Refining LLM Trading — Decision Frequency Impact

## Motivation & Novelty Assessment

### Why This Research Matters
LLM-based trading agents have emerged as a major research direction, with 70+ papers published since 2023. However, virtually all systems operate at daily trading frequency, and recent rigorous evaluations (FINSABER, DeepFund) show that daily LLM trading **consistently underperforms Buy-and-Hold** over longer periods. This raises a fundamental question: is the problem the LLM approach itself, or the decision frequency? If LLMs are better at strategic reasoning than rapid tactical decisions, they may excel at longer-term trading where fundamental analysis and macro reasoning matter more than daily noise.

### Gap in Existing Work
Based on the literature review:
- **No paper systematically compares LLM trading across different decision frequencies** (daily vs. weekly vs. monthly). Every published system uses daily decisions.
- FINSABER showed daily LLM strategies collapse over 20-year evaluations, with regime miscalibration as a key failure mode.
- DeepFund's live testing showed the only profitable LLM strategy used conservative, low-frequency trading.
- InvestorBench found ETF trading (the most strategic/long-term task) was hardest for LLMs but improved with more capable models.

### Our Novel Contribution
We are the first to systematically test whether LLM trading agents perform better when making decisions at weekly or monthly frequency versus daily frequency. We control for stock selection, time period, and information structure, isolating decision frequency as the independent variable.

### Experiment Justification
- **Experiment 1 (Multi-frequency comparison):** Core experiment comparing daily/weekly/monthly LLM trading on the same stocks and period. Tests whether longer horizons improve risk-adjusted returns.
- **Experiment 2 (Market regime analysis):** Splits results by bull/bear periods to test whether longer horizons reduce regime miscalibration (the key failure mode identified by FINSABER).
- **Experiment 3 (Cost-adjusted analysis):** Computes transaction costs and API costs at each frequency, testing practical viability.

---

## Research Question
**Do LLM trading agents achieve better risk-adjusted returns when making less frequent (weekly or monthly) trading decisions compared to daily decisions?**

## Hypothesis Decomposition

- **H1 (Primary):** LLM agents making weekly decisions achieve higher Sharpe ratios than daily decisions over a multi-year evaluation period.
- **H2:** LLM agents making monthly decisions outperform daily decisions during bear market regimes (reduced regime miscalibration).
- **H3:** Longer decision horizons reduce portfolio turnover and transaction costs, improving net returns.
- **H4 (Null):** Decision frequency has no significant effect on LLM trading performance — results are driven by the underlying market trend regardless of frequency.

## Proposed Methodology

### Approach
Use a real LLM (GPT-4.1-mini via OpenAI API) as a trading agent that receives price data and technical indicators at each decision point. The agent outputs BUY/HOLD/SELL decisions. We run this at three frequencies (daily, weekly, monthly) on the same stocks and time period, then compare performance metrics.

**Why this approach:**
- Uses a real LLM (not simulated), producing scientifically valid results
- GPT-4.1-mini balances capability with cost-effectiveness (~1,500 API calls total)
- Controlling for stocks and time period isolates frequency as the independent variable
- Technical indicators (SMA, RSI) are standard in the literature

### Stocks
10 tickers available at all three frequencies: AAPL, AMZN, BAC, GOOGL, JPM, META, MSFT, NFLX, NVDA, TSLA

These span tech (AAPL, AMZN, GOOGL, META, MSFT, NVDA), finance (BAC, JPM), entertainment (NFLX), and auto/energy (TSLA).

### Evaluation Period
**January 2023 – December 2024** (2 years)
- Covers post-2022 bear market recovery, 2023 AI rally, 2024 mixed markets
- Includes both bull and bear regimes
- Recent enough to avoid training data contamination concerns

### Experimental Steps
1. **Data preparation:** Load daily/weekly/monthly price data, compute technical indicators (SMA-10, SMA-20, RSI-14) at each frequency
2. **Prompt design:** Create a consistent prompt template for the LLM that presents price history, indicators, and current position
3. **Daily trading:** Run LLM agent at daily frequency (~500 decisions per stock × 10 stocks = ~5000, subsample to 5 stocks for cost)
4. **Weekly trading:** Run LLM agent at weekly frequency (~104 decisions per stock × 5 stocks = ~520)
5. **Monthly trading:** Run LLM agent at monthly frequency (~24 decisions per stock × 5 stocks = ~120)
6. **Baselines:** Compute Buy-and-Hold and SMA crossover returns at each frequency
7. **Analysis:** Compare CR, SR, MDD across frequencies with statistical tests

### Revised Scope (Cost-Optimized)
To keep API costs manageable while maintaining rigor:
- **5 stocks** for main experiment: AAPL, JPM, NVDA, TSLA, MSFT (diverse sectors + volatility profiles)
- **1-year period** (Jan 2024 – Dec 2024) for main experiment
- **2 runs per condition** (temperature=0 for deterministic, temperature=0.3 for stochastic)
- Total API calls: ~(250 + 52 + 12) × 5 × 2 = ~3,140 calls with GPT-4.1-mini

### Baselines
| Baseline | Rationale |
|----------|-----------|
| Buy-and-Hold | Universal passive benchmark used in all papers |
| SMA Crossover (10/20) | Simple technical strategy, frequency-agnostic |

### Evaluation Metrics
| Metric | Description | Why |
|--------|-------------|-----|
| Cumulative Return (CR) | Total % gain/loss | Primary performance measure |
| Sharpe Ratio (SR) | Risk-adjusted return | Standard in all LLM trading papers |
| Maximum Drawdown (MDD) | Worst peak-to-trough | Risk measure |
| Sortino Ratio | Downside-risk-adjusted return | Penalizes downside only |
| Win Rate | % of profitable decisions | Decision quality |
| Turnover Rate | % of periods with position changes | Trading activity |

### Statistical Analysis Plan
- **Primary test:** Paired t-test (or Wilcoxon signed-rank if non-normal) comparing Sharpe ratios across stocks between daily vs weekly, daily vs monthly
- **Significance level:** α = 0.05
- **Effect size:** Cohen's d
- **Multiple comparison correction:** Bonferroni (3 pairwise comparisons)
- **Regime analysis:** Split returns by market regime (bull: SPY > SMA-50, bear: SPY < SMA-50) and compare within regimes

## Expected Outcomes
- **If H1 supported:** Weekly LLM trading achieves higher Sharpe ratios than daily, suggesting LLMs benefit from more context and less noise
- **If H2 supported:** Monthly decisions show particular advantage in bear markets, confirming reduced regime miscalibration
- **If H4 (null):** No significant difference, suggesting the LLM approach has fundamental limitations regardless of frequency
- **Unexpected:** Monthly trading could underperform due to missing shorter-term opportunities

## Timeline and Milestones
1. Environment + data setup: 15 min
2. Implementation (prompt design, trading loop, metrics): 60 min
3. Run experiments (API calls): 45 min
4. Analysis + visualization: 30 min
5. Documentation (REPORT.md, README.md): 30 min

## Potential Challenges
- **API rate limits:** Use exponential backoff; GPT-4.1-mini has generous limits
- **Data format issues:** Multi-level CSV headers need careful parsing
- **Market data alignment:** Ensure daily/weekly/monthly decisions use comparable information windows
- **Cost overruns:** Cap at 5 stocks × 1 year; can extend if budget allows

## Success Criteria
1. Successfully run LLM trading at all 3 frequencies with real API calls
2. Produce valid performance metrics (CR, SR, MDD) for each condition
3. Perform statistical comparison between frequencies
4. Generate clear visualizations showing the comparison
5. Document findings in REPORT.md with actual results
