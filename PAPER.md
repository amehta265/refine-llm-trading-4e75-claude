# Improving an AI-Generated Trading System: Motivations, Proposals, and Results

**Author:** Ankit Mehta\
**Date:** February 2026\
**Subject:** Take-Home Assignment — Pre-Doctoral MPCS Application\
**Repository Under Review:** [Hypogenic-AI/refine-llm-trading-4e75-claude](https://github.com/Hypogenic-AI/refine-llm-trading-4e75-claude)

---

## Table of Contents

1. [How This Paper Works](#1-how-this-paper-works)
2. [Improvement 1: Risk-Aware Position Sizing (Implemented)](#2-improvement-1-risk-aware-position-sizing-implemented)
   - 2.1 [What the Reviews Found](#21-what-the-reviews-found)
   - 2.2 [Why This Matters](#22-why-this-matters)
   - 2.3 [What We Built](#23-what-we-built)
   - 2.4 [Results](#24-results)
   - 2.5 [What We Learned](#25-what-we-learned)
3. [Improvement 2: Broader Stock and Asset Selection](#3-improvement-2-broader-stock-and-asset-selection)
   - 3.1 [What the Reviews Found](#31-what-the-reviews-found)
   - 3.2 [Why This Matters](#32-why-this-matters)
   - 3.3 [Where and How to Implement](#33-where-and-how-to-implement)
   - 3.4 [How to Measure Success](#34-how-to-measure-success)
4. [Improvement 3: Giving the Agent a Memory](#4-improvement-3-giving-the-agent-a-memory)
   - 4.1 [What the Reviews Found](#41-what-the-reviews-found)
   - 4.2 [Why This Matters](#42-why-this-matters)
   - 4.3 [Where and How to Implement](#43-where-and-how-to-implement)
   - 4.4 [How to Measure Success](#44-how-to-measure-success)
5. [Conclusion](#5-conclusion)

---

## 1. How This Paper Works

I propose 3 improvements to an AI-agent-generated trading research repository produced by the [Idea Explorer](https://github.com/ChicagoHAI/idea-explorer) pipeline. The system takes a minimal hypothesis [`idea.yaml`](.idea-explorer/idea.yaml) ("LLM agents may perform better in finance when making longer-term trading decisions rather than optimizing for short-term, day-to-day profit") and autonomously writes code, runs experiments, and generates a research report.

**Where are the reviews?** Rather than writing one long review section here, I placed my code reviews and notes directly in the source files where the issues live — as inline `REVIEW` comments in [`src/data_loader.py`](src/data_loader.py), [`src/backtester.py`](src/backtester.py), [`src/analysis.py`](src/analysis.py), and [`src/trading_agent.py`](src/trading_agent.py). This way, anyone reviewing the [pull request](https://github.com/Hypogenic-AI/refine-llm-trading-4e75-claude) can see each review comment alongside the exact code it refers to, rather than having to cross-reference line numbers from a separate document.

This paper focuses on the **3 improvements** that came out of those reviews: what motivated them, what they'd look like in practice, and (for the first one) what actually happened when I built it.

---

## 2. Improvement 1: Risk-Aware Position Sizing (Implemented)
 In my opinion, this is the most impactful improvement from both a financial and technical perspective.

### 2.1 What the Reviews Found

Several review comments across the codebase pointed to the same gap — the system has no concept of *how much* to invest or *when to cut losses*:

- **`src/trading_agent.py` (line 17):** The LLM prompt asks for a BUY/HOLD/SELL decision but never asks *how confident* the model is. A "slightly bullish" signal gets treated exactly the same as a "strongly bullish" one. Both result in going 100% into the stock.

- **`src/backtester.py` (line 156):** There is no [stop-loss](https://www.investopedia.com/terms/s/stop-lossorder.asp) mechanism. A stop-loss automatically sells your position if the price drops below a threshold. Without one, the LLM can hold through a massive decline simply because it decided to HOLD.

- **`src/backtester.py` (line 197):** The system uses binary positioning — you're either 100% invested or 100% in cash. There's no in-between. Most real trading systems allow partial positions (e.g., "I'm 40% invested because I'm somewhat bullish but not fully convinced").

### 2.2 Why This Matters

To understand why this matters, I've defined a few concepts:

**Position sizing** is deciding *how much* of your money to put into a single trade. Imagine you have $10,000 to invest. Going all-in on a single stock means your entire portfolio rises or falls with that one bet. A position-sizing strategy might instead say "I'll put $6,000 in because I'm fairly confident, but I'll keep $4,000 in cash in case I'm wrong." There is a [school of thought](https://www.heygotrade.com/en/blog/position-sizing-explained/) that position sizing is actually more important than the timing of your trades.

The [Kelly Criterion](https://www.investopedia.com/articles/trading/04/091504.asp) (Kelly, 1956) formalizes this: it calculates the optimal bet size based on your edge (how often you're right) and the odds (how much you win vs. lose).

**Stop-loss** as mentioned before is a price threshold where you automatically exit a losing trade. For example, "if this stock drops 8% from the highest price I've seen since buying it, sell immediately." This is called a [trailing stop](https://www.investopedia.com/terms/t/trailingstop.asp).

Why does the current system need these? Consider TSLA in the experiment: the daily trading strategy experienced a [maximum drawdown](https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp) (MDD) of **-24.2%**. That means at one point, the portfolio lost nearly a quarter of its value from the peak before recovering. With an 8% trailing stop, that loss would have been capped at roughly -8%.

### 2.3 What We Built

The implementation (found in [`improvements/risk_management/`](improvements/risk_management/)) adds three components to the existing backtester. No new API calls were needed. I replayed the existing LLM decisions through the new risk layer.

**Tooling disclosure:** The `RiskManager` class, `replay_with_risk_management`, and `extract_asset_returns_from_decisions` functions were written by me. The `run_full_comparison` output formatting, `write_results_summary` markdown generator and `run_sensitivity_analysis` log printing, were co-authored with Claude as an AI coding assistant. All code was reviewed, understood, and tested by me before inclusion. I also used claude to understand several finance related concepts which helped me better explain and reason with the implementation.

#### Running the Risk-Aware Backtester

From the repository root:

```bash
python3 improvements/risk_management/risk_backtester.py
```

**Prerequisites:** `numpy` and `pandas` (install via `pip install numpy pandas` if needed).

The script reads the original experiment results from `results/` (the 15 per-stock JSON files produced by the main backtester), replays every LLM decision through the risk management layer, and writes outputs to `improvements/risk_management/results/`. No API keys or external datasets are required - everything runs offline against the already-saved decision logs.

#### Component 1: Confidence-Based Position Sizing

Instead of the binary 100%/0% approach, we map each decision to a position fraction:

| Decision | What It Means | Position Size |
|----------|--------------|---------------|
| BUY | Strong conviction — go in | 100% |
| HOLD (while already invested) | Reduced conviction — stay in, but with less | 60% |
| HOLD (while in cash) | No conviction — stay out | 0% |
| SELL | Exit | 0% |

The idea: if the LLM says HOLD while you're already in a stock, that's weaker than a BUY. You should reduce your exposure to reflect that lower confidence.

#### Component 2: Trailing Stop-Loss

A trailing stop tracks the highest price your position has reached and exits if the stock drops more than 8% from that peak. For example:

> You buy a stock at $100. It rises to $120 (new peak). Your stop-loss is now at $120 × 0.92 = $110.40. If the stock drops to $110.40, you automatically sell — locking in a ~$10 gain instead of riding a potential crash.

#### Component 3: Volatility-Adjusted Position Sizing

Stocks that swing wildly (high [volatility](https://www.investopedia.com/terms/v/volatility.asp)) get smaller positions. Stocks that move more predictably get larger positions. This is a simplified version of [risk parity](https://www.investopedia.com/terms/r/risk-parity.asp) (Qian, 2005) — the idea that every position should contribute roughly the same amount of risk to your portfolio. This is scalable and can be applied to different types of securities (cryptocurrency, forex, commodities, e.t.c.)

For example, TSLA (very volatile) might get a 20% position, while AAPL (less volatile) gets a 70% position. 

### 2.4 Results

We ran the risk management layer against all 15 experiment conditions (5 stocks × 3 frequencies). Full results are in [`improvements/risk_management/results/`](improvements/risk_management/results/).

#### Summary by Frequency

| Frequency | Original Sharpe | Risk-Managed Sharpe | Change | Original MDD | Risk-Managed MDD | MDD Improvement |
|-----------|----------------|-------------------|--------|-------------|-----------------|----------------|
| Daily     | 1.17           | 0.78              | -0.38  | -14.0%      | -5.2%           | +8.8%          |
| Weekly    | 0.19           | -0.05             | -0.23  | -17.2%      | -10.5%          | +6.7%          |
| Monthly   | 1.10           | **1.25**          | **+0.15** | -8.9%    | **-4.5%**       | +4.4%          |

> **Quick definitions:** [Sharpe ratio](https://www.investopedia.com/terms/s/sharperatio.asp) measures return per unit of risk — higher is better. A Sharpe above 1.0 is generally considered good. [Maximum drawdown (MDD)](https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp) is the worst peak-to-trough loss — closer to 0% is better.

#### Standout Improvements for Individual Stocks

- **NVDA daily:** Sharpe improved from 2.28 → 2.45 while MDD improved from -20.5% → -5.1%
- **NVDA weekly:** Sharpe improved from 1.26 → 2.16 while MDD improved from -21.8% → -3.7%
- **AAPL monthly:** Sharpe improved from 1.24 → 1.94 while MDD improved from -5.8% → -3.0%

### 2.5 What We Learned

#### Main point: Drawdowns were cut in half across the board.

Average MDD improved from -13.4% to -6.7% — a **50% reduction**. The worst single drawdown (TSLA daily) went from -24.2% to -5.4%. For any real trading application, this is a dramatic improvement in safety.

#### Monthly frequency was the only one where risk management improved *everything*.

Monthly Sharpe went from 1.10 to 1.25 while MDD also improved. This is the most interesting finding that risk management and monthly decision-making are **complementary**. Monthly decisions were already well-timed enough that the trailing stop never fired once (zero triggers across all monthly conditions). The HOLD-to-60% scaling simply refined what was already working.

#### Daily and weekly frequencies told a different story.

- **Daily:** The volatility scaler aggressively reduced position sizes (averaging only ~20% of capital) because short-term volatility is high. This protected against drawdowns but also missed out on rallies. In the bullish 2024 market, that meant lower returns.

- **Weekly:** Stop-losses triggered 7 times total, often at exactly the wrong moment (selling after a dip that then recovered). This is called [whipsaw](https://www.investopedia.com/terms/w/whipsaw.asp), and it's the classic failure mode of mechanical stop-losses. With the corrected frequency-aware volatility scaling (using `sqrt(52)` instead of the erroneous `sqrt(252)` for weekly data), the Sharpe decline is less severe (-0.23 vs. the originally reported -0.63), but weekly remains the weakest frequency even with risk management.

#### Why the "negative" results are still valuable

The daily return reduction and weekly Sharpe decline aren't failures — they're informative:

1. The original daily performance was partly driven by riding volatile rallies at 100% exposure (e.g., NVDA going from +81.5% to +32.0% with risk management). In a bear market, that same exposure would cause severe losses that risk management would prevent.
2. A more sophisticated system would dynamically adjust the stop-loss threshold and volatility target based on [bull vs bear vs sideways trends](https://www.investopedia.com/terms/m/market_regime.asp).
3. The binary nature of the original system (100% or 0%) acts as its own crude risk management i.e. being fully out during sell signals. The fractional approach introduces nuance that interacts differently across frequencies.

---

## 3. Improvement 2: Broader Stock and Asset Selection
### 3.1 What the Reviews Found

- **`src/data_loader.py` (line 10):** The experiment uses 5 stocks, but 4 of them are technology companies (AAPL, NVDA, TSLA, MSFT). Only JPM represents a different sector. This was flagged as a [selection bias](https://en.wikipedia.org/wiki/Selection_bias) — the results might reflect how well the LLM trades *tech stocks* rather than stocks in general.

### 3.2 Why This Matters

There are two specific problems:

**Problem 1: Correlated returns inflate statistical confidence.**

[Correlation](https://www.investopedia.com/terms/c/correlation.asp) measures how much two stocks move together. Tech stocks tend to be highly correlated (often > 0.6) — when Apple rises, Nvidia and Microsoft usually rise too. This means the 5 "independent" stock-level observations in the experiment aren't truly independent. It's a bit like surveying 5 people from the same family and treating their opinions as 5 independent data points.

This directly affects the statistical tests in `src/analysis.py`. The [paired t-test](https://www.investopedia.com/terms/t/t-test.asp) and [Wilcoxon signed-rank test](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test) both assume each observation is independent. With 4 correlated tech stocks, the effective sample size is closer to 2-3, not 5. This means the [p-values](https://www.investopedia.com/terms/p/p-value.asp) (which measure the probability the results are due to chance) are artificially low, overstating confidence.

As noted in the review on `src/analysis.py` (line 131), the [Cohen's d effect sizes](https://en.wikipedia.org/wiki/Effect_size#Cohen's_d) are large (d=1.31 for daily-vs-weekly, d=0.92 for weekly-vs-monthly), but the p-values are still above 0.05. This "large effect but not significant" pattern is a classic sign of insufficient sample size. Expanding to 15–20 stocks across sectors would likely achieve statistical significance.

**Problem 2: Different asset types behave differently at different frequencies.**

- **Cryptocurrency** (BTC, ETH) — High volatility, trades 24/7, strongly driven by narratives and social media. LLMs are trained on enormous amounts of crypto discussion.
- **Utilities stocks** (NEE, SO) — Low volatility, steady dividends, driven by interest rate changes. Very different dynamics from tech.
- **Energy stocks** (XOM, CVX) — Driven by commodity prices and geopolitics. Could test whether the LLM's "general knowledge" translates to macro-driven sectors.
- **Healthcare** (JNJ, UNH) — Driven by FDA approvals, patent cliffs, and policy changes.

If the monthly frequency advantage holds across 4+ sectors, that's much stronger evidence. If some sectors show different patterns (e.g., daily is best for crypto), that's an *even more interesting* finding about the interaction between frequency and asset type.

### 3.3 Where and How to Implement

| File | Change |
|------|--------|
| `src/data_loader.py` | Expand `EXPERIMENT_TICKERS` to include at least 2 stocks per [sector](https://www.msci.com/our-solutions/indexes/gics). Add a `SECTOR_MAP` dictionary for stratified analysis. For crypto, add tickers like `BTC-USD` and `ETH-USD` |
| `src/trading_agent.py` | Add asset-class-specific context to the prompt (e.g., "This is a cryptocurrency that trades 24/7" or "This is a utility stock with typically low volatility"). |
| `src/analysis.py` | Add sector-stratified analysis: compute mean Sharpe/MDD by frequency *within* each sector and test whether the frequency effect varies by sector. |

### 3.4 How to Measure Success

- The frequency effect (monthly > daily > weekly) should be tested independently within each sector.
- If the effect replicates across 4+ sectors, confidence in the finding increases substantially.
- If some sectors show different patterns, that becomes a finding about frequency–sector interaction.

---

## 4. Improvement 3: Giving the Agent Memory

### 4.1 What the Reviews Found

- **`src/backtester.py` (line 167):** The `get_llm_decision()` function is called at each time step to make a BUY/HOLD/SELL decision, but it receives **no information about its prior decisions or their outcomes**. Every decision is made from scratch, as if the agent has amnesia.

- **`src/trading_agent.py` (line 90):** No layered memory context is passed across decisions. The model can't learn from its own track record, which limits its ability to self correct.

### 4.2 Why This Matters
Let think how a human trader works. After a string of bad calls, they step back and ask: *"What am I getting wrong? Is the market in a different mode than I assumed?"* They adjust their approach based on feedback. The current LLM agent can't do this. It makes each decision in a vacuum.

This matters because financial markets are **non-stationary**, meaning their statistical properties change over time. A strategy that works in a bull market can fail dramatically in a bear market These shifts are called [regime changes](https://www.investopedia.com/terms/r/regime.asp).

Markets are also **reflexive** — a concept from George Soros's theory of [reflexivity](https://en.wikipedia.org/wiki/Reflexivity_(social_theory)#In_economics): the actions of market participants change the market itself. Prices aren't just signals to observe; they're influenced by the very trades you make.

The AI-Trader benchmark paper referenced in the repository's literature makes this explicit: *"General intelligence does not automatically translate to effective trading capability."* Simply being a good language model doesn't make you a good trader. You need mechanisms to adapt.

A concrete solution exists in the research literature. The [FinMem architecture](https://arxiv.org/abs/2311.13743) (Yu et al., 2023) introduces layered memory with different time horizons: Short-term memory (14-day window), Medium-term memory (90-day window), and Long-term memory (365-day window).
Each layer has a different decay rate, so recent information is weighted more heavily but older context isn't completely forgotten.

### 4.3 Where and How to Implement

| File | Change |
|------|--------|
| `src/trading_agent.py` | Add a `DecisionMemory` class that stores the last N decisions with their outcomes (return, whether the call was correct). Include this history in the LLM prompt so the model can see its own track record. |
| `src/trading_agent.py` | Add a self-reflection step: after every K decisions, prompt the model with its recent performance statistics and ask it to articulate what's working and what isn't. Inject this updated strategy into subsequent prompts. |
| `src/backtester.py` | Modify the backtest loop to compute running Profit and Loss and pass it to the agent. Add a simple regime detection step — for example, flag whether the stock's price is above or below its [50-day moving average](https://www.investopedia.com/terms/m/movingaverage.asp) as a proxy for "bull" or "bear" conditions. |
| New file: `src/memory.py` | A lightweight memory module with: (1) a rolling window of past decisions and returns, (2) a running [Sharpe ratio](https://www.investopedia.com/terms/s/sharperatio.asp) for self-assessment, and (3) a market regime classifier. |

### 4.4 How to Measure Success

- **Behavioral change:** Does the model actually change its strategy after a string of losses? Compare decision patterns with and without memory.
- **Calibration:** Is there a correlation between the model's expressed reasoning and actual subsequent returns? A well-calibrated agent should be more cautious when it has been wrong recently.
- **Regime transitions:** Test performance specifically during periods where the market shifts. Memory-equipped agents should adapt faster.
- **Strategy drift:** Measure how much the model's behavior changes after receiving feedback compared to the memoryless baseline. Too little drift means the memory isn't helping; too much means the model is overreacting.

---

## 5. Conclusion

These three improvements address the most significant gaps found during my code review:

Detailed review comments motivating each of these improvements can be found inline in the source files. They are designed to be read as part of the [pull request diff](https://github.com/Hypogenic-AI/refine-llm-trading-4e75-claude).
