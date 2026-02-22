# StockBench: Can LLM Agents Trade Stocks Profitably In Real-world Markets?

**Paper:** arXiv 2510.02209 (originally submitted to ICLR 2026, withdrawn)
**Authors:** Yanxu Chen, Yantao Liu, Zijun Yao, Jin Ye, Jianing Yu, Lei Hou, Juanzi Li
**Project page:** https://stockbench.github.io/
**Code:** https://github.com/ChenYXxxx/stockbench (Apache 2.0 license)

**Note:** The local PDF at `papers/stockbench_2505_14488.pdf` is corrupted/mislabeled (contains
a materials science paper about metal oxide heterointerfaces). This summary was compiled from
the arXiv HTML version, project website, and GitHub repository.

---

## 1. What Is StockBench?

StockBench is a **contamination-free benchmark** for evaluating LLM agents in realistic,
multi-month stock trading environments. Unlike static financial QA benchmarks, StockBench
tests the **dynamic and iterative nature of trading**: agents receive daily market signals --
prices, fundamentals, and news -- and must make sequential buy, sell, or hold decisions over
an extended evaluation period.

The benchmark consists of two main building blocks:

- **Back-trading environment**: Contains historical data for stock-trading decision-making and
  simulates realistic stock trading using three critical information sources: investment targets
  (pre-defined stocks), historical market data (prices and fundamentals), and news corpora.
- **Stock-trading agent workflow**: Converts backbone LLMs into agents through four stages:
  1. **Portfolio Overview** -- Agent scans all available stocks receiving news, current holdings,
     historical actions, and opening prices.
  2. **In-Depth Stock Analysis** -- Agent selects specific stocks for deeper analysis with
     fundamental data (market cap, P/E ratio, dividend yield, etc.).
  3. **Decision Generation** -- Agent chooses between three actions: increase, decrease, or
     hold positions.
  4. **Execution and Validation** -- Decisions convert to share quantities; system flags
     liquidity issues requiring revision.

### Contamination Prevention

Data contamination prevention is achieved by using recent market data (post-March 2025)
that falls after the knowledge cutoff of contemporary LLMs. The benchmark is designed to be
continuously updated to avoid overlap with training corpora of future LLMs.

---

## 2. How Does It Evaluate Profitability in Real-World Markets?

StockBench simulates realistic stock trading with the following setup:

- **Initial capital:** $100,000 cash with zero holdings
- **Decision timing:** Daily at market open
- **Evaluation period:** 82 trading days (March 3 to June 30, 2025), capturing both volatility
  and trend reversals
- **Execution:** Each model runs three times with different random seeds; averaged performance
  is reported

### Information Signals Provided to LLMs Daily

**Price Information:**
- Opening prices for daily decisions
- 52-week high/low price ranges
- Historical price data for trend analysis

**Fundamental Indicators:**
- Market capitalization
- Price-to-earnings (P/E) ratio
- Dividend yield
- Recent quarterly dividends

**News & Sentiment:**
- Up to 5 most relevant news articles per stock (previous 48 hours)
- News retrieved via Finnhub's news-search API with time restrictions preventing future
  information leakage

**Historical Context:**
- Agent's current holdings
- Historical trading actions on held positions (past 7 days)
- Portfolio overview information for all investment targets

**Data Sources:** Market data from Polygon API; news from Finnhub API (both have free tiers).

---

## 3. What LLMs Are Tested?

### Proprietary / Closed-Source Models
- OpenAI O3
- OpenAI GPT-5
- Anthropic Claude-4-Sonnet

### Open-Weight Models
- Qwen3-235B-Instruct
- Qwen3-235B-Think (reasoning variant)
- Qwen3-30B-Think
- Qwen3-4B-Instruct
- Qwen3-Coder variant
- DeepSeek-V3
- DeepSeek-V3.1
- Kimi-K2
- GLM-4.5
- GPT-OSS-120B
- GPT-OSS-20B

All models were equipped with 32,768 token context windows and decoded using official
recommended settings.

---

## 4. What Markets/Assets Are Evaluated?

**Primary evaluation:** Top 20 stocks by weight in the **Dow Jones Industrial Average (DJIA)**,
covering six sectors:

| Sector | Stocks |
|--------|--------|
| Technology | Apple, Amazon, IBM, Salesforce, Microsoft |
| Finance | Travelers, JPMorgan Chase, American Express, Visa, Goldman Sachs |
| Consumer | Johnson & Johnson, Procter & Gamble, McDonald's, Home Depot |
| Industrial | Boeing, Honeywell, Caterpillar |
| Medical | Amgen, UnitedHealth Group |
| Materials | Sherwin-Williams |

**Rationale:** High-weighted DJIA stocks are "representative of the global stock market and are
less prone to short-term irrational sentiment-driven events" with transparent, publicly
accessible information.

**Extended analysis:** Additional experiments tested portfolio sizes of 5, 10, 20, and 30 stocks
to assess scalability.

---

## 5. Key Results

### Leaderboard (Main Evaluation Period: March-June 2025)

| Rank | Model | Final Return | Max Drawdown | Sortino Ratio |
|------|-------|-------------|--------------|---------------|
| 1 | Kimi-K2 | +1.9% | -11.8% | 0.0420 |
| 2 | Qwen3-235B-Instruct | +2.4% | -11.2% | 0.0299 |
| 3 | GLM-4.5 | +2.3% | -13.7% | 0.0295 |
| 4 | Qwen3-235B-Think | +2.5% | -14.9% | 0.0309 |
| 5 | OpenAI-O3 | +1.9% | -13.2% | 0.0267 |
| ... | ... | ... | ... | ... |
| 12 | **Passive Baseline** | **+0.4%** | **-15.2%** | **0.0155** |

Lower-performing models: GPT-5 (+0.3%), DeepSeek-V3 (+0.2%), GPT-OSS-20B (-2.8%).

### Key Findings

1. **Most LLM agents outperformed the passive buy-and-hold baseline** (0.4% return),
   though not by large margins.
2. **LLM agents managed downside risk more effectively** than the baseline, limiting
   drawdowns to -11% to -14% vs. the baseline's -15.2%.
3. **Reasoning-tuned models did NOT guarantee superior performance**: Qwen3-235B-Instruct
   outperformed its reasoning-tuned variant (Qwen3-235B-Think) despite the latter's
   stronger mathematical abilities.
4. **Static financial knowledge does not translate to trading success**: Excelling at financial
   QA tasks does not predict dynamic trading performance.
5. **Performance degrades during market downturns**: All LLM agents underperformed the
   passive baseline during the January-April 2025 downturn, while most exceeded it during
   the May-August 2025 upturn.
6. **Model rankings shift considerably between market regimes**, suggesting models excel
   under specific conditions rather than universally.

### Ablation: Data Source Impact

Tests progressively removing information sources (using Kimi-K2 and GPT-OSS-120B):

| Configuration | Kimi-K2 Return | GPT-OSS-120B Return |
|---------------|---------------|---------------------|
| Full input (prices + fundamentals + news) | +1.9% | -1.2% |
| Without news | +1.4% | -1.2% |
| Without news AND fundamentals | +0.6% | -3.4% |

Both news and fundamental data contribute meaningfully to trading decisions. GPT-OSS-120B
relies more heavily on explicit signals.

### Ablation: Portfolio Size

Performance degrades as portfolio size increases:
- Return variability increases substantially (coefficient of variation from 0.2 at 5 stocks to
  2.2-10.2 at 30 stocks)
- Larger models (Kimi-K2) show greater robustness
- Smaller models (GPT-OSS-120B) suffer severe performance deterioration

### Error Analysis

- **Thinking/reasoning models**: Lower arithmetic errors but higher schema/format errors
- **Instruction-tuned models**: Higher arithmetic errors but lower format compliance errors

---

## 6. Trading Horizon / Frequency Effects

- **Trading frequency:** Daily decisions at market open (fixed; no intraday trading)
- **Evaluation period:** 82 trading days (4 months), designed to capture both volatility and
  trend reversals
- **Key finding on temporal effects:** Model performance rankings **shift significantly between
  downturn and upturn periods**:
  - GPT-OSS-120B ranked last during downturns but topped rankings during upturns
  - This indicates "certain models may be better suited to specific market conditions"
  - LLM agents "may struggle to navigate bearish markets, highlighting a key area for
    future improvement"

The paper does not experiment with different trading frequencies (e.g., weekly or hourly);
all experiments use daily decision-making.

---

## 7. Baselines Compared

**Single baseline:** Equal-weight buy-and-hold strategy.
- Allocates initial capital equally across all 20 selected stocks at the period start
- No changes until evaluation period completion
- Represents "widely accepted benchmark in portfolio research, reflecting passive index
  tracking behavior"
- Achieved +0.4% return, -15.2% max drawdown, 0.0155 Sortino ratio

No traditional quantitative trading strategies (e.g., momentum, mean-reversion, MACD-based)
or ML/DL baselines were included. This is a notable limitation -- the only comparison is
against passive holding.

---

## 8. Evaluation Metrics

Three primary metrics:

1. **Final Return (%):** Percentage change in portfolio value from initial V_0 to final V_T,
   calculated as (V_T - V_0) / V_0
2. **Maximum Drawdown (%):** Largest decline from portfolio peak to trough, measuring
   worst-case downside risk
3. **Sortino Ratio:** Risk-adjusted return metric that penalizes only downside volatility
   (not all volatility, unlike Sharpe ratio)

### Composite Ranking

Models are ranked using a composite score derived from z-scores:

```
Composite = [ z(Final Return) - z(Max Drawdown) + z(Sortino Ratio) ] / 3
```

This balances profitability and risk, "rewarding models that achieve high returns while
effectively managing downside exposure."

---

## 9. Code/Data Availability

- **Project website:** https://stockbench.github.io/
- **GitHub repository:** https://github.com/ChenYXxxx/stockbench
- **License:** Apache 2.0
- **Language:** 98.3% Python
- **Status:** Open-source with plans for continuous updates

**Required APIs:** Polygon (market data), Finnhub (news), and an LLM provider API (OpenAI,
etc.). Free tiers available for Polygon and Finnhub.

**Installation:** Python 3.11, conda environment, pip requirements.

**Running:** `scripts/run_benchmark.sh` executes trading simulations; outputs saved to
`storage/reports/backtest/`.

---

## Relevance to Our Project

### Strengths to Build On
- Clean, realistic daily trading simulation framework
- Well-defined agent workflow (overview -> analysis -> decision -> execution)
- Contamination-free evaluation design
- Real market data with news integration
- Multiple ablation studies showing value of different information sources
- Open-source and extensible

### Gaps / Opportunities for Improvement
- **Only one baseline (buy-and-hold)**: No comparison to traditional quant strategies, RL
  agents, or ML models. Adding momentum, mean-reversion, or simple technical analysis
  baselines would be valuable.
- **Only daily trading frequency**: No exploration of weekly, hourly, or multi-day holding
  periods. Trading horizon effects remain unexplored.
- **Limited to DJIA blue-chip stocks**: No small-cap, crypto, or international market testing.
- **Short evaluation period (4 months)**: Longer backtests would better assess robustness
  across full market cycles (bull, bear, sideways).
- **No multi-agent or ensemble approaches tested**.
- **No cost modeling**: Transaction costs, slippage, and market impact not explicitly modeled.
- **Reasoning models underperform expectations**: Opportunity to explore why and design
  better prompting strategies.
- **Market regime sensitivity**: Models that adapt strategy to detected market regime could
  significantly outperform.
