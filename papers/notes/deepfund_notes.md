# DeepFund Paper Notes

**Paper**: "Time Travel is Cheating: Going Live with DeepFund for Real-Time Fund Investment Benchmarking"
**Authors**: Changlun Li, Yao Shi, Chen Wang, Qiqi Duan, Runke Ruan, Weijie Huang, Haonan Long, Lijun Huang, Yuyu Luo, Nan Tang
**ArXiv**: 2505.11065 (May 2025, v2 October 2025)
**Venue**: NeurIPS 2025 (Poster)

**NOTE**: The local file `deepfund_2503_01893.pdf` is **mislabeled** -- it actually contains
a completely different paper ("BiHRNN: Bi-Directional Hierarchical Recurrent Neural
Network for Inflation Forecasting" by Maya Vilenko, Tel Aviv University). The real
DeepFund paper is arXiv 2505.11065. Notes below are sourced from the actual DeepFund
paper via the arXiv HTML version and GitHub repository.

---

## 1. What is DeepFund?

DeepFund is a **live fund investment benchmarking tool** that evaluates LLMs' capability
to manage fund investments in **real-time market conditions**. Unlike existing benchmarks
that rely on historical backtesting, DeepFund connects directly with real-time stock
market data -- specifically data published **after** each model's pretraining cutoff -- to
ensure fair, leakage-free evaluations.

Key features:
- **Live Environment**: Continuously ingests real-time market data, fund assets, and trading history
- **Multi-Agent Architecture**: LLMs assume three roles (Financial Planner, Analyst Team, Portfolio Manager) mirroring real fund management structures
- **Interactive Web Interface**: Displays performance metrics and comparative analysis at https://deepfund.paradoox.ai/
- **Modular API Gateway**: Supports multiple financial data providers (Yahoo Finance, Alpha Vantage)
- **Research only**: The system does NOT execute real trades

---

## 2. "Time Travel is Cheating" -- The Core Problem

The paper's central thesis is that **historical backtesting creates information leakage**
because LLMs may have been pre-trained on the very historical data used for evaluation.
This allows models to "time travel" by leveraging future knowledge embedded in their
training corpora rather than genuinely predicting outcomes.

**Concrete example from paper**: GPT-4o was trained on data up to October 2023,
whereas DeepSeek-V3's training extends until July 2024. If we evaluate DeepSeek-V3
on 2021-2023 data, it will have effectively already "seen" those market conditions during
pre-training, yielding **overly optimistic metrics** that do not reflect true predictive
capability.

This fundamentally undermines the validity of traditional LLM trading strategy evaluations
and is the primary motivation for building a live evaluation framework.

---

## 3. Real-Time vs. Backtesting Evaluation Methodology

### DeepFund's Approach
- Uses market data exclusively published **after** each model's knowledge cutoff date
- **Live testing period**: March 17 -- April 17, 2025 (24 trading days)
- The period captured significant market events:
  - FOMC Meeting (March 18-19)
  - Tariff announcements (April 2-9)
- Each LLM manages **$100,000 initial capital**
- **Daily trading frequency**

### Multi-Agent Workflow (Orchestrator-Worker Paradigm)
1. **Financial Planner** selects and assigns analysts based on market conditions
   - Supports deterministic mode (predefined analyst selection) and dynamic mode (self-reasoning)
2. **Analyst Team** (6 specialized roles) concurrently generates structured signals:
   - Technical Analyst: Price patterns, RSI, volatility, support/resistance
   - Fundamental Analyst: Financial statements, earnings, valuation metrics
   - Insider Analyst: Executive buys/sells, transaction timing
   - Company News Analyst: Sentiment, material events
   - Macro Economic Analyst: GDP, inflation, unemployment, interest rates
   - Policy Analyst: Fiscal/monetary policy, regulation
   - Each generates standardized signals: **Bullish, Bearish, or Neutral** with justifications
3. **Portfolio Manager** synthesizes signals, evaluates risks, decides actions (Buy/Sell/Hold)
   - Manages risk control (position sizing, cash reserves)
   - Maintains **dual-memory architecture** (short-term operational, long-term historical)

### Memory Architecture
- **Short-term memory**: FundState object containing current portfolio positions, recent decisions, active signals
- **Long-term memory**: Comprehensive trading history records enabling pattern recognition and learning

### Signal Validity
The system achieved **4,144 signals (96% validity)** and **1,059 trading decisions (98% validity)** out of 4,320 signals and 1,080 decisions respectively.

---

## 4. LLMs Tested

Nine state-of-the-art models from leading global institutions:

| Provider   | Model              | Release   | Knowledge Cutoff |
|------------|--------------------|-----------|--------------------|
| OpenAI     | GPT-4.1            | Apr 2025  | June 2024          |
| Meta       | Llama 4 Scout      | Apr 2025  | Aug 2024           |
| Google     | Gemini 2.5 Flash   | Apr 2025  | Jan 2025           |
| Anthropic  | Claude 3.7 Sonnet  | Feb 2025  | Oct 2024           |
| xAI        | Grok 3 mini Beta   | Feb 2025  | Nov 2024           |
| DeepSeek   | DeepSeek-V3        | Mar 2025  | Dec 2024           |
| Alibaba    | Qwen2.5-Max        | Jan 2025  | N/A                |
| ByteDance  | Doubao-1.5-pro     | Jan 2025  | N/A                |
| Zhipu      | GLM-4-Air          | Apr 2025  | N/A                |

---

## 5. Markets and Assets Evaluated

### Primary Portfolio
Based on **Berkshire Hathaway's top five holdings** (Q1 2025):
- Apple (AAPL)
- American Express (AXP)
- Bank of America (BAC)
- Coca-Cola (KO)
- Chevron (CVX)

### Extended Analysis (Appendix)
Separate sector-specific universes tested:
- Gold
- Oil & Gas
- Crypto
- Banking

### Exchange
US stock market. Testing period: March 17 -- April 17, 2025.

---

## 6. Key Results -- Real-Time vs. Backtested Performance

### Critical Finding: Only ONE model achieved profitability in live trading.

| Model         | Cumulative Return | Buy & Hold Return | Sharpe Ratio | Max Drawdown | Win Rate |
|---------------|-------------------|--------------------|--------------|--------------|----------|
| **Grok 3**    | **+1.1%**         | -3.09%             | **0.51**     | **5.5%**     | **61%**  |
| Gemini 2.5    | -1.9%             | -1.58%             | -1.37        | 6.4%         | --       |
| Claude 3.7    | -3.7%             | -2.94%             | -1.45        | 10.1%        | --       |
| Llama 4       | -4.3%             | -3.62%             | -2.42        | 8.9%         | --       |
| DeepSeek-V3   | -5.7%             | -5.6%              | -1.39        | 14.5%        | --       |
| GPT-4.1       | -5.9%             | -4.41%             | -1.87        | 12.8%        | --       |
| Qwen2.5-Max   | -6.7%             | -4.86%             | -3.12        | 10.7%        | --       |
| GLM-4-Air     | -7.5%             | -3.90%             | -2.31        | 13.2%        | --       |
| Doubao-1.5    | -8.1%             | -5.37%             | -2.35        | 13.6%        | --       |
| S&P 500       | -6.91%            | N/A                | 0.3          | 13.7%        | --       |

**Risk-Free Rate**: 4.29% (1-month US Treasury bill, April 17, 2025)

### Key Observations
- "Even cutting-edge models such as DeepSeek-V3 and Claude-3.7-Sonnet incur net trading losses within DeepFund's real-time evaluation environment."
- Most US-produced LLMs (except GPT-4.1) demonstrated lower return losses than Chinese-produced LLMs during the bearish tariff period (April 3-9).
- This contrasts sharply with backtesting results where models often appear profitable -- precisely demonstrating the "time travel" problem.

---

## 7. Trading Horizon Effects

### Short-Term (24-day primary evaluation)
- Heightened performance variance across all models
- Market turbulence (tariff announcements, FOMC) created challenging conditions
- Only Grok 3 achieved positive returns

### Extended Evaluation (Full Q2 2025, through June 2025)
- Some models (GPT-4.1, Claude 3.7, DeepSeek-V3) **eventually achieved net profits** over the longer horizon
- Grok 3 maintained leading profitability throughout
- GLM-4, Qwen Max, Gemini 2.5 **remained unprofitable** even over the longer period
- The authors acknowledge: "The evaluation period was short and occurred during a volatile market, which could skew our results toward specific trading approaches."

### Implication
Shorter trading horizons amplify the impact of market volatility and event-driven risk.
Models with conservative cash management (like Grok 3, which maintained ~60% cash
reserves) proved more resilient in short-term turbulence. Aggressive strategies (like
DeepSeek-V3 with ~90% invested) suffered larger drawdowns.

---

## 8. Baselines Compared

1. **Passive Buy & Hold Strategy**: Shows what holding initial positions would yield (computed per-model since each starts with different allocations)
2. **S&P 500**: Market-level performance benchmark (-6.91% during test period)
3. **Risk-Free Rate**: 4.29% (1-month US Treasury bill, April 17, 2025)

No traditional quantitative trading strategies (e.g., momentum, mean-reversion) or
ML-based strategies were included as baselines -- the comparison is primarily
LLM-vs-LLM and LLM-vs-passive/market benchmarks.

---

## 9. Evaluation Metrics

Seven standard financial metrics:

1. **Cumulative Return (CR)**: Total percentage gain/loss from initial investment
2. **Cumulative Return at Buy & Hold (CR_bnh)**: Return if initial positions were passively held
3. **Sharpe Ratio (SR)**: Excess return divided by volatility (annualized); higher = better risk-adjusted return
4. **Maximum Drawdown (MDD)**: Largest portfolio decline from peak to trough
5. **Win Rate (WR)**: Percentage of profitable trades executed
6. **Beta**: Portfolio volatility relative to S&P 500 (market sensitivity)
7. **Alpha**: Excess return compared to market benchmark (true skill measure)

---

## 10. Code and Data Availability

- **GitHub**: https://github.com/HKUSTDial/DeepFund (MIT License)
- **Live Dashboard**: https://deepfund.paradoox.ai/
- **Total operational cost**: ~$100 (LLM APIs 40%, financial data 40%, cloud database 20%)
- **Database options**: Supabase (cloud PostgreSQL) or SQLite (local)
- **Requirements**: Python 3.11+
- Won "Best Open-source Award" at 2025 AI Agent competition

---

## Key Takeaways for Our Research

1. **Backtesting is fundamentally flawed for LLM evaluation** due to training data contamination. Any LLM trading benchmark must account for knowledge cutoff dates.

2. **Real-time performance is dramatically worse** than backtested performance for all tested LLMs. This is the paper's most important finding.

3. **Conservative strategies outperform** in volatile markets: Grok 3's success came from maintaining ~60% cash reserves, low-frequency trading, and diversification -- not from superior prediction.

4. **Most LLMs behave like retail speculators**, not professional fund managers. High-frequency, momentum-driven approaches (DeepSeek-V3, Doubao) led to the worst outcomes.

5. **Signal quality matters**: Grok 3 generated more directional (Bullish/Bearish) signals, while DeepSeek-V3 defaulted to Neutral signals during critical moments, showing less market sensitivity.

6. **Multi-agent architecture** with specialized analyst roles is the current standard approach for LLM trading systems, but the bottleneck is the quality of the LLM's reasoning, not the system architecture.

7. **Longer horizons may help**: Extended Q2 evaluation showed some models recovering, suggesting that LLM trading strategies may need longer evaluation windows to demonstrate value (or that markets simply recovered).

8. **Limitation**: Only 5 stocks tested in the primary evaluation, all large-cap US equities. Sector analysis in appendix is limited. No comparison against traditional quant strategies.
