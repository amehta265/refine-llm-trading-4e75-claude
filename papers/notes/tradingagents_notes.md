# TradingAgents: Multi-Agents LLM Financial Trading Framework

**Paper:** arXiv:2412.20138v7 [q-fin.TR], 3 Jun 2025
**Authors:** Yijia Xiao, Edward Sun, Di Luo, Wei Wang
**Affiliations:** UCLA, MIT, Tauric Research
**Code:** https://github.com/TauricResearch/TradingAgents

---

## 1. What is TradingAgents? (Framework Description)

TradingAgents is a multi-agent LLM-based stock trading framework that simulates the organizational structure and collaborative dynamics of real-world trading firms. Rather than relying on a single LLM agent or independent multi-agent data gathering, TradingAgents assigns specialized roles to different LLM agents that mirror the teams found in professional trading firms: analysts, researchers, traders, risk managers, and fund managers.

The framework addresses two key limitations of prior work:
- **Lack of Realistic Organizational Modeling:** Most prior frameworks fail to capture the complex interactions between agents that mimic real-world trading firms. TradingAgents bridges this gap by simulating multi-agent decision-making processes typical of professional trading teams.
- **Inefficient Communication Interfaces:** Prior systems relying purely on natural language suffer from a "telephone effect" where details are lost over extended conversations. TradingAgents introduces a **hybrid communication protocol** that combines structured outputs (for control, clarity, and reasoning) with natural language dialogue (for debate and collaboration).

Key design principles:
- Agents follow the **ReAct prompting framework** (Yao et al., 2023), synergizing reasoning and acting.
- The environment state is shared and monitored by all agents.
- Communication between agents primarily uses structured documents and reports, with natural language reserved for debates and discussions.
- The framework is designed to be deployed **without GPU**, relying only on API credits, and supports seamless swapping of backbone LLMs.

---

## 2. Agent Roles and Architecture

TradingAgents defines **seven distinct agent roles** organized into five teams:

### I. Analyst Team (4 agents, concurrent operation)
1. **Fundamental Analyst** -- Evaluates company fundamentals: financial statements, earnings reports, insider transactions. Assesses intrinsic value to identify under/overvalued stocks.
2. **Sentiment Analyst** -- Processes social media posts, sentiment scores, and insider sentiments from public information and social media. Predicts short-term impact of collective investor behavior on stock prices.
3. **News Analyst** -- Analyzes news articles, government announcements, macroeconomic indicators. Identifies events that could influence market movements.
4. **Technical Analyst** -- Calculates and selects relevant technical indicators (MACD, RSI, Bollinger Bands, etc.). Analyzes price patterns and trading volumes to forecast price movements.

### II. Researcher Team (2 agents + facilitator, debate format)
5. **Bullish Researcher** -- Advocates for investment opportunities by highlighting positive indicators, growth potential, and favorable conditions.
6. **Bearish Researcher** -- Focuses on potential downsides, risks, and unfavorable signals. Questions viability of investment strategies.
- They engage in **n rounds of debate** (determined by a facilitator agent). The facilitator reviews debate history, selects the prevailing perspective, and records it as a structured entry.

### III. Trader Agent
7. **Trader** -- Executes trading decisions based on analyst reports and researcher perspectives. Evaluates recommendations, decides timing/size of trades, places buy/sell orders, adjusts portfolio allocations. Can operate with varied **risk profiles**.

### IV. Risk Management Team (3 perspectives + facilitator)
- **Risk-seeking (Aggressive)** agent
- **Neutral** agent
- **Risk-conservative (Safe)** agent
- They query the trader's decision and deliberate from three perspectives to adjust the trading plan within risk constraints, engaging in n rounds of discussion guided by a facilitator.

### V. Fund Manager
- Reviews the risk management team's discussion, determines appropriate risk adjustments, and updates the trader's decision. Approves and executes the final trade.

### Workflow Summary
1. Analyst Team concurrently gathers and analyzes market data (structured reports)
2. Researcher Team debates bullish vs. bearish perspectives using analyst reports (natural language debate)
3. Trader makes trading decision based on research output (structured decision + report)
4. Risk Management Team evaluates the decision from aggressive/neutral/conservative perspectives (natural language debate)
5. Fund Manager approves and executes the final trade

### Tools Available to Agents
- **Sentiment Analyst:** Web search engines, Reddit search APIs, X/Twitter search tools, sentiment score calculation algorithms
- **Technical Analyst:** Code execution, technical indicator calculation, trading pattern analysis
- **News Analyst:** EODHD news, Finnhub news APIs
- **Fundamentals Analyst:** Finnhub company profile, financials history, insider sentiment/transactions APIs
- **Market data:** Yahoo Finance (YFin) data retrieval, stockstats indicators

---

## 3. Financial Tasks/Markets Evaluated

- **Task:** Stock trading (buy/sell/hold decisions) on individual equities
- **Market:** US stock market
- **Stocks evaluated:**
  - **Primary evaluation (with full metrics):** AAPL (Apple), GOOGL (Google/Alphabet), AMZN (Amazon)
  - **Mentioned in simulation setup:** Also includes NVDA (Nvidia), MSFT (Microsoft), META (Meta) -- data collected for these but detailed results shown only for AAPL, GOOGL, AMZN
- **Trading frequency:** Daily decisions (each trading day, agents analyze available data and generate buy/sell/hold signals)
- **Backtest period:** January 1, 2024 to March 29, 2024 (approximately 3 months)

---

## 4. What LLMs Are Used?

The framework uses a **two-tier LLM strategy** based on task complexity:

### Quick-thinking models (for fast, low-depth tasks)
- **gpt-4o-mini**
- **gpt-4o**
- Used for: summarization, data retrieval, converting tabular data to text

### Deep-thinking models (for reasoning-intensive tasks)
- **o1-preview**
- Used for: decision-making, evidence-based report writing, data analysis
- Leverages multi-round reasoning architectures

### Task-to-model assignment
- **Analyst nodes:** Deep-thinking models for analysis; quick-thinking models for data retrieval from APIs/tools
- **Researchers and traders:** Deep-thinking models for generating insights and supporting decisions
- **Auxiliary expert models:** Used for specialized tasks like sentiment analysis (specific models not named)

### Design for flexibility
- No GPU required -- relies only on API credits
- Seamless exchangeability of backbone models
- Any locally hosted or API-accessible model can be swapped in
- Future-proof: supports integration of improved reasoning models or finance-tuned models

---

## 5. What Datasets Are Used?

The framework uses a **multi-asset, multi-modal financial dataset** comprising:

1. **Historical Stock Prices:** Open, high, low, close, volume, and adjusted close prices (Jan 1, 2024 -- Mar 29, 2024). Source: Yahoo Finance.

2. **News Articles:** Daily news updates from:
   - Bloomberg
   - Yahoo
   - EODHD
   - FinnHub
   - Reddit
   - Coverage: company-specific developments, global events, macroeconomic trends, government updates

3. **Social Media Posts and Sentiment:**
   - Sources: Reddit (wallstreetbets, stocks, investing, SecurityAnalysis, Finance, Economics subreddits), X/Twitter, other platforms
   - Sentiment scores calculated by auxiliary language models

4. **Insider Sentiments and Transactions:**
   - Source: SEDI, relevant company filings
   - Includes insider transaction data (buys, sells, gifts)

5. **Financial Statements and Earnings Reports:**
   - Quarterly and annual reports filed by companies
   - Source: Finnhub API

6. **Company Profiles and Financial History:**
   - Company descriptions, target industries, financial history
   - Source: Finnhub, third-party reporters

7. **Technical Indicators:**
   - 60 standard technical analysis indicators calculated per asset
   - Includes: MACD, RSI, Bollinger Bands, ADX, ATR, Supertrend, CCI, VWMA, and others
   - Source: stockstats library

---

## 6. Key Results

### Performance Table (Jan 1 -- Mar 29, 2024)

| Metric | AAPL | GOOGL | AMZN |
|--------|------|-------|------|
| **Cumulative Return (CR%)** | 26.62 | 24.36 | 23.21 |
| **Annualized Return (ARR%)** | 30.50 | 27.58 | 24.90 |
| **Sharpe Ratio (SR)** | 8.21 | 6.39 | 5.60 |
| **Max Drawdown (MDD%)** | 0.91 | 1.69 | 2.11 |

### Improvement over best baseline

| Metric | AAPL | GOOGL | AMZN |
|--------|------|-------|------|
| **CR improvement** | +24.57% | +16.58% | +6.10% |
| **ARR improvement** | +28.43% | +19.49% | +7.30% |
| **SR improvement** | +6.57 | +4.26 | +2.07 |

### Key findings:
- TradingAgents achieves at least **23.21% cumulative return** and **24.90% annual return** across three tested stocks, surpassing best baselines by a margin of at least 6.1%.
- On AAPL -- a challenging case due to market volatility (Buy & Hold returned -5.23%) -- TradingAgents achieved **26.62% cumulative return**, demonstrating ability to perform under adverse conditions.
- **Sharpe Ratio** was exceptionally high (up to 8.21), which the authors acknowledge exceeds typical empirical ranges. They attribute this to few pullbacks during the 3-month period and note calculations were verified for correctness.
- **Maximum Drawdown** remained within manageable limits (not exceeding ~2.11%), demonstrating effective risk-return balance.
- TradingAgents maintained relatively low maximum drawdown compared to many baselines despite achieving much higher returns.
- The framework offers **superior explainability** -- all decisions are communicated in natural language with detailed reasoning, tool usage, and thought processes (illustrated via full trading logs in the appendix).

### Important caveats noted by authors:
- Backtesting was limited to **3 months** due to intensive LLM and tool use (11 LLM calls & 20+ tool calls per prediction).
- The exceptionally high Sharpe Ratios may be partially an artifact of the short evaluation period with few pullbacks.
- Future work will optimize LLM reasoning & tool use to enable longer backtesting under limited budgets.

---

## 7. Findings About Trading Horizon or Decision Frequency

- **Decision frequency:** Daily -- agents make one trading decision per trading day.
- **Evaluation horizon:** 3 months (Jan 1 -- Mar 29, 2024), acknowledged as short.
- **Reason for short horizon:** Computational cost -- each daily prediction requires 11 LLM calls and 20+ tool calls, making longer backtesting expensive.
- **No explicit analysis of different trading horizons** (e.g., weekly, monthly) or frequency optimization.
- **Future work** mentions plans to optimize LLM reasoning and tool use to enable longer backtesting under limited budgets, and deploying the framework in a live trading environment.
- The authors note that the Sharpe Ratio exceeding empirical norms may be a consequence of the short evaluation window.

---

## 8. What Baselines Are Compared?

Five established trading strategies are used as baselines:

1. **Buy and Hold (B&H):** Invest equal amounts in all selected stocks and hold throughout the simulation period. This represents the market benchmark.

2. **MACD (Moving Average Convergence Divergence):** A trend-following momentum strategy generating buy/sell signals based on crossover points between the MACD line and signal line.

3. **KDJ + RSI:** A momentum strategy combining KDJ (stochastic oscillator) and RSI indicators to identify overbought and oversold conditions for trading signals.

4. **ZMR (Zero Mean Reversion):** A mean reversion trading strategy generating signals based on price deviations from and subsequent reversions to a zero reference line.

5. **SMA (Simple Moving Average):** A trend-following strategy generating trading signals based on crossovers between short-term and long-term moving averages.

**Notable absence:** No comparison against other LLM-based trading systems (e.g., FinMem, FinAgent, TradingGPT, FinCon), only rule-based/quantitative baselines are used.

---

## 9. Evaluation Metrics Used

Four key metrics are used:

1. **Cumulative Return (CR):** Total return over the simulation period.
   - CR = ((V_end - V_start) / V_start) x 100%

2. **Annualized Return (AR):** Cumulative return normalized over the number of years.
   - AR = ((V_end / V_start)^(1/N) - 1) x 100%

3. **Sharpe Ratio (SR):** Risk-adjusted return comparing portfolio excess return over risk-free rate to volatility.
   - SR = (R_avg - R_f) / sigma
   - Where R_f is the risk-free rate (e.g., 3-month Treasury bill yield)

4. **Maximum Drawdown (MDD):** Largest peak-to-trough decline in portfolio value.
   - MDD = max over t of ((Peak_t - Trough_t) / Peak_t) x 100%

Higher CR, AR, and SR are better. Lower MDD is better.

---

## 10. Code/Data Availability

- **Code:** Available at https://github.com/TauricResearch/TradingAgents
- **Organization:** Tauric Research (https://tauric.ai)
- **Data:** The paper describes a multi-modal dataset collected from multiple APIs (Yahoo Finance, EODHD, Finnhub, Reddit, X/Twitter). It is not clear whether the exact dataset used in experiments is publicly released separately, but the framework code presumably includes data collection tooling.
- **Reproducibility note:** The framework requires API credits for LLMs (OpenAI models) and financial data APIs. No GPU is needed.

---

## Additional Notes

### Strengths
- Novel organizational structure that mirrors real trading firms
- Hybrid communication protocol (structured reports + natural language debate) avoids the "telephone effect"
- Multi-perspective risk management (aggressive, neutral, conservative)
- High explainability -- every decision includes full reasoning chain, tool calls, and evidence
- Modular design allows swapping backbone LLMs

### Limitations and Gaps
- **Very short evaluation period** (3 months) -- results may not generalize
- **No comparison to other LLM-based trading systems** -- only rule-based baselines
- **High computational cost** (11 LLM calls + 20+ tool calls per daily decision) limits scalability
- **No multi-stock portfolio optimization** -- evaluates individual stocks independently
- **Exceptionally high Sharpe Ratios** may be artifacts of the short, favorable test period
- **No analysis of different market regimes** (bull, bear, sideways)
- **No transaction cost modeling** mentioned explicitly
- **Limited stock universe** -- only tech megacaps tested

### Relevance to Our Research
- Demonstrates that multi-agent debate and structured communication improve LLM trading decisions
- The role-specialization approach (fundamental, sentiment, news, technical) is a useful design pattern
- The bull/bear debate mechanism for research and the aggressive/neutral/conservative debate for risk management are interesting architectural choices
- The 3-month limitation highlights the need for more efficient LLM-based trading pipelines
- The absence of LLM-based baselines is a notable gap that could be addressed in follow-up work
