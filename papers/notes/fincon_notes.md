# FinCon: A Synthesized LLM Multi-Agent System with Conceptual Verbal Reinforcement for Enhanced Financial Decision Making

**Paper:** arXiv:2407.06567v3 (NeurIPS 2024)
**Authors:** Yangyang Yu, Zhiyuan Yao, Haohang Li, Zhiyang Deng, Yuechen Jiang, Yupeng Cao, Zhi Chen, Jordan W. Suchow, Zhenyu Cui, Rong Liu, Zhaozhuo Xu, Denghui Zhang, Koduvayur Subbalakshmi, Guojun Xiong, Yueru He, Jimin Huang, Dong Li, Qianqian Xie
**Affiliations:** Stevens Institute of Technology, Harvard University, The Fin AI

---

## 1. What is FinCon? (Multi-Agent System Description)

FinCon is an LLM-based multi-agent framework designed for sequential financial decision-making tasks. Inspired by real-world investment firm structures, it organizes agents in a **manager-analyst hierarchy**:

### Architecture: Two Main Components

**A. Manager-Analyst Agent Group**
- **Analyst Agents:** Multiple specialized agents, each processing a single modality/source of market data in a uni-modal fashion. Seven distinct types of analyst agents are implemented:
  - 3 textual data processing agents (extract insights/sentiments from daily news and financial reports)
  - 1 audio agent (uses Whisper API to interpret ECC audio recordings)
  - 1 data analysis agent (computes financial metrics like momentum and CVaR from tabular time series)
  - 1 stock selection agent (oversees portfolio selection using risk diversification methods)
  - 1 report analyst agent (processes investment research reports, e.g., Zacks)
- **Manager Agent:** The sole decision-maker. Consolidates distilled insights from all analysts, receives risk alerts from the risk-control component, refines investment beliefs, and conducts self-reflection on previous actions. For portfolio management, it calculates portfolio weights using mean-variance convex optimization constrained by directional trading decisions.

**B. Risk-Control Component** (see section 2 below)

### Modular Design of Each Agent
Each agent integrates four modules:
1. **General Configuration & Profiling Module:** Defines task types, trading targets, sector info, and each agent's role/responsibilities.
2. **Perception Module:** Defines how each agent interacts with the market and other agents. Converts raw data and inter-agent communications into LLM-compatible formats.
3. **Memory Module:** Three types:
   - *Working memory:* Observation, distillation, and refinement of events (analogous to human working memory).
   - *Procedural memory:* Records historical actions, outcomes, and reflections during sequential decision-making. Each analyst agent has distinct memory decay rates reflecting the timeliness of its data source. Memory events are retrieved and ranked by recency, relevance, and importance (using an Ebbinghaus forgetting curve-inspired importance decay and cosine similarity-based relevance).
   - *Episodic memory:* Exclusive to the manager agent; stores actions, PnL series from prior episodes, and updated conceptual investment beliefs.
4. **Action Module:** Manager agent conducts trading actions and reflects on reasons/contributions.

### Communication Structure
- Streamlined hierarchical (manager-analyst) communication reduces redundant peer-to-peer interaction.
- Analyst agents send distilled insights upward to the manager.
- Manager sends selective feedback downward to relevant analysts.
- This avoids the high communication costs of peer-to-peer debate/discussion systems like StockAgent.

### POMDP Formulation
The quantitative trading task is formally modeled as an infinite-horizon Partially Observable Markov Decision Process (POMDP). All agent policies are parameterized by textual prompts, and the system optimizes these policies through verbal reinforcement (prompt optimization) rather than traditional gradient descent.

---

## 2. How Does Conceptual Verbal Reinforcement (CVRF) Work?

FinCon uses a **dual-level risk-control mechanism**:

### A. Within-Episode Risk Control (operates during both training and testing)
- Monitors daily **Conditional Value at Risk (CVaR)** -- the average of the worst-performing 1% of daily trading PnLs.
- A sudden drop in CVaR triggers a risk alert: the manager agent immediately adopts a **risk-averse** stance for that day's trading, regardless of prior risk status.
- Also triggered when daily PnL is negative, prompting manager self-reflection.

### B. Over-Episode Risk Control / CVRF (operates only during training)
This is the core innovation -- **Conceptual Verbal Reinforcement (CVRF)**:

1. **Episode Comparison:** After each training episode, the system compares the objective function values (cumulative discounted PnL) of the current episode (k) vs. the previous episode (k-1) to determine which had higher performance.
2. **Trajectory Analysis:** Sustained profitable and losing trades from both episodes are passed to the risk-control component.
3. **Conceptualization:** The risk-control component (an LLM) summarizes **conceptualized investment insights** from each episode -- distilling performance into named conceptual aspects such as "historical momentum," "news insights," "Form 10-Q," "sector trends," etc.
4. **Comparison & Meta-Prompt Generation:** The two sets of conceptualized insights are compared, and the system reasons about why one episode performed better, producing a **textual optimization direction** (meta-prompt).
5. **Learning Rate Calculation:** The **overlapping percentage** of trading decision sequences between the two consecutive episodes serves as the "learning rate" (tau). This is analogous to learning rate in gradient descent -- it controls how much the prompt should change.
6. **Prompt Update (Textual Gradient Descent):** The prompts are updated via: theta <-- Mr(theta, tau, meta_prompt). The risk-control LLM edits the existing prompts based on the meta-prompt feedback, ensuring stable and incremental improvement.
7. **Selective Propagation:** Updated beliefs are first received by the manager, then **selectively propagated** to relevant analyst agents only, minimizing over-communication.

### Key Analogy to Traditional Optimization:
| Factor | Gradient-based model optimizer | LLM-based prompt optimizer |
|--------|-------------------------------|---------------------------|
| Upgrade direction | Model value gradient momentum | Prompt reflection trajectory |
| Update method | Learning rate descent | Overlapping percentage of trading decisions |

### Convergence Behavior:
- Trading action overlap between episodes increases over training (e.g., for GOOG: 46.9% -> 71.4% -> 81.6% across 4 episodes).
- Investment beliefs become progressively more specific and actionable.
- Only **4 training episodes** needed -- far fewer than traditional RL agents.

---

## 3. What Financial Tasks Are Evaluated?

Two main tasks:

### A. Single Stock Trading
- Sequential buy/sell/hold decisions on individual stocks.
- "Sell" means short-selling is allowed (negative positions permitted).
- Daily decision frequency.

### B. Portfolio Management
- Selecting and trading a small portfolio of stocks.
- Stock selection agent constructs a pool considering statistical correlations between returns.
- Manager determines portfolio weights via mean-variance optimization (Equation 1).
- Portfolio weights rebalanced on a **daily basis**.
- Two portfolios tested: Portfolio 1 (TSLA, MSFT, PFE) and Portfolio 2 (AMZN, GM, LLY), selected from a pool of 42 stocks with sufficient news data (>800 articles each).

---

## 4. What LLMs Are Used?

- All LLM-based agent systems (including FinCon and all LLM baselines) use **GPT-4-Turbo** as the backbone model.
- Temperature set at **0.3** for general operation; **0.0** specifically for belief generation (to ensure consistent belief output).
- The framework diagram also references other foundation models (Claude, Gemini, Llama3, Gemma2, Mistral) and financial fine-tuned LLMs (BloombergGPT, FinMA, XuanYuan 2.0, FinGPT) as potential options, but experiments use GPT-4-Turbo exclusively.
- Whisper API used for audio (ECC) transcription/interpretation.

---

## 5. What Datasets/Markets Are Evaluated?

### Data Period
- January 3, 2022 to June 10, 2023 (full period)
- Training: January 3, 2022 to October 4, 2022
- Testing: October 5, 2022 to June 10, 2023
- DRL agents trained on extended period: January 1, 2018 to October 4, 2022 (for convergence)
- High-volatility experiment: Training Jan 17 - Mar 31, 2022; Testing Apr 1 - Oct 15, 2022

### Multi-Modal Data Sources
1. **Stock prices:** Daily OHLCV from Yahoo Finance (via yfinance)
2. **Daily financial news:** From Refinitiv Real-Time News (mainly Reuters)
3. **Company filings:** Form 10-Q (quarterly, from SEC/EDGAR) and Form 10-K (annual, from SEC/EDGAR)
4. **Earnings Conference Calls (ECC):** Audio recordings (unstructured audio data)
5. **Zacks Equity Research:** Zacks Rank (short-term rating), analyst reports (reason to buy/sell, risks)
6. **Investment research reports**

### Stocks Traded (Single Stock)
Eight stocks: TSLA, AMZN, NIO, MSFT, AAPL, GOOG, NFLX, COIN
- Covers bullish (GOOG, MSFT), bearish (NIO), and mixed (TSLA) market conditions.
- COIN (IPO April 2021) used to test LLM advantage over DRL with limited historical data.

### Portfolios (Portfolio Management)
- Portfolio 1: TSLA, MSFT, PFE
- Portfolio 2: AMZN, GM, LLY
- Selected from a pool of 42 stocks with sufficient news data (>800 articles).

### Market: US equities only.

---

## 6. Key Results

### Single Stock Trading
- FinCon **significantly outperforms** all LLM-based and DRL-based approaches in Cumulative Returns (CR) and Sharpe Ratios (SR) across all 8 stocks.
- Achieves one of the lowest Max Drawdown (MDD) values across most assets.
- Highlights:
  - TSLA: CR 82.87%, SR 1.972 (vs. B&H 6.43%, next best LLM FinMem 34.62%)
  - AMZN: CR 24.85%, SR 0.904 (all other agents negative)
  - NIO (bearish): CR 17.46%, SR 0.335 (B&H lost -77.21%)
  - NFLX: CR 69.24%, SR 2.370
  - COIN: CR 57.05%, SR 0.825 (DRL agents could not converge due to limited data)
- Consistent superiority regardless of market conditions (bullish, bearish, mixed).

### Portfolio Management
- Portfolio 1 (TSLA, MSFT, PFE): **CR 113.84%, SR 3.269**, MDD 16.16%
  - vs. Markowitz MV: CR 12.64%, SR 0.614
  - vs. FinRL-A2C: CR 19.46%, SR 0.831
  - vs. Equal-Weighted ETF: CR 9.34%, SR 0.492
- Portfolio 2 (AMZN, GM, LLY): **CR 32.92%, SR 1.371**, MDD 21.50%
  - vs. all baselines: significantly higher CR and SR

### Ablation Studies
- **Within-episode risk control (CVaR):**
  - GOOG: w/ CVaR CR 25.08% vs w/o -1.46%
  - NIO: w/ CVaR CR 17.46% vs w/o -52.89%
  - Portfolio 1: w/ CVaR CR 113.84% vs w/o 14.70%
- **Over-episode belief updates (CVRF):**
  - GOOG: w/ belief CR 25.08% vs w/o -11.94%
  - NIO: w/ belief CR 17.46% vs w/o 8.20%
  - Portfolio 1: w/ belief CR 113.84% vs w/o 28.43%
  - Over-episode belief update is **more critical** than within-episode risk control.
- Achieves these results after **only 4 training episodes** -- far fewer than traditional RL.

### High-Volatility Market Performance
- TSLA during high VIX (>20) period: FinCon CR 22.46%, SR 0.695 (only agent with positive returns; all others deeply negative)
- Portfolio 1 high-vol: FinCon CR -8.43% (best among all; others -15% to -29%)

### Statistical Significance
- Highest and second-highest CRs and SRs tested and found statistically significant using the **Wilcoxon signed-rank test**.

---

## 7. Findings About Trading Horizon / Decision Frequency

- **Decision frequency is daily.** All trading decisions (buy/sell/hold) and portfolio rebalancing are made on a **daily basis**.
- CVaR is monitored daily; risk alerts are triggered day-by-day.
- Different data sources have different timeliness / memory decay rates:
  - Annual filings (10-K): long-term persistence
  - Quarterly filings (10-Q) and ECC: medium-term relevance
  - Daily financial news: most immediate information
- Each analyst agent has **distinct procedural memory decay rates** reflecting these time horizons, which is crucial for aligning multi-type data influencing specific time points.
- The paper does not explicitly explore different trading frequencies (e.g., weekly, monthly). All experiments use daily decision-making.
- Future work mentions scaling to larger portfolios but does not mention different decision frequencies.

---

## 8. What Baselines Are Compared?

### Single Stock Trading Baselines

**Market Baseline:**
- Buy-and-Hold (B&H)

**LLM-based agents:**
1. **Generative Agent (GA)** [Park et al., 2023] -- general-purpose generative agent adapted for trading; memory module with recency/relevance/importance but no layered memory for financial data timeliness.
2. **FinGPT** [Yang et al., 2023] -- open-source LLM framework for financial decision-making; primarily relies on sentiment analysis.
3. **FinMem** [Yu et al., 2023] -- specialized profiling module, self-adaptive risk settings, layered memory. Single-agent framework.
4. **FinAgent** [Zhang et al., 2024] -- builds on FinMem with tool-using capabilities for multi-modal data; single-agent.

**DRL-based agents (from FinRL framework):**
1. **A2C** (Advantage Actor-Critic)
2. **PPO** (Proximal Policy Optimization)
3. **DQN** (Deep Q-Network)

### Portfolio Management Baselines
1. **Markowitz Mean-Variance (MV)** portfolio selection
2. **FinRL-A2C** (RL-based portfolio optimization)
3. **Equal-Weighted ETF** (buy-and-hold equal allocation across all assets)

Note: No LLM-based portfolio management baselines exist because FinCon is the first LLM agent system to address portfolio management.

---

## 9. Evaluation Metrics Used

### Primary Metrics (prioritized):
1. **Cumulative Return (CR%):** Sum of daily log returns over the test period. Higher is better.
2. **Sharpe Ratio (SR):** Risk-adjusted return = (Rp - Rf) / sigma_p. Higher is better. Above 1 is favorable; above 2 is excellent.

### Secondary Metric:
3. **Max Drawdown (MDD%):** Largest peak-to-trough decline. Lower is better. Focuses on potential for significant losses.

### Risk Estimation Metrics (internal to the system):
- **Profit and Loss (PnL):** Daily net trading outcome.
- **Value at Risk (VaR):** Potential loss at a given confidence level.
- **Conditional Value at Risk (CVaR):** Expected loss beyond VaR (average of worst 1% of daily PnLs). Used for within-episode risk control.

### Portfolio-Specific:
- **Portfolio Value:** Total worth of all investments at each time point (initial amount $1,000,000).

### Statistical Testing:
- **Wilcoxon signed-rank test** for statistical significance of performance differences (non-parametric, suitable for non-Gaussian data).
- Performance based on **median CR and SR from 5 repeated epochs**.

---

## 10. Code/Data Availability

- **Code repository:** https://github.com/The-FinAI/FinCon (promised in paper: "We will release the code and demo")
- **Data sources used:**
  - Yahoo Finance (via yfinance) for stock prices
  - Refinitiv Real-Time News for daily news
  - SEC EDGAR (via SEC API) for Form 10-K and 10-Q filings
  - Capital IQ for some data
  - Alpaca News API
  - Zacks Equity Research for analyst reports/ranks
  - ECC audio recordings
- **Related frameworks referenced:** FinRL-Meta (https://github.com/AI4Finance-Foundation/FinRL-Meta)

---

## Key Takeaways for Our Research

1. **Hierarchical multi-agent > flat peer-to-peer:** FinCon's manager-analyst structure with selective communication outperforms systems relying on extensive agent debates, both in performance and cost.
2. **Verbal reinforcement is powerful:** Conceptualized investment beliefs updated via textual gradient descent require only 4 training episodes, far fewer than DRL.
3. **Dual-level risk control is essential:** Both within-episode (CVaR monitoring) and over-episode (belief updates via CVRF) risk control significantly improve performance. Over-episode belief updates are more impactful.
4. **Daily decision frequency:** All experiments are daily. No exploration of other frequencies.
5. **Multi-modal data matters:** Integrating news, filings, ECC audio, and price data through specialized agents improves decision quality.
6. **Portfolio management is harder:** Multi-asset decisions increase hallucination risk due to longer/more complex inputs. FinCon mitigates this but occasionally generates incorrect information.
7. **LLM agents handle limited-data scenarios better than DRL:** For recently IPO'd stocks (COIN), DRL cannot converge but LLM agents perform well.
8. **Memory decay rates aligned to data timeliness** is a key design insight.
9. **GPT-4-Turbo only:** No experiments with other LLMs or cost analysis of different backbone models.
10. **Short-selling allowed:** The "sell" signal permits negative positions, which enables profiting in bearish markets.
