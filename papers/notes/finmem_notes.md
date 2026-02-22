# FinMem: A Performance-Enhanced LLM Trading Agent with Layered Memory and Character Design

**Paper:** arXiv:2311.13743v2 [q-fin.CP], 3 Dec 2023
**Authors:** Yangyang Yu*, Haohang Li*, Zhi Chen*, Yuechen Jiang*, Yang Li*, Denghui Zhang, Rong Liu, Jordan W. Suchow, Khaldoun Khashanah
**Affiliation:** Stevens Institute of Technology, Hoboken, NJ

---

## 1. What is FinMem? (Architecture and Framework Description)

FinMem is an LLM-based autonomous trading agent framework specifically designed for financial decision-making. It addresses the limitations of both Deep Reinforcement Learning (DRL) agents (lack of interpretability, difficulty integrating text with numerical data) and existing LLM-based trading agents (inability to understand varying timeliness of financial data, lack of memory components).

The framework is inspired by Park et al.'s Generative Agents framework but substantially extends it for financial applications. FinMem encompasses **three core modules**:

1. **Profiling Module** -- Customizes the agent's character with:
   - A foundational professional knowledge base (trading sector introduction + historical financial performance overview for the target ticker)
   - Three distinct investment risk inclinations: **risk-seeking**, **risk-averse**, and **self-adaptive**
   - The self-adaptive mode dynamically switches between risk-seeking (when cumulative return is positive) and risk-averse (when cumulative return is negative over a brief period such as 3 days)

2. **Memory Module** -- Layered message processing system with:
   - **Working Memory**: Central workspace for summarization, observation, and reflection operations
   - **Layered Long-term Memory**: Shallow, intermediate, and deep processing layers with varied decay rates

3. **Decision-making Module** -- Converts insights from memories into investment decisions (Buy / Sell / Hold) by integrating top-ranked memory events with current market conditions. Uses Guardrails AI for text validation of trading actions.

**Key innovation:** FinMem's memory module aligns with human cognitive structures, offering interpretability and real-time tuning. Its adjustable cognitive span retains critical information beyond human perceptual limits.

---

## 2. Memory Module Design -- How It Handles Different Time Horizons

### 2.1 Working Memory

Working memory serves as a dynamic "workspace" with three key operations:

- **Summarization**: Condenses external market data (news, reports) into compact informative paragraphs extracting key investment insights and sentiments. Directs insights to appropriate long-term memory layer based on time sensitivity.
- **Observation**:
  - *Training phase*: Accesses actual daily adjusted closing price differences (ground truth labels for Buy/Sell)
  - *Testing phase*: Relies on cumulative return over the last M trading days to infer future market trends (no access to future prices)
- **Reflection**: Two types:
  - *Immediate reflection*: Triggered daily; merges market indications with top-K ranked events from each long-term memory layer; outputs trading direction, rationale, and most influential memory event IDs
  - *Extended reflection*: Reevaluates immediate reflection outcomes over an M-day trace period; summarizes market trends and reassesses cumulative return; stored in deep processing layer

**Key hyperparameters**: K (top events per layer) and M (trace period for extended reflection) control working memory capacity and information retrieval ability.

Working memory capacity = 3 x K (one K per memory layer). Unlike human working memory (7 +/- 2 events), FinMem can expand capacity flexibly.

### 2.2 Layered Long-term Memory

Three processing layers with different decay characteristics, inspired by Craik & Lockhart's (1972) levels of processing framework:

| Layer | Data Type | Stability Q (days) | Decay Base alpha | Update Frequency | Importance Probability (high=80) |
|-------|-----------|---------------------|-------------------|------------------|----------------------------------|
| **Shallow** | Daily market news insights | Q=14 (2 weeks) | alpha=0.9 | Daily | p3=0.05 (low probability of high importance) |
| **Intermediate** | Company 10-Q (quarterly) report insights | Q=90 (1 quarter) | alpha=0.967 | Quarterly | p3=0.15 |
| **Deep** | Company 10-K (annual) report insights + Extended reflections | Q=365 (1 year) | alpha=0.988 | Yearly (+ daily extended reflections) | p3=0.80 (high probability of high importance) |

### 2.3 Memory Retrieval Scoring

Each memory event E in layer l gets an information retrieval score:

```
gamma_l^E = S_Recency_l^E + S_Relevancy_l^E + S_Importance_l^E
```

**Recency Score** (Ebbinghaus forgetting curve):
```
S_Recency_l^E = exp(-delta^E / Q_l)
```
where delta^E = t_P - t_E (time difference between inquiry and event)

**Relevancy Score** (cosine similarity):
```
S_Relevancy_l^E = cosine_similarity(m_E, m_P)
```
Using OpenAI's "text-embedding-ada-002" model for embeddings of memory event text (m_E) and LLM prompt query (m_P).

**Importance Score**:
```
S_Importance_l^E = v_l^E * theta_l
```
where v_l^E is drawn from a piecewise uniform distribution {40, 60, 80} with layer-specific probabilities, and theta_l = (alpha_l)^(delta^E) is an exponential decay factor.

**Purging rules**: Events are removed when S_Recency < 0.05 or S_Importance < 5 (pre-scaling).

### 2.4 Access Counter (Memory Transition Mechanism)

An access counter oversees transfer of memory events between layers:
- Events identified as pivotal for investment decisions receive +5 points to their importance score
- When an event meets criteria for upgrading to a deeper layer, its recency score is reset to 1.0
- Facilitated by Guardrails AI which monitors critical memory IDs across layers
- Significant events ascend from shallower to deeper layers for extended retention
- Less pertinent events gradually diminish

---

## 3. What LLMs Are Used?

FinMem was evaluated with multiple backbone LLMs:

| LLM | Cumulative Return (%) | Sharpe Ratio | Notes |
|-----|----------------------|--------------|-------|
| **GPT-4** | **62.62** | 2.2251 | Highest cumulative return |
| **GPT-4-Turbo** | 54.70 | **2.4960** | Best Sharpe Ratio; chosen as primary backbone |
| **GPT-3.5-Turbo** | 16.15 | 2.1589 | Noteworthy performance, followed closely behind |
| **davinci-003** | 1.63 | 0.8515 | Defaulted to "Hold" during volatility; limited capability as earlier-generation model |
| **Llama2-70b-chat** | -52.72 | -2.8532 | Poor performance; shorter context window; needed to simplify prompts |

**Primary configuration**: GPT-4-Turbo with temperature=0.7 was selected as the backbone for main comparative experiments due to best overall risk-adjusted performance.

**Embedding model**: OpenAI "text-embedding-ada-002" for computing relevancy scores.

**Vector database**: FAISS (open-source) for memory warehouse construction and cosine-similarity-based semantic search.

---

## 4. What Datasets/Markets Are Evaluated?

### Data Sources
- **News data**: Alpaca News API (Benzinga backend provider)
- **Quarterly filings**: SEC Form 10-Q reports
- **Annual filings**: SEC Form 10-K reports
- **Stock prices**: Yahoo Finance (yfinance) -- daily OHLCV data

### Time Period
- **Full dataset span**: August 15, 2021 to April 25, 2023
- **Training period** (main experiments): August 17, 2021 to October 5, 2022
- **Testing period** (main experiments): October 6, 2022 to April 10, 2023
- **Ablation study period**: Training March 14, 2022 to June 15, 2022; Testing June 16, 2022 to December 28, 2022
- **Extended test** (short training): Training Aug 17, 2021 to Feb 10, 2022; Testing Feb 11, 2022 to Apr 25, 2023

### Stock Tickers Evaluated (5 stocks across different sectors)
1. **TSLA** (Tesla, Inc.) -- Consumer Cyclical / EV sector; largest news volume; used for ablation studies
2. **NFLX** (Netflix, Inc.)
3. **AMZN** (Amazon.com, Inc.)
4. **MSFT** (Microsoft Corporation)
5. **COIN** (Coinbase Global, Inc.) -- IPO in April 2021; limited history; DRL agents could not converge

Stocks were selected for having the highest volumes of accessible news text data and being spread across various trading sectors.

---

## 5. Key Results Including Cumulative Return and Sharpe Ratio

### Main Comparative Results (Table 2, testing period)

| Ticker | Model | Cumulative Return (%) | Sharpe Ratio | Daily Vol (%) | Annualized Vol (%) | Max Drawdown (%) |
|--------|-------|-----------------------|--------------|---------------|---------------------|------------------|
| **TSLA** | FinMem | **61.78** | **2.68** | 2.95 | 46.86 | **10.80** |
| TSLA | DQN | 33.34 | 0.97 | 4.40 | 69.89 | 52.00 |
| TSLA | GA | 13.46 | 0.60 | 2.88 | 45.68 | 24.32 |
| TSLA | B&H | -18.63 | -0.54 | 4.41 | 69.98 | 55.32 |
| **NFLX** | FinMem | **36.45** | **2.02** | **2.30** | **36.43** | **15.85** |
| NFLX | B&H | 35.51 | 1.41 | 3.20 | 50.74 | 20.93 |
| **AMZN** | FinMem | **4.89** | **0.23** | 2.69 | 42.66 | **22.93** |
| AMZN | B&H | -10.77 | -0.50 | 2.77 | 43.97 | 33.68 |
| **MSFT** | FinMem | **23.26** | **1.44** | 2.05 | 32.56 | **14.99** |
| MSFT | B&H | 14.69 | 0.84 | 2.23 | 35.44 | 15.01 |
| **COIN** | FinMem | **34.98** | **0.72** | 5.65 | 89.75 | **35.75** |
| COIN | B&H | -30.01 | -0.52 | 6.75 | 107.18 | 60.51 |

**Key findings:**
- FinMem outperforms ALL baselines on Cumulative Return and Sharpe Ratio for every stock
- Statistically significant superiority confirmed via Wilcoxon signed-rank test
- TSLA and NFLX achieve Sharpe Ratios > 2.0
- FinMem maintains lowest volatility and max drawdown in most cases
- DRL agents could not be applied to COIN due to insufficient training data (IPO in 2021)

### Ablation: Risk Inclination Impact (TSLA, shorter period)

| Risk Setting | Cumulative Return (%) | Sharpe Ratio | Max Drawdown (%) |
|-------------|----------------------|--------------|------------------|
| **Self-Adaptive** | **54.70** | **2.50** | **12.57** |
| Risk-Seeking | -19.41 | -0.79 | 45.00 |
| Risk-Averse | -12.47 | -1.58 | 15.99 |
| B&H | -66.95 | -2.08 | 67.33 |

### Ablation: Working Memory Capacity (Top-K, TSLA)

| Top-K | Cumulative Return (%) | Sharpe Ratio | Max Drawdown (%) |
|-------|----------------------|--------------|------------------|
| Top 1 | 52.09 | 1.86 | 25.24 |
| Top 3 | 29.44 | 1.12 | 27.10 |
| **Top 5** | 54.70 | 2.50 | **12.57** |
| **Top 10** | **79.44** | **2.75** | 17.14 |
| B&H | -66.95 | -2.08 | 67.33 |

Top 5 is optimal for balanced performance (lowest volatility + drawdown); Top 10 achieves highest raw returns but with more volatility in stable markets.

---

## 6. Cognitive Span / Memory Layering and Short-term vs Long-term Decisions

The "cognitive span" concept is central to FinMem's design and directly relates to how the agent processes information at different time horizons:

### Mapping to Human Cognition
- **Working memory** mirrors the human cognitive system's temporary storage (Miller's 7+/-2 rule), but FinMem can exceed this limit
- **Layered long-term memory** draws from Craik & Lockhart's (1972) "levels of processing" framework, where deeper processing leads to longer retention

### Time Horizon Mapping
- **Short-term decisions** are informed primarily by the **shallow layer** (daily news, Q_shallow=14 days, fast decay alpha=0.9). These capture immediate market reactions, sentiment shifts, and breaking news.
- **Medium-term decisions** incorporate the **intermediate layer** (10-Q quarterly reports, Q_intermediate=90 days, moderate decay alpha=0.967). These capture quarterly financial performance and trends.
- **Long-term decisions** leverage the **deep layer** (10-K annual reports + extended reflections, Q_deep=365 days, slow decay alpha=0.988). These capture fundamental company performance and accumulated trading wisdom.

### Key Mechanism: Adjustable Cognitive Span
- The Top-K hyperparameter controls how many events from each layer are retrieved per trading decision
- This allows FinMem to process more information than a human trader (who is limited to 5-9 events)
- Higher K values (5-10) generally improve performance but may introduce noise in stable markets
- The access counter mechanism promotes important shallow events to deeper layers, ensuring critical short-term information persists for long-term influence

### Integration for Decision-Making
During immediate reflection, FinMem synthesizes top-K events from ALL three layers simultaneously, allowing:
- Short-term signals (news) to be weighed against medium-term (quarterly) and long-term (annual) fundamentals
- Extended reflections (stored in deep layer) provide meta-cognitive assessment of recent trading patterns
- The self-adaptive risk character modulates how aggressively these signals translate to actions

---

## 7. What Baselines Are Compared?

### Passive Strategy
1. **Buy-and-Hold (B&H)**: Purchase stock and hold regardless of market fluctuations. Standard baseline for stock trading strategies.

### DRL Trading Agents (trained on ~10 years of data: Jan 1, 2012 to Oct 5, 2022)
2. **PPO (Proximal Policy Optimization)**: Balances exploration and exploitation by bounding policy updates. Implemented via Stable Baselines 3.
3. **DQN (Deep Q-Network)**: Adaptation of Q-learning using deep neural networks for scalable Q-value estimation across complex state spaces.
4. **A2C (Advantage Actor-Critic)**: Simultaneously updates policy (actor) and value (critic) functions. Balances exploration and exploitation.

Note: DRL agents take only numeric features as inputs. They could NOT be applied to COIN due to insufficient historical data.

### LLM Trading Agents
5. **General-purpose Generative Agents (GA)**: Adapted from Park et al.'s framework. Has memory module with recency/relevance/importance metrics but NO layered memory and NO self-adaptive risk preference. Prompt template modified for financial tasks.
6. **FinGPT**: Open-source LLM framework by Yang et al. specialized for financial decision-making. Claims superiority over B&H.

---

## 8. Evaluation Metrics Used

Five widely-used financial metrics (all reported as averages from 5 repeated trials):

1. **Cumulative Return**: Sum of daily logarithmic returns over the testing period. Higher = better.
   ```
   CR = sum_{t=1}^{n} [ln(p_{t+1}/p_t) * action_t]
   ```

2. **Sharpe Ratio**: Risk-adjusted return. Average excess return over risk-free rate divided by portfolio volatility. >1 favorable, >2 excellent.
   ```
   SR = (R_p - R_f) / sigma_p
   ```

3. **Annualized Volatility**: Daily volatility (std dev of daily log returns) multiplied by sqrt(252). Lower = less risky.
   ```
   AV = Daily_Volatility * sqrt(252)
   ```

4. **Daily Volatility**: Standard deviation of daily logarithmic returns.

5. **Max Drawdown**: Largest peak-to-trough decline in portfolio value. Smaller = more robust.
   ```
   MDD = max((P_peak - P_trough) / P_peak)
   ```

**Statistical significance**: Wilcoxon signed-rank test (non-parametric, appropriate for non-Gaussian distributed financial data) used to compare FinMem against the second-best strategy.

---

## 9. Code/Data Availability

- **Source code**: Available via the "FINMEM LLM Trading" link referenced on page 1 (footnote 2)
- **Data sources**: All publicly available
  - Yahoo Finance (yfinance) for stock price data
  - Alpaca News API (Benzinga backend) for news data
  - SEC filings for 10-K and 10-Q reports
- **Vector database**: FAISS (open-source) for memory warehouse
- **Validation tool**: Guardrails AI (open-source) for LLM output validation
- **DRL implementation**: Stable Baselines 3

---

## 10. Limitations and Future Work

### Limitations (inferred from text)
- **LLM cost and latency**: The framework relies on commercial LLMs (GPT-4/GPT-4-Turbo), which have API costs. Ablation studies used shorter training periods "for budgetary efficiency."
- **Single-stock trading only**: FinMem is practiced and examined on single-stock trading with discrete actions (Buy/Sell/Hold for a single share), not portfolio-level optimization.
- **Limited data scope**: Used a "limited range and quality of financial news and reports" with general-purpose LLMs rather than finance-specific ones.
- **Context window constraints**: Llama2-70b-chat had to simplify prompts and shorten retrieved memory insights due to shorter context windows, resulting in information loss.
- **Top-K sensitivity**: The optimal K value varies depending on the volume and quality of incoming information; no automatic tuning mechanism is provided.
- **Open-source LLM performance gap**: Llama2-70b-chat performed very poorly (-52.72% cumulative return), indicating the framework is heavily dependent on strong proprietary LLMs.
- **Short evaluation window**: Testing period is roughly 6 months; longer-term robustness is not established.
- **No transaction costs modeled**: The paper does not discuss transaction costs, slippage, or market impact.
- **Optimal risk inclination varies by stock**: Self-adaptive was best for most stocks but risk-seeking was best for MSFT. No automatic way to determine the optimal setting a priori.

### Future Work (stated by authors)
1. **Multi-agent trading system**: Create a system rooted in the FinMem platform for investment portfolio optimization, featuring agents with diverse professional backgrounds in their profiling modules.
2. **Cross-product trading**: Enable concurrent operations and trading across a variety of financial products with dynamic reallocation of investment proportions.
3. **Finance-specific LLMs**: Integrate LLMs fine-tuned specifically for financial applications (the authors anticipate this would elevate performance further).
4. **Larger/higher-quality datasets**: Access to more comprehensive and higher-quality financial data.
5. **Domain generalization**: The framework could extend to domains such as IT consulting and business reporting where actions are driven by time-sensitive information.
6. **Peer-to-peer agent communication**: Multi-agent system leveraging peer-to-peer communication and systematic performance analysis for portfolio assembly.

---

## Key Takeaways for Implementation

1. **Layered memory is the core innovation**: Mapping financial data types to memory layers by timeliness (news -> shallow, 10-Q -> intermediate, 10-K -> deep) with different decay rates is the central contribution.
2. **Self-adaptive risk works best in volatile markets**: The dynamic risk switching mechanism is effective for most stocks, particularly those with high volatility (TSLA, COIN).
3. **GPT-4-Turbo is the recommended backbone**: Best risk-adjusted returns among tested LLMs.
4. **Top-5 memory events per layer is a good default**: Provides best balance of return, volatility, and drawdown.
5. **Short training is sufficient**: 6-12 months of daily data (including at least one 10-Q/10-K publication cycle) is enough for robust results.
6. **Access counter promotes important memories**: Critical events migrate from shallow to deep layers, ensuring persistent influence on decisions.
7. **FAISS + OpenAI embeddings for retrieval**: Standard vector similarity search infrastructure used for memory retrieval.
