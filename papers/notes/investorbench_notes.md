# InvestorBench: A Benchmark for Financial Decision-Making Tasks with LLM-based Agent

**Paper:** arXiv:2412.18174v1 [cs.CE] 24 Dec 2024
**Authors:** Haohang Li, Yupeng Cao, Yangyang Yu, Shashidhar Reddy Javaji, Zhiyang Deng, Yueru He, Yuechen Jiang, Zining Zhu, Koduvayur Subbalakshmi, Guojun Xiong, Jimin Huang, Lingfei Qian, Xueqing Peng, Qianqian Xie, Jordan W. Suchow
**Affiliations:** Stevens Institute of Technology, Columbia University, Harvard University, The Fin AI

---

## 1. What is InvestorBench?

InvestorBench is the **first benchmark specifically designed for evaluating LLM-based agents in diverse financial decision-making contexts**. It addresses two key gaps in the field:

1. The lack of a comprehensive LLM agent framework adaptable to a variety of financial tasks.
2. The absence of standardized benchmarks and consistent datasets for assessing agent performance.

The benchmark is built upon and extends the **FinMem** framework (Yu et al., 2024a), which originally focused on single-stock investment decisions. InvestorBench generalizes this to a broader range of financial tasks.

### Core Architecture

The LLM-based agent in InvestorBench is structured as a **large language model-modulo framework** with the following interconnected modules:

- **Brain/Backbone (LLM):** The core reasoning engine -- understands, processes, and generates natural language; supports complex decision-making, predictive analytics, and reflection on past decisions.
- **Perception:** Converts raw market data (numerical, textual, visual) into structured LLM-compatible format.
- **Profile:** Dual-function module that (a) defines the agent's role as an experienced investor with self-adaptive risk preference (dynamically adjusting based on historical market momentum), and (b) provides background on the decision-making task and target assets.
- **Memory:** Processes and retains market data and historical insights via two components:
  - **Working Memory:** Handles observation, summarization, and reflection (immediate and extended).
  - **Layered Long-Term Memory:** Structures financial insights across multiple layers (shallow/intermediate/deep) with distinct decay rates inspired by human cognitive systems. Uses vector databases with information prioritized and purged based on layer-specific decay rates.
- **Action:** Executes trading decisions -- outputs {"Buy", "Sell", "Hold"} for the traded asset.

### POMDP Formulation

Financial decision-making is modeled as an **infinite horizon Partially Observable Markov Decision Process (POMDP)** with:
- State space: observable + unobservable market components
- Action space: {Buy, Sell, Hold}
- Reward function: daily Profit & Loss (PnL)
- Observation process: multi-dimensional market data
- Reflection process: agent's self-reflection updated daily
- Language-conditioned policy for decision-making

### Four Main Components

1. **Data Sources and Market Environments:** Open-source data + third-party APIs (Yahoo Finance, SEC EDGAR)
2. **LLM Agent:** Advanced agent with Brain, Perception, Profile, Memory, Action modules + external tools
3. **Financial Decision-Making Tasks:** Three distinct tasks differentiated by asset type
4. **Evaluation Metrics:** Standard quantitative finance metrics

---

## 2. Financial Tasks Evaluated

InvestorBench evaluates **three single-asset trading tasks**, each with daily trading frequency:

| Task | Asset Type | Trading Horizon | Characteristics |
|------|-----------|----------------|-----------------|
| **Stock Trading** | Individual equities (7 stocks) | Daily | Requires analyzing company-specific and industry-wide data (market metrics, sector trends, performance reports, news) |
| **Cryptocurrency Trading** | BTC, ETH | Daily | Highly sensitive to crypto-specific news and sentiment; dynamic nature |
| **ETF Trading** | Exchange-traded funds | Daily | Follows passive investment strategies; emphasizes long-term growth and cost efficiency; requires interpreting signals across diverse sectors |

### Time Periods for Each Task

| Task | Warm-up Period | Test Period |
|------|---------------|-------------|
| Stock Trading | 2020-07-01 to 2020-09-30 | 2020-10-01 to 2021-05-06 |
| Cryptocurrency Trading | 2023-02-11 to 2023-04-04 | 2023-04-05 to 2023-11-05 |
| ETF Trading | 2019-07-29 to 2019-12-30 | 2020-01-02 to 2020-09-21 |

---

## 3. LLMs Tested

InvestorBench evaluates **13 LLMs** across three categories:

### Proprietary Models
| Model | Size | Form | Version |
|-------|------|------|---------|
| GPT-4 | N/A | API | 0613 |
| GPT-4o | N/A | API | 0806 |
| GPT-o1-preview | N/A | API | 0912 |

### Open-Source Models
| Model | Size | Form | Version |
|-------|------|------|---------|
| DeepSeek-v2 | 15B | Open | Lite |
| DeepSeek-llm | 67B | Open | Chat |
| Yi-1.5-9b | 9B | Open | Chat |
| Yi-1.5-34b | 34B | Open | Chat |
| Qwen2.5-7b | 7B | Open | Instruct |
| Qwen2.5-32b | 32B | Open | Instruct |
| Qwen2.5-72b | 72B | Open | Instruct |
| Llama3.1-8b | 8B | Open | Instruct |
| Llama3.1-70b | 70B | Open | Instruct |

### Financial Domain-Specific Models
| Model | Size | Form | Version |
|-------|------|------|---------|
| Palmyra-Fin | 70B | Open | 32K |

### Size Categories Used in Analysis
- **Small-size:** <= 10B parameters (Yi-1.5-9B, Llama-3.1-8B, Qwen-2.5-7B)
- **Medium-size:** (10B, 65B] parameters (DeepSeek-V2-Lite 15.7B, Yi-1.5-34B, Qwen2.5-32B)
- **Large-size:** > 65B parameters (DeepSeek-67B, Qwen2.5-72B, Llama-3.1-70B)

---

## 4. Datasets Used

### Stock Market Environment
- **OHLCV data:** Daily stock open/high/low/close/volume from Yahoo Finance
- **Company reports:** Summarized insights from quarterly (10-Q) and annual (10-K) reports from SEC EDGAR
- **News articles:** Daily news for 7 stocks between 2020-07-01 and 2021-05-06
  - MSFT, JNJ, UVV, HON: randomly selected from open-access dataset by Zhou et al. (2021) (500+ records each)
  - TSLA, AAPL, NIO: from Refinitiv Real-Time News (Reuters)
- **Sentiment labels:** Generated by gpt-3.5-turbo-0125 (positive, negative, neutral)
- **Stocks covered:** MSFT, JNJ, UVV, HON, TSLA, AAPL, NIO

### Cryptocurrency Market Environment
- **OHLCV data:** Daily data from CoinMarketCap
- **News data:** Multisource from cryptonews, cryptopotato, cointelegraph (Vanhoucke, 2023)
- **Time span:** 2023-02-13 to 2023-11-05, daily frequency
- **Sentiment labels:** Same approach as stocks
- **Assets covered:** Bitcoin (BTC), Ethereum (ETH)

### ETF Market Environment
- **Source:** News-Informed Financial Trend Yield (NIFTY) dataset (Saqur et al., 2024)
- **Data:** Processed and curated daily news headlines from 2019-07-29 to 2020-09-21
- **Sentiment labels:** Generated for each news headline

---

## 5. Key Results

### 5.1 Stock Trading (Table 2 -- 7 stocks, 13 models)

**Buy & Hold baseline average across 7 stocks:** CR=34.10%, SR=0.732

**Key findings:**

- **Proprietary models dominate stock trading.** GPT-4o achieved the best average risk-adjusted performance (avg CR=39.03%, SR=1.041), followed by GPT-4 (avg CR=43.70%, SR=0.972). These significantly outperformed open-source models.
- **Financial domain models underperformed.** Palmyra-Fin-70B showed negative average CR (-0.45%) and near-zero SR (0.031), suggesting fine-tuning for financial report analysis does not translate to trading decision-making.
- **Larger open-source models outperform smaller ones.** Among open-source models, Qwen2.5-72B-Instruct achieved the best average CR (46.15%) and SR (1.276), while Llama-3.1-70B was also strong (avg CR=38.95%, SR=0.864). Models > 67B showed superior and more consistent results.
- **Small models are highly variable.** Some small models (Yi-1.5-9B, Llama-3.1-8B) performed surprisingly well on some individual stocks but poorly on others.

### 5.2 Cryptocurrency Trading (Table 3 -- BTC and ETH)

**Buy & Hold baseline:** BTC CR=21.82%, SR=0.683; ETH CR=4.53%, SR=0.146

**Key findings:**

- **Proprietary models excel in crypto.** GPT-o1-preview achieved BTC CR=34.06%, SR=1.114, outperforming Buy & Hold. Average proprietary: BTC CR=23.60%, SR=0.825.
- **Mid-sized and small open-source models generally underperform** the market baseline on CR and SR.
- **Large open-source models needed** to effectively capture trading signals in crypto markets (highly sensitive to news and sentiment).
- **Qwen2.5-32B was notable** for very low volatility (AV=15.61% on BTC) and low MDD, though lower CR.

### 5.3 ETF Trading (Table 4)

**Buy & Hold baseline:** CR=2.07%, SR=0.06

**Key findings:**

- **Most models beat the Buy & Hold baseline**, which was very weak during this period (Jul 2019 - Sep 2020, including COVID crash).
- **Palmyra-Fin-70B was the top performer** (CR=24.76%, SR=1.152) -- a notable contrast to its poor stock trading results.
- **GPT-o1-preview** performed best among proprietary models (CR=21.22%, SR=0.849).
- **Proprietary models average:** CR=12.11%, SR=0.445; significantly outperform open-source average (CR=5.83%, SR=0.282).
- **ETF trading requires proprietary models** enriched with extensive pre-trained knowledge for robust reasoning about diverse sectors and long-term strategy.
- **Qwen2.5-32B-Instruct** was the best open-source model (CR=19.62%, SR=0.955).

### 5.4 Cross-Task Discussion

- Performance varies significantly across stock, cryptocurrency, and ETF trading, reflecting inherent market complexity and the importance of model selection.
- Proprietary LLMs generally exhibit better performance in stock trading due to strong training on financial datasets.
- Open-source models struggle especially in more volatile environments (cryptocurrency).
- Agents with **advanced memory systems and dynamic risk assessment** cope better with complex market situations.
- The effectiveness of LLM-based agents depends heavily on their ability to adapt to market fluctuations.

---

## 6. Baselines Compared

- **Buy & Hold strategy:** The primary baseline for all single-asset trading tasks. A passive investment approach where an investor purchases and holds regardless of market fluctuations.
- **Prior frameworks referenced but not directly benchmarked:** FinMem, FinAgent, CryptoTrade, FinRobot, FinCon (these are discussed in related work but the benchmark's focus is on comparing LLM backbones within its own agent framework).

---

## 7. Evaluation Metrics

Four standard quantitative finance metrics are used:

| Metric | Symbol | Direction | Description |
|--------|--------|-----------|-------------|
| **Cumulative Return** | CR (%) | Higher is better | Total value change by summing daily logarithmic returns: CR = sum(ln(p_{t+1}/p_t) * action_t) |
| **Sharpe Ratio** | SR | Higher is better | Risk-adjusted return: SR = (R_p - R_f) / sigma_p |
| **Annualized Volatility** | AV (%) | Lower is better | Return fluctuations scaled to annual: AV = DV * sqrt(252) |
| **Maximum Drawdown** | MDD (%) | Lower is better | Largest portfolio value drop from peak to trough |

**Primary metrics:** CR and SR are considered more essential than AV and MDD due to their focus on long-term gains and risk-adjusted returns.

### Reporting Methodology
- Performance metrics reported for the test trajectory with **median CR, SR, AV, and MDD from 5 repeated epochs**.
- If median metrics don't belong to the same epoch, performance is based on the trajectory with the **median SR**.
- Temperature parameter set to **0.6** for all LLM-based agents (balancing consistency and creativity).

---

## 8. Long-term vs Short-term Trading Performance Findings

### Memory Architecture Encodes Time Horizons

The layered long-term memory system explicitly models different time horizons:
- **Shallow processing layer:** Q_shallow = 14 days (2 weeks) -- handles daily news and short-term insights
- **Intermediate processing layer:** Q_intermediate = 90 days (quarter) -- handles quarterly reports and medium-term news
- **Deep processing layer:** Q_deep = 365 days (year) -- handles annual reports and long-term strategic insights

Each layer has distinct decay rates (alpha_shallow=0.9, alpha_intermediate=0.967, alpha_deep=0.988), with deeper layers retaining information longer.

### Key Observations on Time Horizons

1. **ETF trading (longer-term, passive strategy) is hardest for open-source models.** ETF investment requires strategic, long-term decisions grounded in deep comprehension and reflection, demanding proprietary models with extensive pre-trained knowledge. This suggests LLMs struggle more with long-term strategic reasoning than with shorter-term reactive trading.

2. **Cryptocurrency trading (shorter-term, sentiment-driven) requires large models.** The high sensitivity to news and financial sentiment means only large open-source or proprietary models can effectively capture trading signals.

3. **Stock trading in mixed/volatile markets favors proprietary models.** In monotone bullish markets, the advantage of proprietary models over open-source is less evident. But in complex, mixed market conditions (TSLA, NIO with both upward and downward trends), proprietary models show significantly stronger decision-making.

4. **Proprietary models better handle noisy/delayed investment signals.** They can leverage historical momentum, current holdings, and self-reflection outcomes more effectively.

5. **All tasks are daily trading frequency.** The benchmark does not explicitly compare different trading frequencies (e.g., intraday vs weekly vs monthly). The variation is in the nature of the assets and the strategic horizon required.

---

## 9. Code/Data Availability

- **Open-source benchmark:** InvestorBench is described as an open-source benchmark.
- **License:** Data shared under the **MIT license**.
- **Two modes of engagement offered:**
  1. Users can integrate their fine-tuned LLMs into InvestorBench's agent framework to benchmark against existing results.
  2. Users can incorporate InvestorBench's environment and evaluation metrics into their own agent designs for comparative analysis.
- **LLM deployment:** Uses **vLLM** for deploying open-source models.
  - Small-scale (< 10B): 2x RTX A6000 GPUs (48GB each)
  - Mid-scale (10B-65B): 4x RTX A6000 GPUs
  - Large-scale (> 65B): 8x A100 GPUs (80GB each)
- **No explicit GitHub URL provided in the paper**, but the benchmark is described as open-source and designed for community use.

---

## 10. Limitations and Future Work

### Limitations
1. **Single-asset focus only:** InvestorBench currently focuses on single-asset financial decision-making tasks, without addressing multi-asset tasks such as **portfolio management**.
2. **Copyright restrictions on financial data:** May compromise dataset quality, potentially limiting assessment of model performance.
3. **Limited modalities:** Currently uses text-based data (news, reports, OHLCV) but does not include audio or visual chart data.
4. **No intraday or multi-frequency trading:** All tasks operate at daily frequency only.
5. **Sentiment labels generated by GPT-3.5:** Introduces potential bias/errors from the sentiment labeling model.

### Future Work
- **Additional information modalities:** Audio (e.g., earnings call recordings) and graphs (e.g., K-lines, trade charts) to explore whether these enhance decision-making quality.
- The foundational agent framework is designed to **seamlessly accommodate** these additional modalities.
- **Extending to portfolio management** tasks (multi-asset).

---

## Key Takeaways for Our Research

1. **Proprietary models (GPT-4, GPT-4o, GPT-o1) consistently outperform open-source models** in financial trading decision-making, especially in volatile and complex market conditions.
2. **Model size matters:** Larger open-source models (67B+) significantly outperform smaller ones, with less variance.
3. **Domain-specific fine-tuning does not guarantee trading success.** Palmyra-Fin-70B, fine-tuned for financial report analysis, performed poorly on stock trading but surprisingly well on ETF trading.
4. **The layered memory architecture** (shallow/intermediate/deep with different decay rates) is a key design feature for handling multi-horizon financial information.
5. **Daily trading is the standard frequency** used across all tasks. No exploration of different trading frequencies within the benchmark.
6. **ETF trading (more long-term/strategic) is the most challenging** for LLM agents, requiring the strongest reasoning capabilities.
7. **The benchmark only covers single-asset trading** -- portfolio management is noted as a gap.
8. **Self-adaptive risk preference** (risk-seeking when momentum is positive, risk-averse when negative) is an interesting design choice.
9. **Mixed/volatile markets are the real test** -- most models can handle bullish trends, but complex conditions separate proprietary from open-source models.
