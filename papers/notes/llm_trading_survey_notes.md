# Notes: Large Language Model Agent in Financial Trading: A Survey

**Paper:** Ding, Li, Wang, Chen (2024). "Large Language Model Agent in Financial Trading: A Survey." arXiv:2408.06361v1, received 26 July 2024.

**Authors:** Han Ding (Columbia), Yinheng Li (Columbia), Junhao Wang (Columbia), Hang Chen (NYU).

---

## 1. What Does the Survey Cover?

This is the first survey dedicated to LLM agents for financial trading. It reviews 27 papers (7 of which explicitly use "agent" in their titles) that study using LLMs to make trading decisions or generate trading signals. The survey organizes the literature along three axes:

- **Architecture:** How LLM agents are designed (trader vs. alpha miner, and sub-types).
- **Data:** What types of inputs are fed to the agents (numerical, textual, visual, simulated).
- **Evaluation:** How agents are backtested, what metrics are used, and what performance levels are achieved.

The paper also discusses limitations of the current research and outlines future directions.

---

## 2. Classification of LLM Trading Agent Approaches

The survey introduces a two-level taxonomy:

### 2.1 LLM as a Trader
The LLM directly generates trading decisions (BUY / HOLD / SELL). Four sub-types are identified:

| Sub-type | Description | Key Papers |
|---|---|---|
| **News-Driven** | News and macro updates are fed into the prompt; LLM predicts next-period price movement. The most fundamental and common architecture. | LLMFactor [47], MarketSenseAI [10], Lopez-Lira & Tang [29], Wu [50], Kirtac & Germano [18], Zhang et al. [56] |
| **Reflection-Driven** | Raw inputs (news, reports) are summarized into memories; upon new observations, relevant memories are retrieved and combined to produce reflections (higher-level insights). Reflections then drive trading decisions. Rooted in cognitive science analogies. | FinMem [53], FinAgent [57] |
| **Debate-Driven** | Multiple LLM agents with different roles (mood agent, rhetoric agent, dependency agent, etc.) debate each other's positions to improve sentiment classification and decision robustness. | TradingGPT [27], HAD [51] |
| **RL-Driven** | Reinforcement learning is used to refine the LLM's predictions from backtesting feedback. Correct/incorrect historical predictions serve as reward signals. | SEP [19], Ding et al. [8] |

### 2.2 LLM as an Alpha Miner
The LLM generates alpha factors (quantitative signals) rather than direct trade decisions. These factors feed into downstream quantitative trading systems.

| System | Description |
|---|---|
| **QuantAgent** [48] | Inner-loop/outer-loop architecture. Writer agent generates alpha factor scripts from human ideas; Judge agent refines them. Outer loop tests code in real markets and uses results to improve the judge. |
| **AlphaGPT** [49] | Human-in-the-loop framework for alpha mining on a similar architecture with an experimental environment. |

### 2.3 LLM Selection
- OpenAI models dominate: GPT-3.5 (used in 9 papers) and GPT-4 (8 papers) are by far the most common.
- GPT-3.5 is used even more than GPT-4, reflecting cost-effectiveness and lower latency preferences.
- Long tail of open-source models: BERT, FinBERT, Qwen, Baichuan, OPT, eLang, llama2/70B, Vicuna/7B, FinGPT, BloombergGPT, BLOOM/Z, Gemini (each used in 1-2 papers).

---

## 3. Trading Tasks and Horizons Discussed

### Tasks
- **Single-stock trading:** Direct BUY/HOLD/SELL decisions on individual equities (FinMem, FinAgent).
- **Portfolio management (multi-stock):** Ranking-based long-short strategies across indices like S&P 500 or CSI 300 (FinLlama, Kirtac & Germano, Lopez-Lira & Tang).
- **Alpha factor mining:** Generating quantitative signals for downstream quant systems (QuantAgent, AlphaGPT).
- **Sentiment classification:** Predicting news/social-media sentiment as an intermediate signal.
- **Cryptocurrency trading:** FinAgent extends to ETH on the crypto market.

### Horizons
- Nearly all agents operate on a **daily trading horizon** (next trading period prediction).
- FinAgent computes short-term, mid-term, and long-term signals from price features (e.g., 3-day price changes) that feed into layered reflections.
- **High-frequency trading is explicitly noted as impractical** due to LLM inference latency.
- The median backtesting period across papers is only **1.3 years**, with most evaluations set between 2020 and 2024.

---

## 4. Common Datasets and Benchmarks

### Markets Tested
- **US stock market:** 9 out of 14 real-data papers (dominant market).
- **Chinese stock market:** 5 out of 14 papers.
- **Cryptocurrency:** Only FinAgent (ETH).
- **Other asset classes (bonds, derivatives, commodities):** Notably absent.

### Index / Stock Universes
- **S&P 500** components for multi-stock portfolio agents.
- **CSI 300** components for Chinese market agents.
- High-volume individual stocks for single-stock agents: TSLA, AMZN, MSFT, COIN, NFLX, GOOGL, META, PYPL.

### Data Sources
| Category | Sources |
|---|---|
| Numerical | Stock prices, trading volumes, open/close/high/low, derived features (3-day price change, etc.) |
| Textual -- Fundamental | 10-Q and 10-K filings, analyst reports (e.g., SeekingAlpha) |
| Textual -- Alternative | Bloomberg, WSJ, CNBC, stock research platforms, Twitter, StockTwits, Reddit, StackExchange |
| Visual | Kline charts, volume charts, trading charts (only explored by FinAgent using GPT-4V) |
| Simulated | Synthetic market environments with artificial events (interest rate changes, financial report releases, bulletin board communication among agents) |

### Evaluation Metrics
- **Portfolio Performance:** Cumulative Return, Annualized Return, Sharpe Ratio, Maximum Drawdown.
- **Signal Quality:** F1 score, accuracy (sentiment prediction), win rate, Information Coefficient (IC).
- **System Metrics:** Token generation cost and computational time (only addressed by QuantAgent).

### Baseline Methods
- **Rule-based:** Buy and Hold, Mean Reversion, Short-Term Reversal.
- **ML/DL-based:** Random Forest, LightGBM, LSTM, BERT.
- **RL-based:** PPO, DQN.

---

## 5. Key Findings and Trends

1. **Strong backtesting performance:** LLM agents achieve annualized returns 15%-30% above the strongest baseline during backtesting with real market data.
2. **Long-short strategies outperform:** Ranking-based long-short strategies generally outperform long-only and short-only strategies, but only when the magnitude of the signal is properly utilized (not just sign-based allocation).
3. **Market-cap weighting helps:** Portfolios weighted by market capitalization show slightly higher returns than equally weighted ones, likely because large-cap companies have better quality textual signals due to news coverage bias.
4. **GPT-4 shows strong in-context learning for finance:** General-purpose LLMs like GPT-4 demonstrate great in-context learning capability in financial tasks without fine-tuning.
5. **Fine-tuned models add value:** Domain-specific fine-tuned LLMs (FinGPT, OPT) show further improvement when aligned with financial knowledge.
6. **Multimodal data improves performance:** FinAgent's integration of visual data (Kline charts) via GPT-4V significantly improved trading performance over FinMem (similar architecture, text-only).
7. **Memory and reflection reduce hallucination:** Incorporating layered memory and reflection mechanisms (inspired by cognitive science) mitigates hallucination risk and helps agents develop higher-level understanding.
8. **Debate improves robustness:** Multi-agent debate frameworks enhance sentiment classification accuracy and decision robustness.
9. **Most studies use in-context learning only:** Very few papers fine-tune the LLM; only SEP [19] tunes the LLM during training.
10. **Trading costs are almost universally ignored** in evaluation, which weakens the practical relevance of reported results.

---

## 6. Short-Term vs. Long-Term Trading Performance

The survey does not present a dedicated head-to-head comparison of short-term vs. long-term performance. However, several relevant observations emerge:

- **Most agents are inherently short-term:** They predict next-day or next-period movements based on daily news flow.
- **Short backtesting windows (median 1.3 years)** make it impossible to assess long-term robustness. The authors explicitly flag this as a credibility concern.
- **FinAgent distinguishes time horizons:** It computes short-term, mid-term, and long-term signals from price features, feeding them into layered reflections. This is the most explicit multi-horizon treatment.
- **Backtest period distribution:** 8 papers cover 0-2 years, 2 papers cover 2-5 years, and only 4 papers cover 5+ years.
- **No live trading evidence:** All results are from backtesting only; no paper reports live/paper-trading results.
- **Inference latency rules out HFT:** The authors note that LLM inference latency is a bottleneck that makes these models impractical for high-frequency trading.
- **The authors recommend longer and more diverse backtesting** to improve credibility of results.

---

## 7. Open Challenges and Future Directions

### Architecture Challenges
- **Reliance on closed-source models (GPT-3.5/4):** Raises data privacy concerns and restricts customization.
- **Lack of fine-tuning exploration:** Effectiveness of fine-tuning LLMs specifically for trading remains an open question.
- **Inference latency:** Makes LLM agents impractical for high-frequency trading.
- **Integration with existing trading systems:** Rarely discussed; a practical deployment barrier.

### Data Challenges
- **Social media data is underutilized:** Only SEP [19] incorporates real-time social media data, despite its proven market influence (e.g., GameStop short squeeze).
- **Visual data is nascent:** Only FinAgent experiments with chart data; financial visual understanding by LLMs is still immature.

### Evaluation Challenges
- **Short backtesting periods:** Median of 1.3 years diminishes credibility.
- **Limited market coverage:** Almost exclusively US and Chinese equities. Derivatives, bonds, commodities, and forex are absent.
- **Trading costs ignored:** Few studies account for transaction costs, slippage, or market impact.
- **Arbitrary test period selection:** Start and end dates are chosen rather arbitrarily.

### Behavioral / Interpretability Challenges
- **Lack of ablation studies:** Few studies explore the underlying reasoning processes of LLMs in trading decisions.
- **Agent personality effects:** Agents with different trading styles/personalities perform differently, but this is not well studied.
- **Ethical risks:** LLMs can take unethical actions under pressure (e.g., insider trading, deceptive explanations) as shown in simulated environment studies [41].

### Suggested Future Directions
1. Develop open-source, fine-tuned LLMs for trading to address privacy and customization.
2. Incorporate richer alternative data sources, especially social media.
3. Expand evaluation to more asset classes and longer time periods.
4. Account for trading costs and realistic execution constraints.
5. Use simulated environments for deeper insight into LLM decision-making processes and patterns.
6. Study integration pathways with existing trading infrastructure and human trader workflows.

---

## 8. Notable Papers and Methods Cited

### Core Agent Systems

| Paper | Method | Key Contribution |
|---|---|---|
| **FinMem** [53] (Yu et al., 2023) | Reflection-driven agent | Layered memorization with character design; retrieval based on recency, relevancy, importance |
| **FinAgent** [57] (Zhang et al., 2024) | Multimodal reflection-driven agent | First multimodal trading agent (text + numeric + image via GPT-4V); incorporates MACD, RSI, analyst guidance; tested on crypto (ETH) |
| **TradingGPT** [27] (Li et al., 2023) | Debate-driven agent | Multi-agent system with layered memory and inter-agent debate on actions and reflections |
| **SEP** [19] (Koa et al., 2024) | RL-driven agent | Self-reflective LLM with RL from backtesting feedback; only paper to fine-tune the LLM; uses social media data |
| **QuantAgent** [48] (Wang et al., 2024) | Alpha miner | Inner-loop/outer-loop for alpha factor generation; addresses token cost and computation time |
| **AlphaGPT** [49] (Wang et al., 2023) | Alpha miner | Human-in-the-loop alpha mining framework |
| **LLMFactor** [47] (Wang et al., 2024) | News-driven agent | Uses LLM reasoning to identify factors from historical news-price relationships |
| **MarketSenseAI** [10] (Fatouros et al., 2024) | News-driven agent | Progressive daily news summary, fundamental/macro summary, stock momentum summary; memorization module |
| **HAD** [51] (Xing, 2024) | Debate-driven | Heterogeneous debating framework with specialized agent roles for sentiment analysis |
| **FinLlama** [20] (Konstantinidis et al., 2024) | Sentiment-based | Ranks all S&P 500 stocks; top/bottom 35% long-short strategy |
| **StockAgent** [54] (Zhang et al., 2024) | Simulated environment | LLM agents with varying personalities trade in simulated market with synthetic events |
| **Ding et al.** [8] (2023) | RL-driven (LG + SCRL) | LLM generates embeddings from news headlines; projected into stock feature space; policy trained via PPO |

### Sentiment / Signal Papers
| Paper | Key Detail |
|---|---|
| Lopez-Lira & Tang [29] | Evaluates ChatGPT for stock price movement forecasting; long-short strategy |
| Kirtac & Germano [18] | Sentiment trading with LLMs; fine-tuned models; market-cap weighting |
| Wu [50] | Portfolio based on LLM news scores; sign-based allocation (long-short underperforms long-only) |
| Zhang et al. [56] | Chinese stock market sentiment with LLMs; group-based signal ranking |
| Zheng [60] | API-enhanced ChatGPT for stock prediction |

### Ethics / Safety
| Paper | Key Detail |
|---|---|
| Scheurer et al. [41] | LLMs can strategically deceive users under pressure, including insider trading and deceptive explanations |

---

## Summary Assessment for Our Research

**Relevance to refining LLM trading strategies:**
- The field is nascent (27 papers as of mid-2024) and almost entirely limited to daily-frequency equity trading.
- The strongest results (15-30% annualized excess return) come from reflection-driven and multimodal architectures but rely on short backtests and ignore trading costs.
- Key gaps we could target: (a) longer backtesting horizons, (b) incorporating transaction costs, (c) multi-asset coverage, (d) fine-tuning open-source models, (e) social media data integration, (f) systematic comparison of short-term vs. long-term signal decay.
- The alpha miner paradigm (QuantAgent, AlphaGPT) is especially promising for integration with existing quant pipelines.
- Memory/reflection mechanisms and multi-agent debate are the most architecturally sophisticated approaches and show the best backtesting results.
