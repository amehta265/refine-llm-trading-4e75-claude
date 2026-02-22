# Can Large Language Models Trade? Testing Financial Theories with LLM Agents in Market Simulations

**Paper:** arXiv 2504.10789 (filed as `can_llms_trade_2503_03144.pdf` in this repo)
**Author:** Alejandro Lopez-Lira
**Date:** April 15, 2025
**Venue:** arXiv preprint (q-fin.CP, econ.GN, q-fin.GN, q-fin.TR)

---

## 1. What Is This Paper About?

This paper develops an open-source simulated stock market framework where Large Language Models act as heterogeneous, competing trading agents. The core research question is whether LLMs can execute trading strategies faithfully in a realistic market setting, and what the implications of their behavior are for financial theory and market stability.

The paper addresses three fundamental questions:
- Can LLMs effectively function as trading agents that follow assigned strategies?
- Do the resulting market dynamics exhibit realistic financial phenomena?
- What are the systemic risk implications when many agents share the same underlying LLM architecture?

The key distinction this paper draws is between LLMs as trading agents vs. traditional algorithms or human traders: LLMs follow instructions faithfully but may lack intrinsic profit optimization. They maintain strategic direction even when sustaining financial losses, which differentiates them from both rule-based algorithms and human traders.

---

## 2. How Are LLM Agents Set Up as Trading Agents?

### Two-Tier Prompting Framework

**System Prompt:** Defines the agent's trading philosophy and strategy identity. For example, a Value Investor system prompt establishes that the agent "believes in mean reversion" and tries to "buy undervalued assets and sell overvalued ones."

**User Prompt:** Delivers real-time market state information each round, including position details, order book state, historical prices, and trading constraints.

### Structured Decision Output

Agents provide standardized JSON outputs (validated via Pydantic schema) containing:
- `valuation_reasoning`: Explanation of fundamental value analysis
- `valuation`: Estimated fundamental value of the asset
- `price_target`: Predicted next-round price
- `orders`: Buy/Sell directives with quantity, type (market or limit), and price limits
- `reasoning`: Justification for the trading decision

### Information Sets Provided to Agents

Each agent receives customizable information including:
- **Market State:** Last price, trading volume, price-to-fundamental ratio
- **Order Book:** Best bid/ask, visible depth at multiple price levels
- **Position Data:** Available shares, cash reserves, committed resources
- **Historical Context:** Prior 5 rounds' price and volume history
- **Fundamentals:** Dividend structure, probabilities, redemption information (if applicable)

Information can be customized per agent, enabling experimental designs with asymmetric information.

---

## 3. What Financial Theories Are Tested?

- **Price Discovery:** Whether markets converge toward fundamental value under various starting conditions (above/below fair value)
- **Bubbles and Underreaction:** Whether agent compositions generate speculative bubbles or systematic undervaluation
- **Divergent Beliefs:** How heterogeneous valuation models (optimistic vs. pessimistic agents) affect equilibrium prices and trading volume
- **Market Microstructure:** How different agent types affect liquidity provision, bid-ask spreads, and order book dynamics
- **Mean Reversion:** Whether value-based strategies effectively exploit price deviations from fundamentals
- **Momentum Effects:** Whether trend-following agents can profitably ride price trends
- **Systemic Risk / Correlated Behavior:** Whether agents sharing the same LLM architecture inadvertently create destabilizing correlated trading patterns

---

## 4. What LLMs Are Used?

**Primary Model:** GPT-4o (used consistently across all experimental scenarios).

This single-model choice was deliberate: it enables clean analysis of strategy adherence and market dynamics without confounding effects from heterogeneous model capabilities. The framework is designed to be model-agnostic and could accommodate other LLMs.

---

## 5. What Simulation Environment Is Used?

### Market Structure

**Continuous Double-Auction with Discrete Rounds:** Orders are processed within randomized agent submission sequences to avoid systematic priority advantages.

### Three-Phase Matching Algorithm

1. Non-crossing limit orders are added to a persistent order book
2. Market orders are processed via market-to-market netting at the current price
3. Remaining orders are matched against standing limit orders using price-time priority

### Key Market Features

- Persistent order book with bid-ask spreads
- Support for both market orders and limit orders
- Partial order fills tracked across rounds
- Position management with pre-trade validation
- Cash and share commitment verification
- No short selling allowed
- No borrowing allowed
- Zero transaction costs (in baseline)
- Dividends paid each round

### Fundamental Value Calibration

- **Infinite horizon:** FV = Expected Dividend / Interest Rate = $1.40 / 0.05 = $28.00
- **Finite horizon:** FV = sum of discounted expected dividends + discounted redemption value

### Baseline Market Parameters

- Initial endowments: $1,000,000 cash + 10,000 shares per agent
- Fundamental value: $28.00
- Expected dividend: $1.40 per round (with +/- $1.00 variation, 50% probability each)
- Risk-free rate: 5% per round

---

## 6. Key Results

### Price Discovery Asymmetry

There is a striking asymmetry in price correction:
- **Below fundamental value** ($21 start, $28 FV): Price converges clearly toward the fundamental benchmark. Undervaluation is corrected effectively.
- **Above fundamental value** ($35 start, $28 FV): Price fails to converge downward within 20 rounds. Overvaluation persists despite the presence of value investors.
- This asymmetry is reinforced in extreme mispricing scenarios ($14 and $56 starting prices in infinite-horizon settings).

### Strategy Adherence

LLMs faithfully execute their assigned strategies and maintain strategic consistency over time. However, they lack intrinsic profit optimization -- they continue following instructions even when sustaining financial losses.

### Realistic Market Dynamics

The framework generates emergent phenomena resembling real financial markets:
- Price bubbles under certain agent compositions
- Liquidity provision through market maker agents with realistic bid-ask spread dynamics
- Volume clustering around key price levels
- Momentum-driven overshooting followed by reversals
- Market maker inventory accumulation patterns
- Wealth concentration among successful strategies

### Divergent Beliefs

When agents maintain conflicting valuation models (optimistic vs. pessimistic), prices reflect a weighted average of beliefs rather than converging to objective fundamentals. Disagreement generates substantial trading volume.

### Systemic Risk

A critical finding: agents sharing the same underlying LLM architecture receiving comparable prompts could exhibit correlated responses, inadvertently creating destabilizing trading patterns without explicit coordination.

---

## 7. Findings About Trading Horizon and Strategy Effectiveness

### Horizon Types Tested

- **Finite Horizon (15-20 rounds):** Agents are informed of a terminal date; final wealth is calculated via asset redemption at fundamental value. This creates stronger incentives for convergence.
- **Infinite Horizon (15 rounds):** No terminal information is given; final wealth is based on the last market price. This setting more closely resembles real-world continuous markets.

### Strategy-Specific Observations

- **Value Investors:** Show strong convergence to fundamental-value-based decisions; effective at identifying undervaluation. Decision heatmaps show distinct buy/sell/hold regions based on price-to-fundamental ratios.
- **Momentum Traders:** Less sensitive to fundamental ratios; more responsive to price trends. "The trend is your friend" philosophy is maintained consistently.
- **Market Makers:** Successfully provide liquidity through symmetric bid-ask spreads, typically 1-3% around market price with volatility-adjusted widths.
- **Contrarian Traders:** Look for excessive market moves to trade against.
- **Optimistic/Pessimistic Traders:** Systematically biased valuations (optimistic assumes 80-90% probability of maximum dividends; pessimistic assumes 80-90% minimum probability), which persistently affect price levels.

### Extended Horizon Results

In scenarios with unequal endowments and 100-round horizons:
- Resource advantages enable sustained directional price pressure
- Long-term market dominance by better-endowed agent types
- Initial price biases persist despite corrective trading pressure

---

## 8. What Baselines Are Compared?

### LLM-Based Agent Types

- Value Investors
- Momentum Traders
- Market Makers
- Contrarian Traders
- Speculators
- Optimistic/Pessimistic variants
- Retail Traders

### Deterministic Rule-Based Benchmarks

- Always-buy, always-sell, always-hold agents
- Technical traders (gap trading, mean reversion, momentum-based)
- Algorithmic market makers (fixed-spread strategies)

### Experimental Design

Mixed populations are tested to explore emergent behaviors from heterogeneous agent compositions. The framework supports both LLM-based and rule-based agents in the same simulation.

---

## 9. Evaluation Metrics

### Market-Level Metrics

- Price evolution relative to fundamental value
- Bid-ask spread dynamics over time
- Trading volume patterns
- Order book depth and quality
- Price-to-fundamental ratios

### Agent-Level Metrics

- Absolute position tracking (shares, cash, total wealth)
- Wealth composition and evolution over rounds
- Return calculations (absolute and percentage)
- Trading flow analysis (net buyer/seller identification)
- Decision consistency over time

### Decision Analysis Tools

- Order sizing patterns
- Limit price selection behavior
- Buy/sell/hold frequency distributions
- Reasoning quality assessment via natural language analysis of agent explanations

### Visualization Outputs

- Decision heatmaps revealing strategic patterns and decision boundaries
- Trading flow visualizations tracking position changes
- Reasoning wordclouds extracting decision drivers from agent explanations
- Market quality plots tracking spreads, depth, and efficiency

---

## 10. Code/Data Availability

The paper explicitly describes this as an **open-source framework** designed to "accelerate development in this emerging field." The framework:
- Provides a structured protocol for implementing and validating LLM trading agents
- Offers a controlled market environment with realistic microstructure
- Includes a comprehensive data collection system
- Supports both LLM-based and traditional rule-based agents
- Uses Pydantic-validated structured outputs for reproducibility

No specific GitHub URL was identified in the paper text or search results at the time of review, but the framework is described as open-source and designed for community use.

---

## Key Takeaways for Our Research

1. **GPT-4o as primary model:** Only one LLM was tested, leaving room for multi-model comparisons (e.g., Claude, Llama, Gemini).
2. **Prompt-driven strategy assignment works:** LLMs consistently adhere to assigned trading strategies, validating the approach of using system prompts to define agent behavior.
3. **Asymmetric price discovery:** Undervaluation corrects more readily than overvaluation -- a finding that could inform strategy design.
4. **Correlated behavior risk:** Agents using the same LLM architecture may inadvertently create systemic risk through correlated responses.
5. **Structured output validation (Pydantic):** Using schema-validated JSON outputs ensures clean, parseable agent decisions.
6. **Realistic market microstructure matters:** The persistent order book with limit orders, partial fills, and price-time priority creates meaningful market dynamics that simpler frameworks would miss.
7. **Natural language reasoning as a feature:** Agents explain their decisions, enabling qualitative analysis of strategy execution alongside quantitative metrics.
8. **100-round extended simulations:** Longer horizons reveal strategy dominance and resource-driven market dynamics not visible in short simulations.
