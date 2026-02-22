# Cloned Repositories

Code repositories cloned for the research project: "Refining LLM Trading."

## Repository 1: FINSABER (**Most Important**)
- **URL:** https://github.com/waylonli/FINSABER
- **Purpose:** Comprehensive backtesting framework for evaluating LLM trading strategies over long time horizons with bias mitigation
- **Location:** `code/FINSABER/`
- **Paper:** arXiv:2505.07078 (KDD 2026)
- **Key files:**
  - `backtest/` — Backtesting pipeline with rolling-window evaluation
  - `backtest/data_util/finmem_dataset.py` — Data loader for FINSABER dataset
  - `strategy/` — Strategy implementations (FinMem, FinAgent, traditional, ML, RL)
  - `analysis/` — Performance analysis and visualization
- **License:** CC BY 4.0
- **Notes:** This is the primary framework for our experiments. It provides 20-year bias-mitigated backtesting, support for multiple strategy types, and pre-integrated multi-source data (prices, news, filings). We need to extend it to support weekly/monthly decision frequencies.

## Repository 2: InvestorBench
- **URL:** https://github.com/felis33/INVESTOR-BENCH
- **Purpose:** Benchmark for evaluating LLM agents across stocks, crypto, and ETFs
- **Location:** `code/InvestorBench/`
- **Paper:** arXiv:2412.18174 (ACL 2025)
- **Key files:**
  - Agent framework with Brain/Perception/Profile/Memory/Action modules
  - Layered memory architecture (shallow/intermediate/deep)
  - Evaluation pipeline for 13+ LLMs
- **License:** MIT
- **Notes:** Provides the FinMem-based agent architecture with layered memory. The memory layer design (14-day/90-day/365-day decay) is key for our multi-horizon experiments.

## Repository 3: TradingAgents
- **URL:** https://github.com/TauricResearch/TradingAgents
- **Purpose:** Multi-agent LLM trading framework mimicking trading firm organization
- **Location:** `code/TradingAgents/`
- **Paper:** arXiv:2412.20138
- **Key files:**
  - Multi-agent architecture (analysts, researchers, trader, risk management, fund manager)
  - Hybrid communication protocol (structured reports + natural language debate)
  - Data collection from Bloomberg, Yahoo, Finnhub, Reddit, EODHD
- **License:** MIT
- **Notes:** Useful reference for multi-agent architecture design. Computational cost (~11 LLM calls per decision) motivates less frequent trading.

## Repository 4: FinMem
- **URL:** https://github.com/pipiku915/FinMem-LLM-StockTrading
- **Purpose:** LLM trading agent with layered memory and character design
- **Location:** `code/FinMem/`
- **Paper:** arXiv:2311.13743
- **Key files:**
  - Layered memory module (shallow/intermediate/deep with FAISS)
  - Self-adaptive risk character
  - GPT-4-Turbo trading agent
- **License:** MIT
- **Notes:** Foundational architecture for multi-horizon memory. Key design patterns: Ebbinghaus forgetting curve for recency, cosine similarity for relevancy, layer-specific importance scoring.

## Repository 5: DeepFund
- **URL:** https://github.com/HKUSTDial/DeepFund
- **Purpose:** Live (real-time) fund investment benchmarking tool
- **Location:** `code/DeepFund/`
- **Paper:** arXiv:2505.11065 (NeurIPS 2025)
- **Key files:**
  - Multi-agent architecture (Planner, Analyst Team, Portfolio Manager)
  - Dual memory (short-term operational + long-term historical)
  - Live dashboard integration
- **License:** MIT
- **Notes:** Demonstrates backtesting vs live performance gap. Conservative strategies outperformed aggressive daily trading, supporting our hypothesis.

## Repository 6: FinRL
- **URL:** https://github.com/AI4Finance-Foundation/FinRL
- **Purpose:** Standard RL library for financial applications (baseline)
- **Location:** `code/FinRL/`
- **Key files:**
  - `finrl/meta/data_processors/` — Data processing from multiple sources
  - `finrl/agents/` — DRL agent implementations (PPO, A2C, DQN, SAC, TD3, DDPG)
  - `finrl/meta/env_stock_trading/` — Stock trading environment
- **License:** MIT
- **Notes:** Standard baseline library. Used by FINSABER for RL baselines. Provides data processing, environment setup, and trained DRL agents for comparison.

## Usage for Our Research

### Recommended Workflow
1. **Start with FINSABER:** Use its backtesting framework and dataset as the foundation
2. **Extend for multi-frequency:** Modify the backtesting pipeline to support weekly/monthly decisions
3. **Use FinMem architecture:** Adapt its layered memory for different decision frequencies
4. **Compare against FinRL baselines:** Use FinRL's DRL agents (PPO, A2C) as traditional baselines
5. **Reference TradingAgents/DeepFund:** For multi-agent design patterns and live evaluation insights
