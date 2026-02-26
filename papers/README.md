# Downloaded Papers

Papers downloaded for the research project: "Refining LLM Trading — LLM agents may perform better in finance when making longer-term trading decisions rather than optimizing for short-term, day-to-day profit."

## Core Papers (Deep-Read)

1. **FINSABER: Can LLM-based Financial Investing Strategies Outperform the Market in Long Run?** (2026)
   - Authors: Li, Kim, Cucuringu, Ma
   - arXiv: 2505.07078 | KDD 2026
   - File: `finsaber_2505_07078.pdf`
   - Why relevant: Directly tests long-term LLM trading performance; shows daily trading fails over 20 years

2. **FinMem: A Performance-Enhanced LLM Trading Agent With Layered Memory and Character Design** (2023)
   - Authors: Yu, Li, Chen, Jiang, Li, Suchow, Zhang, Khashanah
   - arXiv: 2311.13743 | 148 citations
   - File: `finmem_2311_13743.pdf`
   - Why relevant: Foundational layered memory architecture for multi-horizon trading

3. **FinCon: A Synthesized LLM Multi-Agent System with Conceptual Verbal Reinforcement** (2024)
   - Authors: Yu et al.
   - arXiv: 2407.06567 | NeurIPS 2024
   - File: `fincon_2407_06567.pdf`
   - Why relevant: Manager-analyst hierarchy with belief updates (form of longer-term learning)

4. **TradingAgents: Multi-Agents LLM Financial Trading Framework** (2025)
   - Authors: Xiao, Sun, Luo, Wang
   - arXiv: 2412.20138 | 92 citations
   - File: `tradingagents_2412_20138.pdf`
   - Why relevant: Trading firm organization; high computational cost motivates less frequent decisions

5. **Large Language Model Agent in Financial Trading: A Survey** (2024)
   - Authors: Ding, Li, Wang, Chen
   - arXiv: 2408.06361 | 53 citations
   - File: `llm_trading_survey_2408_06361.pdf`
   - Why relevant: Comprehensive survey; identifies gap in frequency comparison

6. **FLAG-Trader: Fusion LLM-Agent with Gradient-based RL for Financial Trading** (2025)
   - Authors: Li et al.
   - arXiv: 2502.11433
   - File: `flag_trader_2502_11433.pdf`
   - Why relevant: RL fine-tuned 135M model beats GPT-4 in agentic frameworks

7. **DeepFund: Time Travel is Cheating** (2025)
   - Authors: HKUSTDial
   - arXiv: 2505.11065 | NeurIPS 2025
   - File: `deepfund_2505_11065.pdf`
   - Why relevant: Live trading benchmark; conservative strategies outperform aggressive daily trading

8. **StockBench: Can LLM Agents Trade Stocks Profitably?** (2025)
   - Authors: Chen et al.
   - arXiv: 2510.02209
   - File: `stockbench_2510_02209.pdf`
   - Why relevant: Contamination-free benchmark; regime-dependent performance

9. **InvestorBench: A Benchmark for Financial Decision-Making Tasks** (2024)
   - Authors: Li et al.
   - arXiv: 2412.18174 | ACL 2025
   - File: `investorbench_2412_18174.pdf`
   - Why relevant: Multi-asset benchmark testing 13 LLMs; ETF (long-term) is hardest task

10. **Can Large Language Models Trade? Testing Financial Theories** (2025)
    - Authors: Lopez-Lira
    - arXiv: 2504.10789
    - File: `can_llms_trade_2504_10789.pdf`
    - Why relevant: LLMs excel at value identification; supports longer holding periods

## Additional Papers (Abstract-Level Review)

11. **FinAgent: A Multimodal Foundation Agent for Financial Trading** (2024)
    - arXiv: 2402.18485 | File: `finagent_2402_18485.pdf`

12. **A Survey of Large Language Models for Financial Applications** (2024)
    - arXiv: 2406.11903 | File: `llm_finance_survey_2406_11903.pdf`

13. **TradingGPT: Multi-Agent System with Layered Memory** (2023)
    - arXiv: 2309.03736 | File: `tradinggpt_2309_03736.pdf`

14. **Simulating Financial Market via LLM-based Agents** (2024)
    - arXiv: 2406.19966 | File: `sim_financial_market_2406_19966.pdf`

15. **Automate Strategy Finding with LLM in Quant Investment** (2024)
    - arXiv: 2409.06289 | File: `auto_strategy_2409_06289.pdf`

16. **StockAgent: Large Language Model-based Stock Trading** (2024)
    - arXiv: 2407.18957 | File: `stockagent_2407_18957.pdf`

17. **FinVision: A Multi-Agent Framework for Stock Market Prediction** (2024)
    - arXiv: 2411.08899 | File: `finvision_2411_08899.pdf`

18. **From Deep Learning to LLMs: A Survey of AI in Quant Investment** (2025)
    - arXiv: 2503.21422 | File: `dl_to_llm_survey_2503_21422.pdf`

19. **MarketSenseAI 2.0: Enhancing Stock Analysis through LLM Agents** (2025)
    - arXiv: 2502.00415 | File: `marketsenseai2_2502_00415.pdf`

20. **TradExpert: Revolutionizing Trading with Mixture of Expert LLMs** (2024)
    - arXiv: 2411.00782 | File: `tradexpert_2411_00782.pdf`

21. **Trading-R1: Financial Trading with LLM Reasoning via RL** (2025)
    - arXiv: 2504.17727 | File: `trading_r1_2504_17727.pdf`

22. **MASS: Multi-Agent Simulation Scaling for Portfolio Construction** (2025)
    - arXiv: 2505.10278 | File: `mass_2505_10278.pdf`

23. **FinArena: A Human-Agent Collaboration Framework** (2025)
    - arXiv: 2503.02692 | File: `finarena_2503_02692.pdf`

## Detailed Notes

Deep-reading notes for core papers are in `notes/` directory:
- `notes/finsaber_notes.md`
- `notes/finmem_notes.md`
- `notes/fincon_notes.md`
- `notes/tradingagents_notes.md`
- `notes/llm_trading_survey_notes.md`
- `notes/flag_trader_notes.md`
- `notes/deepfund_notes.md`
- `notes/stockbench_notes.md`
- `notes/investorbench_notes.md`
- `notes/can_llms_trade_notes.md`
- `notes/compiled_summary.json` (structured summary of all deep-read papers)

<!--
    REVIEW (Note):
        Seems as though the literature review covers a breadth of knowledge in the "LLM for trading" space. There is
        inclusion of both survey papers (LLM Finance Survey, LLM Trading survey) and benchmarking papers (InvestorBench, StockBench). This to some degree demonstrates the agent's awareness of the broader field instead of just implementation specific papers.

    REVIEW (Improvement):
        1. Bias towards LLM and Agent based trading. There is no inclusion of quantitative finance or financial economics literature around trading frequency effects and strategies by "humans". The absence of such literature means that this experimental design is informed entirely by the ML/NLP community's conventions rather than by established financial knowledge.
        2. Recency bias: 21/23 papers are from 2024-2025. This may not be entirely significant given this is a new avenue of research. However, exclusion of older literature can reduce generalizability. 
-->
