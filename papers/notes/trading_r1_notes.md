# Notes: trading_r1_2504_12216

**IMPORTANT NOTE ON PAPER IDENTITY:** The file `trading_r1_2504_12216.pdf` (arXiv:2504.12216v2) is actually the paper **"d1: Scaling Reasoning in Diffusion Large Language Models via Reinforcement Learning"** by Siyan Zhao, Devaansh Gupta (UCLA), Qinqing Zheng (Meta AI), and Aditya Grover (UCLA). Published June 3, 2025. This is NOT a paper about financial trading. The filename appears to be mislabeled. The paper concerns applying reinforcement learning to diffusion-based language models (dLLMs) for reasoning tasks (math, planning, coding) -- not trading/finance. Below are comprehensive notes on the actual paper content.

---

## 1. What is d1 (mislabeled as "Trading-R1")?

d1 is a **two-stage post-training framework** for adapting pre-trained **masked diffusion large language models (dLLMs)** into reasoning models. It combines:
1. **Stage 1 -- Supervised Finetuning (SFT):** Finetune the dLLM on high-quality reasoning traces (using the s1K dataset of 1,000 curated reasoning questions).
2. **Stage 2 -- Reinforcement Learning (diffu-GRPO):** A novel policy-gradient based RL algorithm specifically designed for masked dLLMs.

The key contribution is bringing RL-based reasoning improvements (previously limited to autoregressive/AR LLMs like DeepSeek-R1) to the non-autoregressive diffusion LLM paradigm. This is claimed to be the **first application of policy gradient RL to masked dLLMs**.

## 2. How Does It Use Reinforcement Learning to Improve LLM Reasoning?

### The Challenge
Standard RL methods (PPO, GRPO) for AR models rely on computing log-probabilities of generated sequences via sequential factorization (chain rule). dLLMs lack this decomposition because they generate text through iterative denoising (coarse-to-fine), not token-by-token left-to-right.

### diffu-GRPO Algorithm
The paper introduces **diffu-GRPO**, which extends Group Relative Policy Optimization (GRPO) to masked dLLMs via:

1. **Mean-Field Approximation of Sequence Log Probability:** Decomposes sequence-level log-probability into a product of independent per-token log-probabilities.
2. **One-Step Per-Token Log Probability Estimation with Prompt Masking:** Instead of the expensive multi-step Monte Carlo approach (128 forward passes in original LLaDA), uses a single forward pass with a randomly masked prompt to estimate per-token log-probabilities.
3. **Random Prompt Masking as Regularization:** At each gradient update iteration, the prompt is randomly masked with probability p_mask. This creates diverse "views" of the same (prompt, completion) pair, acting as implicit regularization/data augmentation. This allows scaling the number of inner gradient updates (mu) to much higher values (12 or 24 vs. typical 2) while maintaining stable learning, which dramatically reduces the number of expensive online generations needed.

### Advantage Estimation
Uses unnormalized advantage: A_i = r_i - mean(r_j for all j in group), following GRPO with clipping and KL divergence regularization against a reference policy.

### Reward Functions
Composed reward functions combining:
- **Formatting rewards** (XML structure, proper tags)
- **Correctness rewards** (exact match for math, proportion correct for Sudoku, equation correctness for Countdown, unit test pass rate for coding)

## 3. What Tasks Are Evaluated? (NOT Financial Markets)

This paper evaluates on **reasoning tasks**, NOT financial/trading tasks:

### Mathematical Reasoning
- **GSM8K**: Multi-step grade school math problems
- **MATH500**: 500 curated high-school competition math problems from MATH dataset

### Planning
- **4x4 Sudoku**: Constraint satisfaction puzzles
- **Countdown (3 numbers)**: Combinatorial arithmetic game -- reach a target number using basic operations on given numbers

### Coding
- **HumanEval**: 164 hand-crafted Python programming problems
- **MBPP**: 257 crowd-sourced Python tasks

## 4. What LLMs Are Used?

### Primary Model
- **LLaDA-8B-Instruct**: A state-of-the-art open-source masked diffusion LLM (8B parameters), which had NOT undergone RL post-training

### Variants Evaluated
- LLaDA-8B-Instruct (baseline)
- LLaDA + SFT (SFT only on s1K)
- LLaDA + diffu-GRPO (RL only)
- **d1-LLaDA** = LLaDA + SFT + diffu-GRPO (the full d1 recipe)

### Comparison AR Models (from Dream paper)
- DeepSeek 7B
- Mistral 7B
- LLaMA3 8B
- Dream 7B (a dLLM)
- Qwen2.5 7B

## 5. What Datasets Are Used?

### SFT Training
- **s1K**: 1,000 high-quality reasoning questions with detailed step-by-step traces including verification and backtracking

### RL Training
- **GSM8K** training split
- **MATH** training split (for MATH500 evaluation)
- **Countdown** synthetic dataset (from TinyZero project, 3-number instances)
- **4x4 Sudoku** synthetic dataset (1M unique puzzles from Black-Phoenix/4x4-Sudoku-Dataset)
- **KodCode-Light-RL-10K** (for coding tasks, solutions verified by synthetic unit tests)

### Evaluation
- GSM8K test split
- MATH500 (curated 500 problems)
- 256 synthetically generated Countdown questions
- 256 randomly generated Sudoku puzzles
- HumanEval (164 problems)
- MBPP (257 problems)

## 6. Key Results

### Main Performance Table (Accuracy %)

| Model | GSM8K (best) | MATH500 (best) | Countdown (best) | Sudoku (best) |
|-------|-------------|----------------|-------------------|---------------|
| LLaDA-8B-Instruct | 78.2 | 36.2 | 20.7 | 11.7 |
| + SFT | 81.1 | 34.8 | 23.8 | 16.5 |
| + diffu-GRPO | 81.9 | 39.2 | 37.1 | 18.4 |
| **d1-LLaDA** | **82.1** | **40.2** | **42.2** | **22.1** |

### Key Findings
1. **diffu-GRPO consistently outperforms both base LLaDA and SFT** in all 12 evaluation setups (4 tasks x 3 sequence lengths).
2. **d1 recipe (SFT + diffu-GRPO) yields the highest gains** -- outperforms pure diffu-GRPO in 11/12 setups, showing synergistic effects.
3. **Largest gains on planning tasks**: Countdown +26.2% absolute, Sudoku +10.0% absolute. Math tasks show modest gains (+3.9% GSM8K, +4.0% MATH500) likely due to base model saturation.
4. **d1-LLaDA achieves highest GSM8K score (82.1%) and second-highest MATH500 score (40.2%)** among 7B-8B sized models (both dLLMs and AR models), surpassing Qwen2.5-7B on GSM8K.
5. **Unified multi-task model retains strong performance** -- a single model trained on combined datasets performs comparably to per-task models.
6. **Coding improvements**: diffu-GRPO consistently improves coding benchmarks regardless of initialization. HumanEval: 37.8 -> 39.0 (base) or 32.9 -> 37.8 (SFT); MBPP: 41.2 -> 45.5 (base).
7. **"Aha moments" emerge**: At 512-token generation lengths, SFT and d1-LLaDA models show self-correction and backtracking behaviors, instilled from s1K reasoning traces.

### Ablation Findings
- **Random masking >> Fixed masking** for log-probability estimation during RL. Random masking enables scaling mu (inner updates) to 12-24 vs. typical 2, dramatically improving wall-clock efficiency.
- **Optimal masking probability p_mask**: 0.1-0.3 works best. Higher (0.5-0.7) introduces instability. p_mask=0.0 (no masking) slightly underperforms, confirming regularization benefit.
- **RL training generalizes beyond training sequence length**: Though trained at seq_len=256, improvements carry over to 128 and 512.

## 7. Findings About Sequence Length / Scaling

- **GSM8K and MATH500**: Performance improves with increasing generation sequence length (128 -> 256 -> 512), with larger jumps from 128->256 (~7.1%) than 256->512 (~2.5%).
- **Countdown**: Base LLaDA decreases with length; but SFT/diffu-GRPO/d1 peak at 512 length.
- **Sudoku**: Performance DECREASES with increasing sequence length across ALL models -- mixed scaling trend.
- **No significant CoT length growth post-RL** (unlike DeepSeek-R1), since LLaDA was pre-trained on sequences up to 4096 tokens.
- **Limitation**: Fixed-length generation requirement of LLaDA constrains RL training. Slow generation speed makes it infeasible to train with larger generation lengths currently.

## 8. Baselines Compared

1. **LLaDA-8B-Instruct** (base dLLM, no post-training)
2. **LLaDA + SFT** (SFT only on s1K, 20 epochs)
3. **LLaDA + diffu-GRPO** (RL only, task-specific)
4. **d1-LLaDA** (SFT then diffu-GRPO)
5. Cross-architecture comparison with AR models: DeepSeek 7B, Mistral 7B, LLaMA3 8B, Dream 7B, Qwen2.5 7B

## 9. Evaluation Metrics

- **Accuracy (%)**: Primary metric for all tasks -- exact match of extracted answer vs. ground truth
- **Correctness Reward**: Used during RL training (2.0 for correct answer on GSM8K/MATH; proportion of correct cells for Sudoku; 1.0 for reaching target with correct numbers for Countdown; fraction of unit tests passed for coding)
- **Effective Token Usage**: Average number of non-padding, non-EOS tokens per generation -- used to analyze token efficiency
- **Wall Clock Time**: Used for training efficiency comparisons
- Evaluation uses **0-shot prompting** and **greedy decoding** with generation lengths of 128, 256, 512
- For diffu-GRPO and d1-LLaDA: evaluate every 100 steps from step 600, report best results

## 10. Code/Data Availability

- **Code**: https://github.com/dllm-reasoning/d1
- **Project page**: https://dllm-reasoning.github.io/
- Uses TRL library for diffu-GRPO implementation
- Training uses LoRA (rank=128, alpha=64 for RL; rank=128, alpha=256 for SFT)
- Hardware: 8x NVIDIA A100-80G GPUs for math/planning RL; 4x NVIDIA RTX A5000 for coding RL; 2x A6000 for SFT

## Key Takeaways for Our Project

1. **This paper is NOT about financial trading** despite the filename. It is about reasoning in diffusion language models.
2. The core methodological contribution (diffu-GRPO) is specific to masked diffusion LLMs and would not directly apply to standard autoregressive LLMs used in trading.
3. However, the general approach of using GRPO-style RL with composed reward functions (formatting + correctness) could inspire reward design for trading RL systems.
4. The finding that RL generalizes better than SFT (consistent with Chu et al. "SFT memorizes, RL generalizes") is relevant -- this principle would likely hold for trading tasks too.
5. The random masking regularization concept (creating diverse views of same data for RL stability) is an interesting technique that could have analogues in other RL settings.
