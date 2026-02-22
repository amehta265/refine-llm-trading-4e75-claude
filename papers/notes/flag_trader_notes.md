# FLAG-TRADER: Fusion LLM-Agent with Gradient-based Reinforcement Learning for Financial Trading

**Paper:** arXiv:2502.11433v3 (19 Feb 2025)
**Authors:** Guojun Xiong (Harvard), Zhiyang Deng (Stevens), Keyi Wang (Columbia), Yupeng Cao, Haohang Li, Yangyang Yu (Stevens), Xueqing Peng (TheFinAI), Mingquan Lin (U. Minnesota), Kaleb E Smith (NVIDIA), Xiao-Yang Liu Yanglet (Columbia/RPI), Jimin Huang (TheFinAI), Sophia Ananiadou (U. Manchester), Qianqian Xie (TheFinAI)

---

## 1. What is FLAG-TRADER?

FLAG-TRADER (Fusion LLM-Agent with Gradient-based Reinforcement Learning) is a unified architecture that integrates linguistic processing via LLMs with gradient-driven reinforcement learning (RL) policy optimization for financial stock trading. The core idea is to use a **partially fine-tuned LLM as the policy network** in an actor-critic RL framework, leveraging the LLM's pre-trained knowledge while adapting it to financial decision-making through parameter-efficient fine-tuning.

Key innovations:
- A parameter-efficient fine-tuning module that jointly encodes temporal market data and textual streams into unified state representations.
- A hybrid RL component that explicitly incorporates external environment reward gradients into policy updates, ensuring alignment with trading performance metrics.
- The approach enables a small-scale (135M parameter) open-source LLM to surpass much larger proprietary models.

---

## 2. How Does It Combine LLM Reasoning with RL?

### Architecture
FLAG-TRADER uses an **LLM-based actor-critic architecture** with the following structure:

1. **Frozen Base Layers** (`theta_frozen`): Lower layers of the LLM are frozen to retain pre-trained general knowledge and language understanding.
2. **Trainable Top Layers** (`theta_train`): Upper layers of the LLM are fine-tuned for financial domain adaptation.
3. **Policy Head** (`theta_P`): An MLP head on top of trainable layers that outputs a probability distribution over actions {Buy, Sell, Hold}. Includes action masking for invalid trades (e.g., cannot sell when no stocks held).
4. **Value Head** (`theta_V`): A separate MLP head that estimates state value (critic), sharing the same LLM backbone as the policy network.

### State Representation
Market states are converted to structured text prompts (`lang(s_t)`) containing:
- Task description (financial trading objective)
- Legible action space (Buy, Sell, Hold)
- Current state (historical prices, account status with cash/asset/total value, previous decision metrics including recent rewards, net values, past actions)
- Output format specification (JSON)

### Training via PPO (Proximal Policy Optimization)
The system uses online policy gradient learning:
- **Policy loss** (`L_P`): PPO clipped surrogate objective with Generalized Advantage Estimation (GAE)
- **Value loss** (`L_V`): TD error minimized by the critic
- **Combined loss**: `L_total = -L_P + c1*L_V - c2*H(pi)` where H is entropy for exploration
- Updates are applied separately:
  - Policy head: `theta_P <- theta_P - eta * grad(L_P)`
  - Value head: `theta_V <- theta_V - eta * grad(L_V)`
  - Trainable LLM layers: `theta_train <- theta_train - beta * grad(L_total)`

### Reward Design
The reward at each step is the **change in Sharpe ratio**:
- `R(s_t, a_t) = SR_t - SR_{t-1}`
- Where SR_t is the Sharpe ratio computed from the historical PnL from time 0 to time t.
- PnL_t = (C_t - C_{t-1}) + (H_t * P_t - H_{t-1} * P_{t-1})

### MDP Formulation
- **State**: Market observations (price, news/sentiment) + trading account balance (cash, shares held)
- **Action**: Discrete {Sell: -1 (liquidate all), Hold: 0, Buy: +1 (buy with all cash)}
- **Objective**: Maximize expected cumulative discounted reward

---

## 3. Financial Tasks / Markets Evaluated

**Single-asset trading** across:
- **5 US Stocks**: Microsoft (MSFT), Johnson & Johnson (JNJ), UVV Corporation (UVV), Honeywell International (HON), Tesla (TSLA)
- **1 Cryptocurrency**: Bitcoin (BTC)

The task is daily buy/hold/sell decisions on individual assets.

---

## 4. LLMs Used

### FLAG-TRADER's backbone:
- **SmolLM2-135M-Instruct** (135M parameters) -- the primary model fine-tuned with RL

### Baselines (LLM-agentic framework via InvestorBench):

**Financial Domain Models:**
- Palmyra-Fin-70B

**Proprietary Models:**
- GPT-o1-preview
- GPT-4
- GPT-4o

**Open-Source Models:**
- Qwen2.5-72B-Instruct
- Llama-3.1-70B-Instruct
- DeepSeek-67B-Chat
- Yi-1.5-34B-Chat
- Qwen2.5-32B-Instruct
- DeepSeek-V2-Lite (15.7B)
- Yi-1.5-9B-Chat
- Llama-3.1-8B-Instruct
- Qwen-2.5-Instruct-7B

All models used temperature 0.6 during inference.

---

## 5. Datasets Used

The paper uses stock price data and crypto data with the following periods:

**Stocks (MSFT, JNJ, UVV, HON, TSLA):**
- Warm-up period: July 1, 2020 to September 30, 2020
- Test period: October 1, 2020 to May 6, 2021

**Bitcoin (BTC):**
- Warm-up period: February 11, 2023 to April 4, 2023
- Test period: April 5, 2023 to November 5, 2023

The prompt includes historical prices, account status (cash, asset position, total value), recent rewards, net values, and past action history. The data source is from the InvestorBench benchmark framework.

---

## 6. Key Results

### FLAG-TRADER (SmolLM2-135M) Performance Summary:

| Asset | CR (%) | SR | AV (%) | MDD (%) |
|-------|--------|----|--------|---------|
| MSFT | 20.106 | **1.373** | 24.932 | 9.428 |
| JNJ | **33.724** | **3.344** | **17.174** | **9.320** |
| UVV | 46.799 | 1.463 | 67.758 | 35.039 |
| HON | 34.342 | **2.429** | **23.913** | 10.872 |
| TSLA | **50.394** | **1.362** | 64.004 | 37.975 |
| BTC | **45.511** | **1.734** | **30.903** | 24.440 |

(Bold indicates best or near-best across all models.)

### Key Findings:

1. **Superior stock trading performance**: FLAG-TRADER consistently outperforms both buy-and-hold and LLM-agentic baselines across multiple metrics, particularly in Cumulative Return and Sharpe Ratio.

2. **Small model beats large models**: A 135M parameter model (SmolLM2-135M-Instruct) fine-tuned with RL outperforms much larger proprietary models (GPT-4, GPT-o1-preview) and open-source models (up to 72B parameters) used in agentic frameworks. This is the paper's most striking result.

3. **Convergence to stable optimal policy**: The RL training allows the LLM-based agent to converge to a stable policy that becomes less sensitive to initial prompts over time. Early training is prompt-dependent, but this diminishes.

4. **JNJ and BTC standout**: FLAG-TRADER achieved a Sharpe Ratio of 3.344 on JNJ (vs. 1.343 buy-and-hold) and 1.734 on BTC (vs. 0.683 buy-and-hold), representing massive improvements.

5. **TSLA performance**: CR of 50.394% vs. buy-and-hold's 39.244%, with improved SR (1.362 vs. 0.869).

---

## 7. Trading Horizon / Decision Frequency

- **Decision frequency**: Daily trading decisions (buy/sell/hold each day).
- **Action space**: All-in/all-out (buy with all cash, sell all holdings, or hold) -- no partial position sizing.
- **Test horizon**: Approximately 7 months for stocks (Oct 2020 - May 2021) and 7 months for BTC (Apr - Nov 2023).
- **Max episode steps**: 65 (per the hyperparameter table), suggesting roughly 65 trading days per training episode.
- The paper does **not** explicitly investigate the effect of different trading horizons or decision frequencies. All experiments use daily granularity.

---

## 8. Baselines Compared

1. **Buy & Hold**: Passive strategy; buy at start and hold throughout the test period. Standard benchmark.
2. **LLM-Agentic Framework (InvestorBench)**: Uses 13 different LLMs (proprietary and open-source) as backbone models in an agentic trading framework from InvestorBench (Li et al., 2024a). These LLMs generate trading decisions via prompting without RL fine-tuning.

The paper does **not** compare against:
- Traditional RL baselines (DQN, A2C, PPO with MLP/LSTM policy networks)
- Other LLM+RL hybrid methods
- Classical quantitative/technical analysis strategies

---

## 9. Evaluation Metrics

Four financial metrics are used (from Hull, 2007):

1. **Cumulative Return (CR) %** (higher is better): Total value change via sum of log returns. **Primary metric.**
2. **Sharpe Ratio (SR)** (higher is better): Risk-adjusted returns = (avg excess return - risk-free rate) / volatility. **Primary metric.**
3. **Annualized Volatility (AV) %** (lower is better): Standard deviation of daily log returns scaled by sqrt(252). Secondary metric.
4. **Maximum Drawdown (MDD) %** (lower is better): Largest peak-to-trough drop. Secondary metric.

CR and SR are treated as primary metrics. Results are selected from the test trajectory corresponding to the median of these metrics (prioritizing median SR if medians come from different epochs).

---

## 10. Code / Data Availability

The paper does **not** explicitly mention a code repository or public release of code/data in the main text or appendices. However:
- The baseline framework is from **InvestorBench** (Li et al., 2024a; arXiv:2412.18174), which is described as a benchmark.
- LLMs are deployed using the **vLLM** framework.
- The paper references **FinRL** (Liu et al., 2022) as a related open-source RL framework for finance.
- The paper is from **TheFinAI** group, which has released other financial AI tools.

---

## Additional Technical Details

### PPO Hyperparameters (Table 3):
| Parameter | Value |
|-----------|-------|
| total_timesteps | 13,860 |
| learning_rate | 5e-4 |
| num_envs | 1 |
| num_steps | 40 (steps per rollout) |
| gamma (discount) | 0.95 |
| gae_lambda | 0.98 |
| update_epochs | 1 |
| clip_coef (PPO epsilon) | 0.2 |
| ent_coef | 0.05 |
| vf_coef | 0.5 |
| kl_coef | 0.05 |
| max_grad_norm | 0.5 |
| minibatch_size | 32 |
| max_episode_steps | 65 |
| gradient_accumulation_steps | 8 |
| train_dtype | float16 |
| LLM | SmolLM2-135M-Instruct |

### Hardware:
- Small models (<10B): 2x RTX A6000 (48GB each)
- Mid models (10B-65B): 4x RTX A6000
- Large models (>65B): 8x A100 (80GB each)

---

## Limitations (as stated by authors)

1. **Computational cost**: Still expensive when fine-tuning on large-scale market datasets.
2. **Non-stationarity**: Financial markets' high volatility and non-stationarity pose challenges for long-term generalization. Suggest continual learning or meta-learning.
3. **Prompt bias**: Reliance on structured prompts could introduce decision-making biases. Suggest retrieval-augmented methods.
4. **No explicit risk management**: Optimizes for returns without explicitly incorporating risk-sensitive constraints or dynamic portfolio optimization.

---

## Relevance to Our Project

- Demonstrates that RL fine-tuning of even very small LLMs (135M) can beat much larger LLMs used in agentic prompting frameworks for trading.
- Uses a clean actor-critic architecture with parameter-efficient fine-tuning (frozen base + trainable top layers).
- Daily trading frequency with all-in/all-out actions -- a simplified action space.
- Sharpe ratio change as reward signal is an interesting design choice that directly optimizes risk-adjusted returns.
- Does not explore multi-asset portfolio management, only single-asset trading.
- Does not investigate different trading horizons or frequencies.
- The InvestorBench baseline framework and the set of 13 LLMs provide a useful comparison landscape.
- The paper validates that reward-driven RL optimization adds significant value beyond pure LLM prompting for sequential trading decisions.
