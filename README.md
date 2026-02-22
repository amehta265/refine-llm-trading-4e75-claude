# Refining LLM Trading: Does Decision Frequency Matter?

The first systematic comparison of LLM trading agent performance across daily, weekly, and monthly decision frequencies. Uses GPT-4.1-mini with real API calls on 5 stocks during 2024.

## Key Findings

- **Monthly LLM trading achieves comparable Sharpe ratios to daily** (1.10 vs 1.17 mean) while reducing max drawdowns by 36% (-8.9% vs -14.0%)
- **Weekly trading performs worst** (mean Sharpe 0.19) — a "valley of indecision" between daily and monthly
- **Monthly LLM beat Buy-and-Hold on 2/5 stocks** (JPM: +57% vs +43%, TSLA: +104% vs +53%), which is rare in the literature
- **Monthly decisions reduce API costs by 95%** (67K tokens vs 1.4M tokens) while the LLM adopts more strategic, trend-following behavior
- **Monthly max drawdown was lowest**: -7.7% for TSLA vs -24.2% daily and -40.9% B&H

## Project Structure

```
.
├── REPORT.md              # Full research report with results and analysis
├── planning.md            # Research plan and methodology
├── literature_review.md   # Synthesized literature review (23 papers)
├── resources.md           # Catalog of papers, datasets, code
├── src/
│   ├── data_loader.py     # Load and preprocess stock price data
│   ├── trading_agent.py   # LLM trading agent (GPT-4.1-mini)
│   ├── backtester.py      # Backtesting engine and metrics
│   └── analysis.py        # Statistical tests and visualization
├── results/               # JSON results per stock × frequency
├── figures/               # Generated plots (6 figures)
├── datasets/              # Stock price data (daily/weekly/monthly)
├── papers/                # 23 downloaded research papers
└── code/                  # 6 cloned reference repositories
```

## How to Reproduce

```bash
# Setup
uv venv && source .venv/bin/activate
uv add openai pandas numpy scipy matplotlib

# Set API key
export OPENAI_API_KEY="your-key"

# Run experiment (~22 min, ~$1.34 in API costs)
cd src && python backtester.py

# Generate analysis and figures
python analysis.py
```

## Experiment Details

- **Model:** GPT-4.1-mini (temperature=0.0)
- **Stocks:** AAPL, JPM, NVDA, TSLA, MSFT
- **Period:** January–December 2024
- **Decisions:** 1,585 total API calls (1,260 daily + 265 weekly + 60 monthly)
- **Total cost:** $1.34 (1.77M tokens)

See [REPORT.md](REPORT.md) for the complete analysis with statistical tests, visualizations, and discussion.
