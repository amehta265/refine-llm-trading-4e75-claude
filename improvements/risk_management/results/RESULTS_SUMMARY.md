# Risk Management Improvement: Results Summary

## Configuration
- Trailing stop: 8%
- HOLD position fraction: 60%
- Volatility target: 15%

## Per-Stock Results

| Ticker | Freq | Orig CR | Risk CR | Orig SR | Risk SR | Orig MDD | Risk MDD | Stops |
|--------|------|---------|---------|---------|---------|----------|----------|-------|
| AAPL | daily | +24.4% | +7.3% | 1.38 | 0.31 | -8.9% | -5.4% | 0 |
| AAPL | monthly | +23.2% | +28.2% | 1.24 | 1.94 | -5.8% | -3.0% | 0 |
| AAPL | weekly | +7.1% | +2.4% | 0.11 | -0.22 | -12.9% | -11.7% | 0 |
| JPM | daily | +12.8% | +6.7% | 0.60 | 0.24 | -7.9% | -4.9% | 0 |
| JPM | monthly | +57.0% | +26.8% | 2.39 | 1.65 | -6.2% | -3.7% | 0 |
| JPM | weekly | +10.1% | +3.9% | 0.31 | -0.13 | -9.4% | -6.0% | 1 |
| MSFT | daily | +14.5% | +9.7% | 0.70 | 0.58 | -8.6% | -5.4% | 0 |
| MSFT | monthly | -2.7% | +0.1% | -0.60 | -0.44 | -11.9% | -9.3% | 0 |
| MSFT | weekly | -17.4% | -11.2% | -1.50 | -1.59 | -18.3% | -11.7% | 1 |
| NVDA | daily | +81.5% | +32.0% | 2.28 | 2.45 | -20.5% | -5.1% | 1 |
| NVDA | monthly | +16.3% | +31.6% | 0.36 | 0.93 | -13.1% | -1.7% | 0 |
| NVDA | weekly | +46.1% | +43.5% | 1.26 | 2.16 | -21.8% | -3.7% | 2 |
| TSLA | daily | +37.9% | +7.6% | 0.86 | 0.33 | -24.2% | -5.4% | 3 |
| TSLA | monthly | +104.5% | +77.3% | 2.10 | 2.20 | -7.7% | -4.6% | 0 |
| TSLA | weekly | +45.6% | -3.7% | 0.75 | -0.46 | -23.6% | -19.5% | 3 |

## Aggregate by Frequency

| Frequency | Orig SR | Risk SR | SR Change | Orig MDD | Risk MDD | MDD Change |
|-----------|---------|---------|-----------|----------|----------|------------|
| daily | 1.17 | 0.78 | -0.38 | -14.0% | -5.2% | +8.8% |
| weekly | 0.19 | -0.05 | -0.23 | -17.2% | -10.5% | +6.7% |
| monthly | 1.10 | 1.25 | +0.15 | -8.9% | -4.5% | +4.4% |

## Key Findings

- **Max Drawdown** improved by 6.6% on average (less negative = better)
- **Sharpe Ratio** changed by -0.154 on average
- **Cumulative Return** changed by -13.2% on average
- **Worst-case MDD** went from -24.2% to -19.5%
- **Total stop-loss triggers** across all conditions: 11

## Interpretation

The risk management layer successfully reduced maximum drawdowns across the experiment. 
Sharpe ratios declined, suggesting the position sizing was too aggressive in reducing exposure 
during periods that turned out to be profitable.

Whether these results represent improvement or not, they demonstrate that risk management is a 
meaningful dimension of the trading system that interacts with decision frequency in measurable ways.
## Sensitivity Analysis

Swept 4 stop-loss thresholds x 3 hold fractions x 3 volatility targets = 36 configurations.

### Top 5 by Sharpe Ratio

| Stop | Hold | Vol Target | Mean SR | Mean MDD | Mean CR |
|------|------|------------|---------|----------|---------|
| 10% | 80% | 20% | 0.803 | -8.4% | +21.5% |
| 15% | 80% | 20% | 0.803 | -8.4% | +21.5% |
| 8% | 80% | 20% | 0.803 | -8.4% | +21.5% |
| 8% | 60% | 20% | 0.740 | -7.5% | +18.8% |
| 10% | 60% | 20% | 0.740 | -7.5% | +18.8% |

Baseline (no risk management) mean SR: see comparison table for per-frequency values.