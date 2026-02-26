"""
Analysis and visualization for multi-frequency LLM trading experiment.
Computes comparative statistics and generates publication-quality plots.
"""

"""
REVIEW (Note): 
    The statistical rigor is significant as the testing pipeline is well reasoned.
    Shapiro-Wilk determines if the data is normally distributed guiding either a paired t-test or a 
    Wilcoxon signed-rank to determine statistical significance in comparisons.
    Bonferroni correction reduces false positives and Cohen's d gives the effect size making this
    flow methodologically sound
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def load_results(temperature: float = 0.0) -> dict:
    """Load all experiment results from JSON files."""
    results = {}
    for path in RESULTS_DIR.glob(f"*_t{temperature}.json"):
        if path.name.startswith("full_experiment"):
            continue
        key = path.stem.replace(f"_t{temperature}", "")
        with open(path) as f:
            results[key] = json.load(f)
    return results


def build_summary_table(results: dict) -> pd.DataFrame:
    """Build a summary DataFrame comparing all conditions."""
    rows = []
    for key, res in results.items():
        if "error" in res:
            continue
        ticker = res["ticker"]
        freq = res["frequency"]
        llm = res["llm_metrics"]
        bh = res["bh_metrics"]
        sma = res["sma_metrics"]

        rows.append({
            "Ticker": ticker,
            "Frequency": freq,
            "LLM_CR": llm["cumulative_return"],
            "LLM_SR": llm["sharpe_ratio"],
            "LLM_MDD": llm["max_drawdown"],
            "LLM_Sortino": llm["sortino_ratio"],
            "LLM_WinRate": llm["win_rate"],
            "LLM_AnnReturn": llm["annualized_return"],
            "LLM_AnnVol": llm["annualized_volatility"],
            "BH_CR": bh["cumulative_return"],
            "BH_SR": bh["sharpe_ratio"],
            "BH_MDD": bh["max_drawdown"],
            "SMA_CR": sma["cumulative_return"],
            "SMA_SR": sma["sharpe_ratio"],
            "Turnover": res["turnover_rate"],
            "Decisions": res["num_decisions"],
            "BUY_pct": res["decision_counts"]["BUY"] / max(res["num_decisions"], 1),
            "HOLD_pct": res["decision_counts"]["HOLD"] / max(res["num_decisions"], 1),
            "SELL_pct": res["decision_counts"]["SELL"] / max(res["num_decisions"], 1),
            "Tokens": res["total_tokens"],
        })

    df = pd.DataFrame(rows)
    # Order frequencies
    freq_order = {"daily": 0, "weekly": 1, "monthly": 2}
    df["freq_order"] = df["Frequency"].map(freq_order)
    df = df.sort_values(["Ticker", "freq_order"]).drop(columns="freq_order")
    return df


def statistical_tests(summary: pd.DataFrame) -> dict:
    """Run statistical tests comparing frequencies.

    Compares Sharpe ratios across stocks between daily/weekly/monthly.
    Uses paired tests since the same stocks are tested at each frequency.
    """
    results = {}

    # Get Sharpe ratios by frequency
    freq_sharpes = {}
    for freq in ["daily", "weekly", "monthly"]:
        mask = summary["Frequency"] == freq
        freq_sharpes[freq] = summary[mask].sort_values("Ticker")["LLM_SR"].values

    # Paired comparisons
    comparisons = [
        ("daily", "weekly"),
        ("daily", "monthly"),
        ("weekly", "monthly"),
    ]

    for f1, f2 in comparisons:
        s1 = freq_sharpes[f1]
        s2 = freq_sharpes[f2]

        if len(s1) < 3 or len(s2) < 3:
            results[f"{f1}_vs_{f2}"] = {"note": "Insufficient data for test"}
            continue

        # Check normality (Shapiro-Wilk) # This makes a lot of sense to determine which statistical test is safe to use
        diff = s2 - s1
        if len(diff) >= 3:
            _, p_normal = stats.shapiro(diff)
        else:
            p_normal = 0.0

        # Use paired t-test if normal, Wilcoxon if not
        if p_normal > 0.05 and len(diff) >= 3:
            t_stat, p_value = stats.ttest_rel(s1, s2)
            test_name = "paired t-test"
        else:
            try:
                stat, p_value = stats.wilcoxon(s1, s2)
                t_stat = stat
                test_name = "Wilcoxon signed-rank"
            except ValueError:
                t_stat, p_value = 0, 1.0
                test_name = "N/A (identical values)"
        
        """
            REVIEW (Improvement): Need a larger sample size.
                Cohen's d (effect size) is large for daily-vs-weekly (d=1.31) and weekly-vs-monthly (d=0.92) trading, both above the 0.8 threshold.
                Yet the paired t-tests show p > 0.05. These combined express a "large effect but NOT significant." The reason for this is likely the
                sample size: with only 5 stocks. There's too much uncertainty to rule out chance leading to the p > 0.05.
                Expanding to 15-20 stocks would likely achieve significance while preserving similar effect sizes.
        """

        # Effect size (Cohen's d for paired samples)
        if diff.std() > 0:
            cohens_d = diff.mean() / diff.std()
        else:
            cohens_d = 0.0

        # Bonferroni correction (3 comparisons)
        p_adjusted = min(p_value * 3, 1.0) # Without this, the combined false positive risk rises to roughly 14%.

        results[f"{f1}_vs_{f2}"] = {
            "test": test_name,
            "statistic": float(t_stat),
            "p_value": float(p_value),
            "p_adjusted_bonferroni": float(p_adjusted),
            "mean_diff": float(diff.mean()),
            "cohens_d": float(cohens_d),
            f"{f1}_mean_SR": float(s1.mean()),
            f"{f2}_mean_SR": float(s2.mean()),
            "normality_p": float(p_normal),
            "significant_005": p_adjusted < 0.05,
        }

    """
        REVIEW (Improvement): Cumulative Return (CR) differences are not formally tested.
            Cumulative Return is the total percentage gain or loss over the full experiment period.
            The code above runs a full statistical pipeline on Sharpe ratios (Shapiro-Wilk -> paired t-test/Wilcoxon -> Bonferroni),
            but for CR it only computes averages (line 192-200). Based on `results/statistical_tests.json`, the CR means are:
                - Daily:   34.2%
                - Weekly:  18.3%
                - Monthly: 39.7%
            These are meaningful differences — monthly outperformed daily by 5.5% and weekly by 21.4%. Yet no test tells us
            whether these gaps are statistically significant or just noise from the small 5-stock sample.
            The same pipeline used for Sharpe ratios in the first half of this function should be applied to CR as well. Without it,
            we're presenting the most investor-relevant metric without any measure of confidence, while the less intuitive
            Sharpe ratio gets full statistical treatment.
    """
    freq_crs = {}
    for freq in ["daily", "weekly", "monthly"]:
        mask = summary["Frequency"] == freq
        freq_crs[freq] = summary[mask].sort_values("Ticker")["LLM_CR"].values

    results["cr_means"] = {
        freq: float(vals.mean()) for freq, vals in freq_crs.items()
    }

    for f1, f2 in comparisons:
        c1 = freq_crs[f1]
        c2 = freq_crs[f2]

        if len(c1) < 3 or len(c2) < 3:
            results[f"cr_{f1}_vs_{f2}"] = {"note": "Insufficient data for CR test"}
            continue

        diff_cr = c2 - c1
        if len(diff_cr) >= 3:
            _, p_normal_cr = stats.shapiro(diff_cr)
        else:
            p_normal_cr = 0.0

        if p_normal_cr > 0.05 and len(diff_cr) >= 3:
            t_stat_cr, p_value_cr = stats.ttest_rel(c1, c2)
            test_name_cr = "paired t-test"
        else:
            try:
                stat_cr, p_value_cr = stats.wilcoxon(c1, c2)
                t_stat_cr = stat_cr
                test_name_cr = "Wilcoxon signed-rank"
            except ValueError:
                t_stat_cr, p_value_cr = 0, 1.0
                test_name_cr = "N/A (identical values)"

        if diff_cr.std() > 0:
            cohens_d_cr = diff_cr.mean() / diff_cr.std()
        else:
            cohens_d_cr = 0.0

        p_adjusted_cr = min(p_value_cr * 3, 1.0)

        results[f"cr_{f1}_vs_{f2}"] = {
            "test": test_name_cr,
            "statistic": float(t_stat_cr),
            "p_value": float(p_value_cr),
            "p_adjusted_bonferroni": float(p_adjusted_cr),
            "mean_diff": float(diff_cr.mean()),
            "cohens_d": float(cohens_d_cr),
            f"{f1}_mean_CR": float(c1.mean()),
            f"{f2}_mean_CR": float(c2.mean()),
            "normality_p": float(p_normal_cr),
            "significant_005": p_adjusted_cr < 0.05,
        }

    return results


def plot_sharpe_comparison(summary: pd.DataFrame, save_path: Path = None):
    """Bar chart comparing Sharpe ratios across frequencies for each stock."""
    fig, ax = plt.subplots(figsize=(12, 6))

    tickers = summary["Ticker"].unique()
    frequencies = ["daily", "weekly", "monthly"]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    x = np.arange(len(tickers))
    width = 0.25

    for i, freq in enumerate(frequencies):
        mask = summary["Frequency"] == freq
        data = summary[mask].sort_values("Ticker")
        bars = ax.bar(x + i * width, data["LLM_SR"].values, width,
                      label=freq.capitalize(), color=colors[i], alpha=0.85)

    # Add Buy-and-Hold reference
    bh_sharpes = summary[summary["Frequency"] == "daily"].sort_values("Ticker")["BH_SR"].values
    ax.scatter(x + width, bh_sharpes, marker="D", color="red", s=60, zorder=5,
               label="Buy & Hold", edgecolors="darkred", linewidths=0.5)

    ax.set_xlabel("Stock", fontsize=12)
    ax.set_ylabel("Sharpe Ratio", fontsize=12)
    ax.set_title("LLM Trading Sharpe Ratio by Decision Frequency", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(tickers, fontsize=11)
    ax.legend(fontsize=10)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path is None:
        save_path = FIGURES_DIR / "sharpe_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_cumulative_returns(summary: pd.DataFrame, save_path: Path = None):
    """Bar chart comparing cumulative returns across frequencies."""
    fig, ax = plt.subplots(figsize=(12, 6))

    tickers = summary["Ticker"].unique()
    frequencies = ["daily", "weekly", "monthly"]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    x = np.arange(len(tickers))
    width = 0.2

    for i, freq in enumerate(frequencies):
        mask = summary["Frequency"] == freq
        data = summary[mask].sort_values("Ticker")
        vals = data["LLM_CR"].values * 100  # Convert to percentage
        ax.bar(x + i * width, vals, width,
               label=f"LLM {freq.capitalize()}", color=colors[i], alpha=0.85)

    # Add Buy-and-Hold
    bh_cr = summary[summary["Frequency"] == "daily"].sort_values("Ticker")["BH_CR"].values * 100
    ax.bar(x + 3 * width, bh_cr, width, label="Buy & Hold",
           color="#F44336", alpha=0.85)

    ax.set_xlabel("Stock", fontsize=12)
    ax.set_ylabel("Cumulative Return (%)", fontsize=12)
    ax.set_title("Cumulative Returns by Decision Frequency (2024)", fontsize=14, fontweight="bold")
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(tickers, fontsize=11)
    ax.legend(fontsize=10)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path is None:
        save_path = FIGURES_DIR / "cumulative_returns.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_decision_distribution(summary: pd.DataFrame, save_path: Path = None):
    """Stacked bar chart showing BUY/HOLD/SELL distribution by frequency."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    frequencies = ["daily", "weekly", "monthly"]
    for ax, freq in zip(axes, frequencies):
        mask = summary["Frequency"] == freq
        data = summary[mask].sort_values("Ticker")

        tickers = data["Ticker"].values
        buy = data["BUY_pct"].values * 100
        hold = data["HOLD_pct"].values * 100
        sell = data["SELL_pct"].values * 100

        x = np.arange(len(tickers))
        ax.bar(x, buy, label="BUY", color="#4CAF50", alpha=0.85)
        ax.bar(x, hold, bottom=buy, label="HOLD", color="#FFC107", alpha=0.85)
        ax.bar(x, sell, bottom=buy + hold, label="SELL", color="#F44336", alpha=0.85)

        ax.set_title(f"{freq.capitalize()}", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(tickers, rotation=45, fontsize=9)
        ax.set_ylim(0, 105)
        if freq == "daily":
            ax.set_ylabel("Decision Distribution (%)", fontsize=11)
        if freq == "monthly":
            ax.legend(fontsize=9)

    fig.suptitle("LLM Decision Distribution by Frequency", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path is None:
        save_path = FIGURES_DIR / "decision_distribution.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_risk_return_scatter(summary: pd.DataFrame, save_path: Path = None):
    """Scatter plot of annualized return vs volatility by frequency."""
    fig, ax = plt.subplots(figsize=(10, 7))

    freq_colors = {"daily": "#2196F3", "weekly": "#4CAF50", "monthly": "#FF9800"}
    freq_markers = {"daily": "o", "weekly": "s", "monthly": "^"}

    for freq in ["daily", "weekly", "monthly"]:
        mask = summary["Frequency"] == freq
        data = summary[mask]
        ax.scatter(
            data["LLM_AnnVol"] * 100, data["LLM_AnnReturn"] * 100,
            c=freq_colors[freq], marker=freq_markers[freq],
            s=120, label=f"LLM {freq.capitalize()}", alpha=0.8,
            edgecolors="black", linewidths=0.5
        )
        # Label each point with ticker
        for _, row in data.iterrows():
            ax.annotate(
                row["Ticker"],
                (row["LLM_AnnVol"] * 100, row["LLM_AnnReturn"] * 100),
                textcoords="offset points", xytext=(5, 5), fontsize=8
            )

    ax.set_xlabel("Annualized Volatility (%)", fontsize=12)
    ax.set_ylabel("Annualized Return (%)", fontsize=12)
    ax.set_title("Risk-Return Profile by Decision Frequency", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path is None:
        save_path = FIGURES_DIR / "risk_return_scatter.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_mdd_comparison(summary: pd.DataFrame, save_path: Path = None):
    """Bar chart comparing max drawdown across frequencies."""
    fig, ax = plt.subplots(figsize=(12, 6))

    tickers = summary["Ticker"].unique()
    frequencies = ["daily", "weekly", "monthly"]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    x = np.arange(len(tickers))
    width = 0.2

    for i, freq in enumerate(frequencies):
        mask = summary["Frequency"] == freq
        data = summary[mask].sort_values("Ticker")
        vals = data["LLM_MDD"].values * 100
        ax.bar(x + i * width, vals, width,
               label=f"LLM {freq.capitalize()}", color=colors[i], alpha=0.85)

    bh_mdd = summary[summary["Frequency"] == "daily"].sort_values("Ticker")["BH_MDD"].values * 100
    ax.bar(x + 3 * width, bh_mdd, width, label="Buy & Hold", color="#F44336", alpha=0.85)

    ax.set_xlabel("Stock", fontsize=12)
    ax.set_ylabel("Maximum Drawdown (%)", fontsize=12)
    ax.set_title("Maximum Drawdown by Decision Frequency", fontsize=14, fontweight="bold")
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(tickers, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path is None:
        save_path = FIGURES_DIR / "max_drawdown.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_frequency_summary(summary: pd.DataFrame, save_path: Path = None):
    """Aggregate metrics by frequency (mean across stocks)."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    metrics = [
        ("LLM_CR", "Cumulative Return", 100),
        ("LLM_SR", "Sharpe Ratio", 1),
        ("LLM_MDD", "Max Drawdown", 100),
        ("Turnover", "Turnover Rate", 100),
    ]

    frequencies = ["daily", "weekly", "monthly"]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    for ax, (col, title, scale) in zip(axes, metrics):
        means = []
        stds = []
        for freq in frequencies:
            mask = summary["Frequency"] == freq
            vals = summary[mask][col].values * scale
            means.append(vals.mean())
            stds.append(vals.std())

        bars = ax.bar(frequencies, means, color=colors, alpha=0.85,
                      yerr=stds, capsize=5, edgecolor="black", linewidth=0.5)
        ax.set_title(title, fontsize=12, fontweight="bold")
        unit = "%" if scale == 100 else ""
        ax.set_ylabel(f"{title} {unit}", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        # Add value labels
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{mean:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle("Average Performance Metrics by Decision Frequency",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path is None:
        save_path = FIGURES_DIR / "frequency_summary.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def run_full_analysis(temperature: float = 0.0):
    """Run all analysis steps and generate figures."""
    print("Loading results...")
    results = load_results(temperature)

    if not results:
        print("No results found! Run the experiment first.")
        return

    print(f"Loaded {len(results)} results")

    # Build summary table
    summary = build_summary_table(results)
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(summary.to_string(index=False))

    # Save summary CSV
    summary.to_csv(RESULTS_DIR / "summary_table.csv", index=False)

    # Statistical tests
    print("\n" + "=" * 80)
    print("STATISTICAL TESTS")
    print("=" * 80)
    stat_results = statistical_tests(summary)
    for comp, res in stat_results.items():
        print(f"\n{comp}:")
        if isinstance(res, dict):
            for k, v in res.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")

    # Save statistical results (convert numpy types for JSON)
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    with open(RESULTS_DIR / "statistical_tests.json", "w") as f:
        json.dump(make_serializable(stat_results), f, indent=2)

    # Generate plots
    print("\nGenerating visualizations...")
    plot_sharpe_comparison(summary)
    plot_cumulative_returns(summary)
    plot_decision_distribution(summary)
    plot_risk_return_scatter(summary)
    plot_mdd_comparison(summary)
    plot_frequency_summary(summary)

    print("\nAnalysis complete!")
    return summary, stat_results


if __name__ == "__main__":
    run_full_analysis()
