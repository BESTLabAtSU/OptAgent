"""
Comprehensive Benchmark Analysis and Publication-Quality Visualization
For academic paper publication with multi-dimensional analysis
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Statistical analysis
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal, wilcoxon
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Set publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


@dataclass
class AnalysisConfig:
    """Configuration for analysis"""
    results_path: Path
    output_dir: Path
    figure_format: str = "pdf"  # pdf, png, svg
    color_palette: str = "colorblind"
    significance_level: float = 0.05


class BenchmarkAnalyzer:
    """Comprehensive benchmark analysis for academic publication"""

    # Category mappings
    COMPLEXITY_ORDER = [
        "single_agent_single_tool",
        "single_agent_multi_tool",
        "multi_agent_single_tool",
        "multi_agent_multi_tool"
    ]

    COMPLEXITY_LABELS = {
        "single_agent_single_tool": "SA-ST",
        "single_agent_multi_tool": "SA-MT",
        "multi_agent_single_tool": "MA-ST",
        "multi_agent_multi_tool": "MA-MT"
    }

    MODE_ORDER = ["C-1", "C-2", "D"]
    MODE_LABELS = {
        "C-1": "Centralized\n(Single-Stage)",
        "C-2": "Centralized\n(Two-Stage)",
        "D": "Decentralized"
    }

    DOMAIN_CATEGORIES = {
        "CFG": "Configuration",
        "BLD": "Building",
        "HVAC": "HVAC",
        "DER": "DER",
        "CTRL": "Controller",
        "DIST": "Disturbance",
        "ENV": "Environment",
        "SIM": "Simulation",
        "ANL": "Analysis",
        "CMP": "Comparison",
        "COM": "Communication"
    }

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Load and preprocess data
        self.df = self._load_results()
        self._preprocess_data()

        # Color schemes
        self.colors = {
            "mode": {"C-1": "#2ecc71", "C-2": "#3498db", "D": "#e74c3c"},
            "success": {"True": "#27ae60", "False": "#c0392b"},
            "complexity": sns.color_palette("viridis", 4),
            "domain": sns.color_palette("husl", 11)
        }

    def _load_results(self) -> pd.DataFrame:
        """Load benchmark results from JSON"""
        with open(self.config.results_path, 'r') as f:
            data = json.load(f)

        records = []
        for item in data:
            config = item.get("config", {})
            accuracy = item.get("accuracy_metrics", {})

            record = {
                # Config
                "test_id": config.get("test_case_id", ""),
                "test_name": config.get("test_case_name", ""),
                "mode": self._get_mode_label(config),
                "orchestrator_model": config.get("orchestrator_model", ""),
                "agent_model": config.get("agent_model", ""),
                "provider": config.get("provider", ""),
                "two_stage": config.get("two_stage_planning", False),
                "repeat_index": config.get("repeat_index", 0),

                # Category
                "complexity": item.get("category", ""),
                "domain": self._extract_domain(config.get("test_case_id", "")),

                # Metrics
                "success": item.get("success", False),
                "tool_accuracy": accuracy.get("tool_selection_accuracy", 0),
                "agent_accuracy": accuracy.get("agent_selection_accuracy", 0),
                "plan_accuracy": accuracy.get("plan_step_accuracy", 0),
                "param_key_accuracy": accuracy.get("parameter_key_accuracy", 0),
                "param_value_accuracy": accuracy.get("parameter_value_accuracy", 0),

                # Timing
                "total_time": item.get("total_time", 0),
                "planning_time": item.get("planning_time", 0),
                "execution_time": item.get("execution_time", 0),

                # Tokens & Cost
                "total_tokens": item.get("grand_total_tokens", 0),
                "orchestrator_tokens": item.get("orchestrator_total_tokens", 0),
                "agent_tokens": item.get("agent_total_tokens", 0),
                "total_cost": item.get("total_cost", 0),

                # Tool usage
                "tool_calls": item.get("tool_calls", 0),
                "tools_expected": len(accuracy.get("tools_expected", [])),
                "tools_actual": len(accuracy.get("tools_actual", [])),
                "tools_missing": len(accuracy.get("tools_missing", [])),
                "tools_extra": len(accuracy.get("tools_extra", [])),
            }
            records.append(record)

        return pd.DataFrame(records)

    def _get_mode_label(self, config: Dict) -> str:
        intel_mode = config.get("intelligence_mode", "").lower()
        two_stage = config.get("two_stage_planning", False)

        if intel_mode == "centralized":
            return "C-2" if two_stage else "C-1"
        elif intel_mode == "decentralized":
            return "D"
        else:
            # Fallback - log warning for unexpected values
            return "D"

    def _extract_domain(self, test_id: str) -> str:
        parts = test_id.split("_")
        if len(parts) >= 2:
            return parts[1]
        return "UNKNOWN"

    def _preprocess_data(self):
        """Add derived columns"""
        # Composite accuracy
        self.df["avg_accuracy"] = (
                                          self.df["tool_accuracy"] +
                                          self.df["agent_accuracy"] +
                                          self.df["plan_accuracy"]
                                  ) / 3

        # Model tier
        def get_model_tier(row):
            if "gpt-4o" in row["orchestrator_model"]:
                return "High"
            elif "gemma2:9b" in row["orchestrator_model"]:
                return "Medium"
            else:
                return "Low"

        self.df["model_tier"] = self.df.apply(get_model_tier, axis=1)

        # Model configuration label
        self.df["model_config"] = (
                self.df["orchestrator_model"].str.replace(":", "-") + " / " +
                self.df["agent_model"].str.replace(":", "-")
        )

        # Complexity label
        self.df["complexity_label"] = self.df["complexity"].map(self.COMPLEXITY_LABELS)

        # Domain label
        self.df["domain_label"] = self.df["domain"].map(self.DOMAIN_CATEGORIES)

    def get_summary_statistics(self) -> pd.DataFrame:
        """Generate comprehensive summary statistics"""
        metrics = ["tool_accuracy", "agent_accuracy", "plan_accuracy",
                   "avg_accuracy", "total_time", "total_tokens", "total_cost"]

        summary = self.df.groupby("mode")[metrics].agg(["mean", "std", "min", "max", "count"])
        return summary

    def statistical_tests(self) -> Dict[str, Any]:
        """Perform statistical significance tests"""
        results = {}

        metrics = ["tool_accuracy", "agent_accuracy", "plan_accuracy", "avg_accuracy", "total_time"]

        for metric in metrics:
            # Kruskal-Wallis test (non-parametric ANOVA)
            groups = [self.df[self.df["mode"] == m][metric].values for m in self.MODE_ORDER]
            h_stat, p_val = kruskal(*groups)

            # Pairwise Mann-Whitney U tests
            pairwise = {}
            for i, m1 in enumerate(self.MODE_ORDER):
                for m2 in self.MODE_ORDER[i + 1:]:
                    g1 = self.df[self.df["mode"] == m1][metric].values
                    g2 = self.df[self.df["mode"] == m2][metric].values
                    u_stat, p = mannwhitneyu(g1, g2, alternative='two-sided')
                    pairwise[f"{m1} vs {m2}"] = {"U": u_stat, "p": p, "significant": p < 0.05}

            results[metric] = {
                "kruskal_wallis": {"H": h_stat, "p": p_val},
                "pairwise": pairwise
            }

        return results

    # ==================== VISUALIZATION METHODS ====================

    def plot_mode_comparison_radar(self, save: bool = True) -> plt.Figure:
        """Radar chart comparing modes across all metrics"""
        metrics = ["tool_accuracy", "agent_accuracy", "plan_accuracy",
                   "param_key_accuracy", "param_value_accuracy"]
        labels = ["Tool\nSelection", "Agent\nSelection", "Plan\nSequence",
                  "Param\nKeys", "Param\nValues"]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the loop

        for mode in self.MODE_ORDER:
            mode_data = self.df[self.df["mode"] == mode]
            values = [mode_data[m].mean() for m in metrics]
            values += values[:1]

            ax.plot(angles, values, 'o-', linewidth=2,
                    label=mode, color=self.colors["mode"][mode])
            ax.fill(angles, values, alpha=0.25, color=self.colors["mode"][mode])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"])
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.set_title("Accuracy Comparison Across Intelligence Modes", pad=20)

        if save:
            fig.savefig(self.config.output_dir / f"radar_mode_comparison.{self.config.figure_format}")
        return fig

    def plot_complexity_performance(self, save: bool = True) -> plt.Figure:
        """Grouped bar chart: performance by complexity and mode"""
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))

        metrics = [("tool_accuracy", "Tool Selection Accuracy"),
                   ("agent_accuracy", "Agent Selection Accuracy"),
                   ("plan_accuracy", "Plan Sequence Accuracy")]

        x = np.arange(len(self.COMPLEXITY_ORDER))
        width = 0.25

        for ax_idx, (metric, title) in enumerate(metrics):
            ax = axes[ax_idx]

            for i, mode in enumerate(self.MODE_ORDER):
                means = []
                stds = []
                for complexity in self.COMPLEXITY_ORDER:
                    subset = self.df[(self.df["mode"] == mode) &
                                     (self.df["complexity"] == complexity)]
                    means.append(subset[metric].mean())
                    stds.append(subset[metric].std())

                ax.bar(x + (i - 1) * width, means, width,
                       label=mode, color=self.colors["mode"][mode],
                       yerr=stds, capsize=3, alpha=0.85)

            ax.set_xlabel("Task Complexity")
            ax.set_ylabel(title)
            ax.set_xticks(x)
            ax.set_xticklabels([self.COMPLEXITY_LABELS[c] for c in self.COMPLEXITY_ORDER])
            ax.set_ylim(0, 1.1)
            ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
            ax.legend(title="Mode")
            ax.set_title(title)

        plt.tight_layout()

        if save:
            fig.savefig(self.config.output_dir / f"complexity_performance.{self.config.figure_format}")
        return fig

    def plot_domain_heatmap(self, save: bool = True) -> plt.Figure:
        """Heatmap of performance across domains and modes"""
        pivot = self.df.pivot_table(
            values="avg_accuracy",
            index="domain_label",
            columns="mode",
            aggfunc="mean"
        ).reindex(columns=self.MODE_ORDER)

        fig, ax = plt.subplots(figsize=(8, 10))

        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn",
                    vmin=0, vmax=1, center=0.5, ax=ax,
                    cbar_kws={"label": "Average Accuracy"})

        ax.set_xlabel("Intelligence Mode")
        ax.set_ylabel("Domain Category")
        ax.set_title("Performance Heatmap: Domain × Mode")

        plt.tight_layout()

        if save:
            fig.savefig(self.config.output_dir / f"domain_heatmap.{self.config.figure_format}")
        return fig

    def plot_model_comparison(self, save: bool = True) -> plt.Figure:
        """Compare different model configurations"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. Accuracy vs Model Config
        ax1 = axes[0, 0]
        model_order = self.df.groupby("model_config")["avg_accuracy"].mean().sort_values(ascending=False).index

        sns.boxplot(data=self.df, x="model_config", y="avg_accuracy",
                    order=model_order, ax=ax1, palette="viridis")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        ax1.set_xlabel("Model Configuration (Orchestrator / Agent)")
        ax1.set_ylabel("Average Accuracy")
        ax1.set_title("(a) Accuracy by Model Configuration")

        # 2. Cost vs Accuracy scatter
        ax2 = axes[0, 1]
        for mode in self.MODE_ORDER:
            subset = self.df[self.df["mode"] == mode]
            ax2.scatter(subset["total_cost"], subset["avg_accuracy"],
                        label=mode, color=self.colors["mode"][mode], alpha=0.6)
        ax2.set_xlabel("Total Cost ($)")
        ax2.set_ylabel("Average Accuracy")
        ax2.set_title("(b) Cost-Accuracy Trade-off")
        ax2.legend()

        # 3. Time vs Complexity
        ax3 = axes[1, 0]
        complexity_order = [self.COMPLEXITY_LABELS[c] for c in self.COMPLEXITY_ORDER]
        sns.violinplot(data=self.df, x="complexity_label", y="total_time",
                       order=complexity_order, hue="mode", hue_order=self.MODE_ORDER,
                       split=False, ax=ax3, palette=self.colors["mode"])
        ax3.set_xlabel("Task Complexity")
        ax3.set_ylabel("Execution Time (s)")
        ax3.set_title("(c) Execution Time by Complexity and Mode")
        ax3.legend(title="Mode")

        # 4. Token usage by mode
        ax4 = axes[1, 1]
        token_data = self.df.groupby("mode")[["orchestrator_tokens", "agent_tokens"]].mean()
        token_data = token_data.reindex(self.MODE_ORDER)

        x = np.arange(len(self.MODE_ORDER))
        width = 0.35
        ax4.bar(x - width / 2, token_data["orchestrator_tokens"], width,
                label="Orchestrator", color="#3498db")
        ax4.bar(x + width / 2, token_data["agent_tokens"], width,
                label="Agent", color="#e74c3c")
        ax4.set_xticks(x)
        ax4.set_xticklabels(self.MODE_ORDER)
        ax4.set_xlabel("Intelligence Mode")
        ax4.set_ylabel("Average Tokens")
        ax4.set_title("(d) Token Usage by Mode")
        ax4.legend()

        plt.tight_layout()

        if save:
            fig.savefig(self.config.output_dir / f"model_comparison.{self.config.figure_format}")
        return fig

    def plot_accuracy_breakdown(self, save: bool = True) -> plt.Figure:
        """Stacked bar showing accuracy breakdown by mode"""
        fig, ax = plt.subplots(figsize=(10, 6))

        metrics = ["tools_correct", "tools_missing", "tools_extra"]
        labels = ["Correct", "Missing", "Extra"]
        colors = ["#27ae60", "#e74c3c", "#f39c12"]

        # Calculate proportions
        mode_data = []
        for mode in self.MODE_ORDER:
            subset = self.df[self.df["mode"] == mode]
            total = subset["tools_expected"].sum()
            correct = total - subset["tools_missing"].sum()
            missing = subset["tools_missing"].sum()
            extra = subset["tools_extra"].sum()
            mode_data.append([correct / total if total > 0 else 0,
                              missing / total if total > 0 else 0,
                              extra / (total + extra) if (total + extra) > 0 else 0])

        mode_data = np.array(mode_data)
        x = np.arange(len(self.MODE_ORDER))

        bottom = np.zeros(len(self.MODE_ORDER))
        for i, (label, color) in enumerate(zip(labels, colors)):
            ax.bar(x, mode_data[:, i], bottom=bottom, label=label, color=color)
            bottom += mode_data[:, i]

        ax.set_xticks(x)
        ax.set_xticklabels(self.MODE_ORDER)
        ax.set_xlabel("Intelligence Mode")
        ax.set_ylabel("Proportion")
        ax.set_title("Tool Selection Breakdown by Mode")
        ax.legend(loc='upper right')
        ax.set_ylim(0, 1.5)

        if save:
            fig.savefig(self.config.output_dir / f"accuracy_breakdown.{self.config.figure_format}")
        return fig

    def plot_efficiency_frontier(self, save: bool = True) -> plt.Figure:
        """Pareto frontier: accuracy vs efficiency"""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Group by model config and mode
        grouped = self.df.groupby(["model_config", "mode"]).agg({
            "avg_accuracy": "mean",
            "total_time": "mean",
            "total_cost": "mean",
            "total_tokens": "mean"
        }).reset_index()

        for mode in self.MODE_ORDER:
            subset = grouped[grouped["mode"] == mode]
            scatter = ax.scatter(
                subset["total_time"],
                subset["avg_accuracy"],
                c=[self.colors["mode"][mode]] * len(subset),
                s=subset["total_tokens"] / 100,  # Size by tokens
                alpha=0.7,
                label=mode,
                edgecolors='white',
                linewidth=0.5
            )

            # Add Pareto frontier
            pareto_points = self._get_pareto_frontier(
                subset["total_time"].values,
                subset["avg_accuracy"].values,
                maximize_y=True
            )
            if len(pareto_points) > 1:
                ax.plot(pareto_points[:, 0], pareto_points[:, 1],
                        '--', color=self.colors["mode"][mode], alpha=0.5)

        ax.set_xlabel("Average Execution Time (s)")
        ax.set_ylabel("Average Accuracy")
        ax.set_title("Efficiency Frontier: Accuracy vs Time\n(Point size indicates token usage)")
        ax.legend(title="Mode")

        # Add size legend
        handles = [plt.scatter([], [], s=size / 100, c='gray', alpha=0.7,
                               label=f'{size} tokens')
                   for size in [1000, 5000, 10000]]
        legend2 = ax.legend(handles=handles, title="Token Usage",
                            loc='lower right', framealpha=0.9)
        ax.add_artist(ax.legend(title="Mode", loc='upper left'))

        if save:
            fig.savefig(self.config.output_dir / f"efficiency_frontier.{self.config.figure_format}")
        return fig

    def _get_pareto_frontier(self, x: np.ndarray, y: np.ndarray,
                             maximize_y: bool = True) -> np.ndarray:
        """Calculate Pareto frontier points"""
        sorted_indices = np.argsort(x)
        x_sorted = x[sorted_indices]
        y_sorted = y[sorted_indices]

        pareto_points = []
        if maximize_y:
            max_y = -np.inf
            for xi, yi in zip(x_sorted, y_sorted):
                if yi > max_y:
                    pareto_points.append([xi, yi])
                    max_y = yi
        else:
            min_y = np.inf
            for xi, yi in zip(x_sorted, y_sorted):
                if yi < min_y:
                    pareto_points.append([xi, yi])
                    min_y = yi

        return np.array(pareto_points) if pareto_points else np.array([[0, 0]])

    def plot_scaling_analysis(self, save: bool = True) -> plt.Figure:
        """How performance scales with task complexity"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        complexity_num = {c: i for i, c in enumerate(self.COMPLEXITY_ORDER)}
        self.df["complexity_num"] = self.df["complexity"].map(complexity_num)

        metrics = [
            ("avg_accuracy", "Average Accuracy", axes[0, 0]),
            ("total_time", "Execution Time (s)", axes[0, 1]),
            ("total_tokens", "Total Tokens", axes[1, 0]),
            ("tool_calls", "Tool Calls", axes[1, 1])
        ]

        for metric, ylabel, ax in metrics:
            for mode in self.MODE_ORDER:
                subset = self.df[self.df["mode"] == mode]
                means = subset.groupby("complexity_num")[metric].mean()
                stds = subset.groupby("complexity_num")[metric].std()

                ax.errorbar(means.index, means.values, yerr=stds.values,
                            marker='o', label=mode, color=self.colors["mode"][mode],
                            capsize=4, linewidth=2, markersize=8)

            ax.set_xticks(range(len(self.COMPLEXITY_ORDER)))
            ax.set_xticklabels([self.COMPLEXITY_LABELS[c] for c in self.COMPLEXITY_ORDER])
            ax.set_xlabel("Task Complexity")
            ax.set_ylabel(ylabel)
            ax.legend(title="Mode")
            ax.grid(True, alpha=0.3)

        fig.suptitle("Performance Scaling with Task Complexity", fontsize=14)
        plt.tight_layout()

        if save:
            fig.savefig(self.config.output_dir / f"scaling_analysis.{self.config.figure_format}")
        return fig

    def plot_failure_analysis(self, save: bool = True) -> plt.Figure:
        """Analyze failure patterns"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 1. Failure rate by mode
        ax1 = axes[0]
        failure_rates = self.df.groupby("mode")["success"].apply(
            lambda x: 1 - x.mean()
        ).reindex(self.MODE_ORDER)

        bars = ax1.bar(self.MODE_ORDER, failure_rates.values,
                       color=[self.colors["mode"][m] for m in self.MODE_ORDER])
        ax1.set_xlabel("Intelligence Mode")
        ax1.set_ylabel("Failure Rate")
        ax1.set_title("(a) Failure Rate by Mode")
        ax1.set_ylim(0, 1)

        for bar, rate in zip(bars, failure_rates.values):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f'{rate:.1%}', ha='center', va='bottom')

        # 2. Failure by complexity
        ax2 = axes[1]
        failure_by_complexity = self.df.groupby(["complexity", "mode"])["success"].apply(
            lambda x: 1 - x.mean()
        ).unstack(fill_value=0)
        failure_by_complexity = failure_by_complexity.reindex(
            self.COMPLEXITY_ORDER, columns=self.MODE_ORDER
        )

        failure_by_complexity.plot(kind="bar", ax=ax2,
                                   color=[self.colors["mode"][m] for m in self.MODE_ORDER])
        ax2.set_xticklabels([self.COMPLEXITY_LABELS[c] for c in self.COMPLEXITY_ORDER],
                            rotation=0)
        ax2.set_xlabel("Task Complexity")
        ax2.set_ylabel("Failure Rate")
        ax2.set_title("(b) Failure Rate by Complexity")
        ax2.legend(title="Mode")

        # 3. Error type breakdown
        ax3 = axes[2]
        failed = self.df[~self.df["success"]]

        # Categorize failures
        def categorize_failure(row):
            if row["tools_missing"] > 0:
                return "Missing Tools"
            elif row["tools_extra"] > 0:
                return "Extra Tools"
            elif row["plan_accuracy"] < 0.5:
                return "Plan Error"
            else:
                return "Other"

        if len(failed) > 0:
            failed = failed.copy()
            failed["failure_type"] = failed.apply(categorize_failure, axis=1)
            failure_counts = failed["failure_type"].value_counts()

            colors_failure = ["#e74c3c", "#f39c12", "#9b59b6", "#7f8c8d"]
            ax3.pie(failure_counts.values, labels=failure_counts.index,
                    autopct='%1.1f%%', colors=colors_failure[:len(failure_counts)])
        else:
            ax3.text(0.5, 0.5, "No Failures", ha='center', va='center', fontsize=14)
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)

        ax3.set_title("(c) Failure Type Distribution")

        plt.tight_layout()

        if save:
            fig.savefig(self.config.output_dir / f"failure_analysis.{self.config.figure_format}")
        return fig

    def plot_comprehensive_dashboard(self, save: bool = True) -> plt.Figure:
        """Create a comprehensive dashboard for paper"""
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

        # Row 1: Mode comparison
        ax1 = fig.add_subplot(gs[0, :2])  # Accuracy by mode
        mode_means = self.df.groupby("mode")[["tool_accuracy", "agent_accuracy", "plan_accuracy"]].mean()
        mode_means = mode_means.reindex(self.MODE_ORDER)
        x = np.arange(len(self.MODE_ORDER))
        width = 0.25

        for i, (col, label) in enumerate([("tool_accuracy", "Tool"),
                                          ("agent_accuracy", "Agent"),
                                          ("plan_accuracy", "Plan")]):
            ax1.bar(x + (i - 1) * width, mode_means[col], width, label=label)

        ax1.set_xticks(x)
        ax1.set_xticklabels(self.MODE_ORDER)
        ax1.set_ylabel("Accuracy")
        ax1.set_title("(a) Accuracy Metrics by Mode")
        ax1.legend()
        ax1.set_ylim(0, 1.1)

        # Cost and time
        ax2 = fig.add_subplot(gs[0, 2:])
        ax2_twin = ax2.twinx()

        mode_metrics = self.df.groupby("mode")[["total_cost", "total_time"]].mean()
        mode_metrics = mode_metrics.reindex(self.MODE_ORDER)

        bars1 = ax2.bar(x - 0.2, mode_metrics["total_cost"] * 1000, 0.4,
                        label="Cost (×10⁻³ $)", color="#3498db")
        bars2 = ax2_twin.bar(x + 0.2, mode_metrics["total_time"], 0.4,
                             label="Time (s)", color="#e74c3c")

        ax2.set_xticks(x)
        ax2.set_xticklabels(self.MODE_ORDER)
        ax2.set_ylabel("Cost (×10⁻³ $)", color="#3498db")
        ax2_twin.set_ylabel("Time (s)", color="#e74c3c")
        ax2.set_title("(b) Cost and Time by Mode")

        # Row 2: Complexity analysis
        ax3 = fig.add_subplot(gs[1, :2])
        pivot = self.df.pivot_table(values="avg_accuracy",
                                    index="complexity", columns="mode",
                                    aggfunc="mean")
        pivot = pivot.reindex(self.COMPLEXITY_ORDER, columns=self.MODE_ORDER)

        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn",
                    vmin=0, vmax=1, ax=ax3, cbar_kws={"shrink": 0.8})
        ax3.set_yticklabels([self.COMPLEXITY_LABELS[c] for c in self.COMPLEXITY_ORDER], rotation=0)
        ax3.set_title("(c) Accuracy: Complexity × Mode")

        # Domain analysis
        ax4 = fig.add_subplot(gs[1, 2:])
        domain_acc = self.df.groupby("domain_label")["avg_accuracy"].mean().sort_values(ascending=True)

        colors_domain = plt.cm.viridis(np.linspace(0.2, 0.8, len(domain_acc)))
        ax4.barh(range(len(domain_acc)), domain_acc.values, color=colors_domain)
        ax4.set_yticks(range(len(domain_acc)))
        ax4.set_yticklabels(domain_acc.index)
        ax4.set_xlabel("Average Accuracy")
        ax4.set_title("(d) Accuracy by Domain")
        ax4.axvline(x=domain_acc.mean(), color='red', linestyle='--', label='Mean')

        # Row 3: Model comparison
        ax5 = fig.add_subplot(gs[2, :2])
        model_perf = self.df.groupby("model_config")["avg_accuracy"].mean().sort_values(ascending=False)
        top_models = model_perf.head(7)

        bars = ax5.barh(range(len(top_models)), top_models.values, color="#27ae60")
        ax5.set_yticks(range(len(top_models)))
        ax5.set_yticklabels(top_models.index)
        ax5.set_xlabel("Average Accuracy")
        ax5.set_title("(e) Top Model Configurations")
        ax5.invert_yaxis()

        # Success rate by model tier
        ax6 = fig.add_subplot(gs[2, 2:])
        tier_success = self.df.groupby(["model_tier", "mode"])["success"].mean().unstack()
        tier_success = tier_success.reindex(["High", "Medium", "Low"], columns=self.MODE_ORDER)

        tier_success.plot(kind="bar", ax=ax6,
                          color=[self.colors["mode"][m] for m in self.MODE_ORDER])
        ax6.set_xlabel("Model Tier")
        ax6.set_ylabel("Success Rate")
        ax6.set_title("(f) Success Rate by Model Tier")
        ax6.set_xticklabels(ax6.get_xticklabels(), rotation=0)
        ax6.legend(title="Mode")
        ax6.set_ylim(0, 1.1)

        fig.suptitle("Multi-Agent System Benchmark Analysis", fontsize=16, y=1.02)

        if save:
            fig.savefig(self.config.output_dir / f"comprehensive_dashboard.{self.config.figure_format}",
                        bbox_inches='tight')
        return fig

    def generate_latex_tables(self) -> Dict[str, str]:
        """Generate LaTeX tables for paper"""
        tables = {}

        # Table 1: Summary by mode
        summary = self.df.groupby("mode").agg({
            "success": ["count", "mean"],
            "tool_accuracy": ["mean", "std"],
            "agent_accuracy": ["mean", "std"],
            "plan_accuracy": ["mean", "std"],
            "total_time": ["mean", "std"],
            "total_cost": ["mean", "std"]
        }).round(3)

        latex = summary.to_latex(multicolumn=True, multirow=True)
        tables["summary_by_mode"] = latex

        # Table 2: Performance by complexity
        complexity_table = self.df.pivot_table(
            values=["avg_accuracy", "total_time", "total_tokens"],
            index="complexity",
            columns="mode",
            aggfunc="mean"
        ).round(3)

        tables["complexity_comparison"] = complexity_table.to_latex(multicolumn=True)

        # Table 3: Statistical tests
        stats_results = self.statistical_tests()
        stat_rows = []
        for metric, results in stats_results.items():
            kw = results["kruskal_wallis"]
            row = {
                "Metric": metric.replace("_", " ").title(),
                "H-statistic": f"{kw['H']:.2f}",
                "p-value": f"{kw['p']:.4f}",
                "Significant": "Yes" if kw['p'] < 0.05 else "No"
            }
            stat_rows.append(row)

        stat_df = pd.DataFrame(stat_rows)
        tables["statistical_tests"] = stat_df.to_latex(index=False)

        return tables

    def generate_all_figures(self):
        """Generate all publication figures"""
        print("Generating figures...")

        figures = {
            "radar": self.plot_mode_comparison_radar(),
            "complexity": self.plot_complexity_performance(),
            "heatmap": self.plot_domain_heatmap(),
            "model_comparison": self.plot_model_comparison(),
            "accuracy_breakdown": self.plot_accuracy_breakdown(),
            "efficiency": self.plot_efficiency_frontier(),
            "scaling": self.plot_scaling_analysis(),
            "failure": self.plot_failure_analysis(),
            "dashboard": self.plot_comprehensive_dashboard()
        }

        plt.close('all')
        print(f"Saved {len(figures)} figures to {self.config.output_dir}")

        return figures

    def generate_paper_report(self) -> str:
        """Generate complete analysis report for paper"""
        report = []
        report.append("# Benchmark Analysis Report\n")
        report.append(f"**Generated**: {pd.Timestamp.now()}\n")
        report.append(f"**Total Tests**: {len(self.df)}\n\n")

        # Summary statistics
        report.append("## Summary Statistics\n")
        summary = self.get_summary_statistics()
        report.append(summary.to_markdown())
        report.append("\n\n")

        # Key findings
        report.append("## Key Findings\n\n")

        # Best mode
        mode_acc = self.df.groupby("mode")["avg_accuracy"].mean()
        best_mode = mode_acc.idxmax()
        report.append(f"- **Best Overall Mode**: {best_mode} (Avg Accuracy: {mode_acc[best_mode]:.1%})\n")

        # Best model config
        model_acc = self.df.groupby("model_config")["avg_accuracy"].mean()
        best_model = model_acc.idxmax()
        report.append(f"- **Best Model Config**: {best_model} (Acc: {model_acc[best_model]:.1%})\n")

        # Complexity impact
        complexity_acc = self.df.groupby("complexity")["avg_accuracy"].mean()
        report.append(
            f"- **Easiest Task Type**: {self.COMPLEXITY_LABELS.get(complexity_acc.idxmax(), complexity_acc.idxmax())}\n")
        report.append(
            f"- **Hardest Task Type**: {self.COMPLEXITY_LABELS.get(complexity_acc.idxmin(), complexity_acc.idxmin())}\n")

        # Statistical significance
        report.append("\n## Statistical Analysis\n\n")
        stats = self.statistical_tests()
        for metric, results in stats.items():
            kw = results["kruskal_wallis"]
            sig = "✓" if kw['p'] < 0.05 else "✗"
            report.append(f"- **{metric}**: H={kw['H']:.2f}, p={kw['p']:.4f} {sig}\n")

        report_text = "".join(report)

        # Save report
        with open(self.config.output_dir / "analysis_report.md", 'w') as f:
            f.write(report_text)

        return report_text


def main():
    """Run complete analysis"""
    config = AnalysisConfig(
        results_path=Path("benchmark_results/old_results/results_20260108_064758.json"),
        output_dir=Path("benchmark_results/figures"),
        figure_format="pdf"
    )

    analyzer = BenchmarkAnalyzer(config)

    # Generate all outputs
    analyzer.generate_all_figures()
    tables = analyzer.generate_latex_tables()
    report = analyzer.generate_paper_report()

    print("\n" + "=" * 50)
    print("Analysis Complete!")
    print("=" * 50)
    print(f"\nFigures saved to: {config.output_dir}")
    print(f"Report saved to: {config.output_dir / 'analysis_report.md'}")

    # Print key statistics
    print("\n" + analyzer.generate_paper_report())


if __name__ == "__main__":
    main()