"""
Enhanced Benchmark Analysis with Multi-Run Support and Model Size Visualizations
"""
import json
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import warnings
import re

warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns

from scipy import stats
from scipy.stats import mannwhitneyu, kruskal

# Publication style
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
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


@dataclass
class AnalysisConfig:
    """Configuration for analysis"""
    results_paths: List[Path]  # Multiple JSON files
    db_paths: List[Path] = None  # Optional: SQLite databases
    output_dir: Path = Path("benchmark_results/figures")
    figure_format: str = "pdf"
    significance_level: float = 0.05


class ModelSizeMapper:
    """Map model names to approximate sizes for visualization"""

    # Model size estimates (in billions of parameters)
    MODEL_SIZES = {
        # OpenAI
        "gpt-4o": 200,  # Estimated
        "gpt-4o-mini": 8,  # Estimated
        "gpt-4": 170,
        "gpt-3.5-turbo": 20,

        # Ollama/Open source
        "qwen3:1.7b": 1.7,
        "qwen3:4b": 4,
        "qwen3:8b": 8,
        "qwen3:14b": 14,
        "qwen3:30b": 30,
        "qwen3:72b": 72,
        "gemma2:2b": 2,
        "gemma2:9b": 9,
        "gemma2:27b": 27,
        "llama3.2:1b": 1,
        "llama3.2:3b": 3,
        "llama3:8b": 8,
        "llama3:70b": 70,
        "mistral:7b": 7,
        "mixtral:8x7b": 47,  # MoE effective
    }

    # Size categories for grouping
    SIZE_CATEGORIES = {
        "XS": (0, 2),
        "S": (2, 5),
        "M": (5, 12),
        "L": (12, 35),
        "XL": (35, 100),
        "XXL": (100, 500),
    }

    @classmethod
    def get_size(cls, model_name: str) -> float:
        """Get model size in billions"""
        model_lower = model_name.lower()

        # Direct match
        if model_lower in cls.MODEL_SIZES:
            return cls.MODEL_SIZES[model_lower]

        # Try to extract size from name
        size_match = re.search(r'(\d+\.?\d*)b', model_lower)
        if size_match:
            return float(size_match.group(1))

        # Check partial matches
        for key, size in cls.MODEL_SIZES.items():
            if key in model_lower or model_lower in key:
                return size

        return 10.0  # Default fallback

    @classmethod
    def get_category(cls, model_name: str) -> str:
        """Get size category"""
        size = cls.get_size(model_name)
        for cat, (low, high) in cls.SIZE_CATEGORIES.items():
            if low <= size < high:
                return cat
        return "XXL"

    @classmethod
    def get_size_label(cls, model_name: str) -> str:
        """Get readable size label"""
        size = cls.get_size(model_name)
        if size >= 100:
            return f"{size:.0f}B"
        elif size >= 10:
            return f"{size:.0f}B"
        else:
            return f"{size:.1f}B"


class MultiRunBenchmarkAnalyzer:
    """Analyzer that combines multiple benchmark runs"""

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
        "C-1": "Centralized (1-Stage)",
        "C-2": "Centralized (2-Stage)",
        "D": "Decentralized"
    }

    DOMAIN_CATEGORIES = {
        "CFG": "Configuration", "BLD": "Building", "HVAC": "HVAC",
        "DER": "DER", "CTRL": "Controller", "DIST": "Disturbance",
        "ENV": "Environment", "SIM": "Simulation", "ANL": "Analysis",
        "CMP": "Comparison", "COM": "Communication"
    }

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.size_mapper = ModelSizeMapper()

        # Load and combine all results
        self.df = self._load_all_results()
        self._preprocess_data()

        # Color schemes
        self.colors = {
            "mode": {"C-1": "#2ecc71", "C-2": "#3498db", "D": "#e74c3c"},
            "complexity": sns.color_palette("viridis", 4),
        }

    def _load_all_results(self) -> pd.DataFrame:
        """Load and combine results from multiple sources"""
        all_records = []

        # Load from JSON files
        for json_path in self.config.results_paths:
            if json_path.exists():
                print(f"Loading: {json_path}")
                records = self._load_json_results(json_path)
                all_records.extend(records)

        # Optionally load from SQLite databases
        if self.config.db_paths:
            for db_path in self.config.db_paths:
                if db_path.exists():
                    print(f"Loading DB: {db_path}")
                    records = self._load_db_results(db_path)
                    all_records.extend(records)

        df = pd.DataFrame(all_records)

        # Remove duplicates based on unique test configuration
        if len(df) > 0:
            df = df.drop_duplicates(subset=[
                'test_id', 'mode', 'orchestrator_model',
                'agent_model', 'repeat_index'
            ], keep='last')

        print(f"Total records loaded: {len(df)}")
        return df

    def _load_json_results(self, path: Path) -> List[Dict]:
        """Load results from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)

        records = []
        for item in data:
            config = item.get("config", {})
            accuracy = item.get("accuracy_metrics", {})

            record = {
                "test_id": config.get("test_case_id", ""),
                "test_name": config.get("test_case_name", ""),
                "mode": self._get_mode_label(config),
                "orchestrator_model": config.get("orchestrator_model", ""),
                "agent_model": config.get("agent_model", ""),
                "provider": config.get("provider", ""),
                "two_stage": config.get("two_stage_planning", False),
                "repeat_index": config.get("repeat_index", 0),
                "complexity": item.get("category", ""),
                "domain": self._extract_domain(config.get("test_case_id", "")),
                "success": item.get("success", False),
                "tool_accuracy": accuracy.get("tool_selection_accuracy", 0),
                "agent_accuracy": accuracy.get("agent_selection_accuracy", 0),
                "plan_accuracy": accuracy.get("plan_step_accuracy", 0),
                "total_time": item.get("total_time", 0),
                "total_tokens": item.get("grand_total_tokens", 0),
                "orchestrator_tokens": item.get("orchestrator_total_tokens", 0),
                "agent_tokens": item.get("agent_total_tokens", 0),
                "total_cost": item.get("total_cost", 0),
                "tool_calls": item.get("tool_calls", 0),
                "tools_expected": len(accuracy.get("tools_expected", [])),
                "tools_actual": len(accuracy.get("tools_actual", [])),
                "tools_missing": len(accuracy.get("tools_missing", [])),
                "tools_extra": len(accuracy.get("tools_extra", [])),
                "source_file": str(path.name),
            }
            records.append(record)
        return records

    def _load_db_results(self, db_path: Path) -> List[Dict]:
        """Load results from SQLite checkpoint database"""
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        rows = conn.execute("""
            SELECT result_json FROM test_results WHERE status = 'completed'
        """).fetchall()

        records = []
        for row in rows:
            item = json.loads(row["result_json"])
            config = item.get("config", {})
            accuracy = item.get("accuracy_metrics", {})

            record = {
                "test_id": config.get("test_case_id", ""),
                "test_name": config.get("test_case_name", ""),
                "mode": self._get_mode_label(config),
                "orchestrator_model": config.get("orchestrator_model", ""),
                "agent_model": config.get("agent_model", ""),
                "provider": config.get("provider", ""),
                "two_stage": config.get("two_stage_planning", False),
                "repeat_index": config.get("repeat_index", 0),
                "complexity": item.get("category", ""),
                "domain": self._extract_domain(config.get("test_case_id", "")),
                "success": item.get("success", False),
                "tool_accuracy": accuracy.get("tool_selection_accuracy", 0),
                "agent_accuracy": accuracy.get("agent_selection_accuracy", 0),
                "plan_accuracy": accuracy.get("plan_step_accuracy", 0),
                "total_time": item.get("total_time", 0),
                "total_tokens": item.get("grand_total_tokens", 0),
                "orchestrator_tokens": item.get("orchestrator_total_tokens", 0),
                "agent_tokens": item.get("agent_total_tokens", 0),
                "total_cost": item.get("total_cost", 0),
                "tool_calls": item.get("tool_calls", 0),
                "tools_expected": len(accuracy.get("tools_expected", [])),
                "tools_actual": len(accuracy.get("tools_actual", [])),
                "tools_missing": len(accuracy.get("tools_missing", [])),
                "tools_extra": len(accuracy.get("tools_extra", [])),
                "source_file": str(db_path.name),
            }
            records.append(record)

        conn.close()
        return records

    def _get_mode_label(self, config: Dict) -> str:
        intel_mode = config.get("intelligence_mode", "").lower()
        two_stage = config.get("two_stage_planning", False)
        if intel_mode == "centralized":
            return "C-2" if two_stage else "C-1"
        return "D"

    def _extract_domain(self, test_id: str) -> str:
        parts = test_id.split("_")
        return parts[1] if len(parts) >= 2 else "UNKNOWN"

    def _preprocess_data(self):
        """Add derived columns"""
        if len(self.df) == 0:
            return

        # Composite accuracy
        self.df["avg_accuracy"] = (
                                          self.df["tool_accuracy"] +
                                          self.df["agent_accuracy"] +
                                          self.df["plan_accuracy"]
                                  ) / 3

        # Model sizes
        self.df["orch_size"] = self.df["orchestrator_model"].apply(
            self.size_mapper.get_size
        )
        self.df["agent_size"] = self.df["agent_model"].apply(
            self.size_mapper.get_size
        )
        self.df["orch_category"] = self.df["orchestrator_model"].apply(
            self.size_mapper.get_category
        )
        self.df["agent_category"] = self.df["agent_model"].apply(
            self.size_mapper.get_category
        )

        # Model config label
        self.df["model_config"] = (
                self.df["orchestrator_model"].str.replace(":", "-") + " / " +
                self.df["agent_model"].str.replace(":", "-")
        )

        # Short model config
        self.df["model_config_short"] = (
                self.df["orch_category"] + "->" + self.df["agent_category"]
        )

        # Labels
        self.df["complexity_label"] = self.df["complexity"].map(self.COMPLEXITY_LABELS)
        self.df["domain_label"] = self.df["domain"].map(self.DOMAIN_CATEGORIES)

    # ==================== MODEL SIZE VISUALIZATIONS ====================

    def plot_model_size_heatmap(self, metric: str = "avg_accuracy",
                                save: bool = True) -> plt.Figure:
        """
        Heatmap with orchestrator size on Y-axis, agent size on X-axis
        Color intensity shows performance
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for idx, mode in enumerate(self.MODE_ORDER):
            ax = axes[idx]
            mode_data = self.df[self.df["mode"] == mode]

            if len(mode_data) == 0:
                ax.set_title(f"{mode} (No Data)")
                continue

            # Create pivot table
            pivot = mode_data.pivot_table(
                values=metric,
                index="orchestrator_model",
                columns="agent_model",
                aggfunc="mean"
            )

            # Sort by model size
            orch_order = sorted(pivot.index,
                                key=lambda x: self.size_mapper.get_size(x))
            agent_order = sorted(pivot.columns,
                                 key=lambda x: self.size_mapper.get_size(x))

            pivot = pivot.reindex(index=orch_order, columns=agent_order)

            # Create heatmap
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn",
                        vmin=0, vmax=1, ax=ax, cbar_kws={"shrink": 0.8})

            # Add size labels
            orch_labels = [f"{m}\n({self.size_mapper.get_size_label(m)})"
                           for m in orch_order]
            agent_labels = [f"{m}\n({self.size_mapper.get_size_label(m)})"
                            for m in agent_order]

            ax.set_yticklabels(orch_labels, rotation=0, fontsize=8)
            ax.set_xticklabels(agent_labels, rotation=45, ha='right', fontsize=8)
            ax.set_xlabel("Agent Model (Size)")
            ax.set_ylabel("Orchestrator Model (Size)")
            ax.set_title(f"{self.MODE_LABELS[mode]}")

        metric_label = metric.replace("_", " ").title()
        fig.suptitle(f"Model Size Combination: {metric_label}", fontsize=14)
        plt.tight_layout()

        if save:
            fig.savefig(self.config.output_dir /
                        f"model_size_heatmap_{metric}.{self.config.figure_format}")
        return fig

    def plot_model_size_scatter(self, save: bool = True) -> plt.Figure:
        """
        Scatter plot: X=Orchestrator size, Y=Agent size
        Color=Performance, Size=Token usage
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Aggregate by model combination and mode
        grouped = self.df.groupby(
            ["orchestrator_model", "agent_model", "mode"]
        ).agg({
            "avg_accuracy": "mean",
            "total_tokens": "mean",
            "total_time": "mean",
            "orch_size": "first",
            "agent_size": "first",
        }).reset_index()

        for idx, mode in enumerate(self.MODE_ORDER):
            ax = axes[idx]
            mode_data = grouped[grouped["mode"] == mode]

            if len(mode_data) == 0:
                ax.set_title(f"{mode} (No Data)")
                continue

            scatter = ax.scatter(
                mode_data["orch_size"],
                mode_data["agent_size"],
                c=mode_data["avg_accuracy"],
                s=mode_data["total_tokens"] / 50 + 50,
                cmap="RdYlGn",
                vmin=0, vmax=1,
                alpha=0.7,
                edgecolors='black',
                linewidth=0.5
            )

            # Add model labels
            for _, row in mode_data.iterrows():
                ax.annotate(
                    f"{row['orchestrator_model'].split(':')[0]}/\n{row['agent_model'].split(':')[0]}",
                    (row["orch_size"], row["agent_size"]),
                    fontsize=6, alpha=0.7,
                    xytext=(5, 5), textcoords='offset points'
                )

            ax.set_xlabel("Orchestrator Size (B params)")
            ax.set_ylabel("Agent Size (B params)")
            ax.set_title(f"{self.MODE_LABELS[mode]}")
            ax.set_xscale('log')
            ax.set_yscale('log')

            # Add diagonal line (equal size)
            lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                    max(ax.get_xlim()[1], ax.get_ylim()[1])]
            ax.plot(lims, lims, 'k--', alpha=0.3, label='Equal size')
            ax.set_xlim(lims)
            ax.set_ylim(lims)

        # Shared colorbar
        cbar = fig.colorbar(scatter, ax=axes, shrink=0.8, label="Avg Accuracy")

        fig.suptitle("Model Size Combinations\n(Point size = Token usage)", fontsize=14)
        plt.tight_layout()

        if save:
            fig.savefig(self.config.output_dir /
                        f"model_size_scatter.{self.config.figure_format}")
        return fig

    def plot_model_size_matrix(self, save: bool = True) -> plt.Figure:
        """
        Matrix plot using categorical size bins
        Shows clear patterns of large-orch/small-agent vs small-orch/large-agent
        """
        size_order = ["XS", "S", "M", "L", "XL", "XXL"]

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        metrics = [
            ("avg_accuracy", "Average Accuracy", axes[0, 0]),
            ("total_time", "Execution Time (s)", axes[0, 1]),
            ("total_tokens", "Total Tokens", axes[1, 0]),
            ("success", "Success Rate", axes[1, 1]),
        ]

        for metric, title, ax in metrics:
            pivot = self.df.pivot_table(
                values=metric,
                index="orch_category",
                columns="agent_category",
                aggfunc="mean"
            )

            # Reindex to ensure order
            valid_orch = [s for s in size_order if s in pivot.index]
            valid_agent = [s for s in size_order if s in pivot.columns]
            pivot = pivot.reindex(index=valid_orch, columns=valid_agent)

            # Choose colormap based on metric
            if metric in ["avg_accuracy", "success"]:
                cmap = "RdYlGn"
                vmin, vmax = 0, 1
            else:
                cmap = "YlOrRd"
                vmin, vmax = None, None

            sns.heatmap(pivot, annot=True, fmt=".2f", cmap=cmap,
                        vmin=vmin, vmax=vmax, ax=ax,
                        cbar_kws={"shrink": 0.8})

            ax.set_xlabel("Agent Size Category")
            ax.set_ylabel("Orchestrator Size Category")
            ax.set_title(title)

        fig.suptitle("Performance by Model Size Categories", fontsize=14)
        plt.tight_layout()

        if save:
            fig.savefig(self.config.output_dir /
                        f"model_size_matrix.{self.config.figure_format}")
        return fig

    # ==================== MODE COMPARISON WITH MODEL HUE ====================

    def plot_mode_by_model_config(self, save: bool = True) -> plt.Figure:
        """
        Compare modes with model configuration as hue
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Get top N model configs for clarity
        top_configs = (self.df.groupby("model_config_short")
                       .size()
                       .nlargest(6)
                       .index.tolist())

        plot_df = self.df[self.df["model_config_short"].isin(top_configs)]

        # 1. Accuracy by mode, grouped by model config
        ax1 = axes[0, 0]
        sns.barplot(data=plot_df, x="mode", y="avg_accuracy",
                    hue="model_config_short", ax=ax1,
                    order=self.MODE_ORDER, palette="husl",
                    errorbar="sd")
        ax1.set_xlabel("Intelligence Mode")
        ax1.set_ylabel("Average Accuracy")
        ax1.set_title("(a) Accuracy by Mode & Model Config")
        ax1.legend(title="Model Config", bbox_to_anchor=(1.02, 1))
        ax1.set_ylim(0, 1.1)

        # 2. Time by mode
        ax2 = axes[0, 1]
        sns.barplot(data=plot_df, x="mode", y="total_time",
                    hue="model_config_short", ax=ax2,
                    order=self.MODE_ORDER, palette="husl",
                    errorbar="sd")
        ax2.set_xlabel("Intelligence Mode")
        ax2.set_ylabel("Execution Time (s)")
        ax2.set_title("(b) Time by Mode & Model Config")
        ax2.legend(title="Model Config", bbox_to_anchor=(1.02, 1))

        # 3. Tokens by mode
        ax3 = axes[1, 0]
        sns.barplot(data=plot_df, x="mode", y="total_tokens",
                    hue="model_config_short", ax=ax3,
                    order=self.MODE_ORDER, palette="husl",
                    errorbar="sd")
        ax3.set_xlabel("Intelligence Mode")
        ax3.set_ylabel("Total Tokens")
        ax3.set_title("(c) Token Usage by Mode & Model Config")
        ax3.legend(title="Model Config", bbox_to_anchor=(1.02, 1))

        # 4. Success rate
        ax4 = axes[1, 1]
        success_data = plot_df.groupby(
            ["mode", "model_config_short"]
        )["success"].mean().reset_index()
        sns.barplot(data=success_data, x="mode", y="success",
                    hue="model_config_short", ax=ax4,
                    order=self.MODE_ORDER, palette="husl")
        ax4.set_xlabel("Intelligence Mode")
        ax4.set_ylabel("Success Rate")
        ax4.set_title("(d) Success Rate by Mode & Model Config")
        ax4.legend(title="Model Config", bbox_to_anchor=(1.02, 1))
        ax4.set_ylim(0, 1.1)

        plt.tight_layout()

        if save:
            fig.savefig(self.config.output_dir /
                        f"mode_by_model_config.{self.config.figure_format}")
        return fig

    def plot_complexity_by_model(self, save: bool = True) -> plt.Figure:
        """
        Complexity analysis with model config facets
        """
        top_configs = (self.df.groupby("model_config_short")
                       .size()
                       .nlargest(4)
                       .index.tolist())

        plot_df = self.df[self.df["model_config_short"].isin(top_configs)]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, config in enumerate(top_configs):
            ax = axes[idx]
            config_data = plot_df[plot_df["model_config_short"] == config]

            # Pivot for heatmap
            pivot = config_data.pivot_table(
                values="avg_accuracy",
                index="complexity_label",
                columns="mode",
                aggfunc="mean"
            )

            # Reorder
            complexity_order = [self.COMPLEXITY_LABELS[c]
                                for c in self.COMPLEXITY_ORDER
                                if self.COMPLEXITY_LABELS[c] in pivot.index]
            pivot = pivot.reindex(index=complexity_order,
                                  columns=self.MODE_ORDER)

            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn",
                        vmin=0, vmax=1, ax=ax)
            ax.set_title(f"Model Config: {config}")
            ax.set_xlabel("Mode")
            ax.set_ylabel("Complexity")

        fig.suptitle("Complexity × Mode Performance by Model Configuration",
                     fontsize=14)
        plt.tight_layout()

        if save:
            fig.savefig(self.config.output_dir /
                        f"complexity_by_model.{self.config.figure_format}")
        return fig

    def plot_domain_by_model(self, save: bool = True) -> plt.Figure:
        """
        Domain performance faceted by model configuration
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))

        for idx, mode in enumerate(self.MODE_ORDER):
            ax = axes[idx]
            mode_data = self.df[self.df["mode"] == mode]

            if len(mode_data) == 0:
                continue

            # Pivot: domain vs model config
            pivot = mode_data.pivot_table(
                values="avg_accuracy",
                index="domain_label",
                columns="model_config_short",
                aggfunc="mean"
            )

            # Keep only domains with enough data
            pivot = pivot.dropna(thresh=2)

            if len(pivot) > 0:
                sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn",
                            vmin=0, vmax=1, ax=ax,
                            cbar_kws={"shrink": 0.8})

            ax.set_title(f"{self.MODE_LABELS[mode]}")
            ax.set_xlabel("Model Configuration")
            ax.set_ylabel("Domain")

        fig.suptitle("Domain Performance: Mode × Model Configuration", fontsize=14)
        plt.tight_layout()

        if save:
            fig.savefig(self.config.output_dir /
                        f"domain_by_model.{self.config.figure_format}")
        return fig

    # ==================== COMPREHENSIVE DASHBOARD ====================

    def plot_comprehensive_model_dashboard(self, save: bool = True) -> plt.Figure:
        """
        Master dashboard showing model size effects
        """
        fig = plt.figure(figsize=(18, 14))
        gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35)

        # Row 1: Model size effects
        # 1a. Size category matrix
        ax1 = fig.add_subplot(gs[0, :2])
        pivot = self.df.pivot_table(
            values="avg_accuracy",
            index="orch_category",
            columns="agent_category",
            aggfunc="mean"
        )
        size_order = ["XS", "S", "M", "L", "XL"]
        valid_idx = [s for s in size_order if s in pivot.index]
        valid_col = [s for s in size_order if s in pivot.columns]
        pivot = pivot.reindex(index=valid_idx, columns=valid_col)

        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn",
                    vmin=0, vmax=1, ax=ax1)
        ax1.set_xlabel("Agent Size")
        ax1.set_ylabel("Orchestrator Size")
        ax1.set_title("(a) Accuracy by Model Size Categories")

        # 1b. Size ratio analysis
        ax2 = fig.add_subplot(gs[0, 2:])
        self.df["size_ratio"] = self.df["orch_size"] / self.df["agent_size"]
        self.df["size_ratio_cat"] = pd.cut(
            self.df["size_ratio"],
            bins=[0, 0.5, 1, 2, 100],
            labels=["Small→Large", "Equal", "Large→Small", "Much Larger"]
        )

        ratio_perf = self.df.groupby(["size_ratio_cat", "mode"]).agg({
            "avg_accuracy": "mean",
            "total_time": "mean"
        }).reset_index()

        sns.barplot(data=ratio_perf, x="size_ratio_cat", y="avg_accuracy",
                    hue="mode", hue_order=self.MODE_ORDER, ax=ax2,
                    palette=self.colors["mode"])
        ax2.set_xlabel("Orchestrator/Agent Size Ratio")
        ax2.set_ylabel("Average Accuracy")
        ax2.set_title("(b) Performance by Size Ratio")
        ax2.legend(title="Mode")

        # Row 2: Mode comparisons with model grouping
        ax3 = fig.add_subplot(gs[1, :2])
        top_configs = self.df.groupby("model_config_short").size().nlargest(5).index
        plot_df = self.df[self.df["model_config_short"].isin(top_configs)]

        sns.boxplot(data=plot_df, x="mode", y="avg_accuracy",
                    hue="model_config_short", ax=ax3,
                    order=self.MODE_ORDER, palette="husl")
        ax3.set_xlabel("Intelligence Mode")
        ax3.set_ylabel("Average Accuracy")
        ax3.set_title("(c) Mode Performance by Model Config")
        ax3.legend(title="Config", bbox_to_anchor=(1.02, 1), fontsize=8)

        # Complexity by model
        ax4 = fig.add_subplot(gs[1, 2:])
        complexity_model = self.df.groupby(
            ["complexity_label", "model_config_short"]
        )["avg_accuracy"].mean().reset_index()
        complexity_model = complexity_model[
            complexity_model["model_config_short"].isin(top_configs)
        ]

        sns.barplot(data=complexity_model, x="complexity_label", y="avg_accuracy",
                    hue="model_config_short", ax=ax4, palette="husl")
        ax4.set_xlabel("Task Complexity")
        ax4.set_ylabel("Average Accuracy")
        ax4.set_title("(d) Complexity Performance by Model Config")
        ax4.legend(title="Config", bbox_to_anchor=(1.02, 1), fontsize=8)

        # Row 3: Summary stats
        ax5 = fig.add_subplot(gs[2, :2])
        model_summary = self.df.groupby("model_config").agg({
            "avg_accuracy": "mean",
            "total_time": "mean",
            "total_tokens": "mean",
            "success": "mean",
            "orch_size": "first",
            "agent_size": "first"
        }).reset_index()
        model_summary = model_summary.nlargest(10, "avg_accuracy")

        y_pos = range(len(model_summary))
        ax5.barh(y_pos, model_summary["avg_accuracy"], color="#27ae60")
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(model_summary["model_config"], fontsize=8)
        ax5.set_xlabel("Average Accuracy")
        ax5.set_title("(e) Top 10 Model Configurations")
        ax5.invert_yaxis()

        # Efficiency scatter
        ax6 = fig.add_subplot(gs[2, 2:])
        for mode in self.MODE_ORDER:
            mode_data = self.df[self.df["mode"] == mode]
            grouped = mode_data.groupby("model_config_short").agg({
                "avg_accuracy": "mean",
                "total_time": "mean"
            }).reset_index()

            ax6.scatter(grouped["total_time"], grouped["avg_accuracy"],
                        label=mode, color=self.colors["mode"][mode],
                        s=100, alpha=0.7)

        ax6.set_xlabel("Average Time (s)")
        ax6.set_ylabel("Average Accuracy")
        ax6.set_title("(f) Efficiency: Accuracy vs Time by Mode")
        ax6.legend()

        fig.suptitle("Multi-Agent Benchmark: Model Size & Configuration Analysis",
                     fontsize=16, y=1.02)

        if save:
            fig.savefig(self.config.output_dir /
                        f"comprehensive_model_dashboard.{self.config.figure_format}",
                        bbox_inches='tight')
        return fig

    def generate_all_figures(self):
        """Generate all publication figures"""
        print("Generating figures...")

        figures = {}

        # Model size visualizations
        figures["size_heatmap_acc"] = self.plot_model_size_heatmap("avg_accuracy")
        figures["size_heatmap_time"] = self.plot_model_size_heatmap("total_time")
        figures["size_scatter"] = self.plot_model_size_scatter()
        figures["size_matrix"] = self.plot_model_size_matrix()

        # Mode comparisons with model hue
        figures["mode_by_model"] = self.plot_mode_by_model_config()
        figures["complexity_by_model"] = self.plot_complexity_by_model()
        figures["domain_by_model"] = self.plot_domain_by_model()

        # Comprehensive dashboard
        figures["model_dashboard"] = self.plot_comprehensive_model_dashboard()

        plt.close('all')
        print(f"Saved {len(figures)} figures to {self.config.output_dir}")

        return figures

    def generate_summary_report(self) -> str:
        """Generate analysis summary"""
        report = []
        report.append("# Multi-Run Benchmark Analysis\n")
        report.append(f"**Total Records**: {len(self.df)}\n")
        report.append(f"**Source Files**: {self.df['source_file'].nunique()}\n")
        report.append(f"**Model Configurations**: {self.df['model_config'].nunique()}\n\n")

        # Best configurations
        report.append("## Top Model Configurations\n")
        top = self.df.groupby("model_config")["avg_accuracy"].mean().nlargest(5)
        for config, acc in top.items():
            report.append(f"- {config}: {acc:.1%}\n")

        # Size analysis
        report.append("\n## Model Size Insights\n")
        size_pivot = self.df.pivot_table(
            values="avg_accuracy",
            index="orch_category",
            columns="agent_category",
            aggfunc="mean"
        )
        best_combo = size_pivot.stack().idxmax()
        report.append(f"- Best size combination: Orch={best_combo[0]}, Agent={best_combo[1]}\n")

        # Save report
        report_text = "".join(report)
        with open(self.config.output_dir / "multi_run_report.md", 'w') as f:
            f.write(report_text)

        return report_text


def main():
    """Run analysis on multiple result files"""
    # Configure paths - add all your result files here
    config = AnalysisConfig(
        results_paths=[
            Path("benchmark_results/results_benchmark_eva.json"),
            # Path("benchmark_results/old_results/results_20260108_195506.json"),
        ],
        db_paths=[
            Path("benchmark_results/benchmark_checkpoint.db"),
        ],
        output_dir=Path("benchmark_results/figures_combined"),
        figure_format="pdf"
    )

    analyzer = MultiRunBenchmarkAnalyzer(config)

    # Generate all outputs
    analyzer.generate_all_figures()
    report = analyzer.generate_summary_report()

    print("\n" + "=" * 50)
    print("Analysis Complete!")
    print("=" * 50)
    print(report)


if __name__ == "__main__":
    main()