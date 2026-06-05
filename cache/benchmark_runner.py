"""
Benchmarking Suite with Checkpointing, Progress Monitoring, and Rich Analysis
REVISED: Added plan vs execution comparison, agent_results persistence, and fixed accuracy extraction
"""

import sys
import os
import asyncio
import json
import sqlite3
import hashlib
import signal
import threading
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from itertools import product
from collections import defaultdict

# Suppress all logging before any other imports
def suppress_all_logging():
    """Completely disable all logging output"""
    logging.disable(logging.CRITICAL)
    noisy_loggers = [
        "", "root", "httpx", "httpcore", "http", "urllib3", "requests", "aiohttp",
        "openai", "anthropic", "ollama", "asyncio", "concurrent",
        "mcp", "mcp_client", "mcp_server",
        "src", "src.orchestrator", "src.agent", "src.llm", "src.core",
        "orchestrator", "Orchestrator", "agent", "Agent",
        "llm", "llm_interface", "BenchmarkSuite", "benchmark",
        "websockets", "anyio",
    ]
    for name in noisy_loggers:
        logger = logging.getLogger(name)
        logger.setLevel(logging.CRITICAL + 100)
        logger.disabled = True
        logger.propagate = False
        logger.handlers = []
        logger.addHandler(logging.NullHandler())

suppress_all_logging()

# Progress monitoring with rich
try:
    from rich.console import Console
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn, TextColumn,
        TimeElapsedColumn, TimeRemainingColumn, TaskID
    )
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from src.core.data_class import IntelligenceMode, AgentGenerationMode, FailureType
from src.orchestrator.orchestrator import Orchestrator
from src.llm.llm_interface import create_llm_client
from TESTCASE import TestCategory, BenchmarkTestCase, get_all_test_cases, get_test_cases_summary


class RunStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanVsExecutionMetrics:
    """Metrics comparing orchestrator's plan vs actual agent execution"""
    # What orchestrator planned (centralized mode only)
    planned_tools_by_step: List[Dict] = field(default_factory=list)  # [{step_id, agent_id, tools, parameters}]
    planned_tool_count: int = 0

    # What agents actually executed
    executed_tools_by_step: List[Dict] = field(default_factory=list)  # [{step_id, agent_id, tools, parameters}]
    executed_tool_count: int = 0

    # Comparison metrics
    plan_adherence_rate: float = 0.0  # % of planned tools that were executed
    tools_added_by_agents: List[str] = field(default_factory=list)  # Tools agents added beyond plan
    tools_removed_by_agents: List[str] = field(default_factory=list)  # Planned tools agents didn't execute
    tools_modified_parameters: List[Dict] = field(default_factory=list)  # Tools where params changed

    # Analysis flags
    agent_followed_plan: bool = True
    agent_added_tools: bool = False
    agent_removed_tools: bool = False
    agent_modified_params: bool = False


@dataclass
class AccuracyMetrics:
    """Detailed accuracy metrics"""
    # Tool selection accuracy
    tool_selection_accuracy: float = 0.0
    tools_expected: List[str] = field(default_factory=list)
    tools_actual: List[str] = field(default_factory=list)
    tools_correct: List[str] = field(default_factory=list)
    tools_missing: List[str] = field(default_factory=list)
    tools_extra: List[str] = field(default_factory=list)

    # Agent selection accuracy
    agent_selection_accuracy: float = 0.0
    agents_expected: List[str] = field(default_factory=list)
    agents_actual: List[str] = field(default_factory=list)
    agents_correct: List[str] = field(default_factory=list)
    agents_missing: List[str] = field(default_factory=list)
    agents_extra: List[str] = field(default_factory=list)

    # Plan step accuracy
    plan_step_accuracy: float = 0.0
    expected_step_count: int = 0
    matched_step_count: int = 0
    plan_sequence_details: List[Dict] = field(default_factory=list)
    expected_plan_sequence: List[Dict] = field(default_factory=list)
    actual_plan_sequence: List[Dict] = field(default_factory=list)

    # Parameter accuracy
    parameter_key_accuracy: float = 0.0
    parameter_value_accuracy: float = 0.0
    parameter_details: List[Dict] = field(default_factory=list)

    # NEW: Plan vs Execution comparison (centralized mode)
    plan_vs_execution: PlanVsExecutionMetrics = field(default_factory=PlanVsExecutionMetrics)

@dataclass
class RetryStats:
    """Retry statistics for a benchmark run"""
    # Step-level (orchestrator retry)
    total_steps: int = 0
    steps_retried: int = 0
    steps_retry_succeeded: int = 0       # Retried AND eventually succeeded
    steps_retry_failed: int = 0          # Retried but still failed
    step_retry_rate: float = 0.0         # % of steps that needed retry

    # Tool-level (agent retry)
    total_tool_calls: int = 0
    tool_calls_retried: int = 0
    tool_calls_retry_succeeded: int = 0
    tool_calls_retry_failed: int = 0
    tool_retry_rate: float = 0.0

    # Retry effectiveness
    retry_recovery_rate: float = 0.0     # Of retried items, % that recovered

@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run"""
    intelligence_mode: IntelligenceMode
    agent_generation_mode: AgentGenerationMode
    orchestrator_model: str
    agent_model: str
    test_case_id: str
    test_case_name: str
    run_id: str
    provider: str
    two_stage_planning: bool = False
    repeat_index: int = 0

    def get_mode_label(self) -> str:
        if self.intelligence_mode == IntelligenceMode.SINGLE_AGENT_REACT:
            return "ReAct"
        elif self.intelligence_mode == IntelligenceMode.CENTRALIZED:
            return "C-2" if self.two_stage_planning else "C-1"
        return "D"

    def get_unique_key(self) -> str:
        key_parts = [
            self.run_id,  # Include run_id to make each run independent
            self.test_case_id, self.intelligence_mode.value,
            self.orchestrator_model, self.agent_model,
            str(self.two_stage_planning), str(self.repeat_index)
        ]
        return hashlib.md5("_".join(key_parts).encode()).hexdigest()[:16]


@dataclass
class BenchmarkResult:
    """Comprehensive result of a single benchmark run"""
    config: BenchmarkConfig
    category: str
    test_category: str = ""

    # Timing metrics
    total_time: float = 0.0
    planning_time: float = 0.0
    execution_time: float = 0.0
    synthesis_time: float = 0.0

    # Token metrics
    orchestrator_prompt_tokens: int = 0
    orchestrator_completion_tokens: int = 0
    orchestrator_total_tokens: int = 0
    agent_total_tokens: int = 0
    grand_total_tokens: int = 0

    # Cost metrics
    orchestrator_cost: float = 0.0
    agent_cost: float = 0.0
    total_cost: float = 0.0

    # Execution details
    tool_calls: int = 0
    tools_used: List[str] = field(default_factory=list)
    agents_used: List[str] = field(default_factory=list)
    agent_metrics: Dict[str, Dict] = field(default_factory=dict)

    # Success and accuracy
    success: bool = False
    failure_type: str = "none"  # FailureType enum value
    retry_stats: RetryStats = field(default_factory=RetryStats)
    accuracy_metrics: AccuracyMetrics = field(default_factory=AccuracyMetrics)

    # Raw data for re-evaluation (NEW: added agent_results)
    execution_plan: List[Dict] = field(default_factory=list)
    agent_results: Dict[str, Any] = field(default_factory=dict)  # NEW: persist for re-evaluation

    # Response
    response_summary: str = ""
    error: str = None
    timestamp: str = ""
    status: RunStatus = RunStatus.PENDING

    def extract_test_category(self) -> str:
        parts = self.config.test_case_id.split("_")
        if len(parts) >= 2:
            return parts[1]
        return "UNKNOWN"


class CheckpointManager:
    """SQLite-based checkpoint manager for resumable benchmarks"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = None
        self._init_db()

    def _init_db(self):
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS benchmark_runs (
                run_id TEXT PRIMARY KEY, start_time TEXT, end_time TEXT,
                status TEXT, total_tests INTEGER, completed_tests INTEGER, config_json TEXT
            );
            CREATE TABLE IF NOT EXISTS test_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, config_key TEXT UNIQUE,
                test_case_id TEXT, test_case_name TEXT, category TEXT, test_category TEXT,
                status TEXT, result_json TEXT, created_at TEXT, updated_at TEXT,
                FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id)
            );
            CREATE INDEX IF NOT EXISTS idx_run_id ON test_results(run_id);
            CREATE INDEX IF NOT EXISTS idx_status ON test_results(status);
            CREATE INDEX IF NOT EXISTS idx_config_key ON test_results(config_key);
        """)
        self.conn.commit()

    def create_run(self, run_id: str, total_tests: int, config: Dict) -> None:
        self.conn.execute("""
            INSERT OR REPLACE INTO benchmark_runs 
            (run_id, start_time, status, total_tests, completed_tests, config_json)
            VALUES (?, ?, ?, ?, 0, ?)
        """, (run_id, datetime.now().isoformat(), "running", total_tests, json.dumps(config)))
        self.conn.commit()

    def get_run_status(self, run_id: str) -> Optional[Dict]:
        row = self.conn.execute(
            "SELECT * FROM benchmark_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return dict(row) if row else None

    def is_test_completed(self, config_key: str) -> bool:
        row = self.conn.execute(
            "SELECT status FROM test_results WHERE config_key = ?", (config_key,)
        ).fetchone()
        return row and row["status"] == RunStatus.COMPLETED.value

    def save_result(self, result: BenchmarkResult) -> None:
        config_key = result.config.get_unique_key()
        result_dict = self._result_to_dict(result)
        self.conn.execute("""
            INSERT OR REPLACE INTO test_results
            (run_id, config_key, test_case_id, test_case_name, category, 
             test_category, status, result_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.config.run_id, config_key, result.config.test_case_id,
            result.config.test_case_name, result.category, result.extract_test_category(),
            result.status.value, json.dumps(result_dict, default=str),
            datetime.now().isoformat(), datetime.now().isoformat()
        ))
        self.conn.execute("""
            UPDATE benchmark_runs SET completed_tests = (
                SELECT COUNT(*) FROM test_results WHERE run_id = ? AND status = ?
            ) WHERE run_id = ?
        """, (result.config.run_id, RunStatus.COMPLETED.value, result.config.run_id))
        self.conn.commit()

    def get_all_results(self, run_id: str) -> List[BenchmarkResult]:
        rows = self.conn.execute(
            "SELECT result_json FROM test_results WHERE run_id = ? AND status = ?",
            (run_id, RunStatus.COMPLETED.value)
        ).fetchall()
        results = []
        for row in rows:
            result_dict = json.loads(row["result_json"])
            results.append(self._dict_to_result(result_dict))
        return results

    def get_progress(self, run_id: str) -> Dict:
        row = self.conn.execute("""
            SELECT total_tests, completed_tests,
                (SELECT COUNT(*) FROM test_results WHERE run_id = ? AND status = ?) as failed
            FROM benchmark_runs WHERE run_id = ?
        """, (run_id, RunStatus.FAILED.value, run_id)).fetchone()
        if row:
            return {
                "total": row["total_tests"], "completed": row["completed_tests"],
                "failed": row["failed"], "remaining": row["total_tests"] - row["completed_tests"] - row["failed"]
            }
        return {"total": 0, "completed": 0, "failed": 0, "remaining": 0}

    def finalize_run(self, run_id: str, status: str = "completed"):
        self.conn.execute("""
            UPDATE benchmark_runs SET end_time = ?, status = ? WHERE run_id = ?
        """, (datetime.now().isoformat(), status, run_id))
        self.conn.commit()

    def _result_to_dict(self, result: BenchmarkResult) -> Dict:
        d = asdict(result)
        d["config"]["intelligence_mode"] = result.config.intelligence_mode.value
        d["config"]["agent_generation_mode"] = result.config.agent_generation_mode.value
        d["status"] = result.status.value
        return d

    def _dict_to_result(self, d: Dict) -> BenchmarkResult:
        config_dict = d.pop("config")
        accuracy_dict = d.pop("accuracy_metrics", {})

        # Handle nested PlanVsExecutionMetrics
        plan_vs_exec_dict = accuracy_dict.pop("plan_vs_execution", {})
        plan_vs_exec = PlanVsExecutionMetrics(**plan_vs_exec_dict) if plan_vs_exec_dict else PlanVsExecutionMetrics()

        config = BenchmarkConfig(
            intelligence_mode=IntelligenceMode(config_dict["intelligence_mode"]),
            agent_generation_mode=AgentGenerationMode(config_dict["agent_generation_mode"]),
            orchestrator_model=config_dict["orchestrator_model"],
            agent_model=config_dict["agent_model"],
            test_case_id=config_dict["test_case_id"],
            test_case_name=config_dict["test_case_name"],
            run_id=config_dict["run_id"],
            provider=config_dict["provider"],
            two_stage_planning=config_dict.get("two_stage_planning", False),
            repeat_index=config_dict.get("repeat_index", 0)
        )
        retry_dict = d.pop("retry_stats", {})
        retry_stats = RetryStats(**retry_dict) if retry_dict else RetryStats()
        accuracy = AccuracyMetrics(**accuracy_dict) if accuracy_dict else AccuracyMetrics()
        accuracy.plan_vs_execution = plan_vs_exec

        return BenchmarkResult(
            config=config, accuracy_metrics=accuracy,
            retry_stats=retry_stats,
            status=RunStatus(d.get("status", "completed")),
            **{k: v for k, v in d.items() if k not in ["status", "retry_stats"]}
        )

    def close(self):
        if self.conn:
            self.conn.close()


class ProgressMonitor:
    """Real-time progress monitoring with ETA"""

    def __init__(self, total: int, run_id: str):
        self.total = total
        self.run_id = run_id
        self.completed = 0
        self.failed = 0
        self.start_time = time.time()
        self.test_times: List[float] = []
        self.current_test = ""
        self.lock = threading.Lock()
        self._last_line_length = 0

        if RICH_AVAILABLE:
            self.console = Console(stderr=True, force_terminal=True)
            self.progress = Progress(
                SpinnerColumn(spinner_name="dots"),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40, complete_style="green"),
                TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
                TextColumn("•"),
                TextColumn("[green]✓{task.fields[passed]}[/green]"),
                TextColumn("[red]✗{task.fields[failed]}[/red]"),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("→"),
                TimeRemainingColumn(),
                console=self.console,
                refresh_per_second=4,
                transient=False,
            )
            self.task_id = None
        else:
            self.console = None
            self.progress = None
            self.task_id = None

    def start(self):
        if RICH_AVAILABLE:
            self.console.print(f"\n[bold]Benchmark: {self.run_id}[/bold] | Tests: {self.total}\n")
            self.progress.start()
            self.task_id = self.progress.add_task(
                description=f"[cyan]Starting...",
                total=self.total,
                passed=0,
                failed=0
            )
        else:
            print(f"\nBenchmark: {self.run_id} | Tests: {self.total}\n")

    def update(self, test_name: str, elapsed: float, success: bool):
        with self.lock:
            self.current_test = test_name
            self.test_times.append(elapsed)
            if success:
                self.completed += 1
            else:
                self.failed += 1

            if RICH_AVAILABLE and self.task_id is not None:
                display_name = test_name[:35] + "..." if len(test_name) > 35 else test_name
                self.progress.update(
                    self.task_id, advance=1,
                    description=f"[cyan]{display_name}",
                    passed=self.completed, failed=self.failed
                )
                self.progress.refresh()
            else:
                self._print_plain_progress()

    def _print_plain_progress(self):
        done = self.completed + self.failed
        pct = (done / self.total) * 100 if self.total > 0 else 0
        elapsed = time.time() - self.start_time

        if done > 0:
            recent = self.test_times[-10:]
            avg_time = sum(recent) / len(recent)
            remaining = (self.total - done) * avg_time
            eta = f"{remaining:.0f}s" if remaining < 60 else f"{remaining/60:.1f}m"
        else:
            eta = "..."

        bar_width = 30
        filled = int(bar_width * done / self.total) if self.total > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)
        test_display = self.current_test[:25] + "..." if len(self.current_test) > 25 else self.current_test

        status = f"\r[{bar}] {pct:5.1f}% | {done}/{self.total} | ✓{self.completed} ✗{self.failed} | ETA:{eta} | {test_display}"
        sys.stderr.write("\r" + " " * self._last_line_length + "\r" + status)
        sys.stderr.flush()
        self._last_line_length = len(status)

    def get_stats(self) -> Dict:
        with self.lock:
            elapsed = time.time() - self.start_time
            done = self.completed + self.failed
            avg_time = sum(self.test_times) / len(self.test_times) if self.test_times else 0
            return {
                "total": self.total, "completed": self.completed, "failed": self.failed,
                "remaining": self.total - done, "elapsed_seconds": elapsed,
                "avg_time_per_test": avg_time, "success_rate": self.completed / done if done > 0 else 0
            }

    def print_summary(self):
        stats = self.get_stats()
        if RICH_AVAILABLE:
            table = Table(title="Summary", show_header=True, header_style="bold magenta")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("Total", str(stats["total"]))
            table.add_row("Passed", f"[green]{stats['completed']}[/green]")
            table.add_row("Failed", f"[red]{stats['failed']}[/red]")
            table.add_row("Success Rate", f"{stats['success_rate']:.1%}")
            table.add_row("Total Time", f"{stats['elapsed_seconds']:.1f}s")
            table.add_row("Avg Time/Test", f"{stats['avg_time_per_test']:.2f}s")
            self.console.print("\n")
            self.console.print(table)
        else:
            print(f"\n{'='*50}")
            print(f"Total: {stats['total']} | Passed: {stats['completed']} | Failed: {stats['failed']}")
            print(f"Success Rate: {stats['success_rate']:.1%} | Time: {stats['elapsed_seconds']:.1f}s")
            print(f"{'='*50}")

    def stop(self):
        if RICH_AVAILABLE and self.progress:
            self.progress.stop()
        else:
            sys.stderr.write("\r" + " " * self._last_line_length + "\r")
            sys.stderr.flush()
        self.print_summary()


class EnhancedBenchmarkSuite:
    """Enhanced benchmarking suite with checkpointing and monitoring"""

    MODEL_TIERS = [
        {"tier": "S",     "model": "qwen3:1.7b",   "provider": "ollama", "params": "1.7B"},
        {"tier": "M",     "model": "qwen3:4b",     "provider": "ollama", "params": "4B"},
        {"tier": "L",     "model": "qwen3:8b",     "provider": "ollama", "params": "8B"},
        {"tier": "XL",    "model": "qwen3:14b",    "provider": "ollama", "params": "14B"},
        {"tier": "API-1", "model": "gpt-4o-mini",  "provider": "openai", "params": "N/A"},
        {"tier": "API-2", "model": "gpt-5.2",      "provider": "openai", "params": "N/A"},
        # ^^^ VERIFY exact GPT-5.2 model string before running
    ]

    TIMEOUT_BY_MODEL = {
        "qwen3:1.7b": 90, "qwen3:4b": 90, "qwen3:8b": 90,
        "qwen3:14b": 90, "gpt-4o-mini": 90, "gpt-5.2": 90,
    }

    def __init__(self,
                 mcp_server_path: str,
                 config_dir: Path = Path("config"),
                 output_dir: Path = Path("benchmark_results"),
                 openai_api_key: str = None,
                 num_repeats: int = 1):

        self.mcp_server_path = mcp_server_path
        self.config_dir = config_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.openai_api_key = openai_api_key
        self.num_repeats = num_repeats

        self.test_cases = get_all_test_cases()
        self.model_configs = self._create_model_configs()

        self.checkpoint_db = self.output_dir / "benchmark_checkpoint.db"
        self.checkpoint = CheckpointManager(self.checkpoint_db)
        self.monitor: Optional[ProgressMonitor] = None

        self._shutdown_requested = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print("\n[!] Shutdown requested. Saving checkpoint...")
        self._shutdown_requested = True

    def _create_model_configs(self) -> List[Dict]:
        """
        Generate all pairwise orchestrator × specialist model configurations.
        6 tiers × 6 tiers = 36 combinations.
        """
        configs = []
        for orch_tier in self.MODEL_TIERS:
            for agent_tier in self.MODEL_TIERS:
                orch_prov = orch_tier["provider"]
                agent_prov = agent_tier["provider"]

                if orch_prov == agent_prov:
                    provider = orch_prov
                else:
                    provider = "mixed"

                configs.append({
                    "provider":     provider,
                    "orchestrator": orch_tier["model"],
                    "agent":        agent_tier["model"],
                    "orch_tier":    orch_tier["tier"],
                    "agent_tier":   agent_tier["tier"],
                })
        return configs

    def _generate_all_configs(self, categories: List[TestCategory], model_configs: List[Dict],
                              run_id: str) -> List[Tuple[BenchmarkConfig, BenchmarkTestCase]]:
        configs = []

        # --- Multi-agent configs (existing: C-1, C-2, D) ---
        multi_agent_modes = [IntelligenceMode.CENTRALIZED, IntelligenceMode.DECENTRALIZED]
        two_stage_options = [False, True]

        for category in categories:
            for test_case in self.test_cases.get(category, []):
                # Multi-agent: all model pairings × all modes
                for model_config in model_configs:
                    for intel_mode in multi_agent_modes:
                        for two_stage in two_stage_options:
                            if two_stage and intel_mode != IntelligenceMode.CENTRALIZED:
                                continue
                            for repeat_idx in range(self.num_repeats):
                                config = BenchmarkConfig(
                                    intelligence_mode=intel_mode,
                                    agent_generation_mode=AgentGenerationMode.STATIC,
                                    orchestrator_model=model_config["orchestrator"],
                                    agent_model=model_config["agent"],
                                    test_case_id=test_case.test_id,
                                    test_case_name=test_case.name,
                                    run_id=run_id,
                                    provider=model_config["provider"],
                                    two_stage_planning=two_stage,
                                    repeat_index=repeat_idx
                                )
                                configs.append((config, test_case))

                # Single-agent ReAct baseline: one run per unique model in model_configs
                react_models_seen = set()
                for mc in model_configs:
                    # Use orchestrator model as the single-agent model (deduplicate)
                    react_model = mc["orchestrator"]
                    if react_model in react_models_seen:
                        continue
                    react_models_seen.add(react_model)
                    for repeat_idx in range(self.num_repeats):
                        config = BenchmarkConfig(
                            intelligence_mode=IntelligenceMode.SINGLE_AGENT_REACT,
                            agent_generation_mode=AgentGenerationMode.STATIC,
                            orchestrator_model=react_model,
                            agent_model=react_model,
                            test_case_id=test_case.test_id,
                            test_case_name=test_case.name,
                            run_id=run_id,
                            provider=mc["provider"],
                            two_stage_planning=False,
                            repeat_index=repeat_idx
                        )
                        configs.append((config, test_case))

        return configs

    async def run_single_benchmark(self, config: 'BenchmarkConfig',
                                   test_case: 'BenchmarkTestCase') -> 'BenchmarkResult':
        start_time = time.time()

        # ─── LLM client setup (UNCHANGED from your current code) ───
        if config.provider == "mixed":
            def get_provider_for_model(model: str) -> str:
                openai_prefixes = ("gpt-", "davinci", "curie", "babbage", "ada")
                if model.lower().startswith(openai_prefixes):
                    return "openai"
                return "ollama"

            orch_provider = get_provider_for_model(config.orchestrator_model)
            agent_provider = get_provider_for_model(config.agent_model)

            llm_client = create_llm_client(
                provider=orch_provider,
                model=config.orchestrator_model,
                api_key=self.openai_api_key if orch_provider == "openai" else None,
                host="http://localhost:11434" if orch_provider == "ollama" else None
            )

            if agent_provider != orch_provider:
                agent_llm_client = create_llm_client(
                    provider=agent_provider,
                    model=config.agent_model,
                    api_key=self.openai_api_key if agent_provider == "openai" else None,
                    host="http://localhost:11434" if agent_provider == "ollama" else None
                )
            else:
                agent_llm_client = None
        else:
            llm_client = create_llm_client(
                provider=config.provider,
                model=config.orchestrator_model,
                api_key=self.openai_api_key if config.provider == "openai" else None,
                host="http://localhost:11434" if config.provider == "ollama" else None
            )
            agent_llm_client = None

        orchestrator = Orchestrator(
            llm_client=llm_client,
            mcp_server_path=self.mcp_server_path,
            intelligence_mode=config.intelligence_mode,
            agent_generation_mode=config.agent_generation_mode,
            orchestrator_model=config.orchestrator_model,
            agent_model=config.agent_model,
            agent_llm_client=agent_llm_client,
            config_dir=self.config_dir,
            log_dir=Path(f"logs/benchmark_{config.run_id}"),
            use_two_stage_planning=config.two_stage_planning
        )

        result = BenchmarkResult(
            config=config, category=test_case.category.value,
            timestamp=datetime.now().isoformat()
        )

        # ████████████████████████████████████████████████████████████████
        # ████  EVERYTHING BELOW THIS LINE IS CHANGED                ████
        # ████████████████████████████████████████████████████████████████

        try:
            await orchestrator.initialize()
            response = await orchestrator.process_request(test_case.request)

            # ─── Extract metrics from response (UNCHANGED) ───
            benchmark_data = response.get("benchmark", {})
            metrics = benchmark_data.get("metrics", {})
            timing = metrics.get("timing", {})
            tokens = metrics.get("tokens", {})
            cost = metrics.get("cost", {})
            agents = metrics.get("agents", {})

            result.total_time = timing.get("total_time", time.time() - start_time)
            result.planning_time = timing.get("planning_time", 0)
            result.execution_time = timing.get("execution_time", 0)
            result.synthesis_time = timing.get("synthesis_time", 0)
            result.orchestrator_prompt_tokens = tokens.get("orchestrator_prompt", 0)
            result.orchestrator_completion_tokens = tokens.get("orchestrator_completion", 0)
            result.orchestrator_total_tokens = tokens.get("orchestrator_total", 0)
            result.agent_total_tokens = tokens.get("agents_total", 0)
            result.grand_total_tokens = tokens.get("grand_total", 0)
            result.orchestrator_cost = cost.get("orchestrator", 0)
            result.agent_cost = cost.get("agents", 0)
            result.total_cost = cost.get("total", 0)
            result.tool_calls = metrics.get("tool_calls", 0)
            result.tools_used = list(set(
                tool for a in agents.values() for tool in a.get("tools", [])
            ))
            result.agents_used = list(agents.keys())
            result.agent_metrics = agents
            result.execution_plan = benchmark_data.get("execution_plan", [])
            result.agent_results = benchmark_data.get("agent_results", {})
            result.response_summary = response.get("summary", "")

            # ─── Calculate accuracy (UNCHANGED) ───
            result.accuracy_metrics = self._calculate_accuracy(
                test_case=test_case,
                agents_used=result.agents_used,
                tools_used=result.tools_used,
                execution_plan=result.execution_plan,
                agent_results=result.agent_results,
                intelligence_mode=config.intelligence_mode
            )
            result.retry_stats = self._calculate_retry_stats(result.agent_results)
            # ████ NEW: Classify success/failure based on what actually happened ████
            #
            # The orchestrator returned a response (no exception was thrown).
            # But that doesn't mean everything worked. Three scenarios:
            #
            # Scenario A: orchestrator itself reported an error
            #   → response["status"] == "error"
            #   → This means something crashed inside process_request but was
            #     caught internally (e.g., planning failed, agent not found)
            #
            # Scenario B: orchestrator reported "failed"
            #   → The workflow ran but produced bad results
            #   → This is a reasoning failure
            #
            # Scenario C: orchestrator reported "success" or "partial"
            #   → But accuracy might still be zero (wrong tools/params)
            #   → Check accuracy to decide if it's really a success

            response_status = response.get("status", "unknown")

            if response_status == "error":
                # Scenario A: internal error that was caught
                result.success = False
                result.status = RunStatus.FAILED
                result.error = response.get("error", "Unknown internal error")
                result.failure_type = response.get(
                    "failure_type", FailureType.RUNTIME_FAILURE.value
                )

            elif response_status == "failed":
                # Scenario B: workflow ran but failed
                result.success = False
                result.status = RunStatus.COMPLETED  # It completed, just poorly
                result.failure_type = FailureType.REASONING_FAILURE.value

            else:
                # Scenario C: "success" or "partial" — check accuracy
                result.success = True
                result.status = RunStatus.COMPLETED

                # Post-hoc check: if accuracy is zero despite having expected tools,
                # the LLM chose completely wrong tools/agents → reasoning failure
                combined_accuracy = (
                        result.accuracy_metrics.tool_selection_accuracy
                        + result.accuracy_metrics.plan_step_accuracy
                )
                has_expected_tools = len(test_case.expected_tools) > 0

                if combined_accuracy == 0 and has_expected_tools:
                    result.failure_type = FailureType.REASONING_FAILURE.value
                else:
                    result.failure_type = FailureType.NONE.value

        # ████ NEW: Separate except blocks for different failure types ████
        #
        # Previously this was just:
        #     except Exception as e:
        #         result.error = str(e)
        #         result.success = False
        #         result.status = RunStatus.FAILED
        #         result.total_time = time.time() - start_time
        #
        # Now we distinguish timeout, JSON parse, and everything else.
        # This is what Reviewer 2 #17 asked for: "report runtime failure
        # rates explicitly" — you can't do that if all failures are lumped.

        except asyncio.TimeoutError:
            # The entire workflow timed out (not just one step)
            result.error = "Overall workflow timeout"
            result.success = False
            result.status = RunStatus.FAILED
            result.failure_type = FailureType.RUNTIME_FAILURE.value
            result.total_time = time.time() - start_time

        except json.JSONDecodeError as e:
            # A JSON parse error escaped safe_json_parse somehow
            # (shouldn't happen after patch C7, but just in case)
            result.error = f"JSON parse error: {str(e)}"
            result.success = False
            result.status = RunStatus.FAILED
            result.failure_type = FailureType.JSON_PARSE_FAILURE.value
            result.total_time = time.time() - start_time

        except Exception as e:
            # Everything else: OOM, network error, MCP crash, etc.
            result.error = str(e)
            result.success = False
            result.status = RunStatus.FAILED
            result.failure_type = FailureType.RUNTIME_FAILURE.value
            result.total_time = time.time() - start_time

        # ████████████████████████████████████████████████████████████████
        # ████  END OF CHANGES — finally block is UNCHANGED          ████
        # ████████████████████████████████████████████████████████████████

        finally:
            await orchestrator.shutdown()
            await llm_client.close()
            if agent_llm_client is not None:
                await agent_llm_client.close()

        return result

    # ==================== ACCURACY CALCULATION METHODS ====================

    def _extract_planned_tools_from_execution_plan(self, execution_plan: List[Dict]) -> List[Dict]:
        """
        Extract what the orchestrator PLANNED from execution_plan.
        Only meaningful for centralized mode.
        Returns: [{step_id, agent_id, tools: [tool_names], parameters: {tool: params}}]
        """
        planned = []
        for step in sorted(execution_plan, key=lambda s: s.get("step_id", "")):
            step_id = step.get("step_id", "")
            agent_id = step.get("agent_id", "")

            tools = []
            parameters = {}

            guidance = step.get("orchestrator_guidance", {})
            if guidance:
                tool_instructions = guidance.get("tool_instructions", [])
                for instr in tool_instructions:
                    tool_name = instr.get("tool", "")
                    if tool_name:
                        tools.append(tool_name)
                        parameters[tool_name] = instr.get("parameters", {})

            planned.append({
                "step_id": step_id,
                "agent_id": agent_id,
                "tools": tools,
                "parameters": parameters
            })

        return planned

    def _extract_executed_tools_from_agent_results(self, agent_results: Dict[str, Any]) -> List[Dict]:
        """
        Extract what agents ACTUALLY EXECUTED from agent_results.
        Returns: [{step_id, agent_id, tools: [tool_names], parameters: {tool: params}}]
        """
        executed = []

        for step_id, result in sorted(agent_results.items(), key=lambda x: x[0]):
            if not isinstance(result, dict):
                continue

            agent_id = result.get("agent", "unknown")
            tools = []
            parameters = {}

            for tool_info in result.get("tools_used", []):
                if isinstance(tool_info, dict):
                    tool_name = tool_info.get("tool", "")
                    if tool_name:
                        tools.append(tool_name)
                        parameters[tool_name] = tool_info.get("parameters", {})

            executed.append({
                "step_id": step_id,
                "agent_id": agent_id,
                "tools": tools,
                "parameters": parameters
            })

        return executed

    def _calculate_plan_vs_execution(self, execution_plan: List[Dict],
                                      agent_results: Dict[str, Any]) -> PlanVsExecutionMetrics:
        """
        Compare orchestrator's plan vs actual agent execution.
        This reveals if agents are overthinking, improving, or degrading the plan.
        """
        metrics = PlanVsExecutionMetrics()

        # Extract planned and executed
        planned = self._extract_planned_tools_from_execution_plan(execution_plan)
        executed = self._extract_executed_tools_from_agent_results(agent_results)

        metrics.planned_tools_by_step = planned
        metrics.executed_tools_by_step = executed

        # Flatten tool lists for comparison
        all_planned_tools = []
        all_executed_tools = []
        planned_params = {}  # tool -> params
        executed_params = {}  # tool -> params

        for step in planned:
            all_planned_tools.extend(step["tools"])
            planned_params.update(step["parameters"])

        for step in executed:
            all_executed_tools.extend(step["tools"])
            executed_params.update(step["parameters"])

        metrics.planned_tool_count = len(all_planned_tools)
        metrics.executed_tool_count = len(all_executed_tools)

        planned_set = set(all_planned_tools)
        executed_set = set(all_executed_tools)

        # Tools added by agents (not in plan)
        metrics.tools_added_by_agents = list(executed_set - planned_set)

        # Tools removed by agents (in plan but not executed)
        metrics.tools_removed_by_agents = list(planned_set - executed_set)

        # Tools where parameters were modified
        common_tools = planned_set & executed_set
        for tool in common_tools:
            p_params = planned_params.get(tool, {})
            e_params = executed_params.get(tool, {})
            if p_params != e_params:
                metrics.tools_modified_parameters.append({
                    "tool": tool,
                    "planned_params": p_params,
                    "executed_params": e_params
                })

        # Plan adherence rate
        if planned_set:
            adhered_tools = planned_set & executed_set
            metrics.plan_adherence_rate = len(adhered_tools) / len(planned_set)
        else:
            metrics.plan_adherence_rate = 1.0 if not executed_set else 0.0

        # Set analysis flags
        metrics.agent_followed_plan = (
            not metrics.tools_added_by_agents and
            not metrics.tools_removed_by_agents and
            not metrics.tools_modified_parameters
        )
        metrics.agent_added_tools = len(metrics.tools_added_by_agents) > 0
        metrics.agent_removed_tools = len(metrics.tools_removed_by_agents) > 0
        metrics.agent_modified_params = len(metrics.tools_modified_parameters) > 0

        return metrics

    def _extract_actual_parameters_by_tool(self, agent_results: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract actual parameters used for each tool from agent results.
        Returns a LIST of parameter dicts per tool to handle multiple calls.
        """
        tool_parameters = {}  # tool_name -> list of {parameters, step_id}

        for step_id, result in agent_results.items():
            if not isinstance(result, dict):
                continue

            for tool_info in result.get("tools_used", []):
                tool_name = tool_info.get("tool", "")
                parameters = tool_info.get("parameters", {})

                if tool_name:
                    if tool_name not in tool_parameters:
                        tool_parameters[tool_name] = []
                    tool_parameters[tool_name].append({
                        "parameters": parameters,
                        "step_id": step_id
                    })

        return tool_parameters

    def _values_match(self, expected: Any, actual: Any) -> bool:
        """Check if two values match with type-aware comparison"""
        if expected is None:
            return actual is None
        if type(expected) != type(actual):
            if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
                return abs(float(expected) - float(actual)) < 1e-9
            return False
        if isinstance(expected, str):
            return expected.lower().strip() == actual.lower().strip()
        if isinstance(expected, (int, float)):
            return abs(expected - actual) < 1e-9
        if isinstance(expected, bool):
            return expected == actual
        if isinstance(expected, list):
            if len(expected) != len(actual):
                return False
            return all(self._values_match(e, a) for e, a in zip(expected, actual))
        if isinstance(expected, dict):
            if set(expected.keys()) != set(actual.keys()):
                return False
            return all(self._values_match(expected[k], actual[k]) for k in expected)
        return expected == actual

    def _calculate_parameter_accuracy(self, test_case, agent_results: Dict[str, Any]) -> Tuple[float, float, List[Dict]]:
        """
        Calculate parameter key and value accuracy.
        Handles multiple calls to same tool.
        """
        actual_params_by_tool = self._extract_actual_parameters_by_tool(agent_results)

        # Build expected parameters from test case
        expected_params_by_tool = {}
        for step in test_case.expected_steps:
            if hasattr(step, 'expected_parameters') and step.expected_parameters:
                for tool_name, params in step.expected_parameters.items():
                    if params:
                        if tool_name not in expected_params_by_tool:
                            expected_params_by_tool[tool_name] = []
                        expected_params_by_tool[tool_name].append(params)

        if not expected_params_by_tool:
            return 1.0, 1.0, []

        details = []
        total_keys = 0
        correct_keys = 0
        correct_values = 0

        for tool_name, expected_params_list in expected_params_by_tool.items():
            actual_calls = actual_params_by_tool.get(tool_name, [])

            for exp_idx, expected_params in enumerate(expected_params_list):
                # Find best matching actual call
                best_match = None
                best_match_score = -1

                for act_call in actual_calls:
                    actual_params = act_call.get("parameters", {})
                    match_score = len(set(expected_params.keys()) & set(actual_params.keys()))
                    if match_score > best_match_score:
                        best_match_score = match_score
                        best_match = actual_params

                if best_match is None:
                    best_match = {}

                tool_total_keys = len(expected_params)
                tool_correct_keys = 0
                tool_correct_values = 0
                key_results = {}

                for key, expected_value in expected_params.items():
                    total_keys += 1
                    key_exists = key in best_match

                    if key_exists:
                        tool_correct_keys += 1
                        correct_keys += 1
                        actual_value = best_match[key]
                        value_match = self._values_match(expected_value, actual_value)
                        if value_match:
                            tool_correct_values += 1
                            correct_values += 1
                        key_results[key] = {
                            "expected": expected_value,
                            "actual": actual_value,
                            "key_correct": True,
                            "value_correct": value_match
                        }
                    else:
                        key_results[key] = {
                            "expected": expected_value,
                            "actual": None,
                            "key_correct": False,
                            "value_correct": False
                        }

                details.append({
                    "tool": tool_name,
                    "expected_call_index": exp_idx,
                    "expected_params": expected_params,
                    "actual_params": best_match,
                    "key_accuracy": tool_correct_keys / tool_total_keys if tool_total_keys else 0,
                    "value_accuracy": tool_correct_values / tool_total_keys if tool_total_keys else 0,
                    "total_keys": tool_total_keys,
                    "correct_keys": tool_correct_keys,
                    "correct_values": tool_correct_values,
                    "key_results": key_results
                })

        key_accuracy = correct_keys / total_keys if total_keys else 1.0
        value_accuracy = correct_values / total_keys if total_keys else 1.0

        return key_accuracy, value_accuracy, details


    def _calculate_retry_stats(self, agent_results: Dict[str, Any]) -> RetryStats:
        """Calculate retry statistics from agent execution results"""
        stats = RetryStats()

        for step_id, result in agent_results.items():
            if not isinstance(result, dict):
                continue

            stats.total_steps += 1

            # Step-level retry info (injected by orchestrator)
            retry_info = result.get("retry_info", {})
            retries_used = retry_info.get("retries_used", 0)
            if retries_used > 0:
                stats.steps_retried += 1
                if retry_info.get("final_success", False):
                    stats.steps_retry_succeeded += 1
                else:
                    stats.steps_retry_failed += 1

            # Tool-level retry info (from agent's tools_used)
            for tool_info in result.get("tools_used", []):
                if not isinstance(tool_info, dict):
                    continue
                stats.total_tool_calls += 1
                attempts = tool_info.get("attempts", 1)
                if attempts > 1:
                    stats.tool_calls_retried += 1
                    if tool_info.get("success", False):
                        stats.tool_calls_retry_succeeded += 1
                    else:
                        stats.tool_calls_retry_failed += 1

        # Calculate rates
        if stats.total_steps > 0:
            stats.step_retry_rate = stats.steps_retried / stats.total_steps
        if stats.total_tool_calls > 0:
            stats.tool_retry_rate = stats.tool_calls_retried / stats.total_tool_calls

        total_retried = stats.steps_retried + stats.tool_calls_retried
        total_retry_recovered = stats.steps_retry_succeeded + stats.tool_calls_retry_succeeded
        if total_retried > 0:
            stats.retry_recovery_rate = total_retry_recovered / total_retried

        return stats

    def _calculate_accuracy(self, test_case, agents_used: List[str], tools_used: List[str],
                            execution_plan: List[Dict], agent_results: Dict[str, Any],
                            intelligence_mode: IntelligenceMode) -> AccuracyMetrics:
        """
        Calculate all accuracy metrics with proper extraction for all modes.
        """
        metrics = AccuracyMetrics()

        # ===== Tool Accuracy =====
        expected_tools = set(test_case.expected_tools)
        actual_tools = set(tools_used)
        metrics.tools_expected = list(expected_tools)
        metrics.tools_actual = list(actual_tools)
        metrics.tools_correct = list(expected_tools & actual_tools)
        metrics.tools_missing = list(expected_tools - actual_tools)
        metrics.tools_extra = list(actual_tools - expected_tools)
        metrics.tool_selection_accuracy = (
            len(metrics.tools_correct) / len(expected_tools) if expected_tools else 1.0
        )

        # ===== Agent Accuracy =====
        expected_agents = set(test_case.expected_agents)
        actual_agents = set(agents_used)

        if intelligence_mode == IntelligenceMode.SINGLE_AGENT_REACT:
            # Single-agent baseline: agent selection is not meaningful
            # (always "universal_agent"), so score as N/A → 1.0 to avoid
            # unfairly penalizing the baseline in aggregate metrics
            metrics.agents_expected = list(expected_agents)
            metrics.agents_actual = list(actual_agents)
            metrics.agents_correct = []
            metrics.agents_missing = []
            metrics.agents_extra = []
            metrics.agent_selection_accuracy = -1.0  # Sentinel: skip in aggregation
        else:
            metrics.agents_expected = list(expected_agents)
            metrics.agents_actual = list(actual_agents)
            metrics.agents_correct = list(expected_agents & actual_agents)
            metrics.agents_missing = list(expected_agents - actual_agents)
            metrics.agents_extra = list(actual_agents - expected_agents)
            metrics.agent_selection_accuracy = (
                len(metrics.agents_correct) / len(expected_agents) if expected_agents else 1.0
            )

        # ===== Plan Step Accuracy (based on ACTUAL execution) =====
        expected_seq = test_case.get_expected_step_sequence()

        # Extract actual sequence from agent_results (what was ACTUALLY executed)
        executed = self._extract_executed_tools_from_agent_results(agent_results)
        actual_seq = [(step["agent_id"], step["tools"]) for step in executed]

        metrics.expected_plan_sequence = [
            {"step": i + 1, "agent": a, "tools": t}
            for i, (a, t) in enumerate(expected_seq)
        ]
        metrics.actual_plan_sequence = [
            {"step": i + 1, "agent": a, "tools": t}
            for i, (a, t) in enumerate(actual_seq)
        ]

        # Subsequence matching
        matched = 0
        exp_idx = 0
        for act_agent, act_tools in actual_seq:
            if exp_idx >= len(expected_seq):
                break
            exp_agent, exp_tools = expected_seq[exp_idx]
            if exp_agent == act_agent and set(exp_tools) <= set(act_tools):
                matched += 1
                exp_idx += 1

        metrics.expected_step_count = len(expected_seq)
        metrics.matched_step_count = matched
        metrics.plan_step_accuracy = matched / len(expected_seq) if expected_seq else 1.0

        # ===== Parameter Accuracy =====
        param_key_acc, param_val_acc, param_details = self._calculate_parameter_accuracy(
            test_case, agent_results
        )
        metrics.parameter_key_accuracy = param_key_acc
        metrics.parameter_value_accuracy = param_val_acc
        metrics.parameter_details = param_details

        # ===== Plan vs Execution (Centralized mode only) =====
        if intelligence_mode == IntelligenceMode.CENTRALIZED:
            metrics.plan_vs_execution = self._calculate_plan_vs_execution(
                execution_plan, agent_results
            )
        # For decentralized mode, plan_vs_execution stays as default (empty)

        return metrics

    # ==================== BENCHMARK EXECUTION ====================

    async def run_benchmark(self, categories: List[TestCategory] = None,
                            model_configs: List[Dict] = None,
                            run_id: str = None, resume: bool = True) -> List[BenchmarkResult]:

        categories = categories or list(TestCategory)
        model_configs = model_configs or self.model_configs
        run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

        all_configs = self._generate_all_configs(categories, model_configs, run_id)
        total_tests = len(all_configs)

        self.checkpoint.create_run(run_id, total_tests, {
            "categories": [c.value for c in categories], "num_repeats": self.num_repeats
        })

        pending_configs = []
        for config, test_case in all_configs:
            config_key = config.get_unique_key()
            if resume and self.checkpoint.is_test_completed(config_key):
                continue
            pending_configs.append((config, test_case))

        if len(pending_configs) < total_tests:
            print(f"Resuming: {total_tests - len(pending_configs)} already completed")

        self.monitor = ProgressMonitor(len(pending_configs), run_id)
        self.monitor.start()

        results = []

        try:
            for config, test_case in pending_configs:
                if self._shutdown_requested:
                    print("\nShutdown requested. Progress saved.")
                    break

                start = time.time()
                result = await self.run_single_benchmark(config, test_case)
                elapsed = time.time() - start

                self.checkpoint.save_result(result)
                results.append(result)
                self.monitor.update(
                    f"{config.test_case_name} [{config.get_mode_label()}]",
                    elapsed, result.success
                )

        finally:
            self.monitor.stop()
            if not self._shutdown_requested:
                self.checkpoint.finalize_run(run_id)
            all_results = self.checkpoint.get_all_results(run_id)
            self._save_results(run_id, all_results)
            self._generate_report(run_id, all_results)

        return self.checkpoint.get_all_results(run_id)

    def _save_results(self, run_id: str, results: List[BenchmarkResult]):
        output_file = self.output_dir / f"results_{run_id}.json"
        data = []
        for r in results:
            d = asdict(r)
            d["config"]["intelligence_mode"] = r.config.intelligence_mode.value
            d["config"]["agent_generation_mode"] = r.config.agent_generation_mode.value
            d["status"] = r.status.value
            d["test_category"] = r.extract_test_category()
            data.append(d)
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"\nResults saved to: {output_file}")

    def _generate_report(self, run_id: str, results: List[BenchmarkResult]):
        report_file = self.output_dir / f"report_{run_id}.md"
        with open(report_file, 'w') as f:
            f.write("# Multi-Agent Benchmark Report\n\n")
            f.write(f"**Run ID**: {run_id}\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            n = len(results)
            if n == 0:
                f.write("No results to report.\n")
                return

            f.write("## Executive Summary\n\n")
            f.write(f"- **Total Tests**: {n}\n")
            f.write(f"- **Success Rate**: {sum(r.success for r in results) / n:.1%}\n")
            f.write(f"- **Avg Tool Accuracy**: {sum(r.accuracy_metrics.tool_selection_accuracy for r in results) / n:.1%}\n")
            f.write(f"- **Avg Agent Accuracy**: {sum(r.accuracy_metrics.agent_selection_accuracy for r in results) / n:.1%}\n")
            f.write(f"- **Avg Plan Accuracy**: {sum(r.accuracy_metrics.plan_step_accuracy for r in results) / n:.1%}\n")
            f.write(f"- **Avg Param Key Accuracy**: {sum(r.accuracy_metrics.parameter_key_accuracy for r in results) / n:.1%}\n")
            f.write(f"- **Avg Param Value Accuracy**: {sum(r.accuracy_metrics.parameter_value_accuracy for r in results) / n:.1%}\n")
            f.write(f"- **Avg Tokens**: {sum(r.grand_total_tokens for r in results) / n:.0f}\n")
            f.write(f"- **Avg Cost**: ${sum(r.total_cost for r in results) / n:.4f}\n")
            f.write(f"- **Avg Time**: {sum(r.total_time for r in results) / n:.2f}s\n\n")

            # Plan vs Execution Analysis (Centralized mode only)
            centralized_results = [r for r in results if r.config.intelligence_mode == IntelligenceMode.CENTRALIZED]
            if centralized_results:
                f.write("## Plan vs Execution Analysis (Centralized Mode)\n\n")

                followed_plan = sum(1 for r in centralized_results if r.accuracy_metrics.plan_vs_execution.agent_followed_plan)
                added_tools = sum(1 for r in centralized_results if r.accuracy_metrics.plan_vs_execution.agent_added_tools)
                removed_tools = sum(1 for r in centralized_results if r.accuracy_metrics.plan_vs_execution.agent_removed_tools)
                modified_params = sum(1 for r in centralized_results if r.accuracy_metrics.plan_vs_execution.agent_modified_params)
                avg_adherence = sum(r.accuracy_metrics.plan_vs_execution.plan_adherence_rate for r in centralized_results) / len(centralized_results)

                f.write(f"- **Tests where agent followed plan exactly**: {followed_plan}/{len(centralized_results)} ({followed_plan/len(centralized_results):.1%})\n")
                f.write(f"- **Tests where agent added tools**: {added_tools}/{len(centralized_results)} ({added_tools/len(centralized_results):.1%})\n")
                f.write(f"- **Tests where agent removed tools**: {removed_tools}/{len(centralized_results)} ({removed_tools/len(centralized_results):.1%})\n")
                f.write(f"- **Tests where agent modified parameters**: {modified_params}/{len(centralized_results)} ({modified_params/len(centralized_results):.1%})\n")
                f.write(f"- **Average plan adherence rate**: {avg_adherence:.1%}\n\n")

            f.write("## Results by Mode\n\n")
            mode_groups = defaultdict(list)
            for r in results:
                mode_groups[r.config.get_mode_label()].append(r)

            f.write("| Mode | Tests | Success | Tool Acc | Agent Acc | Plan Acc | Param Key | Param Val | Tokens | Time |\n")
            f.write("|------|-------|---------|----------|-----------|----------|-----------|-----------|--------|------|\n")
            for mode in ["C-1", "C-2", "D", "ReAct"]:
                if mode in mode_groups:
                    rs = mode_groups[mode]
                    m = len(rs)
                    f.write(f"| {mode} | {m} | {sum(r.success for r in rs) / m:.1%} | ")
                    f.write(f"{sum(r.accuracy_metrics.tool_selection_accuracy for r in rs) / m:.1%} | ")
                    f.write(f"{sum(r.accuracy_metrics.agent_selection_accuracy for r in rs) / m:.1%} | ")
                    f.write(f"{sum(r.accuracy_metrics.plan_step_accuracy for r in rs) / m:.1%} | ")
                    f.write(f"{sum(r.accuracy_metrics.parameter_key_accuracy for r in rs) / m:.1%} | ")
                    f.write(f"{sum(r.accuracy_metrics.parameter_value_accuracy for r in rs) / m:.1%} | ")
                    f.write(f"{sum(r.grand_total_tokens for r in rs) / m:.0f} | ")
                    f.write(f"{sum(r.total_time for r in rs) / m:.1f}s |\n")

            # Failure type breakdown
            f.write("\\n## Failure Type Analysis\\n\\n")
            failure_groups = defaultdict(int)
            for r in results:
                ft = getattr(r, 'failure_type', 'none') or 'none'
                failure_groups[ft] += 1
            f.write("| Failure Type | Count | % |\\n")
            f.write("|-------------|-------|---|\\n")
            for ft, count in sorted(failure_groups.items(), key=lambda x: -x[1]):
                pct = count / n * 100
                f.write(f"| {ft} | {count} | {pct:.1f}% |\\n")


            # Retry analysis
            f.write("\\n## Retry Policy Analysis\\n\\n")

            runs_with_step_retry = [r for r in results if r.retry_stats.steps_retried > 0]
            runs_with_tool_retry = [r for r in results if r.retry_stats.tool_calls_retried > 0]
            total_steps = sum(r.retry_stats.total_steps for r in results)
            total_step_retries = sum(r.retry_stats.steps_retried for r in results)
            total_step_recovered = sum(r.retry_stats.steps_retry_succeeded for r in results)
            total_tools = sum(r.retry_stats.total_tool_calls for r in results)
            total_tool_retries = sum(r.retry_stats.tool_calls_retried for r in results)
            total_tool_recovered = sum(r.retry_stats.tool_calls_retry_succeeded for r in results)

            f.write(f"### Step-Level Retries (Orchestrator)\\n\\n")
            f.write(f"- Total steps executed: {total_steps}\\n")
            f.write(f"- Steps that needed retry: {total_step_retries} ({total_step_retries/max(total_steps,1)*100:.1f}%)\\n")
            f.write(f"- Retries that recovered: {total_step_recovered} ({total_step_recovered/max(total_step_retries,1)*100:.1f}% recovery rate)\\n")
            f.write(f"- Runs affected by step retry: {len(runs_with_step_retry)}/{n} ({len(runs_with_step_retry)/n*100:.1f}%)\\n\\n")

            f.write(f"### Tool-Level Retries (Agent)\\n\\n")
            f.write(f"- Total tool calls: {total_tools}\\n")
            f.write(f"- Tool calls that needed retry: {total_tool_retries} ({total_tool_retries/max(total_tools,1)*100:.1f}%)\\n")
            f.write(f"- Retries that recovered: {total_tool_recovered} ({total_tool_recovered/max(total_tool_retries,1)*100:.1f}% recovery rate)\\n")
            f.write(f"- Runs affected by tool retry: {len(runs_with_tool_retry)}/{n} ({len(runs_with_tool_retry)/n*100:.1f}%)\\n\\n")

            # Retry rate by model tier
            f.write(f"### Retry Rate by Orchestrator Model\\n\\n")
            f.write("| Model | Runs | Step Retry % | Tool Retry % | Recovery % |\\n")
            f.write("|-------|------|-------------|-------------|------------|\\n")
            model_retry = defaultdict(lambda: {"runs": 0, "steps": 0, "step_retried": 0,
                                               "tools": 0, "tool_retried": 0, "recovered": 0, "retried_total": 0})
            for r in results:
                m = r.config.orchestrator_model
                d = model_retry[m]
                d["runs"] += 1
                d["steps"] += r.retry_stats.total_steps
                d["step_retried"] += r.retry_stats.steps_retried
                d["tools"] += r.retry_stats.total_tool_calls
                d["tool_retried"] += r.retry_stats.tool_calls_retried
                d["recovered"] += r.retry_stats.steps_retry_succeeded + r.retry_stats.tool_calls_retry_succeeded
                d["retried_total"] += r.retry_stats.steps_retried + r.retry_stats.tool_calls_retried

            for model in sorted(model_retry.keys()):
                d = model_retry[model]
                step_pct = d["step_retried"] / max(d["steps"], 1) * 100
                tool_pct = d["tool_retried"] / max(d["tools"], 1) * 100
                recov_pct = d["recovered"] / max(d["retried_total"], 1) * 100
                f.write(f"| {model} | {d['runs']} | {step_pct:.1f}% | {tool_pct:.1f}% | {recov_pct:.1f}% |\\n")


            f.write("\n## Results by Model Configuration\n\n")
            model_groups = defaultdict(list)
            for r in results:
                key = f"{r.config.orchestrator_model} / {r.config.agent_model}"
                model_groups[key].append(r)

            f.write("| Orchestrator / Agent | Tests | Success | Avg Accuracy | Tokens | Cost |\n")
            f.write("|---------------------|-------|---------|--------------|--------|------|\n")
            for key, rs in sorted(model_groups.items()):
                m = len(rs)
                avg_acc = sum(
                    (r.accuracy_metrics.tool_selection_accuracy +
                     r.accuracy_metrics.agent_selection_accuracy +
                     r.accuracy_metrics.plan_step_accuracy +
                     r.accuracy_metrics.parameter_key_accuracy +
                     r.accuracy_metrics.parameter_value_accuracy) / 5 for r in rs
                ) / m
                f.write(f"| {key} | {m} | {sum(r.success for r in rs) / m:.1%} | ")
                f.write(f"{avg_acc:.1%} | {sum(r.grand_total_tokens for r in rs) / m:.0f} | ")
                f.write(f"${sum(r.total_cost for r in rs) / m:.4f} |\n")

            f.write("\n## Results by Test Category\n\n")
            cat_groups = defaultdict(list)
            for r in results:
                cat_groups[r.extract_test_category()].append(r)

            f.write("| Category | Tests | Success | Tool Acc | Agent Acc | Plan Acc | Param Acc |\n")
            f.write("|----------|-------|---------|----------|-----------|----------|----------|\n")
            for cat in sorted(cat_groups.keys()):
                rs = cat_groups[cat]
                m = len(rs)
                param_acc = sum((r.accuracy_metrics.parameter_key_accuracy + r.accuracy_metrics.parameter_value_accuracy) / 2 for r in rs) / m
                f.write(f"| {cat} | {m} | {sum(r.success for r in rs) / m:.1%} | ")
                f.write(f"{sum(r.accuracy_metrics.tool_selection_accuracy for r in rs) / m:.1%} | ")
                f.write(f"{sum(r.accuracy_metrics.agent_selection_accuracy for r in rs) / m:.1%} | ")
                f.write(f"{sum(r.accuracy_metrics.plan_step_accuracy for r in rs) / m:.1%} | ")
                f.write(f"{param_acc:.1%} |\n")

            f.write("\n## Results by Task Complexity\n\n")
            complexity_groups = defaultdict(list)
            for r in results:
                complexity_groups[r.category].append(r)

            f.write("| Complexity | Tests | Success | Tool Acc | Agent Acc | Plan Acc | Time |\n")
            f.write("|------------|-------|---------|----------|-----------|----------|------|\n")
            for complexity in ["single_agent_single_tool", "single_agent_multi_tool",
                               "multi_agent_single_tool", "multi_agent_multi_tool"]:
                if complexity in complexity_groups:
                    rs = complexity_groups[complexity]
                    m = len(rs)
                    f.write(f"| {complexity} | {m} | {sum(r.success for r in rs) / m:.1%} | ")
                    f.write(f"{sum(r.accuracy_metrics.tool_selection_accuracy for r in rs) / m:.1%} | ")
                    f.write(f"{sum(r.accuracy_metrics.agent_selection_accuracy for r in rs) / m:.1%} | ")
                    f.write(f"{sum(r.accuracy_metrics.plan_step_accuracy for r in rs) / m:.1%} | ")
                    f.write(f"{sum(r.total_time for r in rs) / m:.1f}s |\n")


            # Extra tool/agent analysis
            f.write("\n## Failure Type Analysis\n\n")
            f.write("Models may use more tools than strictly required (e.g., querying before adding, "
                    "validating after saving). This section quantifies that overhead.\\n\\n")

            # Per-model extra tool stats
            f.write("### Extra Tools by Model\\n\\n")
            f.write("| Model | Runs | Avg Extra Tools | Avg Extra Agents | "
                    "Avg Total Tool Calls | Common Extra Tools |\\n")
            f.write("|-------|------|----------------|-----------------|"
                    "--------------------|-------------------|\\n")

            from collections import Counter
            model_stats = defaultdict(lambda: {
                "runs": 0, "extra_tools_total": 0, "extra_agents_total": 0,
                "tool_calls_total": 0, "extra_tool_names": Counter()
            })

            for r in results:
                m = r.config.orchestrator_model
                d = model_stats[m]
                d["runs"] += 1
                extras = r.accuracy_metrics.tools_extra
                d["extra_tools_total"] += len(extras)
                d["extra_agents_total"] += len(r.accuracy_metrics.agents_extra)
                d["tool_calls_total"] += r.tool_calls
                for t in extras:
                    d["extra_tool_names"][t] += 1

            for model in sorted(model_stats.keys()):
                d = model_stats[model]
                n_runs = d["runs"]
                avg_extra_tools = d["extra_tools_total"] / n_runs
                avg_extra_agents = d["extra_agents_total"] / n_runs
                avg_tool_calls = d["tool_calls_total"] / n_runs
                # Top 3 most common extra tools
                top_extras = d["extra_tool_names"].most_common(3)
                top_str = ", ".join(f"{t}({c})" for t, c in top_extras) if top_extras else "none"
                f.write(f"| {model} | {n_runs} | {avg_extra_tools:.1f} | "
                        f"{avg_extra_agents:.1f} | {avg_tool_calls:.1f} | {top_str} |\\n")

            # Per-mode extra tool stats
            f.write("\\n### Extra Tools by Intelligence Mode\\n\\n")
            f.write("| Mode | Runs | Avg Extra Tools | Avg Extra Agents |\\n")
            f.write("|------|------|----------------|-----------------|\\n")

            mode_stats = defaultdict(lambda: {"runs": 0, "extra_tools": 0, "extra_agents": 0})
            for r in results:
                mode = r.config.get_mode_label()
                d = mode_stats[mode]
                d["runs"] += 1
                d["extra_tools"] += len(r.accuracy_metrics.tools_extra)
                d["extra_agents"] += len(r.accuracy_metrics.agents_extra)

            for mode in ["C-1", "C-2", "D", "ReAct"]:
                if mode in mode_stats:
                    d = mode_stats[mode]
                    n_runs = d["runs"]
                    f.write(f"| {mode} | {n_runs} | "
                            f"{d['extra_tools'] / n_runs:.1f} | "
                            f"{d['extra_agents'] / n_runs:.1f} |\\n")

            # Missing tools analysis (model failed to use required tools)
            f.write("\\n### Missing Tool Analysis\\n\\n")
            missing_counter = Counter()
            runs_with_missing = 0
            for r in results:
                missing = r.accuracy_metrics.tools_missing
                if missing:
                    runs_with_missing += 1
                    for t in missing:
                        missing_counter[t] += 1

            if runs_with_missing > 0:
                f.write(f"Runs with missing required tools: "
                        f"{runs_with_missing}/{n} ({runs_with_missing / n * 100:.1f}%)\\n\\n")
                f.write("| Missing Tool | Count | % of Runs |\\n")
                f.write("|-------------|-------|-----------|\\n")
                for tool, count in missing_counter.most_common(10):
                    f.write(f"| {tool} | {count} | {count / n * 100:.1f}% |\\n")
            else:
                f.write("No runs had missing required tools.\\n")

        print(f"Report saved to: {report_file}")


async def main():
    from API import API_KEY

    ROOT = Path(__file__).resolve().parent

    suite = EnhancedBenchmarkSuite(
        mcp_server_path=str(ROOT / "src" / "mcp_center" / "mcp_server.py"),
        config_dir=ROOT / "config",
        output_dir=ROOT / "benchmark_results",
        openai_api_key=API_KEY,
        num_repeats=1,
    )

    results = await suite.run_benchmark(
        categories=[
            TestCategory.SINGLE_AGENT_SINGLE_TOOL,
            TestCategory.SINGLE_AGENT_MULTI_TOOL,
            TestCategory.MULTI_AGENT_SINGLE_TOOL,
            TestCategory.MULTI_AGENT_MULTI_TOOL
        ],
        run_id="benchmark_eva",
        resume=True
    )

    print(f"\n✓ Completed {len(results)} benchmark tests")


if __name__ == "__main__":
    asyncio.run(main())