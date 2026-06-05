"""
Comprehensive Benchmarking Suite for Multi-Agent System
UPDATED: Enhanced reporting with detailed comparisons
"""
import asyncio
import json
from pathlib import Path

from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
import logging
from itertools import product

from src.core.data_class import IntelligenceMode, AgentGenerationMode
from src.orchestrator.orchestrator import Orchestrator
from src.llm.llm_interface import create_llm_client, LLMProvider

from TESTCASE import (
    TestCategory, ExpectedStep, BenchmarkTestCase,
    get_all_test_cases, get_test_cases_by_category,
    get_test_case_by_id, get_test_cases_summary,
)


@dataclass
class AccuracyMetrics:
    """Detailed accuracy metrics for benchmark evaluation"""
    tool_selection_accuracy: float = 0.0
    tools_expected: List[str] = field(default_factory=list)
    tools_actual: List[str] = field(default_factory=list)
    tools_correct: List[str] = field(default_factory=list)
    tools_missing: List[str] = field(default_factory=list)
    tools_extra: List[str] = field(default_factory=list)

    agent_selection_accuracy: float = 0.0
    agents_expected: List[str] = field(default_factory=list)
    agents_actual: List[str] = field(default_factory=list)
    agents_correct: List[str] = field(default_factory=list)
    agents_missing: List[str] = field(default_factory=list)
    agents_extra: List[str] = field(default_factory=list)

    plan_step_accuracy: float = 0.0
    expected_step_count: int = 0
    matched_step_count: int = 0
    plan_sequence_details: List[Dict] = field(default_factory=list)

    # NEW: Store raw expected/actual sequences for detailed reporting
    expected_plan_sequence: List[Dict] = field(default_factory=list)
    actual_plan_sequence: List[Dict] = field(default_factory=list)

    parameter_key_accuracy: float = 0.0
    parameter_value_accuracy: float = 0.0
    parameter_details: List[Dict] = field(default_factory=list)


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

    def get_mode_label(self) -> str:
        """Get short mode label: C-1, C-2, or D"""
        if self.intelligence_mode == IntelligenceMode.CENTRALIZED:
            return "C-2" if self.two_stage_planning else "C-1"
        return "D"


@dataclass
class BenchmarkResult:
    """Comprehensive result of a single benchmark run"""
    config: 'BenchmarkConfig'
    category: str

    # Timing
    total_time: float = 0.0
    planning_time: float = 0.0
    execution_time: float = 0.0
    synthesis_time: float = 0.0

    # Tokens
    orchestrator_prompt_tokens: int = 0
    orchestrator_completion_tokens: int = 0
    orchestrator_total_tokens: int = 0
    agent_total_tokens: int = 0
    grand_total_tokens: int = 0

    # Cost
    orchestrator_cost: float = 0.0
    agent_cost: float = 0.0
    total_cost: float = 0.0

    # Tool usage
    tool_calls: int = 0
    tools_used: List[str] = field(default_factory=list)

    # Agent details
    agents_used: List[str] = field(default_factory=list)
    agent_metrics: Dict[str, Dict] = field(default_factory=dict)
    agent_modifications: List[Dict] = field(default_factory=list)

    # Quality
    success: bool = False
    accuracy_metrics: AccuracyMetrics = field(default_factory=AccuracyMetrics)

    # Plan
    execution_plan: List[Dict] = field(default_factory=list)

    # Reasoning
    reasoning_trace: List[Dict] = field(default_factory=list)
    orchestrator_reasoning: str = ""

    # Response
    response_summary: str = ""
    error: str = None
    timestamp: str = ""

    # Two-stage specific
    two_stage_planning: bool = False
    planning_llm_calls: int = 0

    @classmethod
    def from_comprehensive_metrics(cls, config: 'BenchmarkConfig',
                                   category: str,
                                   metrics: Dict[str, Any],
                                   execution_plan: List[Dict] = None) -> 'BenchmarkResult':
        """Create from ComprehensiveMetrics.to_dict() output"""
        timing = metrics.get("timing", {})
        tokens = metrics.get("tokens", {})
        cost = metrics.get("cost", {})
        quality = metrics.get("quality", {})
        agents = metrics.get("agents", {})

        reasoning_trace = metrics.get("reasoning_trace", [])
        planning_calls = sum(1 for step in reasoning_trace
                           if "planning" in step.get("step", "").lower())

        return cls(
            config=config,
            category=category,
            execution_plan=execution_plan or [],
            total_time=timing.get("total_time", 0),
            planning_time=timing.get("planning_time", 0),
            execution_time=timing.get("execution_time", 0),
            synthesis_time=timing.get("synthesis_time", 0),
            orchestrator_prompt_tokens=tokens.get("orchestrator_prompt", 0),
            orchestrator_completion_tokens=tokens.get("orchestrator_completion", 0),
            orchestrator_total_tokens=tokens.get("orchestrator_total", 0),
            agent_total_tokens=tokens.get("agents_total", 0),
            grand_total_tokens=tokens.get("grand_total", 0),
            orchestrator_cost=cost.get("orchestrator", 0),
            agent_cost=cost.get("agents", 0),
            total_cost=cost.get("total", 0),
            tool_calls=metrics.get("tool_calls", 0),
            tools_used=list(set(
                tool for a in agents.values() for tool in a.get("tools", [])
            )),
            agents_used=list(agents.keys()),
            agent_metrics=agents,
            success=quality.get("success", False),
            reasoning_trace=reasoning_trace,
            timestamp=metrics.get("timestamp", ""),
            two_stage_planning=config.two_stage_planning,
            planning_llm_calls=planning_calls
        )


class BenchmarkSuite:
    """Main benchmarking suite with two-stage planning support"""

    def __init__(self,
                 mcp_server_path: str,
                 config_dir: Path = Path("../config"),
                 output_dir: Path = Path("../benchmark_results"),
                 log_level: str = "INFO",
                 openai_api_key: str = None):
        self.mcp_server_path = mcp_server_path
        self.config_dir = config_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.openai_api_key = openai_api_key

        logging.basicConfig(level=log_level)
        self.logger = logging.getLogger("BenchmarkSuite")

        self.results: List[BenchmarkResult] = []
        self.test_cases = get_all_test_cases()
        self.configurations = self._create_configurations()

        summary = get_test_cases_summary()
        self.logger.info(f"Loaded {summary['total_cases']} test cases: {summary['by_category']}")

    def _create_configurations(self) -> list:
        """Create benchmark configurations including two_stage_planning variants"""
        intelligence_modes = [IntelligenceMode.CENTRALIZED, IntelligenceMode.DECENTRALIZED]
        agent_gen_modes = [AgentGenerationMode.STATIC]
        two_stage_options = [False, True]

        model_configs = [
            # {"provider": "openai", "orchestrator": "gpt-4o-mini", "agent": "gpt-4o-mini"},  # ceiling
            {"provider": "ollama", "orchestrator": "gemma2:9b", "agent": "gemma2:9b"},  # mid baseline
            # {"provider": "ollama", "orchestrator": "qwen3:4b", "agent": "qwen3:4b"},  # small baseline
            #
            # {"provider": "mixed", "orchestrator": "gpt-4o-mini", "agent": "qwen3:1.7b"},  # smart orch + weak agent
            # {"provider": "ollama", "orchestrator": "gemma2:9b", "agent": "qwen3:1.7b"},  # mid orch + weak agent
            #
            # {"provider": "mixed", "orchestrator": "qwen3:1.7b", "agent": "gpt-4o-mini"},  # weak orch + smart agent
            # {"provider": "ollama", "orchestrator": "qwen3:1.7b", "agent": "gemma2:9b"},  # weak orch + mid agent

        ]

        configurations = []
        for intel_mode, agent_mode, model_config, two_stage in product(
                intelligence_modes, agent_gen_modes, model_configs, two_stage_options
        ):
            if two_stage and intel_mode != IntelligenceMode.CENTRALIZED:
                continue

            configurations.append({
                "intelligence_mode": intel_mode,
                "agent_generation_mode": agent_mode,
                "orchestrator_model": model_config["orchestrator"],
                "agent_model": model_config["agent"],
                "two_stage_planning": two_stage,
                **model_config
            })

        return configurations

    def _calculate_tool_selection_accuracy(self, expected_tools: List[str],
                                           actual_tools: List[str]) -> Tuple[float, Dict]:
        expected_set = set(expected_tools)
        actual_set = set(actual_tools)
        correct = expected_set & actual_set
        missing = expected_set - actual_set
        extra = actual_set - expected_set
        if not expected_set:
            accuracy = 1.0 if not actual_set else 0.0
        else:
            accuracy = len(correct) / len(expected_set)
        return accuracy, {
            "expected": list(expected_set), "actual": list(actual_set),
            "correct": list(correct), "missing": list(missing), "extra": list(extra)
        }

    def _calculate_agent_selection_accuracy(self, expected_agents: List[str],
                                            actual_agents: List[str]) -> Tuple[float, Dict]:
        expected_set = set(expected_agents)
        actual_set = set(actual_agents)
        correct = expected_set & actual_set
        missing = expected_set - actual_set
        extra = actual_set - expected_set
        if not expected_set:
            accuracy = 1.0 if not actual_set else 0.0
        else:
            accuracy = len(correct) / len(expected_set)
        return accuracy, {
            "expected": list(expected_set), "actual": list(actual_set),
            "correct": list(correct), "missing": list(missing), "extra": list(extra)
        }

    def _calculate_plan_step_accuracy(self, test_case: BenchmarkTestCase,
                                      execution_plan: List[Dict],
                                      agent_results: Dict[str, Any] = None) -> Tuple[float, Dict]:
        expected_sequence = test_case.get_expected_step_sequence()
        actual_sequence = []
        for step in sorted(execution_plan, key=lambda s: s.get("step_id", "")):
            agent_id = step.get("agent_id", "")
            step_id = step.get("step_id", "")
            tools = []
            guidance = step.get("orchestrator_guidance", {})
            if guidance:
                tool_instructions = guidance.get("tool_instructions", [])
                tools = [t.get("tool", "") for t in tool_instructions if t.get("tool")]
            if not tools and agent_results and step_id in agent_results:
                result = agent_results[step_id]
                if isinstance(result, dict):
                    tools_used = result.get("tools_used", [])
                    tools = [t.get("tool", "") for t in tools_used
                             if isinstance(t, dict) and t.get("tool")]
            actual_sequence.append((agent_id, tools))

        expected_count = len(expected_sequence)

        # Build expected/actual plan lists for detailed reporting
        expected_plan_list = [{"step": i+1, "agent": a, "tools": t}
                             for i, (a, t) in enumerate(expected_sequence)]
        actual_plan_list = [{"step": i+1, "agent": a, "tools": t}
                           for i, (a, t) in enumerate(actual_sequence)]

        if expected_count == 0:
            return 1.0, {
                "expected_steps": expected_plan_list,
                "actual_steps": actual_plan_list,
                "matched_count": 0, "expected_count": 0,
                "details": [], "is_valid_subsequence": True
            }

        matched_count = 0
        expected_idx = 0
        details = []
        match_positions = []

        for actual_idx, (act_agent, act_tools) in enumerate(actual_sequence):
            if expected_idx >= expected_count:
                break
            exp_agent, exp_tools = expected_sequence[expected_idx]
            agent_match = (exp_agent == act_agent)
            tools_match = set(exp_tools) <= set(act_tools)
            step_match = agent_match and tools_match
            if step_match:
                details.append({
                    "expected_position": expected_idx + 1, "actual_position": actual_idx + 1,
                    "expected_agent": exp_agent, "actual_agent": act_agent,
                    "expected_tools": exp_tools, "actual_tools": act_tools,
                    "agent_match": True, "tools_match": True, "step_match": True
                })
                match_positions.append(actual_idx)
                matched_count += 1
                expected_idx += 1

        for i in range(expected_idx, expected_count):
            exp_agent, exp_tools = expected_sequence[i]
            details.append({
                "expected_position": i + 1, "actual_position": None,
                "expected_agent": exp_agent, "actual_agent": None,
                "expected_tools": exp_tools, "actual_tools": [],
                "agent_match": False, "tools_match": False, "step_match": False
            })

        accuracy = matched_count / expected_count
        return accuracy, {
            "expected_steps": expected_plan_list,
            "actual_steps": actual_plan_list,
            "matched_count": matched_count, "expected_count": expected_count,
            "is_valid_subsequence": matched_count == expected_count,
            "match_positions": match_positions, "details": details
        }

    def _extract_actual_parameters_by_tool(self, agent_results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        tool_parameters = {}
        for step_id, result in agent_results.items():
            if not isinstance(result, dict):
                continue
            for tool_info in result.get("tools_used", []):
                tool_name = tool_info.get("tool", "")
                parameters = tool_info.get("parameters", {})
                if tool_name and tool_name not in tool_parameters:
                    tool_parameters[tool_name] = parameters
        return tool_parameters

    def _values_match(self, expected: Any, actual: Any) -> bool:
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

    def _calculate_parameter_accuracy(self, test_case: BenchmarkTestCase,
                                      agent_results: Dict[str, Any]) -> Tuple[float, float, List[Dict]]:
        actual_params_by_tool = self._extract_actual_parameters_by_tool(agent_results)
        expected_params_by_tool = {}
        for step in test_case.expected_steps:
            if hasattr(step, 'expected_parameters') and step.expected_parameters:
                for tool_name, params in step.expected_parameters.items():
                    if params:
                        expected_params_by_tool[tool_name] = params

        if not expected_params_by_tool:
            return 1.0, 1.0, []

        details = []
        total_keys = correct_keys = correct_values = 0

        for tool_name, expected_params in expected_params_by_tool.items():
            actual_params = actual_params_by_tool.get(tool_name, {})
            tool_total_keys = len(expected_params)
            tool_correct_keys = tool_correct_values = 0
            key_results = {}

            for key, expected_value in expected_params.items():
                total_keys += 1
                key_exists = key in actual_params
                if key_exists:
                    tool_correct_keys += 1
                    correct_keys += 1
                    actual_value = actual_params[key]
                    value_match = self._values_match(expected_value, actual_value)
                    if value_match:
                        tool_correct_values += 1
                        correct_values += 1
                    key_results[key] = {"expected": expected_value, "actual": actual_value,
                                       "key_correct": True, "value_correct": value_match}
                else:
                    key_results[key] = {"expected": expected_value, "actual": None,
                                       "key_correct": False, "value_correct": False}

            details.append({
                "tool": tool_name, "expected_params": expected_params,
                "actual_params": actual_params,
                "key_accuracy": tool_correct_keys / tool_total_keys if tool_total_keys else 0,
                "value_accuracy": tool_correct_values / tool_total_keys if tool_total_keys else 0,
                "total_keys": tool_total_keys, "correct_keys": tool_correct_keys,
                "correct_values": tool_correct_values, "key_results": key_results
            })

        return (correct_keys / total_keys if total_keys else 1.0,
                correct_values / total_keys if total_keys else 1.0, details)

    def _calculate_accuracy_metrics(self, test_case: BenchmarkTestCase,
                                    agents_used: List[str], tools_used: List[str],
                                    execution_plan: List[Dict],
                                    agent_results: Dict[str, Any]) -> AccuracyMetrics:
        tool_acc, tool_details = self._calculate_tool_selection_accuracy(test_case.expected_tools, tools_used)
        agent_acc, agent_details = self._calculate_agent_selection_accuracy(test_case.expected_agents, agents_used)
        plan_acc, plan_details = self._calculate_plan_step_accuracy(test_case, execution_plan, agent_results)
        param_key_acc, param_value_acc, param_details = self._calculate_parameter_accuracy(test_case, agent_results)

        return AccuracyMetrics(
            tool_selection_accuracy=tool_acc, tools_expected=tool_details["expected"],
            tools_actual=tool_details["actual"], tools_correct=tool_details["correct"],
            tools_missing=tool_details["missing"], tools_extra=tool_details["extra"],
            agent_selection_accuracy=agent_acc, agents_expected=agent_details["expected"],
            agents_actual=agent_details["actual"], agents_correct=agent_details["correct"],
            agents_missing=agent_details["missing"], agents_extra=agent_details["extra"],
            plan_step_accuracy=plan_acc, expected_step_count=plan_details.get("expected_count", 0),
            matched_step_count=plan_details.get("matched_count", 0),
            plan_sequence_details=plan_details.get("details", []),
            expected_plan_sequence=plan_details.get("expected_steps", []),
            actual_plan_sequence=plan_details.get("actual_steps", []),
            parameter_key_accuracy=param_key_acc, parameter_value_accuracy=param_value_acc,
            parameter_details=param_details,
        )

    async def run_single_benchmark(self, config: BenchmarkConfig,
                                   test_case: BenchmarkTestCase) -> BenchmarkResult:
        """Run benchmark with two_stage_planning option"""
        two_stage_str = " [2-stage]" if config.two_stage_planning else ""
        self.logger.info(
            f"Running: {test_case.name} | {config.provider} | "
            f"{config.intelligence_mode.value}{two_stage_str} | "
            f"{config.orchestrator_model}/{config.agent_model}"
        )

        llm_client = create_llm_client(
            provider=config.provider,
            model=config.orchestrator_model,
            api_key=self.openai_api_key if config.provider == "openai" else None,
            host="http://localhost:11434" if config.provider == "ollama" else None
        )

        orchestrator = Orchestrator(
            llm_client=llm_client,
            mcp_server_path=self.mcp_server_path,
            intelligence_mode=config.intelligence_mode,
            agent_generation_mode=config.agent_generation_mode,
            orchestrator_model=config.orchestrator_model,
            agent_model=config.agent_model,
            config_dir=self.config_dir,
            log_dir=Path(f"logs/benchmark_{config.run_id}"),
            use_two_stage_planning=config.two_stage_planning
        )

        try:
            await orchestrator.initialize()
            response = await orchestrator.process_request(test_case.request)

            benchmark_data = response.get("benchmark", {})
            metrics = benchmark_data.get("metrics", {})
            execution_plan = benchmark_data.get("execution_plan", [])

            result = BenchmarkResult.from_comprehensive_metrics(
                config=config, category=test_case.category.value,
                metrics=metrics, execution_plan=execution_plan
            )

            result.agent_modifications = benchmark_data.get("agent_modifications", [])
            result.response_summary = response.get("summary", "")
            result.accuracy_metrics = self._calculate_accuracy_metrics(
                test_case=test_case, agents_used=result.agents_used,
                tools_used=result.tools_used, execution_plan=execution_plan,
                agent_results=benchmark_data.get("agent_results", {})
            )

        except Exception as e:
            self.logger.error(f"Benchmark error: {e}")
            result = BenchmarkResult(config=config, category=test_case.category.value,
                                    success=False, error=str(e))
        finally:
            await orchestrator.shutdown()
            await llm_client.close()

        return result

    async def run_full_benchmark(self, categories: List[TestCategory] = None,
                                 configurations: List[Dict] = None) -> List[BenchmarkResult]:
        """Run complete benchmark suite"""
        categories = categories or list(TestCategory)
        configs = configurations or self.configurations

        total_tests = sum(len(self.test_cases.get(cat, [])) for cat in categories)
        total_runs = total_tests * len(configs)

        self.logger.info(f"Starting benchmark: {total_runs} runs across {len(categories)} categories")

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        current = 0

        for category in categories:
            self.logger.info(f"\n{'=' * 50}\nCategory: {category.value}\n{'=' * 50}")

            for test_case in self.test_cases.get(category, []):
                for config_dict in configs:
                    current += 1
                    self.logger.info(f"Progress: {current}/{total_runs}")

                    config = BenchmarkConfig(
                        intelligence_mode=config_dict["intelligence_mode"],
                        agent_generation_mode=config_dict["agent_generation_mode"],
                        orchestrator_model=config_dict["orchestrator_model"],
                        agent_model=config_dict["agent_model"],
                        test_case_id=test_case.test_id,
                        test_case_name=test_case.name,
                        run_id=run_id,
                        provider=config_dict["provider"],
                        two_stage_planning=config_dict.get("two_stage_planning", False)
                    )

                    result = await self.run_single_benchmark(config, test_case)
                    self.results.append(result)

                    if current % 10 == 0:
                        self._save_results(run_id)

        self._save_results(run_id)
        self._generate_report(run_id)
        return self.results

    def _save_results(self, run_id: str):
        """Save results to JSON"""
        results_file = self.output_dir / f"benchmark_results_{run_id}.json"
        data = []
        for r in self.results:
            d = asdict(r)
            d["config"]["intelligence_mode"] = r.config.intelligence_mode.value
            d["config"]["agent_generation_mode"] = r.config.agent_generation_mode.value
            data.append(d)
        with open(results_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def _generate_report(self, run_id: str):
        """Generate markdown report with detailed comparisons"""
        report_file = self.output_dir / f"benchmark_report_{run_id}.md"

        with open(report_file, 'w') as f:
            f.write("# Multi-Agent System Benchmark Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("**Mode Legend**: C-1 = Centralized Single-Stage, C-2 = Centralized Two-Stage, D = Decentralized\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")
            n = max(len(self.results), 1)
            f.write(f"- Total runs: {len(self.results)}\n")
            f.write(f"- Success rate: {sum(r.success for r in self.results) / n:.1%}\n")

            avg_tool = sum(r.accuracy_metrics.tool_selection_accuracy for r in self.results) / n
            avg_agent = sum(r.accuracy_metrics.agent_selection_accuracy for r in self.results) / n
            avg_plan = sum(r.accuracy_metrics.plan_step_accuracy for r in self.results) / n
            avg_param_key = sum(r.accuracy_metrics.parameter_key_accuracy for r in self.results) / n
            avg_param_val = sum(r.accuracy_metrics.parameter_value_accuracy for r in self.results) / n

            f.write(f"- **Avg Tool Selection Accuracy**: {avg_tool:.1%}\n")
            f.write(f"- **Avg Agent Selection Accuracy**: {avg_agent:.1%}\n")
            f.write(f"- **Avg Plan Step Accuracy**: {avg_plan:.1%}\n")
            f.write(f"- **Avg Parameter Key Accuracy**: {avg_param_key:.1%}\n")
            f.write(f"- **Avg Parameter Value Accuracy**: {avg_param_val:.1%}\n")
            f.write(f"- Avg tokens: {sum(r.grand_total_tokens for r in self.results) / n:.0f}\n")
            f.write(f"- Avg cost: ${sum(r.total_cost for r in self.results) / n:.4f}\n")
            f.write(f"- Avg time: {sum(r.total_time for r in self.results) / n:.2f}s\n\n")

            # Accuracy Breakdown with Mode Column
            f.write("## Accuracy Metrics Breakdown\n\n")

            f.write("### Tool Selection Accuracy\n\n")
            f.write("| Test | Mode | Expected | Actual | Correct | Missing | Extra | Accuracy |\n")
            f.write("|------|------|----------|--------|---------|---------|-------|----------|\n")
            for r in self.results:
                am = r.accuracy_metrics
                mode = r.config.get_mode_label()
                f.write(f"| {r.config.test_case_name[:25]} | {mode} | {len(am.tools_expected)} | ")
                f.write(f"{len(am.tools_actual)} | {len(am.tools_correct)} | ")
                f.write(f"{len(am.tools_missing)} | {len(am.tools_extra)} | {am.tool_selection_accuracy:.1%} |\n")

            f.write("\n### Agent Selection Accuracy\n\n")
            f.write("| Test | Mode | Expected | Actual | Correct | Missing | Extra | Accuracy |\n")
            f.write("|------|------|----------|--------|---------|---------|-------|----------|\n")
            for r in self.results:
                am = r.accuracy_metrics
                mode = r.config.get_mode_label()
                f.write(f"| {r.config.test_case_name[:25]} | {mode} | {len(am.agents_expected)} | ")
                f.write(f"{len(am.agents_actual)} | {len(am.agents_correct)} | ")
                f.write(f"{len(am.agents_missing)} | {len(am.agents_extra)} | {am.agent_selection_accuracy:.1%} |\n")

            f.write("\n### Plan Step Accuracy (Order Matters)\n\n")
            f.write("| Test | Mode | Expected Steps | Matched Steps | Accuracy |\n")
            f.write("|------|------|----------------|---------------|----------|\n")
            for r in self.results:
                am = r.accuracy_metrics
                mode = r.config.get_mode_label()
                f.write(f"| {r.config.test_case_name[:25]} | {mode} | {am.expected_step_count} | ")
                f.write(f"{am.matched_step_count} | {am.plan_step_accuracy:.1%} |\n")

            f.write("\n### Parameter Accuracy\n\n")
            f.write("| Test | Mode | Key Accuracy | Value Accuracy |\n")
            f.write("|------|------|--------------|----------------|\n")
            for r in self.results:
                am = r.accuracy_metrics
                mode = r.config.get_mode_label()
                f.write(f"| {r.config.test_case_name[:25]} | {mode} | {am.parameter_key_accuracy:.1%} | ")
                f.write(f"{am.parameter_value_accuracy:.1%} |\n")

            # NEW: Detailed Comparison Tables
            f.write("\n---\n")
            f.write("## Detailed Expected vs Actual Comparisons\n\n")

            # Group results by test case
            test_case_groups = {}
            for r in self.results:
                tc_name = r.config.test_case_name
                if tc_name not in test_case_groups:
                    test_case_groups[tc_name] = []
                test_case_groups[tc_name].append(r)

            for tc_name, tc_results in test_case_groups.items():
                f.write(f"### Test Case: {tc_name}\n\n")

                # Get expected values from first result (same for all modes)
                first_result = tc_results[0]
                am = first_result.accuracy_metrics

                # Expected Tools & Agents (same for all modes)
                f.write("#### Expected Configuration\n\n")
                f.write(f"- **Expected Agents**: {', '.join(am.agents_expected) or 'None'}\n")
                f.write(f"- **Expected Tools**: {', '.join(am.tools_expected) or 'None'}\n")

                # Expected Plan
                if am.expected_plan_sequence:
                    f.write("\n**Expected Plan Sequence:**\n\n")
                    f.write("| Step | Agent | Tools |\n")
                    f.write("|------|-------|-------|\n")
                    for step in am.expected_plan_sequence:
                        tools_str = ', '.join(step['tools']) if step['tools'] else '-'
                        f.write(f"| {step['step']} | {step['agent']} | {tools_str} |\n")
                f.write("\n")

                # Actual Results by Mode
                f.write("#### Actual Results by Mode\n\n")

                # Agent Comparison Table
                f.write("**Agents:**\n\n")
                f.write("| Mode | Actual Agents | Correct | Missing | Extra |\n")
                f.write("|------|---------------|---------|---------|-------|\n")
                for r in tc_results:
                    am = r.accuracy_metrics
                    mode = r.config.get_mode_label()
                    actual = ', '.join(am.agents_actual) or '-'
                    correct = ', '.join(am.agents_correct) or '-'
                    missing = ', '.join(am.agents_missing) or '-'
                    extra = ', '.join(am.agents_extra) or '-'
                    f.write(f"| {mode} | {actual} | {correct} | {missing} | {extra} |\n")
                f.write("\n")

                # Tool Comparison Table
                f.write("**Tools:**\n\n")
                f.write("| Mode | Actual Tools | Correct | Missing | Extra |\n")
                f.write("|------|--------------|---------|---------|-------|\n")
                for r in tc_results:
                    am = r.accuracy_metrics
                    mode = r.config.get_mode_label()
                    actual = ', '.join(am.tools_actual) or '-'
                    correct = ', '.join(am.tools_correct) or '-'
                    missing = ', '.join(am.tools_missing) or '-'
                    extra = ', '.join(am.tools_extra) or '-'
                    f.write(f"| {mode} | {actual} | {correct} | {missing} | {extra} |\n")
                f.write("\n")

                # Plan Comparison - Show actual plan for each mode
                f.write("**Plan Sequences:**\n\n")
                for r in tc_results:
                    am = r.accuracy_metrics
                    mode = r.config.get_mode_label()
                    f.write(f"*Mode {mode}* (Accuracy: {am.plan_step_accuracy:.1%}):\n\n")

                    if am.actual_plan_sequence:
                        f.write("| Step | Agent | Tools | Match |\n")
                        f.write("|------|-------|-------|-------|\n")

                        # Create lookup for expected steps
                        expected_lookup = {s['step']: s for s in am.expected_plan_sequence}

                        for step in am.actual_plan_sequence:
                            tools_str = ', '.join(step['tools']) if step['tools'] else '-'
                            step_num = step['step']

                            # Check if this step matches expected
                            match = "✗"
                            if step_num <= len(am.expected_plan_sequence):
                                exp_step = expected_lookup.get(step_num)
                                if exp_step:
                                    agent_ok = exp_step['agent'] == step['agent']
                                    tools_ok = set(exp_step['tools']) <= set(step['tools'])
                                    if agent_ok and tools_ok:
                                        match = "✓"
                                    elif agent_ok:
                                        match = "agent✓"
                                    elif tools_ok:
                                        match = "tools✓"

                            f.write(f"| {step_num} | {step['agent']} | {tools_str} | {match} |\n")
                    else:
                        f.write("*No steps executed*\n")
                    f.write("\n")

                f.write("---\n\n")

            # Parameter Details
            f.write("## Parameter Details by Tool\n\n")
            for r in self.results:
                if r.accuracy_metrics.parameter_details:
                    mode = r.config.get_mode_label()
                    f.write(f"### {r.config.test_case_name} [{mode}]\n\n")
                    for detail in r.accuracy_metrics.parameter_details:
                        f.write(f"**Tool: `{detail['tool']}`**\n")
                        f.write(f"- Key Accuracy: {detail['key_accuracy']:.1%} ({detail['correct_keys']}/{detail['total_keys']})\n")
                        f.write(f"- Value Accuracy: {detail['value_accuracy']:.1%} ({detail['correct_values']}/{detail['total_keys']})\n\n")

                        if detail.get('key_results'):
                            f.write("| Parameter | Expected | Actual | Status |\n")
                            f.write("|-----------|----------|--------|--------|\n")
                            for key, res in detail['key_results'].items():
                                status = "✓" if res['value_correct'] else ("key✓ val✗" if res['key_correct'] else "✗")
                                f.write(f"| `{key}` | `{res['expected']}` | `{res['actual']}` | {status} |\n")
                        f.write("\n")

            # Mode Comparison Summary
            f.write("## Mode Comparison Summary\n\n")

            single_stage = [r for r in self.results if r.config.get_mode_label() == "C-1"]
            two_stage = [r for r in self.results if r.config.get_mode_label() == "C-2"]
            decentralized = [r for r in self.results if r.config.get_mode_label() == "D"]

            f.write("| Metric | C-1 | C-2 | D |\n")
            f.write("|--------|-----|-----|---|\n")

            for metric_name, getter in [
                ("Tests", lambda rs: str(len(rs))),
                ("Success Rate", lambda rs: f"{sum(r.success for r in rs) / max(len(rs),1):.1%}"),
                ("Tool Acc", lambda rs: f"{sum(r.accuracy_metrics.tool_selection_accuracy for r in rs) / max(len(rs),1):.1%}"),
                ("Agent Acc", lambda rs: f"{sum(r.accuracy_metrics.agent_selection_accuracy for r in rs) / max(len(rs),1):.1%}"),
                ("Plan Acc", lambda rs: f"{sum(r.accuracy_metrics.plan_step_accuracy for r in rs) / max(len(rs),1):.1%}"),
                ("Param Key Acc", lambda rs: f"{sum(r.accuracy_metrics.parameter_key_accuracy for r in rs) / max(len(rs),1):.1%}"),
                ("Param Val Acc", lambda rs: f"{sum(r.accuracy_metrics.parameter_value_accuracy for r in rs) / max(len(rs),1):.1%}"),
                ("Avg Tokens", lambda rs: f"{sum(r.grand_total_tokens for r in rs) / max(len(rs),1):.0f}"),
                ("Avg Time", lambda rs: f"{sum(r.total_time for r in rs) / max(len(rs),1):.2f}s"),
            ]:
                c1_val = getter(single_stage) if single_stage else "-"
                c2_val = getter(two_stage) if two_stage else "-"
                d_val = getter(decentralized) if decentralized else "-"
                f.write(f"| {metric_name} | {c1_val} | {c2_val} | {d_val} |\n")

            f.write("\n")

            # Detailed Results Table
            f.write("## Detailed Results\n\n")
            f.write("| Test | Category | Mode | Tool Acc | Agent Acc | Plan Acc | Key Acc | Val Acc | Tokens | Time |\n")
            f.write("|------|----------|------|----------|-----------|----------|---------|---------|--------|------|\n")
            for r in self.results:
                am = r.accuracy_metrics
                mode = r.config.get_mode_label()
                f.write(f"| {r.config.test_case_name[:20]} | {r.category[:15]} | {mode} | ")
                f.write(f"{am.tool_selection_accuracy:.1%} | {am.agent_selection_accuracy:.1%} | ")
                f.write(f"{am.plan_step_accuracy:.1%} | {am.parameter_key_accuracy:.1%} | ")
                f.write(f"{am.parameter_value_accuracy:.1%} | {r.grand_total_tokens} | {r.total_time:.2f}s |\n")

        self.logger.info(f"Report saved: {report_file}")

    def _write_mode_section(self, f, title: str, results: List[BenchmarkResult]):
        """Helper to write a mode section with full metrics"""
        cn = len(results)
        f.write(f"### {title}\n\n")
        f.write(f"| Metric | Value |\n|--------|-------|\n")
        f.write(f"| Tests | {cn} |\n")
        f.write(f"| Success Rate | {sum(r.success for r in results) / cn:.1%} |\n")
        f.write(f"| Tool Selection Acc | {sum(r.accuracy_metrics.tool_selection_accuracy for r in results) / cn:.1%} |\n")
        f.write(f"| Agent Selection Acc | {sum(r.accuracy_metrics.agent_selection_accuracy for r in results) / cn:.1%} |\n")
        f.write(f"| Plan Step Acc | {sum(r.accuracy_metrics.plan_step_accuracy for r in results) / cn:.1%} |\n")
        f.write(f"| Param Key Acc | {sum(r.accuracy_metrics.parameter_key_accuracy for r in results) / cn:.1%} |\n")
        f.write(f"| Param Value Acc | {sum(r.accuracy_metrics.parameter_value_accuracy for r in results) / cn:.1%} |\n")
        f.write(f"| Avg Tokens | {sum(r.grand_total_tokens for r in results) / cn:.0f} |\n")
        f.write(f"| Avg Cost | ${sum(r.total_cost for r in results) / cn:.4f} |\n")
        f.write(f"| Avg Time | {sum(r.total_time for r in results) / cn:.2f}s |\n\n")


async def run_benchmark():
    """Example benchmark execution"""
    from API import API_KEY
    ROOT = Path(__file__).resolve().parent
    suite = BenchmarkSuite(
        mcp_server_path=str(ROOT / "src" / "mcp_center" / "mcp_server.py"),
        config_dir=ROOT / "config",
        output_dir=ROOT / "benchmark_results",
        openai_api_key=API_KEY
    )

    print(f"\nTest Case Summary: {get_test_cases_summary()}")

    results = await suite.run_full_benchmark(
        categories=[TestCategory.SINGLE_AGENT_SINGLE_TOOL,
                    TestCategory.SINGLE_AGENT_MULTI_TOOL,
                    TestCategory.MULTI_AGENT_SINGLE_TOOL,
                    TestCategory.MULTI_AGENT_MULTI_TOOL]
    )



    print(f"\nBenchmark complete! {len(results)} tests run.")
    print(f"Results saved to: {suite.output_dir}")


if __name__ == "__main__":
    asyncio.run(run_benchmark())