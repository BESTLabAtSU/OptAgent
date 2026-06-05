"""
Metrics Collection for Multi-Agent Benchmarking
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
from datetime import datetime


# ============================================================
# COST CONFIGURATION
# ============================================================

MODEL_COSTS = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "gpt-5.2":       {"input": 1.75,  "output": 14.00},

    "qwen3:30b": {"input": 0.0, "output": 0.0},
    "deepseek-r1:32b": {"input": 0.0, "output": 0.0},
    "gpt-oss:20b": {"input": 0.0, "output": 0.0},

    "qwen3:14b": {"input": 0.0, "output": 0.0},

    "gemma2:9b": {"input": 0.0, "output": 0.0},
    "deepseek-r1:8b": {"input": 0.0, "output": 0.0},
    "qwen3:8b": {"input": 0.0, "output": 0.0},
    "phi3:3.8b": {"input": 0.0, "output": 0.0},

    "gemma2:2b": {"input": 0.0, "output": 0.0},
    "llama3.2:3b": {"input": 0.0, "output": 0.0},
    "llama3.2:1b": {"input": 0.0, "output": 0.0},
    "qwen3:4b": {"input": 0.0, "output": 0.0},
    "qwen3:1.7b": {"input": 0.0, "output": 0.0},
    "qwen3:0.6b": {"input": 0.0, "output": 0.0},
    "deepseek-r1:1.5b": {"input": 0.0, "output": 0.0},

    "llama3.2": {"input": 0.0, "output": 0.0},
    "mistral": {"input": 0.0, "output": 0.0},
    "phi3": {"input": 0.0, "output": 0.0},
    "qwen2.5:7b": {"input": 0.0, "output": 0.0},
    "qwen2.5:3b": {"input": 0.0, "output": 0.0},
}


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost in USD for a given model and token counts"""
    costs = MODEL_COSTS.get(model, {"input": 0.0, "output": 0.0})
    input_cost = (prompt_tokens / 1_000_000) * costs["input"]
    output_cost = (completion_tokens / 1_000_000) * costs["output"]
    return input_cost + output_cost


# ============================================================
# DETAILED METRICS DATA CLASSES
# ============================================================
@dataclass
class TokenMetrics:
    """Token usage breakdown"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    def add(self, prompt: int, completion: int):
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion


@dataclass
class TimingMetrics:
    """Timing breakdown"""
    planning_time: float = 0.0       # Orchestrator planning
    execution_time: float = 0.0      # Agent execution
    synthesis_time: float = 0.0      # Response synthesis
    total_time: float = 0.0
    
    def finalize(self, agent_check_time: float = 0.0):
        self.total_time = (
            agent_check_time +
            self.planning_time + 
            self.execution_time + 
            self.synthesis_time
        )


@dataclass
class AgentMetrics:
    """Per-agent detailed metrics"""
    agent_id: str
    model: str
    tokens: TokenMetrics = field(default_factory=TokenMetrics)
    cost: float = 0.0
    execution_time: float = 0.0
    tool_calls: int = 0
    tools_used: List[str] = field(default_factory=list)
    planning_tokens: int = 0      # Tokens for tool planning
    synthesis_tokens: int = 0     # Tokens for response synthesis
    reasoning_trace: List[Dict] = field(default_factory=list)  # NEW: Store reasoning
    
    def calculate_cost(self):
        self.cost = calculate_cost(
            self.model, 
            self.tokens.prompt_tokens, 
            self.tokens.completion_tokens
        )


@dataclass
class OrchestratorMetrics:
    """Orchestrator-specific metrics"""
    model: str = ""
    tokens: TokenMetrics = field(default_factory=TokenMetrics)
    cost: float = 0.0
    planning_time: float = 0.0
    synthesis_time: float = 0.0
    agent_check_time: float = 0.0  # Dynamic mode: time to check/create agents
    agent_modifications: List[Dict] = field(default_factory=list)
    execution_plan: List[Dict] = field(default_factory=list)
    reasoning_trace: List[Dict] = field(default_factory=list)  # NEW
    
    def calculate_cost(self):
        self.cost = calculate_cost(
            self.model,
            self.tokens.prompt_tokens,
            self.tokens.completion_tokens
        )


@dataclass 
class ComprehensiveMetrics:
    """Complete metrics for a benchmark run"""
    # Configuration
    intelligence_mode: str = ""
    agent_generation_mode: str = ""
    orchestrator_model: str = ""
    agent_model: str = ""
    provider: str = ""
    two_stage_planning: bool = False

    # High-level metrics
    timing: TimingMetrics = field(default_factory=TimingMetrics)
    total_tokens: int = 0
    total_cost: float = 0.0
    total_tool_calls: int = 0
    
    # Component breakdown
    orchestrator: OrchestratorMetrics = field(default_factory=OrchestratorMetrics)
    agents: Dict[str, AgentMetrics] = field(default_factory=dict)
    
    # Quality metrics
    success: bool = False
    accuracy_score: float = 0.0
    completeness_score: float = 0.0  # NEW: How much of the task was completed
    
    # Reasoning traces (for paper)
    full_reasoning_trace: List[Dict] = field(default_factory=list)
    
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def finalize(self):
        """Calculate aggregate metrics"""
        self.timing.finalize(self.orchestrator.agent_check_time)
        self.orchestrator.calculate_cost()
        
        # Aggregate from agents
        agent_tokens = 0
        agent_cost = 0.0
        for agent in self.agents.values():
            agent.calculate_cost()
            agent_tokens += agent.tokens.total_tokens
            agent_cost += agent.cost
            self.total_tool_calls += agent.tool_calls
        
        # Totals
        self.total_tokens = self.orchestrator.tokens.total_tokens + agent_tokens
        self.total_cost = self.orchestrator.cost + agent_cost
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "configuration": {
                "intelligence_mode": self.intelligence_mode,
                "agent_generation_mode": self.agent_generation_mode,
                "orchestrator_model": self.orchestrator_model,
                "agent_model": self.agent_model,
                "provider": self.provider,
            },
            "timing": {
                "agent_check_time": self.orchestrator.agent_check_time,
                "planning_time": self.timing.planning_time,
                "execution_time": self.timing.execution_time,
                "synthesis_time": self.timing.synthesis_time,
                "total_time": self.timing.total_time,
            },
            "tokens": {
                "orchestrator_prompt": self.orchestrator.tokens.prompt_tokens,
                "orchestrator_completion": self.orchestrator.tokens.completion_tokens,
                "orchestrator_total": self.orchestrator.tokens.total_tokens,
                "agents_total": sum(a.tokens.total_tokens for a in self.agents.values()),
                "grand_total": self.total_tokens,
            },
            "cost": {
                "orchestrator": self.orchestrator.cost,
                "agents": sum(a.cost for a in self.agents.values()),
                "total": self.total_cost,
            },
            "tool_calls": self.total_tool_calls,
            "quality": {
                "success": self.success,
                "accuracy": self.accuracy_score,
                "completeness": self.completeness_score,
            },
            "agents": {
                aid: {
                    "model": a.model,
                    "tokens": a.tokens.total_tokens,
                    "cost": a.cost,
                    "time": a.execution_time,
                    "tool_calls": a.tool_calls,
                    "tools": a.tools_used,
                }
                for aid, a in self.agents.items()
            },
            "reasoning_trace": self.full_reasoning_trace,
            "timestamp": self.timestamp,
        }


# ============================================================
# METRICS COLLECTOR (to use in Orchestrator/Agent)
# ============================================================
class MetricsCollector:
    """Helper class to collect metrics during execution"""
    
    def __init__(self):
        self.metrics = ComprehensiveMetrics()
        self._step_start_times: Dict[str, float] = {}

    def set_config(self, intelligence_mode: str, agent_gen_mode: str,
                   orchestrator_model: str, agent_model: str, provider: str,
                   two_stage_planning: bool = False):
        self.metrics.intelligence_mode = intelligence_mode
        self.metrics.agent_generation_mode = agent_gen_mode
        self.metrics.orchestrator_model = orchestrator_model
        self.metrics.agent_model = agent_model
        self.metrics.provider = provider
        self.metrics.orchestrator.model = orchestrator_model
        self.metrics.two_stage_planning = two_stage_planning
    
    def start_phase(self, phase: str):
        """Start timing a phase"""
        import time
        self._step_start_times[phase] = time.time()
    
    def end_phase(self, phase: str) -> float:
        """End timing a phase and return duration"""
        import time
        start = self._step_start_times.get(phase, time.time())
        duration = time.time() - start
        
        if phase == "planning":
            self.metrics.timing.planning_time = duration
            self.metrics.orchestrator.planning_time = duration
        elif phase == "execution":
            self.metrics.timing.execution_time = duration
        elif phase == "synthesis":
            self.metrics.timing.synthesis_time = duration
            self.metrics.orchestrator.synthesis_time = duration
        elif phase == "agent_check":
            self.metrics.timing.agent_check_time = duration
            self.metrics.orchestrator.agent_check_time = duration
            
        return duration
    
    def record_orchestrator_llm_call(self, response, phase: str = ""):
        """Record an orchestrator LLM call"""
        self.metrics.orchestrator.tokens.add(
            response.prompt_tokens,
            response.completion_tokens
        )
        # Add to reasoning trace
        self.metrics.orchestrator.reasoning_trace.append({
            "phase": phase,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "model": response.model,
        })
    
    def record_agent_result(self, agent_id: str, model: str, result: Dict[str, Any]):
        """Record results from an agent execution"""
        if agent_id not in self.metrics.agents:
            self.metrics.agents[agent_id] = AgentMetrics(agent_id=agent_id, model=model)
        
        agent = self.metrics.agents[agent_id]
        metrics = result.get("metrics", {})
        
        agent.tokens.add(
            metrics.get("prompt_tokens", 0),
            metrics.get("completion_tokens", metrics.get("tokens_used", 0))
        )
        agent.execution_time += metrics.get("execution_time", 0)
        agent.tool_calls += metrics.get("tool_calls", 0)
        
        # Record tools used
        if "tools_used" in result:
            for tool in result["tools_used"]:
                tool_name = tool.get("tool") if isinstance(tool, dict) else str(tool)
                if tool_name and tool_name not in agent.tools_used:
                    agent.tools_used.append(tool_name)
        
        # Record reasoning if available
        if "reasoning" in result:
            agent.reasoning_trace.append(result["reasoning"])
    
    def add_reasoning_step(self, step_type: str, content: Dict[str, Any]):
        """Add a reasoning step to the trace"""
        self.metrics.full_reasoning_trace.append({
            "type": step_type,
            "timestamp": datetime.now().isoformat(),
            **content
        })
    
    def set_quality_metrics(self, success: bool, accuracy: float, completeness: float = 0.0):
        self.metrics.success = success
        self.metrics.accuracy_score = accuracy
        self.metrics.completeness_score = completeness
    
    def finalize(self) -> ComprehensiveMetrics:
        self.metrics.finalize()
        return self.metrics