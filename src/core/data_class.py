import yaml
import json
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from dataclasses import dataclass, asdict
import json
from typing import Dict, Any, List


"""
Specialist Agent with MCP Integration and Response Logging
"""

import json
import time
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod


class IntelligenceMode(Enum):
    CENTRALIZED = "centralized"
    DECENTRALIZED = "decentralized"
    SINGLE_AGENT_REACT = "single_agent_react"

class FailureType(Enum):
    NONE = "none"
    PLANNING_FAILURE = "planning"          # Orchestrator produced invalid plan
    TOOL_EXECUTION_FAILURE = "tool_execution"  # MCP tool call failed
    RUNTIME_FAILURE = "runtime"            # Timeout, crash, OOM
    REASONING_FAILURE = "reasoning"        # Wrong tool/agent/params selected
    JSON_PARSE_FAILURE = "json_parse"      # LLM returned unparseable output
    CASCADE_FAILURE = "cascade"            # Failed because dependency failed

@dataclass
class AgentCard:
    """Agent identity and configuration"""
    agent_id: str
    name: str
    role: str
    description: str
    example_tasks: str
    available_tools: str
    model: str = "gpt-4"
    temperature: float = 0.3
    capabilities: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentCard':
        """Create from dictionary"""
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class SerializableResponse:
    """Base class for responses with JSON serialization"""

    def to_json(self) -> str:
        """Convert to JSON string with custom serialization"""

        def serialize(obj):
            if isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize(item) for item in obj]
            elif hasattr(obj, '__dict__'):
                return serialize(vars(obj))
            else:
                return obj

        # Get dict and clean it
        data = asdict(self)
        data = serialize(data)

        return json.dumps(data, indent=2, default=str)


@dataclass
class AgentResponse(SerializableResponse):
    """Structured agent response for logging"""
    agent_id: str
    timestamp: str
    request: Dict[str, Any]
    response: Dict[str, Any]
    tools_used: List[Dict[str, Any]]
    token_usage: Dict[str, int]
    execution_time: float
    mode: str
    model: str


@dataclass
class OrchestratorResponse(SerializableResponse):
    """Structured orchestrator response for logging"""
    timestamp: str
    request: str
    execution_plan: List[Dict[str, Any]]
    agent_results: Dict[str, Any]
    final_response: Dict[str, Any]
    metrics: Dict[str, Any]
    mode: str
    models: Dict[str, str]


class AgentGenerationMode(Enum):
    """How agents are managed during orchestration"""
    STATIC = "static"    # Use agents as defined in cards, no modifications
    DYNAMIC = "dynamic"  # Orchestrator can create/revise agents based on task needs


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


@dataclass
class BenchmarkResult:
    """Results from a benchmark run"""
    config: BenchmarkConfig
    execution_time: float
    orchestrator_tokens: int
    agent_tokens: int
    total_tokens: int
    tool_calls: int
    success: bool
    accuracy_score: float
    error: Optional[str] = None
    timestamp: str = None
    response_summary: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class BenchmarkTestCase:
    """Test case for benchmarking"""

    def __init__(self, test_id: str, name: str, request: str,
                 expected_tools: List[str] = None,
                 expected_output: Dict[str, Any] = None,
                 validation_criteria: Dict[str, Any] = None):
        self.test_id = test_id
        self.name = name
        self.request = request
        self.expected_tools = expected_tools or []
        self.expected_output = expected_output or {}
        self.validation_criteria = validation_criteria or {}