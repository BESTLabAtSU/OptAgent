"""
Agent Factory
"""
import yaml
import json
from pathlib import Path
import logging
from typing import Dict, Any, Optional, List

from ..core.data_class import IntelligenceMode, AgentCard
from .baseagent import SpecialistAgent


class AgentFactory:
    """Factory for creating agents from configuration files"""

    def __init__(self,
                 mcp_client,
                 llm_client,
                 config_dir: Path = Path("config/agents"),
                 response_log_dir: Path = Path("logs/agent_responses"),
                 default_model: str = "gpt-4",
                 intelligence_mode: IntelligenceMode = IntelligenceMode.DECENTRALIZED):
        self.mcp_client = mcp_client
        self.llm_client = llm_client
        self.config_dir = config_dir
        self.response_log_dir = response_log_dir
        self.default_model = default_model
        self.intelligence_mode = intelligence_mode

        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.response_log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("AgentFactory")

    async def initialize(self):
        """Initialize factory"""
        if not self.mcp_client.is_connected:
            raise RuntimeError("MCP client must be connected before initializing")
        self.logger.info("Agent factory initialized")

    async def create_agent(self, agent_id: str, model_override: str = None) -> SpecialistAgent:
        """Create agent from config file"""
        config_path = self.config_dir / f"{agent_id}.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Agent config not found: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if model_override:
            config["model"] = model_override

        # Handle legacy field
        if 'specialization' in config and 'tool_categories' not in config:
            config['tool_categories'] = config['specialization']

        agent_card = AgentCard.from_dict(config)

        agent = SpecialistAgent(
            agent_card=agent_card,
            llm_client=self.llm_client,
            mcp_client=self.mcp_client,
            intelligence_mode=self.intelligence_mode,
            response_log_dir=self.response_log_dir / agent_id
        )

        self.logger.info(f"Created agent {agent_id} with {len(agent.available_tools)} tools")
        return agent

    async def create_all_agents(self, model_override: str = None) -> Dict[str, SpecialistAgent]:
        """Create all agents from config files"""
        agents = {}

        config_files = list(self.config_dir.glob("*.yaml"))
        if not config_files:
            raise FileNotFoundError(f"No agent configs found in {self.config_dir}")

        for config_file in config_files:
            agent_id = config_file.stem
            try:
                agent = await self.create_agent(agent_id, model_override)
                agents[agent_id] = agent
            except Exception as e:
                self.logger.error(f"Failed to create agent {agent_id}: {e}")

        if not agents:
            raise RuntimeError("No agents could be created")

        return agents

    async def create_agent_from_spec(self, spec: Dict[str, Any]) -> SpecialistAgent:
        """
        Create agent from specification dict (used by orchestrator in dynamic mode)

        Args:
            spec: Agent specification with agent_id, name, role, tools, etc.
        """
        required = ["agent_id", "name", "role", "available_tools"]
        for field in required:
            if field not in spec:
                raise ValueError(f"Missing required field: {field}")

        config = {
            "agent_id": spec["agent_id"],
            "name": spec["name"],
            "role": spec["role"],
            "description": spec.get("description", spec["role"]),
            "example_tasks": spec.get("example_tasks", ""),
            "available_tools": spec["available_tools"],
            "model": spec.get("model", self.default_model),
            "temperature": spec.get("temperature", 0.3),
            "capabilities": spec.get("capabilities", []),
            "constraints": spec.get("constraints", [])
        }

        agent_card = AgentCard.from_dict(config)

        if self.mcp_client and self.mcp_client.is_connected:
            all_server_tools = set(self.mcp_client.available_tools.keys())
            requested_tools = config["available_tools"]
            if isinstance(requested_tools, list):
                valid_tools = [t for t in requested_tools if t in all_server_tools]
                invalid_tools = [t for t in requested_tools if t not in all_server_tools]
                if invalid_tools:
                    self.logger.warning(
                        f"Agent '{config['agent_id']}': stripped invalid tools: {invalid_tools}"
                    )
                if not valid_tools:
                    raise ValueError(
                        f"Agent '{config['agent_id']}' has no valid tools. "
                        f"Invalid: {invalid_tools}"
                    )
                config["available_tools"] = valid_tools

        agent = SpecialistAgent(
            agent_card=agent_card,
            llm_client=self.llm_client,
            mcp_client=self.mcp_client,
            intelligence_mode=self.intelligence_mode,
            response_log_dir=self.response_log_dir / spec["agent_id"]
        )

        self.logger.info(f"Created agent from spec: {spec['agent_id']} with {len(agent.available_tools)} tools")
        return agent

    async def revise_agent(self, agent: SpecialistAgent, new_tools: List[str]) -> SpecialistAgent:
        """
        Revise existing agent with new tools (used by orchestrator in dynamic mode)
        """
        agent.update_tools(new_tools)
        self.logger.info(f"Revised agent {agent.agent_card.agent_id} with {len(new_tools)} tools")
        return agent

    def get_all_server_tools(self) -> Dict[str, Any]:
        """Get all tools available on MCP server"""
        return self.mcp_client.available_tools

    def get_tool_categories(self) -> Dict[str, List[str]]:
        """Get tools organized by category"""
        return self.mcp_client.get_tools_summary()