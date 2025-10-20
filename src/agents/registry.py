"""
Agent registry for managing agents in the system
"""
from typing import Dict, Any, List, Optional
from ..core.base_agent import AgentCard, AgentCapability, AgentStatus


class AgentRegistry:
    """Registry for managing agents"""

    def __init__(self):
        self._agents: Dict[str, Dict[str, Any]] = {}
        self._capability_index: Dict[AgentCapability, List[str]] = {}

    async def initialize(self) -> None:
        """Initialize the registry"""
        # Could load agent configurations from disk/database
        pass

    async def shutdown(self) -> None:
        """Shutdown the registry"""
        pass

    async def register_agent(
            self,
            agent_id: str,
            agent_card: AgentCard
    ) -> None:
        """Register an agent"""
        self._agents[agent_id] = {
            "agent_id": agent_id,
            "card": agent_card,
            "status": AgentStatus.IDLE,
            "metrics": {
                "tasks_completed": 0,
                "avg_response_time": 0,
                "success_rate": 1.0
            }
        }

        # Update capability index
        for capability in agent_card.capabilities:
            if capability not in self._capability_index:
                self._capability_index[capability] = []
            self._capability_index[capability].append(agent_id)

    async def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent"""
        if agent_id in self._agents:
            agent = self._agents[agent_id]

            # Remove from capability index
            for capability in agent["card"].capabilities:
                if capability in self._capability_index:
                    self._capability_index[capability].remove(agent_id)

            del self._agents[agent_id]

    async def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent information"""
        return self._agents.get(agent_id)

    async def get_agents_by_capability(
            self,
            capabilities: List[AgentCapability]
    ) -> List[Dict[str, Any]]:
        """Get agents that have all specified capabilities"""
        matching_agents = []

        for agent_id, agent in self._agents.items():
            agent_caps = set(agent["card"].capabilities)
            required_caps = set(capabilities)

            if required_caps.issubset(agent_caps):
                matching_agents.append(agent)

        # Sort by performance metrics (success rate, response time)
        matching_agents.sort(
            key=lambda a: (
                -a["metrics"]["success_rate"],
                a["metrics"]["avg_response_time"]
            )
        )

        return matching_agents

    async def update_agent_status(
            self,
            agent_id: str,
            status: AgentStatus
    ) -> None:
        """Update agent status"""
        if agent_id in self._agents:
            self._agents[agent_id]["status"] = status

    async def update_agent_metrics(
            self,
            agent_id: str,
            task_completed: bool,
            response_time: float
    ) -> None:
        """Update agent performance metrics"""
        if agent_id in self._agents:
            metrics = self._agents[agent_id]["metrics"]

            # Update task count
            metrics["tasks_completed"] += 1

            # Update average response time
            n = metrics["tasks_completed"]
            current_avg = metrics["avg_response_time"]
            metrics["avg_response_time"] = (
                    (current_avg * (n - 1) + response_time) / n
            )

            # Update success rate
            if task_completed:
                success_count = int(metrics["success_rate"] * (n - 1)) + 1
            else:
                success_count = int(metrics["success_rate"] * (n - 1))
            metrics["success_rate"] = success_count / n

    def get_all_agents(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered agents"""
        return self._agents.copy()