
# =====================================
# src/orchestrator/agent_selector.py
# =====================================
"""
Agent selection logic for the orchestrator
"""
from typing import List, Dict, Any, Optional
from ..core.base_agent import AgentCapability


class AgentSelector:
    """Selects appropriate agents for tasks"""

    def __init__(self, agent_registry):
        self.agent_registry = agent_registry

    async def select_agent(
            self,
            required_capabilities: List[AgentCapability],
            task_context: Dict[str, Any]
    ) -> Optional[str]:
        """Select the best agent for a task"""
        # Get agents with required capabilities
        available_agents = await self.agent_registry.get_agents_by_capability(
            required_capabilities
        )

        if not available_agents:
            return None

        # Score agents based on various factors
        scored_agents = []
        for agent in available_agents:
            score = await self._score_agent(agent, task_context)
            scored_agents.append((agent["agent_id"], score))

        # Sort by score and return best
        scored_agents.sort(key=lambda x: x[1], reverse=True)
        return scored_agents[0][0] if scored_agents else None

    async def _score_agent(
            self,
            agent: Dict[str, Any],
            context: Dict[str, Any]
    ) -> float:
        """Score an agent for task suitability"""
        score = 0.0

        # Base score from success rate
        metrics = agent.get("metrics", {})
        score += metrics.get("success_rate", 0) * 100

        # Penalty for slow response time
        avg_time = metrics.get("avg_response_time", 0)
        if avg_time > 0:
            score -= min(avg_time, 10)  # Cap penalty at 10

        # Bonus for matching model preferences
        if "preferred_model" in context:
            model_prefs = agent["card"].model_preferences
            if model_prefs.get("model") == context["preferred_model"]:
                score += 10

        return score
