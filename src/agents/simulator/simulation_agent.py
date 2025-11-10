"""
Simulation Agent - A proper agent for running simulations
Following the same pattern as your DER Manager Agent
"""
import json
from typing import Dict, Any, Optional
from pathlib import Path

from ...core.base_agent import BaseAgent, AgentCard, AgentStatus, AgentCapability, AgentContext
from ...core.message_bus import Message, MessageType
from ...tools.tool_registry import get_tool_registry
from ...tools.simulation_tool import SimulationTool
from ...llm.ollama_client import OllamaClient


class SimulationAgent(BaseAgent):
    """
    Simulation Agent - handles simulation requests through the agent framework
    """

    def __init__(
            self,
            agent_id: str = "simulation_agent_001",
            llm_client: Optional[OllamaClient] = None,
            config_dir: Optional[Path] = None,
            agent_card: Optional[AgentCard] = None
    ):
        self.config_dir = config_dir or Path(
            r"D:\Building_Simulator\BuildGPT_Agentic_AI_for_Autonomous_Building\bestopt\examples\SFH_1_Building\config_setup.json"
        )

        # Register simulation tool
        self._register_tools_in_registry()

        super().__init__(
            agent_id=agent_id,
            agent_card=agent_card,
            llm_client=llm_client
        )

    def _register_tools_in_registry(self):
        """Register simulation tools in the global tool registry"""
        registry = get_tool_registry()

        if "simulation" not in registry.list_tools():
            registry.register_tool(SimulationTool(config_dir=self.config_dir))

    def _load_agent_card(self) -> AgentCard:
        """Load the Simulation agent card"""
        yaml_path = Path(__file__).parent / "cards" / "simulation_agent.yaml"
        if yaml_path.exists():
            return AgentCard.from_yaml(yaml_path)

        return AgentCard(
            name="Simulation Agent",
            description="Runs building energy simulations and calculates KPIs",
            version="1.0.0",
            capabilities=[
                AgentCapability.SIMULATION,
            ],
            supported_tools=[
                "simulation"
            ],
            required_context=[
               "user_request",
               "simulation_config"
            ],
            model_preferences={
                "model": "qwen3:1.7b",
                "temperature": 0.2,
                "max_tokens": 2048,
                "system_prompt": "You are a simulation specialist. Help run simulations and analyze results."
            }
        )

    async def process_task(self, task: Dict[str, Any], context: AgentContext) -> Dict[str, Any]:
        """
        Process simulation tasks
        """
        action = task.get("context", {}).get("action_type", None)

        if action == "run":
            return await self._run_simulation(task, context)
        elif action == "compare":
            return await self._compare_simulations(task, context)
        else:
            return await self._handle_generic(task, context)

    async def _run_simulation(self, task: Dict[str, Any], context: AgentContext) -> Dict[str, Any]:
        """Run a simulation with given configuration"""
        request_context = task.get("context", {})

        # Get configuration from context or previous step results
        config = None

        # Check if we have results from a previous step (e.g., DER query)
        if "result_from_step_1" in request_context:
            prev_result = request_context["result_from_step_1"]
            if "data" in prev_result:
                config = prev_result["data"]

        # Or use shared context
        if not config and "shared_context" in request_context:
            config = request_context["shared_context"].get("current_config")

        # Run simulation
        result = await self.use_tool("simulation", {
            "action": "run",
            "config": config,
            "parameters": request_context.get("parameters", {})
        })

        if result.success:
            # Calculate KPIs
            kpis = self._calculate_flexibility_kpis(result.data)

            response = {
                "status": "success",
                "simulation_results": result.data,
                "flexibility_kpis": kpis,
                "message": "Simulation completed successfully"
            }

            # Generate summary if needed
            if request_context.get("generate_summary", False):
                summary = await self._generate_summary(kpis)
                response["summary"] = summary

            return response
        else:
            return {
                "status": "error",
                "error": result.error
            }

    async def _compare_simulations(self, task: Dict[str, Any], context: AgentContext) -> Dict[str, Any]:
        """Compare results from multiple simulations"""
        request_context = task.get("context", {})

        # Get baseline and comparison results
        baseline = request_context.get("result_from_step_2", {})
        comparison = request_context.get("result_from_step_4", {})

        if not baseline or not comparison:
            return {
                "status": "error",
                "error": "Missing simulation results to compare"
            }

        # Extract KPIs
        baseline_kpis = baseline.get("flexibility_kpis", {})
        comparison_kpis = comparison.get("flexibility_kpis", {})

        # Calculate differences
        analysis = {
            "baseline": baseline_kpis,
            "updated": comparison_kpis,
            "improvements": {}
        }

        for kpi in baseline_kpis:
            if kpi in comparison_kpis:
                baseline_val = baseline_kpis[kpi]
                updated_val = comparison_kpis[kpi]
                if baseline_val != 0:
                    improvement_pct = ((updated_val - baseline_val) / baseline_val) * 100
                    analysis["improvements"][kpi] = {
                        "absolute": updated_val - baseline_val,
                        "percentage": improvement_pct
                    }

        # Generate comparison summary
        summary = await self._generate_comparison_summary(analysis, task.get("request", ""))

        return {
            "status": "success",
            "analysis": analysis,
            "summary": summary
        }

    def _calculate_flexibility_kpis(self, simulation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate flexibility KPIs from simulation results"""
        # This should match your actual KPI calculations
        return {
            "peak_reduction": simulation_data.get("peak_reduction", 0),
            "self_consumption": simulation_data.get("self_consumption", 0),
            "grid_independence": simulation_data.get("grid_independence", 0),
            "cost_savings": simulation_data.get("cost_savings", 0),
            "carbon_reduction": simulation_data.get("carbon_reduction", 0),
            "flexibility_score": simulation_data.get("flexibility_score", 0)
        }

    async def _generate_summary(self, kpis: Dict[str, Any]) -> str:
        """Generate a summary of simulation results"""
        prompt = f"""
        Summarize these simulation KPIs in a clear, concise way:
        {json.dumps(kpis, indent=2)}
        
        Focus on the most important metrics and what they mean for the building's energy performance.
        """

        summary = await self.think(prompt)
        return summary

    async def _generate_comparison_summary(self, analysis: Dict[str, Any], request: str) -> str:
        """Generate a comparison summary"""
        prompt = f"""
        Create a clear summary of this flexibility comparison:
        
        User request: {request}
        Analysis: {json.dumps(analysis, indent=2)}
        
        Highlight:
        - Key improvements
        - Most significant changes
        - Overall recommendation
        
        Be specific with numbers and percentages.
        """

        summary = await self.think(prompt)
        return summary

    async def _handle_generic(self, task: Dict[str, Any], context: AgentContext) -> Dict[str, Any]:
        """Handle generic simulation requests"""
        return {
            "status": "success",
            "message": "Simulation agent ready. Specify 'run' or 'compare' action."
        }