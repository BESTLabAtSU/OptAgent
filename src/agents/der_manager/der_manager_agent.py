"""
DER Manager Agent - LLM-based configuration handling
"""
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
import random

from ...core.base_agent import BaseAgent, AgentCard, AgentStatus, AgentCapability, AgentContext
from ...core.message_bus import Message, MessageType
from ...tools.tool_registry import get_tool_registry
from ...tools.der_query_tool import DERQueryTool
from ...tools.der_system_tool import DERSystemTool
from ...tools.der_controller_tool import DERControllerTool
from ...llm.ollama_client import OllamaClient


class DERManagerAgent(BaseAgent):
    """
    DER Manager Agent - properly integrated with BaseAgent
    """

    def __init__(
            self,
            agent_id: str = "der_manager_001",
            llm_client: Optional[OllamaClient] = None,
            config_dir: Optional[Path] = None,
            agent_card: Optional[AgentCard] = None
    ):
        # Set config directory before calling super().__init__
        self.config_dir = config_dir or Path(
            r"D:\Building_Simulator\BuildGPT_Agentic_AI_for_Autonomous_Building\bestopt\examples\SFH_1_Building\config_setup.json"
        )

        # Register tools in the global registry BEFORE calling super().__init__
        # This ensures tools are available when BaseAgent._load_tools() is called
        self._register_tools_in_registry()

        # Now initialize the base agent
        super().__init__(
            agent_id=agent_id,
            agent_card=agent_card,
            llm_client=llm_client
        )

    def _register_tools_in_registry(self):
        """Register DER tools in the global tool registry"""
        registry = get_tool_registry()

        # Create and register tools if they're not already registered
        if "der_query" not in registry.list_tools():
            registry.register_tool(DERQueryTool(config_dir=self.config_dir))

        if "der_system" not in registry.list_tools():
            registry.register_tool(DERSystemTool(config_dir=self.config_dir))

        if "der_controller" not in registry.list_tools():
            registry.register_tool(DERControllerTool(config_dir=self.config_dir))

    def _load_agent_card(self) -> AgentCard:
        """Load the DER Manager agent card from YAML or create default"""
        # Try to load from YAML file
        yaml_path = Path(__file__).parent / "cards" / "der_manager.yaml"
        if yaml_path.exists():
            return AgentCard.from_yaml(yaml_path)

        # Return default card with DER Manager specifics
        return AgentCard(
            name="DER Manager",
            description="Manages DER system configurations including PV, Battery, and EV systems",
            version="1.0.0",
            capabilities=[
                AgentCapability.DER_MANAGEMENT,
                AgentCapability.CONFIGURATION,
                AgentCapability.MONITORING
            ],
            supported_tools=[
                "der_query",  # Note: using base names without '_tool' suffix
                "der_system",
                "der_controller"
            ],
            required_context=[
                "user_request",
                "system_config",
                "controller_config"
            ],
            model_preferences={
                "model": "qwen3:1.7b",
                "temperature": 0.3,
                "max_tokens": 2048,
                "system_prompt": "You are a DER management expert assistant. Help users configure and optimize their distributed energy resources."
            },
            performance_metrics={
                "avg_response_time": 2.5,
                "success_rate": 0.95,
                "max_concurrent_tasks": 5
            }
        )

    async def use_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """
        Override to handle both base tool names and legacy names with '_tool' suffix
        """
        # Handle legacy tool names with '_tool' suffix
        if tool_name.endswith("_tool"):
            tool_base_name = tool_name.replace("_tool", "")
        else:
            tool_base_name = tool_name

        # Try to use tool through parent's method (which uses registry)
        if tool_base_name in self._tools:
            return await super().use_tool(tool_base_name, parameters)
        elif tool_name in self._tools:
            return await super().use_tool(tool_name, parameters)
        else:
            # Fallback to registry lookup
            registry = get_tool_registry()
            tool = registry.get_tool(tool_base_name)
            if tool:
                result = await tool.execute(parameters)
                # Store in context
                if self.current_context:
                    self.current_context.tool_results.append({
                        "tool": tool_name,
                        "parameters": parameters,
                        "result": result
                    })
                return result
            else:
                raise ValueError(f"Tool {tool_name} not available")

    async def process_task(self, task: Dict[str, Any], context: AgentContext) -> Dict[str, Any]:
        """
        Process a DER management task
        """
        request = task.get("request", "")
        task_context = task.get("context", {})
        intent = task_context.get("intent", "unknown")

        print(f"DER Agent processing: {intent} - {request}")

        # Route to appropriate handler based on intent
        handlers = {
            "system_query": self._handle_system_query,
            "controller_query": self._handle_controller_query,
            "system_add": self._handle_system_add,
            "system_update": self._handle_system_update,
            "controller_add": self._handle_controller_add,
            "controller_update": self._handle_controller_update,
            "greeting": self._handle_greeting,
        }

        handler = handlers.get(intent, self._handle_generic_request)
        result = await handler(request, context)

        return result

    async def _handle_system_query(self, request: str, context: AgentContext) -> Dict[str, Any]:
        """Handle system query requests"""
        query_prompt = f"""
        Extract the query parameters from this request about DER systems:
        Request: {request}

        Return as JSON with fields:
        - system_id: specific system ID if mentioned (or null for all)
        - query_type: "full" or "specific" (for specific parameters)
        - parameters: list of specific parameters to query if mentioned

        Respond only with valid JSON, no other text.
        """

        try:
            extraction = await self.think(query_prompt)
            extraction = extraction.strip()
            if extraction.startswith("```json"):
                extraction = extraction[7:]
            if extraction.endswith("```"):
                extraction = extraction[:-3]
            query_params = json.loads(extraction.strip())
        except:
            query_params = {"system_id": None, "query_type": "full"}

        # Use tool (can use either name)
        result = await self.use_tool("der_query", {
            "query_type": "system",
            "system_id": query_params.get("system_id"),
            "parameters": query_params.get("parameters")
        })

        if result.success:
            response_prompt = f"""
            {self.agent_card.model_preferences.get('system_prompt', '')}

            Generate a clear, friendly response about the DER systems:
            Data: {json.dumps(result.data, indent=2)}
            User request: {request}

            Format the response in a readable way with proper units and explanations.
            """

            response_text = await self.think(response_prompt)

            return {
                "response": response_text,
                "data": result.data,
                "status": "success"
            }
        else:
            return {
                "response": f"I couldn't retrieve the system information: {result.error}",
                "status": "error",
                "error": result.error
            }

    async def shutdown(self):
        """Cleanup on shutdown"""
        # Unregister tools from the registry if needed
        # (optional - depends on your cleanup strategy)
        await super().shutdown()

    async def _handle_controller_query(self, request: str, context: AgentContext) -> Dict[str, Any]:
        """Handle controller query requests"""
        # Parse query parameters
        query_prompt = f"""
        Extract the query parameters from this request about DER controllers:
        Request: {request}

        Return as JSON with fields:
        - controller_id: specific controller ID if mentioned (or null for all)
        - system_id: associated system ID if mentioned
        - query_type: "full" or "specific"
        - parameters: list of specific parameters to query if mentioned
        
        Respond only with valid JSON, no other text.
        """

        try:
            extraction = await self.think(query_prompt)
            # Clean extraction
            extraction = extraction.strip()
            if extraction.startswith("```json"):
                extraction = extraction[7:]
            if extraction.endswith("```"):
                extraction = extraction[:-3]
            query_params = json.loads(extraction.strip())
        except:
            query_params = {"controller_id": None, "query_type": "full"}

        # Execute query
        result = await self.use_tool("der_query_tool", {
            "query_type": "controller",
            "controller_id": query_params.get("controller_id"),
            "system_id": query_params.get("system_id"),
            "parameters": query_params.get("parameters")
        })

        if result.success:
            response_prompt = f"""
            {self.agent_card.model_preferences.get('system_prompt', '')}
            
            Generate a clear response about the DER controllers:
            Data: {json.dumps(result.data, indent=2)}
            User request: {request}

            Explain the controller settings in simple terms.
            """

            response_text = await self.think(response_prompt)

            return {
                "response": response_text,
                "data": result.data,
                "status": "success"
            }
        else:
            return {
                "response": f"I couldn't retrieve the controller information: {result.error}",
                "status": "error",
                "error": result.error
            }

    async def _handle_system_add(self, request: str, context: AgentContext) -> Dict[str, Any]:
        """Handle system add requests"""
        # Parse system configuration from request
        parse_prompt = f"""
        Extract the DER system configuration from this request:
        Request: {request}

        Return as JSON with:
        - system_id: unique identifier for the system
        - system_type: "der_systems" (always)
        - components: object with component configurations (pv, bat, ev, etc.)

        Example components:
        - pv: {{"rated_capacity_kW": 5}}
        - bat: {{"rated_capacity_kWh": 10, "initial_soc": 0.5}}
        - ev: {{"rated_capacity_kWh": 75, "initial_soc": 0.8}}
        
        Respond only with valid JSON, no other text.
        """

        try:
            extraction = await self.think(parse_prompt)
            # Clean extraction
            extraction = extraction.strip()
            if extraction.startswith("```json"):
                extraction = extraction[7:]
            if extraction.endswith("```"):
                extraction = extraction[:-3]
            config = json.loads(extraction.strip())
        except Exception as e:
            return {
                "response": f"I couldn't parse the system configuration. Please provide clear details about the system components.",
                "status": "error",
                "error": str(e)
            }

        # Add system using tool
        result = await self.use_tool("der_system_tool", {
            "action": "add",
            "config": config
        })

        if result.success:
            return {
                "response": f"Successfully added DER system '{config['system_id']}' with the specified components.",
                "data": result.data,
                "status": "success"
            }
        else:
            return {
                "response": f"Failed to add the system: {result.error}",
                "status": "error",
                "error": result.error
            }

    async def _handle_system_update(self, request: str, context: AgentContext) -> Dict[str, Any]:
        """Handle system update requests"""
        # Parse update parameters
        parse_prompt = f"""
        Extract the update parameters from this DER system update request:
        Request: {request}

        Return as JSON with:
        - system_id: ID of system to update (required)
        - updates: object with parameters to update
        - merge: true to merge with existing, false to replace (default: true)

        The updates object should contain:
        - components: object with component updates (optional)
          - pv: PV configuration changes
          - bat: Battery configuration changes  
          - ev: EV configuration changes
        - system_name: new system name (optional)

        Example response format:
        {{
            "system_id": "der_001",
            "updates": {{
                "components": {{
                    "pv": {{"rated_capacity_kW": 10}},
                    "bat": {{"rated_capacity_kWh": 20, "initial_soc": 0.8}}
                }},
                "system_name": "Updated DER System"
            }},
            "merge": true
        }}

        Respond only with valid JSON, no other text.
        """

        try:
            extraction = await self.think(parse_prompt)
            # Clean extraction
            extraction = extraction.strip()
            if extraction.startswith("```json"):
                extraction = extraction[7:]
            if extraction.endswith("```"):
                extraction = extraction[:-3]
            update_params = json.loads(extraction.strip())

            # Validate required fields
            if not update_params.get("system_id"):
                return {
                    "response": "Please specify which system to update (system_id is required).",
                    "status": "error",
                    "error": "Missing system_id"
                }

            if not update_params.get("updates"):
                return {
                    "response": "Please specify what parameters to update.",
                    "status": "error",
                    "error": "Missing updates"
                }

        except json.JSONDecodeError as e:
            return {
                "response": "I couldn't parse the update request. Please specify which system to update and what parameters to change.",
                "status": "error",
                "error": f"JSON parsing error: {str(e)}"
            }
        except Exception as e:
            return {
                "response": "An error occurred while parsing your request. Please try again.",
                "status": "error",
                "error": str(e)
            }

        # Update system using tool - note the parameter structure
        try:
            result = await self.use_tool("der_system", {
                "action": "update",
                "system_id": update_params.get("system_id"),
                "updates": update_params.get("updates"),
                "merge": update_params.get("merge", True)
            })
        except Exception as e:
            return {
                "response": f"Failed to call the update tool: {str(e)}",
                "status": "error",
                "error": str(e)
            }

        if result.success:
            # Generate a user-friendly response
            response_parts = [f"Successfully updated DER system '{update_params['system_id']}'."]

            # Add details about what was updated
            updates = update_params.get("updates", {})
            if "components" in updates:
                components = updates["components"]
                updated_components = []

                if "pv" in components:
                    pv_changes = components["pv"]
                    if "rated_capacity_kW" in pv_changes:
                        updated_components.append(f"PV capacity to {pv_changes['rated_capacity_kW']} kW")

                if "bat" in components:
                    bat_changes = components["bat"]
                    changes = []
                    if "rated_capacity_kWh" in bat_changes:
                        changes.append(f"capacity to {bat_changes['rated_capacity_kWh']} kWh")
                    if "initial_soc" in bat_changes:
                        changes.append(f"SOC to {bat_changes['initial_soc'] * 100:.0f}%")
                    if changes:
                        updated_components.append(f"Battery {' and '.join(changes)}")

                if "ev" in components:
                    ev_changes = components["ev"]
                    changes = []
                    if "rated_capacity_kWh" in ev_changes:
                        changes.append(f"capacity to {ev_changes['rated_capacity_kWh']} kWh")
                    if "initial_soc" in ev_changes:
                        changes.append(f"SOC to {ev_changes['initial_soc'] * 100:.0f}%")
                    if changes:
                        updated_components.append(f"EV {' and '.join(changes)}")

                if updated_components:
                    response_parts.append(f"Updated: {', '.join(updated_components)}.")

            if "system_name" in updates:
                response_parts.append(f"System name changed to '{updates['system_name']}'.")

            return {
                "response": " ".join(response_parts),
                "data": result.data,
                "status": "success"
            }
        else:
            # Provide more specific error messages
            error_msg = result.error
            if "not found" in error_msg.lower():
                return {
                    "response": f"System '{update_params['system_id']}' was not found. Please check the system ID.",
                    "status": "error",
                    "error": error_msg
                }
            else:
                return {
                    "response": f"Failed to update the system: {error_msg}",
                    "status": "error",
                    "error": error_msg
                }

    async def _handle_controller_add(self, request: str, context: AgentContext) -> Dict[str, Any]:
        """Handle controller add requests"""
        # Parse controller configuration
        parse_prompt = f"""
        Extract the DER controller configuration from this request:
        Request: {request}

        Return as JSON with:
        - controller_id: unique identifier
        - system_id: associated DER system ID
        - parameters: controller settings like:
          - type: "rule-based" or "optimization"
          - mode: "self_consumption", "peak_shaving", etc.
          - bat_soc_min/max: battery SOC limits
          - ev_v2g_enabled: true/false
          - max_grid_import/export: power limits
          
        Respond only with valid JSON, no other text.
        """

        try:
            extraction = await self.think(parse_prompt)
            # Clean extraction
            extraction = extraction.strip()
            if extraction.startswith("```json"):
                extraction = extraction[7:]
            if extraction.endswith("```"):
                extraction = extraction[:-3]
            config = json.loads(extraction.strip())
        except Exception as e:
            return {
                "response": "Please provide controller configuration details.",
                "status": "error",
                "error": str(e)
            }

        # Add controller using tool
        result = await self.use_tool("der_controller_tool", {
            "action": "add",
            "config": config
        })

        if result.success:
            return {
                "response": f"Successfully added controller '{config['controller_id']}' for DER system.",
                "data": result.data,
                "status": "success"
            }
        else:
            return {
                "response": f"Failed to add controller: {result.error}",
                "status": "error",
                "error": result.error
            }

    async def _handle_controller_update(self, request: str, context: AgentContext) -> Dict[str, Any]:
        """Handle controller update requests"""
        # Parse update parameters
        parse_prompt = f"""
        Extract controller update parameters from this request:
        Request: {request}

        Return as JSON with:
        - controller_id: ID of controller to update
        - updates: parameters to change (setpoints, modes, limits, etc.)
        - merge: true to merge, false to replace
        
        Respond only with valid JSON, no other text.
        """

        try:
            extraction = await self.think(parse_prompt)
            # Clean extraction
            extraction = extraction.strip()
            if extraction.startswith("```json"):
                extraction = extraction[7:]
            if extraction.endswith("```"):
                extraction = extraction[:-3]
            update_params = json.loads(extraction.strip())
        except Exception as e:
            return {
                "response": "Please specify which controller to update and parameters to change.",
                "status": "error",
                "error": str(e)
            }

        # Update controller using tool
        result = await self.use_tool("der_controller_tool", {
            "action": "update",
            "controller_id": update_params.get("controller_id"),
            "updates": update_params.get("updates"),
            "merge": update_params.get("merge", True)
        })

        if result.success:
            return {
                "response": f"Successfully updated controller '{update_params['controller_id']}'.",
                "data": result.data,
                "status": "success"
            }
        else:
            return {
                "response": f"Failed to update controller: {result.error}",
                "status": "error",
                "error": result.error
            }

    async def _handle_greeting(self, request: str, context: AgentContext) -> Dict[str, Any]:
        """Handle greeting messages"""
        greetings = [
            "Hello! I'm your DER Manager. I can help you configure and monitor your distributed energy resources.",
            "Hi there! Ready to assist with your DER systems. What would you like to know or configure?",
            "Greetings! I manage your DER systems. Feel free to ask about configurations or make updates."
        ]

        response = random.choice(greetings)

        try:
            result = await self.use_tool("der_query", {"query_type": "system"})
            if result.success and result.data:
                systems = result.data.get("systems", {})
                if systems:
                    response += f"\n\nCurrently managing {len(systems)} DER system(s)."
        except:
            pass

        return {
            "response": response,
            "status": "success"
        }

    async def _handle_generic_request(self, request: str, context: AgentContext) -> Dict[str, Any]:
        """Handle requests when intent is unclear"""
        routing_prompt = f"""
        This DER management request doesn't have a clear intent. Analyze and provide response:
        Request: {request}

        Provide a helpful response or ask for clarification.
        """

        response_text = await self.think(routing_prompt)

        return {
            "response": response_text,
            "status": "success",
            "note": "Generic handler - intent unclear"
        }