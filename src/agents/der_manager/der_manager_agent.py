"""
DER Manager Agent - LLM-based configuration handling
"""
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

from ...core.base_agent import BaseAgent, AgentCard, AgentContext, AgentCapability
from ...core.message_bus import Message, MessageType


class DERManagerAgent(BaseAgent):
    """
    Agent responsible for managing DER systems configuration
    Uses LLM for intelligent parameter extraction and configuration
    """

    def __init__(self, agent_id: Optional[str] = None):
        super().__init__(agent_id=agent_id or "der_manager_001")
        self.current_config: Dict[str, Any] = {
            "pv": {"rated_capacity_kW": 2},
            "bat": {"rated_capacity_kWh": 5, "initial_soc": 0.3},
            "ev": {"rated_capacity_kWh": 5, "initial_soc": 0.3}
        }

    def _load_agent_card(self) -> AgentCard:
        """Load the DER Manager agent card"""
        card_path = Path(__file__).parent / "agent_card.yaml"

        if not card_path.exists():
            return AgentCard(
                name="DER Manager",
                description="Manages DER system configurations and analyzes flexibility",
                version="1.0.0",
                capabilities=[
                    AgentCapability.DER_MANAGEMENT,
                    AgentCapability.CONFIGURATION,
                    AgentCapability.OPTIMIZATION
                ],
                supported_tools=[
                    "der_config",
                    "flexibility_analyzer",
                    "system_query"
                ],
                required_context=[
                    "system_id",
                    "cluster_id",
                    "configuration_parameters"
                ],
                model_preferences={
                    "model": "qwen3:1.7b",
                    "temperature": 0.3,
                    "max_tokens": 2048
                },
                performance_metrics={
                    "avg_response_time": 2.5,
                    "success_rate": 0.95
                }
            )

        return AgentCard.from_yaml(card_path)

    async def process_task(self, task: Dict[str, Any], context: AgentContext) -> Dict[str, Any]:
        """Process DER management tasks based on orchestrator's intent"""
        try:
            user_request = task.get("request", "")
            intent = task.get("context", {}).get("intent", "unknown")

            print(f"DER Agent processing intent: {intent}")
            print(f"Request: {user_request}")

            if intent == "config_update":
                result = await self._handle_config_update_llm(user_request, context)
            elif intent == "flexibility_analysis":
                result = await self._handle_flexibility_analysis_llm(user_request, context)
            elif intent == "status_query":
                result = await self._handle_system_query(user_request, context)
            elif intent == "greeting":
                result = await self._handle_greeting(user_request, context)
            else:
                result = await self._handle_complex_request(user_request, context)

            return {
                "status": "success",
                "intent": intent,
                "result": result,
                "agent": self.agent_id
            }

        except Exception as e:
            print(f"DER Agent error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "error": str(e),
                "agent": self.agent_id
            }

    async def _handle_config_update_llm(
            self,
            request: str,
            context: AgentContext
    ) -> Dict[str, Any]:
        print("config ongoing")
        """
        Handle configuration update using LLM for parameter extraction
        This combines parameter extraction and configuration generation
        """

        # Use LLM to extract parameters and generate configuration
        extraction_prompt = f"""
        Extract configuration parameters from this user request about updating their DER system:
        
        User request: "{request}"
        
        Current system configuration:
        {json.dumps(self.current_config, indent=2)}
        
        DER components:
        - "pv" (photovoltaic/solar): capacity in kW
        - "bat" (battery storage): capacity in kWh, state of charge (soc) as decimal
        - "ev" (electric vehicle): capacity in kWh, state of charge (soc) as decimal
        
        Extract and return a JSON object with:
        {{
            "component": "pv" or "bat" or "ev",
            "updates": {{
                "field_name": new_value
            }},
            "needs_clarification": false,
            "clarification_message": null
        }}
        
        Examples:
        - "update my pv to 5kW" -> {{"component": "pv", "updates": {{"rated_capacity_kW": 5}}, "needs_clarification": false}}
        - "change battery to 10" -> {{"component": "bat", "updates": {{"rated_capacity_kWh": 10}}, "needs_clarification": false}}
        - "update my solar" -> {{"component": "pv", "updates": {{}}, "needs_clarification": true, "clarification_message": "What capacity would you like for your PV system?"}}
        
        Return ONLY valid JSON, no additional text.
        """

        try:
            # Get LLM to extract parameters
            extraction_result = await self.think(extraction_prompt)
            print("DER-internal-config-------------")
            print(extraction_result)
            # Parse the JSON response
            params = json.loads(extraction_result.strip())

            # Check if clarification is needed
            if params.get("needs_clarification", False):
                return {
                    "needs_clarification": True,
                    "response": params.get("clarification_message",
                                           "Could you please specify the value you'd like to update?")
                }

            component = params.get("component")
            updates = params.get("updates", {})

            if not component or not updates:
                return {
                    "error": "Could not understand the configuration request",
                    "response": "I couldn't understand which component or value you want to update. Could you please specify?"
                }

            # Build the complete system configuration
            new_config = self.current_config.copy()
            if component in new_config:
                new_config[component].update(updates)

            # Prepare DER configuration parameters for the tool
            config_params = {
                "cluster_id": "residential_cluster_1",
                "system_id": "der_system_1",
                "system_type": "der_systems",
                "parameters": {
                    "system_name": "PV-Battery-EV System",
                    "system_config": new_config
                },
                "class_path": "bestopt.env.modules.ders.system.der.DERModule"
            }

            # Use the DER config tool
            tool_result = await self.use_tool("der_config", {
                "action": "update",
                "config": config_params
            })

            # Update internal state
            self.current_config = new_config

            # Generate user-friendly response
            response_prompt = f"""
            The user requested: "{request}"
            
            Configuration update completed:
            - Component: {component}
            - Updates: {json.dumps(updates)}
            - Tool result: {tool_result.get('success', False)}
            
            Generate a brief, friendly confirmation message that:
            1. Confirms what was updated
            2. States the new values
            3. Is conversational and helpful
            
            Keep it to 1-2 sentences.
            """

            response = await self.think(response_prompt)

            return {
                "config_updated": True,
                "component": component,
                "updates": updates,
                "response": response,
                "technical_details": tool_result,
                "new_configuration": new_config
            }

        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            # Fallback to a more conversational approach
            fallback_response = await self.think(f"""
            The user wants to update their DER configuration: "{request}"
            
            I had trouble parsing the specific parameters. 
            Provide a helpful response asking for clarification about:
            - Which component (PV, battery, or EV)
            - What specific value they want to set
            
            Be friendly and give examples.
            """)

            return {
                "needs_clarification": True,
                "response": fallback_response
            }
        except Exception as e:
            print(f"Error in config update: {e}")
            return {
                "error": str(e),
                "response": "I encountered an error updating the configuration. Please try again."
            }

    async def _handle_flexibility_analysis_llm(
            self,
            request: str,
            context: AgentContext
    ) -> Dict[str, Any]:
        """Analyze flexibility using LLM-based parameter extraction"""

        # Extract parameters using LLM
        extraction_prompt = f"""
        Extract flexibility analysis parameters from this request:
        
        User request: "{request}"
        Current configuration: {json.dumps(self.current_config, indent=2)}
        
        Return JSON with:
        {{
            "component": "pv" or "bat" or "ev",
            "from_value": current_value,
            "to_value": new_value,
            "needs_values": true/false
        }}
        
        If specific values aren't mentioned, set needs_values to true.
        Return ONLY JSON.
        """

        try:
            params_str = await self.think(extraction_prompt)
            params = json.loads(params_str.strip())

            if params.get("needs_values", False):
                response = await self.think(f"""
                The user wants flexibility analysis but didn't specify values.
                Request: "{request}"
                
                Ask them to provide the current and target values for their analysis.
                Be helpful and conversational.
                """)
                return {"needs_clarification": True, "response": response}

            # Use flexibility analyzer tool
            analysis_result = await self.use_tool("flexibility_analyzer", {
                "component": params["component"],
                "baseline_value": params["from_value"],
                "new_value": params["to_value"],
                "analysis_type": "delta"
            })

            # Generate insights
            insights = await self.think(f"""
            Flexibility analysis results for {params['component']} 
            from {params['from_value']} to {params['to_value']}:
            
            {json.dumps(analysis_result, indent=2)}
            
            Provide clear insights about:
            1. Flexibility improvement
            2. Grid service potential
            3. Economic benefits
            4. Recommendations
            
            Keep it concise and actionable.
            """)

            return {
                "analysis_type": "flexibility_delta",
                "parameters": params,
                "metrics": analysis_result,
                "response": insights
            }

        except Exception as e:
            print(f"Flexibility analysis error: {e}")
            return {
                "error": str(e),
                "response": "I had trouble analyzing the flexibility. Could you please rephrase your request?"
            }

    async def _handle_system_query(self, request: str, context: AgentContext) -> Dict[str, Any]:
        """Handle system status queries"""
        current_config = await self.use_tool("system_query", {
            "query_type": "current_configuration",
            "system_id": "der_system_1"
        })

        response = await self.think(f"""
        User query: "{request}"
        System configuration: {json.dumps(current_config, indent=2)}
        
        Provide a clear, informative response about the DER system status.
        Include capacities and state of charge where relevant.
        Be conversational and helpful.
        """)

        return {
            "query_type": "system_status",
            "configuration": current_config,
            "response": response
        }

    async def _handle_greeting(self, request: str, context: AgentContext) -> Dict[str, Any]:
        """Handle greetings"""
        response = await self.think(f"""
        User greeted with: "{request}"
        
        Respond warmly and offer to help with their DER system.
        Mention you can help with PV, battery, and EV configuration and flexibility analysis.
        Keep it brief and welcoming.
        """)

        return {"response_type": "greeting", "response": response}

    async def _handle_complex_request(self, request: str, context: AgentContext) -> Dict[str, Any]:
        """Handle complex requests"""
        response = await self.think(f"""
        Complex request about DER system: "{request}"
        Context: {json.dumps(context.shared_memory if context else {}, indent=2)}
        
        Provide a helpful response. Explain available capabilities if needed.
        Be specific and actionable.
        """)

        return {"request_type": "complex", "response": response}