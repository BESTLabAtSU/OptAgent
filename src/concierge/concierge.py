"""
Concierge
"""
from typing import Dict, Any, Optional, List, Union
import asyncio
import json
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Awaitable

ReasoningCallback = Callable[[Dict[str, Any]], Awaitable[None]]

from ..core.data_class import IntelligenceMode, AgentGenerationMode
from ..orchestrator.orchestrator import Orchestrator
from ..llm.llm_interface import BaseLLMClient, create_llm_client, LLMResponse


@dataclass
class ConciergeResponse:
    """Structured response from concierge"""
    message: str
    routed_to_orchestrator: bool = False
    orchestrator_response: Optional[Dict[str, Any]] = None

    @property
    def benchmark(self) -> Optional[Dict[str, Any]]:
        if self.orchestrator_response:
            return self.orchestrator_response.get("benchmark")
        return None

    @property
    def execution_plan(self) -> List[Dict]:
        if self.benchmark:
            return self.benchmark.get("execution_plan", [])
        return []

    @property
    def agent_results(self) -> Dict[str, Any]:
        if self.benchmark:
            return self.benchmark.get("agent_results", {})
        return {}

    @property
    def metrics(self) -> Dict[str, Any]:
        if self.benchmark:
            return self.benchmark.get("metrics", {})
        return {}


@dataclass
class ConversationContext:
    """Maintains conversation history and session state"""
    conversation_id: str
    messages: List[Dict[str, str]] = field(default_factory=list)

    # Track what's been done in this session
    session_state: Dict[str, Any] = field(default_factory=lambda: {
        "completed_tasks": [],
        "last_orchestrator_result": None,
        "workspace_context": {},
        "simulation_active": False,
        "last_simulation_id": None,
    })

    def add_message(self, role: str, content: str):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

    def get_history(self, max_turns: int = 10) -> List[Dict[str, str]]:
        """Get recent conversation history"""
        recent = self.messages[-(max_turns * 2):]
        # Return only role and content for LLM compatibility
        return [{"role": m["role"], "content": m["content"]} for m in recent]

    def get_full_history(self) -> List[Dict[str, str]]:
        """Get full conversation history with timestamps"""
        return self.messages.copy()

    def update_session_state(self, orchestrator_response: Dict[str, Any], user_request: str):
        """Update session state after orchestrator completes a task"""

        # Extract key information from the response
        tools_used = self._extract_tools_used(orchestrator_response)
        data_available = self._extract_available_data(orchestrator_response)

        self.session_state["last_orchestrator_result"] = {
            "request": user_request,
            "summary": orchestrator_response.get("summary", ""),
            "status": orchestrator_response.get("status", ""),
            "tools_used": tools_used,
            "data_available": data_available,
            "timestamp": datetime.now().isoformat()
        }

        # Check if simulation was run
        if any("simulation" in tool.lower() for tool in tools_used):
            self.session_state["simulation_active"] = True
            # Try to extract simulation ID if available
            details = orchestrator_response.get("details", {})
            if "simulation_id" in details:
                self.session_state["last_simulation_id"] = details["simulation_id"]

        # Update workspace context
        self.session_state["workspace_context"].update(data_available)

        # Track completed task
        self.session_state["completed_tasks"].append({
            "request": user_request[:100],
            "summary": orchestrator_response.get("summary", "")[:200],
            "tools_used": tools_used,
            "timestamp": datetime.now().isoformat()
        })

        # Keep only last 10 tasks
        if len(self.session_state["completed_tasks"]) > 10:
            self.session_state["completed_tasks"] = self.session_state["completed_tasks"][-10:]



    def _extract_tools_used(self, response: Dict) -> List[str]:
        """Extract which tools were used from orchestrator response"""
        tools = []

        try:
            # Try to get from benchmark
            benchmark = response.get("benchmark", {})
            for agent_id, agent_result in benchmark.get("agent_results", {}).items():
                tools_used = agent_result.get("tools_used", [])
                # Ensure we only add strings, not dicts
                for tool in tools_used:
                    if isinstance(tool, str):
                        tools.append(tool)
                    elif isinstance(tool, dict):
                        # Extract tool name from dict if present
                        tool_name = tool.get("name") or tool.get("tool") or str(tool)
                        tools.append(str(tool_name))

            # Also check details
            details = response.get("details", {})
            if isinstance(details, dict) and "tools_executed" in details:
                for tool in details["tools_executed"]:
                    if isinstance(tool, str):
                        tools.append(tool)
                    elif isinstance(tool, dict):
                        tool_name = tool.get("name") or tool.get("tool") or str(tool)
                        tools.append(str(tool_name))
        except Exception as e:
            # Log but don't fail
            print(f"Warning: Error extracting tools: {e}")

        # Remove duplicates safely
        return list(dict.fromkeys(tools))  # Preserves order, removes duplicates

    def _extract_available_data(self, response: Dict) -> Dict[str, Any]:
        """Extract what data/files are now available"""
        data = {}

        # Convert response to string for keyword search
        response_str = json.dumps(response, default=str).lower()

        # Check for simulation results
        if "simulation" in response_str:
            data["simulation_completed"] = True

        # Check for specific analysis types
        analysis_types = [
            "cost", "energy", "flexibility", "emissions",
            "comfort", "peak", "load", "battery", "solar",
            "hvac", "thermal", "optimization"
        ]

        for analysis_type in analysis_types:
            if analysis_type in response_str:
                data[f"{analysis_type}_analyzed"] = True

        # Check for file outputs
        if "saved" in response_str or "exported" in response_str or "file" in response_str:
            data["files_created"] = True

        return data

    def get_context_summary(self) -> str:
        """Get a summary of current session context for the orchestrator"""
        if not self.session_state["last_orchestrator_result"]:
            return ""

        last = self.session_state["last_orchestrator_result"]

        context_parts = [
            "=== SESSION CONTEXT (Previous work in this conversation) ==="
        ]

        # Last task info
        context_parts.append(f"\nMost recent task:")
        context_parts.append(f"  Request: {last.get('request', 'N/A')}")
        context_parts.append(f"  Result: {last.get('summary', 'N/A')[:300]}")
        context_parts.append(f"  Status: {last.get('status', 'N/A')}")

        # Tools used
        tools = last.get('tools_used', [])
        if tools:
            context_parts.append(f"  Tools used: {', '.join(tools)}")

        # What's available now
        workspace = self.session_state.get("workspace_context", {})
        if workspace:
            available = [k.replace('_analyzed', '').replace('_completed', '').replace('_created', '')
                        for k, v in workspace.items() if v]
            if available:
                context_parts.append(f"\nData/analyses available in workspace: {', '.join(set(available))}")

        # Simulation status
        if self.session_state.get("simulation_active"):
            context_parts.append("\n[IMPORTANT: A simulation has been run. Results are available in the workspace for further analysis.]")
            if self.session_state.get("last_simulation_id"):
                context_parts.append(f"  Simulation ID: {self.session_state['last_simulation_id']}")

        # Recent task history (if more than one task)
        if len(self.session_state["completed_tasks"]) > 1:
            context_parts.append("\nPrevious tasks in this session:")
            for task in self.session_state["completed_tasks"][-5:-1]:  # Skip the most recent (already shown)
                context_parts.append(f"  - {task['request'][:60]}...")

        context_parts.append("\n=== END SESSION CONTEXT ===\n")

        return "\n".join(context_parts)

    def clear_session(self):
        """Clear session state but keep conversation history"""
        self.session_state = {
            "completed_tasks": [],
            "last_orchestrator_result": None,
            "workspace_context": {},
            "simulation_active": False,
            "last_simulation_id": None,
        }


class EnhancedConcierge:
    """
    LLM-based Concierge with session context tracking for follow-up questions
    """

    SYSTEM_PROMPT = """You are a friendly assistant for a Building Energy and DER (Distributed Energy Resources) simulation system.

IMPORTANT: You are a ROUTING layer. The orchestrator behind you has access to default configurations and is smart enough to handle incomplete requests.

Your jobs:
1. Handle greetings and small talk naturally
2. Route ANY technical request to the orchestrator 
3. Ask for clarification based on error message if the process failed 

When routing, respond with a brief acknowledgment + ACTION: ROUTE_TO_ORCHESTRATOR

Examples:
User: "analyze flexibility after upgrading battery to 20kwh"
You: "I'll run that flexibility analysis for you. ACTION: ROUTE_TO_ORCHESTRATOR"

User: "compare solar options"  
You: "Let me compare those for you. ACTION: ROUTE_TO_ORCHESTRATOR"

User: "how about energy?" (after a simulation)
You: "I'll analyze the energy metrics from the simulation. ACTION: ROUTE_TO_ORCHESTRATOR"

User: "and the cost?"
You: "Running the cost analysis now. ACTION: ROUTE_TO_ORCHESTRATOR"

DO NOT route:
- "Hi" / "Hello" / greetings
- "What can you do?" / capability questions  
- "Thanks" / acknowledgments

Available agents: {agent_info}
"""

    RESPONSE_FORMATTER_PROMPT = """Format this simulation result for the user in a friendly way.

Result from system:
{result}

User's request: {request}

Guidelines:
- Lead with the key finding/answer
- Show important numbers clearly
- Keep it concise (5 sentences max)
- Suggest ONE relevant follow-up if appropriate

If there's an error, briefly explain and suggest trying again."""

    # Follow-up question indicators
    FOLLOWUP_INDICATORS = [
        "how about", "what about", "and the", "also show", "show me",
        "now analyze", "then", "also", "same for", "similar",
        "what's the", "can you also", "and also", "next",
        "continue", "more", "another", "other", "else"
    ]

    def __init__(self,
                 mcp_server_path: str,
                 config_dir: Path = Path("config"),
                 log_dir: Path = Path("logs"),
                 default_provider: str = "ollama",
                 default_orchestrator_model: str = "gemma2:9b",
                 default_agent_model: str = "gemma2:9b",
                 openai_api_key: str = None):

        self.mcp_server_path = mcp_server_path
        self.config_dir = config_dir
        self.log_dir = log_dir
        self.openai_api_key = openai_api_key

        self.provider = default_provider
        self.orchestrator_model = default_orchestrator_model
        self.agent_model = default_agent_model

        self.intelligence_mode = IntelligenceMode.CENTRALIZED
        self.two_stage_planning = True

        self.orchestrator: Optional[Orchestrator] = None
        self.llm_client: Optional[BaseLLMClient] = None
        self.agent_llm_client: Optional[BaseLLMClient] = None

        self.conversation: Optional[ConversationContext] = None
        self.is_initialized = False
        self._agent_info_cache: str = ""

        self._last_orchestrator_response: Optional[Dict[str, Any]] = None

    async def initialize(self) -> bool:
        """Initialize the concierge and orchestrator"""
        try:
            self.llm_client = create_llm_client(
                provider=self.provider,
                model=self.orchestrator_model,
                api_key=self.openai_api_key if self.provider == "openai" else None,
                host="http://localhost:11434" if self.provider == "ollama" else None
            )

            if self.orchestrator_model != self.agent_model:
                agent_provider = "openai" if "gpt" in self.agent_model else "ollama"
                self.agent_llm_client = create_llm_client(
                    provider=agent_provider,
                    model=self.agent_model,
                    api_key=self.openai_api_key if agent_provider == "openai" else None,
                    host="http://localhost:11434" if agent_provider == "ollama" else None
                )

            self.orchestrator = Orchestrator(
                llm_client=self.llm_client,
                mcp_server_path=self.mcp_server_path,
                intelligence_mode=self.intelligence_mode,
                agent_generation_mode=AgentGenerationMode.STATIC,
                orchestrator_model=self.orchestrator_model,
                agent_model=self.agent_model,
                agent_llm_client=self.agent_llm_client,
                config_dir=self.config_dir,
                log_dir=self.log_dir / "concierge",
                use_two_stage_planning=self.two_stage_planning
            )

            await self.orchestrator.initialize()
            self._cache_agent_info()

            self.conversation = ConversationContext(
                conversation_id=f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

            self.is_initialized = True
            return True

        except Exception as e:
            self.is_initialized = False
            raise RuntimeError(f"Failed to initialize: {e}")

    def _cache_agent_info(self):
        if not self.orchestrator:
            self._agent_info_cache = "No agents available"
            return

        info_parts = []
        for aid, agent in self.orchestrator.agents.items():
            tools = list(agent.available_tools.keys())
            info_parts.append(
                f"- {agent.agent_card.name} ({aid}): {agent.agent_card.role}\n"
                f"  Tools: {', '.join(tools[:5])}{'...' if len(tools) > 5 else ''}"
            )

        self._agent_info_cache = "\n".join(info_parts)

    async def shutdown(self):
        if self.orchestrator:
            await self.orchestrator.shutdown()
        if self.llm_client:
            await self.llm_client.close()
        if self.agent_llm_client:
            await self.agent_llm_client.close()

    def _get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT.format(agent_info=self._agent_info_cache)

    def _is_followup_question(self, request: str) -> bool:
        """Check if this appears to be a follow-up question"""
        request_lower = request.lower().strip()

        # Check for explicit follow-up indicators
        for indicator in self.FOLLOWUP_INDICATORS:
            if indicator in request_lower:
                return True

        # Check if it's a short question (likely relies on context)
        word_count = len(request.split())
        if word_count <= 5:
            return True

        # Check if it starts with conjunctions
        if request_lower.startswith(("and ", "but ", "so ", "then ", "also ")):
            return True

        return False

    def _enrich_followup_request(self, request: str) -> str:
        """Enrich vague follow-up requests with session context"""

        # Only enrich if we have previous context and this looks like a follow-up
        has_context = self.conversation.session_state["last_orchestrator_result"] is not None
        is_followup = self._is_followup_question(request)

        if has_context and is_followup:
            context_summary = self.conversation.get_context_summary()

            enriched = f"""{context_summary}
Current user request: {request}

IMPORTANT: This is a follow-up question. The user is likely referring to the previous simulation/analysis results. 
Use the existing workspace data to fulfill this request without re-running the simulation unless explicitly asked."""

            return enriched

        # Even if not a clear follow-up, include minimal context if available
        if has_context:
            last = self.conversation.session_state["last_orchestrator_result"]
            minimal_context = f"""[Context: Previous task was "{last.get('request', '')[:50]}..." with status: {last.get('status', 'unknown')}]

User request: {request}"""
            return minimal_context

        return request

    async def chat(self, user_message: str, return_full_response: bool = False) -> Union[str, ConciergeResponse]:
        """
        Main entry point for user interaction.
        """
        if not self.is_initialized:
            msg = "I'm not ready yet. Please wait for initialization to complete."
            return ConciergeResponse(message=msg) if return_full_response else msg

        self.conversation.add_message("user", user_message)

        messages = [
            {"role": "system", "content": self._get_system_prompt()}
        ]
        messages.extend(self.conversation.get_history())

        try:
            llm_response: LLMResponse = await self.llm_client.chat(
                messages=messages,
                model=self.orchestrator_model,
                temperature=0.7,
                max_tokens=500
            )

            response_text = llm_response.content.strip()
            orchestrator_response = None
            routed = False

            if "ACTION: ROUTE_TO_ORCHESTRATOR" in response_text:
                conversational_part = response_text.replace("ACTION: ROUTE_TO_ORCHESTRATOR", "").strip()

                orchestrator_response = await self._route_to_orchestrator(user_message)
                self._last_orchestrator_response = orchestrator_response
                routed = True

                final_response = await self._format_orchestrator_response(
                    user_message,
                    orchestrator_response,
                    conversational_part
                )
            else:
                final_response = response_text

            self.conversation.add_message("assistant", final_response)

            if return_full_response:
                return ConciergeResponse(
                    message=final_response,
                    routed_to_orchestrator=routed,
                    orchestrator_response=orchestrator_response
                )
            return final_response

        except Exception as e:
            error_response = f"I encountered an issue: {str(e)}. Could you try rephrasing your request?"
            self.conversation.add_message("assistant", error_response)

            if return_full_response:
                return ConciergeResponse(message=error_response)
            return error_response

    async def chat_verbose(self, user_message: str) -> ConciergeResponse:
        """Convenience method - always returns full response with reasoning data."""
        return await self.chat(user_message, return_full_response=True)

    def get_last_orchestrator_response(self) -> Optional[Dict[str, Any]]:
        """Get the last orchestrator response"""
        return self._last_orchestrator_response

    async def _route_to_orchestrator(self, request: str) -> Dict[str, Any]:
        """Route request to orchestrator with enriched context"""
        try:
            # Enrich the request with session context
            enriched_request = self._enrich_followup_request(request)

            # Build context dict
            context = {
                "conversation_history": self.conversation.get_history(5),
                "session_state": self.conversation.session_state,
                "is_followup": self._is_followup_question(request),
            }

            response = await self.orchestrator.process_request(
                enriched_request,
                context=context
            )

            # Update session state with results
            self.conversation.update_session_state(response, request)

            return response

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "summary": f"Failed to process request: {str(e)}"
            }

    async def _format_orchestrator_response(self,
                                            user_request: str,
                                            result: Dict[str, Any],
                                            conversational_intro: str = "") -> str:
        status = result.get("status", "unknown")
        summary = result.get("summary", "")
        error = result.get("error", "")
        recommendations = result.get("recommendations", [])

        # Safely build result summary - convert complex objects to strings
        try:
            details = result.get("details", {})
            # Convert details to a safe format
            if isinstance(details, dict):
                safe_details = {}
                for k, v in details.items():
                    if isinstance(v, (str, int, float, bool)):
                        safe_details[k] = v
                    elif isinstance(v, (list, dict)):
                        # Convert complex objects to string representation
                        safe_details[k] = json.dumps(v, default=str)[:200]  # Truncate
                    else:
                        safe_details[k] = str(v)[:200]
                details = safe_details
            else:
                details = {"raw": str(details)[:500]}
        except Exception:
            details = {}

        result_summary = {
            "status": status,
            "summary": summary[:500] if summary else "",
            "error": error if status != "success" else None,
            "recommendations": recommendations[:3] if recommendations else None,
            "details": details
        }
        result_summary = {k: v for k, v in result_summary.items() if v}

        formatter_prompt = self.RESPONSE_FORMATTER_PROMPT.format(
            result=json.dumps(result_summary, indent=2, default=str),
            request=user_request
        )

        try:
            llm_response: LLMResponse = await self.llm_client.chat(
                messages=[{"role": "user", "content": formatter_prompt}],
                model=self.orchestrator_model,
                temperature=0.5,
                max_tokens=400
            )

            formatted = llm_response.content.strip()

            if conversational_intro:
                return f"{conversational_intro}\n\n{formatted}"
            return formatted

        except Exception as e:
            # Better fallback with error info
            print(f"Warning: Error formatting response: {e}")
            if status == "success":
                return summary if summary else "Task completed successfully."
            else:
                return f"I had some trouble with that: {error or summary or str(e)}"

    async def chat_streaming(
            self,
            user_message: str,
            reasoning_callback: Optional[ReasoningCallback] = None
    ) -> Dict[str, Any]:
        """
        Chat with real-time reasoning callbacks and session context tracking.
        """
        if not self.is_initialized:
            return {
                "message": "I'm not ready yet. Please wait for initialization.",
                "routed": False
            }

        self.conversation.add_message("user", user_message)

        # Emit: Concierge analyzing
        if reasoning_callback:
            await reasoning_callback({
                "phase": "concierge_routing",
                "source": "concierge",
                "title": "Analyzing request...",
                "content": {
                    "user_message": user_message[:100],
                    "is_followup": self._is_followup_question(user_message),
                    "has_session_context": self.conversation.session_state["last_orchestrator_result"] is not None
                }
            })

        messages = [
            {"role": "system", "content": self._get_system_prompt()}
        ]
        messages.extend(self.conversation.get_history())

        try:
            llm_response: LLMResponse = await self.llm_client.chat(
                messages=messages,
                model=self.orchestrator_model,
                temperature=0.7,
                max_tokens=500
            )

            response_text = llm_response.content.strip()
            orchestrator_response = None
            routed = False

            if "ACTION: ROUTE_TO_ORCHESTRATOR" in response_text:
                conversational_part = response_text.replace("ACTION: ROUTE_TO_ORCHESTRATOR", "").strip()
                routed = True

                # Emit: Routing decision
                if reasoning_callback:
                    await reasoning_callback({
                        "phase": "concierge_routing",
                        "source": "concierge",
                        "title": "Routing to orchestrator",
                        "content": {
                            "decision": "route_to_orchestrator",
                            "is_followup": self._is_followup_question(user_message),
                            "acknowledgment": conversational_part[:100] if conversational_part else "Processing..."
                        }
                    })

                # Enrich request with context
                enriched_request = self._enrich_followup_request(user_message)

                # Build context
                context = {
                    "conversation_history": self.conversation.get_history(5),
                    "session_state": self.conversation.session_state,
                    "is_followup": self._is_followup_question(user_message),
                }

                # Route to orchestrator WITH the callback
                orchestrator_response = await self.orchestrator.process_request(
                    enriched_request,
                    context=context,
                    reasoning_callback=reasoning_callback
                )

                self._last_orchestrator_response = orchestrator_response

                # Update session state
                self.conversation.update_session_state(orchestrator_response, user_message)

                # Format the final response
                final_response = await self._format_orchestrator_response(
                    user_message,
                    orchestrator_response,
                    conversational_part
                )
            else:
                # Direct response from concierge
                if reasoning_callback:
                    await reasoning_callback({
                        "phase": "concierge_routing",
                        "source": "concierge",
                        "title": "Handling directly",
                        "content": {
                            "decision": "direct_response",
                            "response_preview": response_text[:100]
                        }
                    })

                final_response = response_text

            self.conversation.add_message("assistant", final_response)

            return {
                "message": final_response,
                "routed": routed,
                "orchestrator_response": orchestrator_response
            }

        except Exception as e:
            error_response = f"I encountered an issue: {str(e)}. Could you try rephrasing?"
            self.conversation.add_message("assistant", error_response)

            return {
                "message": error_response,
                "routed": False,
                "error": str(e)
            }

    # === Configuration Methods ===

    def set_mode(self, mode: str, two_stage: bool = None):
        if mode.lower() == "centralized":
            self.intelligence_mode = IntelligenceMode.CENTRALIZED
        elif mode.lower() == "decentralized":
            self.intelligence_mode = IntelligenceMode.DECENTRALIZED

        if two_stage is not None:
            self.two_stage_planning = two_stage

        if self.orchestrator:
            self.orchestrator.set_intelligence_mode(self.intelligence_mode)
            self.orchestrator.set_two_stage_planning(self.two_stage_planning)

    def set_models(self, orchestrator_model: str = None, agent_model: str = None):
        if orchestrator_model:
            self.orchestrator_model = orchestrator_model
        if agent_model:
            self.agent_model = agent_model
        if self.orchestrator:
            self.orchestrator.update_models(orchestrator_model, agent_model)

    def get_available_agents(self) -> Dict[str, Dict]:
        if not self.orchestrator:
            return {}
        return {
            aid: {
                "name": agent.agent_card.name,
                "role": agent.agent_card.role,
                "description": agent.agent_card.description,
                "tools": list(agent.available_tools.keys())
            }
            for aid, agent in self.orchestrator.agents.items()
        }

    def get_available_tools(self) -> Dict[str, Dict]:
        if not self.orchestrator:
            return {}
        tools = {}
        for agent in self.orchestrator.agents.values():
            for name, info in agent.available_tools.items():
                if name not in tools:
                    tools[name] = {
                        "description": info.get("description", ""),
                        "parameters": info.get("parameters", [])
                    }
        return tools

    def clear_conversation(self):
        """Clear conversation history and session state"""
        self.conversation = ConversationContext(
            conversation_id=f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self._last_orchestrator_response = None

    def clear_session_only(self):
        """Clear session state but keep conversation history"""
        if self.conversation:
            self.conversation.clear_session()
        self._last_orchestrator_response = None

    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session state"""
        if not self.conversation:
            return {}

        state = self.conversation.session_state
        return {
            "tasks_completed": len(state.get("completed_tasks", [])),
            "simulation_active": state.get("simulation_active", False),
            "last_simulation_id": state.get("last_simulation_id"),
            "workspace_data": list(state.get("workspace_context", {}).keys()),
            "last_task": state.get("last_orchestrator_result", {}).get("request", "None")
        }

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the full conversation history"""
        if not self.conversation:
            return []
        return self.conversation.get_full_history()