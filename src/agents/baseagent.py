"""
Specialist Agent
"""
import json
from typing import Dict, Any, List, Optional, Callable, Awaitable

ReasoningCallback = Callable[[Dict[str, Any]], Awaitable[None]]
import time
import logging
from pathlib import Path
from datetime import datetime

from ..core.data_class import IntelligenceMode, AgentCard, AgentResponse
from ..llm.llm_interface import BaseLLMClient, LLMResponse


class SpecialistAgent:
    """Specialist agent with LLM-based execution in both modes"""
    MAX_TOOL_RETRIES = 1
    def __init__(self,
                 agent_card: AgentCard,
                 llm_client: BaseLLMClient,
                 mcp_client,
                 intelligence_mode: IntelligenceMode = IntelligenceMode.DECENTRALIZED,
                 response_log_dir: Path = Path("logs/agent_responses")):
        self.agent_card = agent_card
        self.llm_client = llm_client
        self.mcp_client = mcp_client
        self.intelligence_mode = intelligence_mode
        self.response_log_dir = response_log_dir

        self.response_log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"Agent.{agent_card.agent_id}")

        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_requests = 0
        self.tool_calls = 0

        self.available_tools = {}
        self._load_tools_from_card()

    def _load_tools_from_card(self):
        """Load tools based on agent card definition"""
        if not self.mcp_client or not self.mcp_client.is_connected:
            self.logger.warning("MCP client not connected")
            return

        all_tools = self.mcp_client.available_tools

        if hasattr(self.agent_card, 'available_tools') and self.agent_card.available_tools:
            for tool_name in self.agent_card.available_tools:
                if tool_name in all_tools:
                    self.available_tools[tool_name] = all_tools[tool_name]
                else:
                    self.logger.warning(f"Tool '{tool_name}' not found on server")
            self.logger.info(f"Loaded {len(self.available_tools)} tools for {self.agent_card.agent_id}")

    def _serialize_tool_result(self, result: Any) -> Any:
        """Convert CallToolResult to JSON-serializable format"""
        if result is None:
            return None
        if hasattr(result, '__dict__'):
            if hasattr(result, 'content'):
                if isinstance(result.content, list):
                    serialized = []
                    for item in result.content:
                        if hasattr(item, 'text'):
                            serialized.append({"type": "text", "text": item.text})
                        elif hasattr(item, '__dict__'):
                            serialized.append(vars(item))
                        else:
                            serialized.append(item)
                    return serialized
                elif hasattr(result.content, '__dict__'):
                    return vars(result.content)
                else:
                    return result.content
            else:
                return {k: self._serialize_tool_result(v) for k, v in vars(result).items()}
        elif isinstance(result, list):
            return [self._serialize_tool_result(item) for item in result]
        elif isinstance(result, dict):
            return {k: self._serialize_tool_result(v) for k, v in result.items()}
        else:
            return result

    async def process_request(
            self,
            request: Dict[str, Any],
            reasoning_callback: Optional[ReasoningCallback] = None
    ) -> Dict[str, Any]:
        """Process request with real-time reasoning callbacks"""
        start_time = time.time()
        tools_used = []
        reasoning_trace = []
        self.total_requests += 1

        tokens_before = {
            "total": self.total_tokens,
            "prompt": self.prompt_tokens,
            "completion": self.completion_tokens
        }

        try:
            if self.intelligence_mode == IntelligenceMode.CENTRALIZED:
                response = await self._process_centralized(
                    request, tools_used, reasoning_trace, reasoning_callback
                )
            else:
                response = await self._process_decentralized(
                    request, tools_used, reasoning_trace, reasoning_callback
                )

            execution_time = time.time() - start_time

            request_prompt_tokens = self.prompt_tokens - tokens_before["prompt"]
            request_completion_tokens = self.completion_tokens - tokens_before["completion"]
            request_total_tokens = self.total_tokens - tokens_before["total"]

            response["metrics"] = {
                "prompt_tokens": request_prompt_tokens,
                "completion_tokens": request_completion_tokens,
                "tokens_used": request_total_tokens,
                "execution_time": execution_time,
                "tool_calls": len(tools_used),
                "mode": self.intelligence_mode.value,
                "model": self.agent_card.model
            }
            response["tools_used"] = tools_used
            response["reasoning_trace"] = reasoning_trace

            await self._log_response(request, response, tools_used, execution_time)
            return response

        except Exception as e:
            self.logger.error(f"Error processing request: {e}", exc_info=True)
            return {"status": "error", "error": str(e), "agent": self.agent_card.agent_id}

    async def _process_centralized(
            self,
            request: Dict[str, Any],
            tools_used: List,
            reasoning_trace: List,
            reasoning_callback: Optional[ReasoningCallback] = None
    ) -> Dict[str, Any]:
        """Centralized processing with streaming"""

        guidance = request.get("orchestrator_guidance", {})
        tool_instructions = guidance.get("tool_instructions", [])
        task = request.get("task", "")

        # === VALIDATION PHASE ===
        validation_prompt = f"""
You are {self.agent_card.name}, a specialist agent.
Role: {self.agent_card.role}

The orchestrator has provided these tool instructions for the task:
Task: {task}

Instructions: {json.dumps(tool_instructions, indent=2)}

Your available tools: {list(self.available_tools.keys())}

Analyze and respond with JSON:
{{
    "validation": "valid" or "needs_adjustment",
    "reasoning": "your analysis of the instructions",
    "refined_instructions": [
        {{"tool": "name", "parameters": {{}}, "reason": "why"}}
    ]
}}

If instructions are valid, return them as-is in refined_instructions.
If adjustments needed, provide corrections.
"""

        try:
            validation_response: LLMResponse = await self.llm_client.chat(
                messages=[{"role": "user", "content": validation_prompt}],
                model=self.agent_card.model,
                temperature=0.1,
                json_mode=True
            )
            self._update_tokens(validation_response)

            validation = json.loads(validation_response.content)
            reasoning_trace.append({
                "phase": "validation",
                "reasoning": validation.get("reasoning", ""),
                "validation_result": validation.get("validation", "")
            })

            if reasoning_callback:
                await reasoning_callback({
                    "phase": "agent_validation",
                    "source": f"agent:{self.agent_card.agent_id}",
                    "title": f"{self.agent_card.name} validating instructions",
                    "content": {
                        "validation_result": validation.get("validation", ""),
                        "reasoning": validation.get("reasoning", ""),
                        "instructions_received": len(tool_instructions),
                        "instructions_refined": len(validation.get("refined_instructions", [])),
                        "tokens_used": validation_response.total_tokens
                    }
                })

            instructions_to_execute = validation.get("refined_instructions", tool_instructions)

        except Exception as e:
            self.logger.warning(f"Validation failed, using original: {e}")
            instructions_to_execute = tool_instructions

            # FIX: Add to reasoning trace on validation failure (was in original)
            reasoning_trace.append({
                "phase": "validation",
                "error": str(e),
                "fallback": "using original instructions"
            })

            if reasoning_callback:
                await reasoning_callback({
                    "phase": "agent_validation",
                    "source": f"agent:{self.agent_card.agent_id}",
                    "title": f"{self.agent_card.name} validation skipped",
                    "content": {
                        "error": str(e),
                        "fallback": "using original instructions"
                    }
                })

        # === EXECUTE TOOLS ===
        results, errors = {}, []

        for instruction in instructions_to_execute:
            tool_name = instruction.get("tool")
            parameters = instruction.get("parameters", {})

            if tool_name not in self.available_tools:
                self.logger.warning(f"Tool '{tool_name}' not available")
                errors.append(f"Tool '{tool_name}' not available")
                continue

            if reasoning_callback:
                await reasoning_callback({
                    "phase": "tool_call_start",
                    "source": f"agent:{self.agent_card.agent_id}",
                    "title": f"Calling: {tool_name}",
                    "content": {
                        "tool": tool_name,
                        "parameters": parameters
                    }
                })

            tool_success = False
            for tool_attempt in range(self.MAX_TOOL_RETRIES + 1):
                try:
                    result = await self.mcp_client.call_tool(tool_name, parameters)
                    serialized = self._serialize_tool_result(result)
                    results[tool_name] = serialized
                    tools_used.append({
                        "tool": tool_name,
                        "parameters": parameters,
                        "result": serialized,
                        "reasoning": instruction.get("reason", "orchestrator instruction"),
                        "success": True,
                        "attempts": tool_attempt + 1
                    })
                    self.tool_calls += 1
                    tool_success = True
                    break
                except Exception as e:
                    if tool_attempt < self.MAX_TOOL_RETRIES:
                        self.logger.warning(f"Tool '{tool_name}' attempt {tool_attempt + 1} failed: {e}, retrying...")
                        continue
                    # Final attempt failed
                    self.logger.error(f"Tool '{tool_name}' failed after {tool_attempt + 1} attempts: {e}")
                    errors.append(f"Error with tool '{tool_name}': {str(e)}")
                    tools_used.append({
                        "tool": tool_name,
                        "parameters": parameters,
                        "error": str(e),
                        "success": False,
                        "attempts": tool_attempt + 1
                    })

        # === SYNTHESIS ===
        response = await self._synthesize_response(
            request, results, errors, reasoning_trace, reasoning_callback
        )
        response["orchestrator_guided"] = True
        response["tool_instructions_received"] = len(tool_instructions)
        response["tool_instructions_executed"] = len(tools_used)
        return response

    async def _suggest_recovery(self, tool_name: str, parameters: Dict, error: str) -> Optional[str]:
        """Use LLM to suggest recovery from tool error"""
        prompt = f"""
Tool '{tool_name}' failed with error: {error}
Parameters used: {json.dumps(parameters)}

Briefly suggest how to handle this error (1-2 sentences).
"""
        try:
            response: LLMResponse = await self.llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.agent_card.model,
                temperature=0.3,
                max_tokens=100
            )
            self._update_tokens(response)
            return response.content
        except:
            return None

    async def _process_decentralized(
            self,
            request: Dict[str, Any],
            tools_used: List,
            reasoning_trace: List,
            reasoning_callback: Optional[ReasoningCallback] = None
    ) -> Dict[str, Any]:
        """Process autonomously with streaming"""
        try:
            # === PLANNING PHASE ===
            tool_plan, planning_reasoning = await self._plan_tool_usage(request)
            reasoning_trace.append({
                "phase": "planning",
                "reasoning": planning_reasoning,
                "planned_tools": [t.get("tool") for t in tool_plan]
            })

            if reasoning_callback:
                await reasoning_callback({
                    "phase": "agent_planning",
                    "source": f"agent:{self.agent_card.agent_id}",
                    "title": f"{self.agent_card.name} planning tool usage",
                    "content": {
                        "reasoning": planning_reasoning[:200],
                        "tools_planned": [t.get("tool") for t in tool_plan],
                        "tool_count": len(tool_plan)
                    }
                })

            results, errors = {}, []

            for tool_call in tool_plan:
                tool_name = tool_call.get("tool")
                parameters = tool_call.get("parameters", {})
                tool_reasoning = tool_call.get("reasoning", "")

                if tool_name in self.available_tools:
                    if reasoning_callback:
                        await reasoning_callback({
                            "phase": "tool_call_start",
                            "source": f"agent:{self.agent_card.agent_id}",
                            "title": f"Calling: {tool_name}",
                            "content": {
                                "tool": tool_name,
                                "parameters": parameters,
                                "reasoning": tool_reasoning
                            }
                        })

                    tool_success = False
                    for tool_attempt in range(self.MAX_TOOL_RETRIES + 1):
                        try:
                            result = await self.mcp_client.call_tool(tool_name, parameters)
                            serialized = self._serialize_tool_result(result)
                            results[tool_name] = serialized
                            tools_used.append({
                                "tool": tool_name,
                                "parameters": parameters,
                                "result": serialized,
                                "reasoning": tool_reasoning,
                                "success": True,
                                "attempts": tool_attempt + 1
                            })
                            self.tool_calls += 1
                            tool_success = True
                            break
                        except Exception as e:
                            if tool_attempt < self.MAX_TOOL_RETRIES:
                                self.logger.warning(f"Tool '{tool_name}' attempt {tool_attempt + 1} failed: {e}, retrying...")
                                continue
                            self.logger.error(f"Tool '{tool_name}' failed after {tool_attempt + 1} attempts: {e}")
                            errors.append(f"Error with tool '{tool_name}': {str(e)}")
                            tools_used.append({
                                "tool": tool_name,
                                "parameters": parameters,
                                "error": str(e),
                                "success": False,
                                "attempts": tool_attempt + 1
                            })
                else:
                    self.logger.warning(f"Tool {tool_name} not available")
                    errors.append(f"Tool '{tool_name}' not available")

            response = await self._synthesize_response(
                request, results, errors, reasoning_trace, reasoning_callback
            )
            response["autonomous_planning"] = True
            response["tools_planned"] = len(tool_plan)
            return response

        except Exception as e:
            self.logger.error(f"Error in decentralized processing: {e}")
            return {"status": "error", "error": str(e), "agent": self.agent_card.agent_id}

    async def _plan_tool_usage(self, request: Dict[str, Any]) -> tuple[List[Dict], str]:
        """Use LLM to plan tool usage - returns plan and reasoning"""
        tool_descriptions = {
            name: {
                "description": info.get("description", ""),
                "parameters": [
                    {"name": p["name"], "type": p["type"], "required": p["required"],
                     "description": p["description"], "default": p.get("default")}
                    for p in info.get("parameters", [])
                ]
            }
            for name, info in self.available_tools.items()
        }

        prompt = f"""
You are {self.agent_card.name}, a specialist agent.
Role: {self.agent_card.role}

Available Tools:
{json.dumps(tool_descriptions, indent=2)}

Task Request: {json.dumps(request, indent=2)}

Analyze the task and plan which tools to use.

Return JSON:
{{
    "reasoning": "step-by-step analysis of the task and why you chose these tools",
    "tool_calls": [
        {{"tool": "name", "parameters": {{}}, "reasoning": "why this tool"}}
    ]
}}
"""

        try:
            response: LLMResponse = await self.llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.agent_card.model,
                temperature=self.agent_card.temperature,
                json_mode=True
            )
            self._update_tokens(response)

            result = json.loads(response.content)
            return result.get("tool_calls", []), result.get("reasoning", "")

        except Exception as e:
            self.logger.error(f"Error planning tool usage: {e}")
            return [], f"Planning failed: {str(e)}"

    async def _synthesize_response(
            self,
            request: Dict[str, Any],
            results: Dict[str, Any],
            errors: List[str],
            reasoning_trace: List,
            reasoning_callback: Optional[ReasoningCallback] = None
    ) -> Dict[str, Any]:
        """Synthesize final response with streaming"""

        # Quick path for simple cases
        if not results and errors:
            return {
                "status": "failed",
                "summary": "Failed to execute tools",
                "data": {},
                "recommendations": [],
                "errors": errors,
                "agent": self.agent_card.agent_id,
                "mode": self.intelligence_mode.value
            }

        if results and not errors and len(results) == 1:
            tool_name = list(results.keys())[0]
            return {
                "status": "success",
                "summary": f"Successfully executed {tool_name}",
                "data": results,
                "recommendations": [],  # FIX: Was missing in revised version
                "errors": None,
                "agent": self.agent_card.agent_id,
                "mode": self.intelligence_mode.value
            }

        # Use LLM for complex synthesis
        prompt = f"""
Synthesize a response from the tool execution results.

Original Request: {json.dumps(request.get('task', request), indent=2)}
Tool Results: {json.dumps(results, indent=2)}
Errors: {json.dumps(errors) if errors else "None"}

Return JSON:
{{
    "status": "success" or "partial" or "failed",
    "summary": "clear summary of what was accomplished",
    "key_findings": ["important finding 1", "finding 2"],
    "data": {{}},
    "recommendations": ["actionable recommendation"],
    "errors": []
}}
"""

        try:
            response: LLMResponse = await self.llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.agent_card.model,
                temperature=self.agent_card.temperature,
                json_mode=True
            )
            self._update_tokens(response)

            synthesis = json.loads(response.content)
            synthesis["agent"] = self.agent_card.agent_id
            synthesis["mode"] = self.intelligence_mode.value

            reasoning_trace.append({
                "phase": "synthesis",
                "summary": synthesis.get("summary", "")
            })

            if reasoning_callback:
                await reasoning_callback({
                    "phase": "agent_synthesis",
                    "source": f"agent:{self.agent_card.agent_id}",
                    "title": f"{self.agent_card.name} synthesized results",
                    "content": {
                        "status": synthesis.get("status", ""),
                        "summary": synthesis.get("summary", "")[:200],
                        "key_findings": synthesis.get("key_findings", [])[:3],
                        "tokens_used": response.total_tokens
                    }
                })

            return synthesis

        except Exception as e:
            self.logger.error(f"Error synthesizing response: {e}")
            return {
                "status": "partial" if results else "failed",
                "summary": "Error during synthesis",
                "data": results,
                "recommendations": [],  # FIX: Was missing in revised version
                "errors": (errors + [f"Synthesis error: {str(e)}"]) if errors else [str(e)],
                "agent": self.agent_card.agent_id,
                "mode": self.intelligence_mode.value
            }

    def _update_tokens(self, response: LLMResponse):
        """Update token counts from LLM response"""
        self.prompt_tokens += response.prompt_tokens
        self.completion_tokens += response.completion_tokens
        self.total_tokens += response.total_tokens

    async def _log_response(self, request: Dict[str, Any], response: Dict[str, Any],
                            tools_used: List[Dict[str, Any]], execution_time: float):
        """Log agent response"""
        timestamp = datetime.now().isoformat()
        agent_response = AgentResponse(
            agent_id=self.agent_card.agent_id,
            timestamp=timestamp,
            request=request,
            response=response,
            tools_used=tools_used,
            token_usage={
                "prompt": self.prompt_tokens,
                "completion": self.completion_tokens,
                "total": self.total_tokens
            },
            execution_time=execution_time,
            mode=self.intelligence_mode.value,
            model=self.agent_card.model
        )
        log_file = self.response_log_dir / f"{self.agent_card.agent_id}_{timestamp.replace(':', '-')}.json"
        try:
            with open(log_file, 'w') as f:
                f.write(agent_response.to_json())
        except Exception as e:
            self.logger.error(f"Failed to log response: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get detailed agent performance metrics"""
        return {
            "agent_id": self.agent_card.agent_id,
            "total_requests": self.total_requests,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "total_tool_calls": self.tool_calls,
            "available_tools": len(self.available_tools),
            "mode": self.intelligence_mode.value,
            "model": self.agent_card.model
        }

    def update_model(self, model: str):
        """Update the LLM model"""
        self.agent_card.model = model

    def update_tools(self, tool_names: List[str]):
        """Update agent's available tools (used by orchestrator in dynamic mode)"""
        all_tools = self.mcp_client.available_tools
        self.available_tools = {}
        for name in tool_names:
            if name in all_tools:
                self.available_tools[name] = all_tools[name]
        self.logger.info(f"Updated tools: {len(self.available_tools)} tools available")

    def get_capability_summary(self) -> Dict[str, Any]:
        """Return agent capability summary for orchestrator"""
        return {
            "agent_id": self.agent_card.agent_id,
            "name": self.agent_card.name,
            "role": self.agent_card.role,
            "description": self.agent_card.description,
            "available_tools": list(self.available_tools.keys()),
            "tool_count": len(self.available_tools),
            "model": self.agent_card.model
        }