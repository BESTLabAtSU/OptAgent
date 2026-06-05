"""
Orchestrator
"""
from typing import Dict, Any, List, Optional, Callable, Awaitable
import json
from pathlib import Path
import logging
import asyncio
from datetime import datetime

ReasoningCallback = Callable[[Dict[str, Any]], Awaitable[None]]

from ..core.data_class import IntelligenceMode, AgentGenerationMode, OrchestratorResponse, FailureType
from ..core.utils import safe_json_parse
from ..core.matrix import MetricsCollector
from ..agents.factory import AgentFactory
from ..mcp_center.mcp_client import MCPClient, MCPClientPool
from ..llm.llm_interface import BaseLLMClient, LLMResponse


class Orchestrator:
    """Orchestrator with static/dynamic agent generation and dual LLM support"""

    WORKFLOW_PATTERNS = """
    ## Common Workflow Patterns (chain of thought):

    ### Pattern 1: Comparison Study 
    Keywords: "compare", "change", "upgrade", "after", "before vs after", "how will X change", "what if"
    When user asks about impact of changes or comparisons:
    1. FIRST: Run simulation with CURRENT config and save this run with a name such as "sim_old"
    2. THEN: Update relevant system parameters (HVAC, DER, controllers...) with requested changes
    3. Save the updated configuration using config_save with descriptive name such as "new_config"
    4. THEN: Run simulation with UPDATED config and this run with descriptive name such as "sim_new"
    5. FINALLY: Call comparison tools to compare differences with name "sim_old" and "sim_new"

    ### Pattern 2: Configuration
    Keywords: "update", "change", "set", "modify", "replace", "upgrade"
    When modifying system parameters:
    Use 'Config Agent' for file level operation like: save, validate, create, or list 
    Use 'Component Agent' such as 'hvac_agent, der_agent, controller_agent, building_agent, disturbance_agent, environment_agent' for add, update, query...

    ### Pattern 3: Full Analysis
    Keywords: "analyze", "evaluate", "assess performance"
    After any simulation:
    1. Use analysis_comprehensive OR multiple analysis tools (comfort, energy, cost, flexibility)
    2. Analysis requires a completed simulation name

    ## Key Dependencies:
    - comparison_* tools REQUIRE TWO completed simulations
    - analysis_* tools REQUIRE ONE completed simulation  
    - "Baseline" simulation should be run BEFORE making changes for comparison studies
    """

    MAX_STEP_RETRIES = 2          # Max retries per execution step
    TIMEOUT_BY_MODEL = {
        "qwen3:1.7b": 90, "qwen3:4b": 90, "qwen3:8b": 90,
        "qwen3:14b": 90, "gpt-4o-mini": 90, "gpt-5.2": 90,
    }
    STEP_TIMEOUT_DEFAULT = 120

    def __init__(self,
                 llm_client: BaseLLMClient,
                 mcp_server_path: str = None,
                 intelligence_mode: IntelligenceMode = IntelligenceMode.DECENTRALIZED,
                 agent_generation_mode: AgentGenerationMode = AgentGenerationMode.STATIC,
                 orchestrator_model: str = "gpt-4o-mini",
                 agent_model: str = "gpt-3.5-turbo",
                 agent_llm_client: BaseLLMClient = None,
                 config_dir: Path = Path("config"),
                 log_dir: Path = Path("logs"),
                 use_client_pool: bool = False,
                 pool_size: int = 3,
                 use_two_stage_planning: bool = False):

        self.llm_client = llm_client
        self.agent_llm_client = agent_llm_client or llm_client
        self.mcp_server_path = mcp_server_path
        self.intelligence_mode = intelligence_mode
        self.agent_generation_mode = agent_generation_mode
        self.orchestrator_model = orchestrator_model
        self.agent_model = agent_model
        self.config_dir = config_dir
        self.log_dir = log_dir
        self.use_client_pool = use_client_pool
        self.pool_size = pool_size
        self.use_two_stage_planning = use_two_stage_planning
        self.current_metrics: MetricsCollector = None

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.orchestrator_log_dir = log_dir / "orchestrator"
        self.orchestrator_log_dir.mkdir(exist_ok=True)

        self.logger = logging.getLogger("Orchestrator")
        self.mcp_client = MCPClient(server_script_path=mcp_server_path)
        self.mcp_client_pool = None
        self.agent_factory = None
        self.agents = {}

    async def initialize(self):
        """Initialize orchestrator"""
        try:
            await self.mcp_client.connect()
            self.logger.info("Connected to MCP server")

            if self.use_client_pool:
                self.mcp_client_pool = MCPClientPool(
                    pool_size=self.pool_size,
                    server_script_path=self.mcp_server_path
                )
                await self.mcp_client_pool.initialize()

            self.agent_factory = AgentFactory(
                mcp_client=self.mcp_client,
                llm_client=self.agent_llm_client,
                config_dir=self.config_dir / "agents",
                response_log_dir=self.log_dir / "agent_responses",
                default_model=self.agent_model,
                intelligence_mode=self.intelligence_mode
            )

            await self.agent_factory.initialize()

            if self.intelligence_mode == IntelligenceMode.SINGLE_AGENT_REACT:
                # Create single universal agent with ALL tools
                all_tool_names = list(self.mcp_client.available_tools.keys())
                universal_agent = await self.agent_factory.create_agent_from_spec({
                    "agent_id": "universal_agent",
                    "name": "Universal Agent",
                    "role": "General-purpose building energy agent with access to all tools",
                    "description": "Handles all building energy tasks autonomously using any available tool",
                    "available_tools": all_tool_names,
                    "model": self.agent_model,
                    "temperature": 0.3,
                })
                # Force decentralized behavior so agent plans its own tools
                universal_agent.intelligence_mode = IntelligenceMode.DECENTRALIZED
                self.agents = {"universal_agent": universal_agent}
            else:
                self.agents = await self.agent_factory.create_all_agents(model_override=self.agent_model)

            self.logger.info(
                f"Initialized {len(self.agents)} agents "
                f"(intelligence: {self.intelligence_mode.value}, "
                f"generation: {self.agent_generation_mode.value}, "
                f"two_stage: {self.use_two_stage_planning})"
            )

            orch_provider = getattr(self.llm_client, 'provider', 'unknown')
            agent_provider = getattr(self.agent_llm_client, 'provider', 'unknown')
            self.logger.info(
                f"LLM clients - Orchestrator: {orch_provider}/{self.orchestrator_model}, "
                f"Agents: {agent_provider}/{self.agent_model}"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize: {e}")
            raise

    async def shutdown(self):
        """Shutdown orchestrator"""
        try:
            if self.mcp_client_pool:
                await self.mcp_client_pool.shutdown()
            if self.mcp_client:
                await self.mcp_client.disconnect()
            self.logger.info("Orchestrator shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

    async def process_request(
            self,
            user_request: str,
            context: Optional[Dict[str, Any]] = None,
            reasoning_callback: Optional[ReasoningCallback] = None
    ) -> Dict[str, Any]:
        """Process user request with real-time reasoning callbacks AND complete metrics"""

        self.current_metrics = MetricsCollector()
        self.current_metrics.set_config(
            intelligence_mode=self.intelligence_mode.value,
            agent_gen_mode=self.agent_generation_mode.value,
            orchestrator_model=self.orchestrator_model,
            agent_model=self.agent_model,
            provider=getattr(self.llm_client, 'provider', 'unknown').value
            if hasattr(self.llm_client, 'provider') else 'unknown',
            two_stage_planning=self.use_two_stage_planning
        )

        agent_modifications = []

        try:
            # === DYNAMIC AGENT CHECK ===
            if self.agent_generation_mode == AgentGenerationMode.DYNAMIC:
                self.current_metrics.start_phase("agent_check")

                if reasoning_callback:
                    await reasoning_callback({
                        "phase": "agent_check",
                        "source": "orchestrator",
                        "title": "Checking agent capabilities...",
                        "content": {"request": user_request[:100]}
                    })

                agent_modifications = await self._check_and_prepare_agents(
                    user_request, context, reasoning_callback
                )
                self.current_metrics.end_phase("agent_check")
                self.current_metrics.metrics.orchestrator.agent_modifications = agent_modifications

            # === PLANNING PHASE ===
            self.current_metrics.start_phase("planning")

            # FIX: Add reasoning step for planning start (was in original)
            self.current_metrics.add_reasoning_step("planning_start", {
                "request": user_request,
                "mode": self.intelligence_mode.value,
                "two_stage": self.use_two_stage_planning
            })

            if reasoning_callback:
                await reasoning_callback({
                    "phase": "planning_start",
                    "source": "orchestrator",
                    "title": "Creating execution plan...",
                    "content": {
                        "mode": self.intelligence_mode.value,
                        "two_stage": self.use_two_stage_planning,
                        "available_agents": list(self.agents.keys())
                    }
                })

            execution_plan = await self._create_execution_plan(
                user_request, context, reasoning_callback
            )

            self.current_metrics.end_phase("planning")
            self.current_metrics.metrics.orchestrator.execution_plan = execution_plan

            # FIX: Add reasoning step for planning complete (was in original)
            self.current_metrics.add_reasoning_step("planning_complete", {
                "steps": len(execution_plan),
                "agents": list(set(s.get("agent_id") for s in execution_plan))
            })

            # === EXECUTION PHASE ===
            self.current_metrics.start_phase("execution")
            agent_results = await self._execute_plan(
                execution_plan, context, reasoning_callback
            )
            self.current_metrics.end_phase("execution")

            # FIX: Record agent results (MISSING in revised version)
            for step_id, result in agent_results.items():
                agent_id = result.get("agent", "unknown")
                agent_model = result.get("metrics", {}).get("model", self.agent_model)
                self.current_metrics.record_agent_result(agent_id, agent_model, result)

                if "reasoning_trace" in result:
                    self.current_metrics.add_reasoning_step(f"agent_{agent_id}", {
                        "trace": result["reasoning_trace"]
                    })

            # === SYNTHESIS PHASE ===
            self.current_metrics.start_phase("synthesis")

            if reasoning_callback:
                await reasoning_callback({
                    "phase": "synthesis_start",
                    "source": "orchestrator",
                    "title": "Synthesizing final response...",
                    "content": {"steps_completed": len(agent_results)}
                })

            final_response = await self._synthesize_response(
                user_request, agent_results, reasoning_callback
            )
            self.current_metrics.end_phase("synthesis")

            # FIX: Calculate quality metrics (MISSING in revised version)
            success = final_response.get("status") == "success"
            accuracy = self._calculate_accuracy_score(final_response, agent_results)
            completeness = self._calculate_completeness_score(execution_plan, agent_results)
            self.current_metrics.set_quality_metrics(success, accuracy, completeness)

            # FIX: Finalize metrics (MISSING in revised version)
            final_metrics = self.current_metrics.finalize()

            # FIX: Attach benchmark data to response (MISSING in revised version)
            final_response["benchmark"] = {
                "metrics": final_metrics.to_dict(),
                "agent_modifications": agent_modifications,
                "agent_results": agent_results,
                "execution_plan": execution_plan,
                "models": {
                    "orchestrator": self.orchestrator_model,
                    "agents": self.agent_model
                },
                "two_stage_planning": self.use_two_stage_planning,
            }

            # FIX: Log response (MISSING in revised version)
            await self._log_response(user_request, execution_plan, agent_results,
                                     final_response, final_metrics.to_dict())

            return final_response


        except Exception as e:
            self.logger.error(f"Error processing request: {e}")
            error_response = {
                "status": "error",
                "error": str(e),
                "request": user_request,
                "failure_type": FailureType.RUNTIME_FAILURE.value,
            }
            # Preserve partial metrics
            if self.current_metrics:
                try:
                    partial_metrics = self.current_metrics.finalize()
                    error_response["benchmark"] = {
                        "metrics": partial_metrics.to_dict(),
                        "agent_modifications": agent_modifications if 'agent_modifications' in dir() else [],
                        "agent_results": agent_results if 'agent_results' in dir() else {},
                        "execution_plan": execution_plan if 'execution_plan' in dir() else [],
                        "models": {
                            "orchestrator": self.orchestrator_model,
                            "agents": self.agent_model
                        },
                        "two_stage_planning": self.use_two_stage_planning,
                    }
                except Exception:
                    pass
            return error_response


    async def _check_and_prepare_agents(
            self,
            user_request: str,
            context: Optional[Dict[str, Any]],
            reasoning_callback: Optional[ReasoningCallback] = None
    ) -> List[Dict]:
        """Dynamic mode: Check if current agents can handle the task."""
        modifications = []

        agent_capabilities = {
            aid: agent.get_capability_summary() for aid, agent in self.agents.items()
        }

        all_server_tools = list(self.mcp_client.available_tools.keys())
        tools_in_use = set()
        for agent in self.agents.values():
            tools_in_use.update(agent.available_tools.keys())
        unused_tools = set(all_server_tools) - tools_in_use

        prompt = f"""
Analyze if current agents can handle this task, or if modifications are needed.

User Request: {user_request}
Context: {json.dumps(context) if context else "None"}

Current Agents:
{json.dumps(agent_capabilities, indent=2)}

All Server Tools: {json.dumps(all_server_tools)}
Tools Not Assigned to Any Agent: {json.dumps(list(unused_tools))}

Return JSON:
{{
    "can_handle": true/false,
    "analysis": "explanation",
    "modifications": [
        {{
            "action": "create" or "revise",
            "agent_id": "id",
            "name": "name (for create)",
            "role": "role (for create)",
            "description": "description (for create)",
            "tools": ["tool_names"],
            "reason": "why needed"
        }}
    ]
}}
"""

        response: LLMResponse = await self.llm_client.chat(
            messages=[{"role": "user", "content": prompt}],
            model=self.orchestrator_model,
            temperature=0.2,
            json_mode=True
        )

        self.current_metrics.record_orchestrator_llm_call(response, "agent_check")
        result, parsed_ok, parse_error = safe_json_parse(
          response.content, fallback={"can_handle": True, "analysis": "", "modifications": []}
        )
        if not parsed_ok:
            self.logger.error(f"Agent check JSON parse failed: {parse_error}")

        # FIX: Keep original logging (was in original)
        self.logger.info(f"Agent check: can_handle={result['can_handle']}")

        if reasoning_callback:
            await reasoning_callback({
                "phase": "agent_check",
                "source": "orchestrator",
                "title": "Agent capability check complete",
                "content": {
                    "can_handle": result.get("can_handle", True),
                    "analysis": result.get("analysis", ""),
                    "modifications_needed": len(result.get("modifications", [])),
                    "modifications": [
                        {
                            "action": m.get("action"),
                            "agent_id": m.get("agent_id"),
                            "reason": m.get("reason")
                        }
                        for m in result.get("modifications", [])
                    ],
                    "tokens_used": response.total_tokens
                }
            })

        for mod in result.get("modifications", []):
            try:
                if mod["action"] == "create":
                    if reasoning_callback:
                        await reasoning_callback({
                            "phase": "agent_check",
                            "source": "orchestrator",
                            "title": f"Creating new agent: {mod['agent_id']}",
                            "content": {
                                "action": "create",
                                "agent_id": mod["agent_id"],
                                "name": mod.get("name", ""),
                                "role": mod.get("role", ""),
                                "tools": mod.get("tools", []),
                                "reason": mod.get("reason", "")
                            }
                        })

                    new_agent = await self.agent_factory.create_agent_from_spec({
                        "agent_id": mod["agent_id"],
                        "name": mod["name"],
                        "role": mod["role"],
                        "description": mod.get("description", mod["role"]),
                        "available_tools": mod["tools"],
                        "model": self.agent_model
                    })
                    self.agents[mod["agent_id"]] = new_agent
                    modifications.append({
                        "action": "created",
                        "agent_id": mod["agent_id"],
                        "reason": mod["reason"]
                    })

                elif mod["action"] == "revise":
                    if mod["agent_id"] in self.agents:
                        if reasoning_callback:
                            await reasoning_callback({
                                "phase": "agent_check",
                                "source": "orchestrator",
                                "title": f"Revising agent: {mod['agent_id']}",
                                "content": {
                                    "action": "revise",
                                    "agent_id": mod["agent_id"],
                                    "new_tools": mod.get("tools", []),
                                    "reason": mod.get("reason", "")
                                }
                            })

                        await self.agent_factory.revise_agent(
                            self.agents[mod["agent_id"]],
                            mod["tools"]
                        )
                        modifications.append({
                            "action": "revised",
                            "agent_id": mod["agent_id"],
                            "reason": mod["reason"]
                        })

            except Exception as e:
                self.logger.error(f"Failed to apply modification {mod}: {e}")
                if reasoning_callback:
                    await reasoning_callback({
                        "phase": "agent_check",
                        "source": "orchestrator",
                        "title": f"Failed to modify agent: {mod.get('agent_id', '?')}",
                        "content": {"error": str(e), "modification": mod}
                    })

        return modifications

    async def _create_execution_plan(
            self,
            user_request: str,
            context: Optional[Dict[str, Any]],
            reasoning_callback: Optional[ReasoningCallback] = None
    ) -> List[Dict[str, Any]]:
        """Create execution plan based on intelligence mode"""
        if self.intelligence_mode == IntelligenceMode.SINGLE_AGENT_REACT:
            return await self._create_react_single_agent_plan(
                user_request, context, reasoning_callback
            )
        elif self.intelligence_mode == IntelligenceMode.CENTRALIZED:
            if self.use_two_stage_planning:
                return await self._create_centralized_plan_two_stage(
                    user_request, context, reasoning_callback
                )
            else:
                return await self._create_centralized_plan(
                    user_request, context, reasoning_callback
                )
        else:
            return await self._create_decentralized_plan(
                user_request, context, reasoning_callback
            )

    async def _create_react_single_agent_plan(
            self,
            user_request: str,
            context: Optional[Dict[str, Any]],
            reasoning_callback: Optional[ReasoningCallback] = None
    ) -> List[Dict[str, Any]]:
        """
        ReAct single-agent baseline: NO orchestrator LLM call.
        Creates a trivial one-step plan that delegates everything to the universal agent.
        The agent itself will plan and execute autonomously (decentralized-style).
        """
        plan = [{
            "step_id": "step_1",
            "agent_id": "universal_agent",
            "task": user_request,
            "depends_on": [],
            "expected_outcome": "Complete the full user request using available tools"
        }]

        self.current_metrics.add_reasoning_step("react_passthrough", {
            "mode": "single_agent_react",
            "note": "No orchestrator planning — full delegation to universal agent"
        })

        if reasoning_callback:
            await reasoning_callback({
                "phase": "planning_stage1",
                "source": "orchestrator",
                "title": "ReAct baseline: delegating to universal agent",
                "content": {
                    "mode": "single_agent_react",
                    "agent": "universal_agent",
                    "tools_available": len(self.agents["universal_agent"].available_tools),
                    "tokens_used": 0
                }
            })

        return plan

    async def _create_centralized_plan(
            self,
            user_request: str,
            context: Optional[Dict[str, Any]],
            reasoning_callback: Optional[ReasoningCallback] = None
    ) -> List[Dict[str, Any]]:
        """Centralized plan with detailed tool instructions (ONE-STAGE)"""

        agent_info = {}
        for agent_id, agent in self.agents.items():
            tool_descriptions = {
                name: {
                    "description": info.get("description", ""),
                    "parameters": [
                        {
                            "name": p["name"],
                            "type": p["type"],
                            "required": p["required"],
                            "description": p["description"],
                            "default": p.get("default")
                        }
                        for p in info.get("parameters", [])
                    ]
                }
                for name, info in agent.available_tools.items()
            }

            agent_info[agent_id] = {
                "name": agent.agent_card.name,
                "role": agent.agent_card.role,
                "description": agent.agent_card.description,
                "available_tools": tool_descriptions
            }

        prompt = f"""
Create detailed execution plan for CENTRALIZED mode.

{self.WORKFLOW_PATTERNS}

## User Request: 
{user_request}

## Context: 
{json.dumps(context) if context else "None"}

## Available Agents (with full tool schemas):
{json.dumps(agent_info, indent=2)}

Return JSON:
{{
    "understanding": "what the user wants",
    "reasoning": "analysis referencing the workflow patterns",
    "steps": [
        {{
            "step_id": "step_1",
            "agent_id": "agent_id",
            "task": "task description",
            "depends_on": [],
            "orchestrator_guidance": {{
                "tool_instructions": [
                    {{
                        "tool": "tool_name", 
                        "parameters": {{"param_name": "value"}},
                        "expected_output": "..."
                    }}
                ],
                "validation": "how to verify"
            }}
        }}
    ]
}}
"""

        response: LLMResponse = await self.llm_client.chat(
            messages=[{"role": "user", "content": prompt}],
            model=self.orchestrator_model,
            temperature=0.2,
            json_mode=True
        )

        self.current_metrics.record_orchestrator_llm_call(response, "planning")
        # plan = json.loads(response.content)
        self.current_metrics.add_reasoning_step("raw_planning_response", {
            "phase": "centralized_one_stage",
            "raw_content": response.content[:2000],
            "model": self.orchestrator_model
        })

        plan, parsed_ok, parse_error = safe_json_parse(
              response.content, fallback={"understanding": "", "reasoning": "", "steps": []}
          )
        if not parsed_ok:
              self.logger.error(f"Planning JSON parse failed: {parse_error}")
              self.current_metrics.add_reasoning_step("json_parse_failure", {
                  "phase": "planning", "error": parse_error,
                  "raw_content": response.content[:500]
              })
        # FIX: Add reasoning step (was in original)
        self.current_metrics.add_reasoning_step("orchestrator_planning", {
            "understanding": plan.get("understanding", ""),
            "reasoning": plan.get("reasoning", ""),
            "steps_planned": len(plan.get("steps", []))
        })

        if reasoning_callback:
            await reasoning_callback({
                "phase": "planning_stage1",
                "source": "orchestrator",
                "title": "Execution plan created (one-stage centralized)",
                "content": {
                    "understanding": plan.get("understanding", ""),
                    "reasoning": plan.get("reasoning", ""),
                    "steps": [
                        {
                            "step_id": s.get("step_id"),
                            "agent": s.get("agent_id"),
                            "task": s.get("task"),
                            "tools": [
                                t.get("tool")
                                for t in s.get("orchestrator_guidance", {}).get("tool_instructions", [])
                            ],
                            "params": [
                                {
                                    "tool": t.get("tool"),
                                    "parameters": t.get("parameters", {})
                                }
                                for t in s.get("orchestrator_guidance", {}).get("tool_instructions", [])
                            ]
                        }
                        for s in plan.get("steps", [])
                    ],
                    "tokens_used": response.total_tokens,
                    "mode": "one_stage_centralized"
                }
            })

        self.logger.info(f"Created centralized plan with {len(plan['steps'])} steps")
        return plan['steps']

    async def _create_centralized_plan_two_stage(
            self,
            user_request: str,
            context: Optional[Dict[str, Any]],
            reasoning_callback: Optional[ReasoningCallback] = None
    ) -> List[Dict[str, Any]]:
        """Two-stage centralized planning with streaming"""

        # Stage 1: Basic info for initial planning
        basic_agent_info = {}
        for agent_id, agent in self.agents.items():
            basic_agent_info[agent_id] = {
                "name": agent.agent_card.name,
                "role": agent.agent_card.role,
                "description": agent.agent_card.description,
                "tools": list(agent.available_tools.keys())
            }

        stage1_prompt = f"""
Create a high-level execution plan. Select which agents and tools to use.

{self.WORKFLOW_PATTERNS}

## User Request: 
{user_request}

## Context: 
{json.dumps(context) if context else "None"}

## Available Agents (with tool names only):
{json.dumps(basic_agent_info, indent=2)}

Return JSON:
{{
    "understanding": "what the user wants",
    "reasoning": "why these agents/tools, in this order, referencing the workflow patterns",
    "steps": [
        {{
            "step_id": "step_1",
            "agent_id": "agent_id",
            "task": "task description",
            "depends_on": [],
            "tools_to_use": ["tool_name1", "tool_name2"]
        }}
    ]
}}
"""

        response1: LLMResponse = await self.llm_client.chat(
            messages=[{"role": "user", "content": stage1_prompt}],
            model=self.orchestrator_model,
            temperature=0.2,
            json_mode=True
        )

        self.current_metrics.record_orchestrator_llm_call(response1, "planning_stage1")

        self.current_metrics.add_reasoning_step("raw_planning_response", {
            "phase": "centralized_two_stage_1",
            "raw_content": response1.content[:2000],
            "model": self.orchestrator_model
        })

        # stage1_plan = json.loads(response1.content)
        stage1_plan, parsed_ok, parse_error = safe_json_parse(
           response1.content, fallback={"understanding": "", "reasoning": "", "steps": []}
        )
        if not parsed_ok:
              self.logger.error(f"Stage 1 JSON parse failed: {parse_error}")
              self.current_metrics.add_reasoning_step("json_parse_failure", {
                  "phase": "planning_stage1", "error": parse_error,
                  "raw_content": response1.content[:500]
              })

        # FIX: Add reasoning step for stage 1 (was in original)
        self.current_metrics.add_reasoning_step("orchestrator_planning_stage1", {
            "understanding": stage1_plan.get("understanding", ""),
            "reasoning": stage1_plan.get("reasoning", ""),
            "steps_planned": len(stage1_plan.get("steps", []))
        })

        if reasoning_callback:
            await reasoning_callback({
                "phase": "planning_stage1",
                "source": "orchestrator",
                "title": "Stage 1: High-level plan created",
                "content": {
                    "understanding": stage1_plan.get("understanding", ""),
                    "reasoning": stage1_plan.get("reasoning", ""),
                    "steps": [
                        {
                            "step_id": s.get("step_id"),
                            "agent": s.get("agent_id"),
                            "task": s.get("task"),
                            "tools": s.get("tools_to_use", [])
                        }
                        for s in stage1_plan.get("steps", [])
                    ],
                    "tokens_used": response1.total_tokens
                }
            })

        self.logger.info(f"Stage 1: Created plan with {len(stage1_plan['steps'])} steps")

        # Stage 2: Get detailed parameters for selected tools only
        selected_tools = set()
        for step in stage1_plan.get("steps", []):
            selected_tools.update(step.get("tools_to_use", []))

        detailed_tool_info = {}
        for agent_id, agent in self.agents.items():
            for tool_name, tool_info in agent.available_tools.items():
                if tool_name in selected_tools:
                    detailed_tool_info[tool_name] = {
                        "description": tool_info.get("description", ""),
                        "parameters": [
                            {
                                "name": p["name"],
                                "type": p["type"],
                                "required": p["required"],
                                "description": p["description"],
                                "default": p.get("default")
                            }
                            for p in tool_info.get("parameters", [])
                        ]
                    }

        stage2_prompt = f"""
Complete the execution plan with specific tool parameters.

User Request: {user_request}
Plan from Stage 1: {json.dumps(stage1_plan['steps'], indent=2)}

Tool Parameter Details (for selected tools only):
{json.dumps(detailed_tool_info, indent=2)}

For each step, fill in the orchestrator_guidance with exact parameters.
Return JSON:
{{
    "steps": [
        {{
            "step_id": "step_1",
            "agent_id": "agent_id",
            "task": "task description",
            "depends_on": [],
            "orchestrator_guidance": {{
                "tool_instructions": [
                    {{
                        "tool": "tool_name",
                        "parameters": {{"param_name": "value"}},
                        "expected_output": "..."
                    }}
                ],
                "validation": "how to verify"
            }}
        }}
    ]
}}
"""

        response2: LLMResponse = await self.llm_client.chat(
            messages=[{"role": "user", "content": stage2_prompt}],
            model=self.orchestrator_model,
            temperature=0.2,
            json_mode=True
        )

        self.current_metrics.record_orchestrator_llm_call(response2, "planning_stage2")

        self.current_metrics.add_reasoning_step("raw_planning_response", {
            "phase": "centralized_two_stage_2",
            "raw_content": response2.content[:2000],
            "model": self.orchestrator_model
        })

        # stage2_plan = json.loads(response2.content)
        stage2_plan, parsed_ok, parse_error = safe_json_parse(
              response2.content, fallback={"steps": []}
          )
        if not parsed_ok:
              self.logger.error(f"Stage 2 JSON parse failed: {parse_error}")
              self.current_metrics.add_reasoning_step("json_parse_failure", {
                  "phase": "planning_stage2", "error": parse_error,
                  "raw_content": response2.content[:500]
              })

        # FIX: Add reasoning step for stage 2 (was in original)
        self.current_metrics.add_reasoning_step("orchestrator_planning_stage2", {
            "tools_detailed": list(selected_tools),
            "steps_completed": len(stage2_plan.get("steps", []))
        })

        if reasoning_callback:
            await reasoning_callback({
                "phase": "planning_stage2",
                "source": "orchestrator",
                "title": "Stage 2: Detailed parameters filled",
                "content": {
                    "tools_detailed": list(selected_tools),
                    "steps_with_params": [
                        {
                            "step_id": s.get("step_id"),
                            "agent": s.get("agent_id"),
                            "task": s.get("task", "")[:60],
                            "tools": [
                                {
                                    "tool": t.get("tool"),
                                    "params": t.get("parameters", {})
                                }
                                for t in s.get("orchestrator_guidance", {}).get("tool_instructions", [])
                            ]
                        }
                        for s in stage2_plan.get("steps", [])
                    ],
                    "tokens_used": response2.total_tokens
                }
            })

        self.logger.info(f"Stage 2: Completed plan with detailed parameters")
        return stage2_plan['steps']

    async def _create_decentralized_plan(
            self,
            user_request: str,
            context: Optional[Dict[str, Any]],
            reasoning_callback: Optional[ReasoningCallback] = None
    ) -> List[Dict[str, Any]]:
        """Decentralized plan - high level only, agents decide tools"""

        agent_info = {
            aid: {
                "name": a.agent_card.name,
                "role": a.agent_card.role,
                "description": a.agent_card.description,
                "tool_count": len(a.available_tools)
            }
            for aid, a in self.agents.items()
        }

        prompt = f"""
Create high-level plan for DECENTRALIZED mode. Agents will decide their own tool usage.

{self.WORKFLOW_PATTERNS}

## User Request: 
{user_request}

## Context: 
{json.dumps(context) if context else "None"}

## Available Agents:
{json.dumps(agent_info, indent=2)}

Return JSON:
{{
    "understanding": "what the user wants",
    "reasoning": "why these agents in this order, referencing the patterns",
    "steps": [
        {{
            "step_id": "step_1",
            "agent_id": "agent_id",
            "task": "high-level task",
            "depends_on": [],
            "expected_outcome": "what should be achieved"
        }}
    ]
}}
"""

        response: LLMResponse = await self.llm_client.chat(
            messages=[{"role": "user", "content": prompt}],
            model=self.orchestrator_model,
            temperature=0.2,
            json_mode=True
        )

        self.current_metrics.record_orchestrator_llm_call(response, "planning")

        self.current_metrics.add_reasoning_step("raw_planning_response", {
            "phase": "decentralized",
            "raw_content": response.content[:2000],
            "model": self.orchestrator_model
        })

        # plan = json.loads(response.content)
        plan, parsed_ok, parse_error = safe_json_parse(
                      response.content, fallback={"understanding": "", "reasoning": "", "steps": []}
                  )
        if not parsed_ok:
              self.logger.error(f"Decentralized planning JSON parse failed: {parse_error}")
        # FIX: Add reasoning step (was in original)
        self.current_metrics.add_reasoning_step("orchestrator_planning", {
            "understanding": plan.get("understanding", ""),
            "reasoning": plan.get("reasoning", ""),
            "steps_planned": len(plan.get("steps", []))
        })

        if reasoning_callback:
            await reasoning_callback({
                "phase": "planning_stage1",
                "source": "orchestrator",
                "title": "High-level plan created (decentralized)",
                "content": {
                    "understanding": plan.get("understanding", ""),
                    "reasoning": plan.get("reasoning", ""),
                    "steps": [
                        {
                            "step_id": s.get("step_id"),
                            "agent": s.get("agent_id"),
                            "task": s.get("task"),
                            "expected_outcome": s.get("expected_outcome", ""),
                            "note": "Agent will autonomously select tools"
                        }
                        for s in plan.get("steps", [])
                    ],
                    "tokens_used": response.total_tokens,
                    "mode": "decentralized"
                }
            })

        self.logger.info(f"Created decentralized plan with {len(plan['steps'])} steps")
        return plan['steps']

    async def _execute_plan(
            self,
            execution_plan: List[Dict[str, Any]],
            context: Optional[Dict[str, Any]],
            reasoning_callback: Optional[ReasoningCallback] = None
    ) -> Dict[str, Any]:
        """Execute plan with retry, timeout, and cascade failure handling"""
        results = {}
        completed = set()
        step_failure_types = {}  # Track failure type per step

        for step in execution_plan:
            step_id = step['step_id']
            agent_id = step['agent_id']
            depends_on = step.get('depends_on', [])

            # --- Check dependency cascade ---
            failed_deps = [
                d for d in depends_on
                if d in completed and results.get(d, {}).get("status") == "error"
            ]
            if failed_deps:
                results[step_id] = {
                    "status": "error",
                    "error": f"Skipped: dependencies failed: {failed_deps}",
                    "agent": agent_id,
                    "failure_type": FailureType.CASCADE_FAILURE.value,
                    "tools_used": []
                }
                step_failure_types[step_id] = FailureType.CASCADE_FAILURE
                completed.add(step_id)
                self.logger.warning(f"Step {step_id} skipped due to cascade failure")

                if reasoning_callback:
                    await reasoning_callback({
                        "phase": "agent_skipped",
                        "source": f"agent:{agent_id}",
                        "title": f"Skipped {agent_id} (dependency failed)",
                        "content": {"step_id": step_id, "failed_deps": failed_deps}
                    })
                continue

            # --- Wait for non-failed dependencies ---
            pending_deps = [d for d in depends_on if d not in completed]
            wait_time = 0
            while pending_deps and wait_time < 30:
                await asyncio.sleep(0.1)
                wait_time += 0.1
                pending_deps = [d for d in depends_on if d not in completed]

            # --- Resolve agent ---
            agent = self.agents.get(agent_id)
            if not agent:
                self.logger.error(f"Agent {agent_id} not found")
                results[step_id] = {
                    "status": "error",
                    "error": f"Agent {agent_id} not found",
                    "failure_type": FailureType.PLANNING_FAILURE.value,
                    "tools_used": []
                }
                step_failure_types[step_id] = FailureType.PLANNING_FAILURE
                completed.add(step_id)
                continue

            if reasoning_callback:
                await reasoning_callback({
                    "phase": "agent_start",
                    "source": f"agent:{agent_id}",
                    "title": f"Starting {agent.agent_card.name}",
                    "content": {
                        "step_id": step_id,
                        "task": step['task'],
                        "tools_planned": [
                            t.get("tool")
                            for t in step.get('orchestrator_guidance', {}).get('tool_instructions', [])
                        ]
                    }
                })

            # --- Build agent request ---
            agent_request = {
                "task": step['task'],
                "context": context,
                "previous_results": {sid: results[sid] for sid in depends_on if sid in results}
            }
            if self.intelligence_mode == IntelligenceMode.CENTRALIZED:
                agent_request["orchestrator_guidance"] = step.get('orchestrator_guidance', {})

            # --- Execute with retry + timeout ---
            # Resolve timeout for this agent's model
            agent_model = getattr(agent.agent_card, 'model', self.agent_model)
            step_timeout = self.TIMEOUT_BY_MODEL.get(
                agent_model, self.STEP_TIMEOUT_DEFAULT
            ) if hasattr(self, 'TIMEOUT_BY_MODEL') else self.STEP_TIMEOUT_DEFAULT

            result = None
            last_error = None

            for attempt in range(self.MAX_STEP_RETRIES + 1):
                try:
                    result = await asyncio.wait_for(
                        agent.process_request(agent_request, reasoning_callback),
                        timeout=step_timeout
                    )

                    # ── FIX: Check the agent's returned status ──────────────
                    # Previously this was just `break` — agent errors were invisible
                    # to the retry mechanism because agent.process_request() catches
                    # all exceptions and returns {"status": "error", ...}.
                    agent_status = (result.get("status", "unknown")
                                    if isinstance(result, dict) else "unknown")

                    if agent_status in ("success", "partial"):
                        break  # Good result, stop retrying
                    else:
                        # Soft failure: agent returned error without raising
                        last_error = result.get("error",
                                                f"Agent returned status: {agent_status}")
                        self.logger.warning(
                            f"Step {step_id} soft failure "
                            f"(attempt {attempt + 1}/{self.MAX_STEP_RETRIES + 1}): "
                            f"{last_error}"
                        )
                        if attempt < self.MAX_STEP_RETRIES:
                            if reasoning_callback:
                                await reasoning_callback({
                                    "phase": "agent_retry",
                                    "source": f"agent:{agent_id}",
                                    "title": f"Retrying {agent.agent_card.name} "
                                             f"(soft failure)",
                                    "content": {
                                        "attempt": attempt + 2,
                                        "max": self.MAX_STEP_RETRIES + 1,
                                        "error": last_error[:150],
                                        "failure_type": "soft"
                                    }
                                })
                            result = None  # Clear so next attempt overwrites
                        # else: final attempt, keep the error result

                except asyncio.TimeoutError:
                    last_error = (f"Timeout after {step_timeout}s "
                                  f"(attempt {attempt + 1})")
                    self.logger.warning(
                        f"Step {step_id} timeout "
                        f"(attempt {attempt + 1}/{self.MAX_STEP_RETRIES + 1})"
                    )
                    result = None

                    if reasoning_callback and attempt < self.MAX_STEP_RETRIES:
                        await reasoning_callback({
                            "phase": "agent_retry",
                            "source": f"agent:{agent_id}",
                            "title": f"Retrying {agent.agent_card.name} (timeout)",
                            "content": {
                                "attempt": attempt + 2,
                                "max": self.MAX_STEP_RETRIES + 1,
                                "failure_type": "hard"
                            }
                        })

                except Exception as e:
                    last_error = str(e)
                    self.logger.warning(
                        f"Step {step_id} exception (attempt {attempt + 1}): {e}"
                    )
                    result = None

                    if reasoning_callback and attempt < self.MAX_STEP_RETRIES:
                        await reasoning_callback({
                            "phase": "agent_retry",
                            "source": f"agent:{agent_id}",
                            "title": f"Retrying {agent.agent_card.name} (exception)",
                            "content": {
                                "attempt": attempt + 2,
                                "error": str(e)[:150],
                                "failure_type": "hard"
                            }
                        })

            # --- Handle final result ---
            if result is None:
                result = {
                    "status": "error",
                    "error": last_error or "Unknown failure after retries",
                    "agent": agent_id,
                    "failure_type": FailureType.RUNTIME_FAILURE.value,
                    "tools_used": []
                }
                step_failure_types[step_id] = FailureType.RUNTIME_FAILURE

            # Retry metadata — now actually populated
            final_ok = (isinstance(result, dict)
                        and result.get("status") in ("success", "partial"))
            result["retry_info"] = {
                "retries_used": attempt,
                "max_retries": self.MAX_STEP_RETRIES,
                "was_retried": attempt > 0,
                "final_success": final_ok,
            }
            results[step_id] = result
            completed.add(step_id)


            if reasoning_callback:
                await reasoning_callback({
                    "phase": "agent_complete",
                    "source": f"agent:{agent_id}",
                    "title": f"Completed {agent.agent_card.name}",
                    "content": {
                        "step_id": step_id,
                        "status": result.get("status", "unknown"),
                        "summary": result.get("summary", "")[:150],
                        "tools_used": [t.get("tool") for t in result.get("tools_used", [])],
                        "tokens_used": result.get("metrics", {}).get("tokens_used", 0),
                        "retries_needed": attempt if result.get("status") != "error" else self.MAX_STEP_RETRIES
                    }
                })

            self.logger.info(f"Completed step {step_id}")

        return results

    async def _synthesize_response(
            self,
            user_request: str,
            agent_results: Dict[str, Any],
            reasoning_callback: Optional[ReasoningCallback] = None
    ) -> Dict[str, Any]:
        """Synthesize final response"""
        prompt = f"""
Synthesize final response from agent results.

User Request: {user_request}
Agent Results: {json.dumps(agent_results, indent=2)}

Interpret the numerical data and provide meaningful insights.

Return JSON:
{{
    "status": "success" or "partial" or "failed",
    "summary": "2-3 sentence executive summary with KEY FINDINGS and actual numbers",
    "reasoning": "how results address request",
    "details": {{}},
    "recommendations": ["actionable recommendations based on data"],
    "next_steps": []
}}
"""

        response: LLMResponse = await self.llm_client.chat(
            messages=[{"role": "user", "content": prompt}],
            model=self.orchestrator_model,
            temperature=0.3,
            json_mode=True
        )

        self.current_metrics.record_orchestrator_llm_call(response, "synthesis")

        result, parsed_ok, parse_error = safe_json_parse(
          response.content,
          fallback={"status": "failed", "summary": "JSON parse failed", "reasoning": "", "details": {}, "recommendations": [], "next_steps": []}
        )
        if not parsed_ok:
            self.logger.error(f"Synthesis JSON parse failed: {parse_error}")
            # FIX: Add reasoning step (was in original)
            self.current_metrics.add_reasoning_step("orchestrator_synthesis", {
                "status": result.get("status"),
                "reasoning": result.get("reasoning", "")
            })

        if reasoning_callback:
            await reasoning_callback({
                "phase": "synthesis_complete",
                "source": "orchestrator",
                "title": "Final synthesis complete",
                "content": {
                    "status": result.get("status"),
                    "summary": result.get("summary", "")[:200],
                    "reasoning": result.get("reasoning", "")[:150],
                    "recommendations_count": len(result.get("recommendations", [])),
                    "tokens_used": response.total_tokens
                }
            })

        return result

    def _calculate_accuracy_score(self, response: Dict, agent_results: Dict) -> float:
        """Calculate accuracy based on response quality"""
        score = 0.0
        status = response.get("status", "failed")
        if status == "success":
            score += 0.4
        elif status == "partial":
            score += 0.2

        agent_scores = []
        for result in agent_results.values():
            if isinstance(result, dict):
                if result.get("status") == "success":
                    agent_scores.append(1.0)
                elif result.get("status") == "partial":
                    agent_scores.append(0.5)
                else:
                    agent_scores.append(0.0)

        if agent_scores:
            score += 0.4 * (sum(agent_scores) / len(agent_scores))

        if response.get("summary") and len(response.get("summary", "")) > 20:
            score += 0.2

        return min(score, 1.0)

    def _calculate_completeness_score(self, plan: List[Dict], results: Dict) -> float:
        """Calculate plan completion rate"""
        if not plan:
            return 0.0

        completed = 0
        for step in plan:
            step_id = step.get("step_id")
            if step_id in results:
                result = results[step_id]
                if isinstance(result, dict) and result.get("status") in ["success", "partial"]:
                    completed += 1

        return completed / len(plan)

    async def _log_response(self, user_request: str, execution_plan: List[Dict[str, Any]],
                            agent_results: Dict[str, Any], final_response: Dict[str, Any],
                            metrics: Dict[str, Any]):
        """Log response"""
        timestamp = datetime.now().isoformat()
        orchestrator_response = OrchestratorResponse(
            timestamp=timestamp, request=user_request, execution_plan=execution_plan,
            agent_results=agent_results, final_response=final_response, metrics=metrics,
            mode=self.intelligence_mode.value,
            models={"orchestrator": self.orchestrator_model, "agents": self.agent_model}
        )
        log_file = self.orchestrator_log_dir / f"response_{timestamp.replace(':', '-')}.json"
        with open(log_file, 'w') as f:
            f.write(orchestrator_response.to_json())

    def update_models(self, orchestrator_model: str = None, agent_model: str = None):
        """Update models"""
        if orchestrator_model:
            self.orchestrator_model = orchestrator_model
        if agent_model:
            self.agent_model = agent_model
            for agent in self.agents.values():
                agent.update_model(agent_model)

    def set_intelligence_mode(self, mode: IntelligenceMode):
        """Change intelligence mode"""
        self.intelligence_mode = mode
        for agent in self.agents.values():
            agent.intelligence_mode = mode

    def set_agent_generation_mode(self, mode: AgentGenerationMode):
        """Change agent generation mode"""
        self.agent_generation_mode = mode
        self.logger.info(f"Set agent generation mode to {mode.value}")

    def set_two_stage_planning(self, enabled: bool):
        """Enable or disable two-stage planning"""
        self.use_two_stage_planning = enabled
        self.logger.info(f"Set two_stage_planning to {enabled}")
