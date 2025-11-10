"""
Main orchestrator for managing agent coordination and task routing - FIXED VERSION
"""
import asyncio
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import json

from ..core.message_bus import Message, MessageType, get_message_bus
from ..core.base_agent import AgentCapability, AgentStatus
from ..agents.registry import AgentRegistry
from ..llm.ollama_client import OllamaClient


@dataclass
class Task:
    """Enhanced task representation with execution plan"""
    id: str
    user_request: str
    intent: Optional[str] = None
    execution_plan: List[Dict[str, Any]] = field(default_factory=list)  # Added
    selected_agents: List[str] = field(default_factory=list)
    subtasks: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    results: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


class Orchestrator:
    """
    Central orchestrator that:
    1. Receives user requests from the concierge
    2. Analyzes and decomposes tasks
    3. Selects appropriate agents
    4. Coordinates agent execution
    5. Aggregates results
    """

    def __init__(self, llm_client: Optional[OllamaClient] = None):
        self.llm_client = llm_client or OllamaClient()
        self.message_bus = get_message_bus()
        self.agent_registry = AgentRegistry()
        self.active_tasks: Dict[str, Task] = {}
        self.task_history: List[Task] = []
        self._running = False
        self._task_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the orchestrator"""
        await self.message_bus.subscribe(
            MessageType.TASK_REQUEST.value,
            self._handle_task_request
        )
        await self.agent_registry.initialize()
        self._running = True
        print("Orchestrator initialized successfully")

    async def shutdown(self) -> None:
        """Shutdown the orchestrator"""
        self._running = False
        await self.agent_registry.shutdown()

    async def _handle_task_request(self, message: Message) -> None:
        """Handle incoming task requests"""
        try:
            payload = message.payload
            request = payload.get("request", "")
            context = payload.get("context", {})

            # Create task
            async with self._task_lock:
                task = Task(
                    id=message.id,
                    user_request=request,
                    context=context
                )
                self.active_tasks[task.id] = task

            # Process the task
            result = await self.process_task(task)

            # Send response
            response = Message(
                type=MessageType.TASK_RESPONSE,
                sender="orchestrator",
                recipient=message.sender,
                payload=result,
                correlation_id=message.id
            )
            await self.message_bus.publish(response)

            # Move to history
            async with self._task_lock:
                task.status = "completed"
                task.completed_at = datetime.now()
                self.task_history.append(task)
                if task.id in self.active_tasks:
                    del self.active_tasks[task.id]

        except Exception as e:
            print(f"Error handling task request: {e}")
            import traceback
            traceback.print_exc()

            # Send error response
            error_response = Message(
                type=MessageType.ERROR,
                sender="orchestrator",
                recipient=message.sender,
                payload={"error": str(e)},
                correlation_id=message.id
            )
            await self.message_bus.publish(error_response)

            # Clean up task
            async with self._task_lock:
                if message.id in self.active_tasks:
                    task = self.active_tasks[message.id]
                    task.status = "failed"
                    self.task_history.append(task)
                    del self.active_tasks[message.id]

    async def process_task(self, task: Task) -> Dict[str, Any]:
        """
        UPDATED: Enhanced task processing with multi-step support
        """
        try:
            # Step 1: Analyze task complexity and intent
            analysis = await self._analyze_task_complexity(task.user_request)
            task.intent = analysis.get("intent", "unknown")
            is_complex = analysis.get("is_complex", False)

            print(f"Task analysis - Intent: {task.intent}, Complex: {is_complex}")

            if is_complex:
                # Step 2: Create execution plan for complex tasks
                task.execution_plan = await self._create_execution_plan(task)
                # Step 3: Execute multi-step plan
                results = await self._execute_multi_step_plan(task)
            else:
                # Simple task - use existing logic
                task.intent = await self._analyze_intent(task.user_request)
                task.subtasks = await self._decompose_task(task)
                task.selected_agents = await self._select_agents(task)
                results = await self._execute_with_agents(task)

            task.results = results

            # Step 4: Aggregate and format results
            final_result = await self._aggregate_results(task)
            return final_result

        except Exception as e:
            task.status = "failed"
            raise e

    async def _analyze_task_complexity(self, user_request: str) -> Dict[str, Any]:
        """
        Determine if task is simple or requires multi-step execution
        """
        prompt = f"""
        Analyze this request and determine if it requires multiple steps:

        User request: {user_request}

        Determine:
        1. The intent (what the user wants to achieve)
        2. Whether this requires multiple steps
        3. Whether this task (is_complex: true/false, true when need multiple steps)
        4. If complex, what type: comparison, sequential, unclear

        Examples of complex tasks:
        - "Compare flexibility with current vs 20kWh battery" (comparison)
        - "Update battery then run simulation" (sequential)

        Examples of simple tasks:
        - "Show current DER configuration" (simple query)
        - "Update battery to 20kWh" (simple update)
        - "Add a 5kW PV" (simple add)
        - "What's the controller setpoint?" (simple query)

        Return as JSON:
        {{
            "intent": "brief description",
            "is_complex": true/false,
            "complexity_type": "comparison/sequential/optimization/none"
        }}

        Respond only with valid JSON.
        """

        try:
            response = await self.llm_client.generate(prompt, temperature=0.1)
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            return json.loads(response.strip())
        except Exception as e:
            print(f"Warning: Failed to analyze complexity: {e}")
            return {"intent": "unknown", "is_complex": False}

    async def _create_execution_plan(self, task: Task) -> List[Dict[str, Any]]:
        """
        DYNAMIC: Create execution plan using LLM to understand task requirements
        """
        # First, analyze what the task requires
        analysis_prompt = f"""
        Analyze this request and determine what steps are needed:
        Request: {task.user_request}

        Identify the pattern:
        1. Is this a comparison task? (needs baseline + change + comparison)
        2. Is this a sequential task? (needs specific order of operations)
        3. Is this an optimization task? (needs analysis + optimization)
        4. Is this a simple multi-step task? (needs multiple operations but no comparison)

        Also identify:
        - What needs to be queried/retrieved
        - What needs to be modified/updated
        - What needs to be simulated/calculated
        - What needs to be compared/analyzed

        Return as JSON:
        {{
            "pattern": "comparison/sequential/optimization/multi-step",
            "requires_baseline": true/false,
            "requires_update": true/false,
            "requires_simulation": true/false,
            "requires_comparison": true/false,
            "update_details": "what to update if applicable"
        }}

        Respond only with valid JSON.
        """

        try:
            analysis_response = await self.llm_client.generate(analysis_prompt, temperature=0.1)
            if analysis_response.startswith("```json"):
                analysis_response = analysis_response[7:]
            if analysis_response.endswith("```"):
                analysis_response = analysis_response[:-3]
            task_analysis = json.loads(analysis_response.strip())
        except Exception as e:
            print(f"Warning: Failed to analyze task pattern: {e}")
            task_analysis = {"pattern": "multi-step"}

        # Now create the execution plan based on the analysis
        plan_prompt = f"""
        Create an execution plan for this task:
        Request: {task.user_request}
        Task Pattern: {task_analysis.get('pattern', 'multi-step')}
        Task Analysis: {json.dumps(task_analysis)}

        Available agents and their capabilities:
        - der_manager: 
          * Query DER configurations (action_type: "query")
          * Update DER systems (action_type: "update") 
          * Add new DER systems (action_type: "add")
        - simulation:
          * Run simulations with given configs (action_type: "run")
        - analyzer:
          * Compare simulation results (action_type: "compare")
          * Analyze single results (action_type: "analyze")
        - optimizer:
          * Optimize configurations (action_type: "optimize")

        Based on the task analysis, create the appropriate step sequence.

        Guidelines:
        - For queries: Use der_manager with action_type "query"
        - For updates: Use der_manager with action_type "update" and specify what to update
        - For simulations: Use simulation with action_type "run"
        - For comparisons: Run baseline, make change, run again, then compare
        - Each step should have clear dependencies
        - Save important results for later steps using save_as

        Return a JSON with 'steps' array where each step has:
        {{
            "step_id": "step_1", "step_2", etc.,
            "agent": "der_manager/simulation/analyzer/optimizer",
            "action": "Human-readable description of what this step does",
            "action_type": "query/update/add/run/compare/analyze/optimize",
            "depends_on": ["step_ids this depends on"],
            "save_as": "key_name to save result",
            "parameters": {{
                // Specific parameters based on action_type
                // For query: {{"query_type": "system/controller"}}
                // For update: {{"system_id": "...", "updates": {{...}}}}
                // For run: {{"use_config": "saved_config_key"}}
                // For compare: {{"baseline": "key1", "updated": "key2"}}
            }}
        }}

        Examples based on pattern:

        If pattern is "comparison" (like "compare flexibility with 20kWh battery"):
        - Step 1: Query current config (der_manager, query)
        - Step 2: Run baseline simulation (simulation, run)
        - Step 3: Update config with changes (der_manager, update)
        - Step 4: Run updated simulation (simulation, run)
        - Step 5: Compare results (analyzer, compare)

        If pattern is "sequential" (like "update battery then run simulation"):
        - Step 1: Update configuration (der_manager, update)
        - Step 2: Run simulation (simulation, run)

        If pattern is "multi-step" (like "add PV and battery then optimize"):
        - Step 1: Add PV system (der_manager, add)
        - Step 2: Add battery system (der_manager, add)
        - Step 3: Optimize configuration (optimizer, optimize)

        Now generate the specific plan for this request.
        Respond only with valid JSON.
        """

        try:
            plan_response = await self.llm_client.generate(plan_prompt, temperature=0.1)
            if plan_response.startswith("```json"):
                plan_response = plan_response[7:]
            if plan_response.endswith("```"):
                plan_response = plan_response[:-3]
            plan = json.loads(plan_response.strip())

            steps = plan.get("steps", [])

            # Validate and enhance the plan
            enhanced_steps = []
            for step in steps:
                # Ensure all required fields exist
                enhanced_step = {
                    "step_id": step.get("step_id", f"step_{len(enhanced_steps) + 1}"),
                    "agent": step.get("agent", "der_manager"),
                    "action": step.get("action", "Execute task"),
                    "action_type": step.get("action_type", "query"),
                    "depends_on": step.get("depends_on", []),
                    "save_as": step.get("save_as", f"result_{len(enhanced_steps) + 1}"),
                    "parameters": step.get("parameters", {})
                }

                # Auto-enhance parameters based on action_type and agent
                if enhanced_step["agent"] == "der_manager":
                    if enhanced_step["action_type"] == "query" and "query_type" not in enhanced_step["parameters"]:
                        enhanced_step["parameters"]["query_type"] = "system"
                    elif enhanced_step["action_type"] == "update":
                        # Extract update details from the action description if not in parameters
                        if "updates" not in enhanced_step["parameters"]:
                            enhanced_step["parameters"] = self._extract_update_parameters(
                                enhanced_step["action"],
                                task.user_request
                            )

                elif enhanced_step["agent"] == "simulation":
                    if enhanced_step["action_type"] == "run":
                        # Determine which config to use based on dependencies
                        if not enhanced_step["parameters"].get("use_config"):
                            # If depends on an update step, use updated config
                            for dep in enhanced_step["depends_on"]:
                                if "step_3" in dep or "update" in dep:
                                    enhanced_step["parameters"]["use_config"] = "updated_config"
                                    break
                            else:
                                # Otherwise use current/baseline config
                                enhanced_step["parameters"]["use_config"] = "current_config"

                elif enhanced_step["agent"] == "analyzer":
                    if enhanced_step["action_type"] == "compare":
                        # Set baseline and updated keys based on dependencies
                        if not enhanced_step["parameters"].get("baseline"):
                            enhanced_step["parameters"]["baseline"] = "baseline_results"
                        if not enhanced_step["parameters"].get("updated"):
                            enhanced_step["parameters"]["updated"] = "updated_results"

                enhanced_steps.append(enhanced_step)

            return enhanced_steps

        except Exception as e:
            print(f"Warning: Failed to create execution plan: {e}")
            import traceback
            traceback.print_exc()

            # Emergency fallback - create a simple plan based on keywords
            return self._create_emergency_fallback_plan(task)

    async def _extract_update_parameters(self, action_description: str, user_request: str) -> Dict[str, Any]:
        """
        Use LLM to extract update parameters from action description
        """
        extraction_prompt = f"""
        Extract the DER system update parameters from this description:
        Action: {action_description}
        Original request: {user_request}

        Identify what needs to be updated and the specific values.
        Common components and their parameters:
        - Battery (bat): rated_capacity_kWh, initial_soc, max_charge_kW, max_discharge_kW
        - PV/Solar (pv): rated_capacity_kW
        - EV (ev): rated_capacity_kWh, initial_soc
        - HVAC: cooling_capacity_kW, heating_capacity_kW

        Return as JSON:
        {{
            "system_id": "der_001",  
            "updates": {{
                "components": {{
                    // Only include components that need updating
                    "bat": {{"rated_capacity_kWh": 20}},  // example for battery
                    "pv": {{"rated_capacity_kW": 5}}  // example for PV
                }}
            }}
        }}

        Important:
        - Extract numeric values from any format (20kWh, twenty kilowatt hours, 20,000 Wh, etc.)
        - Only include components explicitly mentioned for update
        - Use standard units (kW for power, kWh for energy)

        Respond only with valid JSON.
        """

        try:
            response = await self.llm_client.generate(extraction_prompt, temperature=0.1)
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            parameters = json.loads(response.strip())

            # Validate the structure
            if "updates" not in parameters:
                parameters["updates"] = {"components": {}}
            if "components" not in parameters["updates"]:
                parameters["updates"]["components"] = {}
            if "system_id" not in parameters:
                parameters["system_id"] = "der_001"

            return parameters

        except Exception as e:
            print(f"Warning: Failed to extract parameters via LLM: {e}")
            # Fallback to simple extraction
            return {
                "system_id": "der_001",
                "updates": {"components": {}}
            }

    def _create_emergency_fallback_plan(self, task: Task) -> List[Dict[str, Any]]:
        """
        Create a very simple fallback plan when LLM fails
        """
        # Detect if this looks like a comparison
        is_comparison = any(word in task.user_request.lower()
                            for word in ["compare", "difference", "versus", "vs", "better"])

        if is_comparison:
            # Basic comparison plan
            return [
                {
                    "step_id": "step_1",
                    "agent": "der_manager",
                    "action": "Query current configuration",
                    "action_type": "query",
                    "depends_on": [],
                    "save_as": "current_config",
                    "parameters": {"query_type": "system"}
                },
                {
                    "step_id": "step_2",
                    "agent": "simulation",
                    "action": "Run baseline simulation",
                    "action_type": "run",
                    "depends_on": ["step_1"],
                    "save_as": "baseline_results",
                    "parameters": {"use_config": "current_config"}
                },
                {
                    "step_id": "step_3",
                    "agent": "der_manager",
                    "action": "Update configuration",
                    "action_type": "update",
                    "depends_on": ["step_1"],
                    "save_as": "updated_config",
                    "parameters": self._extract_update_parameters("", task.user_request)
                },
                {
                    "step_id": "step_4",
                    "agent": "simulation",
                    "action": "Run updated simulation",
                    "action_type": "run",
                    "depends_on": ["step_3"],
                    "save_as": "updated_results",
                    "parameters": {"use_config": "updated_config"}
                },
                {
                    "step_id": "step_5",
                    "agent": "analyzer",
                    "action": "Compare results",
                    "action_type": "compare",
                    "depends_on": ["step_2", "step_4"],
                    "save_as": "comparison_results",
                    "parameters": {"baseline": "baseline_results", "updated": "updated_results"}
                }
            ]
        else:
            # Basic single step plan
            return [
                {
                    "step_id": "step_1",
                    "agent": "der_manager",
                    "action": task.user_request,
                    "action_type": "query",
                    "depends_on": [],
                    "save_as": "result",
                    "parameters": {"query_type": "system"}
                }
            ]

    async def _execute_multi_step_plan(self, task: Task) -> List[Dict[str, Any]]:
        """
        NEW: Execute multi-step plan with dependency management
        """
        results = []
        completed_steps = set()
        step_results = {}  # step_id -> result

        while len(completed_steps) < len(task.execution_plan):
            # Find steps ready to execute
            ready_steps = []
            for step in task.execution_plan:
                step_id = step["step_id"]
                if step_id not in completed_steps:
                    # Check dependencies
                    deps = step.get("depends_on", [])
                    if all(d in completed_steps for d in deps):
                        ready_steps.append(step)

            if not ready_steps:
                print("Warning: No steps ready to execute, possible dependency issue")
                break

            # Execute ready steps (could be parallel in future)
            for step in ready_steps:
                step_id = step["step_id"]
                agent = step["agent"]
                action = step["action"]
                actiontype = step["action_type"]

                # Build request with context from previous steps
                request_context = {
                    "action": action,
                    "action_type": actiontype,
                    "step_id": step_id,
                    "full_request": task.user_request,
                    "shared_context": task.context
                }

                # Add dependent results to context
                for dep_id in step.get("depends_on", []):
                    if dep_id in step_results:
                        request_context[f"result_from_{dep_id}"] = step_results[dep_id]

                # Send to appropriate agent
                agent_id = await self._get_agent_for_type(agent)
                if not agent_id:
                    print(f"No agent available for type: {agent}")
                    step_results[step_id] = {"error": f"No agent for {agent}"}
                    completed_steps.add(step_id)
                    continue

                try:
                    agent_request = Message(
                        type=MessageType.AGENT_REQUEST,
                        sender="orchestrator",
                        recipient=agent_id,
                        payload={
                            "task_id": task.id,
                            "request": actiontype,
                            "context": request_context
                        }
                    )

                    response = await asyncio.wait_for(
                        self.message_bus.request_response(agent_request, timeout=30.0),
                        timeout=35.0
                    )

                    if response:
                        result = response.payload
                        step_results[step_id] = result

                        # Save to shared context if specified
                        save_as = step.get("save_as")
                        if save_as:
                            task.context[save_as] = result

                        results.append({
                            "step_id": step_id,
                            "agent": agent,
                            "result": result
                        })
                    else:
                        step_results[step_id] = {"error": "Agent timeout"}

                except Exception as e:
                    step_results[step_id] = {"error": str(e)}

                completed_steps.add(step_id)

        return results

    async def _get_agent_for_type(self, agent_type: str) -> Optional[str]:
        """
        NEW: Map agent types to actual agent IDs
        """
        # Map agent types to registered agent IDs
        agent_map = {
            "der_manager": "der_manager_001",
            "simulation": "simulation_agent_001",
            "analyzer": "analyzer_agent_001",
            "optimizer": "optimizer_agent_001",
            "hvac_manager": "hvac_manager_001"
        }

        agent_id = agent_map.get(agent_type)

        # Verify agent exists in registry
        if agent_id:
            agent = await self.agent_registry.get_agent(agent_id)
            if agent:
                return agent_id

        return None

    async def _analyze_intent(self, user_request: str) -> str:
        """Analyze user intent from the request - Updated for DER operations"""
        prompt = f"""
        Analyze the following user request and determine the primary intent.

        User request: {user_request}

        Available intents for DER system:
        - system_query: User wants to know about current DER systems and their parameters
        - controller_query: User wants to know about DER controllers and their parameters
        - system_add: User wants to add a new DER system
        - system_update: User wants to update/modify an existing DER system
        - controller_add: User wants to add a new DER controller
        - controller_update: User wants to update/modify an existing DER controller
        - greeting: User is greeting or engaging in casual conversation
        - complex: Multiple intents or complex reasoning needed

        Examples:
        - "What DER systems do we have?" -> system_query
        - "Show me the battery capacity" -> system_query
        - "What controller are we using for the DER?" -> controller_query
        - "Add a new PV system with 5kW capacity" -> system_add
        - "Update the battery capacity to 20kWh" -> system_update
        - "Change the controller setpoints" -> controller_update
        - "Hello" -> greeting

        Return only the intent category name.
        """

        try:
            intent = await self.llm_client.generate(prompt, temperature=0.1)
            return intent.strip().lower()
        except Exception as e:
            print(f"Warning: Failed to analyze intent: {e}")
            return "unknown"

    async def _decompose_task(self, task: Task) -> List[Dict[str, Any]]:
        """Decompose complex tasks into subtasks"""
        # For simple tasks, no decomposition needed
        if task.intent in ["config_update", "status_query", "greeting"]:
            return [{
                "id": f"{task.id}_0",
                "type": task.intent,
                "request": task.user_request
            }]

        # For complex tasks, use LLM to decompose
        if task.intent == "complex":
            prompt = f"""
            Decompose this complex request into simpler subtasks:

            Request: {task.user_request}

            Provide a JSON list of subtasks, each with:
            - type: The subtask type
            - description: What needs to be done
            - dependencies: List of subtask IDs this depends on

            Example format:
            [{{"id": "0", "type": "query", "description": "...", "dependencies": []}}]
            """

            try:
                response = await self.llm_client.generate(prompt, temperature=0.2)
                subtasks = json.loads(response)
                return subtasks
            except Exception as e:
                print(f"Warning: Failed to decompose task: {e}")
                # Fallback to single task
                return [{
                    "id": f"{task.id}_0",
                    "type": task.intent,
                    "request": task.user_request
                }]

        return [{
            "id": f"{task.id}_0",
            "type": task.intent,
            "request": task.user_request
        }]

    async def _select_agents(self, task: Task) -> List[str]:
        """Select appropriate agents for the task"""
        selected_agents = []

        # Map intents to required capabilities
        capability_map = {
            "config_update": [AgentCapability.DER_MANAGEMENT, AgentCapability.CONFIGURATION],
            "flexibility_analysis": [AgentCapability.DER_MANAGEMENT, AgentCapability.OPTIMIZATION],
            "status_query": [AgentCapability.DER_MANAGEMENT, AgentCapability.MONITORING],
            "optimization": [AgentCapability.OPTIMIZATION],
            "forecast": [AgentCapability.FORECASTING],
            "greeting": [AgentCapability.DER_MANAGEMENT]
        }

        required_capabilities = capability_map.get(task.intent, [AgentCapability.DER_MANAGEMENT])

        # Get available agents from registry
        available_agents = await self.agent_registry.get_agents_by_capability(required_capabilities)

        # For now, select the first available agent
        # In production, this would consider load balancing, performance metrics, etc.
        if available_agents:
            selected_agents = [available_agents[0]["agent_id"]]
        else:
            # Fallback to DER Manager if available
            der_manager = await self.agent_registry.get_agent("der_manager_001")
            if der_manager:
                selected_agents = ["der_manager_001"]

        return selected_agents

    async def _execute_with_agents(self, task: Task) -> List[Dict[str, Any]]:
        """Execute task with selected agents"""
        results = []

        for subtask in task.subtasks:
            for agent_id in task.selected_agents:
                try:
                    # Send request to agent
                    agent_request = Message(
                        type=MessageType.AGENT_REQUEST,
                        sender="orchestrator",
                        recipient=agent_id,
                        payload={
                            "task_id": task.id,
                            "subtask_id": subtask.get("id"),
                            "request": subtask.get("request", task.user_request),
                            "context": {
                                "intent": task.intent,
                                "full_request": task.user_request
                            }
                        }
                    )

                    # Wait for response with timeout
                    response = await asyncio.wait_for(
                        self.message_bus.request_response(agent_request, timeout=30.0),
                        timeout=35.0
                    )

                    if response:
                        results.append({
                            "agent_id": agent_id,
                            "subtask_id": subtask.get("id"),
                            "result": response.payload
                        })
                    else:
                        results.append({
                            "agent_id": agent_id,
                            "subtask_id": subtask.get("id"),
                            "error": "Agent timeout"
                        })

                except asyncio.TimeoutError:
                    results.append({
                        "agent_id": agent_id,
                        "subtask_id": subtask.get("id"),
                        "error": "Agent request timeout"
                    })
                except Exception as e:
                    results.append({
                        "agent_id": agent_id,
                        "subtask_id": subtask.get("id"),
                        "error": f"Agent execution error: {str(e)}"
                    })

        return results

    async def _aggregate_results(self, task: Task) -> Dict[str, Any]:
        """
        UPDATED: Handle both simple and complex task results
        """
        if len(task.results) == 1:
            single_result = task.results[0]
            if "error" in single_result:
                return {
                    "error": single_result["error"],
                    "agent": single_result.get("agent_id", "unknown")
                }
            elif "result" in single_result:
                return single_result["result"]
            else:
                return {"error": "Invalid agent response format"}

        # For multi-step results, create a summary
        if task.execution_plan:
            # Find the final step (usually comparison or analysis)
            final_steps = [r for r in task.results if
                           "compare" in r.get("step_id", "") or "analyze" in r.get("step_id", "")]
            if final_steps:
                return final_steps[-1].get("result", {})

        # Default aggregation
        aggregated = {
            "task_id": task.id,
            "intent": task.intent,
            "request": task.user_request,
            "results": task.results
        }

        return aggregated

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a task"""
        task = self.active_tasks.get(task_id)
        if task:
            return {
                "task_id": task.id,
                "status": task.status,
                "intent": task.intent,
                "selected_agents": task.selected_agents,
                "created_at": task.created_at.isoformat(),
                "progress": len(task.results) / max(len(task.subtasks), 1)
            }

        # Check history
        for hist_task in self.task_history:
            if hist_task.id == task_id:
                return {
                    "task_id": hist_task.id,
                    "status": hist_task.status,
                    "intent": hist_task.intent,
                    "selected_agents": hist_task.selected_agents,
                    "created_at": hist_task.created_at.isoformat(),
                    "completed_at": hist_task.completed_at.isoformat() if hist_task.completed_at else None,
                    "results": hist_task.results
                }

        return None