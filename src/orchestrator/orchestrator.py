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
    """Task representation"""
    id: str
    user_request: str
    intent: Optional[str] = None
    selected_agents: List[str] = field(default_factory=list)
    subtasks: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    results: List[Dict[str, Any]] = field(default_factory=list)


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
        # Subscribe to task requests
        await self.message_bus.subscribe(
            MessageType.TASK_REQUEST.value,
            self._handle_task_request
        )

        # Initialize agent registry
        await self.agent_registry.initialize()

        self._running = True
        print("Orchestrator initialized successfully")

    async def shutdown(self) -> None:
        """Shutdown the orchestrator"""
        self._running = False
        await self.agent_registry.shutdown()

    async def _handle_task_request(self, message: Message) -> None:
        """Handle incoming task requests - FIXED VERSION"""
        try:
            payload = message.payload
            request = payload.get("request", "")
            context = payload.get("context", {}) # placeholder
            conversation_history = payload.get("conversation_history", []) # placeholder

            # Create task
            async with self._task_lock:
                task = Task(
                    id=message.id,
                    user_request=request
                )
                self.active_tasks[task.id] = task

            # Process the task (this is the long-running operation)
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
        Main task processing pipeline

        Args:
            task: Task to process

        Returns:
            Processing result
        """
        try:
            # Step 1: Analyze task intent
            task.intent = await self._analyze_intent(task.user_request)
            print("intent from orchestrator: {}".format(task.intent))

            # Step 2: Decompose into subtasks if needed
            task.subtasks = await self._decompose_task(task)

            # Step 3: Select appropriate agents
            task.selected_agents = await self._select_agents(task)

            # Step 4: Execute task(s) with selected agents
            results = await self._execute_with_agents(task)
            task.results = results

            # Step 5: Aggregate and format results
            final_result = await self._aggregate_results(task)
            print(final_result)
            print("planning end")
            return final_result

        except Exception as e:
            task.status = "failed"
            raise e

    async def _analyze_intent(self, user_request: str) -> str:
        """Analyze user intent from the request"""
        prompt = f"""
        Analyze the following user request and determine the primary intent.

        User request: {user_request}

        Common intents:
        - config_update: User wants to update DER configuration
        - flexibility_analysis: User wants to analyze flexibility or optimization
        - status_query: User wants to know current system status
        - optimization: User wants to optimize operations
        - forecast: User wants predictions or forecasts
        - greeting: User is greeting or engaging in casual conversation
        - complex: Multiple intents or complex reasoning needed

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
            "greeting": [AgentCapability.DER_MANAGEMENT]  # Default to DER manager for greetings
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
        """Aggregate results from multiple agents/subtasks - FIXED"""
        if len(task.results) == 1:
            # Single result, check for error first
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

        # Multiple results, aggregate them
        aggregated = {
            "task_id": task.id,
            "intent": task.intent,
            "request": task.user_request,
            "results": []
        }

        for result in task.results:
            if "error" in result:
                aggregated["results"].append({
                    "status": "error",
                    "error": result["error"],
                    "agent": result["agent_id"]
                })
            else:
                aggregated["results"].append({
                    "status": "success",
                    "data": result.get("result", {}),
                    "agent": result["agent_id"]
                })

        # Use LLM to create a coherent summary if multiple results
        if len(task.results) > 1:
            try:
                summary_prompt = f"""
                Summarize these results into a coherent response:

                User request: {task.user_request}
                Results: {json.dumps(aggregated['results'], indent=2)}

                Provide a clear, unified response.
                """

                summary = await self.llm_client.generate(summary_prompt)
                aggregated["summary"] = summary
            except Exception as e:
                print(f"Warning: Failed to generate summary: {e}")
                aggregated["summary"] = "Multiple results obtained. Check details for more information."

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