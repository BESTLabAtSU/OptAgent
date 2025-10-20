# src/core/base_agent.py
"""
Base agent class for all agents in the system
"""
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid
import yaml
from pathlib import Path

from ..llm.ollama_client import OllamaClient
from .message_bus import Message, MessageType, get_message_bus


class AgentStatus(Enum):
    """Agent status states"""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"
    INITIALIZING = "initializing"


class AgentCapability(Enum):
    """Standard agent capabilities"""
    DER_MANAGEMENT = "der_management"
    OPTIMIZATION = "optimization"
    FORECASTING = "forecasting"
    MONITORING = "monitoring"
    REPORTING = "reporting"
    CONFIGURATION = "configuration"


@dataclass
class AgentCard:
    """Agent capability card"""
    name: str
    description: str
    version: str = "1.0.0"
    capabilities: List[AgentCapability] = field(default_factory=list)
    supported_tools: List[str] = field(default_factory=list)
    required_context: List[str] = field(default_factory=list)
    model_preferences: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'AgentCard':
        """Load agent card from YAML file"""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        # Convert string capabilities to enum
        if 'capabilities' in data:
            data['capabilities'] = [
                AgentCapability(cap) if isinstance(cap, str) else cap
                for cap in data['capabilities']
            ]

        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "capabilities": [cap.value for cap in self.capabilities],
            "supported_tools": self.supported_tools,
            "required_context": self.required_context,
            "model_preferences": self.model_preferences,
            "performance_metrics": self.performance_metrics
        }


@dataclass
class AgentContext:
    """Runtime context for an agent"""
    task_id: str
    user_request: str
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    shared_memory: Dict[str, Any] = field(default_factory=dict)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Abstract base class for all agents
    """

    def __init__(
            self,
            agent_id: Optional[str] = None,
            agent_card: Optional[AgentCard] = None,
            llm_client: Optional[OllamaClient] = None
    ):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.agent_card = agent_card or self._load_agent_card()
        self.llm_client = llm_client or OllamaClient()
        self.status = AgentStatus.INITIALIZING
        self.current_context: Optional[AgentContext] = None
        self.message_bus = get_message_bus()
        self._tools: Dict[str, Any] = {}
        self._running = False
        self._task_queue: asyncio.Queue = asyncio.Queue()

    @abstractmethod
    def _load_agent_card(self) -> AgentCard:
        """Load the agent's capability card"""
        pass

    @abstractmethod
    async def process_task(self, task: Dict[str, Any], context: AgentContext) -> Dict[str, Any]:
        """
        Process a task

        Args:
            task: Task details
            context: Execution context

        Returns:
            Task result
        """
        pass

    async def initialize(self) -> None:
        """Initialize the agent"""
        try:
            # Load tools
            await self._load_tools()

            # Subscribe to agent-specific channel
            await self.message_bus.subscribe(
                f"agent:{self.agent_id}",
                self._handle_message
            )

            # Subscribe to broadcast channels for this agent's capabilities
            for capability in self.agent_card.capabilities:
                await self.message_bus.subscribe(
                    f"capability:{capability.value}",
                    self._handle_message
                )

            self.status = AgentStatus.IDLE
            self._running = True

            # Start task processor
            asyncio.create_task(self._process_task_queue())

        except Exception as e:
            self.status = AgentStatus.ERROR
            raise Exception(f"Agent initialization failed: {e}")

    async def shutdown(self) -> None:
        """Shutdown the agent"""
        self._running = False
        self.status = AgentStatus.OFFLINE

        # Unsubscribe from channels
        # ... (implementation)

    async def _load_tools(self) -> None:
        """Load tools for this agent"""
        from ..tools.tool_registry import get_tool_registry

        registry = get_tool_registry()
        for tool_name in self.agent_card.supported_tools:
            tool = registry.get_tool(tool_name)
            if tool:
                self._tools[tool_name] = tool

    async def _handle_message(self, message: Message) -> None:
        """Handle incoming messages"""
        try:
            if message.type == MessageType.AGENT_REQUEST:
                # Add task to queue
                await self._task_queue.put(message)

            elif message.type == MessageType.A2A_MESSAGE:
                # Handle agent-to-agent communication
                await self._handle_a2a_message(message)

        except Exception as e:
            await self._send_error(message, str(e))

    async def _process_task_queue(self) -> None:
        """Process tasks from the queue"""
        while self._running:
            try:
                # Get task from queue
                message = await self._task_queue.get()

                if self.status == AgentStatus.IDLE:
                    self.status = AgentStatus.BUSY

                    # Create context
                    context = AgentContext(
                        task_id=message.id,
                        user_request=message.payload.get("request", ""),
                        metadata=message.metadata
                    )
                    self.current_context = context

                    # Process the task
                    result = await self.process_task(message.payload, context)

                    # Send response
                    response = Message(
                        type=MessageType.AGENT_RESPONSE,
                        sender=self.agent_id,
                        recipient=message.sender,
                        payload=result,
                        correlation_id=message.id
                    )
                    await self.message_bus.publish(response)

                    self.status = AgentStatus.IDLE

            except Exception as e:
                self.status = AgentStatus.ERROR
                print(f"Error processing task: {e}")
                await asyncio.sleep(1)  # Prevent tight loop on error
                self.status = AgentStatus.IDLE

    async def _handle_a2a_message(self, message: Message) -> None:
        """Handle agent-to-agent messages"""
        # Override in specific agents for A2A communication
        pass

    async def _send_error(self, original_message: Message, error: str) -> None:
        """Send error response"""
        response = Message(
            type=MessageType.ERROR,
            sender=self.agent_id,
            recipient=original_message.sender,
            payload={"error": error},
            correlation_id=original_message.id
        )
        await self.message_bus.publish(response)

    async def use_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """
        Use a tool

        Args:
            tool_name: Name of the tool
            parameters: Tool parameters

        Returns:
            Tool execution result
        """
        if tool_name not in self._tools:
            raise ValueError(f"Tool {tool_name} not available")

        tool = self._tools[tool_name]
        result = await tool.execute(parameters)

        # Store in context
        if self.current_context:
            self.current_context.tool_results.append({
                "tool": tool_name,
                "parameters": parameters,
                "result": result
            })

        return result

    async def think(self, prompt: str) -> str:
        """
        Use LLM to think/reason about something

        Args:
            prompt: The prompt to send to LLM

        Returns:
            LLM response
        """
        # Add context to prompt
        if self.current_context:
            context_str = f"""
            Task: {self.current_context.user_request}
            Previous tool results: {self.current_context.tool_results}
            """
            prompt = f"{context_str}\n\n{prompt}"

        response = await self.llm_client.generate(
            prompt=prompt,
            model=self.agent_card.model_preferences.get("model")
        )

        return response

    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "agent_id": self.agent_id,
            "name": self.agent_card.name,
            "status": self.status.value,
            "capabilities": [cap.value for cap in self.agent_card.capabilities],
            "current_task": self.current_context.task_id if self.current_context else None
        }