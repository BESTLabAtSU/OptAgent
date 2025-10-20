"""
Central message bus for inter-component communication - FIXED VERSION
"""
import asyncio
from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import json
from abc import ABC, abstractmethod


class MessageType(Enum):
    """Message types in the system"""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    AGENT_REQUEST = "agent_request"
    AGENT_RESPONSE = "agent_response"
    TOOL_EXECUTE = "tool_execute"
    TOOL_RESULT = "tool_result"
    SYSTEM_EVENT = "system_event"
    ERROR = "error"
    A2A_MESSAGE = "a2a_message"  # Agent-to-Agent communication


@dataclass
class Message:
    """Base message structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.SYSTEM_EVENT
    sender: str = ""
    recipient: Optional[str] = None  # None means broadcast
    payload: Dict[Any, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None  # For tracking related messages

    def to_json(self) -> str:
        """Serialize message to JSON"""
        return json.dumps({
            "id": self.id,
            "type": self.type.value,
            "sender": self.sender,
            "recipient": self.recipient,
            "payload": self.payload,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id
        })

    @classmethod
    def from_json(cls, json_str: str) -> 'Message':
        """Deserialize message from JSON"""
        data = json.loads(json_str)
        return cls(
            id=data["id"],
            type=MessageType(data["type"]),
            sender=data["sender"],
            recipient=data.get("recipient"),
            payload=data["payload"],
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            correlation_id=data.get("correlation_id")
        )


class MessageBusBackend(ABC):
    """Abstract base for message bus backends"""

    @abstractmethod
    async def publish(self, channel: str, message: Message) -> None:
        """Publish a message to a channel"""
        pass

    @abstractmethod
    async def subscribe(self, channel: str, callback: Callable) -> None:
        """Subscribe to a channel"""
        pass

    @abstractmethod
    async def unsubscribe(self, channel: str, callback: Callable) -> None:
        """Unsubscribe from a channel"""
        pass


class InMemoryBackend(MessageBusBackend):
    """In-memory message bus backend for single-process applications"""

    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        """Lazy initialization of lock to ensure it's created in the correct event loop"""
        if self._lock is None:
            try:
                self._lock = asyncio.Lock()
            except RuntimeError:
                # No event loop running, will be created later
                pass
        return self._lock

    async def _ensure_lock(self) -> asyncio.Lock:
        """Ensure lock is created in current event loop"""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def publish(self, channel: str, message: Message) -> None:
        """Publish message to all subscribers of a channel - FIXED"""
        lock = await self._ensure_lock()

        # Initialize subscribers list outside the lock
        subscribers = []

        async with lock:
            if channel in self.subscribers:
                # Create a copy of subscribers to avoid modification during iteration
                subscribers = self.subscribers[channel].copy()

        # Execute callbacks outside the lock to avoid deadlocks
        if subscribers:
            tasks = []
            for callback in subscribers:
                try:
                    # Create task for each callback
                    task = asyncio.create_task(callback(message))
                    tasks.append(task)
                except Exception as e:
                    print(f"Error creating task for callback: {e}")

            # Wait for all callbacks to complete
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                # Log any exceptions
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        print(f"Callback {i} raised exception: {result}")

    async def subscribe(self, channel: str, callback: Callable) -> None:
        """Subscribe to a channel"""
        lock = await self._ensure_lock()

        async with lock:
            if channel not in self.subscribers:
                self.subscribers[channel] = []
            if callback not in self.subscribers[channel]:
                self.subscribers[channel].append(callback)

    async def unsubscribe(self, channel: str, callback: Callable) -> None:
        """Unsubscribe from a channel"""
        lock = await self._ensure_lock()

        async with lock:
            if channel in self.subscribers and callback in self.subscribers[channel]:
                self.subscribers[channel].remove(callback)
                if not self.subscribers[channel]:
                    del self.subscribers[channel]


class MessageBus:
    """
    Central message bus for the multi-agent system
    Handles all inter-component communication
    """

    def __init__(self, backend: Optional[MessageBusBackend] = None):
        self.backend = backend or InMemoryBackend()
        self._message_history: List[Message] = []
        self._max_history_size = 1000
        self._handlers: Dict[str, List[Callable]] = {}

    async def publish(self, message: Message, channel: Optional[str] = None) -> None:
        """
        Publish a message to the bus

        Args:
            message: The message to publish
            channel: Optional specific channel, defaults to message type
        """
        # Store in history
        self._message_history.append(message)
        if len(self._message_history) > self._max_history_size:
            self._message_history.pop(0)

        # Determine channel
        if channel is None:
            channel = message.type.value

        # Publish to backend
        await self.backend.publish(channel, message)

        # If message has a specific recipient, also publish to recipient's channel
        if message.recipient:
            await self.backend.publish(f"agent:{message.recipient}", message)

    async def subscribe(self, channel: str, handler: Callable[[Message], None]) -> None:
        """
        Subscribe to a channel

        Args:
            channel: Channel name or pattern
            handler: Async callback function
        """
        await self.backend.subscribe(channel, handler)

    async def unsubscribe(self, channel: str, handler: Callable[[Message], None]) -> None:
        """
        Unsubscribe from a channel

        Args:
            channel: Channel name
            handler: The handler to remove
        """
        await self.backend.unsubscribe(channel, handler)

    async def request_response(
            self,
            message: Message,
            timeout: float = 30.0
    ) -> Optional[Message]:
        """
        Send a message and wait for a response

        Args:
            message: The request message
            timeout: Timeout in seconds

        Returns:
            Response message or None if timeout
        """
        response_received = asyncio.Event()
        response_message = None

        async def response_handler(msg: Message):
            nonlocal response_message
            if msg.correlation_id == message.id:
                response_message = msg
                response_received.set()

        # Subscribe to responses
        await self.subscribe(MessageType.TASK_RESPONSE.value, response_handler)
        await self.subscribe(MessageType.AGENT_RESPONSE.value, response_handler)

        try:
            # Send the request
            await self.publish(message)

            # Wait for response
            try:
                await asyncio.wait_for(response_received.wait(), timeout=timeout)
                return response_message
            except asyncio.TimeoutError:
                return None

        finally:
            # Cleanup - always unsubscribe
            try:
                await self.unsubscribe(MessageType.TASK_RESPONSE.value, response_handler)
                await self.unsubscribe(MessageType.AGENT_RESPONSE.value, response_handler)
            except Exception as e:
                print(f"Warning: Error during unsubscribe: {e}")

    def get_history(
            self,
            message_type: Optional[MessageType] = None,
            sender: Optional[str] = None,
            limit: int = 100
    ) -> List[Message]:
        """
        Get message history with optional filters

        Args:
            message_type: Filter by message type
            sender: Filter by sender
            limit: Maximum number of messages to return

        Returns:
            List of messages matching the criteria
        """
        filtered = self._message_history

        if message_type:
            filtered = [m for m in filtered if m.type == message_type]

        if sender:
            filtered = [m for m in filtered if m.sender == sender]

        return filtered[-limit:]

    async def broadcast(
            self,
            message_type: MessageType,
            payload: Dict[Any, Any],
            sender: str = "system"
    ) -> None:
        """
        Broadcast a message to all subscribers

        Args:
            message_type: Type of message
            payload: Message payload
            sender: Sender identifier
        """
        message = Message(
            type=message_type,
            sender=sender,
            recipient=None,  # Broadcast
            payload=payload
        )
        await self.publish(message)


# Global message bus instance
_message_bus: Optional[MessageBus] = None


def get_message_bus() -> MessageBus:
    """Get the global message bus instance"""
    global _message_bus
    if _message_bus is None:
        _message_bus = MessageBus()
    return _message_bus


def set_message_bus(bus: MessageBus) -> None:
    """Set the global message bus instance"""
    global _message_bus
    _message_bus = bus