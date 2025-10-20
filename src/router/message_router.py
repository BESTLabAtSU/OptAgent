# =====================================
# src/router/message_router.py
# =====================================
"""
Message routing for the multi-agent system
"""
from typing import Dict, Any, List, Optional, Callable
from ..core.message_bus import Message, MessageType


class MessageRouter:
    """Routes messages between components"""

    def __init__(self):
        self.routes: Dict[str, List[Callable]] = {}
        self.filters: List[Callable] = []

    def add_route(self, pattern: str, handler: Callable) -> None:
        """Add a routing rule"""
        if pattern not in self.routes:
            self.routes[pattern] = []
        self.routes[pattern].append(handler)

    def add_filter(self, filter_func: Callable) -> None:
        """Add a message filter"""
        self.filters.append(filter_func)

    async def route_message(self, message: Message) -> None:
        """Route a message to appropriate handlers"""
        # Apply filters
        for filter_func in self.filters:
            if not await filter_func(message):
                return  # Message filtered out

        # Route to handlers
        for pattern, handlers in self.routes.items():
            if self._matches_pattern(message, pattern):
                for handler in handlers:
                    await handler(message)

    def _matches_pattern(self, message: Message, pattern: str) -> bool:
        """Check if message matches pattern"""
        # Simple pattern matching
        if pattern == "*":
            return True
        if pattern == message.type.value:
            return True
        if pattern.startswith("agent:") and message.recipient:
            return pattern == f"agent:{message.recipient}"
        return False