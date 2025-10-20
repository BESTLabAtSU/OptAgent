"""
Concierge - User-facing interface for the multi-agent system
"""
from typing import Dict, Any, Optional, List
import asyncio
from datetime import datetime

from ..core.message_bus import Message, MessageType, get_message_bus
from ..llm.ollama_client import OllamaClient


class Concierge:
    """
    The Concierge acts as the primary interface between users and the multi-agent system.
    It handles user requests, provides context, and manages conversations.
    """

    def __init__(self, llm_client: Optional[OllamaClient] = None):
        self.llm_client = llm_client or OllamaClient()
        self.message_bus = get_message_bus()
        self.conversation_history: List[Dict[str, Any]] = []
        self.user_context: Dict[str, Any] = {}

    async def initialize(self) -> None:
        """Initialize the concierge"""
        # Subscribe to relevant message types
        await self.message_bus.subscribe(
            MessageType.SYSTEM_EVENT.value,
            self._handle_system_event
        )
        print("Concierge initialized")

    async def handle_user_request(self, request: str) -> Dict[str, Any]:
        """
        Process a user request

        Args:
            request: Natural language user request

        Returns:
            Response dictionary
        """
        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": request,
            "timestamp": datetime.now().isoformat()
        })

        # Enhance request with context if needed
        enhanced_request = await self._enhance_request(request)

        # Create task request message
        task_message = Message(
            type=MessageType.TASK_REQUEST,
            sender="concierge",
            payload={
                "request": enhanced_request,
                "original_request": request,
                "context": self.user_context,
                "conversation_history": self.conversation_history[-5:]  # Last 5 messages
            }
        )

        # Send to orchestrator and wait for response
        response = await self.message_bus.request_response(
            task_message,
            timeout=60.0
        )

        if response:
            # Process and format response
            formatted_response = await self._format_response(response.payload)

            # Add to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": formatted_response.get("message", ""),
                "timestamp": datetime.now().isoformat()
            })

            return formatted_response
        else:
            return {
                "status": "error",
                "message": "Request timeout. Please try again."
            }

    async def _enhance_request(self, request: str) -> str:
        """Enhance user request with context"""
        # For now, return as-is
        # Could add context enhancement, spell correction, etc.
        return request

    async def _format_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Format response for user presentation"""
        if "error" in response:
            return {
                "status": "error",
                "message": f"I encountered an error: {response['error']}"
            }

        # Extract key information
        if "response" in response:
            message = response["response"]
        elif "result" in response and "response" in response["result"]:
            message = response["result"]["response"]
        else:
            message = "Task completed successfully."

        return {
            "status": "success",
            "message": message,
            "details": response
        }

    async def _handle_system_event(self, message: Message) -> None:
        """Handle system events"""
        # Log or process system events
        pass