"""
Main application entry point for the Multi-Agent DER Framework
"""
import asyncio
import click
from pathlib import Path
import signal
import sys
from typing import Dict, Optional, List, Any

from src.core.message_bus import MessageBus, Message, MessageType, get_message_bus
from src.orchestrator.orchestrator import Orchestrator
from src.concierge.concierge import Concierge  # Import Concierge
from src.agents.der_manager.der_manager_agent import DERManagerAgent
from src.tools.der_config_tool import DERConfigTool
from src.tools.tool_registry import get_tool_registry
from src.llm.ollama_client import OllamaClient
from config.settings import get_settings


class DERMultiAgentSystem:
    """Main application class for the DER Multi-Agent System"""

    def __init__(self):
        self.settings = get_settings()
        self.message_bus = get_message_bus()
        self.orchestrator = None
        self.concierge = None  # Add Concierge
        self.agents = []
        self.ollama_client = None
        self._running = False
        self._loop = None

    async def initialize(self):
        """Initialize all system components"""
        # Store the event loop
        self._loop = asyncio.get_running_loop()

        print("🚀 Initializing DER Multi-Agent System...")

        # Initialize Ollama client
        print("📡 Connecting to Ollama...")
        self.ollama_client = OllamaClient()

        # Check Ollama health
        if not await self.ollama_client.health_check():
            print("⚠️  Warning: Ollama service is not available. Please ensure Ollama is running.")
            print(f"   Expected at: {self.settings.ollama.host}")
        else:
            print("✅ Ollama connection established")

            # List available models
            try:
                models = await self.ollama_client.list_models()
                if models:
                    print(f"📦 Available models: {[m['name'] for m in models]}")
                else:
                    print("⚠️  No models found. You may need to pull a model first.")
            except:
                pass

        # Initialize tools registry
        print("🔧 Registering tools...")
        registry = get_tool_registry()

        # Register DER Config Tool
        der_config_tool = DERConfigTool()
        registry.register_tool(der_config_tool)
        print("  ✓ DER Config Tool registered")

        # Register placeholder tools for now
        from src.tools.base_tool import BaseTool, ToolResult

        class FlexibilityAnalyzer(BaseTool):
            def __init__(self):
                super().__init__(
                    name="flexibility_analyzer",
                    description="Analyze flexibility potential",
                    parameters_schema={}
                )

            async def execute(self, parameters):
                # Simplified flexibility analysis
                component = parameters.get("component")
                baseline = parameters.get("baseline_value", 0)
                new_value = parameters.get("new_value", 0)

                flexibility_gain = (new_value - baseline) * 0.3  # Simplified calculation

                return ToolResult(
                    success=True,
                    data={
                        "flexibility_gain_kW": flexibility_gain,
                        "potential_revenue_daily": flexibility_gain * 24 * 0.15,  # $0.15/kWh
                        "grid_services": ["frequency_regulation", "peak_shaving"]
                    }
                )

        class SystemQuery(BaseTool):
            def __init__(self):
                super().__init__(
                    name="system_query",
                    description="Query system status",
                    parameters_schema={}
                )

            async def execute(self, parameters):
                # Return mock system status
                return ToolResult(
                    success=True,
                    data={
                        "pv": {"capacity_kW": 2, "current_output_kW": 1.5},
                        "battery": {"capacity_kWh": 5, "soc": 0.3, "power_kW": 0},
                        "ev": {"capacity_kWh": 5, "soc": 0.3, "connected": True}
                    }
                )

        registry.register_tool(FlexibilityAnalyzer())
        registry.register_tool(SystemQuery())
        print("  ✓ Analysis tools registered")

        # Initialize Orchestrator
        print("🎯 Starting Orchestrator...")
        self.orchestrator = Orchestrator(llm_client=self.ollama_client)
        await self.orchestrator.initialize()

        # Initialize Concierge
        print("🎨 Starting Concierge...")
        self.concierge = Concierge(llm_client=self.ollama_client)
        await self.concierge.initialize()

        # Initialize DER Manager Agent
        print("🤖 Starting DER Manager Agent...")
        der_agent = DERManagerAgent()
        await der_agent.initialize()
        self.agents.append(der_agent)

        # Register agent with orchestrator's registry
        await self.orchestrator.agent_registry.register_agent(
            agent_id=der_agent.agent_id,
            agent_card=der_agent.agent_card
        )

        self._running = True
        print("✅ System initialization complete!")
        print("-" * 50)

    async def shutdown(self):
        """Shutdown all system components"""
        print("\n🛑 Shutting down DER Multi-Agent System...")
        self._running = False

        # Shutdown agents
        for agent in self.agents:
            await agent.shutdown()

        # Shutdown orchestrator
        if self.orchestrator:
            await self.orchestrator.shutdown()

        # Close Ollama client
        if self.ollama_client:
            await self.ollama_client.close()

        print("👋 System shutdown complete")

    async def process_user_request(self, request: str) -> Dict[str, Any]:
        """
        Process a user request through the Concierge

        Args:
            request: Natural language user request

        Returns:
            Processing result
        """
        print(f"\n📨 Processing request: {request}")

        # Use Concierge to handle the request
        if self.concierge:
            result = await self.concierge.handle_user_request(request)
            return result
        else:
            # Fallback to direct orchestrator communication if Concierge not available
            task_message = Message(
                type=MessageType.TASK_REQUEST,
                sender="user",
                payload={"request": request}
            )

            response = await self.message_bus.request_response(
                task_message,
                timeout=60.0
            )

            if response:
                return response.payload
            else:
                return {"error": "Request timeout"}

    async def interactive_mode(self):
        """Run in interactive mode with Concierge"""
        print("\n🎮 Interactive Mode")
        print("Type your requests or 'quit' to exit")
        print("-" * 50)

        # Use the stored event loop
        loop = self._loop or asyncio.get_running_loop()

        # Welcome message from Concierge
        if self.concierge:
            welcome_message = """
Hello! I'm your DER System Assistant. I can help you with:
• 🔧 Updating your PV, battery, or EV configurations
• 📊 Analyzing flexibility and optimization potential
• 📈 Checking system status and performance
• 💡 Providing recommendations for your energy system

What would you like to do today?
            """
            print(welcome_message)

        while self._running:
            try:
                # Get user input
                user_input = await loop.run_in_executor(
                    None,
                    lambda: input("\n💬 You: ")
                )

                if user_input.lower() in ['quit', 'exit', 'q']:
                    break

                if user_input.strip():
                    # Process the request through Concierge
                    result = await self.process_user_request(user_input)

                    # Display result
                    print("\n🤖 Assistant:")

                    # Check the result structure from Concierge
                    if isinstance(result, dict):
                        if "status" in result:
                            # This is from the Concierge
                            if result["status"] == "error":
                                print(f"❌ {result.get('message', 'An error occurred')}")
                            else:
                                print(result.get('message', 'Task completed'))
                                print(result["details"])
                                # # Optionally show technical details if available
                                # if "details" in result and self.settings.get("show_technical_details", False):
                                #     print("\n📋 Technical Details:")
                                #     import json
                                #     print(json.dumps(result["details"], indent=2))

                        elif "error" in result:
                            # Direct error response
                            print(f"❌ Error: {result['error']}")

                        elif "response" in result:
                            # Direct response without Concierge formatting
                            print(result["response"])

                        else:
                            # Fallback display
                            print(f"Result: {result}")
                    else:
                        print(result)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                import traceback
                traceback.print_exc()


@click.command()
@click.option('--mode', type=click.Choice(['interactive', 'server']), default='interactive', help='Run mode')
@click.option('--port', default=8000, help='Server port (for server mode)')
@click.option('--debug', is_flag=True, help='Enable debug mode')
def main(mode: str, port: int, debug: bool):
    """DER Multi-Agent Framework"""

    # Set debug mode
    if debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    # Create and run the system
    system = DERMultiAgentSystem()

    async def run():
        try:
            # Initialize system
            await system.initialize()

            if mode == 'interactive':
                # Run interactive mode
                await system.interactive_mode()
            else:
                # Server mode (to be implemented)
                print(f"Server mode on port {port} - Not yet implemented")
                # Keep running
                while system._running:
                    await asyncio.sleep(1)

        except KeyboardInterrupt:
            print("\n\nReceived interrupt signal...")
        except Exception as e:
            print(f"\n❌ Fatal error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await system.shutdown()

    # Run the async main
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()