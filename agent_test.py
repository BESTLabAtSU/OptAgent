"""
Test examples for DER management system
Shows how the orchestrator, agent, and tools work together
"""
import asyncio
import json
from pathlib import Path

from src.orchestrator.orchestrator import Orchestrator
from src.agents.der_manager.der_manager_agent import DERManagerAgent
from src.agents.simulator.simulation_agent import SimulationAgent
from src.core.message_bus import Message, MessageType, get_message_bus
from src.tools.tool_registry import get_tool_registry
from src.llm.ollama_client import OllamaClient
from src.llm.openai_client import OpenAIClient

async def test_der_operations():
    """Test various DER operations"""

    # Initialize LLM client first
    ollama_client = OllamaClient()
    openai_client = OpenAIClient()
    # Check Ollama health
    if not await ollama_client.health_check():
        print("⚠️  Warning: Ollama service is not available")
        return

    # Initialize components with LLM client
    # orchestrator = Orchestrator(llm_client=ollama_client)
    orchestrator = Orchestrator(llm_client=openai_client)
    der_agent = DERManagerAgent(llm_client=ollama_client)
    simulation_agent = SimulationAgent(llm_client=ollama_client)
    message_bus = get_message_bus()

    # Get tool registry and register agent's tools
    tool_registry = get_tool_registry()

    await orchestrator.initialize()
    await der_agent.initialize()
    await simulation_agent.initialize()

    # Register agent with orchestrator
    await orchestrator.agent_registry.register_agent(
        agent_id=der_agent.agent_id,
        agent_card=der_agent.agent_card
    )

    await orchestrator.agent_registry.register_agent(
        agent_id=simulation_agent.agent_id,
        agent_card=simulation_agent.agent_card
    )

    # Test cases for each of the 7 core functions
    test_cases = [
        # {
        #     "name": "1. System Query",
        #     "request": "What DER systems do we have and what are their capacities?",
        #     "expected_intent": "system_query"
        # },
        # {
        #     "name": "2. Controller Query",
        #     "request": "Show me the controller settings for the DER system",
        #     "expected_intent": "controller_query"
        # },
        # {
        #     "name": "3. Greeting",
        #     "request": "Hello, can you help me with the DER configuration?",
        #     "expected_intent": "greeting"
        # },
        # {
        #     "name": "4. System Add",
        #     "request": "Add a new DER system with 8kW PV and 15kWh battery",
        #     "expected_intent": "system_add"
        # },
        # {
        #     "name": "5. System Update",
        #     "request": "Update the battery capacity to 20kWh for der_system_1",
        #     "expected_intent": "system_update"
        # },
        # {
        #     "name": "6. Controller Add",
        #     "request": "Add a new controller for peak shaving with battery SOC limits 20-80%",
        #     "expected_intent": "controller_add"
        # },
        # {
        #     "name": "7. Controller Update",
        #     "request": "Change the controller to enable V2G and set max export to 7kW",
        #     "expected_intent": "controller_update"
        # },
        {
            "name": "8. Compare DER update",
            "request": "I want to compare the flexibility difference if I update my battery to 20kWh",
            "expected_intent": "complex"
        }
    ]

    print("=" * 60)
    print("DER MANAGEMENT SYSTEM TEST")
    print("=" * 60)

    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        print(f"Request: {test_case['request']}")
        print(f"Expected Intent: {test_case['expected_intent']}")

        # Create task request
        task_request = Message(
            type=MessageType.TASK_REQUEST,
            sender="test_client",
            recipient="orchestrator",
            payload={
                "request": test_case["request"],
                "context": {},
                "conversation_history": []
            }
        )

        # Send to orchestrator and wait for response
        try:
            response = await message_bus.request_response(
                task_request,
                timeout=60.0
            )
            if response:
                print(f"Response received:")
                print(json.dumps(response.payload, indent=2))
            else:
                print("No response received")

        except asyncio.TimeoutError:
            print("Request timed out")
        except Exception as e:
            print(f"Error: {str(e)}")

        print("-" * 60)
        await asyncio.sleep(1)

    # Cleanup
    await orchestrator.shutdown()
    await der_agent.shutdown()
    await ollama_client.close()

    print("\nTest completed!")


async def test_der_tools_with_registry():
    """Test DER tools through the tool registry"""
    from src.tools.der_query_tool import DERQueryTool
    from src.tools.der_system_tool import DERSystemTool
    from src.tools.der_controller_tool import DERControllerTool
    from src.tools.tool_registry import get_tool_registry

    print("=" * 60)
    print("TOOL REGISTRY TESTING")
    print("=" * 60)

    # Get tool registry
    registry = get_tool_registry()

    # Register tools
    query_tool = DERQueryTool()
    system_tool = DERSystemTool()
    controller_tool = DERControllerTool()

    registry.register_tool(query_tool)
    registry.register_tool(system_tool)
    registry.register_tool(controller_tool)

    print(f"\nRegistered tools: {registry.list_tools()}")

    # Test 1: Query through registry
    print("\n1. Querying all DER systems:")
    tool = registry.get_tool("der_query")
    if tool:
        result = await tool.execute({
            "query_type": "system"
        })
        print(f"Success: {result.success}")
        if result.success:
            print(f"Systems found: {result.data.get('total', 0)}")
            print(json.dumps(result.data, indent=2))
    else:
        print("Tool not found in registry")

    # Test 2: Add system through registry
    print("\n2. Adding new DER system:")
    tool = registry.get_tool("der_system")
    if tool:
        result = await tool.execute({
            "action": "add",
            "config": {
                "system_id": "der_system_test",
                "system_type": "der_systems",
                "components": {
                    "pv": {"rated_capacity_kW": 10},
                    "bat": {
                        "rated_capacity_kWh": 25,
                        "initial_soc": 0.5,
                        "charge_speed": 0.25,
                        "discharge_speed": 0.5,
                        "charge_efficiency": 0.95
                    }
                }
            }
        })
        print(f"Success: {result.success}")
        if result.success:
            print(result.data.get("message"))

    # Test 3: Update system through registry
    print("\n3. Updating PV capacity:")
    tool = registry.get_tool("der_system")
    if tool:
        result = await tool.execute({
            "action": "update",
            "system_id": "der_system_test",
            "updates": {
                "components": {
                    "pv": {"rated_capacity_kW": 12}
                }
            },
            "merge": True
        })
        print(f"Success: {result.success}")
        if result.success:
            print(result.data.get("message"))

    print("\nTool registry testing completed!")


async def test_integrated_system():
    """Test the fully integrated system like in main.py"""
    from src.core.message_bus import MessageBus, Message, MessageType, get_message_bus
    from src.orchestrator.orchestrator import Orchestrator
    from src.concierge.concierge import Concierge
    from src.agents.der_manager.der_manager_agent import DERManagerAgent
    from src.tools.der_query_tool import DERQueryTool
    from src.tools.der_system_tool import DERSystemTool
    from src.tools.der_controller_tool import DERControllerTool
    from src.tools.tool_registry import get_tool_registry
    from src.llm.ollama_client import OllamaClient

    print("=" * 60)
    print("INTEGRATED SYSTEM TEST")
    print("=" * 60)

    # Initialize LLM client
    ollama_client = OllamaClient()

    if not await ollama_client.health_check():
        print("⚠️  Ollama service required but not available")
        return

    # Initialize tools registry
    print("🔧 Registering tools...")
    registry = get_tool_registry()

    # Register DER tools
    registry.register_tool(DERQueryTool())
    registry.register_tool(DERSystemTool())
    registry.register_tool(DERControllerTool())
    print(f"  ✓ Registered tools: {registry.list_tools()}")

    # Initialize Orchestrator
    print("🎯 Starting Orchestrator...")
    orchestrator = Orchestrator(llm_client=ollama_client)
    await orchestrator.initialize()

    # Initialize Concierge
    print("🎨 Starting Concierge...")
    concierge = Concierge(llm_client=ollama_client)
    await concierge.initialize()

    # Initialize DER Manager Agent
    print("🤖 Starting DER Manager Agent...")
    der_agent = DERManagerAgent(llm_client=ollama_client)
    await der_agent.initialize()

    # Register agent with orchestrator
    await orchestrator.agent_registry.register_agent(
        agent_id=der_agent.agent_id,
        agent_card=der_agent.agent_card
    )

    # Test user requests through Concierge
    test_requests = [
        "Hello, what can you help me with?",
        "Show me all DER systems",
        "Update the PV capacity to 5kW for der_system_1",
        "What's the current battery state of charge?"
    ]

    for request in test_requests:
        print(f"\n📨 User: {request}")
        result = await concierge.handle_user_request(request)

        if isinstance(result, dict):
            if result.get("status") == "error":
                print(f"❌ {result.get('message', 'Error occurred')}")
            else:
                print(f"🤖 Assistant: {result.get('message', 'Completed')}")
                if "details" in result:
                    print(f"   Details: {result['details']}")
        else:
            print(f"🤖 Assistant: {result}")

        await asyncio.sleep(1)

    # Cleanup
    await orchestrator.shutdown()
    await der_agent.shutdown()
    await ollama_client.close()

    print("\n✅ Integrated system test completed!")


def main():
    """Main entry point"""
    print("DER Management System Test Suite")
    print("1. Test full system with message bus")
    print("2. Test tools with registry")
    print("3. Test integrated system (like main.py)")
    print("4. Run all tests")

    choice = input("\nSelect test option (1-4): ").strip()

    if choice == "1":
        asyncio.run(test_der_operations())
    elif choice == "2":
        asyncio.run(test_der_tools_with_registry())
    elif choice == "3":
        asyncio.run(test_integrated_system())
    elif choice == "4":
        asyncio.run(test_der_operations())
        print("\n" + "=" * 60 + "\n")
        asyncio.run(test_der_tools_with_registry())
        print("\n" + "=" * 60 + "\n")
        asyncio.run(test_integrated_system())
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()