import asyncio
from src.llm.openai_client import OpenAIClient

async def quick_test():
    client = OpenAIClient()
    ok = await client.health_check()
    print("✅ LLM ready!" if ok else "❌ LLM not responding")

    if ok:
        result = await client.generate("Say hello from the agentic AI framework.")
        print("🧠 Model output:", result)

    await client.close()

asyncio.run(quick_test())
