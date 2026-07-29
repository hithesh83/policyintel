import asyncio
from app.core.config import settings
from app.llm.dependency import get_ai_service_instance

async def run_integration_test():
    print(f"Loading environment variables. Model: {settings.ollama_model}")
    
    # Instantiate the full stack via DI
    print("Instantiating AIService via DI (LLMClient -> OllamaClient)")
    ai_service = get_ai_service_instance()
    
    # 1. Health check
    print("\n--- Running Health Check ---")
    health = await ai_service.health()
    print(f"Status: {health.status}, Model: {health.model}, Latency: {health.latency_ms}ms")
    
    # 2. Sample Prompt (generate)
    print("\n--- Testing generate() ---")
    prompt = "What is the capital of India? Give a one-word answer."
    print(f"Prompt: {prompt}")
    response = await ai_service.generate(prompt)
    print(f"Response: {response.text.strip()}")
    print(f"Request ID: {response.request_id}, Latency: {response.latency_ms}ms, Tokens: {response.completion_tokens}")

    # 3. Clean up
    if hasattr(ai_service.client, 'aclose'):
        await ai_service.client.aclose()
        
    print("\nIntegration test complete.")

if __name__ == "__main__":
    asyncio.run(run_integration_test())
