import os
from dotenv import load_dotenv

from llm_client import (
    GeminiProvider,
    OllamaProvider,
    LLMClient
)

# Load environment variables
load_dotenv(override=True)

# Select provider
provider_name = input(
    "Choose provider (gemini/ollama): "
).strip().lower()

if provider_name == "gemini":

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("Gemini API key not found in .env")

    provider = GeminiProvider(api_key)

elif provider_name == "ollama":

    provider = OllamaProvider("llama3.2")

else:
    raise ValueError("Invalid provider selected")

# Create common client
client = LLMClient(provider)

# Get user prompt
prompt = input("Enter your question: ")

# Generate response
answer = client.generate(prompt)

print("\nLLM Response:")
print(answer)
