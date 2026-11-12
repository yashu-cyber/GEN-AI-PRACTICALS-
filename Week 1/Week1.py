import ollama

# Step 1: Connect to the local Llama model
response = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "user",
            "content": "what is the full form of ai"
        }
    ]
)

# Step 2: Display the generated response
print("\nLLM Response:")
print(response["message"]["content"])