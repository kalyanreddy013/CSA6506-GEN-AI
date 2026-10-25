from transformers import pipeline

# Load pre-trained LLM (GPT-2)
generator = pipeline("text-generation", model="gpt2")

# Multiple prompts
prompts = [
    "Artificial Intelligence is",
    "Machine Learning helps",
    "The future of technology is"
]

# Generate responses for each prompt
for prompt in prompts:
    result = generator(
        prompt,
        max_length=50,
        num_return_sequences=1
    )

    print("\nPrompt:")
    print(prompt)

    print("Generated Response:")
    print(result[0]["generated_text"])