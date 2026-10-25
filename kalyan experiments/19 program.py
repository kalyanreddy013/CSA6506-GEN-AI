from transformers import pipeline

# Load pre-trained GPT-2 language model
generator = pipeline("text-generation", model="gpt2")

# Input prompt
prompt = "Artificial Intelligence will"

# Generate text
result = generator(
    prompt,
    max_length=50,
    num_return_sequences=1
)

# Display generated text
print("Prompt:")
print(prompt)

print("\nGenerated Text:")
print(result[0]["generated_text"])