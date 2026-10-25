from transformers import pipeline

# Load pre-trained GPT-2 model
text_generator = pipeline("text-generation", model="gpt2")

# Input prompt
prompt = "Artificial Intelligence is"

# Generate text
result = text_generator(
    prompt,
    max_length=50,
    num_return_sequences=1
)

# Display generated text
print("Generated Text:")
print(result[0]["generated_text"])