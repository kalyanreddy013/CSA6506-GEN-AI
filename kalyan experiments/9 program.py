from transformers import pipeline

# Load GPT-2 pre-trained model
text_generator = pipeline("text-generation", model="gpt2")

# Give input prompt
prompt = "The future of Artificial Intelligence is"

# Generate text
output = text_generator(
    prompt,
    max_length=50,
    num_return_sequences=1
)

# Print generated text
print("Generated Text:")
print(output[0]["generated_text"])