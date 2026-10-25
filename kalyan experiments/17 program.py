import torch
from transformers import BertTokenizer, BertModel

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Input sentence
sentence = "Artificial Intelligence is changing the world."

# Tokenize the sentence
inputs = tokenizer(sentence, return_tensors="pt")

# Generate contextual embeddings
with torch.no_grad():
    outputs = model(**inputs)

# Get last hidden state embeddings
embeddings = outputs.last_hidden_state

# Display results
print("Original Sentence:")
print(sentence)

print("\nToken IDs:")
print(inputs["input_ids"])

print("\nEmbedding Shape:")
print(embeddings.shape)

print("\nContextual Embeddings:")
print(embeddings)