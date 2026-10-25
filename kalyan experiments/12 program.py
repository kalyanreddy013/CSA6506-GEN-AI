import torch
from transformers import BertTokenizer, BertModel

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Input sentence
sentence = "Artificial Intelligence is transforming the world."

# Tokenize input text
inputs = tokenizer(sentence, return_tensors="pt")

# Generate contextual embeddings
with torch.no_grad():
    outputs = model(**inputs)

# Get embeddings from last hidden state
embeddings = outputs.last_hidden_state

# Display output
print("Input Sentence:")
print(sentence)

print("\nEmbedding Shape:")
print(embeddings.shape)

print("\nContextual Embedding:")
print(embeddings)