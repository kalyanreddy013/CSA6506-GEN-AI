import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Two semantically similar sentences
sentence1 = "I love artificial intelligence."
sentence2 = "I like AI technology."

# Convert sentences into tokens
inputs1 = tokenizer(sentence1, return_tensors="pt")
inputs2 = tokenizer(sentence2, return_tensors="pt")

# Generate embeddings
with torch.no_grad():
    output1 = model(**inputs1)
    output2 = model(**inputs2)

# Take CLS token embedding as sentence representation
embedding1 = output1.last_hidden_state[:, 0, :]
embedding2 = output2.last_hidden_state[:, 0, :]

# Calculate cosine similarity
similarity = cosine_similarity(
    embedding1.numpy(),
    embedding2.numpy()
)

# Display results
print("Sentence 1:")
print(sentence1)

print("\nSentence 2:")
print(sentence2)

print("\nCosine Similarity Score:")
print(similarity[0][0])