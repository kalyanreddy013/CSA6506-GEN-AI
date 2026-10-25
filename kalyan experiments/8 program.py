from transformers import BertTokenizer

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Input sentence
text = "Artificial Intelligence is transforming the world."

# Tokenize the text
tokens = tokenizer.tokenize(text)

# Convert tokens into token IDs
token_ids = tokenizer.convert_tokens_to_ids(tokens)

# Display results
print("Original Text:")
print(text)

print("\nTokens:")
print(tokens)

print("\nToken IDs:")
print(token_ids)