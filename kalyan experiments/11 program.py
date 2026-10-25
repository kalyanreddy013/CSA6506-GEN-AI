from transformers import BertTokenizer

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Input sentence
sentence = "Artificial Intelligence is changing the world."

# Tokenize the sentence
tokens = tokenizer.tokenize(sentence)

# Convert tokens into token IDs
token_ids = tokenizer.convert_tokens_to_ids(tokens)

# Display results
print("Original Sentence:")
print(sentence)

print("\nTokens:")
print(tokens)

print("\nToken IDs:")
print(token_ids)