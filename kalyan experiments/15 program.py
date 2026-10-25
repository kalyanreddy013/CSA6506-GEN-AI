from transformers import BertTokenizer

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Input sentence
sentence = "Natural Language Processing is a part of Artificial Intelligence."

# Generate tokens
tokens = tokenizer.tokenize(sentence)

# Convert tokens to token IDs
token_ids = tokenizer.convert_tokens_to_ids(tokens)

# Display results
print("Original Sentence:")
print(sentence)

print("\nGenerated Tokens:")
print(tokens)

print("\nToken IDs:")
print(token_ids)