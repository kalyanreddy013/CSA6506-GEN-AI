from transformers import BertTokenizer, GPT2Tokenizer

# Load BERT tokenizer
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load GPT-2 tokenizer
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Input sentence
sentence = "Artificial Intelligence is transforming the world."

# BERT Tokenization
bert_tokens = bert_tokenizer.tokenize(sentence)

# GPT-2 Tokenization
gpt2_tokens = gpt2_tokenizer.tokenize(sentence)

# Display results
print("Original Sentence:")
print(sentence)

print("\nBERT Tokens:")
print(bert_tokens)

print("\nGPT-2 Tokens:")
print(gpt2_tokens)