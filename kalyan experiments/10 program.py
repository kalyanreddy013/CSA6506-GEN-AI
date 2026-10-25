from transformers import pipeline

# Load pre-trained transformer model using pipeline API
sentiment_pipeline = pipeline("sentiment-analysis")

# Input text
text = "I really enjoyed learning Artificial Intelligence. It is very interesting."

# Perform sentiment analysis
result = sentiment_pipeline(text)

# Display output
print("Input Text:")
print(text)

print("\nSentiment:")
print(result[0]["label"])

print("Confidence Score:")
print(result[0]["score"])