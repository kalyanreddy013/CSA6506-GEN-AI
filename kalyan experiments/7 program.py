from transformers import pipeline

# Load pre-trained sentiment analysis model
sentiment_model = pipeline("sentiment-analysis")

# Input text
text = "I love this AI course. It is very interesting and useful."

# Perform sentiment analysis
result = sentiment_model(text)

# Display result
print("Text:", text)
print("Sentiment:", result[0]["label"])
print("Confidence:", result[0]["score"])