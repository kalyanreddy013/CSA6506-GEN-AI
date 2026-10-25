from transformers import pipeline

# Load pre-trained question answering model
qa_pipeline = pipeline("question-answering")

# Context paragraph
context = """
Artificial Intelligence (AI) is a branch of computer science that enables
machines to learn, reason, and make decisions. Machine Learning is a part
of AI that allows computers to learn from data.
"""

# Question
question = "What is Machine Learning?"

# Perform question answering
result = qa_pipeline(
    question=question,
    context=context
)

# Display answer
print("Question:")
print(question)

print("\nAnswer:")
print(result["answer"])

print("\nConfidence Score:")
print(result["score"])