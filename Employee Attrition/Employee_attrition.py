import pandas as pd
import numpy as np
import re
import joblib
import openai
from langchain.embeddings import OpenAIEmbeddings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Set OpenAI API Key (Replace 'your-api-key' with your actual key)
openai.api_key = "your-api-key"

# Load OpenAI Embeddings Model
embedding_model = OpenAIEmbeddings()

# Preprocessing Function
def preprocess_text(text):
    """Cleans and normalizes the text by converting to lowercase and removing non-alphabetic characters."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

# Load dataset
df = pd.read_csv("employee_reviews.csv")  # Ensure 'Review' and 'Attrition' columns exist

# Preprocess text
df['Cleaned_Review'] = df['Review'].apply(preprocess_text)

# Convert text to embeddings using OpenAI
X = np.array([embedding_model.embed_query(text) for text in df['Cleaned_Review']])
y = df['Attrition']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model for classification
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Save the trained model for future use
joblib.dump(model, "attrition_model.pkl")

# Function for real-time prediction using LangChain embeddings
def predict_attrition(review):
    """Predicts whether an employee is at risk of attrition based on their review."""
    review = preprocess_text(review)  # Clean the input text
    review_vectorized = np.array([embedding_model.embed_query(review)])  # Convert to embedding
    prediction = model.predict(review_vectorized)  # Make prediction
    return "Attrition Risk" if prediction[0] == 1 else "No Attrition Risk"

# Example usage: Predicting attrition risk for a sample review
sample_review = "I feel stuck in my career with no growth opportunities."
print(predict_attrition(sample_review))