import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import nltk
from nltk.corpus import stopwords
import re
import os
import requests
import zipfile

# Download NLTK data
nltk.download('stopwords')

def download_and_extract_dataset():
    """Download the dataset zip from Kaggle and extract its contents."""
    output_dir = "dataset"
    zip_path = os.path.join(output_dir, "data.zip")
    url = "https://www.kaggle.com/api/v1/datasets/download/ozlerhakan/spam-or-not-spam-dataset"

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Download the zip file
    with requests.get(url, stream=True, allow_redirects=True) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

def preprocess_text(text):
    """Clean and preprocess text data"""
    # Handle non-string values (NaN, None, etc.)
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def train_spam_model():
    """Train a spam detection model"""
    # Download and extract dataset
    download_and_extract_dataset()

    df = pd.read_csv('dataset/spam_or_not_spam.csv')

    # Preprocess text
    df['processed_text'] = df['email'].apply(preprocess_text)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], df['label'], test_size=0.2, random_state=42
    )

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train Naive Bayes classifier
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    # Evaluate model
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)

    # Save model and vectorizer
    joblib.dump(model, 'models/spam_model.joblib')
    joblib.dump(vectorizer, 'models/vectorizer.joblib')
    print("\nModel saved successfully!")

if __name__ == "__main__":
    train_spam_model()
