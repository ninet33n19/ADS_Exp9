import joblib
import re
import nltk
from nltk.corpus import stopwords

class SpamDetector:
    def __init__(self):
        # Load the trained model and vectorizer
        self.model = joblib.load('models/spam_model.joblib')
        self.vectorizer = joblib.load('models/vectorizer.joblib')
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        """Clean and preprocess text"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = text.split()
        words = [word for word in words if word not in self.stop_words]
        return ' '.join(words)

    def predict(self, text):
        """Predict if text is spam or not"""
        processed_text = self.preprocess_text(text)
        text_tfidf = self.vectorizer.transform([processed_text])
        prediction = self.model.predict(text_tfidf)[0]
        probability = self.model.predict_proba(text_tfidf)[0]

        return {
            'is_spam': bool(prediction),
            'confidence': float(max(probability)),
            'spam_probability': float(probability[1])
        }
