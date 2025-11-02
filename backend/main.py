from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import io
import base64
from mnist.model import MNISTNet
from spam_ham.model import SpamDetector

class MNISTService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model = MNISTNet().to(self.device)
            self.model.load_state_dict(torch.load('models/mnist_model.pth', map_location=self.device))
            self.model.eval()
            self.loaded = True
        except Exception as e:
            print(f"Error loading MNIST model: {e}")
            self.loaded = False

    def preprocess_image(self, image_data):
        # Convert base64 to PIL Image
        image_data = image_data.split(',')[1]  # Remove data:image/png;base64, prefix
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('L')  # Convert to grayscale

        # Resize to 28x28 (MNIST size)
        image = image.resize((28, 28), Image.Resampling.LANCZOS)

        # Convert to numpy array and normalize
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = 1 - image_array  # Invert colors (MNIST has white digits on black background)

        # Add batch and channel dimensions
        image_tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0)

        # Normalize with MNIST mean and std
        image_tensor = (image_tensor - 0.1307) / 0.3081

        return image_tensor.to(self.device)

    def predict(self, image_data):
        if not self.loaded:
            raise Exception("MNIST model not loaded")
        image_tensor = self.preprocess_image(image_data)
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        return {
            'prediction': int(predicted.item()),
            'confidence': float(confidence.item()),
            'probabilities': probabilities.squeeze().tolist()
        }

class SpamService:
    def __init__(self):
        try:
            self.detector = SpamDetector()
            self.loaded = True
            print("Spam detection model loaded successfully!")
        except Exception as e:
            print(f"Error loading spam model: {e}")
            print("Please run train_model.py first to create the model.")
            self.loaded = False

    def predict(self, text):
        if not self.loaded:
            raise Exception("Spam model not loaded")
        return self.detector.predict(text)

    def batch_predict(self, texts):
        if not self.loaded:
            raise Exception("Spam model not loaded")
        results = []
        for text in texts:
            if text and text.strip():
                result = self.detector.predict(text)
                results.append({
                    'text': text,
                    'is_spam': result['is_spam'],
                    'confidence': result['confidence'],
                    'spam_probability': result['spam_probability']
                })
        return results

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize services
mnist_service = MNISTService()
spam_service = SpamService()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'mnist_model_loaded': mnist_service.loaded,
        'spam_model_loaded': spam_service.loaded
    })

@app.route('/mnist/predict', methods=['POST'])
def mnist_predict():
    try:
        data = request.get_json()
        image_data = data['image']
        result = mnist_service.predict(image_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/spam/predict', methods=['POST'])
def spam_predict():
    """Predict if the given text is spam or not"""
    try:
        if not spam_service.loaded:
            return jsonify({
                'error': 'Spam model not loaded. Please train the model first.'
            }), 500

        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({
                'error': 'Please provide text in the request body'
            }), 400

        text = data['text']

        if not text.strip():
            return jsonify({
                'error': 'Text cannot be empty'
            }), 400

        # Make prediction
        result = spam_service.predict(text)

        return jsonify({
            'text': text,
            'is_spam': result['is_spam'],
            'confidence': result['confidence'],
            'spam_probability': result['spam_probability'],
            'message': 'This message is likely SPAM!' if result['is_spam'] else 'This message appears to be HAM (not spam).'
        })

    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}'
        }), 500

@app.route('/spam/batch_predict', methods=['POST'])
def batch_predict():
    """Predict spam for multiple texts"""
    try:
        if not spam_service.loaded:
            return jsonify({
                'error': 'Spam model not loaded. Please train the model first.'
            }), 500

        data = request.get_json()

        if not data or 'texts' not in data:
            return jsonify({
                'error': 'Please provide texts array in the request body'
            }), 400

        texts = data['texts']

        if not isinstance(texts, list):
            return jsonify({
                'error': 'Texts must be an array'
            }), 400

        results = spam_service.batch_predict(texts)

        return jsonify({
            'results': results,
            'total': len(results),
            'spam_count': sum(1 for r in results if r['is_spam'])
        })

    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
