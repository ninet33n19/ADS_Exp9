from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import io
import base64
from model import MNISTNet

app = Flask(__name__)
CORS(app)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNISTNet().to(device)
model.load_state_dict(torch.load('models/mnist_model.pth', map_location=device))
model.eval()

def preprocess_image(image_data):
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

    return image_tensor.to(device)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data['image']

        # Preprocess the image
        image_tensor = preprocess_image(image_data)

        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        result = {
            'prediction': int(predicted.item()),
            'confidence': float(confidence.item()),
            'probabilities': probabilities.squeeze().tolist()
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
