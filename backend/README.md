# Backend API Server

Flask-based REST API server providing machine learning model endpoints for MNIST digit recognition and spam detection.

## 🏗️ Architecture

The backend consists of two main services:
- **MNISTService**: Handles digit recognition using a PyTorch CNN model
- **SpamService**: Handles spam detection using scikit-learn Naive Bayes

## 📦 Dependencies

Install using `uv` (recommended):
```bash
uv pip install -e .
```

Or using pip:
```bash
pip install -r requirements.txt
```

### Key Dependencies
- `flask>=3.1.2`: Web framework
- `flask-cors>=6.0.1`: CORS support
- `torch`: PyTorch for MNIST model
- `torchvision`: MNIST dataset handling
- `scikit-learn>=1.7.2`: Spam detection model
- `nltk>=3.9.2`: Text preprocessing
- `pandas>=2.3.3`: Data handling
- `joblib>=1.5.2`: Model serialization

## 🚀 Setup

### 1. Install NLTK Data

```bash
python -c "import nltk; nltk.download('stopwords')"
```

### 2. Train MNIST Model

```bash
cd mnist
python train.py
cd ..
```

This will:
- Download MNIST dataset
- Train the CNN for 10 epochs
- Save model to `models/mnist_model.pth`

### 3. Train Spam Detection Model

```bash
cd spam_ham
python train.py
cd ..
```

This will:
- Download the spam dataset from Kaggle
- Preprocess and train Naive Bayes classifier
- Save model and vectorizer to `models/`

### 4. Run the Server

```bash
python main.py
```

The server will start on `http://0.0.0.0:5000`

## 📡 API Endpoints

### Health Check

**GET** `/health`

Check server and model status.

**Response:**
```json
{
  "status": "healthy",
  "mnist_model_loaded": true,
  "spam_model_loaded": true
}
```

### MNIST Prediction

**POST** `/mnist/predict`

Predict digit from drawn image.

**Request Body:**
```json
{
  "image": "data:image/png;base64,iVBORw0KGgo..."
}
```

**Response:**
```json
{
  "prediction": 7,
  "confidence": 0.95,
  "probabilities": [0.01, 0.02, ..., 0.95, ...]
}
```

**Error Responses:**
- `500`: Model not loaded or processing error

### Spam Detection

**POST** `/spam/predict`

Detect if text is spam.

**Request Body:**
```json
{
  "text": "Check out this amazing offer!"
}
```

**Response:**
```json
{
  "text": "Check out this amazing offer!",
  "is_spam": true,
  "confidence": 0.87,
  "spam_probability": 0.87,
  "message": "This message is likely SPAM!"
}
```

**Error Responses:**
- `400`: Missing or empty text
- `500`: Model not loaded or processing error

### Batch Spam Detection

**POST** `/spam/batch_predict`

Detect spam for multiple texts.

**Request Body:**
```json
{
  "texts": ["Message 1", "Message 2", "Message 3"]
}
```

**Response:**
```json
{
  "results": [
    {
      "text": "Message 1",
      "is_spam": false,
      "confidence": 0.92,
      "spam_probability": 0.08
    },
    ...
  ],
  "total": 3,
  "spam_count": 1
}
```

## 🧠 Model Details

### MNIST Model (`mnist/model.py`)

**Architecture:**
- Conv2d(1, 32, 3) → ReLU
- Conv2d(32, 64, 3) → ReLU → MaxPool2d
- Dropout(0.25)
- Linear(9216, 128) → ReLU
- Dropout(0.5)
- Linear(128, 10) → LogSoftmax

**Training:**
- Optimizer: Adam (lr=0.001)
- Loss: Negative Log Likelihood
- Epochs: 10
- Batch size: 64

### Spam Detection Model (`spam_ham/model.py`)

**Preprocessing:**
- Convert to lowercase
- Remove non-alphabetic characters
- Remove stopwords (NLTK English stopwords)

**Model:**
- Vectorizer: TF-IDF (max_features=1000)
- Classifier: Multinomial Naive Bayes
- Output: Binary classification (spam/ham)

## 📁 Project Structure

```
backend/
├── main.py              # Flask application
├── mnist/
│   ├── model.py        # MNIST CNN model
│   └── train.py        # Training script
├── spam_ham/
│   ├── model.py        # SpamDetector class
│   └── train.py        # Training script
├── models/             # Trained models (created after training)
│   ├── mnist_model.pth
│   ├── spam_model.joblib
│   └── vectorizer.joblib
├── data/               # MNIST dataset (downloaded automatically)
└── dataset/            # Spam dataset (downloaded automatically)
```

## 🔧 Configuration

### Model Loading

Models are loaded on server startup. If a model fails to load:
- Server will still start
- The `/health` endpoint will show which models are loaded
- API calls to unloaded models will return errors

### CORS

CORS is enabled for all routes to allow frontend access from different origins.

### Device Support

MNIST model automatically uses GPU if available, otherwise falls back to CPU.

## 🐛 Troubleshooting

### Model Not Loading

1. Ensure models are trained:
   ```bash
   ls models/
   ```

2. Check model paths in `main.py`:
   - MNIST: `models/mnist_model.pth`
   - Spam: `models/spam_model.joblib` and `models/vectorizer.joblib`

### NLTK Data Missing

```bash
python -c "import nltk; nltk.download('stopwords')"
```

### Port Already in Use

Change the port in `main.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

## 📝 Notes

- Models must be trained before the API can make predictions
- The spam dataset is automatically downloaded from Kaggle during training
- MNIST dataset is downloaded via torchvision during training
- All models support batch processing where applicable

