# ADS Experiment 9: Machine Learning Web Application

A full-stack web application showcasing two machine learning models: MNIST digit recognition and spam detection. Built with Flask (backend) and Next.js (frontend).

## 🚀 Features

### 1. MNIST Digit Recognition
- Interactive drawing canvas for digit input (0-9)
- Real-time prediction using a trained CNN model
- Confidence scores and probability distributions
- Responsive UI with visual feedback

### 2. Spam Detection
- Text-based spam classification using Naive Bayes
- Single and batch prediction support
- Confidence metrics and spam probability scores
- Clean, intuitive interface

## 📁 Project Structure

```
ADS_Exp9/
├── backend/           # Flask API server
│   ├── mnist/        # MNIST model and training
│   ├── spam_ham/     # Spam detection model and training
│   ├── models/       # Trained model files
│   └── main.py       # Flask application
├── frontend/         # Next.js web application
│   ├── app/          # Pages and routes
│   └── components/   # React components
└── README.md         # This file
```

## 🛠️ Tech Stack

### Backend
- **Flask**: REST API framework
- **PyTorch**: Deep learning for MNIST
- **scikit-learn**: Machine learning for spam detection
- **NLTK**: Natural language processing
- **NumPy/Pandas**: Data processing

### Frontend
- **Next.js 16**: React framework
- **TypeScript**: Type safety
- **Tailwind CSS**: Styling
- **shadcn/ui**: UI components

## 📋 Prerequisites

- Python 3.12+
- Node.js 18+ (or Bun)
- npm/yarn/bun

## 🚦 Getting Started

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Install dependencies (using uv recommended):
```bash
uv pip install -e .
# Or using pip
pip install -r requirements.txt
```

3. Download NLTK data:
```bash
python -c "import nltk; nltk.download('stopwords')"
```

4. Train the models:
```bash
# Train MNIST model
cd mnist
python train.py
cd ..

# Train spam detection model
cd spam_ham
python train.py
cd ..
```

5. Start the Flask server:
```bash
python main.py
```

The API will be available at `http://localhost:5000`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
# or
bun install
```

3. Start the development server:
```bash
npm run dev
# or
bun dev
```

The application will be available at `http://localhost:3000`

## 📡 API Endpoints

### Health Check
```
GET /health
```
Returns the status of the API and model loading state.

### MNIST Prediction
```
POST /mnist/predict
Body: { "image": "base64_image_data" }
```

### Spam Detection
```
POST /spam/predict
Body: { "text": "message text" }
```

### Batch Spam Detection
```
POST /spam/batch_predict
Body: { "texts": ["text1", "text2", ...] }
```

## 🎯 Usage

1. **MNIST Recognition**:
   - Navigate to `/mnist` in the web app
   - Draw a digit (0-9) on the canvas
   - Click "Predict Digit" to get predictions

2. **Spam Detection**:
   - Navigate to `/spam-detection` in the web app
   - Enter text in the textarea
   - Click "Detect Spam" to analyze

## 📝 Model Details

### MNIST Model
- Architecture: Convolutional Neural Network (CNN)
- Layers: 2 Conv2d, 2 Fully Connected, Dropout
- Accuracy: Trained on MNIST dataset
- Input: 28x28 grayscale images
- Output: Digit classification (0-9)

### Spam Detection Model
- Algorithm: Multinomial Naive Bayes
- Features: TF-IDF vectorization (1000 features)
- Preprocessing: Lowercase, stopword removal, punctuation removal
- Dataset: Spam or Not Spam dataset from Kaggle

## 🔧 Development

### Backend Development
- Model files are saved in `backend/models/`
- Training scripts are in respective module directories
- API uses CORS for cross-origin requests

### Frontend Development
- Pages are in `frontend/app/`
- Reusable components in `frontend/components/`
- UI components use shadcn/ui pattern
