# Frontend Web Application

Next.js 16 web application providing interactive interfaces for MNIST digit recognition and spam detection.

## 🚀 Getting Started

### Prerequisites

- Node.js 18+ (or Bun)
- Backend API server running on `http://localhost:5000`

### Installation

```bash
npm install
# or
bun install
```

### Development

Start the development server:

```bash
npm run dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## 📱 Pages

### Home Page (`/`)
- Landing page with navigation to both ML features
- Simple card-based layout

### MNIST Digit Recognition (`/mnist`)
- Interactive drawing canvas (280x280px)
- Real-time digit prediction
- Visual probability distribution for all digits (0-9)
- Confidence scores and progress bars

### Spam Detection (`/spam-detection`)
- Text input area for message analysis
- Single message spam detection
- Confidence metrics and spam probability
- Visual indicators (green for ham, red for spam)

## 🎨 Features

### Drawing Canvas (`components/DrawingCanvas.tsx`)
- Touch and mouse support
- 280x280 pixel canvas
- White background with black strokes
- Clear canvas functionality
- Automatic image data capture

### UI Components
- Built with **shadcn/ui** components
- **Tailwind CSS** for styling
- Responsive design
- Loading states and error handling
- Progress bars and badges
- Card-based layouts

## 🛠️ Tech Stack

- **Next.js 16**: React framework with App Router
- **TypeScript**: Type safety
- **Tailwind CSS 4**: Utility-first styling
- **shadcn/ui**: Accessible component library
- **Lucide React**: Icon library
- **React 19**: Latest React features

## 📁 Project Structure

```
frontend/
├── app/
│   ├── page.tsx              # Home page
│   ├── mnist/
│   │   └── page.tsx         # MNIST digit recognition
│   ├── spam-detection/
│   │   └── page.tsx         # Spam detection interface
│   ├── layout.tsx           # Root layout
│   └── globals.css          # Global styles
├── components/
│   ├── DrawingCanvas.tsx    # Canvas component
│   └── ui/                  # shadcn/ui components
│       ├── button.tsx
│       ├── card.tsx
│       ├── input.tsx
│       ├── textarea.tsx
│       ├── badge.tsx
│       ├── alert.tsx
│       └── progress.tsx
└── lib/
    └── utils.ts             # Utility functions
```

## 🔌 API Integration

### MNIST Endpoint
```typescript
fetch("http://localhost:5000/mnist/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ image: base64ImageData })
})
```

### Spam Detection Endpoint
```typescript
fetch("http://localhost:5000/spam/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ text: messageText })
})
```

## 🎯 Key Components

### DrawingCanvas
- Handles mouse and touch events
- Converts canvas to base64 PNG
- Provides clear functionality
- Callback on draw completion

### MNIST Page
- Manages prediction state
- Displays results with visual feedback
- Shows probability distribution
- Error handling and loading states

### Spam Detection Page
- Text input with validation
- Real-time spam analysis
- Color-coded results
- Confidence visualization

## 🎨 Styling

- **Tailwind CSS**: Utility classes for styling
- **Custom Colors**: Gradient backgrounds
- **Responsive**: Mobile-friendly layouts
- **Dark/Light**: Adapts to system preferences (via Tailwind)

## 🔧 Configuration

### API Endpoint

Update the API URL in the page components if your backend runs on a different port:

```typescript
const API_URL = "http://localhost:5000";
```

### Build for Production

```bash
npm run build
npm start
```

## 📝 Available Scripts

- `npm run dev`: Start development server
- `npm run build`: Build for production
- `npm start`: Start production server
- `npm run lint`: Run ESLint

## 🐛 Troubleshooting

### API Connection Issues

1. Ensure backend is running on `http://localhost:5000`
2. Check CORS settings in backend
3. Verify API endpoints match backend routes

### Canvas Not Drawing

- Check browser console for errors
- Ensure touch events are properly handled on mobile
- Verify canvas ref is initialized

### Build Errors

- Clear `.next` directory: `rm -rf .next`
- Reinstall dependencies: `rm -rf node_modules && npm install`

## 🎯 Future Enhancements

- Batch spam detection UI
- Model confidence visualization improvements
- Export/import functionality
- Dark mode toggle
- History of predictions
- Real-time drawing feedback
