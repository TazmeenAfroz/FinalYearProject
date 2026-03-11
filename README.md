# MetaGaze - Gaze Estimation System

A real-time gaze estimation system using deep learning with a React frontend and FastAPI backend.

## 📁 Project Structure

```
latest/
├── core/                    # Core ML components (shared)
│   ├── model.py            # GazeSymCAT model architecture
│   ├── dataset.py          # Data loading and eye extraction
│   ├── utils.py            # Utility functions
│   └── config.py           # Training configuration
│
├── weights/                 # Model weights (see weights/README.md)
│   └── best_model.pth      # Trained model (1.1GB - download separately)
│
├── backend/                 # FastAPI backend server
│   ├── main.py             # API endpoints & WebSocket
│   ├── inference.py        # Model inference pipeline
│   ├── requirements.txt    # Python dependencies
│   └── uploads/            # Temporary image uploads
│
├── frontend/               # React web interface
│   ├── src/
│   │   ├── components/     # React components
│   │   │   ├── LiveCamera.js      # Live webcam gaze tracking
│   │   │   ├── ImageUploader.js   # Static image upload
│   │   │   ├── ResultDisplay.js   # Gaze visualization
│   │   │   └── GazeStats.js       # Statistics display
│   │   ├── services/
│   │   │   └── api.js      # Backend API client
│   │   └── App.js          # Main React app
│   ├── package.json        # Node dependencies
│   └── public/
│
├── live_test.py            # Standalone live gaze estimation (no UI)
├── requirements.txt        # Python dependencies (root)
├── .gitignore             # Git ignore rules
└── README.md              # This file

```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ with pip
- Node.js 16+ with npm
- Webcam (for live testing)

### 0. Download Model Weights ⚠️

**IMPORTANT**: The trained model file (1.1GB) is not included in the repository.

You need to download `best_model.pth` and place it in the `weights/` directory:

```bash
# Create weights directory if it doesn't exist
mkdir -p weights

# Download the model file (provide your own cloud link)
# Place best_model.pth in: weights/best_model.pth
```

See [weights/README.md](weights/README.md) for details.

### 1. Install Python Dependencies

```bash
# Install all Python packages
pip install -r requirements.txt
```

### 2. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

### 3. Run the Application

**Option A: Full Stack (Backend + Frontend)**

Terminal 1 - Start Backend:
```bash
cd backend
python main.py
# Backend runs on http://localhost:8000
```

Terminal 2 - Start Frontend:
```bash
cd frontend
npm start
# Frontend runs on http://localhost:3000
```

Open your browser to `http://localhost:3000`

**Option B: Standalone Live Testing (No UI)**

```bash
python live_test.py
# Press Q to quit, S to save snapshot
```

## 🎯 Features

### 1. **Live Camera Gaze Tracking**
- Real-time gaze estimation from webcam
- WebSocket streaming for minimal latency
- Visual feedback with gaze and head pose arrows

### 2. **Image Upload Mode**
- Upload static images for gaze analysis
- Download annotated results
- Batch processing support

### 3. **Dual Display**
- Raw camera feed
- Annotated output with gaze vectors
- Real-time FPS counter

## 📦 Dependencies

### Python (Backend + Core)
- **FastAPI** - Web framework
- **PyTorch** - Deep learning
- **OpenCV** - Image processing
- **MediaPipe** - Face & eye detection
- **Pillow** - Image handling

### JavaScript (Frontend)
- **React** - UI framework
- **Axios** - HTTP client
- Minimal dependencies for fast builds

## 🔧 API Endpoints

### HTTP Endpoints
- `GET /` - Health check
- `POST /api/predict` - Single image gaze prediction
- `GET /api/models` - List available models

### WebSocket
- `WS /api/ws/live` - Real-time gaze estimation stream

## 🧹 Maintenance

**Clean cache files:**
```bash
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete
```

**Reinstall dependencies:**
```bash
# Backend
pip install -r requirements.txt --upgrade

# Frontend
cd frontend && npm install
```

## 📝 Notes

- Model weights are stored in `core/model/`
- Uploaded images are temporarily stored in `backend/uploads/`
- Live test snapshots are saved to `live_out/`
- Frontend proxies API requests to backend via port 8000

## 🐛 Troubleshooting

**Import errors:**
- Make sure you're running from the project root
- Check that `core/` module is accessible

**WebSocket connection failed:**
- Ensure backend is running on port 8000
- Check firewall settings

**Low FPS / Quality:**
- Adjust JPEG quality in `frontend/src/components/LiveCamera.js`
- Check GPU availability for faster inference

## 📄 License

Academic/Research Project
