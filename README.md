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



## 📄 License

Academic/Research Project
