# ClearDrive-ADAS

**ClearDrive** is a Hybrid AI and Image Processing Dashboard designed for low-visibility driving conditions (like fog and night-time) using a standard dashcam. 

## 🌟 Main Features

1. **AI Weather Classification:** Real-time identification of weather conditions (foggy, night, or clear).
2. **Dynamic Image Enhancement:** 
   - *Fog Mode:* Mathematically removes haze.
   - *Night Mode:* Intelligently boosts brightness (CLAHE).
3. **Lane Detection:** Uses classical edge detection to find lane markings.
4. **Drivable Area Semantic Segmentation:** Uses a deep learning model to highlight the drivable road area when lane lines are invisible.
5. **Live Dashboard:** Displays a real-time driver HUD and streams performance metrics (FPS, Visibility Score, Contrast Gain) to the frontend.

## 🛠 Tech Stack

**Backend (Python / Image Processing / AI):**
- **Framework:** FastAPI, Uvicorn
- **Computer Vision:** OpenCV, NumPy
- **Deep Learning:** PyTorch

**Frontend (Web Dashboard):**
- **Framework:** React 19, Vite
- **Styling/Architecture:** Web-based HUD

## 🚀 Setup Instructions

### 1. Backend Setup
Navigate to the `backend` directory and install the Python dependencies:

```bash
cd backend
pip install -r requirements.txt
```

Run the FastAPI server:
```bash
uvicorn main:app --reload
```

### 2. Frontend Setup
Navigate to the `frontend` directory and install the Node dependencies:

```bash
cd frontend
npm install
```

Start the Vite development server:
```bash
npm run dev
```

*The React dashboard will be accessible via your browser, consuming the video stream and live metrics from the backend.*
