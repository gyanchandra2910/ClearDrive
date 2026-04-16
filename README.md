# 🚘 ClearDrive-ADAS: Hybrid AI & DIP Dashboard for Low-Visibility Driving

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![React](https://img.shields.io/badge/React-19-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)

**ClearDrive** is a real-time hybrid **Advanced Driver Assistance System (ADAS)** designed to handle **low-visibility driving conditions** such as heavy fog and dark roads using a standard RGB dashcam.

Unlike expensive LiDAR systems that lack semantic understanding of road markings, **ClearDrive** uses a cascaded pipeline of **lightweight Deep Learning models + classical Digital Image Processing (DIP)** techniques to:

- Restore visibility dynamically
- Detect lane boundaries
- Stream real-time safety metrics

The processed data is displayed on a **web-based dashboard at ~20 FPS**.

---

# 🌟 Core Features

### 🧠 AI Weather Classification
Automatically classifies incoming frames into:

- `FOGGY`
- `NIGHT`
- `CLEAR`

This allows the system to dynamically apply the correct enhancement algorithm.

---

### 🪄 Dynamic Image Enhancement

#### Fog Mode
Uses **Dark Channel Prior (DCP)** to mathematically reverse the haze model and remove fog.

#### Night Mode
Uses **CLAHE (Contrast Limited Adaptive Histogram Equalization)** on the **CIE-LAB L channel** to boost brightness without distorting colors.

---

### 🛣️ Dual-Track Lane Detection

#### Classical Vision Pipeline
- Canny Edge Detection
- Probabilistic Hough Transform

Detects strict mathematical lane boundaries.

#### AI Semantic Segmentation
Uses **MobileNetV3 LR-ASPP** segmentation to detect the **drivable road area** when lane lines are invisible.

---

### 📊 Live Telemetry Dashboard

Real-time dashboard displays:

- Processed video stream
- FPS
- Visibility Score (%)
- Contrast Gain (%)

Built with **React + Tailwind**.

---

# 🏗️ System Architecture & Mathematical Pipeline

## 1️⃣ Temporal Smoothing (Anti-Flicker Queue)

To prevent rapid switching between weather modes:

- A **10-frame FIFO buffer** is used.
- Majority voting determines the final weather state.

At **20 FPS**, this provides **0.5 seconds of temporal stability**.

Atmospheric light estimation also uses a **5-frame moving average**.

---

## 2️⃣ Fog Restoration (Dark Channel Prior)

Fog follows the **Koschmieder Haze Model**:

```
I(x) = J(x)t(x) + A(1 - t(x))
```

Where:

- `I(x)` = observed image
- `J(x)` = true scene radiance
- `t(x)` = transmission map
- `A` = atmospheric light

### Atmospheric Light (A)

Estimated using:

- Top **0.1% brightest pixels**
- Within a **15 × 15 dark channel window**

This ensures we sample fog brightness rather than white vehicles.

---

### Transmission Map

Computed using:

- **7 × 7 minimum filter**
- Aggressiveness factor:

```
ω = 0.95
```

This retains **5% aerial depth** so the scene does not look artificial.

---

## 3️⃣ Night Enhancement (CLAHE Optimization)

Pipeline:

1. Convert RGB → **CIE-LAB**
2. Apply CLAHE only on **L channel**

Configuration:

- Grid size: **8 × 8**
- Clip limit: **3.0**

Benefits:

- Prevents headlight glare
- Preserves color balance
- Improves shadow visibility

---

## 4️⃣ Ego-Vehicle Masking

To avoid false detections from the sky:

```python
road_mask[0:int(height*0.6), :] = 0
```

This restricts segmentation to the **bottom 40% of the frame**, removing sky false positives **with zero computational cost**.

---

# 🛠️ Tech Stack

## Backend (Vision + AI Engine)

- **FastAPI**
- **Uvicorn**
- **OpenCV**
- **NumPy**
- **PyTorch**
- **Torchvision**

### Models Used

- **MobileNetV2** → Weather classification
- **MobileNetV3 LR-ASPP** → Road segmentation

Both models are used via **transfer learning**.

---

## Frontend (Dashboard UI)

- **React 19**
- **Vite**
- **Tailwind CSS**
- **WebSockets / HTTP Streaming**

---

# 🚀 Setup & Installation

## Prerequisites

- Python **3.9+**
- Node.js **18+**

---

# 1️⃣ Backend Setup

```bash
# Clone repository
git clone https://github.com/YourUsername/ClearDrive-ADAS.git

cd ClearDrive-ADAS/backend

# Create virtual environment
python -m venv venv

# Activate environment

# Linux / Mac
source venv/bin/activate

# Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
uvicorn main:app --reload
```

Backend will run at:

```
http://localhost:8000
```

---

# 2️⃣ Frontend Setup

Open a new terminal:

```bash
cd ClearDrive-ADAS/frontend

npm install

npm run dev
```

Dashboard will run at:

```
http://localhost:5173
```

---

# 📈 Evaluation Metrics

ClearDrive logs frame-by-frame analytics into a CSV file.

## Contrast Gain (%)

Measures improvement in image contrast:

```
Contrast Gain = increase in standard deviation (σ) of grayscale image
```

Higher value → better lane visibility.

---

## Visibility Score (%)

Computed as:

```
Pixels where transmission map t(x) ≥ 0.65
```

Safety rule:

- If **Visibility < 60%**
- UI triggers **⚠️ CAUTION ALERT**

---

# 👨‍💻 Author

**Gyan Chandra**  
B.Tech + M.Tech Dual Degree  
Computer Science & Engineering  
IIITDM Kancheepuram

LinkedIn | Portfolio

---

**Built for safer highways, powered by mathematics and AI.**
