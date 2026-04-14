import cv2
import numpy as np
import time
import torch
import tempfile
import os
import uuid
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# DIP Modules
from dehaze import get_dark_channel, get_atmospheric_light, get_transmission, recover_image
from lanes import canny_edge_detection, region_of_interest, display_lines
from visibility import generate_visibility_map, apply_hud
from evaluation import get_performance_metrics
from enhancement import enhance_low_light
from alerts import get_alerts

# AI Modules
from classifier import WeatherClassifier
from segmentation import RoadSegmentor 

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("ClearDrive FastAPI: Booting Hybrid AI-DIP Web Server...")

# Temp file registry: maps stream_id -> file path
stream_registry = {}

# Global status for metrics endpoint
system_status = {
    "fps": 0,
    "visibility": 0,
    "contrast": 0,
    "weather": "CLEAR"
}

brain = WeatherClassifier()
segmentor = RoadSegmentor()

# ─────────────────────────────────────────────────────────────────────────────
# CORE PROCESSING ENGINE — shared by all 3 modes
# ─────────────────────────────────────────────────────────────────────────────
def process_and_stream(cap, loop=True):
    """
    Receives an OpenCV VideoCapture object (file, webcam, or uploaded).
    Runs the full ClearDrive AI-DIP pipeline on each frame.
    Yields MJPEG bytes for StreamingResponse.
    """
    proc_w, proc_h = 640, 480
    mode_history = []
    A_history = []
    frame_count = 0
    cached_weather_mode = "CLEAR"
    cached_road_mask = np.zeros((proc_h, proc_w, 3), dtype=np.uint8)

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()

        if not ret:
            if loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        frame = cv2.resize(frame, (proc_w, proc_h))
        frame_count += 1

        # ── Stage 1: AI Weather Classification (Frame Skipped) ───────────────
        if frame_count % 10 == 1:
            raw_mode = brain.predict(frame)
            mode_history.append(raw_mode)
            if len(mode_history) > 10:
                mode_history.pop(0)
            cached_weather_mode = max(set(mode_history), key=mode_history.count)
        weather_mode = cached_weather_mode
        
        # ── HYBRID LOGIC: Brightness Fallback ───────────────────────────────
        # In case AI misclassifies dark night as fog
        avg_brightness = np.mean(frame)
        if avg_brightness < 45: 
            weather_mode = "NIGHT"

        if weather_mode == "NIGHT":
            status_mode = "AI MODE: NIGHT"
            processed_img = enhance_low_light(frame)
            # Night visibility score
            t_visual = np.ones((proc_h, proc_w)) * 0.7
        elif weather_mode == "FOGGY":
            status_mode = "AI MODE: FOGGY"
            img_float = frame.astype('float64')
            dark = get_dark_channel(img_float)
            current_A = get_atmospheric_light(img_float, dark)
            A_history.append(current_A)
            if len(A_history) > 5: A_history.pop(0)
            A_smoothed = np.mean(A_history, axis=0)
            
            # Aggressive Transmission logic for deep fog
            t = get_transmission(img_float, A_smoothed, window_size=7, omega=0.97)
            t_smoothed = cv2.GaussianBlur(t, (3, 3), 0)
            dehazed = recover_image(img_float, t_smoothed, A_smoothed, t0=0.05)
            
            # REINFORCEMENT: Apply CLAHE to stretch the contrast of the dehazed image
            # This is key for "white-out" visibility
            lab = cv2.cvtColor(dehazed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l_enhanced = clahe.apply(l)
            lab_enhanced = cv2.merge((l_enhanced, a, b))
            processed_img = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
            
            t_visual = t_smoothed
        else:
            status_mode = "AI MODE: CLEAR"
            processed_img = frame.copy()
            t_visual = np.ones((proc_h, proc_w))

        # ── Stage 2: Road Segmentation (Frame Skipped) ───────────────────────
        if frame_count % 3 == 1:
            cached_road_mask = segmentor.get_road_mask(processed_img)
        road_mask = cached_road_mask

        # ── Stage 3: Lane Detection ───────────────────────────────────────────
        canny_img = canny_edge_detection(processed_img)
        roi_img = region_of_interest(canny_img)
        lines = cv2.HoughLinesP(roi_img, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
        lane_assist_output = cv2.addWeighted(processed_img, 0.7, road_mask, 0.3, 0)
        if lines is not None:
            line_layer = display_lines(processed_img, lines)
            lane_assist_output = cv2.addWeighted(lane_assist_output, 1, line_layer, 1, 0)

        # ── Stage 4: HUD & Dashboard ──────────────────────────────────────────
        vis_map = generate_visibility_map(t_visual)
        hud_final = apply_hud(processed_img, vis_map, alpha=0.35)
        c_gain, v_score = get_performance_metrics(frame, processed_img, t_visual)
        active_alerts = get_alerts(v_score, lines)
        fps = 1.0 / (time.time() - start_time)

        # Update global status for metrics API
        system_status["fps"] = round(fps, 1)
        system_status["visibility"] = round(v_score, 1)
        system_status["contrast"] = round(c_gain, 1)
        system_status["weather"] = weather_mode

        grid_w, grid_h = 540, 380
        s1 = cv2.resize(frame, (grid_w, grid_h))
        s2 = cv2.resize(processed_img, (grid_w, grid_h))
        s3 = cv2.resize(hud_final, (grid_w, grid_h))
        s4 = cv2.resize(lane_assist_output, (grid_w, grid_h))

        y_alert = 80
        for msg, color in active_alerts:
            cv2.putText(s1, msg, (15, y_alert), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_alert += 30

        cv2.putText(s1, "RAW INPUT",    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(s2, "AI-ENHANCED",  (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),     2)
        cv2.putText(s3, "HUD",          (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255),   2)
        cv2.putText(s4, "AI SEGMENT",   (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),     2)

        dashboard = np.vstack((np.hstack((s1, s2)), np.hstack((s3, s4))))
        header = np.zeros((70, dashboard.shape[1], 3), dtype=np.uint8)
        header_color = (0, 0, 255) if active_alerts else (0, 255, 0)
        info_str = f"SYSTEM: {status_mode} | FPS: {fps:.1f} | Visibility: {v_score:.1f}%"
        cv2.putText(header, info_str, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, header_color, 2)

        final_display = cv2.resize(np.vstack((header, dashboard)), (1280, 720))

        ret2, buffer = cv2.imencode('.jpg', final_display, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret2: continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK — frontend pings this to check if backend is alive
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "engine": "ClearDrive ADAS v2.0"}


@app.get("/metrics")
def get_metrics():
    return system_status


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 1: Pre-recorded dashcam video (loops forever)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/video_feed")
def video_feed():
    cap = cv2.VideoCapture("foggy_dashcam.mp4")
    return StreamingResponse(
        process_and_stream(cap, loop=True),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 2: Live Webcam feed (opens camera index 0)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/webcam_feed")
def webcam_feed():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        return {"error": "No webcam found on this server."}
    return StreamingResponse(
        process_and_stream(cap, loop=False),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 3a: Upload video — saves file, returns a stream_id
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    stream_id = str(uuid.uuid4())
    stream_registry[stream_id] = tmp_path
    return {"stream_id": stream_id}


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 3b: GET stream by ID — streams the uploaded video via MJPEG
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/stream/{stream_id}")
def stream_uploaded(stream_id: str):
    tmp_path = stream_registry.get(stream_id)
    if not tmp_path or not os.path.exists(tmp_path):
        return {"error": "Stream not found or expired."}
    cap = cv2.VideoCapture(tmp_path)

    def stream_and_cleanup():
        yield from process_and_stream(cap, loop=False)
        try:
            os.remove(tmp_path)
            del stream_registry[stream_id]
        except: pass

    return StreamingResponse(
        stream_and_cleanup(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
