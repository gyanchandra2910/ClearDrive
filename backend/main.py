import cv2
import numpy as np
import time
import torch

# Classical DIP pipeline modules
from dehaze import get_dark_channel, get_atmospheric_light, get_transmission, recover_image
from lanes import canny_edge_detection, region_of_interest, display_lines
from visibility import generate_visibility_map, apply_hud
from evaluation import get_performance_metrics
from enhancement import enhance_low_light
from alerts import get_alerts

# Deep Learning AI modules and performance telemetry
from classifier import WeatherClassifier
from segmentation import RoadSegmentor
from logger import PerformanceLogger

def run_cleardrive_master(video_path):
    print("ClearDrive: Booting Hybrid AI-DIP Engine (Temporal Smoothing Mode)...")

    # Instantiate AI models and the CSV performance logger
    brain = WeatherClassifier()
    segmentor = RoadSegmentor()
    logger = PerformanceLogger()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Video file not found. Check the provided path.")
        return

    # Configure the output video writer at standard 720p HD resolution
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('ClearDrive_Final_Output.avi', fourcc, 20.0, (1280, 720))

    # Target processing resolution for the pipeline
    proc_w, proc_h = 640, 480

    # ── Temporal Smoothing Buffers ──────────────────────────────────────────────
    # mode_history: stores past N weather predictions to compute a majority vote
    #               and prevent rapid mode flickering between frames
    mode_history = []
    # A_history: smooths Atmospheric Light estimation across frames
    #            to eliminate sudden brightness jumps during fog processing
    A_history = []

    # Frame skip caches — heavy AI models run every N frames for performance
    frame_count = 0
    cached_weather_mode = "CLEAR"
    cached_road_mask = np.zeros((proc_h, proc_w), dtype=np.uint8)

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break  # End of video stream

        frame = cv2.resize(frame, (proc_w, proc_h))
        frame_count += 1

        # ── Stage 1: AI Weather Classification (runs every 10 frames) ──────────
        # Running inference on every frame is computationally expensive;
        # we skip 9 out of 10 frames and use the cached majority-vote result
        if frame_count % 10 == 1:
            raw_mode = brain.predict(frame)
            mode_history.append(raw_mode)

            # Maintain a rolling window of the last 10 predictions (~0.5 seconds)
            # and select the most frequent label to suppress transient misclassifications
            if len(mode_history) > 10:
                mode_history.pop(0)
            cached_weather_mode = max(set(mode_history), key=mode_history.count)

        weather_mode = cached_weather_mode

        # ── Stage 1b: Hybrid Override — Brightness Heuristic ───────────────────
        # MobileNetV2 can misclassify very dark scenes as FOGGY due to uniform low values.
        # If the global mean pixel brightness is below 45, force NIGHT mode regardless.
        avg_brightness = np.mean(frame)
        if avg_brightness < 45:
            weather_mode = "NIGHT"

        # ── Stage 1c: Apply Weather-Specific Enhancement ────────────────────────
        if weather_mode == "NIGHT":
            status_mode = "AI MODE: NIGHT (CLAHE Enhancement)"
            processed_img = enhance_low_light(frame)
            # Use a fixed-high transmission map for night scenes (no fog scattering)
            t_visual = np.ones((proc_h, proc_w)) * 0.8

        elif weather_mode == "FOGGY":
            status_mode = "AI MODE: FOGGY (Aggressive DCP)"
            img_float = frame.astype('float64')
            dark = get_dark_channel(img_float)

            # Smooth Atmospheric Light across the last 5 frames to prevent brightness pops
            current_A = get_atmospheric_light(img_float, dark)
            A_history.append(current_A)
            if len(A_history) > 5:
                A_history.pop(0)
            A_smoothed = np.mean(A_history, axis=0)

            # Apply Dark Channel Prior with aggressive omega=0.95 for severe fog removal
            t = get_transmission(img_float, A_smoothed, window_size=7, omega=0.95)
            t_smoothed = cv2.GaussianBlur(t, (5, 5), 0)
            processed_img = recover_image(img_float, t_smoothed, A_smoothed, t0=0.10)
            t_visual = t_smoothed

        else:
            # CLEAR — pass the original frame through without any modification
            status_mode = "AI MODE: CLEAR (Pass-Through)"
            processed_img = frame.copy()
            t_visual = np.ones((proc_h, proc_w))

        # ── Stage 2: AI Road Segmentation (runs every 3 frames) ────────────────
        # LR-ASPP segmentation is expensive; cache the road mask across 3 frames
        if frame_count % 3 == 1:
            cached_road_mask = segmentor.get_road_mask(processed_img)
        road_mask = cached_road_mask

        # ── Stage 3: Classical Lane Detection ──────────────────────────────────
        # Apply Canny + ROI masking + Probabilistic Hough Transform for geometric lines
        canny_img = canny_edge_detection(processed_img)
        roi_img = region_of_interest(canny_img)
        lines = cv2.HoughLinesP(
            roi_img, 2, np.pi / 180, 100,
            np.array([]), minLineLength=40, maxLineGap=5
        )

        # Blend the AI road segmentation mask (30%) with the enhanced frame (70%)
        lane_assist_output = cv2.addWeighted(processed_img, 0.7, road_mask, 0.3, 0)
        if lines is not None:
            # Overlay geometric Hough lines on top of the blended frame
            line_layer = display_lines(processed_img, lines)
            lane_assist_output = cv2.addWeighted(lane_assist_output, 1, line_layer, 1, 0)

        # ── Stage 4: HUD, Metrics & Alert Generation ───────────────────────────
        vis_map = generate_visibility_map(t_visual)
        hud_final = apply_hud(processed_img, vis_map, alpha=0.35)

        c_gain, v_score = get_performance_metrics(frame, processed_img, t_visual)
        active_alerts = get_alerts(v_score, lines)
        fps = 1.0 / (time.time() - start_time)

        # Persist per-frame telemetry to CSV for later analytics and report generation
        logger.log(status_mode, fps, c_gain, v_score)

        # ── Display Assembly — Build the 4-Panel Dashboard Grid ──────────────
        grid_w, grid_h = 540, 380
        s1 = cv2.resize(frame, (grid_w, grid_h))               # Panel 1: Raw input
        s2 = cv2.resize(processed_img, (grid_w, grid_h))       # Panel 2: AI-enhanced
        s3 = cv2.resize(hud_final, (grid_w, grid_h))           # Panel 3: Reliability HUD
        s4 = cv2.resize(lane_assist_output, (grid_w, grid_h))  # Panel 4: Road segmentation

        # Render alert text overlays on the raw input panel
        y_alert = 80
        for msg, color in active_alerts:
            cv2.putText(s1, msg, (15, y_alert), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_alert += 30

        # Label each panel for driver orientation
        cv2.putText(s1, "RAW INPUT FEED",      (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(s2, "AI-ENHANCED STREAM",  (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),     2)
        cv2.putText(s3, "RELIABILITY HUD",     (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255),   2)
        cv2.putText(s4, "AI ROAD SEGMENTATION",(15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),     2)

        # Stitch the 4 panels into a 2x2 grid layout
        dashboard = np.vstack((np.hstack((s1, s2)), np.hstack((s3, s4))))

        # Build a status header bar with live telemetry readout
        header = np.zeros((70, dashboard.shape[1], 3), dtype=np.uint8)
        header_color = (0, 0, 255) if active_alerts else (0, 255, 0)
        info_str = f"SYSTEM: {status_mode} | FPS: {int(fps)} | Visibility: {v_score}% | Contrast: +{c_gain}%"
        cv2.putText(header, info_str, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, header_color, 2)

        final_display = np.vstack((header, dashboard))

        # Write the current frame to the output video at 720p resolution
        save_frame = cv2.resize(final_display, (1280, 720))
        out.write(save_frame)

        # Display the live dashboard window (press 'q' to quit)
        cv2.namedWindow("ClearDrive: Intelligent Autonomous Dashboard", cv2.WINDOW_NORMAL)
        cv2.imshow("ClearDrive: Intelligent Autonomous Dashboard", final_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources and finalize output file
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("System shutdown complete. 'ClearDrive_Final_Output.avi' saved successfully.")

if __name__ == "__main__":
    run_cleardrive_master("foggy_dashcam.mp4")