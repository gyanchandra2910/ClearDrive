import cv2
import numpy as np
import time
import torch

# DIP Modules
from dehaze import get_dark_channel, get_atmospheric_light, get_transmission, recover_image
from lanes import canny_edge_detection, region_of_interest, display_lines
from visibility import generate_visibility_map, apply_hud
from evaluation import get_performance_metrics
from enhancement import enhance_low_light
from alerts import get_alerts

# AI Modules & Analytics Engine
from classifier import WeatherClassifier
from segmentation import RoadSegmentor 
from logger import PerformanceLogger 

def run_cleardrive_master(video_path):
    print("🚀 ClearDrive: Booting Hybrid AI-DIP Engine (Smooth Temporal Mode)...")
    
    # Initialize GPU-accelerated Brains & Data Logger
    brain = WeatherClassifier()
    segmentor = RoadSegmentor()
    logger = PerformanceLogger()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: Video file not found!")
        return

    # Video Saver setup - Standard 720p HD resolution
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('ClearDrive_Final_Output.avi', fourcc, 20.0, (1280, 720))

    # Processing parameters
    proc_w, proc_h = 640, 480

    # ==========================================
    # 🧠 TEMPORAL MEMORY BUFFERS (The New Fix)
    # ==========================================
    mode_history = []  # AI Class fluctuation rokne ke liye
    A_history = []     # Brightness popping rokne ke liye
    
    # --- FPS OPTIMIZATION: Frame Skipping Caches ---
    frame_count = 0
    cached_weather_mode = "CLEAR"
    cached_road_mask = np.zeros((proc_h, proc_w), dtype=np.uint8)

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.resize(frame, (proc_w, proc_h))
        frame_count += 1
        
        # ==========================================
        # STAGE 1: AI WEATHER CLASSIFICATION (Frame Skipped)
        # ==========================================
        # Only run heavy AI once every 10 frames
        if frame_count % 10 == 1:
            raw_mode = brain.predict(frame)
            mode_history.append(raw_mode)
            
            # Pichle 10 frames (0.5 sec) ka Majority Vote lo taaki flicker na ho
            if len(mode_history) > 10:
                mode_history.pop(0)
            cached_weather_mode = max(set(mode_history), key=mode_history.count)
            
        weather_mode = cached_weather_mode
        
        if weather_mode == "NIGHT":
            status_mode = "AI MODE: NIGHT (CLAHE Enhancement)"
            processed_img = enhance_low_light(frame)
            t_visual = np.ones((proc_h, proc_w)) * 0.8 
        
        elif weather_mode == "FOGGY":
            status_mode = "AI MODE: FOGGY (Aggressive DCP)"
            img_float = frame.astype('float64')
            dark = get_dark_channel(img_float)
            
            # 💡 FIX: Atmospheric Light Smoothing (No Brightness Popping)
            current_A = get_atmospheric_light(img_float, dark)
            A_history.append(current_A)
            if len(A_history) > 5:  # Pichle 5 frames ka memory
                A_history.pop(0)
            
            # Calculate mean of A over the history buffer
            A_smoothed = np.mean(A_history, axis=0)
            
            # Aggressive Fog Removal Parameters
            t = get_transmission(img_float, A_smoothed, window_size=7, omega=0.95)
            t_smoothed = cv2.GaussianBlur(t, (5, 5), 0)
            processed_img = recover_image(img_float, t_smoothed, A_smoothed, t0=0.10)
            t_visual = t_smoothed
            
        else:
            status_mode = "AI MODE: CLEAR (Pass-Through)"
            processed_img = frame.copy()
            t_visual = np.ones((proc_h, proc_w))

        # ==========================================
        # STAGE 2: AI ROAD SEGMENTATION (Frame Skipped)
        # ==========================================
        # Only run heavy segmentation every 3 frames
        if frame_count % 3 == 1:
            cached_road_mask = segmentor.get_road_mask(processed_img)
        road_mask = cached_road_mask

        # ==========================================
        # STAGE 3: CLASSICAL LANE TRACKING
        # ==========================================
        canny_img = canny_edge_detection(processed_img)
        roi_img = region_of_interest(canny_img)
        lines = cv2.HoughLinesP(roi_img, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
        
        lane_assist_output = cv2.addWeighted(processed_img, 0.7, road_mask, 0.3, 0)
        if lines is not None:
            line_layer = display_lines(processed_img, lines)
            lane_assist_output = cv2.addWeighted(lane_assist_output, 1, line_layer, 1, 0)

        # ==========================================
        # STAGE 4: HUD, METRICS & ALERTS
        # ==========================================
        vis_map = generate_visibility_map(t_visual)
        hud_final = apply_hud(processed_img, vis_map, alpha=0.35)
        
        c_gain, v_score = get_performance_metrics(frame, processed_img, t_visual)
        active_alerts = get_alerts(v_score, lines)
        fps = 1.0 / (time.time() - start_time)

        # Log Data to CSV
        logger.log(status_mode, fps, c_gain, v_score)

        # ==========================================
        # DISPLAY ASSEMBLY
        # ==========================================
        grid_w, grid_h = 540, 380
        s1 = cv2.resize(frame, (grid_w, grid_h))
        s2 = cv2.resize(processed_img, (grid_w, grid_h))
        s3 = cv2.resize(hud_final, (grid_w, grid_h))
        s4 = cv2.resize(lane_assist_output, (grid_w, grid_h))

        # Alert Overlay
        y_alert = 80
        for msg, color in active_alerts:
            cv2.putText(s1, msg, (15, y_alert), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_alert += 30

        cv2.putText(s1, "RAW INPUT FEED", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(s2, "AI-ENHANCED STREAM", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(s3, "RELIABILITY HUD", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(s4, "AI ROAD SEGMENTATION", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        dashboard = np.vstack((np.hstack((s1, s2)), np.hstack((s3, s4))))
        
        header = np.zeros((70, dashboard.shape[1], 3), dtype=np.uint8)
        header_color = (0, 0, 255) if active_alerts else (0, 255, 0)
        info_str = f"SYSTEM: {status_mode} | FPS: {int(fps)} | Visibility: {v_score}% | Contrast: +{c_gain}%"
        cv2.putText(header, info_str, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, header_color, 2)

        final_display = np.vstack((header, dashboard))
        
        # Save output in standard HD
        save_frame = cv2.resize(final_display, (1280, 720))
        out.write(save_frame)

        # Live Display
        cv2.namedWindow("ClearDrive: Intelligent Autonomous Dashboard", cv2.WINDOW_NORMAL)
        cv2.imshow("ClearDrive: Intelligent Autonomous Dashboard", final_display)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("✅ System Shutdown. 'ClearDrive_Final_Output.avi' saved successfully in HD.")

if __name__ == "__main__":
    run_cleardrive_master("foggy_dashcam.mp4")