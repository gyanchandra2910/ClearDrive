import cv2
import numpy as np

def generate_visibility_map(t_map):
    """
    Stage 3: Transmission map (t) ko ek Red/Yellow/Green HUD mein convert karna.
    t_map: 0.0 se 1.0 ke beech ki values wala 2D array.
    """
    h, w = t_map.shape
    # Ek blank black image (canvas) banao
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    # Mathematical Logic for Zones:
    # 1. Safe Zone (Green): Jahan t >= 0.65 hai
    overlay[t_map >= 0.65] = [0, 255, 0]  # BGR format mein Green

    # 2. Caution Zone (Yellow): Jahan t 0.35 aur 0.65 ke beech hai
    overlay[(t_map >= 0.35) & (t_map < 0.65)] = [0, 255, 255]  # BGR format mein Yellow

    # 3. Danger Zone (Red): Jahan t < 0.35 hai (Heavy fog)
    overlay[t_map < 0.35] = [0, 0, 255]  # BGR format mein Red

    # Map ko soft aur natural banane ke liye heavy blur laga rahe hain
    overlay_smoothed = cv2.GaussianBlur(overlay, (51, 51), 0)
    
    return overlay_smoothed

def apply_hud(original_img, visibility_map, alpha=0.3):
    """
    HUD ko original image par transparent glass ki tarah chipkana.
    """
    return cv2.addWeighted(original_img, 1 - alpha, visibility_map, alpha, 0)