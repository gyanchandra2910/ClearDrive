import cv2
import numpy as np

def generate_visibility_map(t_map):
    """
    Stage 3: Convert physical transmission constraints (t) into a color-coded visual HUD.
    t_map is an abstract float matrix evaluating localized environmental density.
    """
    h, w = t_map.shape
    # Instantiate blank canvas array for HUD assembly
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    # Mathematical Logic for Zones:
    # 1. Safe Zone (Green): Clear visibility where transmission bounds t >= 0.65
    overlay[t_map >= 0.65] = [0, 255, 0]  # Standard OpenCV BGR mappings

    # 2. Caution Zone (Yellow): Partial density interference where t is bounded below 0.65
    overlay[(t_map >= 0.35) & (t_map < 0.65)] = [0, 255, 255]

    # 3. Danger Zone (Red): Extreme density, structural occlusion (t < 0.35)
    overlay[t_map < 0.35] = [0, 0, 255]

    # Apply aggressive Gaussian blur to emulate graphical color gradients across vector bounds
    overlay_smoothed = cv2.GaussianBlur(overlay, (51, 51), 0)
    
    return overlay_smoothed

def apply_hud(original_img, visibility_map, alpha=0.3):
    """
    Synthesize GUI overlays directly onto the primary optical feed seamlessly.
    """
    return cv2.addWeighted(original_img, 1 - alpha, visibility_map, alpha, 0)