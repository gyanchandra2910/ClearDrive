import cv2
import numpy as np

def calculate_contrast(img):
    """Image ka Global Contrast nikalna (Standard Deviation of Luma)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.std()

def get_performance_metrics(original, dehazed, t_map):
    """
    Original aur Dehazed image ko compare karke stats nikalna.
    """
    c_orig = calculate_contrast(original)
    c_final = calculate_contrast(dehazed)
    
    # Contrast kitna % badha
    contrast_improvement = ((c_final - c_orig) / (c_orig + 1e-6)) * 100
    
    # Visibility Score: Kitne % pixels Safe (t > 0.65) hain
    safe_pixels = np.sum(t_map >= 0.65)
    total_pixels = t_map.size
    visibility_score = (safe_pixels / total_pixels) * 100
    
    return round(contrast_improvement, 2), round(visibility_score, 2)