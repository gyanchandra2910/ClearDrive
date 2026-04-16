import cv2
import numpy as np

def calculate_contrast(img):
    """Calculate the Global Contrast of the frame (Standard Deviation of Luma axis)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.std()

def get_performance_metrics(original, dehazed, t_map):
    """
    Compare original and processed images to extract physical analytic performance stats.
    """
    c_orig = calculate_contrast(original)
    c_final = calculate_contrast(dehazed)
    
    # Calculate contrast gain percentage
    contrast_improvement = ((c_final - c_orig) / (c_orig + 1e-6)) * 100
    
    # Visibility Score: Percentage of structural pixels marked as Safe (t > 0.65)
    safe_pixels = np.sum(t_map >= 0.65)
    total_pixels = t_map.size
    visibility_score = (safe_pixels / total_pixels) * 100
    
    return round(contrast_improvement, 2), round(visibility_score, 2)