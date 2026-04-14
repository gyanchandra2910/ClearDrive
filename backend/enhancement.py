import cv2
import numpy as np

def enhance_low_light(img):
    """
    Advanced Night Enhancement for Dashcams:
    Combines CLAHE (Local Contrast) + Gamma Correction (Global Brightness)
    """
    # 1. Gamma Correction to lift shadows
    # Gamma > 1.0 makes image brighter
    gamma = 1.5
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    brightened = cv2.LUT(img, table)

    # 2. LAB CLAHE for local detail sharpening
    lab = cv2.cvtColor(brightened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Adaptive thresholding support
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(10,10))
    l_enhanced = clahe.apply(l)
    
    lab_final = cv2.merge((l_enhanced, a, b))
    enhanced = cv2.cvtColor(lab_final, cv2.COLOR_LAB2BGR)
    
    # 3. Optional: Subtle bilateral filter to reduce high-ISO noise
    denoised = cv2.bilateralFilter(enhanced, d=5, sigmaColor=50, sigmaSpace=50)
    
    return denoised