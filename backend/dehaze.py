import cv2
import numpy as np

def get_dark_channel(img, window_size=15):
    """
    Step 1: Dark Channel Calculate karna.
    Pehle har pixel par R,G,B ka minimum nikalte hain.
    Phir cv2.erode use karke ek local window (patch) ka minimum nikalte hain.
    """
    # 1. Color channels mein minimum dhoondo
    min_channel = np.min(img, axis=2)
    
    # 2. Window (patch) mein minimum dhoondo (using morphological erosion)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel

def get_atmospheric_light(img, dark_channel, top_percent=0.001):
    """
    Step 2: Atmospheric Light (A) nikalna.
    Dark channel ke top 0.1% sabse bright pixels nikal kar
    unka original image mein average color (A) dhoondte hain.
    """
    h, w = dark_channel.shape
    num_pixels = h * w
    num_top_pixels = int(max(num_pixels * top_percent, 1))

    # Arrays ko 1D mein flatten karo taaki sort karna easy ho
    dark_flat = dark_channel.reshape(-1)
    img_flat = img.reshape(-1, 3)

    # Dark channel ke sabse bright pixels ke index nikalna
    indices = np.argsort(dark_flat)[-num_top_pixels:]

    # Un indexes par original image ka color average lena
    A = np.mean(img_flat[indices], axis=0)
    return A


def get_transmission(img, A, window_size=7, omega=0.85):
    """
    Step 3: Transmission Map (t) estimate karna.
    Formula: t(x) = 1 - omega * dark_channel(I/A)
    """
    # Image ko Atmospheric light se divide karo
    norm_img = img / A
    
    # Normalized image ka dark channel nikal kar transmission calculate karo
    transmission = 1 - omega * get_dark_channel(norm_img, window_size)
    return transmission


def recover_image(img, t, A, t0=0.25):
    """
    Step 4: Asli image (J) recover karna.
    Formula: J = (I - A) / max(t, t0) + A
    """
    # t ko t0 (0.1) se neeche mat jaane do warna divide-by-zero aayega
    t_matrix = np.maximum(t, t0)
    
    # t_matrix ko 3 channels (RGB) mein duplicate karo broadcasting ke liye
    t_matrix = cv2.merge([t_matrix, t_matrix, t_matrix])

    # Final recovery formula
    J = (img - A) / t_matrix + A
    
    # Values ko 0 se 255 ke beech limit karke wapas image format mein lao
    J = np.clip(J, 0, 255).astype(np.uint8)
    return J

def run_dehaze_pipeline(image_path):
    # 1. Image read karo
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image nahi mili! Path check karo.")
        return
        
    # Math operations ke liye float64 mein convert karna zaroori hai
    img_float = img.astype('float64') 

    # 2. Pipeline execute karo
    print("Processing Stage 1: Dehazing...")
    dark = get_dark_channel(img_float)
    A = get_atmospheric_light(img_float, dark)
    t = get_transmission(img_float, A)
    
    # (Pro-Tip: Transmission map thoda blocky hota hai. 
    # Isko smooth karne ke liye hum aage chalkar Guided Filter lagayenge, 
    # filhal ek basic Gaussian Blur usse theek kar dega)
    t_smoothed = cv2.GaussianBlur(t, (15, 15), 0)

    recovered = recover_image(img_float, t_smoothed, A)

    # 3. Output display karo
    cv2.imshow("Original Foggy Image", img)
    cv2.imshow("Recovered Clear Image", recovered)
    
    # Optional: t_map dekhna ho toh
    # cv2.imshow("Transmission Map", (t_smoothed * 255).astype(np.uint8))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Yahan apni foggy image ka naam daalna (e.g., 'fog.jpg')
if __name__ == "__main__":
    run_dehaze_pipeline("test_fog.jpg")