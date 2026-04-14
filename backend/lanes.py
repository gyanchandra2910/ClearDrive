import cv2
import numpy as np

def canny_edge_detection(img):
    """
    Step 1 & 2: Grayscale, Blur, aur Canny Edges
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # 50 aur 150 thresholds hain. Inko video ke hisaab se tweak kar sakte hain.
    canny = cv2.Canny(blur, 50, 150) 
    return canny

def region_of_interest(img):
    """
    Step 3: Sirf road wale hisse par focus karna (Masking)
    """
    height = img.shape[0]
    width = img.shape[1]
    
    # Ek triangular/trapezoidal area define kar rahe hain jahan road hoti hai
    polygons = np.array([
        [(50, height), (width - 50, height), (width // 2, height // 2 + 50)]
    ], dtype=np.int32)
    
    # Ek black image (mask) banao
    mask = np.zeros_like(img)
    
    # Us mask par apne polygon ko white (255) color se bhar do
    cv2.fillPoly(mask, polygons, 255)
    
    # Original edge image aur mask ka 'AND' operation karo
    # Isse sirf polygon ke andar ke edges bachenge
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def display_lines(img, lines):
    """
    Step 4(b): Draw lines on an empty image
    """
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # (0, 255, 0) Green color hai, 5 line ki thickness hai
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return line_image

def run_lane_detection(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image nahi mili!")
        return

    # Pipeline execute karo
    canny_image = canny_edge_detection(img)
    cropped_canny = region_of_interest(canny_image)
    
    # Step 4(a): Hough Transform
    # 2 distance resolution hai, np.pi/180 angle resolution hai, 
    # 100 minimum votes (intersections) hain line banne ke liye
    lines = cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    
    # Lines draw karo original image ke upar
    line_image = display_lines(img, lines)
    # Original image aur line image ko blend karo (alpha = 0.8)
    combo_image = cv2.addWeighted(img, 0.8, line_image, 1, 1)
    
    # Outputs display karo
    cv2.imshow("1. Canny Edges + Mask", cropped_canny)
    cv2.imshow("2. Final Lane Detection", combo_image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Test karne ke liye ek clear road ki image ka naam yahan daalna
    run_lane_detection("test_clear_road.jpg")