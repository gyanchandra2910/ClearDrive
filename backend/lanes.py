import cv2
import numpy as np

def canny_edge_detection(img):
    """
    Stage 1 & 2: Convert to grayscale, apply Gaussian smoothing, then detect edges.
    Canny edge detection operates on a blurred grayscale image to suppress noise
    before identifying strong gradient boundaries in the scene.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Hysteresis thresholds: edges with gradient > 150 are strong, 50–150 are weak (linked if connected)
    canny = cv2.Canny(blur, 50, 150)
    return canny


def region_of_interest(img):
    """
    Stage 3: Mask the edge image to retain only the road-relevant triangular region.
    This eliminates edges from the sky, dashboard, and surrounding environment,
    ensuring the Hough Transform focuses exclusively on the drivable lane area.
    """
    height = img.shape[0]
    width = img.shape[1]

    # Define a trapezoidal ROI that covers the lower field of view where road lanes appear
    polygons = np.array([
        [(50, height), (width - 50, height), (width // 2, height // 2 + 50)]
    ], dtype=np.int32)

    # Create a blank mask of the same size as the edge image
    mask = np.zeros_like(img)

    # Paint the polygon region white (255) on the mask
    cv2.fillPoly(mask, polygons, 255)

    # Apply bitwise AND to isolate edges only within the defined polygon boundary
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def display_lines(img, lines):
    """
    Stage 4b: Render detected lane lines onto a blank transparent overlay image.
    The result is blended into the original frame by the caller.
    """
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Draw each line segment in green with 5px thickness for visibility
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return line_image


def run_lane_detection(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found. Check the provided file path.")
        return

    # Run the full lane detection pipeline sequentially
    canny_image = canny_edge_detection(img)
    cropped_canny = region_of_interest(canny_image)

    # Stage 4a: Probabilistic Hough Transform
    # Parameters: rho=2 (distance resolution), theta=pi/180 (angle resolution),
    # threshold=100 (minimum vote intersections to confirm a line segment)
    lines = cv2.HoughLinesP(
        cropped_canny, 2, np.pi / 180, 100,
        np.array([]), minLineLength=40, maxLineGap=5
    )

    # Draw detected lines onto a transparent layer
    line_image = display_lines(img, lines)

    # Alpha-blend the line overlay onto the original frame
    combo_image = cv2.addWeighted(img, 0.8, line_image, 1, 1)

    # Display the intermediate edge mask and final lane detection result
    cv2.imshow("1. Canny Edges + ROI Mask", cropped_canny)
    cv2.imshow("2. Final Lane Detection Overlay", combo_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Standalone test — provide a clear road image to validate detection
    run_lane_detection("test_clear_road.jpg")