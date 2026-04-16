import cv2
import numpy as np

def get_dark_channel(img, window_size=15):
    """
    Step 1: Compute the Dark Channel of an image.
    For each pixel, take the minimum value across all RGB color channels.
    Then apply morphological erosion over a local patch window to extract the
    minimum intensity within each neighbourhood, forming the dark channel map.
    """
    # Find the per-pixel minimum value across the three color channels
    min_channel = np.min(img, axis=2)

    # Apply local patch minimum using morphological erosion over window_size x window_size
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel


def get_atmospheric_light(img, dark_channel, top_percent=0.001):
    """
    Step 2: Estimate the global Atmospheric Light (A).
    Identify the top 0.1% brightest pixels in the dark channel map,
    then average their corresponding colors in the original image to
    approximate the dominant environmental illumination vector.
    """
    h, w = dark_channel.shape
    num_pixels = h * w
    num_top_pixels = int(max(num_pixels * top_percent, 1))

    # Flatten spatial arrays to 1D for efficient index-based sorting
    dark_flat = dark_channel.reshape(-1)
    img_flat = img.reshape(-1, 3)

    # Get indices of the highest-intensity pixels from the dark channel
    indices = np.argsort(dark_flat)[-num_top_pixels:]

    # Average the RGB color vectors at those indices in the original image
    A = np.mean(img_flat[indices], axis=0)
    return A


def get_transmission(img, A, window_size=7, omega=0.85):
    """
    Step 3: Estimate the Transmission Map (t).
    Formula: t(x) = 1 - omega * dark_channel(I / A)
    The omega parameter controls dehazing aggressiveness (0.85–0.97 for automotive use).
    """
    # Normalize the image by the atmospheric light to isolate haze contribution
    norm_img = img / A

    # Compute the transmission coefficient from the normalized dark channel
    transmission = 1 - omega * get_dark_channel(norm_img, window_size)
    return transmission


def recover_image(img, t, A, t0=0.25):
    """
    Step 4: Recover the original scene radiance (J).
    Formula: J(x) = (I(x) - A) / max(t(x), t0) + A
    The floor value t0 prevents division-by-zero in regions of near-zero transmission.
    """
    # Clamp transmission values at t0 to avoid numerical instability
    t_matrix = np.maximum(t, t0)

    # Expand the single-channel transmission map to 3 channels for RGB broadcasting
    t_matrix = cv2.merge([t_matrix, t_matrix, t_matrix])

    # Apply the atmospheric scattering recovery formula
    J = (img - A) / t_matrix + A

    # Clip the output values to valid 8-bit image range [0, 255]
    J = np.clip(J, 0, 255).astype(np.uint8)
    return J


def run_dehaze_pipeline(image_path):
    # Load the source image from disk
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found. Verify the file path and try again.")
        return

    # Convert to float64 for high-precision arithmetic throughout the pipeline
    img_float = img.astype('float64')

    # Execute the full Dark Channel Prior dehazing pipeline
    print("Processing: Executing Dark Channel Prior dehazing pipeline...")
    dark = get_dark_channel(img_float)
    A = get_atmospheric_light(img_float, dark)
    t = get_transmission(img_float, A)

    # Smooth the raw transmission map using Gaussian blur.
    # The raw map tends to have block artifacts at patch boundaries;
    # smoothing improves visual quality without requiring a guided filter.
    t_smoothed = cv2.GaussianBlur(t, (15, 15), 0)

    recovered = recover_image(img_float, t_smoothed, A)

    # Display the before/after comparison
    cv2.imshow("Original Foggy Scene", img)
    cv2.imshow("Dehazed Output", recovered)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Run standalone test using a foggy test image
    run_dehaze_pipeline("test_fog.jpg")