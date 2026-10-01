import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def ensure_dir(directory: str) -> None:
    """Ensure output directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def inspect_image(image_path: str) -> dict:
    """Task 1: Load image and return measured image-data properties."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image from path: {image_path}")

    height, width, channels = img_bgr.shape
    shape_list = [height, width, channels]
    pixel_count = width * height
    estimated_bytes = pixel_count * channels * 1  # 8-bit per channel = 1 byte per channel

    return {
        "width": width,
        "height": height,
        "channels": channels,
        "shape": shape_list,
        "pixel_count": pixel_count,
        "estimated_bytes": estimated_bytes,
        "color_order": "BGR"  # OpenCV loads images as BGR by default
    }

def create_pixel_views(image_path: str, output_dir: str) -> dict:
    """Task 2: Create red, green, blue channels, grayscale, and downsampled views."""
    ensure_dir(output_dir)
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Extract channels
    R = img_rgb[:, :, 0]
    G = img_rgb[:, :, 1]
    B = img_rgb[:, :, 2]
    
    # Grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Downsample (half width, half height)
    h, w = img_bgr.shape[:2]
    downsampled_w, downsampled_h = w // 2, h // 2
    downsampled = cv2.resize(img_rgb, (downsampled_w, downsampled_h), interpolation=cv2.INTER_AREA)

    # Plotting montage
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    axes[0, 0].imshow(R, cmap='Reds')
    axes[0, 0].set_title('Red Channel')
    axes[0, 1].imshow(G, cmap='Greens')
    axes[0, 1].set_title('Green Channel')
    axes[0, 2].imshow(B, cmap='Blues')
    axes[0, 2].set_title('Blue Channel')
    
    axes[1, 0].imshow(gray, cmap='gray')
    axes[1, 0].set_title('Grayscale')
    axes[1, 1].imshow(downsampled)
    axes[1, 1].set_title(f'Downsampled ({downsampled_w}x{downsampled_h})')
    
    axes[1, 2].axis('off')  # Hide empty subplot slot
    
    for ax in axes.flat:
        ax.axis('off')
        
    plt.tight_layout()
    output_path = os.path.join(output_dir, "pixel_views.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    return {
        "original_size": [w, h],
        "downsampled_size": [downsampled_w, downsampled_h],
        "output_path": output_path
    }

def create_adjustments(
    image_path: str,
    output_dir: str,
    brightness_delta: int = 40,
    contrast_factor: float = 1.5,
    threshold: int = 127,
) -> dict:
    """Task 3: Create brightness, contrast, and thresholding results."""
    ensure_dir(output_dir)
    if not (0 <= threshold <= 255):
        raise ValueError("Threshold must be between 0 and 255")

    img_bgr = cv2.imread(image_path)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Brightness adjustment
    brighter = np.clip(gray.astype(np.int16) + brightness_delta, 0, 255).astype(np.uint8)

    # Contrast adjustment
    contrast = np.clip(gray.astype(np.float32) * contrast_factor, 0, 255).astype(np.uint8)

    # Binary Thresholding
    _, thresh_img = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Plot montage
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('Original Grayscale')
    
    axes[0, 1].imshow(brighter, cmap='gray')
    axes[0, 1].set_title(f'Brighter (+{brightness_delta})')
    
    axes[1, 0].imshow(contrast, cmap='gray')
    axes[1, 0].set_title(f'High Contrast (x{contrast_factor})')
    
    axes[1, 1].imshow(thresh_img, cmap='gray')
    axes[1, 1].set_title(f'Thresholded ({threshold})')

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    output_path = os.path.join(output_dir, "adjustments.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    return {
        "brightness_delta": brightness_delta,
        "contrast_factor": contrast_factor,
        "threshold": threshold,
        "output_path": output_path
    }

def create_blur_and_edges(
    image_path: str,
    output_dir: str,
    kernel_size: int = 5,
) -> dict:
    """Task 4: Create mean-blur and Sobel edge detection results."""
    ensure_dir(output_dir)
    if kernel_size % 2 == 0 or kernel_size <= 0:
        raise ValueError("kernel_size must be a positive odd integer")

    img_bgr = cv2.imread(image_path)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Mean blur
    blurred = cv2.blur(gray, (kernel_size, kernel_size))

    # Function for combined Sobel edge detection magnitude
    def get_sobel(img):
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(sobelx, sobely)
        return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    sobel_orig = get_sobel(gray)
    sobel_blur = get_sobel(blurred)

    # Plot montage
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('Original Grayscale')
    
    axes[0, 1].imshow(blurred, cmap='gray')
    axes[0, 1].set_title(f'Mean Blur (Kernel {kernel_size}x{kernel_size})')
    
    axes[1, 0].imshow(sobel_orig, cmap='gray')
    axes[1, 0].set_title('Sobel Edges (Original)')
    
    axes[1, 1].imshow(sobel_blur, cmap='gray')
    axes[1, 1].set_title('Sobel Edges (Blurred)')

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    output_path = os.path.join(output_dir, "blur_and_edges.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    return {
        "kernel_size": kernel_size,
        "output_path": output_path
    }

def run_lab(image_path: str, output_dir: str) -> dict:
    """Task 5: Run Tasks 1–4 and return consolidated results."""
    t1 = inspect_image(image_path)
    t2 = create_pixel_views(image_path, output_dir)
    t3 = create_adjustments(image_path, output_dir)
    t4 = create_blur_and_edges(image_path, output_dir)

    return {
        "task1": t1,
        "task2": t2,
        "task3": t3,
        "task4": t4
    }

def main() -> None:
    """Main execution entry point for repository root."""
    image_path = "images/original.jpg"
    output_dir = "outputs"
    ensure_dir(output_dir)
    
    results = run_lab(image_path, output_dir)
    
    # Print Task 1 summary to console for reporting verification
    t1 = results["task1"]
    print("--- Task 1: Image Inspection Data ---")
    print(f"Width: {t1['width']} px")
    print(f"Height: {t1['height']} px")
    print(f"Channels: {t1['channels']}")
    print(f"Shape: {t1['shape']}")
    print(f"Pixel Count: {t1['pixel_count']}")
    print(f"Estimated Size: {t1['estimated_bytes']} bytes")
    print(f"Color Order: {t1['color_order']}")
    print("-----------------------------------")
    print("Lab execution complete. Outputs saved in 'outputs/'.")

def main() -> None:
    """Main execution entry point for repository root."""
    # Get the directory where this script file is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    image_path = os.path.join(base_dir, "images", "original.jpg")
    output_dir = os.path.join(base_dir, "outputs")
    
    ensure_dir(output_dir)
    
    results = run_lab(image_path, output_dir)
    print("Lab execution completed successfully. Outputs saved to outputs/.")

if __name__ == "__main__":
    main()