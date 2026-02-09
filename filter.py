import cv2
import numpy as np
import torch
import torch.nn as nn
from rembg import new_session, remove
import time

print("--- RIG INITIALIZED ---")
print("--- Loading background removal model ---")
print("--- Mode: FULL GPU ACCELERATION (PyTorch + CUDA) ---")

# Check PyTorch CUDA availability
if torch.cuda.is_available():
    print(f"--- PyTorch CUDA: Enabled ---")
    print(f"--- GPU: {torch.cuda.get_device_name(0)} ---")
    device = torch.device('cuda')
else:
    print("--- PyTorch CUDA not available, falling back to CPU ---")
    device = torch.device('cpu')

# Create a session for background removal with CUDA
try:
    session = new_session("u2net", providers=["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"])
    print("--- rembg model loaded on GPU! ---")
except Exception as e:
    print(f"--- GPU not available for rembg ({e}), falling back to CPU ---")
    session = new_session("u2net", providers=["CPUExecutionProvider"])

print("--- Make sure you have a webcam connected ---")

cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("--- Starting video stream. Press 'q' to quit ---")

# FPS tracking variables
fps_start_time = time.time()
frame_count = 0
fps = 0
display_fps = 0

# Create Gaussian kernel once
def create_gaussian_kernel(kernel_size=11, sigma=2.0, channels=1):
    """Create Gaussian kernel for convolution"""
    if sigma == 0:
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8

    # Create 1D Gaussian kernel
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / kernel.sum()

    # Reshape for conv2d: (out_channels, in_channels, height, width)
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    if channels > 1:
        kernel = kernel.repeat(channels, 1, 1, 1)

    return kernel.to(device)

# Pre-create Gaussian kernels
gaussian_kernel_1ch = create_gaussian_kernel(kernel_size=11, sigma=2.0, channels=1)
gaussian_kernel_3ch = create_gaussian_kernel(kernel_size=15, sigma=2.0, channels=3)

def to_gpu(image):
    """Convert numpy array to GPU tensor (make writable first)"""
    if not image.flags.writeable:
        image = image.copy()
    return torch.from_numpy(image).to(device)

def to_cpu(tensor):
    """Convert GPU tensor back to numpy"""
    return tensor.detach().cpu().numpy()

def gaussian_blur_gpu(image, kernel_size=11, sigma=2.0):
    """Apply Gaussian blur on GPU using PyTorch conv2d"""
    # Convert to tensor and add dimensions
    if len(image.shape) == 2:
        # Grayscale
        tensor = to_gpu(image).float().unsqueeze(0).unsqueeze(0) / 255.0
        kernel = gaussian_kernel_1ch
    else:
        # RGB
        tensor = to_gpu(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        kernel = gaussian_kernel_3ch

    # Apply convolution with Gaussian kernel
    with torch.no_grad():
        # Use reflection padding for better edge handling
        padding = kernel_size // 2
        tensor_padded = pad(tensor, (padding, padding, padding, padding), mode='reflect')

        # Apply convolution
        blurred = conv2d(tensor_padded, kernel, groups=tensor.shape[1])

    # Convert back
    if len(image.shape) == 2:
        result = (blurred.squeeze(0).squeeze(0) * 255).clamp(0, 255)
    else:
        result = (blurred.squeeze(0).permute(1, 2, 0) * 255).clamp(0, 255)

    return to_cpu(result).astype(np.uint8)

def canny_edge_detection_gpu(frame, threshold1=20, threshold2=60):
    """Apply Canny edge detection using OpenCV (already optimized)"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Less blur to preserve more details
    gray_blurred = cv2.GaussianBlur(gray, (3, 3), 0.5)
    return cv2.Canny(gray_blurred, threshold1, threshold2)

def blend_images_gpu(background, foreground, mask, threshold=50):
    """Blend images using mask - GPU accelerated"""
    # Convert to GPU tensors (make writable)
    bg = background.copy() if not background.flags.writeable else background
    fg = foreground.copy() if not foreground.flags.writeable else foreground
    m = mask.copy() if not mask.flags.writeable else mask

    bg_tensor = torch.from_numpy(bg).to(device).float()
    fg_tensor = torch.from_numpy(fg).to(device).float()
    mask_tensor = torch.from_numpy(m).to(device).float()

    # Normalize threshold from 0-255 range to 0-1 range
    threshold_normalized = threshold / 255.0

    # Expand mask to 3 channels
    mask_tensor = mask_tensor.unsqueeze(-1).expand(-1, -1, 3) / 255.0

    # Blend on GPU
    with torch.no_grad():
        condition = mask_tensor > threshold_normalized
        result = torch.where(condition, fg_tensor, bg_tensor)

    return to_cpu(result).astype(np.uint8)

def apply_bloom_gpu(output, kernel_size=15):
    """Apply bloom effect on GPU"""
    img = output.copy() if not output.flags.writeable else output
    output_tensor = torch.from_numpy(img).to(device).float() / 255.0
    output_tensor = output_tensor.permute(2, 0, 1).unsqueeze(0)

    # Apply Gaussian blur on GPU using convolution
    with torch.no_grad():
        padding = kernel_size // 2
        output_padded = pad(output_tensor, (padding, padding, padding, padding), mode='reflect')
        bloom_tensor = conv2d(output_padded, gaussian_kernel_3ch, groups=3)

        # Make sure dimensions match after padding and convolution
        if bloom_tensor.shape == output_tensor.shape:
            # Blend
            result = (output_tensor * 1.0 + bloom_tensor * 0.5).clamp(0, 1)
        else:
            # Fallback if dimensions don't match
            result = output_tensor

    # Convert back
    result = (result.squeeze(0).permute(1, 2, 0) * 255).clamp(0, 255)
    return to_cpu(result).astype(np.uint8)

def create_scanlines_gpu(height, width):
    """Create scanlines pattern on GPU"""
    # Create scanline mask on GPU
    scanline_mask = torch.zeros(height, width, device=device, dtype=torch.uint8)

    # Use vectorized indexing for scanlines (GPU accelerated)
    indices = torch.arange(0, height, 10, device=device)
    scanline_mask[indices, :] = 30

    return to_cpu(scanline_mask)

# Pre-compute scanlines pattern (will be created on first frame)
scanlines_pattern = None
frame_height, frame_width = 0, 0

# Warmup frames
warmup_frames = 5

# Import required functions
from torch.nn.functional import pad, conv2d

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Initialize frame dimensions on first frame
    if frame_height == 0:
        frame_height, frame_width = frame.shape[:2]
        # Pre-create scanlines pattern
        scanlines_pattern = create_scanlines_gpu(frame_height, frame_width)

    # Remove background - GPU accelerated
    result = remove(frame, session=session)

    # Convert RGBA result to get the alpha channel as our mask
    if result.shape[2] == 4:  # Has alpha channel
        mask = result[:, :, 3]  # Keep as 0-255 for efficiency

        # Blur the mask for smoother edges - GPU accelerated
        mask_blurred = gaussian_blur_gpu(mask, kernel_size=11)

        # THE OSCILLOSCOPE (Edges of you)
        edges = canny_edge_detection_gpu(frame, 20, 60)

        person_lines = np.zeros_like(frame)
        person_lines[edges > 0] = [255, 255, 255]

        # THE BACKGROUND (Black)
        bg = np.zeros_like(frame, dtype=np.uint8)

        # Add pre-computed scanlines
        bg[scanlines_pattern > 0] = [30, 30, 30]

        # COMBINE - GPU accelerated blending
        output = blend_images_gpu(bg, person_lines, mask_blurred, threshold=50)

        # Add a final "glow" - GPU accelerated
        final_view = apply_bloom_gpu(output, kernel_size=15)

        # Calculate FPS
        frame_count += 1
        current_time = time.time()
        elapsed = current_time - fps_start_time

        if elapsed >= 1.0:  # Update FPS every second
            fps = frame_count / elapsed
            display_fps = fps
            frame_count = 0
            fps_start_time = current_time

        # Draw FPS counter on frame
        fps_text = f"FPS: {display_fps:.1f}"
        device_text = f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}"
        cv2.putText(final_view, fps_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(final_view, device_text, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Music Video Filter', final_view)

    # Skip warmup frames display
    if warmup_frames > 0:
        warmup_frames -= 1
        if warmup_frames == 0:
            print(f"--- GPU warmup complete! ---")
            print(f"--- Running on: {torch.cuda.get_device_name(0)} ---")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"--- RIG CLOSED ---")
print(f"--- Average FPS: {display_fps:.1f} ---")
print(f"--- GPU acceleration: {'CUDA' if torch.cuda.is_available() else 'CPU'} ---")
