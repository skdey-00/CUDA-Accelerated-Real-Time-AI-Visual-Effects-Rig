import cv2
import numpy as np
import torch
import torch.nn as nn
from rembg import new_session, remove
import time
import datetime
import os
import subprocess

def get_clipboard_text():
    """Get text from Windows clipboard using PowerShell"""
    try:
        result = subprocess.run(['powershell', '-command', 'Get-Clipboard'],
                              capture_output=True, text=True, timeout=1)
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    return ""

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

# Stats window dimensions (base size - will scale proportionally)
BASE_STATS_WIDTH = 650
BASE_STATS_HEIGHT = 550  # More compact layout

# Current window size (updated each frame)
current_window_width = BASE_STATS_WIDTH
current_window_height = BASE_STATS_HEIGHT

# Recording state
is_recording = False
video_writer = None
recording_start_time = None
recording_frame_count = 0

# FPS tracking variables
fps_start_time = time.time()
frame_count = 0
fps = 0
display_fps = 0
fps_history = []

# Performance tracking
processing_times = []

# Color state
current_line_color = "#FFFFFF"  # Default white
line_color_bgr = (255, 255, 255)  # Default white in BGR
color_input_text = "#FFFFFF"
color_input_active = False
color_input_cursor_visible = True
color_input_cursor_timer = 0

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

def hex_to_bgr(hex_color):
    """Convert hex color code to BGR tuple"""
    # Remove # if present
    hex_color = hex_color.replace('#', '')

    # Validate hex code
    if len(hex_color) != 6:
        return None

    try:
        # Parse hex to RGB
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)

        # Return as BGR for OpenCV
        return (b, g, r)
    except ValueError:
        return None

def get_gpu_memory():
    """Get GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**2  # MB
        reserved = torch.cuda.memory_reserved(0) / 1024**2  # MB
        return allocated, reserved
    return 0, 0

def get_window_size():
    """Get current window size"""
    global current_window_width, current_window_height
    try:
        rect = cv2.getWindowImageRect('Stats & Control')
        if rect[2] > 0 and rect[3] > 0:
            current_window_width = rect[2]
            current_window_height = rect[3]
    except:
        current_window_width = BASE_STATS_WIDTH
        current_window_height = BASE_STATS_HEIGHT
    return current_window_width, current_window_height

def draw_stats_panel(fps, processing_time, gpu_allocated, gpu_reserved, recording_status, recording_time, input_text, input_active, cursor_visible, current_color):
    """Draw the stats/control panel with compact layout"""
    # Get actual window size
    width, height = get_window_size()

    # Calculate scale factors
    scale_x = width / BASE_STATS_WIDTH
    scale_y = height / BASE_STATS_HEIGHT
    scale = min(scale_x, scale_y)

    # Create stats panel
    stats = np.zeros((height, width, 3), dtype=np.uint8)
    stats[:] = (20, 20, 30)

    # Header
    header_height = int(55 * scale)
    cv2.rectangle(stats, (0, 0), (width, header_height), (30, 30, 50), -1)
    cv2.putText(stats, "STATS & CONTROL", (int(25 * scale), int(38 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.85 * scale, (0, 200, 255), 2)

    # Two-column layout
    col1_x = int(25 * scale)
    col2_x = int(340 * scale)
    y = int(75 * scale)
    line_h = int(28 * scale)
    font_size = 0.5 * scale

    # LEFT COLUMN - Stats
    cv2.putText(stats, "PERFORMANCE", (col1_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (100, 200, 100), 2)
    y += line_h

    fps_color = (0, 255, 0) if fps >= 20 else (0, 165, 255) if fps >= 10 else (0, 0, 255)
    cv2.putText(stats, f"FPS: {fps:.1f}", (col1_x + int(15 * scale), y), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, fps_color, 2)
    y += line_h

    cv2.putText(stats, f"Time: {processing_time:.1f}ms", (col1_x + int(15 * scale), y), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, (180, 180, 180), 1)

    # RIGHT COLUMN - GPU
    y = int(75 * scale)
    cv2.putText(stats, "GPU", (col2_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (100, 200, 100), 2)
    y += line_h

    if torch.cuda.is_available():
        cv2.putText(stats, "Device: GPU", (col2_x + int(15 * scale), y), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, (0, 200, 255), 1)
        y += line_h
        cv2.putText(stats, f"Mem: {gpu_allocated:.0f}/{gpu_reserved:.0f}MB", (col2_x + int(15 * scale), y), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale, (180, 180, 180), 1)
    else:
        cv2.putText(stats, "Device: CPU", (col2_x + int(15 * scale), y), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, (255, 100, 100), 1)

    # Recording Section
    y = int(170 * scale)
    cv2.putText(stats, "RECORDING", (col1_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (100, 200, 100), 2)
    y += line_h

    if recording_status:
        cv2.putText(stats, "● RECORDING", (col1_x + int(15 * scale), y), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, (0, 0, 255), 2)
        cv2.putText(stats, recording_time, (col1_x + int(140 * scale), y), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (200, 200, 200), 1)
    else:
        cv2.putText(stats, "READY", (col1_x + int(15 * scale), y), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, (0, 255, 0), 2)

    # Record Button
    button_x = int(200 * scale)
    button_y = int(215 * scale)
    button_width = int(250 * scale)
    button_height = int(50 * scale)
    button_color = (0, 0, 255) if not recording_status else (100, 0, 0)
    cv2.rectangle(stats, (button_x, button_y), (button_x + button_width, button_y + button_height), button_color, -1)
    cv2.rectangle(stats, (button_x, button_y), (button_x + button_width, button_y + button_height), (255, 255, 255), 2)

    button_text = "● RECORD" if not recording_status else "■ STOP"
    text_size = cv2.getTextSize(button_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65 * scale, 2)[0]
    text_x = button_x + (button_width - text_size[0]) // 2
    text_y = button_y + (button_height + text_size[1]) // 2
    cv2.putText(stats, button_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65 * scale, (255, 255, 255), 2)

    current_button_pos = {
        'x': button_x,
        'y': button_y,
        'w': button_width,
        'h': button_height
    }

    # Color Section
    y = int(290 * scale)
    cv2.putText(stats, "LINE COLOR", (col1_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (100, 200, 100), 2)

    # Color preview and label on same line
    preview_size = int(30 * scale)
    cv2.rectangle(stats, (col1_x, y + int(8 * scale)), (col1_x + preview_size, y + int(8 * scale) + preview_size), current_color, -1)
    cv2.rectangle(stats, (col1_x, y + int(8 * scale)), (col1_x + preview_size, y + int(8 * scale) + preview_size), (255, 255, 255), 2)
    cv2.putText(stats, "Hex:", (col1_x + preview_size + int(10 * scale), y + int(25 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, (180, 180, 180), 1)

    # Textbox
    textbox_x = int(130 * scale)
    textbox_y = y + int(35 * scale)
    textbox_width = int(495 * scale)
    textbox_height = int(45 * scale)

    textbox_bg = (40, 40, 60) if not input_active else (60, 80, 100)
    cv2.rectangle(stats, (textbox_x, textbox_y), (textbox_x + textbox_width, textbox_y + textbox_height), textbox_bg, -1)
    border_color = (100, 150, 200) if input_active else (100, 100, 150)
    cv2.rectangle(stats, (textbox_x, textbox_y), (textbox_x + textbox_width, textbox_y + textbox_height), border_color, 2)

    # Input text
    display_text = input_text
    if input_active and cursor_visible:
        display_text += "|"

    cv2.putText(stats, display_text, (textbox_x + int(12 * scale), textbox_y + int(30 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.65 * scale, (255, 255, 255), 2)

    current_textbox_pos = {
        'x': textbox_x,
        'y': textbox_y,
        'w': textbox_width,
        'h': textbox_height
    }

    # Helper text
    cv2.putText(stats, "Type hex (e.g. #FF0000) | Ctrl+V to paste | Enter to apply", (col1_x, textbox_y + textbox_height + int(18 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.35 * scale, (120, 120, 120), 1)

    # Footer
    footer_height = int(38 * scale)
    cv2.rectangle(stats, (0, height - footer_height), (width, height), (30, 30, 50), -1)
    cv2.putText(stats, "Press 'q' to quit | Click textbox to type | ESC to cancel",
                (int(25 * scale), height - int(14 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.38 * scale, (150, 150, 150), 1)

    return stats, current_button_pos, current_textbox_pos

# Global button and textbox positions (updated each frame)
current_button_pos = {}
current_textbox_pos = {}

def is_button_clicked(x, y):
    """Check if coordinates are within the button area"""
    if not current_button_pos:
        return False
    return current_button_pos['x'] <= x <= current_button_pos['x'] + current_button_pos['w'] and \
           current_button_pos['y'] <= y <= current_button_pos['y'] + current_button_pos['h']

def is_textbox_clicked(x, y):
    """Check if coordinates are within the textbox area"""
    if not current_textbox_pos:
        return False
    return current_textbox_pos['x'] <= x <= current_textbox_pos['x'] + current_textbox_pos['w'] and \
           current_textbox_pos['y'] <= y <= current_textbox_pos['y'] + current_textbox_pos['h']

# Mouse callback for button clicks and textbox
def mouse_callback(event, x, y, flags, param):
    global is_recording, video_writer, recording_start_time, recording_frame_count
    global color_input_active, color_input_text, current_line_color
    global current_button_pos, current_textbox_pos

    if event == cv2.EVENT_LBUTTONDOWN:
        if is_button_clicked(x, y):
            if not is_recording:
                # Start recording
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = "recordings"
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"recording_{timestamp}.mp4")

                # Get video properties
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                frame_height, frame_width = param['frame_shape'][:2]
                video_writer = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))

                is_recording = True
                recording_start_time = time.time()
                recording_frame_count = 0
                print(f"--- Recording started: {output_path} ---")
            else:
                # Stop recording
                is_recording = False
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                recording_time = time.time() - recording_start_time
                print(f"--- Recording stopped: {recording_frame_count} frames in {recording_time:.1f}s ---")
                print(f"--- Average FPS: {recording_frame_count / recording_time:.1f} ---")
        elif is_textbox_clicked(x, y):
            color_input_active = True
        else:
            color_input_active = False

# Pre-compute scanlines pattern (will be created on first frame)
scanlines_pattern = None
frame_height, frame_width = 0, 0

# Warmup frames
warmup_frames = 5

# Import required functions
from torch.nn.functional import pad, conv2d

# Create stats window and set mouse callback (resizable)
cv2.namedWindow('Stats & Control', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Stats & Control', BASE_STATS_WIDTH, BASE_STATS_HEIGHT)
cv2.setMouseCallback('Stats & Control', mouse_callback, param={'frame_shape': (480, 640, 3)})

# Helper function to handle keyboard input
def handle_keyboard_input(key):
    global color_input_text, line_color_bgr, current_line_color, color_input_active

    if key == ord('q'):
        return False  # Signal to quit
    elif color_input_active:
        # Check for Ctrl+V (paste)
        if key == 22:  # Ctrl+V
            clipboard_text = get_clipboard_text()
            # Filter to only valid hex characters
            filtered = ''.join(c for c in clipboard_text.upper() if c in '0123456789ABCDEF#')
            if filtered:
                # Add # if not present
                if '#' not in filtered and len(filtered) == 6:
                    filtered = '#' + filtered
                # Limit to 7 characters (#XXXXXX)
                color_input_text = filtered[:7]
                print(f"--- Pasted from clipboard: {color_input_text} ---")
        elif key == 13:  # Enter key - apply color
            # Parse and apply hex color
            new_color = hex_to_bgr(color_input_text)
            if new_color is not None:
                line_color_bgr = new_color
                current_line_color = color_input_text.upper()
                print(f"--- Color updated to: {color_input_text} ---")
            else:
                print(f"--- Invalid hex color: {color_input_text} ---")
            color_input_active = False
        elif key == 27:  # Escape key - cancel
            color_input_active = False
            color_input_text = current_line_color
        elif key == 8:  # Backspace
            color_input_text = color_input_text[:-1]
        elif 32 <= key <= 126:  # Printable characters
            # Allow hex characters and #
            char = chr(key)
            if char.upper() in '0123456789ABCDEF#':
                if len(color_input_text) < 7:  # Max length for hex color
                    color_input_text += char.upper()
    return True  # Continue running

while cap.isOpened():
    frame_start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Initialize frame dimensions on first frame
    if frame_height == 0:
        frame_height, frame_width = frame.shape[:2]
        # Pre-create scanlines pattern
        scanlines_pattern = create_scanlines_gpu(frame_height, frame_width)
        # Update mouse callback with correct frame shape
        cv2.setMouseCallback('Stats & Control', mouse_callback, param={'frame_shape': frame.shape})

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
        person_lines[edges > 0] = line_color_bgr

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
            fps_history.append(fps)
            if len(fps_history) > 60:
                fps_history.pop(0)
            frame_count = 0
            fps_start_time = current_time

        # Calculate processing time
        processing_time = (time.time() - frame_start_time) * 1000  # ms
        processing_times.append(processing_time)
        if len(processing_times) > 60:
            processing_times.pop(0)

        # Get GPU memory info
        gpu_allocated, gpu_reserved = get_gpu_memory()

        # Recording
        if is_recording and video_writer is not None:
            video_writer.write(final_view)
            recording_frame_count += 1

        # Calculate recording time
        recording_time_str = ""
        if is_recording and recording_start_time:
            rec_time = time.time() - recording_start_time
            minutes = int(rec_time // 60)
            seconds = int(rec_time % 60)
            recording_time_str = f"{minutes:02d}:{seconds:02d}"

        # Update cursor blink
        color_input_cursor_timer += 1
        if color_input_cursor_timer >= 30:  # Blink every ~0.5 seconds
            color_input_cursor_visible = not color_input_cursor_visible
            color_input_cursor_timer = 0

        # Draw stats panel
        stats_panel, btn_pos, txt_pos = draw_stats_panel(
            display_fps,
            np.mean(processing_times) if processing_times else 0,
            gpu_allocated,
            gpu_reserved,
            is_recording,
            recording_time_str,
            color_input_text,
            color_input_active,
            color_input_cursor_visible,
            line_color_bgr
        )

        # Update global positions for click detection
        current_button_pos = btn_pos
        current_textbox_pos = txt_pos

        # Display windows
        cv2.imshow('Music Video Filter', final_view)
        cv2.imshow('Stats & Control', stats_panel)

    # Skip warmup frames display
    if warmup_frames > 0:
        warmup_frames -= 1
        if warmup_frames == 0:
            print(f"--- GPU warmup complete! ---")
            print(f"--- Running on: {torch.cuda.get_device_name(0)} ---")

    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if not handle_keyboard_input(key):
        break

# Cleanup
cap.release()
if video_writer is not None:
    video_writer.release()
cv2.destroyAllWindows()

print(f"--- RIG CLOSED ---")
print(f"--- Average FPS: {display_fps:.1f} ---")
print(f"--- GPU acceleration: {'CUDA' if torch.cuda.is_available() else 'CPU'} ---")
if is_recording:
    print(f"--- Recording saved ---")
