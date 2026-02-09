# CUDA-Accelerated Real-Time AI Visual Effects Rig

A high-performance, real-time video effects rig powered by CUDA GPU acceleration and AI background removal.

## Features

- **GPU-Accelerated Background Removal** - Uses rembg AI model with CUDA support
- **Real-Time Edge Detection** - Canny edge detection for oscilloscope-style visual effects
- **GPU Image Processing** - PyTorch-powered Gaussian blur, blending, and bloom effects
- **Live FPS Monitoring** - Real-time performance tracking
- **CRT Scanline Effect** - Retro visual aesthetics

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA 12.4+ support
- Webcam

## Installation

```bash
# Install dependencies
pip install opencv-python numpy rembg torch torchvision onnxruntime-gpu
```

## Usage

```bash
python filter.py
```

Press `q` to quit the application.

## GPU Acceleration

This project utilizes:
- **PyTorch CUDA** - For Gaussian blur, image blending, and bloom effects
- **ONNX Runtime GPU** - For AI background removal model inference
- **NumPy/PyTorch** - For vectorized operations on GPU

## Performance

Expected performance on NVIDIA RTX 4050 Laptop GPU:
- 15-30 FPS depending on resolution
- Background removal: GPU-accelerated
- Image processing: GPU-accelerated

## Controls

- `q` - Quit application

## License

MIT License

## Author

Created with CUDA GPU acceleration for real-time visual effects processing.
