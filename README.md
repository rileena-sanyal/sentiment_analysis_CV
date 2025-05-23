# Video Sentiment Analyser

A real-time video sentiment analysis tool that uses GPU acceleration (when available) to detect faces and classify their emotions/sentiments. It draws bounding boxes and overlays sentiment labels on a live webcam feed or video file, and tracks per-person sentiment history and performance statistics.

---

## Features

- **GPU-accelerated face detection** via [MTCNN (facenet-pytorch)](https://github.com/timesler/facenet-pytorch) or fallback to [RetinaFace](https://github.com/serengil/retinaface).
- **Emotion recognition**:
  - Preferred: [`FER`](https://github.com/justinshenk/fer) library (with MTCNN).
  - Fallback: lightweight custom PyTorch CNN model.
- **Batch processing** on the GPU for maximal throughput.
- **Frame-skipping + overlay caching** to maintain smooth display while saving compute.
- **Per-person tracking** and history of last 30 frames.
- **On-screen performance** stats: FPS, average latency, device, people count, sentiment distribution.
- Save output video or snapshots on key press.
- “Reset” tracking on demand.

---

## Table of Contents

1. [Requirements](#requirements)  
2. [Installation](#installation)  
3. [Usage](#usage)  
4. [Configuration & Options](#configuration--options)  
5. [How It Works](#how-it-works)  
6. [Extending & Retraining](#extending--retraining)  
7. [License](#license)  

---

## Requirements

- Python 3.7+  
- [PyTorch](https://pytorch.org/) & CUDA toolkit (optional but recommended)  
- [torchvision](https://github.com/pytorch/vision)  
- [opencv-python](https://pypi.org/project/opencv-python/)  
- [numpy](https://numpy.org/)  

Optional (for best accuracy and speed):

- [facenet-pytorch](https://pypi.org/project/facenet-pytorch/)  
- [fer](https://pypi.org/project/fer/)  
- [retinaface](https://pypi.org/project/retinaface/)  

---

## Installation

1. **Clone this repository**  
   ```bash
   git clone https://github.com/your-org/cuda-video-sentiment.git
   cd cuda-video-sentiment

