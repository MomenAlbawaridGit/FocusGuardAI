# FocusGuard AI 👁️📱

> A real-time computer vision application that acts as a strict digital supervisor. It uses deep learning to monitor your focus and instantly triggers a visual intervention (playing a custom video) if you look away from your screen or pick up your smartphone.


## 🚀 The Concept
I built this project to solve a common problem: getting distracted while working. Instead of relying on willpower, I engineered a zero-latency pipeline using OpenCV, YOLOv8, and MediaPipe to enforce productivity. 

The application runs seamlessly in the background with a lightweight, "always-on-top" picture-in-picture widget, leveraging CUDA on an RTX 4060 for buttery-smooth real-time inference.

## ✨ Features
* **Real-Time Phone Detection:** Utilizes a YOLO nano model optimized via PyTorch/CUDA to instantly detect if a smartphone enters the frame.
* **3D Head Pose Estimation:** Uses MediaPipe Face Mesh and OpenCV's `solvePnP` to calculate pitch and yaw, accurately determining if you are looking at your screen or distracted.
* **State Management:** Implements a customizable "grace period" timer before triggering the penalty to prevent false positives from quick glances.
* **System Integration:** Automatically takes over your screen using OS-level subprocesses to play a penalty video once the distraction threshold is breached.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Computer Vision:** OpenCV (`cv2`)
* **Object Detection:** Ultralytics YOLOv8 (COCO Dataset, Class 67)
* **Pose Estimation:** Google MediaPipe
* **Hardware Acceleration:** CUDA / cuDNN

## ⚙️ Installation & Quick Start

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/FocusGuard-AI.git](https://github.com/yourusername/FocusGuard-AI.git)
   cd FocusGuard-AI