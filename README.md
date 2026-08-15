# 🌌 Spatial Vision AR

> A computer vision-based interactive system enabling real-time 3D hand tracking, spatial wireframe rendering, and air-writing recognition using OpenCV and MediaPipe.

---

## 📌 Overview

**Spatial Vision AR** bridges physical interactions with augmented reality by tracking hands in 3D spatial coordinates using standard camera inputs. Built with **OpenCV** and **MediaPipe**, the system interprets hand gestures to draw air-written text and graphics, render interactive 3D spatial wireframes, and project virtual AR overlays seamlessly in real time.

---

## ✨ Key Features

- 🖐️ **Real-Time 3D Hand Tracking** - Tracks 21 hand keypoints with depth estimation ($Z$-coordinate spatial positioning).
- ✍️ **Air-Writing & Gesture Canvas** - Draw, sketch, and erase in 3D air space with high precision and stroke smoothing.
- 📐 **Spatial Wireframe Rendering** - Renders dynamic 3D geometric wireframes, coordinate grids, and bounding shapes around objects or hands.
- 🎨 **Interactive AR Canvas** - Dynamic color palettes, stroke width adjustments, and gesture-triggered canvas resets.
- ⚡ **High FPS & Low Latency** - Optimized video processing pipeline using OpenCV and MediaPipe Hands solution.

---

## 🛠️ Tech Stack & Dependencies

- **Language** - Python 3.9+
- **Computer Vision** - `opencv-python`
- **Hand Tracking ML Pipeline** - `mediapipe`
- **Numerical Computations** - `numpy`

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python installed on your system.

```bash
python --version
```

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/rabeyanoor/spatial-vision-ar.git
   cd spatial-vision-ar
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install Required Packages**
   ```bash
   pip install opencv-python mediapipe numpy
   ```

---

## 🎮 How It Works

1. **Video Stream Capture** - OpenCV captures live frame buffers from your webcam.
2. **Landmark Extraction** - MediaPipe Hands extracts 21 3D landmarks ($X, Y, Z$) per hand.
3. **Gesture & Coordinate Mapping**
   - **Index Fingertip (Landmark 8)** - Serves as the primary drawing pointer.
   - **Pinch / Finger Distance** - Toggles selection, drawing modes, or wireframe scaling.
4. **Air-Writing & Wireframe Engine** - Tracks coordinate histories to render continuous trajectories and project 3D spatial wireframe meshes over the live feed.

---

## 📁 Repository Structure

```text
spatial-vision-ar/
├── README.md              # Project documentation
├── main.py                # Core application entry point
├── src/
│   ├── hand_tracker.py    # MediaPipe 3D landmark extraction pipeline
│   ├── air_canvas.py      # Air-writing recognition & rendering logic
│   └── wireframe_3d.py    # 3D spatial mesh & geometry projection
└── requirements.txt       # Project dependencies
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/rabeyanoor/spatial-vision-ar/issues).

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more details.
