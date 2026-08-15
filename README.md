# 🌌 Spatial Vision AR (URANTUNE_WL_OT)

> A computer vision-based interactive system enabling real-time 3D hand tracking, spatial wireframe rendering, and air-writing recognition using OpenCV, MediaPipe, and EasyOCR.

---

## 📌 Overview

**Spatial Vision AR** bridges physical interactions with augmented reality by tracking hands in 3D spatial coordinates using standard camera inputs. Built with **OpenCV**, **MediaPipe**, and **EasyOCR**, the system interprets hand gestures to draw air-written text on a translucent Notepad overlay, render interactive 3D spatial wireframe meshes, and project virtual AR controls seamlessly in real time.

---

## ✨ Key Features

- 🖐️ **2D & 3D Landmark Detection** - Tracks 21 hand keypoints ($X, Y, Z$) using Google MediaPipe Hands pre-trained deep learning pipeline.
- 📐 **Pose & Geometry Wireframe Engine** - Renders 3D spatial meshes, polygon faces, and prints real-time $(X, Y)$ coordinate labels on landmarks.
- 📑 **Virtual UI & Lined Notepad Card** - Interactive right-side glassmorphic toolbar with Light, Notepad, and Power buttons, plus a left-side notebook paper card.
- ✍️ **Air-Writing & Optical Character Recognition (OCR)** - Draw text in the air using finger gestures and recognize handwritten words via EasyOCR.
- ⚡ **High FPS & Low Latency** - Optimized OpenCV frame processing with top-left `( 33 FPS )` status pill.

---

## 🛠️ System Architecture & Computer Vision Pipeline

1. **Palm & Hand Landmark Detection**
   - MediaPipe Palm Detector locates hand regions in the frame.
   - Hand Landmark Model places 21 keypoints per hand with 3D depth tracking.
2. **Mathematical Geometry & 3D Mesh Rendering**
   - Calculates spatial vectors between keypoints across hands.
   - Renders semi-transparent wireframe meshes using `cv2.fillPoly` and `cv2.polylines`.
   - Displays dynamic $(X, Y)$ coordinate tags adjacent to fingertips.
3. **Air Writing & OCR Engine**
   - Finger movement trajectories are recorded onto the Notepad overlay.
   - EasyOCR converts drawn strokes into digital text (e.g. "Hello, My name is P Khang").

---

## 📁 Repository Structure

```text
spatial-vision-ar/
├── main.py                # Main application entry point (URANTUNE_WL_OT)
├── src/
│   ├── config.py          # Color codes, resolution, and window configuration
│   ├── hand_tracker.py    # MediaPipe 3D landmark extraction & EMA smoothing
│   ├── wireframe_engine.py# 3D spatial mesh & coordinate rendering
│   ├── air_scribble.py    # Air-writing canvas & notepad ink logic
│   ├── ocr_engine.py      # EasyOCR text recognition engine
│   └── ui_manager.py      # Right toolbar, Light/Notepad/Power buttons & FPS badge
├── tests/
│   └── test_tracker.py    # Unit tests suite
├── requirements.txt       # Dependencies list
├── .gitignore             # Git ignore rules
├── LICENSE                # MIT License
└── README.md              # Documentation
```

---

## 🚀 Getting Started

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/rabeyanoor/spatial-vision-ar.git
   cd spatial-vision-ar
   ```

2. **Create a Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Application**
   ```bash
   python main.py
   ```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Check the [issues page](https://github.com/rabeyanoor/spatial-vision-ar/issues).

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more details.
