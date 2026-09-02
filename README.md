# 🌌 Spatial Vision AR (AirGesture 3D System)

> High-precision 3D spatial vision and touchless gesture interaction system featuring real-time 3D hand tracking, touchless phalanx virtual keyboard, AR 3D mesh volume rendering, glassmorphic UI, and natural language auto-capitalization.

---

## 📌 Overview

**Spatial Vision AR** is a cutting-edge computer vision application that transforms regular RGB camera input into a 3D touchless spatial computing interface. Powered by **OpenCV** and **Google MediaPipe**, the system tracks 21 3D hand keypoints per hand to enable intuitive gesture controls, air-typing with two-handed phalanx key selection, 3D wireframe mesh projection, and real-time AR UI overlay elements.

---

## ✨ Key Features

- 🖐️ **3D Hand Tracking & Landmark Smoothing** - Extracts 21 keypoint coordinates ($X, Y, Z$) per hand with real-time landmark smoothing and confidence filtering.
- 📐 **3D Mesh & Wireframe Projection** - Visualizes translucent 3D mesh hulls, fingertip callout tags, and dual-hand volumetric bounding wireframes.
- ⌨️ **Phalanx Touchless Virtual Keyboard** - Touchless dual-hand character matrix selection (A–Z) with strict 3D depth verification and touch debouncing.
- ✍️ **Air Drawing & Stroke Recognition** - Captures real-time finger motion strokes and performs character gesture recognition.
- 📝 **NLP Auto-Capitalization Engine** - Applies automatic sentence-start capitalization, standalone 'i' correction, and natural language formatting.
- 🖥️ **Glassmorphic AR UI & Lined Notepad** - Interactive HUD featuring FPS counter, status indicators, animated sidebar controls, and virtual lined notepad.

---

## 📁 Repository Architecture

```text
spatial-vision-ar/
├── main.py                  # Main entry point and real-time interaction loop
├── hand_tracker.py          # MediaPipe 3D hand tracking and coordinate extraction
├── gesture_recognizer.py    # Phalanx touchless typing keyboard and gesture engine
├── ar_mesh_3d.py            # 3D mesh renderer and spatial wireframe volume projection
├── ar_ui_renderer.py        # Glassmorphic AR UI, HUD, and notepad renderer
├── auto_capitalizer.py      # NLP sentence formatting and auto-capitalization engine
├── air_drawing_ocr.py       # Fingertip stroke capture and air drawing recognition
├── run_video_test.py        # Test runner and video processing benchmark script
├── requirements.txt         # Project dependencies list
├── LICENSE                  # MIT License
└── README.md                # System documentation
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Webcam or video file source

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/rabeyanoor/spatial-vision-ar.git
   cd spatial-vision-ar
   ```

2. **Set Up Environment**
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

5. **Run Video Test Benchmark**
   ```bash
   python run_video_test.py
   ```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more details.
