---
title: AirGesture 3D System
emoji: 🖐️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
---

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
- ⌨️ **Hand Keyboard** - Letters drawn on the finger joints; touch a joint with the thumb to type, palm centre = DEL (optional on-screen keyboard too).
- ✍️ **Air Drawing & Stroke Recognition** - Captures real-time finger motion strokes and performs character gesture recognition.
- 📝 **NLP Auto-Capitalization Engine** - Applies automatic sentence-start capitalization, standalone 'i' correction, and natural language formatting.
- 🧊 **Hand-Controlled 3D Boxes** - Sidebar opens a glowing pink particle-liquid box that sloshes under gravity and a Rubik's cube with animated layer turns; rotate, carry, resize with two hands, or squeeze with both palms to burst it.
- 🖥️ **Glassmorphic AR UI & Lined Notepad** - Interactive HUD featuring FPS counter, status indicators, animated sidebar controls, and virtual lined notepad.

---

## 📁 Repository Architecture

```text
spatial-vision-ar/
├── main.py                  # Desktop entry point (webcam window + keyboard shortcuts)
├── app.py                   # Gradio / Hugging Face Space entry point
├── pipeline.py              # Shared per-frame pipeline used by main.py, app.py and run_video_test.py
├── hand_tracker.py          # MediaPipe 3D hand tracking and coordinate extraction
├── gesture_recognizer.py    # Phalanx touchless typing keyboard and gesture engine
├── ar_mesh_3d.py            # 3D mesh renderer and spatial wireframe volume projection
├── ar_ui_renderer.py        # Glassmorphic AR UI, HUD, and notepad renderer
├── ar_objects_3d.py         # Hand-controlled 3D liquid box and Rubik's cube
├── auto_capitalizer.py      # NLP sentence formatting and auto-capitalization engine
├── air_writing.py           # Air writing: pinch and write letters in the air ($P recogniser)
├── air_keyboard.py          # Optional on-screen keyboard
├── run_video_test.py        # Test runner and video processing benchmark script
├── web/                     # Phone / browser version (index.html, js/, serve_https.py)
├── requirements.txt         # Hugging Face Space dependencies (headless OpenCV)
├── requirements-desktop.txt # Local desktop dependencies (OpenCV with GUI window support)
├── assets/fonts/            # Architects Daughter handwriting font (SIL OFL) for the notepad
├── LICENSE                  # MIT License
└── README.md                # System documentation
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 – 3.11 (MediaPipe 0.10.9 has no wheels for newer Python)
- Webcam or video file source

### Installation

```bash
git clone https://github.com/rabeyanoor/spatial-vision-ar.git
cd spatial-vision-ar

# With uv (picks a compatible Python automatically)
uv venv venv --python 3.11
uv pip install --python venv/bin/python -r requirements-desktop.txt

# ...or with plain venv + pip
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements-desktop.txt
```

> `requirements.txt` uses `opencv-python-headless` for the Hugging Face Space, which cannot open a window.
> For the desktop app, install `requirements-desktop.txt` instead.

### Run

```bash
venv/bin/python main.py                          # webcam (mirrored, 1280x720)
venv/bin/python main.py --source my_video.mp4    # video file
venv/bin/python main.py --debug                  # show mode / hands / gesture status line
venv/bin/python app.py                           # Gradio web demo on http://localhost:7860
venv/bin/python run_video_test.py video.mp4      # offline: writes output_demo.mp4 + demo_frames/
```

On GNOME Wayland, prefix with `QT_QPA_PLATFORM=xcb` if the window does not appear.

Performance: the desktop app uses MediaPipe's lite hand model and runs hand tracking on its own thread, so the
picture stays at the camera's 30 fps while tracking updates in the background; the webcam is opened in MJPG
mode (1280x720 @ 30 fps) and air-writing recognition also runs in the background.

---

## 📱 Phone / Web Version

`web/` is a browser version of the whole app (hand mesh, notepad, sidebar, liquid box, Rubik's cube).
Hand tracking runs **on the phone itself** with MediaPipe Tasks for Web, so there is no server lag.

```bash
venv/bin/python web/serve_https.py      # prints https://<computer-ip>:8443
```

1. Connect the phone to the same Wi-Fi as the computer and open the printed `https://` link.
2. The certificate is self-signed: tap **Advanced → Proceed** once (phones only allow the camera on https pages).
3. Tap **Start (front camera)** or **Start with back camera**, and allow camera access.

**Over a USB cable (Android, no certificate warning):** turn on *Developer options → USB debugging* on the
phone, plug it in, tap *Allow* on the phone, keep `web/serve_https.py` running and run
`venv/bin/python web/open_on_phone.py` — it opens the app in the phone's Chrome with the back camera.

Layout: full-screen camera with the notepad floating on top. Add `?split` to the link for a split layout
(camera on the top 2/3, notepad panel on the bottom 1/3).

Extras on the web version:
- **● Record** saves the screen (camera + all effects) as a video (MP4 where supported, otherwise WebM);
  on phones the share sheet offers "Save video".
- **Switch camera** flips between front and back camera at any time.
- Touch: tap the right-edge handle or a tile; drag a box to rotate it; long-press a box (0.4 s) then drag to move it; pinch with two fingers to resize / move;
  double-tap a box to burst it; on the Rubik's cube drag a block with one finger to turn its layer live (drag outside the cube to turn the whole cube).
- URL presets: `?camera=back&autostart&box=liquid|rubik&notepad&sidebar`.

The phone needs internet access the first time (MediaPipe and the model load from jsDelivr / Google storage).
Hand tracking runs in a Web Worker (`web/js/hand_worker.js`, CPU/XNNPACK), so the picture and recordings stay
smooth on slower phones while hand positions update in the background. `web/vendor/vision_bundle.js` is a local
copy of `@mediapipe/tasks-vision` 0.10.14 (Apache-2.0) because workers cannot load the CDN's `.cjs` file.

---

## 🖐️ Gestures

| Gesture | Action |
|---|---|
| Show 1 hand | Cyan dots with `x, y` coordinates on the extended fingertips |
| Show 2 hands | Cyan dots with coordinates on both hands' fingertips |
| Swipe right / move hand into the right edge | Slide in the sidebar |
| Index tip on **Bulb** tile | Toggle dark theme |
| Index tip on **Document** tile | Show / hide the notepad |
| Index tip on **Liquid Box** tile | Open / close the 3D liquid box |
| Index tip on **Rubik's Cube** tile | Open / close the Rubik's cube |
| Index tip on **Power** tile | Clear notepad text, close boxes and hide the sidebar |

**3D boxes** (one open at a time)

| Gesture | Action |
|---|---|
| Pinch thumb + index inside the box and drag | Rotate the box (twist the wrist to roll); it keeps spinning briefly after release |
| Rest an open palm on the box for ~0.5 s (or make a fist on it), then move the hand | Carry the box anywhere — up, down, left, right |
| Pinch with **both** hands, then pull apart / push together | Make the box bigger / smaller (moving both hands also moves and rolls it) |
| Put both open palms on either side of the box and press them together | The box **bursts** into pieces, then pops back in fresh |
| Turn the liquid box | The glowing liquid flows to the new bottom under gravity |
| **Rubik's cube:** pinch on a block (sticker) and drag | The row / column through that block turns **live with your hand** and locks in at 90°; keep moving for more turns. Let go early: past ~35° it finishes, otherwise it snaps back |
| **Rubik's cube:** pinch on a block and twist the wrist | The face that block is on turns the same way (clockwise / anticlockwise), like a real cube |
| Pinch ring | A ring follows each hand's pinch point (filled while pinching); the grabbed block / layer gets a white outline |
| **Rubik's cube:** make a fist on the cube and move | Turn the whole cube to see the other sides |
| **Rubik's cube:** point with only the index finger and swipe across a block | Also turns that block's layer |
| Hold / turn the cube with one hand and turn layers with the other | Two-hand solving; "SOLVED!" appears when every face is one colour |

**Air writing** (default when the notepad is open) — works with one hand, so it also suits a phone.

| Gesture | Action |
|---|---|
| Hold **2 fingers together** (index + middle, ring and pinky folded) or **3 fingers together** (thumb + index + middle, like holding a pen) | Pen down: pink ink follows your fingertips |
| Spread the fingers | Pen up — nothing is written while the fingers are apart (a ring shows where the pen is) |
| Pause ~0.5 s after a letter | The letter is recognised and typed, then write the next one (whole words letter by letter) |
| Capital (A–Z) or small (a–z) letters | Both are recognised; a plain vertical line is `l` (a lone `l` becomes the word "I"), a line with a dot is `i` |
| Letters with several strokes (T, H, E, …) | Lift the pen between strokes without pausing |
| Straight line left → right / right → left | Space / delete one letter |
| Quick tap with the fingers together | `.` |

**Hand keyboard** (optional: press `m` on desktop, or open the web app with `?handkeys`) — nothing is drawn on screen: touch a finger joint with the
thumb of the same hand and hold it there for a moment (~0.15 s) to type its letter; lift the thumb before the next
letter. To see the letters on the joints while learning the layout, press `k` (desktop) or open the web app with `?labels`.

| Target | Left hand | Right hand |
|---|---|---|
| Index tip / DIP / PIP | a / b / c | m / n / o |
| Middle tip / DIP / PIP | d / e / f | p / q / r |
| Ring tip / DIP / PIP | g / h / i | s / t / u |
| Pinky tip / DIP / PIP | j / k / l | v / w / x |
| Index knuckle | space (SPC) | space (SPC) |
| Pinky knuckle | `,` | `.` |
| Palm centre | **DEL** — deletes one letter, hold to keep deleting | same |

**On-screen Air Keyboard** (optional: press `m` on desktop, or open the web app with `?keyboard`) — point the
index fingertip at a key and pinch thumb + index to press it; pinch-and-hold DEL keeps deleting.

Both thumb tips together = **y**, both index tips together = **z**, right index on left palm = **space**.
Sentence starts and a standalone "i" are capitalised automatically.

**Keyboard shortcuts (desktop):** `q` quit · `m` switch mode · `n` notepad · `s` sidebar · `b` backspace · `w` delete word · `c` clear · `d` debug HUD · `1` liquid box · `2` Rubik's cube · `x` scramble cube · `r` reset box (contents, rotation, size and position) · `v` start / stop video recording (saved to `recordings/`)

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more details.
