import sys
import os

# Automatically append local venv site-packages if running with system python
venv_site = os.path.join(os.path.dirname(__file__), "venv", "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
if os.path.exists(venv_site) and venv_site not in sys.path:
    sys.path.insert(0, venv_site)

import cv2
import time
import argparse
import numpy as np
from src.hand_tracker import HandTracker
from src.air_scribble import AirScribble
from src.wireframe_engine import WireframeEngine
from src.ocr_engine import OCREngine
from src.ui_manager import UIManager
from src.knuckle_gesture import KnuckleGestureEngine, extract_xy
from src.config import FRAME_WIDTH, FRAME_HEIGHT, WINDOW_TITLE

def main():
    parser = argparse.ArgumentParser(description="Spatial Vision AR - URANTUNE_WL_OT")
    parser.add_argument("--source", default=0, help="Camera index or IP stream URL")
    args = parser.parse_args()

    source = args.source
    if str(source).isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # Warmup camera frames
    for _ in range(5):
        cap.read()

    tracker = HandTracker()
    scribble = AirScribble()
    wireframe = WireframeEngine()
    ocr = OCREngine()
    ui = UIManager()
    knuckle_engine = KnuckleGestureEngine(cooldown=0.35)

    # Initialize text buffer with default text matching notepad lines
    if not scribble.text_buffer:
        scribble.text_buffer = "Hello World"

    # Default to WRITE mode so Notepad paper lines are active immediately!
    active_mode = "WRITE"
    ui.sidebar_visible = True
    light_on = False

    fps_eval_time = time.time()
    fps = 30

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera Feed Unavailable - Connect Webcam", (300, 360),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # FPS Calculation
        curr_time = time.time()
        fps = int(1.0 / (curr_time - fps_eval_time + 1e-5))
        fps_eval_time = curr_time

        # Step 1: Process Hand Tracking
        landmarks_list, handedness_list, raw_results = tracker.process(frame)

        norm_landmarks = None
        target_landmarks = None

        if landmarks_list:
            target_landmarks = landmarks_list[0]
        if raw_results and raw_results.multi_hand_landmarks:
            norm_landmarks = raw_results.multi_hand_landmarks[0].landmark
            if not target_landmarks:
                target_landmarks = norm_landmarks

        if target_landmarks:
            # Step 2: Fist Gesture Word Erase (✊ mut)
            scribble.text_buffer, erased = knuckle_engine.check_word_erase(
                target_landmarks, scribble.text_buffer
            )
            if erased:
                cv2.putText(frame, "[ ERASED 1 WORD ]", (w // 2 - 120, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            # Step 3: Direct Finger Joint Touch Air Typing
            char, touch_pt = knuckle_engine.detect_finger_joint_typing(target_landmarks)
            if char is not None:
                active_mode = "WRITE"
                ui.sidebar_visible = True
                scribble.text_buffer += char
                print(f"Typed Character: '{char}' -> Current Notepad Buffer: '{scribble.text_buffer}'")

            if touch_pt is not None:
                px, py = int(touch_pt[0]), int(touch_pt[1])
                if px <= 1 and py <= 1:
                    px, py = int(px * w), int(py * h)
                cv2.circle(frame, (px, py), 14, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.putText(frame, f"+ '{char}'", (px + 15, py - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)

        # Step 4: Handle Toolbar Interaction
        active_mode, light_on, quit_signal = ui.check_interaction(
            frame, landmarks_list, norm_landmarks, w, h, active_mode, light_on
        )
        if quit_signal:
            break

        # Step 5: Execute Selected Mode
        if active_mode == "WRITE":
            ui.draw_notepad_card(frame, scribble.text_buffer)
            scribble.update(frame, landmarks_list)
        else:
            wireframe.draw_3d_spatial_mesh(frame, landmarks_list)

        # Step 6: Render Dynamic Sidebar
        ui.draw_right_toolbar(frame, active_mode, light_on)

        # Step 7: Render Top-Left ( 28 FPS ) Capsule Badge
        ui.draw_top_fps_badge(frame, fps)

        cv2.imshow(WINDOW_TITLE, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('c'):
            scribble.clear()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
