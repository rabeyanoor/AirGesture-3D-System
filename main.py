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

    active_mode = "WIREFRAME"
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
        landmarks_list, handedness_list, _ = tracker.process(frame)

        # Step 2: Handle Gesture Triggering & Interaction
        active_mode, light_on, quit_signal = ui.check_interaction(
            frame, landmarks_list, w, h, active_mode, light_on
        )
        if quit_signal:
            break

        # Step 3: Execute Selected Mode
        if active_mode == "WRITE":
            ui.draw_notepad_card(frame, scribble.text_buffer)
            scribble.update(frame, landmarks_list)
            frame = scribble.merge_with_frame(frame)
        else:
            # WIREFRAME Mode: Clean 3D Mesh & Coordinates matching video 0:00 to 0:09
            wireframe.draw_3d_spatial_mesh(frame, landmarks_list)

        # Step 4: Render Dynamic Sidebar (ONLY when sidebar_visible == True)
        ui.draw_right_toolbar(frame, active_mode, light_on)

        # Step 5: Render Top-Left ( 28 FPS ) Capsule Badge
        ui.draw_top_fps_badge(frame, fps)

        cv2.imshow(WINDOW_TITLE, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            ocr_text = ocr.recognize(scribble.canvas)
            if ocr_text:
                scribble.text_buffer = " ".join(ocr_text)
            print("OCR Text Output:", scribble.text_buffer)
        elif key == ord('c'):
            scribble.clear()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
