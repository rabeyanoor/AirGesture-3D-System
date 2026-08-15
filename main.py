import sys
import os

venv_site = os.path.join(os.path.dirname(__file__), "venv", "lib",
                         f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
if os.path.exists(venv_site) and venv_site not in sys.path:
    sys.path.insert(0, venv_site)

import cv2
import time
import argparse
import numpy as np

from src.hand_tracker    import HandTracker
from src.wireframe_engine import WireframeEngine
from src.ui_manager      import UIManager
from src.air_typer       import AirTyper
from src.ai_solver       import AISolver
from src.config          import FRAME_WIDTH, FRAME_HEIGHT, WINDOW_TITLE


def open_camera(preferred=0):
    """Auto-detects active webcam across indices [0, 1, 2, 4] and V4L2 backends."""
    if str(preferred).isdigit():
        src_int = int(preferred)
        for idx in [src_int, 0, 1, 2, 4]:
            for backend in [cv2.CAP_V4L2, cv2.CAP_ANY]:
                try:
                    cap = cv2.VideoCapture(idx, backend)
                    if cap.isOpened():
                        ret, f = cap.read()
                        if ret and f is not None:
                            print(f"[CAM OK] Camera Index {idx} connected.")
                            return cap
                        cap.release()
                except Exception:
                    pass
    return None


def main():
    parser = argparse.ArgumentParser(description="Spatial Vision AR")
    parser.add_argument("--source", default=0)
    args = parser.parse_args()

    cap = open_camera(args.source)
    if cap is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    tracker   = HandTracker(max_hands=2)
    wireframe = WireframeEngine()
    ui        = UIManager()
    typer     = AirTyper()
    ai        = AISolver()

    active_mode = "WIREFRAME"
    ui.sidebar_visible = False
    light_on    = False
    text_buffer = ""
    ai_answer   = ""
    shift_mode  = False

    fps_time = time.time()
    last_cam_retry = 0
    fps = 30

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

    while True:
        ret = False
        frame = None

        if cap is not None and cap.isOpened():
            ret, frame = cap.read()

        if not ret or frame is None:
            if cap is not None:
                cap.release()
                cap = None

            # Attempt auto-reconnect every 2 seconds
            if time.time() - last_cam_retry > 2.0:
                last_cam_retry = time.time()
                cap = open_camera(args.source)
                if cap is not None:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(frame, "Webcam Feed Unavailable - Please Connect Webcam", (240, 360),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
        else:
            frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape

        # FPS
        now = time.time()
        fps = int(1.0 / (now - fps_time + 1e-5))
        fps_time = now

        # ── Hand Tracking ──────────────────────────────────────────────
        lm_list, hand_labels, raw = tracker.process(frame)

        left_px,  left_norm  = tracker.get_hand_by_label(lm_list, hand_labels, raw, "Left")
        right_px, right_norm = tracker.get_hand_by_label(lm_list, hand_labels, raw, "Right")

        # ── Two-Hand Air Typing (only in WRITE mode) ───────────────────
        if active_mode == "WRITE" and left_norm and right_norm:

            # Shift toggle (V-gesture on right hand)
            shift_mode = typer.check_shift(right_norm)

            # Word erase (right fist)
            text_buffer, erased = typer.check_word_erase(right_norm, text_buffer)
            if erased:
                ai_answer = ""
                cv2.putText(frame, "[ WORD ERASED ]", (w//2-110, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 240), 2, cv2.LINE_AA)

            # Character detection
            char, touch_pt = typer.detect(left_norm, right_norm)
            if char is not None:
                if char == '\b':
                    text_buffer = text_buffer[:-1]
                else:
                    text_buffer += char
                ai_answer = ""
                print(f"[TYPE] '{char}'  buffer='{text_buffer}'")

            # Touch point indicator
            if touch_pt is not None:
                px = int(touch_pt[0] * w)
                py = int(touch_pt[1] * h)
                cv2.circle(frame, (px, py), 16, (0, 255, 255), -1, cv2.LINE_AA)
                lbl = char if char and char != ' ' else 'SPC'
                cv2.putText(frame, f"'{lbl}'", (px+18, py-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)

            # Visual keyboard targets on left hand
            ui.draw_left_hand_targets(frame, left_px, shift_mode)

        # ── Sidebar interaction ────────────────────────────────────────
        sidebar_lm_list = lm_list
        sidebar_norm = right_norm if right_norm else (left_norm if left_norm else None)
        active_mode, light_on, _ = ui.check_interaction(
            frame, sidebar_lm_list, sidebar_norm, w, h, active_mode, light_on
        )

        # ── AI Solve ───────────────────────────────────────────────────
        if text_buffer.endswith('?') and not ai_answer:
            ai_answer = ai.solve(text_buffer)

        # ── Render Mode ────────────────────────────────────────────────
        if active_mode == "WRITE":
            ui.draw_notepad_card(frame, text_buffer, ai_answer)
        else:
            wireframe.draw_3d_spatial_mesh(frame, lm_list)

        ui.draw_right_toolbar(frame, active_mode, light_on)
        ui.draw_top_fps_badge(frame, fps)
        ui.draw_shift_indicator(frame, shift_mode)

        cv2.imshow(WINDOW_TITLE, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('c'):
            text_buffer = ""
            ai_answer   = ""

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
