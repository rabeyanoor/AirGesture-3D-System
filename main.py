"""
Spatial Vision AR - AirGesture 3D System
Main Entry Point

Features:
- Dual Hand 3D Wireframe Box & Single Hand Mesh Polygon with Coordinate Callouts
- Fist Gesture (Delete Last Word) with Debounce Trigger Flag
- Palm Touch / Tap Gesture (Space Bar Insertion)
- Dual-Hand A-Z Virtual Grid Keyboard (Left Hand Group Selector + Right Hand Letter Selector)
- Air Drawing Stroke Character Recognition
- NLP Auto-Capitalization Engine (Sentence start, 'I', formatting)
- Glassmorphic AR Lined Notepad & Animated Right Sidebar
"""

import cv2
import time
import sys
import argparse
import numpy as np

from hand_tracker import HandTracker
from gesture_recognizer import GestureRecognizer
from ar_mesh_3d import ARMesh3D
from air_drawing_ocr import AirDrawingOCR
from auto_capitalizer import AutoCapitalizer
from ar_ui_renderer import ARUIRenderer


def open_webcam(initial_index=0):
    """
    Attempts to open webcam at initial_index with frame warm-up retries.
    If it fails, automatically probes available camera indices (0 to 5).
    """
    indices_to_try = [initial_index] + [i for i in range(6) if i != initial_index]
    for idx in indices_to_try:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            for _ in range(10):
                ret, frame = cap.read()
                if ret and frame is not None:
                    if idx != initial_index:
                        print(f"[INFO] Camera index {initial_index} unavailable. Auto-switched to active camera index {idx}.")
                    return cap, idx
                time.sleep(0.05)
            cap.release()

    return None, initial_index


def main():
    parser = argparse.ArgumentParser(description="Spatial Vision AR - AirGesture 3D System")
    parser.add_argument("--source", type=str, default="0",
                        help="Video source: '0' for Webcam or path to video file (e.g. video_2026-08-28_01-24-08.mp4)")
    args = parser.parse_args()

    # Determine camera index or file path
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source

    if isinstance(source, int):
        cap, active_idx = open_webcam(source)
        if cap is None:
            print(f"Error: Unable to open any webcam (tried camera indices 0-5).")
            sys.exit(1)
        source = active_idx
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Unable to open video source: {args.source}")
            sys.exit(1)

    # Initialize Modules
    tracker = HandTracker(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
    recognizer = GestureRecognizer()
    mesh_renderer = ARMesh3D()
    air_ocr = AirDrawingOCR()
    ui_renderer = ARUIRenderer()

    # Initial State Variables
    text_buffer = ""
    active_mode = "PHALANX KEYBOARD"  # Modes: "PHALANX KEYBOARD", "SPATIAL AR 3D", "GRID KEYBOARD", "AIR DRAW"
    sidebar_open = False
    active_gesture = "IDLE"

    prev_time = time.time()

    print("=========================================================")
    print(" Spatial Vision AR - AirGesture 3D System Started")
    print(" Press 'q' to Quit | 'm' to Switch Modes | 'c' to Clear")
    print("=========================================================")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # Loop video file if playing video file source
            if isinstance(source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        # Horizontal mirror for webcam feed
        if source == 0:
            frame = cv2.flip(frame, 1)

        h, w, c = frame.shape

        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-5)
        prev_time = curr_time

        # Process Hands
        all_hands, raw_left, raw_right = tracker.process(frame)

        # Filter hands for single-hand gestures
        hands_data = [h for h in all_hands if tracker.is_palm_facing_camera(h)]
        left_hand = raw_left
        right_hand = raw_right
        hand_count = len(all_hands)

        active_gesture = "IDLE"

        # -------------------------------------------------------------
        # 1. AR 3D Mesh & Spatial Wireframe Rendering
        # -------------------------------------------------------------
        if len(all_hands) >= 2:
            frame = mesh_renderer.draw_dual_hand_3d_wireframe(frame, all_hands[0], all_hands[1])
        elif len(all_hands) == 1:
            frame = mesh_renderer.draw_fingertip_polygon(frame, all_hands[0])

        # -------------------------------------------------------------
        # 2. Gesture Logic Processing
        # -------------------------------------------------------------
        # A. Fist Gesture Display Status
        fist_detected = (left_hand and recognizer.is_fist(left_hand)) or (right_hand and recognizer.is_fist(right_hand))
        if fist_detected:
            active_gesture = "FIST GESTURE"

        # B. Palm Touch / Press for Space Insertion
        if left_hand and right_hand:
            is_touch, touch_pt = recognizer.check_palm_touch(left_hand, right_hand, w, h)
            if is_touch:
                if not recognizer.palm_touch_triggered:
                    text_buffer += " "
                    text_buffer = AutoCapitalizer.format_text(text_buffer)
                    recognizer.palm_touch_triggered = True
                active_gesture = "PALM TOUCH (SPACE)"
                # Visual feedback glowing circle
                cv2.circle(frame, touch_pt, 18, (0, 255, 0), cv2.FILLED, cv2.LINE_AA)
                cv2.circle(frame, touch_pt, 24, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                recognizer.palm_touch_triggered = False

        # C. Right Boundary Entry / Motion (Sidebar Toggle & Hold Trigger)
        sidebar_open = recognizer.update_sidebar_state(hands_data, w, sidebar_open)
        sidebar_open, click_action = recognizer.check_sidebar_clicks(hands_data, ui_renderer, sidebar_open)
        if click_action == "EXIT_APP":
            print("--> Power Button Clicked: Exiting Application.")
            break
        elif click_action:
            active_gesture = click_action
        elif sidebar_open and active_gesture == "IDLE":
            active_gesture = "SIDEBAR OPEN"

        # D. Finger Phalanx Keyboard (Chorded Joint Mapping A-Z)
        if active_mode in ["PHALANX KEYBOARD", "SPATIAL AR 3D"]:
            phalanx_char = recognizer.detect_phalanx_keyboard(left_hand, right_hand, w, h)
            if phalanx_char == '<DELETE_WORD>':
                text_buffer = text_buffer.rstrip()
                if ' ' in text_buffer:
                    text_buffer = text_buffer.rsplit(' ', 1)[0] + ' '
                else:
                    text_buffer = ""
                active_gesture = "DELETED WORD"
            elif phalanx_char:
                text_buffer += phalanx_char
                text_buffer = AutoCapitalizer.process_notepad_text(text_buffer)
                active_gesture = f"PHALANX KEY '{phalanx_char.upper()}'"

        # E. Dual-Hand Grid Keypad Matrix Selection (A-Z Virtual Tap)
        elif left_hand and right_hand and active_mode == "GRID KEYBOARD":
            sel_group, typed_char = recognizer.process_grid_keypad(left_hand, right_hand, w, h)
            if sel_group:
                active_gesture = f"KEYPAD ({sel_group})"
            if typed_char:
                text_buffer += typed_char.lower()
                text_buffer = AutoCapitalizer.process_notepad_text(text_buffer)
                active_gesture = f"TYPED '{typed_char}'"

        # E. Air Drawing & Character Recognition
        if right_hand and active_mode == "AIR DRAW":
            if recognizer.is_index_pointing(right_hand):
                r_index_tip = right_hand['px'][8]
                air_ocr.add_point((r_index_tip[0], r_index_tip[1]))
                active_gesture = "AIR DRAWING"
            else:
                if air_ocr.drawing and time.time() - air_ocr.last_point_time > air_ocr.pause_threshold:
                    drawn_char = air_ocr.recognize_stroke()
                    if drawn_char:
                        text_buffer += drawn_char.lower()
                        text_buffer = AutoCapitalizer.format_text(text_buffer)
                        active_gesture = f"RECOGNIZED '{drawn_char}'"

        # Draw Air Drawing trail if points exist
        frame = air_ocr.draw_path(frame)

        # -------------------------------------------------------------
        # 3. UI Overlays Rendering
        # -------------------------------------------------------------
        # Apply Screen Brightness / Dark Theme Filter if toggled on via Tile 1 (Bulb)
        frame = ui_renderer.apply_theme_filter(frame)

        # Format text buffer automatically using NLP process_notepad_text rules
        text_buffer = AutoCapitalizer.process_notepad_text(text_buffer)

        # Draw AR Virtual Lined Notepad Overlay
        frame = ui_renderer.draw_notepad_overlay(frame, text_buffer)

        # Draw Animated Right Sidebar
        frame = ui_renderer.draw_sidebar(frame, sidebar_open)

        # Draw Top HUD Status Banner
        frame = ui_renderer.draw_top_hud(frame, fps, active_mode, hand_count, active_gesture)

        # Show Output Window
        cv2.imshow("Camera", frame)

        # Keyboard Controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            sidebar_open = not sidebar_open
            print(f"--> Sidebar toggled: {sidebar_open}")
        elif key == ord('n'):
            ui_renderer.show_notepad = not ui_renderer.show_notepad
            print(f"--> AR Notepad toggled: {ui_renderer.show_notepad}")
        elif key == ord('m'):
            # Cycle Modes
            modes = ["PHALANX KEYBOARD", "SPATIAL AR 3D", "GRID KEYBOARD", "AIR DRAW"]
            cur_idx = modes.index(active_mode) if active_mode in modes else 0
            active_mode = modes[(cur_idx + 1) % len(modes)]
            print(f"--> Switched Active Mode to: {active_mode}")
        elif key in [8, ord('b')]:  # ASCII Backspace key (8) or 'b'
            text_buffer = text_buffer[:-1]
            print("--> Backspace: Deleted last character.")
        elif key == ord('w'):  # Delete Word key
            text_buffer = text_buffer.rstrip()
            if ' ' in text_buffer:
                text_buffer = text_buffer.rsplit(' ', 1)[0] + ' '
            else:
                text_buffer = ""
            print("--> Word Delete: Deleted last word.")
        elif key == ord('c'):
            text_buffer = ""
            air_ocr.clear()
            print("--> Notepad and Canvas Cleared.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
