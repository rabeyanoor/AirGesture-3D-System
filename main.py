"""
Spatial Vision AR - AirGesture 3D System
Main Entry Point (desktop window)

Features:
- Single-hand frosted-glass fingertip fan & dual-hand 3D glass volume with coordinate callouts
- Right swipe / right-edge entry reveals the sidebar (Bulb = dark theme, Document = notepad,
  Liquid box, Rubik's cube, Power = clear notepad + close boxes + hide sidebar)
- Hand-controlled 3D boxes: pinch-drag to rotate, index-swipe to turn Rubik's cube layers
- Finger Phalanx Keyboard (thumb touches finger joints: A-Z, space, comma, period, delete word)
- Palm Touch (right index on left palm) inserts a space
- Dual-Hand A-Z Grid Keypad and Air Drawing modes
- NLP Auto-Capitalization Engine
"""

import argparse
import os
import sys
import time
from datetime import datetime

import cv2

from pipeline import SpatialVisionPipeline

WINDOW_NAME = "URANTUNE_WL_OT"
RECORD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")


def open_webcam(initial_index=0, width=1280, height=720):
    """
    Attempts to open webcam at initial_index with frame warm-up retries.
    If it fails, automatically probes available camera indices (0 to 5).
    """
    indices_to_try = [initial_index] + [i for i in range(6) if i != initial_index]
    for idx in indices_to_try:
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2) if sys.platform.startswith("linux") else cv2.VideoCapture(idx)
        if cap.isOpened():
            # MJPG gives full HD-ish resolution at 30 fps on most webcams (raw YUYV is often 640x480 / 10 fps)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always the newest frame, no lag
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
                        help="Video source: camera index (e.g. '0') or path to a video file")
    parser.add_argument("--width", type=int, default=1280, help="Requested webcam capture width")
    parser.add_argument("--height", type=int, default=720, help="Requested webcam capture height")
    parser.add_argument("--no-mirror", action="store_true", help="Do not mirror the webcam feed")
    parser.add_argument("--debug", action="store_true", help="Show mode / hand count / gesture status line")
    args = parser.parse_args()

    is_camera = args.source.isdigit()
    if is_camera:
        cap, _ = open_webcam(int(args.source), args.width, args.height)
        if cap is None:
            print("Error: Unable to open any webcam (tried camera indices 0-5).")
            sys.exit(1)
    else:
        cap = cv2.VideoCapture(args.source)
        if not cap.isOpened():
            print(f"Error: Unable to open video source: {args.source}")
            sys.exit(1)

    # Live camera: hand tracking runs on its own thread so the picture stays at camera speed
    pipeline = SpatialVisionPipeline(debug_hud=args.debug, async_tracking=is_camera)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    print("=========================================================")
    print(" Spatial Vision AR - AirGesture 3D System Started")
    print(" q: Quit | m: Switch Mode | n: Notepad | s: Sidebar")
    print(" b: Backspace | w: Delete Word | c: Clear | d: Debug HUD")
    print(" 1: Liquid Box | 2: Rubik's Cube | x: Scramble | r: Reset Box")
    print(" v: Start / stop video recording (saved in recordings/) | k: Show / hide letters on fingers")
    print("=========================================================")

    writer, record_path = None, None

    def stop_recording():
        nonlocal writer
        if writer is not None:
            writer.release()
            writer = None
            print(f"--> Recording saved: {record_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if not is_camera:
                # Loop video files
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            # Webcam dropped (e.g. USB reset): try to reconnect for up to 10 s instead of quitting
            print("[WARN] Camera disconnected - reconnecting...")
            cap.release()
            deadline = time.time() + 10
            while time.time() < deadline:
                cap, _ = open_webcam(int(args.source), args.width, args.height)
                if cap is not None:
                    print("[INFO] Camera reconnected.")
                    break
                cv2.waitKey(500)
            if cap is None:
                print("Error: Camera lost and could not reconnect.")
                break
            continue

        # Mirror webcam feed so on-screen motion matches the user's hands
        if is_camera and not args.no_mirror:
            frame = cv2.flip(frame, 1)

        frame = pipeline.process(frame)
        if writer is not None:
            writer.write(frame)
            # REC badge only on screen, not in the saved video
            shown = frame.copy()
            h, w = shown.shape[:2]
            if int(time.time() * 2) % 2 == 0:
                cv2.circle(shown, (w - 150, 34), 9, (40, 40, 230), -1, cv2.LINE_AA)
            cv2.putText(shown, "REC", (w - 132, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 230), 2, cv2.LINE_AA)
            cv2.imshow(WINDOW_NAME, shown)
        else:
            cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break
        elif key == ord('s'):
            pipeline.sidebar_open = not pipeline.sidebar_open
            print(f"--> Sidebar toggled: {pipeline.sidebar_open}")
        elif key == ord('n'):
            pipeline.ui_renderer.show_notepad = not pipeline.ui_renderer.show_notepad
            print(f"--> AR Notepad toggled: {pipeline.ui_renderer.show_notepad}")
        elif key == ord('m'):
            print(f"--> Switched Active Mode to: {pipeline.cycle_mode()}")
        elif key == ord('d'):
            pipeline.debug_hud = not pipeline.debug_hud
        elif key == ord('k'):
            pipeline.show_key_labels = not pipeline.show_key_labels
            print(f"--> Letters on fingers: {'shown' if pipeline.show_key_labels else 'hidden'}")
        elif key in (8, ord('b')):  # Backspace or 'b'
            pipeline.backspace()
        elif key == ord('w'):
            pipeline.delete_last_word()
        elif key == ord('1'):
            pipeline.toggle_box('liquid')
        elif key == ord('2'):
            pipeline.toggle_box('rubik')
        elif key == ord('x'):
            pipeline.rubik_cube.scramble()
        elif key == ord('r'):
            box = pipeline.active_box()
            if box is pipeline.rubik_cube:
                box.reset_cube()
            elif box is pipeline.liquid_box:
                box.reset_liquid()
            if box is not None:
                box.reset_orientation()
                box.reset_placement()
        elif key == ord('c'):
            pipeline.clear()
            print("--> Notepad and Canvas Cleared.")
        elif key == ord('v'):
            if writer is None:
                os.makedirs(RECORD_DIR, exist_ok=True)
                record_path = os.path.join(RECORD_DIR, f"airgesture-{datetime.now():%Y%m%d-%H%M%S}.mp4")
                fps = min(30.0, max(10.0, pipeline._fps or 20.0))  # match the real processing rate
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(record_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                print(f"--> Recording started ({fps:.0f} fps): {record_path}")
            else:
                stop_recording()

    stop_recording()
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
