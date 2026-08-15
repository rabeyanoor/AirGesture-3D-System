import cv2
import time
import argparse
from src.hand_tracker import HandTracker
from src.air_scribble import AirScribble
from src.wireframe_engine import WireframeEngine
from src.ocr_engine import OCREngine
from src.ui_manager import UIManager
from src.config import FRAME_WIDTH, FRAME_HEIGHT

def main():
    parser = argparse.ArgumentParser(description="Spatial Vision AR System")
    parser.add_argument("--source", default=0, help="Camera index or stream URL (e.g., 0, 1, or http://ip:port/video)")
    args = parser.parse_args()

    # Parse camera input source
    source = args.source
    if str(source).isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # Initialize Core Engines
    tracker = HandTracker()
    scribble = AirScribble()
    wireframe = WireframeEngine()
    ocr = OCREngine()
    ui = UIManager()

    active_mode = "WRITE"
    recognized_text = []
    text_display_timer = 0

    fps_eval_time = time.time()
    fps = 0

    print("=========================================")
    print("  Spatial Vision AR - Interactive System ")
    print("=========================================")
    print("Controls:")
    print("  - Pinch (Thumb + Index Tip): Draw in Air / Interact")
    print("  - Hover over UI buttons on the right to switch modes")
    print("  - Press 's' on keyboard: Trigger OCR text recognition")
    print("  - Press 'c' on keyboard: Clear canvas")
    print("  - Press 'q' or ESC: Quit application")
    print("=========================================")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Notice: Camera stream unavailable or ended.")
            break

        # Flip horizontally for natural selfie view
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Measure FPS
        current_time = time.time()
        fps = int(1.0 / (current_time - fps_eval_time + 1e-5))
        fps_eval_time = current_time

        # Step 1: Process Hand Landmarks
        landmarks_list, handedness_list, _ = tracker.process(frame)

        # Step 2: Handle UI Interactions (Button Hover & Dwell)
        active_mode, trigger_ocr = ui.check_hover_interaction(
            frame, landmarks_list, w, active_mode, scribble
        )

        # Step 3: Execute Active Mode Logic
        if active_mode == "WRITE":
            scribble.update(frame, landmarks_list)
        elif active_mode == "WIREFRAME":
            wireframe.draw_3d_spatial_mesh(frame, landmarks_list)

        # Merge drawn air canvas onto video feed
        frame = scribble.merge_with_frame(frame)

        # Draw UI Overlay Panels
        ui.draw_ui(frame, active_mode, scribble.color, scribble.is_erasing)

        # Handle OCR Recognition Trigger
        if trigger_ocr:
            recognized_text = ocr.recognize(scribble.canvas)
            text_display_timer = time.time()
            print("OCR Recognition Output:", recognized_text)

        # Display Recognized Text HUD Banner
        if recognized_text and (time.time() - text_display_timer < 5.0):
            banner_str = "Recognized: " + " ".join(recognized_text)
            cv2.rectangle(frame, (20, h - 60), (w - 20, h - 15), (0, 0, 0), -1)
            cv2.rectangle(frame, (20, h - 60), (w - 20, h - 15), (0, 255, 0), 2)
            cv2.putText(frame, banner_str, (35, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Display FPS in bottom corner
        cv2.putText(frame, f"FPS: {fps}", (20, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # Display Viewport Window
        cv2.imshow("Spatial Vision AR Interface", frame)

        # Keyboard Shortcut Triggers
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            recognized_text = ocr.recognize(scribble.canvas)
            text_display_timer = time.time()
            print("OCR Text Recognition Output:", recognized_text)
        elif key == ord('c'):
            scribble.clear()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
