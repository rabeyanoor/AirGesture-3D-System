import cv2
import time
import argparse
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

    tracker = HandTracker()
    scribble = AirScribble()
    wireframe = WireframeEngine()
    ocr = OCREngine()
    ui = UIManager()

    active_mode = "WIREFRAME"
    light_on = False
    recognized_text = []

    fps_eval_time = time.time()
    fps = 30

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # FPS Calculation
        curr_time = time.time()
        fps = int(1.0 / (curr_time - fps_eval_time + 1e-5))
        fps_eval_time = curr_time

        # Step 1: Process Hand Tracking
        landmarks_list, handedness_list, _ = tracker.process(frame)

        # Step 2: Handle Right Toolbar Interaction
        active_mode, light_on, trigger_ocr = ui.check_interaction(
            frame, landmarks_list, w, h, active_mode, light_on, scribble
        )

        if trigger_ocr:
            recognized_text = ocr.recognize(scribble.canvas)

        # Step 3: Mode Execution
        if active_mode == "WRITE":
            ui.draw_notepad_card(frame, recognized_text)
            scribble.update(frame, landmarks_list)
            frame = scribble.merge_with_frame(frame)
        else:
            wireframe.draw_3d_spatial_mesh(frame, landmarks_list)

        # Step 4: Render Right Vertical Toolbar Panel
        ui.draw_right_toolbar(frame, active_mode, light_on)

        # Step 5: Render Top-Left ( 33 FPS ) Badge
        ui.draw_top_fps_badge(frame, fps)

        cv2.imshow(WINDOW_TITLE, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            recognized_text = ocr.recognize(scribble.canvas)
            print("OCR Output:", recognized_text)
        elif key == ord('c'):
            scribble.clear()
            recognized_text = []

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
