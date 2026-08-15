import cv2
from src.hand_tracker import HandTracker
from src.air_scribble import AirScribble
from src.wireframe_engine import WireframeEngine
from src.ocr_engine import OCREngine
from src.ui_manager import UIManager

def main():
    tracker = HandTracker()
    scribble = AirScribble()
    ocr = OCREngine()

    cap = cv2.VideoCapture(0)
    active_mode = "MENU"

    print("Spatial Vision AR System Started")
    print("Press 'q' to exit, 's' to trigger OCR recognition.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        landmarks_list, _ = tracker.process(frame)
        UIManager.draw_virtual_gui(frame)

        mode_change = UIManager.check_interaction(landmarks_list, w, scribble)
        if mode_change:
            active_mode = mode_change

        if active_mode == "WRITE":
            scribble.update(frame, landmarks_list)
        elif active_mode == "WIREFRAME":
            WireframeEngine.draw_3d_mesh(frame, landmarks_list)

        frame = scribble.merge_with_frame(frame)

        cv2.imshow("Spatial AR Interface", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            text_output = ocr.recognize(scribble.canvas)
            print("Recognized Text:")
            for line in text_output:
                print(line)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
