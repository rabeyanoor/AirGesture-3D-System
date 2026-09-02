"""
Test script to process video_2026-08-28_01-24-08.mp4 and export demo frames.
"""

import cv2
import time
from hand_tracker import HandTracker
from gesture_recognizer import GestureRecognizer
from ar_mesh_3d import ARMesh3D
from air_drawing_ocr import AirDrawingOCR
from auto_capitalizer import AutoCapitalizer
from ar_ui_renderer import ARUIRenderer


def test_video_processing():
    cap = cv2.VideoCapture("video_2026-08-28_01-24-08.mp4")
    if not cap.isOpened():
        print("Error opening video")
        return

    tracker = HandTracker()
    recognizer = GestureRecognizer()
    mesh_renderer = ARMesh3D()
    air_ocr = AirDrawingOCR()
    ui_renderer = ARUIRenderer()

    text_buffer = "Hello My name is"
    sidebar_open = False

    frame_idx = 0
    saved_count = 0

    print("Processing video_2026-08-28_01-24-08.mp4...")

    while cap.isOpened() and frame_idx < 150:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        h, w, c = frame.shape

        hands_data, left_hand, right_hand = tracker.process(frame)
        hand_count = len(hands_data)

        active_gesture = "IDLE"

        # 3D AR Mesh & Wireframe
        if hand_count == 1:
            frame = mesh_renderer.draw_fingertip_polygon(frame, hands_data[0])
        elif hand_count >= 2:
            frame = mesh_renderer.draw_fingertip_polygon(frame, left_hand)
            frame = mesh_renderer.draw_fingertip_polygon(frame, right_hand)
            frame = mesh_renderer.draw_dual_hand_3d_wireframe(frame, left_hand, right_hand)

        # Sidebar trigger check
        sidebar_open = recognizer.check_sidebar_trigger(hands_data, w)
        if sidebar_open:
            active_gesture = "SIDEBAR OPEN"

        # Render Overlays
        formatted_text = AutoCapitalizer.format_text(text_buffer)
        frame = ui_renderer.draw_notepad_overlay(frame, formatted_text)
        frame = ui_renderer.draw_sidebar(frame, sidebar_open)
        frame = ui_renderer.draw_top_hud(frame, 30.0, "SPATIAL AR 3D", hand_count, active_gesture)

        # Save sample frame snapshots for validation
        if frame_idx in [10, 40, 70, 110, 140]:
            out_name = f"/tmp/output_frame_{frame_idx:03d}.jpg"
            cv2.imwrite(out_name, frame)
            saved_count += 1
            print(f"Saved snapshot: {out_name}")

    cap.release()
    print(f"Successfully processed {frame_idx} frames, saved {saved_count} snapshot images!")


if __name__ == "__main__":
    test_video_processing()
