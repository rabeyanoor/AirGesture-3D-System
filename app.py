"""
Spatial Vision AR - Hugging Face Space Entry Point
Gradio Real-Time Webcam / Video Processor for Touchless Phalanx Typing & AR 3D Hand Mesh.
"""

import os
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import cv2
import numpy as np
import gradio as gr
import time

from hand_tracker import HandTracker
from gesture_recognizer import GestureRecognizer
from ar_mesh_3d import ARMesh3D
from ar_ui_renderer import ARUIRenderer
from auto_capitalizer import AutoCapitalizer

# Initialize Core AR Modules
tracker = HandTracker(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
recognizer = GestureRecognizer()
mesh_renderer = ARMesh3D()
ui_renderer = ARUIRenderer()

# Global state across frames
state = {
    'text_buffer': "",
    'sidebar_open': False,
    'last_time': time.time(),
    'active_mode': "PHALANX KEYBOARD"
}


def process_frame(frame, mode, show_notepad_check):
    if frame is None:
        return None, ""

    # Convert RGB (Gradio format) to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    h, w, _ = frame_bgr.shape

    # Calculate FPS
    curr_time = time.time()
    fps = 1.0 / (curr_time - state['last_time'] + 1e-5)
    state['last_time'] = curr_time

    # Process Hand Landmarks
    all_hands, raw_left, raw_right = tracker.process(frame_bgr)
    hands_data = [h_data for h_data in all_hands if tracker.is_palm_facing_camera(h_data)]

    active_gesture = "IDLE"
    ui_renderer.show_notepad = show_notepad_check

    # 1. 3D Wireframe Overlay
    if len(all_hands) >= 2:
        frame_bgr = mesh_renderer.draw_dual_hand_3d_wireframe(frame_bgr, all_hands[0], all_hands[1])
    elif len(all_hands) == 1:
        frame_bgr = mesh_renderer.draw_fingertip_polygon(frame_bgr, all_hands[0])

    # 2. Gesture Logic
    fist_detected = (raw_left and recognizer.is_fist(raw_left)) or (raw_right and recognizer.is_fist(raw_right))
    if fist_detected:
        active_gesture = "FIST GESTURE"

    # Palm Touch (Space)
    if raw_left and raw_right:
        is_touch, touch_pt = recognizer.check_palm_touch(raw_left, raw_right, w, h)
        if is_touch:
            if not recognizer.palm_touch_triggered:
                state['text_buffer'] += " "
                state['text_buffer'] = AutoCapitalizer.format_text(state['text_buffer'])
                recognizer.palm_touch_triggered = True
            active_gesture = "PALM TOUCH (SPACE)"
            cv2.circle(frame_bgr, touch_pt, 18, (0, 255, 0), cv2.FILLED, cv2.LINE_AA)
        else:
            recognizer.palm_touch_triggered = False

    # Sidebar Click check
    state['sidebar_open'] = recognizer.update_sidebar_state(hands_data, w, state['sidebar_open'])
    state['sidebar_open'], click_action = recognizer.check_sidebar_clicks(hands_data, ui_renderer, state['sidebar_open'])

    if click_action == "NOTEPAD_TOGGLED":
        ui_renderer.show_notepad = not ui_renderer.show_notepad
        active_gesture = "NOTEPAD TOGGLED"

    # Typing
    if mode in ["PHALANX KEYBOARD", "SPATIAL AR 3D"] and ui_renderer.show_notepad:
        phalanx_char = recognizer.detect_phalanx_keyboard(raw_left, raw_right, w, h)
        if phalanx_char == '<DELETE_WORD>':
            state['text_buffer'] = state['text_buffer'].rstrip()
            if ' ' in state['text_buffer']:
                state['text_buffer'] = state['text_buffer'].rsplit(' ', 1)[0] + ' '
            else:
                state['text_buffer'] = ""
            active_gesture = "DELETED WORD"
        elif phalanx_char:
            state['text_buffer'] += phalanx_char
            state['text_buffer'] = AutoCapitalizer.process_notepad_text(state['text_buffer'])
            active_gesture = f"PHALANX KEY '{phalanx_char.upper()}'"

    # Render UI
    frame_bgr = ui_renderer.apply_theme_filter(frame_bgr)
    frame_bgr = ui_renderer.draw_notepad_overlay(frame_bgr, state['text_buffer'])
    frame_bgr = ui_renderer.draw_sidebar(frame_bgr, state['sidebar_open'])
    frame_bgr = ui_renderer.draw_top_hud(frame_bgr, fps, mode, len(all_hands), active_gesture)

    # Convert back to RGB for Gradio
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return frame_rgb, state['text_buffer']


def clear_buffer():
    state['text_buffer'] = ""
    return ""


# Build Gradio UI
with gr.Blocks(title="Spatial Vision AR - AirGesture 3D System") as demo:
    gr.Markdown("# 🖐️ Spatial Vision AR - AirGesture 3D Typing System")
    gr.Markdown(
        "Touchless 3D Phalanx Keyboard & Gesture-Driven AR Notepad powered by MediaPipe and OpenCV."
    )

    with gr.Row():
        with gr.Column(scale=2):
            input_image = gr.Image(sources=["webcam"], streaming=True, label="Live Webcam Feed")
        with gr.Column(scale=1):
            mode_dropdown = gr.Dropdown(
                choices=["PHALANX KEYBOARD", "SPATIAL AR 3D", "GRID KEYBOARD"],
                value="PHALANX KEYBOARD",
                label="Active Operating Mode"
            )
            notepad_toggle = gr.Checkbox(value=False, label="Show AR Notepad Overlay")
            typed_output = gr.Textbox(label="Typed Output Stream", value="", interactive=False)
            clear_btn = gr.Button("Clear Typed Buffer")

    input_image.stream(
        fn=process_frame,
        inputs=[input_image, mode_dropdown, notepad_toggle],
        outputs=[input_image, typed_output]
    )

    clear_btn.click(fn=clear_buffer, outputs=[typed_output])

if __name__ == "__main__":
    demo.launch()
