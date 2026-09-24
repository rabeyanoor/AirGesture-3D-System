"""
Spatial Vision AR - Hugging Face Space Entry Point
Gradio Real-Time Webcam / Video Processor for Touchless Phalanx Typing & AR 3D Hand Mesh.
"""

import os
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

try:
    import spaces
    @spaces.GPU
    def _zero_gpu_startup_check():
        pass
    _zero_gpu_startup_check()
except Exception:
    pass

import cv2
import gradio as gr

from pipeline import MODES, SpatialVisionPipeline

# Single shared pipeline (the Space serves one webcam stream at a time)
pipeline = SpatialVisionPipeline()


def process_frame(frame, mode):
    if frame is None:
        return None, pipeline.text_buffer

    pipeline.active_mode = mode
    # Gradio delivers RGB; the pipeline works in BGR
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_bgr = pipeline.process(frame_bgr)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), pipeline.text_buffer


def set_notepad(show):
    pipeline.ui_renderer.show_notepad = show


def set_box(choice):
    pipeline.ui_renderer.show_liquid = choice == "Liquid Box"
    pipeline.ui_renderer.show_rubik = choice == "Rubik's Cube"


def scramble_cube():
    pipeline.rubik_cube.scramble()


def clear_buffer():
    pipeline.clear()
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
                choices=MODES,
                value="AIR WRITE",
                label="Active Operating Mode"
            )
            notepad_toggle = gr.Checkbox(value=False, label="Show AR Notepad Overlay")
            box_radio = gr.Radio(choices=["None", "Liquid Box", "Rubik's Cube"], value="None",
                                 label="3D Box (pinch-drag to rotate, index-swipe turns cube layers)")
            scramble_btn = gr.Button("Scramble Rubik's Cube")
            typed_output = gr.Textbox(label="Typed Output Stream", value="", interactive=False)
            clear_btn = gr.Button("Clear Typed Buffer")

    input_image.stream(
        fn=process_frame,
        inputs=[input_image, mode_dropdown],
        outputs=[input_image, typed_output]
    )

    notepad_toggle.change(fn=set_notepad, inputs=[notepad_toggle])
    box_radio.change(fn=set_box, inputs=[box_radio])
    scramble_btn.click(fn=scramble_cube)

    clear_btn.click(fn=clear_buffer, outputs=[typed_output])

demo.queue()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
