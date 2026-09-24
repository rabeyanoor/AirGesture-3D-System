"""
Offline test runner: pushes a recorded video through the full AR pipeline and writes
an annotated output video plus a few sample frames.

Usage:
    python run_video_test.py [input.mp4] [--out output_demo.mp4] [--frames 0] [--notepad]
"""

import argparse
import glob
import os
import time

import cv2

from pipeline import SpatialVisionPipeline


def default_input():
    videos = sorted(glob.glob("video_*.mp4"))
    return videos[-1] if videos else "input.mp4"


def test_video_processing():
    parser = argparse.ArgumentParser(description="Process a video file through the AR pipeline")
    parser.add_argument("input", nargs="?", default=default_input())
    parser.add_argument("--out", default="output_demo.mp4")
    parser.add_argument("--frames", type=int, default=0, help="Max frames to process (0 = all)")
    parser.add_argument("--notepad", action="store_true", help="Start with the notepad open")
    parser.add_argument("--samples-dir", default="demo_frames")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error opening video: {args.input}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    limit = args.frames or total

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    os.makedirs(args.samples_dir, exist_ok=True)

    pipeline = SpatialVisionPipeline(debug_hud=True)
    pipeline.ui_renderer.show_notepad = args.notepad

    print(f"Processing {args.input} ({w}x{h}, {total} frames @ {fps:.1f} fps)...")
    sample_every = max(1, limit // 8)
    start = time.time()
    frame_idx = 0
    gestures = {}

    while frame_idx < limit:
        ret, frame = cap.read()
        if not ret:
            break

        out = pipeline.process(frame)
        writer.write(out)
        if pipeline.active_gesture != "IDLE":
            gestures[pipeline.active_gesture] = gestures.get(pipeline.active_gesture, 0) + 1

        if frame_idx % sample_every == 0:
            cv2.imwrite(os.path.join(args.samples_dir, f"frame_{frame_idx:04d}.jpg"), out)
        frame_idx += 1

    cap.release()
    writer.release()

    elapsed = time.time() - start
    print(f"Done: {frame_idx} frames in {elapsed:.1f}s ({frame_idx / max(elapsed, 1e-5):.1f} fps)")
    print(f"Output video: {args.out} | Sample frames: {args.samples_dir}/")
    print(f"Final notepad text: {pipeline.text_buffer!r}")
    for g, n in sorted(gestures.items(), key=lambda kv: -kv[1]):
        print(f"  {g}: {n} frames")


if __name__ == "__main__":
    test_video_processing()
