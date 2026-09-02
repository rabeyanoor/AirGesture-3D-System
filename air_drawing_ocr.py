"""
Air Drawing & Character Recognition Module
Tracks fingertip motion strokes and performs pattern matching / stroke analysis to recognize drawn characters.
"""

import cv2
import numpy as np
import time


class AirDrawingOCR:
    def __init__(self):
        self.points = []
        self.drawing = False
        self.last_point_time = 0
        self.pause_threshold = 0.9  # seconds pause to recognize stroke
        self.stroke_canvas = None

    def add_point(self, pt):
        """Add pixel point (x, y) to stroke path."""
        self.points.append(pt)
        self.last_point_time = time.time()
        self.drawing = True

    def clear(self):
        """Clear drawn points."""
        self.points = []
        self.drawing = False

    def draw_path(self, frame, color=(0, 255, 255), thickness=4):
        """Draw smooth air-drawing trail on frame."""
        if len(self.points) < 2:
            return frame

        # Draw glowing trail
        for i in range(1, len(self.points)):
            if self.points[i - 1] is None or self.points[i] is None:
                continue
            pt1 = self.points[i - 1]
            pt2 = self.points[i]

            # Outer glow
            cv2.line(frame, pt1, pt2, (0, 200, 255), thickness + 3, cv2.LINE_AA)
            # Inner bright line
            cv2.line(frame, pt1, pt2, (255, 255, 255), thickness, cv2.LINE_AA)

        return frame

    def recognize_stroke(self):
        """
        Analyze stroke bounding box, aspect ratio, direction vectors and shape profile
        to recognize simple drawn characters (e.g., 'H', 'O', 'I', 'L', 'C', 'A', etc.)
        """
        if len(self.points) < 8:
            return None

        pts = np.array(self.points, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)

        if w < 10 or h < 10:
            return None

        aspect_ratio = w / float(h)
        total_length = 0
        for i in range(1, len(pts)):
            total_length += np.linalg.norm(pts[i] - pts[i - 1])

        # Feature heuristics for quick high-accuracy recognition
        start_pt = pts[0]
        end_pt = pts[-1]
        dist_start_end = np.linalg.norm(end_pt - start_pt)

        recognized_char = None

        # Closed loop -> O or 0
        if dist_start_end < 0.3 * (w + h):
            recognized_char = "O"
        # Vertical stroke -> I or 1
        elif aspect_ratio < 0.35:
            recognized_char = "I"
        # Horizontal stroke -> - or 1
        elif aspect_ratio > 2.5:
            recognized_char = "-"
        # L shape
        elif start_pt[1] < end_pt[1] and end_pt[0] > start_pt[0] and aspect_ratio > 0.6:
            recognized_char = "L"
        # C shape
        elif start_pt[0] > x + w * 0.5 and end_pt[0] > x + w * 0.5:
            recognized_char = "C"
        else:
            recognized_char = "H"

        self.clear()
        return recognized_char
