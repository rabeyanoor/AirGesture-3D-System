import cv2
import numpy as np
from src.config import PINCH_THRESHOLD, DEFAULT_STROKE_THICKNESS, COLOR_WHITE, COLOR_RED

class AirScribble:
    def __init__(self, thickness=DEFAULT_STROKE_THICKNESS, pinch_thresh=PINCH_THRESHOLD):
        self.canvas = None
        self.prev_x = 0
        self.prev_y = 0
        self.thickness = thickness
        self.color = COLOR_WHITE
        self.pinch_thresh = pinch_thresh
        self.stroke_history = []
        self.is_erasing = False

    def update(self, frame, landmarks_list):
        """Processes finger movement and draws smoothed strokes on canvas."""
        h, w, _ = frame.shape
        if self.canvas is None:
            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)

        if not landmarks_list:
            self.prev_x, self.prev_y = 0, 0
            return

        hand = landmarks_list[0]
        index_tip = hand[8][:2]
        thumb_tip = hand[4][:2]
        middle_tip = hand[12][:2]

        # Calculate distance between Index Tip and Thumb Tip for Pinch detection
        pinch_dist = np.hypot(index_tip[0] - thumb_tip[0], index_tip[1] - thumb_tip[1])
        
        # Calculate distance between Index Tip and Middle Tip for Eraser gesture detection
        eraser_dist = np.hypot(index_tip[0] - middle_tip[0], index_tip[1] - middle_tip[1])

        # Active Drawing condition: Pinch detected (Index & Thumb close together)
        if pinch_dist < self.pinch_thresh:
            if self.prev_x == 0 and self.prev_y == 0:
                self.prev_x, self.prev_y = index_tip

            draw_color = (0, 0, 0) if self.is_erasing else self.color
            draw_thick = self.thickness * 5 if self.is_erasing else self.thickness

            # Draw anti-aliased smooth line segment
            cv2.line(self.canvas, (self.prev_x, self.prev_y), index_tip, draw_color, draw_thick, cv2.LINE_AA)
            self.prev_x, self.prev_y = index_tip

            # Visual Indicator on Pointer Tip
            indicator_color = COLOR_RED if self.is_erasing else self.color
            cv2.circle(frame, index_tip, self.thickness + 2, indicator_color, -1, cv2.LINE_AA)
        else:
            self.prev_x, self.prev_y = 0, 0

    def set_color(self, new_color):
        self.color = new_color
        self.is_erasing = False

    def set_eraser(self, enable=True):
        self.is_erasing = enable

    def set_thickness(self, thickness):
        self.thickness = max(1, min(30, thickness))

    def clear(self):
        if self.canvas is not None:
            self.canvas.fill(0)

    def merge_with_frame(self, frame):
        """Blends written canvas with video frame using alpha thresholding."""
        if self.canvas is None:
            return frame

        gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, inv_canvas = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY_INV)
        inv_canvas = cv2.cvtColor(inv_canvas, cv2.COLOR_GRAY2BGR)

        frame = cv2.bitwise_and(frame, inv_canvas)
        frame = cv2.bitwise_or(frame, self.canvas)
        return frame
