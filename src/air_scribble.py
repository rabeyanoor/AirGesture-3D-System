import cv2
import numpy as np
from src.config import PINCH_THRESHOLD, COLOR_BLACK

class AirScribble:
    def __init__(self, pinch_thresh=PINCH_THRESHOLD):
        self.canvas = None
        self.prev_x = 0
        self.prev_y = 0
        self.pinch_thresh = pinch_thresh

    def update(self, frame, landmarks_list):
        """Processes finger movements and draws smooth dark ink strokes on notepad."""
        h, w, _ = frame.shape
        if self.canvas is None:
            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)

        if not landmarks_list:
            self.prev_x, self.prev_y = 0, 0
            return

        hand = landmarks_list[0]
        index_tip = hand[8][:2]
        thumb_tip = hand[4][:2]

        distance = np.hypot(index_tip[0] - thumb_tip[0], index_tip[1] - thumb_tip[1])

        # Active writing when Index & Thumb pinch close together or index tip trace
        if distance < self.pinch_thresh or distance < 60:
            if self.prev_x == 0 and self.prev_y == 0:
                self.prev_x, self.prev_y = index_tip

            # Draw ink line on canvas
            cv2.line(self.canvas, (self.prev_x, self.prev_y), index_tip, (30, 30, 30), 3, cv2.LINE_AA)
            self.prev_x, self.prev_y = index_tip

            # Visual red/cyan pointer dot on fingertip
            cv2.circle(frame, index_tip, 4, (0, 0, 255), -1, cv2.LINE_AA)
        else:
            self.prev_x, self.prev_y = 0, 0

    def clear(self):
        if self.canvas is not None:
            self.canvas.fill(0)

    def merge_with_frame(self, frame):
        if self.canvas is None:
            return frame

        gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, inv_canvas = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY_INV)
        inv_canvas = cv2.cvtColor(inv_canvas, cv2.COLOR_GRAY2BGR)

        frame = cv2.bitwise_and(frame, inv_canvas)
        frame = cv2.bitwise_or(frame, self.canvas)
        return frame
