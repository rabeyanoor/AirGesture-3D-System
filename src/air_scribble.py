import cv2
import time
import numpy as np

class AirScribble:
    def __init__(self, pinch_thresh=35):
        self.canvas = None
        self.prev_x = 0
        self.prev_y = 0
        self.pinch_thresh = pinch_thresh
        self.text_buffer = ""
        self.is_pinching = False

    def update(self, frame, landmarks_list):
        """
        No ink drawing lines rendered on screen. Shows sleek pointer dot for clean writing.
        """
        h, w, _ = frame.shape
        if self.canvas is None:
            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)

        if not landmarks_list:
            self.prev_x, self.prev_y = 0, 0
            self.is_pinching = False
            return

        hand = landmarks_list[0]
        index_tip = hand[8][:2]
        thumb_tip = hand[4][:2]

        distance = np.hypot(index_tip[0] - thumb_tip[0], index_tip[1] - thumb_tip[1])

        if distance < self.pinch_thresh:
            self.is_pinching = True
            # Write active indicator dot (red)
            cv2.circle(frame, index_tip, 6, (0, 0, 255), -1, cv2.LINE_AA)
        else:
            self.is_pinching = False
            # Hover cursor dot (green) on index tip when hand is open
            cv2.circle(frame, index_tip, 5, (0, 255, 0), -1, cv2.LINE_AA)

    def clear(self):
        if self.canvas is not None:
            self.canvas.fill(0)
        self.text_buffer = ""

    def merge_with_frame(self, frame):
        # Return frame cleanly without drawing lines overlay
        return frame
