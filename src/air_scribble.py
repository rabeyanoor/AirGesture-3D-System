import cv2
import numpy as np

class AirScribble:
    def __init__(self, pinch_thresh=35):
        self.canvas = None
        self.prev_x = 0
        self.prev_y = 0
        self.pinch_thresh = pinch_thresh
        self.text_buffer = "Hello, My name is P Kha"
        self.is_pinching = False

    def update(self, frame, landmarks_list):
        """
        Processes hand gestures. Pinching index + thumb triggers air writing / stroke tracing.
        Hovering index finger displays a green cursor without drawing unwanted lines.
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

        # Active Ink Drawing ONLY when Pinching (Index + Thumb touching)
        if distance < self.pinch_thresh:
            self.is_pinching = True
            if self.prev_x == 0 and self.prev_y == 0:
                self.prev_x, self.prev_y = index_tip

            # Draw ink line segment
            cv2.line(self.canvas, (self.prev_x, self.prev_y), index_tip, (30, 30, 30), 4, cv2.LINE_AA)
            self.prev_x, self.prev_y = index_tip

            # Active red indicator dot on pinch point
            cv2.circle(frame, index_tip, 6, (0, 0, 255), -1, cv2.LINE_AA)
        else:
            self.is_pinching = False
            self.prev_x, self.prev_y = 0, 0
            # Hover cursor dot (green) on index tip when hand is open
            cv2.circle(frame, index_tip, 5, (0, 255, 0), -1, cv2.LINE_AA)

    def clear(self):
        if self.canvas is not None:
            self.canvas.fill(0)
        self.text_buffer = ""

    def merge_with_frame(self, frame):
        if self.canvas is None:
            return frame

        gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, inv_canvas = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY_INV)
        inv_canvas = cv2.cvtColor(inv_canvas, cv2.COLOR_GRAY2BGR)

        frame = cv2.bitwise_and(frame, inv_canvas)
        frame = cv2.bitwise_or(frame, self.canvas)
        return frame
