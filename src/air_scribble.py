import cv2
import numpy as np

class AirScribble:
    def __init__(self, thickness=4, pinch_thresh=40):
        self.canvas = None
        self.prev_x = 0
        self.prev_y = 0
        self.thickness = thickness
        self.pinch_thresh = pinch_thresh

    def update(self, frame, landmarks_list):
        h, w, _ = frame.shape
        if self.canvas is None:
            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)

        if len(landmarks_list) > 0:
            index_tip = landmarks_list[0][8]
            thumb_tip = landmarks_list[0][4]

            distance = np.hypot(index_tip[0] - thumb_tip[0], index_tip[1] - thumb_tip[1])

            if distance < self.pinch_thresh:
                if self.prev_x == 0 and self.prev_y == 0:
                    self.prev_x, self.prev_y = index_tip
                cv2.line(self.canvas, (self.prev_x, self.prev_y), index_tip, (255, 255, 255), self.thickness)
                self.prev_x, self.prev_y = index_tip
            else:
                self.prev_x, self.prev_y = 0, 0

    def clear(self):
        if self.canvas is not None:
            self.canvas.fill(0)

    def merge_with_frame(self, frame):
        if self.canvas is None:
            return frame
        gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, inv_canvas = cv2.threshold(gray_canvas, 50, 255, cv2.THRESH_BINARY_INV)
        inv_canvas = cv2.cvtColor(inv_canvas, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, inv_canvas)
        frame = cv2.bitwise_or(frame, self.canvas)
        return frame
