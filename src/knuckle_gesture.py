import time
import numpy as np

class KnuckleGestureEngine:
    def __init__(self, touch_threshold=35, cooldown=0.35):
        self.touch_threshold = touch_threshold
        self.cooldown = cooldown
        self.last_input_time = 0

    def detect_knuckle_touch(self, hand_landmarks):
        """
        Calculates Euclidean distance between Thumb Tip (Landmark 4) and index/middle joints:
        - 8: Index Tip
        - 6: Index PIP Joint
        - 5: Index MCP Joint
        - 12: Middle Tip
        - 10: Middle PIP Joint
        """
        if not hand_landmarks or len(hand_landmarks) < 21:
            return None

        thumb_tip = np.array(hand_landmarks[4][:2])
        index_tip = np.array(hand_landmarks[8][:2])
        index_pip = np.array(hand_landmarks[6][:2])
        index_mcp = np.array(hand_landmarks[5][:2])
        middle_tip = np.array(hand_landmarks[12][:2])
        middle_pip = np.array(hand_landmarks[10][:2])

        # Euclidean distances
        dist_idx_tip = np.linalg.norm(thumb_tip - index_tip)
        dist_idx_pip = np.linalg.norm(thumb_tip - index_pip)
        dist_idx_mcp = np.linalg.norm(thumb_tip - index_mcp)
        dist_mid_tip = np.linalg.norm(thumb_tip - middle_tip)
        dist_mid_pip = np.linalg.norm(thumb_tip - middle_pip)

        now = time.time()
        if now - self.last_input_time < self.cooldown:
            return None

        char = None
        if dist_idx_tip < self.touch_threshold:
            char = "PINCH_DRAW"  # Active Air Scribble Mode
        elif dist_idx_pip < self.touch_threshold:
            char = " "
            self.last_input_time = now
        elif dist_idx_mcp < self.touch_threshold:
            char = "BACKSPACE"
            self.last_input_time = now
        elif dist_mid_tip < self.touch_threshold:
            char = "NEXT_LINE"
            self.last_input_time = now

        return char
