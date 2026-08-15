import time
import numpy as np

def extract_xy(lm):
    """Safely extracts (x, y) coordinates from landmark objects or (x, y, z) tuples."""
    if hasattr(lm, 'x') and hasattr(lm, 'y'):
        return np.array([lm.x, lm.y], dtype=np.float32)
    elif isinstance(lm, (tuple, list)) and len(lm) >= 2:
        return np.array([lm[0], lm[1]], dtype=np.float32)
    return np.array([0.0, 0.0], dtype=np.float32)

class KnuckleGestureEngine:
    def __init__(self, cooldown=0.35):
        self.cooldown = cooldown
        self.last_input_time = 0
        self.last_fist_time = 0

    def is_fist_gesture(self, landmarks):
        """
        Detects Closed Fist Gesture (✊ mut).
        Checks if index, middle, ring, pinky tips are folded down towards palm.
        """
        if not landmarks or len(landmarks) < 21:
            return False

        idx_tip_y = extract_xy(landmarks[8])[1]
        idx_pip_y = extract_xy(landmarks[6])[1]

        mid_tip_y = extract_xy(landmarks[12])[1]
        mid_pip_y = extract_xy(landmarks[10])[1]

        rng_tip_y = extract_xy(landmarks[16])[1]
        rng_pip_y = extract_xy(landmarks[14])[1]

        pnk_tip_y = extract_xy(landmarks[20])[1]
        pnk_pip_y = extract_xy(landmarks[18])[1]

        return (idx_tip_y > idx_pip_y) and (mid_tip_y > mid_pip_y) and \
               (rng_tip_y > rng_pip_y) and (pnk_tip_y > pnk_pip_y)

    def check_word_erase(self, landmarks, text_buffer):
        """
        If Closed Fist (✊ mut) is detected, erases one word from text_buffer.
        """
        now = time.time()
        if self.is_fist_gesture(landmarks):
            if now - self.last_fist_time > self.cooldown:
                self.last_fist_time = now
                words = text_buffer.strip().split()
                if words:
                    words.pop()
                    return " ".join(words), True
                elif text_buffer:
                    return "", True
        return text_buffer, False

    def detect_finger_joint_typing(self, landmarks):
        """
        Ultra-robust finger joint typing engine. Supports both pixel tuples and normalized landmark objects!
        - 8: Index Tip -> "H"
        - 7: Index PIP -> "e"
        - 6: Index MCP -> "l"
        - 12: Middle Tip -> "o"
        - 10: Middle PIP -> " " (Space)
        - 16: Ring Tip -> ","
        - 20: Pinky Tip -> " Spatial AR"
        """
        if not landmarks or len(landmarks) < 21:
            return None, None

        now = time.time()
        if now - self.last_input_time < self.cooldown:
            return None, None

        wrist = extract_xy(landmarks[0])
        middle_mcp = extract_xy(landmarks[9])
        hand_scale = np.linalg.norm(wrist - middle_mcp)

        # Determine threshold dynamically based on coordinate scale (normalized vs pixels)
        if hand_scale > 10.0:
            # Pixel space coordinates (e.g. 1280x720)
            touch_threshold = max(45.0, 0.50 * hand_scale)
        else:
            # Normalized float coordinates [0.0, 1.0]
            touch_threshold = max(0.09, 0.50 * hand_scale)

        thumb_tip = extract_xy(landmarks[4])
        index_tip = extract_xy(landmarks[8])
        index_pip = extract_xy(landmarks[7])
        index_mcp = extract_xy(landmarks[6])
        middle_tip = extract_xy(landmarks[12])
        middle_pip = extract_xy(landmarks[10])
        ring_tip = extract_xy(landmarks[16])
        pinky_tip = extract_xy(landmarks[20])

        d_index_tip = np.linalg.norm(thumb_tip - index_tip)
        d_index_pip = np.linalg.norm(thumb_tip - index_pip)
        d_index_mcp = np.linalg.norm(thumb_tip - index_mcp)
        d_middle_tip = np.linalg.norm(thumb_tip - middle_tip)
        d_middle_pip = np.linalg.norm(thumb_tip - middle_pip)
        d_ring_tip = np.linalg.norm(thumb_tip - ring_tip)
        d_pinky_tip = np.linalg.norm(thumb_tip - pinky_tip)

        char = None
        touch_pt = None

        if d_index_tip < touch_threshold:
            char = "H"
            touch_pt = index_tip
        elif d_index_pip < touch_threshold:
            char = "e"
            touch_pt = index_pip
        elif d_index_mcp < touch_threshold:
            char = "l"
            touch_pt = index_mcp
        elif d_middle_tip < touch_threshold:
            char = "o"
            touch_pt = middle_tip
        elif d_middle_pip < touch_threshold:
            char = " "
            touch_pt = middle_pip
        elif d_ring_tip < touch_threshold:
            char = ","
            touch_pt = ring_tip
        elif d_pinky_tip < touch_threshold:
            char = " Spatial AR"
            touch_pt = pinky_tip

        if char is not None:
            self.last_input_time = now

        return char, touch_pt
