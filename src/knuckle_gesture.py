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
    def __init__(self, cooldown=0.5):
        self.cooldown = cooldown
        self.last_input_time = 0
        self.last_fist_time = 0
        self.last_char = None

    def is_fist_gesture(self, landmarks):
        """
        Detects Closed Fist Gesture (✊ mut).
        All 4 finger tips must be BELOW their respective PIP joints.
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
            if now - self.last_fist_time > 0.7:
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
        TIGHT threshold finger joint typing — only deliberate thumb-to-joint
        pinch triggers a character. Normal open hand will NOT trigger.

        Uses 0.18 * hand_scale as threshold. For a typical hand_scale of ~0.30,
        this gives ~0.054 — only a real pinch/touch reaches this distance.

        Mapping:
        - Thumb Tip (4) + Index Tip (8)   -> "H"
        - Thumb Tip (4) + Index PIP (7)   -> "e"
        - Thumb Tip (4) + Index MCP (6)   -> "l"
        - Thumb Tip (4) + Middle Tip (12) -> "o"
        - Thumb Tip (4) + Middle PIP (10) -> " " (Space)
        - Thumb Tip (4) + Ring Tip (16)   -> ","
        """
        if not landmarks or len(landmarks) < 21:
            return None, None

        now = time.time()
        if now - self.last_input_time < self.cooldown:
            return None, None

        wrist = extract_xy(landmarks[0])
        middle_mcp = extract_xy(landmarks[9])
        hand_scale = np.linalg.norm(wrist - middle_mcp)

        # TIGHT threshold: only deliberate pinch (thumb touching joint) triggers
        if hand_scale > 10.0:
            # Pixel space (e.g. 1280x720)
            touch_threshold = max(25.0, 0.18 * hand_scale)
        else:
            # Normalized [0.0, 1.0]
            touch_threshold = max(0.035, 0.18 * hand_scale)

        thumb_tip = extract_xy(landmarks[4])
        index_tip = extract_xy(landmarks[8])
        index_pip = extract_xy(landmarks[7])
        index_mcp = extract_xy(landmarks[6])
        middle_tip = extract_xy(landmarks[12])
        middle_pip = extract_xy(landmarks[10])
        ring_tip = extract_xy(landmarks[16])

        # Calculate all distances
        distances = [
            (np.linalg.norm(thumb_tip - index_tip),  "H", index_tip),
            (np.linalg.norm(thumb_tip - index_pip),   "e", index_pip),
            (np.linalg.norm(thumb_tip - index_mcp),   "l", index_mcp),
            (np.linalg.norm(thumb_tip - middle_tip),  "o", middle_tip),
            (np.linalg.norm(thumb_tip - middle_pip),  " ", middle_pip),
            (np.linalg.norm(thumb_tip - ring_tip),    ",", ring_tip),
        ]

        # Find the closest joint
        distances.sort(key=lambda x: x[0])
        closest_dist, closest_char, closest_pt = distances[0]

        if closest_dist < touch_threshold:
            # Prevent same character from repeating if finger stays touching
            if closest_char == self.last_char:
                return None, None
            self.last_char = closest_char
            self.last_input_time = now
            return closest_char, closest_pt
        else:
            # Finger lifted — allow same char to be typed again next pinch
            self.last_char = None

        return None, None
