import time
import numpy as np

class KnuckleGestureEngine:
    def __init__(self, cooldown=0.45):
        self.cooldown = cooldown
        self.last_input_time = 0
        self.last_fist_time = 0

    def is_fist_gesture(self, landmarks_normalized):
        """
        Detects Closed Fist Gesture (✊ mut) using normalized MediaPipe landmarks.
        Checks if index, middle, ring, pinky tips are folded down towards palm.
        """
        if not landmarks_normalized or len(landmarks_normalized) < 21:
            return False

        index_folded = landmarks_normalized[8].y > landmarks_normalized[6].y
        middle_folded = landmarks_normalized[12].y > landmarks_normalized[10].y
        ring_folded = landmarks_normalized[16].y > landmarks_normalized[14].y
        pinky_folded = landmarks_normalized[20].y > landmarks_normalized[18].y

        return index_folded and middle_folded and ring_folded and pinky_folded

    def check_word_erase(self, landmarks_normalized, text_buffer):
        """
        If Closed Fist (✊ mut) is detected, erases one word from text_buffer.
        """
        now = time.time()
        if self.is_fist_gesture(landmarks_normalized):
            if now - self.last_fist_time > self.cooldown:
                self.last_fist_time = now
                words = text_buffer.strip().split()
                if words:
                    words.pop()
                    return " ".join(words), True
                elif text_buffer:
                    return "", True
        return text_buffer, False

    def detect_finger_joint_typing(self, landmarks_normalized):
        """
        Hand-scale invariant Euclidean distance calculation.
        Uses hand_scale = distance(wrist(0), middle_mcp(9)) to adapt dynamically
        to any camera distance!
        - 8: Index Tip -> "H"
        - 7: Index PIP -> "e"
        - 6: Index MCP -> "l"
        - 12: Middle Tip -> "o"
        - 10: Middle PIP -> " " (Space)
        - 16: Ring Tip -> ","
        """
        if not landmarks_normalized or len(landmarks_normalized) < 21:
            return None, None

        now = time.time()
        if now - self.last_input_time < self.cooldown:
            return None, None

        wrist = np.array([landmarks_normalized[0].x, landmarks_normalized[0].y])
        middle_mcp = np.array([landmarks_normalized[9].x, landmarks_normalized[9].y])
        hand_scale = np.linalg.norm(wrist - middle_mcp)
        if hand_scale < 1e-4:
            hand_scale = 0.20

        # Scale-invariant touch threshold
        touch_threshold = 0.42 * hand_scale

        thumb_tip = np.array([landmarks_normalized[4].x, landmarks_normalized[4].y])
        index_tip = np.array([landmarks_normalized[8].x, landmarks_normalized[8].y])
        index_pip = np.array([landmarks_normalized[7].x, landmarks_normalized[7].y])
        index_mcp = np.array([landmarks_normalized[6].x, landmarks_normalized[6].y])
        middle_tip = np.array([landmarks_normalized[12].x, landmarks_normalized[12].y])
        middle_pip = np.array([landmarks_normalized[10].x, landmarks_normalized[10].y])
        ring_tip = np.array([landmarks_normalized[16].x, landmarks_normalized[16].y])

        d_index_tip = np.linalg.norm(thumb_tip - index_tip)
        d_index_pip = np.linalg.norm(thumb_tip - index_pip)
        d_index_mcp = np.linalg.norm(thumb_tip - index_mcp)
        d_middle_tip = np.linalg.norm(thumb_tip - middle_tip)
        d_middle_pip = np.linalg.norm(thumb_tip - middle_pip)
        d_ring_tip = np.linalg.norm(thumb_tip - ring_tip)

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

        if char is not None:
            self.last_input_time = now

        return char, touch_pt
