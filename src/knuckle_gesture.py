import time
import numpy as np

class KnuckleGestureEngine:
    def __init__(self, touch_threshold=35, cooldown=0.45):
        self.touch_threshold = touch_threshold
        self.cooldown = cooldown
        self.last_input_time = 0
        self.last_fist_time = 0

    def is_fist_gesture(self, hand_landmarks):
        """
        Detects Closed Fist Gesture (✊ mut).
        Checks if index, middle, ring, pinky tips are folded down towards palm.
        """
        if not hand_landmarks or len(hand_landmarks) < 21:
            return False

        # Tips vs PIP/MCP joints
        index_folded = hand_landmarks[8][1] > hand_landmarks[6][1]
        middle_folded = hand_landmarks[12][1] > hand_landmarks[10][1]
        ring_folded = hand_landmarks[16][1] > hand_landmarks[14][1]
        pinky_folded = hand_landmarks[20][1] > hand_landmarks[18][1]

        return index_folded and middle_folded and ring_folded and pinky_folded

    def check_word_erase(self, hand_landmarks, text_buffer):
        """
        If Closed Fist (✊ mut) is detected, erases one word from text_buffer.
        Clenching fist twice (dubar mut) erases two words.
        """
        now = time.time()
        if self.is_fist_gesture(hand_landmarks):
            if now - self.last_fist_time > self.cooldown:
                self.last_fist_time = now
                words = text_buffer.strip().split()
                if words:
                    words.pop()
                    return " ".join(words), True
        return text_buffer, False

    def detect_knuckle_touch(self, hand_landmarks):
        """
        Calculates Euclidean distance between Thumb Tip (Landmark 4) and index/middle joints.
        """
        if not hand_landmarks or len(hand_landmarks) < 21:
            return None

        thumb_tip = np.array(hand_landmarks[4][:2])
        index_tip = np.array(hand_landmarks[8][:2])
        index_pip = np.array(hand_landmarks[6][:2])
        index_mcp = np.array(hand_landmarks[5][:2])
        middle_tip = np.array(hand_landmarks[12][:2])

        dist_idx_pip = np.linalg.norm(thumb_tip - index_pip)
        dist_idx_mcp = np.linalg.norm(thumb_tip - index_mcp)
        dist_mid_tip = np.linalg.norm(thumb_tip - middle_tip)

        now = time.time()
        if now - self.last_input_time < self.cooldown:
            return None

        char = None
        if dist_idx_pip < self.touch_threshold:
            char = " "
            self.last_input_time = now
        elif dist_idx_mcp < self.touch_threshold:
            char = "BACKSPACE"
            self.last_input_time = now

        return char
