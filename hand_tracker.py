"""
Hand Tracking Module using MediaPipe Hands
Provides hand detection, 21 landmark extraction, and spatial 3D coordinate utilities.
"""

import cv2
import mediapipe as mp
try:
    import mediapipe.python.solutions.hands as mp_hands
    import mediapipe.python.solutions.drawing_utils as mp_draw
    import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
except ImportError:
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
import math
import numpy as np


class HandTracker:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        try:
            self.mp_hands = mp.solutions.hands
            self.mp_draw = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
        except AttributeError:
            self.mp_hands = mp_hands
            self.mp_draw = mp_draw
            self.mp_drawing_styles = mp_drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # Fingertip and Joint IDs
        self.FINGERTIP_IDS = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        self.PIP_JOINT_IDS = [2, 6, 10, 14, 18]

    def process(self, frame):
        """
        Process BGR frame and return detected hand landmarks and handedness labels.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        left_hand = None
        right_hand = None
        hands_data = []

        h, w, c = frame.shape

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label  # 'Left' or 'Right'
                score = handedness.classification[0].score

                # Extract pixel coordinates
                landmarks_px = []
                landmarks_norm = []
                for lm in hand_landmarks.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks_px.append((cx, cy, lm.z))
                    landmarks_norm.append((lm.x, lm.y, lm.z))

                hand_dict = {
                    'label': label,
                    'score': score,
                    'landmarks': hand_landmarks,
                    'px': landmarks_px,
                    'norm': landmarks_norm
                }
                hands_data.append(hand_dict)

                if label == 'Left':
                    left_hand = hand_dict
                elif label == 'Right':
                    right_hand = hand_dict

        return hands_data, left_hand, right_hand

    @staticmethod
    def calculate_distance(p1, p2, width=1, height=1):
        """
        Calculate Euclidean distance between two points (pixel or normalized).
        """
        if len(p1) >= 2 and len(p2) >= 2:
            dx = (p1[0] - p2[0]) * width
            dy = (p1[1] - p2[1]) * height
            return math.hypot(dx, dy)
        return float('inf')

    @staticmethod
    def is_palm_facing_camera(hand_dict):
        """
        Determines whether the hand is facing the camera.
        Returns True for detected hands to ensure landmarks and UI overlays are always rendered.
        """
        if not hand_dict or 'px' not in hand_dict:
            return False

        px = hand_dict['px']
        p0 = px[0]   # Wrist
        p5 = px[5]   # Index MCP
        p17 = px[17] # Pinky MCP

        # Check palm width span to filter out edge cases where hand is extremely far
        dx_palm = abs(p5[0] - p17[0])
        dy_palm = abs(p5[1] - p17[1])
        palm_span = math.hypot(dx_palm, dy_palm)

        wrist_to_middle = math.hypot(px[9][0] - p0[0], px[9][1] - p0[1])

        if wrist_to_middle > 0 and (palm_span / wrist_to_middle) < 0.15:
            return False

        return True

