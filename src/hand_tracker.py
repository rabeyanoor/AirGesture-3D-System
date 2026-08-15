import cv2
import numpy as np
import mediapipe as mp
from src.config import MAX_NUM_HANDS, MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE, SMOOTHING_FACTOR

class HandTracker:
    def __init__(self, max_hands=MAX_NUM_HANDS, detection_con=MIN_DETECTION_CONFIDENCE, tracking_con=MIN_TRACKING_CONFIDENCE):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_con,
            min_tracking_confidence=tracking_con
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.prev_landmarks = {}

    def process(self, frame):
        """
        Processes video frame and extracts smoothed 2D/3D hand landmarks.
        Returns:
            landmarks_list: List of 21 (x, y, z) tuple positions per detected hand
            handedness_list: List of 'Left' or 'Right' labels per hand
            raw_results: MediaPipe raw output
        """
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        landmarks_list = []
        handedness_list = []

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Extract handedness label
                handedness_label = "Right"
                if results.multi_handedness and idx < len(results.multi_handedness):
                    handedness_label = results.multi_handedness[idx].classification[0].label
                handedness_list.append(handedness_label)

                # Draw landmark connections on frame
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style()
                )

                # Extract coordinates with exponential moving average smoothing
                pts = []
                for lm_id, lm in enumerate(hand_landmarks.landmark):
                    cx, cy, cz = int(lm.x * w), int(lm.y * h), lm.z * w

                    # Apply smoothing if previous landmark exists
                    key = (idx, lm_id)
                    if key in self.prev_landmarks:
                        prev_cx, prev_cy, prev_cz = self.prev_landmarks[key]
                        cx = int(SMOOTHING_FACTOR * cx + (1 - SMOOTHING_FACTOR) * prev_cx)
                        cy = int(SMOOTHING_FACTOR * cy + (1 - SMOOTHING_FACTOR) * prev_cy)
                        cz = SMOOTHING_FACTOR * cz + (1 - SMOOTHING_FACTOR) * prev_cz
                    
                    self.prev_landmarks[key] = (cx, cy, cz)
                    pts.append((cx, cy, cz))

                landmarks_list.append(pts)

        return landmarks_list, handedness_list, results

    @staticmethod
    def calculate_distance(p1, p2):
        """Calculates 2D Euclidean distance between two points."""
        return np.hypot(p1[0] - p2[0], p1[1] - p2[1])

    @staticmethod
    def calculate_3d_distance(p1, p2):
        """Calculates 3D Euclidean distance between two points."""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)
