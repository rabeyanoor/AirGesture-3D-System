import cv2
import numpy as np
import mediapipe as mp
from src.config import MAX_NUM_HANDS, MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE, SMOOTHING_FACTOR


class HandTracker:
    def __init__(self, max_hands=2, detection_con=MIN_DETECTION_CONFIDENCE, tracking_con=MIN_TRACKING_CONFIDENCE):
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
            landmarks_list:   List of 21 (x, y, z) pixel-space tuples per hand
            handedness_list:  List of 'Left' or 'Right' per hand
            raw_results:      MediaPipe raw output (normalized landmarks)
        """
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        landmarks_list = []
        handedness_list = []

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Handedness label
                handedness_label = "Right"
                if results.multi_handedness and idx < len(results.multi_handedness):
                    handedness_label = results.multi_handedness[idx].classification[0].label
                handedness_list.append(handedness_label)

                # Draw landmarks
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style()
                )

                # Extract + smooth pixel-space coords
                pts = []
                for lm_id, lm in enumerate(hand_landmarks.landmark):
                    cx = int(lm.x * w)
                    cy = int(lm.y * h)
                    cz = lm.z * w

                    key = (idx, lm_id)
                    if key in self.prev_landmarks:
                        px, py, pz = self.prev_landmarks[key]
                        cx = int(SMOOTHING_FACTOR * cx + (1 - SMOOTHING_FACTOR) * px)
                        cy = int(SMOOTHING_FACTOR * cy + (1 - SMOOTHING_FACTOR) * py)
                        cz = SMOOTHING_FACTOR * cz + (1 - SMOOTHING_FACTOR) * pz

                    self.prev_landmarks[key] = (cx, cy, cz)
                    pts.append((cx, cy, cz))

                landmarks_list.append(pts)

        return landmarks_list, handedness_list, results

    @staticmethod
    def get_hand_by_label(landmarks_list, handedness_list, norm_results, label):
        """
        Returns (pixel_landmarks, norm_landmarks) for the requested label ('Left'/'Right').
        norm_landmarks are raw mediapipe landmark objects with .x .y .z attributes.
        """
        pixel_lm = None
        norm_lm  = None

        for i, side in enumerate(handedness_list):
            if side == label:
                pixel_lm = landmarks_list[i] if i < len(landmarks_list) else None
                if (norm_results and norm_results.multi_hand_landmarks
                        and i < len(norm_results.multi_hand_landmarks)):
                    norm_lm = norm_results.multi_hand_landmarks[i].landmark
                break

        return pixel_lm, norm_lm

    @staticmethod
    def calculate_distance(p1, p2):
        return np.hypot(p1[0] - p2[0], p1[1] - p2[1])

    @staticmethod
    def calculate_3d_distance(p1, p2):
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
