"""
Hand Tracking Module using MediaPipe Hands
Provides hand detection, 21 landmark extraction, and spatial 3D coordinate utilities.
"""

import math
import threading
import time

import cv2
import mediapipe as mp


class HandTracker:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7,
                 max_infer_width=960, model_complexity=0, async_mode=False):
        """
        model_complexity=0 is MediaPipe's lite hand model (~1.5x faster than the full one on CPU).
        async_mode=True runs detection on a worker thread: process() returns immediately with the latest
        landmarks, so the video keeps flowing at camera speed while tracking catches up.
        """
        self.max_infer_width = max_infer_width
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # Fingertip and Joint IDs
        self.FINGERTIP_IDS = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        self.PIP_JOINT_IDS = [2, 6, 10, 14, 18]

        self.async_mode = async_mode
        self._latest = []            # [(label, score, landmarks proto)] from the last finished detection
        self._pending = None         # newest frame waiting for the worker
        self._cond = threading.Condition()
        self.detect_ms = 0.0
        if async_mode:
            threading.Thread(target=self._worker, daemon=True).start()

    def _detect(self, frame):
        """Run MediaPipe on a (downscaled) copy of the frame; returns normalised results."""
        h, w = frame.shape[:2]
        small = frame
        if w > self.max_infer_width:
            scale = self.max_infer_width / float(w)
            small = cv2.resize(frame, (self.max_infer_width, int(h * scale)), interpolation=cv2.INTER_AREA)
        t0 = time.perf_counter()
        results = self.hands.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        self.detect_ms = (time.perf_counter() - t0) * 1000
        out = []
        if results.multi_hand_landmarks and results.multi_handedness:
            for lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                c = handedness.classification[0]
                out.append((c.label, c.score, lms))
        return out

    def _worker(self):
        while True:
            with self._cond:
                while self._pending is None:
                    self._cond.wait()
                frame, self._pending = self._pending, None
            self._latest = self._detect(frame)

    def process(self, frame):
        """
        Returns (hands_data, left_hand, right_hand) with landmarks in this frame's pixel coordinates.
        In async mode the frame is queued for detection and the most recent finished result is returned.
        """
        h, w = frame.shape[:2]
        if self.async_mode:
            with self._cond:
                # Only the newest frame matters: an older queued frame is simply replaced
                self._pending = cv2.resize(frame, (self.max_infer_width, int(h * self.max_infer_width / w))) \
                    if w > self.max_infer_width else frame.copy()
                self._cond.notify()
            detections = self._latest
        else:
            detections = self._detect(frame)

        left_hand = None
        right_hand = None
        hands_data = []
        for label, score, hand_landmarks in detections:
            landmarks_px = []
            landmarks_norm = []
            for lm in hand_landmarks.landmark:
                landmarks_px.append((int(lm.x * w), int(lm.y * h), lm.z))
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

