"""
Gesture Recognizer Module
Detects hand gestures:
1. Fist Gesture (Delete Last Word)
2. Palm Tap / Touch (Space Bar Insertion)
3. Right Motion / Boundary Entry (Sidebar Trigger)
4. Dual-Hand Grid Keypad Matrix Selection (A-Z Virtual Tap)
5. Index Pointing & Drawing Gesture
"""

import math
import time


class GestureRecognizer:
    def __init__(self):
        # A-Z Grid Matrix Mapping
        self.CHAR_MATRIX = {
            "GRP1": ["A", "B", "C", "D", "E"],
            "GRP2": ["F", "G", "H", "I", "J"],
            "GRP3": ["K", "L", "M", "N", "O"],
            "GRP4": ["P", "Q", "R", "S", "T"],
            "GRP5": ["U", "V", "W", "X", "Y", "Z"]
        }

        # Debounce Flags
        self.fist_triggered = False
        self.palm_touch_triggered = False
        self.keypad_triggered = False
        self.last_keypad_time = 0
        self.cooldown_period = 0.4  # seconds cooldown for typing

    @staticmethod
    def is_fist(hand_dict):
        """
        Check if index, middle, ring, and pinky tips are tightly folded into palm (3D distance check).
        Prevents false positive triggers when hand is resting or pointing downwards.
        """
        if not hand_dict:
            return False

        landmarks = hand_dict['landmarks'].landmark
        # Pairs: (Fingertip ID, MCP Joint Base ID)
        pairs = [(8, 5), (12, 9), (16, 13), (20, 17)]

        folded_count = 0
        for tip, mcp in pairs:
            dist = math.hypot(landmarks[tip].x - landmarks[mcp].x, landmarks[tip].y - landmarks[mcp].y)
            dz = abs(landmarks[tip].z - landmarks[mcp].z)
            if dist < 0.085 and dz < 0.06:
                folded_count += 1

        return folded_count >= 4

    @staticmethod
    def is_index_pointing(hand_dict):
        """
        Check if only index finger is extended for drawing/pointing.
        """
        if not hand_dict:
            return False

        lm = hand_dict['landmarks'].landmark
        index_extended = lm[8].y < lm[6].y
        middle_folded = lm[12].y > lm[10].y
        ring_folded = lm[16].y > lm[14].y
        pinky_folded = lm[20].y > lm[18].y

        return index_extended and middle_folded and ring_folded and pinky_folded

    def check_palm_touch(self, left_hand, right_hand, frame_w, frame_h, distance_threshold=40):
        """
        Check if right index tip touches left palm center (landmark 9 or 0).
        """
        if not left_hand or not right_hand:
            return False, None

        # Left palm center (Landmark 9 Middle MCP or Landmark 0 Wrist)
        l_palm = left_hand['px'][9]
        r_index_tip = right_hand['px'][8]

        dist = math.hypot(r_index_tip[0] - l_palm[0], r_index_tip[1] - l_palm[1])

        if dist < distance_threshold:
            return True, (l_palm[0], l_palm[1])
        return False, None

    def update_sidebar_state(self, hands_data, frame_w, current_state, threshold_ratio=0.80):
        """
        Dual-Trigger Sidebar Mechanic:
        1. Boundary Threshold: Hand enters right 80% boundary zone of screen.
        2. Swipe Gesture: Rapid rightward hand movement vector (dx / dt velocity).
        """
        now = time.time()
        is_triggered = False
        right_threshold = int(frame_w * threshold_ratio)

        for hand in hands_data:
            index_x = hand['px'][8][0]
            wrist_x = hand['px'][0][0]

            # 1. Boundary Threshold Check (80% boundary)
            if index_x > right_threshold or wrist_x > right_threshold:
                is_triggered = True
                break

            # 2. Rightward Velocity / Motion Vector Check
            hand_label = hand['label']
            last_x = getattr(self, f'prev_{hand_label}_x', None)
            last_time = getattr(self, f'prev_{hand_label}_time', None)

            if last_x is not None and last_time is not None:
                dt = now - last_time
                if 0.01 < dt < 0.3:
                    dx = index_x - last_x
                    if dx > 35 and index_x > int(frame_w * 0.60):
                        is_triggered = True
                        break

            setattr(self, f'prev_{hand_label}_x', index_x)
            setattr(self, f'prev_{hand_label}_time', now)

        if is_triggered:
            if not getattr(self, 'sidebar_triggered', False) and (now - getattr(self, 'last_sidebar_toggle', 0) > 0.5):
                current_state = True  # Opens sidebar on right motion / boundary
                self.last_sidebar_toggle = now
                self.sidebar_triggered = True
        else:
            self.sidebar_triggered = False

        return current_state

    def check_sidebar_clicks(self, hands_data, ui_renderer, sidebar_open):
        """
        Check if index finger clicks/touches any of the sidebar tiles:
        - Tile 3 (Power Off Button): Closes the sidebar! (sidebar_open = False)
        - Tile 2 (Document Button): Toggles AR Notepad!
        - Tile 1 (Bulb Button): Light action
        """
        if not sidebar_open or not ui_renderer.tile_rects:
            return sidebar_open, None

        now = time.time()
        for hand in hands_data:
            index_tip = hand['px'][8]  # Index fingertip
            ix, iy = index_tip[0], index_tip[1]

            # Tile 1: Bulb Icon -> Brightness / Dark Theme Toggle
            if 'tile_1' in ui_renderer.tile_rects:
                tx, ty, tw, th = ui_renderer.tile_rects['tile_1']
                if tx <= ix <= tx + tw and ty <= iy <= ty + th:
                    if now - getattr(self, 'last_click_time', 0) > 0.5:
                        self.last_click_time = now
                        ui_renderer.dark_mode = not ui_renderer.dark_mode
                        return sidebar_open, "THEME_TOGGLED"

            # Tile 2: Document / Sheet Icon -> AR Notepad Canvas Toggle (On/Off)
            if 'tile_2' in ui_renderer.tile_rects:
                tx, ty, tw, th = ui_renderer.tile_rects['tile_2']
                if tx <= ix <= tx + tw and ty <= iy <= ty + th:
                    if now - getattr(self, 'last_click_time', 0) > 0.5:
                        self.last_click_time = now
                        ui_renderer.show_notepad = not ui_renderer.show_notepad
                        return sidebar_open, "NOTEPAD_TOGGLED"

            # Tile 3: Power Icon -> Application Exit / Streaming Off
            if 'tile_3' in ui_renderer.tile_rects:
                tx, ty, tw, th = ui_renderer.tile_rects['tile_3']
                if tx <= ix <= tx + tw and ty <= iy <= ty + th:
                    if now - getattr(self, 'last_click_time', 0) > 0.5:
                        self.last_click_time = now
                        return False, "EXIT_APP"

        return sidebar_open, None

    def process_grid_keypad(self, left_hand, right_hand, frame_w, frame_h):
        """
        Dual-Hand Grid Keypad Matrix Selection:
        - Left hand pinch selects Group (GRP1-GRP5)
        - Right hand pinch selects character from group (1st to 5th char)
        """
        if not left_hand or not right_hand:
            return None, None

        selected_group = None
        typed_char = None

        l_px = left_hand['px']
        r_px = right_hand['px']

        l_thumb = l_px[4]

        # 1. Left Hand: Select Letter Group via Thumb Pinch
        d_index = math.hypot(l_thumb[0] - l_px[8][0], l_thumb[1] - l_px[8][1])
        d_middle = math.hypot(l_thumb[0] - l_px[12][0], l_thumb[1] - l_px[12][1])
        d_ring = math.hypot(l_thumb[0] - l_px[16][0], l_thumb[1] - l_px[16][1])
        d_pinky = math.hypot(l_thumb[0] - l_px[20][0], l_thumb[1] - l_px[20][1])

        pinch_thresh = 40

        if d_index < pinch_thresh:
            selected_group = "GRP1"
        elif d_middle < pinch_thresh:
            selected_group = "GRP2"
        elif d_ring < pinch_thresh:
            selected_group = "GRP3"
        elif d_pinky < pinch_thresh:
            selected_group = "GRP4"
        else:
            # Default open left palm selects GRP5
            selected_group = "GRP5"

        # 2. Right Hand: Select Exact Character via Thumb Pinch with Finger
        r_thumb = r_px[4]
        rd_index = math.hypot(r_thumb[0] - r_px[8][0], r_thumb[1] - r_px[8][1])
        rd_middle = math.hypot(r_thumb[0] - r_px[12][0], r_thumb[1] - r_px[12][1])
        rd_ring = math.hypot(r_thumb[0] - r_px[16][0], r_thumb[1] - r_px[16][1])
        rd_pinky = math.hypot(r_thumb[0] - r_px[20][0], r_thumb[1] - r_px[20][1])

        char_idx = None
        if rd_index < pinch_thresh:
            char_idx = 0
        elif rd_middle < pinch_thresh:
            char_idx = 1
        elif rd_ring < pinch_thresh:
            char_idx = 2
        elif rd_pinky < pinch_thresh:
            char_idx = 3

        now = time.time()
        if selected_group and char_idx is not None:
            group_chars = self.CHAR_MATRIX[selected_group]
            if char_idx < len(group_chars):
                if now - self.last_keypad_time > self.cooldown_period:
                    typed_char = group_chars[char_idx]
                    self.last_keypad_time = now

        return selected_group, typed_char

    def detect_phalanx_keyboard(self, left_hand, right_hand, frame_w, frame_h, touch_threshold=28):
        """
        Finger Phalanx Keyboard (Chorded Finger Mapping A-Z):
        Uses Dynamic Scale Adaptability + Responsive 2D Touch Threshold + Nearest-Neighbor Selection.
        - Thumb-to-Palm Touch (Thumb 4 touches Palm 9): Deletes Last Word
        - Left Hand (Thumb Tip 4 touches joints 8,7,6 | 12,11,10 | 16,15,14 | 20,19,18): Letters A to L
        - Right Hand (Thumb Tip 4 touches joints 8,7,6 | 12,11,10 | 16,15,14 | 20,19,18): Letters M to X
        - Dual Thumb Touch (Left Thumb 4 + Right Thumb 4): Letter Y
        - Dual Index Touch (Left Index 8 + Right Index 8): Letter Z
        """
        now = time.time()

        # Cooldown guard (0.35s debounce delay)
        if now - getattr(self, 'last_phalanx_time', 0) < 0.35:
            return None

        # 1. Thumb-to-Palm Touch Gesture (Delete Last Word)
        if left_hand:
            l_thumb = left_hand['px'][4]
            l_palm = left_hand['px'][9]
            dist_l_palm = math.hypot(l_thumb[0] - l_palm[0], l_thumb[1] - l_palm[1])
            if dist_l_palm < 35:
                if now - getattr(self, 'last_thumb_delete_time', 0) > 0.6:
                    self.last_thumb_delete_time = now
                    self.last_phalanx_time = now
                    return '<DELETE_WORD>'

        if right_hand:
            r_thumb = right_hand['px'][4]
            r_palm = right_hand['px'][9]
            dist_r_palm = math.hypot(r_thumb[0] - r_palm[0], r_thumb[1] - r_palm[1])
            if dist_r_palm < 35:
                if now - getattr(self, 'last_thumb_delete_time', 0) > 0.6:
                    self.last_thumb_delete_time = now
                    self.last_phalanx_time = now
                    return '<DELETE_WORD>'

        # 2. Dual-Hand Gestures (Letters Y and Z)
        if left_hand and right_hand:
            l_px = left_hand['px']
            r_px = right_hand['px']

            # Letter Y: Left Thumb 4 + Right Thumb 4
            d_thumbs_2d = math.hypot(l_px[4][0] - r_px[4][0], l_px[4][1] - r_px[4][1])
            if d_thumbs_2d < 30:
                self.last_phalanx_time = now
                return 'y'

            # Letter Z: Left Index 8 + Right Index 8
            d_indices_2d = math.hypot(l_px[8][0] - r_px[8][0], l_px[8][1] - r_px[8][1])
            if d_indices_2d < 30:
                self.last_phalanx_time = now
                return 'z'

        candidates = []

        # 3. Left Hand Phalanx Mapping (Letters A to L, Space)
        if left_hand:
            l_px = left_hand['px']
            l_thumb_pt = l_px[4]

            # Dynamic threshold based on hand scale (distance from wrist 0 to middle MCP 9)
            hand_scale = math.hypot(l_px[9][0] - l_px[0][0], l_px[9][1] - l_px[0][1])
            dynamic_thresh = max(26.0, hand_scale * 0.35)

            left_map = [
                (8, 'a'), (7, 'b'), (6, 'c'),          # Index Finger (a, b, c)
                (12, 'd'), (11, 'e'), (10, 'f'),       # Middle Finger (d, e, f)
                (16, 'g'), (15, 'h'), (14, 'i'),       # Ring Finger (g, h, i)
                (20, 'j'), (19, 'k'), (18, 'l'),        # Pinky Finger (j, k, l)
                (5, ' ')                               # Knuckle 5 (Space)
            ]

            for lm_id, letter in left_map:
                joint_pt = l_px[lm_id]
                dist_2d = math.hypot(l_thumb_pt[0] - joint_pt[0], l_thumb_pt[1] - joint_pt[1])

                if dist_2d < dynamic_thresh:
                    candidates.append((dist_2d, letter))

        # 4. Right Hand Phalanx Mapping (Letters M to X, Space)
        if right_hand:
            r_px = right_hand['px']
            r_thumb_pt = r_px[4]

            hand_scale = math.hypot(r_px[9][0] - r_px[0][0], r_px[9][1] - r_px[0][1])
            dynamic_thresh = max(26.0, hand_scale * 0.35)

            right_map = [
                (8, 'm'), (7, 'n'), (6, 'o'),          # Index Finger (m, n, o)
                (12, 'p'), (11, 'q'), (10, 'r'),       # Middle Finger (p, q, r)
                (16, 's'), (15, 't'), (14, 'u'),       # Ring Finger (s, t, u)
                (20, 'v'), (19, 'w'), (18, 'x'),        # Pinky Finger (v, w, x)
                (5, ' ')                               # Knuckle 5 (Space)
            ]

            for lm_id, letter in right_map:
                joint_pt = r_px[lm_id]
                dist_2d = math.hypot(r_thumb_pt[0] - joint_pt[0], r_thumb_pt[1] - joint_pt[1])

                if dist_2d < dynamic_thresh:
                    candidates.append((dist_2d, letter))

        # 5. Nearest-Neighbor Selection
        if candidates:
            candidates.sort(key=lambda x: x[0])
            best_dist, best_letter = candidates[0]
            self.last_phalanx_time = now
            return best_letter

        return None
