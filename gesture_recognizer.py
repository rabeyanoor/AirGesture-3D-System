"""
Gesture Recognizer Module
Detects hand gestures:
1. Fist Gesture (status)
2. Palm Tap / Touch (Space Bar Insertion)
3. Right Motion / Boundary Entry (Sidebar Trigger) + Sidebar Tile Taps
4. Finger Phalanx Keyboard (thumb-to-joint typing, A-Z + space , .)
5. Dual-Hand Grid Keypad Matrix Selection (A-Z Virtual Tap)
6. Index Pointing & Drawing Gesture
"""

import math
import time


def hand_scale(hand_dict):
    """Wrist (0) to middle MCP (9) distance in pixels - reference length for resolution-independent thresholds."""
    px = hand_dict['px']
    return math.hypot(px[9][0] - px[0][0], px[9][1] - px[0][1])


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

    def check_palm_touch(self, left_hand, right_hand, frame_w, frame_h, distance_ratio=0.3):
        """
        Check if right index tip touches left palm center (landmark 9).
        Threshold scales with the left hand's size so it works at any camera resolution / distance.
        """
        if not left_hand or not right_hand:
            return False, None

        l_palm = left_hand['px'][9]
        r_index_tip = right_hand['px'][8]

        distance_threshold = max(20.0, hand_scale(left_hand) * distance_ratio)
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
                    if dx > 35 * frame_w / 1280.0 and index_x > int(frame_w * 0.60):
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
        Index fingertip taps on the sidebar tiles (fires once on entry, not repeatedly while hovering):
        - Bulb: Dark theme toggle
        - Document: AR Notepad toggle
        - Liquid box / Rubik's cube: open or close that 3D box (only one open at a time)
        - Power: Clears the notepad text, closes open boxes and hides the sidebar
        Taps are ignored until the sidebar has finished sliding in, so the hand that swipes it open
        does not click a tile by accident.
        """
        if not sidebar_open or not ui_renderer.tile_rects:
            self._hovered_tile = None
            self._sidebar_ready_time = None
            return sidebar_open, None

        now = time.time()
        if not ui_renderer.sidebar_fully_open():
            return sidebar_open, None
        if getattr(self, '_sidebar_ready_time', None) is None:
            self._sidebar_ready_time = now
        if now - self._sidebar_ready_time < 0.3:
            return sidebar_open, None

        hovered = None
        for hand in hands_data:
            ix, iy = hand['px'][8][0], hand['px'][8][1]
            for name, (tx, ty, tw, th) in ui_renderer.tile_rects.items():
                if tx <= ix <= tx + tw and ty <= iy <= ty + th:
                    hovered = name
                    break
            if hovered:
                break

        prev = getattr(self, '_hovered_tile', None)
        self._hovered_tile = hovered
        if hovered is None or hovered == prev or now - getattr(self, 'last_click_time', 0) < 0.6:
            return sidebar_open, None

        self.last_click_time = now
        if hovered == 'bulb':
            ui_renderer.dark_mode = not ui_renderer.dark_mode
            return sidebar_open, "THEME_TOGGLED"
        if hovered == 'notepad':
            ui_renderer.show_notepad = not ui_renderer.show_notepad
            return sidebar_open, "NOTEPAD_TOGGLED"
        if hovered == 'liquid':
            # Only one 3D box is open at a time
            ui_renderer.show_liquid = not ui_renderer.show_liquid
            ui_renderer.show_rubik = False
            return sidebar_open, "LIQUID_BOX_TOGGLED"
        if hovered == 'rubik':
            ui_renderer.show_rubik = not ui_renderer.show_rubik
            ui_renderer.show_liquid = False
            return sidebar_open, "RUBIK_CUBE_TOGGLED"
        self._hovered_tile = None
        return False, "POWER_OFF"

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

        pinch_thresh = max(20.0, (hand_scale(left_hand) + hand_scale(right_hand)) * 0.5 * 0.3)

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

    # Thumb-tip targets per hand: (landmark id, character)
    LEFT_PHALANX_MAP = [
        (8, 'a'), (7, 'b'), (6, 'c'),      # Index finger
        (12, 'd'), (11, 'e'), (10, 'f'),   # Middle finger
        (16, 'g'), (15, 'h'), (14, 'i'),   # Ring finger
        (20, 'j'), (19, 'k'), (18, 'l'),   # Pinky finger
        (5, ' '),                          # Index knuckle -> space
        (17, ','),                         # Pinky knuckle -> comma
    ]
    RIGHT_PHALANX_MAP = [
        (8, 'm'), (7, 'n'), (6, 'o'),
        (12, 'p'), (11, 'q'), (10, 'r'),
        (16, 's'), (15, 't'), (14, 'u'),
        (20, 'v'), (19, 'w'), (18, 'x'),
        (5, ' '),
        (17, '.'),                         # Pinky knuckle -> period
    ]

    def phalanx_hover(self, hand, key_map, reach=0.6):
        """The joint the thumb tip is closest to (within reach * hand size): (landmark id, char, ratio) or None."""
        if not hand:
            return None
        px = hand['px']
        scale = max(hand_scale(hand), 1e-6)
        best = None
        for lm_id, char in list(key_map) + [(9, '<DELETE_WORD>')]:
            d = math.hypot(px[4][0] - px[lm_id][0], px[4][1] - px[lm_id][1]) / scale
            if d < reach and (best is None or d < best[2]):
                best = (lm_id, char, d)
        return best

    def _phalanx_contact(self, left_hand, right_hand):
        """Return the key currently being touched (no debounce), or None."""
        # 1. Thumb tip on palm centre (9) -> delete last word
        for hand in (left_hand, right_hand):
            if hand:
                px = hand['px']
                if math.hypot(px[4][0] - px[9][0], px[4][1] - px[9][1]) < hand_scale(hand) * 0.3:
                    return '<DELETE_WORD>'

        # 2. Dual-hand chords: thumbs together -> y, index tips together -> z
        if left_hand and right_hand:
            l_px, r_px = left_hand['px'], right_hand['px']
            thresh = max(18.0, (hand_scale(left_hand) + hand_scale(right_hand)) * 0.5 * 0.25)
            if math.hypot(l_px[4][0] - r_px[4][0], l_px[4][1] - r_px[4][1]) < thresh:
                return 'y'
            if math.hypot(l_px[8][0] - r_px[8][0], l_px[8][1] - r_px[8][1]) < thresh:
                return 'z'

        # 3. Thumb tip on a finger joint -> nearest joint wins
        candidates = []
        for hand, key_map in ((left_hand, self.LEFT_PHALANX_MAP), (right_hand, self.RIGHT_PHALANX_MAP)):
            if not hand:
                continue
            px = hand['px']
            thumb = px[4]
            thresh = max(14.0, hand_scale(hand) * 0.3)
            for lm_id, char in key_map:
                d = math.hypot(thumb[0] - px[lm_id][0], thumb[1] - px[lm_id][1])
                if d < thresh:
                    candidates.append((d, char))
        if candidates:
            return min(candidates)[1]
        return None

    def detect_phalanx_keyboard(self, left_hand, right_hand, frame_w, frame_h, dwell=0.15):
        """
        Hand keyboard (thumb tip touches finger joints):
        - Left hand: a-l on index/middle/ring/pinky (tip, DIP, PIP), index knuckle = space, pinky knuckle = ','
        - Right hand: m-x likewise, index knuckle = space, pinky knuckle = '.'
        - Both thumbs together: y | Both index tips together: z
        - Thumb on palm centre: delete one letter (keep holding to keep deleting)
        The thumb has to rest on a joint for `dwell` seconds (so brushing past a joint types nothing),
        and a key fires once per touch: lift the thumb before the next key.
        """
        now = time.time()
        key = self._phalanx_contact(left_hand, right_hand)

        # Release detection with a short grace period to ride out single-frame tracking dropouts
        if key is None:
            if now - getattr(self, '_phalanx_last_contact', 0) > 0.12:
                self._phalanx_latched = False
                self._phalanx_candidate = None
            return None
        self._phalanx_last_contact = now

        if getattr(self, '_phalanx_latched', False):
            # Held on DEL: repeat
            if key == '<DELETE_WORD>' and self._phalanx_fired == key and now - self._phalanx_fire_t > 0.5 \
                    and now - self.last_phalanx_time > 0.15:
                self.last_phalanx_time = now
                return key
            return None

        # Dwell: the same key must stay touched for a moment
        cand = getattr(self, '_phalanx_candidate', None)
        if cand is None or cand[0] != key:
            self._phalanx_candidate = (key, now)
            return None
        if now - cand[1] < dwell or now - getattr(self, 'last_phalanx_time', 0) < 0.2:
            return None

        self._phalanx_latched = True
        self._phalanx_fired, self._phalanx_fire_t = key, now
        self.last_phalanx_time = now
        return key
