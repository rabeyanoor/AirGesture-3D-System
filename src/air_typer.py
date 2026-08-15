import time
import numpy as np


def extract_xy(lm):
    """Safely extracts (x, y) from landmark object or (x,y,z) tuple."""
    if hasattr(lm, 'x') and hasattr(lm, 'y'):
        return np.array([lm.x, lm.y], dtype=np.float32)
    elif isinstance(lm, (tuple, list)) and len(lm) >= 2:
        return np.array([lm[0], lm[1]], dtype=np.float32)
    return np.array([0.0, 0.0], dtype=np.float32)


class AirTyper:
    """
    Two-Hand Air Typing Engine.

    LEFT HAND  = Virtual Keyboard (12 knuckle targets)
    RIGHT HAND = Pointer / Trigger (right index tip touches left knuckles)

    Normal Mode   (A-L) : right_index -> left hand 12 joints
    Shift Mode    (M-X) : triggered by right hand Peace/V gesture
    Y, Z                : left thumb tip & left thumb IP
    Space               : right index -> left palm (Landmark 9)
    Backspace           : right index -> left pinky tip (Landmark 20)
    """

    # Normal Mode: right index touches left joints -> A-L
    NORMAL_MAP = {
        8:  'A',   # Left Index Tip
        7:  'B',   # Left Index PIP
        6:  'C',   # Left Index MCP
        12: 'D',   # Left Middle Tip
        11: 'E',   # Left Middle PIP
        10: 'F',   # Left Middle MCP
        16: 'G',   # Left Ring Tip
        15: 'H',   # Left Ring PIP
        14: 'I',   # Left Ring MCP
        20: 'J',   # Left Pinky Tip
        19: 'K',   # Left Pinky PIP
        18: 'L',   # Left Pinky MCP
    }

    # Shift Mode: same joints -> M-X
    SHIFT_MAP = {
        8:  'M',
        7:  'N',
        6:  'O',
        12: 'P',
        11: 'Q',
        10: 'R',
        16: 'S',
        15: 'T',
        14: 'U',
        20: 'V',
        19: 'W',
        18: 'X',
    }

    TOUCH_THRESHOLD_NORM = 0.045   # Normalized space tight pinch distance
    COOLDOWN = 0.45

    def __init__(self):
        self.last_char_time = 0
        self.last_char = None
        self.shift_mode = False

    # ------------------------------------------------------------------ #
    # Gesture Helpers                                                       #
    # ------------------------------------------------------------------ #

    def _is_v_gesture(self, lm):
        """Right hand Peace/V gesture: index + middle up, ring + pinky folded."""
        if not lm or len(lm) < 21:
            return False
        index_up  = extract_xy(lm[8])[1]  < extract_xy(lm[6])[1]
        middle_up = extract_xy(lm[12])[1] < extract_xy(lm[10])[1]
        ring_dn   = extract_xy(lm[16])[1] > extract_xy(lm[14])[1]
        pinky_dn  = extract_xy(lm[20])[1] > extract_xy(lm[18])[1]
        return index_up and middle_up and ring_dn and pinky_dn

    def _is_fist(self, lm):
        """All four fingers folded -> fist / word erase."""
        if not lm or len(lm) < 21:
            return False
        return (
            extract_xy(lm[8])[1]  > extract_xy(lm[6])[1]  and
            extract_xy(lm[12])[1] > extract_xy(lm[10])[1] and
            extract_xy(lm[16])[1] > extract_xy(lm[14])[1] and
            extract_xy(lm[20])[1] > extract_xy(lm[18])[1]
        )

    def _hand_scale(self, lm):
        """Wrist-to-MiddleMCP distance as scale reference."""
        wrist  = extract_xy(lm[0])
        m_mcp  = extract_xy(lm[9])
        s = np.linalg.norm(wrist - m_mcp)
        return s if s > 0.01 else 0.15

    def _cooldown_ok(self, char):
        now = time.time()
        if char == self.last_char:
            return False
        if now - self.last_char_time < self.COOLDOWN:
            return False
        return True

    def _register(self, char):
        self.last_char = char
        self.last_char_time = time.time()

    # ------------------------------------------------------------------ #
    # Public API                                                            #
    # ------------------------------------------------------------------ #

    def check_shift(self, right_lm):
        """Toggle shift mode based on V-gesture of right hand."""
        self.shift_mode = self._is_v_gesture(right_lm)
        return self.shift_mode

    def check_word_erase(self, right_lm, text_buffer):
        """Right hand fist -> erase last word."""
        if self._is_fist(right_lm):
            now = time.time()
            if now - self.last_char_time > 0.7:
                self.last_char_time = now
                self.last_char = None
                words = text_buffer.strip().split()
                if words:
                    return ' '.join(words[:-1]), True
                elif text_buffer:
                    return '', True
        return text_buffer, False

    def detect(self, left_lm, right_lm):
        """
        Main detection: right index tip proximity to left hand joints.

        Returns (char, touch_point_normalized) or (None, None).
        """
        if not left_lm or not right_lm or len(left_lm) < 21 or len(right_lm) < 21:
            return None, None

        now = time.time()
        if now - self.last_char_time < self.COOLDOWN:
            # Allow reset after finger lifts
            r_idx = extract_xy(right_lm[8])
            closest_dist = min(
                np.linalg.norm(r_idx - extract_xy(left_lm[j]))
                for j in self.NORMAL_MAP
            )
            threshold = max(self.TOUCH_THRESHOLD_NORM, 0.18 * self._hand_scale(left_lm))
            if closest_dist >= threshold:
                self.last_char = None   # finger lifted, ready for next
            return None, None

        char_map = self.SHIFT_MAP if self.shift_mode else self.NORMAL_MAP
        r_idx = extract_xy(right_lm[8])
        threshold = max(self.TOUCH_THRESHOLD_NORM, 0.20 * self._hand_scale(left_lm))

        # --- Special keys ---
        # Space: right index -> left palm base (Landmark 9)
        left_palm  = extract_xy(left_lm[9])
        if np.linalg.norm(r_idx - left_palm) < threshold:
            if self._cooldown_ok('SPACE'):
                self._register('SPACE')
                return ' ', left_palm

        # Y: right index -> left thumb tip (Landmark 4)
        left_thumb = extract_xy(left_lm[4])
        if np.linalg.norm(r_idx - left_thumb) < threshold:
            if self._cooldown_ok('Y'):
                self._register('Y')
                return 'Y', left_thumb

        # Z: right index -> left thumb IP (Landmark 3)
        left_thumb_ip = extract_xy(left_lm[3])
        if np.linalg.norm(r_idx - left_thumb_ip) < threshold:
            if self._cooldown_ok('Z'):
                self._register('Z')
                return 'Z', left_thumb_ip

        # Backspace: right index -> left pinky tip? No — use separate gesture:
        # right middle tip (LM 12) -> left wrist (LM 0)
        r_mid = extract_xy(right_lm[12])
        left_wrist = extract_xy(left_lm[0])
        if np.linalg.norm(r_mid - left_wrist) < threshold:
            if self._cooldown_ok('BS'):
                self._register('BS')
                return '\b', left_wrist

        # --- Normal / Shift alphabet keys (A-L  or  M-X) ---
        candidates = []
        for joint_id, char in char_map.items():
            joint_pt = extract_xy(left_lm[joint_id])
            d = np.linalg.norm(r_idx - joint_pt)
            candidates.append((d, char, joint_pt))

        candidates.sort(key=lambda x: x[0])
        best_dist, best_char, best_pt = candidates[0]

        if best_dist < threshold and self._cooldown_ok(best_char):
            self._register(best_char)
            return best_char, best_pt

        if best_dist >= threshold:
            self.last_char = None   # finger lifted

        return None, None
