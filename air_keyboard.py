"""
Air Keyboard Module
On-screen glass QWERTY keyboard for the AR notepad.

- Point the index fingertip at a key (a ring follows it) and pinch thumb + index to press it.
- DEL deletes one character; keep the pinch held on DEL to keep deleting.
- SPACE, ENTER and CLEAR work the same way.
"""

import math
import time

import cv2

ROWS = [
    [("q", 1), ("w", 1), ("e", 1), ("r", 1), ("t", 1), ("y", 1), ("u", 1), ("i", 1), ("o", 1), ("p", 1)],
    [("a", 1), ("s", 1), ("d", 1), ("f", 1), ("g", 1), ("h", 1), ("j", 1), ("k", 1), ("l", 1), ("'", 1)],
    [("z", 1), ("x", 1), ("c", 1), ("v", 1), ("b", 1), ("n", 1), ("m", 1), (",", 1), (".", 1), ("?", 1)],
    [("<CLEAR>", 1.5), (" ", 4.5), ("<BKSP>", 2.5), ("<ENTER>", 1.5)],
]
LABELS = {"<CLEAR>": "CLEAR", " ": "SPACE", "<BKSP>": "DEL", "<ENTER>": "ENTER"}
REPEATABLE = {"<BKSP>"}


class AirKeyboard:
    def __init__(self):
        self.keys = []            # [(key, (x1, y1, x2, y2))]
        self.rect = None
        self._pinching = {}       # hand label -> bool (hysteresis)
        self._held = {}           # hand label -> (key, press_time, last_repeat)
        self._hover = {}          # hand label -> (point, key, pinching)
        self._flash = {}          # key -> time pressed (visual feedback)
        self._prev_key = {}       # hand label -> key hovered in the previous frame

    # ------------------------------------------------------------------
    def layout(self, w, h):
        x1, x2 = int(0.037 * w), int(0.80 * w)
        y1, y2 = int(0.675 * h), int(0.975 * h)
        self.rect = (x1, y1, x2, y2)
        gap = max(3, int(w / 320))
        row_h = (y2 - y1 - gap * (len(ROWS) + 1)) / len(ROWS)
        self.keys = []
        for r, row in enumerate(ROWS):
            units = sum(u for _, u in row)
            unit_w = (x2 - x1 - gap * (len(row) + 1)) / units
            x = x1 + gap
            y = y1 + gap + r * (row_h + gap)
            for key, u in row:
                kw = unit_w * u
                self.keys.append((key, (int(x), int(y), int(x + kw), int(y + row_h))))
                x += kw + gap

    def contains(self, pt, margin=0):
        if self.rect is None:
            return False
        x1, y1, x2, y2 = self.rect
        return x1 - margin <= pt[0] <= x2 + margin and y1 - margin <= pt[1] <= y2 + margin

    def key_at(self, pt):
        for key, (x1, y1, x2, y2) in self.keys:
            if x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2:
                return key
        return None

    @staticmethod
    def cursor(hand):
        """Index fingertip: stays put while the thumb closes in for the pinch."""
        return hand['px'][8][0], hand['px'][8][1]

    def owns(self, hand):
        """True when this hand is working the keyboard (so 3D boxes should ignore it)."""
        return self.contains(self.cursor(hand), margin=10)

    # ------------------------------------------------------------------
    def update(self, hands, w, h, now=None):
        """Returns the list of keys pressed this frame."""
        now = time.time() if now is None else now
        self.layout(w, h)
        pressed = []
        seen = set()
        self._hover = {}
        for hand in hands:
            label = hand['label']
            seen.add(label)
            px = hand['px']
            scale = math.hypot(px[9][0] - px[0][0], px[9][1] - px[0][1])
            d = math.hypot(px[4][0] - px[8][0], px[4][1] - px[8][1])
            was = self._pinching.get(label, False)
            pinch = d < scale * (0.6 if was else 0.4)
            self._pinching[label] = pinch
            pt = self.cursor(hand)
            key = self.key_at(pt)
            self._hover[label] = (pt, key, pinch)

            if pinch and not was:
                # Press on the pinch edge; prefer the key aimed at just before the fingers closed
                target = self._prev_key.get(label) or key
                if target is not None:
                    pressed.append(target)
                    self._flash[target] = now
                    self._held[label] = (target, now, now)
            elif pinch and label in self._held:
                hkey, t0, last = self._held[label]
                if hkey in REPEATABLE and now - t0 > 0.5 and now - last > 0.1:
                    pressed.append(hkey)
                    self._flash[hkey] = now
                    self._held[label] = (hkey, t0, now)
            elif not pinch:
                self._held.pop(label, None)
            if not pinch:
                self._prev_key[label] = key
        for label in list(self._pinching):
            if label not in seen:
                self._pinching.pop(label)
                self._held.pop(label, None)
                self._prev_key.pop(label, None)
        return pressed

    # ------------------------------------------------------------------
    def draw(self, frame, now=None):
        now = time.time() if now is None else now
        h, w = frame.shape[:2]
        if not self.keys:
            self.layout(w, h)
        hovered = {k for _, k, _ in self._hover.values() if k}
        x1, y1, x2, y2 = self.rect
        roi = frame[y1:y2, x1:x2]
        overlay = roi.copy()
        cv2.rectangle(overlay, (0, 0), (x2 - x1, y2 - y1), (226, 228, 225), -1)
        for key, (kx1, ky1, kx2, ky2) in self.keys:
            flash = now - self._flash.get(key, -9) < 0.18
            col = (180, 105, 255) if flash else (255, 255, 255) if key in hovered else (240, 242, 240)
            cv2.rectangle(overlay, (kx1 - x1, ky1 - y1), (kx2 - x1, ky2 - y1), col, -1)
        roi[:] = cv2.addWeighted(overlay, 0.55, roi, 0.45, 0)

        fs = max(0.45, (y2 - y1) / 4 / 60.0)
        for key, (kx1, ky1, kx2, ky2) in self.keys:
            if key in hovered:
                cv2.rectangle(frame, (kx1, ky1), (kx2, ky2), (255, 255, 255), max(2, int(w / 640)), cv2.LINE_AA)
            label = LABELS.get(key, key.upper())
            scale = fs * (0.62 if len(label) > 1 else 1.0)
            tw, th = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0]
            cv2.putText(frame, label, ((kx1 + kx2 - tw) // 2, (ky1 + ky2 + th) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (40, 40, 45), 2, cv2.LINE_AA)

        # Pinch cursor for hands over the keyboard
        r = max(6, int(w / 128))
        for pt, key, pinch in self._hover.values():
            if not self.contains(pt, margin=40):
                continue
            c = (int(pt[0]), int(pt[1]))
            if pinch:
                cv2.circle(frame, c, r, (80, 230, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, c, r + 3, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, c, r + 5, (40, 40, 45), 1, cv2.LINE_AA)
        return frame
