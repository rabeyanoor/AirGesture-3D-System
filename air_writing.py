"""
Air Writing Module
Write letters in the air and they are typed into the notepad.

- Hold 2 fingers together (index + middle, ring and pinky folded) or 3 fingers together (thumb + index +
  middle, like holding a pen) and write; spread the fingers and the pen lifts. Lift between strokes of
  multi-stroke letters (T, H, X, ...) without pausing. A short pause (~0.5 s) ends the letter and types it
  (a stroke started far to the right also starts the next letter at once), so words flow letter by letter.
- A horizontal line left -> right = space, right -> left = delete one letter, a quick tap = "."

Recognition: $P point-cloud recogniser (Vatavu, Anthony & Wobbrock, 2012) against the A-Z templates in
web/letter_templates.json (capital A-Z and small a-z shapes) - stroke order and direction do not matter,
multi-stroke letters are fine.
"""

import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "letter_templates.json")
N_POINTS = 32


# ----------------------------------------------------------------------------- $P recogniser
def _resample(strokes, n=N_POINTS):
    """Resample multi-stroke input to n points spread evenly along the total ink length."""
    pts = [(np.asarray(s, dtype=np.float64), sid) for sid, s in enumerate(strokes) if len(s) > 0]
    total = sum(np.linalg.norm(np.diff(p, axis=0), axis=1).sum() for p, _ in pts if len(p) > 1)
    if total == 0:
        allp = np.vstack([p for p, _ in pts])
        return np.repeat(allp[:1], n, axis=0)
    step = total / (n - 1)
    out, acc = [pts[0][0][0]], 0.0
    for p, _ in pts:
        prev = p[0]
        for q in p[1:]:
            d = np.linalg.norm(q - prev)
            while acc + d >= step and d > 0:
                t = (step - acc) / d
                new = prev + t * (q - prev)
                out.append(new)
                prev, d, acc = new, np.linalg.norm(q - new), 0.0
            acc += d
            prev = q
    while len(out) < n:
        out.append(pts[-1][0][-1])
    return np.array(out[:n])


def _normalize(strokes):
    """Resample, then scale into a unit square so wide / narrow handwriting matches the same template.
    Nearly one-dimensional ink (an I, a dash) keeps its aspect ratio instead of being blown up."""
    p = _resample(strokes)
    mn, mx = p.min(0), p.max(0)
    w, h = max(mx[0] - mn[0], 1e-6), max(mx[1] - mn[1], 1e-6)
    if min(w, h) / max(w, h) < 0.3:
        w = h = max(w, h)
    p = (p - mn) / np.array([w, h])
    return p - p.mean(0)


def _cloud_distance(d, start):
    """Greedy weighted matching over a precomputed distance matrix d (rows: points of a, cols: points of b)."""
    n = len(d)
    matched = np.zeros(n, dtype=bool)
    total, i = 0.0, start
    for k in range(n):
        row = np.where(matched, np.inf, d[i])
        j = int(np.argmin(row))
        matched[j] = True
        total += (1 - k / n) * row[j]
        i = (i + 1) % n
    return total


def _greedy_match(a, b):
    n = len(a)
    step = max(1, int(n ** 0.5))
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    dt = d.T.copy()
    return min(min(_cloud_distance(d, i), _cloud_distance(dt, i)) for i in range(0, n, step))


class PRecognizer:
    def __init__(self, path=TEMPLATE_PATH):
        with open(path) as f:
            self.templates = [(t["char"], _normalize(t["strokes"])) for t in json.load(f)]

    def recognize(self, strokes):
        pts = _normalize(strokes)
        char, score = min(((c, _greedy_match(pts, tp)) for c, tp in self.templates), key=lambda x: x[1])
        # Point clouds blur D and O together: D is the round letter with a straight left side
        if char in ("O", "Q", "C", "D", "o", "c"):
            straight = _straight_left(strokes)
            if straight and char in ("O", "Q", "C", "o", "c"):
                char = "D"
            elif not straight and char == "D":
                char = "O"
        # A single plain vertical line is "l" (a dotted line is "i"; "I" alone becomes "I" via auto-caps)
        if char == "I" and len(strokes) == 1:
            char = "l"
        return char, score


def _straight_left(strokes):
    """True when the ink's top-left and bottom-left corners reach its left edge (a D; an O is round there)."""
    p = np.array([q for s in strokes for q in s], dtype=np.float64)
    mn, mx = p.min(0), p.max(0)
    bw, bh = max(mx[0] - mn[0], 1e-6), max(mx[1] - mn[1], 1e-6)
    top = p[p[:, 1] < mn[1] + 0.1 * bh]
    bottom = p[p[:, 1] > mx[1] - 0.1 * bh]
    if len(top) == 0 or len(bottom) == 0:
        return False
    return top[:, 0].min() - mn[0] < 0.18 * bw and bottom[:, 0].min() - mn[0] < 0.18 * bw


# ----------------------------------------------------------------------------- pen + gestures
def _extended(px, tip, pip, mcp):
    d_tip = math.hypot(px[tip][0] - px[mcp][0], px[tip][1] - px[mcp][1])
    d_pip = math.hypot(px[pip][0] - px[mcp][0], px[pip][1] - px[mcp][1])
    return d_pip > 0 and d_tip / d_pip > 1.3


def pen_pose(px, was_down=False):
    """
    Writing pose: 2 or 3 fingertips held together.
      - two fingers: index + middle tips together, ring and pinky folded
      - three fingers: thumb + index + middle tips together (like holding a pen)
    Returns (pen_down, pen_point, hover_point). Hysteresis: once down, fingers may part a little more.
    """
    scale = max(math.hypot(px[9][0] - px[0][0], px[9][1] - px[0][1]), 1e-6)
    d = lambda a, b: math.hypot(px[a][0] - px[b][0], px[a][1] - px[b][1]) / scale
    k = 1.5 if was_down else 1.0
    two = d(8, 12) < 0.25 * k and not _extended(px, 16, 14, 13) and not _extended(px, 20, 18, 17)
    three = d(4, 8) < 0.22 * k and d(4, 12) < 0.22 * k and d(8, 12) < 0.22 * k
    tip = ((px[8][0] + px[12][0]) / 2.0, (px[8][1] + px[12][1]) / 2.0)
    return two or three, tip, tip


class AirWriter:
    PAUSE = 0.55       # seconds of pen-up that end a letter
    DEBOUNCE = 2       # frames a pen change must persist (tracking jitter never breaks a stroke)

    def __init__(self):
        self.recognizer = PRecognizer()
        self.strokes = []          # finished strokes of the current letter
        self.current = None        # stroke being drawn
        self.pen_label = None
        self._pen = {}             # hand label -> [pen down?, frames the raw state has disagreed]
        self.hover = {}            # hand label -> (point, pen down) for the on-screen pen cursor
        self.last_pen_up = 0.0
        self.popup = None          # (text, (x, y), time) - shows what was recognised
        # Recognition takes ~0.1-0.2 s: run it off the video thread so the picture never freezes
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._pending = []         # futures of letters being recognised, in writing order

    def _pen_state(self, hand):
        st = self._pen.setdefault(hand['label'], [False, 0])
        raw, pt, hover = pen_pose(hand['px'], st[0])
        if raw != st[0]:
            st[1] += 1
            if st[1] >= self.DEBOUNCE:
                st[0], st[1] = raw, 0
        else:
            st[1] = 0
        return st[0], pt, hover

    @property
    def writing(self):
        return self.current is not None or bool(self.strokes) or bool(self._pending)

    def _letter_box(self):
        p = np.array([q for s in self.strokes for q in s], dtype=np.float64)
        return p.min(0), p.max(0)

    def update(self, hands, w, h, now=None, blocked=None):
        """
        Returns the typed key ('a'-'z', ' ', '.', '<BKSP>') when a letter is finished, else None.
        blocked(pt) -> True for screen points that belong to something else (e.g. a 3D box).
        """
        now = time.time() if now is None else now
        pen = None
        self.hover = {}
        for hand in hands:
            down, pt, hover = self._pen_state(hand)
            self.hover[hand['label']] = (hover, down)
            if self.pen_label is not None and hand['label'] != self.pen_label:
                continue
            if down and pen is None:
                pen = (hand['label'], pt)
        seen = {hd['label'] for hd in hands}
        for label in list(self._pen):
            if label not in seen:
                self._pen.pop(label)

        if pen is not None:
            label, pt = pen
            if self.current is None:
                if blocked is not None and blocked(pt) and not self.writing:
                    return None
                # Writing a word: a stroke that starts clearly to the right (or left) of the letter
                # being written begins the next letter, so the previous one is typed right away
                if self.strokes:
                    mn, mx = self._letter_box()
                    size = max(mx[1] - mn[1], mx[0] - mn[0], 0.04 * h)
                    if pt[0] > mx[0] + 0.8 * size or pt[0] < mn[0] - 0.8 * size:
                        self._finish(w, h)
                self.current = [pt]
                self.pen_label = label
            else:
                last = self.current[-1]
                if math.hypot(pt[0] - last[0], pt[1] - last[1]) > 1.5:
                    self.current.append(pt)
            return self._ready()

        # Pen up
        if self.current is not None:
            self.strokes.append(self.current)
            self.current = None
            self.last_pen_up = now
        if self.strokes and now - self.last_pen_up > self.PAUSE:
            self._finish(w, h)
            self.pen_label = None
        if not self.strokes:
            self.pen_label = None
        return self._ready()

    def _finish(self, w, h):
        """Queue the current letter for recognition (off the video thread)."""
        strokes, self.strokes = self.strokes, []
        self._pending.append(self._pool.submit(self._classify, strokes, w, h))

    def _classify(self, strokes, w, h):
        allp = np.array([p for s in strokes for p in s], dtype=np.float64)
        mn, mx = allp.min(0), allp.max(0)
        bw, bh = mx - mn
        diag = math.hypot(w, h)
        centre = (float((mn[0] + mx[0]) / 2), float(mn[1]))
        if max(bw, bh) < 0.03 * diag:
            key = '.'                                   # quick tap
        elif len(strokes) == 1 and bw > 2.5 * max(bh, 1) and bw > 0.08 * diag:
            s = strokes[0]
            key = ' ' if s[-1][0] > s[0][0] else '<BKSP>'  # horizontal swipe
        else:
            char, score = self.recognizer.recognize(strokes)
            key = char.lower()
        label = {' ': 'SPACE', '<BKSP>': 'DEL', '.': '.'}.get(key, key.upper())
        return key, (label, centre)

    def _ready(self):
        """The oldest finished letter, in writing order (None while it is still being recognised)."""
        if self._pending and self._pending[0].done():
            key, (label, centre) = self._pending.pop(0).result()
            self.popup = (label, centre, time.time())
            return key
        return None

    def clear(self):
        self.strokes, self.current, self.pen_label = [], None, None
        self._pen, self.hover = {}, {}
        self._pending = []

    # ------------------------------------------------------------------
    def draw(self, frame):
        """Glowing pink ink for the letter being written, and the recognised letter popping up."""
        h, w = frame.shape[:2]
        thick = max(3, int(w / 220))
        # Pen cursor: hollow ring while the fingers are apart, filled pink while writing
        r = max(6, int(w / 140))
        for (x, y), down in self.hover.values():
            c = (int(x), int(y))
            if down:
                cv2.circle(frame, c, r, (180, 105, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, c, r + 3, (255, 255, 255), 2, cv2.LINE_AA)
        for s in self.strokes + ([self.current] if self.current else []):
            if len(s) < 2:
                if s:
                    cv2.circle(frame, (int(s[0][0]), int(s[0][1])), thick, (180, 105, 255), -1, cv2.LINE_AA)
                continue
            pts = np.array(s, dtype=np.int32)
            cv2.polylines(frame, [pts], False, (150, 60, 230), thick + 5, cv2.LINE_AA)
            cv2.polylines(frame, [pts], False, (230, 200, 255), thick, cv2.LINE_AA)
        if self.popup and time.time() - self.popup[2] < 0.8:
            text, (x, y), t0 = self.popup
            a = 1 - (time.time() - t0) / 0.8
            scale = (1.6 if len(text) == 1 else 0.9) * w / 1280
            tw, th = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)[0]
            org = (int(x - tw / 2), int(y - 20 - (1 - a) * 30))
            cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (40, 20, 90), 6, cv2.LINE_AA)
            cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (220, 180, 255), 3, cv2.LINE_AA)
        return frame
