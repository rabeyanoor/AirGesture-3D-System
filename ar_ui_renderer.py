"""
AR UI Renderer Module
Renders the glassmorphic UI from the reference video (all sizes scale with the frame):
1. Minimal FPS pill badge (top-left); optional debug status line
2. Glassmorphic right sidebar with 5 action tiles (Bulb, Document, Liquid Box, Rubik's Cube, Power);
   active tiles are tinted
3. Toggleable lined "NOTEPAD" overlay with handwritten-style text
"""

import math
import os
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "assets", "fonts", "ArchitectsDaughter-Regular.ttf")


def _rounded_rect(img, pt1, pt2, color, radius, thickness=-1):
    """Draw a filled or outlined rounded rectangle."""
    x1, y1 = pt1
    x2, y2 = pt2
    r = int(max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2)))
    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
            cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)


def _blend_region(frame, overlay, x1, y1, x2, y2, alpha):
    """Alpha-blend overlay onto frame inside a clipped rectangle only (cheaper than full-frame blending)."""
    fh, fw = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(fw, x2), min(fh, y2)
    if x2 <= x1 or y2 <= y1:
        return frame
    frame[y1:y2, x1:x2] = cv2.addWeighted(overlay[y1:y2, x1:x2], alpha, frame[y1:y2, x1:x2], 1 - alpha, 0)
    return frame


class ARUIRenderer:
    TILE_ORDER = ('bulb', 'notepad', 'liquid', 'rubik', 'power')

    def __init__(self):
        self.sidebar_anim_x = 0.0  # Animation factor (0.0 = closed, 1.0 = fully open)
        self.show_notepad = False  # Notepad closed by default for clean initial camera screen
        self.dark_mode = False     # Dark mode / brightness overlay flag
        self.show_liquid = False   # 3D liquid box (sidebar tile)
        self.show_rubik = False    # Rubik's cube (sidebar tile)
        self.tile_rects = {}       # Bounding boxes for click interactions
        self._fonts = {}

    def _font(self, size):
        if size not in self._fonts:
            try:
                self._fonts[size] = ImageFont.truetype(FONT_PATH, size)
            except OSError:
                self._fonts[size] = ImageFont.load_default()
        return self._fonts[size]

    # ------------------------------------------------------------------
    # HUD
    # ------------------------------------------------------------------
    def draw_top_hud(self, frame, fps, mode_name="PHALANX KEYBOARD", hand_count=0, active_gesture="IDLE",
                     debug=False):
        """Minimal FPS pill (top-left). With debug=True a small status line shows mode / hands / gesture."""
        h, w, _ = frame.shape
        s = w / 1280.0
        font_scale = 0.4 * s
        fps_text = f"{int(round(fps))} FPS"
        tw, th = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]

        x1, y1 = int(14 * s), int(10 * s)
        x2, y2 = x1 + tw + int(26 * s), y1 + th + int(14 * s)
        overlay = frame.copy()
        _rounded_rect(overlay, (x1, y1), (x2, y2), (95, 95, 95), (y2 - y1) // 2)
        frame = _blend_region(frame, overlay, x1, y1, x2 + 1, y2 + 1, 0.7)
        cv2.putText(frame, fps_text, (x1 + int(13 * s), y2 - int(7 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (235, 235, 235), 1, cv2.LINE_AA)

        if debug:
            status = f"{mode_name} | HANDS: {hand_count} | {active_gesture}"
            cv2.putText(frame, status, (int(16 * s), h - int(16 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45 * s, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, status, (int(16 * s), h - int(16 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45 * s, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------
    def draw_sidebar(self, frame, open_state):
        """Animated right sidebar: Bulb (theme), Document (notepad), Liquid box, Rubik's cube, Power (clear/close)."""
        frame_h, frame_w, _ = frame.shape
        s = frame_w / 1280.0
        target_anim = 1.0 if open_state else 0.0
        self.sidebar_anim_x += (target_anim - self.sidebar_anim_x) * 0.3

        if self.sidebar_anim_x < 0.02:
            self.tile_rects = {}
            return frame

        margin = int(12 * s)
        sidebar_w = int(frame_w * 0.105)
        start_y = int(frame_h * 0.055)
        end_y = int(frame_h * 0.955)
        open_x = frame_w - margin - sidebar_w
        cur_x = int(frame_w + (open_x - frame_w) * self.sidebar_anim_x)

        overlay = frame.copy()
        _rounded_rect(overlay, (cur_x, start_y), (cur_x + sidebar_w, end_y), (218, 222, 216), int(14 * s))
        frame = _blend_region(frame, overlay, cur_x, start_y, cur_x + sidebar_w + 1, end_y + 1, 0.42)

        # Top highlight bar
        cv2.line(frame, (cur_x + int(sidebar_w * 0.12), start_y + int(4 * s)),
                 (cur_x + int(sidebar_w * 0.88), start_y + int(4 * s)), (255, 255, 255), max(1, int(2 * s)),
                 cv2.LINE_AA)

        tile = int(frame_w * 0.05)
        tile_x = cur_x + (sidebar_w - tile) // 2
        first_y = start_y + int(frame_h * 0.085)
        gap = int(frame_h * 0.12)
        icon_color = (60, 60, 60)
        lw = max(1, int(round(1.6 * s)))

        active_flags = {
            'bulb': self.dark_mode,
            'notepad': self.show_notepad,
            'liquid': self.show_liquid,
            'rubik': self.show_rubik,
        }

        overlay = frame.copy()
        tiles = []
        self.tile_rects = {}
        for i, name in enumerate(self.TILE_ORDER):
            ty = first_y + i * gap
            self.tile_rects[name] = (tile_x, ty, tile, tile)
            fill = (205, 195, 170) if active_flags.get(name) else (238, 240, 238)
            cv2.rectangle(overlay, (tile_x, ty), (tile_x + tile, ty + tile), fill, -1)
            tiles.append((name, tile_x, ty))
        last_y = first_y + (len(self.TILE_ORDER) - 1) * gap
        frame = _blend_region(frame, overlay, tile_x, first_y, tile_x + tile + 1, last_y + tile + 1, 0.55)

        for name, tx, ty in tiles:
            cv2.rectangle(frame, (tx, ty), (tx + tile, ty + tile), (250, 250, 250), 1, cv2.LINE_AA)
            cx, cy = tx + tile // 2, ty + tile // 2
            u = tile / 64.0
            if name == 'bulb':
                cv2.circle(frame, (cx, cy - int(5 * u)), int(12 * u), icon_color, lw, cv2.LINE_AA)
                cv2.line(frame, (cx - int(6 * u), cy + int(10 * u)), (cx + int(6 * u), cy + int(10 * u)),
                         icon_color, lw, cv2.LINE_AA)
                cv2.line(frame, (cx - int(4 * u), cy + int(15 * u)), (cx + int(4 * u), cy + int(15 * u)),
                         icon_color, lw, cv2.LINE_AA)
            elif name == 'notepad':
                cv2.rectangle(frame, (cx - int(14 * u), cy - int(18 * u)), (cx + int(14 * u), cy + int(18 * u)),
                              icon_color, lw, cv2.LINE_AA)
                for dy in (-8, 0, 8):
                    cv2.line(frame, (cx - int(9 * u), cy + int(dy * u)), (cx + int(9 * u), cy + int(dy * u)),
                             icon_color, lw, cv2.LINE_AA)
            elif name == 'liquid':
                # Wireframe cube with liquid in the lower half
                o = int(7 * u)
                front = np.array([[cx - 15 * u, cy - 9 * u], [cx + 9 * u, cy - 9 * u],
                                  [cx + 9 * u, cy + 15 * u], [cx - 15 * u, cy + 15 * u]], dtype=np.int32)
                back = front + np.array([o, -o], dtype=np.int32)
                liquid = np.array([front[3], front[2], [front[2][0], cy + 3 * u], [front[3][0], cy + 3 * u]],
                                  dtype=np.int32)
                cv2.fillPoly(frame, [liquid], (180, 105, 255), cv2.LINE_AA)
                cv2.polylines(frame, [front], True, icon_color, lw, cv2.LINE_AA)
                cv2.polylines(frame, [back], True, icon_color, lw, cv2.LINE_AA)
                for a, b in zip(front, back):
                    cv2.line(frame, tuple(int(v) for v in a), tuple(int(v) for v in b), icon_color, lw, cv2.LINE_AA)
            elif name == 'rubik':
                cell = int(9 * u)
                colors = [(255, 255, 255), (40, 30, 210), (60, 175, 30),
                          (0, 215, 255), (0, 130, 255), (200, 90, 10),
                          (40, 30, 210), (255, 255, 255), (0, 215, 255)]
                x0, y0 = cx - int(cell * 1.5) - 1, cy - int(cell * 1.5) - 1
                cv2.rectangle(frame, (x0 - 2, y0 - 2), (x0 + cell * 3 + 3, y0 + cell * 3 + 3), (30, 30, 30), -1)
                for k, col in enumerate(colors):
                    gx, gy = x0 + (k % 3) * (cell + 1), y0 + (k // 3) * (cell + 1)
                    cv2.rectangle(frame, (gx, gy), (gx + cell - 1, gy + cell - 1), col, -1)
            else:  # power
                cv2.ellipse(frame, (cx, cy + int(2 * u)), (int(15 * u), int(15 * u)), -90, 35, 325,
                            icon_color, lw, cv2.LINE_AA)
                cv2.line(frame, (cx, cy - int(16 * u)), (cx, cy + int(1 * u)), icon_color, lw, cv2.LINE_AA)

        return frame

    def sidebar_fully_open(self):
        return self.sidebar_anim_x > 0.95

    # ------------------------------------------------------------------
    # Notepad
    # ------------------------------------------------------------------
    def draw_notepad_overlay(self, frame, text_buffer, caret=False):
        """Lined glass notepad (left side) with handwritten-style text, only when toggled on."""
        if not self.show_notepad:
            return frame

        frame_h, frame_w, _ = frame.shape
        s = frame_w / 1280.0
        x1, y1 = int(frame_w * 0.037), int(frame_h * 0.085)
        x2, y2 = int(frame_w * 0.505), int(frame_h * 0.65)

        overlay = frame.copy()
        _rounded_rect(overlay, (x1, y1), (x2, y2), (225, 228, 226), int(16 * s))
        frame = _blend_region(frame, overlay, x1, y1, x2 + 1, y2 + 1, 0.42)

        # Rule lines
        pad_x = int((x2 - x1) * 0.05)
        first_rule = y1 + int(frame_h * 0.105)
        spacing = int(frame_h * 0.048)
        rules = list(range(first_rule, y2 - int(spacing * 0.6), spacing))
        for ry in rules:
            cv2.line(frame, (x1 + pad_x, ry), (x2 - pad_x, ry), (250, 250, 250), max(1, int(round(1.4 * s))),
                     cv2.LINE_AA)

        # Text via PIL for the handwritten font (only the notepad ROI is converted)
        roi = frame[y1:y2, x1:x2]
        img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)

        header_font = self._font(max(10, int(16 * s)))
        draw.text((pad_x, int(frame_h * 0.03)), "NOTEPAD", font=header_font, fill=(255, 255, 255))

        text_font = self._font(max(12, int(frame_h * 0.036)))
        text_x = pad_x + int((x2 - x1) * 0.06)
        max_w = (x2 - x1) - text_x - pad_x
        # Blinking caret while typing so it is clear where the next letter goes
        shown_text = text_buffer + ("|" if caret and int(time.time() * 2) % 2 == 0 else "")
        lines = self._wrap(shown_text, text_font, max_w)
        # Keep the latest lines visible when the text is longer than the page
        for idx, line in enumerate(lines[-len(rules):] if rules else []):
            baseline = rules[idx] - y1 - int(spacing * 0.18)
            draw.text((text_x, baseline), line, font=text_font, fill=(15, 15, 15), anchor="ls")

        roi[:] = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        return frame

    @staticmethod
    def _wrap(text, font, max_w):
        lines = []
        for paragraph in text.split('\n'):
            cur = ""
            for word in paragraph.split(' '):
                cand = f"{cur} {word}" if cur else word
                if font.getlength(cand) <= max_w or not cur:
                    cur = cand
                else:
                    lines.append(cur)
                    cur = word
            lines.append(cur)
        return lines

    def draw_hand_keys(self, frame, hands_maps, flash=None):
        """
        Hand keyboard: a letter tag on every finger joint of each hand.
        hands_maps: [(hand, key_map, hover)], hover = (landmark id, char, distance ratio) or None.
        flash: (hand label, landmark id) of the key just typed.
        """
        h, w = frame.shape[:2]
        for hand, key_map, hover in hands_maps:
            px = hand['px']
            scale = math.hypot(px[9][0] - px[0][0], px[9][1] - px[0][1])
            r = int(max(9, min(22, scale * 0.13)))
            fs = r / 22.0
            items = list(key_map) + [(9, '<DELETE_WORD>')]
            for lm_id, char in items:
                x, y = int(px[lm_id][0]), int(px[lm_id][1])
                label = {' ': 'SPC', '<DELETE_WORD>': 'DEL'}.get(char, char.upper())
                is_hover = hover is not None and hover[0] == lm_id
                is_flash = flash is not None and flash == (hand['label'], lm_id)
                fill = (180, 105, 255) if (is_flash or is_hover) else (245, 245, 245)
                rr = int(r * (1.35 if is_hover else 1.0)) + (4 if len(label) > 1 else 0)
                cv2.circle(frame, (x, y), rr, fill, -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), rr, (40, 40, 45), 1, cv2.LINE_AA)
                scale_txt = fs * (0.5 if len(label) > 1 else 0.75) * (1.2 if is_hover else 1.0)
                tw, th = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale_txt, 2)[0]
                cv2.putText(frame, label, (x - tw // 2, y + th // 2), cv2.FONT_HERSHEY_SIMPLEX, scale_txt,
                            (255, 255, 255) if (is_flash or is_hover) else (35, 35, 40), 2, cv2.LINE_AA)
            # The thumb tip is the "pen"
            tx, ty = int(px[4][0]), int(px[4][1])
            cv2.circle(frame, (tx, ty), max(5, r // 2), (80, 230, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (tx, ty), max(5, r // 2) + 2, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    def apply_theme_filter(self, frame):
        """Dims the scene when the bulb tile toggles dark mode."""
        if self.dark_mode:
            frame = cv2.convertScaleAbs(frame, alpha=0.55, beta=0)
        return frame
