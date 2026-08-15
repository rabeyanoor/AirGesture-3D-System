import cv2
import time
import numpy as np
from src.config import COLOR_WHITE, COLOR_BLACK, COLOR_CYAN

class UIManager:
    def __init__(self):
        self.hover_start_time = None
        self.hovered_button = None
        self.dwell_time = 0.5

    def draw_top_fps_badge(self, img, fps):
        """Draws top-left rounded FPS badge like ( 33 FPS )."""
        badge_str = f"{fps} FPS"
        overlay = img.copy()
        
        # Rounded pill rectangle
        cv2.rectangle(overlay, (20, 20), (100, 50), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
        cv2.rectangle(img, (20, 20), (100, 50), (100, 100, 100), 1, cv2.LINE_AA)

        # Center text
        cv2.putText(img, badge_str, (32, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)

    def draw_right_toolbar(self, img, active_mode, light_on):
        """Draws right vertical toolbar panel with Light, Notepad, and Power icons."""
        h, w, _ = img.shape
        panel_w = 75
        panel_h = 320
        px1 = w - panel_w - 20
        py1 = (h - panel_h) // 2
        px2 = w - 20
        py2 = py1 + panel_h

        # Draw semi-transparent panel background with rounded appearance
        overlay = img.copy()
        cv2.rectangle(overlay, (px1, py1), (px2, py2), (230, 235, 235), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
        cv2.rectangle(img, (px1, py1), (px2, py2), (180, 190, 190), 1, cv2.LINE_AA)

        buttons = self.get_button_rects(w, h)
        for btn in buttons:
            name, (bx1, by1, bx2, by2) = btn["name"], btn["rect"]

            is_active = (name == "NOTEPAD" and active_mode == "WRITE") or \
                        (name == "LIGHT" and light_on)

            # Draw button background card
            btn_overlay = img.copy()
            bg_col = (180, 220, 255) if is_active else (255, 255, 255)
            cv2.rectangle(btn_overlay, (bx1, by1), (bx2, by2), bg_col, -1)
            cv2.addWeighted(btn_overlay, 0.85, img, 0.15, 0, img)
            cv2.rectangle(img, (bx1, by1), (bx2, by2), (150, 150, 150), 1, cv2.LINE_AA)

            # Draw Icon line-art
            cx = (bx1 + bx2) // 2
            cy = (by1 + by2) // 2

            if name == "LIGHT":
                # Light Bulb Icon
                icon_col = (0, 180, 255) if light_on else (40, 40, 40)
                cv2.circle(img, (cx, cy - 3), 11, icon_col, 2, cv2.LINE_AA)
                cv2.rectangle(img, (cx - 5, cy + 8), (cx + 5, cy + 13), icon_col, -1)
                cv2.line(img, (cx - 3, cy + 16), (cx + 3, cy + 16), icon_col, 2)
            elif name == "NOTEPAD":
                # Notepad Document Icon
                icon_col = (0, 100, 220) if active_mode == "WRITE" else (40, 40, 40)
                cv2.rectangle(img, (cx - 12, cy - 14), (cx + 12, cy + 14), icon_col, 2, cv2.LINE_AA)
                # Lined text inside icon
                cv2.line(img, (cx - 7, cy - 6), (cx + 7, cy - 6), icon_col, 1)
                cv2.line(img, (cx - 7, cy), (cx + 7, cy), icon_col, 1)
                cv2.line(img, (cx - 7, cy + 6), (cx + 7, cy + 6), icon_col, 1)
            elif name == "POWER":
                # Power Button Icon
                icon_col = (0, 0, 200)
                cv2.ellipse(img, (cx, cy + 2), (11, 11), 0, 40, 320, icon_col, 2, cv2.LINE_AA)
                cv2.line(img, (cx, cy - 11), (cx, cy - 1), icon_col, 2, cv2.LINE_AA)

    def draw_notepad_card(self, img):
        """Draws left-side translucent Notepad card with horizontal notebook ruling lines."""
        h, w, _ = img.shape
        nx1, ny1 = 40, 80
        nw, nh = 450, 400
        nx2, ny2 = nx1 + nw, ny1 + nh

        # Card Translucent Overlay
        overlay = img.copy()
        cv2.rectangle(overlay, (nx1, ny1), (nx2, ny2), (240, 240, 240), -1)
        cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
        cv2.rectangle(img, (nx1, ny1), (nx2, ny2), (200, 200, 200), 1, cv2.LINE_AA)

        # Title NOTEPAD
        cv2.putText(img, "NOTEPAD", (nx1 + 25, ny1 + 35), cv2.FONT_HERSHEY_DUPLEX, 0.6, (100, 100, 100), 1, cv2.LINE_AA)

        # Horizontal Notebook Ruling Lines
        line_spacing = 30
        for y in range(ny1 + 65, ny2 - 20, line_spacing):
            cv2.line(img, (nx1 + 20, y), (nx2 - 20, y), (200, 200, 200), 1, cv2.LINE_AA)

    def get_button_rects(self, w, h):
        panel_h = 320
        py1 = (h - panel_h) // 2
        bw, bh = 55, 55
        bx1 = w - 20 - 65
        
        return [
            {"name": "LIGHT", "rect": (bx1, py1 + 20, bx1 + bw, py1 + 20 + bh)},
            {"name": "NOTEPAD", "rect": (bx1, py1 + 105, bx1 + bw, py1 + 105 + bh)},
            {"name": "POWER", "rect": (bx1, py1 + 190, bx1 + bw, py1 + 190 + bh)},
        ]

    def check_interaction(self, img, landmarks_list, w, h, active_mode, light_on):
        """Checks finger collision / dwell on right toolbar icons."""
        if not landmarks_list:
            self.hover_start_time = None
            self.hovered_button = None
            return active_mode, light_on, False

        index_pt = landmarks_list[0][8][:2]
        ix, iy = index_pt

        buttons = self.get_button_rects(w, h)
        hovered_now = None

        for btn in buttons:
            name, (bx1, by1, bx2, by2) = btn["name"], btn["rect"]
            if bx1 <= ix <= bx2 and by1 <= iy <= by2:
                hovered_now = name
                break

        quit_signal = False

        if hovered_now:
            if self.hovered_button == hovered_now:
                elapsed = time.time() - self.hover_start_time
                progress_angle = int((elapsed / self.dwell_time) * 360)
                cv2.ellipse(img, (ix, iy), (16, 16), 0, 0, progress_angle, (0, 255, 0), 2)

                if elapsed >= self.dwell_time:
                    if hovered_now == "NOTEPAD":
                        active_mode = "WRITE" if active_mode != "WRITE" else "WIREFRAME"
                    elif hovered_now == "LIGHT":
                        light_on = not light_on
                    elif hovered_now == "POWER":
                        quit_signal = True
                    self.hover_start_time = time.time() + 0.5
            else:
                self.hovered_button = hovered_now
                self.hover_start_time = time.time()
        else:
            self.hovered_button = None
            self.hover_start_time = None

        return active_mode, light_on, quit_signal
