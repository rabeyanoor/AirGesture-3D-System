import cv2
import time
import numpy as np
from src.config import COLOR_WHITE, COLOR_BLACK, COLOR_CYAN

class UIManager:
    def __init__(self):
        self.hover_start_time = None
        self.hovered_button = None
        self.dwell_time = 0.45  # Hover dwell duration for selection

    def draw_top_fps_badge(self, img, fps):
        """Draws top-left rounded FPS capsule badge matching video (e.g. 19 FPS / 28 FPS / 31 FPS)."""
        badge_str = f"{fps} FPS"
        overlay = img.copy()
        cv2.rectangle(overlay, (25, 22), (110, 58), (50, 50, 50), -1)
        cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
        cv2.rectangle(img, (25, 22), (110, 58), (140, 140, 140), 1, cv2.LINE_AA)
        cv2.putText(img, badge_str, (36, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)

    def draw_right_toolbar(self, img, active_mode, light_on):
        """
        Draws right vertical translucent toolbar panel with 3 buttons matching video_2026-08-15_16-15-48.mp4:
        1. Light Bulb 💡
        2. Notepad 📑
        3. Power ⏻
        """
        h, w, _ = img.shape
        panel_w = 80
        panel_h = 340
        px1 = w - panel_w - 25
        py1 = (h - panel_h) // 2
        px2 = w - 25
        py2 = py1 + panel_h

        # Panel Translucent Background
        overlay = img.copy()
        cv2.rectangle(overlay, (px1, py1), (px2, py2), (230, 235, 235), -1)
        cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
        cv2.rectangle(img, (px1, py1), (px2, py2), (190, 195, 195), 1, cv2.LINE_AA)

        buttons = self.get_button_rects(w, h)
        for btn in buttons:
            name, (bx1, by1, bx2, by2) = btn["name"], btn["rect"]

            is_active = (name == "NOTEPAD" and active_mode == "WRITE") or \
                        (name == "LIGHT" and light_on)

            btn_overlay = img.copy()
            bg_col = (200, 215, 185) if is_active else (255, 255, 255)
            cv2.rectangle(btn_overlay, (bx1, by1), (bx2, by2), bg_col, -1)
            cv2.addWeighted(btn_overlay, 0.85, img, 0.15, 0, img)
            cv2.rectangle(img, (bx1, by1), (bx2, by2), (160, 160, 160), 1, cv2.LINE_AA)

            cx = (bx1 + bx2) // 2
            cy = (by1 + by2) // 2

            if name == "LIGHT":
                # Light Bulb Icon 💡
                icon_col = (0, 165, 255) if light_on else (50, 50, 50)
                cv2.circle(img, (cx, cy - 3), 12, icon_col, 2, cv2.LINE_AA)
                cv2.rectangle(img, (cx - 5, cy + 9), (cx + 5, cy + 14), icon_col, -1)
                cv2.line(img, (cx - 3, cy + 17), (cx + 3, cy + 17), icon_col, 2)
            elif name == "NOTEPAD":
                # Notepad Document Icon 📑
                icon_col = (0, 100, 220) if active_mode == "WRITE" else (50, 50, 50)
                cv2.rectangle(img, (cx - 13, cy - 15), (cx + 13, cy + 15), icon_col, 2, cv2.LINE_AA)
                cv2.line(img, (cx - 8, cy - 7), (cx + 8, cy - 7), icon_col, 1)
                cv2.line(img, (cx - 8, cy), (cx + 8, cy), icon_col, 1)
                cv2.line(img, (cx - 8, cy + 7), (cx + 8, cy + 7), icon_col, 1)
            elif name == "POWER":
                # Power Icon ⏻ (Circle outline with vertical top line)
                icon_col = (40, 40, 180)
                cv2.ellipse(img, (cx, cy + 2), (12, 12), 0, 40, 320, icon_col, 2, cv2.LINE_AA)
                cv2.line(img, (cx, cy - 12), (cx, cy - 1), icon_col, 2, cv2.LINE_AA)

    def draw_notepad_card(self, img, text_content="Hello, My name is P Kha"):
        """Draws left-side translucent Notepad card with ruling lines and typed text overlay matching video."""
        h, w, _ = img.shape
        nx1, ny1 = 45, 80
        nw, nh = 470, 420
        nx2, ny2 = nx1 + nw, ny1 + nh

        # Card Translucent Overlay
        overlay = img.copy()
        cv2.rectangle(overlay, (nx1, ny1), (nx2, ny2), (240, 240, 240), -1)
        cv2.addWeighted(overlay, 0.40, img, 0.60, 0, img)
        cv2.rectangle(img, (nx1, ny1), (nx2, ny2), (190, 190, 190), 1, cv2.LINE_AA)

        # Title NOTEPAD
        cv2.putText(img, "NOTEPAD", (nx1 + 25, ny1 + 35), cv2.FONT_HERSHEY_DUPLEX, 0.65, (70, 70, 70), 1, cv2.LINE_AA)

        # Horizontal Notebook Ruling Lines
        line_spacing = 32
        lines_y = []
        for y in range(ny1 + 65, ny2 - 20, line_spacing):
            cv2.line(img, (nx1 + 20, y), (nx2 - 20, y), (220, 220, 220), 1, cv2.LINE_AA)
            lines_y.append(y)

        # Render Text onto Notepad Line 1 matching video_2026-08-15_16-15-48.mp4
        if text_content:
            first_line_y = lines_y[0] - 6
            cv2.putText(img, text_content, (nx1 + 30, first_line_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)

    def get_button_rects(self, w, h):
        panel_h = 340
        py1 = (h - panel_h) // 2
        bw, bh = 60, 60
        bx1 = w - 25 - 70
        
        return [
            {"name": "LIGHT", "rect": (bx1, py1 + 20, bx1 + bw, py1 + 20 + bh)},
            {"name": "NOTEPAD", "rect": (bx1, py1 + 115, bx1 + bw, py1 + 115 + bh)},
            {"name": "POWER", "rect": (bx1, py1 + 210, bx1 + bw, py1 + 210 + bh)},
        ]

    def check_interaction(self, img, landmarks_list, w, h, active_mode, light_on):
        """Checks finger hover dwell selection on right toolbar icons."""
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
