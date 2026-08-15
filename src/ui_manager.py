import cv2
import time
import numpy as np
from src.config import AVAILABLE_COLORS, COLOR_WHITE, COLOR_GREEN, COLOR_RED, COLOR_YELLOW, COLOR_DARK_GRAY

class UIManager:
    def __init__(self):
        self.hover_start_time = None
        self.hovered_button = None
        self.dwell_time = 0.6  # Seconds needed to trigger button dwell action

    def draw_ui(self, img, active_mode, current_color, is_erasing):
        """Renders modern glassmorphic control panel and floating buttons."""
        h, w, _ = img.shape

        # Draw semi-transparent header panel
        header = img.copy()
        cv2.rectangle(header, (0, 0), (w, 60), (30, 30, 30), -1)
        cv2.addWeighted(header, 0.6, img, 0.4, 0, img)
        
        # Header Title and Mode Indicator
        mode_str = f"MODE: {active_mode}"
        if is_erasing:
            mode_str += " (ERASER)"
        cv2.putText(img, "SPATIAL VISION AR", (20, 38), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, mode_str, (320, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # Floating Right Side Toolbar Buttons
        buttons = self.get_buttons(w)
        for btn in buttons:
            name, (bx1, by1, bx2, by2) = btn["name"], btn["rect"]
            
            # Highlight active button
            btn_bg = (60, 60, 60)
            text_color = (220, 220, 220)
            
            if active_mode == name or (name == "Eraser" and is_erasing):
                btn_bg = (0, 180, 255)
                text_color = (0, 0, 0)

            # Button semi-transparent box
            btn_overlay = img.copy()
            cv2.rectangle(btn_overlay, (bx1, by1), (bx2, by2), btn_bg, -1)
            cv2.addWeighted(btn_overlay, 0.7, img, 0.3, 0, img)
            cv2.rectangle(img, (bx1, by1), (bx2, by2), (100, 100, 100), 1, cv2.LINE_AA)

            # Center text in button
            text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            tx = bx1 + (bx2 - bx1 - text_size[0]) // 2
            ty = by1 + (by2 - by1 + text_size[1]) // 2
            cv2.putText(img, name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)

        # Color Selector Bar (shown when in WRITE mode)
        if active_mode == "WRITE":
            self._draw_color_palette(img, w, current_color)

    def _draw_color_palette(self, img, frame_w, current_color):
        """Renders color selection swatches."""
        start_x = 550
        y1, y2 = 12, 48
        swatch_w = 32

        for idx, (cname, color_bgr) in enumerate(AVAILABLE_COLORS):
            cx1 = start_x + idx * (swatch_w + 8)
            cx2 = cx1 + swatch_w
            
            cv2.rectangle(img, (cx1, y1), (cx2, y2), color_bgr, -1)
            
            # Border for selected color
            if color_bgr == current_color:
                cv2.rectangle(img, (cx1 - 2, y1 - 2), (cx2 + 2, y2 + 2), (0, 255, 255), 2)
            else:
                cv2.rectangle(img, (cx1, y1), (cx2, y2), (200, 200, 200), 1)

    def get_buttons(self, w):
        return [
            {"name": "WIREFRAME", "rect": (w - 130, 80, w - 20, 130)},
            {"name": "WRITE", "rect": (w - 130, 145, w - 20, 195)},
            {"name": "Eraser", "rect": (w - 130, 210, w - 20, 260)},
            {"name": "Clear", "rect": (w - 130, 275, w - 20, 325)},
            {"name": "OCR Text", "rect": (w - 130, 340, w - 20, 390)},
        ]

    def check_hover_interaction(self, img, landmarks_list, frame_w, active_mode, scribble):
        """
        Checks if index fingertip hovers over UI buttons and handles dwell timing / clicks.
        Returns updated active_mode, trigger_ocr_flag
        """
        if not landmarks_list:
            self.hover_start_time = None
            self.hovered_button = None
            return active_mode, False

        index_pt = landmarks_list[0][8][:2]
        ix, iy = index_pt

        # Check Color Palette click
        if active_mode == "WRITE" and 12 <= iy <= 48:
            start_x = 550
            swatch_w = 32
            for idx, (_, color_bgr) in enumerate(AVAILABLE_COLORS):
                cx1 = start_x + idx * (swatch_w + 8)
                cx2 = cx1 + swatch_w
                if cx1 <= ix <= cx2:
                    scribble.set_color(color_bgr)

        # Check Button Hover / Dwell
        buttons = self.get_buttons(frame_w)
        hovered_now = None

        for btn in buttons:
            name, (bx1, by1, bx2, by2) = btn["name"], btn["rect"]
            if bx1 <= ix <= bx2 and by1 <= iy <= by2:
                hovered_now = name
                break

        trigger_ocr = False

        if hovered_now:
            if self.hovered_button == hovered_now:
                elapsed = time.time() - self.hover_start_time
                
                # Render Dwell Progress Ring around index tip
                progress_angle = int((elapsed / self.dwell_time) * 360)
                cv2.ellipse(img, (ix, iy), (18, 18), 0, 0, progress_angle, (0, 255, 0), 3)

                if elapsed >= self.dwell_time:
                    # Action Triggered!
                    if hovered_now in ["WIREFRAME", "WRITE"]:
                        active_mode = hovered_now
                        scribble.set_eraser(False)
                    elif hovered_now == "Eraser":
                        scribble.set_eraser(True)
                    elif hovered_now == "Clear":
                        scribble.clear()
                    elif hovered_now == "OCR Text":
                        trigger_ocr = True

                    # Reset timer after activation
                    self.hover_start_time = time.time() + 0.5
            else:
                self.hovered_button = hovered_now
                self.hover_start_time = time.time()
        else:
            self.hovered_button = None
            self.hover_start_time = None

        return active_mode, trigger_ocr
