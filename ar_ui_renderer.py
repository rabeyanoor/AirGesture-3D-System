"""
AR UI Renderer Module
Renders high-quality glassmorphic UI elements matching user screenshot:
1. Clean minimal FPS Pill Badge (Top-Left)
2. Minimalist Top Center Banner ("URANTUNE_WL_OT")
3. Glassmorphic Animated Right Sidebar with 3 exact action tiles (Bulb, Document, Power Off)
4. Toggleable AR Lined Notepad Overlay
"""

import cv2
import numpy as np
import time


class ARUIRenderer:
    def __init__(self):
        self.sidebar_anim_x = 0.0  # Animation factor (0.0 = closed, 1.0 = fully open)
        self.cursor_blink_time = time.time()
        self.show_cursor = True
        self.show_notepad = False  # Notepad closed by default for clean initial camera screen
        self.dark_mode = False     # Dark mode / brightness overlay flag
        self.tile_rects = {}      # Bounding boxes for click interactions

    def draw_top_hud(self, frame, fps, mode_name="PHALANX KEYBOARD", hand_count=0, active_gesture="IDLE"):
        """
        Render clean minimal FPS badge on top-left, mode status, and title header.
        """
        h, w, _ = frame.shape
        overlay = frame.copy()

        # 1. Top-Left FPS Pill Badge
        fps_text = f"{int(fps)} FPS"
        cv2.rectangle(overlay, (25, 20), (120, 48), (40, 40, 45), -1)
        cv2.rectangle(overlay, (25, 20), (120, 48), (120, 120, 130), 1, cv2.LINE_AA)

        # 2. Top-Center Minimal Title Header & Active Mode Status
        title_text = f"URANTUNE_WL_OT | MODE: {mode_name}"
        tw, th = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        tx = (w - tw) // 2
        cv2.rectangle(overlay, (tx - 15, 15), (tx + tw + 15, 42), (20, 20, 25), -1)
        cv2.rectangle(overlay, (tx - 15, 15), (tx + tw + 15, 42), (0, 200, 255), 1, cv2.LINE_AA)

        # 3. Active Gesture Action Pill (Top-Right HUD)
        if active_gesture != "IDLE":
            gw, gh = cv2.getTextSize(active_gesture, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            gx = w - gw - 40
            cv2.rectangle(overlay, (gx - 10, 15), (gx + gw + 10, 42), (0, 180, 80), -1)
            cv2.rectangle(overlay, (gx - 10, 15), (gx + gw + 10, 42), (255, 255, 255), 1, cv2.LINE_AA)

        alpha = 0.65
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        cv2.putText(frame, fps_text, (38, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 230), 1, cv2.LINE_AA)
        cv2.putText(frame, title_text, (tx, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        if active_gesture != "IDLE":
            gw, gh = cv2.getTextSize(active_gesture, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            gx = w - gw - 40
            cv2.putText(frame, active_gesture, (gx, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    def draw_sidebar(self, frame, open_state):
        """
        Render Animated Interactive Right Sidebar with 3 glassmorphic action tiles matching exact screenshot.
        """
        frame_h, frame_w, _ = frame.shape
        target_anim = 1.0 if open_state else 0.0

        # Smooth animation interpolation
        self.sidebar_anim_x += (target_anim - self.sidebar_anim_x) * 0.3

        if self.sidebar_anim_x < 0.02:
            self.tile_rects = {}
            return frame

        sidebar_width = 110
        sidebar_height = int(frame_h * 0.78)
        cur_x = int(frame_w - (sidebar_width * self.sidebar_anim_x))
        start_y = (frame_h - sidebar_height) // 2

        overlay = frame.copy()

        # Translucent glassmorphic panel background
        cv2.rectangle(overlay, (cur_x, start_y), (cur_x + sidebar_width, start_y + sidebar_height),
                      (220, 226, 228), -1)
        cv2.rectangle(overlay, (cur_x, start_y), (cur_x + sidebar_width, start_y + sidebar_height),
                      (255, 255, 255), 2, cv2.LINE_AA)
        # Accent top highlight line
        cv2.line(overlay, (cur_x + 15, start_y + 6), (cur_x + sidebar_width - 15, start_y + 6), (255, 255, 255), 3, cv2.LINE_AA)

        tile_size = 56
        tile_x = cur_x + (sidebar_width - tile_size) // 2
        icon_color = (45, 45, 50)

        # --- Tile 1 (Top): Light / Bulb Icon ---
        t1_y = start_y + 40
        self.tile_rects['tile_1'] = (tile_x, t1_y, tile_size, tile_size)
        cv2.rectangle(overlay, (tile_x, t1_y), (tile_x + tile_size, t1_y + tile_size),
                      (242, 245, 248), -1)
        cv2.rectangle(overlay, (tile_x, t1_y), (tile_x + tile_size, t1_y + tile_size),
                      (175, 185, 190), 1, cv2.LINE_AA)
        
        c1 = (tile_x + tile_size // 2, t1_y + tile_size // 2)
        # Bulb circle top & base lines
        cv2.circle(overlay, (c1[0], c1[1] - 4), 10, icon_color, 2, cv2.LINE_AA)
        cv2.line(overlay, (c1[0] - 8, c1[1] + 9), (c1[0] + 8, c1[1] + 9), icon_color, 2, cv2.LINE_AA)
        cv2.line(overlay, (c1[0] - 4, c1[1] + 13), (c1[0] + 4, c1[1] + 13), icon_color, 2, cv2.LINE_AA)

        # --- Tile 2 (Middle): Document / Notepad Icon ---
        t2_y = start_y + 120
        self.tile_rects['tile_2'] = (tile_x, t2_y, tile_size, tile_size)
        cv2.rectangle(overlay, (tile_x, t2_y), (tile_x + tile_size, t2_y + tile_size),
                      (242, 245, 248), -1)
        cv2.rectangle(overlay, (tile_x, t2_y), (tile_x + tile_size, t2_y + tile_size),
                      (175, 185, 190), 1, cv2.LINE_AA)
        
        c2 = (tile_x + tile_size // 2, t2_y + tile_size // 2)
        # Document outline and 3 horizontal lines
        cv2.rectangle(overlay, (c2[0] - 12, c2[1] - 15), (c2[0] + 12, c2[1] + 15), icon_color, 2, cv2.LINE_AA)
        cv2.line(overlay, (c2[0] - 7, c2[1] - 7), (c2[0] + 7, c2[1] - 7), icon_color, 2, cv2.LINE_AA)
        cv2.line(overlay, (c2[0] - 7, c2[1]), (c2[0] + 7, c2[1]), icon_color, 2, cv2.LINE_AA)
        cv2.line(overlay, (c2[0] - 7, c2[1] + 7), (c2[0] + 7, c2[1] + 7), icon_color, 2, cv2.LINE_AA)

        # --- Tile 3 (Bottom): Power / Off Icon ---
        t3_y = start_y + 200
        self.tile_rects['tile_3'] = (tile_x, t3_y, tile_size, tile_size)
        cv2.rectangle(overlay, (tile_x, t3_y), (tile_x + tile_size, t3_y + tile_size),
                      (242, 245, 248), -1)
        cv2.rectangle(overlay, (tile_x, t3_y), (tile_x + tile_size, t3_y + tile_size),
                      (175, 185, 190), 1, cv2.LINE_AA)
        
        c3 = (tile_x + tile_size // 2, t3_y + tile_size // 2)
        # Power arc and top vertical line
        cv2.ellipse(overlay, (c3[0], c3[1] + 2), (11, 11), 0, 45, 315, icon_color, 2, cv2.LINE_AA)
        cv2.line(overlay, (c3[0], c3[1] - 12), (c3[0], c3[1] - 2), icon_color, 2, cv2.LINE_AA)

        alpha = 0.75
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        return frame

    def draw_notepad_overlay(self, frame, text_buffer, x=40, y=70, w=520, h=250):
        """
        Render Virtual Notepad AR Overlay only when toggled on.
        """
        if not self.show_notepad:
            return frame

        frame_h, frame_w, _ = frame.shape
        overlay = frame.copy()

        # Glass Panel Background
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (230, 235, 240), -1)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 255), 2, cv2.LINE_AA)

        # Lined Paper Rules
        line_spacing = 32
        for line_y in range(y + 60, y + h - 10, line_spacing):
            cv2.line(overlay, (x + 20, line_y), (x + w - 20, line_y), (200, 205, 215), 1, cv2.LINE_AA)

        alpha = 0.72
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Header Title
        cv2.circle(frame, (x + 30, y + 28), 6, (0, 150, 255), cv2.FILLED, cv2.LINE_AA)
        cv2.putText(frame, "AR NOTEPAD", (x + 48, y + 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 60), 2, cv2.LINE_AA)

        # Cursor Blinking Logic
        if time.time() - self.cursor_blink_time > 0.5:
            self.show_cursor = not self.show_cursor
            self.cursor_blink_time = time.time()

        display_text = text_buffer + ("|" if self.show_cursor else "")

        max_chars_per_line = 34
        words = display_text.split(' ')
        lines = []
        cur_line = ""

        for word in words:
            if len(cur_line + " " + word) <= max_chars_per_line:
                cur_line = (cur_line + " " + word).strip()
            else:
                lines.append(cur_line)
                cur_line = word
        if cur_line:
            lines.append(cur_line)

        for idx, line in enumerate(lines[:6]):
            text_y = y + 55 + idx * line_spacing
            cv2.putText(frame, line, (x + 30, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 30), 2, cv2.LINE_AA)

        return frame

    def apply_theme_filter(self, frame):
        """
        Applies Dark Mode / Screen Brightness Filter overlay if dark_mode is active.
        """
        if self.dark_mode:
            dark_overlay = np.zeros_like(frame)
            frame = cv2.addWeighted(frame, 0.55, dark_overlay, 0.45, 0)
        return frame
