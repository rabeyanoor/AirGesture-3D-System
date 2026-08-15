import cv2
import time
import numpy as np


class UIManager:
    def __init__(self):
        self.hover_start = None
        self.hovered_btn = None
        self.dwell_time = 0.5
        self.sidebar_visible = False

    # ------------------------------------------------------------------ #
    # FPS Badge                                                            #
    # ------------------------------------------------------------------ #
    def draw_top_fps_badge(self, img, fps):
        badge = f"{fps} FPS"
        overlay = img.copy()
        cv2.rectangle(overlay, (20, 18), (108, 54), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
        cv2.rectangle(img, (20, 18), (108, 54), (120, 120, 120), 1, cv2.LINE_AA)
        cv2.putText(img, badge, (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (240, 240, 240), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    # Shift Mode Indicator                                                 #
    # ------------------------------------------------------------------ #
    def draw_shift_indicator(self, img, shift_on):
        col = (0, 200, 80) if shift_on else (80, 80, 80)
        label = "SHIFT: ON (M-X)" if shift_on else "SHIFT: OFF (A-L)"
        overlay = img.copy()
        cv2.rectangle(overlay, (20, 60), (220, 88), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
        cv2.putText(img, label, (28, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    col, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    # Right Sidebar                                                        #
    # ------------------------------------------------------------------ #
    def draw_right_toolbar(self, img, active_mode, light_on):
        if not self.sidebar_visible:
            return
        h, w, _ = img.shape
        pw, ph = 80, 300
        px1 = w - pw - 20
        py1 = (h - ph) // 2
        px2, py2 = px1 + pw, py1 + ph

        overlay = img.copy()
        cv2.rectangle(overlay, (px1, py1), (px2, py2), (228, 232, 232), -1)
        cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
        cv2.rectangle(img, (px1, py1), (px2, py2), (180, 185, 185), 1, cv2.LINE_AA)

        for btn in self._buttons(w, h):
            name, (bx1, by1, bx2, by2) = btn["name"], btn["rect"]
            is_active = (name == "NOTEPAD" and active_mode == "WRITE") or \
                        (name == "LIGHT" and light_on)
            bov = img.copy()
            cv2.rectangle(bov, (bx1, by1), (bx2, by2),
                          (190, 210, 175) if is_active else (255, 255, 255), -1)
            cv2.addWeighted(bov, 0.85, img, 0.15, 0, img)
            cv2.rectangle(img, (bx1, by1), (bx2, by2), (150, 150, 150), 1, cv2.LINE_AA)

            cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
            if name == "LIGHT":
                col = (0, 165, 255) if light_on else (50, 50, 50)
                cv2.circle(img, (cx, cy - 3), 12, col, 2, cv2.LINE_AA)
                cv2.rectangle(img, (cx-5, cy+9), (cx+5, cy+14), col, -1)
                cv2.line(img, (cx-3, cy+17), (cx+3, cy+17), col, 2)
            elif name == "NOTEPAD":
                col = (0, 100, 220) if active_mode == "WRITE" else (50, 50, 50)
                cv2.rectangle(img, (cx-13, cy-15), (cx+13, cy+15), col, 2, cv2.LINE_AA)
                for dy in [-7, 0, 7]:
                    cv2.line(img, (cx-8, cy+dy), (cx+8, cy+dy), col, 1)
            elif name == "POWER":
                col = (170, 35, 35)
                cv2.ellipse(img, (cx, cy+3), (13, 13), 0, 45, 315, col, 2, cv2.LINE_AA)
                cv2.line(img, (cx, cy-13), (cx, cy-1), col, 2, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    # Notepad Card                                                         #
    # ------------------------------------------------------------------ #
    def draw_notepad_card(self, img, text_buffer, ai_answer=""):
        h, w, _ = img.shape
        nx1, ny1 = 40, 70
        nw, nh = min(520, w - 140), 420
        nx2, ny2 = nx1 + nw, ny1 + nh

        overlay = img.copy()
        cv2.rectangle(overlay, (nx1, ny1), (nx2, ny2), (242, 242, 242), -1)
        cv2.addWeighted(overlay, 0.42, img, 0.58, 0, img)
        cv2.rectangle(img, (nx1, ny1), (nx2, ny2), (185, 185, 185), 1, cv2.LINE_AA)

        cv2.putText(img, "NOTEPAD", (nx1+20, ny1+30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (60, 60, 60), 1, cv2.LINE_AA)

        # Ruling lines
        line_gap = 30
        lines_y = []
        for y in range(ny1+55, ny2-15, line_gap):
            cv2.line(img, (nx1+15, y), (nx2-15, y), (215, 215, 215), 1, cv2.LINE_AA)
            lines_y.append(y)

        # Typed text
        display = text_buffer if text_buffer else ""
        if display:
            cv2.putText(img, display, (nx1+20, lines_y[0]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.68, (15, 15, 15), 2, cv2.LINE_AA)

        # AI answer
        if ai_answer and len(lines_y) > 2:
            cv2.putText(img, f"AI: {ai_answer}", (nx1+20, lines_y[2]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 100, 200), 2, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    # Left-hand Joint Overlay (visual keyboard targets)                   #
    # ------------------------------------------------------------------ #
    def draw_left_hand_targets(self, img, left_pixel_lm, shift_mode):
        """Draws colored circles on left hand joints to show typing targets."""
        if not left_pixel_lm:
            return

        NORMAL_MAP = {8:'A',7:'B',6:'C',12:'D',11:'E',10:'F',
                      16:'G',15:'H',14:'I',20:'J',19:'K',18:'L',
                      4:'Y', 3:'Z', 9:'_', 0:'⌫'}
        SHIFT_MAP  = {8:'M',7:'N',6:'O',12:'P',11:'Q',10:'R',
                      16:'S',15:'T',14:'U',20:'V',19:'W',18:'X',
                      4:'Y', 3:'Z', 9:'_', 0:'⌫'}

        char_map = SHIFT_MAP if shift_mode else NORMAL_MAP
        col = (0, 220, 60) if shift_mode else (0, 200, 255)

        for joint_id, char in char_map.items():
            if joint_id >= len(left_pixel_lm):
                continue
            px, py = left_pixel_lm[joint_id][0], left_pixel_lm[joint_id][1]
            cv2.circle(img, (px, py), 10, col, -1, cv2.LINE_AA)
            cv2.putText(img, char, (px + 12, py + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    # Interaction (sidebar trigger + button hover)                        #
    # ------------------------------------------------------------------ #
    def _buttons(self, w, h):
        ph = 300
        py1 = (h - ph) // 2
        bx1 = w - 20 - 70
        bw = bh = 55
        return [
            {"name": "LIGHT",   "rect": (bx1, py1+15,  bx1+bw, py1+15+bh)},
            {"name": "NOTEPAD", "rect": (bx1, py1+95,  bx1+bw, py1+95+bh)},
            {"name": "POWER",   "rect": (bx1, py1+175, bx1+bw, py1+175+bh)},
        ]

    def get_button_rects(self, w, h):
        """Public alias for test compatibility."""
        return self._buttons(w, h)

    def check_interaction(self, img, landmarks_list, norm_landmarks, w, h, active_mode, light_on):
        # Index finger x > 0.88 → show sidebar
        if norm_landmarks and len(norm_landmarks) >= 9:
            if norm_landmarks[8].x > 0.88:
                self.sidebar_visible = True

        if not self.sidebar_visible and active_mode != "WRITE":
            return active_mode, light_on, False

        if not landmarks_list:
            self.hover_start = None
            self.hovered_btn = None
            return active_mode, light_on, False

        ix, iy = landmarks_list[0][8][:2]
        hovered_now = None
        for btn in self._buttons(w, h):
            n, (bx1, by1, bx2, by2) = btn["name"], btn["rect"]
            if bx1 <= ix <= bx2 and by1 <= iy <= by2:
                hovered_now = n
                break

        if hovered_now:
            if self.hovered_btn == hovered_now:
                elapsed = time.time() - self.hover_start
                ang = int(min(elapsed / self.dwell_time, 1.0) * 360)
                cv2.ellipse(img, (ix, iy), (18, 18), 0, 0, ang, (0, 230, 80), 2)
                if elapsed >= self.dwell_time:
                    if hovered_now == "NOTEPAD":
                        active_mode = "WRITE" if active_mode != "WRITE" else "WIREFRAME"
                        if active_mode == "WIREFRAME":
                            self.sidebar_visible = False
                    elif hovered_now == "LIGHT":
                        light_on = not light_on
                    elif hovered_now == "POWER":
                        active_mode = "WIREFRAME"
                        self.sidebar_visible = False
                    self.hover_start = time.time() + 0.6
            else:
                self.hovered_btn = hovered_now
                self.hover_start = time.time()
        else:
            self.hovered_btn = None
            self.hover_start = None

        return active_mode, light_on, False
