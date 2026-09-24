"""
Spatial Vision AR - Shared Frame Pipeline
Single per-frame processing path used by main.py (desktop), app.py (Gradio) and run_video_test.py.

Flow per frame:
1. MediaPipe hand tracking
2. Gestures: sidebar reveal + tile taps, 3D box control, phalanx keyboard, palm-touch space,
   grid keypad, air drawing
3. 3D box (liquid box / Rubik's cube) with the AR glass hand mesh drawn on top
4. UI: FPS pill, lined notepad, right sidebar
"""

import time

import cv2

from hand_tracker import HandTracker
from gesture_recognizer import GestureRecognizer
from ar_mesh_3d import ARMesh3D
from ar_objects_3d import LiquidCube, RubiksCube
from air_keyboard import AirKeyboard, LABELS as KEY_LABELS
from air_writing import AirWriter
from auto_capitalizer import AutoCapitalizer
from ar_ui_renderer import ARUIRenderer

MODES = ["AIR WRITE", "HAND KEYBOARD", "AIR KEYBOARD", "GRID KEYBOARD"]


class SpatialVisionPipeline:
    def __init__(self, debug_hud=False, async_tracking=False):
        # async_tracking: hand detection on a worker thread (live camera) so the video never waits for it
        self.tracker = HandTracker(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.6,
                                   async_mode=async_tracking)
        self.recognizer = GestureRecognizer()
        self.mesh_renderer = ARMesh3D()
        self.writer = AirWriter()
        self.ui_renderer = ARUIRenderer()
        self.liquid_box = LiquidCube()
        self.rubik_cube = RubiksCube()
        self.keyboard = AirKeyboard()

        self.debug_hud = debug_hud
        self.text_buffer = ""
        self.active_mode = "AIR WRITE"
        self._key_flash = None  # (hand label, landmark id, time) of the last finger key typed
        self.show_key_labels = False  # letters on the finger joints are hidden unless toggled on ('k')
        self.sidebar_open = False
        self.active_gesture = "IDLE"
        self.exit_requested = False

        self._prev_time = None
        self._fps = 0.0

    # ------------------------------------------------------------------
    # Text editing helpers
    # ------------------------------------------------------------------
    def append_text(self, chars):
        self.text_buffer = AutoCapitalizer.process_notepad_text(self.text_buffer + chars)

    def backspace(self):
        self.text_buffer = self.text_buffer[:-1]

    def delete_last_word(self):
        text = self.text_buffer.rstrip()
        self.text_buffer = text.rsplit(' ', 1)[0] + ' ' if ' ' in text else ""

    def press_key(self, key):
        """Apply one Air Keyboard key to the notepad text."""
        if key == "<BKSP>":
            self.backspace()
        elif key == "<CLEAR>":
            self.text_buffer = ""
        elif key == "<ENTER>":
            self.text_buffer += "\n"
        else:
            self.append_text(key)

    def clear(self):
        self.text_buffer = ""
        self.writer.clear()

    def active_box(self):
        if self.ui_renderer.show_liquid:
            return self.liquid_box
        if self.ui_renderer.show_rubik:
            return self.rubik_cube
        return None

    def toggle_box(self, name):
        ui = self.ui_renderer
        if name == 'liquid':
            ui.show_liquid, ui.show_rubik = not ui.show_liquid, False
        else:
            ui.show_rubik, ui.show_liquid = not ui.show_rubik, False

    def cycle_mode(self):
        idx = MODES.index(self.active_mode) if self.active_mode in MODES else 0
        self.active_mode = MODES[(idx + 1) % len(MODES)]
        return self.active_mode

    def _update_fps(self, now):
        if self._prev_time is not None:
            inst = 1.0 / max(now - self._prev_time, 1e-5)
            # Exponential smoothing keeps the FPS pill readable
            self._fps = inst if self._fps == 0 else self._fps * 0.9 + inst * 0.1
        self._prev_time = now

    # ------------------------------------------------------------------
    # Main per-frame entry point
    # ------------------------------------------------------------------
    def process(self, frame, now=None):
        now = time.time() if now is None else now
        self._update_fps(now)
        h, w, _ = frame.shape

        all_hands, left_hand, right_hand = self.tracker.process(frame)
        hands_data = [hd for hd in all_hands if self.tracker.is_palm_facing_camera(hd)]
        gesture = "IDLE"

        # 2. Gestures
        # 2A. Fist status
        if (left_hand and self.recognizer.is_fist(left_hand)) or (right_hand and self.recognizer.is_fist(right_hand)):
            gesture = "FIST GESTURE"

        notepad_on = self.ui_renderer.show_notepad

        # 2B. Right index touching left palm -> space (phalanx typing mode only)
        if notepad_on and self.active_mode == "HAND KEYBOARD" and left_hand and right_hand:
            is_touch, touch_pt = self.recognizer.check_palm_touch(left_hand, right_hand, w, h)
            if is_touch:
                if not self.recognizer.palm_touch_triggered:
                    self.append_text(" ")
                    self.recognizer.palm_touch_triggered = True
                gesture = "PALM TOUCH (SPACE)"
                cv2.circle(frame, touch_pt, 16, (255, 220, 120), 2, cv2.LINE_AA)
            else:
                self.recognizer.palm_touch_triggered = False

        # 2C. Sidebar reveal + tile taps
        self.sidebar_open = self.recognizer.update_sidebar_state(hands_data, w, self.sidebar_open)
        self.sidebar_open, click_action = self.recognizer.check_sidebar_clicks(
            hands_data, self.ui_renderer, self.sidebar_open)
        if click_action == "POWER_OFF":
            # Power tile: wipe the notepad, close any 3D box and slide the sidebar away
            self.clear()
            self.ui_renderer.show_liquid = self.ui_renderer.show_rubik = False
            gesture = click_action
        elif click_action:
            gesture = click_action
        elif self.sidebar_open and gesture == "IDLE":
            gesture = "SIDEBAR OPEN"

        # 2D. Air Keyboard (notepad open): point at a key and pinch to press it
        keyboard_on = notepad_on and self.active_mode == "AIR KEYBOARD"
        writing_on = notepad_on and self.active_mode == "AIR WRITE"
        box_hands = all_hands
        if writing_on and self.writer.writing:
            # The hand holding the pen is writing, not grabbing a box
            box_hands = [hd for hd in all_hands if hd['label'] != self.writer.pen_label]
        if keyboard_on:
            for key in self.keyboard.update(all_hands, w, h, now):
                self.press_key(key)
                gesture = f"KEY '{KEY_LABELS.get(key, key.upper())}'"
            # Hands working the keyboard are not grabbing 3D boxes
            box_hands = [hd for hd in all_hands if not self.keyboard.owns(hd)]

        # 2E. 3D box interaction (pinch-drag rotate, Rubik layer swipes)
        box = self.active_box()
        box_busy = False
        if box is not None:
            box_gesture = box.update(box_hands, w, h, now)
            box_busy = box.grabbed
            if box_gesture and gesture in ("IDLE", "SIDEBAR OPEN", "FIST GESTURE"):
                gesture = box_gesture

        # 2F. Phalanx keyboard (only while the notepad is visible, and not while holding a box)
        if box_busy:
            pass
        elif self.active_mode == "HAND KEYBOARD" and self.ui_renderer.show_notepad:
            hovers = {'Left': self.recognizer.phalanx_hover(left_hand, self.recognizer.LEFT_PHALANX_MAP),
                      'Right': self.recognizer.phalanx_hover(right_hand, self.recognizer.RIGHT_PHALANX_MAP)}
            key = self.recognizer.detect_phalanx_keyboard(left_hand, right_hand, w, h)
            if key:
                # Flash the joint that was touched
                for label, hv in hovers.items():
                    if hv is not None and hv[1] == key:
                        self._key_flash = (label, hv[0], now)
            if key == '<DELETE_WORD>':
                self.backspace()  # palm = DEL: one letter (hold to keep deleting)
                gesture = "DEL"
            elif key:
                self.append_text(key)
                gesture = f"HAND KEY '{key.upper() if key.strip() else 'SPACE'}'"

        # 2G. Dual-hand grid keypad
        elif self.active_mode == "GRID KEYBOARD" and left_hand and right_hand:
            sel_group, typed_char = self.recognizer.process_grid_keypad(left_hand, right_hand, w, h)
            if sel_group:
                gesture = f"KEYPAD ({sel_group})"
            if typed_char:
                self.append_text(typed_char.lower())
                gesture = f"TYPED '{typed_char}'"

        # 2H. Air writing: pinch (hold the pen) and write a letter in the air
        if writing_on:
            sidebar_x = w * 0.86 if self.sidebar_open else w + 1
            grabbing = box.mode_label if box is not None and box.grabbed else None

            def blocked(pt):
                # Pinches on a 3D box or on the sidebar are not pen strokes
                return pt[0] > sidebar_x or (box is not None and box.contains(pt))

            pen_hands = [hd for hd in all_hands if hd['label'] != grabbing]
            key = self.writer.update(pen_hands, w, h, now, blocked=blocked)
            if key == '<BKSP>':
                self.backspace()
                gesture = "DEL"
            elif key:
                self.append_text(key)
                gesture = f"WROTE '{key.upper() if key.strip() else 'SPACE'}'"
            elif self.writer.writing:
                gesture = "WRITING..."

        # 3. 3D box, then the hand mesh on top so the hand appears in front of the box
        if box is not None:
            frame = box.render(frame, now)
        hand_keys = notepad_on and self.active_mode == "HAND KEYBOARD"
        if hand_keys and self.show_key_labels:
            # Letters sit on the finger joints (the coordinate dots would only clutter them)
            maps = []
            for hand, key_map in ((left_hand, self.recognizer.LEFT_PHALANX_MAP),
                                  (right_hand, self.recognizer.RIGHT_PHALANX_MAP)):
                if hand:
                    maps.append((hand, key_map, self.recognizer.phalanx_hover(hand, key_map)))
            flash = None
            if self._key_flash and now - self._key_flash[2] < 0.3:
                flash = self._key_flash[:2]
            frame = self.ui_renderer.draw_hand_keys(frame, maps, flash)
        elif len(all_hands) >= 2:
            frame = self.mesh_renderer.draw_dual_hand_3d_wireframe(frame, all_hands[0], all_hands[1])
        elif len(all_hands) == 1:
            frame = self.mesh_renderer.draw_fingertip_polygon(frame, all_hands[0])

        if writing_on:
            frame = self.writer.draw(frame)

        # 4. UI overlays
        frame = self.ui_renderer.apply_theme_filter(frame)
        frame = self.ui_renderer.draw_notepad_overlay(frame, self.text_buffer, caret=keyboard_on or hand_keys or writing_on)
        if keyboard_on:
            frame = self.keyboard.draw(frame, now)
        frame = self.ui_renderer.draw_sidebar(frame, self.sidebar_open)
        frame = self.ui_renderer.draw_top_hud(frame, self._fps, self.active_mode, len(all_hands), gesture,
                                              debug=self.debug_hud)

        self.active_gesture = gesture
        return frame
