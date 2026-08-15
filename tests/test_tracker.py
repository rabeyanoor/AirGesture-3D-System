import unittest
import numpy as np
from src.hand_tracker import HandTracker
from src.air_scribble import AirScribble
from src.wireframe_engine import WireframeEngine
from src.ui_manager import UIManager

class TestSpatialVisionAR(unittest.TestCase):
    def test_hand_tracker_init(self):
        tracker = HandTracker()
        self.assertIsNotNone(tracker.hands)

    def test_air_scribble_canvas_init(self):
        scribble = AirScribble()
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        scribble.update(dummy_frame, [])
        self.assertIsNotNone(scribble.canvas)
        self.assertEqual(scribble.canvas.shape, (720, 1280, 3))

    def test_ui_manager_buttons(self):
        ui = UIManager()
        buttons = ui.get_buttons(1280)
        self.assertGreater(len(buttons), 0)

if __name__ == '__main__':
    unittest.main()
