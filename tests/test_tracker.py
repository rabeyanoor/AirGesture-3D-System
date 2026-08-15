import unittest
import numpy as np
from src.hand_tracker import HandTracker
from src.wireframe_engine import WireframeEngine
from src.ui_manager import UIManager
from src.air_typer import AirTyper
from src.ai_solver import AISolver

class TestSpatialVisionAR(unittest.TestCase):
    def test_hand_tracker_init(self):
        tracker = HandTracker(max_hands=2)
        self.assertIsNotNone(tracker.hands)

    def test_ui_manager_buttons(self):
        ui = UIManager()
        buttons = ui.get_button_rects(1280, 720)
        self.assertGreater(len(buttons), 0)

    def test_air_typer_init(self):
        typer = AirTyper()
        self.assertEqual(typer.last_char, None)

    def test_ai_solver_init(self):
        ai = AISolver()
        self.assertIsNotNone(ai)

if __name__ == '__main__':
    unittest.main()
