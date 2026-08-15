# Configuration Settings for Spatial Vision AR

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Color Palette (BGR format)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_CYAN = (255, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_MAGENTA = (255, 0, 255)
COLOR_ORANGE = (0, 165, 255)
COLOR_DARK_GRAY = (40, 40, 40)
COLOR_LIGHT_GRAY = (200, 200, 200)

AVAILABLE_COLORS = [
    ("White", COLOR_WHITE),
    ("Red", COLOR_RED),
    ("Green", COLOR_GREEN),
    ("Blue", COLOR_BLUE),
    ("Yellow", COLOR_YELLOW),
    ("Cyan", COLOR_CYAN),
    ("Magenta", COLOR_MAGENTA),
]

# Hand Tracking Settings
MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.75
MIN_TRACKING_CONFIDENCE = 0.75

# Gesture & Interaction Settings
PINCH_THRESHOLD = 35
HOVER_SELECTION_TIME = 0.8  # Seconds needed to dwell on a button
SMOOTHING_FACTOR = 0.5       # Exponential moving average factor for landmarks

# Default Canvas Settings
DEFAULT_STROKE_THICKNESS = 5
ERASER_THICKNESS = 30
