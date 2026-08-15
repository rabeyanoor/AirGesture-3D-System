import cv2
import numpy as np

class OCREngine:
    def __init__(self, languages=['en'], gpu=False):
        self.reader = None
        self.gpu = gpu
        self.languages = languages
        self._init_reader()

    def _init_reader(self):
        try:
            import easyocr
            self.reader = easyocr.Reader(self.languages, gpu=self.gpu)
        except Exception as e:
            print("OCR Engine Notice - EasyOCR loading deferred or falling back:", e)

    def recognize(self, canvas):
        """
        Preprocesses air-written canvas and recognizes handwritten text.
        Returns:
            list of recognized strings
        """
        if canvas is None or np.sum(canvas) == 0:
            return []

        if self.reader is None:
            self._init_reader()
            if self.reader is None:
                return ["OCR Library Not Installed"]

        # Step 1: Convert to Grayscale
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

        # Step 2: Crop non-empty bounding box of drawn text
        coords = cv2.findNonZero(gray)
        if coords is None:
            return []
        x, y, w, h = cv2.boundingRect(coords)

        # Add padding around bounding box
        pad = 20
        h_canvas, w_canvas = gray.shape
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w_canvas, x + w + pad), min(h_canvas, y + h + pad)

        cropped = gray[y1:y2, x1:x2]

        # Step 3: Contrast Enhancement & Inversion (Black text on White Background)
        _, thresh = cv2.threshold(cropped, 50, 255, cv2.THRESH_BINARY)
        inv = cv2.bitwise_not(thresh)

        # Step 4: Perform OCR Recognition
        try:
            results = self.reader.readtext(inv)
            text_list = [res[1] for res in results if res[2] > 0.2] # Filter by confidence
            return text_list
        except Exception as err:
            print("OCR Error:", err)
            return []
