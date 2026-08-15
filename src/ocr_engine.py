import easyocr

class OCREngine:
    def __init__(self, languages=['en'], gpu=False):
        self.reader = easyocr.Reader(languages, gpu=gpu)

    def recognize(self, canvas):
        if canvas is None:
            return []
        results = self.reader.readtext(canvas)
        text_list = [res[1] for res in results]
        return text_list
