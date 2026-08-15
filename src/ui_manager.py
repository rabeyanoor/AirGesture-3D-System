import cv2

class UIManager:
    @staticmethod
    def draw_virtual_gui(img):
        h, w, _ = img.shape
        cv2.rectangle(img, (w - 120, 50), (w - 20, 150), (200, 200, 200), -1)
        cv2.putText(img, "Wireframe", (w - 110, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.rectangle(img, (w - 120, 180), (w - 20, 280), (200, 200, 200), -1)
        cv2.putText(img, "Notepad", (w - 110, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.rectangle(img, (w - 120, 310), (w - 20, 410), (200, 200, 200), -1)
        cv2.putText(img, "Clear", (w - 110, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    @staticmethod
    def check_interaction(landmarks_list, frame_width, air_scribble):
        active_mode = None
        if len(landmarks_list) > 0:
            ix, iy = landmarks_list[0][8]
            if ix > frame_width - 120 and ix < frame_width - 20:
                if 50 < iy < 150:
                    active_mode = "WIREFRAME"
                elif 180 < iy < 280:
                    active_mode = "WRITE"
                elif 310 < iy < 410:
                    air_scribble.clear()
        return active_mode
