import cv2
import numpy as np

class WireframeEngine:
    @staticmethod
    def draw_3d_mesh(frame, landmarks_list):
        if len(landmarks_list) == 2:
            hand1_pts = landmarks_list[0]
            hand2_pts = landmarks_list[1]
            
            # Connect key joints from both hands to create a 3D wireframe volume
            poly_pts = np.array([
                hand1_pts[4], hand1_pts[8], hand1_pts[12],
                hand2_pts[12], hand2_pts[8], hand2_pts[4]
            ], np.int32)
            
            cv2.polylines(frame, [poly_pts], isClosed=True, color=(0, 255, 255), thickness=2)
            cv2.fillPoly(frame, [poly_pts], color=(100, 100, 250))
