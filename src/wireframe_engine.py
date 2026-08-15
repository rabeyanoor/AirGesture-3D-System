import cv2
import numpy as np

class WireframeEngine:
    def draw_3d_spatial_mesh(self, frame, landmarks_list):
        """
        Renders 3D spatial wireframe and mesh polygons exactly matching reference demo images.
        Draws cyan landmark keypoints, coordinate labels (X, Y), and translucent grid mesh fill.
        """
        if not landmarks_list:
            return

        # 1. Render Cyan keypoints and (X, Y) text labels for all detected hand landmarks
        for hand_pts in landmarks_list:
            for fid in [4, 8, 12, 16, 20]:
                pt = hand_pts[fid][:2]
                # Draw cyan dot
                cv2.circle(frame, pt, 5, (255, 255, 0), -1, cv2.LINE_AA)
                
                # Coordinate label string: (X, Y)
                coord_str = f"({pt[0]}, {pt[1]})"
                cv2.putText(frame, coord_str, (pt[0] - 15, pt[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, coord_str, (pt[0] - 15, pt[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        # 2. Render 3D Spatial Wireframe Mesh Polygon connecting hands
        if len(landmarks_list) >= 1:
            if len(landmarks_list) == 2:
                hand1 = landmarks_list[0]
                hand2 = landmarks_list[1]
                
                # Build 3D spatial polygon vertices across both hands
                pts1 = [hand1[4][:2], hand1[8][:2], hand1[12][:2], hand1[16][:2], hand1[20][:2]]
                pts2 = [hand2[20][:2], hand2[16][:2], hand2[12][:2], hand2[8][:2], hand2[4][:2]]
                poly_pts = np.array(pts1 + pts2, np.int32)
            else:
                # Single hand spatial mesh across fingers
                hand = landmarks_list[0]
                poly_pts = np.array([
                    hand[4][:2], hand[8][:2], hand[12][:2], hand[16][:2], hand[20][:2], hand[0][:2]
                ], np.int32)

            # Draw translucent stippled / textured mesh fill
            overlay = frame.copy()
            cv2.fillPoly(overlay, [poly_pts], color=(200, 200, 200))
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

            # Draw white wireframe outline
            cv2.polylines(frame, [poly_pts], isClosed=True, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

            # Draw interior spatial cross wireframe lines
            if len(landmarks_list) == 2:
                for fid in [4, 8, 12, 16, 20]:
                    p1 = hand1[fid][:2]
                    p2 = hand2[fid][:2]
                    cv2.line(frame, p1, p2, (255, 255, 255), 1, cv2.LINE_AA)
