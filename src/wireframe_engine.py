import cv2
import numpy as np

class WireframeEngine:
    def draw_3d_spatial_mesh(self, frame, landmarks_list):
        """
        Renders 3D spatial wireframe and mesh polygons matching frame_2.png in video_2026-08-15_16-15-48.mp4.
        Draws cyan landmark keypoints, (X, Y) text tags, and translucent wireframe polygon fill.
        """
        if not landmarks_list:
            return

        # 1. Render Cyan keypoints and (X, Y) text labels for key hand landmarks
        for hand_pts in landmarks_list:
            for fid in [4, 8, 12, 16, 20, 0, 5, 9, 13, 17]:
                pt = hand_pts[fid][:2]
                # Cyan keypoint circle
                cv2.circle(frame, pt, 4, (255, 255, 0), -1, cv2.LINE_AA)
                
                # Coordinate string: (X, Y)
                coord_str = f"{pt[0]}, {pt[1]}"
                # Text with black outline for visibility
                cv2.putText(frame, coord_str, (pt[0] - 14, pt[1] - 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, coord_str, (pt[0] - 14, pt[1] - 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)

        # 2. Render 3D Spatial Wireframe Mesh Polygon connecting hands
        if len(landmarks_list) >= 1:
            if len(landmarks_list) == 2:
                hand1 = landmarks_list[0]
                hand2 = landmarks_list[1]
                
                # 3D polygon vertices connecting fingertips and wrists across both hands
                pts1 = [hand1[4][:2], hand1[8][:2], hand1[12][:2], hand1[16][:2], hand1[20][:2], hand1[0][:2]]
                pts2 = [hand2[0][:2], hand2[20][:2], hand2[16][:2], hand2[12][:2], hand2[8][:2], hand2[4][:2]]
                poly_pts = np.array(pts1 + pts2, np.int32)
            else:
                hand = landmarks_list[0]
                poly_pts = np.array([
                    hand[4][:2], hand[8][:2], hand[12][:2], hand[16][:2], hand[20][:2], hand[0][:2]
                ], np.int32)

            # Translucent Mesh Polygon Fill
            overlay = frame.copy()
            cv2.fillPoly(overlay, [poly_pts], color=(180, 180, 180))
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

            # White Wireframe Edge Outlines
            cv2.polylines(frame, [poly_pts], isClosed=True, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

            # Interior Connecting Wireframe Lines
            if len(landmarks_list) == 2:
                for fid in [4, 8, 12, 16, 20]:
                    cv2.line(frame, hand1[fid][:2], hand2[fid][:2], (255, 255, 255), 1, cv2.LINE_AA)
