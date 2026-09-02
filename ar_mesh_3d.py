"""
AR 3D Mesh & Wireframe Renderer Module
Matches exact high-precision visual aesthetics of reference video:
1. Extended-Fingertip Mesh Polygon (highlights ONLY extended fingertips with cyan dots, coordinate callouts, and translucent mesh hull).
2. Single-Finger Extended Mode (renders clean line between tip and PIP joint with tip/joint coordinates).
3. Dynamic Dual-Hand 3D Wireframe Volume (connects extended fingertips of Hand 1 to Hand 2 with translucent glass box faces).
"""

import cv2
import numpy as np
import math


class ARMesh3D:
    def __init__(self):
        self.FINGERTIP_IDS = [4, 8, 12, 16, 20]
        self.poly_color = (200, 205, 210)        # Translucent light grey/white polygon fill
        self.wireframe_color = (255, 255, 255)   # Crisp white outline
        self.dot_color = (255, 180, 0)            # Vivid cyan/blue dot (BGR: 255, 180, 0)
        self.text_color = (255, 255, 255)        # Clean white coordinate text

        # Mapping tip to PIP joint for single-finger indicator
        self.PIP_MAP = {8: 6, 12: 10, 16: 14, 20: 18, 4: 2}

    def get_extended_fingertip_ids(self, hand_dict):
        """
        Identify ONLY fingertips [4, 8, 12, 16, 20] that are genuinely extended.
        Uses relative 2D landmark ratios and wrist geometry, calibrated for both near and far hand distances.
        """
        if not hand_dict or 'px' not in hand_dict:
            return []

        px = hand_dict['px']
        p0 = px[0]  # Wrist landmark 0

        extended_ids = []
        fingers_map = [
            (8, 6, 5),    # Index Finger: (tip, pip, mcp)
            (12, 10, 9),  # Middle Finger
            (16, 14, 13), # Ring Finger
            (20, 18, 17)  # Pinky Finger
        ]

        for tip_id, pip_id, mcp_id in fingers_map:
            # 2D Distances from MCP joint
            d_tip_mcp = math.hypot(px[tip_id][0] - px[mcp_id][0], px[tip_id][1] - px[mcp_id][1])
            d_pip_mcp = math.hypot(px[pip_id][0] - px[mcp_id][0], px[pip_id][1] - px[mcp_id][1])

            # 2D Distances from Wrist
            d_tip_wrist = math.hypot(px[tip_id][0] - p0[0], px[tip_id][1] - p0[1])
            d_pip_wrist = math.hypot(px[pip_id][0] - p0[0], px[pip_id][1] - p0[1])

            if d_pip_mcp == 0:
                continue

            ratio_2d = d_tip_mcp / d_pip_mcp

            # Calibrated ratio > 1.15 to work reliably when hand is far from camera
            # AND tip is farther from wrist than PIP joint (d_tip_wrist > d_pip_wrist - 8)
            if ratio_2d > 1.15 and d_tip_wrist > (d_pip_wrist - 8):
                extended_ids.append(tip_id)

        # Precise Thumb extension check (Thumb tip 4, IP 3, MCP 2, Index MCP 5)
        p2 = px[2]
        p3 = px[3]
        p4 = px[4]
        p5 = px[5]

        d_thumb_tip_mcp = math.hypot(p4[0] - p2[0], p4[1] - p2[1])
        d_thumb_ip_mcp = math.hypot(p3[0] - p2[0], p3[1] - p2[1])
        d_thumb_tip_index_mcp = math.hypot(p4[0] - p5[0], p4[1] - p5[1])
        d_thumb_mcp_index_mcp = math.hypot(p2[0] - p5[0], p2[1] - p5[1])

        d_thumb_wrist = math.hypot(p4[0] - p0[0], p4[1] - p0[1])
        d_ip_wrist = math.hypot(p3[0] - p0[0], p3[1] - p0[1])

        if d_thumb_ip_mcp > 0:
            ratio_thumb = d_thumb_tip_mcp / d_thumb_ip_mcp
            if (ratio_thumb > 1.15) and (d_thumb_tip_index_mcp > d_thumb_mcp_index_mcp * 1.05) and (d_thumb_wrist > (d_ip_wrist - 5)):
                extended_ids.append(4)

        return sorted(extended_ids)

    def draw_fingertip_polygon(self, frame, hand_dict, alpha=0.35):
        """
        Draw white mesh lines, cyan dots, and coordinates ONLY on extended fingertips.
        - Single extended finger: cyan dots + coordinates at Tip & PIP joint with connecting white line (Image 2).
        - 2+ extended fingers: translucent mesh polygon + white outline + cyan dots + coordinates (Image 3).
        """
        if not hand_dict or 'px' not in hand_dict:
            return frame

        px_list = hand_dict['px']
        extended_ids = self.get_extended_fingertip_ids(hand_dict)

        if not extended_ids:
            return frame

        # -------------------------------------------------------------
        # Single Extended Finger Mode (Matches Reference Image 2)
        # -------------------------------------------------------------
        if len(extended_ids) == 1:
            tip_id = extended_ids[0]
            pip_id = self.PIP_MAP.get(tip_id, tip_id)

            t_x, t_y = int(px_list[tip_id][0]), int(px_list[tip_id][1])
            p_x, p_y = int(px_list[pip_id][0]), int(px_list[pip_id][1])

            # White connecting line between Fingertip and PIP Joint
            cv2.line(frame, (t_x, t_y), (p_x, p_y), self.wireframe_color, 1, cv2.LINE_AA)

            # Render Cyan Dot & Coordinates on Fingertip
            cv2.circle(frame, (t_x, t_y), 5, self.dot_color, cv2.FILLED, cv2.LINE_AA)
            cv2.circle(frame, (t_x, t_y), 7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{t_x},{t_y}", (t_x - 20, t_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.text_color, 1, cv2.LINE_AA)

            # Render Cyan Dot & Coordinates on PIP Joint
            cv2.circle(frame, (p_x, p_y), 4, self.dot_color, cv2.FILLED, cv2.LINE_AA)
            cv2.circle(frame, (p_x, p_y), 6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{p_x},{p_y}", (p_x + 8, p_y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.text_color, 1, cv2.LINE_AA)

            return frame

        # -------------------------------------------------------------
        # Multiple Extended Fingers Mesh Mode (Matches Reference Image 3)
        # -------------------------------------------------------------
        pts = [(int(px_list[tip_id][0]), int(px_list[tip_id][1])) for tip_id in extended_ids]
        pts_arr = np.array(pts, dtype=np.int32)

        overlay = frame.copy()
        if len(extended_ids) >= 3:
            hull = cv2.convexHull(pts_arr)
            cv2.fillPoly(overlay, [hull], self.poly_color)
            cv2.polylines(frame, [hull], True, self.wireframe_color, 1, cv2.LINE_AA)
        else:
            cv2.line(frame, pts[0], pts[1], self.wireframe_color, 1, cv2.LINE_AA)

        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Render cyan dots and coordinate text on each extended fingertip
        for tip_id in extended_ids:
            cx, cy = int(px_list[tip_id][0]), int(px_list[tip_id][1])
            cv2.circle(frame, (cx, cy), 4, self.dot_color, cv2.FILLED, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{cx},{cy}", (cx - 20, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.text_color, 1, cv2.LINE_AA)

        return frame

    def draw_dual_hand_3d_wireframe(self, frame, hand1, hand2, alpha=0.35):
        """
        Dynamically connects extended fingertips of Hand 1 directly to extended fingertips of Hand 2.
        Renders 3D glass box wireframe volume between hands, guaranteed to connect near or far.
        """
        if not hand1 or not hand2:
            return frame

        h1_ext = self.get_extended_fingertip_ids(hand1)
        h2_ext = self.get_extended_fingertip_ids(hand2)

        # Fallback to all fingertips [4, 8, 12, 16, 20] if specific extended list is empty
        # ensuring 2 hands ALWAYS connect seamlessly when detected!
        if not h1_ext:
            h1_ext = [4, 8, 12, 16, 20]
        if not h2_ext:
            h2_ext = [4, 8, 12, 16, 20]

        h1_px = hand1['px']
        h2_px = hand2['px']

        h1_pts = [(int(h1_px[idx][0]), int(h1_px[idx][1])) for idx in h1_ext]
        h2_pts = [(int(h2_px[idx][0]), int(h2_px[idx][1])) for idx in h2_ext]

        overlay = frame.copy()

        # Mesh for Hand 1 extended fingers
        if len(h1_pts) >= 3:
            h1_hull = cv2.convexHull(np.array(h1_pts, dtype=np.int32))
            cv2.fillPoly(overlay, [h1_hull], self.poly_color)
            cv2.polylines(frame, [h1_hull], True, self.wireframe_color, 1, cv2.LINE_AA)
        elif len(h1_pts) == 2:
            cv2.line(frame, h1_pts[0], h1_pts[1], self.wireframe_color, 1, cv2.LINE_AA)

        # Mesh for Hand 2 extended fingers
        if len(h2_pts) >= 3:
            h2_hull = cv2.convexHull(np.array(h2_pts, dtype=np.int32))
            cv2.fillPoly(overlay, [h2_hull], self.poly_color)
            cv2.polylines(frame, [h2_hull], True, self.wireframe_color, 1, cv2.LINE_AA)
        elif len(h2_pts) == 2:
            cv2.line(frame, h2_pts[0], h2_pts[1], self.wireframe_color, 1, cv2.LINE_AA)

        # Connect matching or union tips across Hand 1 and Hand 2
        common_tips = set(h1_ext).intersection(set(h2_ext))
        if not common_tips:
            common_tips = set(h1_ext).union(set(h2_ext))

        for tip_id in common_tips:
            pt1 = (int(h1_px[tip_id][0]), int(h1_px[tip_id][1]))
            pt2 = (int(h2_px[tip_id][0]), int(h2_px[tip_id][1]))
            cv2.line(frame, pt1, pt2, self.wireframe_color, 1, cv2.LINE_AA)

        # Fill side faces for 3D glass box volume (Matches Reference Image 4 & 5)
        sorted_common = sorted(list(common_tips))
        for i in range(len(sorted_common) - 1):
            t1 = sorted_common[i]
            t2 = sorted_common[i + 1]
            face_pts = np.array([
                (int(h1_px[t1][0]), int(h1_px[t1][1])),
                (int(h2_px[t1][0]), int(h2_px[t1][1])),
                (int(h2_px[t2][0]), int(h2_px[t2][1])),
                (int(h1_px[t2][0]), int(h1_px[t2][1]))
            ], dtype=np.int32)
            cv2.fillPoly(overlay, [face_pts], (150, 155, 160))

        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Render cyan dots and coordinate text on extended fingertips of both hands
        for tip_id in h1_ext:
            cx, cy = int(h1_px[tip_id][0]), int(h1_px[tip_id][1])
            cv2.circle(frame, (cx, cy), 4, self.dot_color, cv2.FILLED, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{cx},{cy}", (cx - 20, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.text_color, 1, cv2.LINE_AA)

        for tip_id in h2_ext:
            cx, cy = int(h2_px[tip_id][0]), int(h2_px[tip_id][1])
            cv2.circle(frame, (cx, cy), 4, self.dot_color, cv2.FILLED, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{cx},{cy}", (cx - 20, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.text_color, 1, cv2.LINE_AA)

        return frame
