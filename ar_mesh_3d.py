"""
AR 3D Mesh & Wireframe Renderer Module
Minimal hand overlays: cyan dots with "x, y" coordinate callouts (no lines, no fill):
1. Single hand: a dot on every extended fingertip.
2. Single extended finger: dots on the tip and PIP joint.
3. Two hands: dots on the matching extended fingertips of both hands.
"""

import math

import cv2


class ARMesh3D:
    def __init__(self):
        self.FINGERTIP_IDS = [4, 8, 12, 16, 20]
        self.dot_color = (235, 200, 20)          # Cyan dot (BGR)
        self.text_color = (255, 255, 255)        # White coordinate text

        # Mapping tip to PIP joint for single-finger indicator
        self.PIP_MAP = {8: 6, 12: 10, 16: 14, 20: 18, 4: 2}

    # ------------------------------------------------------------------
    # Finger state
    # ------------------------------------------------------------------
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
            d_tip_mcp = math.hypot(px[tip_id][0] - px[mcp_id][0], px[tip_id][1] - px[mcp_id][1])
            d_pip_mcp = math.hypot(px[pip_id][0] - px[mcp_id][0], px[pip_id][1] - px[mcp_id][1])

            d_tip_wrist = math.hypot(px[tip_id][0] - p0[0], px[tip_id][1] - p0[1])
            d_pip_wrist = math.hypot(px[pip_id][0] - p0[0], px[pip_id][1] - p0[1])

            if d_pip_mcp == 0:
                continue

            ratio_2d = d_tip_mcp / d_pip_mcp

            # Calibrated ratio > 1.15 to work reliably when hand is far from camera
            # AND tip is farther from wrist than PIP joint
            if ratio_2d > 1.15 and d_tip_wrist > (d_pip_wrist - 8):
                extended_ids.append(tip_id)

        # Thumb extension check (Thumb tip 4, IP 3, MCP 2, Index MCP 5)
        p2, p3, p4, p5 = px[2], px[3], px[4], px[5]

        d_thumb_tip_mcp = math.hypot(p4[0] - p2[0], p4[1] - p2[1])
        d_thumb_ip_mcp = math.hypot(p3[0] - p2[0], p3[1] - p2[1])
        d_thumb_tip_index_mcp = math.hypot(p4[0] - p5[0], p4[1] - p5[1])
        d_thumb_mcp_index_mcp = math.hypot(p2[0] - p5[0], p2[1] - p5[1])

        d_thumb_wrist = math.hypot(p4[0] - p0[0], p4[1] - p0[1])
        d_ip_wrist = math.hypot(p3[0] - p0[0], p3[1] - p0[1])

        if d_thumb_ip_mcp > 0:
            ratio_thumb = d_thumb_tip_mcp / d_thumb_ip_mcp
            if (ratio_thumb > 1.15) and (d_thumb_tip_index_mcp > d_thumb_mcp_index_mcp * 1.05) \
                    and (d_thumb_wrist > (d_ip_wrist - 5)):
                extended_ids.append(4)

        return sorted(extended_ids)

    # ------------------------------------------------------------------
    # Drawing primitives
    # ------------------------------------------------------------------
    def _draw_point(self, frame, pt, label_offset=(-22, -10), radius=4):
        x, y = pt
        cv2.circle(frame, (x, y), radius, self.dot_color, cv2.FILLED, cv2.LINE_AA)
        scale = max(0.35, frame.shape[1] / 1280.0 * 0.38)
        cv2.putText(frame, f"{x}, {y}", (x + label_offset[0], y + label_offset[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, self.text_color, 1, cv2.LINE_AA)

    @staticmethod
    def _tip_pts(hand_dict, tip_ids):
        return [(int(hand_dict['px'][i][0]), int(hand_dict['px'][i][1])) for i in tip_ids]

    # ------------------------------------------------------------------
    # Single hand
    # ------------------------------------------------------------------
    def draw_fingertip_polygon(self, frame, hand_dict):
        """
        - 1 extended finger: tip + PIP joint dots.
        - 2+ extended fingers: a dot on each extended fingertip.
        """
        if not hand_dict or 'px' not in hand_dict:
            return frame

        px_list = hand_dict['px']
        extended_ids = self.get_extended_fingertip_ids(hand_dict)
        if not extended_ids:
            return frame

        if len(extended_ids) == 1:
            tip_id = extended_ids[0]
            pip_id = self.PIP_MAP.get(tip_id, tip_id)
            tip = (int(px_list[tip_id][0]), int(px_list[tip_id][1]))
            pip = (int(px_list[pip_id][0]), int(px_list[pip_id][1]))
            self._draw_point(frame, tip)
            self._draw_point(frame, pip, label_offset=(8, 4))
            return frame

        pts = self._tip_pts(hand_dict, extended_ids)

        for pt in pts:
            self._draw_point(frame, pt)
        return frame

    # ------------------------------------------------------------------
    # Two hands
    # ------------------------------------------------------------------
    def draw_dual_hand_3d_wireframe(self, frame, hand1, hand2):
        """
        Marks the fingertips that are extended on both hands.
        """
        if not hand1 or not hand2:
            return frame

        # Order hands left -> right on screen so the volume polygon never self-intersects
        if hand1['px'][0][0] > hand2['px'][0][0]:
            hand1, hand2 = hand2, hand1

        h1_ext = self.get_extended_fingertip_ids(hand1)
        h2_ext = self.get_extended_fingertip_ids(hand2)

        shared = [t for t in self.FINGERTIP_IDS if t in h1_ext and t in h2_ext]
        if len(shared) < 2:
            # Hands always connect when both are visible
            shared = list(self.FINGERTIP_IDS)

        chain1 = self._tip_pts(hand1, shared)
        chain2 = self._tip_pts(hand2, shared)


        for pt in chain1 + chain2:
            self._draw_point(frame, pt)
        return frame
