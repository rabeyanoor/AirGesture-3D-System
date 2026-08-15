import cv2
import numpy as np

class WireframeEngine:
    def __init__(self):
        self.angle = 0

    def draw_3d_spatial_mesh(self, frame, landmarks_list):
        """
        Renders rich, futuristic 3D spatial wireframe structures.
        Supports both single-hand rotatable 3D geometry and dual-hand volumetric mesh.
        """
        if not landmarks_list:
            return

        self.angle += 0.05  # Auto-rotation increment for 3D elements

        # Mode 1: Dual-Hand Volumetric Bounding Mesh
        if len(landmarks_list) == 2:
            self._draw_dual_hand_mesh(frame, landmarks_list[0], landmarks_list[1])
        
        # Mode 2: Single-Hand Anchored 3D Rotatable Cube & Skeleton Node Mesh
        for hand_pts in landmarks_list:
            self._draw_hand_skeleton_mesh(frame, hand_pts)
            self._draw_floating_3d_cube(frame, hand_pts)

    def _draw_dual_hand_mesh(self, frame, hand1_pts, hand2_pts):
        """Creates dynamic 3D poly-mesh volume between two hands."""
        # Connect key fingertips across hands (Thumb, Index, Middle, Ring, Pinky)
        fingertip_ids = [4, 8, 12, 16, 20]
        
        mesh_points = []
        for fid in fingertip_ids:
            mesh_points.append(hand1_pts[fid][:2])
        for fid in reversed(fingertip_ids):
            mesh_points.append(hand2_pts[fid][:2])

        poly_pts = np.array(mesh_points, np.int32)

        # Create glowing semi-transparent overlay
        overlay = frame.copy()
        cv2.fillPoly(overlay, [poly_pts], color=(255, 180, 0))
        cv2.polylines(frame, [poly_pts], isClosed=True, color=(0, 255, 255), thickness=2)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        # Render connecting spatial cross-lines
        for fid in fingertip_ids:
            p1 = hand1_pts[fid][:2]
            p2 = hand2_pts[fid][:2]
            cv2.line(frame, p1, p2, (0, 255, 128), 1, cv2.LINE_AA)

    def _draw_hand_skeleton_mesh(self, frame, hand_pts):
        """Draws glowing spatial nodes and bounding box around a single hand."""
        pts_2d = np.array([pt[:2] for pt in hand_pts])
        x, y, w, h = cv2.boundingRect(pts_2d)

        # Draw futuristic corner bracket bounding box
        length = 20
        thick = 2
        color = (0, 255, 0)
        
        # Top-left corner
        cv2.line(frame, (x, y), (x + length, y), color, thick)
        cv2.line(frame, (x, y), (x, y + length), color, thick)
        # Top-right corner
        cv2.line(frame, (x + w, y), (x + w - length, y), color, thick)
        cv2.line(frame, (x + w, y), (x + w, y + length), color, thick)
        # Bottom-left corner
        cv2.line(frame, (x, y + h), (x + length, y + h), color, thick)
        cv2.line(frame, (x, y + h), (x, y + h - length), color, thick)
        # Bottom-right corner
        cv2.line(frame, (x + w, y + h), (x + w - length, y + h), color, thick)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - length), color, thick)

        # Render glowing coordinate rings at key joints
        for pt in [hand_pts[4], hand_pts[8], hand_pts[12], hand_pts[16], hand_pts[20]]:
            cv2.circle(frame, pt[:2], 6, (255, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, pt[:2], 10, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_floating_3d_cube(self, frame, hand_pts):
        """Projects a rotatable 3D wireframe cube floating above the wrist point."""
        wrist = hand_pts[0][:2]
        cx, cy = wrist[0], wrist[1] - 80
        size = 35

        # 3D Cube vertices in local space
        vertices = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1]
        ], dtype=float) * size

        # Rotation matrix around Y and X axes
        cos_a, sin_a = np.cos(self.angle), np.sin(self.angle)
        rot_y = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])
        rot_x = np.array([[1, 0, 0], [0, cos_a, -sin_a], [0, sin_a, cos_a]])
        rot_matrix = np.dot(rot_x, rot_y)

        rotated_verts = np.dot(vertices, rot_matrix.T)

        # Project 3D coordinates onto 2D viewport
        proj_verts = []
        for v in rotated_verts:
            px = int(cx + v[0])
            py = int(cy + v[1])
            proj_verts.append((px, py))

        # Cube edges definition
        edges = [
            (0,1), (1,2), (2,3), (3,0), # Back face
            (4,5), (5,6), (6,7), (7,4), # Front face
            (0,4), (1,5), (2,6), (3,7)  # Connecting edges
        ]

        # Draw illuminated edges
        for e in edges:
            p1 = proj_verts[e[0]]
            p2 = proj_verts[e[1]]
            cv2.line(frame, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)
