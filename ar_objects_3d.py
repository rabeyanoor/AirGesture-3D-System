"""
AR 3D Objects Module
Hand-controlled 3D boxes opened from the sidebar:
1. LiquidCube  - wireframe glass box filled with glowing particle liquid that flows under gravity
                 as the box is turned.
2. RubiksCube  - 3x3x3 cube with animated layer turns.

Interaction (shared):
- Pinch (thumb tip + index tip) inside the box and drag -> rotate the box (twist the wrist to roll).
  Released boxes keep spinning briefly with decaying momentum.
- Make a fist inside the box and move -> carry the box anywhere on screen.
- Pinch with BOTH hands and pull apart / push together -> resize (and move / roll) the box.
- Press the box from both sides with two open palms -> it bursts into pieces, then pops back fresh.
- Rubik's Cube: pinch on a sticker, then drag OR twist the wrist -> the layer follows the hand live and
  locks in at 90 degrees (keep moving for more turns; let go early and it finishes past ~35 degrees or snaps
  back). The grabbed sticker / layer is outlined in white and a ring shows each hand's pinch point.
  A fist on the cube turns the whole cube; pinching just beside it rotates it too.
  Pointing with only the index finger and swiping across a sticker also turns its layer.
  One hand can hold/turn the cube while the other turns layers (two-hand solving).

World frame: x right, y down (screen), z into the screen. Camera sits at z = -CAM_DIST.
"""

import math
import random
import time

import cv2
import numpy as np

CAM_DIST = 7.0


def rodrigues(axis_angle):
    """Rotation matrix for a rotation vector (axis * angle, right-hand rule)."""
    theta = float(np.linalg.norm(axis_angle))
    if theta < 1e-9:
        return np.eye(3)
    k = np.asarray(axis_angle, dtype=np.float64) / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


def orthonormalize(R):
    u, _, vt = np.linalg.svd(R)
    return u @ vt


def _hand_scale(px):
    return math.hypot(px[9][0] - px[0][0], px[9][1] - px[0][1])


def _extended(px, tip, pip, mcp):
    d_tip = math.hypot(px[tip][0] - px[mcp][0], px[tip][1] - px[mcp][1])
    d_pip = math.hypot(px[pip][0] - px[mcp][0], px[pip][1] - px[mcp][1])
    return d_pip > 0 and d_tip / d_pip > 1.3


def _is_fist(px):
    """At least 3 of the 4 fingers folded (not extended, tip pulled back towards the wrist)."""
    folded = 0
    for tip, pip, mcp in ((8, 6, 5), (12, 10, 9), (16, 14, 13), (20, 18, 17)):
        d_tip = math.hypot(px[tip][0] - px[0][0], px[tip][1] - px[0][1])
        d_pip = math.hypot(px[pip][0] - px[0][0], px[pip][1] - px[0][1])
        if not _extended(px, tip, pip, mcp) and d_tip < d_pip * 1.15:
            folded += 1
    return folded >= 3


def _smoothstep(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


class Object3DBase:
    """
    Screen placement, perspective projection and hand interaction shared by both boxes:
    - Pinch inside + drag (one hand)       -> rotate (wrist twist = roll), momentum after release
    - Fist inside + move, or open palm held on the box ~0.4 s -> carry the box around the screen
    - Pinch with both hands                -> resize (pinch distance), move (midpoint), roll (hand angle)
    - Two open palms squeezing the box     -> burst into fragments, then the box pops back in fresh
    """

    BURST_DURATION = 1.5
    APPEAR_DURATION = 0.35
    MIN_SCALE, MAX_SCALE = 0.4, 2.4    # relative to the default size

    def __init__(self, center=(0.6, 0.52), size=0.17):
        self.default_center = tuple(center)
        self.default_size = size
        self.center_ratio = list(center)
        self.size_ratio = size          # world unit -> fraction of frame height
        self.R = rodrigues([0.45, -0.6, 0.0])
        self.omega = np.zeros(3)        # per-frame rotation vector (momentum)

        self.mode = None                # None | 'rotate' | 'carry' | 'scale'
        self.mode_label = None          # hand label driving 'rotate' / 'carry'
        self._pinching = {}             # hand label -> pinch state (hysteresis)
        self._last = None               # per-mode reference data

        self._burst_armed_t = None
        self.burst_t = None
        self._burst_last_t = None
        self.fragments = None
        self.appear_t = None
        self._appear_scale = 1.0

        self.bbox = None                # (x1, y1, x2, y2) screen bbox of last projection
        self.frame_size = (1280, 720)

    @property
    def grabbed(self):
        return self.mode is not None

    @property
    def bursting(self):
        return self.burst_t is not None

    # --------------------------------------------------------------
    # Projection
    # --------------------------------------------------------------
    def project(self, pts_world):
        w, h = self.frame_size
        cx, cy = self.center_ratio[0] * w, self.center_ratio[1] * h
        s = self.size_ratio * h * self._appear_scale
        pts = np.atleast_2d(pts_world)
        f = CAM_DIST / (CAM_DIST + pts[:, 2])
        return np.stack([cx + pts[:, 0] * s * f, cy + pts[:, 1] * s * f], axis=1)

    @staticmethod
    def facing_camera(center_world, normal_world):
        return float(np.dot(normal_world, center_world - np.array([0.0, 0.0, -CAM_DIST]))) < 0

    def extent(self):
        """Half-size of the object in world units (for the grab bbox)."""
        return 1.0

    def _update_bbox(self):
        e = self.extent()
        corners = np.array([[x, y, z] for x in (-e, e) for y in (-e, e) for z in (-e, e)]) @ self.R.T
        sp = self.project(corners)
        self.bbox = (sp[:, 0].min(), sp[:, 1].min(), sp[:, 0].max(), sp[:, 1].max())

    def contains(self, pt, margin=0.15):
        if self.bbox is None:
            return False
        x1, y1, x2, y2 = self.bbox
        mx, my = (x2 - x1) * margin, (y2 - y1) * margin
        return x1 - mx <= pt[0] <= x2 + mx and y1 - my <= pt[1] <= y2 + my

    # --------------------------------------------------------------
    # Hand features
    # --------------------------------------------------------------
    def _hand_info(self, hand):
        px = hand['px']
        label = hand['label']
        scale = _hand_scale(px)
        fist = _is_fist(px)
        d = math.hypot(px[4][0] - px[8][0], px[4][1] - px[8][1])
        was = self._pinching.get(label, False)
        pinch = (not fist) and d < scale * (0.6 if was else 0.4)
        self._pinching[label] = pinch
        n_ext = sum(_extended(px, t, p, m) for t, p, m in ((8, 6, 5), (12, 10, 9), (16, 14, 13), (20, 18, 17)))
        palm = ((px[0][0] + px[5][0] + px[17][0]) / 3.0, (px[0][1] + px[5][1] + px[17][1]) / 3.0)
        return {
            'label': label,
            'pinch': pinch,
            'fist': fist,
            'open': n_ext >= 3 and not pinch and not fist,
            'pt': ((px[4][0] + px[8][0]) / 2.0, (px[4][1] + px[8][1]) / 2.0),
            'roll': math.atan2(px[9][1] - px[0][1], px[9][0] - px[0][0]),
            'palm': palm,
        }

    # --------------------------------------------------------------
    # Interaction state machine
    # --------------------------------------------------------------
    def update_interaction(self, hands, w, h, now):
        """Returns a gesture label for the HUD (or None)."""
        self.frame_size = (w, h)
        if self.bursting:
            return self._update_burst(now)

        self._appear_scale = 1.0 if self.appear_t is None else \
            _smoothstep((now - self.appear_t) / self.APPEAR_DURATION)
        self._update_bbox()
        infos = [self._hand_info(hd) for hd in hands]
        self._cursors = [(i['pt'], i['pinch']) for i in infos]
        gesture = None

        # 1. Two open palms squeezing the box -> burst
        if self.mode is None and self._check_burst(infos, now):
            return "BOX BURST!"

        # 2. Both hands pinching -> resize / move / roll
        pinchers = [i for i in infos if i['pinch']]
        if len(pinchers) >= 2:
            a, b = sorted(pinchers[:2], key=lambda i: i['pt'][0])
            dist = max(1.0, math.hypot(b['pt'][0] - a['pt'][0], b['pt'][1] - a['pt'][1]))
            mid = ((a['pt'][0] + b['pt'][0]) / 2.0, (a['pt'][1] + b['pt'][1]) / 2.0)
            ang = math.atan2(b['pt'][1] - a['pt'][1], b['pt'][0] - a['pt'][0])
            if self.mode != 'scale' and (self.mode == 'rotate' or self.contains(a['pt'], 0.35)
                                         or self.contains(b['pt'], 0.35)):
                self.mode, self.omega[:] = 'scale', 0
                self._last = {'dist': dist, 'size': self.size_ratio, 'mid': mid,
                              'center': list(self.center_ratio), 'ang': ang}
            if self.mode == 'scale':
                ref = self._last
                lo, hi = self.default_size * self.MIN_SCALE, self.default_size * self.MAX_SCALE
                self.size_ratio = min(hi, max(lo, ref['size'] * dist / ref['dist']))
                self._set_center(ref['center'][0] + (mid[0] - ref['mid'][0]) / w,
                                 ref['center'][1] + (mid[1] - ref['mid'][1]) / h)
                dang = (ang - ref['ang'] + math.pi) % (2 * math.pi) - math.pi
                self.R = orthonormalize(rodrigues([0.0, 0.0, dang]) @ self.R)
                ref['ang'] = ang
                self._update_bbox()
                return "RESIZING BOX"
        elif self.mode == 'scale':
            self.mode = None

        # 3. Continue a one-hand grab
        if self.mode is not None and self.mode != 'scale':
            info = next((i for i in infos if i['label'] == self.mode_label), None)
            gesture = self._continue_grab(info, w, h, now) if info else None
            if gesture is None:
                self.mode = None

        # 4. Start a one-hand grab
        if self.mode is None:
            for info in infos:
                gesture = self._start_grab(info, now)
                if gesture:
                    self.mode_label = info['label']
                    break

        if self.mode not in ('rotate', 'spin') and np.linalg.norm(self.omega) > 1e-4:
            self.R = orthonormalize(rodrigues(self.omega) @ self.R)
            self.omega *= 0.92
        self._update_bbox()
        return gesture

    def _rotate_step(self, dx, dy, droll, w):
        """Drag right -> front face moves right; drag down -> front face moves down; droll = in-plane twist."""
        k = 0.011 * 1280.0 / w  # radians per pixel
        step = np.array([dy * k, -dx * k, droll])
        self.omega = self.omega * 0.5 + step * 0.5
        self.R = orthonormalize(rodrigues(step) @ self.R)

    def _palm_hold(self, info, now, hold=0.4):
        """True once an open palm has rested on the box for `hold` seconds (a quick pass never grabs)."""
        dwell = self.__dict__.setdefault('_palm_dwell', {})
        if info['open'] and self.contains(info['palm'], 0.0):
            t0 = dwell.setdefault(info['label'], now)
            return now - t0 >= hold
        dwell.pop(info['label'], None)
        return False

    def _start_grab(self, info, now):
        """Default boxes: fist inside or palm held on it = carry, pinch inside = rotate. Returns a status or None."""
        if (info['fist'] and self.contains(info['palm'], 0.1)) or self._palm_hold(info, now):
            self.mode, self._last = 'carry', {'palm': info['palm']}
            self.__dict__.get('_palm_dwell', {}).pop(info['label'], None)
            return "GRABBED BOX (MOVE)"
        if info['pinch'] and self.contains(info['pt']):
            self.mode, self._last = 'rotate', {'pt': info['pt'], 'roll': info['roll']}
            self.omega[:] = 0
            return "GRABBED BOX (ROTATE)"
        return None

    def _continue_grab(self, info, w, h, now):
        """Returns a status label while the grab continues, or None to release it."""
        if self.mode == 'rotate' and info['pinch']:
            droll = (info['roll'] - self._last['roll'] + math.pi) % (2 * math.pi) - math.pi
            self._rotate_step(info['pt'][0] - self._last['pt'][0], info['pt'][1] - self._last['pt'][1], droll, w)
            self._last = {'pt': info['pt'], 'roll': info['roll']}
            return "ROTATING BOX"
        if self.mode == 'carry' and (info['fist'] or info['open']):
            dx, dy = info['palm'][0] - self._last['palm'][0], info['palm'][1] - self._last['palm'][1]
            self._set_center(self.center_ratio[0] + dx / w, self.center_ratio[1] + dy / h)
            self._last = {'palm': info['palm']}
            return "MOVING BOX"
        return None

    def _set_center(self, cx, cy):
        self.center_ratio = [min(0.92, max(0.08, cx)), min(0.92, max(0.08, cy))]

    # --------------------------------------------------------------
    # Burst
    # --------------------------------------------------------------
    def _check_burst(self, infos, now):
        opens = [i for i in infos if i['open']]
        if len(opens) < 2 or self.bbox is None:
            self._burst_armed_t = None
            return False
        a, b = sorted(opens[:2], key=lambda i: i['palm'][0])
        x1, y1, x2, y2 = self.bbox
        bw, bh, cx = x2 - x1, y2 - y1, (x1 + x2) / 2.0
        beside = all(y1 - 0.3 * bh <= i['palm'][1] <= y2 + 0.3 * bh for i in (a, b)) \
            and a['palm'][0] < cx < b['palm'][0]
        if not beside:
            self._burst_armed_t = None
            return False
        gap = b['palm'][0] - a['palm'][0]
        if gap > 0.85 * bw:
            # Palms on both sides of the box: armed, waiting for the squeeze
            self._burst_armed_t = now
            return False
        if self._burst_armed_t is not None and now - self._burst_armed_t < 2.0 and gap < 0.5 * bw:
            self._burst_armed_t = None
            self.start_burst(now)
            return True
        return False

    def fragment_seeds(self):
        """(points_world Nx3, colors Nx3 BGR, sizes N in world units, shapes N: 0 = drop, 1 = shard)."""
        raise NotImplementedError

    def on_respawn(self):
        self.reset_orientation()

    def start_burst(self, now):
        P, C, S, shape = self.fragment_seeds()
        rng = np.random.default_rng()
        n = len(P)
        dirs = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-3) + rng.normal(0, 0.35, (n, 3))
        V = dirs * rng.uniform(3.0, 7.5, (n, 1)) + np.array([0.0, -2.5, 0.0])
        self.fragments = {'P': P.astype(np.float64), 'V': V, 'C': C, 'S': S, 'shape': shape,
                          'ang': rng.uniform(0, math.pi, n), 'spin': rng.uniform(-9, 9, n)}
        self.burst_t = self._burst_last_t = now
        self.mode, self.omega[:] = None, 0

    def _update_burst(self, now):
        dt = min(max(now - self._burst_last_t, 0.0), 0.05)
        self._burst_last_t = now
        F = self.fragments
        F['V'][:, 1] += 12.0 * dt
        F['P'] += F['V'] * dt
        F['ang'] += F['spin'] * dt
        if now - self.burst_t >= self.BURST_DURATION:
            self.burst_t, self.fragments = None, None
            self.on_respawn()
            self.appear_t = now
            self._appear_scale = 0.0
            return None
        return "BOX BURST!"

    def render_burst(self, frame, now):
        h, w = frame.shape[:2]
        self.frame_size = (w, h)
        F = self.fragments
        t = (now - self.burst_t) / self.BURST_DURATION
        overlay = frame.copy()

        # Shock ring
        if t < 0.3 and self.bbox is not None:
            x1, y1, x2, y2 = self.bbox
            c = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            r = int((x2 - x1) * (0.4 + t * 3.0))
            cv2.circle(overlay, c, r, (255, 240, 200), max(2, int(8 * (1 - t / 0.3))), cv2.LINE_AA)

        ps = self.project(F['P'])
        f = CAM_DIST / (CAM_DIST + F['P'][:, 2])
        s = self.size_ratio * h
        order = np.argsort(-F['P'][:, 2])
        for i in order:
            x, y = ps[i]
            if not (-50 < x < w + 50 and -50 < y < h + 50):
                continue
            r = max(1.5, F['S'][i] * s * f[i] * 0.5)
            col = tuple(int(c) for c in F['C'][i])
            if F['shape'][i] == 0:
                cv2.circle(overlay, (int(x), int(y)), int(r), col, -1, cv2.LINE_AA)
            else:
                a = F['ang'][i]
                ca, sa = math.cos(a) * r, math.sin(a) * r
                quad = np.array([[x + ca - sa, y + sa + ca], [x - ca - sa, y - sa + ca],
                                 [x - ca + sa, y - sa - ca], [x + ca + sa, y + sa - ca]], dtype=np.int32)
                cv2.fillConvexPoly(overlay, quad, col, cv2.LINE_AA)
        alpha = max(0.0, 1.0 - t ** 2)
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    def draw_cursors(self, frame):
        """Pinch cursor for each hand: hollow ring = open, filled = pinching (shows where a grab will land)."""
        r = max(6, int(frame.shape[1] / 128))
        for pt, pinch in getattr(self, '_cursors', []):
            c = (int(pt[0]), int(pt[1]))
            if pinch:
                cv2.circle(frame, c, r, (80, 230, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, c, r + 3, (255, 255, 255), 2, cv2.LINE_AA)

    def reset_orientation(self):
        self.R = rodrigues([0.45, -0.6, 0.0])
        self.omega[:] = 0

    def reset_placement(self):
        self.center_ratio = list(self.default_center)
        self.size_ratio = self.default_size


# ======================================================================
# Liquid Cube
# ======================================================================
class LiquidCube(Object3DBase):
    CUBE_EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    CUBE_FACES = [  # (vertex indices, outward normal)
        ((0, 1, 3, 2), (-1, 0, 0)), ((4, 6, 7, 5), (1, 0, 0)),
        ((0, 4, 5, 1), (0, -1, 0)), ((2, 3, 7, 6), (0, 1, 0)),
        ((0, 2, 6, 4), (0, 0, -1)), ((1, 5, 7, 3), (0, 0, 1)),
    ]

    def __init__(self, n_particles=240):
        super().__init__(center=(0.6, 0.5), size=0.2)
        self.n = n_particles
        self.radius = 0.08               # particle radius (local units, box half-size = 1)
        self.h = 0.22                    # repulsion range
        self.gravity = 7.0
        self.corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
        self.reset_liquid()
        self._prev_R = self.R.copy()
        self._last_t = None

    def reset_liquid(self):
        rng = np.random.default_rng()
        # Start as a loose block in the lower part of the box (in world "down" direction)
        world = rng.uniform([-0.9, 0.0, -0.9], [0.9, 0.9, 0.9], size=(self.n, 3))
        self.p = (world @ self.R).astype(np.float32)  # world -> local (R^T applied to row vectors)
        self.v = np.zeros_like(self.p)

    def _step(self, dt):
        g_local = (self.R.T @ np.array([0.0, self.gravity, 0.0])).astype(np.float32)
        self.v += g_local * dt

        # Short-range pairwise repulsion keeps the particles spread out like a fluid
        # (dense matrix form: sum_j w_ij (p_i - p_j) = rowsum(w)_i * p_i - (w @ p)_i)
        p = self.p
        sq = (p * p).sum(1)
        dist = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * (p @ p.T), 0.0))
        near = (dist < self.h) & (dist > 1e-6)
        wgt = np.where(near, (self.h - dist) / (self.h * np.maximum(dist, 1e-6)), 0.0)
        push = wgt.sum(1)[:, None] * p - wgt @ p
        self.v += push * 55.0 * dt

        # Light XSPH-style viscosity: blend towards neighbours' mean velocity
        nf = near.astype(np.float32)
        cnt = nf.sum(1)[:, None]
        vmean = np.where(cnt > 0, (nf @ self.v) / np.maximum(cnt, 1), self.v)
        self.v += (vmean - self.v) * 0.08
        self.v *= 0.995

        self.p += self.v * dt

        lim = 1.0 - self.radius
        over = np.abs(self.p) > lim
        self.p = np.clip(self.p, -lim, lim)
        self.v[over] *= -0.3

    def fragment_seeds(self):
        drops = (self.p @ self.R.T).astype(np.float64)
        n = len(drops)
        drop_cols = np.tile(np.array([180, 105, 255]), (n, 1))  # pink drops, every 4th light pink
        drop_cols[::4] = (235, 210, 255)
        # Glass shards along the 12 edges
        t = np.linspace(0.1, 0.9, 4)
        edge_pts = np.array([self.corners[a] + (self.corners[b] - self.corners[a]) * ti
                             for a, b in self.CUBE_EDGES for ti in t]) @ self.R.T
        glass = np.tile(np.array([60, 60, 60]), (len(edge_pts), 1))
        glass[::2] = (215, 218, 220)
        P = np.vstack([drops, edge_pts])
        C = np.vstack([drop_cols, glass])
        S = np.concatenate([np.full(n, 0.12), np.full(len(edge_pts), 0.28)])
        shape = np.concatenate([np.zeros(n, int), np.ones(len(edge_pts), int)])
        return P, C, S, shape

    def on_respawn(self):
        self.reset_orientation()
        self.reset_liquid()
        self._prev_R = self.R.copy()

    def update(self, hands, w, h, now=None):
        now = time.time() if now is None else now
        gesture = self.update_interaction(hands, w, h, now)
        if self.bursting:
            self._last_t = now
            return gesture
        dt = 1 / 30.0 if self._last_t is None else min(max(now - self._last_t, 1e-3), 1 / 20.0)
        self._last_t = now

        # Keep world-space momentum when the box turns (liquid lags behind the walls -> sloshing)
        turn = (self.R.T @ self._prev_R).astype(np.float32)
        self.v = self.v @ turn.T
        self._prev_R = self.R.copy()

        # At most 2 sub-steps so a slow frame never snowballs into slower physics
        steps = 1 if dt <= 1 / 40.0 else 2
        for _ in range(steps):
            self._step(dt / steps)
        return gesture

    def render(self, frame, now=None):
        now = time.time() if now is None else now
        if self.bursting:
            return self.render_burst(frame, now)
        h, w = frame.shape[:2]
        self.frame_size = (w, h)
        corners_w = self.corners @ self.R.T
        cs = self.project(corners_w)
        ci = cs.astype(np.int32)

        # Translucent glass faces (only those facing the camera)
        x1, y1 = max(0, ci[:, 0].min() - 2), max(0, ci[:, 1].min() - 2)
        x2, y2 = min(w, ci[:, 0].max() + 3), min(h, ci[:, 1].max() + 3)
        overlay = frame[y1:y2, x1:x2].copy()
        for idx, n in self.CUBE_FACES:
            nw = self.R @ np.array(n, dtype=np.float64)
            center = corners_w[list(idx)].mean(0)
            if self.facing_camera(center, nw):
                cv2.fillPoly(overlay, [ci[list(idx)] - [x1, y1]], (205, 208, 210), cv2.LINE_AA)
        if x2 > x1 and y2 > y1:
            frame[y1:y2, x1:x2] = cv2.addWeighted(overlay, 0.28, frame[y1:y2, x1:x2], 0.72, 0)

        # Glowing liquid particles
        pw = self.p @ self.R.T
        ps = self.project(pw)
        order = np.argsort(-pw[:, 2])  # far first
        s = self.size_ratio * h * self._appear_scale
        pr = max(2, int(self.radius * s * 1.1))
        pad = pr * 6
        gx1, gy1 = max(0, int(ps[:, 0].min()) - pad), max(0, int(ps[:, 1].min()) - pad)
        gx2, gy2 = min(w, int(ps[:, 0].max()) + pad), min(h, int(ps[:, 1].max()) + pad)
        if gx2 > gx1 and gy2 > gy1:
            # Soft liquid body: particle discs at 1/3 resolution, blurred, then alpha-composited in pink
            ds = 3
            mh, mw = (gy2 - gy1 + ds - 1) // ds, (gx2 - gx1 + ds - 1) // ds
            mask = np.zeros((mh, mw), dtype=np.uint8)
            rs = max(1, pr // ds + 1)
            for i in order:
                cv2.circle(mask, (int((ps[i, 0] - gx1) / ds), int((ps[i, 1] - gy1) / ds)), rs, 255, -1)
            k = rs * 4 + 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)
            mask = cv2.resize(mask, (gx2 - gx1, gy2 - gy1), interpolation=cv2.INTER_LINEAR)
            # Keep the liquid strictly inside the box: clip to the box's on-screen silhouette
            inside = np.zeros_like(mask)
            cv2.fillConvexPoly(inside, cv2.convexHull(ci) - np.array([gx1, gy1], dtype=np.int32), 255, cv2.LINE_AA)
            mask = cv2.multiply(mask, inside, scale=1.0 / 255)
            alpha = (np.clip(mask.astype(np.float32) / 255.0 * 1.6, 0, 1) * 0.85)[..., None]
            roi = frame[gy1:gy2, gx1:gx2]
            liquid_col = np.array([180, 105, 255], dtype=np.float32)  # BGR hot pink
            roi[:] = (roi.astype(np.float32) * (1.0 - alpha) + liquid_col * alpha).astype(np.uint8)
            # Sparkling particle cores (only those inside the silhouette)
            for i in order:
                c = (int(ps[i, 0]) - gx1, int(ps[i, 1]) - gy1)
                if 0 <= c[1] < inside.shape[0] and 0 <= c[0] < inside.shape[1] and inside[c[1], c[0]]:
                    cv2.circle(roi, c, 1, (245, 225, 255), -1, cv2.LINE_AA)

        # Dark wireframe edges on top
        lw = max(2, int(round(w / 640)))
        for a, b in self.CUBE_EDGES:
            cv2.line(frame, tuple(ci[a]), tuple(ci[b]), (25, 25, 25), lw, cv2.LINE_AA)
        self.draw_cursors(frame)
        return frame


# ======================================================================
# Rubik's Cube
# ======================================================================
class RubiksCube(Object3DBase):
    # Sticker colours (BGR) by solved-state face normal
    COLORS = {
        (0, -1, 0): (255, 255, 255),   # top    - white
        (0, 1, 0): (0, 215, 255),      # bottom - yellow
        (0, 0, -1): (60, 175, 30),     # front  - green
        (0, 0, 1): (200, 90, 10),      # back   - blue
        (-1, 0, 0): (0, 130, 255),     # left   - orange
        (1, 0, 0): (40, 30, 210),      # right  - red
    }
    FACE_DIRS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    DRAG_GAIN = 2.0

    def __init__(self):
        super().__init__(center=(0.6, 0.5), size=0.11)
        self.turn_duration = 0.28
        self.reset_cube()
        self._trail = []                # recent (t, x, y) of the pointing fingertip
        self._last_turn_time = 0.0
        self._grab_seen = {}            # hand label -> last time it was holding the cube
        self._was_unsolved = False
        self.solved_t = None

    def extent(self):
        return 1.5

    def reset_cube(self):
        self.cubies = []
        for x in (-1, 0, 1):
            for y in (-1, 0, 1):
                for z in (-1, 0, 1):
                    if (x, y, z) == (0, 0, 0):
                        continue
                    pos = np.array([x, y, z])
                    stickers = {}
                    for n in self.FACE_DIRS:
                        if np.dot(pos, n) == 1:
                            stickers[n] = self.COLORS[n]
                    self.cubies.append({'pos': pos, 'ori': np.eye(3, dtype=int), 'stickers': stickers})
        self.anim = None     # (axis, layer, sign, t_start, start_fraction)
        self.queue = []
        self.preview = None  # (axis, layer, angle) while a hand is turning a layer live
        self._was_unsolved = False
        self.solved_t = None

    def is_solved(self):
        faces = {}
        for c in self.cubies:
            for n, col in c['stickers'].items():
                faces.setdefault(tuple(c['ori'] @ np.array(n)), set()).add(col)
        return all(len(cols) == 1 for cols in faces.values())

    def fragment_seeds(self):
        P, C, S = [], [], []
        for c in self.cubies:
            center = self.R @ c['pos'].astype(np.float64)
            P.append(center); C.append((18, 18, 18)); S.append(0.95)
            for n, col in c['stickers'].items():
                P.append(center + self.R @ (c['ori'] @ np.array(n)) * 0.55)
                C.append(col); S.append(0.8)
        n = len(P)
        return np.array(P), np.array(C), np.array(S), np.ones(n, int)

    def on_respawn(self):
        self.reset_orientation()
        self.reset_cube()

    # --------------------------------------------------------------
    # Layer turns
    # --------------------------------------------------------------
    @staticmethod
    def _quarter(axis, sign):
        v = np.zeros(3)
        v[axis] = sign * math.pi / 2
        return np.rint(rodrigues(v)).astype(int)

    def _apply_turn(self, axis, layer, sign):
        Rq = self._quarter(axis, sign)
        for c in self.cubies:
            if c['pos'][axis] == layer:
                c['pos'] = Rq @ c['pos']
                c['ori'] = Rq @ c['ori']

    def request_turn(self, axis, layer, sign):
        self.queue.append((axis, layer, sign))

    def scramble(self, n=20):
        self.anim, self.queue = None, []
        for _ in range(n):
            self._apply_turn(random.randrange(3), random.choice((-1, 0, 1)), random.choice((-1, 1)))
        self._was_unsolved = not self.is_solved()
        self.solved_t = None

    def _after_turn(self, now):
        if self.is_solved():
            if self._was_unsolved:
                self.solved_t = now
            self._was_unsolved = False
        else:
            self._was_unsolved = True

    def _advance_anim(self, now):
        if self.anim is None and self.queue:
            axis, layer, sign = self.queue.pop(0)
            self.anim = (axis, layer, sign, now, 0.0)
        if self.anim is not None:
            axis, layer, sign, t0, f0 = self.anim
            if now - t0 >= self.turn_duration * (1.0 - f0):
                self._apply_turn(axis, layer, sign)
                self.anim = None
                self._after_turn(now)
                if self.queue:
                    self._advance_anim(now)

    def _outer_faces(self):
        """Visible outer stickers: (depth, screen quad, cubie pos, face normal) - cube-local, no animation."""
        faces = []
        cam = np.array([0.0, 0.0, -CAM_DIST])
        for c in self.cubies:
            center_w = self.R @ c['pos'].astype(np.float64)
            for n in self.FACE_DIRS:
                n_cur = c['ori'] @ np.array(n)
                if np.max(np.abs(c['pos'] + n_cur)) != 2:
                    continue
                n_w = self.R @ n_cur
                fc = center_w + n_w * 0.5
                if not self.facing_camera(fc, n_w):
                    continue
                u = np.array([n_cur[1], n_cur[2], n_cur[0]])
                u_w, v_w = self.R @ u, self.R @ np.cross(n_cur, u)
                quad = np.array([fc + (su * u_w + sv * v_w) * 0.5 for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
                faces.append((float(np.linalg.norm(fc - cam)), self.project(quad).astype(np.float32),
                              c['pos'].copy(), n_cur))
        return faces

    def pick(self, pt):
        """The sticker under a screen point (nearest to the camera), as (cubie pos, face normal), or None."""
        for _, quad, pos, n in sorted(self._outer_faces(), key=lambda f: f[0]):
            if cv2.pointPolygonTest(quad, (float(pt[0]), float(pt[1])), False) >= 0:
                return pos, n
        return None

    def _drag_axis(self, face, dx, dy):
        """
        For a drag on a sticker: the in-plane rotation axis k whose motion best matches the drag, the screen
        direction the sticker moves for a positive turn about k, pixels per radian, and the match score.
        """
        pos, n = face
        drag = np.array([dx, dy], dtype=np.float64)
        drag /= max(np.linalg.norm(drag), 1e-6)
        P = self.R @ (pos + 0.5 * n)
        p0 = self.project(P[None])[0]
        best = None
        for k in range(3):
            if n[k] != 0:
                continue  # the rotation axis must lie in the sticker's plane
            omega = self.R @ np.eye(3)[k]
            vs = self.project((P + 0.05 * np.cross(omega, P))[None])[0] - p0
            nv = np.linalg.norm(vs)
            if nv < 1e-6:
                continue
            score = float(np.dot(vs / nv, drag))
            if best is None or abs(score) > abs(best[3]):
                best = (k, vs / nv, nv / 0.05, score)
        return best

    def drag_turn(self, face, dx, dy):
        """Turn the layer through the picked sticker so that the sticker follows the drag direction."""
        best = self._drag_axis(face, dx, dy)
        if best is None:
            return False
        k, _, _, score = best
        self.request_turn(k, int(face[0][k]), 1 if score > 0 else -1)
        return True

    def pick_near(self, pt):
        """pick(), or the nearest visible sticker when the pinch lands just beside one."""
        face = self.pick(pt)
        if face is not None:
            return face
        best = None
        for _, quad, pos, n in self._outer_faces():
            c = quad.mean(0)
            size = np.linalg.norm(quad[0] - quad[2])
            d = math.hypot(pt[0] - c[0], pt[1] - c[1])
            if d < 0.8 * size and (best is None or d < best[0]):
                best = (d, (pos, n))
        return best[1] if best else None

    def _swipe_to_turn(self, x0, y0, dx, dy):
        """Map a screen swipe starting at (x0, y0) onto a layer turn."""
        face = self.pick((x0, y0))
        if face is not None and self.drag_turn(face, dx, dy):
            return
        # Fallback (swipe started beside the stickers): row / column nearest to the start point
        horizontal = abs(dx) >= abs(dy)
        comp = 1 if horizontal else 0
        axis = int(np.argmax(np.abs(self.R[comp, :])))
        axis_w = self.R[:, axis]
        centers = self.project(np.array([axis_w * j for j in (-1, 0, 1)]))
        coord = y0 if horizontal else x0
        layer = (-1, 0, 1)[int(np.argmin(np.abs(centers[:, 1 if horizontal else 0] - coord)))]
        v = np.cross(axis_w, np.array([0.0, 0.0, -1.5]))
        moving = v[0] if horizontal else v[1]
        swipe = dx if horizontal else dy
        self.request_turn(axis, layer, 1 if moving * swipe > 0 else -1)

    # --------------------------------------------------------------
    # Hand grabs: pinch a sticker + drag = layer turn, fist = turn the whole cube
    # --------------------------------------------------------------
    def _idle(self):
        return self.anim is None and not self.queue

    def _start_grab(self, info, now):
        if info['fist'] and self.contains(info['palm'], 0.1):
            self.mode, self._last = 'spin', {'palm': info['palm']}
            self.omega[:] = 0
            return "GRABBED CUBE (TURN)"
        if info['pinch'] and self.contains(info['pt'], 0.05):
            face = self.pick_near(info['pt'])
            if face is not None:
                self.mode = 'layer'
                self._last = {'start': info['pt'], 'roll': info['roll'], 'face': face, 'axis': None}
                return "GRABBED LAYER"
        return super()._start_grab(info, now)

    def _grip_angle(self, info):
        """Live layer angle from the hand: drag along the sticker's path, or twist the wrist to turn the face."""
        g = self._last
        if g['face'] is None:
            return None
        dx, dy = info['pt'][0] - g['start'][0], info['pt'][1] - g['start'][1]
        droll = (info['roll'] - g['roll'] + math.pi) % (2 * math.pi) - math.pi
        bw = self.bbox[2] - self.bbox[0]
        pos, n = g['face']
        if g['axis'] is None:
            if abs(droll) > math.radians(15) and math.hypot(dx, dy) < 0.15 * bw:
                # Twist: turn the face the sticker is on (axis = sticker normal) like a real cube
                k = int(np.argmax(np.abs(n)))
                g.update(axis=k, twist=True, zsign=1.0 if self.R[2, k] >= 0 else -1.0)
            elif math.hypot(dx, dy) > 0.04 * bw:
                best = self._drag_axis(g['face'], dx, dy)
                if best is None:
                    return None
                k, vhat, px_per_rad, _ = best
                g.update(axis=k, twist=False, vhat=vhat, pxr=px_per_rad)
            else:
                return None
        k = g['axis']
        if g['twist']:
            # Screen-clockwise wrist twist -> screen-clockwise face turn
            angle = droll * g['zsign']
        else:
            # Gain > 1: a quarter turn needs ~70% of the cube width of hand travel, not the full sticker arc
            angle = self.DRAG_GAIN * float(np.dot([dx, dy], g['vhat'])) / max(g['pxr'], 1e-6)
        return k, int(pos[k]), max(-math.pi / 2, min(math.pi / 2, angle))

    def _finish_grip(self, now):
        """Hand let go of a layer: finish the turn if it is past ~35 degrees, otherwise snap back."""
        if self.preview is None:
            return
        axis, layer, angle = self.preview
        self.preview = None
        if abs(angle) > math.radians(35):
            self.anim = (axis, layer, 1 if angle > 0 else -1, now, abs(angle) / (math.pi / 2))

    def _continue_grab(self, info, w, h, now):
        if self.mode == 'layer':
            if not info['pinch']:
                self._finish_grip(now)
                return None
            if not self._idle():
                return "HOLDING LAYER"
            res = self._grip_angle(info)
            if res is None:
                self.preview = None
                return "HOLDING LAYER"
            axis, layer, angle = res
            if abs(angle) >= math.radians(85):
                # Full quarter turn reached while holding: lock it in and keep following the hand
                self._apply_turn(axis, layer, 1 if angle > 0 else -1)
                self._after_turn(now)
                self._last_turn_time = now
                self.preview = None
                # Keep holding (even with no sticker under the hand) until the pinch is released,
                # so a long drag never turns into a whole-cube rotation
                face = self.pick_near(info['pt'])
                self._last = {'start': info['pt'], 'roll': info['roll'], 'face': face, 'axis': None}
                return "LAYER TURN"
            self.preview = (axis, layer, angle)
            return "TURNING LAYER"
        if self.mode == 'spin':
            if not info['fist']:
                return None
            dx, dy = info['palm'][0] - self._last['palm'][0], info['palm'][1] - self._last['palm'][1]
            self._rotate_step(dx, dy, 0.0, w)
            self._last = {'palm': info['palm']}
            return "TURNING CUBE"
        return super()._continue_grab(info, w, h, now)

    def _update_swipe(self, hands, now):
        # The hand holding the cube never swipes; the OTHER hand can turn layers meanwhile (two-hand solving)
        if self.mode is not None and self.mode != 'scale':
            self._grab_seen[self.mode_label] = now
        if self.mode == 'scale' or now - self._last_turn_time < 0.4:
            self._trail = []
            return None

        pointer = None
        for hand in hands:
            if now - self._grab_seen.get(hand['label'], -1.0) < 0.35:
                continue
            px = hand['px']
            if _extended(px, 8, 6, 5) and not _extended(px, 12, 10, 9) and not _extended(px, 16, 14, 13):
                pointer = hand
                break
        if pointer is None:
            self._trail = []
            return None

        tip = pointer['px'][8]
        self._trail.append((now, tip[0], tip[1]))
        self._trail = [p for p in self._trail if now - p[0] <= 0.45]

        t0, x0, y0 = self._trail[0]
        if not self.contains((x0, y0), margin=0.1):
            return None
        dx, dy = tip[0] - x0, tip[1] - y0
        need = 0.28 * (self.bbox[2] - self.bbox[0])
        if self._idle() and math.hypot(dx, dy) > need and (abs(dx) > 1.8 * abs(dy) or abs(dy) > 1.8 * abs(dx)):
            self._swipe_to_turn(x0, y0, dx, dy)
            self._last_turn_time = now
            self._trail = []
            return "LAYER TURN"
        return None

    def update(self, hands, w, h, now=None):
        now = time.time() if now is None else now
        gesture = self.update_interaction(hands, w, h, now)
        if self.bursting:
            self.preview = None
            return gesture
        if self.mode != 'layer' and self.preview is not None:
            self._finish_grip(now)  # hand lost / switched to two-hand resize mid-turn
        swipe = self._update_swipe(hands, now)
        self._advance_anim(now)
        if self.solved_t is not None and now - self.solved_t < 2.5:
            return "CUBE SOLVED!"
        return swipe or gesture

    # --------------------------------------------------------------
    # Rendering
    # --------------------------------------------------------------
    def render(self, frame, now=None):
        now = time.time() if now is None else now
        if self.bursting:
            return self.render_burst(frame, now)
        h, w = frame.shape[:2]
        self.frame_size = (w, h)

        anim_R, anim_axis, anim_layer = np.eye(3), None, None
        if self.anim is not None:
            axis, layer, sign, t0, f0 = self.anim
            t = min(1.0, (now - t0) / max(self.turn_duration * (1.0 - f0), 1e-3))
            t = f0 + (1.0 - f0) * t * t * (3 - 2 * t)  # smoothstep from where the hand left it
            v = np.zeros(3)
            v[axis] = sign * t * math.pi / 2
            anim_R, anim_axis, anim_layer = rodrigues(v), axis, layer
        elif self.preview is not None:
            axis, layer, angle = self.preview
            v = np.zeros(3)
            v[axis] = angle
            anim_R, anim_axis, anim_layer = rodrigues(v), axis, layer
        turning = anim_axis is not None

        # Highlight what the hand is holding: the whole layer once its axis is known, else the sticker
        hl_layer, hl_face = None, None
        if self.mode == 'layer' and self._last and self._last.get('face') is not None:
            if self._last.get('axis') is not None:
                k = self._last['axis']
                hl_layer = (k, int(self._last['face'][0][k]))
            else:
                hl_face = self._last['face']

        half = 0.5
        faces = []  # (depth, body_pts, sticker_pts, color)
        for c in self.cubies:
            M = self.R @ (anim_R if anim_axis is not None and c['pos'][anim_axis] == anim_layer else np.eye(3))
            center_w = M @ c['pos'].astype(np.float64)
            for n in self.FACE_DIRS:
                n_cur = c['ori'] @ np.array(n)                # face normal in cube-local frame
                n_w = M @ n_cur
                fc = center_w + n_w * half
                if not self.facing_camera(fc, n_w):
                    continue
                # Only outer faces need drawing, plus inner faces exposed while a layer is turning
                outer = np.max(np.abs(c['pos'] + n_cur)) == 2
                if not outer and not turning:
                    continue
                # Two in-plane axes of the face
                u = np.array([n_cur[1], n_cur[2], n_cur[0]])
                vv = np.cross(n_cur, u)
                u_w, v_w = M @ u, M @ vv
                quad = np.array([fc + (su * u_w + sv * v_w) * half for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
                inset = np.array([fc + (su * u_w + sv * v_w) * half * 0.8
                                  for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
                color = c['stickers'].get(n)  # stickers are keyed by the solved-state normal
                hl = outer and color is not None and (
                    (hl_layer is not None and c['pos'][hl_layer[0]] == hl_layer[1]) or
                    (hl_face is not None and np.array_equal(c['pos'], hl_face[0]) and np.array_equal(n_cur, hl_face[1])))
                faces.append((float(np.linalg.norm(fc - np.array([0, 0, -CAM_DIST]))), quad, inset, color, hl))

        faces.sort(key=lambda f: -f[0])
        hl_w = max(2, int(round(w / 500)))
        for _, quad, inset, color, hl in faces:
            q = self.project(quad).astype(np.int32)
            cv2.fillPoly(frame, [q], (18, 18, 18), cv2.LINE_AA)
            if color is not None:
                si = self.project(inset).astype(np.int32)
                cv2.fillPoly(frame, [si], color, cv2.LINE_AA)
                cv2.polylines(frame, [si], True, (255, 255, 255) if hl else (35, 35, 35), hl_w if hl else 1,
                              cv2.LINE_AA)

        # "SOLVED!" pill above the cube
        if self.solved_t is not None and now - self.solved_t < 2.5 and self.bbox is not None:
            x1, y1, x2, _ = self.bbox
            text, scale = "SOLVED!", 0.9 * w / 1280.0
            tw, th = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0]
            cx, ty = int((x1 + x2) / 2), max(th + 20, int(y1) - 20)
            cv2.rectangle(frame, (cx - tw // 2 - 14, ty - th - 12), (cx + tw // 2 + 14, ty + 10), (60, 175, 30), -1)
            cv2.putText(frame, text, (cx - tw // 2, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2,
                        cv2.LINE_AA)
        self.draw_cursors(frame)
        return frame
