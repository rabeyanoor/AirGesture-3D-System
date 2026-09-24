// Hand-controlled 3D boxes (port of ar_objects_3d.py).
// World frame: x right, y down, z into the screen; camera at z = -CAM.
import { dist, handScale, extended, isFist } from './gestures.js';

const CAM = 7.0;

// ---------------------------------------------------------------- 3x3 math
const I3 = () => [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
const mul = (A, B) => A.map((r) => [0, 1, 2].map((j) => r[0] * B[0][j] + r[1] * B[1][j] + r[2] * B[2][j]));
const mv = (A, v) => [A[0][0] * v[0] + A[0][1] * v[1] + A[0][2] * v[2],
  A[1][0] * v[0] + A[1][1] * v[1] + A[1][2] * v[2], A[2][0] * v[0] + A[2][1] * v[1] + A[2][2] * v[2]];
const tr = (A) => [[A[0][0], A[1][0], A[2][0]], [A[0][1], A[1][1], A[2][1]], [A[0][2], A[1][2], A[2][2]]];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scl = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const norm = (a) => Math.hypot(a[0], a[1], a[2]);

export function rodrigues(v) {
  const th = norm(v);
  if (th < 1e-9) return I3();
  const k = scl(v, 1 / th);
  const K = [[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]];
  const K2 = mul(K, K), s = Math.sin(th), c = 1 - Math.cos(th);
  return [0, 1, 2].map((i) => [0, 1, 2].map((j) => (i === j ? 1 : 0) + s * K[i][j] + c * K2[i][j]));
}

function orthonormalize(R) {
  // Gram-Schmidt on the columns
  let x = [R[0][0], R[1][0], R[2][0]];
  let y = [R[0][1], R[1][1], R[2][1]];
  x = scl(x, 1 / norm(x));
  y = add(y, scl(x, -dot(x, y)));
  y = scl(y, 1 / norm(y));
  const z = cross(x, y);
  return [[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]];
}

const smooth = (t) => { t = Math.min(1, Math.max(0, t)); return t * t * (3 - 2 * t); };
const wrap = (a) => ((a + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
/** Convex hull of 2D points (monotone chain), counter-clockwise. */
function convexHull(points) {
  const p = points.map((q) => [q[0], q[1]]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cr = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [], upper = [];
  for (const q of p) { while (lower.length >= 2 && cr(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop(); lower.push(q); }
  for (const q of p.slice().reverse()) { while (upper.length >= 2 && cr(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop(); upper.push(q); }
  return lower.slice(0, -1).concat(upper.slice(0, -1));
}

// ---------------------------------------------------------------- shared base
class Box3D {
  constructor(center, size) {
    this.defaultCenter = center.slice();
    this.defaultSize = size;
    this.center = center.slice();     // fraction of W, H
    this.size = size;                 // world unit -> fraction of min(W, H)
    this.R = rodrigues([0.45, -0.6, 0]);
    this.omega = [0, 0, 0];
    this.mode = null;                 // null | rotate | carry | scale
    this.modeLabel = null;
    this.pinching = {};
    this.last = null;
    this.burstArmed = null;
    this.burstT = null;
    this.burstLast = 0;
    this.frags = null;
    this.appearT = null;
    this.appear = 1;
    this.bbox = null;
    this.W = 1280; this.H = 720;
  }

  get grabbed() { return this.mode !== null; }
  get bursting() { return this.burstT !== null; }
  extent() { return 1; }

  project(p) {
    const s = this.size * Math.min(this.W, this.H) * this.appear;
    const f = CAM / (CAM + p[2]);
    return [this.center[0] * this.W + p[0] * s * f, this.center[1] * this.H + p[1] * s * f];
  }

  facing(center, n) { return dot(n, [center[0], center[1], center[2] + CAM]) < 0; }

  updateBBox() {
    const e = this.extent();
    let x1 = Infinity, y1 = Infinity, x2 = -Infinity, y2 = -Infinity;
    for (const x of [-e, e]) for (const y of [-e, e]) for (const z of [-e, e]) {
      const [sx, sy] = this.project(mv(this.R, [x, y, z]));
      x1 = Math.min(x1, sx); y1 = Math.min(y1, sy); x2 = Math.max(x2, sx); y2 = Math.max(y2, sy);
    }
    this.bbox = [x1, y1, x2, y2];
  }

  contains(pt, margin = 0.15) {
    if (!this.bbox) return false;
    const [x1, y1, x2, y2] = this.bbox;
    const mx = (x2 - x1) * margin, my = (y2 - y1) * margin;
    return pt[0] >= x1 - mx && pt[0] <= x2 + mx && pt[1] >= y1 - my && pt[1] <= y2 + my;
  }

  /** Drag right -> front face moves right; drag down -> front face moves down; roll = in-plane twist. */
  rotateBy(dx, dy, droll = 0, keepMomentum = true) {
    const k = 0.011 * 1280 / Math.max(this.W, this.H);
    const step = [dy * k, -dx * k, droll];
    if (keepMomentum) this.omega = this.omega.map((o, i) => o * 0.5 + step[i] * 0.5);
    this.R = orthonormalize(mul(rodrigues(step), this.R));
  }

  setCenter(cx, cy) { this.center = [Math.min(0.92, Math.max(0.08, cx)), Math.min(0.92, Math.max(0.08, cy))]; }
  setSize(s) { this.size = Math.min(this.defaultSize * 2.4, Math.max(this.defaultSize * 0.4, s)); }

  handInfo(h) {
    const px = h.px;
    const fist = isFist(px);
    const was = this.pinching[h.label] || false;
    const pinch = !fist && dist(px[4], px[8]) < handScale(px) * (was ? 0.6 : 0.4);
    this.pinching[h.label] = pinch;
    const nExt = [[8, 6, 5], [12, 10, 9], [16, 14, 13], [20, 18, 17]].filter((f) => extended(px, ...f)).length;
    return {
      label: h.label, pinch, fist, open: nExt >= 3 && !pinch && !fist,
      pt: [(px[4][0] + px[8][0]) / 2, (px[4][1] + px[8][1]) / 2],
      roll: Math.atan2(px[9][1] - px[0][1], px[9][0] - px[0][0]),
      palm: [(px[0][0] + px[5][0] + px[17][0]) / 3, (px[0][1] + px[5][1] + px[17][1]) / 3],
    };
  }

  /** Hand interaction state machine; returns a status label or null. */
  interact(hands, W, H, now) {
    this.W = W; this.H = H;
    if (this.bursting) return this.updateBurst(now);
    this.appear = this.appearT === null ? 1 : smooth((now - this.appearT) / 0.35);
    this.updateBBox();
    const infos = hands.map((h) => this.handInfo(h));
    this.cursors = infos.map((i) => [i.pt, i.pinch]);
    let status = null;

    if (this.mode === null && this.checkBurst(infos, now)) return 'BOX BURST!';

    const pinchers = infos.filter((i) => i.pinch);
    if (pinchers.length >= 2) {
      const [a, b] = pinchers.slice(0, 2).sort((p, q) => p.pt[0] - q.pt[0]);
      const d = Math.max(1, dist(a.pt, b.pt));
      const mid = [(a.pt[0] + b.pt[0]) / 2, (a.pt[1] + b.pt[1]) / 2];
      const ang = Math.atan2(b.pt[1] - a.pt[1], b.pt[0] - a.pt[0]);
      if (this.mode !== 'scale' && (this.mode === 'rotate' || this.contains(a.pt, 0.35) || this.contains(b.pt, 0.35))) {
        this.mode = 'scale';
        this.omega = [0, 0, 0];
        this.last = { d, size: this.size, mid, center: this.center.slice(), ang };
      }
      if (this.mode === 'scale') {
        const r = this.last;
        this.setSize(r.size * d / r.d);
        this.setCenter(r.center[0] + (mid[0] - r.mid[0]) / W, r.center[1] + (mid[1] - r.mid[1]) / H);
        this.R = orthonormalize(mul(rodrigues([0, 0, wrap(ang - r.ang)]), this.R));
        r.ang = ang;
        this.updateBBox();
        return 'RESIZING BOX';
      }
    } else if (this.mode === 'scale') {
      this.mode = null;
    }

    if (this.mode !== null && this.mode !== 'scale') {
      const info = infos.find((i) => i.label === this.modeLabel);
      status = info ? this.continueGrab(info, W, H, now) : null;
      if (status === null) this.mode = null;
    }

    if (this.mode === null) {
      for (const info of infos) {
        status = this.startGrab(info, now);
        if (status) { this.modeLabel = info.label; break; }
      }
    }
    this.coast();
    this.updateBBox();
    return status;
  }

  /** True once an open palm has rested on the box for `hold` s (a quick pass never grabs). */
  palmHold(info, now, hold = 0.4) {
    this.palmDwell ||= {};
    if (info.open && this.contains(info.palm, 0)) {
      this.palmDwell[info.label] ??= now;
      return now - this.palmDwell[info.label] >= hold;
    }
    delete this.palmDwell[info.label];
    return false;
  }

  /** Default boxes: fist inside or palm held on it = carry, pinch inside = rotate. */
  startGrab(info, now) {
    if ((info.fist && this.contains(info.palm, 0.1)) || this.palmHold(info, now)) {
      this.mode = 'carry'; this.last = { palm: info.palm };
      if (this.palmDwell) delete this.palmDwell[info.label];
      return 'GRABBED BOX (MOVE)';
    }
    if (info.pinch && this.contains(info.pt)) {
      this.mode = 'rotate'; this.last = { pt: info.pt, roll: info.roll };
      this.omega = [0, 0, 0];
      return 'GRABBED BOX (ROTATE)';
    }
    return null;
  }

  /** Status while the grab continues, or null to release it. */
  continueGrab(info, W, H, now) {
    if (this.mode === 'rotate' && info.pinch) {
      this.rotateBy(info.pt[0] - this.last.pt[0], info.pt[1] - this.last.pt[1], wrap(info.roll - this.last.roll));
      this.last = { pt: info.pt, roll: info.roll };
      return 'ROTATING BOX';
    }
    if (this.mode === 'carry' && (info.fist || info.open)) {
      this.setCenter(this.center[0] + (info.palm[0] - this.last.palm[0]) / W,
        this.center[1] + (info.palm[1] - this.last.palm[1]) / H);
      this.last = { palm: info.palm };
      return 'MOVING BOX';
    }
    return null;
  }

  /** Momentum after a release (hand or touch). */
  coast() {
    if (this.mode !== 'rotate' && this.mode !== 'spin' && !this.touchHeld && norm(this.omega) > 1e-4) {
      this.R = orthonormalize(mul(rodrigues(this.omega), this.R));
      this.omega = scl(this.omega, 0.92);
    }
  }

  checkBurst(infos, now) {
    const opens = infos.filter((i) => i.open);
    if (opens.length < 2 || !this.bbox) { this.burstArmed = null; return false; }
    const [a, b] = opens.slice(0, 2).sort((p, q) => p.palm[0] - q.palm[0]);
    const [x1, y1, x2, y2] = this.bbox;
    const bw = x2 - x1, bh = y2 - y1, cx = (x1 + x2) / 2;
    const beside = [a, b].every((i) => i.palm[1] >= y1 - 0.3 * bh && i.palm[1] <= y2 + 0.3 * bh) &&
      a.palm[0] < cx && cx < b.palm[0];
    if (!beside) { this.burstArmed = null; return false; }
    const gap = b.palm[0] - a.palm[0];
    if (gap > 0.85 * bw) { this.burstArmed = now; return false; }
    if (this.burstArmed !== null && now - this.burstArmed < 2 && gap < 0.5 * bw) {
      this.burstArmed = null;
      this.startBurst(now);
      return true;
    }
    return false;
  }

  startBurst(now) {
    const seeds = this.fragmentSeeds();
    this.frags = seeds.map((f) => {
      const n = Math.max(norm(f.p), 1e-3);
      const dir = [f.p[0] / n + (Math.random() - 0.5) * 0.7, f.p[1] / n + (Math.random() - 0.5) * 0.7,
        f.p[2] / n + (Math.random() - 0.5) * 0.7];
      const sp = 3 + Math.random() * 4.5;
      return { ...f, v: [dir[0] * sp, dir[1] * sp - 2.5, dir[2] * sp], ang: Math.random() * Math.PI,
        spin: (Math.random() - 0.5) * 18 };
    });
    this.burstT = this.burstLast = now;
    this.mode = null;
    this.omega = [0, 0, 0];
  }

  updateBurst(now) {
    const dt = Math.min(Math.max(now - this.burstLast, 0), 0.05);
    this.burstLast = now;
    for (const f of this.frags) {
      f.v[1] += 12 * dt;
      f.p = add(f.p, scl(f.v, dt));
      f.ang += f.spin * dt;
    }
    if (now - this.burstT >= 1.5) {
      this.burstT = null;
      this.frags = null;
      this.respawn();
      this.appearT = now;
      this.appear = 0;
      return null;
    }
    return 'BOX BURST!';
  }

  drawBurst(ctx, now) {
    const t = (now - this.burstT) / 1.5;
    ctx.save();
    ctx.globalAlpha = Math.max(0, 1 - t * t);
    if (t < 0.3 && this.bbox) {
      const [x1, y1, x2, y2] = this.bbox;
      ctx.strokeStyle = 'rgba(200,240,255,0.9)';
      ctx.lineWidth = Math.max(2, 8 * (1 - t / 0.3));
      ctx.beginPath();
      ctx.arc((x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * (0.4 + t * 3), 0, Math.PI * 2);
      ctx.stroke();
    }
    const s = this.size * Math.min(this.W, this.H);
    const order = this.frags.slice().sort((a, b) => b.p[2] - a.p[2]);
    for (const f of order) {
      const [x, y] = this.project(f.p);
      const r = Math.max(1.5, f.s * s * (CAM / (CAM + f.p[2])) * 0.5);
      ctx.fillStyle = f.c;
      if (f.shape === 0) {
        ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
      } else {
        ctx.save(); ctx.translate(x, y); ctx.rotate(f.ang); ctx.fillRect(-r, -r, 2 * r, 2 * r); ctx.restore();
      }
    }
    ctx.restore();
  }

  /** Pinch cursor per hand: hollow ring = open, filled = pinching. */
  drawCursors(ctx) {
    const r = Math.max(6, Math.min(this.W, this.H) / 90);
    for (const [pt, pinch] of this.cursors || []) {
      ctx.beginPath(); ctx.arc(pt[0], pt[1], r, 0, Math.PI * 2);
      if (pinch) { ctx.fillStyle = '#ffe650'; ctx.fill(); }
      ctx.beginPath(); ctx.arc(pt[0], pt[1], r + 3, 0, Math.PI * 2);
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();
    }
  }

  resetOrientation() { this.R = rodrigues([0.45, -0.6, 0]); this.omega = [0, 0, 0]; }
  resetPlacement() { this.center = this.defaultCenter.slice(); this.size = this.defaultSize; }
}

// ---------------------------------------------------------------- liquid box
const CORNERS = [];
for (const x of [-1, 1]) for (const y of [-1, 1]) for (const z of [-1, 1]) CORNERS.push([x, y, z]);
const EDGES = [[0, 1], [1, 3], [3, 2], [2, 0], [4, 5], [5, 7], [7, 6], [6, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
const FACES = [[[0, 1, 3, 2], [-1, 0, 0]], [[4, 6, 7, 5], [1, 0, 0]], [[0, 4, 5, 1], [0, -1, 0]],
  [[2, 3, 7, 6], [0, 1, 0]], [[0, 2, 6, 4], [0, 0, -1]], [[1, 5, 7, 3], [0, 0, 1]]];

export class LiquidBox extends Box3D {
  constructor(n = 240) {
    super([0.6, 0.5], 0.2 * 720 / 720);
    this.n = n;
    this.radius = 0.08;
    this.h = 0.22;
    this.gravity = 7;
    this.p = new Float32Array(n * 3);
    this.v = new Float32Array(n * 3);
    this.acc = new Float32Array(n * 3);
    this.vs = new Float32Array(n * 3);
    this.cnt = new Float32Array(n);
    this.resetLiquid();
    this.prevR = this.R;
    this.lastT = null;
  }

  resetLiquid() {
    const RT = tr(this.R);
    for (let i = 0; i < this.n; i++) {
      const w = [(Math.random() * 2 - 1) * 0.9, Math.random() * 0.9, (Math.random() * 2 - 1) * 0.9];
      const l = mv(RT, w);
      this.p.set(l, i * 3);
      this.v[i * 3] = this.v[i * 3 + 1] = this.v[i * 3 + 2] = 0;
    }
  }

  step(dt) {
    const { n, p, v, acc, vs, cnt, h } = this;
    const g = mv(tr(this.R), [0, this.gravity, 0]);
    acc.fill(0); vs.fill(0); cnt.fill(0);
    const h2 = h * h;
    for (let i = 0; i < n; i++) {
      const ix = i * 3;
      for (let j = i + 1; j < n; j++) {
        const jx = j * 3;
        const dx = p[ix] - p[jx], dy = p[ix + 1] - p[jx + 1], dz = p[ix + 2] - p[jx + 2];
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 >= h2 || d2 < 1e-12) continue;
        const d = Math.sqrt(d2);
        const w = (h - d) / (h * d);
        acc[ix] += w * dx; acc[ix + 1] += w * dy; acc[ix + 2] += w * dz;
        acc[jx] -= w * dx; acc[jx + 1] -= w * dy; acc[jx + 2] -= w * dz;
        vs[ix] += v[jx]; vs[ix + 1] += v[jx + 1]; vs[ix + 2] += v[jx + 2];
        vs[jx] += v[ix]; vs[jx + 1] += v[ix + 1]; vs[jx + 2] += v[ix + 2];
        cnt[i] += 1; cnt[j] += 1;
      }
    }
    const lim = 1 - this.radius;
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < 3; k++) {
        const q = i * 3 + k;
        v[q] += g[k] * dt + acc[q] * 55 * dt;
        if (cnt[i] > 0) v[q] += (vs[q] / cnt[i] - v[q]) * 0.08;
        v[q] *= 0.995;
        p[q] += v[q] * dt;
        if (p[q] > lim) { p[q] = lim; v[q] *= -0.3; } else if (p[q] < -lim) { p[q] = -lim; v[q] *= -0.3; }
      }
    }
  }

  update(hands, W, H, now) {
    const status = this.interact(hands, W, H, now);
    if (this.bursting) { this.lastT = now; return status; }
    this.physics(now);
    return status;
  }

  physics(now) {
    const dt = this.lastT === null ? 1 / 30 : Math.min(Math.max(now - this.lastT, 1e-3), 1 / 20);
    this.lastT = now;
    // Keep world-space momentum when the box turns -> the liquid sloshes
    const turn = mul(tr(this.R), this.prevR);
    for (let i = 0; i < this.n; i++) {
      const r = mv(turn, [this.v[i * 3], this.v[i * 3 + 1], this.v[i * 3 + 2]]);
      this.v.set(r, i * 3);
    }
    this.prevR = this.R;
    const steps = dt <= 1 / 40 ? 1 : 2;
    for (let s = 0; s < steps; s++) this.step(dt / steps);
  }

  fragmentSeeds() {
    const out = [];
    for (let i = 0; i < this.n; i++) {
      out.push({ p: mv(this.R, [this.p[i * 3], this.p[i * 3 + 1], this.p[i * 3 + 2]]),
        c: i % 4 ? 'rgb(255,105,180)' : 'rgb(255,210,235)', s: 0.12, shape: 0 });
    }
    EDGES.forEach(([a, b], e) => {
      for (const t of [0.1, 0.37, 0.63, 0.9]) {
        const q = add(CORNERS[a], scl(add(CORNERS[b], scl(CORNERS[a], -1)), t));
        out.push({ p: mv(this.R, q), c: (e + t * 10) % 2 < 1 ? 'rgb(60,60,60)' : 'rgb(220,218,215)', s: 0.28, shape: 1 });
      }
    });
    return out;
  }

  respawn() { this.resetOrientation(); this.resetLiquid(); this.prevR = this.R; }

  draw(ctx, now) {
    if (this.bursting) { this.drawBurst(ctx, now); return; }
    const cw = CORNERS.map((c) => mv(this.R, c));
    const cs = cw.map((c) => this.project(c));
    ctx.fillStyle = 'rgba(210,208,205,0.28)';
    for (const [idx, n] of FACES) {
      const nw = mv(this.R, n);
      const ctr = scl(idx.reduce((acc, i) => add(acc, cw[i]), [0, 0, 0]), 0.25);
      if (!this.facing(ctr, nw)) continue;
      ctx.beginPath();
      idx.forEach((i, k) => (k ? ctx.lineTo(...cs[i]) : ctx.moveTo(...cs[i])));
      ctx.closePath();
      ctx.fill();
    }

    const s = this.size * Math.min(this.W, this.H) * this.appear;
    const pr = Math.max(2, this.radius * s * 1.1);
    const pts = [];
    for (let i = 0; i < this.n; i++) {
      const w = mv(this.R, [this.p[i * 3], this.p[i * 3 + 1], this.p[i * 3 + 2]]);
      pts.push([...this.project(w), w[2]]);
    }
    pts.sort((a, b) => b[2] - a[2]);
    // Keep the liquid strictly inside the box: clip to the convex hull of the projected corners
    ctx.save();
    const hull = convexHull(cs);
    ctx.beginPath();
    hull.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
    ctx.closePath();
    ctx.clip();
    // Soft body: large translucent discs merge into a solid blue volume
    ctx.fillStyle = 'rgba(255,105,180,0.33)';
    for (const [x, y] of pts) { ctx.beginPath(); ctx.arc(x, y, pr * 2.2, 0, Math.PI * 2); ctx.fill(); }
    ctx.fillStyle = 'rgba(255,120,190,0.55)';
    for (const [x, y] of pts) { ctx.beginPath(); ctx.arc(x, y, pr, 0, Math.PI * 2); ctx.fill(); }
    ctx.fillStyle = 'rgba(255,225,245,0.95)';
    for (const [x, y] of pts) ctx.fillRect(x - 1, y - 1, 2, 2);
    ctx.restore();

    ctx.strokeStyle = '#191919';
    ctx.lineWidth = Math.max(2, Math.min(this.W, this.H) / 360);
    ctx.lineJoin = 'round';
    ctx.beginPath();
    for (const [a, b] of EDGES) { ctx.moveTo(...cs[a]); ctx.lineTo(...cs[b]); }
    ctx.stroke();
    this.drawCursors(ctx);
  }
}

// ---------------------------------------------------------------- Rubik's cube
const DIRS = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
const COLORS = { '0,-1,0': '#ffffff', '0,1,0': '#ffd700', '0,0,-1': '#1eaf3c', '0,0,1': '#0a5ac8',
  '-1,0,0': '#ff8200', '1,0,0': '#d21e28' };
const key = (v) => v.map((x) => Math.round(x) + 0).join(',');

export class RubiksCube extends Box3D {
  constructor() {
    super([0.6, 0.5], 0.11);
    this.turnDur = 0.28;
    this.trail = [];
    this.lastTurn = 0;
    this.grabSeen = {};
    this.resetCube();
  }

  extent() { return 1.5; }

  resetCube() {
    this.cubies = [];
    for (const x of [-1, 0, 1]) for (const y of [-1, 0, 1]) for (const z of [-1, 0, 1]) {
      if (!x && !y && !z) continue;
      const pos = [x, y, z];
      const stickers = {};
      for (const n of DIRS) if (dot(pos, n) === 1) stickers[key(n)] = COLORS[key(n)];
      this.cubies.push({ pos, ori: I3(), stickers });
    }
    this.anim = null;     // {axis, layer, sign, t0, f0}
    this.queue = [];
    this.preview = null;  // {axis, layer, angle} while a hand / finger turns a layer live
    this.grip = null;
    this.wasUnsolved = false;
    this.solvedT = null;
  }

  quarter(axis, sign) {
    const v = [0, 0, 0]; v[axis] = sign * Math.PI / 2;
    return rodrigues(v).map((r) => r.map((x) => Math.round(x)));
  }

  applyTurn(axis, layer, sign) {
    const Q = this.quarter(axis, sign);
    for (const c of this.cubies) {
      if (c.pos[axis] === layer) { c.pos = mv(Q, c.pos).map(Math.round); c.ori = mul(Q, c.ori).map((r) => r.map(Math.round)); }
    }
  }

  isSolved() {
    const faces = {};
    for (const c of this.cubies) {
      for (const [n, col] of Object.entries(c.stickers)) {
        const cur = key(mv(c.ori, n.split(',').map(Number)));
        (faces[cur] ||= new Set()).add(col);
      }
    }
    return Object.values(faces).every((s) => s.size === 1);
  }

  scramble(n = 20) {
    this.anim = null; this.queue = [];
    for (let i = 0; i < n; i++) {
      this.applyTurn(Math.floor(Math.random() * 3), [-1, 0, 1][Math.floor(Math.random() * 3)], Math.random() < 0.5 ? -1 : 1);
    }
    this.wasUnsolved = !this.isSolved();
    this.solvedT = null;
  }

  afterTurn(now) {
    if (this.isSolved()) { if (this.wasUnsolved) this.solvedT = now; this.wasUnsolved = false; }
    else this.wasUnsolved = true;
  }

  advance(now) {
    if (!this.anim && this.queue.length) this.anim = { ...this.queue.shift(), t0: now, f0: 0 };
    if (this.anim && now - this.anim.t0 >= this.turnDur * (1 - this.anim.f0)) {
      const { axis, layer, sign } = this.anim;
      this.applyTurn(axis, layer, sign);
      this.anim = null;
      this.afterTurn(now);
      if (this.queue.length) this.advance(now);
    }
  }

  /** Visible outer stickers (no animation): {d, quad (screen), pos, n}. */
  outerFaces() {
    const out = [];
    for (const c of this.cubies) {
      const cw = mv(this.R, c.pos);
      for (const n of DIRS) {
        const nc = mv(c.ori, n);
        if (Math.max(...add(c.pos, nc).map(Math.abs)) !== 2) continue;
        const nw = mv(this.R, nc);
        const fc = add(cw, scl(nw, 0.5));
        if (!this.facing(fc, nw)) continue;
        const u = [nc[1], nc[2], nc[0]];
        const uw = mv(this.R, u), vw = mv(this.R, cross(nc, u));
        const quad = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([a, b]) =>
          this.project(add(fc, add(scl(uw, a * 0.5), scl(vw, b * 0.5)))));
        out.push({ d: norm([fc[0], fc[1], fc[2] + CAM]), quad, pos: c.pos.slice(), n: nc });
      }
    }
    return out;
  }

  /** The sticker under a screen point (nearest to the camera), or null. */
  pick(pt) {
    const inside = (q) => {
      let sgn = 0;
      for (let i = 0; i < 4; i++) {
        const a = q[i], b = q[(i + 1) % 4];
        const c = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0]);
        if (c !== 0) { if (sgn && Math.sign(c) !== sgn) return false; sgn = Math.sign(c); }
      }
      return true;
    };
    const faces = this.outerFaces().sort((a, b) => a.d - b.d);
    return faces.find((f) => inside(f.quad)) || null;
  }

  /** In-plane rotation axis that best matches a drag on a sticker: {k, vhat, pxr, score}. */
  dragAxis(face, dx, dy) {
    const len = Math.hypot(dx, dy) || 1;
    const drag = [dx / len, dy / len];
    const P = mv(this.R, add(face.pos, scl(face.n, 0.5)));
    const p0 = this.project(P);
    let best = null;
    for (let k = 0; k < 3; k++) {
      if (face.n[k] !== 0) continue; // rotation axis must lie in the sticker plane
      const e = [0, 0, 0]; e[k] = 1;
      const p1 = this.project(add(P, scl(cross(mv(this.R, e), P), 0.05)));
      const vs = [p1[0] - p0[0], p1[1] - p0[1]];
      const nv = Math.hypot(vs[0], vs[1]);
      if (nv < 1e-6) continue;
      const score = (vs[0] * drag[0] + vs[1] * drag[1]) / nv;
      if (!best || Math.abs(score) > Math.abs(best.score)) best = { k, vhat: [vs[0] / nv, vs[1] / nv], pxr: nv / 0.05, score };
    }
    return best;
  }

  /** Turn the layer through the picked sticker so the sticker follows the drag. */
  dragTurn(face, dx, dy) {
    const best = this.dragAxis(face, dx, dy);
    if (!best) return false;
    this.queue.push({ axis: best.k, layer: face.pos[best.k], sign: best.score > 0 ? 1 : -1 });
    return true;
  }

  /** pick(), or the nearest visible sticker when the point lands just beside one. */
  pickNear(pt) {
    const f = this.pick(pt);
    if (f) return f;
    let best = null;
    for (const face of this.outerFaces()) {
      const c = face.quad.reduce((a, p) => [a[0] + p[0] / 4, a[1] + p[1] / 4], [0, 0]);
      const size = Math.hypot(face.quad[0][0] - face.quad[2][0], face.quad[0][1] - face.quad[2][1]);
      const d = Math.hypot(pt[0] - c[0], pt[1] - c[1]);
      if (d < 0.8 * size && (!best || d < best.d)) best = { d, face };
    }
    return best ? best.face : null;
  }

  // ---- live layer grip (shared by hand pinch and touch) ----
  beginGrip(pt, roll, face) { this.grip = { start: pt, roll, face, axis: null }; }

  gripAngle(pt, roll) {
    const g = this.grip;
    if (!g || !g.face) return null;
    const dx = pt[0] - g.start[0], dy = pt[1] - g.start[1];
    const droll = wrap(roll - g.roll);
    const bw = this.bbox[2] - this.bbox[0];
    if (g.axis === null) {
      if (Math.abs(droll) > 15 * Math.PI / 180 && Math.hypot(dx, dy) < 0.15 * bw) {
        // Twist: turn the face the sticker is on (axis = sticker normal), like a real cube
        let k = 0;
        for (let i = 1; i < 3; i++) if (Math.abs(g.face.n[i]) > Math.abs(g.face.n[k])) k = i;
        Object.assign(g, { axis: k, twist: true, zsign: this.R[2][k] >= 0 ? 1 : -1 });
      } else if (Math.hypot(dx, dy) > 0.04 * bw) {
        const b = this.dragAxis(g.face, dx, dy);
        if (!b) return null;
        Object.assign(g, { axis: b.k, twist: false, vhat: b.vhat, pxr: b.pxr });
      } else return null;
    }
    const angle = g.twist ? droll * g.zsign
      : 2.0 * (dx * g.vhat[0] + dy * g.vhat[1]) / Math.max(g.pxr, 1e-6); // gain: 90deg ~ 70% of cube width
    return { axis: g.axis, layer: g.face.pos[g.axis], angle: Math.max(-Math.PI / 2, Math.min(Math.PI / 2, angle)) };
  }

  /** Returns a status label; locks in a quarter turn at ~85 degrees and keeps following. */
  gripUpdate(pt, roll, now) {
    if (!this.idle()) return 'HOLDING LAYER';
    const r = this.gripAngle(pt, roll);
    if (!r) { this.preview = null; return 'HOLDING LAYER'; }
    if (Math.abs(r.angle) >= 85 * Math.PI / 180) {
      this.applyTurn(r.axis, r.layer, r.angle > 0 ? 1 : -1);
      this.afterTurn(now);
      this.lastTurn = now;
      this.preview = null;
      this.beginGrip(pt, roll, this.pickNear(pt)); // keep holding, even with no sticker under the hand
      return 'LAYER TURN';
    }
    this.preview = r;
    return 'TURNING LAYER';
  }

  /** Let go: finish the turn if past ~35 degrees, otherwise snap back. */
  endGrip(now) {
    const p = this.preview;
    this.preview = null;
    this.grip = null;
    if (p && Math.abs(p.angle) > 35 * Math.PI / 180) {
      this.anim = { axis: p.axis, layer: p.layer, sign: p.angle > 0 ? 1 : -1, t0: now, f0: Math.abs(p.angle) / (Math.PI / 2) };
    }
  }

  idle() { return !this.anim && !this.queue.length; }

  /** Map a screen swipe starting at (x0, y0) onto a layer turn. */
  swipeTurn(x0, y0, dx, dy) {
    const face = this.pick([x0, y0]);
    if (face && this.dragTurn(face, dx, dy)) return;
    const horiz = Math.abs(dx) >= Math.abs(dy);
    const comp = horiz ? 1 : 0;
    let axis = 0;
    for (let k = 1; k < 3; k++) if (Math.abs(this.R[comp][k]) > Math.abs(this.R[comp][axis])) axis = k;
    const aw = [this.R[0][axis], this.R[1][axis], this.R[2][axis]];
    const coord = horiz ? y0 : x0;
    let layer = 0, bestD = Infinity;
    for (const j of [-1, 0, 1]) {
      const c = this.project(scl(aw, j))[horiz ? 1 : 0];
      if (Math.abs(c - coord) < bestD) { bestD = Math.abs(c - coord); layer = j; }
    }
    const vel = cross(aw, [0, 0, -1.5]);
    const moving = horiz ? vel[0] : vel[1];
    this.queue.push({ axis, layer, sign: moving * (horiz ? dx : dy) > 0 ? 1 : -1 });
  }

  /** Hand grabs: pinch a sticker, then drag or twist = live layer turn; fist = turn the whole cube. */
  startGrab(info, now) {
    if (info.fist && this.contains(info.palm, 0.1)) {
      this.mode = 'spin'; this.last = { palm: info.palm }; this.omega = [0, 0, 0];
      return 'GRABBED CUBE (TURN)';
    }
    if (info.pinch && this.contains(info.pt, 0.05)) {
      const face = this.pickNear(info.pt);
      if (face) { this.mode = 'layer'; this.beginGrip(info.pt, info.roll, face); return 'GRABBED LAYER'; }
    }
    return super.startGrab(info, now);
  }

  continueGrab(info, W, H, now) {
    if (this.mode === 'layer') {
      if (!info.pinch) { this.endGrip(now); return null; }
      return this.gripUpdate(info.pt, info.roll, now);
    }
    if (this.mode === 'spin') {
      if (!info.fist) return null;
      this.rotateBy(info.palm[0] - this.last.palm[0], info.palm[1] - this.last.palm[1]);
      this.last = { palm: info.palm };
      return 'TURNING CUBE';
    }
    return super.continueGrab(info, W, H, now);
  }

  updateSwipe(hands, now) {
    if (this.mode !== null && this.mode !== 'scale') this.grabSeen[this.modeLabel] = now;
    if (this.mode === 'scale' || now - this.lastTurn < 0.4) { this.trail = []; return null; }
    let ptr = null;
    for (const h of hands) {
      if (now - (this.grabSeen[h.label] ?? -1) < 0.35) continue;
      const px = h.px;
      if (extended(px, 8, 6, 5) && !extended(px, 12, 10, 9) && !extended(px, 16, 14, 13)) { ptr = h; break; }
    }
    if (!ptr) { this.trail = []; return null; }
    const tip = ptr.px[8];
    this.trail.push([now, tip[0], tip[1]]);
    this.trail = this.trail.filter((p) => now - p[0] <= 0.45);
    const [, x0, y0] = this.trail[0];
    if (!this.contains([x0, y0], 0.1)) return null;
    const dx = tip[0] - x0, dy = tip[1] - y0;
    const need = 0.28 * (this.bbox[2] - this.bbox[0]);
    if (this.idle() && Math.hypot(dx, dy) > need && (Math.abs(dx) > 1.8 * Math.abs(dy) || Math.abs(dy) > 1.8 * Math.abs(dx))) {
      this.swipeTurn(x0, y0, dx, dy);
      this.lastTurn = now;
      this.trail = [];
      return 'LAYER TURN';
    }
    return null;
  }

  update(hands, W, H, now) {
    const status = this.interact(hands, W, H, now);
    if (this.bursting) { this.preview = null; this.grip = null; return status; }
    // Hand lost / switched to two-hand resize mid-turn (touch grips are ended by the touch handler)
    if (this.mode !== 'layer' && this.grip && !this.touchGrip) this.endGrip(now);
    const swipe = this.updateSwipe(hands, now);
    this.advance(now);
    if (this.solvedT !== null && now - this.solvedT < 2.5) return 'CUBE SOLVED!';
    return swipe || status;
  }

  fragmentSeeds() {
    const out = [];
    for (const c of this.cubies) {
      const ctr = mv(this.R, c.pos);
      out.push({ p: ctr, c: '#121212', s: 0.95, shape: 1 });
      for (const [n, col] of Object.entries(c.stickers)) {
        out.push({ p: add(ctr, scl(mv(this.R, mv(c.ori, n.split(',').map(Number))), 0.55)), c: col, s: 0.8, shape: 1 });
      }
    }
    return out;
  }

  respawn() { this.resetOrientation(); this.resetCube(); }

  draw(ctx, now) {
    if (this.bursting) { this.drawBurst(ctx, now); return; }
    let animR = null, turnAxis = null, turnLayer = null;
    if (this.anim) {
      const a = this.anim;
      const t = a.f0 + (1 - a.f0) * smooth((now - a.t0) / Math.max(this.turnDur * (1 - a.f0), 1e-3));
      const v = [0, 0, 0]; v[a.axis] = a.sign * t * Math.PI / 2;
      animR = rodrigues(v); turnAxis = a.axis; turnLayer = a.layer;
    } else if (this.preview) {
      const v = [0, 0, 0]; v[this.preview.axis] = this.preview.angle;
      animR = rodrigues(v); turnAxis = this.preview.axis; turnLayer = this.preview.layer;
    }
    // Highlight what is held: the whole layer once its axis is known, otherwise the grabbed sticker
    const g = this.grip && this.grip.face ? this.grip : null;
    const hl = (c, nc) => !!g && (g.axis !== null
      ? c.pos[g.axis] === g.face.pos[g.axis]
      : key(c.pos) === key(g.face.pos) && key(nc) === key(g.face.n));
    const faces = [];
    for (const c of this.cubies) {
      const turning = animR && c.pos[turnAxis] === turnLayer;
      const M = turning ? mul(this.R, animR) : this.R;
      const cw = mv(M, c.pos);
      for (const n of DIRS) {
        const nc = mv(c.ori, n);
        const nw = mv(M, nc);
        const fc = add(cw, scl(nw, 0.5));
        if (!this.facing(fc, nw)) continue;
        const outer = Math.max(...add(c.pos, nc).map(Math.abs)) === 2;
        if (!outer && !animR) continue;
        const u = [nc[1], nc[2], nc[0]];
        const uw = mv(M, u), vw = mv(M, cross(nc, u));
        const corner = (k) => [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([a, b]) =>
          this.project(add(fc, add(scl(uw, a * 0.5 * k), scl(vw, b * 0.5 * k)))));
        const col = c.stickers[key(n)];
        faces.push({ d: norm([fc[0], fc[1], fc[2] + CAM]), quad: corner(1), inset: corner(0.8), col,
          hl: outer && !!col && hl(c, nc) });
      }
    }
    faces.sort((a, b) => b.d - a.d);
    const poly = (pts) => { ctx.beginPath(); pts.forEach((p, i) => (i ? ctx.lineTo(...p) : ctx.moveTo(...p))); ctx.closePath(); };
    for (const f of faces) {
      ctx.fillStyle = '#121212'; poly(f.quad); ctx.fill();
      if (f.col) { ctx.fillStyle = f.col; poly(f.inset); ctx.fill(); }
      if (f.hl) { ctx.strokeStyle = '#fff'; ctx.lineWidth = Math.max(2, Math.min(this.W, this.H) / 260); ctx.stroke(); }
    }

    if (this.solvedT !== null && now - this.solvedT < 2.5 && this.bbox) {
      const [x1, y1, x2] = this.bbox;
      const fs = Math.max(16, 26 * Math.min(this.W, this.H) / 720);
      ctx.font = `700 ${fs}px system-ui, sans-serif`;
      const tw = ctx.measureText('SOLVED!').width;
      const cx = (x1 + x2) / 2, ty = Math.max(fs + 16, y1 - 18);
      ctx.fillStyle = '#1eaf3c';
      ctx.fillRect(cx - tw / 2 - 14, ty - fs - 8, tw + 28, fs + 18);
      ctx.fillStyle = '#fff';
      ctx.fillText('SOLVED!', cx - tw / 2, ty);
    }
    this.drawCursors(ctx);
  }
}
