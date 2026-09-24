// Air writing (port of air_writing.py): hold 2 fingers together (index + middle, ring and pinky folded)
// or 3 fingers together (thumb + index + middle) and write in the air; spread the fingers to lift the pen.
// A short pause (~0.5 s) ends the letter: it is recognised ($P point-cloud recogniser) and typed.
// Horizontal line left -> right = space, right -> left = delete, quick tap = ".".
import { dist, handScale, extended } from './gestures.js';

/** Writing pose: 2 or 3 fingertips together. Returns {down, pt}. Hysteresis once the pen is down. */
export function penPose(px, wasDown = false) {
  const s = Math.max(handScale(px), 1e-6);
  const d = (a, b) => dist(px[a], px[b]) / s;
  const k = wasDown ? 1.5 : 1;
  const two = d(8, 12) < 0.25 * k && !extended(px, 16, 14, 13) && !extended(px, 20, 18, 17);
  const three = d(4, 8) < 0.22 * k && d(4, 12) < 0.22 * k && d(8, 12) < 0.22 * k;
  return { down: two || three, pt: [(px[8][0] + px[12][0]) / 2, (px[8][1] + px[12][1]) / 2] };
}

const N = 32;

function resample(strokes, n = N) {
  const pts = strokes.filter((s) => s.length).map((s) => s.map((p) => [p[0], p[1]]));
  let total = 0;
  for (const s of pts) for (let i = 1; i < s.length; i++) total += dist(s[i - 1], s[i]);
  if (total === 0) return Array.from({ length: n }, () => pts[0][0].slice());
  const step = total / (n - 1);
  const out = [pts[0][0].slice()];
  let acc = 0;
  for (const s of pts) {
    let prev = s[0];
    for (let i = 1; i < s.length; i++) {
      let q = s[i], d = dist(prev, q);
      while (acc + d >= step && d > 0) {
        const t = (step - acc) / d;
        const nw = [prev[0] + t * (q[0] - prev[0]), prev[1] + t * (q[1] - prev[1])];
        out.push(nw);
        prev = nw; d = dist(nw, q); acc = 0;
      }
      acc += d;
      prev = q;
    }
  }
  const last = pts[pts.length - 1];
  while (out.length < n) out.push(last[last.length - 1].slice());
  return out.slice(0, n);
}

/** Scale into a unit square (1-D ink such as an I keeps its aspect ratio), centre on the centroid. */
function normalize(strokes) {
  const p = resample(strokes);
  let mnx = Infinity, mny = Infinity, mxx = -Infinity, mxy = -Infinity;
  for (const [x, y] of p) { mnx = Math.min(mnx, x); mny = Math.min(mny, y); mxx = Math.max(mxx, x); mxy = Math.max(mxy, y); }
  let w = Math.max(mxx - mnx, 1e-6), h = Math.max(mxy - mny, 1e-6);
  if (Math.min(w, h) / Math.max(w, h) < 0.3) w = h = Math.max(w, h);
  const q = p.map(([x, y]) => [(x - mnx) / w, (y - mny) / h]);
  const cx = q.reduce((a, v) => a + v[0], 0) / q.length, cy = q.reduce((a, v) => a + v[1], 0) / q.length;
  return q.map(([x, y]) => [x - cx, y - cy]);
}

function cloudDistance(a, b, start) {
  const n = a.length, matched = new Array(n).fill(false);
  let sum = 0, i = start;
  for (let k = 0; k < n; k++) {
    let best = Infinity, bj = -1;
    for (let j = 0; j < n; j++) {
      if (matched[j]) continue;
      const d = Math.hypot(a[i][0] - b[j][0], a[i][1] - b[j][1]);
      if (d < best) { best = d; bj = j; }
    }
    matched[bj] = true;
    sum += (1 - k / n) * best;
    i = (i + 1) % n;
  }
  return sum;
}

function greedyMatch(a, b) {
  const step = Math.max(1, Math.floor(Math.sqrt(a.length)));
  let min = Infinity;
  for (let i = 0; i < a.length; i += step) min = Math.min(min, cloudDistance(a, b, i), cloudDistance(b, a, i));
  return min;
}

/** D vs O: a D's top-left and bottom-left corners reach its left edge; an O is round there. */
function straightLeft(strokes) {
  const p = strokes.flat();
  const xs = p.map((q) => q[0]), ys = p.map((q) => q[1]);
  const mnx = Math.min(...xs), bw = Math.max(Math.max(...xs) - mnx, 1e-6);
  const mny = Math.min(...ys), mxy = Math.max(...ys), bh = Math.max(mxy - mny, 1e-6);
  const top = p.filter((q) => q[1] < mny + 0.1 * bh).map((q) => q[0]);
  const bottom = p.filter((q) => q[1] > mxy - 0.1 * bh).map((q) => q[0]);
  if (!top.length || !bottom.length) return false;
  return Math.min(...top) - mnx < 0.18 * bw && Math.min(...bottom) - mnx < 0.18 * bw;
}

export class AirWriter {
  constructor() {
    this.templates = [];
    fetch('letter_templates.json').then((r) => r.json())
      .then((t) => { this.templates = t.map((x) => ({ ch: x.char, pts: normalize(x.strokes) })); })
      .catch((e) => console.error('letter templates', e));
    this.strokes = [];
    this.current = null;
    this.penLabel = null;
    this.pen = {};      // label -> {down, n}: debounced pen state (2 frames) so jitter never breaks a stroke
    this.hover = {};    // label -> {pt, down} for the pen cursor
    this.lastPenUp = 0;
    this.popup = null;
  }

  penState(h) {
    const st = (this.pen[h.label] ||= { down: false, n: 0 });
    const raw = penPose(h.px, st.down);
    if (raw.down !== st.down) { if (++st.n >= 2) { st.down = raw.down; st.n = 0; } } else st.n = 0;
    return { down: st.down, pt: raw.pt };
  }

  letterBox() {
    const all = this.strokes.flat();
    const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]);
    return { mnx: Math.min(...xs), mxx: Math.max(...xs), mny: Math.min(...ys), mxy: Math.max(...ys) };
  }

  get writing() { return this.current !== null || this.strokes.length > 0; }

  recognize(strokes) {
    if (!this.templates.length) return '?';
    const p = normalize(strokes);
    let best = null;
    for (const t of this.templates) {
      const d = greedyMatch(p, t.pts);
      if (!best || d < best.d) best = { ch: t.ch, d };
    }
    let ch = best.ch;
    if ('OQCDoc'.includes(ch)) {
      const straight = straightLeft(strokes);
      if (straight && ch !== 'D') ch = 'D';
      else if (!straight && ch === 'D') ch = 'O';
    }
    // A single plain vertical line is "l" (a dotted line is "i"; "l" alone becomes "I" via auto-caps)
    if (ch === 'I' && strokes.length === 1) ch = 'l';
    return ch;
  }

  /** Returns the typed key ('a'-'z', ' ', '.', '<BKSP>') when a letter is finished, else null. */
  update(hands, W, H, now, blocked = null) {
    let pen = null;
    this.hover = {};
    for (const h of hands) {
      const st = this.penState(h);
      this.hover[h.label] = st;
      if (this.penLabel !== null && h.label !== this.penLabel) continue;
      if (st.down && !pen) pen = { label: h.label, pt: st.pt };
    }
    const seen = new Set(hands.map((h) => h.label));
    for (const l of Object.keys(this.pen)) if (!seen.has(l)) delete this.pen[l];

    if (pen) {
      let key = null;
      if (this.current === null) {
        if (blocked && blocked(pen.pt) && !this.writing) return null;
        // A stroke started far to the right (or left) of the letter being written begins the next letter
        if (this.strokes.length) {
          const b = this.letterBox();
          const size = Math.max(b.mxy - b.mny, b.mxx - b.mnx, 0.04 * H);
          if (pen.pt[0] > b.mxx + 0.8 * size || pen.pt[0] < b.mnx - 0.8 * size) key = this.finish(W, H, now);
        }
        this.current = [pen.pt];
        this.penLabel = pen.label;
      } else if (dist(pen.pt, this.current[this.current.length - 1]) > 1.5) {
        this.current.push(pen.pt);
      }
      return key;
    }
    if (this.current !== null) { this.strokes.push(this.current); this.current = null; this.lastPenUp = now; }
    if (this.strokes.length && now - this.lastPenUp > 0.55) {
      const k = this.finish(W, H, now);
      this.penLabel = null;
      return k;
    }
    if (!this.strokes.length) this.penLabel = null;
    return null;
  }

  finish(W, H, now) {
    const strokes = this.strokes;
    this.strokes = [];
    const all = strokes.flat();
    const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]);
    const bw = Math.max(...xs) - Math.min(...xs), bh = Math.max(...ys) - Math.min(...ys);
    const diag = Math.hypot(W, H);
    let key;
    if (Math.max(bw, bh) < 0.03 * diag) key = '.';
    else if (strokes.length === 1 && bw > 2.5 * Math.max(bh, 1) && bw > 0.08 * diag) {
      const s = strokes[0];
      key = s[s.length - 1][0] > s[0][0] ? ' ' : '<BKSP>';
    } else key = this.recognize(strokes).toLowerCase();
    const label = key === ' ' ? 'SPACE' : key === '<BKSP>' ? 'DEL' : key.toUpperCase();
    this.popup = { label, x: (Math.min(...xs) + Math.max(...xs)) / 2, y: Math.min(...ys), t: now };
    return key;
  }

  clear() { this.strokes = []; this.current = null; this.penLabel = null; this.pen = {}; this.hover = {}; }

  draw(ctx, now, u) {
    const lw = Math.max(3, 5 * u);
    // Pen cursor: hollow ring while the fingers are apart, filled pink while writing
    const r = Math.max(6, 9 * u);
    for (const { pt, down } of Object.values(this.hover)) {
      ctx.beginPath(); ctx.arc(pt[0], pt[1], r, 0, Math.PI * 2);
      if (down) { ctx.fillStyle = '#ff69b4'; ctx.fill(); }
      ctx.beginPath(); ctx.arc(pt[0], pt[1], r + 3, 0, Math.PI * 2);
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();
    }
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    for (const s of [...this.strokes, ...(this.current ? [this.current] : [])]) {
      ctx.beginPath();
      s.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
      if (s.length === 1) ctx.lineTo(s[0][0] + 0.1, s[0][1]);
      ctx.strokeStyle = 'rgba(230,60,150,0.85)'; ctx.lineWidth = lw + 5; ctx.stroke();
      ctx.strokeStyle = '#ffd2eb'; ctx.lineWidth = lw; ctx.stroke();
    }
    if (this.popup && now - this.popup.t < 0.8) {
      const a = 1 - (now - this.popup.t) / 0.8;
      const fs = Math.max(20, (this.popup.label.length === 1 ? 56 : 30) * u);
      ctx.font = `800 ${fs}px system-ui, sans-serif`;
      const tw = ctx.measureText(this.popup.label).width;
      const x = this.popup.x - tw / 2, y = this.popup.y - 16 - (1 - a) * 30;
      ctx.globalAlpha = a;
      ctx.lineWidth = 6; ctx.strokeStyle = '#5a1440'; ctx.strokeText(this.popup.label, x, y);
      ctx.fillStyle = '#ffb4dc'; ctx.fillText(this.popup.label, x, y);
      ctx.globalAlpha = 1;
    }
  }
}
