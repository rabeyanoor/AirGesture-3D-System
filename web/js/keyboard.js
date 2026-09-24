// On-screen glass QWERTY keyboard for the notepad (port of air_keyboard.py).
// Point the index fingertip at a key and pinch thumb + index to press it (hold DEL to keep deleting).
// On a phone the keys can simply be tapped.
import { dist, handScale } from './gestures.js';

const ROWS = [
  [['q', 1], ['w', 1], ['e', 1], ['r', 1], ['t', 1], ['y', 1], ['u', 1], ['i', 1], ['o', 1], ['p', 1]],
  [['a', 1], ['s', 1], ['d', 1], ['f', 1], ['g', 1], ['h', 1], ['j', 1], ['k', 1], ['l', 1], ["'", 1]],
  [['z', 1], ['x', 1], ['c', 1], ['v', 1], ['b', 1], ['n', 1], ['m', 1], [',', 1], ['.', 1], ['?', 1]],
  [['<CLEAR>', 1.5], [' ', 4.5], ['<BKSP>', 2.5], ['<ENTER>', 1.5]],
];
export const KEY_LABELS = { '<CLEAR>': 'CLEAR', ' ': 'SPACE', '<BKSP>': '⌫ DEL', '<ENTER>': 'ENTER' };
const REPEATABLE = new Set(['<BKSP>']);

export class AirKeyboard {
  constructor() {
    this.keys = [];
    this.rect = null;
    this.pinching = {};
    this.held = {};
    this.hover = {};
    this.prevKey = {};
    this.flash = {};
  }

  layout(W, H, notepadRect) {
    let x1, y1, x2, y2;
    if (H > W) {
      // Portrait phone: full-width keyboard right under the notepad
      x1 = 0.03 * W; x2 = 0.97 * W;
      y1 = notepadRect[3] + 10; y2 = Math.min(H - 70, y1 + 0.28 * H);
    } else {
      x1 = 0.037 * W; x2 = 0.80 * W; y1 = 0.675 * H; y2 = 0.975 * H - 50;
    }
    this.rect = [x1, y1, x2, y2];
    const gap = Math.max(3, W / 320);
    const rowH = (y2 - y1 - gap * (ROWS.length + 1)) / ROWS.length;
    this.keys = [];
    ROWS.forEach((row, r) => {
      const units = row.reduce((a, [, u]) => a + u, 0);
      const unitW = (x2 - x1 - gap * (row.length + 1)) / units;
      let x = x1 + gap;
      const y = y1 + gap + r * (rowH + gap);
      for (const [key, u] of row) {
        this.keys.push({ key, r: [x, y, x + unitW * u, y + rowH] });
        x += unitW * u + gap;
      }
    });
  }

  contains(pt, m = 0) {
    if (!this.rect) return false;
    const [x1, y1, x2, y2] = this.rect;
    return pt[0] >= x1 - m && pt[0] <= x2 + m && pt[1] >= y1 - m && pt[1] <= y2 + m;
  }

  keyAt(pt) {
    const k = this.keys.find(({ r }) => pt[0] >= r[0] && pt[0] <= r[2] && pt[1] >= r[1] && pt[1] <= r[3]);
    return k ? k.key : null;
  }

  /** Index fingertip: stays put while the thumb closes in for the pinch. */
  static cursor(hand) { return [hand.px[8][0], hand.px[8][1]]; }

  owns(hand) { return this.contains(AirKeyboard.cursor(hand), 10); }

  /** Returns the keys pressed this frame by hands. */
  update(hands, now) {
    const pressed = [];
    const seen = new Set();
    this.hover = {};
    for (const h of hands) {
      const label = h.label;
      seen.add(label);
      const was = this.pinching[label] || false;
      const pinch = dist(h.px[4], h.px[8]) < handScale(h.px) * (was ? 0.6 : 0.4);
      this.pinching[label] = pinch;
      const pt = AirKeyboard.cursor(h);
      const key = this.keyAt(pt);
      this.hover[label] = { pt, key, pinch };
      if (pinch && !was) {
        const target = this.prevKey[label] || key; // key aimed at just before the fingers closed
        if (target !== null && target !== undefined) {
          pressed.push(target);
          this.flash[target] = now;
          this.held[label] = { key: target, t0: now, last: now };
        }
      } else if (pinch && this.held[label]) {
        const hk = this.held[label];
        if (REPEATABLE.has(hk.key) && now - hk.t0 > 0.5 && now - hk.last > 0.1) {
          pressed.push(hk.key);
          this.flash[hk.key] = now;
          hk.last = now;
        }
      } else if (!pinch) {
        delete this.held[label];
      }
      if (!pinch) this.prevKey[label] = key;
    }
    for (const label of Object.keys(this.pinching)) {
      if (!seen.has(label)) { delete this.pinching[label]; delete this.held[label]; delete this.prevKey[label]; }
    }
    return pressed;
  }

  /** Touch press (phone): returns the key under the finger, if any. */
  tap(pt, now) {
    const key = this.keyAt(pt);
    if (key !== null) this.flash[key] = now;
    return key;
  }

  draw(ctx, now) {
    if (!this.rect) return;
    const [x1, y1, x2, y2] = this.rect;
    const hovered = new Set(Object.values(this.hover).map((h) => h.key).filter(Boolean));
    ctx.fillStyle = 'rgba(226, 228, 225, 0.35)';
    ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
    const rowH = this.keys[0].r[3] - this.keys[0].r[1];
    for (const { key, r } of this.keys) {
      const flash = now - (this.flash[key] ?? -9) < 0.18;
      ctx.fillStyle = flash ? 'rgba(255,105,180,0.85)' : hovered.has(key) ? 'rgba(255,255,255,0.8)' : 'rgba(240,242,240,0.5)';
      ctx.fillRect(r[0], r[1], r[2] - r[0], r[3] - r[1]);
      if (hovered.has(key)) {
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 2;
        ctx.strokeRect(r[0] + 1, r[1] + 1, r[2] - r[0] - 2, r[3] - r[1] - 2);
      }
      const label = KEY_LABELS[key] || key.toUpperCase();
      const fs = Math.max(11, rowH * (label.length > 1 ? 0.3 : 0.45));
      ctx.font = `600 ${fs}px system-ui, sans-serif`;
      ctx.fillStyle = '#28282d';
      const tw = ctx.measureText(label).width;
      ctx.fillText(label, (r[0] + r[2] - tw) / 2, (r[1] + r[3]) / 2 + fs * 0.35);
    }
    const rad = Math.max(6, Math.min(ctx.canvas.width, ctx.canvas.height) / 180);
    for (const { pt, pinch } of Object.values(this.hover)) {
      if (!this.contains(pt, 40)) continue;
      ctx.beginPath(); ctx.arc(pt[0], pt[1], rad, 0, Math.PI * 2);
      if (pinch) { ctx.fillStyle = '#ffe650'; ctx.fill(); }
      ctx.beginPath(); ctx.arc(pt[0], pt[1], rad + 3, 0, Math.PI * 2);
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();
    }
  }
}
