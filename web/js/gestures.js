// Hand geometry helpers + gesture recognizer (port of gesture_recognizer.py).
// Hands: { label: 'Left' | 'Right', px: [[x, y, z] x 21] } in canvas CSS pixels.

export const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
export const handScale = (px) => dist(px[0], px[9]);

export function extended(px, tip, pip, mcp, ratio = 1.3) {
  const dp = dist(px[pip], px[mcp]);
  return dp > 0 && dist(px[tip], px[mcp]) / dp > ratio;
}

const FINGERS = [[8, 6, 5], [12, 10, 9], [16, 14, 13], [20, 18, 17]];

/** Fingertip ids that are genuinely extended (same rules as ar_mesh_3d.py). */
export function extendedTipIds(px) {
  const p0 = px[0];
  const ids = [];
  for (const [tip, pip, mcp] of FINGERS) {
    const dPip = dist(px[pip], px[mcp]);
    if (dPip === 0) continue;
    if (dist(px[tip], px[mcp]) / dPip > 1.15 && dist(px[tip], p0) > dist(px[pip], p0) - 8) ids.push(tip);
  }
  const dIp = dist(px[3], px[2]);
  if (dIp > 0 && dist(px[4], px[2]) / dIp > 1.15 && dist(px[4], px[5]) > dist(px[2], px[5]) * 1.05 &&
      dist(px[4], p0) > dist(px[3], p0) - 5) ids.push(4);
  return ids.sort((a, b) => a - b);
}

/** At least 3 of the 4 fingers folded. */
export function isFist(px) {
  let folded = 0;
  for (const [tip, pip, mcp] of FINGERS) {
    if (!extended(px, tip, pip, mcp) && dist(px[tip], px[0]) < dist(px[pip], px[0]) * 1.15) folded++;
  }
  return folded >= 3;
}

export const LEFT_MAP = [[8, 'a'], [7, 'b'], [6, 'c'], [12, 'd'], [11, 'e'], [10, 'f'], [16, 'g'], [15, 'h'],
  [14, 'i'], [20, 'j'], [19, 'k'], [18, 'l'], [5, ' '], [17, ',']];
export const RIGHT_MAP = [[8, 'm'], [7, 'n'], [6, 'o'], [12, 'p'], [11, 'q'], [10, 'r'], [16, 's'], [15, 't'],
  [14, 'u'], [20, 'v'], [19, 'w'], [18, 'x'], [5, ' '], [17, '.']];

export class GestureRecognizer {
  constructor() {
    this.palmTouchTriggered = false;
    this.sidebarTriggered = false;
    this.lastSidebarToggle = 0;
    this.prevX = {};
    this.hoveredTile = null;
    this.lastClick = 0;
    this.sidebarReadyT = null;
    this.phLatched = false;
    this.phLastContact = 0;
    this.lastPhalanx = 0;
  }

  /** Right index tip touching the left palm centre -> space. */
  checkPalmTouch(left, right) {
    if (!left || !right) return null;
    const palm = left.px[9];
    return dist(right.px[8], palm) < Math.max(20, handScale(left.px) * 0.3) ? palm : null;
  }

  /** Hand entering the right 20% of the screen or swiping right opens the sidebar. */
  updateSidebar(hands, W, open, now) {
    let trig = false;
    for (const h of hands) {
      const ix = h.px[8][0], wx = h.px[0][0];
      if (ix > W * 0.8 || wx > W * 0.8) { trig = true; break; }
      const prev = this.prevX[h.label];
      if (prev && now - prev.t > 0.01 && now - prev.t < 0.3 && ix - prev.x > 35 * W / 1280 && ix > W * 0.6) {
        trig = true; break;
      }
      this.prevX[h.label] = { x: ix, t: now };
    }
    if (trig) {
      if (!this.sidebarTriggered && now - this.lastSidebarToggle > 0.5) {
        open = true;
        this.lastSidebarToggle = now;
        this.sidebarTriggered = true;
      }
    } else {
      this.sidebarTriggered = false;
    }
    return open;
  }

  /** Index fingertip entering a tile fires it once. Returns the tile name or null. */
  checkSidebarClicks(hands, ui, open, now) {
    if (!open || !ui.fullyOpen()) { this.hoveredTile = null; this.sidebarReadyT = null; return null; }
    if (this.sidebarReadyT === null) this.sidebarReadyT = now;
    if (now - this.sidebarReadyT < 0.3) return null;
    let hovered = null;
    for (const h of hands) {
      hovered = ui.tileAt(h.px[8][0], h.px[8][1]);
      if (hovered) break;
    }
    const prev = this.hoveredTile;
    this.hoveredTile = hovered;
    if (!hovered || hovered === prev || now - this.lastClick < 0.6) return null;
    this.lastClick = now;
    if (hovered === 'power') this.hoveredTile = null;
    return hovered;
  }

  /** Joint the thumb tip is closest to (within reach x hand size): {id, ch} or null. DEL = palm centre (9). */
  phalanxHover(hand, map, reach = 0.6) {
    if (!hand) return null;
    const s = Math.max(handScale(hand.px), 1e-6);
    let best = null;
    for (const [id, ch] of [...map, [9, '<DEL>']]) {
      const d = dist(hand.px[4], hand.px[id]) / s;
      if (d < reach && (!best || d < best.d)) best = { id, ch, d };
    }
    return best;
  }

  phalanxContact(left, right) {
    for (const h of [left, right]) {
      if (h && dist(h.px[4], h.px[9]) < handScale(h.px) * 0.3) return '<DEL>';
    }
    if (left && right) {
      const t = Math.max(18, (handScale(left.px) + handScale(right.px)) * 0.5 * 0.25);
      if (dist(left.px[4], right.px[4]) < t) return 'y';
      if (dist(left.px[8], right.px[8]) < t) return 'z';
    }
    let best = null;
    for (const [h, map] of [[left, LEFT_MAP], [right, RIGHT_MAP]]) {
      if (!h) continue;
      const thresh = Math.max(14, handScale(h.px) * 0.3);
      for (const [id, ch] of map) {
        const d = dist(h.px[4], h.px[id]);
        if (d < thresh && (!best || d < best.d)) best = { d, ch };
      }
    }
    return best ? best.ch : null;
  }

  /**
   * Thumb-to-joint typing. The thumb must rest on a joint for `dwell` s (brushing past types nothing);
   * one key per touch; holding DEL (palm centre) keeps deleting.
   */
  detectPhalanx(left, right, now, dwell = 0.15) {
    const key = this.phalanxContact(left, right);
    if (key === null) {
      if (now - this.phLastContact > 0.12) { this.phLatched = false; this.phCand = null; }
      return null;
    }
    this.phLastContact = now;
    if (this.phLatched) {
      if (key === '<DEL>' && this.phFired === key && now - this.phFireT > 0.5 && now - this.lastPhalanx > 0.15) {
        this.lastPhalanx = now;
        return key;
      }
      return null;
    }
    if (!this.phCand || this.phCand.key !== key) { this.phCand = { key, t: now }; return null; }
    if (now - this.phCand.t < dwell || now - this.lastPhalanx < 0.2) return null;
    this.phLatched = true;
    this.phFired = key; this.phFireT = now;
    this.lastPhalanx = now;
    return key;
  }
}

/** Sentence-start, post-punctuation and standalone "i" capitalisation (auto_capitalizer.py). */
export function autoCapitalize(text) {
  if (!text) return text;
  text = text.replace(/\b[ilL](?=[\s.,!?])/g, 'I'); // standalone i (or a lone air-written "l") is the pronoun I
  text = text.replace(/^(\s*)([a-z])/, (m, s, c) => s + c.toUpperCase());
  text = text.replace(/([.!?]\s*)([a-z])/g, (m, s, c) => s + c.toUpperCase());
  return text;
}
