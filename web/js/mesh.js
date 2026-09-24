// Minimal hand overlays (port of ar_mesh_3d.py): cyan dots + coordinates, no lines or fill.
import { extendedTipIds } from './gestures.js';

const TIPS = [4, 8, 12, 16, 20];
const PIP_OF = { 4: 2, 8: 6, 12: 10, 16: 14, 20: 18 };
function point(ctx, p, u, dx = -22, dy = -10) {
  ctx.fillStyle = '#14c8eb';
  ctx.beginPath();
  ctx.arc(p[0], p[1], Math.max(3, 4 * u), 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = '#fff';
  ctx.font = `${Math.max(9, Math.round(11 * u))}px system-ui, sans-serif`;
  ctx.fillText(`${Math.round(p[0])}, ${Math.round(p[1])}`, p[0] + dx * u, p[1] + dy * u);
}

export function drawHandMesh(ctx, hands, u) {
  if (hands.length >= 2) {
    const [a, b] = hands[0].px[0][0] <= hands[1].px[0][0] ? [hands[0], hands[1]] : [hands[1], hands[0]];
    const ea = extendedTipIds(a.px), eb = extendedTipIds(b.px);
    let shared = TIPS.filter((t) => ea.includes(t) && eb.includes(t));
    if (shared.length < 2) shared = TIPS;
    const c1 = shared.map((t) => a.px[t]);
    const c2 = shared.map((t) => b.px[t]);
    for (const p of c1.concat(c2)) point(ctx, p, u);
    return;
  }
  if (hands.length === 1) {
    const px = hands[0].px;
    const ext = extendedTipIds(px);
    if (!ext.length) return;
    if (ext.length === 1) {
      point(ctx, px[ext[0]], u);
      point(ctx, px[PIP_OF[ext[0]]], u, 8, 4);
      return;
    }
    const pts = ext.map((t) => px[t]);
    for (const p of pts) point(ctx, p, u);
  }
}
