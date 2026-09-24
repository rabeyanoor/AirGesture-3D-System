// Glass UI: FPS pill, lined notepad, right sidebar with 5 tiles (port of ar_ui_renderer.py).

export const TILE_ORDER = ['bulb', 'notepad', 'liquid', 'rubik', 'power'];

function rrect(ctx, x, y, w, h, r) {
  r = Math.max(0, Math.min(r, w / 2, h / 2));
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

export class UI {
  constructor() {
    this.dark = false;
    this.showNotepad = false;
    this.showLiquid = false;
    this.showRubik = false;
    this.anim = 0;
    this.tileRects = {};
    this.handle = null;
    this.fontReady = false;
    document.fonts?.load('24px "Architects Daughter"').then(() => { this.fontReady = true; }).catch(() => {});
  }

  layout(W, H) {
    const u = Math.min(W, H) / 720;
    const portrait = H > W;
    const margin = Math.max(8, 12 * u);
    const tile = Math.max(46, 64 * u);
    const sidebarW = Math.max(tile + 20, Math.min(0.105 * W, 140), portrait ? 0 : 0.105 * W);
    const gap = Math.max(58, 86 * u);
    const top = Math.max(14, 0.055 * H);
    const first = top + Math.max(26, 50 * u);
    const bottom = Math.min(H - Math.max(14, 0.045 * H), first + 4 * gap + tile + Math.max(24, 40 * u));
    return { u, portrait, margin, tile, sidebarW, gap, top, first, bottom };
  }

  fullyOpen() { return this.anim > 0.95; }

  tileAt(x, y) {
    for (const [name, [tx, ty, ts]] of Object.entries(this.tileRects)) {
      if (x >= tx && x <= tx + ts && y >= ty && y <= ty + ts) return name;
    }
    return null;
  }

  handleAt(x, y) {
    if (!this.handle) return false;
    const [hx, hy, hw, hh] = this.handle;
    return x >= hx - 14 && x <= hx + hw + 14 && y >= hy - 14 && y <= hy + hh + 14;
  }

  drawFps(ctx, fps, W, H) {
    const { u } = this.layout(W, H);
    const text = `${Math.round(fps)} FPS`;
    const fs = Math.max(10, 11 * u);
    ctx.font = `${fs}px system-ui, sans-serif`;
    const tw = ctx.measureText(text).width;
    const x = Math.max(10, 14 * u), y = Math.max(10, 10 * u) + (window.visualViewport ? 0 : 0);
    const h = fs + 12;
    ctx.fillStyle = 'rgba(95, 95, 95, 0.7)';
    rrect(ctx, x, y, tw + 24, h, h / 2);
    ctx.fill();
    ctx.fillStyle = '#ebebeb';
    ctx.fillText(text, x + 12, y + h / 2 + fs * 0.35);
  }

  drawStatus(ctx, text, W, H) {
    if (!text || text === 'IDLE') return;
    const { u } = this.layout(W, H);
    const fs = Math.max(11, 13 * u);
    ctx.font = `600 ${fs}px system-ui, sans-serif`;
    const tw = ctx.measureText(text).width;
    const x = Math.max(10, 14 * u), y = H - Math.max(56, 70 * u);
    ctx.fillStyle = 'rgba(0,0,0,0.45)';
    rrect(ctx, x, y - fs - 6, tw + 20, fs + 14, 8);
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.fillText(text, x + 10, y + 2);
  }

  notepadRect(W, H) {
    if (this.panel) {
      const p = this.panel;
      return [p.x + 12, p.y + 10, p.x + p.w - 12, p.y + p.h - (p.portrait ? 64 : 64)];
    }
    const L = this.layout(W, H);
    if (L.portrait) {
      const x1 = 0.04 * W, y1 = Math.max(52, 0.07 * H);
      return [x1, y1, W - L.sidebarW - 2 * L.margin - 8, y1 + 0.34 * H];
    }
    return [0.037 * W, 0.085 * H, 0.505 * W, 0.65 * H];
  }

  /** Bottom (or side) panel: the notepad always lives here; dimmed with a hint until writing is turned on. */
  drawPanel(ctx, text) {
    const p = this.panel;
    if (!p) return;
    ctx.fillStyle = '#16171b';
    ctx.fillRect(p.x, p.y, p.w, p.h);
    ctx.fillStyle = 'rgba(255,255,255,0.08)';
    ctx.fillRect(p.x, p.y, p.portrait ? p.w : 1, p.portrait ? 1 : p.h);
    const was = this.showNotepad;
    this.showNotepad = true;
    ctx.save();
    if (!was) ctx.globalAlpha = 0.45;
    this.drawNotepad(ctx, was ? text : text.replace(/\|$/, ''), p.w, p.h);
    ctx.restore();
    this.showNotepad = was;
    if (!was) {
      const [x1, y1, x2, y2] = this.notepadRect();
      ctx.font = '600 14px system-ui, sans-serif';
      ctx.fillStyle = '#fff';
      const msg = 'Tap the notepad tile in the sidebar to start writing';
      const tw = ctx.measureText(msg).width;
      ctx.fillText(msg, (x1 + x2 - tw) / 2, (y1 + y2) / 2);
    }
  }

  drawNotepad(ctx, text, W, H) {
    if (!this.showNotepad) return;
    const L = this.layout(W, H);
    const [x1, y1, x2, y2] = this.notepadRect(W, H);
    const u = L.u;
    ctx.fillStyle = 'rgba(226, 228, 225, 0.42)';
    rrect(ctx, x1, y1, x2 - x1, y2 - y1, 16 * u);
    ctx.fill();

    const padX = (x2 - x1) * 0.05;
    const spacing = Math.max(24, 34.5 * u);
    const firstRule = y1 + Math.max(46, 75 * u);
    const rules = [];
    for (let ry = firstRule; ry < y2 - spacing * 0.6; ry += spacing) rules.push(ry);
    ctx.strokeStyle = 'rgba(250,250,250,0.95)';
    ctx.lineWidth = Math.max(1, 1.4 * u);
    for (const ry of rules) {
      ctx.beginPath();
      ctx.moveTo(x1 + padX, ry);
      ctx.lineTo(x2 - padX, ry);
      ctx.stroke();
    }

    const hand = this.fontReady ? '"Architects Daughter", cursive' : 'cursive';
    ctx.fillStyle = '#fff';
    ctx.font = `${Math.max(12, 16 * u)}px ${hand}`;
    ctx.fillText('NOTEPAD', x1 + padX, y1 + Math.max(24, 36 * u));

    ctx.fillStyle = '#0f0f0f';
    ctx.font = `${Math.max(15, 26 * u)}px ${hand}`;
    const textX = x1 + padX + (x2 - x1) * 0.06;
    const maxW = x2 - padX - textX;
    const lines = [];
    for (const para of text.split('\n')) {
      let cur = '';
      for (const word of para.split(' ')) {
        const cand = cur ? `${cur} ${word}` : word;
        if (ctx.measureText(cand).width <= maxW || !cur) cur = cand;
        else { lines.push(cur); cur = word; }
      }
      lines.push(cur);
    }
    const shown = lines.slice(-rules.length);
    shown.forEach((line, i) => ctx.fillText(line, textX, rules[i] - spacing * 0.18));
  }

  drawSidebar(ctx, open, W, H) {
    const L = this.layout(W, H);
    this.anim += ((open ? 1 : 0) - this.anim) * 0.3;

    // Small handle on the right edge (touch target to open / close the sidebar)
    const hh = Math.max(64, 90 * L.u), hw = 6;
    this.handle = [W - hw - 4, H / 2 - hh / 2, hw, hh];
    if (this.anim < 0.5) {
      ctx.fillStyle = 'rgba(255,255,255,0.55)';
      rrect(ctx, ...this.handle, 3);
      ctx.fill();
    }

    if (this.anim < 0.02) { this.tileRects = {}; return; }
    const openX = W - L.margin - L.sidebarW;
    const x = W + (openX - W) * this.anim;
    ctx.fillStyle = 'rgba(216, 222, 218, 0.42)';
    rrect(ctx, x, L.top, L.sidebarW, L.bottom - L.top, 14 * L.u);
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = Math.max(1.5, 2 * L.u);
    ctx.beginPath();
    ctx.moveTo(x + L.sidebarW * 0.12, L.top + 5 * L.u);
    ctx.lineTo(x + L.sidebarW * 0.88, L.top + 5 * L.u);
    ctx.stroke();

    const active = { bulb: this.dark, notepad: this.showNotepad, liquid: this.showLiquid, rubik: this.showRubik };
    const tx = x + (L.sidebarW - L.tile) / 2;
    this.tileRects = {};
    TILE_ORDER.forEach((name, i) => {
      const ty = L.first + i * L.gap;
      this.tileRects[name] = [tx, ty, L.tile];
      ctx.fillStyle = active[name] ? 'rgba(170, 195, 205, 0.75)' : 'rgba(238, 240, 238, 0.55)';
      ctx.fillRect(tx, ty, L.tile, L.tile);
      ctx.strokeStyle = 'rgba(250,250,250,0.95)';
      ctx.lineWidth = 1;
      ctx.strokeRect(tx + 0.5, ty + 0.5, L.tile - 1, L.tile - 1);
      this.drawIcon(ctx, name, tx + L.tile / 2, ty + L.tile / 2, L.tile / 64);
    });
  }

  /** Hand keyboard: a letter tag on every finger joint; hovered joint grows pink, typed joint flashes. */
  drawHandKeys(ctx, maps, flash) {
    for (const { hand, map, hover } of maps) {
      const px = hand.px;
      const s = Math.hypot(px[9][0] - px[0][0], px[9][1] - px[0][1]);
      const r = Math.max(9, Math.min(22, s * 0.13));
      for (const [id, ch] of [...map, [9, '<DEL>']]) {
        const label = ch === ' ' ? 'SPC' : ch === '<DEL>' ? 'DEL' : ch.toUpperCase();
        const isHover = hover && hover.id === id;
        const isFlash = flash && flash.label === hand.label && flash.id === id;
        const rr = r * (isHover ? 1.35 : 1) + (label.length > 1 ? 4 : 0);
        const [x, y] = px[id];
        ctx.beginPath(); ctx.arc(x, y, rr, 0, Math.PI * 2);
        ctx.fillStyle = isHover || isFlash ? '#ff69b4' : 'rgba(245,245,245,0.95)';
        ctx.fill();
        ctx.strokeStyle = '#28282d'; ctx.lineWidth = 1; ctx.stroke();
        const fs = rr * (label.length > 1 ? 0.62 : 1.0);
        ctx.font = `700 ${fs}px system-ui, sans-serif`;
        ctx.fillStyle = isHover || isFlash ? '#fff' : '#23232a';
        const tw = ctx.measureText(label).width;
        ctx.fillText(label, x - tw / 2, y + fs * 0.36);
      }
      // Thumb tip = the "pen"
      ctx.beginPath(); ctx.arc(px[4][0], px[4][1], Math.max(5, r / 2), 0, Math.PI * 2);
      ctx.fillStyle = '#ffe650'; ctx.fill();
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();
    }
  }

  drawIcon(ctx, name, cx, cy, s) {
    ctx.save();
    ctx.translate(cx, cy);
    ctx.scale(s, s);
    ctx.strokeStyle = '#3c3c3c';
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';
    const line = (a, b, c, d) => { ctx.beginPath(); ctx.moveTo(a, b); ctx.lineTo(c, d); ctx.stroke(); };
    if (name === 'bulb') {
      ctx.beginPath(); ctx.arc(0, -5, 12, 0, Math.PI * 2); ctx.stroke();
      line(-6, 10, 6, 10); line(-4, 15, 4, 15);
    } else if (name === 'notepad') {
      ctx.strokeRect(-14, -18, 28, 36);
      for (const dy of [-8, 0, 8]) line(-9, dy, 9, dy);
    } else if (name === 'liquid') {
      ctx.fillStyle = '#ff69b4';
      ctx.fillRect(-15, 3, 24, 12);
      ctx.strokeRect(-15, -9, 24, 24);
      ctx.strokeRect(-8, -16, 24, 24);
      line(-15, -9, -8, -16); line(9, -9, 16, -16); line(9, 15, 16, 8); line(-15, 15, -8, 8);
    } else if (name === 'rubik') {
      const cols = ['#fff', '#d21e28', '#1eaf3c', '#ffd700', '#ff8200', '#0a5ac8', '#d21e28', '#fff', '#ffd700'];
      ctx.fillStyle = '#1e1e1e';
      ctx.fillRect(-16, -16, 32, 32);
      cols.forEach((c, k) => {
        ctx.fillStyle = c;
        ctx.fillRect(-14 + (k % 3) * 10, -14 + Math.floor(k / 3) * 10, 8, 8);
      });
    } else {
      ctx.beginPath(); ctx.arc(0, 2, 15, -Math.PI / 2 + 0.6, -Math.PI / 2 - 0.6 + Math.PI * 2); ctx.stroke();
      line(0, -16, 0, 1);
    }
    ctx.restore();
  }
}
