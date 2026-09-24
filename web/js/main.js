// AirGesture 3D - web / phone version. Hand tracking runs on-device with MediaPipe Tasks Vision.
import { FilesetResolver, HandLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';
import { GestureRecognizer, autoCapitalize, dist, LEFT_MAP, RIGHT_MAP } from './gestures.js';
import { drawHandMesh } from './mesh.js';
import { UI } from './ui.js';
import { LiquidBox, RubiksCube } from './boxes.js';
import { AirKeyboard, KEY_LABELS } from './keyboard.js';
import { AirWriter } from './airwrite.js';

const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

const canvas = document.getElementById('stage');
const ctx = canvas.getContext('2d');
const video = document.getElementById('video');
const startScreen = document.getElementById('start');
const startBtn = document.getElementById('startBtn');
const msg = document.getElementById('msg');
const toolbar = document.getElementById('toolbar');
const scrambleBtn = document.getElementById('scrambleBtn');

const ui = new UI();
const rec = new GestureRecognizer();
const liquid = new LiquidBox();
const rubik = new RubiksCube();
const keyboard = new AirKeyboard();
// Typing in the notepad: air writing by default (pinch and write letters in the air);
// ?handkeys = thumb-to-finger-joint keyboard, ?keyboard = on-screen keyboard
const qs = new URLSearchParams(location.search);
const typingMode = qs.has('keyboard') ? 'keyboard' : qs.has('handkeys') ? 'handkeys' : 'write';
const useScreenKeyboard = typingMode === 'keyboard';
const usePhalanx = typingMode === 'handkeys';
const writer = new AirWriter();
const showKeyLabels = new URLSearchParams(location.search).has('labels'); // letters on joints: hidden by default
let keyFlash = null;

const state = {
  text: '',
  sidebarOpen: false,
  facing: 'user',
  stream: null,
  landmarker: null,
  hands: [],
  lastVideoTime: -1,
  fps: 0,
  lastFrame: 0,
  status: 'IDLE',
  statusUntil: 0,
  W: 0,
  H: 0,
};

// ------------------------------------------------------------------ layout
// Full screen camera by default. With ?split the camera "stage" takes 2/3 of the screen (top in portrait,
// left in landscape) and the rest is a panel holding the notepad and the buttons. state.W / state.H are the
// stage size (it starts at 0,0), so all AR code and touch coordinates work in stage space.
const splitLayout = new URLSearchParams(location.search).has('split');
let laidOut = false;
function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 1.5); // 1.5x is sharp enough and much lighter than 3x
  const FW = window.innerWidth, FH = window.innerHeight;
  const portrait = FH >= FW;
  state.fullW = FW; state.fullH = FH;
  if (splitLayout) {
    state.W = portrait ? FW : Math.round(FW * 2 / 3);
    state.H = portrait ? Math.round(FH * 2 / 3) : FH;
    state.panel = portrait ? { x: 0, y: state.H, w: FW, h: FH - state.H, portrait }
      : { x: state.W, y: 0, w: FW - state.W, h: FH, portrait };
  } else {
    state.W = FW; state.H = FH; state.panel = null;
    if (!laidOut && portrait) {
      // Portrait phone, full screen: park the boxes below the floating notepad
      for (const b of [liquid, rubik]) { b.defaultCenter = [0.45, 0.68]; b.center = [0.45, 0.68]; }
    }
  }
  laidOut = true;
  ui.panel = state.panel;
  canvas.width = Math.round(FW * dpr);
  canvas.height = Math.round(FH * dpr);
  canvas.style.width = `${FW}px`;
  canvas.style.height = `${FH}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  // Buttons live in the panel
  toolbar.style.left = !state.panel || portrait ? '50%' : `${state.W + (FW - state.W) / 2}px`;
  toolbar.style.maxWidth = `${(state.panel ? state.panel.w : FW) - 16}px`;
}
window.addEventListener('resize', resize);
window.visualViewport?.addEventListener('resize', resize);
resize();

const activeBox = () => (ui.showLiquid ? liquid : ui.showRubik ? rubik : null);

function toggleBox(name) {
  if (name === 'liquid') { ui.showLiquid = !ui.showLiquid; ui.showRubik = false; }
  else { ui.showRubik = !ui.showRubik; ui.showLiquid = false; }
  scrambleBtn.hidden = !ui.showRubik;
}

function setStatus(s, hold = 0.6) {
  if (!s) return;
  state.status = s;
  state.statusUntil = performance.now() / 1000 + hold;
}

function runTile(name) {
  if (name === 'bulb') { ui.dark = !ui.dark; setStatus('THEME TOGGLED'); }
  else if (name === 'notepad') { ui.showNotepad = !ui.showNotepad; setStatus('NOTEPAD TOGGLED'); }
  else if (name === 'liquid' || name === 'rubik') { toggleBox(name); setStatus(name === 'liquid' ? 'LIQUID BOX' : "RUBIK'S CUBE"); }
  else if (name === 'power') {
    state.text = '';
    ui.showLiquid = ui.showRubik = false;
    scrambleBtn.hidden = true;
    state.sidebarOpen = false;
    setStatus('POWER OFF');
  }
}

function appendText(chars) { state.text = autoCapitalize(state.text + chars); }
function pressKey(key) {
  if (key === '<BKSP>') state.text = state.text.slice(0, -1);
  else if (key === '<CLEAR>') state.text = '';
  else if (key === '<ENTER>') state.text += '\n';
  else appendText(key);
  setStatus(`KEY '${(KEY_LABELS[key] || key.toUpperCase()).replace('\u232B ', '')}'`, 0.5);
}

// ------------------------------------------------------------------ camera + model
async function startCamera() {
  if (state.stream) state.stream.getTracks().forEach((t) => t.stop());
  state.stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    // 960x540 is plenty for hand tracking and much lighter than 720p on mid-range phones
    video: { facingMode: state.facing, width: { ideal: 960 }, height: { ideal: 540 } },
  });
  video.srcObject = state.stream;
  await video.play();
  state.lastVideoTime = -1;
}

// Hand tracking in a Web Worker so drawing / recording never waits for MediaPipe (big win on slower phones).
function startWorker() {
  return new Promise((resolve, reject) => {
    let w;
    try { w = new Worker('js/hand_worker.js'); } catch (e) { reject(e); return; }
    const timer = setTimeout(() => reject(new Error('worker timeout')), 30000);
    w.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'ready') {
        clearTimeout(timer);
        w.onmessage = (ev) => {
          const r = ev.data;
          if (r.type !== 'result') return;
          state.hands = toHands(r);
          state.trackMs = r.ms;
          state.trackCount = (state.trackCount || 0) + 1;
          state.workerBusy = false;
        };
        resolve({ worker: w, delegate: m.delegate });
      } else if (m.type === 'error') {
        clearTimeout(timer);
        reject(new Error(m.message));
      }
    };
    w.onerror = (e) => { clearTimeout(timer); reject(e); };
    // CPU (XNNPACK) inside the worker is the reliable choice: WebGL in workers is flaky on some phones, and
    // being off the main thread it never slows the picture down. ?delegate=GPU to experiment.
    w.postMessage({ type: 'init', delegate: new URLSearchParams(location.search).get('delegate') || 'CPU' });
  });
}

async function createLandmarker() {
  const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: 'VIDEO',
    numHands: 2,
    minHandDetectionConfidence: 0.6,
    minHandPresenceConfidence: 0.6,
    minTrackingConfidence: 0.5,
  });
  try {
    return await HandLandmarker.createFromOptions(fileset, opts('GPU'));
  } catch (e) {
    console.warn('GPU delegate unavailable, using CPU', e);
    return HandLandmarker.createFromOptions(fileset, opts('CPU'));
  }
}

async function start(facing) {
  state.facing = facing;
  msg.className = '';
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    msg.className = 'error';
    msg.textContent = 'The camera needs a secure (https) page. Open the https:// link from serve_https.py.';
    return;
  }
  startBtn.disabled = startBackBtn.disabled = true;
  try {
    msg.textContent = 'Starting camera...';
    await startCamera();
    msg.textContent = 'Loading hand tracking model...';
    try {
      const { worker, delegate } = await startWorker();
      state.worker = worker;
      setStatus(`HAND TRACKING READY (${delegate})`, 2);
    } catch (err) {
      console.warn('Worker tracking unavailable, using the main thread', err);
      state.landmarker = await createLandmarker();
    }
    startScreen.style.display = 'none';
    toolbar.style.display = 'flex';
    requestAnimationFrame(loop);
  } catch (e) {
    console.error(e);
    msg.className = 'error';
    msg.textContent = e.name === 'NotAllowedError'
      ? 'Camera permission was denied. Allow camera access for this site and try again.'
      : `Could not start: ${e.message || e}`;
    startBtn.disabled = false;
    startBackBtn.disabled = false;
  }
}
const startBackBtn = document.getElementById('startBackBtn');
startBtn.addEventListener('click', () => start('user'));
startBackBtn.addEventListener('click', () => start('environment'));

// URL options (handy for testing / sharing a preset): ?autostart&camera=back&box=liquid|rubik&notepad&sidebar
const params = new URLSearchParams(location.search);
if (params.get('box') === 'liquid' || params.get('box') === 'rubik') toggleBox(params.get('box'));
if (params.has('notepad')) ui.showNotepad = true;
if (params.has('sidebar')) state.sidebarOpen = true;
if (params.has('debug')) window.airgesture = { state, ui, liquid, rubik, keyboard, writer };
if (params.has('autostart')) start(params.get('camera') === 'back' ? 'environment' : 'user');

document.getElementById('flipBtn').addEventListener('click', async () => {
  state.facing = state.facing === 'user' ? 'environment' : 'user';
  try { await startCamera(); } catch (e) { state.facing = 'user'; await startCamera(); }
});
scrambleBtn.addEventListener('click', () => rubik.scramble());

// ------------------------------------------------------------------ recording (canvas -> video file)
const recBtn = document.getElementById('recBtn');
const rec_ = { recorder: null, chunks: [], mime: '', t0: 0 };

function pickMime() {
  const types = ['video/mp4;codecs=avc1', 'video/mp4', 'video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
  return types.find((t) => window.MediaRecorder?.isTypeSupported?.(t)) || '';
}

async function saveVideo(blob, ext) {
  const name = `airgesture-${new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)}.${ext}`;
  const file = new File([blob], name, { type: blob.type });
  // Phones: the share sheet offers "Save video" to the gallery; otherwise fall back to a download
  if (navigator.canShare?.({ files: [file] })) {
    try { await navigator.share({ files: [file], title: 'AirGesture 3D' }); return; } catch (e) { /* cancelled */ }
  }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 60000);
}

recBtn.addEventListener('click', () => {
  if (rec_.recorder) { rec_.recorder.stop(); return; }
  if (!window.MediaRecorder || !canvas.captureStream) { setStatus('RECORDING NOT SUPPORTED', 2); return; }
  rec_.mime = pickMime();
  const stream = canvas.captureStream(30);
  const r = new MediaRecorder(stream, rec_.mime ? { mimeType: rec_.mime, videoBitsPerSecond: 3_500_000 } : undefined);
  rec_.chunks = [];
  r.ondataavailable = (e) => { if (e.data.size) rec_.chunks.push(e.data); };
  r.onstop = () => {
    const type = r.mimeType || rec_.mime || 'video/webm';
    saveVideo(new Blob(rec_.chunks, { type }), type.includes('mp4') ? 'mp4' : 'webm');
    rec_.recorder = null;
    recBtn.classList.remove('recording');
    recBtn.innerHTML = '&#9679; Record';
  };
  r.start(500);
  rec_.recorder = r;
  rec_.t0 = performance.now();
  recBtn.classList.add('recording');
});

function drawRecIndicator() {
  if (!rec_.recorder) return;
  const s = Math.floor((performance.now() - rec_.t0) / 1000);
  const label = `\u25A0 Stop ${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
  if (recBtn.textContent !== label) recBtn.textContent = label;
}
document.getElementById('clearBtn').addEventListener('click', () => { state.text = ''; });

// ------------------------------------------------------------------ video -> canvas mapping
function coverRect() {
  const vw = video.videoWidth || 1280, vh = video.videoHeight || 720;
  const s = Math.max(state.W / vw, state.H / vh);
  const dw = vw * s, dh = vh * s;
  return { ox: (state.W - dw) / 2, oy: (state.H - dh) / 2, dw, dh };
}

function toHands(result) {
  const { ox, oy, dw, dh } = coverRect();
  const mirror = state.facing === 'user';
  const hands = [];
  (result.landmarks || []).forEach((lms, i) => {
    const raw = result.handedness?.[i]?.[0]?.categoryName || 'Right';
    // MediaPipe labels assume a mirrored (selfie) image; the raw video is not mirrored, so swap
    const label = raw === 'Left' ? 'Right' : 'Left';
    const px = lms.map((l) => {
      const x = ox + l.x * dw;
      return [mirror ? state.W - x : x, oy + l.y * dh, l.z * dw];
    });
    hands.push({ label, px });
  });
  // Two hands with the same label: relabel by screen position so both still work
  if (hands.length === 2 && hands[0].label === hands[1].label) {
    hands.sort((a, b) => a.px[0][0] - b.px[0][0]);
    hands[0].label = 'Left'; hands[1].label = 'Right';
  }
  return hands;
}

// ------------------------------------------------------------------ main loop
function loop() {
  requestAnimationFrame(loop);
  const nowMs = performance.now();
  const now = nowMs / 1000;
  const { W, H } = state; // camera stage size

  if (state.lastFrame) {
    const inst = 1000 / Math.max(nowMs - state.lastFrame, 1);
    state.fps = state.fps ? state.fps * 0.9 + inst * 0.1 : inst;
  }
  state.lastFrame = nowMs;

  if (video.readyState >= 2 && video.currentTime !== state.lastVideoTime) {
    if (state.worker && !state.workerBusy) {
      // Send the newest camera frame to the worker; frames arriving while it is busy are skipped
      state.lastVideoTime = video.currentTime;
      state.workerBusy = true;
      const ts = state.lastTs = Math.max((state.lastTs || 0) + 1, nowMs);
      createImageBitmap(video)
        .then((bitmap) => state.worker.postMessage({ type: 'frame', bitmap, ts }, [bitmap]))
        .catch(() => { state.workerBusy = false; });
    } else if (state.landmarker) {
      state.lastVideoTime = video.currentTime;
      state.hands = toHands(state.landmarker.detectForVideo(video, nowMs));
    }
  }
  const hands = state.hands;
  const left = hands.find((h) => h.label === 'Left');
  const right = hands.find((h) => h.label === 'Right');

  // Gestures
  if (ui.showNotepad && usePhalanx && left && right) {
    const touch = rec.checkPalmTouch(left, right);
    if (touch && !rec.palmTouchTriggered) { appendText(' '); rec.palmTouchTriggered = true; setStatus('SPACE'); }
    if (!touch) rec.palmTouchTriggered = false;
  }
  state.sidebarOpen = rec.updateSidebar(hands, W, state.sidebarOpen, now);
  const tile = rec.checkSidebarClicks(hands, ui, state.sidebarOpen, now);
  if (tile) runTile(tile);

  // Air Keyboard: point at a key and pinch to press it; hands on the keyboard don't grab boxes
  const keyboardOn = ui.showNotepad && useScreenKeyboard;
  const writingOn = ui.showNotepad && typingMode === 'write';
  let boxHands = hands;
  // The hand holding the air-writing pen is writing, not grabbing a box
  if (writingOn && writer.writing) boxHands = hands.filter((h) => h.label !== writer.penLabel);
  if (keyboardOn) {
    keyboard.layout(W, H, ui.notepadRect(W, H));
    for (const k of keyboard.update(hands, now)) pressKey(k);
    boxHands = hands.filter((h) => !keyboard.owns(h));
  }

  const box = activeBox();
  let busy = false;
  if (box) {
    setStatus(box.update(boxHands, W, H, now), 0.3);
    busy = box.grabbed || touch.onBox;
  }
  if (writingOn && !touch.onBox) {
    const L = ui.layout(W, H);
    const sidebarX = state.sidebarOpen ? W - L.sidebarW - 2 * L.margin : W + 1;
    const grabbing = box && box.grabbed ? box.modeLabel : null;
    const k = writer.update(hands.filter((h) => h.label !== grabbing), W, H, now,
      (pt) => pt[0] > sidebarX || (box && box.contains(pt)));
    if (k === '<BKSP>') { state.text = state.text.slice(0, -1); setStatus('DEL'); }
    else if (k) { appendText(k); setStatus(`WROTE '${k === ' ' ? 'SPACE' : k.toUpperCase()}'`); }
    else if (writer.writing) setStatus('WRITING...', 0.2);
  }

  const handKeys = ui.showNotepad && usePhalanx;
  const hoverL = handKeys ? rec.phalanxHover(left, LEFT_MAP) : null;
  const hoverR = handKeys ? rec.phalanxHover(right, RIGHT_MAP) : null;
  if (!busy && handKeys) {
    const k = rec.detectPhalanx(left, right, now);
    if (k) {
      for (const [label, hv] of [['Left', hoverL], ['Right', hoverR]]) {
        if (hv && hv.ch === k) keyFlash = { label, id: hv.id, t: now };
      }
    }
    if (k === '<DEL>') { state.text = state.text.slice(0, -1); setStatus('DEL'); }
    else if (k) { appendText(k); setStatus(`KEY '${k === ' ' ? 'SPACE' : k.toUpperCase()}'`); }
  }

  // Draw: AR content is clipped to the camera stage; the notepad lives in the panel
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, state.fullW, state.fullH);
  ctx.save();
  ctx.beginPath();
  ctx.rect(0, 0, W, H);
  ctx.clip();
  if (video.readyState >= 2) {
    const { ox, oy, dw, dh } = coverRect();
    ctx.save();
    if (state.facing === 'user') { ctx.translate(W, 0); ctx.scale(-1, 1); }
    ctx.drawImage(video, ox, oy, dw, dh);
    ctx.restore();
  }
  if (ui.dark) { ctx.fillStyle = 'rgba(0,0,0,0.45)'; ctx.fillRect(0, 0, W, H); }
  if (box) box.draw(ctx, now);
  if (handKeys && showKeyLabels) {
    const maps = [];
    if (left) maps.push({ hand: left, map: LEFT_MAP, hover: hoverL });
    if (right) maps.push({ hand: right, map: RIGHT_MAP, hover: hoverR });
    ui.drawHandKeys(ctx, maps, keyFlash && now - keyFlash.t < 0.3 ? keyFlash : null);
  } else {
    drawHandMesh(ctx, hands, Math.min(W, H) / 720);
  }
  if (writingOn) writer.draw(ctx, now, Math.min(W, H) / 720);
  const shownText = state.text + ((keyboardOn || handKeys || writingOn) && Math.floor(now * 2) % 2 === 0 ? '|' : '');
  if (!state.panel) ui.drawNotepad(ctx, shownText, W, H); // full screen: notepad floats over the camera
  if (keyboardOn) keyboard.draw(ctx, now);
  ui.drawSidebar(ctx, state.sidebarOpen, W, H);
  ui.drawFps(ctx, state.fps, W, H);
  ui.drawStatus(ctx, now < state.statusUntil ? state.status : '', W, H);
  ctx.restore();
  if (state.panel) ui.drawPanel(ctx, shownText);
  drawRecIndicator();
}

// ------------------------------------------------------------------ touch control
// Tap: sidebar handle / tiles. On a box: 1 finger drag = rotate (Rubik's: swipe on the cube = layer turn,
// drag from outside = rotate), 2 fingers = resize / move / roll, double-tap = burst.
const touch = { pts: new Map(), onBox: false, mode: null, ref: null, lastTap: { t: 0, x: 0, y: 0 } };

canvas.addEventListener('pointerdown', (e) => {
  try { canvas.setPointerCapture(e.pointerId); } catch (err) { /* synthetic or already-released pointer */ }
  touch.pts.set(e.pointerId, { x: e.clientX, y: e.clientY, sx: e.clientX, sy: e.clientY, t: performance.now() });
  if (e.clientX > state.W || e.clientY > state.H) { touch.mode = 'panel'; return; } // panel area
  if (ui.showNotepad && !usePhalanx && keyboard.contains([e.clientX, e.clientY]) && !ui.tileAt(e.clientX, e.clientY)) {
    // Phone: tap a key; holding DEL keeps deleting
    const key = keyboard.tap([e.clientX, e.clientY], performance.now() / 1000);
    if (key !== null) {
      pressKey(key);
      if (key === '<BKSP>') {
        touch.repeat = setTimeout(function again() {
          pressKey('<BKSP>');
          keyboard.flash['<BKSP>'] = performance.now() / 1000;
          touch.repeat = setTimeout(again, 100);
        }, 500);
      }
    }
    touch.mode = 'key';
    return;
  }
  const box = activeBox();
  if (touch.pts.size === 1) {
    touch.mode = null;
    if (box && !box.bursting && !ui.tileAt(e.clientX, e.clientY)) {
      const face = box === rubik && rubik.idle() ? rubik.pickNear([e.clientX, e.clientY]) : null;
      if (face) { touch.mode = 'grip'; rubik.beginGrip([e.clientX, e.clientY], 0, face); rubik.touchGrip = true; }
      else if (box.contains([e.clientX, e.clientY], 0.1) || box === rubik) touch.mode = 'rotate';
      if (touch.mode) {
        touch.onBox = true; box.touchHeld = true; box.omega = [0, 0, 0];
        // Long-press (0.4 s without moving) on the box -> drag moves it
        if (box.contains([e.clientX, e.clientY], 0)) {
          clearTimeout(touch.moveTimer);
          touch.moveTimer = setTimeout(() => {
            const p = touch.pts.get(e.pointerId);
            if (touch.pts.size === 1 && p && Math.hypot(p.x - p.sx, p.y - p.sy) < 12) {
              if (touch.mode === 'grip') { rubik.endGrip(performance.now() / 1000); rubik.touchGrip = false; rubik.preview = null; }
              touch.mode = 'move';
              setStatus('MOVING BOX', 1);
              if (navigator.vibrate) navigator.vibrate(30);
            }
          }, 400);
        }
      }
    }
  } else if (touch.pts.size === 2 && box && !box.bursting) {
    if (touch.mode === 'grip') { rubik.endGrip(performance.now() / 1000); rubik.touchGrip = false; }
    const [a, b] = [...touch.pts.values()];
    if (box.contains([a.x, a.y], 0.35) || box.contains([b.x, b.y], 0.35) || touch.onBox) {
      touch.mode = 'scale';
      touch.onBox = true;
      box.touchHeld = true;
      touch.ref = { d: Math.max(1, dist([a.x, a.y], [b.x, b.y])), size: box.size, center: box.center.slice(),
        mid: [(a.x + b.x) / 2, (a.y + b.y) / 2], ang: Math.atan2(b.y - a.y, b.x - a.x) };
    }
  }
});

canvas.addEventListener('pointermove', (e) => {
  const p = touch.pts.get(e.pointerId);
  if (!p) return;
  const dx = e.clientX - p.x, dy = e.clientY - p.y;
  p.x = e.clientX; p.y = e.clientY;
  const box = activeBox();
  if (!box || box.bursting) return;
  if (touch.mode === 'move' && touch.pts.size === 1) {
    box.setCenter(box.center[0] + dx / state.W, box.center[1] + dy / state.H);
  } else if (touch.mode === 'grip' && box === rubik && touch.pts.size === 1) {
    setStatus(rubik.gripUpdate([e.clientX, e.clientY], 0, performance.now() / 1000), 0.3);
  } else if (touch.mode === 'rotate' && touch.pts.size === 1) {
    box.rotateBy(dx, dy);
  } else if (touch.mode === 'scale' && touch.pts.size >= 2) {
    const [a, b] = [...touch.pts.values()];
    const r = touch.ref;
    const d = Math.max(1, dist([a.x, a.y], [b.x, b.y]));
    const mid = [(a.x + b.x) / 2, (a.y + b.y) / 2];
    const ang = Math.atan2(b.y - a.y, b.x - a.x);
    box.setSize(r.size * d / r.d);
    box.setCenter(r.center[0] + (mid[0] - r.mid[0]) / state.W, r.center[1] + (mid[1] - r.mid[1]) / state.H);
    let dang = ang - r.ang;
    dang = Math.atan2(Math.sin(dang), Math.cos(dang));
    box.rotateBy(0, 0, dang, false);
    r.ang = ang;
  }
});

function endPointer(e) {
  const p = touch.pts.get(e.pointerId);
  if (!p) return;
  touch.pts.delete(e.pointerId);
  if (touch.repeat) { clearTimeout(touch.repeat); touch.repeat = null; }
  clearTimeout(touch.moveTimer);
  if (touch.mode === 'key') { if (touch.pts.size === 0) touch.mode = null; return; }
  const box = activeBox();
  const moved = Math.hypot(p.x - p.sx, p.y - p.sy);
  const quick = performance.now() - p.t < 350;

  if (touch.mode === 'grip' && box === rubik) {
    rubik.endGrip(performance.now() / 1000);
    rubik.touchGrip = false;
  }
  if (touch.mode === 'grip' && moved >= 12) {
    // handled above (live layer turn)
  } else if (moved < 12 && quick && touch.mode !== 'scale') {
    // Tap
    const now = performance.now();
    const tileName = ui.tileAt(p.x, p.y);
    if (tileName) runTile(tileName);
    else if (ui.handleAt(p.x, p.y)) state.sidebarOpen = !state.sidebarOpen;
    else if (box && box.contains([p.x, p.y], 0)) {
      const lt = touch.lastTap;
      if (now - lt.t < 320 && Math.hypot(p.x - lt.x, p.y - lt.y) < 40) {
        box.startBurst(now / 1000);
        setStatus('BOX BURST!', 1);
        touch.lastTap = { t: 0, x: 0, y: 0 };
      } else {
        touch.lastTap = { t: now, x: p.x, y: p.y };
      }
    }
  }

  if (touch.pts.size === 0) {
    touch.mode = null;
    touch.onBox = false;
    if (box) box.touchHeld = false;
  } else if (touch.mode === 'scale') {
    touch.mode = null; // lifting one finger ends the pinch
  }
}
canvas.addEventListener('pointerup', endPointer);
canvas.addEventListener('pointercancel', endPointer);
