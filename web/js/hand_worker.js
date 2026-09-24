// Hand tracking off the main thread: the page keeps drawing (and recording) at camera speed while
// MediaPipe runs here. The page sends ImageBitmaps; we reply with landmarks + handedness.
// Classic (non-module) worker: MediaPipe loads its WASM glue with importScripts, which module workers forbid,
// so the CommonJS bundle is loaded with a tiny module/exports shim.
self.module = { exports: {} };
self.exports = self.module.exports;
// Local copy of @mediapipe/tasks-vision 0.10.14 vision_bundle.cjs (Apache-2.0): the CDN serves .cjs as
// application/node, which importScripts refuses
importScripts('../vendor/vision_bundle.js');
const { FilesetResolver, HandLandmarker } = self.module.exports;

const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

let landmarker = null;

async function create(fileset, delegate) {
  return HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: 'VIDEO',
    numHands: 2,
    minHandDetectionConfidence: 0.6,
    minHandPresenceConfidence: 0.6,
    minTrackingConfidence: 0.5,
  });
}

self.onmessage = async (e) => {
  const m = e.data;
  if (m.type === 'init') {
    try {
      const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
      let delegate = m.delegate || 'GPU';
      try {
        landmarker = await create(fileset, delegate);
      } catch (err) {
        delegate = 'CPU';
        landmarker = await create(fileset, 'CPU');
      }
      self.postMessage({ type: 'ready', delegate });
    } catch (err) {
      self.postMessage({ type: 'error', message: String(err && err.message || err) });
    }
  } else if (m.type === 'frame') {
    const t0 = performance.now();
    let r = { landmarks: [], handedness: [] };
    try {
      r = landmarker.detectForVideo(m.bitmap, m.ts);
    } catch (err) {
      // a bad frame just yields no hands
    }
    m.bitmap.close();
    self.postMessage({
      type: 'result',
      id: m.id,
      ms: performance.now() - t0,
      landmarks: (r.landmarks || []).map((hand) => hand.map((p) => ({ x: p.x, y: p.y, z: p.z }))),
      handedness: (r.handedness || []).map((c) => [{ categoryName: c[0]?.categoryName }]),
    });
  }
};
