// CropDog auto-crop core — in-browser port of scripts/crop.py.
//
// Runs entirely client-side using MediaPipe Tasks (WASM), so the app deploys
// to Vercel as a static site with no backend. The algorithm mirrors crop.py:
//
//   1. Detect every person with the Pose Landmarker.
//   2. Per person: pixel box from visible landmarks, extended upward for hair
//      by a fraction of the ear->shoulder head-height.
//   3. Refine edges with the Image Segmenter person mask, attributing each
//      person to their connected mask blob (catches hands / crossed arms /
//      stray hair the pose skeleton misses).
//   4. Union all people's boxes (no "primary" person).
//   5. Pad a fixed % on all sides, clamp to image edges, crop.
//   6. No person -> return the original unchanged.

// ---------------------------------------------------------------------------
// Tunable constants (kept in lockstep with scripts/crop.py).
// ---------------------------------------------------------------------------
export const POSE_VISIBILITY_THRESHOLD = 0.5;
export const HEAD_EXTENSION_RATIO = 0.7;
export const HEAD_HEIGHT_FALLBACK_RATIO = 0.2;
export const PADDING_RATIO = 0.08;
export const MIN_MASK_COMPONENT_AREA_RATIO = 0.005;
export const MASK_PERSON_CLASS = 15; // DeepLabV3 (Pascal VOC) "person"
export const MAX_POSES = 6;

// Detection runs on a downscaled copy for speed; the final crop uses full res.
const DETECT_MAX_DIM = 1024;

// MediaPipe pose landmark indices for the head-height proxy.
const L_EAR = 7;
const R_EAR = 8;
const L_SHOULDER = 11;
const R_SHOULDER = 12;

// Same models the Python version uses.
const WASM_BASE =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const POSE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task";
const SEG_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/latest/deeplab_v3.tflite";

// ---------------------------------------------------------------------------
// Lazy singletons — models load once, on first crop.
// ---------------------------------------------------------------------------
let _tasksPromise = null;

async function getTasks() {
  if (!_tasksPromise) {
    _tasksPromise = (async () => {
      const { FilesetResolver, PoseLandmarker, ImageSegmenter } = await import(
        "@mediapipe/tasks-vision"
      );
      const vision = await FilesetResolver.forVisionTasks(WASM_BASE);

      const poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: POSE_MODEL_URL },
        runningMode: "IMAGE",
        numPoses: MAX_POSES,
        minPoseDetectionConfidence: 0.5,
      });

      const imageSegmenter = await ImageSegmenter.createFromOptions(vision, {
        baseOptions: { modelAssetPath: SEG_MODEL_URL },
        runningMode: "IMAGE",
        outputCategoryMask: true,
        outputConfidenceMasks: false,
      });

      return { poseLandmarker, imageSegmenter };
    })();
  }
  return _tasksPromise;
}

// ---------------------------------------------------------------------------
// Geometry helpers (mirror crop.py).
// ---------------------------------------------------------------------------
function estimateHeadHeight(pts, boxHeight) {
  const dists = [];
  for (const [ear, shoulder] of [
    [L_EAR, L_SHOULDER],
    [R_EAR, R_SHOULDER],
  ]) {
    if (pts[ear] && pts[shoulder]) {
      dists.push(Math.abs(pts[shoulder].y - pts[ear].y));
    }
  }
  if (dists.length) return dists.reduce((a, b) => a + b, 0) / dists.length;
  return HEAD_HEIGHT_FALLBACK_RATIO * Math.max(boxHeight, 1);
}

// Returns [x0, y0, x1, y1] in detection-space px, or null.
function personBoxFromLandmarks(landmarks, w, h) {
  const xs = [];
  const ys = [];
  const pts = {};
  landmarks.forEach((lm, i) => {
    const vis = lm.visibility == null ? 1 : lm.visibility;
    if (vis < POSE_VISIBILITY_THRESHOLD) return;
    const x = lm.x * w;
    const y = lm.y * h;
    xs.push(x);
    ys.push(y);
    pts[i] = { x, y };
  });
  if (!xs.length) return null;

  const x0 = Math.min(...xs);
  const x1 = Math.max(...xs);
  const y0 = Math.min(...ys);
  const y1 = Math.max(...ys);

  const headHeight = estimateHeadHeight(pts, y1 - y0);
  const y0Extended = y0 - HEAD_EXTENSION_RATIO * headHeight;
  return [x0, y0Extended, x1, y1];
}

function unionBoxes(boxes) {
  return [
    Math.min(...boxes.map((b) => b[0])),
    Math.min(...boxes.map((b) => b[1])),
    Math.max(...boxes.map((b) => b[2])),
    Math.max(...boxes.map((b) => b[3])),
  ];
}

function padBox(box, ratio) {
  const [x0, y0, x1, y1] = box;
  const bw = x1 - x0;
  const bh = y1 - y0;
  return [x0 - ratio * bw, y0 - ratio * bh, x1 + ratio * bw, y1 + ratio * bh];
}

function clampBox(box, w, h) {
  return [
    Math.max(0, Math.floor(box[0])),
    Math.max(0, Math.floor(box[1])),
    Math.min(w, Math.ceil(box[2])),
    Math.min(h, Math.ceil(box[3])),
  ];
}

// Grow `box` so its aspect ratio matches the input image's, then keep it inside
// the image. We only ever ADD background on the short axis — the subject region
// is never cut. The target dimension can never exceed the image (bh*R <= imgW
// and bw/R <= imgH), so the box always fits; if it's already the full image we
// simply return the whole frame.
function fitBoxToAspect(box, imgW, imgH) {
  const [x0, y0, x1, y1] = box;
  const bw = x1 - x0;
  const bh = y1 - y0;
  if (bw <= 0 || bh <= 0) return box;

  const targetRatio = imgW / imgH; // width / height
  const boxRatio = bw / bh;
  const EPS = 1e-3;
  if (Math.abs(boxRatio - targetRatio) < EPS) return box;

  if (boxRatio < targetRatio) {
    // Too tall/narrow -> add width, keep height.
    const tw = Math.min(imgW, Math.round(bh * targetRatio));
    const cx = (x0 + x1) / 2;
    let nx0 = Math.round(cx - tw / 2);
    nx0 = Math.max(0, Math.min(nx0, imgW - tw));
    return [nx0, y0, nx0 + tw, y1];
  }
  // Too wide/short -> add height, keep width.
  const th = Math.min(imgH, Math.round(bw / targetRatio));
  const cy = (y0 + y1) / 2;
  let ny0 = Math.round(cy - th / 2);
  ny0 = Math.max(0, Math.min(ny0, imgH - th));
  return [x0, ny0, x1, ny0 + th];
}

// 4-connected connected-component labelling over the person mask.
function labelComponents(mask, w, h) {
  const labels = new Int32Array(w * h); // 0 = unlabeled/background
  let next = 0;
  const stack = [];
  for (let start = 0; start < mask.length; start++) {
    if (mask[start] !== MASK_PERSON_CLASS || labels[start] !== 0) continue;
    next += 1;
    stack.push(start);
    labels[start] = next;
    while (stack.length) {
      const idx = stack.pop();
      const x = idx % w;
      const y = (idx - x) / w;
      // neighbors
      if (x > 0) tryPush(idx - 1);
      if (x < w - 1) tryPush(idx + 1);
      if (y > 0) tryPush(idx - w);
      if (y < h - 1) tryPush(idx + w);
    }
    // eslint-disable-next-line no-inner-declarations
    function tryPush(n) {
      if (mask[n] === MASK_PERSON_CLASS && labels[n] === 0) {
        labels[n] = next;
        stack.push(n);
      }
    }
  }
  return { labels, count: next };
}

// Refine using the person mask: grow to the full extent of every mask blob a
// pose box sits on. Returns [x0,y0,x1,y1] in mask-space px, or null.
function maskBoxForBoxes(poseBoxes, mask, w, h) {
  const { labels, count } = labelComponents(mask, w, h);
  if (count === 0) return null;

  // Area per label, for dropping tiny specks.
  const areas = new Int32Array(count + 1);
  for (let i = 0; i < labels.length; i++) areas[labels[i]]++;
  const minArea = MIN_MASK_COMPONENT_AREA_RATIO * w * h;

  const selected = new Set();
  for (const box of poseBoxes) {
    const x0 = Math.max(0, Math.floor(box[0]));
    const y0 = Math.max(0, Math.floor(box[1]));
    const x1 = Math.min(w, Math.ceil(box[2]));
    const y1 = Math.min(h, Math.ceil(box[3]));
    if (x1 <= x0 || y1 <= y0) continue;

    // Dominant label covering this person's landmark box.
    const counts = new Map();
    for (let y = y0; y < y1; y++) {
      const row = y * w;
      for (let x = x0; x < x1; x++) {
        const lbl = labels[row + x];
        if (lbl > 0) counts.set(lbl, (counts.get(lbl) || 0) + 1);
      }
    }
    let best = 0;
    let bestCount = 0;
    for (const [lbl, c] of counts) {
      if (c > bestCount) {
        bestCount = c;
        best = lbl;
      }
    }
    if (best > 0 && areas[best] >= minArea) selected.add(best);
  }
  if (!selected.size) return null;

  let mx0 = Infinity;
  let my0 = Infinity;
  let mx1 = -Infinity;
  let my1 = -Infinity;
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      if (selected.has(labels[row + x])) {
        if (x < mx0) mx0 = x;
        if (x > mx1) mx1 = x;
        if (y < my0) my0 = y;
        if (y > my1) my1 = y;
      }
    }
  }
  if (mx1 < mx0) return null;
  return [mx0, my0, mx1 + 1, my1 + 1];
}

// ---------------------------------------------------------------------------
// Public API.
// ---------------------------------------------------------------------------

/**
 * Crop a person/people out of an image file.
 * @param {File|Blob} file
 * @returns {Promise<{detected:boolean, numPeople:number, box:number[]|null, blob:Blob, width:number, height:number}>}
 */
export async function autoCrop(file) {
  const { poseLandmarker, imageSegmenter } = await getTasks();

  const bitmap = await createImageBitmap(file);
  const fullW = bitmap.width;
  const fullH = bitmap.height;

  // Downscaled canvas for detection.
  const scale = Math.min(1, DETECT_MAX_DIM / Math.max(fullW, fullH));
  const dw = Math.max(1, Math.round(fullW * scale));
  const dh = Math.max(1, Math.round(fullH * scale));
  const dCanvas = document.createElement("canvas");
  dCanvas.width = dw;
  dCanvas.height = dh;
  const dCtx = dCanvas.getContext("2d", { willReadFrequently: true });
  dCtx.drawImage(bitmap, 0, 0, dw, dh);

  // --- Pose detection ---
  const poseResult = poseLandmarker.detect(dCanvas);
  const people = poseResult.landmarks || [];

  const makeOriginal = async () => {
    const c = document.createElement("canvas");
    c.width = fullW;
    c.height = fullH;
    c.getContext("2d").drawImage(bitmap, 0, 0);
    const blob = await canvasToBlob(c);
    return { detected: false, numPeople: 0, box: null, blob, width: fullW, height: fullH };
  };

  if (!people.length) return makeOriginal();

  // Per-person landmark boxes (detection space).
  const boxes = [];
  for (const landmarks of people) {
    const box = personBoxFromLandmarks(landmarks, dw, dh);
    if (box) boxes.push(box);
  }
  if (!boxes.length) return makeOriginal();

  let merged = unionBoxes(boxes);

  // --- Segmentation refine ---
  try {
    const maskData = await segmentPersonMask(imageSegmenter, dCanvas, dw, dh);
    if (maskData) {
      const maskBox = maskBoxForBoxes(boxes, maskData, dw, dh);
      if (maskBox) merged = unionBoxes([merged, maskBox]);
    }
  } catch (e) {
    // Segmentation is a refinement; never fatal.
    // eslint-disable-next-line no-console
    console.warn("segmentation skipped:", e);
  }

  // Scale detection-space box to full-res, then pad + clamp.
  const sx = fullW / dw;
  const sy = fullH / dh;
  const fullBox = [merged[0] * sx, merged[1] * sy, merged[2] * sx, merged[3] * sy];
  const padded = padBox(fullBox, PADDING_RATIO);
  const base = clampBox(padded, fullW, fullH);
  // Match the output aspect ratio to the input file's (grows only; never cuts).
  const [x0, y0, x1, y1] = fitBoxToAspect(base, fullW, fullH);

  if (x1 <= x0 || y1 <= y0) return makeOriginal();

  const out = document.createElement("canvas");
  out.width = x1 - x0;
  out.height = y1 - y0;
  out.getContext("2d").drawImage(bitmap, x0, y0, x1 - x0, y1 - y0, 0, 0, x1 - x0, y1 - y0);
  const blob = await canvasToBlob(out);

  return {
    detected: true,
    numPeople: boxes.length,
    box: [x0, y0, x1, y1],
    blob,
    width: out.width,
    height: out.height,
  };
}

function segmentPersonMask(segmenter, canvas, w, h) {
  return new Promise((resolve, reject) => {
    try {
      segmenter.segment(canvas, (result) => {
        try {
          const categoryMask = result.categoryMask;
          if (!categoryMask) {
            resolve(null);
            return;
          }
          // Copy out before the mask is closed/freed.
          const data = Uint8Array.from(categoryMask.getAsUint8Array());
          categoryMask.close();
          resolve(data);
        } catch (err) {
          reject(err);
        }
      });
    } catch (err) {
      reject(err);
    }
  });
}

function canvasToBlob(canvas) {
  return new Promise((resolve) =>
    canvas.toBlob((b) => resolve(b), "image/png")
  );
}
