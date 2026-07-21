#!/usr/bin/env python3
"""
CropDog Phase 2 — subject detection (no cropping).

Usage:
    python detect.py <image_path>

The input image is expected to already be EXIF-corrected (display-oriented);
the API route normalizes it with sharp before calling this. Prints a single
JSON line to stdout describing everything detected:

    {
      "detected": bool,
      "numPeople": int,
      "image": {"width": W, "height": H},
      "people": [
        {
          "index": 0,
          "confidence": 0.0-1.0,          # mean visibility of key landmarks
          "landmarks": [ {"i","name","x","y","visibility"}, ... ],  # vis > 0.5
          "boxes": {
            "landmark": [x0,y0,x1,y1] | null,
            "segmentation": [x0,y0,x1,y1] | null,
            "union": [x0,y0,x1,y1] | null,
            "hairExtended": [x0,y0,x1,y1] | null
          }
        }, ...
      ],
      "combinedBox": [x0,y0,x1,y1] | null   # only when 2 people
    }

Detection only: NO cropping is performed or returned.
"""

import json
import os
import sys
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ----------------------------------------------------------------------------
# Tunable constants.
# ----------------------------------------------------------------------------
POSE_VISIBILITY_THRESHOLD = 0.5
MAX_POSES = 2  # detect up to 2 people
# Extend the box top upward by this fraction of face height (eye->chin) for hair.
HAIR_EXTENSION_FACE_RATIO = 0.25
MASK_PERSON_CLASS = 15  # DeepLabV3 (Pascal VOC) "person"
MIN_MASK_COMPONENT_AREA_RATIO = 0.005

# ----------------------------------------------------------------------------
# Models (shared with the rest of the project; downloaded on first run).
# ----------------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
POSE_MODEL_PATH = os.path.join(MODELS_DIR, "pose_landmarker_full.task")
SEG_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "deeplab_v3/float32/latest/deeplab_v3.tflite"
)
SEG_MODEL_PATH = os.path.join(MODELS_DIR, "deeplab_v3.tflite")


def ensure_model(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    sys.stderr.write(f"[detect.py] downloading model: {os.path.basename(dest)}\n")
    urllib.request.urlretrieve(url, dest)


# ----------------------------------------------------------------------------
# Landmark reference (MediaPipe Pose, 33 points).
# ----------------------------------------------------------------------------
LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
    "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb", "left_hip",
    "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
]

NOSE, LEFT_EYE, RIGHT_EYE, MOUTH_L, MOUTH_R = 0, 2, 5, 9, 10
# Key landmarks the phase cares about, used for the confidence proxy.
KEY_LANDMARKS = [2, 5, 0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


# ----------------------------------------------------------------------------
# Geometry helpers.
# ----------------------------------------------------------------------------
def pixel_points(landmarks, w, h):
    """dict index -> (x_px, y_px, visibility) for landmarks above threshold."""
    pts = {}
    for i, lm in enumerate(landmarks):
        vis = getattr(lm, "visibility", 1.0)
        if vis >= POSE_VISIBILITY_THRESHOLD:
            pts[i] = (lm.x * w, lm.y * h, vis)
    return pts


def landmark_box(pts):
    if not pts:
        return None
    xs = [p[0] for p in pts.values()]
    ys = [p[1] for p in pts.values()]
    return [min(xs), min(ys), max(xs), max(ys)]


def face_height(pts):
    """Approximate eye->chin distance in px (pose has no chin landmark, so we
    extrapolate from eye->mouth or eye->nose). Returns None if unavailable."""
    def y_of(idx):
        return pts[idx][1] if idx in pts else None

    eye_ys = [y_of(LEFT_EYE), y_of(RIGHT_EYE)]
    eye_ys = [y for y in eye_ys if y is not None]
    eye_y = sum(eye_ys) / len(eye_ys) if eye_ys else None
    if eye_y is None:
        return None

    mouth_ys = [y_of(MOUTH_L), y_of(MOUTH_R)]
    mouth_ys = [y for y in mouth_ys if y is not None]
    if mouth_ys:
        mouth_y = sum(mouth_ys) / len(mouth_ys)
        # eye->mouth is ~2/3 of eye->chin.
        return max(0.0, (mouth_y - eye_y) * 1.5)

    nose_y = y_of(NOSE)
    if nose_y is not None:
        # eye->nose is ~0.45 of eye->chin.
        return max(0.0, (nose_y - eye_y) * 2.2)

    return None


def union(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def hair_extend(box, fh):
    if box is None or not fh:
        return box
    return [box[0], box[1] - HAIR_EXTENSION_FACE_RATIO * fh, box[2], box[3]]


def int_box(box, w, h):
    if box is None:
        return None
    x0 = int(max(0, np.floor(box[0])))
    y0 = int(max(0, np.floor(box[1])))
    x1 = int(min(w, np.ceil(box[2])))
    y1 = int(min(h, np.ceil(box[3])))
    return [x0, y0, x1, y1]


def per_person_seg_boxes(person_mask, centers_x):
    """Per-person segmentation bboxes. Each person-mask pixel is assigned to the
    nearest person by horizontal distance to that person's landmark-box center,
    so side-by-side (even touching) subjects get distinct silhouettes. Returns a
    list aligned with centers_x (None where a person got no pixels)."""
    n = len(centers_x)
    result = [None] * n
    ys, xs = np.where(person_mask)
    if xs.size == 0:
        return result
    centers = np.asarray(centers_x, dtype=np.float32)
    # Nearest center by |x - center_x|.
    assign = np.argmin(np.abs(xs[:, None].astype(np.float32) - centers[None, :]),
                       axis=1)
    for p in range(n):
        sel = assign == p
        if not np.any(sel):
            continue
        pxs = xs[sel]
        pys = ys[sel]
        result[p] = [int(pxs.min()), int(pys.min()),
                     int(pxs.max()) + 1, int(pys.max()) + 1]
    return result


# ----------------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------------
def run_detection(image_bgr):
    """Run pose + segmentation detection on a display-oriented BGR image and
    return the JSON-serializable result dict. The detection logic here is
    identical to the original CLI script — only the entry point changed."""
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    ensure_model(POSE_MODEL_URL, POSE_MODEL_PATH)
    ensure_model(SEG_MODEL_URL, SEG_MODEL_PATH)

    # --- Pose Landmarker (up to 2 people) ------------------------------------
    pose_opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=MAX_POSES,
        min_pose_detection_confidence=0.5,
    )
    with vision.PoseLandmarker.create_from_options(pose_opts) as landmarker:
        pose_result = landmarker.detect(mp_image)

    people_landmarks = pose_result.pose_landmarks or []

    base = {"image": {"width": w, "height": h}}
    if not people_landmarks:
        return {"detected": False, "numPeople": 0, "people": [],
                "combinedBox": None, **base}

    # --- Per-person pixel points + landmark boxes ----------------------------
    all_pts = [pixel_points(lm, w, h) for lm in people_landmarks]
    lm_boxes = [landmark_box(pts) for pts in all_pts]

    # --- Image Segmenter (person mask -> split per nearest person) -----------
    seg_boxes = [None] * len(people_landmarks)
    try:
        seg_opts = vision.ImageSegmenterOptions(
            base_options=mp_python.BaseOptions(model_asset_path=SEG_MODEL_PATH),
            running_mode=vision.RunningMode.IMAGE,
            output_category_mask=True,
        )
        with vision.ImageSegmenter.create_from_options(seg_opts) as segmenter:
            seg_result = segmenter.segment(mp_image)
        category_mask = seg_result.category_mask.numpy_view()
        person_mask = category_mask == MASK_PERSON_CLASS
        centers_x = [
            (b[0] + b[2]) / 2 if b is not None else w / 2 for b in lm_boxes
        ]
        seg_boxes = per_person_seg_boxes(person_mask, centers_x)
    except Exception as e:  # segmentation is optional; never fatal
        sys.stderr.write(f"[detect.py] segmentation skipped: {e}\n")

    # --- Assemble per person -------------------------------------------------
    people = []
    hair_boxes = []
    for idx, pts in enumerate(all_pts):
        lm_box = lm_boxes[idx]
        seg_box = seg_boxes[idx]

        uni = union(lm_box, seg_box)
        fh = face_height(pts)
        hair = hair_extend(uni, fh)

        # Confidence proxy: mean visibility of the key landmarks that are visible.
        key_vis = [pts[i][2] for i in KEY_LANDMARKS if i in pts]
        confidence = round(sum(key_vis) / len(key_vis), 3) if key_vis else 0.0

        landmark_list = [
            {
                "i": i,
                "name": LANDMARK_NAMES[i],
                "x": round(pts[i][0], 1),
                "y": round(pts[i][1], 1),
                "visibility": round(pts[i][2], 3),
            }
            for i in sorted(pts.keys())
        ]

        hair_int = int_box(hair, w, h)
        hair_boxes.append(hair_int)
        people.append(
            {
                "index": idx,
                "confidence": confidence,
                "landmarks": landmark_list,
                "boxes": {
                    "landmark": int_box(lm_box, w, h),
                    "segmentation": int_box(seg_box, w, h),
                    "union": int_box(uni, w, h),
                    "hairExtended": hair_int,
                },
            }
        )

    combined = None
    if len(hair_boxes) >= 2:
        c = hair_boxes[0]
        for b in hair_boxes[1:]:
            c = union(c, b)
        combined = int_box(c, w, h)

    return {
        "detected": True,
        "numPeople": len(people),
        "people": people,
        "combinedBox": combined,
        **base,
    }


def main():
    """CLI entry point (kept for local/debug use): detect.py <image_path>."""
    if len(sys.argv) != 2:
        sys.stderr.write("usage: detect.py <image_path>\n")
        sys.exit(2)

    data = np.fromfile(sys.argv[1], dtype=np.uint8)
    image_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image_bgr is None:
        sys.stderr.write("failed to decode input image\n")
        sys.exit(1)

    print(json.dumps(run_detection(image_bgr)))


if __name__ == "__main__":
    main()
