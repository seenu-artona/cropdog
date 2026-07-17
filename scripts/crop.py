#!/usr/bin/env python3
"""
CropDog auto-crop core.

Usage:
    python crop.py <input_image_path> <output_image_path>

Prints a single JSON line to stdout describing the result:
    {"detected": bool, "numPeople": int, "box": [x0, y0, x1, y1] | null}

Algorithm (see project spec):
  1. Detect every person with MediaPipe Pose Landmarker.
     - Keep landmarks with visibility above a threshold.
     - Build each person's pixel bounding box.
     - Extend the box upward above the topmost landmark to account for hair,
       using a fraction of the person's own head-height (ear->shoulder proxy).
     - Refine box edges using the Image Segmenter's person mask (catches
       hands / stray hair the pose landmarks miss).
  2. If >1 person, take the UNION of all boxes (no "primary" person).
  3. Pad the merged box outward by a fixed percentage on all four sides.
  4. Clamp to the image edges (never invent pixels).
  5. Grow the box (only ever adding background on the short side -- the subject
     is never cut) so the output aspect ratio matches the input file's. No
     rule-of-thirds, no orientation bias.
  6. Crop to that final rectangle. If NO person is detected, write the original
     image unchanged and report detected=False.
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
# Tunable constants (the whole point of the MVP is to tune these).
# ----------------------------------------------------------------------------

# Minimum pose-landmark visibility to trust a landmark.
POSE_VISIBILITY_THRESHOLD = 0.5

# How far to extend the box ABOVE the topmost landmark, as a fraction of the
# person's own head-height (ear->shoulder distance). Spec says ~60-80%.
HEAD_EXTENSION_RATIO = 0.7

# Fallback head-height as a fraction of the person's landmark-box height,
# used only when ear/shoulder landmarks aren't both visible.
HEAD_HEIGHT_FALLBACK_RATIO = 0.20

# Fixed padding added on all four sides of the merged subject box, as a
# fraction of the box's width/height.
PADDING_RATIO = 0.08

# Ignore person-mask blobs smaller than this fraction of the image area
# (segmentation noise) when refining boxes.
MIN_MASK_COMPONENT_AREA_RATIO = 0.005

# DeepLabV3 (Pascal VOC) label index for "person".
MASK_PERSON_CLASS = 15

# Max number of people to detect.
MAX_POSES = 6

# ----------------------------------------------------------------------------
# Model management (downloaded on first run).
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
    sys.stderr.write(f"[crop.py] downloading model: {os.path.basename(dest)}\n")
    urllib.request.urlretrieve(url, dest)


# ----------------------------------------------------------------------------
# Geometry helpers.
# ----------------------------------------------------------------------------

# MediaPipe pose landmark indices we care about for the head-height proxy.
L_EAR, R_EAR = 7, 8
L_SHOULDER, R_SHOULDER = 11, 12

# Head landmarks used to find the face's horizontal center for centering:
# nose, both eyes (inner/center), both ears.
FACE_LANDMARKS = [0, 2, 5, 7, 8]


def face_center_x(landmarks, w):
    """Horizontal center of the face from visible head landmarks (px), or None."""
    xs = []
    for i in FACE_LANDMARKS:
        lm = landmarks[i]
        if getattr(lm, "visibility", 1.0) < POSE_VISIBILITY_THRESHOLD:
            continue
        xs.append(lm.x * w)
    if not xs:
        return None
    return sum(xs) / len(xs)


def person_box_from_landmarks(landmarks, w, h):
    """Return [x0, y0, x1, y1] pixel box from visible landmarks, extended
    upward for hair. Returns None if no landmark is visible enough."""
    xs, ys = [], []
    pts = {}  # index -> (x_px, y_px) for the landmarks we need
    for i, lm in enumerate(landmarks):
        if getattr(lm, "visibility", 1.0) < POSE_VISIBILITY_THRESHOLD:
            continue
        x_px = lm.x * w
        y_px = lm.y * h
        xs.append(x_px)
        ys.append(y_px)
        pts[i] = (x_px, y_px)

    if not xs:
        return None

    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)

    # --- Head extension above the topmost landmark ---------------------------
    head_height = estimate_head_height(pts, box_height=(y1 - y0))
    y0_extended = y0 - HEAD_EXTENSION_RATIO * head_height

    return [x0, y0_extended, x1, y1]


def estimate_head_height(pts, box_height):
    """Head-height proxy = vertical ear->shoulder distance, averaged over any
    visible side. Falls back to a fraction of the box height."""
    dists = []
    for ear, shoulder in ((L_EAR, L_SHOULDER), (R_EAR, R_SHOULDER)):
        if ear in pts and shoulder in pts:
            dists.append(abs(pts[shoulder][1] - pts[ear][1]))
    if dists:
        return sum(dists) / len(dists)
    return HEAD_HEIGHT_FALLBACK_RATIO * max(box_height, 1.0)


def mask_box_for_boxes(pose_boxes, person_mask, w, h):
    """Refine using the person mask by connected components.

    Each detected person's landmark box sits on top of that person's blob in
    the segmentation mask. We take the full extent of every mask blob that a
    pose box lands on — this recovers crossed arms, hands, and stray hair that
    extend well beyond the pose skeleton, while ignoring background regions
    that no detected person overlaps. Returns a box [x0,y0,x1,y1] or None."""
    mask_u8 = person_mask.astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(mask_u8)
    if num_labels <= 1:
        return None  # no person pixels

    min_area = MIN_MASK_COMPONENT_AREA_RATIO * (w * h)

    selected = set()
    for box in pose_boxes:
        x0 = max(0, int(np.floor(box[0])))
        y0 = max(0, int(np.floor(box[1])))
        x1 = min(w, int(np.ceil(box[2])))
        y1 = min(h, int(np.ceil(box[3])))
        if x1 <= x0 or y1 <= y0:
            continue
        region = labels[y0:y1, x0:x1]
        region = region[region > 0]
        if region.size == 0:
            continue
        # The blob this person occupies = the label covering the most of
        # their landmark box.
        vals, counts = np.unique(region, return_counts=True)
        selected.add(int(vals[np.argmax(counts)]))

    if not selected:
        return None

    keep = np.isin(labels, list(selected)) & (labels > 0)
    # Drop tiny specks that survived only via a grazing overlap.
    for lbl in list(selected):
        if int((labels == lbl).sum()) < min_area:
            keep &= labels != lbl

    ys, xs = np.where(keep)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def union_boxes(boxes):
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    return [x0, y0, x1, y1]


def pad_box(box, ratio):
    x0, y0, x1, y1 = box
    bw = x1 - x0
    bh = y1 - y0
    return [
        x0 - ratio * bw,
        y0 - ratio * bh,
        x1 + ratio * bw,
        y1 + ratio * bh,
    ]


def clamp_box(box, w, h):
    x0, y0, x1, y1 = box
    x0 = int(max(0, np.floor(x0)))
    y0 = int(max(0, np.floor(y0)))
    x1 = int(min(w, np.ceil(x1)))
    y1 = int(min(h, np.ceil(y1)))
    return [x0, y0, x1, y1]


def fit_box_to_aspect(box, w, h, face_cx=None):
    """Grow `box` so its aspect ratio matches the input image's, staying inside
    the image. Only ever ADDS background on the short axis -- the subject region
    is never cut. The target dimension can't exceed the image (bh*R <= w and
    bw/R <= h), so it always fits; a full-image box just returns the frame.

    When adding width, center the crop horizontally on the face (`face_cx`, px)
    instead of the box center, but only within the slack that still keeps the
    whole subject box in frame and the window inside the image. Falls back to the
    box center if face_cx is None. Height stays head-anchored (box-centered)."""
    x0, y0, x1, y1 = box
    bw = x1 - x0
    bh = y1 - y0
    if bw <= 0 or bh <= 0:
        return box

    target_ratio = w / h  # width / height
    box_ratio = bw / bh
    if abs(box_ratio - target_ratio) < 1e-3:
        return box

    if box_ratio < target_ratio:
        # Too tall/narrow -> add width, keep height.
        tw = min(w, int(round(bh * target_ratio)))
        anchor = (x0 + x1) / 2 if face_cx is None else face_cx
        # Keep the whole subject box in frame: nx0 in [x1 - tw, x0]. Also inside
        # the image: nx0 in [0, w - tw]. (tw >= bw guarantees this is valid.)
        lo = max(0, x1 - tw)
        hi = min(x0, w - tw)
        nx0 = int(round(anchor - tw / 2))
        nx0 = max(lo, min(nx0, hi))
        return [nx0, y0, nx0 + tw, y1]

    # Too wide/short -> add height, keep width (head-anchored: box-centered).
    th = min(h, int(round(bw / target_ratio)))
    cy = (y0 + y1) / 2
    ny0 = int(round(cy - th / 2))
    ny0 = max(0, min(ny0, h - th))
    return [x0, ny0, x1, ny0 + th]


# ----------------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------------

def main():
    if len(sys.argv) != 3:
        sys.stderr.write("usage: crop.py <input> <output>\n")
        sys.exit(2)

    input_path, output_path = sys.argv[1], sys.argv[2]

    # Read image (BGR). cv2 handles most formats; decode robustly.
    data = np.fromfile(input_path, dtype=np.uint8)
    image_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image_bgr is None:
        sys.stderr.write("failed to decode input image\n")
        sys.exit(1)

    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    ensure_model(POSE_MODEL_URL, POSE_MODEL_PATH)
    ensure_model(SEG_MODEL_URL, SEG_MODEL_PATH)

    # --- Pose Landmarker (multi-person) --------------------------------------
    pose_opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=MAX_POSES,
        min_pose_detection_confidence=0.5,
    )
    with vision.PoseLandmarker.create_from_options(pose_opts) as landmarker:
        pose_result = landmarker.detect(mp_image)

    people = pose_result.pose_landmarks or []

    # No person detected -> return original unchanged.
    if not people:
        cv2.imwrite(output_path, image_bgr)
        print(json.dumps({"detected": False, "numPeople": 0, "box": None}))
        return

    # --- Image Segmenter (person mask) ---------------------------------------
    person_mask = None
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
    except Exception as e:  # segmentation is a refinement; never fatal
        sys.stderr.write(f"[crop.py] segmentation skipped: {e}\n")
        person_mask = None

    # --- Per-person landmark boxes (with head extension) + face centers ------
    boxes = []
    face_xs = []
    for landmarks in people:
        box = person_box_from_landmarks(landmarks, w, h)
        if box is None:
            continue
        boxes.append(box)
        fx = face_center_x(landmarks, w)
        if fx is not None:
            face_xs.append(fx)

    if not boxes:
        # People detected but no landmark cleared the visibility threshold.
        cv2.imwrite(output_path, image_bgr)
        print(json.dumps({"detected": False, "numPeople": 0, "box": None}))
        return

    # Horizontal center of all faces (mean), or None if none were visible.
    face_cx = sum(face_xs) / len(face_xs) if face_xs else None

    # --- Union -> refine with mask -> pad -> clamp ---------------------------
    merged = union_boxes(boxes)

    # Refine edges with the person mask: grow the box to the full extent of
    # every mask blob a person sits on (hands / crossed arms / stray hair).
    if person_mask is not None:
        mask_box = mask_box_for_boxes(boxes, person_mask, w, h)
        if mask_box is not None:
            merged = union_boxes([merged, mask_box])

    padded = pad_box(merged, PADDING_RATIO)
    base = clamp_box(padded, w, h)
    # Match the output aspect ratio to the input file's (grows only; never cuts),
    # centering horizontally on the face.
    x0, y0, x1, y1 = fit_box_to_aspect(base, w, h, face_cx)

    if x1 <= x0 or y1 <= y0:
        cv2.imwrite(output_path, image_bgr)
        print(json.dumps({"detected": False, "numPeople": 0, "box": None}))
        return

    cropped = image_bgr[y0:y1, x0:x1]
    cv2.imwrite(output_path, cropped)

    print(
        json.dumps(
            {
                "detected": True,
                "numPeople": len(boxes),
                "box": [x0, y0, x1, y1],
            }
        )
    )


if __name__ == "__main__":
    main()
