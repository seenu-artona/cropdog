# CropDog — Auto-Crop Validation MVP

Throwaway MVP to prove the auto-crop algorithm works. One image in, one
cropped image out. Stateless, local-only. No accounts, storage, or config.

## What it does

Upload an image, click **Crop**. The server:

1. Detects every person with **MediaPipe Pose Landmarker**.
2. Builds each person's bounding box from visible landmarks, extends it
   upward for hair (fraction of ear→shoulder head-height), and refines the
   edges using the **Image Segmenter** person mask (catches hands / stray hair).
3. Takes the **union** of all people's boxes (no "primary" person).
4. Pads the box by a fixed % on all sides, then clamps to the image edges.
5. Crops to that rectangle — no fixed aspect ratio, no rule-of-thirds, no
   orientation bias.
6. If no person is detected, returns the original image and says so.

Tunable constants live at the top of [`scripts/crop.py`](scripts/crop.py)
(`HEAD_EXTENSION_RATIO`, `PADDING_RATIO`, etc.).

## Setup

Requires Node 18+ and Python 3.9–3.12.

```bash
# 1. Python environment
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt

# 2. Node deps
npm install

# 3. Run
npm run dev
```

Then open http://localhost:3000.

> On the **first crop**, MediaPipe downloads its two models (~a few tens of MB)
> into `scripts/models/`. That request may take several seconds.

If your Python interpreter isn't at `./.venv/bin/python`, set `CROPDOG_PYTHON`
to the interpreter path before `npm run dev`.

## Layout

- `app/page.jsx` — the single page (upload, Crop, result, download).
- `app/api/crop/route.js` — API route; spawns the Python script.
- `scripts/crop.py` — the cropping algorithm (MediaPipe).
