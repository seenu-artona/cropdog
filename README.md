# CropDog — Auto-Crop Validation MVP

Throwaway MVP to prove the auto-crop algorithm works. One image in, one
cropped image out. Stateless, no accounts, no storage.

Everything runs **client-side in the browser** via MediaPipe Tasks (WASM), so
the app is a static Next.js site that deploys to Vercel with **no backend**.
Images never leave the browser.

## What it does

Upload an image, click **Crop**. In the browser:

1. Detects every person with **MediaPipe Pose Landmarker**.
2. Builds each person's bounding box from visible landmarks, extends it upward
   for hair (fraction of ear→shoulder head-height), and refines the edges with
   the **Image Segmenter** person mask — attributing each person to their
   connected mask blob to catch hands / crossed arms / stray hair.
3. Takes the **union** of all people's boxes (no "primary" person).
4. Pads the box by a fixed % on all sides, then clamps to the image edges.
5. Grows the box (only ever adding background on the short side — the subject
   is never cut) so the **output aspect ratio matches the input file's**, then
   crops. No rule-of-thirds, no orientation bias.
6. If no person is detected, returns the original image and says so.

Tunable constants live at the top of
[`app/lib/autocrop.js`](app/lib/autocrop.js)
(`HEAD_EXTENSION_RATIO`, `PADDING_RATIO`, etc.). The models are downloaded from
the MediaPipe CDN on first crop and cached by the browser.

## Run locally

Requires Node 18+.

```bash
npm install
npm run dev
```

Then open http://localhost:3000.

## Deploy to Vercel

It's a standard Next.js app with no backend, so:

1. Push to GitHub (already at `github.com/seenu-artona/cropdog`).
2. In Vercel: **New Project → import the repo → Deploy.** Framework
   auto-detects as Next.js; no environment variables or settings needed.

## Layout

- `app/page.jsx` — the single page (upload, Crop, result, download).
- `app/lib/autocrop.js` — the cropping algorithm (MediaPipe WASM, client-side).
- `scripts/crop.py` — the original Python reference implementation of the same
  algorithm (not used by the app; kept for reference / server-side use).
