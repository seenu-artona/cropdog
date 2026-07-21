# CropDog

Auto-crop tool for studio photos. Split into two deployables:

- **`/` (repo root)** — the **Next.js app** (deploys to **Vercel**). Owns the web
  UI, EXIF normalization, and orchestration. Calls the detection service over HTTP.
- **`detection-service/`** — the **Python detection service** (deploys to
  **Render**). Owns `detect.py` and all MediaPipe / computer-vision work. A minimal
  FastAPI app exposing `POST /detect` and `GET /health`.

```
Browser ──upload──▶ Next.js /api/detect ──(sharp EXIF)──▶ Render FastAPI /detect ──▶ MediaPipe
                    (Vercel)                                (detection-service)
```

## Local development

You need **two processes**: the Next.js app and the detection service.

**1. Detection service** (Docker):

```bash
docker-compose up --build      # serves on http://localhost:8000
```

Or, without Docker, run it directly (needs a Python env with the
`detection-service/requirements.txt` deps installed):

```bash
cd detection-service
uvicorn main:app --host 0.0.0.0 --port 8000
```

**2. Next.js app**:

```bash
npm install
npm run dev                    # http://localhost:3000
```

The app reads `DETECTION_SERVICE_URL` from `.env.local` (defaults to
`http://localhost:8000`). Open http://localhost:3000, upload a JPEG/PNG, and
click **Process** to see the detection debug overlay.

> First request after the service is idle may be slow while models load. The
> models are baked into the Docker image at build time, so cold starts stay fast.

## Deploy

### Detection service → Render

1. Push to GitHub.
2. On [render.com](https://render.com): **New +** → **Blueprint** → connect the
   `seenu-artona/cropdog` repo. Render reads [`render.yaml`](render.yaml) and sets
   up the `cropdog-detection` Docker web service automatically (free plan,
   health check at `/health`).
3. Wait for the first build (installs deps + downloads MediaPipe models), then
   copy the public URL (e.g. `https://cropdog-detection.onrender.com`).

### Next.js app → Vercel

4. In the Vercel project, set env var **`DETECTION_SERVICE_URL`** to the Render URL.
5. Redeploy.

> Render's free tier sleeps after ~15 min idle; the next request cold-starts in
> ~30–60s. The Next.js route waits up to 90s and returns a clear "waking up /
> unavailable" message if the service is down.

## Layout

- `app/page.jsx` — single page: upload, Process, detection debug overlay.
- `app/api/detect/route.js` — EXIF-normalizes with sharp, POSTs to the detection service.
- `app/api/process/route.js` — Phase 1 echo endpoint (EXIF normalization).
- `detection-service/detect.py` — MediaPipe pose + segmentation, bounding boxes.
- `detection-service/main.py` — FastAPI wrapper (`/detect`, `/health`).
- `detection-service/Dockerfile` — installs deps, pre-downloads models, runs uvicorn.
- `render.yaml` — Render Blueprint for the detection service.
- `docker-compose.yml` — runs the detection service locally on port 8000.
- `scripts/crop.py` — earlier standalone crop reference (not used by the app).
