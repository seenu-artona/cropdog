"""CropDog detection service — minimal FastAPI wrapper around detect.py.

Endpoints:
  GET  /health  -> {"ok": true}
  POST /detect  -> accepts an image (multipart form field "image", or JSON
                   {"image_base64": "..."}) and returns exactly the JSON that
                   detect.run_detection produces. No detection logic lives here.
"""

import base64

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request

from detect import run_detection

app = FastAPI(title="CropDog Detection Service")


@app.get("/health")
def health():
    return {"ok": True}


def _decode(data: bytes):
    arr = np.frombuffer(data, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return image_bgr


@app.post("/detect")
async def detect(request: Request):
    content_type = request.headers.get("content-type", "")

    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("image") or form.get("file")
        if upload is None or not hasattr(upload, "read"):
            raise HTTPException(status_code=400, detail="No image field in form data")
        data = await upload.read()
    else:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Expected multipart form-data or JSON with image_base64",
            )
        b64 = body.get("image_base64") or body.get("image")
        if not b64:
            raise HTTPException(status_code=400, detail="Missing image_base64")
        if isinstance(b64, str) and b64.startswith("data:"):
            b64 = b64.split(",", 1)[1]
        try:
            data = base64.b64decode(b64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image")

    image_bgr = _decode(data)
    return run_detection(image_bgr)
