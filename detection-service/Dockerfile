FROM python:3.11-slim

# System libraries MediaPipe / OpenCV need at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download the MediaPipe model files at BUILD time so the container is
# self-contained and cold starts are fast (important on Render's free tier,
# which sleeps after 15 min idle and cold-starts on the next request).
RUN python -c "from detect import ensure_model, POSE_MODEL_URL, POSE_MODEL_PATH, SEG_MODEL_URL, SEG_MODEL_PATH; ensure_model(POSE_MODEL_URL, POSE_MODEL_PATH); ensure_model(SEG_MODEL_URL, SEG_MODEL_PATH)"

# Render injects $PORT; default to 8000 for local runs.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
