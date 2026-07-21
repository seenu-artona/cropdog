"use client";

import { useState, useRef, useEffect } from "react";

const ACCEPTED = ["image/jpeg", "image/png"];

// Per-person colors (index 0, 1).
const PERSON_COLORS = ["#e6194b", "#3b82f6"];
const COMBINED_COLOR = "#111111";

// Load the file as a display-oriented ImageBitmap (EXIF baked in).
async function loadOrientedBitmap(file) {
  try {
    return await createImageBitmap(file, { imageOrientation: "from-image" });
  } catch {
    return await createImageBitmap(file);
  }
}

function drawBox(ctx, box, color, lineWidth, dashed) {
  if (!box) return;
  const [x0, y0, x1, y1] = box;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.setLineDash(dashed ? [lineWidth * 4, lineWidth * 3] : []);
  ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
  ctx.restore();
}

// Draw the original image + all debug overlays onto the canvas.
function renderOverlay(canvas, bitmap, data) {
  const W = data.image.width;
  const H = data.image.height;
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(bitmap, 0, 0, W, H);

  const unit = Math.max(1.5, W / 500); // scale strokes/dots to image size

  data.people.forEach((person, i) => {
    const color = PERSON_COLORS[i % PERSON_COLORS.length];
    // Boxes: landmark (thin), segmentation (thin dashed), union+hair (thick).
    drawBox(ctx, person.boxes.landmark, color, unit, false);
    drawBox(ctx, person.boxes.segmentation, color, unit, true);
    drawBox(ctx, person.boxes.hairExtended, color, unit * 2.4, false);

    // Landmark dots.
    ctx.fillStyle = color;
    const r = unit * 2;
    for (const lm of person.landmarks) {
      ctx.beginPath();
      ctx.arc(lm.x, lm.y, r, 0, Math.PI * 2);
      ctx.fill();
    }
  });

  // Combined subject box (only when 2 people).
  if (data.combinedBox) {
    drawBox(ctx, data.combinedBox, COMBINED_COLOR, unit * 2.4, true);
  }
}

export default function Home() {
  const [file, setFile] = useState(null);
  const [bitmap, setBitmap] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const canvasRef = useRef(null);

  // (Re)draw whenever we have both the image and a detection result.
  useEffect(() => {
    if (data?.detected && bitmap && canvasRef.current) {
      renderOverlay(canvasRef.current, bitmap, data);
    }
  }, [data, bitmap]);

  async function handleFileChange(e) {
    const f = e.target.files?.[0] || null;
    setData(null);
    setError(null);
    if (!f) {
      setFile(null);
      setBitmap(null);
      setPreviewUrl(null);
      return;
    }
    if (!ACCEPTED.includes(f.type)) {
      setFile(null);
      setBitmap(null);
      setPreviewUrl(null);
      setError("Please choose a JPEG or PNG image.");
      return;
    }
    setFile(f);
    const bmp = await loadOrientedBitmap(f);
    setBitmap(bmp);
    // display-oriented preview
    const c = document.createElement("canvas");
    c.width = bmp.width;
    c.height = bmp.height;
    c.getContext("2d").drawImage(bmp, 0, 0);
    c.toBlob((b) => setPreviewUrl(URL.createObjectURL(b)), f.type);
  }

  async function handleProcess() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setData(null);
    try {
      const form = new FormData();
      form.append("image", file);
      const res = await fetch("/api/detect", { method: "POST", body: form });
      const json = await res.json();
      if (!res.ok) throw new Error(json?.error || `Server error (${res.status})`);
      setData(json);
    } catch (err) {
      setError(err?.message || String(err));
    } finally {
      setLoading(false);
    }
  }

  const box = { border: "1px solid #ddd", borderRadius: 6, background: "#fff" };

  return (
    <main style={{ maxWidth: 820, margin: "0 auto", padding: "40px 20px" }}>
      <h1 style={{ marginBottom: 4 }}>CropDog</h1>
      <p style={{ marginTop: 0, color: "#666" }}>
        Phase 2 — subject detection debug view (no cropping yet).
      </p>

      <div
        style={{
          display: "flex",
          gap: 12,
          alignItems: "center",
          margin: "24px 0",
          flexWrap: "wrap",
        }}
      >
        <input type="file" accept="image/jpeg,image/png" onChange={handleFileChange} />
        <button
          onClick={handleProcess}
          disabled={!file || loading}
          style={{
            padding: "8px 20px",
            fontSize: 16,
            cursor: !file || loading ? "not-allowed" : "pointer",
            background: !file || loading ? "#ccc" : "#0070f3",
            color: "#fff",
            border: "none",
            borderRadius: 6,
          }}
        >
          {loading ? "Detecting…" : "Process"}
        </button>
      </div>

      {error && <p style={{ color: "#c00", fontWeight: 600 }}>Error: {error}</p>}

      {/* Plain uploaded image (before detection) */}
      {previewUrl && !data && (
        <section>
          <h2 style={{ fontSize: 16, marginBottom: 8 }}>Uploaded image</h2>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={previewUrl}
            alt="Uploaded"
            style={{ ...box, maxWidth: "100%", maxHeight: 520, height: "auto" }}
          />
        </section>
      )}

      {/* No subject */}
      {data && !data.detected && (
        <p
          style={{
            background: "#fff3cd",
            border: "1px solid #ffe69c",
            padding: "12px 16px",
            borderRadius: 6,
            fontWeight: 600,
          }}
        >
          No subject detected.
        </p>
      )}

      {/* Detection overlay */}
      {data?.detected && (
        <section>
          <h2 style={{ fontSize: 16, marginBottom: 8 }}>
            Detection — {data.numPeople}{" "}
            {data.numPeople === 1 ? "person" : "people"}
          </h2>
          <canvas
            ref={canvasRef}
            style={{ ...box, maxWidth: "100%", maxHeight: 620, height: "auto" }}
          />
          <Legend data={data} />
        </section>
      )}
    </main>
  );
}

function Swatch({ color, dashed, thick }) {
  return (
    <span
      style={{
        display: "inline-block",
        width: 22,
        height: 0,
        borderTop: `${thick ? 4 : 2}px ${dashed ? "dashed" : "solid"} ${color}`,
        verticalAlign: "middle",
        marginRight: 8,
      }}
    />
  );
}

function Legend({ data }) {
  const row = { display: "flex", alignItems: "center", marginBottom: 6, fontSize: 14 };
  return (
    <div
      style={{
        marginTop: 14,
        padding: "12px 16px",
        border: "1px solid #eee",
        borderRadius: 6,
        background: "#fafafa",
      }}
    >
      <div style={{ fontWeight: 600, marginBottom: 8 }}>Legend</div>
      {data.people.map((p, i) => (
        <div key={i} style={{ marginBottom: 10 }}>
          <div style={{ ...row, fontWeight: 600, color: PERSON_COLORS[i % 2] }}>
            <span
              style={{
                display: "inline-block",
                width: 12,
                height: 12,
                borderRadius: "50%",
                background: PERSON_COLORS[i % 2],
                marginRight: 8,
              }}
            />
            Person {i + 1} — confidence {p.confidence}, {p.landmarks.length} landmarks
          </div>
          <div style={{ ...row, marginLeft: 20 }}>
            <Swatch color={PERSON_COLORS[i % 2]} /> landmark box
          </div>
          <div style={{ ...row, marginLeft: 20 }}>
            <Swatch color={PERSON_COLORS[i % 2]} dashed /> segmentation box
          </div>
          <div style={{ ...row, marginLeft: 20 }}>
            <Swatch color={PERSON_COLORS[i % 2]} thick /> subject box (union + hair)
          </div>
        </div>
      ))}
      {data.combinedBox && (
        <div style={row}>
          <Swatch color={COMBINED_COLOR} dashed thick /> combined subject box (both
          people)
        </div>
      )}
    </div>
  );
}
