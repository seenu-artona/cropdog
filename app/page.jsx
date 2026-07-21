"use client";

import { useState } from "react";

const ACCEPTED = ["image/jpeg", "image/png"];

// Return an object URL for the file with EXIF orientation baked into the pixels,
// so the displayed original is in display-oriented coordinates (not relying on
// the browser's implicit orientation handling). Falls back to a plain object
// URL if createImageBitmap can't honor orientation.
async function toDisplayOrientedURL(file) {
  try {
    const bitmap = await createImageBitmap(file, {
      imageOrientation: "from-image",
    });
    const canvas = document.createElement("canvas");
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    canvas.getContext("2d").drawImage(bitmap, 0, 0);
    const blob = await new Promise((res) =>
      canvas.toBlob((b) => res(b), file.type || "image/png")
    );
    return URL.createObjectURL(blob);
  } catch {
    return URL.createObjectURL(file);
  }
}

export default function Home() {
  const [file, setFile] = useState(null);
  const [originalUrl, setOriginalUrl] = useState(null);
  const [processedUrl, setProcessedUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleFileChange(e) {
    const f = e.target.files?.[0] || null;
    setProcessedUrl(null);
    setError(null);
    if (!f) {
      setFile(null);
      setOriginalUrl(null);
      return;
    }
    if (!ACCEPTED.includes(f.type)) {
      setFile(null);
      setOriginalUrl(null);
      setError("Please choose a JPEG or PNG image.");
      return;
    }
    setFile(f);
    setOriginalUrl(await toDisplayOrientedURL(f));
  }

  async function handleProcess() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setProcessedUrl(null);
    try {
      const form = new FormData();
      form.append("image", file);
      const res = await fetch("/api/process", { method: "POST", body: form });
      if (!res.ok) {
        let msg = `Server error (${res.status})`;
        try {
          const j = await res.json();
          if (j?.error) msg = j.error;
        } catch {}
        throw new Error(msg);
      }
      const blob = await res.blob();
      setProcessedUrl(URL.createObjectURL(blob));
    } catch (err) {
      setError(err?.message || String(err));
    } finally {
      setLoading(false);
    }
  }

  const imgStyle = {
    maxWidth: "100%",
    maxHeight: 480,
    height: "auto",
    border: "1px solid #ddd",
    borderRadius: 6,
    background: "#fff",
    imageOrientation: "from-image",
  };

  return (
    <main style={{ maxWidth: 720, margin: "0 auto", padding: "40px 20px" }}>
      <h1 style={{ marginBottom: 4 }}>CropDog</h1>
      <p style={{ marginTop: 0, color: "#666" }}>
        Phase 1 — upload an image, send it to the server, get it back.
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
        <input
          type="file"
          accept="image/jpeg,image/png"
          onChange={handleFileChange}
        />
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
          {loading ? "Processing…" : "Process"}
        </button>
      </div>

      {error && <p style={{ color: "#c00", fontWeight: 600 }}>Error: {error}</p>}

      {originalUrl && (
        <section style={{ marginTop: 16 }}>
          <h2 style={{ fontSize: 16, marginBottom: 8 }}>Uploaded image</h2>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={originalUrl} alt="Uploaded" style={imgStyle} />
        </section>
      )}

      {processedUrl && (
        <section style={{ marginTop: 24 }}>
          <h2 style={{ fontSize: 16, marginBottom: 8 }}>Returned from server</h2>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={processedUrl} alt="Returned from server" style={imgStyle} />
        </section>
      )}
    </main>
  );
}
