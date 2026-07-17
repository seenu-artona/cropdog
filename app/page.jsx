"use client";

import { useState } from "react";
import { autoCrop } from "./lib/autocrop";

export default function Home() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null); // { url, detected, numPeople, width, height }
  const [error, setError] = useState(null);

  async function handleCrop() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const r = await autoCrop(file);
      const url = URL.createObjectURL(r.blob);
      setResult({
        url,
        detected: r.detected,
        numPeople: r.numPeople,
        width: r.width,
        height: r.height,
      });
    } catch (e) {
      setError(e?.message || String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <main style={{ maxWidth: 720, margin: "0 auto", padding: "40px 20px" }}>
      <h1 style={{ marginBottom: 4 }}>CropDog</h1>
      <p style={{ marginTop: 0, color: "#666" }}>
        Auto-crop MVP — one image in, one cropped image out. Runs entirely in
        your browser.
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
          accept="image/*"
          onChange={(e) => {
            setFile(e.target.files?.[0] || null);
            setResult(null);
            setError(null);
          }}
        />
        <button
          onClick={handleCrop}
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
          {loading ? "Cropping…" : "Crop"}
        </button>
      </div>

      {loading && (
        <p style={{ color: "#666" }}>
          Processing… the first run downloads the MediaPipe models (a few tens
          of MB), so it may take a few seconds.
        </p>
      )}

      {error && <p style={{ color: "#c00", fontWeight: 600 }}>Error: {error}</p>}

      {result && !result.detected && (
        <p
          style={{
            background: "#fff3cd",
            border: "1px solid #ffe69c",
            padding: "12px 16px",
            borderRadius: 6,
            fontWeight: 600,
          }}
        >
          No subject detected — showing the original image unmodified.
        </p>
      )}

      {result && result.detected && (
        <p style={{ color: "#666" }}>
          Detected {result.numPeople}{" "}
          {result.numPeople === 1 ? "person" : "people"} — cropped to the padded
          subject box ({result.width}×{result.height}).
        </p>
      )}

      {result && (
        <div style={{ marginTop: 16 }}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={result.url}
            alt="Cropped result"
            style={{
              maxWidth: "100%",
              height: "auto",
              border: "1px solid #ddd",
              borderRadius: 6,
              background: "#fff",
            }}
          />
          <div style={{ marginTop: 12 }}>
            <a
              href={result.url}
              download="cropped.png"
              style={{
                display: "inline-block",
                padding: "8px 20px",
                background: "#111",
                color: "#fff",
                borderRadius: 6,
                textDecoration: "none",
              }}
            >
              Download cropped image
            </a>
          </div>
        </div>
      )}
    </main>
  );
}
