"use client";

import { useState } from "react";

export default function Home() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null); // { imageDataUrl, detected, numPeople }
  const [error, setError] = useState(null);

  async function handleCrop() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const form = new FormData();
      form.append("image", file);

      const res = await fetch("/api/crop", { method: "POST", body: form });
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || "Crop failed");
      }
      setResult(data);
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <main
      style={{
        maxWidth: 720,
        margin: "0 auto",
        padding: "40px 20px",
      }}
    >
      <h1 style={{ marginBottom: 4 }}>CropDog</h1>
      <p style={{ marginTop: 0, color: "#666" }}>
        Auto-crop MVP — one image in, one cropped image out.
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

      {error && (
        <p style={{ color: "#c00", fontWeight: 600 }}>Error: {error}</p>
      )}

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
          {result.numPeople === 1 ? "person" : "people"} — cropped to the
          padded subject box.
        </p>
      )}

      {result && (
        <div style={{ marginTop: 16 }}>
          <img
            src={result.imageDataUrl}
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
              href={result.imageDataUrl}
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
