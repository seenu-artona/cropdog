import sharp from "sharp";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
// Allow long waits: Render's free tier cold-starts (~30-60s) when it has been
// idle > 15 min. Requires a Vercel plan that permits extended function
// duration (Pro/Team); on Hobby this is capped lower.
export const maxDuration = 120;

// Render free-tier cold starts can take ~30-60s, so wait generously.
const FETCH_TIMEOUT_MS = 90000;

export async function POST(req) {
  try {
    const serviceUrl = process.env.DETECTION_SERVICE_URL;
    if (!serviceUrl) {
      return Response.json(
        { error: "Detection service is not configured (DETECTION_SERVICE_URL missing)." },
        { status: 500 }
      );
    }

    const formData = await req.formData();
    const image = formData.get("image");
    if (!image || typeof image.arrayBuffer !== "function") {
      return Response.json({ error: "No image uploaded" }, { status: 400 });
    }

    // EXIF-normalize server-side (display-oriented) before sending on, so the
    // coordinates the service returns match what the client displays.
    const inputBuffer = Buffer.from(await image.arrayBuffer());
    const normalized = await sharp(inputBuffer).rotate().png().toBuffer();

    const endpoint = `${serviceUrl.replace(/\/+$/, "")}/detect`;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);

    let res;
    try {
      const forward = new FormData();
      forward.append(
        "image",
        new Blob([normalized], { type: "image/png" }),
        "image.png"
      );
      res = await fetch(endpoint, {
        method: "POST",
        body: forward,
        signal: controller.signal,
      });
    } catch (e) {
      if (e?.name === "AbortError") {
        return Response.json(
          {
            error:
              "Detection service timed out — it may be waking up. Please try again in a moment.",
          },
          { status: 504 }
        );
      }
      return Response.json(
        { error: "Detection service unavailable. Please try again shortly." },
        { status: 503 }
      );
    } finally {
      clearTimeout(timer);
    }

    if (!res.ok) {
      let detail = "";
      try {
        const body = await res.json();
        detail = body?.detail || body?.error || "";
      } catch {}
      return Response.json(
        {
          error: `Detection service error (${res.status})${detail ? `: ${detail}` : ""}`,
        },
        { status: 502 }
      );
    }

    // Pass the service's JSON through unchanged — identical response shape.
    const payload = await res.json();
    return Response.json(payload);
  } catch (err) {
    return Response.json({ error: err?.message || String(err) }, { status: 500 });
  }
}
