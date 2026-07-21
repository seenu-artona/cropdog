import sharp from "sharp";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Phase 1: accept an uploaded JPEG/PNG and return it back unchanged — except we
// bake in EXIF orientation first (sharp.rotate() with no args auto-rotates from
// the EXIF Orientation tag and strips it), so everything downstream operates in
// display-oriented coordinates.
export async function POST(req) {
  try {
    const formData = await req.formData();
    const image = formData.get("image");

    if (!image || typeof image.arrayBuffer !== "function") {
      return Response.json({ error: "No image uploaded" }, { status: 400 });
    }

    const inputBuffer = Buffer.from(await image.arrayBuffer());

    const pipeline = sharp(inputBuffer).rotate(); // apply EXIF orientation
    const format = (await pipeline.metadata()).format;

    let outBuffer;
    let contentType;
    if (format === "png") {
      outBuffer = await pipeline.png().toBuffer();
      contentType = "image/png";
    } else if (format === "jpeg" || format === "jpg") {
      outBuffer = await pipeline.jpeg().toBuffer();
      contentType = "image/jpeg";
    } else {
      return Response.json(
        { error: `Unsupported image format: ${format}. Use JPEG or PNG.` },
        { status: 415 }
      );
    }

    return new Response(outBuffer, {
      status: 200,
      headers: {
        "Content-Type": contentType,
        "Cache-Control": "no-store",
      },
    });
  } catch (err) {
    return Response.json(
      { error: err?.message || String(err) },
      { status: 500 }
    );
  }
}
