import { spawn } from "node:child_process";
import { mkdtemp, writeFile, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";

export const runtime = "nodejs";
// Cropping can take a while on first run (model download); avoid static opt.
export const dynamic = "force-dynamic";

// Prefer the project venv python; fall back to system python3.
function pythonBin() {
  const venvPy = path.join(process.cwd(), ".venv", "bin", "python");
  return process.env.CROPDOG_PYTHON || venvPy;
}

function runPython(scriptPath, inputPath, outputPath) {
  return new Promise((resolve, reject) => {
    const proc = spawn(pythonBin(), [scriptPath, inputPath, outputPath], {
      cwd: process.cwd(),
    });

    let stdout = "";
    let stderr = "";
    proc.stdout.on("data", (d) => (stdout += d.toString()));
    proc.stderr.on("data", (d) => (stderr += d.toString()));

    proc.on("error", (err) => reject(err));
    proc.on("close", (code) => {
      if (code !== 0) {
        reject(
          new Error(
            `crop.py exited with code ${code}. stderr:\n${stderr}`
          )
        );
        return;
      }
      resolve({ stdout, stderr });
    });
  });
}

export async function POST(req) {
  let workDir;
  try {
    const formData = await req.formData();
    const image = formData.get("image");

    if (!image || typeof image.arrayBuffer !== "function") {
      return Response.json({ error: "No image uploaded" }, { status: 400 });
    }

    const bytes = Buffer.from(await image.arrayBuffer());

    workDir = await mkdtemp(path.join(tmpdir(), "cropdog-"));
    const inputPath = path.join(workDir, "input");
    const outputPath = path.join(workDir, "output.png");
    await writeFile(inputPath, bytes);

    const scriptPath = path.join(process.cwd(), "scripts", "crop.py");
    const { stdout } = await runPython(scriptPath, inputPath, outputPath);

    // crop.py prints a single JSON line describing the result.
    const lastLine = stdout.trim().split("\n").filter(Boolean).pop() || "{}";
    let meta;
    try {
      meta = JSON.parse(lastLine);
    } catch {
      throw new Error(`Could not parse crop.py output: ${stdout}`);
    }

    const outBytes = await readFile(outputPath);
    const imageDataUrl = `data:image/png;base64,${outBytes.toString("base64")}`;

    return Response.json({
      detected: !!meta.detected,
      numPeople: meta.numPeople || 0,
      box: meta.box || null,
      imageDataUrl,
    });
  } catch (err) {
    return Response.json(
      { error: err.message || String(err) },
      { status: 500 }
    );
  } finally {
    if (workDir) {
      rm(workDir, { recursive: true, force: true }).catch(() => {});
    }
  }
}
