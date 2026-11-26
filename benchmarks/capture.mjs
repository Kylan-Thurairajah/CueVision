import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { arch, cpus, platform, release, totalmem } from "node:os";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
const requireBackend = createRequire(new URL("../apps/backend/package.json", import.meta.url));
const playwright = process.env.CUEVISION_PLAYWRIGHT_MODULE
  ? createRequire(import.meta.url)(resolve(process.env.CUEVISION_PLAYWRIGHT_MODULE))
  : requireBackend("playwright-core");
const captureUrl = new URL("/capture", process.env.CUEVISION_WORKFLOW_URL ?? "http://127.0.0.1:8081").href;
const executablePath = process.env.CUEVISION_BROWSER_PATH;
const fixture = await readFile(new URL("../sample_pool_table1.jpg", import.meta.url));

function count(name, fallback, minimum) {
  const value = Number(process.env[name] ?? fallback);
  if (!Number.isInteger(value) || value < minimum) throw new Error(`${name} must be an integer >= ${minimum}`);
  return value;
}

const warmups = count("CUEVISION_BENCH_WARMUPS", 5, 0);
const iterations = count("CUEVISION_BENCH_ITERATIONS", 30, 1);
const output = resolve(root, process.argv[2] ?? "benchmarks/capture-baseline.local.json");

function summarize(values) {
  const sorted = values.toSorted((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return {
    median: Number((sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2).toFixed(1)),
    p95: Number(sorted[Math.ceil(0.95 * sorted.length) - 1].toFixed(1)),
  };
}

async function runCapture(page) {
  await page.goto(captureUrl);
  await page.evaluate(() => {
    const status = document.querySelector("#status");
    const results = document.querySelector("#jobs");
    const afterPaint = () => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));

    document.querySelector("#simulate").addEventListener("click", () => {
      performance.clearResourceTimings();
      const started = performance.now();
      let advicePaint;
      const observer = new MutationObserver(() => {
        if (!advicePaint && results.querySelector(".advice")?.textContent?.trim()) {
          advicePaint = afterPaint().then(() => performance.now() - started);
        }
        if (status.textContent !== "Capture finished.") return;
        observer.disconnect();
        void (async () => {
          const original = results.querySelector('img[alt^="Original capture"]');
          const overlay = results.querySelector('img[alt^="Analysis overlay"]');
          if (!original || !overlay || !advicePaint) {
            throw new Error("Expected advice, original, and overlay for the sample capture.");
          }
          await Promise.all([original.decode(), overlay.decode()]);
          await afterPaint();
          const clickToVisualMs = performance.now() - started;
          const imageWasLaidOut = image => image.getBoundingClientRect().width > 0;
          if (!imageWasLaidOut(original) || !imageWasLaidOut(overlay)) {
            throw new Error("Completed image previews were not laid out.");
          }
          const resources = performance.getEntriesByType("resource");
          const duration = path => resources.find(entry => new URL(entry.name).pathname === path)?.duration ?? null;
          const overlayPath = resources.map(entry => new URL(entry.name).pathname)
            .find(path => /^\/jobs\/[^/]+\/overlay$/.test(path));
          window.captureMeasurement = {
            clickToAdviceMs: await advicePaint,
            clickToVisualMs,
            sampleFetchMs: duration("/capture/sample.jpg"),
            submitRequestMs: duration("/jobs"),
            overlayFetchMs: duration(overlayPath),
            advice: results.querySelector(".advice").textContent,
          };
        })().catch(error => { window.captureMeasurement = { error: error.message }; });
      });
      observer.observe(status, { childList: true, subtree: true, characterData: true });
      observer.observe(results, { childList: true, subtree: true, characterData: true });
    }, { capture: true, once: true });
  });

  const completed = page.waitForResponse(async response => {
    if (!response.ok() || response.request().method() !== "GET"
      || !/^\/jobs\/[^/]+$/.test(new URL(response.url()).pathname)) return false;
    return (await response.json()).status === "completed";
  }, { timeout: 15000 });
  await page.locator("#simulate").click();
  const [jobResponse, measurement] = await Promise.all([
    completed,
    page.waitForFunction(() => window.captureMeasurement, null, { timeout: 15000 })
      .then(handle => handle.jsonValue()),
  ]);
  if (measurement.error) throw new Error(measurement.error);
  const job = await jobResponse.json();
  if (job.result?.outcome !== "recommendation" || !job.result.best
    || !measurement.advice.includes(`Pot ball #${job.result.best.object_idx} `)) {
    throw new Error("Expected the visible recommendation to identify the backend's target ball.");
  }
  const { advice, ...timings } = measurement;
  return {
    ...timings,
    queuedMs: Date.parse(job.startedAt) - Date.parse(job.createdAt),
    workerMs: Date.parse(job.completedAt) - Date.parse(job.startedAt),
  };
}

const browser = await playwright.chromium.launch({ executablePath });
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const runs = [];
  const total = warmups + iterations;
  for (let index = 0; index < total; index += 1) {
    const timing = await runCapture(page);
    if (index >= warmups) runs.push(timing);
  }
  const summary = {};
  for (const metric of Object.keys(runs[0])) {
    const values = runs.map(run => run[metric]);
    if (values.every(Number.isFinite)) summary[metric] = summarize(values);
  }
  const report = {
    recordedAt: new Date().toISOString(),
    client: {
      os: `${platform()} ${release()} ${arch()}`,
      cpu: cpus()[0]?.model,
      logicalProcessors: cpus().length,
      memoryBytes: totalmem(),
      node: process.version,
      browser: browser.version(),
      viewport: "1280×900",
    },
    workload: {
      captureUrl,
      file: "sample_pool_table1.jpg",
      sha256: createHash("sha256").update(fixture).digest("hex"),
      bytes: fixture.length,
      warmups,
      iterations,
      concurrency: 1,
      freshSessionPerCapture: true,
    },
    summaryMs: summary,
    samplesMs: runs,
  };
  await writeFile(output, `${JSON.stringify(report, null, 2)}\n`);
  console.table(summary);
  console.log(`Saved ${output}`);
} finally {
  await browser.close();
}
