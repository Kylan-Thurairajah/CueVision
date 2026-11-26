import { randomUUID } from "node:crypto";
import { readFile, rename, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { setTimeout as wait } from "node:timers/promises";
import { createClient } from "redis";
import { z } from "zod";
import type { Store } from "./store.js";

const STREAM = "cuevision:analysis";
const GROUP = "analysis-workers";
const ANALYSIS_TIMEOUT_MS = 20_000;
const RECLAIM_AFTER_MS = 90_000;
const MAX_RESPONSE_BYTES = 72 * 1024 * 1024;
const pngMagic = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);

const point = z.tuple([z.number(), z.number()]);
const analysisResult = z.object({
  advice: z.string(),
  outcome: z.enum(["recommendation", "no_usable_table", "no_cue_ball", "no_clear_pot"]),
  rectified: z.boolean(),
  num_balls: z.number().int().nonnegative(),
  cue_idx: z.number().int().nonnegative().nullable(),
  best: z.object({
    score: z.number(),
    object_idx: z.number().int().nonnegative(),
    pocket: point,
    ghost: point,
    cut_deg: z.number(),
  }).nullable(),
  overlay_width: z.number().int().positive(),
  overlay_height: z.number().int().positive(),
  overlay_png_base64: z.string().min(1),
});
const jobId = z.uuid();
const streamBatches = z.array(z.object({
  messages: z.array(z.object({ id: z.string(), message: z.record(z.string(), z.string()) })),
}));

interface WorkerOptions {
  store: Store;
  dataDir: string;
  redisUrl: string;
  analysisUrl: string;
}

class AnalysisFailure extends Error {
  constructor(message: string, readonly retryable: boolean) {
    super(message);
  }
}

function log(event: string, details: Record<string, unknown> = {}) {
  console.log(JSON.stringify({ event, ...details }));
}

async function readAnalysisResponse(response: Response) {
  const reader = response.body?.getReader();
  if (!reader) throw new AnalysisFailure("Analysis returned an empty response.", true);
  const chunks: Uint8Array[] = [];
  let bytes = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      bytes += value.length;
      if (bytes > MAX_RESPONSE_BYTES) {
        await reader.cancel();
        throw new AnalysisFailure("Analysis response exceeded the size limit.", false);
      }
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }
  return analysisResult.parse(JSON.parse(Buffer.concat(chunks, bytes).toString("utf8")));
}

async function processJob({ store, dataDir, analysisUrl }: WorkerOptions, id: string) {
  const lease = randomUUID();
  const job = await store.claim(id, lease);
  if (!job) return;

  const started = performance.now();
  log("job_started", { job_id: id, attempt: job.attempts });
  const overlay = `${id}-${lease}.png`;
  const target = join(dataDir, overlay);
  const temporary = `${target}.tmp`;
  let metadata: Omit<z.infer<typeof analysisResult>, "overlay_png_base64">;
  try {
    const original = await readFile(join(dataDir, job.original));
    const form = new FormData();
    form.set("file", new Blob([new Uint8Array(original)]), "capture");
    const url = new URL(analysisUrl);
    url.searchParams.set("felt", job.felt);
    const response = await fetch(url, {
      method: "POST", body: form, signal: AbortSignal.timeout(ANALYSIS_TIMEOUT_MS), redirect: "error",
    });
    if (!response.ok) {
      await response.body?.cancel();
      const invalid = [400, 413, 415, 422].includes(response.status);
      throw new AnalysisFailure(
        invalid ? "The image was rejected by the analysis service." : "The analysis service could not process the image.",
        !invalid,
      );
    }
    const { overlay_png_base64: encoded, ...result } = await readAnalysisResponse(response);
    const png = Buffer.from(encoded, "base64");
    if (!png.subarray(0, pngMagic.length).equals(pngMagic)) {
      throw new AnalysisFailure("The analysis overlay is invalid.", true);
    }
    metadata = result;
    await writeFile(temporary, png, { flag: "wx", mode: 0o600 });
    await rename(temporary, target);
  } catch (error) {
    await rm(temporary, { force: true }).catch(() => log("artifact_cleanup_failed", { job_id: id }));
    const message = error instanceof AnalysisFailure ? error.message : "Analysis or artifact storage failed.";
    const retryable = !(error instanceof AnalysisFailure) || error.retryable;
    const applied = await store.fail(id, lease, message, retryable);
    log("job_attempt_failed", { job_id: id, attempt: job.attempts, retryable, applied, duration_ms: Math.round(performance.now() - started) });
    return;
  }

  // A database error can occur after commit. Leave the artifact in place when
  // commit visibility is unknown, so the persisted result can still be read.
  const applied = await store.succeed(id, lease, metadata, overlay);
  if (!applied) await rm(target, { force: true }).catch(() => log("artifact_cleanup_failed", { job_id: id }));
  log(applied ? "job_completed" : "job_lease_lost", {
    job_id: id, attempt: job.attempts, duration_ms: Math.round(performance.now() - started),
  });
}

export async function runWorker(options: WorkerOptions, signal: AbortSignal) {
  const redisOptions = {
    url: options.redisUrl,
    disableOfflineQueue: true,
    socket: { connectTimeout: 5000, reconnectStrategy: (retries: number) => Math.min(250 * (retries + 1), 2000) },
  };
  const relay = createClient(redisOptions);
  const reader = createClient(redisOptions);
  relay.on("error", () => log("redis_connection_error", { role: "relay" }));
  reader.on("error", () => log("redis_connection_error", { role: "reader" }));
  const consumer = randomUUID();
  const stop = () => {
    if (relay.isOpen) relay.destroy();
    if (reader.isOpen) reader.destroy();
  };
  signal.addEventListener("abort", stop, { once: true });
  try {
    if (signal.aborted) return;
    await Promise.all([relay.connect(), reader.connect()]);

    const relayLoop = async () => {
      while (!signal.aborted) {
        try {
          const recovered = await options.store.recover();
          if (recovered) log("jobs_recovered", { count: recovered });
          for (const id of await options.store.pending()) {
            if (signal.aborted) break;
            await relay.xAdd(STREAM, "*", { jobId: id });
            await options.store.markPublished(id);
          }
        } catch {
          if (!signal.aborted) log("job_relay_failed");
        }
        if (!signal.aborted) await wait(500);
      }
    };

    const establishGroup = async () => {
      try {
        await reader.xGroupCreate(STREAM, GROUP, "0", { MKSTREAM: true });
      } catch (error) {
        if (!(error instanceof Error) || !error.message.startsWith("BUSYGROUP")) throw error;
      }
    };
    const consume = async (message: { id: string; message: Record<string, string> }) => {
      const id = jobId.safeParse(message.message.jobId);
      if (id.success) await processJob(options, id.data);
      if (!signal.aborted) {
        await reader.multi().xAck(STREAM, GROUP, message.id).xDel(STREAM, message.id).exec();
      }
    };
    const consumeLoop = async () => {
      let groupReady = false;
      let reclaimCursor = "0-0";
      let nextSweep = 0;
      while (!signal.aborted) {
        try {
          if (!groupReady) {
            await establishGroup();
            groupReady = true;
          }
          if (Date.now() >= nextSweep) {
            const reclaimed = await reader.xAutoClaim(STREAM, GROUP, consumer, RECLAIM_AFTER_MS, reclaimCursor, { COUNT: 1 });
            reclaimCursor = reclaimed.nextId;
            nextSweep = reclaimCursor === "0-0" ? Date.now() + 5000 : 0;
            for (const message of reclaimed.messages) {
              if (message && !signal.aborted) await consume(message);
            }
          }
          if (signal.aborted) break;
          const batches = await reader.xReadGroup(GROUP, consumer, { key: STREAM, id: ">" }, { BLOCK: 1000, COUNT: 1 });
          for (const batch of streamBatches.parse(batches ?? [])) {
            for (const message of batch.messages) {
              if (message && !signal.aborted) await consume(message);
            }
          }
        } catch {
          groupReady = false;
          reclaimCursor = "0-0";
          nextSweep = 0;
          if (!signal.aborted) {
            log("job_consumer_failed");
            await wait(1000);
          }
        }
      }
    };

    log("worker_started", { consumer });
    await Promise.all([relayLoop(), consumeLoop()]);
  } catch (error) {
    if (!signal.aborted) throw error;
  } finally {
    signal.removeEventListener("abort", stop);
    stop();
    log("worker_stopped", { consumer });
  }
}
