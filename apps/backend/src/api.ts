import { File } from "node:buffer";
import { createHash, randomBytes, randomUUID } from "node:crypto";
import { mkdir, open, rename, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import Fastify, { LogController, type FastifyRequest } from "fastify";

import { type Job, type Store } from "./store.js";

const MAX_IMAGE_BYTES = 8 * 1024 * 1024;
const MAX_MULTIPART_BYTES = MAX_IMAGE_BYTES + 64 * 1024;
const PNG_SIGNATURE = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
const UUID = /^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$/;
const ARTIFACT = /^[0-9a-f-]+\.(?:jpg|png)$/;

class ApiError extends Error {
  constructor(readonly status: number, readonly code: string, message: string) {
    super(message);
  }
}

function sha256(value: Buffer | string) {
  return createHash("sha256").update(value).digest("hex");
}

function describeJob(job: Job) {
  return {
    id: job.id,
    status: job.status,
    attempts: job.attempts,
    createdAt: job.created_at,
    updatedAt: job.updated_at,
    startedAt: job.started_at,
    attemptStartedAt: job.attempt_started_at,
    completedAt: job.completed_at,
    result: job.result,
    error: job.error,
    originalUrl: `/jobs/${job.id}/original`,
    overlayUrl: job.overlay ? `/jobs/${job.id}/overlay` : null,
  };
}

function imageExtension(bytes: Buffer) {
  if (bytes.subarray(0, PNG_SIGNATURE.length).equals(PNG_SIGNATURE)) return "png";
  if (bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) return "jpg";
  throw new ApiError(415, "unsupported_image", "Upload a JPEG or PNG image.");
}

async function uploadedImage(request: FastifyRequest<{ Body: Buffer }>) {
  const contentType = request.headers["content-type"];
  if (!contentType || !/^multipart\/form-data(?:\s*;|$)/i.test(contentType)) {
    throw new ApiError(415, "unsupported_media_type", "Use a multipart form with an image field.");
  }

  let form: FormData;
  try {
    form = await new Response(new Uint8Array(request.body), {
      headers: { "content-type": contentType },
    }).formData();
  } catch {
    throw new ApiError(400, "invalid_upload", "Could not parse the multipart upload.");
  }

  const parts = [...form];
  if (parts.length !== 1 || parts[0][0] !== "file" || !(parts[0][1] instanceof File)) {
    throw new ApiError(400, "invalid_upload", "Provide exactly one file in the file field.");
  }
  const file = parts[0][1];
  if (file.size === 0) throw new ApiError(400, "invalid_upload", "The image is empty.");
  if (file.size > MAX_IMAGE_BYTES) throw new ApiError(413, "image_too_large", "Image exceeds 8 MiB.");
  return Buffer.from(await file.arrayBuffer());
}

export function buildApi({ store, dataDir }: { store: Store; dataDir: string }) {
  const uploadSessions = new WeakMap<FastifyRequest, { id: string }>();
  const app = Fastify({
    logger: { level: process.env.LOG_LEVEL ?? "info" },
    logController: new LogController({ disableRequestLogging: true }),
    requestIdHeader: false,
    requestTimeout: 30_000,
    bodyLimit: 64 * 1024,
  });
  app.addContentTypeParser("multipart/form-data", {
    parseAs: "buffer",
    bodyLimit: MAX_MULTIPART_BYTES,
  }, (_request, payload, done) => done(null, payload));

  app.addHook("onRequest", async (_request, reply) => {
    reply.header("Cache-Control", "no-store");
    reply.header("X-Content-Type-Options", "nosniff");
  });
  app.addHook("onResponse", async (request, reply) => {
    request.log.info({
      event: "request_finished",
      method: request.method,
      route: request.routeOptions.url,
      status: reply.statusCode,
      durationMs: reply.elapsedTime,
    });
  });

  app.setNotFoundHandler((_request, reply) => reply.code(404).send({
    error: { code: "not_found", message: "Route not found." },
  }));
  app.setErrorHandler((error, request, reply) => {
    if (error instanceof ApiError) {
      if (error.status === 401) reply.header("WWW-Authenticate", "Bearer");
      void reply.code(error.status).send({ error: { code: error.code, message: error.message } });
      return;
    }
    const status = typeof error === "object" && error !== null && "statusCode" in error
      ? error.statusCode
      : undefined;
    if (typeof status === "number" && status >= 400 && status < 500) {
      const code = status === 413 ? "image_too_large" : "invalid_request";
      const message = status === 413 ? "Request body exceeds the upload limit." : "Invalid request.";
      void reply.code(status).send({ error: { code, message } });
      return;
    }
    request.log.error({ event: "request_failed", errorName: error instanceof Error ? error.name : "unknown" });
    void reply.code(500).send({ error: { code: "internal_error", message: "Request could not be completed." } });
  });

  async function authenticated(request: FastifyRequest) {
    const bearer = /^Bearer ([\w-]{43})$/i.exec(request.headers.authorization ?? "");
    const session = bearer && await store.session(sha256(bearer[1]));
    if (!session) throw new ApiError(401, "unauthorized", "Supply a valid session bearer token.");
    return session;
  }

  async function ownedJob(request: FastifyRequest<{ Params: { id: string } }>) {
    const session = await authenticated(request);
    const { id } = request.params;
    const job = UUID.test(id) ? await store.getJob(session.id, id) : undefined;
    if (!job) throw new ApiError(404, "job_not_found", "Job not found.");
    return job;
  }

  app.get("/health", async () => ({ status: "ok" }));
  app.get("/ready", async () => {
    try {
      await store.pool.query("SELECT 1");
      return { status: "ok" };
    } catch {
      throw new ApiError(503, "not_ready", "Job storage is unavailable.");
    }
  });

  app.post<{ Body: { source?: unknown } }>("/sessions", async (request, reply) => {
    const source = request.body?.source;
    if (source !== "upload" && source !== "simulated" && source !== "replay") {
      throw new ApiError(400, "invalid_source", "Source must be upload, simulated, or replay.");
    }
    const id = randomUUID();
    const token = randomBytes(32).toString("base64url");
    await store.createSession(id, sha256(token), source);
    return reply.code(201).send({ id, token, source });
  });

  app.post<{ Body: Buffer; Querystring: { felt?: unknown } }>("/jobs", {
    bodyLimit: MAX_MULTIPART_BYTES,
    onRequest: async (request) => {
      uploadSessions.set(request, await authenticated(request));
    },
  }, async (request, reply) => {
    const session = uploadSessions.get(request);
    if (!session) throw new Error("Missing authenticated upload session");
    const key = request.headers["idempotency-key"];
    if (typeof key !== "string" || !/^[!-~]{1,128}$/.test(key)) {
      throw new ApiError(400, "invalid_idempotency_key", "Supply an Idempotency-Key of 1–128 visible ASCII characters.");
    }
    const felt = request.query.felt ?? "auto";
    if (felt !== "auto" && felt !== "green" && felt !== "blue") {
      throw new ApiError(400, "invalid_felt", "Felt must be auto, green, or blue.");
    }
    const bytes = await uploadedImage(request);
    const original = `${randomUUID()}.${imageExtension(bytes)}`;
    const staging = join(dataDir, `${original}.part`);
    const destination = join(dataDir, original);
    await mkdir(dataDir, { recursive: true });
    try {
      await writeFile(staging, bytes, { flag: "wx", mode: 0o600 });
      await rename(staging, destination);
    } catch (error) {
      await rm(staging, { force: true }).catch(() => undefined);
      throw error;
    }

    // A failed database connection may leave an ambiguous commit. Keep the
    // uploaded file unless the store positively confirms that it did not use it.
    const job = await store.submit({
      id: randomUUID(), sessionId: session.id, key, digest: sha256(bytes), original, felt,
    });
    if (!job.created) {
      await rm(destination, { force: true }).catch(() => {
        request.log.warn({ event: "unused_upload_cleanup_failed" });
      });
    }
    if (job.conflict) {
      throw new ApiError(409, "idempotency_conflict", "This key is already associated with different input.");
    }
    request.log.info({ event: job.created ? "job_queued" : "job_replayed", jobId: job.id, sessionId: session.id });
    return reply.code(job.created ? 202 : 200).header("Location", `/jobs/${job.id}`).send({ id: job.id });
  });

  app.get("/jobs", async (request) => {
    const session = await authenticated(request);
    return { jobs: (await store.listJobs(session.id)).map(describeJob) };
  });
  app.get<{ Params: { id: string } }>("/jobs/:id", async (request) => describeJob(await ownedJob(request)));
  for (const kind of ["original", "overlay"] as const) {
    app.get<{ Params: { id: string } }>(`/jobs/:id/${kind}`, async (request, reply) => {
      const job = await ownedJob(request);
      const name = job[kind];
      if (!name || !ARTIFACT.test(name)) throw new ApiError(404, "artifact_not_found", "Image not found.");
      const file = await open(join(dataDir, name), "r").catch((error: NodeJS.ErrnoException) => {
        if (error.code === "ENOENT") throw new ApiError(404, "artifact_not_found", "Image not found.");
        throw error;
      });
      return reply.type(name.endsWith(".png") ? "image/png" : "image/jpeg").send(file.createReadStream());
    });
  }

  return app;
}
