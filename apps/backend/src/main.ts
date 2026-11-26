import { mkdir } from "node:fs/promises";
import { resolve } from "node:path";
import pg from "pg";
import { z } from "zod";
import { buildApi } from "./api.js";
import { Store } from "./store.js";
import { runWorker } from "./worker.js";

const host = z.string().min(1).default("127.0.0.1");
const port = z.coerce.number().int().min(1).max(65535).default(8081);
const url = (schemes: string[]) => z.url().refine(value => schemes.includes(new URL(value).protocol));
const sharedConfig = z.object({
  DATABASE_URL: url(["postgres:", "postgresql:"]),
  DATA_DIR: z.string().min(1),
});
const apiConfig = sharedConfig.extend({ HOST: host, PORT: port });
const workerConfig = sharedConfig.extend({
  REDIS_URL: url(["redis:", "rediss:"]),
  ANALYSIS_URL: url(["http:", "https:"]),
});

async function main() {
  const role = process.argv[2];
  if (role !== "api" && role !== "worker") throw new Error("Expected api or worker role");

  const config = role === "api"
    ? { role: "api" as const, ...apiConfig.parse(process.env) }
    : { role: "worker" as const, ...workerConfig.parse(process.env) };
  const dataDir = resolve(config.DATA_DIR);
  const pool = new pg.Pool({
    connectionString: config.DATABASE_URL,
    connectionTimeoutMillis: 5000,
    statement_timeout: 5000,
  });
  pool.on("error", () => console.error(JSON.stringify({ event: "database_connection_error" })));
  const store = new Store(pool);
  const shutdown = new AbortController();
  const stop = () => shutdown.abort();
  process.once("SIGINT", stop);
  process.once("SIGTERM", stop);
  try {
    await mkdir(dataDir, { recursive: true });
    await store.migrate();
    if (config.role === "worker") {
      await runWorker({ store, dataDir, redisUrl: config.REDIS_URL, analysisUrl: config.ANALYSIS_URL }, shutdown.signal);
    } else {
      const app = buildApi({ store, dataDir });
      try {
        if (!shutdown.signal.aborted) {
          await app.listen({ host: config.HOST, port: config.PORT });
          if (!shutdown.signal.aborted) {
            await new Promise<void>(resolveStop => shutdown.signal.addEventListener("abort", () => resolveStop(), { once: true }));
          }
        }
      } finally {
        await app.close();
      }
    }
  } finally {
    process.removeListener("SIGINT", stop);
    process.removeListener("SIGTERM", stop);
    await pool.end();
  }
}

main().catch((error: unknown) => {
  const details = error instanceof z.ZodError
    ? { invalidSettings: error.issues.map(issue => issue.path.join(".")) }
    : { errorType: error instanceof Error ? error.name : "UnknownError",
        code: error instanceof Error && "code" in error ? error.code : undefined };
  console.error(JSON.stringify({ event: "startup_failed", ...details }));
  process.exitCode = 1;
});
