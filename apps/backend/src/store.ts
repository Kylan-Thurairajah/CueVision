import { readdir, readFile } from "node:fs/promises";
import type { Pool } from "pg";

export type Felt = "auto" | "green" | "blue";

export interface Job {
  id: string;
  session_id: string;
  original: string;
  felt: Felt;
  status: "queued" | "processing" | "completed" | "failed";
  attempts: number;
  created_at: Date;
  updated_at: Date;
  started_at: Date | null;
  attempt_started_at: Date | null;
  completed_at: Date | null;
  available_at: Date;
  result: unknown;
  overlay: string | null;
  error: string | null;
}

interface Submission {
  id: string;
  sessionId: string;
  key: string;
  digest: string;
  original: string;
  felt: Felt;
}

const jobColumns = `
  id, session_id, original, felt, status, attempts, created_at, updated_at,
  started_at, attempt_started_at, completed_at, available_at, result, overlay, error
`;
const maxAttempts = 3;
const migrationsDirectory = new URL("../migrations/", import.meta.url);

export class Store {
  constructor(readonly pool: Pool) {}

  async migrate(): Promise<void> {
    const names = (await readdir(migrationsDirectory))
      .filter((name) => /^\d+_[a-z0-9_]+\.sql$/.test(name))
      .sort();
    if (names.length === 0) {
      throw new Error("No database migrations found");
    }
    const client = await this.pool.connect();
    try {
      await client.query("BEGIN");
      // Serialize migration startup when the API and worker start together.
      await client.query("SELECT pg_advisory_xact_lock(134987211)");
      await client.query(`
        CREATE TABLE IF NOT EXISTS schema_migrations (
          name text PRIMARY KEY,
          applied_at timestamptz NOT NULL DEFAULT now()
        )
      `);
      const applied = await client.query<{ name: string }>("SELECT name FROM schema_migrations");
      const appliedNames = new Set(applied.rows.map(({ name }) => name));
      for (const name of names) {
        if (appliedNames.has(name)) {
          continue;
        }
        await client.query(await readFile(new URL(name, migrationsDirectory), "utf8"));
        await client.query("INSERT INTO schema_migrations (name) VALUES ($1)", [name]);
      }
      await client.query("COMMIT");
    } catch (error) {
      await client.query("ROLLBACK");
      throw error;
    } finally {
      client.release();
    }
  }

  async createSession(id: string, tokenHash: string, source: string): Promise<void> {
    await this.pool.query("INSERT INTO sessions (id, token_hash, source) VALUES ($1, $2, $3)", [
      id, tokenHash, source,
    ]);
  }

  async session(tokenHash: string): Promise<{ id: string; source: string } | undefined> {
    const { rows } = await this.pool.query<{ id: string; source: string }>(
      "SELECT id, source FROM sessions WHERE token_hash = $1", [tokenHash],
    );
    return rows[0];
  }

  async submit(input: Submission): Promise<{ id: string; created: boolean; conflict: boolean }> {
    const { id, sessionId, key, digest, original, felt } = input;
    const inserted = await this.pool.query(`
      INSERT INTO jobs (id, session_id, idempotency_key, digest, original, felt)
      VALUES ($1, $2, $3, $4, $5, $6)
      ON CONFLICT (session_id, idempotency_key) DO NOTHING
    `, [id, sessionId, key, digest, original, felt]);
    if (inserted.rowCount === 1) {
      return { id, created: true, conflict: false };
    }
    const { rows } = await this.pool.query<{ id: string; digest: string; felt: Felt }>(`
      SELECT id, digest, felt FROM jobs
      WHERE session_id = $1 AND idempotency_key = $2
    `, [sessionId, key]);
    const existing = rows[0];
    if (!existing) {
      throw new Error("Idempotent submission disappeared");
    }
    return {
      id: existing.id,
      created: false,
      conflict: existing.digest !== digest || existing.felt !== felt,
    };
  }

  async getJob(sessionId: string, id: string): Promise<Job | undefined> {
    const { rows } = await this.pool.query<Job>(`
      SELECT ${jobColumns} FROM jobs WHERE session_id = $1 AND id = $2
    `, [sessionId, id]);
    return rows[0];
  }

  async listJobs(sessionId: string): Promise<Job[]> {
    const { rows } = await this.pool.query<Job>(`
      SELECT ${jobColumns} FROM jobs WHERE session_id = $1
      ORDER BY created_at DESC, id DESC LIMIT 50
    `, [sessionId]);
    return rows;
  }

  async pending(): Promise<string[]> {
    const { rows } = await this.pool.query<{ id: string }>(`
      SELECT id FROM jobs
      WHERE status = 'queued' AND available_at <= now()
        AND (published_at IS NULL OR published_at <= now() - interval '15 seconds')
      ORDER BY available_at, id LIMIT 100
    `);
    return rows.map(({ id }) => id);
  }

  async markPublished(id: string): Promise<void> {
    await this.pool.query(`
      UPDATE jobs SET published_at = now() WHERE id = $1 AND status = 'queued'
    `, [id]);
  }

  async claim(id: string, lease: string): Promise<Job | undefined> {
    const { rows } = await this.pool.query<Job>(`
      UPDATE jobs SET status = 'processing', attempts = attempts + 1,
        started_at = COALESCE(started_at, now()), attempt_started_at = now(),
        updated_at = now(), lease_token = $2, lease_expires_at = now() + interval '90 seconds'
      WHERE id = $1 AND status = 'queued' AND available_at <= now() AND attempts < $3
      RETURNING ${jobColumns}
    `, [id, lease, maxAttempts]);
    return rows[0];
  }

  async succeed(id: string, lease: string, result: unknown, overlay: string): Promise<boolean> {
    const updated = await this.pool.query(`
      UPDATE jobs SET status = 'completed', completed_at = now(), updated_at = now(),
        result = $3::jsonb, overlay = $4, lease_token = NULL, lease_expires_at = NULL
      WHERE id = $1 AND lease_token = $2 AND lease_expires_at > now()
    `, [id, lease, JSON.stringify(result), overlay]);
    return updated.rowCount === 1;
  }

  async fail(id: string, lease: string, message: string, retryable: boolean): Promise<boolean> {
    const updated = await this.pool.query(`
      UPDATE jobs SET
        status = CASE WHEN $4 AND attempts < $5 THEN 'queued' ELSE 'failed' END,
        available_at = now() + attempts * interval '1 second',
        error = CASE WHEN $4 AND attempts < $5 THEN NULL ELSE left($3, 500) END,
        completed_at = CASE WHEN $4 AND attempts < $5 THEN NULL ELSE now() END,
        updated_at = now(), published_at = NULL, lease_token = NULL, lease_expires_at = NULL
      WHERE id = $1 AND lease_token = $2 AND lease_expires_at > now()
    `, [id, lease, message, retryable, maxAttempts]);
    return updated.rowCount === 1;
  }

  async recover(): Promise<number> {
    const updated = await this.pool.query(`
      UPDATE jobs SET
        status = CASE WHEN attempts < $1 THEN 'queued' ELSE 'failed' END,
        available_at = now(),
        error = CASE WHEN attempts < $1 THEN NULL ELSE 'Analysis worker stopped responding.' END,
        completed_at = CASE WHEN attempts < $1 THEN NULL ELSE now() END,
        updated_at = now(), published_at = NULL, lease_token = NULL, lease_expires_at = NULL
      WHERE status = 'processing' AND lease_expires_at <= now()
    `, [maxAttempts]);
    return updated.rowCount ?? 0;
  }
}
