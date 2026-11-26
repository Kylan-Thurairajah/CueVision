import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { randomUUID } from "node:crypto";
import { readFile } from "node:fs/promises";
import { setTimeout as sleep } from "node:timers/promises";
import test from "node:test";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../../../", import.meta.url));
const baseUrl = process.env.CUEVISION_TEST_URL ?? "http://127.0.0.1:8081";
const sample = await readFile(new URL("../../../sample_pool_table1.jpg", import.meta.url));
const runRecovery = process.env.CUEVISION_TEST_RECOVERY === "1";

function request(path, init) {
  return fetch(new URL(path, baseUrl), { ...init, signal: AbortSignal.timeout(5000) });
}

function authorization(session) {
  return { Authorization: `Bearer ${session.token}` };
}

async function createSession(source = "simulated") {
  const response = await request("/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source }),
  });
  assert.equal(response.status, 201, await response.clone().text());
  const session = await response.json();
  assert.ok(session.id);
  assert.ok(session.token);
  return session;
}

function submit(session, key, { content = sample, felt = "auto" } = {}) {
  const multipart = new FormData();
  multipart.append("file", new Blob([content], { type: "image/jpeg" }), "table.jpg");
  return request(`/jobs?felt=${felt}`, {
    method: "POST",
    headers: { ...authorization(session), "Idempotency-Key": key },
    body: multipart,
  });
}

async function getJob(session, id) {
  const response = await request(`/jobs/${id}`, { headers: authorization(session) });
  assert.equal(response.status, 200, await response.clone().text());
  return response.json();
}

async function waitForJob(session, id, expectedStatus, timeoutMs = 30_000) {
  const deadline = performance.now() + timeoutMs;
  let job;
  do {
    job = await getJob(session, id);
    if (job.status === expectedStatus) return job;
    assert.notEqual(job.status, "failed", JSON.stringify(job));
    await sleep(200);
  } while (performance.now() < deadline);
  assert.fail(`Job ${id} did not reach ${expectedStatus}. Last state: ${job.status}`);
}

function compose(...args) {
  const result = spawnSync("docker", ["compose", ...args], {
    cwd: root,
    encoding: "utf8",
    timeout: 30_000,
  });
  assert.equal(result.status, 0, result.error?.message ?? result.stderr);
  return result.stdout.trim();
}

test("concurrent submissions complete with session-scoped results and artifacts", { timeout: 40_000 }, async () => {
  assert.equal((await request("/ready")).status, 200);
  const [session, otherSession] = await Promise.all([
    createSession(),
    createSession("upload"),
  ]);
  const key = randomUUID();
  const submissions = await Promise.all([
    submit(session, key),
    submit(session, key),
    submit(otherSession, key),
  ]);
  assert.deepEqual(submissions.map((response) => response.status).sort(), [200, 202, 202]);
  const [first, duplicate, other] = await Promise.all(submissions.map((response) => response.json()));
  assert.equal(first.id, duplicate.id);
  assert.notEqual(first.id, other.id);

  assert.equal((await submit(session, key, { felt: "green" })).status, 409);
  assert.equal((await submit(session, randomUUID(), { content: "not an image" })).status, 415);
  assert.equal((await request(`/jobs/${first.id}`)).status, 401);
  for (const path of [`/jobs/${first.id}`, `/jobs/${first.id}/original`, `/jobs/${first.id}/overlay`]) {
    assert.equal((await request(path, { headers: authorization(otherSession) })).status, 404);
  }

  const [job, otherJob] = await Promise.all([
    waitForJob(session, first.id, "completed"),
    waitForJob(otherSession, other.id, "completed"),
  ]);
  assert.equal(job.attempts, 1);
  assert.equal(otherJob.attempts, 1);
  assert.equal(job.result.outcome, "recommendation");
  assert.ok(job.result.advice);
  assert.equal(job.error, null);
  assert.ok(job.startedAt);
  assert.ok(job.completedAt);

  const history = await request("/jobs", { headers: authorization(session) });
  assert.equal(history.status, 200);
  assert.deepEqual((await history.json()).jobs.map(({ id }) => id), [first.id]);

  const artifacts = await Promise.all([
    request(`/jobs/${first.id}/original`, { headers: authorization(session) }),
    request(`/jobs/${first.id}/overlay`, { headers: authorization(session) }),
  ]);
  assert.deepEqual(artifacts.map((response) => response.status), [200, 200]);
  const [original, overlay] = await Promise.all(artifacts.map(async (response) => Buffer.from(await response.arrayBuffer())));
  assert.deepEqual(original, sample);
  assert.deepEqual(overlay.subarray(0, 8), Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]));
});

test("decoder-rejected uploads fail after one attempt", { timeout: 40_000 }, async () => {
  const session = await createSession();
  const response = await submit(session, randomUUID(), { content: Buffer.from([255, 216, 255, 224]) });
  assert.equal(response.status, 202, await response.clone().text());
  const { id } = await response.json();
  const job = await waitForJob(session, id, "failed");
  assert.equal(job.attempts, 1);
  assert.match(job.error, /rejected/i);
  assert.equal(job.overlayUrl, null);
});

test("queued jobs survive an unavailable Redis service", { skip: !runRecovery, timeout: 70_000 }, async () => {
  const session = await createSession();
  let id;
  compose("stop", "redis");
  try {
    const response = await submit(session, randomUUID());
    assert.equal(response.status, 202, await response.clone().text());
    ({ id } = await response.json());
    assert.equal((await getJob(session, id)).status, "queued");
  } finally {
    compose("start", "redis");
  }
  assert.equal((await waitForJob(session, id, "completed")).attempts, 1);
});

test("an unavailable analyzer exhausts the retry budget", { skip: !runRecovery, timeout: 90_000 }, async () => {
  const session = await createSession();
  compose("stop", "analysis");
  try {
    const response = await submit(session, randomUUID());
    assert.equal(response.status, 202, await response.clone().text());
    const { id } = await response.json();
    const job = await waitForJob(session, id, "failed", 70_000);
    assert.equal(job.attempts, 3);
    assert.ok(job.error);
    assert.equal(job.overlayUrl, null);
  } finally {
    compose("start", "--wait", "analysis");
  }
});

test("a published job is redispatched after stream loss", { skip: !runRecovery, timeout: 90_000 }, async () => {
  const session = await createSession();
  let id;
  compose("stop", "worker");
  try {
    const response = await submit(session, randomUUID());
    assert.equal(response.status, 202, await response.clone().text());
    ({ id } = await response.json());
    assert.match(id, /^[0-9a-f-]{36}$/);
    // Reproduce the window between successful publish and first consumption.
    compose("exec", "-T", "redis", "redis-cli", "XADD", "cuevision:analysis", "*", "jobId", id);
    compose("exec", "-T", "postgres", "psql", "-v", "ON_ERROR_STOP=1", "-U", "cuevision", "-d", "cuevision",
      "-c", `UPDATE jobs SET published_at = NOW() WHERE id = '${id}'`);
    compose("exec", "-T", "redis", "redis-cli", "DEL", "cuevision:analysis");
    assert.equal((await getJob(session, id)).status, "queued");
  } finally {
    compose("start", "worker");
  }
  assert.equal((await waitForJob(session, id, "completed")).attempts, 1);
});

test("an interrupted worker lease can be recovered", { skip: !runRecovery, timeout: 150_000 }, async () => {
  const session = await createSession();
  let id;
  compose("pause", "analysis");
  try {
    const response = await submit(session, randomUUID());
    assert.equal(response.status, 202, await response.clone().text());
    ({ id } = await response.json());
    assert.equal((await waitForJob(session, id, "processing", 15_000)).attempts, 1);
    compose("kill", "worker");
  } finally {
    compose("unpause", "analysis");
    const deadline = performance.now() + 20_000;
    while (compose("ps", "--format", "{{.Health}}", "analysis") !== "healthy") {
      assert.ok(performance.now() < deadline, "Analysis did not recover after unpause");
      await sleep(250);
    }
    compose("start", "worker");
  }
  assert.equal((await waitForJob(session, id, "completed", 110_000)).attempts, 2);
});

test("reclaiming a completed job's pending delivery keeps the committed result", { skip: !runRecovery, timeout: 65_000 }, async () => {
  const session = await createSession();
  const response = await submit(session, randomUUID());
  assert.equal(response.status, 202, await response.clone().text());
  const { id } = await response.json();
  const completed = await waitForJob(session, id, "completed");
  const redis = (...args) => compose("exec", "-T", "redis", "redis-cli", "--raw", ...args);
  const stream = "cuevision:analysis";
  const group = "analysis-workers";
  let delivery;
  compose("stop", "worker");
  try {
    delivery = redis("XADD", stream, "*", "jobId", id);
    const delivered = redis("XREADGROUP", "GROUP", group, "interrupted-test", "COUNT", "1", "STREAMS", stream, ">");
    assert.ok(delivered.includes(delivery), delivered);
    assert.equal(redis("XCLAIM", stream, group, "interrupted-test", "0", delivery, "IDLE", "91000", "JUSTID"), delivery);
  } finally {
    compose("start", "worker");
  }

  const deadline = performance.now() + 12_000;
  while (redis("XRANGE", stream, delivery, delivery) !== "") {
    assert.ok(performance.now() < deadline, "Worker did not acknowledge the reclaimed delivery");
    await sleep(200);
  }
  assert.deepEqual(await getJob(session, id), completed);
});
