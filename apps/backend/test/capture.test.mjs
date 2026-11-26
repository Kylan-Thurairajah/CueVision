import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { setTimeout as sleep } from "node:timers/promises";
import test from "node:test";

import { createCaptureClient } from "../capture/client.mjs";

const mockBaseUrl = "http://cuevision.test";
const liveBaseUrl = process.env.CUEVISION_CAPTURE_TEST_URL;
const jpeg = new File([Uint8Array.of(0xff, 0xd8, 0xff, 0xe0)], "table.jpg", { type: "image/jpeg" });

test("submission of the same prepared capture retries with the original session and idempotency key", async () => {
  let sessions = 0;
  const attempts = [];
  const client = createCaptureClient({
    baseUrl: mockBaseUrl,
    async fetchImpl(url, init) {
      const requestUrl = new URL(url);
      if (requestUrl.pathname === "/sessions") {
        sessions += 1;
        assert.equal(init.method, "POST");
        assert.deepEqual(JSON.parse(init.body), { source: "simulated" });
        return Response.json({ id: "session-1", token: "token-1" }, { status: 201 });
      }
      assert.equal(requestUrl.pathname, "/jobs");
      assert.equal(requestUrl.searchParams.get("felt"), "green");
      assert.equal(init.method, "POST");
      assert.deepEqual(Array.from(init.body.keys()), ["file"]);
      const file = init.body.get("file");
      assert.equal(file.name, "capture.jpg");
      assert.deepEqual(await file.bytes(), await jpeg.bytes());
      const headers = new Headers(init.headers);
      attempts.push({
        token: headers.get("authorization"),
        key: headers.get("idempotency-key"),
      });
      if (attempts.length === 1) throw new TypeError("Connection lost after backend accepted upload");
      return Response.json({ id: "job-1" }, { status: 200 });
    },
  });
  const capture = client.prepare("simulated", jpeg, "green");

  await assert.rejects(client.submit(capture), (error) => error.retryable === true);
  const retry = await client.submit(capture);

  assert.equal(sessions, 1);
  assert.deepEqual(attempts, Array(2).fill({ token: "Bearer token-1", key: capture.key }));
  assert.deepEqual(retry, { id: "job-1", status: 200, sessionId: "session-1" });
});

test("capture sources use separate sessions while captures of a source share one session", async () => {
  const sessions = [];
  const credentials = new Map();
  const client = createCaptureClient({
    baseUrl: mockBaseUrl,
    async fetchImpl(url, init) {
      if (new URL(url).pathname === "/sessions") {
        const { source } = JSON.parse(init.body);
        sessions.push(source);
        return Response.json({ id: `session-${source}`, token: `token-${source}` }, { status: 201 });
      }
      assert.equal(new URL(url).pathname, "/jobs");
      const headers = new Headers(init.headers);
      const key = headers.get("idempotency-key");
      assert.ok(key);
      credentials.set(key, headers.get("authorization"));
      return Response.json({ id: `job-${key}` }, { status: 202 });
    },
  });
  const captures = ["upload", "simulated", "replay", "upload"].map((source) => client.prepare(source, jpeg));
  const receipts = await Promise.all(captures.map((capture) => client.submit(capture)));

  assert.deepEqual(sessions.sort(), ["replay", "simulated", "upload"]);
  assert.equal(credentials.size, 4);
  for (const [index, capture] of captures.entries()) {
    assert.equal(credentials.get(capture.key), `Bearer token-${capture.source}`);
    assert.equal(receipts[index].sessionId, `session-${capture.source}`);
    assert.equal(receipts[index].status, 202);
  }
});

test("invalid captures fail locally before any session or job request", async () => {
  const client = createCaptureClient({
    baseUrl: mockBaseUrl,
    fetchImpl() {
      assert.fail("Invalid captures must not reach the backend");
    },
  });
  const empty = new File([], "empty.png", { type: "image/png" });
  const disguisedGif = new File(["GIF89a"], "table.jpg", { type: "image/jpeg" });
  const oversized = new File([new Uint8Array(8 * 1024 * 1024 + 1)], "large.png", { type: "image/png" });

  assert.throws(() => client.prepare("other", jpeg));
  assert.throws(() => client.prepare("upload", jpeg, "red"));
  assert.throws(() => client.prepare("upload", empty));
  assert.throws(() => client.prepare("upload", oversized));
  await assert.rejects(client.submit(client.prepare("upload", disguisedGif)), /JPEG or PNG/);
});

test("external artifact links are rejected before a bearer token can be sent", async () => {
  const client = createCaptureClient({
    baseUrl: mockBaseUrl,
    fetchImpl() {
      assert.fail("External URLs must not reach fetch");
    },
  });
  const capture = client.prepare("upload", jpeg);
  await assert.rejects(client.artifact(capture, "//external.test/jobs/1234/original"), /Invalid artifact URL/);
  await assert.rejects(client.artifact(capture, "https://external.test/jobs/1234/overlay"), /Invalid artifact URL/);
});

test("an interrupted artifact body can be fetched again with the same session", async () => {
  const receipts = [];
  const client = createCaptureClient({
    baseUrl: mockBaseUrl,
    async fetchImpl(url, init) {
      if (url.pathname === "/sessions") return Response.json({ id: "session-1", token: "token-1" });
      receipts.push(new Headers(init.headers).get("authorization"));
      if (receipts.length === 1) {
        return new Response(new ReadableStream({
          start(controller) { controller.error(new TypeError("Connection lost during image download")); },
        }));
      }
      return new Response("image bytes");
    },
  });
  const capture = client.prepare("upload", jpeg);
  await assert.rejects(client.artifact(capture, "/jobs/1234/overlay"), { retryable: true });
  assert.equal(await (await client.artifact(capture, "/jobs/1234/overlay")).text(), "image bytes");
  assert.deepEqual(receipts, ["Bearer token-1", "Bearer token-1"]);
});

test("the served capture page and replay client use the live job pipeline", {
  skip: !liveBaseUrl,
  timeout: 45_000,
}, async () => {
  const get = (path) => fetch(new URL(path, liveBaseUrl), { signal: AbortSignal.timeout(5000) });
  const [page, clientScript, browserScript, sampleResponse] = await Promise.all([
    get("/capture"),
    get("/capture/client.mjs"),
    get("/capture/page.mjs"),
    get("/capture/sample.jpg"),
  ]);
  assert.equal(page.status, 200);
  assert.match(page.headers.get("content-type") ?? "", /^text\/html\b/);
  assert.ok(await page.text());
  for (const script of [clientScript, browserScript]) {
    assert.equal(script.status, 200);
    assert.match(script.headers.get("content-type") ?? "", /^text\/javascript\b/);
    assert.ok(await script.text());
  }

  const sample = await readFile(new URL("../../../sample_pool_table1.jpg", import.meta.url));
  assert.equal(sampleResponse.status, 200);
  assert.deepEqual(Buffer.from(await sampleResponse.arrayBuffer()), sample);
  const file = new File([sample], "table.jpg", { type: "image/jpeg" });
  const client = createCaptureClient({ baseUrl: liveBaseUrl, timeoutMs: 5000 });
  const upload = client.prepare("upload", file);
  const simulated = client.prepare("simulated", file);
  const replay = client.prepare("replay", file);
  const receipts = await Promise.all([upload, simulated, replay].map((capture) => client.submit(capture)));
  const playback = receipts[2];
  assert.deepEqual(receipts.map(({ status }) => status), [202, 202, 202]);
  assert.equal(new Set(receipts.map(({ sessionId }) => sessionId)).size, 3);
  assert.equal(new Set(receipts.map(({ id }) => id)).size, 3);
  assert.deepEqual(await client.submit(replay), { ...playback, status: 200 });

  const deadline = performance.now() + 30_000;
  let job;
  do {
    job = await client.job(replay, playback.id);
    if (job.status === "completed") break;
    assert.notEqual(job.status, "failed", JSON.stringify(job));
    await sleep(200);
  } while (performance.now() < deadline);

  assert.equal(job.status, "completed");
  assert.equal(job.id, playback.id);
  assert.equal(job.result.outcome, "recommendation");
  const [original, overlay] = await Promise.all([
    client.artifact(replay, job.originalUrl),
    client.artifact(replay, job.overlayUrl),
  ]);
  assert.deepEqual(Buffer.from(await original.arrayBuffer()), sample);
  assert.equal(overlay.type, "image/png");
  assert.ok(overlay.size > 0);
  await assert.rejects(client.job(simulated, playback.id));
});
