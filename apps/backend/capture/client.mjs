const sources = new Set(["upload", "simulated", "replay"]);
const feltModes = new Set(["auto", "green", "blue"]);
const maxImageBytes = 8 * 1024 * 1024;
const png = [137, 80, 78, 71, 13, 10, 26, 10];

export class CaptureError extends Error {
  constructor(message, retryable = false) {
    super(message);
    this.retryable = retryable;
  }
}

export function createCaptureClient({
  baseUrl = globalThis.location?.origin,
  fetchImpl = fetch,
  timeoutMs = 10_000,
} = {}) {
  const api = new URL(baseUrl);
  const sessions = new Map();

  async function request(path, options) {
    let response;
    try {
      response = await fetchImpl(new URL(path, api), {
        ...options, signal: AbortSignal.timeout(timeoutMs), redirect: "error",
      });
    } catch {
      throw new CaptureError("Connection interrupted. Reconnect and retry this capture.", true);
    }
    if (!response.ok) {
      const body = await response.json().catch(() => null);
      throw new CaptureError(body?.error?.message ?? `Request failed (HTTP ${response.status}).`,
        response.status === 429 || response.status >= 500);
    }
    return response;
  }

  async function json(response) {
    try {
      return await response.json();
    } catch {
      throw new CaptureError("Could not read the server response. Retry this capture.", true);
    }
  }

  function session(source) {
    if (!sessions.has(source)) {
      const pending = request("/sessions", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ source }),
      }).then(json).catch(error => {
        sessions.delete(source);
        throw error;
      });
      sessions.set(source, pending);
    }
    return sessions.get(source);
  }

  function prepare(source, file, felt = "auto") {
    if (!sources.has(source) || !feltModes.has(felt)) throw new CaptureError("Invalid capture source or felt mode.");
    if (!(file instanceof Blob) || !file.size) throw new CaptureError("Select a nonempty JPEG or PNG image.");
    if (file.size > maxImageBytes) throw new CaptureError("Image exceeds 8 MiB.");
    return Object.freeze({ source, file, felt, key: crypto.randomUUID() });
  }

  async function submit(capture) {
    const signature = new Uint8Array(await capture.file.slice(0, 8).arrayBuffer());
    const isPng = png.every((byte, index) => byte === signature[index]);
    const isJpeg = signature[0] === 255 && signature[1] === 216 && signature[2] === 255;
    if (!isPng && !isJpeg) throw new CaptureError("Use a JPEG or PNG image.");
    const { id: sessionId, token } = await session(capture.source);
    const form = new FormData();
    form.set("file", capture.file, isPng ? "capture.png" : "capture.jpg");
    const response = await request(`/jobs?felt=${capture.felt}`, {
      method: "POST", body: form,
      headers: { Authorization: `Bearer ${token}`, "Idempotency-Key": capture.key },
    });
    return { ...(await json(response)), status: response.status, sessionId };
  }

  async function authorized(capture, path) {
    const { token } = await session(capture.source);
    return request(path, { headers: { Authorization: `Bearer ${token}` } });
  }

  return {
    prepare,
    submit,
    job: async (capture, id) => json(await authorized(capture, `/jobs/${encodeURIComponent(id)}`)),
    artifact: async (capture, path) => {
      if (!/^\/jobs\/[0-9a-f-]+\/(original|overlay)$/.test(path)) {
        throw new CaptureError("Invalid artifact URL.");
      }
      const response = await authorized(capture, path);
      try {
        return await response.blob();
      } catch {
        throw new CaptureError("Connection interrupted while loading the image. Reconnect and retry.", true);
      }
    },
  };
}
