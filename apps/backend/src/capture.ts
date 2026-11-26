import type { FastifyInstance } from "fastify";
import { createReadStream } from "node:fs";

const assets = [
  ["/capture", "../capture/index.html", "text/html; charset=utf-8"],
  ["/capture/client.mjs", "../capture/client.mjs", "text/javascript; charset=utf-8"],
  ["/capture/page.mjs", "../capture/page.mjs", "text/javascript; charset=utf-8"],
  ["/capture/page.css", "../capture/page.css", "text/css; charset=utf-8"],
  ["/capture/sample.jpg", "../../../sample_pool_table1.jpg", "image/jpeg"],
] as const;

export function registerCapture(app: FastifyInstance) {
  for (const [path, file, type] of assets) {
    app.get(path, (_request, reply) => reply.type(type).header("Content-Security-Policy",
      "default-src 'self'; img-src 'self' blob:; object-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
    ).send(createReadStream(new URL(file, import.meta.url))));
  }
}
