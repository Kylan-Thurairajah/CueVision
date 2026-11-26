# CueVision job backend

Run `docker compose up --build --wait` from the repository root with a running Linux Docker engine. The TypeScript API is available at `http://127.0.0.1:8081`. Set `CUEVISION_PORT` in your shell to change the host port. PostgreSQL, Redis, and the Python service are internal to the Compose network. Local Compose uses trusted PostgreSQL access inside that network.

## Submit a sample

Run from the repository root in PowerShell.

```powershell
$base = 'http://127.0.0.1:8081'
$session = Invoke-RestMethod -Method Post -Uri "$base/sessions" -ContentType 'application/json' -Body '{"source":"simulated"}'
$headers = @{ Authorization = "Bearer $($session.token)" }
$key = [guid]::NewGuid().ToString()
$job = curl.exe --fail-with-body -sS -H "Authorization: $($headers.Authorization)" -H "Idempotency-Key: $key" -F "file=@sample_pool_table1.jpg" "$base/jobs?felt=auto" | ConvertFrom-Json
Invoke-RestMethod -Uri "$base/jobs/$($job.id)" -Headers $headers
```

Poll the last URL until it returns `completed` or `failed`. When available, download the overlay with `Invoke-WebRequest -Uri "$base/jobs/$($job.id)/overlay" -Headers $headers -OutFile overlay.png`.

| Endpoint | Behavior |
| --- | --- |
| `POST /sessions` | Create a session for `upload`, `simulated`, or `replay`. Keep the returned token for subsequent requests. |
| `POST /jobs?felt=auto` | Submit one JPEG or PNG in multipart field `file`, using Bearer authentication and an `Idempotency-Key`. Returns 202 with the job ID. Reusing a key with the same bytes and felt mode returns the same ID with 200. Reusing it for different input returns 409. |
| `GET /jobs` | List up to 50 recent jobs for the authenticated session. |
| `GET /jobs/:id` | Get status, timestamps, attempts, result metadata, and artifact URLs. |
| `GET /jobs/:id/original` and `/overlay` | Stream an artifact owned by the authenticated session. |
| `GET /health` and `/ready` | Check process liveness and PostgreSQL availability. Submissions can be accepted while Redis or the analyzer is unavailable. |

The API bounds multipart bytes and checks image signatures before enqueueing. Python checks encoded content and decoded dimensions. Corrupt or oversized decoded images can therefore be accepted as jobs and subsequently fail. An analyzer response with no recommended shot is a completed job with its own outcome.

## Delivery and recovery

PostgreSQL assigns each job a durable state and a unique session-scoped idempotency key. The queued job row also acts as the outbox. The relay polls every 500 ms and publishes job IDs to a Redis Streams consumer group. Queued jobs become eligible for republishing after 15 seconds, which restores work when Redis loses its state. Under sustained backlog this can produce duplicate stream entries. PostgreSQL claims keep those entries from triggering duplicate active attempts.

The worker gives each processing attempt a 90-second lease and allows 20 seconds for the analyzer call. Expired work becomes eligible again, up to three attempts. A lease token fences result and failure writes. Stream entries are acknowledged after durable state changes and removed from the single consumer group's stream. Redis recovery can redeliver work, so an analyzer invocation may repeat after a worker interruption even though the API exposes one job.

Originals and overlays live on a shared volume. The API and worker write files before publishing their references, and each processing attempt uses its own overlay filename. A process crash or uncertain database commit can leave orphaned files. Automatic reconciliation and retention limits are future work. Keep this stack local while those controls and external access policy are pending.

## Development and tests

The backend is one Node.js 22 package with API and worker entry points. Install with `npm ci` and build with `npm run build` from `apps/backend`. Both roles require `DATABASE_URL` and `DATA_DIR`. The worker also requires `REDIS_URL` and the full Python endpoint as `ANALYSIS_URL`. The API defaults to `127.0.0.1:8081` and accepts `HOST` and `PORT` overrides.

From the repository root, run `npm --prefix apps/backend run test:integration` against the running stack. Set `CUEVISION_TEST_URL` for a different API address. Set `CUEVISION_TEST_RECOVERY=1` only against a disposable Compose project to include tests that interrupt its services. Use the same Compose project name and port settings for startup and test execution. GitHub Actions builds and starts real PostgreSQL and Redis services for these checks.

Request logs include response duration, and worker logs include per-attempt processing duration. They omit session credentials, upload bytes, and result content.
