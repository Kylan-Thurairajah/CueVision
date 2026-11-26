# Grade CueVision milestones 1–3

Run the commands from the repository root, where `compose.yaml` and `sample_pool_table1.jpg` are located. The examples use Windows PowerShell. You need Python 3.13, Node.js 22, Docker Desktop running Linux containers, and a browser.

## 1. Python analysis

Create an isolated environment, run the Python tests, and analyze the included photo.

```powershell
py -3.13 -m venv .venv
$python = '.\.venv\Scripts\python.exe'
& $python -m pip install -r requirements-dev.txt
& $python -m pytest tests/analysis
$overlay = Join-Path $env:TEMP ("cuevision-{0}.png" -f [guid]::NewGuid())
& $python -m pool_ai_full.cli sample_pool_table1.jpg --overlay $overlay
Invoke-Item $overlay
```

**Pass:** All 17 tests pass. The CLI prints a shot recommendation and opens a 1024×512 top-down image with the aim and pot paths annotated. The API tests also check JSON output, upload limits, invalid images, and explicit outcomes when a shot cannot be recommended.

To inspect the HTTP contract yourself, start the Python API in one terminal.

```powershell
.\.venv\Scripts\python.exe -m pool_ai_full.server
```

Then run this in a second terminal from the repository root.

```powershell
$analysis = 'http://127.0.0.1:8080'
Invoke-RestMethod "$analysis/ready"
$result = curl.exe --fail-with-body -sS -F "file=@sample_pool_table1.jpg" "$analysis/analyze?felt=auto" | ConvertFrom-Json
$result | Select-Object outcome, advice, rectified, num_balls, overlay_width, overlay_height
curl.exe -sS -i -F "file=@README.md" "$analysis/analyze"
```

**Pass:** Readiness returns `ok`. The sample returns `recommendation` and `rectified=True`. The text upload returns HTTP 415 with `unsupported_image`. Press Ctrl+C in the server terminal when finished.

## 2. Backend jobs and recovery

Start an isolated Compose project. These settings use port 18083 and keep the test stack separate from the default project. If that port is occupied, change `CUEVISION_PORT` before starting.

```powershell
$env:COMPOSE_PROJECT_NAME = "cuevision-grade-$PID"
$env:CUEVISION_PORT = '18083'
$env:CUEVISION_TEST_URL = "http://127.0.0.1:$($env:CUEVISION_PORT)"
docker compose up --build --wait
Invoke-RestMethod "$env:CUEVISION_TEST_URL/ready"
npm.cmd --prefix apps/backend run test:integration
```

**Pass:** Readiness returns `ok`. Two integration tests pass. They submit concurrent jobs, verify that identical requests share one job ID, reject an idempotency conflict, check session isolation, retrieve both images, and verify a corrupt JPEG fails after one attempt. The five service-interruption tests are skipped in this run.

For the full recovery check, run the following in the same terminal with the isolated project still active. It intentionally stops services in that project and takes roughly two minutes because a worker lease actually expires.

```powershell
$env:CUEVISION_TEST_RECOVERY = '1'
npm.cmd --prefix apps/backend run test:integration
Remove-Item Env:CUEVISION_TEST_RECOVERY
```

**Pass:** All seven tests pass. Queued work survives both a Redis outage and stream loss. Temporary analysis failures retry to the limit, interrupted jobs finish on another attempt, and a redelivered completed job preserves its result.

## 3. Hardware-free capture

With the Compose stack still running, open the capture page.

```powershell
Start-Process "$env:CUEVISION_TEST_URL/capture"
$env:CUEVISION_CAPTURE_TEST_URL = $env:CUEVISION_TEST_URL
npm.cmd --prefix apps/backend run test:capture
```

**Pass:** The capture tests pass, including the live checks selected by `CUEVISION_CAPTURE_TEST_URL`. Without that setting, the command runs the standalone tests.

Then try these browser checks.

| Check | What to do | Pass condition |
| --- | --- | --- |
| Simulated wearable | Click **Capture sample**. | A real backend job ID appears. The sample completes with a recommendation and the page displays the overlay. |
| Local image | Select `sample_pool_table1.jpg` from the repository and click **Upload photo**. | The upload completes with its own job ID. |
| Frame replay | Take copies of two JPEG or PNG files, give them names starting with `01` and `02`, select both, and click **Replay frames**. You can use the sample photo and the PNG from Milestone 1. | Frames are submitted in filename order and each gets its own job ID. |
| Invalid file | Choose **All files** in the picker if needed and select a non-image file, such as `README.md`. | The page reports the invalid file and remains usable for another capture. |
| Disconnection | Select a local photo. In browser DevTools, change the Network throttling option to **Offline**. Click **Upload photo**, restore the normal connection, then click **Retry**. | The page explains the interruption. Retry resumes the pending submission and produces one job ID. If the server had accepted the upload before the interruption, retry reuses that job. |

Keep the capture tab open during retry. It stores the session and pending capture in memory until the page reloads. This milestone grades browser upload, simulated capture, and frame replay against the real backend. The [capture guide](capture.md#official-meta-wearable-path) records the native Meta SDK findings separately.

When you finish, stop the grading project. Add `--volumes` only if you want to delete this disposable project's saved jobs and images.

```powershell
docker compose down
```
