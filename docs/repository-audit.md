# CueVision repository audit

## Starting point

The checkout started clean on `main` at `52a56be` (September 2, 2025), matching the locally recorded `origin/main`. The audit used the local Git ref. It did not contact the project remote.

| Area | Current behavior and disposition |
| --- | --- |
| Vision | Preserve the felt mask, table rectification, Hough-circle detector, and brightness/saturation cue-ball heuristic in `pool_ai_full`. |
| Planning | Preserve the direct-pot ghost-ball heuristic. It uses fixed 1024×512 pocket geometry even when rectification fails and analysis falls back to original image coordinates. Verify that case before returning shot advice for it. |
| CLI | Loads one image, prints advice, and optionally writes an overlay. Keep it as a standalone diagnostic entry point. |
| API | FastAPI serves multipart `POST /analyze`, sharing the CLI's analysis logic. Its instructions omit `python-multipart` and pass `felt` as form data even though it is a query argument. Successful recommendations contain NumPy arrays and scalars that require explicit serialization. CPU-heavy analysis runs on the async handler thread. There are no application upload or decoded-size bounds, and the default bind is all interfaces. |
| Accelerator | TensorRT is an unimplemented detector. Defer GPU optimization, Jetson integration and multi-shot planning. |
| Messenger | Preserve the sender and receiver as legacy source. They use private chat selectors, saved browser cookies, and disabled Chromium sandbox flags. The receiver saves message-row screenshots and logs text. The sender dispatches a live message when run. Neither calls the analyzer. |
| Assets and tooling | Keep the 159,230-byte sample JPEG. Treat the empty `result.json` and tracked Python bytecode as generated artifacts. Root README is a placeholder, Node has no real test script, and there is no Python lockfile, automated suite, Compose, or CI at the starting commit. |

## Credential handling

`scripts/cookies.json` contained eight nonempty Facebook or Messenger cookie entries, including four associated with session state. Values were never printed or used. The introducing commit, `88d4268`, is reachable from both local refs. Treat those sessions as potentially exposed and revoke them through Meta account controls. The index removal is staged, and the retained local copy is ignored. Existing commits still contain the data. Keep the browser scripts out of the demo, tests, and container builds.

## Baseline procedure

Use the unchanged source at `52a56be`, an isolated Python environment, and the included JPEG. Record the versions resolved by the dependency install before comparing behavior later. The additional multipart dependency is necessary to get as far as registering the original upload endpoint.

```text
python -m pip install opencv-python==5.1.0.95 numpy==2.5.5 fastapi==0.165.0 uvicorn==0.54.0 python-multipart==0.0.34
$env:PYTHONIOENCODING='utf-8'
python -B -m pool_ai_full.cli sample_pool_table1.jpg --overlay overlay.png
python -B -m uvicorn pool_ai_full.server:app --host 127.0.0.1 --port 8080
curl.exe -i -F "file=@sample_pool_table1.jpg" "http://127.0.0.1:8080/analyze?felt=auto"
```

| Check | Observed in the default checkout environment |
| --- | --- |
| CLI startup with sample path | Python 3.13.1 stops at import with missing `cv2`. |
| API startup | Stops at import with missing `fastapi`; there is no listener to accept the sample upload. |
| Isolated setup | Initial virtualenv bootstrapping failed at `ensurepip`. A later install into an isolated package directory from PyPI succeeded. |
| JavaScript | Both legacy scripts pass `node --check` on Node 22.14.0. No live Messenger automation was run. |
| Containers | Docker CLI is installed. `docker info` cannot connect to the Docker Desktop Linux engine. |

With the versions above on Python 3.13.1, the unchanged CLI analyzed the sample. Redirected Windows CP1252 output could not encode the approximation symbol. With UTF-8 output, the CLI exited successfully, recommended potting ball #1 to the top-left pocket at roughly a 1° cut, and wrote a 213,289-byte PNG. The loopback HTTP server started and served OpenAPI with HTTP 200. Uploading the sample returned HTTP 500 because FastAPI could not serialize a NumPy float. Uploading the README returned HTTP 400. Both outcomes came from real multipart requests.

This legacy check establishes the starting behavior and environment failures. No latency benchmark was run. The first successful full-stack run will be the baseline for comparing end-to-end performance. Use a separate direct-analysis baseline for analyzing OpenCV execution time.
