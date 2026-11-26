# CueVision analysis service

The CLI and API share the original OpenCV felt segmentation, table rectification, Hough ball detection, and ghost-ball shot planner. Both run locally on a CPU with no wearable hardware or model downloads.

## Setup

From the repository root, create a virtual environment with Python 3.13. The runtime uses headless OpenCV and does not require GUI libraries.

```sh
python -m venv .venv
```

On Windows PowerShell, activate the environment and install the pinned dependencies.

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell execution policy blocks activation, run `.\.venv\Scripts\python.exe` directly in place of `python`.

On Linux, use your shell to activate and install.

```sh
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## CLI

```sh
python -m pool_ai_full.cli sample_pool_table1.jpg --overlay overlay.png
python -m pool_ai_full.cli sample_pool_table1.jpg --felt auto
```

`--felt` accepts `auto`, `green`, or `blue`. `--overlay` is optional and writes an annotated PNG when supplied.

## HTTP API

```sh
python -m pool_ai_full.server
curl -F "file=@sample_pool_table1.jpg" "http://127.0.0.1:8080/analyze?felt=auto"
```

On Windows PowerShell, use `curl.exe` in the example above. The service listens on `127.0.0.1:8080` by default. Set `HOST` or `PORT` in the process environment to override the defaults.

| Method and path | Purpose |
| --- | --- |
| `GET /health` | Process liveness |
| `GET /ready` | Readiness, including PNG encoder availability |
| `POST /analyze` | Upload an image as multipart field `file` |

`/analyze` accepts JPEG and PNG images up to 8 MiB and 16 million decoded pixels. The overall multipart request has a separate cap of 8 MiB plus 64 KiB for multipart overhead. The optional query parameter `felt` defaults to `auto` and takes the same values as the CLI. The optional `detector` query parameter accepts `hough` and defaults to it.

A successful analysis returns HTTP 200 with `advice`, `rectified`, `num_balls`, `cue_idx`, `best`, `overlay_png_base64`, `overlay_width`, `overlay_height`, and `outcome`. The outcome distinguishes `recommendation`, `no_usable_table`, `no_cue_ball`, and `no_clear_pot`. With a recommendation, `best` includes the target ball index, pocket, ghost point, score, and cut angle. The positions refer to pixels in the overlay. An analysis without a recommendation has a null `best`. If table rectification fails, the service skips ball detection and returns an unannotated PNG at the decoded image dimensions.

Request errors use an `error` object with `code` and `message`.

| HTTP status | Error code | Cause |
| --- | --- | --- |
| 400 | `bad_image` | The image could not be read |
| 400 | `invalid_upload` | Malformed multipart request |
| 413 | `image_too_large` | Request, file, or pixel limit exceeded |
| 415 | `unsupported_image` | Format other than JPEG or PNG |
| 422 | `validation_error` | Missing upload or invalid query parameter |
| 500 | `analysis_failed` | Unexpected processing error |
| 503 | `not_ready` | PNG encoder unavailable |

## Tests

```sh
python -m pip install -r requirements-dev.txt
python -m pytest tests/analysis
```
