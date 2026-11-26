# Simulated capture latency

Use the same Compose configuration, browser and workload for later comparisons. This first baseline is a sequential, warm-stack smoke workload using the included table photo, one new simulated session per capture, and speech disabled. It does not measure physical wearable latency, audio onset, throughput under concurrency, or generalization to other photos.

## Run

Start CueVision with `docker compose up --build --wait`. Set `CUEVISION_WORKFLOW_URL` if the backend uses a host port other than 8081. From the repository root in PowerShell, point the runner at an installed Chromium browser.

```powershell
npm.cmd --prefix apps/backend ci
$env:CUEVISION_BROWSER_PATH = 'C:\Program Files\Google\Chrome\Application\chrome.exe'
node benchmarks/capture.mjs
```

The script writes `benchmarks/capture-baseline.local.json` by default. Pass a different output path as its first argument to save a comparison run. Defaults are five warmups and 30 measured iterations. Override them with `CUEVISION_BENCH_WARMUPS` and `CUEVISION_BENCH_ITERATIONS`. Set `CUEVISION_PLAYWRIGHT_MODULE` to use a separately installed Playwright Core module.

## Measurement

- **Click to advice** begins at the simulated shutter click. It ends after the recommendation text appears in the workspace and a browser rendering opportunity has passed.
- **Click to visual result** ends after both image previews decode and have been laid out. Every recorded capture must produce a real recommendation and display the target ball returned by the backend.
- **Browser request times** come from same-origin resource timing. Sample fetch, job submission, and overlay fetch include their response bodies. Polling and session creation contribute to the total click-to-result metrics.
- **Queue and worker times** use timestamps from the same PostgreSQL clock. The worker span includes analyzer HTTP, image handling, and result persistence.

The two click-to-result metrics use the browser's monotonic clock. Queue and worker intervals are reported independently. Stage medians and P95s are not additive. Percentiles use nearest-rank P95, and the JSON report includes each measured sample, fixture digest, browser version, and host specifications.

## First local baseline

On November 26, 2026, the included 159,230-byte JPEG completed all 30 measured captures after five warmups. The run used headless Chrome 158 on Windows with a Ryzen 7 5700X and a local Compose stack running one worker. Chrome and the services shared the same physical host. [Raw measurements and runner configuration](capture-baseline.json)

| Interval | Median | P95 |
| --- | ---: | ---: |
| Click to advice | 564.1 ms | 580.7 ms |
| Click to decoded previews | 564.2 ms | 580.7 ms |
| Sample fetch | 2.8 ms | 3.2 ms |
| Job submission | 5.7 ms | 10.2 ms |
| Overlay fetch | 3.3 ms | 4.5 ms |
| Queued | 235.5 ms | 455 ms |
| Worker | 29 ms | 40 ms |
