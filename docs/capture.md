# CueVision capture

The backend serves a small browser capture workspace at `/capture`. **Capture sample** simulates a wearable shutter using the included JPEG. **Upload photo** accepts a local JPEG or PNG. **Replay frames** sorts up to 12 JPEG or PNG files by filename and submits them in that order. Prefix names with `01-`, `02-`, and so on for explicit ordering.

All three entry points use the same capture contract. A capture has a source, immutable image bytes, a felt selection, and an idempotency key. Each source gets its own backend session. The client retains the bearer token in memory and reports submission, queueing, analysis, and the result as state becomes available. Completed jobs show the original photo, the analysis image, and advice. A job with no usable shot displays the analyzer's explanation. A processing failure displays its error.

If a connection drops during submission or result retrieval, reconnect and select **Retry**. The page keeps the current capture's bytes, key, and session. When it already received a job ID, it resumes polling that job. A new capture replaces the previous batch on screen and cancels its speech. Any already accepted jobs can still finish on the server. Their late responses cannot update or speak over the active capture. Reloading the page ends its in-memory sessions and discards pending retries. A dropped response during session creation may leave an empty session. Unsupported images are rejected before submission by checking their file signatures, and Python performs full decoding and pixel-limit validation.

## Audio and browser verification

On a browser with speech synthesis and an installed voice, click **Capture sample**, then **Play advice**. To read advice as soon as a result arrives, enable **Read advice aloud automatically** before capturing. For replay, automatic speech reads the final frame's advice. Earlier frames in the active replay have their own playback buttons. If the browser declines playback, the page keeps the text and offers manual playback. [Web Speech API](https://developer.mozilla.org/en-US/docs/Web/API/SpeechSynthesis) · [Chromium user activation requirement](https://blog.chromium.org/2018/10/chrome-71-beta-relative-time-formats.html)

With Compose running, run the browser tests on Windows PowerShell. Point the URL at your stack and use an installed Chromium browser. To use Playwright's Chromium instead, run `npx playwright-core install chromium` inside `apps/backend` and leave `CUEVISION_BROWSER_PATH` unset.

```powershell
npm.cmd --prefix apps/backend ci
$env:CUEVISION_WORKFLOW_URL = 'http://127.0.0.1:8081'
$env:CUEVISION_BROWSER_PATH = 'C:/Program Files/Google/Chrome/Application/chrome.exe'
npm.cmd --prefix apps/backend run test:workflow
```

The automated tests use the real job stack and observe what the page sends to the speech API. Use the manual step above to verify sound through your device's speakers.

## Official Meta wearable path

Meta's Android DAT 1.1.0 CameraAccess sample supports taking a photo during camera preview. Its debug MockDeviceKit can supply a JPEG or PNG photo and simulate registration, permission decisions, device lifecycle and disconnection. The iOS SDK also supports mock-device testing. Physical glasses and audio routing were outside this validation. [Android changelog](https://github.com/meta/meta-wearables-dat-android/blob/main/CHANGELOG.md) · [MockDeviceKit guide](https://github.com/meta/meta-wearables-dat-android/blob/main/plugins/mwdat-android/skills/mockdevice-testing/SKILL.md) · [iOS SDK](https://github.com/meta/meta-wearables-dat-ios)

The native spike verified the sample prerequisites against this Windows host. JDK 21 is installed, and Android Studio, Android SDK 36, an emulator and an Android virtual device were absent from the locations checked. A native build or mock session could not be run. On a provisioned Windows machine, build the official CameraAccess debug sample, exercise a MockDeviceKit capture and submit its photo with the same session and multipart job contract. The browser simulator does not use Meta's SDK. [CameraAccess prerequisites](https://github.com/meta/meta-wearables-dat-android/blob/main/samples/CameraAccess/README.md#prerequisites) · [Backend contract](backend.md)

See the [grading guide](grading.md#3-hardware-free-capture) for hands-on checks.
