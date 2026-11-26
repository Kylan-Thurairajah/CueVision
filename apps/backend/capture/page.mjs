import { CaptureError, createCaptureClient } from "./client.mjs";

const client = createCaptureClient();
const element = id => document.getElementById(id);
const status = element("status");
const results = element("jobs");
const retry = element("retry");
const autoSpeech = element("auto-speech");
const audioStatus = element("audio-status");
const speech = globalThis.speechSynthesis;
const canSpeak = Boolean(speech && globalThis.SpeechSynthesisUtterance);
const objectUrls = [];
const pocketNames = {
  TL: "top-left", TM: "top-middle", TR: "top-right",
  BL: "bottom-left", BM: "bottom-middle", BR: "bottom-right",
};
const stageNames = {
  queued: "Queued", processing: "Analyzing table", completed: "Analysis complete", failed: "Analysis failed",
};
let active;
let currentUtterance;

function message(text) {
  if (status.textContent !== text) status.textContent = text;
}

function audioMessage(text) {
  audioStatus.textContent = text;
  audioStatus.hidden = !text;
}

function stopSpeech() {
  currentUtterance = undefined;
  speech?.cancel();
}

function speak(batch, text) {
  if (active !== batch || !canSpeak) return;
  stopSpeech();
  audioMessage("");
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "en";
  const failed = () => {
    if (active === batch && currentUtterance === utterance) {
      audioMessage("Audio playback was unavailable. Try Play advice again or read the text below.");
    }
  };
  utterance.addEventListener("error", failed);
  utterance.addEventListener("end", () => {
    if (currentUtterance === utterance) currentUtterance = undefined;
  });
  currentUtterance = utterance;
  try {
    speech.speak(utterance);
  } catch {
    failed();
  }
}

function createFigure(blob, caption, alt) {
  const figure = document.createElement("figure");
  const label = document.createElement("figcaption");
  const image = document.createElement("img");
  const url = URL.createObjectURL(blob);
  objectUrls.push(url);
  image.src = url;
  image.alt = alt;
  label.textContent = caption;
  figure.append(label, image);
  return figure;
}

function displayResult(batch, frame, job, overlay) {
  const advice = job.result?.advice
    ?.replace(/\b(TL|TM|TR|BL|BM|BR)\b/g, pocket => `${pocketNames[pocket]} pocket`)
    .replace("ghost point", "marked aim point")
    .replace("≈", "about")
    .replace("°", " degrees");
  frame.advice.textContent = advice ?? job.error ?? "Analysis finished without a recommendation.";
  if (advice && canSpeak) {
    const play = document.createElement("button");
    play.type = "button";
    play.textContent = "Play advice";
    play.addEventListener("click", () => speak(batch, advice));
    frame.row.append(play);
  }
  const images = document.createElement("div");
  images.className = "capture-images";
  images.append(createFigure(frame.capture.file, "Original photo", `Original capture for ${frame.name}`));
  if (overlay) {
    const caption = job.result?.rectified ? "Annotated top-down view" : "Analysis image";
    images.append(createFigure(overlay, caption, `Analysis overlay for ${frame.name}`));
  }
  frame.row.append(images);
  return advice;
}

async function waitForJob(batch, frame) {
  const deadline = performance.now() + 120_000;
  while (active === batch && performance.now() < deadline) {
    const job = await client.job(frame.capture, frame.submission.id);
    if (active !== batch) return;
    const stage = stageNames[job.status] ?? job.status;
    frame.details.textContent = `${stage} · Job ${job.id}`;
    message(`${stage} · Capture ${batch.index + 1} of ${batch.frames.length}`);
    if (job.status === "completed" || job.status === "failed") return job;
    await new Promise(resolve => setTimeout(resolve, 500));
  }
  if (active === batch) throw new CaptureError("Analysis is still pending. Retry to check its status.", true);
}

function createFrames(batch, files) {
  if (!files.length) throw new CaptureError("Select an image first.");
  if (files.length > 12) throw new CaptureError("Select at most 12 frames per replay.");
  return files.map(file => {
    const capture = client.prepare(batch.source, file, batch.felt);
    const row = document.createElement("li");
    const title = document.createElement("strong");
    const details = document.createElement("p");
    const session = document.createElement("p");
    const advice = document.createElement("p");
    title.textContent = file.name;
    details.textContent = "Waiting to submit";
    details.className = session.className = "id";
    advice.className = "advice";
    row.append(title, details, session, advice);
    return { capture, name: file.name, row, details, session, advice };
  });
}

async function continueBatch(batch) {
  if (!batch || active !== batch || batch.running) return;
  batch.running = true;
  retry.hidden = true;
  try {
    if (!batch.frames) {
      message("Preparing capture…");
      const files = await batch.load();
      if (active !== batch) return;
      batch.frames = createFrames(batch, files);
      results.append(...batch.frames.map(frame => frame.row));
    }
    while (batch.index < batch.frames.length) {
      const frame = batch.frames[batch.index];
      message(`Submitting capture ${batch.index + 1} of ${batch.frames.length}…`);
      frame.submission ??= await client.submit(frame.capture);
      if (active !== batch) return;
      frame.session.textContent = `Session ${frame.submission.sessionId}`;
      frame.details.textContent = `Submitted · Job ${frame.submission.id}`;
      const job = await waitForJob(batch, frame);
      if (!job || active !== batch) return;
      const overlay = job.overlayUrl ? await client.artifact(frame.capture, job.overlayUrl) : null;
      if (active !== batch) return;
      const advice = displayResult(batch, frame, job, overlay);
      batch.index++;
      if (advice && autoSpeech.checked && batch.index === batch.frames.length) speak(batch, advice);
    }
    message(batch.source === "replay" ? "Replay finished." : "Capture finished.");
  } catch (error) {
    if (active !== batch) return;
    message(error.message);
    retry.hidden = !(error instanceof CaptureError && error.retryable);
  } finally {
    batch.running = false;
  }
}

function start(source, load) {
  const batch = { source, load, felt: element("felt").value, index: 0, running: false };
  active = batch;
  stopSpeech();
  if (canSpeak) audioMessage("");
  for (const url of objectUrls.splice(0)) URL.revokeObjectURL(url);
  results.replaceChildren();
  void continueBatch(batch);
}

async function loadSample() {
  try {
    const response = await fetch("/capture/sample.jpg", { signal: AbortSignal.timeout(10_000) });
    if (!response.ok) throw new Error("Sample unavailable");
    return [new File([await response.blob()], "sample_pool_table1.jpg", { type: "image/jpeg" })];
  } catch {
    throw new CaptureError("Could not load the sample. Reconnect and retry.", true);
  }
}

autoSpeech.disabled = !canSpeak;
if (!canSpeak) audioMessage("Speech is unavailable in this browser. Advice will appear as text.");
autoSpeech.addEventListener("change", () => { if (!autoSpeech.checked) stopSpeech(); });
element("simulate").addEventListener("click", () => start("simulated", loadSample));
element("upload").addEventListener("click", () => {
  const files = [...element("photo").files];
  start("upload", () => files);
});
element("replay").addEventListener("click", () => {
  const files = [...element("frames").files].sort((a, b) => a.name < b.name ? -1 : Number(a.name > b.name));
  start("replay", () => files);
});
retry.addEventListener("click", () => void continueBatch(active));
