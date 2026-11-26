import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { readFile } from "node:fs/promises";
import { setTimeout as sleep } from "node:timers/promises";
import test from "node:test";

const url = process.env.CUEVISION_WORKFLOW_URL;
const executablePath = process.env.CUEVISION_BROWSER_PATH;
const require = createRequire(import.meta.url);
const sample = new URL("../../../sample_pool_table1.jpg", import.meta.url);
const options = { skip: !url, timeout: 45_000 };

async function withPage(callback) {
  const { chromium } = require(process.env.CUEVISION_PLAYWRIGHT_PATH ?? "playwright-core");
  const browser = await chromium.launch({ executablePath });
  try {
    const page = await browser.newPage();
    page.setDefaultTimeout(12_000);
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.addInitScript(() => {
      window.cueVisionSpeech = [];
      window.cueVisionProgress = [];
      new MutationObserver(() => {
        const status = document.getElementById("status")?.textContent;
        if (status) window.cueVisionProgress.push(status);
      }).observe(document, { childList: true, subtree: true });
      const speech = window.speechSynthesis;
      const stop = () => window.cueVisionSpeech.push({ type: "cancel" });
      Object.defineProperties(speech, {
        speak: {
          value(utterance) { window.cueVisionSpeech.push({ type: "speak", text: utterance.text }); },
          configurable: true,
        },
        cancel: { value: stop, configurable: true },
      });
    });
    await page.goto(new URL("/capture", url).href);
    await callback(page);
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
  }
}

function speech(page) {
  return page.evaluate(() => window.cueVisionSpeech);
}

async function awaitImage(page, alt) {
  const image = page.getByRole("img", { name: alt, exact: true });
  await image.waitFor();
  await page.waitForFunction(name => {
    const img = Array.from(document.images).find(candidate => candidate.alt === name);
    return img?.complete && img.naturalWidth > 0;
  }, alt);
}

async function upload(page, name, bytes, mediaType = "image/jpeg") {
  await page.locator("#photo").setInputFiles({ name, mimeType: mediaType, buffer: bytes });
  await page.getByRole("button", { name: "Upload photo", exact: true }).click();
}

test("the real sample job displays progress, both images, and playable advice", options, async () => {
  await withPage(async page => {
    const submitted = page.waitForResponse(response =>
      new URL(response.url()).pathname === "/jobs" && response.request().method() === "POST");
    await page.getByRole("button", { name: "Capture sample", exact: true }).click();
    const response = await submitted;
    assert.equal(response.status(), 202);
    const { id } = await response.json();
    await page.getByText("Capture finished.", { exact: true }).waitFor();
    const progress = await page.evaluate(() => window.cueVisionProgress);
    assert.ok(progress.some(text => /loading sample|preparing capture|processing capture|submitting capture|queued|analyzing table/i.test(text)),
      `Expected observable progress, received ${JSON.stringify(progress)}`);
    assert.match(await page.locator("#jobs").textContent(), new RegExp(id));
    await awaitImage(page, "Original capture for sample_pool_table1.jpg");
    await awaitImage(page, "Analysis overlay for sample_pool_table1.jpg");
    assert.match(await page.locator("#jobs").textContent(), /pot ball/i);
    assert.equal((await speech(page)).filter(({ type }) => type === "speak").length, 0);
    await page.getByRole("button", { name: "Play advice", exact: true }).click();
    const utterances = (await speech(page)).filter(({ type }) => type === "speak");
    assert.equal(utterances.length, 1);
    assert.match(utterances[0].text, /pot ball/i);
  });
});

test("a late older job cannot replace or speak over the newer capture", options, async () => {
  await withPage(async page => {
    const bytes = await readFile(sample);
    const seen = Promise.withResolvers();
    const release = Promise.withResolvers();
    const finished = Promise.withResolvers();
    void finished.promise.catch(() => undefined);
    let firstId;
    let held = false;

    await page.route(/\/jobs\/[0-9a-f-]+$/, async route => {
      if (held || route.request().method() !== "GET") {
        await route.continue();
        return;
      }
      held = true;
      firstId = new URL(route.request().url()).pathname.split("/").at(-1);
      seen.resolve();
      try {
        await release.promise;
        const authorization = await route.request().headerValue("authorization");
        assert.ok(authorization);
        const deadline = performance.now() + 5000;
        let response;
        while (performance.now() < deadline) {
          response = await page.request.get(route.request().url(), { headers: { authorization } });
          if ((await response.json()).status === "completed") break;
          await sleep(100);
        }
        assert.equal((await response?.json()).status, "completed");
        await route.fulfill({ response }).catch(() => {
          // A superseded browser request may already have been aborted.
        });
        finished.resolve();
      } catch (error) {
        finished.reject(error);
      }
    });
    try {
      await page.getByRole("checkbox", { name: /Read advice aloud automatically/i }).check();
      await upload(page, "older.jpg", bytes);
      await Promise.race([
        seen.promise,
        sleep(10_000, undefined, { ref: false }).then(() => { throw new Error("Older job was not polled"); }),
      ]);
      await upload(page, "newer.jpg", bytes);
      await page.getByText("Capture finished.", { exact: true }).waitFor();
      await awaitImage(page, "Analysis overlay for newer.jpg");
      const newerUtterances = (await speech(page)).filter(({ type }) => type === "speak");
      assert.equal(newerUtterances.length, 1);
      assert.match(newerUtterances[0].text, /pot ball/i);
      release.resolve();
      await finished.promise;
      await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
      const result = await page.locator("#jobs").textContent();
      assert.ok(result.includes("newer.jpg"));
      assert.ok(!result.includes("older.jpg"));
      assert.ok(!result.includes(firstId));
      const utterances = (await speech(page)).filter(({ type }) => type === "speak");
      assert.deepEqual(utterances, newerUtterances);
    } finally {
      release.resolve();
    }
  });
});

test("no-shot and decoder failures remain understandable in the workspace", options, async () => {
  await withPage(async page => {
    const blackImage = await page.evaluate(() => {
      const canvas = document.createElement("canvas");
      canvas.width = canvas.height = 32;
      const context = canvas.getContext("2d");
      context.fillStyle = "#000000";
      context.fillRect(0, 0, 32, 32);
      return canvas.toDataURL().split(",")[1];
    });
    await upload(page, "blank.png", Buffer.from(blackImage, "base64"), "image/png");
    await page.getByText("Capture finished.", { exact: true }).waitFor();
    await awaitImage(page, "Original capture for blank.png");
    assert.match(await page.locator("#jobs").textContent(), /could not find the table/i);

    await upload(page, "corrupt.jpg", Buffer.from([0xff, 0xd8, 0xff, 0xe0]));
    await page.getByText("The image was rejected by the analysis service.", { exact: true }).waitFor();
    assert.equal(await page.getByRole("button", { name: "Play advice", exact: true }).count(), 0);
    assert.equal((await speech(page)).filter(({ type }) => type === "speak").length, 0);
  });
});
