const puppeteer = require("puppeteer-extra")
const StealthPlugin = require("puppeteer-extra-plugin-stealth")
const fs = require("fs")
const path = require("path")

puppeteer.use(StealthPlugin())

const COOKIES_PATH = "./cookies.json"
const CHAT_NAME = "Kylan Thurairajah"
const IMAGE_DIR = "./images"

if (!fs.existsSync(IMAGE_DIR)) {
  fs.mkdirSync(IMAGE_DIR)
}

async function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

async function start() {
  const browser = await puppeteer.launch({
    headless: false,
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
  })

  const page = await browser.newPage()

  if (fs.existsSync(COOKIES_PATH)) {
    const cookies = JSON.parse(fs.readFileSync(COOKIES_PATH))
    await page.setCookie(...cookies)
    console.log("✅ Logged in using saved cookies")
  }

  await page.goto("https://www.messenger.com", { waitUntil: "networkidle2" })

  if (!(await page.$('[aria-label="Chats"]'))) {
    console.log("👉 Please log in manually...")
    await sleep(60000) 
    const cookies = await page.cookies()
    fs.writeFileSync(COOKIES_PATH, JSON.stringify(cookies, null, 2))
    console.log("✅ Saved cookies for next time")
  }

  await page.waitForSelector('[aria-label="Search Messenger"]')
  await page.type('[aria-label="Search Messenger"]', CHAT_NAME)
  await page.keyboard.press("Enter")
  await sleep(3000)

  console.log(`🔎 Watching chat with ${CHAT_NAME}...`)

  let lastMessage = ""

  while (true) {
    try {
      const newText = await page.evaluate((lastMessage) => {
        const rows = document.querySelectorAll('[role="row"]')
        if (!rows.length) return null

        const latestRow = rows[rows.length - 1]
        const textEl = latestRow.querySelector('[dir="auto"]')
        const text = textEl ? textEl.innerText.trim() : ""

        if (text && text !== lastMessage) {
          return text
        }

        return lastMessage === "" ? "init" : null
      }, lastMessage)

      if (newText) {
        const rows = await page.$$('[role="row"]')
        const lastRow = rows[rows.length - 1]

        const filename = `msg_${Date.now()}.png`
        const filePath = path.join(IMAGE_DIR, filename)

        await lastRow.screenshot({ path: filePath })
        console.log(`📸 Screenshot saved: ${filePath}`)

        if (newText !== "init") {
          lastMessage = newText
          console.log("💬 New text:", newText)
        }
      }

      await sleep(5000)
    } catch (err) {
      console.error("⚠️ Error in loop:", err.message)
      await sleep(10000)
    }
  }
}

start()
