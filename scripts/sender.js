const puppeteer = require("puppeteer-extra")
const StealthPlugin = require("puppeteer-extra-plugin-stealth")
const fs = require("fs")

puppeteer.use(StealthPlugin())

const COOKIES_PATH = "./cookies.json"
const CHAT_NAME = "Kylan Thurairajah"

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

  const inputSelector = '[aria-label="Message"]'
  await page.waitForSelector(inputSelector)

  const messageText =
    "Test message"
  await page.type(inputSelector, messageText)
  await page.keyboard.press("Enter")

  console.log("📤 Sent message:", messageText)

  await sleep(5000)
  await browser.close()
}

start()
