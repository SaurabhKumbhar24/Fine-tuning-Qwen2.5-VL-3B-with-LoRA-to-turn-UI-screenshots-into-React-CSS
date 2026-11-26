// Phase 3, step 2: render every answer and take a screenshot.
//
// For each row of a .jsonl file (from evaluate.py or prepare_refs.py):
//   1. write App.jsx, style.css and a small entry file to a build folder
//   2. bundle them with esbuild into one page (out.js + out.css + index.html)
//   3. open the page in headless Chromium at the size of the input screenshot
//   4. screenshot the visible window
//
// Run:     node render.mjs outputs/eval/finetuned.jsonl
// Writes:  outputs/eval/renders/finetuned/<problem_id>.png
//          outputs/eval/finetuned.render.jsonl   (did it render? which error?)

import fs from "fs";
import path from "path";
import { pathToFileURL } from "url";
import * as esbuild from "esbuild";
import { chromium } from "playwright";

const EVAL_DIR = "outputs/eval";
const REFS_DIR = path.join(EVAL_DIR, "refs");
const LOAD_TIMEOUT_MS = 10000;   // max wait for the page to load
const SETTLE_MS = 500;           // extra wait for effects / fonts after load

const inputFile = process.argv[2];
if (!inputFile) {
  console.error("Usage: node render.mjs outputs/eval/<name>.jsonl");
  process.exit(1);
}
const name = path.basename(inputFile, ".jsonl");
const buildRoot = path.resolve(EVAL_DIR, "build", name);  // inside the project, so node_modules is found
const shotDir = path.join(EVAL_DIR, "renders", name);
const outputFile = path.join(EVAL_DIR, `${name}.render.jsonl`);
fs.mkdirSync(shotDir, { recursive: true });

const records = fs
  .readFileSync(inputFile, "utf-8")
  .split("\n")
  .filter((line) => line.trim() !== "")
  .map((line) => JSON.parse(line));

// ---------- the fixed files of every page ----------
const ENTRY = `
import "./style.css";
import * as Module from "./App.jsx";
import { createRoot } from "react-dom/client";

const App = Module.default;
if (typeof App !== "function" && typeof App !== "object") {
  throw new Error("App.jsx has no default export");
}
createRoot(document.getElementById("root")).render(<App />);
`;

const HTML = `<!doctype html>
<html>
  <head><meta charset="utf-8"><link rel="stylesheet" href="out.css"></head>
  <body><div id="root"></div><script src="out.js"></script></body>
</html>`;

// The model's code may import its CSS under any name ("./App.css", "./styles.css").
// There is only one CSS block, so every relative .css import points to style.css.
const cssRedirect = (cssPath) => ({
  name: "css-redirect",
  setup(build) {
    build.onResolve({ filter: /^\.\.?\/.*\.css$/ }, () => ({ path: cssPath }));
  },
});

// ---------- step 1 + 2: write files and bundle ----------
async function buildPage(record) {
  const dir = path.join(buildRoot, record.problem_id);
  fs.rmSync(dir, { recursive: true, force: true });
  fs.mkdirSync(dir, { recursive: true });

  const cssPath = path.join(dir, "style.css");
  fs.writeFileSync(path.join(dir, "App.jsx"), record.jsx);
  fs.writeFileSync(cssPath, record.css ?? "");
  fs.writeFileSync(path.join(dir, "entry.jsx"), ENTRY);
  fs.writeFileSync(path.join(dir, "index.html"), HTML);
  // esbuild only writes out.css when there is CSS; make sure the <link> never 404s
  fs.writeFileSync(path.join(dir, "out.css"), "");

  await esbuild.build({
    entryPoints: [path.join(dir, "entry.jsx")],
    outfile: path.join(dir, "out.js"),
    bundle: true,
    jsx: "automatic",                       // no need for `import React` to use JSX
    loader: { ".js": "jsx" },
    define: { "process.env.NODE_ENV": '"production"' },
    plugins: [cssRedirect(cssPath)],
    logLevel: "silent",
  });
  return path.join(dir, "index.html");
}

// PNG files store width and height at bytes 16-23 (big-endian)
function pngSize(file) {
  const buf = fs.readFileSync(file);
  return { width: buf.readUInt32BE(16), height: buf.readUInt32BE(20) };
}

// esbuild error -> short text, e.g. 'Could not resolve "react-icons/fa"'
function buildErrorText(err) {
  const first = err.errors?.[0];
  return first ? first.text : String(err.message).split("\n")[0];
}

// ---------- step 3 + 4: open and screenshot ----------
async function screenshotPage(browser, htmlPath, size, shotPath) {
  const page = await browser.newPage({ viewport: size });
  const pageErrors = [];
  page.on("pageerror", (err) => pageErrors.push(err.message.split("\n")[0]));

  // Block internet requests (images from URLs, web fonts): renders are then
  // offline, fast and the same every run. Local files load as normal.
  await page.route("**/*", (route) =>
    route.request().url().startsWith("file:") ? route.continue() : route.abort()
  );

  try {
    await page.goto(pathToFileURL(htmlPath).href, { waitUntil: "load", timeout: LOAD_TIMEOUT_MS });
  } catch {
    // a slow page still gets a screenshot of whatever is there
  }
  await page.waitForTimeout(SETTLE_MS);

  // React removes everything from #root when the component crashes
  const isBlank = await page.evaluate(() => document.getElementById("root").childElementCount === 0);
  await page.screenshot({ path: shotPath, fullPage: false });
  await page.close();
  return { pageErrors, isBlank };
}

// ---------- main loop ----------
const browser = await chromium.launch();
const results = [];

for (const [i, record] of records.entries()) {
  const result = {
    problem_id: record.problem_id,
    complexity_level: record.complexity_level,
    render_ok: false,
    stage: null,        // where it failed: "no jsx" | "build" | "runtime"
    error: null,
    page_errors: [],
  };
  const shotPath = path.join(shotDir, `${record.problem_id}.png`);
  fs.rmSync(shotPath, { force: true });

  if (record.jsx === null) {
    result.stage = "no jsx";
    result.error = "no jsx block";
  } else {
    let htmlPath = null;
    try {
      htmlPath = await buildPage(record);
    } catch (err) {
      result.stage = "build";
      result.error = buildErrorText(err);
    }

    if (htmlPath !== null) {
      const size = pngSize(path.join(REFS_DIR, `${record.problem_id}.png`));
      const { pageErrors, isBlank } = await screenshotPage(browser, htmlPath, size, shotPath);
      result.page_errors = pageErrors;
      if (isBlank) {
        result.stage = "runtime";
        result.error = pageErrors[0] ?? "page is blank";
      } else {
        result.render_ok = true;
      }
    }
  }

  results.push(result);
  const status = result.render_ok ? "ok" : `FAILED (${result.stage}): ${result.error}`;
  console.log(`[${i + 1}/${records.length}] ${record.problem_id}  ${status}`);
}

await browser.close();
fs.writeFileSync(outputFile, results.map((r) => JSON.stringify(r)).join("\n") + "\n");

// ---------- summary ----------
const pct = (n, total) => `${n}/${total} (${total ? Math.round((100 * n) / total) : 0}%)`;
console.log(`\nRendered: ${pct(results.filter((r) => r.render_ok).length, results.length)}`);
const levels = [...new Set(results.map((r) => r.complexity_level))].sort();
for (const level of levels) {
  const rows = results.filter((r) => r.complexity_level === level);
  console.log(`  complexity ${level}: ${pct(rows.filter((r) => r.render_ok).length, rows.length)}`);
}

const byStage = {};
for (const r of results.filter((r) => !r.render_ok)) byStage[r.stage] = (byStage[r.stage] ?? 0) + 1;
if (Object.keys(byStage).length > 0) console.log("Failures by stage:", byStage);

// Packages the code needs but which are not installed -> candidates for `npm install`
const missing = {};
for (const r of results) {
  const m = r.error?.match(/^Could not resolve "([^"]+)"/);
  if (m && !m[1].startsWith(".")) missing[m[1]] = (missing[m[1]] ?? 0) + 1;
}
if (Object.keys(missing).length > 0) {
  console.log("Missing packages (rows):");
  for (const [pkg, n] of Object.entries(missing).sort((a, b) => b[1] - a[1])) {
    console.log(`  ${pkg.padEnd(40)} ${n}`);
  }
}
console.log(`\nScreenshots in ${shotDir}, details in ${outputFile}`);
