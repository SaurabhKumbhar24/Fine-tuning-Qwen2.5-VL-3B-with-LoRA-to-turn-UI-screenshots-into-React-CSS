// Phase 2 of evaluation: does the generated code parse?
//
// Reads the .jsonl file written by evaluate.py and, for every row, checks:
//   - does the JSX parse?           (@babel/parser)
//   - does it have `export default`?
//   - does the CSS parse?           (postcss)
//   - which packages does it import?
//
// Setup (once):  npm install @babel/parser postcss
// Run:           node syntax_check.mjs outputs/eval/finetuned.jsonl
// Writes:        outputs/eval/finetuned.syntax.jsonl

import fs from "fs";
import { parse } from "@babel/parser";
import postcss from "postcss";

const inputFile = process.argv[2];
if (!inputFile) {
  console.error("Usage: node syntax_check.mjs outputs/eval/<name>.jsonl");
  process.exit(1);
}
const outputFile = inputFile.replace(/\.jsonl$/, ".syntax.jsonl");

// ---------- read the phase 1 results ----------
const records = fs
  .readFileSync(inputFile, "utf-8")
  .split("\n")
  .filter((line) => line.trim() !== "")
  .map((line) => JSON.parse(line));

// ---------- check one JSX string ----------
function checkJsx(jsx) {
  const result = { jsx_ok: false, has_export: false, imports: [], jsx_error: null };
  if (jsx === null) {
    result.jsx_error = "no jsx block";
    return result;
  }

  let ast;
  try {
    ast = parse(jsx, { sourceType: "module", plugins: ["jsx"] });
  } catch (err) {
    // err.message looks like: "Unexpected token, expected "," (12:4)"
    result.jsx_error = err.message;
    return result;
  }

  result.jsx_ok = true;
  // ast.program.body is the list of top-level statements
  for (const node of ast.program.body) {
    if (node.type === "ImportDeclaration") {
      result.imports.push(node.source.value);  // e.g. "react", "@mui/material"
    }
    if (node.type === "ExportDefaultDeclaration") {
      result.has_export = true;
    }
  }
  return result;
}

// ---------- check one CSS string ----------
function checkCss(css) {
  if (css === null) return { css_ok: false, css_error: "no css block" };
  try {
    postcss.parse(css);
    return { css_ok: true, css_error: null };
  } catch (err) {
    return { css_ok: false, css_error: err.reason ?? err.message };
  }
}

// "@mui/material/Button" -> "@mui/material",  "lodash/map" -> "lodash"
function packageName(importPath) {
  const parts = importPath.split("/");
  return importPath.startsWith("@") ? parts.slice(0, 2).join("/") : parts[0];
}

const isRelative = (importPath) => importPath.startsWith(".");

// ---------- check every row ----------
const results = records.map((r) => {
  const jsx = checkJsx(r.jsx);
  const css = checkCss(r.css);
  return {
    problem_id: r.problem_id,
    complexity_level: r.complexity_level,
    hit_max_tokens: r.hit_max_tokens,
    ...jsx,
    ...css,
    has_relative_import: jsx.imports.some(isRelative),
  };
});

fs.writeFileSync(outputFile, results.map((r) => JSON.stringify(r)).join("\n") + "\n");

// ---------- summary ----------
const pct = (n, total) => `${n}/${total} (${total ? Math.round((100 * n) / total) : 0}%)`;

function printRates(rows, label) {
  const n = rows.length;
  const jsxOk = rows.filter((r) => r.jsx_ok).length;
  const exportOk = rows.filter((r) => r.has_export).length;
  const cssOk = rows.filter((r) => r.css_ok).length;
  const relative = rows.filter((r) => r.has_relative_import).length;
  console.log(
    `${label.padEnd(14)} JSX ${pct(jsxOk, n).padEnd(12)} export ${pct(exportOk, n).padEnd(12)} ` +
      `CSS ${pct(cssOk, n).padEnd(12)} relative imports ${relative}`
  );
}

console.log(`\n${results.length} rows from ${inputFile}\n`);
printRates(results, "all");
const levels = [...new Set(results.map((r) => r.complexity_level))].sort();
for (const level of levels) {
  printRates(results.filter((r) => r.complexity_level === level), `complexity ${level}`);
}

// most imported packages (each counted once per row)
const counts = {};
for (const r of results) {
  for (const pkg of new Set(r.imports.filter((p) => !isRelative(p)).map(packageName))) {
    counts[pkg] = (counts[pkg] ?? 0) + 1;
  }
}
console.log("\nMost imported packages:");
for (const [pkg, count] of Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 10)) {
  console.log(`  ${pkg.padEnd(30)} ${count}`);
}

// a few failures, so you can see what went wrong
const failures = results.filter((r) => !r.jsx_ok);
if (failures.length > 0) {
  console.log(`\nJSX failures (first 5 of ${failures.length}):`);
  for (const r of failures.slice(0, 5)) {
    const cut = r.hit_max_tokens ? "  [hit max tokens]" : "";
    console.log(`  ${r.problem_id}: ${r.jsx_error}${cut}`);
  }
}

console.log(`\nDetails written to ${outputFile}`);
