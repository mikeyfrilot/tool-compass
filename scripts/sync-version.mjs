#!/usr/bin/env node
// ci-infra-03: sync-version.mjs — keep the npm wrapper's embedded version + git
// tag in lockstep with the release version. Referenced by release.yml (the
// "Verify npm/bin/tool-compass.js references the tag" step) and by the header
// comment in npm/bin/tool-compass.js — this script used to be missing, so those
// references were dead. This restores the real tool.
//
// The npm wrapper lives in the npm/ subdir:
//   - npm/package.json      carries `"version": "X.Y.Z"`  (source of truth)
//   - npm/bin/tool-compass.js embeds `version: "X.Y.Z"` and `tag: "vX.Y.Z"`
//     inside a JSON.stringify(...) launch-config object; npm-launcher uses the
//     tag to fetch that exact GitHub Release's binaries.
//
// Behavior:
//   - Target version: argv[2] if given (accepts both "2.5.0" and "v2.5.0"),
//     else the `version` field from npm/package.json.
//   - Rewrites the bin shim's `version:` and `tag:` lines to match, via precise
//     regex replace — the rest of the file is left byte-for-byte untouched.
//   - If an explicit version arg was passed, package.json `version` is updated
//     too; otherwise package.json is treated as the source of truth and only the
//     bin shim is (re)synced.
//   - Prints what changed; exits 0 on success (including a clean no-op), and
//     exits non-zero with a clear message if an expected line can't be found.
//
// No external deps — node: builtins only. ESM (.mjs).

import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = join(HERE, "..");
const PKG_PATH = join(REPO_ROOT, "npm", "package.json");
const BIN_PATH = join(REPO_ROOT, "npm", "bin", "tool-compass.js");

const SEMVER_RE = /^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$/;

function fail(msg) {
  console.error(`sync-version: ${msg}`);
  process.exit(1);
}

// Determine target version + whether it was explicitly supplied.
const rawArg = process.argv[2];
const explicit = typeof rawArg === "string" && rawArg.trim() !== "";

let pkgRaw;
try {
  pkgRaw = readFileSync(PKG_PATH, "utf8");
} catch {
  fail(`cannot read ${PKG_PATH}`);
}

let pkg;
try {
  pkg = JSON.parse(pkgRaw);
} catch {
  fail(`${PKG_PATH} is not valid JSON`);
}

let version;
if (explicit) {
  version = rawArg.trim().replace(/^v/, "");
} else {
  version = pkg.version;
  if (typeof version !== "string" || version.trim() === "") {
    fail(`no version arg given and npm/package.json has no "version" field`);
  }
}

if (!SEMVER_RE.test(version)) {
  fail(`"${version}" is not a valid semver version (expected e.g. 2.5.0 or v2.5.0)`);
}

const tag = `v${version}`;
const changes = [];

// --- 1. npm/package.json --------------------------------------------------
// Only rewrite package.json when an explicit version was supplied; otherwise it
// IS the source of truth and we leave it alone.
if (explicit && pkg.version !== version) {
  // Precise line replace so we don't reserialize/reorder the whole file.
  const pkgVersionRe = /("version"\s*:\s*")[^"]+(")/;
  if (!pkgVersionRe.test(pkgRaw)) {
    fail(`could not find a "version" line to update in ${PKG_PATH}`);
  }
  const newPkgRaw = pkgRaw.replace(pkgVersionRe, `$1${version}$2`);
  writeFileSync(PKG_PATH, newPkgRaw);
  changes.push(`npm/package.json version: ${pkg.version} -> ${version}`);
}

// --- 2. npm/bin/tool-compass.js ------------------------------------------
let binRaw;
try {
  binRaw = readFileSync(BIN_PATH, "utf8");
} catch {
  fail(`cannot read ${BIN_PATH}`);
}

// The launch-config object embeds:  version: "X.Y.Z",   and   tag: "vX.Y.Z",
const binVersionRe = /(version:\s*")[^"]+(")/;
const binTagRe = /(tag:\s*")[^"]+(")/;

if (!binVersionRe.test(binRaw)) {
  fail(`could not find a \`version: "..."\` line in ${BIN_PATH}`);
}
if (!binTagRe.test(binRaw)) {
  fail(`could not find a \`tag: "..."\` line in ${BIN_PATH}`);
}

const curVersion = binRaw.match(binVersionRe)[0].match(/"([^"]+)"/)[1];
const curTag = binRaw.match(binTagRe)[0].match(/"([^"]+)"/)[1];

let newBinRaw = binRaw;
if (curVersion !== version) {
  newBinRaw = newBinRaw.replace(binVersionRe, `$1${version}$2`);
  changes.push(`npm/bin/tool-compass.js version: ${curVersion} -> ${version}`);
}
if (curTag !== tag) {
  newBinRaw = newBinRaw.replace(binTagRe, `$1${tag}$2`);
  changes.push(`npm/bin/tool-compass.js tag: ${curTag} -> ${tag}`);
}

if (newBinRaw !== binRaw) {
  writeFileSync(BIN_PATH, newBinRaw);
}

// --- Report ---------------------------------------------------------------
if (changes.length === 0) {
  console.log(`sync-version: already in sync at ${version} (tag ${tag}); nothing to do.`);
} else {
  console.log(`sync-version: synced npm wrapper to ${version} (tag ${tag}):`);
  for (const c of changes) console.log(`  - ${c}`);
}

process.exit(0);
