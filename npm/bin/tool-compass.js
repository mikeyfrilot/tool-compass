#!/usr/bin/env node
"use strict";

// version/tag refer to the source repo binary release, not the npm wrapper version.
// Both lines are kept in sync at release time by scripts/sync-version.mjs.
process.env.MCPTOOLSHOP_LAUNCH_CONFIG = JSON.stringify({
  toolName: "tool-compass",
  owner: "mcp-tool-shop-org",
  repo: "tool-compass",
  version: "2.5.0",
  tag: "v2.5.0",
});

require("@mcptoolshop/npm-launcher/bin/mcptoolshop-launch.js");
