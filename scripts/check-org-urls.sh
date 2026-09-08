#!/usr/bin/env bash
# check-org-urls.sh — fail if stale org/user URLs reappear
# Run: bash scripts/check-org-urls.sh
# Exit 0 = clean, Exit 1 = stale patterns found

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Patterns that should never appear in tracked files
STALE_PATTERNS=(
  # Written as a character class so the handle is never spelled literally in a
  # PUBLIC repo. grep -rnE below treats it identically. Spelling it out here
  # published the exact string this check exists to keep out -- flagged by
  # identity-scan.py on 2026-09-08, and it was a true positive, not noise.
  "mikey[f]rilot"
  "github\.com/mcp-tool-shop/"    # mcp-tool-shop without -org
  "your-repo"
  "your-org"
)

FAILURES=0

for pattern in "${STALE_PATTERNS[@]}"; do
  # Search tracked files only, exclude .git and binary files
  MATCHES=$(grep -rnE "$pattern" "$REPO_ROOT" \
    --include='*.md' --include='*.py' --include='*.txt' \
    --include='*.yml' --include='*.yaml' --include='*.json' \
    --include='*.toml' --include='*.svg' --include='*.cfg' \
    --include='Dockerfile' --include='Makefile' \
    2>/dev/null || true)

  if [ -n "$MATCHES" ]; then
    echo "FAIL: stale pattern '$pattern' found:"
    echo "$MATCHES"
    echo ""
    FAILURES=$((FAILURES + 1))
  fi
done

if [ "$FAILURES" -gt 0 ]; then
  echo "ERROR: $FAILURES stale URL pattern(s) detected."
  echo "All references should use: github.com/mcp-tool-shop-org/"
  exit 1
else
  echo "OK: no stale org/user URL patterns found."
  exit 0
fi
