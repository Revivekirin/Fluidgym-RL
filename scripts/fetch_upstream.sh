#!/usr/bin/env bash
# Clone upstream repos at the pinned SHAs in UPSTREAM.lock. TD-MPC (v1) is audit-only and not fetched by default.
# Env: FGRL_THIRD_PARTY (dest dir, default <repo>/third_party), FGRL_UPSTREAMS (default "fluidgym tdmpc2")
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TP="${FGRL_THIRD_PARTY:-$ROOT/third_party}"
mkdir -p "$TP"
for name in ${FGRL_UPSTREAMS:-fluidgym tdmpc2}; do
  read -r _ url sha _ < <(grep -E "^$name[[:space:]]" "$ROOT/UPSTREAM.lock")
  [ -d "$TP/$name/.git" ] || git clone --quiet "$url" "$TP/$name"
  git -C "$TP/$name" fetch --quiet origin "$sha" 2>/dev/null || git -C "$TP/$name" fetch --quiet --all
  git -C "$TP/$name" checkout --quiet "$sha"
  echo "$name @ $(git -C "$TP/$name" rev-parse HEAD)"
done
