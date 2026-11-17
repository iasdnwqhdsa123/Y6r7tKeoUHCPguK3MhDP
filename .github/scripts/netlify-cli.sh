#!/usr/bin/env bash
set -euo pipefail
N=/tmp/netlify-cli
[ -x "$N/node_modules/.bin/netlify" ] || npm install --prefix "$N" --silent --no-fund --no-audit netlify-cli@17.38.1 >&2
C="$N/node_modules/netlify-cli/dist/utils/deploy/constants.js"
sed -i -E 's/DEFAULT_CONCURRENT_UPLOAD = [0-9]+;/DEFAULT_CONCURRENT_UPLOAD = 128;/' "$C"
grep -q "DEFAULT_CONCURRENT_UPLOAD = 128;" "$C" || echo "::warning::upload concurrency not raised: 5 files at a time" >&2
mkdir -p /tmp/no-functions
echo "$N/node_modules/.bin/netlify"
