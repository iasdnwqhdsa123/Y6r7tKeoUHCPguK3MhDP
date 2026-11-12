#!/usr/bin/env bash
set -uo pipefail
R=/tmp/pre-upload
mkdir -p "$R"
if [ "${1:-}" = "--wait" ]; then
  for f in "$R"/*.pid; do
    [ -e "$f" ] || continue
    m="$(basename "$f" .pid)"
    while kill -0 "$(cat "$f")" 2>/dev/null; do sleep 2; done
    echo "::group::pre-upload $m"; cat "$R/$m.log"; echo "::endgroup::"
    rm -f "$f"
  done
  exit 0
fi
m="$1"
[ -d "dist-data/$m" ] || exit 0
if [ -n "${DATA_S3_BUCKET:-}" ] && [ -n "${DATA_S3_ENDPOINT:-${ARCHIVE_S3_ENDPOINT:-}}" ] && command -v aws > /dev/null; then
  B="$R/$m-r2.files"
  rm -rf "$B"
  if rsync -a --link-dest="$PWD/dist-data/$m/" --exclude '_hours/' --exclude '_quarters/' --exclude '*.building/' --exclude 'latest.json' --exclude '.*' "dist-data/$m/" "$B/"; then
    setsid nohup env AWS_ACCESS_KEY_ID="${DATA_S3_KEY:-${ARCHIVE_S3_KEY:-}}" AWS_SECRET_ACCESS_KEY="${DATA_S3_SECRET:-${ARCHIVE_S3_SECRET:-}}" AWS_DEFAULT_REGION=auto AWS_MAX_ATTEMPTS=5 \
      bash -c 'aws configure set default.s3.max_concurrent_requests 32; aws --endpoint-url "$1" s3 sync "$2/" "s3://$3/data/$4/" --only-show-errors --no-progress && echo "sent: $(du -sh "$2" | cut -f1)"; rm -rf "$2"' _ \
      "${DATA_S3_ENDPOINT:-$ARCHIVE_S3_ENDPOINT}" "$B" "$DATA_S3_BUCKET" "$m" > "$R/$m-r2.log" 2>&1 < /dev/null &
    echo $! > "$R/$m-r2.pid"
    echo "pre-upload $m to the data host: started"
  else
    rm -rf "$B"
  fi
fi
[ -n "${NETLIFY_AUTH_TOKEN:-}" ] && [ -n "${NETLIFY_SITE_ID:-}" ] || { echo "pre-upload $m: no Netlify token, skipped"; exit 0; }
CLI="$(.github/scripts/netlify-cli.sh)" || exit 0
P="${DATA_PATH_SECRET:+/$DATA_PATH_SECRET}"
D="$R/$m.site"
rm -rf "$D"
mkdir -p "$D$P"
rsync -a --link-dest="$PWD/dist-data/$m/" --exclude '_hours/' --exclude '_quarters/' --exclude '*.building/' "dist-data/$m/" "$D$P/$m/" || { echo "pre-upload $m: copy failed, skipped"; rm -rf "$D"; exit 0; }
printf '[build]\n  publish = "."\n' > "$D/netlify.toml"
setsid nohup bash -c 'cd "$1" && "$2" deploy --dir . --functions /tmp/no-functions --message "$3"; rm -rf "$1"' _ "$D" "$CLI" "pre-upload $m ${GITHUB_RUN_ID:-}" > "$R/$m.log" 2>&1 < /dev/null &
echo $! > "$R/$m.pid"
echo "pre-upload $m: started"
