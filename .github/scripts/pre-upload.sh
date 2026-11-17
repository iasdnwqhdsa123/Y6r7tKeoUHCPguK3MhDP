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
  mkdir -p "$B/$m"
  if rsync -a --link-dest="$PWD/dist-data/$m/" --exclude '_hours/' --exclude '_quarters/' --exclude '*.building/' --exclude 'latest.json' --exclude '.*' "dist-data/$m/" "$B/$m/"; then
    setsid nohup bash -c '"$1" --all "$2" && echo "sent: $(du -sh "$2" | cut -f1)"; rm -rf "$2"' _ \
      "$PWD/.github/scripts/r2-bundle.sh" "$B" > "$R/$m-r2.log" 2>&1 < /dev/null &
    echo $! > "$R/$m-r2.pid"
    echo "pre-upload $m to the data host: started"
  else
    rm -rf "$B"
  fi
fi
