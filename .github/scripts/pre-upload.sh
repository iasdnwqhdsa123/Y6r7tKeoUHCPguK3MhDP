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
