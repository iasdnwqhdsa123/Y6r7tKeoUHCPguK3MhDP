#!/usr/bin/env bash
set -uo pipefail
M="$1"; N="$2"; WAIT_MIN="${WAIT_MIN:-50}"
repo="$GITHUB_REPOSITORY"; run="$GITHUB_RUN_ID"
hours="dist-data/$M/_hours"
mkdir -p "$hours"
artifact() { gh api "repos/$repo/actions/runs/$run/artifacts?name=hours-$M-$1" --jq '.artifacts[0].id // empty' 2> /dev/null; }
fetch() {
  local i="$1" id="$2" z="/tmp/hours-$M-$1.zip"
  gh api "repos/$repo/actions/artifacts/$id/zip" > "$z" || { echo "::warning::hour shard $i: download failed"; return; }
  local f; f="$(unzip -Z1 "$z" | head -1)"
  case "$f" in
    *.enc) unzip -p "$z" "$f" | openssl enc -d -aes-256-ctr -pbkdf2 -pass env:DATA_PATH_SECRET | tar -C "$hours" -xf - ;;
    *) unzip -p "$z" "$f" | tar -C "$hours" -xf - ;;
  esac && echo "hour shard $i: $(unzip -Zt "$z" | awk '{print int($3 / 1048576)}') MB in" || echo "::warning::hour shard $i didn't unpack"
  rm -f "$z"
}
left="$(seq 1 $((N - 1)) | tr '\n' ' ')"
end=$(($(date +%s) + WAIT_MIN * 60))
t0=$(date +%s)
while [ -n "${left// /}" ] && [ "$(date +%s)" -lt "$end" ]; do
  next=""
  for i in $left; do
    id="$(artifact "$i")"
    if [ -z "$id" ]; then
      st="$(gh api "repos/$repo/actions/runs/$run/jobs?per_page=100" --jq ".jobs[] | select(.name == \"hours ($i)\") | .status" 2> /dev/null)"
      if [ "$st" = completed ]; then id="$(artifact "$i")"; [ -n "$id" ] || { echo "::warning::hour shard $i ended without its hours"; continue; }; fi
    fi
    if [ -n "$id" ]; then fetch "$i" "$id"; else next="$next $i"; fi
  done
  left="$next"
  if [ -n "${left// /}" ]; then sleep 20; fi
done
[ -z "${left// /}" ] || echo "::warning::hour shards$left not in after $WAIT_MIN min: the build computes their hours"
echo "waited $(($(date +%s) - t0)) s for the hour shards"
