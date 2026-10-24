#!/usr/bin/env bash
set -euo pipefail
name="$1" run="$2" dest="$3"
PARTS="${PARTS:-16}"
repo="${GITHUB_REPOSITORY:?}"
api="https://api.github.com/repos/$repo"
meta="$(gh api "repos/$repo/actions/runs/$run/artifacts?name=$name" --jq '[.artifacts[] | select(.expired | not)] | first | select(. != null) | "\(.id) \(.size_in_bytes) \(.digest // "")"')"
[ -n "$meta" ] || { echo "no $name in run $run"; exit 1; }
read -r id size digest <<<"$meta"
link() { curl -fsS -o /dev/null -w '%{redirect_url}' -H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/vnd.github+json" "$api/actions/artifacts/$id/zip"; }
tmp="$(mktemp -d "${RUNNER_TEMP:-/tmp}/fetch.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT
t0=$(date +%s)
part() {
  local i="$1" from="$2" to="$3" f="$tmp/p$1" n have u
  n=$((to - from + 1))
  : > "$f"
  for _ in 1 2 3 4 5 6; do
    have=$(stat -c %s "$f")
    [ "$have" -lt "$n" ] || break
    u="$(link)" || { sleep 5; continue; }
    curl -fsS --max-time 3600 --speed-limit 50000 --speed-time 60 -r "$((from + have))-$to" "$u" >> "$f" || sleep 5
  done
  [ "$(stat -c %s "$f")" -eq "$n" ] || { echo "$name: part $i has $(stat -c %s "$f") of $n bytes"; return 1; }
}
chunk=$(((size + PARTS - 1) / PARTS))
pids=()
for ((i = 0; i < PARTS; i++)); do
  from=$((i * chunk)); [ "$from" -lt "$size" ] || break
  to=$((from + chunk - 1)); [ "$to" -lt "$size" ] || to=$((size - 1))
  part "$i" "$from" "$to" & pids+=($!)
done
ok=true
for p in "${pids[@]}"; do wait "$p" || ok=false; done
[ "$ok" = true ] || exit 1
for ((i = 0; i < ${#pids[@]}; i++)); do cat "$tmp/p$i" >> "$tmp/a.zip"; rm -f "$tmp/p$i"; done
if [ -n "$digest" ]; then
  got="sha256:$(sha256sum "$tmp/a.zip" | cut -d' ' -f1)"
  [ "$got" = "$digest" ] || { echo "$name: digest $got, expected $digest"; exit 1; }
fi
mkdir -p "$tmp/out"
unzip -q "$tmp/a.zip" -d "$tmp/out"
rm -f "$tmp/a.zip"
mkdir -p "$dest"
(cd "$tmp/out" && find . -mindepth 1 -maxdepth 1 -exec mv -t "$dest" {} +)
s=$(($(date +%s) - t0))
echo "$name (run $run): $((size / 1048576)) MB in ${s}s over ${#pids[@]} connections"
