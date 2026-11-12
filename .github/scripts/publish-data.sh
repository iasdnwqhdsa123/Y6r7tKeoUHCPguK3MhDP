#!/usr/bin/env bash
set -euo pipefail
R="${GITHUB_REPOSITORY:?}" RUN="${FORECAST_RUN:-${GITHUB_RUN_ID:?}}" ATTEMPT="${GITHUB_RUN_ATTEMPT:-1}"
P="${DATA_PATH_SECRET:+/$DATA_PATH_SECRET}"
r2_on() { [ -n "${DATA_S3_BUCKET:-}" ] && [ -n "${DATA_S3_ENDPOINT:-${ARCHIVE_S3_ENDPOINT:-}}" ]; }
r2s3() {
  AWS_ACCESS_KEY_ID="${DATA_S3_KEY:-${ARCHIVE_S3_KEY:-}}" AWS_SECRET_ACCESS_KEY="${DATA_S3_SECRET:-${ARCHIVE_S3_SECRET:-}}" AWS_DEFAULT_REGION=auto \
    AWS_MAX_ATTEMPTS=5 aws --endpoint-url "${DATA_S3_ENDPOINT:-$ARCHIVE_S3_ENDPOINT}" "$@"
}
r2_outs() {
  local t
  t="$(r2s3 s3 cp "s3://$DATA_S3_BUCKET/data/_outs.json" - 2> /tmp/r2-outs.err)" || { grep -q "404\|NoSuchKey\|Not Found" /tmp/r2-outs.err && echo '{}'; return 0; }
  jq -e 'type == "object"' <<<"$t" > /dev/null 2>&1 && echo "$t"
}
old_archive_on() { [ -n "${ARCHIVE_S3_BUCKET:-}" ] && [ -n "${ARCHIVE_S3_ENDPOINT:-}" ]; }
arch_s3() {
  local d="$1"
  shift
  if [ "$d" = data ]; then r2s3 "$@"; else
    AWS_ACCESS_KEY_ID="${ARCHIVE_S3_KEY:-}" AWS_SECRET_ACCESS_KEY="${ARCHIVE_S3_SECRET:-}" AWS_DEFAULT_REGION=auto \
      AWS_MAX_ATTEMPTS=5 aws --endpoint-url "$ARCHIVE_S3_ENDPOINT" "$@"
  fi
}
arch_bucket() { if [ "$1" = data ]; then echo "$DATA_S3_BUCKET"; else echo "$ARCHIVE_S3_BUCKET"; fi; }
arch_prefix() { if [ "$1" = data ]; then echo "archive/"; else echo "${DATA_PATH_SECRET:+$DATA_PATH_SECRET/}"; fi; }
archive_send() {
  local d="$1" staged="$2" B P ROOT day model run there keys list
  B="$(arch_bucket "$d")" P="$(arch_prefix "$d")"
  ROOT="s3://$B${P:+/${P%/}}"
  while read -r day model run; do
    [ -n "$day" ] || continue
    there="$(arch_s3 "$d" s3 cp "$ROOT/$day/$model/latest.json" - 2> /dev/null | jq -r '.runId // ""' 2> /dev/null || true)"
    if [[ -n "$there" && "$there" > "$run" ]]; then echo "archive ($d): $model $day has the newer $there run: $run not sent"; continue; fi
    arch_s3 "$d" s3 sync "$A/$day/$model/" "$ROOT/$day/$model/" --only-show-errors --delete --exclude latest.json --cache-control "public, max-age=3600" || return 1
    arch_s3 "$d" s3 cp "$A/$day/$model/latest.json" "$ROOT/$day/$model/latest.json" --only-show-errors --cache-control "public, max-age=300" --content-type application/json || return 1
    echo "archive ($d): $model $day from the $run run ($(du -sh "$A/$day/$model" | cut -f1))"
  done <<<"$staged"
  keys="/tmp/archive-keys-$d.txt"
  list="$(arch_s3 "$d" s3api list-objects-v2 --bucket "$B" --prefix "$P" --query "Contents[?ends_with(Key, '/latest.json') && !contains(Key, '/req/')].Key" --output text)" || return 1
  tr '\t' '\n' <<<"$list" > "$keys"
  for day in $(python3 pipeline/archive.py old --keys "$keys"); do
    arch_s3 "$d" s3 rm "$ROOT/$day/" --recursive --only-show-errors && echo "archive ($d): $day deleted (past the month)"
  done
  python3 pipeline/archive.py index --keys "$keys" > "/tmp/archive-index-$d.json"
  arch_s3 "$d" s3 cp "/tmp/archive-index-$d.json" "$ROOT/index.json" --only-show-errors --cache-control "no-cache" --content-type application/json || return 1
  echo "archive ($d): $(jq '.days | length' "/tmp/archive-index-$d.json") days listed"
}
archive_seed() {
  local OP list pre day n=0
  r2s3 s3api head-object --bucket "$DATA_S3_BUCKET" --key archive/index.json > /dev/null 2>&1 && return 0
  old_archive_on || return 0
  OP="$(arch_prefix old)"
  list="$(arch_s3 old s3api list-objects-v2 --bucket "$ARCHIVE_S3_BUCKET" --prefix "$OP" --delimiter / --query 'CommonPrefixes[].Prefix' --output text)" || return 1
  echo "archive: the month's days copied into the data host's bucket (once)"
  for pre in $(tr '\t' '\n' <<<"$list" | grep -E '(^|/)[0-9]{8}/$' || true); do
    day="$(basename "$pre")"
    if ! r2s3 s3 sync "s3://$ARCHIVE_S3_BUCKET/$pre" "s3://$DATA_S3_BUCKET/archive/$day/" --only-show-errors > /dev/null 2>&1; then
      rm -rf /tmp/archive-seed
      arch_s3 old s3 sync "s3://$ARCHIVE_S3_BUCKET/$pre" /tmp/archive-seed/ --only-show-errors || return 1
      r2s3 s3 sync /tmp/archive-seed/ "s3://$DATA_S3_BUCKET/archive/$day/" --only-show-errors || return 1
      rm -rf /tmp/archive-seed
    fi
    n=$((n + 1))
  done
  echo "archive: $n days copied"
}
newest() {
  local n
  for n in "$@"; do
    gh api "repos/$R/actions/artifacts?name=out-$n&per_page=20" \
      --jq "[.artifacts[] | select(.expired | not)] | sort_by(.created_at) | last | select(. != null) | \"$n \\(.id) \\(.workflow_run.id)\"" || true
  done
}
take() { # name id run
  local n="$1" t=/tmp/out-take f
  rm -rf "$t"
  if [ "$n" = mapdata ]; then rm -rf dist-data/airspace; else rm -rf "dist-data/$n" "dist-data/free/$n" "dist-data/free/now/$n"; fi
  rm -f "/tmp/outs/$n"
  .github/scripts/fetch-artifact.sh "out-$n" "$3" "$t" || return 1
  f="$(ls "$t"/*.tar* 2> /dev/null | head -1 || true)"
  [ -n "$f" ] || { echo "out-$n: no archive in it"; rm -rf "$t"; return 1; }
  mkdir -p dist-data
  case "$f" in
    *.enc) openssl enc -d -aes-256-ctr -pbkdf2 -pass env:DATA_PATH_SECRET -in "$f" | tar -C dist-data -xf - ;;
    *) tar -C dist-data -xf "$f" ;;
  esac
  rm -rf "$t" dist-data/.changed*
  echo "$2" > "/tmp/outs/$n"
  echo "out-$n taken: $(df -h --output=avail / | tail -1 | tr -d ' ') left on the disk"
}
fallback() { # name; true if its run from before is in dist-data
  local n="$1" run a f t=/tmp/out-old
  run="$(gh api "repos/$R/actions/artifacts?name=store-run-$n&per_page=20" \
    --jq '[.artifacts[] | select(.expired | not)] | sort_by(.created_at) | last | .workflow_run.id // empty' 2> /dev/null || true)"
  [ -n "$run" ] || return 1
  for a in "store-$n" "store-run-$n"; do
    rm -rf "$t"
    .github/scripts/fetch-artifact.sh "$a" "$run" "$t" || { [ "$a" = "store-$n" ] && continue; return 1; }
    for f in "$t"/*.tar*; do
      [ -e "$f" ] || continue
      case "$f" in
        *.enc) openssl enc -d -aes-256-ctr -pbkdf2 -pass env:DATA_PATH_SECRET -in "$f" | tar -C dist-data -xf - ;;
        *) tar -C dist-data -xf "$f" ;;
      esac
    done
  done
  rm -rf "$t" dist-data/.changed*
  [ -f "dist-data/$n/latest.json" ] || return 1
  echo "$n: no output yet, its run from before ($(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['runId'])" "dist-data/$n/latest.json" 2> /dev/null)) kept"
}
deploying() {
  local since busy
  since="$(date -u -d '20 minutes ago' +%Y-%m-%dT%H:%M:%S)"
  busy="$(curl -fsS --max-time 20 -H "Authorization: Bearer $NETLIFY_AUTH_TOKEN" "https://api.netlify.com/api/v1/sites/$NETLIFY_SITE_ID/deploys?per_page=10" 2> /dev/null |
    jq -r --arg since "$since" '[.[] | select((.state | IN("ready", "error", "rejected")) | not) | select(.created_at >= $since)] | length' 2> /dev/null)" || return 1
  [ "${busy:-0}" -gt 0 ] 2> /dev/null
}
through() {
  local left
  left="$(gh api --paginate "repos/$R/actions/runs/$RUN/attempts/$ATTEMPT/jobs?per_page=100" \
    --jq '.jobs[] | select(.name != "publish" and .name != "archive" and .name != "cleanup" and .status != "completed") | .name' 2> /dev/null)" || return 1
  [ -z "$left" ]
}
case "$1" in
sync)
  mkdir -p /tmp/outs dist-data
  names="$(jq -r '.[]' <<<"$ACTIVE") mapdata"
  done_at=""
  for i in $(seq 1 600); do # every 30 s, to the job's limit
    end=false
    if through; then end=true; fi
    r="$(curl -sS --max-time 20 -H "Cache-Control: no-cache" -w '\n%{http_code}' "$DATA_URL$P/_outs.json" 2> /dev/null || true)"
    code="${r##*$'\n'}" live="${r%$'\n'*}"
    if [ "$code" = 404 ]; then live='{}'
    elif [ "$code" != 200 ] || ! jq -e 'type == "object"' <<<"$live" > /dev/null 2>&1; then
      if [ "$end" != true ]; then echo "the data site's _outs.json not read (HTTP ${code:-none}): again in 30 s"; sleep 30; continue; fi
      live='{}'
    fi
    want="$(newest $names)"
    live_r2='{}' new_r2=""
    if r2_on && [ "${r2_fails:-0}" -lt 3 -o "$end" != true ]; then
      live_r2="$(r2_outs)"
      if [ -z "$live_r2" ]; then echo "the data host's _outs.json not read: taken as none"; live_r2='{}'; fi
    fi
    have_all=true
    for n in $names; do
      if ! grep -q "^$n " <<<"$want"; then
        if [ "$n" != mapdata ] && ! jq -e --arg n "$n" 'index($n) != null' <<<"$REQUIRED" > /dev/null; then
          if [ ! -f "/tmp/old-$n" ]; then
            touch "/tmp/old-$n"
            fallback "$n" || echo "::warning::$n has no output on hand and no run from before: deploying without it"
          fi
        else have_all=false; fi
      fi
    done
    new=""
    while read -r n id run; do
      [ -n "$n" ] || continue
      [ "$(jq -r --arg n "$n" '.[$n] // ""' <<<"$live" 2> /dev/null)" = "$id" ] || new="$new $n"
      if r2_on && [ "${r2_fails:-0}" -lt 3 -o "$end" != true ]; then
        [ "$(jq -r --arg n "$n" '.[$n] // ""' <<<"$live_r2" 2> /dev/null)" = "$id" ] || new_r2="$new_r2 $n"
      fi
    done <<<"$want"
    if [ "$have_all" = true ] && [ -n "$new" ] && deploying; then
      echo "another deploy going out: waiting for it"
      sleep 20
      continue
    fi
    if [ "$have_all" = true ] && [ -z "$new" ] && [ -n "$new_r2" ]; then
      echo "::group::data host:$new_r2 (newer than its bucket's)"
      ok=true
      while read -r n id run; do
        [ -n "$n" ] || continue
        [ "$(cat "/tmp/outs/$n" 2> /dev/null)" = "$id" ] && continue
        take "$n" "$id" "$run" || { echo "::warning::out-$n of run $run couldn't be fetched"; ok=false; }
      done <<<"$want"
      if [ "$ok" = true ]; then
        jq -Rn '[inputs | split(" ") | {(.[0]): (.[1] | tonumber)}] | add' <<<"$want" > dist-data/_outs.json
        "$0" r2 || ok=false
      fi
      if [ "$ok" = true ]; then r2_fails=0; else r2_fails=$((${r2_fails:-0} + 1)); echo "::warning::the data host's upload failed ($r2_fails)"; fi
      echo "::endgroup::"
      if [ "$ok" = true ]; then sleep 15; else sleep 60; fi
      continue
    fi
    if [ "$have_all" = true ] && [ -n "$new" ]; then
      echo "::group::deploy:$new (newer than the data site's)"
      ok=true
      while read -r n id run; do
        [ -n "$n" ] || continue
        [ "$(cat "/tmp/outs/$n" 2> /dev/null)" = "$id" ] && continue
        take "$n" "$id" "$run" || { echo "::warning::out-$n of run $run couldn't be fetched"; ok=false; }
      done <<<"$want"
      if [ "$ok" = true ]; then
        jq -Rn '[inputs | split(" ") | {(.[0]): (.[1] | tonumber)}] | add' <<<"$want" > dist-data/_outs.json
        if [ "$(newest $names)" != "$want" ] || deploying; then
          echo "newer outputs or another deploy meanwhile: read again"
          echo "::endgroup::"
          sleep 10
          continue
        fi
        { "$0" assemble && "$0" deploy "$(xargs <<<"$new")"; } || { echo "::warning::the deploy failed"; ok=false; }
        if [ -n "$new_r2" ]; then
          if "$0" r2; then r2_fails=0; else r2_fails=$((${r2_fails:-0} + 1)); echo "::warning::the data host's upload failed ($r2_fails)"; fi
        fi
      fi
      echo "::endgroup::"
      if [ "$ok" = true ]; then sleep 15; else sleep 60; fi
      continue
    fi
    if [ "$end" = true ]; then
      if [ "$have_all" = true ]; then echo "this run's jobs are through and the data site has the newest of each"
      else echo "::warning::this run's jobs are through, but not every model's output is on hand: no deploy"; fi
      exit 0
    fi
    [ $((i % 10)) -ne 1 ] || echo "waiting: $(gh api "repos/$R/actions/runs/$RUN/attempts/$ATTEMPT/jobs?per_page=100" --jq '[.jobs[] | select(.name != "publish" and .name != "archive" and .name != "cleanup" and .status != "completed") | .name] | join(", ")' 2> /dev/null || true)"
    sleep 30
  done
  echo "::warning::still not through at the job's limit"
  exit 1
  ;;
assemble)
  rm -rf site
  mkdir -p "site$P"
  OPEN_EX=(); if [ -n "${DATA_PATH_SECRET:-}" ]; then OPEN_EX=(--exclude '/free/' --exclude '/airspace/'); fi
  rsync -a --delete --link-dest="$PWD/dist-data/" --exclude '_hours/' --exclude '_quarters/' --exclude '*.building/' --exclude '.changed*' --exclude '.model-*' --exclude '.mapdata' "${OPEN_EX[@]}" dist-data/ "site$P/"
  sed "s#^/#$P/#" pipeline/data-site/_headers > site/_headers
  if [ -n "${DATA_PATH_SECRET:-}" ]; then
    O="$OPEN_DIR"
    for d in free airspace; do
      if [ -d "dist-data/$d" ]; then mkdir -p "site$O/$d"; rsync -a --link-dest="$PWD/dist-data/$d/" --exclude '*.building/' "dist-data/$d/" "site$O/$d/"; fi
    done
    printf '%s/free/*/20*\n  Cache-Control: public, max-age=31536000, immutable\n%s/airspace/*\n  Cache-Control: public, max-age=3600\n%s/*\n  X-Robots-Tag: noindex, nofollow\n' "$O" "$O" "$O" >> site/_headers
  fi
  printf "User-agent: *\nDisallow: /\n" > site/robots.txt
  printf '[build]\n  publish = "."\n' > site/netlify.toml
  du -sh site; find site -type f | wc -l
  ;;
r2)
  r2_on || { echo "data host: no bucket set"; exit 0; }
  KEEP_H=6
  D="s3://$DATA_S3_BUCKET/data"
  aws configure set default.s3.max_concurrent_requests 32
  aws configure set default.s3.multipart_threshold 64MB
  t0=$SECONDS
  r2s3 s3 sync dist-data/ "$D/" --only-show-errors --no-progress \
    --exclude '*_hours/*' --exclude '*_quarters/*' --exclude '*.building/*' --exclude '*.changed*' --exclude '.model-*' --exclude '.mapdata' \
    --exclude '*latest.json' --exclude '_outs.json' || exit 1
  sent=0 kept=0
  while IFS= read -r f; do
    rel="${f#dist-data/}"
    mine="$(jq -r '.generated // ""' "$f" 2> /dev/null || true)"
    there="$(r2s3 s3 cp "$D/$rel" - 2> /dev/null | jq -r '.generated // ""' 2> /dev/null || true)"
    if [[ -n "$there" && "$there" > "$mine" ]]; then kept=$((kept + 1)); continue; fi
    r2s3 s3 cp "$f" "$D/$rel" --only-show-errors --content-type application/json || exit 1
    sent=$((sent + 1))
  done < <(find dist-data -name latest.json -not -path '*/_hours/*' -not -path '*/_quarters/*' -not -path '*.building/*' | sort)
  [ ! -f dist-data/_outs.json ] || r2s3 s3 cp dist-data/_outs.json "$D/_outs.json" --only-show-errors --content-type application/json || exit 1
  echo "data host: sent in $((SECONDS - t0)) s, $sent manifests ($kept newer there kept)"
  dirs() { r2s3 s3api list-objects-v2 --bucket "$DATA_S3_BUCKET" --prefix "$1" --delimiter / --query 'CommonPrefixes[].Prefix' --output text 2> /dev/null | tr '\t' '\n' | grep -v '^None$' | grep . || true; }
  folders="$(dirs data/ | grep -vx 'data/free/\|data/airspace/'; dirs data/free/ | grep -vx 'data/free/now/'; dirs data/free/now/)"
  for base in $folders; do
    cur="$(r2s3 s3 cp "s3://$DATA_S3_BUCKET/${base}latest.json" - 2> /dev/null | jq -r '.runId // ""' 2> /dev/null || true)"
    [[ "$cur" =~ ^20[0-9]{8} ]] || continue
    cut="$(date -u -d "${cur:0:8} ${cur:8:2}:00 $KEEP_H hours ago" +%Y%m%d%H)"
    while IFS= read -r r; do
      [ -n "$r" ] || continue
      t="$(sed -E 's#.*/(20[0-9]{8})[^/]*/$#\1#' <<<"$r")"
      if [ "$t" \< "$cut" ] && [ "${r%/}" != "${base}$cur" ]; then
        r2s3 s3 rm "s3://$DATA_S3_BUCKET/$r" --recursive --only-show-errors && echo "data host: ${r#data/} deleted"
      fi
    done < <(dirs "$base" | grep -E '/20[0-9]{8}[^/]*/$' || true)
  done
  ;;
deploy)
  cd site
  CLI="$(../.github/scripts/netlify-cli.sh)"
  msg="forecast $RUN${2:+ $2}"
  if [ "${ALIAS_ONLY:-}" != true ]; then
    "$CLI" deploy --prod --dir . --functions /tmp/no-functions --message "$msg"
  fi
  "$CLI" deploy --alias "$DATA_ALIAS" --dir . --functions /tmp/no-functions --message "$msg ($DATA_ALIAS)" ||
    { [ "${ALIAS_ONLY:-}" != true ] && echo "::warning::the $DATA_ALIAS draft failed (production deployed)"; } || exit 1
  ;;
archive)
  shift
  dests=""
  if old_archive_on; then dests="old"; fi
  if r2_on; then dests="$dests data"; fi
  [ -n "$dests" ] || exit 0
  aws configure set default.s3.max_concurrent_requests 32
  A=/tmp/archive
  rm -rf "$A"
  PY=/tmp/archive-venv/bin/python
  if [ ! -x "$PY" ]; then
    python3 -m venv /tmp/archive-venv && "$PY" -m pip install --quiet --disable-pip-version-check numpy pillow
  fi
  staged="$(PYTHONPATH=pipeline "$PY" pipeline/archive.py stage --dist dist-data --out "$A" "$@")"
  [ -n "$staged" ] || { echo "archive: no model's morning run in this deploy"; exit 0; }
  rc=0
  for d in $dests; do
    if [ "$d" = data ] && ! archive_seed; then echo "::warning::archive: the month not copied into the data host's bucket yet (again next run)"; rc=1; continue; fi
    archive_send "$d" "$staged" || { echo "::warning::archive: the $d bucket's upload failed"; rc=1; }
  done
  rm -rf "$A"
  exit "$rc"
  ;;
archive-run)
  if ! old_archive_on && ! r2_on; then echo "archive: no bucket set"; exit 0; fi
  models="$(gh api "repos/$R/actions/runs/$RUN/artifacts?per_page=100" \
    --jq '.artifacts[] | select((.name | startswith("out-")) and (.expired | not)) | .name | ltrimstr("out-")' | grep -vx 'mapdata' | sort -u || true)"
  [ -n "$models" ] || { echo "archive: no model built in this run"; exit 0; }
  mkdir -p /tmp/outs
  for m in $models; do
    take "$m" "" "$RUN" || { echo "::warning::out-$m of this run couldn't be fetched: not archived"; continue; }
    "$0" archive "$m" || echo "::warning::$m's archive upload failed"
    rm -rf "dist-data/$m" "dist-data/free/$m" "dist-data/free/now/$m"
  done
  ;;
*) echo "usage: publish-data.sh sync|assemble|deploy|r2|archive|archive-run" >&2; exit 2 ;;
esac
