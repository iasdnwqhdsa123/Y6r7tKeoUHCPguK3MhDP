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
ops_on() { [ -n "${OPS_S3_BUCKET:-}" ] && [ -n "${OPS_S3_ENDPOINT:-}" ] && [ -n "${OPS_S3_KEY:-}" ] && [ -n "${OPS_S3_SECRET:-}" ]; }
opss3() {
  AWS_ACCESS_KEY_ID="$OPS_S3_KEY" AWS_SECRET_ACCESS_KEY="$OPS_S3_SECRET" AWS_DEFAULT_REGION=auto \
    AWS_MAX_ATTEMPTS=5 aws --endpoint-url "$OPS_S3_ENDPOINT" "$@"
}
arch_bucket() { if [ "$1" = data ]; then echo "$DATA_S3_BUCKET"; else echo "$ARCHIVE_S3_BUCKET"; fi; }
arch_prefix() { if [ "$1" = data ]; then echo "archive/"; else echo "${DATA_PATH_SECRET:+$DATA_PATH_SECRET/}"; fi; }
archive_send() {
  local d="$1" staged="$2" B P ROOT day model run there keys list try
  B="$(arch_bucket "$d")" P="$(arch_prefix "$d")"
  ROOT="s3://$B${P:+/${P%/}}"
  while read -r day model run; do
    [ -n "$day" ] || continue
    there="$(arch_s3 "$d" s3 cp "$ROOT/$day/$model/latest.json" - 2> /dev/null | jq -r '.runId // ""' 2> /dev/null || true)"
    if [[ -n "$there" && "$there" > "$run" ]]; then echo "archive ($d): $model $day has the newer $there run: $run not sent"; continue; fi
    for try in 1 2 3; do
      arch_s3 "$d" s3 sync "$A/$day/$model/" "$ROOT/$day/$model/" --only-show-errors --delete --exclude latest.json --cache-control "public, max-age=3600" && break
      [ "$try" -lt 3 ] || return 1
      echo "archive ($d): $model $day: an upload failed, the sync again"
      sleep 10
    done
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
  .github/scripts/fetch-artifact.sh "out-$n" "$3" "$t" || { echo "out-$n not fetched: $(df -h --output=avail / | tail -1 | tr -d ' ') left on the disk"; rm -rf "$t"; return 1; }
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
through() {
  local left
  left="$(gh api --paginate "repos/$R/actions/runs/$RUN/attempts/$ATTEMPT/jobs?per_page=100" \
    --jq '.jobs[] | select(.name != "publish" and .name != "archive" and .name != "cleanup" and .status != "completed") | .name' 2> /dev/null)" || return 1
  [ -z "$left" ]
}
case "$1" in
sync)
  r2_on || { echo "::error::no data host bucket (DATA_S3_BUCKET and an endpoint): nothing to publish to"; exit 1; }
  mkdir -p /tmp/outs dist-data
  names="$(jq -r '.[]' <<<"$ACTIVE") mapdata"
  fails=0
  for i in $(seq 1 600); do # every 30 s, to the job's limit
    end=false
    if through; then end=true; fi
    live="$(r2_outs)"
    if [ -z "$live" ]; then
      if [ "$end" != true ]; then echo "the data host's _outs.json not read: again in 30 s"; sleep 30; continue; fi
      live='{}'
    fi
    want="$(newest $names)"
    new=""
    while read -r n id run; do
      [ -n "$n" ] || continue
      [ "$(jq -r --arg n "$n" '.[$n] // ""' <<<"$live" 2> /dev/null)" = "$id" ] || new="$new $n"
    done <<<"$want"
    if [ -n "$new" ] && { [ "$fails" -lt 3 ] || [ "$end" != true ]; }; then
      echo "::group::data host:$new (newer than its bucket's)"
      ok=true sent='{}'
      while read -r n id run; do
        [ -n "$n" ] && [[ " $new " == *" $n "* ]] || continue
        if [ "$(cat "/tmp/outs/$n" 2> /dev/null)" != "$id" ]; then
          take "$n" "$id" "$run" || { echo "::warning::out-$n of run $run couldn't be fetched"; ok=false; continue; }
        fi
        sent="$(jq -c --arg n "$n" --argjson id "$id" '. + {($n): $id}' <<<"$sent")"
      done <<<"$want"
      if [ "$sent" != '{}' ]; then
        cur="$(r2_outs)"
        jq -n --argjson a "${cur:-$live}" --argjson b "$sent" '$a + $b' > dist-data/_outs.json
        if "$0" r2; then
          for n in $(jq -r 'keys[]' <<<"$sent"); do
            if [ "$n" = mapdata ]; then rm -rf dist-data/airspace; else rm -rf "dist-data/$n" "dist-data/free/$n" "dist-data/free/now/$n"; fi
            rm -f "/tmp/outs/$n"
          done
        else ok=false; echo "::warning::the data host's upload failed"; fi
        rm -f dist-data/_outs.json
      fi
      echo "::endgroup::"
      if [ "$ok" = true ]; then fails=0; sleep 15; else fails=$((fails + 1)); sleep 60; fi
      continue
    fi
    if [ "$end" = true ]; then
      if [ -z "$new" ]; then echo "this run's jobs are through and the data host has the newest of each"
      else echo "::warning::this run's jobs are through, but$new didn't reach the data host: left to the next run"; fi
      exit 0
    fi
    [ $((i % 10)) -ne 1 ] || echo "waiting: $(gh api "repos/$R/actions/runs/$RUN/attempts/$ATTEMPT/jobs?per_page=100" --jq '[.jobs[] | select(.name != "publish" and .name != "archive" and .name != "cleanup" and .status != "completed") | .name] | join(", ")' 2> /dev/null || true)"
    sleep 30
  done
  echo "::warning::still not through at the job's limit"
  exit 1
  ;;
r2)
  shift
  r2_on || { echo "data host: no bucket set"; exit 0; }
  D="s3://$DATA_S3_BUCKET/data"
  aws configure set default.s3.max_concurrent_requests 32
  aws configure set default.s3.multipart_threshold 64MB
  t0=$SECONDS
  "$(dirname "$0")/r2-bundle.sh" --all dist-data || exit 1
  roots=()
  if [ "$#" -eq 0 ]; then
    r2s3 s3 sync dist-data/ "$D/" --only-show-errors --no-progress \
      --exclude '*_hours/*' --exclude '*_quarters/*' --exclude '*.building/*' --exclude '*.changed*' --exclude '.model-*' --exclude '.mapdata' \
      --exclude '*latest.json' --exclude '_outs.json' --exclude '*/20[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]*/*' || exit 1
    roots=(dist-data)
  else
    for b in "$@"; do
      [ -d "dist-data/$b" ] || continue
      r2s3 s3 sync "dist-data/$b/" "$D/$b/" --only-show-errors --no-progress \
        --exclude '_hours/*' --exclude '_quarters/*' --exclude '*.building/*' --exclude '.*' --exclude '*/.*' \
        --exclude '*latest.json' --exclude '*latest.json.before' --exclude '20[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]*/*' || exit 1
      roots+=("dist-data/$b")
    done
    [ "${#roots[@]}" -gt 0 ] || { echo "data host: none of $* here"; exit 0; }
  fi
  sent=0 kept=0
  while IFS= read -r f; do
    rel="${f#dist-data/}"
    mine="$(jq -r '.generated // ""' "$f" 2> /dev/null || true)"
    there="$(r2s3 s3 cp "$D/$rel" - 2> /dev/null | jq -r '.generated // ""' 2> /dev/null || true)"
    if [[ -n "$there" && "$there" > "$mine" ]]; then kept=$((kept + 1)); continue; fi
    r2s3 s3 cp "$f" "$D/$rel" --only-show-errors --content-type application/json || exit 1
    sent=$((sent + 1))
  done < <(find "${roots[@]}" -name latest.json -not -path '*/_hours/*' -not -path '*/_quarters/*' -not -path '*.building/*' | sort)
  if [ "$#" -eq 0 ] && [ -f dist-data/_outs.json ]; then r2s3 s3 cp dist-data/_outs.json "$D/_outs.json" --only-show-errors --content-type application/json || exit 1; fi
  echo "data host: sent in $((SECONDS - t0)) s, $sent manifests ($kept newer there kept)"
  "$(dirname "$0")/r2-bundle.sh" --prune "$@" || echo "::warning::data host: older runs not all deleted (again next publish)"
  ;;
model)
  m="${2:?model}" bases=""
  for b in "$m" "free/$m" "free/now/$m"; do [ ! -d "dist-data/$b" ] || bases="$bases $b"; done
  [ -n "$bases" ] || { echo "data host: nothing of $m here"; exit 0; }
  exec "$0" r2 $bases
  ;;
record)
  n="${2:?name}" id="${3:-}"
  r2_on || exit 0
  [[ "$id" =~ ^[0-9]+$ ]] || { echo "::warning::out-$n: no artifact id ($id): not recorded"; exit 0; }
  cur="$(r2_outs)"
  [ -n "$cur" ] || { echo "::warning::the data host's _outs.json not read: out-$n not recorded (the publish sends it)"; exit 0; }
  jq -n --argjson a "$cur" --arg n "$n" --argjson id "$id" '$a + {($n): $id}' > /tmp/outs-record.json
  r2s3 s3 cp /tmp/outs-record.json "s3://$DATA_S3_BUCKET/data/_outs.json" --only-show-errors --content-type application/json
  echo "data host: out-$n $id recorded"
  ;;
archive)
  shift
  dests=""
  if old_archive_on; then dests="old"; fi
  if r2_on; then dests="$dests data"; fi
  [ -n "$dests" ] || exit 0
  keep=""
  while read -r day m run when; do
    [ -n "$m" ] || continue
    if [ "$when" = early ] && r2_on && r2s3 s3api head-object --bucket "$DATA_S3_BUCKET" --key "archive/$day/$m/latest.json" > /dev/null 2>&1; then
      echo "archive: $m $day kept as it is ($run: sent again only in the window's last hours)"
      continue
    fi
    keep="$keep $m"
  done < <(python3 pipeline/archive.py when --dist dist-data "$@")
  [ -n "$keep" ] || exit 0
  set -- $keep
  aws configure set default.s3.max_concurrent_requests 32
  aws configure set default.s3.multipart_threshold 64MB
  aws configure set default.s3.multipart_chunksize 64MB
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
ops)
  m="${2:?model}" run="${3:-}"
  ops_on || { echo "ops: no bucket set"; exit 0; }
  [[ "$run" =~ ^[0-9]{10}$ ]] || { echo "::warning::ops: $m: no run ($run)"; exit 0; }
  day="${run:0:8}" hh="${run:8:2}" O="${RUNNER_TEMP:-/tmp}/ops"
  if [ "${FORCE:-}" != true ]; then
    if there="$(opss3 s3api list-objects-v2 --bucket "$OPS_S3_BUCKET" --prefix "ops/$m/$day/$hh/" --query "Contents[?ends_with(Key, '/done.json')].Key" --output text 2> /tmp/ops-ls.err)"; then
      for k in $there; do
        [ "$k" != None ] || continue
        if opss3 s3 cp "s3://$OPS_S3_BUCKET/$k" - 2> /dev/null | jq -e '.complete != false' > /dev/null 2>&1; then
          echo "ops: $m $run recorded already ($k)"; exit 0
        fi
      done
    else
      echo "::warning::ops: $m $run: the bucket not listed, recorded again: $(head -c 300 /tmp/ops-ls.err)"
    fi
  fi
  grids=""
  if opss3 s3api head-object --bucket "$OPS_S3_BUCKET" --key "ops/$m/grids/$day.json" > /dev/null 2> /tmp/ops-head.err; then grids=--day-has-grids
  elif ! grep -q "404\|NoSuchKey\|Not Found" /tmp/ops-head.err; then
    echo "::warning::ops: $m $run: the day's grids marker not read, the grids left out: $(head -c 300 /tmp/ops-head.err)"; grids=--day-has-grids
  fi
  sent="$(cat "${RUNNER_TEMP:-/tmp}/sent-$m" 2> /dev/null || true)"
  rm -rf "$O"
  python3 pipeline/opslake.py write "dist-data/$m/_hours" "$run" "$O" --model "$m" --build "$RUN-$ATTEMPT" \
    ${PUBLISHED:+--published} ${sent:+--published-t "$sent"} $grids || { echo "::warning::ops: $m $run not written"; rm -rf "$O"; exit 0; }
  put() { # key, file: written once
    local err
    if err="$(opss3 s3api put-object --bucket "$OPS_S3_BUCKET" --key "$1" --body "$2" --if-none-match '*' 2>&1 > /dev/null)"; then return 0; fi
    if grep -q "PreconditionFailed\|ConditionalRequestConflict\|(412)\|(409)" <<<"$err"; then echo "ops: $1 there already"; return 0; fi
    echo "::warning::ops: $1 not sent: $err"
    return 1
  }
  t0=$SECONDS ok=true
  while IFS= read -r f; do put "${f#"$O"/}" "$f" || ok=false; done < <(find "$O" -type f -not -name done.json -not -path "*/grids/*" | sort)
  if [ "$ok" = true ]; then
    while IFS= read -r f; do put "${f#"$O"/}" "$f" || ok=false; done < <(find "$O" -type f -name done.json | sort)
  fi
  if [ "$ok" = true ]; then
    while IFS= read -r f; do put "${f#"$O"/}" "$f" || ok=false; done < <(find "$O" -type f -path "*/grids/*" | sort)
  fi
  echo "ops: $m $run: $(du -sh "$O" | cut -f1) in $((SECONDS - t0)) s"
  rm -rf "$O"
  [ "$ok" = true ] || exit 1
  ;;
*) echo "usage: publish-data.sh model|record|sync|r2|archive|archive-run|ops" >&2; exit 2 ;;
esac
