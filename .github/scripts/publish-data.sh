#!/usr/bin/env bash
set -euo pipefail
R="${GITHUB_REPOSITORY:?}" RUN="${GITHUB_RUN_ID:?}" ATTEMPT="${GITHUB_RUN_ATTEMPT:-1}"
P="${DATA_PATH_SECRET:+/$DATA_PATH_SECRET}"
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
  .github/scripts/fetch-artifact.sh "out-$n" "$3" "$t" || return 1
  f="$(ls "$t"/*.tar* 2> /dev/null | head -1 || true)"
  [ -n "$f" ] || { echo "out-$n: no archive in it"; rm -rf "$t"; return 1; }
  if [ "$n" = mapdata ]; then rm -rf dist-data/airspace; else rm -rf "dist-data/$n" "dist-data/free/$n" "dist-data/free/now/$n"; fi
  mkdir -p dist-data
  case "$f" in
    *.enc) openssl enc -d -aes-256-ctr -pbkdf2 -pass env:DATA_PATH_SECRET -in "$f" | tar -C dist-data -xf - ;;
    *) tar -C dist-data -xf "$f" ;;
  esac
  rm -rf "$t" dist-data/.changed*
  echo "$2" > "/tmp/outs/$n"
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
    --jq '.jobs[] | select(.name != "publish" and .name != "cleanup" and .status != "completed") | .name' 2> /dev/null)" || return 1
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
    done <<<"$want"
    if [ "$have_all" = true ] && [ -n "$new" ] && deploying; then
      echo "another deploy going out: waiting for it"
      sleep 20
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
    [ $((i % 10)) -ne 1 ] || echo "waiting: $(gh api "repos/$R/actions/runs/$RUN/attempts/$ATTEMPT/jobs?per_page=100" --jq '[.jobs[] | select(.name != "publish" and .name != "cleanup" and .status != "completed") | .name] | join(", ")' 2> /dev/null || true)"
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
*) echo "usage: publish-data.sh sync|assemble|deploy" >&2; exit 2 ;;
esac
