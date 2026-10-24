#!/usr/bin/env bash
set -euo pipefail
R="${GITHUB_REPOSITORY:?}" RUN="${GITHUB_RUN_ID:?}" ATTEMPT="${GITHUB_RUN_ATTEMPT:-1}"
case "$1" in
wait)
  skip="${2:-}"
  want=$(jq -r '.[]' <<<"$ACTIVE" | grep -vx -- "$skip" || true)
  for i in $(seq 1 300); do # every 30 s: ~150 min (a model job's limit is 120)
    jobs="$(gh api --paginate "repos/$R/actions/runs/$RUN/attempts/$ATTEMPT/jobs?per_page=100" \
      --jq '.jobs[] | select(.name | startswith("model (")) | "\(.name) \(.status) \(.conclusion)"' 2> /dev/null || true)"
    left=""
    for m in $want; do
      st="$(grep -F "model ($m) " <<<"$jobs" | awk '{print $3}' | head -1)"
      [ "$st" = completed ] || left="$left $m"
    done
    if [ -z "$left" ]; then echo "model jobs done: $(grep -F completed <<<"$jobs" | awk '{print $2 $4}' | tr '\n' ' ')"; exit 0; fi
    [ $((i % 10)) -ne 1 ] || echo "waiting for:$left"
    sleep 30
  done
  echo "::warning::still waiting for:$left (going on without them)"
  exit 1
  ;;
hand)
  dest="$2" only="${3:-}"
  names="$(gh api "repos/$R/actions/runs/$RUN/artifacts?per_page=100" --jq '.artifacts[] | select(.name | startswith("model-")) | .name')"
  [ -z "$only" ] || names="$(grep -x -- "$only" <<<"$names" || true)"
  for n in $names; do .github/scripts/fetch-artifact.sh "$n" "$RUN" "$dest"; done
  ;;
unpack)
  mkdir -p dist-data
  for f in "$2"/*.tar.enc "$2"/*.tar; do
    [ -f "$f" ] || continue
    case "$f" in
      *.enc) openssl enc -d -aes-256-ctr -pbkdf2 -pass env:DATA_PATH_SECRET -in "$f" | tar -C dist-data -xf - ;;
      *) tar -C dist-data -xf "$f" ;;
    esac
    echo "$(basename "$f"): unpacked"
  done
  rm -rf "$2"
  if [ -d dist-data/free ]; then find dist-data/free -mindepth 4 -maxdepth 4 -type d -path '*/h/*' -printf '::add-mask::%f\n'; fi
  ;;
assemble)
  P="${DATA_PATH_SECRET:+/$DATA_PATH_SECRET}"
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
*) echo "usage: publish-data.sh wait|hand|unpack|assemble|deploy" >&2; exit 2 ;;
esac
