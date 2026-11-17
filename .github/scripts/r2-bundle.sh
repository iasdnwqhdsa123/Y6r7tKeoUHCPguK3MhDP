#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUCKET="${DATA_S3_BUCKET:?DATA_S3_BUCKET}"
AWS_CONFIG_FILE="$(mktemp)"
export AWS_CONFIG_FILE
trap 'rm -f "$AWS_CONFIG_FILE"' EXIT
printf '[default]\ns3 =\n  multipart_threshold = 64MB\n  multipart_chunksize = 64MB\n  max_concurrent_requests = 10\n' > "$AWS_CONFIG_FILE"
s3() {
  AWS_ACCESS_KEY_ID="${DATA_S3_KEY:-${ARCHIVE_S3_KEY:-}}" AWS_SECRET_ACCESS_KEY="${DATA_S3_SECRET:-${ARCHIVE_S3_SECRET:-}}" AWS_DEFAULT_REGION=auto \
    AWS_MAX_ATTEMPTS=5 aws --endpoint-url "${DATA_S3_ENDPOINT:-${ARCHIVE_S3_ENDPOINT:-}}" "$@"
}
RUN='20[0-9]{8}[A-Za-z0-9_-]*'
BASE='(free/(now/)?)?[a-z0-9]+'
send() {
  local dir="$1" key="$2" fp there first id idx size n
  [[ "$key" =~ ^data/$BASE/$RUN$ ]] || { echo "r2-bundle: $key: not a run folder" >&2; return 2; }
  [ -d "$dir" ] || { echo "r2-bundle: $dir: no folder" >&2; return 2; }
  fp="$(python3 "$HERE/bundle.py" fp "$dir")" && [ -n "$fp" ] || { echo "r2-bundle: $dir: no fingerprint" >&2; return 1; }
  there="$(s3 s3 cp "s3://$BUCKET/b/$key.json" - 2> /dev/null | jq -r '.fp // ""' 2> /dev/null || true)"
  if [ "$there" = "$fp" ]; then
    echo "${key#data/}: there already"
    return 0
  fi
  first="$(cd "$dir" && find . -type f ! -name '.*' ! -name '*.tmp' ! -path '*/.*' ! -path '*/_hours/*' ! -path '*/_quarters/*' ! -path '*.building/*' | LC_ALL=C sort | sed -n 1p)"
  if [ -z "$there" ] && [ -n "$first" ] && s3 s3api head-object --bucket "$BUCKET" --key "$key/${first#./}" > /dev/null 2>&1; then
    echo "${key#data/}: there as files"
    return 0
  fi
  id="$(date -u +%Y%m%d%H%M%S)$(printf '%04x' "$RANDOM")"
  idx="$(mktemp)"
  python3 "$HERE/bundle.py" index "$dir" --bundle "b/$key.$id.bundle" --id "$id" > "$idx" && size="$(jq -e .size "$idx")" && n="$(jq '.files | length' "$idx")" ||
    { rm -f "$idx"; echo "r2-bundle: $dir: no index" >&2; return 1; }
  if python3 "$HERE/bundle.py" cat "$dir" "$idx" | s3 s3 cp - "s3://$BUCKET/b/$key.$id.bundle" --expected-size "$size" --only-show-errors --content-type application/octet-stream &&
    s3 s3 cp "$idx" "s3://$BUCKET/b/$key.json" --only-show-errors --content-type application/json; then
    echo "${key#data/}: $n files, $((size / 1000000)) MB as one"
    rm -f "$idx"
  else
    rm -f "$idx"
    echo "r2-bundle: ${key#data/} not sent" >&2
    return 1
  fi
}
all() {
  local root="${1%/}" base runs cur pick r rc=0
  [ -d "$root" ] || { echo "r2-bundle: $root: no folder" >&2; return 2; }
  for base in $(cd "$root" && find . -mindepth 1 -maxdepth 3 -type d | sed 's#^\./##' | grep -E "^$BASE\$" | grep -vx 'free\|free/now' | sort); do
    runs="$(cd "$root/$base" && find . -mindepth 1 -maxdepth 1 -type d | sed 's#^\./##' | grep -E "^$RUN\$" | sort || true)"
    [ -n "$runs" ] || continue
    cur="$(jq -r '.runId // ""' "$root/$base/latest.json" 2> /dev/null || true)"
    if [ -n "$cur" ] && grep -qxF -- "$cur" <<< "$runs"; then
      pick="$(awk -v c="$cur" '($0 "") >= (c "")' <<< "$runs")"
    else
      pick="$(tail -n 1 <<< "$runs")"
    fi
    for r in $pick; do send "$root/$base/$r" "data/$base/$r" || rc=1; done
  done
  return $rc
}
KEEP_PREV_H="${KEEP_PREV_H:-2}"
prune() {
  local bases base cur bk fd names prev r k cutoff
  cutoff="$(date -u -d "$KEEP_PREV_H hours ago" +%Y-%m-%dT%H:%M:%S)"
  fresh() {
    local t
    t="$(s3 s3api head-object --bucket "$BUCKET" --key "$1" --query LastModified --output text 2> /dev/null)" || return 0
    [[ ! "$t" =~ ^20[0-9]{2}-[0-9]{2}-[0-9]{2}T ]] || [[ "$t" > "$cutoff" ]]
  }
  dirs() { s3 s3api list-objects-v2 --bucket "$BUCKET" --prefix "$1" --delimiter / --query 'CommonPrefixes[].Prefix' --output text 2> /dev/null | tr '\t' '\n' | grep -v '^None$' | grep . || true; }
  keys() { s3 s3api list-objects-v2 --bucket "$BUCKET" --prefix "$1" --delimiter / --query 'Contents[].Key' --output text 2> /dev/null | tr '\t' '\n' | grep -v '^None$' | grep . || true; }
  if [ "$#" -gt 0 ]; then
    bases="$(printf 'data/%s/\n' "${@%/}")"
  else
    bases="$(dirs data/ | grep -vx 'data/free/\|data/airspace/' || true; dirs data/free/ | grep -vx 'data/free/now/' || true; dirs data/free/now/)"
  fi
  for base in $bases; do
    [[ "$base" =~ ^data/$BASE/$ ]] || continue
    cur="$(s3 s3 cp "s3://$BUCKET/${base}latest.json" - 2> /dev/null | jq -r '.runId // ""' 2> /dev/null || true)"
    [[ "$cur" =~ ^$RUN$ ]] || continue
    bk="$(keys "b/$base")"
    fd="$(dirs "$base" | sed -nE "s#^${base}($RUN)/\$#\\1#p")"
    names="$({ sed -nE "s#^b/${base}($RUN)\\.(json|[0-9a-f]+\\.bundle)\$#\\1#p" <<< "$bk"; printf '%s\n' "$fd"; } | grep . | sort -u || true)"
    prev="$(awk -v c="$cur" '($0 "") < (c "")' <<< "$names" | tail -n 1)"
    if [ -n "$prev" ] && ! fresh "b/${base}${cur}.json"; then prev=""; fi
    for r in $names; do
      if [[ "$r" == "$cur" || "$r" > "$cur" || "$r" == "$prev" ]]; then continue; fi
      if grep -qxF "b/$base$r.json" <<< "$bk"; then s3 s3 rm "s3://$BUCKET/b/$base$r.json" --only-show-errors; fi
      for k in $(grep -E "^b/${base}${r}\\.[0-9a-f]+\\.bundle\$" <<< "$bk" || true); do s3 s3 rm "s3://$BUCKET/$k" --only-show-errors; done
      if grep -qxF -- "$r" <<< "$fd"; then s3 s3 rm "s3://$BUCKET/$base$r/" --recursive --only-show-errors; fi
      echo "data host: ${base#data/}$r deleted"
    done
  done
}
STALE_UPLOAD_H="${STALE_UPLOAD_H:-24}"
stale_uploads() {
  local list cutoff k id t n=0
  list="$(s3 s3api list-multipart-uploads --bucket "$BUCKET" --query 'Uploads[].[Key,UploadId,Initiated]' --output text 2> /dev/null)" ||
    { echo "data host: unfinished uploads not listed" >&2; return 0; }
  cutoff="$(date -u -d "$STALE_UPLOAD_H hours ago" +%Y-%m-%dT%H:%M:%S)"
  while IFS=$'\t' read -r k id t; do
    [ -n "${id:-}" ] && [ "$k" != None ] || continue
    [[ "$t" < "$cutoff" ]] || continue
    if s3 s3api abort-multipart-upload --bucket "$BUCKET" --key "$k" --upload-id "$id" > /dev/null 2>&1; then n=$((n + 1)); fi
  done <<< "$list"
  if [ "$n" -gt 0 ]; then echo "data host: $n unfinished uploads over ${STALE_UPLOAD_H} h old aborted"; fi
  return 0
}
case "${1:-}" in
--all) all "${2:?--all ROOT}" ;;
--prune) shift; prune "$@"; stale_uploads ;;
"" | -*) echo "usage: r2-bundle.sh DIR KEY | --all ROOT | --prune" >&2; exit 2 ;;
*) send "$1" "${2:?KEY}" ;;
esac
