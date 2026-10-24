#!/usr/bin/env bash
set -uo pipefail
cmd="$1" key="$2" arg="$3"
repo="$GITHUB_REPOSITORY"
case "$cmd" in
  restore)
    mkdir -p "$arg"
    id="$(gh api "repos/$repo/actions/artifacts?name=verify-$key&per_page=20" \
      --jq '[.artifacts[] | select(.expired | not)] | sort_by(.created_at) | last | select(. != null) | .id')"
    if [ -z "$id" ]; then echo "verify-$key: none yet"; exit 0; fi
    tmp="$(mktemp -d)"
    if ! gh api "repos/$repo/actions/artifacts/$id/zip" > "$tmp/a.zip" || ! unzip -q -o "$tmp/a.zip" -d "$tmp"; then
      echo "::warning::verify-$key didn't download: starting it afresh"; exit 0
    fi
    f="$(ls "$tmp"/verify-"$key".tar* 2> /dev/null | head -1)"
    case "$f" in
      *.enc) openssl enc -d -aes-256-ctr -pbkdf2 -pass env:DATA_PATH_SECRET -in "$f" | tar -C "$arg" -xf - ;;
      ?*) tar -C "$arg" -xf "$f" ;;
    esac || echo "::warning::verify-$key didn't unpack"
    rm -rf "$tmp"
    echo "verify-$key: $(ls "$arg" | wc -l) files"
    ;;
  pack)
    mkdir -p /tmp/verify-out "$arg"
    if [ -n "${DATA_PATH_SECRET:-}" ]; then
      tar -C "$arg" -cf - . | openssl enc -aes-256-ctr -pbkdf2 -pass env:DATA_PATH_SECRET -out "/tmp/verify-out/verify-$key.tar.enc"
    else tar -C "$arg" -cf "/tmp/verify-out/verify-$key.tar" .; fi
    ls -l /tmp/verify-out
    ;;
  drop)
    gh api "repos/$repo/actions/artifacts?name=verify-$key&per_page=100" --jq '.artifacts[] | .id' |
      while read -r id; do
        [ "$id" = "$arg" ] || gh api -X DELETE "repos/$repo/actions/artifacts/$id" > /dev/null
      done
    ;;
esac
