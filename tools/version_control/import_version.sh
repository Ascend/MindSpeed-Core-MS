#!/usr/bin/env bash
# usage: import commits from repo_commits.txt(only commit ID)
# demo: bash ./checkout_from_commits.sh

set -euo pipefail
mapfile="${1:-repo_commits.txt}"

if [[ ! -f "$mapfile" ]]; then
  echo "❌ version control file not found：$mapfile"
  exit 1
fi

tail -n +2 "$mapfile" | while read -r repo commit_id _; do
  # delete comments and empty lines
  [[ "$repo" =~ ^#.*$ || -z "$repo" ]] && continue

  echo "=== 检出 $repo → $commit_id"

  if [[ "$repo" == "MindSpeed-Core-MS" ]]; then
    git fetch --all --quiet
    git reset --hard --quiet "$commit_id" || { echo "  ↳ checkout failed"; exit 2; }
  elif [[ -d "$repo/.git" ]]; then
    git -C "$repo" fetch --all --quiet
    git -C "$repo" checkout --quiet "$commit_id" || { echo "  ↳ checkout failed"; exit 2; }
  else
    echo "⚠️  $repo not found, or is not a Git repo，skipped"
  fi
done

echo "✅ all repo imported by commit（detached HEAD）"
