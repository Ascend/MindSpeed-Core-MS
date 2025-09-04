#!/usr/bin/env bash
# usage：export commits from MindSpeed-Core-MS and its sub-repositories
# demo: bash ./export_commits_verbose.sh [output_file]   default write to repo_commits.txt

set -euo pipefail
outfile="${1:-repo_commits.txt}"
outfile_abs="$(pwd)/$outfile"     # 固定绝对路径

repos=(
  "MindSpeed-Core-MS"
  "MindSpeed-LLM"
  "MindSpeed-MM"
  "MindSpeed"
  "Megatron-LM"
  "msadapter"
)

printf "%-20s %-40s %-25s %s\n" "#Repo" "CommitID" "CommitTime" "Branch" > "$outfile_abs"

for repo in "${repos[@]}"; do
  if [[ "$repo" == "MindSpeed-Core-MS" ]]; then
    cd .
  else
    if [[ ! -d "$repo/.git" ]]; then
      echo "⚠️  $repo is not a Git repo，skipped" >&2
      continue
    fi
    cd "$repo"
  fi

  commit_id=$(git rev-parse HEAD)
  commit_time=$(git show -s --format=%ci HEAD)
  branch_name=$(git branch --show-current || echo "detached")

  # export to file
  printf "%-20s %-40s %-25s %s\n" "$repo" "$commit_id" "$commit_time" "$branch_name" >> "$outfile_abs"

  cd - > /dev/null
done

echo "✅ export finished：$outfile_abs"
cat "$outfile_abs"
