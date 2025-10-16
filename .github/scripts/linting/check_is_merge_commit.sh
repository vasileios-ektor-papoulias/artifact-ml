#!/bin/bash
set -euo pipefail

# Purpose:
#   Detect whether the last commit (HEAD) is a merge commit by counting its parents.
#
# Usage:
#   .github/scripts/linting/check_is_merge_commit.sh
#
# Accepts:
#   (none) — operates on the current repository’s HEAD.
#
# Stdout on success:
#   <parents_count>
#     e.g., "2" for a typical merge commit (two parents).
#
# Stderr:
#   - Always prints the last commit subject for context.
#   - On success: "This is a merge commit with <N> parents"
#   - On non-merge: "This is not a merge commit. Skipping linting checks."
#
# Exit codes:
#   0  success — the last commit (HEAD) IS a merge commit
#   1  not a merge commit (single-parent or root commit), or unable to determine
#
# Behaviour:
#   - Reads the last commit’s subject and parent list via `git log -1`.
#   - Counts parents (`%P` split by spaces). If count > 1, it’s a merge commit.
#
# Notes:
#   - Run in a Git repository and ensure HEAD exists.
#   - In GitHub Actions, ensure the commit is present locally (use actions/checkout).
#
# Examples:
#   # In a workflow job, fail early if not a merge commit
#   - name: Ensure this run is on a merge commit
#     run: .github/scripts/linting/check_is_merge_commit.sh

LAST_COMMIT_MESSAGE=$(git log -1 --pretty=format:%s)
echo "Last commit message: $LAST_COMMIT_MESSAGE" >&2

PARENTS_COUNT=$(git log -1 --pretty=format:%P | wc -w)

if [ "$PARENTS_COUNT" -le 1 ]; then
  echo "This is not a merge commit. Skipping linting checks." >&2
  exit 1
fi

echo "This is a merge commit with $PARENTS_COUNT parents" >&2
echo "$PARENTS_COUNT"
exit 0
