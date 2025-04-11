#!/bin/bash
set -e

# Usage: .github/scripts/linting/check_is_merge_commit.sh
# Returns: 0 if the last commit is a merge commit, 1 otherwise
# Outputs: The number of parents to stdout if it's a merge commit

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
