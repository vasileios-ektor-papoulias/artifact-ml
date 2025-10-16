#!/bin/bash
set -euo pipefail

# Purpose:
#   Enforce that a PR does NOT change files under one or more forbidden directories (denylist).
#
# Usage:
#   .github/scripts/enforce_path/ensure_changed_files_outside_dirs.sh <base_ref> <dir1> [<dir2> ...]
#
# Accepts:
#   <base_ref>   Git ref representing the PR base (e.g., "main"); used to compute the diff range.
#   <dirN>...    One or more repo-relative directory prefixes that must NOT be modified (e.g., "docs", "scripts").
#
# Stdout on success:
#   (none) — informational messages are written to STDERR for Actions logs.
#
# Stderr:
#   - On success: "All changes are outside the specified directories: <dir1> <dir2> ..."
#   - On failure: ::error::-prefixed diagnostics listing offending paths and the forbidden directories.
#
# Exit codes:
#   0  success — all changed files are OUTSIDE every listed directory
#   1  validation/policy failure (missing args, or at least one change is inside a forbidden dir)
#
# Behaviour:
#   - Fetches the remote <base_ref> (shallow) and computes merge-base with HEAD.
#   - Lists changed paths (names only) from <merge-base>..HEAD; empty diff counts as success.
#   - Checks that no path starts with any "<dirN>/" (simple repo-relative prefix test).
#
# Notes:
#   - Intended for PR validation jobs; <base_ref> should be the PR’s base branch.
#   - Ensure sufficient history for <base_ref> exists locally
#     (e.g., actions/checkout@v4 with `fetch-depth: 0`, or enough to reach the merge-base).
#   - Pass directory names without leading "./".
#
# Examples:
#   # Disallow changes under docs/ and scripts/ when comparing to main
#   .github/scripts/enforce_path/ensure_changed_files_outside_dirs.sh main docs scripts
#
#   # In a GitHub Actions step:
#   - name: Forbid touching third_party/ and tooling/
#     run: |
#       .github/scripts/enforce_path/ensure_changed_files_outside_dirs.sh \
#         "${{ github.base_ref }}" third_party tooling




if [ "$#" -lt 2 ]; then
  echo "::error::Missing required parameters!" >&2
  echo "::error::Usage: $0 <base_ref> <dir1> [<dir2> ...]" >&2
  echo "::error::Example: $0 main docs scripts third_party" >&2
  exit 1
fi

BASE_REF="${1:-}"
shift 1

git fetch --no-tags --prune --depth=2 origin "$BASE_REF"
MB=$(git merge-base "origin/$BASE_REF" HEAD)
CHANGED_FILES=$(git diff --name-only "$MB" HEAD)

echo "Files changed in this PR:" >&2
printf '%s\n' "$CHANGED_FILES" >&2
if [ -z "$CHANGED_FILES" ]; then
  echo "No changes detected." >&2
  exit 0
fi

echo "Forbidden directory prefixes:" >&2
for DIR in "$@"; do
  printf '  - %s/\n' "${DIR%/}" >&2
done

INVALID_FILES=""

for DIR in "$@"; do
  DIR="${DIR%/}/"
  DIR_ESC=$(printf '%s' "$DIR" | sed 's/[.[\*^$+?{}()|\\]/\\&/g')
  MATCHING_FILES=$(printf '%s\n' "$CHANGED_FILES" | grep -E "^${DIR_ESC}" || true)
  
  if [ -n "$MATCHING_FILES" ]; then
    if [ -n "$INVALID_FILES" ]; then
      INVALID_FILES="$INVALID_FILES
$MATCHING_FILES"
    else
      INVALID_FILES="$MATCHING_FILES"
    fi
  fi
done

if [ -n "$INVALID_FILES" ]; then
  INVALID_FILES=$(printf '%s\n' "$INVALID_FILES" | awk 'NF' | sort -u)
fi

if [ -n "$INVALID_FILES" ]; then
  echo "::error::The following files are inside forbidden directories:" >&2
  echo "::error::$INVALID_FILES" >&2
  echo "::error::Changes must not modify files under: $*" >&2
  exit 1
else
  echo "All changes are outside the specified directories: $*" >&2
  exit 0
fi
