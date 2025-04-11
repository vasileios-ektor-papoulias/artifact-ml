#!/bin/bash
set -e

# Usage: .github/scripts/enforce_path/ensure_changed_files_outside_dirs.sh <base_ref> <head_ref> <dir1> [<dir2> ...]
# Returns: 0 if all changed files are outside the specified directories, 1 otherwise
# Example: .github/scripts/enforce_path/ensure_changed_files_outside_dirs.sh main feature-branch dir1 dir2 dir3

if [ "$#" -lt 3 ]; then
  echo "::error::Missing required parameters!" >&2
  echo "::error::Usage: $0 <base_ref> <head_ref> <dir1> [<dir2> ...]" >&2
  echo "::error::Example: $0 main feature-branch dir1 dir2 dir3" >&2
  exit 1
fi

BASE_REF="$1"
HEAD_REF="$2"
shift 2

if [ -z "$BASE_REF" ] || [ -z "$HEAD_REF" ] || [ $# -eq 0 ]; then
  echo "::error::Missing required parameters!" >&2
  echo "::error::Usage: $0 <base_ref> <head_ref> <dir1> [<dir2> ...]" >&2
  echo "::error::Example: $0 main feature-branch dir1 dir2 dir3" >&2
  exit 1
fi

CHANGED_FILES=$(git diff --name-only origin/$BASE_REF origin/$HEAD_REF)
echo "Files changed in this PR:" >&2
echo "$CHANGED_FILES" >&2

INVALID_FILES=""

for DIR in "$@"; do
  MATCHING_FILES=$(echo "$CHANGED_FILES" | grep -E "^$DIR/" || true)
  
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
  echo "::error::The following files are inside the forbidden directories:" >&2
  echo "::error::$INVALID_FILES" >&2
  echo "::error::Changes must not modify files in the following directories: $@" >&2
  exit 1
else
  echo "All changes are outside the specified directories: $@" >&2
  exit 0
fi
