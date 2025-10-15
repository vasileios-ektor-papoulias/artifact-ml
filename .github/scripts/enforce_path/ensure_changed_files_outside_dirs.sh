#!/bin/bash
set -euo pipefail

# Usage: .github/scripts/enforce_path/ensure_changed_files_outside_dirs.sh <base_ref> <dir1> [<dir2> ...]
# Returns: 0 if all changed files are outside the specified directories, 1 otherwise



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
