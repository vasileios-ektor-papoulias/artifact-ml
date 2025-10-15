#!/bin/bash
set -euo pipefail

# Usage: .github/scripts/enforce_path/ensure_changed_files_in_dir.sh <component_name> <base_ref>
# Returns: 0 if all changed files are within the component directory, 1 otherwise


COMPONENT_NAME="${1:-}"
BASE_REF="${2:-}"

if [ -z "$COMPONENT_NAME" ] || [ -z "$BASE_REF" ]; then
  echo "::error::Missing required parameters!" >&2
  echo "::error::Usage: $0 <component_name> <base_ref>" >&2
  exit 1
fi


git fetch --no-tags --prune --depth=2 origin "$BASE_REF"
MB=$(git merge-base "origin/$BASE_REF" HEAD)
CHANGED_FILES=$(git diff --name-only "$MB" HEAD)
echo "Files changed in this PR:" >&2
echo "$CHANGED_FILES" >&2

ALLOWED_PREFIX="${COMPONENT_NAME}/"
INVALID_FILES=$(printf '%s\n' "$CHANGED_FILES" | grep -v "^${ALLOWED_PREFIX}" || true)

if [ -n "$INVALID_FILES" ]; then
  echo "::error::The following files are outside the allowed ./${ALLOWED_PREFIX} directory:" >&2
  echo "::error::$INVALID_FILES" >&2
  echo "::error::Changes to dev-${COMPONENT_NAME} branches must only modify files in ${ALLOWED_PREFIX}" >&2
  exit 1
else
  echo "All changes are within the allowed ${ALLOWED_PREFIX} directory." >&2
  exit 0
fi
