#!/bin/bash
set -e

# Usage: .github/scripts/enforce_path/ensure_changed_files_in_dir.sh <component_name> <base_ref> <head_ref>
# Returns: 0 if all changed files are within the component directory, 1 otherwise

COMPONENT_NAME="$1"
BASE_REF="$2"
HEAD_REF="$3"

if [ -z "$COMPONENT_NAME" ] || [ -z "$BASE_REF" ] || [ -z "$HEAD_REF" ]; then
  echo "::error::Missing required parameters!" >&2
  echo "::error::Usage: $0 <component_name> <base_ref> <head_ref>" >&2
  exit 1
fi

CHANGED_FILES=$(git diff --name-only origin/$BASE_REF origin/$HEAD_REF)
echo "Files changed in this PR:" >&2
echo "$CHANGED_FILES" >&2

INVALID_FILES=$(echo "$CHANGED_FILES" | grep -v "^$COMPONENT_NAME/" || true)

if [ -n "$INVALID_FILES" ]; then
  echo "::error::The following files are outside the allowed ./$COMPONENT_NAME directory:" >&2
  echo "::error::$INVALID_FILES" >&2
  echo "::error::Changes to dev-$COMPONENT_NAME branch must only modify files in the cicd_sandbox/$COMPONENT_NAME/ directory." >&2
  exit 1
else
  echo "All changes are within the allowed cicd_sandbox/$COMPONENT_NAME/ directory." >&2
  exit 0
fi
