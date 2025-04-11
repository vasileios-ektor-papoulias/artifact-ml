#!/bin/bash
set -e

# Usage: .github/scripts/linting/lint_commit_description.sh
# Returns: 0 if the commit description follows the convention, 1 otherwise
# Outputs: The bump type (patch, minor, major) to stdout if successful

chmod +x .github/scripts/linting/detect_bump_pattern.sh

LAST_COMMIT_DESCRIPTION=$(git log -1 --pretty=format:%b)

if [ -z "$LAST_COMMIT_DESCRIPTION" ]; then
  echo "::error::Merge commit has no description!" >&2
  echo "::error::You must add a description that starts with one of: 'patch:', 'minor:', or 'major:'." >&2
  echo "::error::This is required for the automatic version bumping to work correctly." >&2
  exit 1
fi

echo "Commit description: $LAST_COMMIT_DESCRIPTION" >&2

BUMP_TYPE=$(.github/scripts/linting/detect_bump_pattern.sh "$LAST_COMMIT_DESCRIPTION")

echo "$BUMP_TYPE"
exit 0
