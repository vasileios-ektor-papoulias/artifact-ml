#!/bin/bash
set -e

# Usage: .github/scripts/version_bump/get_bump_type.sh
# Returns: The version bump type extracted from commit description
# Returns 1 if no bump type can be extracted

chmod +x .github/scripts/linting/lint_commit_description.sh

BUMP_TYPE=$(.github/scripts/linting/lint_commit_description.sh)

echo "Determined bump_type: $BUMP_TYPE" >&2

echo "$BUMP_TYPE"
