#!/bin/bash
set -e

# Usage: .github/scripts/linting/lint_merge_commit_description.sh
# Returns: 0 if the commit is not a merge commit or if it's a merge commit with a valid description
# Returns: 1 if it's a merge commit with an invalid description

chmod +x .github/scripts/linting/check_is_merge_commit.sh
chmod +x .github/scripts/linting/lint_commit_description.sh

if .github/scripts/linting/check_is_merge_commit.sh &>/dev/null; then
    echo "This is a merge commit, checking description..."
    
    # Lint the commit description
    BUMP_TYPE=$(.github/scripts/linting/lint_commit_description.sh)
    echo "Bump type: $BUMP_TYPE"
    exit 0
else
    echo "Not a merge commit, skipping linting"
    exit 0
fi
