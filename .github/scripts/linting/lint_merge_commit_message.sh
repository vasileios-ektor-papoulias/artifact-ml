#!/bin/bash
set -e

# Usage: .github/scripts/linting/lint_merge_commit_message.sh
# Returns: 0 if the commit is not a merge commit or if it's a merge commit with a valid message
# Returns: 1 if it's a merge commit with an invalid message

chmod +x .github/scripts/linting/check_is_merge_commit.sh
chmod +x .github/scripts/linting/lint_commit_message.sh

if .github/scripts/linting/check_is_merge_commit.sh &>/dev/null; then
    echo "This is a merge commit, checking message..."
    
    # Lint the commit message
    COMPONENT_NAME=$(.github/scripts/linting/lint_commit_message.sh)
    echo "Component name: $COMPONENT_NAME"
    exit 0
else
    echo "Not a merge commit, skipping linting"
    exit 0
fi
