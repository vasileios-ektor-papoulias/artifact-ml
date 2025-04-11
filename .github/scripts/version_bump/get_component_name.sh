#!/bin/bash
set -e

# Usage: .github/scripts/version_bump/get_component_name.sh
# Returns: The component name extracted from the merge commit message
# If no component name can be extracted, returns an empty string
# Uses the lint_commit_message.sh script to validate and extract the component name

chmod +x .github/scripts/linting/check_is_merge_commit.sh
chmod +x .github/scripts/linting/lint_commit_message.sh

if .github/scripts/linting/check_is_merge_commit.sh &>/dev/null; then
    COMPONENT_NAME=$(.github/scripts/linting/lint_commit_message.sh)
    if [ -n "$COMPONENT_NAME" ]; then
        echo "Extracted component name '$COMPONENT_NAME' from merge commit message" >&2
        echo "$COMPONENT_NAME"
        exit 0
    fi
fi

echo "No component name found" >&2
echo ""
