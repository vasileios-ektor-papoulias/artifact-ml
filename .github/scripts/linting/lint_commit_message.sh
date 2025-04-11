#!/bin/bash
set -e

# Usage: .github/scripts/linting/lint_commit_message.sh
# Returns: 0 if the commit message follows the branch naming convention, 1 otherwise
# Outputs: The component name to stdout if successful

chmod +x .github/scripts/linting/extract_branch_info.sh

LAST_COMMIT_MESSAGE=$(git log -1 --pretty=format:%s)
echo "Last commit message: $LAST_COMMIT_MESSAGE" >&2

BRANCH_NAME=""
if [[ "$LAST_COMMIT_MESSAGE" =~ Merge\ pull\ request\ .*\ from\ ([^/]+)\/([a-zA-Z0-9_/-]+) ]]; then
  USERNAME="${BASH_REMATCH[1]}"
  BRANCH_NAME="${BASH_REMATCH[2]}"
  echo "Extracted username: $USERNAME" >&2
  echo "Extracted branch name: $BRANCH_NAME" >&2
else
  echo "::error::Merge commit message does not follow the expected format!" >&2
  echo "::error::Expected format: 'Merge pull request #123 from username/branch-name'" >&2
  exit 1
fi

BRANCH_INFO_RESULT=$(.github/scripts/linting/extract_branch_info.sh "$BRANCH_NAME" 2>&1)
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
  echo "$BRANCH_INFO_RESULT" >&2
  echo "Merge commit message does not follow the branch naming convention!" >&2
  echo "::error::Merge commit message does not follow the branch naming convention!" >&2
  echo "::error::Merge commit message should be from a branch named 'dev-<component_name>', 'hotfix-<component_name>/<some_other_name>', or 'setup-<component_name>/<some_other_name>'." >&2
  echo "::error::Examples:" >&2
  echo "::error::  'Merge pull request #123 from username/dev-mycomponent'" >&2
  echo "::error::  'Merge pull request #123 from username/hotfix-mycomponent/fix-critical-bug'" >&2
  echo "::error::  'Merge pull request #123 from username/setup-mycomponent/initial-config'" >&2
  echo "::error::This is required for the automatic version bumping to work correctly." >&2
  exit 1
fi

COMPONENT_NAME=$(echo "$BRANCH_INFO_RESULT" | grep -o '"component_name":"[^"]*"' | cut -d'"' -f4)
BRANCH_TYPE=$(echo "$BRANCH_INFO_RESULT" | grep -o '"branch_type":"[^"]*"' | cut -d'"' -f4)

echo "Merge commit message follows the branch naming convention ($BRANCH_TYPE branch)." >&2
echo "Extracted component name: $COMPONENT_NAME" >&2
echo "$COMPONENT_NAME"
exit 0
