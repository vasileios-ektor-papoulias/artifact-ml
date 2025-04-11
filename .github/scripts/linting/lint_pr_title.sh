#!/bin/bash
set -e

# Usage: .github/scripts/linting/lint_pr_title.sh "PR Title" [branch_name]
# Returns: 0 if the PR title follows the convention, 1 otherwise
# Outputs: The bump type (patch, minor, major) to stdout if successful
# If branch_name is provided and it's a root component branch, ensures PR title uses no-bump prefix

chmod +x .github/scripts/linting/detect_bump_pattern.sh
chmod +x .github/scripts/linting/extract_branch_info.sh

PR_TITLE="$1"
BRANCH_NAME="$2"

if [ -z "$PR_TITLE" ]; then
  echo "::error::No PR title provided!" >&2
  echo "::error::Usage: $0 \"PR Title\" [branch_name]" >&2
  exit 1
fi

echo "PR Title: $PR_TITLE" >&2

if [ -n "$BRANCH_NAME" ]; then
  echo "Branch name: $BRANCH_NAME" >&2
  
  BRANCH_INFO=$(.github/scripts/linting/extract_branch_info.sh "$BRANCH_NAME" 2>/dev/null || echo '{"component_name":""}')
  
  COMPONENT=$(echo "$BRANCH_INFO" | grep -o '"component_name":"[^"]*"' | cut -d'"' -f4)
  
  echo "Branch component: $COMPONENT" >&2
  
  if [ "$COMPONENT" = "root" ]; then
    PR_TITLE_LOWER=$(echo "$PR_TITLE" | tr '[:upper:]' '[:lower:]')
    
    if [[ "$PR_TITLE_LOWER" =~ ^no-bump: ]] || [[ "$PR_TITLE_LOWER" =~ ^no-bump\( ]]; then
      echo "Root component PR has correct no-bump prefix" >&2
    else
      echo "::error::PRs from root component branches must use 'no-bump:' prefix" >&2
      echo "::error::Root component changes should not trigger version bumps" >&2
      exit 1
    fi
  fi
fi

BUMP_TYPE=$(.github/scripts/linting/detect_bump_pattern.sh "$PR_TITLE")

echo "$BUMP_TYPE"
exit 0
