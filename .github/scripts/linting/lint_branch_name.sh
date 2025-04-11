#!/bin/bash
set -e

# Usage: .github/scripts/linting/lint_branch_name.sh <branch_name> <allowed_components>
# Returns: 0 if the branch name follows the convention, 1 otherwise
# Example: .github/scripts/linting/lint_branch_name.sh "dev-subrepo" "subrepo content"

chmod +x .github/scripts/linting/extract_branch_info.sh

BRANCH_NAME="$1"
ALLOWED_COMPONENTS="$2"

if [ -z "$BRANCH_NAME" ] || [ -z "$ALLOWED_COMPONENTS" ]; then
  echo "::error::Missing required parameters!" >&2
  echo "::error::Usage: $0 <branch_name> <allowed_components>" >&2
  echo "::error::Example: $0 \"dev-subrepo\" \"subrepo content\"" >&2
  exit 1
fi

echo "Checking branch name: $BRANCH_NAME" >&2
echo "Allowed components: $ALLOWED_COMPONENTS" >&2

IFS=' ' read -r -a COMPONENTS <<< "$ALLOWED_COMPONENTS"

VALID_BRANCH=false

BRANCH_INFO_RESULT=$(.github/scripts/linting/extract_branch_info.sh "$BRANCH_NAME" 2>/dev/null || echo '{"branch_type":"","component_name":""}')
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  COMPONENT_NAME=$(echo "$BRANCH_INFO_RESULT" | grep -o '"component_name":"[^"]*"' | cut -d'"' -f4)
  BRANCH_TYPE=$(echo "$BRANCH_INFO_RESULT" | grep -o '"branch_type":"[^"]*"' | cut -d'"' -f4)
  
  echo "Extracted branch type: $BRANCH_TYPE" >&2
  echo "Extracted component name: $COMPONENT_NAME" >&2
  
  for COMPONENT in "${COMPONENTS[@]}"; do
    if [ "$COMPONENT_NAME" = "$COMPONENT" ]; then
      echo "Branch follows the $BRANCH_TYPE-$COMPONENT_NAME naming convention." >&2
      VALID_BRANCH=true
      break
    fi
  done
fi

if [ "$VALID_BRANCH" = false ]; then
  echo "::error::Branch name does not follow the required naming convention!" >&2
  echo "::error::Branch name must be one of:" >&2
  for COMPONENT in "${COMPONENTS[@]}"; do
    echo "::error::  - dev-$COMPONENT" >&2
    echo "::error::  - hotfix-$COMPONENT/<descriptive-name>" >&2
    echo "::error::  - setup-$COMPONENT/<descriptive-name>" >&2
  done
  exit 1
fi

exit 0
