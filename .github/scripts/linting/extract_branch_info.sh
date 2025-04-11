#!/bin/bash
set -e

# Usage: .github/scripts/linting/extract_branch_info.sh <branch_name>
# Returns: JSON-formatted string with branch_type and component_name
# Example output: {"branch_type":"dev","component_name":"mycomponent"}
# Exits with error if the branch name doesn't follow the convention

BRANCH_NAME="$1"

if [ -z "$BRANCH_NAME" ]; then
  echo "::error::No branch name provided!" >&2
  echo "::error::Usage: $0 <branch_name>" >&2
  exit 1
fi

echo "Branch name: $BRANCH_NAME" >&2

BRANCH_TYPE=""
COMPONENT_NAME=""

if [[ "$BRANCH_NAME" =~ ^dev-([^/]+) ]]; then
  BRANCH_TYPE="dev"
  COMPONENT_NAME="${BASH_REMATCH[1]}"
elif [[ "$BRANCH_NAME" =~ ^hotfix-([^/]+)/ ]]; then
  BRANCH_TYPE="hotfix"
  COMPONENT_NAME="${BASH_REMATCH[1]}"
elif [[ "$BRANCH_NAME" =~ ^setup-([^/]+)/ ]]; then
  BRANCH_TYPE="setup"
  COMPONENT_NAME="${BASH_REMATCH[1]}"
else
  echo "::error::Branch name does not follow the convention!" >&2
  echo "::error::Branch name should be 'dev-<component_name>', 'hotfix-<component_name>/<some_other_name>', or 'setup-<component_name>/<some_other_name>'." >&2
  echo "::error::Examples:" >&2
  echo "::error::  'dev-mycomponent'" >&2
  echo "::error::  'hotfix-mycomponent/fix-critical-bug'" >&2
  echo "::error::  'setup-mycomponent/initial-config'" >&2
  exit 1
fi

echo "{\"branch_type\":\"$BRANCH_TYPE\",\"component_name\":\"$COMPONENT_NAME\"}"
exit 0
