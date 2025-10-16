#!/bin/bash
set -euo pipefail

# Usage: .github/scripts/linting/lint_commit_message.sh
# Returns: 0 if the commit message follows the branch naming convention, 1 otherwise
# Stdout (on success): the component name

chmod +x .github/scripts/linting/lint_branch_name.sh

# ---- Policy (declare allowed sets here; env can override) ----
ALLOWED_COMPONENTS_DEFAULT="root core experiment torch"
ALLOWED_BRANCH_TYPES_DEFAULT="dev hotfix setup"

ALLOWED_COMPONENTS="${ALLOWED_COMPONENTS:-$ALLOWED_COMPONENTS_DEFAULT}"
ALLOWED_BRANCH_TYPES="${ALLOWED_BRANCH_TYPES:-$ALLOWED_BRANCH_TYPES_DEFAULT}"
# --------------------------------------------------------------

LAST_COMMIT_MESSAGE=$(git log -1 --pretty=format:%s)
echo "Last commit message: $LAST_COMMIT_MESSAGE" >&2

BRANCH_NAME=""
if [[ "$LAST_COMMIT_MESSAGE" =~ ^Merge\ pull\ request\ #[0-9]+(\ .*)?\ from\ ([^:/[:space:]]+)[/:]([A-Za-z0-9._/-]+)$ ]]; then
  USERNAME="${BASH_REMATCH[2]}"
  BRANCH_NAME="${BASH_REMATCH[3]}"
  echo "Extracted username: $USERNAME" >&2
  echo "Extracted branch name: $BRANCH_NAME" >&2
else
  echo "::error::Merge commit subject didn’t match expected formats." >&2
  echo "::error::Examples:" >&2
  echo "::error::  'Merge pull request #123 from username/feature/foo-bar'" >&2
  echo "::error::  'Merge pull request #123 from username:feature/foo-bar'" >&2
  exit 1
fi

# Call the branch linter (use declared allowed sets)
if ! BRANCH_INFO_JSON=$(.github/scripts/linting/lint_branch_name.sh \
      "$BRANCH_NAME" \
      "$ALLOWED_COMPONENTS" \
      "$ALLOWED_BRANCH_TYPES" 2>&1); then
  # The linter writes diagnostics to stderr; we captured both, so echo them and fail
  echo "$BRANCH_INFO_JSON" >&2
  echo "::error::Merge commit message does not follow the required branch naming convention (type-component[/desc])." >&2
  exit 1
fi

# Extract fields from returned JSON
COMPONENT_NAME=$(echo "$BRANCH_INFO_JSON" | grep -o '"component_name":"[^"]*"' | cut -d'"' -f4 || true)
BRANCH_TYPE=$(echo "$BRANCH_INFO_JSON" | grep -o '"branch_type":"[^"]*"' | cut -d'"' -f4 || true)

if [ -z "${COMPONENT_NAME}" ] || [ -z "${BRANCH_TYPE}" ]; then
  echo "::error::Failed to parse component/type from linter output." >&2
  echo "::error::Output was: $BRANCH_INFO_JSON" >&2
  exit 1
fi

echo "Merge commit message follows the branch naming convention ($BRANCH_TYPE branch)." >&2
echo "Extracted component name: $COMPONENT_NAME" >&2

# Print component to stdout (contract preserved)
echo "$COMPONENT_NAME"
exit 0