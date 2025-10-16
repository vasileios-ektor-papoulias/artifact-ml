#!/bin/bash
set -euo pipefail

# Usage: .github/scripts/linting/lint_pr_title.sh "PR Title" [branch_name]
# Returns: 0 if the PR title follows the convention, 1 otherwise
# Stdout on success: bump type (patch|minor|major|no-bump)
# If branch_name is provided and it resolves to the 'root' component, PR title must use 'no-bump:'.

chmod +x .github/scripts/linting/detect_bump_pattern.sh
chmod +x .github/scripts/linting/lint_branch_name.sh

# ---- Policy (declare allowed sets here; env can override) ----
ALLOWED_COMPONENTS_DEFAULT="root core experiment torch"
ALLOWED_BRANCH_TYPES_DEFAULT="dev hotfix setup feature fix"

ALLOWED_COMPONENTS="${ALLOWED_COMPONENTS:-$ALLOWED_COMPONENTS_DEFAULT}"
ALLOWED_BRANCH_TYPES="${ALLOWED_BRANCH_TYPES:-$ALLOWED_BRANCH_TYPES_DEFAULT}"
# --------------------------------------------------------------

PR_TITLE="${1-}"
BRANCH_NAME="${2-}"

if [ -z "$PR_TITLE" ]; then
  echo "::error::No PR title provided!" >&2
  echo "::error::Usage: $0 \"PR Title\" [branch_name]" >&2
  exit 1
fi

echo "PR Title: $PR_TITLE" >&2

COMPONENT=""
if [ -n "$BRANCH_NAME" ]; then
  echo "Branch name: $BRANCH_NAME" >&2

  # Validate/parse branch via linter using declared allowed sets
  if ! BRANCH_INFO_JSON=$(.github/scripts/linting/lint_branch_name.sh \
        "$BRANCH_NAME" \
        "$ALLOWED_COMPONENTS" \
        "$ALLOWED_BRANCH_TYPES" 2>&1); then
    echo "$BRANCH_INFO_JSON" >&2
    echo "::error::Failed to validate/parse branch name for PR title checks." >&2
    exit 1
  fi

  COMPONENT=$(echo "$BRANCH_INFO_JSON" | grep -o '"component_name":"[^"]*"' | cut -d'"' -f4 || true)
  echo "Branch component: $COMPONENT" >&2

  if [ "$COMPONENT" = "root" ]; then
    # Root component PRs must always be no-bump
    PR_TITLE_LOWER=$(echo "$PR_TITLE" | tr '[:upper:]' '[:lower:]')
    if [[ "$PR_TITLE_LOWER" =~ ^no-bump: ]] || [[ "$PR_TITLE_LOWER" =~ ^no-bump\( ]]; then
      echo "Root component PR has correct no-bump prefix." >&2
    else
      echo "::error::PRs from 'root' component branches must use 'no-bump:' prefix." >&2
      echo "::error::Root component changes should not trigger version bumps." >&2
      exit 1
    fi
  fi
fi

# Detect the bump type from the PR title (patch|minor|major|no-bump)
BUMP_TYPE=$(.github/scripts/linting/detect_bump_pattern.sh "$PR_TITLE")

# Success: print bump type to stdout
echo "$BUMP_TYPE"
exit 0

