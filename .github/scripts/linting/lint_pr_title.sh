#!/bin/bash
set -euo pipefail

# Purpose:
#   Validate that a given PR title adheres to the repository’s semantic versioning
#   convention and emit the parsed bump type. If an optional source branch name
#   is provided, parse it and enforce that PRs targeting the **root** component
#   must use the **no-bump** bump type.
#
# Usage:
#   .github/scripts/linting/lint_pr_title.sh "PR Title" [branch_name]
#
# Accepts:
#   "PR Title"    (required)  The PR title to validate, e.g., "minor: add feature".
#   [branch_name] (optional)  Source branch to parse for the component, e.g., "dev-core",
#                             "hotfix-root/fix-ci". Parsed strictly by extract_branch_info.sh.
#
# Stdout on success:
#   One of: patch | minor | major | no-bump
#
# Stderr on failure:
#   ::error::-prefixed diagnostics indicating:
#     - missing PR title, or
#     - invalid/unsupported title prefix, or
#     - invalid branch name (when branch_name is provided), or
#     - root component PR missing required "no-bump" bump type.
#
# Exit codes:
#   0 — title prefix is valid (bump type printed to stdout) and, if a branch_name
#       is provided, it parses successfully; root rule (no-bump) honored.
#   1 — any error: missing title, invalid prefix, branch parsing failure,
#       or root component PR without "no-bump:".
#
# Behaviour:
#   - Validates the PR title’s leading token (case-insensitive):
#       • "patch:" | "minor:" | "major:" | "no-bump:"
#       • Scoped variants allowed: "<type>(scope): ..."
#   - If [branch_name] is supplied:
#       • Delegates to extract_branch_info.sh to parse the branch shape and extract component_name.
#       • If component_name == "root", require the title to start with "no-bump:"
#         (or scoped "no-bump(scope):").
#   - On success, prints only the bump type to STDOUT.
#
# Notes:
#   - Title matching is case-insensitive; only the leading token is validated.
#
# Examples:
#   .github/scripts/linting/lint_pr_title.sh "patch: fix login bug"
#     --> patch        # exit 0
#
#   .github/scripts/linting/lint_pr_title.sh "minor(ui): add tabs"
#     --> minor        # exit 0
#
#   .github/scripts/linting/lint_pr_title.sh "no-bump: docs" dev-core
#     --> no-bump      # exit 0 (non-root branch; any valid bump allowed)
#
#   .github/scripts/linting/lint_pr_title.sh "no-bump(docs): housekeeping" hotfix-root/fix-ci
#     --> no-bump      # exit 0 (root branch requires no-bump)
#
#   .github/scripts/linting/lint_pr_title.sh "patch: fix" hotfix-root/fix-ci
#     --> (stderr explains root requires "no-bump:")   # exit 1

chmod +x .github/scripts/linting/detect_bump_pattern.sh
chmod +x .github/scripts/linting/extract_branch_info.sh

PR_TITLE="${1-}"
BRANCH_NAME="${2-}"

if [ -z "$PR_TITLE" ]; then
  echo "::error::No PR title provided!" >&2
  echo "::error::Usage: $0 \"PR Title\" [branch_name]" >&2
  exit 1
fi

echo "PR Title: $PR_TITLE" >&2

# If a branch is provided, parse its shape and enforce the root rule.
if [ -n "$BRANCH_NAME" ]; then
  echo "Branch name: $BRANCH_NAME" >&2

  if ! BRANCH_INFO_JSON=$(.github/scripts/linting/extract_branch_info.sh "$BRANCH_NAME" 2>&1); then
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