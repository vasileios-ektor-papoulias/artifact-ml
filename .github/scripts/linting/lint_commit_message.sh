#!/bin/bash
set -euo pipefail

# Purpose:
#   Validate a commit’s *subject line* that encodes a source branch name, ensuring the
#   branch both (1) matches the repository’s branch-naming SHAPE and (2) conforms to
#   allowed component/type policy. On success, print the parsed component name.
#
# Usage:
#   .github/scripts/linting/lint_commit_message.sh
#
# Accepts:
#   (no positional args; reads from the local Git repo)
#   Environment (optional overrides):
#     ALLOWED_COMPONENTS    space-separated list; default: "root core experiment torch"
#     ALLOWED_BRANCH_TYPES  space-separated list; default: "dev hotfix setup"
#
# Stdout on success:
#   <component_name>
#     e.g. core
#
# Stderr on failure:
#   ::error::-prefixed diagnostics explaining either:
#     - subject could not be parsed to extract a branch name, or
#     - extracted branch failed SHAPE validation, or
#     - extracted branch failed allowed component/type policy (with guidance).
#
# Exit codes:
#   0 — subject parsed; source branch SHAPE valid and allowed; component printed to STDOUT
#   1 — subject not parseable to a branch, or SHAPE invalid, or policy disallowed
#
# Behaviour:
#   - Reads the last commit’s subject:  git log -1 --pretty=format:%s
#   - Extracts a branch name from the subject (supports "<user>/<branch>" and "<user>:<branch>" forms).
#   - Delegates SHAPE + policy validation to:
#       .github/scripts/linting/lint_branch_name.sh "<branch>" "$ALLOWED_COMPONENTS" "$ALLOWED_BRANCH_TYPES"
#   - On success, parses the returned JSON and prints only component_name to STDOUT.
#
# Notes:
#   - Common usage is on merge commits created by GitHub PRs; those subjects include the
#     source branch name, which this script validates against the repo’s branch rules.
#   - SHAPE rules (dev-<component> vs <type>-<component>/<desc>) are enforced by
#     lint_branch_name.sh (which uses extract_branch_info.sh).
#
# Examples:
#   # Defaults allow dev + core
#   # Subject encodes: dev-core
#   ALLOWED_COMPONENTS="root core experiment torch" ALLOWED_BRANCH_TYPES="dev hotfix setup" \
#   .github/scripts/linting/lint_commit_message.sh
#   --> core    # exit 0
#
#   # Policy allow-list narrowed to hotfix only (setup disallowed)
#   # Subject encodes: setup-core/init
#   ALLOWED_COMPONENTS="root core" ALLOWED_BRANCH_TYPES="dev hotfix" \
#   .github/scripts/linting/lint_commit_message.sh
#   --> (stderr explains disallowed type 'setup')  # exit 1

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