#!/bin/bash
set -euo pipefail

# Purpose:
#   Parse the last commit’s *subject line*, ensuring it encodes a valid branch
#   name as per the repository's conventions. On success, print the parsed
#   component name.
#
# Usage:
#   .github/scripts/linting/lint_commit_message.sh
#
# Accepts:
#   (no positional args; reads from the local Git repo)
#
# Stdout on success:
#   <component_name>
#     e.g. core
#
# Stderr on failure:
#   ::error::-prefixed diagnostics explaining:
#     - subject could not be parsed to extract a branch name, or
#     - extracted branch failed SHAPE validation (per extract_branch_info.sh).
#
# Exit codes:
#   0 — subject parsed; source branch SHAPE valid; component printed to STDOUT
#   1 — subject not parseable to a branch or SHAPE invalid
#
# Behaviour:
#   - Reads the last commit’s subject:  git log -1 --pretty=format:%s
#   - Extracts a branch name from the subject (supports "<user>/<branch>" and "<user>:<branch>" forms).
#   - Delegates SHAPE parsing ONLY to:
#       .github/scripts/linting/extract_branch_info.sh "<branch>"
#   - On success, parses the returned JSON and prints only component_name to STDOUT.
#
# Notes:
#   - Common usage is on merge commits created by GitHub PRs; those subjects include the
#     source branch name, which this script validates against the repo’s branch SHAPE rules.
#   - SHAPE rules (dev-<component> vs <type>-<component>/<desc>) are enforced solely by
#     extract_branch_info.sh; this script does not apply any allowed-type/component policy.
#
# Examples:
#   # Subject encodes: dev-core  → prints "core"
#   .github/scripts/linting/lint_commit_message.sh
#
#   # Subject encodes: hotfix-core/fix-ci  → prints "core"
#   .github/scripts/linting/lint_commit_message.sh

chmod +x .github/scripts/linting/extract_branch_info.sh

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
  echo "::error::Commit subject didn’t match expected formats to extract a branch." >&2
  echo "::error::Examples:" >&2
  echo "::error::  'Merge pull request #123 from username/feature/foo-bar'" >&2
  echo "::error::  'Merge pull request #123 from username:feature/foo-bar'" >&2
  exit 1
fi

# Validate SHAPE & parse via extractor
if ! BRANCH_INFO_JSON=$(.github/scripts/linting/extract_branch_info.sh "$BRANCH_NAME" 2>&1); then
  echo "$BRANCH_INFO_JSON" >&2
  echo "::error::Source branch does not follow the required naming convention (type-component[/desc])." >&2
  exit 1
fi

# Extract fields from returned JSON
COMPONENT_NAME=$(echo "$BRANCH_INFO_JSON" | grep -o '"component_name":"[^"]*"' | cut -d'"' -f4 || true)
BRANCH_TYPE=$(echo "$BRANCH_INFO_JSON" | grep -o '"branch_type":"[^"]*"' | cut -d'"' -f4 || true)

if [ -z "${COMPONENT_NAME}" ] || [ -z "${BRANCH_TYPE}" ]; then
  echo "::error::Failed to parse component/type from extractor output." >&2
  echo "::error::Output was: $BRANCH_INFO_JSON" >&2
  exit 1
fi

# Normalize type allow-list to lowercase for comparison
IFS=' ' read -r -a _types_raw <<< "$ALLOWED_BRANCH_TYPES"
ALLOWED_TYPES_LC=()
for t in "${_types_raw[@]}"; do ALLOWED_TYPES_LC+=("${t,,}"); done

# Policy checks
type_ok=false
comp_ok=false

for t in "${ALLOWED_TYPES_LC[@]}"; do
  if [ "${BRANCH_TYPE,,}" = "$t" ]; then
    type_ok=true; break
  fi
done

IFS=' ' read -r -a _components <<< "$ALLOWED_COMPONENTS"
for c in "${_components[@]}"; do
  if [ "$COMPONENT_NAME" = "$c" ]; then
    comp_ok=true; break
  fi
done

if [ "$type_ok" != true ] || [ "$comp_ok" != true ]; then
  echo "::error::Branch is syntactically valid, but fails policy checks." >&2
  echo "::error::Allowed branch types: $ALLOWED_BRANCH_TYPES" >&2
  echo "::error::Allowed components: $ALLOWED_COMPONENTS" >&2
  exit 1
fi

echo "Commit subject encodes a valid and allowed branch ($BRANCH_TYPE-$COMPONENT_NAME)." >&2

# Print component to stdout (contract)
echo "$COMPONENT_NAME"
exit 0