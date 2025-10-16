#!/bin/bash
set -euo pipefail

# Purpose:
#   Enforce that a PR only changes files under a single component directory (e.g., "artifact-core/").
#
# Usage:
#   .github/scripts/enforce_path/ensure_changed_files_in_dir.sh <component_dir> <base_ref>
#
# Accepts:
#   <component_dir>  Repo-relative directory that ALL changed files must be under (e.g., "artifact-core").
#   <base_ref>       Git ref representing the PR base (e.g., "main", "dev-core") used to compute the diff.
#
# Stdout on success:
#   (none) — informational messages are written to STDERR for Actions logs.
#
# Stderr:
#   - On success: "All changes are within the allowed <component_dir>/ directory."
#   - On failure: ::error::-prefixed diagnostics listing offending paths and guidance.
#
# Exit codes:
#   0  success — all changed files are under <component_dir>/
#   1  validation/policy failure (missing args, unable to diff, or at least one change outside <component_dir>/)
#
# Behaviour:
#   - Fetches the remote <base_ref> (shallow) and computes merge-base with HEAD.
#   - Lists changed paths (names only) from <merge-base>..HEAD.
#   - Verifies each path begins with "<component_dir>/" (simple repo-relative prefix check).
#   - Fails if any path lies outside "<component_dir>/".
#
# Notes:
#   - Intended for PR validation; set <base_ref> to the PR’s base branch.
#   - Ensure sufficient history for <base_ref> exists locally
#     (e.g., actions/checkout@v4 with `fetch-depth: 0`, or enough to reach the merge-base).
#   - Pass <component_dir> without leading "./".
#
# Examples:
#   # Enforce that a PR to main only touches artifact-core/
#   .github/scripts/enforce_path/ensure_changed_files_in_dir.sh artifact-core main
#
#   # Enforce that a PR into dev-experiment only touches artifact-experiment/
#   .github/scripts/enforce_path/ensure_changed_files_in_dir.sh artifact-experiment dev-experiment




COMPONENT_DIR="${1:-}"
BASE_REF="${2:-}"

if [ -z "$COMPONENT_DIR" ] || [ -z "$BASE_REF" ]; then
  echo "::error::Missing required parameters!" >&2
  echo "::error::Usage: $0 <component_dir> <base_ref>" >&2
  exit 1
fi


git fetch --no-tags --prune --depth=2 origin "$BASE_REF"
MB=$(git merge-base "origin/$BASE_REF" HEAD)
CHANGED_FILES=$(git diff --name-only "$MB" HEAD)
echo "Files changed in this PR:" >&2
echo "$CHANGED_FILES" >&2

ALLOWED_PREFIX="${COMPONENT_DIR}/"
INVALID_FILES=$(printf '%s\n' "$CHANGED_FILES" | grep -v "^${ALLOWED_PREFIX}" || true)

if [ -n "$INVALID_FILES" ]; then
  echo "::error::The following files are outside the allowed ./${ALLOWED_PREFIX} directory:" >&2
  echo "::error::$INVALID_FILES" >&2
  echo "::error::Changes to dev-${COMPONENT_DIR} branches must only modify files in ${ALLOWED_PREFIX}" >&2
  exit 1
else
  echo "All changes are within the allowed ${ALLOWED_PREFIX} directory." >&2
  exit 0
fi
