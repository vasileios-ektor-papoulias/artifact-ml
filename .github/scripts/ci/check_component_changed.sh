#!/usr/bin/env bash
set -euo pipefail

# Purpose:
#   Check if a specific component directory has changes between two Git refs.
#
# Usage:
#   .github/scripts/ci/check_component_changed.sh <component_dir> [base_ref]
#
# Accepts:
#   <component_dir>  Directory to check for changes (e.g., "artifact-core")
#   [base_ref]       Optional base ref to compare against (default: HEAD~1)
#
# Stdout on success:
#   "true" if the component has changes, "false" otherwise
#
# Exit codes:
#   0 — always (outputs true/false to stdout)
#   1 — missing required arguments
#
# Examples:
#   .github/scripts/ci/check_component_changed.sh artifact-core
#     --> true (if artifact-core/ has changes)
#
#   .github/scripts/ci/check_component_changed.sh artifact-torch HEAD~2
#     --> false (if artifact-torch/ has no changes since HEAD~2)

COMPONENT_DIR="${1:-}"
BASE_REF="${2:-HEAD~1}"

if [[ -z "$COMPONENT_DIR" ]]; then
    echo "Error: Component directory is required." >&2
    echo "Usage: $0 <component_dir> [base_ref]" >&2
    exit 1
fi

# Get changed files between base_ref and HEAD
CHANGED_FILES=$(git diff --name-only "$BASE_REF" HEAD 2>/dev/null || echo "")

# Check if any changed file starts with the component directory
if echo "$CHANGED_FILES" | grep -q "^${COMPONENT_DIR}/"; then
    echo "true"
else
    echo "false"
fi

