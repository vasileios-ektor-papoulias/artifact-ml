#!/bin/bash
set -euo pipefail

# Purpose:
#   Emit the component name associated with the last commit when it’s a
#   GitHub-style merge commit whose subject encodes the source branch.
#   This is a thin wrapper around existing linters and is intended for
#   version-bump workflows that need to know which component changed.
#
# Usage:
#   .github/scripts/version_bump/get_component_name.sh
#
# Accepts:
#   (no CLI args) — inspects the repository’s HEAD commit.
#
# Stdout on success:
#   <component_name>
#     e.g. core
#
# Stderr on failure:
#   Informational logs and ::error::-style messages bubbled up from the
#   underlying linters (or a note that no component name was found).
#
# Exit codes:
#   0 — script completed; if no component was found, stdout will be empty
#       (consumers should treat empty stdout as “not found”).
#
# Behaviour:
#   - Verifies whether HEAD is a merge commit via:
#       .github/scripts/linting/check_is_merge_commit.sh
#   - If it is a merge commit, parses/validates the encoded source branch
#     using:
#       .github/scripts/linting/lint_commit_message.sh
#     and prints only the component name to STDOUT (e.g., core).
#   - If it is not a merge commit, or parsing/validation yields no component,
#     prints an informational line to STDERR and outputs an empty line on STDOUT.
#
# Notes:
#   - Typical usage is on merge commits created by PR merges, where GitHub
#     sets the subject to include the source branch (e.g., “from user/dev-core”).
#   - This script intentionally does not fail the job when no component is
#     found; downstream steps should check for empty stdout.
#
# Examples:
#   # HEAD is a merge of dev-core → prints “core”
#   .github/scripts/version_bump/get_component_name.sh
#     --> core

chmod +x .github/scripts/linting/check_is_merge_commit.sh
chmod +x .github/scripts/linting/lint_commit_message.sh

if .github/scripts/linting/check_is_merge_commit.sh &>/dev/null; then
    COMPONENT_NAME=$(.github/scripts/linting/lint_commit_message.sh)
    if [ -n "$COMPONENT_NAME" ]; then
        echo "Extracted component name '$COMPONENT_NAME' from merge commit message" >&2
        echo "$COMPONENT_NAME"
        exit 0
    fi
fi

echo "No component name found" >&2
echo ""
