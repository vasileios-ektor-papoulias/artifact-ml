#!/bin/bash
set -e

# Purpose:
#   Gate merge commits by ensuring their **subject/message** encodes a valid source
#   branch per repository policy, and emit the parsed **component name**. Non-merge
#   commits are ignored (treated as pass).
#
# Usage:
#   .github/scripts/linting/lint_merge_commit_message.sh
#
# Accepts:
#   (no positional args) — operates on HEAD.
#   Environment (optional, forwarded to the branch linter):
#     ALLOWED_COMPONENTS    space-separated list (default: "root core experiment torch")
#     ALLOWED_BRANCH_TYPES  space-separated list (default: "dev hotfix setup")
#
# Stdout on success:
#   - If HEAD is a merge commit with a valid message: `Component name: <component>`
#   - If HEAD is not a merge commit: a short note that linting is skipped
#
# Stderr on failure:
#   - Any diagnostics come from the delegated message/branch linter
#     (.github/scripts/linting/lint_commit_message.sh), which emits ::error:: messages.
#
# Exit codes:
#   0 — HEAD is not a merge commit (skipped), **or** it is a merge commit whose message
#       encodes a valid source branch (component printed to stdout)
#   1 — HEAD is a merge commit but the subject does not match the expected pattern,
#       or the extracted branch fails naming/policy validation
#
# Behaviour:
#   - Uses check_is_merge_commit.sh to detect if HEAD has >1 parent.
#   - If not a merge commit: prints a skip message and exits 0.
#   - If a merge commit: runs lint_commit_message.sh to validate the subject and extract
#     the component; on success, echoes `Component name: <name>`; on failure, exits 1.
#
# Notes:
#   - This script is a thin delegator—its only “policy” is to **enforce the message check
#     on merge commits** and skip everything else.
#   - Ensure the repository is checked out so HEAD is available.

chmod +x .github/scripts/linting/check_is_merge_commit.sh
chmod +x .github/scripts/linting/lint_commit_message.sh

if .github/scripts/linting/check_is_merge_commit.sh &>/dev/null; then
    echo "This is a merge commit, checking message..."
    
    # Lint the commit message
    COMPONENT_NAME=$(.github/scripts/linting/lint_commit_message.sh)
    echo "Component name: $COMPONENT_NAME"
    exit 0
else
    echo "Not a merge commit, skipping linting"
    exit 0
fi
