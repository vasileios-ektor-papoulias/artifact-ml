#!/bin/bash
set -euo pipefail

# Purpose:
#   Gate merge commits by ensuring their **description/body** uses a valid semantic
#   bump prefix. Non-merge commits are ignored (treated as pass).
#
# Usage:
#   .github/scripts/linting/lint_merge_commit_description.sh
#
# Accepts:
#   (no positional args) — operates on HEAD.
#
# Stdout on success:
#   - If HEAD is a merge commit with a valid description: prints `Bump type: <patch|minor|major|no-bump>`
#   - If HEAD is not a merge commit: prints a short note that linting is skipped
#
# Stderr on failure:
#   - Any diagnostics come from the delegated description linter
#     (.github/scripts/linting/lint_commit_description.sh), which emits ::error:: messages.
#
# Exit codes:
#   0 — HEAD is not a merge commit (skipped), **or** it is a merge commit with a valid bump prefix
#   1 — HEAD is a merge commit but the description/body fails the bump-prefix check
#
# Behaviour:
#   - Uses check_is_merge_commit.sh to determine if HEAD has >1 parent.
#   - If not a merge commit: prints a skip message and exits 0.
#   - If a merge commit: runs lint_commit_description.sh to validate the description/body;
#     on success, echoes the detected bump type; on failure, exits 1.
#
# Notes:
#   - This script is a thin delegator—its only “policy” is to **enforce the description
#     check on merge commits** and skip everything else.
#   - Ensure the repository is checked out so HEAD is available.

chmod +x .github/scripts/linting/check_is_merge_commit.sh
chmod +x .github/scripts/linting/lint_commit_description.sh

if .github/scripts/linting/check_is_merge_commit.sh &>/dev/null; then
    echo "This is a merge commit, checking description..."
    
    # Lint the commit description
    BUMP_TYPE=$(.github/scripts/linting/lint_commit_description.sh)
    echo "Bump type: $BUMP_TYPE"
    exit 0
else
    echo "Not a merge commit, skipping linting"
    exit 0
fi
