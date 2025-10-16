#!/bin/bash
set -euo pipefail

# Purpose:
#   Determine the semantic version bump type for the last commit by validating
#   its description/body and emitting the parsed type.
#
# Usage:
#   .github/scripts/version_bump/get_bump_type.sh
#
# Accepts:
#   (no CLI args) — reads the last commit’s description via helper linter.
#
# Stdout on success:
#   One of: patch | minor | major | no-bump
#
# Stderr on failure:
#   ::error::-prefixed diagnostics bubbled up from the description linter
#   (e.g., missing description or invalid prefix).
#
# Exit codes:
#   0 — bump type successfully determined and printed to stdout
#   1 — unable to determine bump type (e.g., description missing/invalid)
#
# Behaviour:
#   - Delegates validation/parsing to:
#       .github/scripts/linting/lint_commit_description.sh
#     which enforces the allowed prefixes and prints the type.
#   - Echoes a brief informational line to STDERR (for CI logs), then prints
#     only the bump type to STDOUT for downstream steps.
#
# Notes:
#   - Intended for use in version-bump workflows after merge-to-main.
#   - Requires the last commit to have a description that starts with one of:
#       "patch:", "minor:", "major:", or "no-bump:"
#
# Examples:
#   # Last commit body: "minor(ui): add tabs"
#   .github/scripts/version_bump/get_bump_type.sh
#     --> minor

chmod +x .github/scripts/linting/lint_commit_description.sh

BUMP_TYPE=$(.github/scripts/linting/lint_commit_description.sh)

echo "Determined bump_type: $BUMP_TYPE" >&2

echo "$BUMP_TYPE"
