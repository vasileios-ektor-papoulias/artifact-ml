#!/bin/bash
set -e

# Purpose:
#   Validate a commit’s *description text* against the repository’s semantic
#   versioning prefix convention and emit the parsed bump type.
#
# Usage:
#   .github/scripts/linting/lint_commit_description.sh
#
# Accepts:
#   (no CLI args) — reads the last commit’s description (body) via `git log -1 --pretty=format:%b`.
#
# Stdout on success:
#   One of: patch | minor | major | no-bump
#
# Stderr on failure:
#   ::error::-prefixed diagnostics describing the required prefix and examples.
#
# Exit codes:
#   0 — description starts with an accepted prefix (type printed to stdout)
#   1 — description missing or does not match the required pattern
#
# Behaviour:
#   - Reads HEAD’s commit *description/body*.
#   - Lowercases it and validates the leading token against:
#       • "patch:" | "minor:" | "major:" | "no-bump:"
#       • scoped variants: "patch(scope):", "minor(scope):", "major(scope):", "no-bump(scope):"
#   - Prints the detected type to STDOUT on success.
#
# Notes:
#   - This script is typically invoked on commits whose description is derived from a **PR title**, hence the common delegation to .github/scripts/linting/detect_bump_pattern.sh.
#
# Examples:
#   "patch: fix login validation bug"      --> patch
#   "minor(ui): add tabbed navigation"     --> minor
#   "major(api): remove deprecated v1"     --> major
#   "no-bump(docs): update README"         --> no-bump


chmod +x .github/scripts/linting/detect_bump_pattern.sh

LAST_COMMIT_DESCRIPTION=$(git log -1 --pretty=format:%b)

if [ -z "$LAST_COMMIT_DESCRIPTION" ]; then
  echo "::error::Merge commit has no description!" >&2
  echo "::error::You must add a description that starts with one of: 'patch:', 'minor:', or 'major:'." >&2
  echo "::error::This is required for the automatic version bumping to work correctly." >&2
  exit 1
fi

echo "Commit description: $LAST_COMMIT_DESCRIPTION" >&2

BUMP_TYPE=$(.github/scripts/linting/detect_bump_pattern.sh "$LAST_COMMIT_DESCRIPTION")

echo "$BUMP_TYPE"
exit 0
