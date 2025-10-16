#!/bin/bash
set -euo pipefail

# Purpose:
#   Read the current version from a given pyproject.toml, compute the next version
#   based on a bump type (patch|minor|major), update the file in place, and print
#   the new version.
#
# Usage:
#   .github/scripts/version_bump/update_pyproject.sh <pyproject_path> <bump_type>
#
# Accepts:
#   <pyproject_path>  Path to the target pyproject.toml (e.g., "pyproject.toml" or "artifact-core/pyproject.toml").
#   <bump_type>       One of: patch | minor | major
#
# Stdout on success:
#   <new_version>
#     e.g., "1.3.0"
#
# Stderr on failure:
#   ::error::-style or plain diagnostics describing missing/invalid arguments,
#   missing file, or invalid current version format.
#
# Exit codes:
#   0 — success (file updated and new version printed)
#   1 — validation failure (bad/missing args; file missing; invalid version format)
#
# Behaviour:
#   - Extracts the current version from the first line matching ^version in <pyproject_path>.
#   - Delegates numeric increment logic to:
#       .github/scripts/version_bump/identify_new_version.sh "<current_version>" "<bump_type>"
#   - Replaces the version in pyproject.toml in place (sed -i).
#   - Prints only the new version to STDOUT.
#
# Notes:
#   - Expects a PEP 621-style line like: version = "X.Y.Z" (semantic version, three integers).
#   - Runs under GNU sed (default on Ubuntu runners). If using macOS locally, ensure GNU sed or adjust -i usage.
#   - Only the first matching version line is updated.
#
# Examples:
#   .github/scripts/version_bump/update_pyproject.sh pyproject.toml minor
#     --> 1.3.0
#
#   .github/scripts/version_bump/update_pyproject.sh artifact-core/pyproject.toml patch
#     --> 0.9.3

PYPROJECT_PATH="${1-}"
BUMP_TYPE="${2-}"

if [[ -z "$PYPROJECT_PATH" || -z "$BUMP_TYPE" ]]; then
    echo "Usage: $0 <pyproject_path> {patch|minor|major}" >&2
    exit 1
fi

if [[ ! -f "$PYPROJECT_PATH" ]]; then
    echo "Error: $PYPROJECT_PATH does not exist" >&2
    exit 1
fi

chmod +x .github/scripts/version_bump/identify_new_version.sh

CURRENT_VERSION=$(grep '^version' "$PYPROJECT_PATH" | head -1 | sed 's/.*"\(.*\)".*/\1/')

NEW_VERSION=$(.github/scripts/version_bump/identify_new_version.sh "$CURRENT_VERSION" "$BUMP_TYPE")

sed -i "s/version = \"$CURRENT_VERSION\"/version = \"$NEW_VERSION\"/" "$PYPROJECT_PATH"

echo "$NEW_VERSION"
