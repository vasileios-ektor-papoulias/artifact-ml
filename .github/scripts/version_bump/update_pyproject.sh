#!/bin/bash
set -euo pipefail

# Purpose:
#   Bump version in pyproject.toml using Poetry's built-in version command.
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
#   missing file, or invalid bump type.
#
# Exit codes:
#   0 — success (file updated and new version printed)
#   1 — validation failure (bad/missing args; file missing; invalid bump type; Poetry error)
#
# Behaviour:
#   - Validates input arguments and file existence.
#   - Navigates to the directory containing pyproject.toml.
#   - Uses Poetry's `version` command to bump the version in place.
#   - Prints only the new version to STDOUT.
#
# Notes:
#   - Requires Poetry to be installed and available in PATH.
#   - Poetry's version command handles reading, calculating, and updating the version.
#   - The --short flag returns just the version number without extra output.
#   - Works with PEP 621-style pyproject.toml files.
#
# Examples:
#   .github/scripts/version_bump/update_pyproject.sh pyproject.toml minor
#     --> 1.3.0
#
#   .github/scripts/version_bump/update_pyproject.sh artifact-core/pyproject.toml patch
#     --> 0.1.1

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

# Validate bump type
case "$BUMP_TYPE" in
  patch|minor|major)
    ;;
  *)
    echo "Error: Unknown bump type: $BUMP_TYPE. Must be one of: patch, minor, major" >&2
    exit 1
    ;;
esac

# Navigate to the directory containing pyproject.toml
PYPROJECT_DIR=$(dirname "$PYPROJECT_PATH")
ORIGINAL_DIR=$(pwd)

cd "$PYPROJECT_DIR"

# Use Poetry to bump the version
# --short flag returns just the version number without extra output
NEW_VERSION=$(poetry version "$BUMP_TYPE" --short)

cd "$ORIGINAL_DIR"

echo "$NEW_VERSION"
