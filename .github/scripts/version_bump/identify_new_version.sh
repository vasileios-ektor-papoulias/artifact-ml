#!/bin/bash
set -euo pipefail

# Purpose:
#   Compute the next semantic version by applying a bump type (patch|minor|major)
#   to a given current version string (X.Y.Z).
#
# Usage:
#   .github/scripts/version_bump/identify_new_version.sh <current_version> <bump_type>
#
# Accepts:
#   <current_version>  required  SemVer in the form X.Y.Z (e.g., 1.2.3)
#   <bump_type>        required  One of: patch | minor | major
#
# Stdout on success:
#   The new version string (e.g., "1.3.0")
#
# Stderr on failure:
#   ::error::-style (or plain "Error: ...") diagnostics describing invalid input
#   or unknown bump types.
#
# Exit codes:
#   0 — success; new version printed to stdout
#   1 — validation failure (missing args, bad version format, or unknown bump type)
#
# Behaviour:
#   - Validates that <current_version> matches X.Y.Z (digits only).
#   - Parses X, Y, Z as decimal numbers (defensive against leading zeros).
#   - Applies bump rules:
#       • patch → (X, Y, Z+1)
#       • minor → (X, Y+1, 0)
#       • major → (X+1, 0, 0)
#   - Prints the new version to STDOUT and a brief “Bumping version …” message to STDERR.
#
# Notes:
#   - This script performs purely mechanical SemVer increments; it does not read files,
#     tags, or commit history.
#   - Leading zeros are not preserved; components are treated as decimal integers.
#
# Examples:
#   .github/scripts/version_bump/identify_new_version.sh 1.2.3 patch
#     --> 1.2.4
#   .github/scripts/version_bump/identify_new_version.sh 1.2.3 minor
#     --> 1.3.0
#   .github/scripts/version_bump/identify_new_version.sh 1.2.3 major
#     --> 2.0.0


CURRENT_VERSION="${1-}"
BUMP_TYPE="${2-}"

if [[ -z "$CURRENT_VERSION" || -z "$BUMP_TYPE" ]]; then
    echo "Usage: $0 <current_version> {patch|minor|major}" >&2
    exit 1
fi

if ! [[ "$CURRENT_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: Current version must be in format X.Y.Z (e.g., 1.2.3)" >&2
    exit 1
fi

IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT_VERSION"

MAJOR=$((10#$MAJOR))
MINOR=$((10#$MINOR))
PATCH=$((10#$PATCH))

case "$BUMP_TYPE" in
  patch)
    PATCH=$((PATCH + 1))
    ;;
  minor)
    MINOR=$((MINOR + 1))
    PATCH=0
    ;;
  major)
    MAJOR=$((MAJOR + 1))
    MINOR=0
    PATCH=0
    ;;
  *)
    echo "Error: Unknown bump type: $BUMP_TYPE. Must be one of: patch, minor, major" >&2
    exit 1
    ;;
esac

NEW_VERSION="$MAJOR.$MINOR.$PATCH"
echo "Bumping version: $CURRENT_VERSION -> $NEW_VERSION" >&2

echo "$NEW_VERSION"
