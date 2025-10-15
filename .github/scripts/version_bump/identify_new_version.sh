#!/bin/bash
set -euo pipefail

# Usage: .github/scripts/version_bump/identify_new_version.sh <current_version> <bump_type>
# Returns: The new version number after applying the bump type to the current version
# Example: .github/scripts/version_bump/identify_new_version.sh "1.2.3" "minor" -> "1.3.0"

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
