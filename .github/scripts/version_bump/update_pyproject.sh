#!/bin/bash
set -euo pipefail

# Usage: .github/scripts/version_bump/update_pyproject.sh <pyproject_path> <bump_type>
# Returns: The new version number
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
