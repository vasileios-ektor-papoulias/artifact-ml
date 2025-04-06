#!/bin/bash
set -e

# Usage: ./scripts/bump_version.sh <subrepo-name> {patch|minor|major}
PACKAGE=$1
BUMP_TYPE=$2

if [[ -z "$PACKAGE" || -z "$BUMP_TYPE" ]]; then
    echo "Usage: $0 <subrepo-name> {patch|minor|major}"
    exit 1
fi

PACKAGE_PATH="packages/$PACKAGE"
PYPROJECT_FILE="$PACKAGE_PATH/pyproject.toml"

if [[ ! -f "$PYPROJECT_FILE" ]]; then
    echo "Error: pyproject.toml not found for package '$PACKAGE'"
    exit 1
fi

# Extract current version
CURRENT_VERSION=$(grep '^version' "$PYPROJECT_FILE" | head -1 | sed 's/.*"\(.*\)".*/\1/')
IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT_VERSION"

# Bump logic
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
    echo "Unknown bump type: $BUMP_TYPE"
    exit 1
    ;;
esac

NEW_VERSION="$MAJOR.$MINOR.$PATCH"
echo "🔖 Bumping $PACKAGE version: $CURRENT_VERSION → $NEW_VERSION"

# Replace version in pyproject.toml
sed -i "s/version = \"$CURRENT_VERSION\"/version = \"$NEW_VERSION\"/" "$PYPROJECT_FILE"

# Git commit + tag
git config --global user.name "github-actions[bot]"
git config --global user.email "github-actions[bot]@users.noreply.github.com"

git add "$PYPROJECT_FILE"
git commit -m "chore($PACKAGE): bump version to v$NEW_VERSION [skip ci]"
git tag -a "$PACKAGE-v$NEW_VERSION" -m "$PACKAGE v$NEW_VERSION"