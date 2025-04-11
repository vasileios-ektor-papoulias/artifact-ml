#!/bin/bash
set -e

# Usage: .github/scripts/version_bump/bump_component_version.sh <bump_type> <component_name> <pyproject_path>
# Returns: 
# Bumps the version of the specified component (subrepo) by editing the relevant pyproject.toml file and pusing a corresponding tag.
# The new version is determined by incrementing the old version by bump_type (patch, minor, major).

BUMP_TYPE=$1
COMPONENT_NAME=$2
PYPROJECT_PATH=$3

NEW_VERSION=$(.github/scripts/version_bump/update_pyproject.sh "$3" "$1")

TAG_NAME=$(.github/scripts/version_bump/get_component_tag.sh "$NEW_VERSION" "$COMPONENT_NAME")
echo "Generated tag name: $TAG_NAME"

.github/scripts/version_bump/push_version_update.sh "$TAG_NAME" "$PYPROJECT_PATH"