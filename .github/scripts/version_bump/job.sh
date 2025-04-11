#!/bin/bash
set -e

# Usage: .github/scripts/version_bump/job.sh
# This script bumps the version based on the merge commit message or branch name
# It's designed to run on main after a PR merge from a dev-<component_name> branch
# It requires no input parameters as it gets all needed information from other scripts

BUMP_TYPE=$(.github/scripts/version_bump/get_bump_type.sh)
echo "Using bump type: $BUMP_TYPE"

if [ "$BUMP_TYPE" = "no-bump" ]; then
  echo "Bump type is 'no-bump', skipping version bump"
  echo "This PR is marked as not requiring a version bump"
  exit 0
fi

COMPONENT_NAME=$(.github/scripts/version_bump/get_component_name.sh)
if [[ -z "$COMPONENT_NAME" ]]; then
    echo "No component name found, will use root pyproject.toml if it exists"
fi

if [ "$COMPONENT_NAME" = "root" ]; then
  echo "Component is 'root', skipping version bump"
  echo "Root component changes should not trigger version bumps"
  exit 0
fi

PYPROJECT_PATH=$(.github/scripts/version_bump/get_pyproject_path.sh "$COMPONENT_NAME") || {
  echo "Error: Failed to find a valid pyproject.toml file for component '$COMPONENT_NAME'"
  echo "Version bump cannot proceed without a valid pyproject.toml file"
  exit 1
}
echo "Using pyproject.toml at: $PYPROJECT_PATH"

.github/scripts/version_bump/bump_component_version.sh "$BUMP_TYPE" "$COMPONENT_NAME" "$PYPROJECT_PATH"
echo "Successfully completed version bump job"
