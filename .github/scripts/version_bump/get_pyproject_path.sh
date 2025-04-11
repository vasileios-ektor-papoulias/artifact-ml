#!/bin/bash
set -e

# Usage: .github/scripts/version_bump/get_pyproject_path.sh [component_name]
# Returns: The path to the pyproject.toml file
# If component_name is provided, returns <component_name>/pyproject.toml if it exists
# If no component_name is provided, returns pyproject.toml if it exists at the project root
# Exits with error if the required pyproject.toml file doesn't exist

COMPONENT_NAME=$1

if [[ -n "$COMPONENT_NAME" ]]; then
    PYPROJECT_PATH="$COMPONENT_NAME/pyproject.toml"
    if [[ ! -f "$PYPROJECT_PATH" ]]; then
        echo "Error: Component pyproject.toml not found at $PYPROJECT_PATH" >&2
        exit 1
    fi
else
    PYPROJECT_PATH="pyproject.toml"
    if [[ ! -f "$PYPROJECT_PATH" ]]; then
        echo "Error: No valid pyproject.toml found at $PYPROJECT_PATH" >&2
        exit 1
    fi
fi

echo "$PYPROJECT_PATH"
