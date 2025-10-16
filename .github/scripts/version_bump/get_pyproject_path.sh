#!/bin/bash
set -euo pipefail

# Purpose:
#   Resolve the path to the appropriate pyproject.toml.
#   If a component name is supplied, return "<component>/pyproject.toml";
#   otherwise, return the repository root "pyproject.toml".
#
# Usage:
#   .github/scripts/version_bump/get_pyproject_path.sh [component_name]
#
# Accepts:
#   [component_name]  (optional) component directory name (repo-relative)
#
# Stdout on success:
#   The resolved path to pyproject.toml (repo-relative), e.g.:
#     • "pyproject.toml"
#     • "artifact-core/pyproject.toml"
#
# Stderr on failure:
#   ::error::-style message (or plain "Error: ...") indicating the missing file/path.
#
# Exit codes:
#   0 — path resolved and printed to stdout
#   1 — required pyproject.toml does not exist at the expected location
#
# Behaviour:
#   - If a component name is provided, checks "<component_name>/pyproject.toml" exists.
#   - If no component name, checks "pyproject.toml" at the repository root exists.
#   - Prints the valid path to stdout; otherwise exits with an error.
#
# Notes:
#   - The component name is used verbatim (no normalization or search).
#   - Must be run from the repository root (paths are repo-relative).
#
# Examples:
#   .github/scripts/version_bump/get_pyproject_path.sh
#     --> pyproject.toml
#   .github/scripts/version_bump/get_pyproject_path.sh artifact-core
#     --> artifact-core/pyproject.toml

COMPONENT_NAME="${1-}"

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
