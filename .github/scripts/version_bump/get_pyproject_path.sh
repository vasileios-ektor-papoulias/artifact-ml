#!/bin/bash
set -euo pipefail

# Purpose:
#   Resolve the path to the appropriate pyproject.toml.
#   Maps component names (core, experiment, torch) to their artifact directories.
#
# Usage:
#   .github/scripts/version_bump/get_pyproject_path.sh [component_name]
#
# Accepts:
#   [component_name]  (optional) component name (core, experiment, torch, or root)
#
# Stdout on success:
#   The resolved path to pyproject.toml (repo-relative), e.g.:
#     • "pyproject.toml" (for root or no component)
#     • "artifact-core/pyproject.toml" (for core)
#
# Stderr on failure:
#   ::error::-style message (or plain "Error: ...") indicating the missing file/path.
#
# Exit codes:
#   0 — path resolved and printed to stdout
#   1 — required pyproject.toml does not exist at the expected location
#
# Behaviour:
#   - Maps component name to directory: core -> artifact-core, etc.
#   - If component is "root" or empty, uses repository root "pyproject.toml".
#   - Prints the valid path to stdout; otherwise exits with an error.
#
# Notes:
#   - Component names are mapped: core -> artifact-core, experiment -> artifact-experiment, etc.
#   - Must be run from the repository root (paths are repo-relative).
#
# Examples:
#   .github/scripts/version_bump/get_pyproject_path.sh
#     --> pyproject.toml
#   .github/scripts/version_bump/get_pyproject_path.sh core
#     --> artifact-core/pyproject.toml
#   .github/scripts/version_bump/get_pyproject_path.sh experiment
#     --> artifact-experiment/pyproject.toml

COMPONENT_NAME="${1-}"

if [[ -n "$COMPONENT_NAME" ]]; then
    # Map component name to directory (core -> artifact-core, etc.)
    if [[ "$COMPONENT_NAME" == "root" ]]; then
        PYPROJECT_PATH="pyproject.toml"
    else
        PYPROJECT_PATH="artifact-$COMPONENT_NAME/pyproject.toml"
    fi
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
