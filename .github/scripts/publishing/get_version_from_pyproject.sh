#!/bin/bash
set -euo pipefail

# Purpose:
#   Extract version from pyproject.toml for a given component.
#
# Usage:
#   .github/scripts/publishing/get_version_from_pyproject.sh <component>
#
# Accepts:
#   <component>  Component name (core, experiment, torch)
#
# Stdout on success:
#   Version string (e.g., "1.2.3")
#
# Exit codes:
#   0 — success
#   1 — validation failure or file not found
#
# Examples:
#   .github/scripts/publishing/get_version_from_pyproject.sh core
#     --> 1.2.3

COMPONENT="${1:-}"

if [ -z "$COMPONENT" ]; then
  echo "::error::Missing required parameter: component" >&2
  echo "::error::Usage: $0 <component>" >&2
  exit 1
fi

# Validate component
if [[ ! "$COMPONENT" =~ ^(core|experiment|torch)$ ]]; then
  echo "::error::Invalid component: $COMPONENT" >&2
  echo "::error::Must be one of: core, experiment, torch" >&2
  exit 1
fi

COMPONENT_DIR="artifact-${COMPONENT}"
PYPROJECT_PATH="${COMPONENT_DIR}/pyproject.toml"

if [ ! -f "$PYPROJECT_PATH" ]; then
  echo "::error::pyproject.toml not found at: $PYPROJECT_PATH" >&2
  exit 1
fi

# Extract version from pyproject.toml
# Looks for: version = "1.2.3"
VERSION=$(grep -E '^version\s*=\s*"[^"]+"\s*$' "$PYPROJECT_PATH" | sed -E 's/^version\s*=\s*"([^"]+)"\s*$/\1/')

if [ -z "$VERSION" ]; then
  echo "::error::Could not extract version from $PYPROJECT_PATH" >&2
  exit 1
fi

# Validate version format
if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "::error::Invalid version format: $VERSION" >&2
  echo "::error::Expected format: X.Y.Z" >&2
  exit 1
fi

echo "$VERSION"
exit 0

