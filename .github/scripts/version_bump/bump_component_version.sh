#!/bin/bash
set -euo pipefail

# Purpose:
#   Bump a component’s version by updating its pyproject.toml, then create/push
#   a corresponding Git tag for that new version.
#
# Usage:
#   .github/scripts/version_bump/bump_component_version.sh <bump_type> <component_name> <pyproject_path>
#
# Accepts:
#   <bump_type>        One of: patch | minor | major
#   <component_name>   Logical component/subrepo name (e.g., core, experiment, torch, root)
#   <pyproject_path>   Path to the component’s pyproject.toml (repo-relative)
#
# Stdout on success:
#   - The new version string (e.g., "1.2.3") for downstream capture.
#
# Stderr on success:
#   - The generated tag name line, e.g.: "Generated tag name: <tag>"
#   - Any informational output emitted by the helper scripts.
#
# Stderr on failure:
#   ::error::-prefixed diagnostics from this script or its helpers explaining
#   missing/invalid arguments, version parsing failures, or Git/tag push errors.
#
# Exit codes:
#   0 — version successfully bumped and tag push attempted/completed
#   1 — validation or operational failure (e.g., bad args, pyproject update failed,
#       tag formatting failed, or push failed)
#
# Behaviour:
#   - Delegates the version computation + in-file update to:
#       .github/scripts/version_bump/update_pyproject.sh <pyproject_path> <bump_type>
#     (returns the NEW version string on stdout).
#   - Formats a component-scoped tag via:
#       .github/scripts/version_bump/get_component_tag.sh <new_version> <component_name>
#   - Commits, tags, and pushes the version change via:
#       .github/scripts/version_bump/push_version_update.sh <tag_name> <pyproject_path>
#
# Notes:
#   - Assumes the helper scripts exist and are executable.
#   - Requires Git to be configured with push permissions for the current repo/branch.
#   - Expects pyproject.toml to contain a parseable version field the helper can update.
#
# Examples:
#   # Bump core's version (minor) and push a tag for it
#   .github/scripts/version_bump/bump_component_version.sh minor core artifact-core/pyproject.toml
#     --> Generated tag name: core-v1.3.0
#
#   # Patch bump for torch
#   .github/scripts/version_bump/bump_component_version.sh patch torch artifact-torch/pyproject.toml
#     --> Generated tag name: torch-v0.4.6

BUMP_TYPE="${1:-}"
COMPONENT_NAME="${2:-}"
PYPROJECT_PATH="${3:-}"

NEW_VERSION=$(.github/scripts/version_bump/update_pyproject.sh "$3" "$1")

TAG_NAME=$(.github/scripts/version_bump/get_component_tag.sh "$NEW_VERSION" "$COMPONENT_NAME")
echo "Generated tag name: $TAG_NAME" >&2

.github/scripts/version_bump/push_version_update.sh "$TAG_NAME" "$PYPROJECT_PATH"

# Output version for downstream steps
echo "$NEW_VERSION"