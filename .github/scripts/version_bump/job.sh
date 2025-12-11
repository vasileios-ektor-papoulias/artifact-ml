#!/bin/bash
set -euo pipefail

# Purpose:
#   Orchestrate a version-bump job given a specific component and bump type.
#   Resolves the correct pyproject.toml and performs the bump+tag step — or
#   skips when policy says so.
#
# Usage:
#   .github/scripts/version_bump/job.sh <component_name> <bump_type>
#
# Accepts:
#   $1 — component_name (e.g., "core", "experiment", "torch", or "root")
#   $2 — bump_type (one of: "patch", "minor", "major", or "no-bump")
#
# Stdout on success:
#   component=<name> — outputs the component name (for logging).
#   version=<version> — outputs the new version (e.g., 1.2.3, for logging).
#   If skipped (no-bump or root), outputs component= and version= (empty values).
#   Note: Workflows do not need to parse these outputs; they are for logging only.
#
# Stderr:
#   Progress messages and ::error::-style diagnostics explaining why the bump
#   could not proceed (e.g., missing pyproject.toml).
#
# Exit codes:
#   0 — success (version bumped) or intentional skip (no-bump, or component==root)
#   1 — failure: missing arguments, invalid bump type, missing pyproject.toml, or delegated script failure
#
# Behaviour:
#   - Validates that component_name and bump_type are provided
#   - Calls:
#       • .github/scripts/version_bump/get_pyproject_path.sh <component?>
#         → resolves the pyproject.toml to edit
#       • .github/scripts/version_bump/bump_component_version.sh <bump> <component> <path>
#         → edits version and pushes an annotated tag
#   - Skips with exit 0 when:
#       • bump type == no-bump
#       • component == root (root changes do not bump versions)
#   - Fails with exit 1 when:
#       • arguments are missing
#       • pyproject.toml cannot be found
#       • any delegated script exits non-zero (due to `set -euo pipefail`)
#
# Notes:
#   - Designed to run on merges to `main` after PRs from component branches.
#   - All user-visible progress is printed to STDERR for clean CI logs.
#   - The component name and bump type are now explicit inputs (extracted by the workflow).
#
# Examples:
#   # PR marked no-bump → job exits 0 without changing versions
#   .github/scripts/version_bump/job.sh core no-bump
#     --> (stderr) "Bump type is 'no-bump', skipping version bump"       # exit 0
#
#   # Component 'core', bump type 'minor' → version updated + tag pushed
#   .github/scripts/version_bump/job.sh core minor
#     --> (stderr) "Using pyproject.toml at: artifact-core/pyproject.toml"  # exit 0
#
#   # Missing pyproject.toml for component → error
#   .github/scripts/version_bump/job.sh core patch
#     --> (stderr) "::error::Failed to find a valid pyproject.toml ..."   # exit 1

# Validate arguments
COMPONENT_NAME="${1:-}"
BUMP_TYPE="${2:-}"

if [[ -z "$COMPONENT_NAME" ]]; then
  echo "::error::Component name is required as the first argument" >&2
  echo "Usage: $0 <component_name> <bump_type>" >&2
  exit 1
fi

if [[ -z "$BUMP_TYPE" ]]; then
  echo "::error::Bump type is required as the second argument" >&2
  echo "Usage: $0 <component_name> <bump_type>" >&2
  exit 1
fi

echo "Component: $COMPONENT_NAME" >&2
echo "Bump type: $BUMP_TYPE" >&2

# Validate bump type
case "$BUMP_TYPE" in
  patch|minor|major|no-bump)
    # Valid bump type
    ;;
  *)
    echo "::error::Invalid bump type '$BUMP_TYPE'. Must be one of: patch, minor, major, no-bump" >&2
    exit 1
    ;;
esac

if [ "$BUMP_TYPE" = "no-bump" ]; then
  echo "Bump type is 'no-bump', skipping version bump" >&2
  echo "This PR is marked as not requiring a version bump" >&2
  echo "component="
  echo "version="
  exit 0
fi

if [ "$COMPONENT_NAME" = "root" ]; then
  echo "Component is 'root', skipping version bump" >&2
  echo "Root component changes should not trigger version bumps" >&2
  echo "component="
  echo "version="
  exit 0
fi

PYPROJECT_PATH=$(.github/scripts/version_bump/get_pyproject_path.sh "$COMPONENT_NAME") || {
  echo "Error: Failed to find a valid pyproject.toml file for component '$COMPONENT_NAME'" >&2
  echo "Version bump cannot proceed without a valid pyproject.toml file" >&2
  exit 1
}
echo "Using pyproject.toml at: $PYPROJECT_PATH" >&2

# bump_component_version.sh outputs the new version to stdout
NEW_VERSION=$(.github/scripts/version_bump/bump_component_version.sh "$BUMP_TYPE" "$COMPONENT_NAME" "$PYPROJECT_PATH")
echo "Successfully completed version bump job" >&2
echo "component=$COMPONENT_NAME"
echo "version=$NEW_VERSION"
