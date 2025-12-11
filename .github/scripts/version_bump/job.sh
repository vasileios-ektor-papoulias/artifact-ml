#!/bin/bash
set -euo pipefail

# Purpose:
#   Orchestrate a version-bump job after changes land (typically on `main`).
#   Determines the bump type and target component, resolves the correct
#   pyproject.toml, and performs the bump+tag step — or skips when policy says so.
#
# Usage:
#   .github/scripts/version_bump/job.sh
#
# Accepts:
#   (no CLI args) — this script delegates to helper scripts to discover:
#     • bump type (from the commit description)
#     • component name (from the commit message / branch naming)
#     • pyproject.toml path (from repo layout and component)
#
# Stdout on success:
#   component=<name> — outputs the component name for downstream steps (e.g., triggering publish).
#   component= — (empty) if skipped due to no-bump or root component.
#
# Stderr on failure:
#   ::error::-style diagnostics from this script or its delegates explaining
#   why the bump could not proceed (e.g., missing pyproject.toml).
#
# Exit codes:
#   0 — success (version bumped) or intentional skip (no-bump, or component==root)
#   1 — failure in any delegated step or required file missing
#
# Behaviour:
#   - Calls:
#       • .github/scripts/version_bump/get_bump_type.sh
#         → expects: patch | minor | major | no-bump
#       • .github/scripts/version_bump/get_component_name.sh
#         → may return empty (root-level)
#       • .github/scripts/version_bump/get_pyproject_path.sh <component?>
#         → resolves the pyproject.toml to edit
#       • .github/scripts/version_bump/bump_component_version.sh <bump> <component> <path>
#         → edits version and pushes an annotated tag
#   - Skips with exit 0 when:
#       • bump type == no-bump
#       • component == root (root changes do not bump versions)
#   - Fails with exit 1 when:
#       • pyproject.toml cannot be found
#       • any delegated script exits non-zero (due to `set -euo pipefail`)
#
# Notes:
#   - Designed to run on merges to `main` after PRs from component branches.
#   - All user-visible progress is printed to STDERR for clean CI logs.
#
# Examples:
#   # PR marked no-bump → job exits 0 without changing versions
#   .github/scripts/version_bump/job.sh
#     --> (stderr) "Bump type is 'no-bump', skipping version bump"       # exit 0
#
#   # Component 'core', bump type 'minor' → version updated + tag pushed
#   .github/scripts/version_bump/job.sh
#     --> (stderr) "Using pyproject.toml at: artifact-core/pyproject.toml"  # exit 0
#
#   # Missing pyproject.toml for component → error
#   .github/scripts/version_bump/job.sh
#     --> (stderr) "::error::Failed to find a valid pyproject.toml ..."   # exit 1


BUMP_TYPE=$(.github/scripts/version_bump/get_bump_type.sh)
echo "Using bump type: $BUMP_TYPE" >&2

if [ "$BUMP_TYPE" = "no-bump" ]; then
  echo "Bump type is 'no-bump', skipping version bump" >&2
  echo "This PR is marked as not requiring a version bump" >&2
  echo "component="
  exit 0
fi

COMPONENT_NAME=$(.github/scripts/version_bump/get_component_name.sh)
if [[ -z "$COMPONENT_NAME" ]]; then
    echo "No component name found, will use root pyproject.toml if it exists" >&2
fi

if [ "$COMPONENT_NAME" = "root" ]; then
  echo "Component is 'root', skipping version bump" >&2
  echo "Root component changes should not trigger version bumps" >&2
  echo "component="
  exit 0
fi

PYPROJECT_PATH=$(.github/scripts/version_bump/get_pyproject_path.sh "$COMPONENT_NAME") || {
  echo "Error: Failed to find a valid pyproject.toml file for component '$COMPONENT_NAME'" >&2
  echo "Version bump cannot proceed without a valid pyproject.toml file" >&2
  exit 1
}
echo "Using pyproject.toml at: $PYPROJECT_PATH" >&2

.github/scripts/version_bump/bump_component_version.sh "$BUMP_TYPE" "$COMPONENT_NAME" "$PYPROJECT_PATH"
echo "Successfully completed version bump job" >&2
echo "component=$COMPONENT_NAME"
