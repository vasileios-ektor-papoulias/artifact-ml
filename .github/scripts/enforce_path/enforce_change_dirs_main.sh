#!/usr/bin/env bash
#
# enforce_change_dirs_for_main.sh
#
# Enforces that PRs to main only modify files in allowed directories
# based on the source branch.
#
# Usage:
#   enforce_change_dirs_for_main.sh <head_ref> <base_ref>
#
# Arguments:
#   head_ref - The source branch of the PR (e.g., "dev-core", "hotfix-torch/fix-bug")
#   base_ref - The target branch of the PR (should be "main")
#
# Exit Codes:
#   0 - Check passed (changes are in allowed directories)
#   1 - Check failed (changes in disallowed directories or invalid branch)
#
# Examples:
#   enforce_change_dirs_for_main.sh "dev-core" "main"
#   enforce_change_dirs_for_main.sh "hotfix-experiment/fix-bug" "main"
#   enforce_change_dirs_for_main.sh "setup-root/add-ci" "main"

set -euo pipefail

# Get the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Allow REPO_ROOT to be overridden for testing
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

HEAD_REF="${1:-}"
BASE_REF="${2:-}"

if [[ -z "$HEAD_REF" ]]; then
    echo "Error: head_ref (source branch) is required" >&2
    echo "Usage: enforce_change_dirs_for_main.sh <head_ref> <base_ref>" >&2
    exit 1
fi

if [[ -z "$BASE_REF" ]]; then
    echo "Error: base_ref (target branch) is required" >&2
    echo "Usage: enforce_change_dirs_for_main.sh <head_ref> <base_ref>" >&2
    exit 1
fi

echo "Checking PR from '$HEAD_REF' to '$BASE_REF'"

# Path to enforce_path scripts
ENFORCE_PATH_DIR="$REPO_ROOT/.github/scripts/enforce_path"

# dev-core → main
if [[ "$HEAD_REF" == "dev-core" ]]; then
    echo "PR from dev-core: enforcing changes only in artifact-core/"
    "$ENFORCE_PATH_DIR/ensure_changed_files_in_dir.sh" "artifact-core" "$BASE_REF"

# dev-experiment → main
elif [[ "$HEAD_REF" == "dev-experiment" ]]; then
    echo "PR from dev-experiment: enforcing changes only in artifact-experiment/"
    "$ENFORCE_PATH_DIR/ensure_changed_files_in_dir.sh" "artifact-experiment" "$BASE_REF"

# dev-torch → main
elif [[ "$HEAD_REF" == "dev-torch" ]]; then
    echo "PR from dev-torch: enforcing changes only in artifact-torch/"
    "$ENFORCE_PATH_DIR/ensure_changed_files_in_dir.sh" "artifact-torch" "$BASE_REF"

# hotfix-core/* → main
elif [[ "$HEAD_REF" == hotfix-core/* ]]; then
    echo "PR from hotfix-core: enforcing changes only in artifact-core/"
    "$ENFORCE_PATH_DIR/ensure_changed_files_in_dir.sh" "artifact-core" "$BASE_REF"

# hotfix-experiment/* → main
elif [[ "$HEAD_REF" == hotfix-experiment/* ]]; then
    echo "PR from hotfix-experiment: enforcing changes only in artifact-experiment/"
    "$ENFORCE_PATH_DIR/ensure_changed_files_in_dir.sh" "artifact-experiment" "$BASE_REF"

# hotfix-torch/* → main
elif [[ "$HEAD_REF" == hotfix-torch/* ]]; then
    echo "PR from hotfix-torch: enforcing changes only in artifact-torch/"
    "$ENFORCE_PATH_DIR/ensure_changed_files_in_dir.sh" "artifact-torch" "$BASE_REF"

# *-root/* → main (setup-root/*, hotfix-root/*)
elif [[ "$HEAD_REF" == *-root/* ]]; then
    echo "PR from root branch: enforcing changes OUTSIDE component source directories"
    "$ENFORCE_PATH_DIR/ensure_changed_files_outside_dirs.sh" "$BASE_REF" \
        "artifact-core/artifact_core" \
        "artifact-experiment/artifact_experiment" \
        "artifact-torch/artifact_torch"

else
    echo "Error: Unrecognized source branch pattern: '$HEAD_REF'" >&2
    echo "Expected patterns: dev-<component>, hotfix-<component>/*, setup-root/*, hotfix-root/*" >&2
    exit 1
fi

echo "✅ Directory enforcement check passed!"

