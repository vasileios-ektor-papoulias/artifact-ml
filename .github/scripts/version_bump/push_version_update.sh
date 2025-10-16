#!/bin/bash
set -euo pipefail

# Purpose:
#   Commit a version-bump change, create an annotated Git tag, and push both the
#   commit and the tag to the remote repository (typically run from CI).
#
# Usage:
#   .github/scripts/version_bump/push_version_update.sh <tag_name> <file_path>
#
# Accepts:
#   <tag_name>   Annotated tag to create and push (e.g., "core-v1.4.0" or "v1.4.0").
#   <file_path>  Path to the file already modified with the new version
#                (e.g., "artifact-core/pyproject.toml" or "pyproject.toml").
#
# Stdout on success:
#   (none) — progress and summaries are written to STDERR for clean CI logs.
#
# Stderr on failure:
#   ::error::-style messages (from this or underlying git commands) describing
#   missing arguments, staging/commit problems, tag creation failures, or push errors.
#
# Exit codes:
#   0 — success (file staged, commit created, tag created, both pushed)
#   1 — validation or git failure (missing args, commit/tag/push failed)
#
# Behaviour:
#   - Stages <file_path> and commits with author/committer set to
#     "github-actions[bot]" and message "Bump version to <tag_name> [skip ci]".
#   - Creates an annotated tag named <tag_name> with message <tag_name>.
#   - Pushes the new commit to the current branch’s remote (origin HEAD).
#   - Pushes tags to the remote (origin).
#
# Notes:
#   - Requires a writable checkout with correct auth to push commits and tags
#     (e.g., GITHUB_TOKEN with permissions: contents: write).
#   - Assumes the working tree is clean except for the version-bumped file.
#   - Fails if the tag already exists on the remote; ensure uniqueness upstream.
#   - The "[skip ci]" marker in the commit message can prevent re-trigger loops.
#
# Examples:
#   # Root project bump to 1.3.0:
#   .github/scripts/version_bump/push_version_update.sh "v1.3.0" "pyproject.toml"
#     --> (stderr) Created git tag: v1.3.0; pushed commit and tag   # exit 0
#
#   # Component bump to 0.9.2 for artifact-core:
#   .github/scripts/version_bump/push_version_update.sh "core-v0.9.2" "artifact-core/pyproject.toml"
#     --> (stderr) Created git tag: core-v0.9.2; pushed commit/tag  # exit 0

TAG_NAME="${1-}"
FILE_PATH="${2-}"

if [[ -z "$TAG_NAME" || -z "$FILE_PATH" ]]; then
    echo "Usage: $0 <tag_name> <file_path>" >&2
    exit 1
fi

git add "$FILE_PATH"

GIT_AUTHOR_NAME="github-actions[bot]"
GIT_AUTHOR_EMAIL="github-actions[bot]@users.noreply.github.com"
GIT_COMMITTER_NAME="github-actions[bot]"
GIT_COMMITTER_EMAIL="github-actions[bot]@users.noreply.github.com"

COMMIT_MSG="Bump version to $TAG_NAME [skip ci]"

git -c user.name="$GIT_AUTHOR_NAME" -c user.email="$GIT_AUTHOR_EMAIL" commit -m "$COMMIT_MSG"

git -c user.name="$GIT_AUTHOR_NAME" -c user.email="$GIT_AUTHOR_EMAIL" tag -a "$TAG_NAME" -m "$TAG_NAME"

echo "Created git tag: $TAG_NAME" >&2

echo "Pushing changes and tags to remote repository..."
git -c user.name="$GIT_AUTHOR_NAME" -c user.email="$GIT_AUTHOR_EMAIL" push origin HEAD
git -c user.name="$GIT_AUTHOR_NAME" -c user.email="$GIT_AUTHOR_EMAIL" push origin --tags

echo "Successfully bumped version to $TAG_NAME"
