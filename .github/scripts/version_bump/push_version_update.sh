#!/bin/bash
set -euo pipefail

# Usage: .github/scripts/version_bump/push_version_update.sh <tag_name> <file_path>
# This script handles git operations: commit version-bump changes, tag and push
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
