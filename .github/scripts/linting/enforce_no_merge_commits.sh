#!/usr/bin/env bash
#
# enforce_no_merge_commits.sh
#
# Ensures that a PR head branch contains no merge commits in its unique
# history on top of the base branch.
#
# Usage:
#   enforce_no_merge_commits.sh <head_ref> <base_ref>
#
# Arguments:
#   head_ref - The source branch of the PR (e.g., "feature-core/add-x").
#   base_ref - The target branch of the PR (e.g., "main", "dev-core").
#
# Exit Codes:
#   0 - Check passed (no merge commits in the branch's unique history, or
#       the head is a dev integration branch, which is exempt).
#   1 - Missing arguments, or merge commits found in the branch.
#
# Behaviour:
#   - Dev integration branches (dev-*) are exempt: their history legitimately
#     contains the merge commits of feature/fix PRs, so PRs from dev-<component>
#     to main are always allowed through.
#   - Fetches origin/<head_ref> and origin/<base_ref> and inspects the range
#     origin/<base_ref>..origin/<head_ref>. This deliberately avoids HEAD:
#     on pull_request events, Actions checks out a synthetic merge commit
#     (refs/pull/N/merge) which would register as a false positive.
#   - Any merge commit in the range fails the check with guidance to rebase.
#     (Merge commits typically appear when the base is merged back into a
#     stale branch, e.g. via GitHub's default "Update branch" button; the
#     repo convention is to rebase instead --- see devops_processes.md.)
#
# Examples:
#   enforce_no_merge_commits.sh "feature-core/add-x" "dev-core"
#   enforce_no_merge_commits.sh "hotfix-torch/fix-y" "main"
#   enforce_no_merge_commits.sh "dev-core" "main"        # exempt, passes

set -euo pipefail

HEAD_REF="${1:-}"
BASE_REF="${2:-}"

if [[ -z "$HEAD_REF" || -z "$BASE_REF" ]]; then
    echo "::error::Usage: enforce_no_merge_commits.sh <head_ref> <base_ref>" >&2
    exit 1
fi

if [[ "$HEAD_REF" == dev-* ]]; then
    echo "✅ Skipping no-merge-commits check: '$HEAD_REF' is a dev integration branch (PR merge commits expected)."
    exit 0
fi

echo "Checking that '$HEAD_REF' contains no merge commits on top of '$BASE_REF'"

git fetch origin "$HEAD_REF" "$BASE_REF"

MERGE_COMMITS=$(git log --merges --format="%h %s" "origin/$BASE_REF..origin/$HEAD_REF")

if [[ -n "$MERGE_COMMITS" ]]; then
    echo "::error::Branch '$HEAD_REF' contains merge commits:" >&2
    echo "$MERGE_COMMITS" >&2
    echo "::error::Do not merge '$BASE_REF' (or any branch) into '$HEAD_REF'." >&2
    echo "::error::Update stale branches by rebasing instead: git rebase origin/$BASE_REF (or GitHub's 'Update with rebase')." >&2
    exit 1
fi

echo "✅ No-merge-commits check passed!"
