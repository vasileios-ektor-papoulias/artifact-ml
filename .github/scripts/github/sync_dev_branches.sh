#!/usr/bin/env bash
#
# sync_dev_branches.sh
#
# Fast-forwards long-lived dev integration branches (dev-core, dev-experiment,
# dev-torch) to the current tip of main.
#
# Usage:
#   sync_dev_branches.sh [target]
#
# Arguments:
#   target - Which dev branch to sync: "all" (default), "core", "experiment",
#            or "torch".
#
# Exit Codes:
#   0 - All requested dev branches are at (or were fast-forwarded to) main.
#   1 - Invalid target, or a dev branch has diverged from main (holds commits
#       not reachable from main) and requires manual reconciliation.
#
# Behaviour:
#   - For each requested component, verifies that origin/dev-<component> is an
#     ancestor of origin/main (i.e., a pure fast-forward is possible).
#   - If the branch is already at main's tip, does nothing.
#   - Otherwise pushes origin/main to the dev branch ref. No merge commits or
#     history rewrites are ever produced.
#   - Divergence aborts the entire run with an error: dev branches are expected
#     to only ever receive component work that flows to main via PRs, so unique
#     commits on a dev branch indicate something needs human attention.
#
# Examples:
#   sync_dev_branches.sh            # sync all three dev branches
#   sync_dev_branches.sh torch      # sync only dev-torch

set -euo pipefail

TARGET="${1:-all}"

case "$TARGET" in
    all)
        COMPONENTS="core experiment torch"
        ;;
    core|experiment|torch)
        COMPONENTS="$TARGET"
        ;;
    *)
        echo "::error::Invalid target '$TARGET'. Expected: all, core, experiment, or torch." >&2
        exit 1
        ;;
esac

git fetch origin main

MAIN_SHA=$(git rev-parse origin/main)
echo "main is at: $MAIN_SHA"

for COMPONENT in $COMPONENTS; do
    BRANCH="dev-$COMPONENT"
    git fetch origin "$BRANCH"
    BRANCH_SHA=$(git rev-parse "origin/$BRANCH")

    if [ "$BRANCH_SHA" = "$MAIN_SHA" ]; then
        echo "✅ $BRANCH is already at main's tip."
        continue
    fi

    if ! git merge-base --is-ancestor "origin/$BRANCH" origin/main; then
        echo "::error::$BRANCH ($BRANCH_SHA) has commits not on main and cannot be fast-forwarded." >&2
        echo "::error::Merge or land the outstanding $BRANCH work via PR before syncing." >&2
        exit 1
    fi

    echo "Fast-forwarding $BRANCH: $BRANCH_SHA -> $MAIN_SHA"
    git push origin "origin/main:refs/heads/$BRANCH"
    echo "✅ $BRANCH synced to main."
done

echo "✅ Dev branch sync complete."
