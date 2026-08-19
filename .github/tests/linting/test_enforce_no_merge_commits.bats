#!/usr/bin/env bats
#
# Unit tests for enforce_no_merge_commits.sh
#
# These tests exercise the real git logic against throwaway local
# repositories: a bare repo standing in for the GitHub remote, and a clone
# in which the script runs (mirroring the Actions checkout).

setup() {
    # Get the directory containing this test file
    TEST_DIR="$(cd "$(dirname "$BATS_TEST_FILENAME")" && pwd)"
    # Navigate to repo root (assuming tests are in .github/tests/linting/)
    REPO_ROOT="$(cd "$TEST_DIR/../../.." && pwd)"
    SCRIPT_PATH="$REPO_ROOT/.github/scripts/linting/enforce_no_merge_commits.sh"

    WORK_DIR="$(mktemp -d)"
    REMOTE_DIR="$WORK_DIR/remote.git"
    CLONE_DIR="$WORK_DIR/clone"

    # Bare repo standing in for the GitHub remote
    git init --bare --quiet "$REMOTE_DIR"

    # Working clone in which branches are authored and the script runs
    git clone --quiet "$REMOTE_DIR" "$CLONE_DIR" 2>/dev/null
    cd "$CLONE_DIR"
    git config user.email "test@example.com"
    git config user.name "Test User"
    git config commit.gpgsign false

    # Seed main with two commits
    git checkout --quiet -b main
    make_commit "initial commit"
    make_commit "second commit on main"
    git push --quiet origin main
}

teardown() {
    cd "$BATS_TMPDIR" || true
    rm -rf "$WORK_DIR"
}

# Helper: create a commit with a unique file in the current repo
make_commit() {
    local msg="$1"
    echo "$msg" > "file_$(date +%s%N).txt"
    git add -A
    git commit --quiet -m "$msg"
}

@test "enforce_no_merge_commits.sh: fails when arguments are missing" {
    run bash "$SCRIPT_PATH" "feature-core/my-feature"

    [ "$status" -eq 1 ]
    [[ "$output" == *"Usage"* ]]
}

@test "enforce_no_merge_commits.sh: passes for a merge-free feature branch" {
    git checkout --quiet -b feature-core/my-feature
    make_commit "feature work 1"
    make_commit "feature work 2"
    git push --quiet origin feature-core/my-feature

    run bash "$SCRIPT_PATH" "feature-core/my-feature" "main"

    [ "$status" -eq 0 ]
    [[ "$output" == *"No-merge-commits check passed"* ]]
}

@test "enforce_no_merge_commits.sh: passes for a stale but merge-free branch" {
    # Branch cut from main, then main advances: branch has no merge commits
    git checkout --quiet -b hotfix-torch/stale-fix
    make_commit "hotfix work"
    git push --quiet origin hotfix-torch/stale-fix
    git checkout --quiet main
    make_commit "main moves on"
    git push --quiet origin main

    run bash "$SCRIPT_PATH" "hotfix-torch/stale-fix" "main"

    [ "$status" -eq 0 ]
    [[ "$output" == *"No-merge-commits check passed"* ]]
}

@test "enforce_no_merge_commits.sh: fails when the base was merged into the branch" {
    # Simulate GitHub's default "Update branch": merge main back into the branch
    git checkout --quiet -b feature-core/back-merged
    make_commit "feature work"
    git checkout --quiet main
    make_commit "main moves on"
    git push --quiet origin main
    git checkout --quiet feature-core/back-merged
    git merge --quiet --no-edit --no-ff main
    git push --quiet origin feature-core/back-merged

    run bash "$SCRIPT_PATH" "feature-core/back-merged" "main"

    [ "$status" -eq 1 ]
    [[ "$output" == *"contains merge commits"* ]]
    [[ "$output" == *"rebasing instead"* ]]
}

@test "enforce_no_merge_commits.sh: skips dev integration branches" {
    # dev branches legitimately contain PR merge commits
    git checkout --quiet -b dev-core
    make_commit "feature work"
    git checkout --quiet main
    git checkout --quiet dev-core
    git merge --quiet --no-edit --no-ff main || true
    git push --quiet origin dev-core

    run bash "$SCRIPT_PATH" "dev-core" "main"

    [ "$status" -eq 0 ]
    [[ "$output" == *"Skipping no-merge-commits check"* ]]
}
