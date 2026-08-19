#!/usr/bin/env bats
#
# Unit tests for sync_dev_branches.sh
#
# These tests exercise the real git fast-forward logic against throwaway
# local repositories: a bare repo standing in for the GitHub remote, and a
# clone in which the script runs (mirroring the Actions checkout).

setup() {
    # Get the directory containing this test file
    TEST_DIR="$(cd "$(dirname "$BATS_TEST_FILENAME")" && pwd)"
    # Navigate to repo root (assuming tests are in .github/tests/github/)
    REPO_ROOT="$(cd "$TEST_DIR/../../.." && pwd)"
    SCRIPT_PATH="$REPO_ROOT/.github/scripts/github/sync_dev_branches.sh"

    WORK_DIR="$(mktemp -d)"
    REMOTE_DIR="$WORK_DIR/remote.git"
    CLONE_DIR="$WORK_DIR/clone"

    # Bare repo standing in for the GitHub remote
    git init --bare --quiet "$REMOTE_DIR"

    # Working clone in which the script runs
    git clone --quiet "$REMOTE_DIR" "$CLONE_DIR" 2>/dev/null
    cd "$CLONE_DIR"
    git config user.email "test@example.com"
    git config user.name "Test User"
    git config commit.gpgsign false

    # Seed main with an initial commit and create the three dev branches at it
    git checkout --quiet -b main
    make_commit "initial commit"
    git push --quiet origin main
    git push --quiet origin main:dev-core main:dev-experiment main:dev-torch

    # Advance main by one commit so the dev branches are behind
    make_commit "second commit on main"
    git push --quiet origin main
    git fetch --quiet origin
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

# Helper: SHA of a branch on the bare remote
remote_sha() {
    git ls-remote "$REMOTE_DIR" "refs/heads/$1" | cut -f1
}

@test "sync_dev_branches.sh: fails on invalid target" {
    run bash "$SCRIPT_PATH" "bogus"

    [ "$status" -eq 1 ]
    [[ "$output" == *"Invalid target 'bogus'"* ]]
}

@test "sync_dev_branches.sh: syncs all dev branches by default" {
    MAIN_SHA="$(remote_sha main)"

    run bash "$SCRIPT_PATH"

    [ "$status" -eq 0 ]
    [ "$(remote_sha dev-core)" = "$MAIN_SHA" ]
    [ "$(remote_sha dev-experiment)" = "$MAIN_SHA" ]
    [ "$(remote_sha dev-torch)" = "$MAIN_SHA" ]
    [[ "$output" == *"Dev branch sync complete."* ]]
}

@test "sync_dev_branches.sh: syncs only the requested component" {
    MAIN_SHA="$(remote_sha main)"
    OLD_CORE_SHA="$(remote_sha dev-core)"

    run bash "$SCRIPT_PATH" "torch"

    [ "$status" -eq 0 ]
    [ "$(remote_sha dev-torch)" = "$MAIN_SHA" ]
    [ "$(remote_sha dev-core)" = "$OLD_CORE_SHA" ]
    [ "$(remote_sha dev-experiment)" = "$OLD_CORE_SHA" ]
}

@test "sync_dev_branches.sh: is a no-op when branches are already at main's tip" {
    bash "$SCRIPT_PATH" "all"

    run bash "$SCRIPT_PATH" "all"

    [ "$status" -eq 0 ]
    [[ "$output" == *"dev-core is already at main's tip."* ]]
    [[ "$output" == *"dev-experiment is already at main's tip."* ]]
    [[ "$output" == *"dev-torch is already at main's tip."* ]]
}

@test "sync_dev_branches.sh: refuses to sync a diverged dev branch" {
    # Put a unique commit on dev-core that is not on main
    git checkout --quiet -b dev-core origin/dev-core
    make_commit "stray commit on dev-core"
    git push --quiet origin dev-core
    DIVERGED_SHA="$(remote_sha dev-core)"
    git checkout --quiet main

    run bash "$SCRIPT_PATH" "core"

    [ "$status" -eq 1 ]
    [[ "$output" == *"cannot be fast-forwarded"* ]]
    [[ "$output" == *"Diverged branches left untouched: dev-core."* ]]
    # The diverged branch must be left untouched
    [ "$(remote_sha dev-core)" = "$DIVERGED_SHA" ]
}

@test "sync_dev_branches.sh: divergence does not block syncing healthy branches" {
    # dev-core diverges; the other two must still be fast-forwarded to main
    git checkout --quiet -b dev-core origin/dev-core
    make_commit "stray commit on dev-core"
    git push --quiet origin dev-core
    DIVERGED_SHA="$(remote_sha dev-core)"
    git checkout --quiet main
    MAIN_SHA="$(remote_sha main)"

    run bash "$SCRIPT_PATH" "all"

    [ "$status" -eq 1 ]
    [ "$(remote_sha dev-core)" = "$DIVERGED_SHA" ]
    [ "$(remote_sha dev-experiment)" = "$MAIN_SHA" ]
    [ "$(remote_sha dev-torch)" = "$MAIN_SHA" ]
    [[ "$output" == *"Diverged branches left untouched: dev-core."* ]]
}
