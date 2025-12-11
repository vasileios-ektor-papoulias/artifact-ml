#!/usr/bin/env bats

# Tests for check_component_changed.sh

setup() {
    # Get the directory of the test file
    TEST_DIR="$(cd "$(dirname "$BATS_TEST_FILENAME")" && pwd)"
    SCRIPT_PATH="$TEST_DIR/../../scripts/ci/check_component_changed.sh"
    
    # Create a temporary git repo for testing
    TEST_REPO=$(mktemp -d)
    cd "$TEST_REPO"
    git init --quiet
    git config user.email "test@test.com"
    git config user.name "Test User"
    
    # Create initial structure
    mkdir -p artifact-core artifact-experiment artifact-torch
    echo "initial" > artifact-core/file.py
    echo "initial" > artifact-experiment/file.py
    echo "initial" > artifact-torch/file.py
    echo "root file" > README.md
    
    git add .
    git commit -m "Initial commit" --quiet
}

teardown() {
    cd /
    rm -rf "$TEST_REPO"
}

@test "returns error when no component directory provided" {
    run bash "$SCRIPT_PATH"
    [ "$status" -eq 1 ]
    [[ "$output" == *"Component directory is required"* ]]
}

@test "returns true when component has changes" {
    # Make a change to artifact-core
    echo "changed" > artifact-core/new_file.py
    git add .
    git commit -m "Change core" --quiet
    
    run bash "$SCRIPT_PATH" "artifact-core"
    [ "$status" -eq 0 ]
    [ "$output" = "true" ]
}

@test "returns false when component has no changes" {
    # Make a change to artifact-core only
    echo "changed" > artifact-core/new_file.py
    git add .
    git commit -m "Change core" --quiet
    
    # Check artifact-experiment (no changes)
    run bash "$SCRIPT_PATH" "artifact-experiment"
    [ "$status" -eq 0 ]
    [ "$output" = "false" ]
}

@test "returns false when only root files changed" {
    # Make a change to root only
    echo "changed" > README.md
    git add .
    git commit -m "Change root" --quiet
    
    run bash "$SCRIPT_PATH" "artifact-core"
    [ "$status" -eq 0 ]
    [ "$output" = "false" ]
}

@test "returns true with nested file changes" {
    # Make a change to a nested file
    mkdir -p artifact-core/subdir
    echo "nested" > artifact-core/subdir/nested.py
    git add .
    git commit -m "Add nested file" --quiet
    
    run bash "$SCRIPT_PATH" "artifact-core"
    [ "$status" -eq 0 ]
    [ "$output" = "true" ]
}

@test "works with custom base ref" {
    # Make two commits
    echo "change1" > artifact-core/file1.py
    git add .
    git commit -m "First change" --quiet
    
    echo "change2" > artifact-experiment/file2.py
    git add .
    git commit -m "Second change" --quiet
    
    # Check against HEAD~2 (should see both changes)
    run bash "$SCRIPT_PATH" "artifact-core" "HEAD~2"
    [ "$status" -eq 0 ]
    [ "$output" = "true" ]
    
    # Check against HEAD~1 (should only see experiment change)
    run bash "$SCRIPT_PATH" "artifact-core" "HEAD~1"
    [ "$status" -eq 0 ]
    [ "$output" = "false" ]
}

@test "handles multiple components changed" {
    # Change multiple components
    echo "changed" > artifact-core/new.py
    echo "changed" > artifact-torch/new.py
    git add .
    git commit -m "Change multiple" --quiet
    
    run bash "$SCRIPT_PATH" "artifact-core"
    [ "$status" -eq 0 ]
    [ "$output" = "true" ]
    
    run bash "$SCRIPT_PATH" "artifact-torch"
    [ "$status" -eq 0 ]
    [ "$output" = "true" ]
    
    run bash "$SCRIPT_PATH" "artifact-experiment"
    [ "$status" -eq 0 ]
    [ "$output" = "false" ]
}

