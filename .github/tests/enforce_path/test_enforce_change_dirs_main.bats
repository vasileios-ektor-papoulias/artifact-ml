#!/usr/bin/env bats
#
# Unit tests for enforce_change_dirs_main.sh
#
# These tests mock the underlying enforce_path scripts to test the routing logic.

setup() {
    # Get the directory containing this test file
    TEST_DIR="$(cd "$(dirname "$BATS_TEST_FILENAME")" && pwd)"
    # Navigate to repo root
    REPO_ROOT="$(cd "$TEST_DIR/../../.." && pwd)"
    SCRIPT_PATH="$REPO_ROOT/.github/scripts/enforce_path/enforce_change_dirs_main.sh"
    
    # Create a temporary directory for mocks
    MOCK_ROOT="$(mktemp -d)"
    mkdir -p "$MOCK_ROOT/.github/scripts/enforce_path"
    
    # Copy the actual script to mock root
    cp "$SCRIPT_PATH" "$MOCK_ROOT/.github/scripts/enforce_path/"
    
    # Create mock enforce_path scripts that succeed by default
    create_passing_mocks
}

teardown() {
    rm -rf "$MOCK_ROOT"
}

create_passing_mocks() {
    cat > "$MOCK_ROOT/.github/scripts/enforce_path/ensure_changed_files_in_dir.sh" << 'EOF'
#!/usr/bin/env bash
echo "Mock: ensure_changed_files_in_dir.sh called with: $@"
exit 0
EOF
    chmod +x "$MOCK_ROOT/.github/scripts/enforce_path/ensure_changed_files_in_dir.sh"
    
    cat > "$MOCK_ROOT/.github/scripts/enforce_path/ensure_changed_files_outside_dirs.sh" << 'EOF'
#!/usr/bin/env bash
echo "Mock: ensure_changed_files_outside_dirs.sh called with: $@"
exit 0
EOF
    chmod +x "$MOCK_ROOT/.github/scripts/enforce_path/ensure_changed_files_outside_dirs.sh"
}

create_failing_mocks() {
    cat > "$MOCK_ROOT/.github/scripts/enforce_path/ensure_changed_files_in_dir.sh" << 'EOF'
#!/usr/bin/env bash
echo "Mock: ensure_changed_files_in_dir.sh FAILED"
exit 1
EOF
    chmod +x "$MOCK_ROOT/.github/scripts/enforce_path/ensure_changed_files_in_dir.sh"
}

# Helper to run the script with mocked REPO_ROOT
run_script() {
    REPO_ROOT="$MOCK_ROOT" bash "$MOCK_ROOT/.github/scripts/enforce_path/enforce_change_dirs_main.sh" "$@"
}

@test "enforce_change_dirs_main.sh: fails when head_ref not provided" {
    run run_script
    
    [ "$status" -eq 1 ]
    [[ "$output" == *"head_ref (source branch) is required"* ]]
}

@test "enforce_change_dirs_main.sh: fails when base_ref not provided" {
    run run_script "dev-core"
    
    [ "$status" -eq 1 ]
    [[ "$output" == *"base_ref (target branch) is required"* ]]
}

@test "enforce_change_dirs_main.sh: routes dev-core to artifact-core" {
    run run_script "dev-core" "main"
    
    [ "$status" -eq 0 ]
    [[ "$output" == *"PR from dev-core"* ]]
    [[ "$output" == *"artifact-core"* ]]
    [[ "$output" == *"Directory enforcement check passed"* ]]
}

@test "enforce_change_dirs_main.sh: routes dev-experiment to artifact-experiment" {
    run run_script "dev-experiment" "main"
    
    [ "$status" -eq 0 ]
    [[ "$output" == *"PR from dev-experiment"* ]]
    [[ "$output" == *"artifact-experiment"* ]]
}

@test "enforce_change_dirs_main.sh: routes dev-torch to artifact-torch" {
    run run_script "dev-torch" "main"
    
    [ "$status" -eq 0 ]
    [[ "$output" == *"PR from dev-torch"* ]]
    [[ "$output" == *"artifact-torch"* ]]
}

@test "enforce_change_dirs_main.sh: routes hotfix-core/* to artifact-core" {
    run run_script "hotfix-core/fix-bug" "main"
    
    [ "$status" -eq 0 ]
    [[ "$output" == *"PR from hotfix-core"* ]]
    [[ "$output" == *"artifact-core"* ]]
}

@test "enforce_change_dirs_main.sh: routes hotfix-experiment/* to artifact-experiment" {
    run run_script "hotfix-experiment/urgent-fix" "main"
    
    [ "$status" -eq 0 ]
    [[ "$output" == *"PR from hotfix-experiment"* ]]
    [[ "$output" == *"artifact-experiment"* ]]
}

@test "enforce_change_dirs_main.sh: routes hotfix-torch/* to artifact-torch" {
    run run_script "hotfix-torch/patch" "main"
    
    [ "$status" -eq 0 ]
    [[ "$output" == *"PR from hotfix-torch"* ]]
    [[ "$output" == *"artifact-torch"* ]]
}

@test "enforce_change_dirs_main.sh: routes setup-root/* to outside dirs check" {
    run run_script "setup-root/add-ci" "main"
    
    [ "$status" -eq 0 ]
    [[ "$output" == *"PR from root branch"* ]]
    [[ "$output" == *"OUTSIDE"* ]]
}

@test "enforce_change_dirs_main.sh: routes hotfix-root/* to outside dirs check" {
    run run_script "hotfix-root/fix-ci" "main"
    
    [ "$status" -eq 0 ]
    [[ "$output" == *"PR from root branch"* ]]
    [[ "$output" == *"OUTSIDE"* ]]
}

@test "enforce_change_dirs_main.sh: fails for unrecognized branch pattern" {
    run run_script "feature-core/my-feature" "main"
    
    [ "$status" -eq 1 ]
    [[ "$output" == *"Unrecognized source branch pattern"* ]]
}

@test "enforce_change_dirs_main.sh: fails for random branch" {
    run run_script "random-branch" "main"
    
    [ "$status" -eq 1 ]
    [[ "$output" == *"Unrecognized source branch pattern"* ]]
}

@test "enforce_change_dirs_main.sh: propagates failure from underlying script" {
    create_failing_mocks
    
    run run_script "dev-core" "main"
    
    [ "$status" -eq 1 ]
    [[ "$output" == *"FAILED"* ]]
}
