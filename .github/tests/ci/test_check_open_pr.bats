#!/usr/bin/env bats
#
# Unit tests for check_open_pr.sh
#
# These tests mock the `gh` CLI to avoid actual API calls.

setup() {
    # Get the directory containing this test file
    TEST_DIR="$(cd "$(dirname "$BATS_TEST_FILENAME")" && pwd)"
    # Navigate to repo root (assuming tests are in .github/tests/ci/)
    REPO_ROOT="$(cd "$TEST_DIR/../../.." && pwd)"
    SCRIPT_PATH="$REPO_ROOT/.github/scripts/ci/check_open_pr.sh"
    
    # Create a temporary directory for mocks
    MOCK_DIR="$(mktemp -d)"
    
    # Export GH_TOKEN for tests (dummy value)
    export GH_TOKEN="test-token"
    
    # Prepend mock directory to PATH
    export PATH="$MOCK_DIR:$PATH"
}

teardown() {
    # Clean up mock directory
    rm -rf "$MOCK_DIR"
}

# Helper to create a mock gh command
create_gh_mock() {
    local output="$1"
    cat > "$MOCK_DIR/gh" << EOF
#!/usr/bin/env bash
echo '$output'
EOF
    chmod +x "$MOCK_DIR/gh"
}

# Helper to create a failing gh mock
create_gh_mock_failure() {
    cat > "$MOCK_DIR/gh" << 'EOF'
#!/usr/bin/env bash
exit 1
EOF
    chmod +x "$MOCK_DIR/gh"
}

@test "check_open_pr.sh: prints 'true' when PR exists" {
    # Mock gh to return a PR number
    create_gh_mock "123"
    
    run bash "$SCRIPT_PATH" "feature-core/my-feature"
    
    [ "$status" -eq 0 ]
    [ "$output" = "true" ]
}

@test "check_open_pr.sh: prints 'false' when no PR exists" {
    # Mock gh to return empty (no PR)
    create_gh_mock ""
    
    run bash "$SCRIPT_PATH" "feature-core/my-feature"
    
    [ "$status" -eq 0 ]
    [ "$output" = "false" ]
}

@test "check_open_pr.sh: prints 'false' when gh command fails" {
    # Mock gh to fail (e.g., network error)
    create_gh_mock_failure
    
    run bash "$SCRIPT_PATH" "feature-core/my-feature"
    
    [ "$status" -eq 0 ]
    [ "$output" = "false" ]
}

@test "check_open_pr.sh: fails when branch name not provided" {
    run bash "$SCRIPT_PATH"
    
    [ "$status" -eq 1 ]
    [[ "$output" == *"Branch name is required"* ]]
}

@test "check_open_pr.sh: fails when GH_TOKEN not set" {
    unset GH_TOKEN
    
    run bash "$SCRIPT_PATH" "feature-core/my-feature"
    
    [ "$status" -eq 1 ]
    [[ "$output" == *"GH_TOKEN environment variable is required"* ]]
}

@test "check_open_pr.sh: handles branch names with slashes" {
    # Mock gh to return a PR number
    create_gh_mock "456"
    
    run bash "$SCRIPT_PATH" "feature-core/deep/nested/branch"
    
    [ "$status" -eq 0 ]
    [ "$output" = "true" ]
}

