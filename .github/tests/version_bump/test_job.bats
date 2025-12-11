#!/usr/bin/env bats
# .github/tests/version_bump/test_job.bats

# Helper function: Extract the final (last non-empty) line from a string.
get_final_line() {
  echo "$1" | sed '/^[[:space:]]*$/d' | tail -n 1 | tr -d '\r\n'
}

setup() {
  TEST_DIR="$(dirname "$BATS_TEST_FILENAME")"
  REPO_ROOT="$(cd "$TEST_DIR"/../../.. && pwd)"
  echo "Repository root is: $REPO_ROOT" >&2
  FAKE_BIN_DIR="$BATS_TMPDIR/fakebin"
  mkdir -p "$FAKE_BIN_DIR"
  cp -r "$REPO_ROOT/.github" "$FAKE_BIN_DIR/"

  SCRIPT="$REPO_ROOT/.github/scripts/version_bump/job.sh"
  echo "Using script path to copy: $SCRIPT" >&2
  cp "$SCRIPT" "$FAKE_BIN_DIR/.github/scripts/version_bump/"
  chmod +x "$FAKE_BIN_DIR/.github/scripts/version_bump/job.sh"

  cat << "EOF" > "$FAKE_BIN_DIR/.github/scripts/version_bump/get_pyproject_path.sh"
#!/bin/bash
echo "Fake get_pyproject_path.sh called with: $@" >&2
echo "subrepo/pyproject.toml"
exit 0
EOF
  chmod +x "$FAKE_BIN_DIR/.github/scripts/version_bump/get_pyproject_path.sh"

  cat << "EOF" > "$FAKE_BIN_DIR/.github/scripts/version_bump/bump_component_version.sh"
#!/bin/bash
echo "Fake bump_component_version.sh called with: $@" >&2
echo "1.2.3"
exit 0
EOF
  chmod +x "$FAKE_BIN_DIR/.github/scripts/version_bump/bump_component_version.sh"

   cd "$FAKE_BIN_DIR"
}

teardown() {
  echo "Teardown: Removing fake bin directory: $BATS_TMPDIR/fakebin" >&2
  rm -rf "$BATS_TMPDIR/fakebin" || echo "Warning: Failed to remove fake bin" >&2
}

@test "exits with error when component name is missing" {
  run ".github/scripts/version_bump/job.sh"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  [[ "$combined" == *"Component name is required as the first argument"* ]]
}

@test "exits with error when bump type is missing" {
  run ".github/scripts/version_bump/job.sh" "core"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  [[ "$combined" == *"Bump type is required as the second argument"* ]]
}

@test "exits with error when bump type is invalid" {
  run ".github/scripts/version_bump/job.sh" "core" "invalid"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  [[ "$combined" == *"Invalid bump type 'invalid'"* ]]
}

@test "skips version bump when bump type is no-bump" {
  run ".github/scripts/version_bump/job.sh" "core" "no-bump"
  [ "$status" -eq 0 ]
  [[ "$output" == *"component="* ]]
  [[ "$output" == *"version="* ]]
  [[ "$stderr" == *"Bump type is 'no-bump', skipping version bump"* ]] || [[ "$output" == *"Bump type is 'no-bump', skipping version bump"* ]]
}

@test "skips version bump when component is root" {
  run ".github/scripts/version_bump/job.sh" "root" "patch"
  [ "$status" -eq 0 ]
  [[ "$output" == *"component="* ]]
  [[ "$output" == *"version="* ]]
  # Check stderr contains skip message
  [[ "$stderr" == *"Component is 'root', skipping version bump"* ]] || [[ "$output" == *"Component is 'root', skipping version bump"* ]]
}

@test "successfully runs the version bump job" {
  run ".github/scripts/version_bump/job.sh" "subrepo" "patch"
  [ "$status" -eq 0 ]
  [[ "$output" == *"component=subrepo"* ]]
  [[ "$output" == *"version=1.2.3"* ]]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Fake bump_component_version.sh called with: patch subrepo subrepo/pyproject.toml"* ]]
  [[ "$combined" == *"Successfully completed version bump job"* ]]
}

@test "handles empty component name correctly when root pyproject.toml exists" {
  cat << "EOF" > "$FAKE_BIN_DIR/.github/scripts/version_bump/get_pyproject_path.sh"
#!/bin/bash
echo "Fake get_pyproject_path.sh called with empty component" >&2
echo "pyproject.toml"
exit 0
EOF
  chmod +x "$FAKE_BIN_DIR/.github/scripts/version_bump/get_pyproject_path.sh"
  run ".github/scripts/version_bump/job.sh" "" "patch"
  [ "$status" -eq 0 ]
  [[ "$output" == *"component="* ]]
  [[ "$output" == *"version=1.2.3"* ]]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Fake bump_component_version.sh called with: patch  pyproject.toml"* ]]
}

@test "exits with error when pyproject.toml doesn't exist for component" {
  cat << "EOF" > "$FAKE_BIN_DIR/.github/scripts/version_bump/get_pyproject_path.sh"
#!/bin/bash
echo "Error: Component pyproject.toml not found at nonexistent/pyproject.toml" >&2
exit 1
EOF
  chmod +x "$FAKE_BIN_DIR/.github/scripts/version_bump/get_pyproject_path.sh"
  run ".github/scripts/version_bump/job.sh" "nonexistent" "patch"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Error: Failed to find a valid pyproject.toml file"* ]]
  [[ "$combined" != *"Fake bump_component_version.sh called with:"* ]]
}
