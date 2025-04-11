#!/usr/bin/env bats
# .github/tests/version_bump/test_update_pyproject.bats

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
  mkdir -p "$FAKE_BIN_DIR/.github/scripts/version_bump"

  SCRIPT="$REPO_ROOT/.github/scripts/version_bump/update_pyproject.sh"
  echo "Using script path to copy: $SCRIPT" >&2
  cp "$SCRIPT" "$FAKE_BIN_DIR/.github/scripts/version_bump/"
  chmod +x "$FAKE_BIN_DIR/.github/scripts/version_bump/update_pyproject.sh"

  cat << "EOF" > "$FAKE_BIN_DIR/.github/scripts/version_bump/identify_new_version.sh"
#!/bin/bash
# Fake identify_new_version.sh for testing update_pyproject.sh
CURRENT_VERSION="$1"
BUMP_TYPE="$2"

if [[ -z "$CURRENT_VERSION" || -z "$BUMP_TYPE" ]]; then
    echo "Usage: $0 <current_version> {patch|minor|major}" >&2
    exit 1
fi

case "$BUMP_TYPE" in
  patch)
    echo "1.2.4"
    ;;
  minor)
    echo "1.3.0"
    ;;
  major)
    echo "2.0.0"
    ;;
  *)
    echo "Error: Unknown bump type: $BUMP_TYPE. Must be one of: patch, minor, major" >&2
    exit 1
    ;;
esac
EOF
  chmod +x "$FAKE_BIN_DIR/.github/scripts/version_bump/identify_new_version.sh"

  cd "$FAKE_BIN_DIR"
}

teardown() {
  echo "Teardown: Removing fake bin directory: $BATS_TMPDIR/fakebin" >&2
  rm -rf "$BATS_TMPDIR/fakebin" || echo "Warning: Failed to remove fake bin" >&2
}

@test "bumps patch version correctly" {
  echo '[tool.poetry]
name = "test-project"
version = "1.2.3"
description = "Test project"' > "$FAKE_BIN_DIR/test-pyproject.toml"
  run ".github/scripts/version_bump/update_pyproject.sh" "test-pyproject.toml" "patch"
  [ "$status" -eq 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$output" == *"1.2.4"* ]]
  grep -q 'version = "1.2.4"' "$FAKE_BIN_DIR/test-pyproject.toml"
  [ "$?" -eq 0 ]
}

@test "bumps minor version correctly" {
  echo '[tool.poetry]
name = "test-project"
version = "1.2.3"
description = "Test project"' > "$FAKE_BIN_DIR/test-pyproject.toml"
  run ".github/scripts/version_bump/update_pyproject.sh" "test-pyproject.toml" "minor"
  [ "$status" -eq 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$output" == *"1.3.0"* ]]
  grep -q 'version = "1.3.0"' "$FAKE_BIN_DIR/test-pyproject.toml"
  [ "$?" -eq 0 ]
}

@test "bumps major version correctly" {
  echo '[tool.poetry]
name = "test-project"
version = "1.2.3"
description = "Test project"' > "$FAKE_BIN_DIR/test-pyproject.toml"
  run ".github/scripts/version_bump/update_pyproject.sh" "test-pyproject.toml" "major"
  [ "$status" -eq 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$output" == *"2.0.0"* ]]
  grep -q 'version = "2.0.0"' "$FAKE_BIN_DIR/test-pyproject.toml"
  [ "$?" -eq 0 ]
}

@test "exits with error when pyproject.toml doesn't exist" {
  run ".github/scripts/version_bump/update_pyproject.sh" "nonexistent.toml" "patch"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Error"* ]] && [[ "$combined" == *"nonexistent.toml"* ]] && [[ "$combined" == *"does not exist"* ]]
}

@test "exits with error when bump type is invalid" {
  echo '[tool.poetry]
name = "test-project"
version = "1.2.3"
description = "Test project"' > "$FAKE_BIN_DIR/test-pyproject.toml"
  run ".github/scripts/version_bump/update_pyproject.sh" "test-pyproject.toml" "invalid"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Unknown bump type"* ]] && [[ "$combined" == *"invalid"* ]]
}

@test "exits with error when no arguments are provided" {
  run ".github/scripts/version_bump/update_pyproject.sh"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Usage"* ]]
}
