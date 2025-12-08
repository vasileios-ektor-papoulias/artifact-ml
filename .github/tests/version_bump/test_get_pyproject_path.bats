#!/usr/bin/env bats
# .github/tests/version_bump/test_get_pyproject_path.bats

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
  mkdir -p "$FAKE_BIN_DIR/artifact-core"
  mkdir -p "$FAKE_BIN_DIR/artifact-experiment"

  SCRIPT="$REPO_ROOT/.github/scripts/version_bump/get_pyproject_path.sh"
  echo "Using script path to copy: $SCRIPT" >&2

  if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: Cannot find the script at: $SCRIPT" >&2
    exit 1
  fi

  cp "$SCRIPT" "$FAKE_BIN_DIR/.github/scripts/version_bump/"
  chmod +x "$FAKE_BIN_DIR/.github/scripts/version_bump/get_pyproject_path.sh"
  
  echo '[tool.poetry]
name = "root-project"
version = "1.0.0"
description = "Root project"' > "$FAKE_BIN_DIR/pyproject.toml"

  echo '[tool.poetry]
name = "artifact-core"
version = "0.1.0"
description = "Artifact Core project"' > "$FAKE_BIN_DIR/artifact-core/pyproject.toml"

  echo '[tool.poetry]
name = "artifact-experiment"
version = "0.1.0"
description = "Artifact Experiment project"' > "$FAKE_BIN_DIR/artifact-experiment/pyproject.toml"

  cd "$FAKE_BIN_DIR"
}

teardown() {
  echo "Teardown: Removing fake bin directory: $BATS_TMPDIR/fakebin" >&2
  rm -rf "$BATS_TMPDIR/fakebin" || echo "Warning: Failed to remove fake bin" >&2
}

@test "returns component pyproject.toml path when component exists" {
  run ".github/scripts/version_bump/get_pyproject_path.sh" "core"
  [ "$status" -eq 0 ]
  [ "$output" = "artifact-core/pyproject.toml" ]
}

@test "returns experiment pyproject.toml path when experiment component" {
  run ".github/scripts/version_bump/get_pyproject_path.sh" "experiment"
  [ "$status" -eq 0 ]
  [ "$output" = "artifact-experiment/pyproject.toml" ]
}

@test "returns root pyproject.toml path when component is root" {
  run ".github/scripts/version_bump/get_pyproject_path.sh" "root"
  [ "$status" -eq 0 ]
  [ "$output" = "pyproject.toml" ]
}

@test "returns root pyproject.toml path when no component is provided" {
  run ".github/scripts/version_bump/get_pyproject_path.sh"
  [ "$status" -eq 0 ]
  [ "$output" = "pyproject.toml" ]
}

@test "exits with error when component pyproject.toml doesn't exist" {
  run ".github/scripts/version_bump/get_pyproject_path.sh" "torch"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Error"* ]] && [[ "$combined" == *"Component pyproject.toml not found"* ]]
}

@test "exits with error when no pyproject.toml exists" {
  rm "$FAKE_BIN_DIR/pyproject.toml"
  rm "$FAKE_BIN_DIR/artifact-core/pyproject.toml"
  rm "$FAKE_BIN_DIR/artifact-experiment/pyproject.toml"
  run ".github/scripts/version_bump/get_pyproject_path.sh"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Error"* ]] && [[ "$combined" == *"No valid pyproject.toml"* ]]
}
