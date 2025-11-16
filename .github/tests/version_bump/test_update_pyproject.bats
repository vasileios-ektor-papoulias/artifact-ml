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
  mkdir -p "$FAKE_BIN_DIR/.github/scripts/version_bump"

  SCRIPT="$REPO_ROOT/.github/scripts/version_bump/update_pyproject.sh"
  echo "Using script path to copy: $SCRIPT" >&2
  cp "$SCRIPT" "$FAKE_BIN_DIR/.github/scripts/version_bump/"
  chmod +x "$FAKE_BIN_DIR/.github/scripts/version_bump/update_pyproject.sh"

  # Create a fake poetry command that simulates version bumping
  cat << "EOF" > "$FAKE_BIN_DIR/poetry"
#!/bin/bash
# Fake poetry command for testing
if [[ "$1" == "version" && "$3" == "--short" ]]; then
  BUMP_TYPE="$2"
  
  # Look for any pyproject.toml or test-pyproject.toml in current directory
  PYPROJECT_FILE=""
  if [[ -f "pyproject.toml" ]]; then
    PYPROJECT_FILE="pyproject.toml"
  elif [[ -f "test-pyproject.toml" ]]; then
    PYPROJECT_FILE="test-pyproject.toml"
  else
    echo "Error: pyproject.toml not found" >&2
    exit 1
  fi
  
  # Read current version from the file
  CURRENT_VERSION=$(grep '^version' "$PYPROJECT_FILE" | head -1 | sed 's/.*"\(.*\)".*/\1/')
  
  # Parse version components
  IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT_VERSION"
  
  # Bump according to type
  case "$BUMP_TYPE" in
    patch)
      PATCH=$((PATCH + 1))
      ;;
    minor)
      MINOR=$((MINOR + 1))
      PATCH=0
      ;;
    major)
      MAJOR=$((MAJOR + 1))
      MINOR=0
      PATCH=0
      ;;
    *)
      echo "Error: Unknown bump type: $BUMP_TYPE" >&2
      exit 1
      ;;
  esac
  
  NEW_VERSION="$MAJOR.$MINOR.$PATCH"
  
  # Update the pyproject.toml file
  sed -i "s/version = \"$CURRENT_VERSION\"/version = \"$NEW_VERSION\"/" "$PYPROJECT_FILE"
  
  # Output just the new version (simulating --short flag)
  echo "$NEW_VERSION"
  exit 0
fi

echo "Error: Invalid poetry command" >&2
exit 1
EOF
  chmod +x "$FAKE_BIN_DIR/poetry"
  
  export PATH="$FAKE_BIN_DIR:$PATH"

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
