#!/usr/bin/env bats
# .github/tests/version_bump/test_push_version_update.bats

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

  SCRIPT="$REPO_ROOT/.github/scripts/version_bump/push_version_update.sh"
  echo "Using script path to copy: $SCRIPT" >&2

  if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: Cannot find the script at: $SCRIPT" >&2
    exit 1
  fi

  cp "$SCRIPT" "$FAKE_BIN_DIR/.github/scripts/version_bump/"
  chmod +x "$FAKE_BIN_DIR/.github/scripts/version_bump/push_version_update.sh"

  cat << "EOF" > "$FAKE_BIN_DIR/git"
#!/bin/bash
# Fake git command for testing push_version_update.sh
echo "git $@" >> "$FAKE_BIN_DIR/git_commands.log"
exit 0
EOF
  chmod +x "$FAKE_BIN_DIR/git"

  export PATH="$FAKE_BIN_DIR:$PATH"
  
  echo "Test file content" > "$FAKE_BIN_DIR/test-file.txt"

  cd "$FAKE_BIN_DIR"
}

teardown() {
  echo "Teardown: Removing fake bin directory: $BATS_TMPDIR/fakebin" >&2
  rm -rf "$BATS_TMPDIR/fakebin" || echo "Warning: Failed to remove fake bin" >&2
}

@test "successfully commits and tags version update" {
  run ".github/scripts/version_bump/push_version_update.sh" "v1.2.3" "test-file.txt"
  [ "$status" -eq 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Created git tag: v1.2.3"* ]]
  [[ "$combined" == *"Pushing changes and tags to remote repository"* ]]
  [[ "$combined" == *"Successfully bumped version to v1.2.3"* ]]
}

@test "exits with error when no tag name is provided" {
  run ".github/scripts/version_bump/push_version_update.sh"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Usage"* ]]
}

@test "exits with error when no file path is provided" {
  cd "$FAKE_BIN_DIR"
  run ".github/scripts/version_bump/push_version_update.sh" "v1.2.3"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Usage"* ]]
}
