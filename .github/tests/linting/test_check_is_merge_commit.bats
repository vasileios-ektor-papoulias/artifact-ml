#!/usr/bin/env bats
# .github/tests/test_check_is_merge_commit.bats

# Helper: Extract the final (last non-empty) line from a string.
get_final_line() {
  echo "$1" | sed '/^[[:space:]]*$/d' | tail -n 1 | tr -d '\r\n'
}

setup() {
    TEST_DIR="$(dirname "$BATS_TEST_FILENAME")"
    REPO_ROOT="$(cd "$TEST_DIR/../../.." && pwd)"
    echo "Repository root is: $REPO_ROOT" >&2
    FAKE_BIN_DIR="$BATS_TMPDIR/fakebin"
    mkdir -p "$FAKE_BIN_DIR"
    mkdir -p "$FAKE_BIN_DIR/.github/scripts/linting"

    cat << 'EOF' > "$FAKE_BIN_DIR/git"
#!/bin/bash
# Fake git command for testing check_is_merge_commit.sh.
if [ "$1" = "log" ] && [ "$2" = "-1" ]; then
  if [ "$3" = "--pretty=format:%s" ]; then
    echo "$FAKE_COMMIT_SUBJECT"
    exit 0
  elif [ "$3" = "--pretty=format:%P" ]; then
    echo "$FAKE_COMMIT_PARENTS"
    exit 0
  fi
fi
echo "Fake git: unhandled args: $@" >&2
exit 1
EOF
  chmod +x "$FAKE_BIN_DIR/git"

  SCRIPT="$REPO_ROOT/.github/scripts/linting/check_is_merge_commit.sh"
  echo "Using script path to copy: $SCRIPT" >&2
  if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: Cannot find the script at: $SCRIPT" >&2
    exit 1
  fi
  cp "$SCRIPT" "$FAKE_BIN_DIR/.github/scripts/linting/"
  chmod +x "$FAKE_BIN_DIR/.github/scripts/linting/$(basename "$SCRIPT")"
  export PATH="$FAKE_BIN_DIR:$PATH"
  cd $FAKE_BIN_DIR
}

teardown() {
  echo "Teardown: Removing fake bin directory: $FAKE_BIN_DIR" >&2
  rm -rf "$FAKE_BIN_DIR" || echo "Warning: Failed to remove $FAKE_BIN_DIR" >&2
}

@test "exits with error when the last commit is not a merge commit" {
  export FAKE_COMMIT_SUBJECT="Regular commit message"
  export FAKE_COMMIT_PARENTS="abc123"
  run "$FAKE_BIN_DIR/.github/scripts/linting/$(basename "$SCRIPT")"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"This is not a merge commit. Skipping linting checks."* ]]
}

@test "returns 2 when the last commit is a merge commit with 2 parents" {
  export FAKE_COMMIT_SUBJECT="Merge commit: Branch merged"
  export FAKE_COMMIT_PARENTS="hash1 hash2"
  run "$FAKE_BIN_DIR/.github/scripts/linting/$(basename "$SCRIPT")"
  [ "$status" -eq 0 ]
  final_line="$(get_final_line "$output")"
  echo "DEBUG: final_line=[$final_line]" >&2
  [ "$final_line" = "2" ]
}

@test "returns 3 when the last commit is a merge commit with 3 parents" {
  export FAKE_COMMIT_SUBJECT="Merge commit: Triple merge"
  export FAKE_COMMIT_PARENTS="hash1 hash2 hash3"
  run "$FAKE_BIN_DIR/.github/scripts/linting/$(basename "$SCRIPT")"
  [ "$status" -eq 0 ]
  final_line="$(get_final_line "$output")"
  echo "DEBUG: final_line=[$final_line]" >&2
  [ "$final_line" = "3" ]
}
