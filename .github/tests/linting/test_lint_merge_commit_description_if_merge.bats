#!/usr/bin/env bats
# .github/tests/linting/test_lint_merge_commit_description.bats

# Helper function: Extract the final (last non-empty) line from a string.
get_final_line() {
  echo "$1" | sed '/^[[:space:]]*$/d' | tail -n 1 | tr -d '\r\n'
}

setup() {
  TEST_DIR="$(dirname "$BATS_TEST_FILENAME")"
  REPO_ROOT="$(cd "$TEST_DIR/../../.." && pwd)"
  echo "Repository root is: $REPO_ROOT" >&2
  FAKE_BIN_DIR="$BATS_TMPDIR/fakebin"
  mkdir -p "$FAKE_BIN_DIR"

   cp -r "$REPO_ROOT/.github" "$FAKE_BIN_DIR/"

  SCRIPT="$REPO_ROOT/.github/scripts/linting/lint_merge_commit_description.sh"
  echo "Using script path to copy: $SCRIPT" >&2
  if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: Cannot find the script at: $SCRIPT" >&2
    exit 1
  fi
  cp "$SCRIPT" "$FAKE_BIN_DIR/.github/scripts/linting/"
  chmod +x "$FAKE_BIN_DIR/.github/scripts/linting/$(basename "$SCRIPT")"


  cat << "EOF" > "$FAKE_BIN_DIR/.github/scripts/linting/check_is_merge_commit.sh"
#!/bin/bash
if [ "$FAKE_IS_MERGE" = "true" ]; then
  exit 0
else
  exit 1
fi
EOF
  chmod +x "$FAKE_BIN_DIR/.github/scripts/linting/check_is_merge_commit.sh"

  cat << "EOF" > "$FAKE_BIN_DIR/.github/scripts/linting/lint_commit_description.sh"
#!/bin/bash
if [ "$FAKE_VALID_DESCRIPTION" = "true" ]; then
  echo "$FAKE_BUMP_TYPE"
  exit 0
else
  echo "::error::Commit description does not follow the convention!" >&2
  exit 1
fi
EOF
  chmod +x "$FAKE_BIN_DIR/.github/scripts/linting/lint_commit_description.sh"

  cd "$FAKE_BIN_DIR"
}

teardown() {
  echo "Teardown: Removing fake bin directory: $BATS_TMPDIR/fakebin" >&2
  rm -rf "$BATS_TMPDIR/fakebin" || echo "Warning: Failed to remove fake bin" >&2
}

@test "skips linting when not a merge commit" {
  export FAKE_IS_MERGE="false"
  export FAKE_COMMIT_DESC="anything"
  run ".github/scripts/linting/lint_merge_commit_description.sh"
  [ "$status" -eq 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Not a merge commit, skipping linting"* ]]
}

@test "checks merge commit with valid description" {
  export FAKE_IS_MERGE="true"
  export FAKE_COMMIT_DESC="patch: fix bug"
  export FAKE_VALID_DESCRIPTION="true"
  export FAKE_BUMP_TYPE="patch"
  run ".github/scripts/linting/lint_merge_commit_description.sh"
  [ "$status" -eq 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"This is a merge commit, checking description..."* ]]
  [[ "$combined" == *"Bump type: patch"* ]]
}

@test "exits with error when merge commit description is invalid" {
  export FAKE_IS_MERGE="true"
  export FAKE_COMMIT_DESC="invalid commit description"
  export FAKE_VALID_DESCRIPTION="false"
  run ".github/scripts/linting/lint_merge_commit_description.sh"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Commit description does not follow the convention!"* ]]
}
