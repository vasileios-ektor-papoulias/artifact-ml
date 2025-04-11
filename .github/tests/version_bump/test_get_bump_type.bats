#!/usr/bin/env bats
# .github/tests/version_bump/test_get_bump_type.bats

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

  SCRIPT="$REPO_ROOT/.github/scripts/version_bump/get_bump_type.sh"
  echo "Using script path to copy: $SCRIPT" >&2

  if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: Cannot find the script at: $SCRIPT" >&2
    exit 1
  fi
  cp "$SCRIPT" "$FAKE_BIN_DIR/.github/scripts/version_bump/"
  chmod +x "$FAKE_BIN_DIR/.github/scripts/version_bump/get_bump_type.sh"

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

@test "successfully determines bump type when description is valid" {
  export FAKE_VALID_DESCRIPTION="true"
  export FAKE_BUMP_TYPE="patch"
  run ".github/scripts/version_bump/get_bump_type.sh"
  [ "$status" -eq 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Determined bump_type: patch"* ]]
  final_line="$(get_final_line "$output")"
  [[ "$final_line" == "patch" ]]
}

@test "exits with error when description is invalid" {
  export FAKE_VALID_DESCRIPTION="false"
  run ".github/scripts/version_bump/get_bump_type.sh"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Commit description does not follow the convention!"* ]]
}
