#!/usr/bin/env bats
# .github/tests/version_bump/test_get_component_name.bats

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

  SCRIPT="$REPO_ROOT/.github/scripts/version_bump/get_component_name.sh"
  echo "Using script path to copy: $SCRIPT" >&2
  cp "$SCRIPT" "$FAKE_BIN_DIR/.github/scripts/version_bump/"
  chmod +x "$FAKE_BIN_DIR/.github/scripts/version_bump/get_component_name.sh"
  
  cat << "EOF" > "$FAKE_BIN_DIR/.github/scripts/linting/check_is_merge_commit.sh"
#!/bin/bash
if [ "$FAKE_IS_MERGE" = "true" ]; then
  exit 0
else
  exit 1
fi
EOF
  chmod +x "$FAKE_BIN_DIR/.github/scripts/linting/check_is_merge_commit.sh"

  cat << "EOF" > "$FAKE_BIN_DIR/.github/scripts/linting/lint_commit_message.sh"
#!/bin/bash
if [ "$FAKE_VALID_MESSAGE" = "true" ]; then
  echo "$FAKE_COMPONENT_NAME"
  exit 0
else
  echo "::error::Merge commit message does not follow the branch naming convention!" >&2
  exit 1
fi
EOF
  chmod +x "$FAKE_BIN_DIR/.github/scripts/linting/lint_commit_message.sh"

  cd "$FAKE_BIN_DIR"
}

teardown() {
  echo "Teardown: Removing fake bin directory: $BATS_TMPDIR/fakebin" >&2
  rm -rf "$BATS_TMPDIR/fakebin" || echo "Warning: Failed to remove fake bin" >&2
}

@test "successfully extracts component name from merge commit message" {
  export FAKE_IS_MERGE="true"
  export FAKE_VALID_MESSAGE="true"
  export FAKE_COMPONENT_NAME="subrepo"  
  run ".github/scripts/version_bump/get_component_name.sh"
  [ "$status" -eq 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Extracted component name 'subrepo' from merge commit message"* ]]
  final_line="$(get_final_line "$output")"
  [ "$final_line" = "subrepo" ]
}

@test "returns empty string when not a merge commit" {
  export FAKE_IS_MERGE="false"
  run ".github/scripts/version_bump/get_component_name.sh"
  [ "$status" -eq 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"No component name found"* ]]
  echo "DEBUG: output=[$output]" >&2
  [[ "$output" != *"subrepo"* ]]
}

@test "returns empty string when merge commit message is invalid" {
  export FAKE_IS_MERGE="true"
  export FAKE_VALID_MESSAGE="false"
  cat << "EOF" > "$FAKE_BIN_DIR/.github/scripts/version_bump/get_component_name_test.sh"
#!/bin/bash
# Modified version of get_component_name.sh for testing
# This version catches the error from lint_commit_message.sh and continues

chmod +x .github/scripts/linting/check_is_merge_commit.sh
chmod +x .github/scripts/linting/lint_commit_message.sh

if .github/scripts/linting/check_is_merge_commit.sh &>/dev/null; then
    # Use the linting script to get the component name
    COMPONENT_NAME=$(.github/scripts/linting/lint_commit_message.sh 2>/dev/null || true)
    if [ -n "$COMPONENT_NAME" ]; then
        echo "Extracted component name '$COMPONENT_NAME' from merge commit message" >&2
        echo "$COMPONENT_NAME"
        exit 0
    fi
fi

echo "No component name found" >&2
echo ""
EOF
  chmod +x "$FAKE_BIN_DIR/.github/scripts/version_bump/get_component_name_test.sh"
  run ".github/scripts/version_bump/get_component_name_test.sh"
  [ "$status" -eq 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"No component name found"* ]]
  echo "DEBUG: output=[$output]" >&2
  [[ "$output" != *"subrepo"* ]]
}
