#!/usr/bin/env bats
# .github/tests/linting/test_lint_commit_message.bats

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
  mkdir -p "$FAKE_BIN_DIR/.github/scripts/linting"

  SCRIPT="$REPO_ROOT/.github/scripts/linting/lint_commit_message.sh"
  echo "Using script path to copy: $SCRIPT" >&2
  if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: Cannot find the script at: $SCRIPT" >&2
    exit 1
  fi
  cp "$SCRIPT" "$FAKE_BIN_DIR/.github/scripts/linting/"
  chmod +x "$FAKE_BIN_DIR/.github/scripts/linting/lint_commit_message.sh"

  cat << "EOF" > "$FAKE_BIN_DIR/.github/scripts/linting/extract_branch_info.sh"
#!/bin/bash
# Fake extract_branch_info.sh for testing lint_commit_message.sh.
BRANCH_NAME="$1"
if [ -z "$BRANCH_NAME" ]; then
  echo "::error::No branch name provided!" >&2
  exit 1
fi

# Debug output to help diagnose issues
echo "DEBUG: Processing branch name: $BRANCH_NAME" >&2

if [[ "$BRANCH_NAME" == "dev-mycomponent" ]]; then
  echo '{"branch_type":"dev","component_name":"mycomponent"}'
  exit 0
elif [[ "$BRANCH_NAME" == "hotfix-mycomponent/fix-critical-bug" ]]; then
  echo '{"branch_type":"hotfix","component_name":"mycomponent"}'
  exit 0
elif [[ "$BRANCH_NAME" == "setup-mycomponent/initial-config" ]]; then
  echo '{"branch_type":"setup","component_name":"mycomponent"}'
  exit 0
elif [[ "$BRANCH_NAME" == "feature-newcomponent" ]]; then
  echo "::error::Branch name does not follow the convention!" >&2
  exit 1
else
  # For testing, we'll handle any other branch name as an error
  echo "::error::Branch name does not follow the convention!" >&2
  echo "DEBUG: Branch name '$BRANCH_NAME' did not match any patterns" >&2
  exit 1
fi
EOF
  chmod +x "$FAKE_BIN_DIR/.github/scripts/linting/extract_branch_info.sh"

  cat << "EOF" > "$FAKE_BIN_DIR/git"
#!/bin/bash
# Fake git command for testing lint_commit_message.sh.
# It handles: git log -1 --pretty=format:%s
if [[ "$@" == *"--pretty=format:%s"* ]]; then
  echo "$FAKE_COMMIT_MSG"
  exit 0
fi
echo "Fake git: unhandled args: $@" >&2
exit 1
EOF
  chmod +x "$FAKE_BIN_DIR/git"
  export PATH="$FAKE_BIN_DIR:$PATH"

  cd "$FAKE_BIN_DIR"
}

teardown() {
  echo "Teardown: Removing fake bin directory: $BATS_TMPDIR/fakebin" >&2
  rm -rf "$BATS_TMPDIR/fakebin" || echo "Warning: Failed to remove fake bin" >&2
}

@test "exits with error when commit message does not follow convention" {
  export FAKE_COMMIT_MSG="Merge pull request #123 from username/feature-newcomponent"
  run ".github/scripts/linting/lint_commit_message.sh"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Extracted branch name: feature-newcomponent"* ]]
}

@test "returns component name for a valid dev branch merge commit message" {
  export FAKE_COMMIT_MSG="Merge pull request #123 from username/dev-mycomponent"
  run ".github/scripts/linting/lint_commit_message.sh"
  [ "$status" -eq 0 ]
  final_line="$(get_final_line "$output")"
  echo "DEBUG: final_line=[$final_line]" >&2
  [ "$final_line" = "mycomponent" ]
}

@test "returns component name for a valid hotfix branch merge commit message" {
  export FAKE_COMMIT_MSG="Merge pull request #123 from username/hotfix-mycomponent/fix-critical-bug"
  run ".github/scripts/linting/lint_commit_message.sh"
  echo "DEBUG: status=[$status]" >&2
  echo "DEBUG: output=[$output]" >&2
  echo "DEBUG: stderr=[$stderr]" >&2
  [ "$status" -eq 0 ]
  final_line="$(get_final_line "$output")"
  echo "DEBUG: final_line=[$final_line]" >&2
  [ "$final_line" = "mycomponent" ]
}

@test "returns component name for a valid setup branch merge commit message" {
  export FAKE_COMMIT_MSG="Merge pull request #123 from username/setup-mycomponent/initial-config"
  run ".github/scripts/linting/lint_commit_message.sh"
  echo "DEBUG: status=[$status]" >&2
  echo "DEBUG: output=[$output]" >&2
  echo "DEBUG: stderr=[$stderr]" >&2
  [ "$status" -eq 0 ]
  final_line="$(get_final_line "$output")"
  echo "DEBUG: final_line=[$final_line]" >&2
  [ "$final_line" = "mycomponent" ]
}
