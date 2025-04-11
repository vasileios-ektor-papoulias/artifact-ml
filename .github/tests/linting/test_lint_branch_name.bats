#!/usr/bin/env bats
# .github/tests/linting/test_lint_branch_name.bats

# Helper function: Extract the final (last non-empty) line from a string.
get_final_line() {
  echo "$1" | sed '/^[[:space:]]*$/d' | tail -n 1 | tr -d '\r\n'
}

setup() {
  TEST_DIR="$(dirname "$BATS_TEST_FILENAME")"
  # Since tests are in .github/tests/linting, go up three levels to get to the repo root.
  REPO_ROOT="$(cd "$TEST_DIR/../../.." && pwd)"
  echo "Repository root is: $REPO_ROOT" >&2

  # Create a fake bin directory inside BATS_TMPDIR.
  FAKE_BIN_DIR="$BATS_TMPDIR/fakebin"
  mkdir -p "$FAKE_BIN_DIR"
  mkdir -p "$FAKE_BIN_DIR/.github/scripts/linting"

  SCRIPT="$REPO_ROOT/.github/scripts/linting/lint_branch_name.sh"
  echo "Using script path to copy: $SCRIPT" >&2
  if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: Cannot find the script at: $SCRIPT" >&2
    exit 1
  fi
  cp "$SCRIPT" "$FAKE_BIN_DIR/.github/scripts/linting/"
  chmod +x "$FAKE_BIN_DIR/.github/scripts/linting/$(basename "$SCRIPT")"
  
  cat << "EOF" > "$FAKE_BIN_DIR/.github/scripts/linting/extract_branch_info.sh"
#!/bin/bash
# Fake extract_branch_info.sh for testing lint_branch_name.sh.
BRANCH_NAME="$1"
if [ -z "$BRANCH_NAME" ]; then
  echo "::error::No branch name provided!" >&2
  exit 1
fi

if [[ "$BRANCH_NAME" == "dev-subrepo" ]]; then
  echo '{"branch_type":"dev","component_name":"subrepo"}'
  exit 0
elif [[ "$BRANCH_NAME" == "hotfix-subrepo/fix-bug" ]]; then
  echo '{"branch_type":"hotfix","component_name":"subrepo"}'
  exit 0
elif [[ "$BRANCH_NAME" == "setup-subrepo/configure-db" ]]; then
  echo '{"branch_type":"setup","component_name":"subrepo"}'
  exit 0
else
  echo "::error::Branch name does not follow the convention!" >&2
  exit 1
fi
EOF
  chmod +x "$FAKE_BIN_DIR/.github/scripts/linting/extract_branch_info.sh"
  export PATH="$FAKE_BIN_DIR:$PATH"
}

teardown() {
  echo "Teardown: Removing fake bin directory: $FAKE_BIN_DIR" >&2
  rm -rf "$FAKE_BIN_DIR" || echo "Warning: Failed to remove $FAKE_BIN_DIR" >&2
}

@test "exits with error when no arguments are provided" {
  cd "$FAKE_BIN_DIR"
  run ".github/scripts/linting/$(basename "$SCRIPT")"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Missing required parameters!"* ]]
}

@test "exits with error when only branch name is provided" {
  cd "$FAKE_BIN_DIR"
  run ".github/scripts/linting/$(basename "$SCRIPT")" "dev-subrepo"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Missing required parameters!"* ]]
}

@test "returns 0 for a valid dev branch naming" {
  cd "$FAKE_BIN_DIR"
  run ".github/scripts/linting/$(basename "$SCRIPT")" "dev-subrepo" "subrepo"
  [ "$status" -eq 0 ]
}

@test "returns 0 for a valid hotfix branch naming" {
  cd "$FAKE_BIN_DIR"
  run ".github/scripts/linting/$(basename "$SCRIPT")" "hotfix-subrepo/fix-bug" "subrepo"
  [ "$status" -eq 0 ]
}

@test "returns 0 for a valid setup branch naming" {
  cd "$FAKE_BIN_DIR"
  run ".github/scripts/linting/$(basename "$SCRIPT")" "setup-subrepo/configure-db" "subrepo"
  [ "$status" -eq 0 ]
}

@test "exits with error when branch name does not match allowed patterns" {
  cd "$FAKE_BIN_DIR"
  run ".github/scripts/linting/$(basename "$SCRIPT")" "feature-newthing" "subrepo"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Branch name does not follow the required naming convention!"* ]]
}

@test "exits with error when branch uses a disallowed component - dev" {
  cd "$FAKE_BIN_DIR"
  run ".github/scripts/linting/$(basename "$SCRIPT")" "dev-other" "subrepo"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Branch name does not follow the required naming convention!"* ]]
}

@test "exits with error when branch uses a disallowed component - hotfix" {
  cd "$FAKE_BIN_DIR"
  run ".github/scripts/linting/$(basename "$SCRIPT")" "hotfix-other" "subrepo"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Branch name does not follow the required naming convention!"* ]]
}

@test "exits with error when branch uses a disallowed component - setup" {
  cd "$FAKE_BIN_DIR"
  run ".github/scripts/linting/$(basename "$SCRIPT")" "setup-other" "subrepo"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Branch name does not follow the required naming convention!"* ]]
}
