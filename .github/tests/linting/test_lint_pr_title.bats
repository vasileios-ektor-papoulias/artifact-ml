#!/usr/bin/env bats
# .github/tests/linting/test_lint_pr_title.bats

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
  mkdir -p "$FAKE_BIN_DIR/.github/scripts/linting"

  cp -r "$REPO_ROOT/.github" "$FAKE_BIN_DIR/"

  cat << "EOF" > "$FAKE_BIN_DIR/.github/scripts/linting/detect_bump_pattern.sh"
#!/bin/bash
# Fake detect_bump_pattern.sh for testing lint_pr_title.sh.
INPUT=$(echo "$1" | tr '[:upper:]' '[:lower:]')
if [ -z "$INPUT" ]; then
  echo "::error::No text provided to check!" >&2
  exit 1
fi
if [[ "$INPUT" == patch:* || "$INPUT" == minor:* || "$INPUT" == major:* || "$INPUT" == no-bump:* ]]; then
  echo "$FAKE_BUMP_TYPE"
  exit 0
else
  echo "::error::Text does not follow the convention!" >&2
  exit 1
fi
EOF
  chmod +x "$FAKE_BIN_DIR/.github/scripts/linting/detect_bump_pattern.sh"

  
  cat << "EOF" > "$FAKE_BIN_DIR/.github/scripts/linting/extract_branch_info.sh"
#!/bin/bash
# Fake extract_branch_info.sh for testing lint_pr_title.sh.
BRANCH_NAME="$1"
if [ -z "$BRANCH_NAME" ]; then
  echo "::error::No branch name provided!" >&2
  exit 1
fi

if [[ "$BRANCH_NAME" == *"root"* ]]; then
  echo '{"branch_type":"hotfix","component_name":"root"}'
  exit 0
elif [[ "$BRANCH_NAME" == *"subrepo"* ]]; then
  echo '{"branch_type":"hotfix","component_name":"subrepo"}'
  exit 0
else
  echo "::error::Branch name does not follow the convention!" >&2
  exit 1
fi
EOF
  chmod +x "$FAKE_BIN_DIR/.github/scripts/linting/extract_branch_info.sh"

  cat << EOF > "$FAKE_BIN_DIR/git"
#!/bin/bash
echo "Fake git: no git call expected" >&2
exit 0
EOF
  chmod +x "$FAKE_BIN_DIR/git"
  export PATH="$FAKE_BIN_DIR:$PATH"

  cd "$FAKE_BIN_DIR"
}

teardown() {
  echo "Teardown: Removing fake bin directory: $BATS_TMPDIR/fakebin" >&2
  rm -rf "$BATS_TMPDIR/fakebin" || echo "Warning: Failed to remove fake bin" >&2
}

@test "exits with error when no PR title is provided" {
  run ".github/scripts/linting/lint_pr_title.sh"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"No PR title provided!"* ]]
}

@test "returns bump type when PR title is valid" {
  export FAKE_BUMP_TYPE="minor"
  # Set a valid PR title (e.g., following the convention: "minor: add feature")
  run ".github/scripts/linting/lint_pr_title.sh" "minor: add feature"
  [ "$status" -eq 0 ]
  final_line="$(get_final_line "$output")"
  echo "DEBUG: final_line=[$final_line]" >&2
  [ "$final_line" = "minor" ]
}

@test "exits with error when PR title does not follow convention" {
  run ".github/scripts/linting/lint_pr_title.sh" "invalid PR title"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Text does not follow the convention!"* ]]
}

@test "exits with error when root component PR does not use no-bump prefix" {
  export FAKE_BUMP_TYPE="patch"
  run ".github/scripts/linting/lint_pr_title.sh" "patch: fix bug" "hotfix-root/fix-bug"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"PRs from root component branches must use 'no-bump:' prefix"* ]]
}

@test "succeeds when root component PR uses no-bump prefix" {
  export FAKE_BUMP_TYPE="no-bump"
  run ".github/scripts/linting/lint_pr_title.sh" "no-bump: update docs" "hotfix-root/update-docs"
  [ "$status" -eq 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Root component PR has correct no-bump prefix"* ]]
  final_line="$(get_final_line "$output")"
  [ "$final_line" = "no-bump" ]
}

@test "succeeds when non-root component PR uses any valid prefix" {
  export FAKE_BUMP_TYPE="patch"
  run ".github/scripts/linting/lint_pr_title.sh" "patch: fix bug" "hotfix-subrepo/fix-bug"
  [ "$status" -eq 0 ]
  final_line="$(get_final_line "$output")"
  [ "$final_line" = "patch" ]
}
