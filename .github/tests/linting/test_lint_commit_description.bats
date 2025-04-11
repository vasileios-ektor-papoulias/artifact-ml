#!/usr/bin/env bats
# .github/tests/test_lint_commit_description.bats

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

  cat << "EOF" > "$FAKE_BIN_DIR/.github/scripts/linting/detect_bump_pattern.sh"
#!/bin/bash
# Fake detect_bump_pattern.sh for testing lint_commit_description.sh.
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

  cat << EOF > "$FAKE_BIN_DIR/git"
#!/bin/bash
# Fake git command for testing lint_commit_description.sh.
if [[ "\$@" == *"--pretty=format:%b"* ]]; then
  echo "\$FAKE_COMMIT_DESC"
  exit 0
fi
echo "Fake git: unhandled args: \$@" >&2
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

@test "exits with error when commit description is empty" {
  export FAKE_COMMIT_DESC=""
  run ".github/scripts/linting/lint_commit_description.sh"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Merge commit has no description!"* ]]
}

@test "returns bump type when commit description is valid" {
  export FAKE_COMMIT_DESC="patch: fix bug"
  export FAKE_BUMP_TYPE="patch"
  run ".github/scripts/linting/lint_commit_description.sh"
  [ "$status" -eq 0 ]
  final_line="$(get_final_line "$output")"
  echo "DEBUG: final_line=[$final_line]" >&2
  [ "$final_line" = "patch" ]
}

@test "exits with error when commit description does not follow convention" {
  export FAKE_COMMIT_DESC="invalid commit description"
  run ".github/scripts/linting/lint_commit_description.sh"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Text does not follow the convention!"* ]]
}
