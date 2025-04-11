#!/usr/bin/env bats
# .github/tests/test_detect_bump_pattern.bats

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

  SCRIPT="$REPO_ROOT/.github/scripts/linting/detect_bump_pattern.sh"
  echo "Using script path to copy: $SCRIPT" >&2
  if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: Cannot find the script at: $SCRIPT" >&2
    exit 1
  fi
  cp "$SCRIPT" "$FAKE_BIN_DIR/.github/scripts/linting/"
  chmod +x "$FAKE_BIN_DIR/.github/scripts/linting/$(basename "$SCRIPT")"
  export PATH="$FAKE_BIN_DIR:$PATH"
}

teardown() {
  echo "Teardown: Removing fake bin directory: $FAKE_BIN_DIR" >&2
  rm -rf "$FAKE_BIN_DIR" || echo "Warning: Failed to remove $FAKE_BIN_DIR" >&2
}

@test "returns patch when input starts with 'patch:'" {
  run "$FAKE_BIN_DIR/.github/scripts/linting/$(basename "$SCRIPT")" "patch: fix a bug"
  [ "$status" -eq 0 ]
  final_line="$(get_final_line "$output")"
  echo "DEBUG: final_line=[$final_line]" >&2
  [ "$final_line" = "patch" ]
}

@test "returns minor when input starts with 'minor:'" {
  run "$FAKE_BIN_DIR/.github/scripts/linting/$(basename "$SCRIPT")" "minor: add a feature"
  [ "$status" -eq 0 ]
  final_line="$(get_final_line "$output")"
  echo "DEBUG: final_line=[$final_line]" >&2
  [ "$final_line" = "minor" ]
}

@test "returns major when input starts with 'major:'" {
  run "$FAKE_BIN_DIR/.github/scripts/linting/$(basename "$SCRIPT")" "major: breaking change"
  [ "$status" -eq 0 ]
  final_line="$(get_final_line "$output")"
  echo "DEBUG: final_line=[$final_line]" >&2
  [ "$final_line" = "major" ]
}

@test "returns no-bump when input starts with 'no-bump:'" {
  run "$FAKE_BIN_DIR/.github/scripts/linting/$(basename "$SCRIPT")" "no-bump: docs only"
  [ "$status" -eq 0 ]
  final_line="$(get_final_line "$output")"
  echo "DEBUG: final_line=[$final_line]" >&2
  [ "$final_line" = "no-bump" ]
}

@test "returns patch when using a scoped prefix 'patch('" {
  run "$FAKE_BIN_DIR/.github/scripts/linting/$(basename "$SCRIPT")" "patch(scope): fix scoped bug"
  [ "$status" -eq 0 ]
  final_line="$(get_final_line "$output")"
  echo "DEBUG: final_line=[$final_line]" >&2
  [ "$final_line" = "patch" ]
}

@test "returns minor when using a scoped prefix 'minor('" {
  run "$FAKE_BIN_DIR/.github/scripts/linting/$(basename "$SCRIPT")" "minor(scope): add scoped feature"
  [ "$status" -eq 0 ]
  final_line="$(get_final_line "$output")"
  echo "DEBUG: final_line=[$final_line]" >&2
  [ "$final_line" = "minor" ]
}

@test "returns major when using a scoped prefix 'major('" {
  run "$FAKE_BIN_DIR/.github/scripts/linting/$(basename "$SCRIPT")" "major(scope): refactor API"
  [ "$status" -eq 0 ]
  final_line="$(get_final_line "$output")"
  echo "DEBUG: final_line=[$final_line]" >&2
  [ "$final_line" = "major" ]
}

@test "returns no-bump when using a scoped prefix 'no-bump('" {
  run "$FAKE_BIN_DIR/.github/scripts/linting/$(basename "$SCRIPT")" "no-bump(scope): non-impacting update"
  [ "$status" -eq 0 ]
  final_line="$(get_final_line "$output")"
  echo "DEBUG: final_line=[$final_line]" >&2
  [ "$final_line" = "no-bump" ]
}

@test "exits with error when no argument is provided" {
  run "$FAKE_BIN_DIR/.github/scripts/linting/$(basename "$SCRIPT")"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"No text provided to check!"* ]]
}

@test "exits with error when input does not match the convention" {
  run "$FAKE_BIN_DIR/.github/scripts/linting/$(basename "$SCRIPT")" "fix typo"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Text does not follow the convention!"* ]]
}
