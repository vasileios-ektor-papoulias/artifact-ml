#!/usr/bin/env bats
# .github/tests/version_bump/test_identify_new_version.bats

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
  
  SCRIPT="$REPO_ROOT/.github/scripts/version_bump/identify_new_version.sh"
  echo "Using script path to copy: $SCRIPT" >&2
  cp "$SCRIPT" "$FAKE_BIN_DIR/.github/scripts/version_bump/"
  chmod +x "$FAKE_BIN_DIR/.github/scripts/version_bump/identify_new_version.sh"

  cd "$FAKE_BIN_DIR"
}

teardown() {
  echo "Teardown: Removing fake bin directory: $BATS_TMPDIR/fakebin" >&2
  rm -rf "$BATS_TMPDIR/fakebin" || echo "Warning: Failed to remove fake bin" >&2
}

@test "exits with error when no arguments are provided" {
  run ".github/scripts/version_bump/identify_new_version.sh"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Usage:"* ]]
}

@test "exits with error when only current version is provided" {
  run ".github/scripts/version_bump/identify_new_version.sh" "1.2.3"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Usage:"* ]]
}

@test "exits with error when current version is not in X.Y.Z format" {
  run ".github/scripts/version_bump/identify_new_version.sh" "1.2" "patch"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Error: Current version must be in format X.Y.Z"* ]]
}

@test "exits with error when bump type is invalid" {
  run ".github/scripts/version_bump/identify_new_version.sh" "1.2.3" "invalid"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Error: Unknown bump type"* ]]
}

@test "correctly bumps patch version" {
  run ".github/scripts/version_bump/identify_new_version.sh" "1.2.3" "patch"
  [ "$status" -eq 0 ]
  final_line="$(get_final_line "$output")"
  echo "DEBUG: final_line=[$final_line]" >&2
  [ "$final_line" = "1.2.4" ]
}

@test "correctly bumps minor version" {
  run ".github/scripts/version_bump/identify_new_version.sh" "1.2.3" "minor"
  [ "$status" -eq 0 ]
  final_line="$(get_final_line "$output")"
  echo "DEBUG: final_line=[$final_line]" >&2
  [ "$final_line" = "1.3.0" ]
}

@test "correctly bumps major version" {
  run ".github/scripts/version_bump/identify_new_version.sh" "1.2.3" "major"
  [ "$status" -eq 0 ]
  final_line="$(get_final_line "$output")"
  echo "DEBUG: final_line=[$final_line]" >&2
  [ "$final_line" = "2.0.0" ]
}

@test "handles version with leading zeros" {
  run ".github/scripts/version_bump/identify_new_version.sh" "1.02.03" "patch"
  [ "$status" -eq 0 ]
  final_line="$(get_final_line "$output")"
  echo "DEBUG: final_line=[$final_line]" >&2
  [ "$final_line" = "1.2.4" ]
}

@test "handles version with large numbers" {
  run ".github/scripts/version_bump/identify_new_version.sh" "10.20.30" "minor"
  [ "$status" -eq 0 ]
  final_line="$(get_final_line "$output")"
  echo "DEBUG: final_line=[$final_line]" >&2
  [ "$final_line" = "10.21.0" ]
}
