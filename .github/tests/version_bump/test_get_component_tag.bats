#!/usr/bin/env bats
# .github/tests/version_bump/test_get_component_tag.bats

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


  SCRIPT="$REPO_ROOT/.github/scripts/version_bump/get_component_tag.sh"
  echo "Using script path to copy: $SCRIPT" >&2
  cp "$SCRIPT" "$FAKE_BIN_DIR/.github/scripts/version_bump/"
  chmod +x "$FAKE_BIN_DIR/.github/scripts/version_bump/get_component_tag.sh"

  cd "$FAKE_BIN_DIR"
}

teardown() {
  echo "Teardown: Removing fake bin directory: $BATS_TMPDIR/fakebin" >&2
  rm -rf "$BATS_TMPDIR/fakebin" || echo "Warning: Failed to remove fake bin" >&2
}

@test "generates tag with component name when provided" {
  run ".github/scripts/version_bump/get_component_tag.sh" "1.2.3" "subrepo"
  [ "$status" -eq 0 ]
  [ "$output" = "subrepo-v1.2.3" ]
}

@test "generates simple version tag when no component name is provided" {
  run ".github/scripts/version_bump/get_component_tag.sh" "1.2.3"
  [ "$status" -eq 0 ]
  [ "$output" = "v1.2.3" ]
}

@test "exits with error when no version is provided" {
  run ".github/scripts/version_bump/get_component_tag.sh"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Usage:"* ]] || [[ "$output" == *"Usage:"* ]]
}
