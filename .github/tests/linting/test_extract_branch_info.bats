#!/usr/bin/env bats
# .github/tests/linting/test_extract_branch_info.bats

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
  chmod +x "$FAKE_BIN_DIR/.github/scripts/linting/extract_branch_info.sh"
  cd "$FAKE_BIN_DIR"
}

teardown() {
  echo "Teardown: Removing fake bin directory: $BATS_TMPDIR/fakebin" >&2
  rm -rf "$BATS_TMPDIR/fakebin" || echo "Warning: Failed to remove fake bin" >&2
}

@test "exits with error when no branch name is provided" {
  run ".github/scripts/linting/extract_branch_info.sh"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"No branch name provided!"* ]]
}

@test "exits with error when branch name does not follow convention" {
  run ".github/scripts/linting/extract_branch_info.sh" "feature-newcomponent"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Branch name does not follow the convention!"* ]]
}

@test "returns branch type and component name for a valid dev branch" {
  run ".github/scripts/linting/extract_branch_info.sh" "dev-mycomponent"
  [ "$status" -eq 0 ]
  final_line="$(get_final_line "$output")"
  echo "DEBUG: final_line=[$final_line]" >&2
  [ "$final_line" = '{"branch_type":"dev","component_name":"mycomponent"}' ]
}

@test "returns branch type and component name for a valid hotfix branch" {
  run ".github/scripts/linting/extract_branch_info.sh" "hotfix-mycomponent/fix-critical-bug"
  [ "$status" -eq 0 ]
  final_line="$(get_final_line "$output")"
  echo "DEBUG: final_line=[$final_line]" >&2
  [ "$final_line" = '{"branch_type":"hotfix","component_name":"mycomponent"}' ]
}

@test "returns branch type and component name for a valid setup branch" {
  run ".github/scripts/linting/extract_branch_info.sh" "setup-mycomponent/initial-config"
  [ "$status" -eq 0 ]
  final_line="$(get_final_line "$output")"
  echo "DEBUG: final_line=[$final_line]" >&2
  [ "$final_line" = '{"branch_type":"setup","component_name":"mycomponent"}' ]
}

@test "correctly handles component name with hyphens" {
  run ".github/scripts/linting/extract_branch_info.sh" "dev-my-component"
  [ "$status" -eq 0 ]
  final_line="$(get_final_line "$output")"
  echo "DEBUG: final_line=[$final_line]" >&2
  [ "$final_line" = '{"branch_type":"dev","component_name":"my-component"}' ]
}

@test "correctly handles component name with underscores" {
  run ".github/scripts/linting/extract_branch_info.sh" "dev-my_component"
  [ "$status" -eq 0 ]
  final_line="$(get_final_line "$output")"
  echo "DEBUG: final_line=[$final_line]" >&2
  [ "$final_line" = '{"branch_type":"dev","component_name":"my_component"}' ]
}

@test "correctly handles root as component name" {
  run ".github/scripts/linting/extract_branch_info.sh" "hotfix-root/update-docs"
  [ "$status" -eq 0 ]
  final_line="$(get_final_line "$output")"
  echo "DEBUG: final_line=[$final_line]" >&2
  [ "$final_line" = '{"branch_type":"hotfix","component_name":"root"}' ]
}