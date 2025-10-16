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

@test "errors when non-dev branch is missing '/<descriptive>' (e.g., feature-foo)" {
  run ".github/scripts/linting/extract_branch_info.sh" "feature-newcomponent"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  # Generic invalid message used by the updated script
  [[ "$combined" == *"does not follow the required convention"* ]]
}

@test "errors when dev branch has a trailing '/…' segment" {
  run ".github/scripts/linting/extract_branch_info.sh" "dev-mycomponent/extra"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  # The script emits a specific message for invalid dev shape
  [[ "$combined" == *"Invalid 'dev' branch shape"* ]] || [[ "$combined" == *"must NOT contain"* ]]
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

@test "returns branch type and component name for a valid feature branch" {
  run ".github/scripts/linting/extract_branch_info.sh" "feature-torch/add-dataloader"
  [ "$status" -eq 0 ]
  final_line="$(get_final_line "$output")"
  echo "DEBUG: final_line=[$final_line]" >&2
  [ "$final_line" = '{"branch_type":"feature","component_name":"torch"}' ]
}

@test "returns branch type and component name for a valid fix branch" {
  run ".github/scripts/linting/extract_branch_info.sh" "fix-core/harden-ci"
  [ "$status" -eq 0 ]
  final_line="$(get_final_line "$output")"
  echo "DEBUG: final_line=[$final_line]" >&2
  [ "$final_line" = '{"branch_type":"fix","component_name":"core"}' ]
}

@test "accepts hyphens inside the component name (dev-my-component)" {
  run ".github/scripts/linting/extract_branch_info.sh" "dev-my-component"
  [ "$status" -eq 0 ]
  final_line="$(get_final_line "$output")"
  echo "DEBUG: final_line=[$final_line]" >&2
  [ "$final_line" = '{"branch_type":"dev","component_name":"my-component"}' ]
}

@test "accepts underscores inside the component name (dev-my_component)" {
  run ".github/scripts/linting/extract_branch_info.sh" "dev-my_component"
  [ "$status" -eq 0 ]
  final_line="$(get_final_line "$output")"
  echo "DEBUG: final_line=[$final_line]" >&2
  [ "$final_line" = '{"branch_type":"dev","component_name":"my_component"}' ]
}

@test "rejects underscore as type-component delimiter (dev_my_component)" {
  run ".github/scripts/linting/extract_branch_info.sh" "dev_my_component"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr")"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"does not follow the required convention"* ]] || [[ "$combined" == *"Invalid 'dev' branch shape"* ]]
}


@test "correctly handles root as component name" {
  run ".github/scripts/linting/extract_branch_info.sh" "hotfix-root/update-docs"
  [ "$status" -eq 0 ]
  final_line="$(get_final_line "$output")"
  echo "DEBUG: final_line=[$final_line]" >&2
  [ "$final_line" = '{"branch_type":"hotfix","component_name":"root"}' ]
}
