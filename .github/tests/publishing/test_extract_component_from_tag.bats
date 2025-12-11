#!/usr/bin/env bats
# .github/tests/publishing/test_extract_component_from_tag.bats

# Helper function: Extract the final (last non-empty) line from a string.
get_final_line() {
  echo "$1" | sed '/^[[:space:]]*$/d' | tail -n 1 | tr -d '\r\n'
}

setup() {
  TEST_DIR="$(dirname "$BATS_TEST_FILENAME")"
  REPO_ROOT="$(cd "$TEST_DIR"/../../.. && pwd)"
  echo "Repository root is: $REPO_ROOT" >&2
  
  SCRIPT="$REPO_ROOT/.github/scripts/publishing/extract_component_from_tag.sh"
  echo "Using script path: $SCRIPT" >&2
  
  if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: Cannot find the script at: $SCRIPT" >&2
    exit 1
  fi
  
  chmod +x "$SCRIPT"
}

@test "extracts component and version from tag (core)" {
  run "$SCRIPT" "push" "core-v1.2.3" ""
  [ "$status" -eq 0 ]
  final_line="$(get_final_line "$output")"
  echo "DEBUG: final_line=[$final_line]" >&2
  [[ "$final_line" == *'"component":"core"'* ]]
  [[ "$final_line" == *'"version":"1.2.3"'* ]]
}

@test "extracts component and version from tag (experiment)" {
  run "$SCRIPT" "push" "experiment-v0.5.1" ""
  [ "$status" -eq 0 ]
  final_line="$(get_final_line "$output")"
  echo "DEBUG: final_line=[$final_line]" >&2
  [[ "$final_line" == *'"component":"experiment"'* ]]
  [[ "$final_line" == *'"version":"0.5.1"'* ]]
}

@test "extracts component and version from tag (torch)" {
  run "$SCRIPT" "push" "torch-v2.0.0" ""
  [ "$status" -eq 0 ]
  final_line="$(get_final_line "$output")"
  echo "DEBUG: final_line=[$final_line]" >&2
  [[ "$final_line" == *'"component":"torch"'* ]]
  [[ "$final_line" == *'"version":"2.0.0"'* ]]
}

@test "handles manual trigger with core" {
  run "$SCRIPT" "workflow_dispatch" "" "core"
  [ "$status" -eq 0 ]
  final_line="$(get_final_line "$output")"
  echo "DEBUG: final_line=[$final_line]" >&2
  [[ "$final_line" == *'"component":"core"'* ]]
  [[ "$final_line" == *'"version":"manual"'* ]]
}

@test "handles manual trigger with experiment" {
  run "$SCRIPT" "workflow_dispatch" "" "experiment"
  [ "$status" -eq 0 ]
  final_line="$(get_final_line "$output")"
  echo "DEBUG: final_line=[$final_line]" >&2
  [[ "$final_line" == *'"component":"experiment"'* ]]
  [[ "$final_line" == *'"version":"manual"'* ]]
}

@test "handles manual trigger with torch" {
  run "$SCRIPT" "workflow_dispatch" "" "torch"
  [ "$status" -eq 0 ]
  final_line="$(get_final_line "$output")"
  echo "DEBUG: final_line=[$final_line]" >&2
  [[ "$final_line" == *'"component":"torch"'* ]]
  [[ "$final_line" == *'"version":"manual"'* ]]
}

@test "exits with error when event_name is missing" {
  run "$SCRIPT"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Missing required parameter"* ]]
}

@test "exits with error when manual component is missing" {
  run "$SCRIPT" "workflow_dispatch" "" ""
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Manual component required"* ]]
}

@test "exits with error when tag is missing for push event" {
  run "$SCRIPT" "push" "" ""
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Tag name required"* ]]
}

@test "exits with error for invalid tag format" {
  run "$SCRIPT" "push" "invalid-tag" ""
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Invalid tag format"* ]]
}

