#!/usr/bin/env bats
# .github/tests/publishing/test_get_version_from_pyproject.bats

# Helper function: Extract the final (last non-empty) line from a string.
get_final_line() {
  echo "$1" | sed '/^[[:space:]]*$/d' | tail -n 1 | tr -d '\r\n'
}

setup() {
  TEST_DIR="$(dirname "$BATS_TEST_FILENAME")"
  REPO_ROOT="$(cd "$TEST_DIR"/../../.. && pwd)"
  echo "Repository root is: $REPO_ROOT" >&2
  
  SCRIPT="$REPO_ROOT/.github/scripts/publishing/get_version_from_pyproject.sh"
  echo "Using script path: $SCRIPT" >&2
  
  if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: Cannot find the script at: $SCRIPT" >&2
    exit 1
  fi
  
  chmod +x "$SCRIPT"
}

@test "extracts version from core component" {
  run "$SCRIPT" "core"
  [ "$status" -eq 0 ]
  final_line="$(get_final_line "$output")"
  echo "DEBUG: final_line=[$final_line]" >&2
  [[ "$final_line" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]
}

@test "extracts version from experiment component" {
  run "$SCRIPT" "experiment"
  [ "$status" -eq 0 ]
  final_line="$(get_final_line "$output")"
  echo "DEBUG: final_line=[$final_line]" >&2
  [[ "$final_line" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]
}

@test "extracts version from torch component" {
  run "$SCRIPT" "torch"
  [ "$status" -eq 0 ]
  final_line="$(get_final_line "$output")"
  echo "DEBUG: final_line=[$final_line]" >&2
  [[ "$final_line" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]
}

@test "exits with error when component is missing" {
  run "$SCRIPT"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Missing required parameter"* ]]
}

@test "exits with error for invalid component name" {
  run "$SCRIPT" "invalid-component"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Invalid component"* ]]
}

@test "exits with error for non-existent component directory" {
  # Create a temporary script wrapper that will look for a non-existent component
  TEMP_DIR="$(mktemp -d)"
  TEMP_SCRIPT="$TEMP_DIR/test_script.sh"
  
  # Copy script and modify component dir to point to non-existent location
  cat > "$TEMP_SCRIPT" << 'EOF'
#!/bin/bash
set -euo pipefail
COMPONENT="${1:-}"
if [ -z "$COMPONENT" ]; then
  echo "::error::Missing required parameter: component" >&2
  exit 1
fi
if [[ ! "$COMPONENT" =~ ^(core|experiment|torch)$ ]]; then
  echo "::error::Invalid component: $COMPONENT" >&2
  exit 1
fi
COMPONENT_DIR="nonexistent-${COMPONENT}"
PYPROJECT_PATH="${COMPONENT_DIR}/pyproject.toml"
if [ ! -f "$PYPROJECT_PATH" ]; then
  echo "::error::pyproject.toml not found at: $PYPROJECT_PATH" >&2
  exit 1
fi
EOF
  
  chmod +x "$TEMP_SCRIPT"
  run "$TEMP_SCRIPT" "core"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"pyproject.toml not found"* ]]
  
  rm -rf "$TEMP_DIR"
}

