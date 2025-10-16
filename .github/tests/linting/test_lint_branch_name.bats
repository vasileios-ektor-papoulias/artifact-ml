#!/usr/bin/env bats
# .github/tests/linting/test_lint_branch_name.bats

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
BRANCH_NAME="$1"
if [ -z "$BRANCH_NAME" ]; then
  echo "::error::No branch name provided!" >&2
  exit 1
fi

case "$BRANCH_NAME" in
  dev-subrepo)
    echo '{"branch_type":"dev","component_name":"subrepo"}'; exit 0 ;;
  hotfix-subrepo/fix-bug)
    echo '{"branch_type":"hotfix","component_name":"subrepo"}'; exit 0 ;;
  setup-subrepo/configure-db)
    echo '{"branch_type":"setup","component_name":"subrepo"}'; exit 0 ;;
  dev-subrepo/*)
    echo "::error::Branch name does not follow the convention!" >&2; exit 1 ;;
  *)
    echo "::error::Branch name does not follow the convention!" >&2; exit 1 ;;
esac
EOF
  chmod +x "$FAKE_BIN_DIR/.github/scripts/linting/extract_branch_info.sh"
  export PATH="$FAKE_BIN_DIR:$PATH"
}

teardown() {
  echo "Teardown: Removing fake bin directory: $FAKE_BIN_DIR" >&2
  rm -rf "$FAKE_BIN_DIR" || echo "Warning: Failed to remove $FAKE_BIN_DIR" >&2
}

# ----------------------------
# Parameter validation
# ----------------------------

@test "errors when no arguments are provided (branch_name required)" {
  cd "$FAKE_BIN_DIR"
  run ".github/scripts/linting/$(basename "$SCRIPT")"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr")"
  [[ "$combined" == *"Missing required parameter: <branch_name>"* || "$combined" == *"Missing required parameter: <branch_name>."* ]]
}

# ============================================================
# 1) Defaults behavior
# (defaults: components = root core experiment torch; types = dev hotfix setup)
# ============================================================

@test "defaults: dev-core passes with no allowed lists provided" {
  run ".github/scripts/linting/$(basename "$SCRIPT")" "dev-core"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = '{"branch_type":"dev","component_name":"core"}' ]
}

@test "defaults: hotfix-core/x passes with no allowed lists provided" {
  run ".github/scripts/linting/$(basename "$SCRIPT")" "hotfix-core/fix-ci"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = '{"branch_type":"hotfix","component_name":"core"}' ]
}

@test "defaults: setup-core/x passes with no allowed lists provided" {
  run ".github/scripts/linting/$(basename "$SCRIPT")" "setup-core/init"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = '{"branch_type":"setup","component_name":"core"}' ]
}

@test "defaults: dev-subrepo fails because 'subrepo' is not in default components" {
  run ".github/scripts/linting/$(basename "$SCRIPT")" "dev-subrepo"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr")"
  [[ "$combined" == *"Allowed components: root core experiment torch"* ]]
  [[ "$combined" == *"does not follow the required naming convention or is not allowed!"* ]]
}

# ============================================================
# 2) Component filtering
# ============================================================

@test "component filter: allow 'core experiment' -> dev-core OK, dev-torch fails" {
  run ".github/scripts/linting/$(basename "$SCRIPT")" "dev-core" "core experiment"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = '{"branch_type":"dev","component_name":"core"}' ]

  run ".github/scripts/linting/$(basename "$SCRIPT")" "dev-torch" "core experiment"
  [ "$status" -ne 0 ]
  [[ "$(printf "%s%s" "$output" "$stderr")" == *"does not follow the required naming convention or is not allowed!"* ]]
}

# ============================================================
# 3) Branch type filtering
# ============================================================

@test "type filter: allow only 'dev' -> dev-core OK, hotfix-core/x fails" {
  run ".github/scripts/linting/$(basename "$SCRIPT")" "dev-core" "core experiment torch" "dev"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = '{"branch_type":"dev","component_name":"core"}' ]

  run ".github/scripts/linting/$(basename "$SCRIPT")" "hotfix-core/fix" "core experiment torch" "dev"
  [ "$status" -ne 0 ]
  [[ "$(printf "%s%s" "$output" "$stderr")" == *"Allowed branch types: dev"* ]]
}

@test "type filter: case-insensitive matching (DEV HOTFIX) -> dev-core OK" {
  run ".github/scripts/linting/$(basename "$SCRIPT")" "dev-core" "core" "DEV HOTFIX"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = '{"branch_type":"dev","component_name":"core"}' ]
}

@test "valid: feature-core/x succeeds when allowed (types include 'feature')" {
  run ".github/scripts/linting/$(basename "$SCRIPT")" "feature-core/x" "core" "dev hotfix setup feature"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = '{"branch_type":"feature","component_name":"core"}' ]
}

# ============================================================
# 4) Both filters (type + component)
# ============================================================

@test "both filters: allow type=hotfix and component=core -> hotfix-core/x OK; dev-core fails" {
  run ".github/scripts/linting/$(basename "$SCRIPT")" "hotfix-core/urgent" "core" "hotfix"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = '{"branch_type":"hotfix","component_name":"core"}' ]

  run ".github/scripts/linting/$(basename "$SCRIPT")" "dev-core" "core" "hotfix"
  [ "$status" -ne 0 ]
  [[ "$(printf "%s%s" "$output" "$stderr")" == *"Allowed branch types: hotfix"* ]]
}

# ============================================================
# 5) Invalid shapes / syntactically invalid
# ============================================================

@test "invalid: dev must not contain a slash (dev-core/something)" {
  run ".github/scripts/linting/$(basename "$SCRIPT")" "dev-core/has-slash"
  [ "$status" -ne 0 ]
  [[ "$(printf "%s%s" "$output" "$stderr")" == *"does not follow the required naming convention or is not allowed!"* ]]
}

@test "invalid: hotfix without trailing '/desc' is invalid (hotfix-core)" {
  run ".github/scripts/linting/$(basename "$SCRIPT")" "hotfix-core"
  [ "$status" -ne 0 ]
  [[ "$(printf "%s%s" "$output" "$stderr")" == *"does not follow the required naming convention or is not allowed!"* ]]
}

@test "invalid: setup without trailing '/desc' is invalid (setup-core)" {
  run ".github/scripts/linting/$(basename "$SCRIPT")" "setup-core"
  [ "$status" -ne 0 ]
  [[ "$(printf "%s%s" "$output" "$stderr")" == *"does not follow the required naming convention or is not allowed!"* ]]
}