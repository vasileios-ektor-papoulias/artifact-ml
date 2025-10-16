#!/usr/bin/env bats
# .github/tests/linting/test_lint_pr_title.bats

# Helper: last non-empty line
get_final_line() {
  echo "$1" | sed '/^[[:space:]]*$/d' | tail -n 1 | tr -d '\r\n'
}

setup() {
  TEST_DIR="$(dirname "$BATS_TEST_FILENAME")"
  REPO_ROOT="$(cd "$TEST_DIR/../../.." && pwd)"
  echo "Repository root is: $REPO_ROOT" >&2

  FAKE_BIN_DIR="$BATS_TMPDIR/fakebin"
  mkdir -p "$FAKE_BIN_DIR/.github/scripts/linting"

  # --- Script under test ---
  SCRIPT_SRC="$REPO_ROOT/.github/scripts/linting/lint_pr_title.sh"
  if [ ! -f "$SCRIPT_SRC" ]; then
    echo "ERROR: Cannot find the script at: $SCRIPT_SRC" >&2
    exit 1
  fi
  cp "$SCRIPT_SRC" "$FAKE_BIN_DIR/.github/scripts/linting/"
  chmod +x "$FAKE_BIN_DIR/.github/scripts/linting/lint_pr_title.sh"

  # --- Stub: detect_bump_pattern.sh ---
  cat << 'EOF' > "$FAKE_BIN_DIR/.github/scripts/linting/detect_bump_pattern.sh"
#!/bin/bash
# Intentionally keep this stub super simple & portable: no bash regex.
INPUT="${1-}"
if [ -z "$INPUT" ]; then
  echo "::error::No text provided to check!" >&2
  exit 1
fi

# normalize to lowercase
LOWER="$(printf "%s" "$INPUT" | tr '[:upper:]' '[:lower:]')"

# Accept either "<type>:" ... or "<type>(scope): ..." (any scope)
# Using case globs instead of [[ =~ ]] avoids regex parsing issues on some environments.
case "$LOWER" in
  patch:*|patch\(*\):*)       echo "patch"; exit 0 ;;
  minor:*|minor\(*\):*)       echo "minor"; exit 0 ;;
  major:*|major\(*\):*)       echo "major"; exit 0 ;;
  no-bump:*|no-bump\(*\):*)   echo "no-bump"; exit 0 ;;
  *)
    echo "::error::Text does not follow the convention!" >&2
    exit 1
    ;;
esac
EOF
  chmod +x "$FAKE_BIN_DIR/.github/scripts/linting/detect_bump_pattern.sh"

  # --- Stub: extract_branch_info.sh (shape-only parser) ---
  # Contract: success -> print JSON {"branch_type":"...","component_name":"..."}; failure -> exit 1
  cat << "EOF" > "$FAKE_BIN_DIR/.github/scripts/linting/extract_branch_info.sh"
#!/bin/bash
set -euo pipefail
BRANCH_NAME="${1-}"

if [ -z "${BRANCH_NAME}" ]; then
  echo "::error::No branch name provided!" >&2
  exit 1
fi

# Accept the full matrix (shape-only, no policy):
# dev-{core,experiment,torch}
# hotfix-{root,core,experiment,torch}/...
# setup-{root,core,experiment,torch}/...
case "$BRANCH_NAME" in
  # dev (no slash)
  dev-core)         echo '{"branch_type":"dev","component_name":"core"}'; exit 0 ;;
  dev-experiment)   echo '{"branch_type":"dev","component_name":"experiment"}'; exit 0 ;;
  dev-torch)        echo '{"branch_type":"dev","component_name":"torch"}'; exit 0 ;;

  # hotfix (must have slash after component)
  hotfix-root/*)        echo '{"branch_type":"hotfix","component_name":"root"}'; exit 0 ;;
  hotfix-core/*)        echo '{"branch_type":"hotfix","component_name":"core"}'; exit 0 ;;
  hotfix-experiment/*)  echo '{"branch_type":"hotfix","component_name":"experiment"}'; exit 0 ;;
  hotfix-torch/*)       echo '{"branch_type":"hotfix","component_name":"torch"}'; exit 0 ;;

  # setup (must have slash after component)
  setup-root/*)         echo '{"branch_type":"setup","component_name":"root"}'; exit 0 ;;
  setup-core/*)         echo '{"branch_type":"setup","component_name":"core"}'; exit 0 ;;
  setup-experiment/*)   echo '{"branch_type":"setup","component_name":"experiment"}'; exit 0 ;;
  setup-torch/*)        echo '{"branch_type":"setup","component_name":"torch"}'; exit 0 ;;

  *)
    echo "::error::Branch name does not follow the required naming convention!" >&2
    exit 1
    ;;
esac
EOF
  chmod +x "$FAKE_BIN_DIR/.github/scripts/linting/extract_branch_info.sh"

  # Fake git (not used by lint_pr_title.sh)
  cat << "EOF" > "$FAKE_BIN_DIR/git"
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

# ======================================================================
# A) BUMP TYPES WITHOUT A BRANCH NAME — FLATTENED
# ======================================================================

@test "A1: no-branch -> patch" {
  run ".github/scripts/linting/lint_pr_title.sh" "patch: fix"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = "patch" ]
}

@test "A2: no-branch -> minor" {
  run ".github/scripts/linting/lint_pr_title.sh" "minor: add"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = "minor" ]
}

@test "A3: no-branch -> major" {
  run ".github/scripts/linting/lint_pr_title.sh" "major: break"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = "major" ]
}

@test "A4: no-branch -> no-bump" {
  run ".github/scripts/linting/lint_pr_title.sh" "no-bump: docs"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = "no-bump" ]
}

@test "A5: no-branch scoped -> patch(scope)" {
  run ".github/scripts/linting/lint_pr_title.sh" "patch(core): fix"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = "patch" ]
}

@test "A6: no-branch scoped -> minor(scope)" {
  run ".github/scripts/linting/lint_pr_title.sh" "minor(ui): add"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = "minor" ]
}

@test "A7: no-branch scoped -> major(scope)" {
  run ".github/scripts/linting/lint_pr_title.sh" "major(api): break"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = "major" ]
}

@test "A8: no-branch scoped -> no-bump(scope)" {
  run ".github/scripts/linting/lint_pr_title.sh" "no-bump(docs): tidy"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = "no-bump" ]
}

# ======================================================================
# B) BUMP TYPES WITH A BRANCH NAME — FLATTENED
# - root requires no-bump
# - non-root accepts any valid bump type
# ======================================================================

# --- dev-core ---
@test "B1: dev-core + patch" {
  run ".github/scripts/linting/lint_pr_title.sh" "patch: x" "dev-core"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = "patch" ]
}
@test "B2: dev-core + minor" {
  run ".github/scripts/linting/lint_pr_title.sh" "minor: x" "dev-core"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = "minor" ]
}
@test "B3: dev-core + major" {
  run ".github/scripts/linting/lint_pr_title.sh" "major: x" "dev-core"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = "major" ]
}
@test "B4: dev-core + no-bump" {
  run ".github/scripts/linting/lint_pr_title.sh" "no-bump: x" "dev-core"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = "no-bump" ]
}

# --- dev-experiment ---
@test "B5: dev-experiment + patch" {
  run ".github/scripts/linting/lint_pr_title.sh" "patch: x" "dev-experiment"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = "patch" ]
}
@test "B6: dev-experiment + minor" {
  run ".github/scripts/linting/lint_pr_title.sh" "minor: x" "dev-experiment"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = "minor" ]
}
@test "B7: dev-experiment + major" {
  run ".github/scripts/linting/lint_pr_title.sh" "major: x" "dev-experiment"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = "major" ]
}
@test "B8: dev-experiment + no-bump" {
  run ".github/scripts/linting/lint_pr_title.sh" "no-bump: x" "dev-experiment"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = "no-bump" ]
}

# --- dev-torch ---
@test "B9: dev-torch + patch" {
  run ".github/scripts/linting/lint_pr_title.sh" "patch: x" "dev-torch"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = "patch" ]
}
@test "B10: dev-torch + minor" {
  run ".github/scripts/linting/lint_pr_title.sh" "minor: x" "dev-torch"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = "minor" ]
}
@test "B11: dev-torch + major" {
  run ".github/scripts/linting/lint_pr_title.sh" "major: x" "dev-torch"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = "major" ]
}
@test "B12: dev-torch + no-bump" {
  run ".github/scripts/linting/lint_pr_title.sh" "no-bump: x" "dev-torch"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = "no-bump" ]
}

# --- hotfix core/experiment/torch ---
@test "B13: hotfix-core + patch" {
  run ".github/scripts/linting/lint_pr_title.sh" "patch: x" "hotfix-core/f1"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = "patch" ]
}
@test "B14: hotfix-experiment + minor" {
  run ".github/scripts/linting/lint_pr_title.sh" "minor: x" "hotfix-experiment/f2"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = "minor" ]
}
@test "B15: hotfix-torch + major" {
  run ".github/scripts/linting/lint_pr_title.sh" "major: x" "hotfix-torch/f3"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = "major" ]
}
@test "B16: hotfix-core + no-bump" {
  run ".github/scripts/linting/lint_pr_title.sh" "no-bump: x" "hotfix-core/f1"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = "no-bump" ]
}

# --- setup core/experiment/torch ---
@test "B17: setup-core + patch" {
  run ".github/scripts/linting/lint_pr_title.sh" "patch: x" "setup-core/i1"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = "patch" ]
}
@test "B18: setup-experiment + minor" {
  run ".github/scripts/linting/lint_pr_title.sh" "minor: x" "setup-experiment/i2"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = "minor" ]
}
@test "B19: setup-torch + major" {
  run ".github/scripts/linting/lint_pr_title.sh" "major: x" "setup-torch/i3"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = "major" ]
}
@test "B20: setup-core + no-bump" {
  run ".github/scripts/linting/lint_pr_title.sh" "no-bump: x" "setup-core/i1"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = "no-bump" ]
}

# --- root (must use no-bump) ---
@test "B21: hotfix-root + no-bump (OK)" {
  run ".github/scripts/linting/lint_pr_title.sh" "no-bump: x" "hotfix-root/r1"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = "no-bump" ]
}
@test "B22: hotfix-root + patch (ERR)" {
  run ".github/scripts/linting/lint_pr_title.sh" "patch: x" "hotfix-root/r1"
  [ "$status" -ne 0 ]
  [[ "$(printf "%s%s" "$output" "$stderr")" == *"must use 'no-bump:' prefix"* ]]
}
@test "B23: hotfix-root + minor (ERR)" {
  run ".github/scripts/linting/lint_pr_title.sh" "minor: x" "hotfix-root/r1"
  [ "$status" -ne 0 ]
  [[ "$(printf "%s%s" "$output" "$stderr")" == *"must use 'no-bump:' prefix"* ]]
}
@test "B24: hotfix-root + major (ERR)" {
  run ".github/scripts/linting/lint_pr_title.sh" "major: x" "hotfix-root/r1"
  [ "$status" -ne 0 ]
  [[ "$(printf "%s%s" "$output" "$stderr")" == *"must use 'no-bump:' prefix"* ]]
}

@test "B25: setup-root + no-bump(scope) (OK)" {
  run ".github/scripts/linting/lint_pr_title.sh" "no-bump(docs): x" "setup-root/r1"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = "no-bump" ]
}
@test "B26: setup-root + patch (ERR)" {
  run ".github/scripts/linting/lint_pr_title.sh" "patch: x" "setup-root/r1"
  [ "$status" -ne 0 ]
  [[ "$(printf "%s%s" "$output" "$stderr")" == *"must use 'no-bump:' prefix"* ]]
}
@test "B27: setup-root + minor (ERR)" {
  run ".github/scripts/linting/lint_pr_title.sh" "minor: x" "setup-root/r1"
  [ "$status" -ne 0 ]
  [[ "$(printf "%s%s" "$output" "$stderr")" == *"must use 'no-bump:' prefix"* ]]
}
@test "B28: setup-root + major (ERR)" {
  run ".github/scripts/linting/lint_pr_title.sh" "major: x" "setup-root/r1"
  [ "$status" -ne 0 ]
  [[ "$(printf "%s%s" "$output" "$stderr")" == *"must use 'no-bump:' prefix"* ]]
}

# ======================================================================
# C) BADLY FORMATTED PR TITLES (explicit)
# ======================================================================

@test "C1: missing colon -> error" {
  run ".github/scripts/linting/lint_pr_title.sh" "minor add feature"
  [ "$status" -ne 0 ]
  [[ "$(printf "%s%s" "$output" "$stderr")" == *"Text does not follow the convention!"* ]]
}

@test "C2: uppercase type without colon -> error" {
  run ".github/scripts/linting/lint_pr_title.sh" "MAJOR breaking change"
  [ "$status" -ne 0 ]
  [[ "$(printf "%s%s" "$output" "$stderr")" == *"Text does not follow the convention!"* ]]
}

@test "C3: random string -> error" {
  run ".github/scripts/linting/lint_pr_title.sh" "update readme and scripts"
  [ "$status" -ne 0 ]
  [[ "$(printf "%s%s" "$output" "$stderr")" == *"Text does not follow the convention!"* ]]
}

@test "C4: empty title -> parameter error" {
  run ".github/scripts/linting/lint_pr_title.sh"
  [ "$status" -ne 0 ]
  [[ "$(printf "%s%s" "$output" "$stderr")" == *"No PR title provided!"* ]]
}
