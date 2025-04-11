#!/usr/bin/env bats
# .github/tests/version_bump/test_bump_component_version.bats

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

  SCRIPT="$REPO_ROOT/.github/scripts/version_bump/bump_component_version.sh"
  cp "$SCRIPT" "$FAKE_BIN_DIR/.github/scripts/version_bump/"
  chmod +x "$FAKE_BIN_DIR/.github/scripts/version_bump/bump_component_version.sh"

  cat << "EOF" > "$FAKE_BIN_DIR/.github/scripts/version_bump/update_pyproject.sh"
#!/bin/bash
echo "Fake update_pyproject.sh called with: $@" >&2
echo "1.2.3"
exit 0
EOF
  chmod +x "$FAKE_BIN_DIR/.github/scripts/version_bump/update_pyproject.sh"

  cat << "EOF" > "$FAKE_BIN_DIR/.github/scripts/version_bump/get_component_tag.sh"
#!/bin/bash
echo "Fake get_component_tag.sh called with: $@" >&2
if [ -n "$2" ]; then
  echo "$2-v$1"
else
  echo "v$1"
fi
exit 0
EOF
  chmod +x "$FAKE_BIN_DIR/.github/scripts/version_bump/get_component_tag.sh"

  cat << "EOF" > "$FAKE_BIN_DIR/.github/scripts/version_bump/push_version_update.sh"
#!/bin/bash
echo "Fake push_version_update.sh called with: $@" >&2
exit 0
EOF
  chmod +x "$FAKE_BIN_DIR/.github/scripts/version_bump/push_version_update.sh"

  SCRIPT="$REPO_ROOT/.github/scripts/version_bump/bump_component_version.sh"
  echo "Using script path to copy: $SCRIPT" >&2

  cd "$FAKE_BIN_DIR"
}

teardown() {
  echo "Teardown: Removing fake bin directory: $BATS_TMPDIR/fakebin" >&2
  rm -rf "$BATS_TMPDIR/fakebin" || echo "Warning: Failed to remove fake bin" >&2
}

@test "successfully bumps version with component name" {
  run ".github/scripts/version_bump/bump_component_version.sh" "patch" "subrepo" "subrepo/pyproject.toml"
  [ "$status" -eq 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Fake update_pyproject.sh called with: subrepo/pyproject.toml patch"* ]]
  [[ "$combined" == *"Fake get_component_tag.sh called with: 1.2.3 subrepo"* ]]
  [[ "$combined" == *"Fake push_version_update.sh called with: subrepo-v1.2.3 subrepo/pyproject.toml"* ]]
}

@test "successfully bumps version without component name" {
  run ".github/scripts/version_bump/bump_component_version.sh" "minor" "" "pyproject.toml"
  [ "$status" -eq 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Fake update_pyproject.sh called with: pyproject.toml minor"* ]]
  [[ "$combined" == *"Fake get_component_tag.sh called with: 1.2.3 "* ]]
  [[ "$combined" == *"Fake push_version_update.sh called with: v1.2.3 pyproject.toml"* ]]
}