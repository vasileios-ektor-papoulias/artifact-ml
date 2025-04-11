#!/usr/bin/env bats
# .github/tests/enforce_path/test_ensure_changed_files_in_dir.bats

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

  cat << EOF > "$FAKE_BIN_DIR/git"
#!/bin/bash
# Fake git command for testing ensure_changed_files_in_dir.sh.
# It handles: git diff --name-only origin/<base_ref> origin/<head_ref>
if [[ "\$@" == *"diff"* && "\$@" == *"--name-only"* ]]; then
  echo "\$FAKE_DIFF_OUTPUT"
  exit 0
fi
echo "Fake git: unhandled args: \$@" >&2
exit 1
EOF
  chmod +x "$FAKE_BIN_DIR/git"
  export PATH="$FAKE_BIN_DIR:$PATH"
  cd "$FAKE_BIN_DIR/.github"
}

teardown() {
  echo "Teardown: Removing fake bin directory: $BATS_TMPDIR/fakebin" >&2
  rm -rf "$BATS_TMPDIR/fakebin" || echo "Warning: Failed to remove fake bin" >&2
}

@test "exits with error when required parameters are missing" {
  run "scripts/enforce_path/ensure_changed_files_in_dir.sh"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Missing required parameters!"* ]]
}

@test "exits with error when some required parameters are missing" {
  run "scripts/enforce_path/ensure_changed_files_in_dir.sh" "subrepo"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Missing required parameters!"* ]]
}

@test "returns 0 when all changed files are within the allowed component directory" {
  export FAKE_DIFF_OUTPUT=$'subrepo/file1.txt\nsubrepo/subdir/file2.txt'
  run "scripts/enforce_path/ensure_changed_files_in_dir.sh" "subrepo" "base" "head"
  [ "$status" -eq 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"All changes are within the allowed cicd_sandbox/subrepo/"* ]]
}

@test "exits with error when there are changed files outside the allowed directory" {
  export FAKE_DIFF_OUTPUT=$'subrepo/file1.txt\nother/file2.txt'
  run "scripts/enforce_path/ensure_changed_files_in_dir.sh" "subrepo" "base" "head"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"The following files are outside the allowed ./"* ]]
}
