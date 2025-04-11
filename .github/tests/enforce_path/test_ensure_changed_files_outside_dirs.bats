#!/usr/bin/env bats
# .github/tests/enforce_path/test_ensure_changed_files_outside_dirs.bats

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

@test "exits with error when required parameters are missing (no arguments)" {
  run "scripts/enforce_path/ensure_changed_files_outside_dirs.sh"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Missing required parameters!"* ]]
}

@test "exits with error when some required parameters are missing (one argument only)" {
  run "scripts/enforce_path/ensure_changed_files_outside_dirs.sh" "main"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Missing required parameters!"* ]]
}

@test "exits with error when some required parameters are missing (base_ref and head_ref only)" {
  run "scripts/enforce_path/ensure_changed_files_outside_dirs.sh" "main" "feature-branch"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"Missing required parameters!"* ]]
}

@test "returns 0 when all changed files are outside the forbidden directories" {
  export FAKE_DIFF_OUTPUT=$'other/file1.txt\nanother/file2.txt'
  run "scripts/enforce_path/ensure_changed_files_outside_dirs.sh" "main" "feature-branch" "dir1" "dir2"
  [ "$status" -eq 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"All changes are outside the specified directories:"* ]]
}

@test "exits with error when some changed files are inside the forbidden directories" {
  export FAKE_DIFF_OUTPUT=$'dir1/file1.txt\nother/file2.txt'
  run "scripts/enforce_path/ensure_changed_files_outside_dirs.sh" "main" "feature-branch" "dir1" "dir2" "dir3"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr" | tr -d '\r\n')"
  echo "DEBUG: combined=[$combined]" >&2
  [[ "$combined" == *"The following files are inside the forbidden directories:"* ]]
}
