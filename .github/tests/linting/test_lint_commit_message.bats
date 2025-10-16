#!/usr/bin/env bats
# .github/tests/linting/test_lint_commit_message.bats
# Tests lint_commit_message.sh.

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
  SCRIPT_SRC="$REPO_ROOT/.github/scripts/linting/lint_commit_message.sh"
  if [ ! -f "$SCRIPT_SRC" ]; then
    echo "ERROR: Cannot find the script at: $SCRIPT_SRC" >&2
    exit 1
  fi
  cp "$SCRIPT_SRC" "$FAKE_BIN_DIR/.github/scripts/linting/"
  chmod +x "$FAKE_BIN_DIR/.github/scripts/linting/lint_commit_message.sh"

  # --- Stub: lint_branch_name.sh (explicit examples only) ---
  cat << "EOF" > "$FAKE_BIN_DIR/.github/scripts/linting/extract_branch_info.sh"
#!/bin/bash
set -euo pipefail
# Contract: on success, print JSON with branch_type & component_name; otherwise exit 1.
# This stub validates SHAPE only (no policy): 
#   - dev-<component>                  (NO trailing "/…")
#   - <type>-<component>/<desc>        (required for non-dev)

BRANCH_NAME="${1-}"

if [ -z "${BRANCH_NAME}" ]; then
  echo "::error::No branch name provided!" >&2
  exit 1
fi

# -----------------------------
# Explicit acceptance table
# -----------------------------
# dev (no slash)
if   [ "$BRANCH_NAME" = "dev-core" ]; then
  echo '{"branch_type":"dev","component_name":"core"}'; exit 0
elif [ "$BRANCH_NAME" = "dev-experiment" ]; then
  echo '{"branch_type":"dev","component_name":"experiment"}'; exit 0
elif [ "$BRANCH_NAME" = "dev-torch" ]; then
  echo '{"branch_type":"dev","component_name":"torch"}'; exit 0
elif [ "$BRANCH_NAME" = "dev-mycomponent" ]; then
  echo '{"branch_type":"dev","component_name":"mycomponent"}'; exit 0

# hotfix (requires "/desc")
elif [ "$BRANCH_NAME" = "hotfix-core/fix-1" ]; then
  echo '{"branch_type":"hotfix","component_name":"core"}'; exit 0
elif [ "$BRANCH_NAME" = "hotfix-experiment/urgent" ]; then
  echo '{"branch_type":"hotfix","component_name":"experiment"}'; exit 0
elif [ "$BRANCH_NAME" = "hotfix-root/update-ci" ]; then
  echo '{"branch_type":"hotfix","component_name":"root"}'; exit 0
elif [ "$BRANCH_NAME" = "hotfix-mycomponent/fix-critical-bug" ]; then
  echo '{"branch_type":"hotfix","component_name":"mycomponent"}'; exit 0

# setup (requires "/desc")
elif [ "$BRANCH_NAME" = "setup-core/init" ]; then
  echo '{"branch_type":"setup","component_name":"core"}'; exit 0
elif [ "$BRANCH_NAME" = "setup-experiment/seed" ]; then
  echo '{"branch_type":"setup","component_name":"experiment"}'; exit 0
elif [ "$BRANCH_NAME" = "setup-root/bootstrap" ]; then
  echo '{"branch_type":"setup","component_name":"root"}'; exit 0
elif [ "$BRANCH_NAME" = "setup-mycomponent/initial-config" ]; then
  echo '{"branch_type":"setup","component_name":"mycomponent"}'; exit 0

# -----------------------------
# Known invalid shapes (helpful errors)
# -----------------------------
elif [[ "$BRANCH_NAME" == dev-*/?* ]]; then
  echo "::error::Invalid 'dev' branch shape: 'dev-<component>' must NOT contain '/<descriptive-name>'." >&2
  exit 1
elif [[ "$BRANCH_NAME" == feature-* && "$BRANCH_NAME" != */* ]]; then
  # Non-dev types must include '/desc'
  echo "::error::Branch name does not follow the required convention! Expected '<type>-<component>/<desc>' for non-dev types." >&2
  exit 1
elif [[ "$BRANCH_NAME" == fix-* && "$BRANCH_NAME" != */* ]]; then
  echo "::error::Branch name does not follow the required convention! Expected '<type>-<component>/<desc>' for non-dev types." >&2
  exit 1

# -----------------------------
# Everything else fails
# -----------------------------
else
  echo "::error::Branch name does not follow the required convention!" >&2
  exit 1
fi
EOF
  chmod +x "$FAKE_BIN_DIR/.github/scripts/linting/extract_branch_info.sh"

  # --- Fake git: emit FAKE_COMMIT_MSG for "git log -1 --pretty=format:%s" ---
  cat << "EOF" > "$FAKE_BIN_DIR/git"
#!/bin/bash
if [[ "$@" == *"log -1"* && "$@" == *"--pretty=format:%s"* ]]; then
  echo "${FAKE_COMMIT_MSG}"
  exit 0
fi
echo "Fake git: unhandled args: $@" >&2
exit 1
EOF
  chmod +x "$FAKE_BIN_DIR/git"

  export PATH="$FAKE_BIN_DIR:$PATH"
  cd "$FAKE_BIN_DIR"
}

teardown() {
  echo "Teardown: Removing fake bin directory: $BATS_TMPDIR/fakebin" >&2
  rm -rf "$BATS_TMPDIR/fakebin" || echo "Warning: Failed to remove fake bin" >&2
}

# ----------------------------
# Invalid branch: rejected by stub
# ----------------------------

@test "fails for invalid feature branch (feature not in defaults)" {
  export FAKE_COMMIT_MSG="Merge pull request #123 from username/feature-newcomponent"
  run ".github/scripts/linting/lint_commit_message.sh"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr")"
  # New extractor/wrapper messages
  [[ "$combined" == *"Extracted branch name: feature-newcomponent"* ]]
  [[ "$combined" == *"Failed to validate/parse branch name from commit subject."* \
     || "$combined" == *"Branch name does not follow the required convention!"* \
     || "$combined" == *"Expected '<type>-<component>/<desc>' for non-dev types."* ]]
}

@test "fails for malformed branch segment" {
  export FAKE_COMMIT_MSG="Merge pull request #77 from username/not-a-valid"
  run ".github/scripts/linting/lint_commit_message.sh"
  [ "$status" -ne 0 ]
  combined="$(printf "%s%s" "$output" "$stderr")"
  [[ "$combined" == *"Extracted branch name: not-a-valid"* ]]
  [[ "$combined" == *"Failed to validate/parse branch name from commit subject."* \
     || "$combined" == *"Branch name does not follow the required convention!"* ]]
}

# ----------------------------
# Valid dev branches
# ----------------------------

@test "returns component for dev-core" {
  export FAKE_COMMIT_MSG="Merge pull request #1 from user/dev-core"
  run ".github/scripts/linting/lint_commit_message.sh"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = "core" ]
}

@test "returns component for dev-experiment" {
  export FAKE_COMMIT_MSG="Merge pull request #2 from user/dev-experiment"
  run ".github/scripts/linting/lint_commit_message.sh"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = "experiment" ]
}

@test "returns component for dev-torch (username:branch form)" {
  export FAKE_COMMIT_MSG="Merge pull request #3 from user:dev-torch"
  run ".github/scripts/linting/lint_commit_message.sh"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = "torch" ]
}

# ----------------------------
# Valid hotfix branches
# ----------------------------

@test "returns component for hotfix-core/fix-1" {
  export FAKE_COMMIT_MSG="Merge pull request #10 from user/hotfix-core/fix-1"
  run ".github/scripts/linting/lint_commit_message.sh"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = "core" ]
}

@test "returns component for hotfix-experiment/urgent" {
  export FAKE_COMMIT_MSG="Merge pull request #11 from user/hotfix-experiment/urgent"
  run ".github/scripts/linting/lint_commit_message.sh"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = "experiment" ]
}

@test "returns component for hotfix-root/update-ci" {
  export FAKE_COMMIT_MSG="Merge pull request #12 from user/hotfix-root/update-ci"
  run ".github/scripts/linting/lint_commit_message.sh"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = "root" ]
}

# ----------------------------
# Valid setup branches
# ----------------------------

@test "returns component for setup-core/init" {
  export FAKE_COMMIT_MSG="Merge pull request #20 from user/setup-core/init"
  run ".github/scripts/linting/lint_commit_message.sh"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = "core" ]
}

@test "returns component for setup-experiment/seed" {
  export FAKE_COMMIT_MSG="Merge pull request #21 from user/setup-experiment/seed"
  run ".github/scripts/linting/lint_commit_message.sh"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = "experiment" ]
}

@test "returns component for setup-root/bootstrap" {
  export FAKE_COMMIT_MSG="Merge pull request #22 from user/setup-root/bootstrap"
  run ".github/scripts/linting/lint_commit_message.sh"
  [ "$status" -eq 0 ]
  [ "$(get_final_line "$output")" = "root" ]
}