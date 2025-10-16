#!/bin/bash
set -euo pipefail

# Usage: .github/scripts/linting/lint_branch_name.sh <branch_name> ["<allowed_components>"] ["<allowed_branch_types>"]
# Returns: 0 if the branch name follows the convention AND is allowed, 1 otherwise.
# Stdout on success: JSON from extract_branch_info.sh, e.g. {"branch_type":"dev","component_name":"core"}
# Examples:
#   .github/scripts/linting/lint_branch_name.sh "dev-core"
#   .github/scripts/linting/lint_branch_name.sh "hotfix-core/fix" "root core experiment torch" "hotfix"
#   .github/scripts/linting/lint_branch_name.sh "setup-experiment/init" "experiment torch" "dev hotfix setup"

chmod +x .github/scripts/linting/extract_branch_info.sh

BRANCH_NAME="${1-}"
DEFAULT_COMPONENTS="root core experiment torch"
DEFAULT_TYPES="dev hotfix setup"

# Optional args (with defaults)
ALLOWED_COMPONENTS_RAW="${2-}"
ALLOWED_TYPES_RAW="${3-}"

if [ -z "${BRANCH_NAME}" ]; then
  echo "::error::Missing required parameter: <branch_name>." >&2
  echo "::error::Usage: $0 <branch_name> [\"<allowed_components>\"] [\"<allowed_branch_types>\"]" >&2
  exit 1
fi

# Apply defaults if not provided
if [ -z "${ALLOWED_COMPONENTS_RAW}" ]; then
  ALLOWED_COMPONENTS_RAW="$DEFAULT_COMPONENTS"
fi
if [ -z "${ALLOWED_TYPES_RAW}" ]; then
  ALLOWED_TYPES_RAW="$DEFAULT_TYPES"
fi

echo "Checking branch name: ${BRANCH_NAME}" >&2
echo "Allowed components: ${ALLOWED_COMPONENTS_RAW}" >&2
echo "Allowed branch types: ${ALLOWED_TYPES_RAW}" >&2

# To arrays
IFS=' ' read -r -a COMPONENTS <<< "${ALLOWED_COMPONENTS_RAW}"
IFS=' ' read -r -a TYPES_RAW <<< "${ALLOWED_TYPES_RAW}"

# Normalize types to lowercase
ALLOWED_TYPES=()
for t in "${TYPES_RAW[@]}"; do
  ALLOWED_TYPES+=( "${t,,}" )
done

VALID_BRANCH=false
TYPE_OK=false
COMP_OK=false
BRANCH_INFO_RESULT=""

# Run extractor; only treat as success if extractor succeeds
if BRANCH_INFO_RESULT=$(.github/scripts/linting/extract_branch_info.sh "${BRANCH_NAME}" 2>/dev/null); then
  COMPONENT_NAME=$(echo "${BRANCH_INFO_RESULT}" | grep -o '"component_name":"[^"]*"' | cut -d'"' -f4)
  BRANCH_TYPE=$(echo "${BRANCH_INFO_RESULT}" | grep -o '"branch_type":"[^"]*"' | cut -d'"' -f4)

  echo "Extracted branch type: ${BRANCH_TYPE}" >&2
  echo "Extracted component name: ${COMPONENT_NAME}" >&2

  # Check type allowed
  for T in "${ALLOWED_TYPES[@]}"; do
    if [ "${BRANCH_TYPE}" = "${T}" ]; then
      TYPE_OK=true
      break
    fi
  done

  # Check component allowed
  for C in "${COMPONENTS[@]}"; do
    if [ "${COMPONENT_NAME}" = "${C}" ]; then
      COMP_OK=true
      break
    fi
  done

  if [ "${TYPE_OK}" = true ] && [ "${COMP_OK}" = true ]; then
    echo "Branch follows the ${BRANCH_TYPE}-${COMPONENT_NAME} naming convention and is allowed." >&2
    VALID_BRANCH=true
  fi
fi

if [ "${VALID_BRANCH}" = false ]; then
  echo "::error::Branch name does not follow the required naming convention or is not allowed!" >&2
  echo "::error::Allowed branch types: ${ALLOWED_TYPES[*]}" >&2
  echo "::error::Allowed components: ${COMPONENTS[*]}" >&2
  echo "::error::Branch name must match one of the following shapes for the allowed types/components:" >&2

  for T in "${ALLOWED_TYPES[@]}"; do
    case "${T}" in
      dev)
        for C in "${COMPONENTS[@]}"; do
          echo "::error::  - dev-${C}" >&2
        done
        ;;
      hotfix)
        for C in "${COMPONENTS[@]}"; do
          echo "::error::  - hotfix-${C}/<descriptive-name>" >&2
        done
        ;;
      setup)
        for C in "${COMPONENTS[@]}"; do
          echo "::error::  - setup-${C}/<descriptive-name>" >&2
        done
        ;;
      *)
        echo "::error::  - (unknown type '${T}' — supported: dev, hotfix, setup)" >&2
        ;;
    esac
  done
  exit 1
fi

# Success: emit the parsed JSON to stdout
echo "${BRANCH_INFO_RESULT}"
exit 0

