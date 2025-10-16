#!/bin/bash
set -euo pipefail

# Usage: .github/scripts/linting/extract_branch_info.sh <branch_name>
# Returns: JSON with branch_type and component_name
#   e.g. {"branch_type":"dev","component_name":"mycomponent"}
# Valid shapes:
#   - dev-<component>                            (NO slash after component)
#   - <branch_type>-<component>/<descriptive>    (slash + name required for non-dev types)

BRANCH_NAME="${1-}"

if [ -z "$BRANCH_NAME" ]; then
  echo "::error::No branch name provided!" >&2
  echo "::error::Usage: $0 <branch_name>" >&2
  exit 1
fi

echo "Branch name: $BRANCH_NAME" >&2

BRANCH_TYPE=""
COMPONENT_NAME=""

# 1) Special case: dev-<component> with NO trailing slash part
if [[ "$BRANCH_NAME" =~ ^dev-([A-Za-z0-9._-]+)$ ]]; then
  BRANCH_TYPE="dev"
  COMPONENT_NAME="${BASH_REMATCH[1]}"

# 2) Explicitly fail dev-<component>/<anything> (dev must not have slash)
elif [[ "$BRANCH_NAME" =~ ^dev-([A-Za-z0-9._-]+)/ ]]; then
  echo "::error::Invalid 'dev' branch shape: 'dev-<component>' must NOT contain '/<descriptive-name>'." >&2
  echo "::error::Example: 'dev-core' (not 'dev-core/add-thing')." >&2
  exit 1

# 3) Generic shape for all other types: <branch_type>-<component>/<descriptive>
elif [[ "$BRANCH_NAME" =~ ^([A-Za-z0-9._-]+)-([A-Za-z0-9._-]+)/.+$ ]]; then
  BRANCH_TYPE="${BASH_REMATCH[1]}"
  COMPONENT_NAME="${BASH_REMATCH[2]}"

# 4) Everything else: invalid
else
  echo "::error::Branch name does not follow the required convention!" >&2
  echo "::error::Valid shapes:" >&2
  echo "::error::  - dev-<component>" >&2
  echo "::error::  - <branch_type>-<component>/<descriptive-name>  (for non-dev types)" >&2
  echo "::error::Examples:" >&2
  echo "::error::  - dev-core" >&2
  echo "::error::  - hotfix-core/fix-critical-bug" >&2
  echo "::error::  - setup-experiment/initial-config" >&2
  echo "::error::  - feature-torch/add-dataloader" >&2
  echo "::error::  - fix-core/harden-ci" >&2
  exit 1
fi

echo "{\"branch_type\":\"$BRANCH_TYPE\",\"component_name\":\"$COMPONENT_NAME\"}"
exit 0