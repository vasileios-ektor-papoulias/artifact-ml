#!/bin/bash
set -euo pipefail

# Purpose:
#   Extract component name and version from a Git tag or use manual input.
#
# Usage:
#   .github/scripts/publishing/extract_component_from_tag.sh <event_name> [tag_or_component] [manual_component]
#
# Accepts:
#   <event_name>        GitHub event name (workflow_dispatch or push)
#   [tag_or_component]  Git tag name (e.g., artifact-core-v1.0.0) or manual component
#   [manual_component]  Component name when manually triggered (core, experiment, torch)
#
# Stdout on success:
#   JSON object with component and version:
#     {"component":"core","version":"1.0.0"}
#   or for manual:
#     {"component":"core","version":"manual"}
#
# Exit codes:
#   0 — success
#   1 — validation failure
#
# Behaviour:
#   - For workflow_dispatch: uses manual component input
#   - For tag-based (push): parses tag to extract component and version
#     Tag format: artifact-<component>-v<version>
#     Example: artifact-core-v1.0.0 → component=core, version=1.0.0
#
# Examples:
#   .github/scripts/publishing/extract_component_from_tag.sh workflow_dispatch "" core
#     --> {"component":"core","version":"manual"}
#
#   .github/scripts/publishing/extract_component_from_tag.sh push artifact-core-v1.0.0 ""
#     --> {"component":"core","version":"1.0.0"}

EVENT_NAME="${1:-}"
TAG_OR_COMPONENT="${2:-}"
MANUAL_COMPONENT="${3:-}"

if [ -z "$EVENT_NAME" ]; then
  echo "::error::Missing required parameter: event_name" >&2
  echo "::error::Usage: $0 <event_name> [tag_or_component] [manual_component]" >&2
  exit 1
fi

if [ "$EVENT_NAME" = "workflow_dispatch" ]; then
  # Manual trigger
  if [ -z "$MANUAL_COMPONENT" ]; then
    echo "::error::Manual component required for workflow_dispatch" >&2
    exit 1
  fi
  
  COMPONENT="$MANUAL_COMPONENT"
  VERSION="manual"
  echo "Manual trigger detected: component=$COMPONENT" >&2
else
  # Tag-based trigger
  if [ -z "$TAG_OR_COMPONENT" ]; then
    echo "::error::Tag name required for tag-based trigger" >&2
    exit 1
  fi
  
  TAG="$TAG_OR_COMPONENT"
  
  # Validate tag format first
  if ! [[ "$TAG" =~ ^artifact-[a-z]+-v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "::error::Invalid tag format: $TAG" >&2
    echo "::error::Expected format: artifact-<component>-v<version>" >&2
    echo "::error::Example: artifact-core-v1.0.0" >&2
    exit 1
  fi
  
  # Extract component: artifact-core-v1.0.0 -> core
  COMPONENT=$(echo "$TAG" | sed 's/artifact-\([^-]*\)-.*/\1/')
  
  # Extract version: artifact-core-v1.0.0 -> 1.0.0
  VERSION=$(echo "$TAG" | sed 's/.*-v\(.*\)/\1/')
  
  echo "Tag-based trigger detected: tag=$TAG, component=$COMPONENT, version=$VERSION" >&2
fi

# Output as JSON
echo "{\"component\":\"$COMPONENT\",\"version\":\"$VERSION\"}"
exit 0

