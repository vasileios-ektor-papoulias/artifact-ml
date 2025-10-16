#!/bin/bash
set -euo pipefail

# Purpose:
#   Generate a tag string from a version and optional component name.
#   Used by release/version-bump steps to produce consistent Git tags.
#
# Usage:
#   .github/scripts/version_bump/get_component_tag.sh <version> [component_name]
#
# Accepts:
#   <version>         (required) version string (e.g., 1.2.3)
#   [component_name]  (optional) component identifier to prefix the tag
#
# Stdout on success:
#   The computed tag:
#     • If component_name omitted →  v<version>           (e.g., v1.2.3)
#     • If component_name given  →  <component>-v<version> (e.g., core-v1.2.3)
#
# Stderr on failure:
#   Usage line explaining required arguments.
#
# Exit codes:
#   0 — tag computed and printed to stdout
#   1 — missing <version> argument
#
# Behaviour:
#   - Validates that <version> is provided.
#   - If component_name is provided, prefixes it to the tag as "<component>-v<version>";
#     otherwise emits "v<version>".
#
# Notes:
#   - This script does not validate semantic version format; callers should ensure <version> is valid.
#   - The component name is used verbatim (no normalization).
#
# Examples:
#   .github/scripts/version_bump/get_component_tag.sh 1.4.0
#     --> v1.4.0
#   .github/scripts/version_bump/get_component_tag.sh 2.0.1 experiment
#     --> experiment-v2.0.1


VERSION="${1-}"
COMPONENT_NAME="${2-}"

if [[ -z "$VERSION" ]]; then
    echo "Usage: $0 <version> [component_name]" >&2
    exit 1
fi

if [[ -z "$COMPONENT_NAME" ]]; then
    TAG_NAME="v${VERSION}"
else
    TAG_NAME="${COMPONENT_NAME}-v${VERSION}"
fi

echo "$TAG_NAME"
