#!/bin/bash
set -e

# Usage: .github/scripts/version_bump/get_component_tag.sh <version> [component_name]
# Returns: The full tag name based on version and optional component name

VERSION=$1
COMPONENT_NAME=$2

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
