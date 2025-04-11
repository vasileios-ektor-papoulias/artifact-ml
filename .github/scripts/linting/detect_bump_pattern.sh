#!/bin/bash
set -e

# Usage: .github/scripts/linting/detect_bump_pattern.sh "text to check"
# Returns: 0 if the text follows the bump pattern, 1 otherwise
# Outputs: The bump type (patch, minor, major) to stdout if successful

TEXT_TO_CHECK="$1"

if [ -z "$TEXT_TO_CHECK" ]; then
  echo "::error::No text provided to check!" >&2
  echo "::error::Usage: $0 \"text to check\"" >&2
  exit 1
fi

echo "Checking text: $TEXT_TO_CHECK" >&2

TEXT_LOWER=$(echo "$TEXT_TO_CHECK" | tr '[:upper:]' '[:lower:]')

if [[ "$TEXT_LOWER" =~ ^patch: ]]; then
  echo "Text follows the convention (patch)." >&2
  echo "patch"
elif [[ "$TEXT_LOWER" =~ ^minor: ]]; then
  echo "Text follows the convention (minor)." >&2
  echo "minor"
elif [[ "$TEXT_LOWER" =~ ^major: ]]; then
  echo "Text follows the convention (major)." >&2
  echo "major"
elif [[ "$TEXT_LOWER" =~ ^no-bump: ]]; then
  echo "Text follows the convention (no-bump)." >&2
  echo "no-bump"
elif [[ "$TEXT_LOWER" =~ ^patch\( ]]; then
  echo "Text follows the convention (patch with scope)." >&2
  echo "patch"
elif [[ "$TEXT_LOWER" =~ ^minor\( ]]; then
  echo "Text follows the convention (minor with scope)." >&2
  echo "minor"
elif [[ "$TEXT_LOWER" =~ ^major\( ]]; then
  echo "Text follows the convention (major with scope)." >&2
  echo "major"
elif [[ "$TEXT_LOWER" =~ ^no-bump\( ]]; then
  echo "Text follows the convention (no-bump with scope)." >&2
  echo "no-bump"
else
  echo "::error::Text does not follow the convention!" >&2
  echo "::error::Text should start with one of: 'patch:', 'minor:', 'major:', or 'no-bump:'." >&2
  echo "::error::Examples:" >&2
  echo "::error::  patch: fix a bug" >&2
  echo "::error::  minor: add a new feature" >&2
  echo "::error::  major: breaking change" >&2
  echo "::error::  no-bump: changes that don't require a version bump" >&2
  echo "::error::Note: For hotfixes, use 'patch:' prefix." >&2
  exit 1
fi

exit 0
