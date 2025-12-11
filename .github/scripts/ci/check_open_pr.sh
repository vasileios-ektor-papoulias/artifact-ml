#!/usr/bin/env bash
#
# check_open_pr.sh
#
# Checks if there is an open pull request for a given branch.
#
# Usage:
#   check_open_pr.sh <branch_name>
#
# Output:
#   Prints "true" to stdout if an open PR exists for the branch.
#   Prints "false" to stdout if no open PR exists.
#
# Exit Codes:
#   0 - Success (check completed, result printed)
#   1 - Error (missing argument or API failure)
#
# Environment:
#   GH_TOKEN - Required. GitHub token for API access.
#
# Example:
#   GH_TOKEN=${{ github.token }} ./check_open_pr.sh "feature-core/my-feature"
#   # Output: "true" or "false"

set -euo pipefail

BRANCH_NAME="${1:-}"

if [[ -z "$BRANCH_NAME" ]]; then
    echo "Error: Branch name is required" >&2
    echo "Usage: check_open_pr.sh <branch_name>" >&2
    exit 1
fi

if [[ -z "${GH_TOKEN:-}" ]]; then
    echo "Error: GH_TOKEN environment variable is required" >&2
    exit 1
fi

# Query GitHub API for open PRs with this branch as head
PR_NUMBER=$(gh pr list --head "$BRANCH_NAME" --state open --json number --jq '.[0].number' 2>/dev/null || echo "")

if [[ -n "$PR_NUMBER" ]]; then
    echo "true"
else
    echo "false"
fi

