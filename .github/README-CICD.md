# Artifact-ML CI/CD

## Overview

CI/CD for Artifact-ML relies on GitHub workflows.

The present details:

- the rules and conventions enforced by these workflows,
- the standard process for contributing to the project,
- CI/CD pipeline implementation details.

### Branch Naming Conventions

- **Development Branches**: `dev-<component_name>`
  - Examples: `dev-core`, `dev-experiment`, `dev-torch`
  - These are the component-specific development branches
  - Feature and bug fix branches are merged into these branches

- **Feature/Bug Fix Branches**: `feature/<some_name>` or similar
  - Example: `feature/add-login`
  - Used for regular development work
  - Should only modify files in one component directory
  - PRs from these branches target the corresponding `dev-<component_name>` branch

- **Hotfix Branches**: `hotfix-<component_name>/<some_other_name>`
  - Examples: `hotfix-core/fix-critical-bug`, `hotfix-experiment/fix-validation-issue`, `hotfix-torch/fix-model-loading`
  - Used for urgent fixes that need to be applied directly to main
  - Should only modify files in the specified component directory
  - PRs from these branches target the main branch

- **Setup Branches**: `setup-<component_name>/<some_other_name>`
  - Examples: `setup-core/initial-config`, `setup-experiment/update-docs`, `setup-torch/add-examples`
  - Used for initial setup or configuration changes
  - Should only modify files in the specified component directory
  - PRs from these branches target the main branch
  - Always use with "no-bump:" prefix as setup changes should not trigger version bumps

> **Note**: If your component name is "root" (changes affecting the monorepo root), you **must** use "no-bump:" prefix regardless of branch type. This is enforced by the PR title linting workflow. The 'root' component refers to the monorepo root and can only be modified by setup-root/* and hotfix-root/* branches. When working with the 'root' component, you cannot edit files in the component directories (artifact-core, artifact-experiment, artifact-torch) - changes must be made only to files outside of these directories.

### PR Title Conventions

**Only PRs to main** must follow semantic versioning conventions:

- `patch:` - For backwards-compatible bug fixes (including hotfixes)
- `minor:` - For backwards-compatible feature additions
- `major:` - For backwards-incompatible changes
- `no-bump:` - For changes that don't require a version bump

**Special rule for root component PRs**: Pull requests from branches with the "root" component (e.g., `hotfix-root/*` or `setup-root/*`) **must** use the "no-bump:" prefix. This is automatically enforced by the PR title linting workflow, which will reject any root component PR that doesn't use this prefix.

Examples of PR titles to main:
- `patch: fix login validation bug`
- `patch: fix critical security vulnerability` (for hotfixes)
- `minor: add user profile page`
- `major: redesign authentication system`
- `no-bump: update documentation`

You can also use scoped versions:
- `patch(auth): fix login validation bug`
- `patch(security): fix critical vulnerability` (for hotfixes)
- `no-bump(docs): update README`

PRs to development branches (e.g., dev-core, dev-experiment, dev-torch) do not need to follow these conventions and can have any descriptive title.

### CI/CD Pipeline Flow

1. **PR Creation and Validation**:
   - PR titles are validated against the convention (enforced by `lint_pr_title_main.yml`)
     - Must start with "patch:", "minor:", "major:", or "no-bump:"
     - PRs from root component branches must use "no-bump:" prefix
   - Branch names for PRs to main are validated to ensure they follow the naming convention (enforced by `enforce_branch_naming.yml`)
   - For PRs to dev-core, only changes to files in the artifact-core directory are allowed
   - For PRs to dev-experiment, only changes to files in the artifact-experiment directory are allowed
   - For PRs to dev-torch, only changes to files in the artifact-torch directory are allowed
   - For PRs from dev-core to main, only changes to files in the artifact-core directory are allowed
   - For PRs from dev-experiment to main, only changes to files in the artifact-experiment directory are allowed
   - For PRs from dev-torch to main, only changes to files in the artifact-torch directory are allowed
   - For PRs from hotfix-core/* branches to main, only changes to files in the artifact-core directory are allowed
   - For PRs from hotfix-experiment/* branches to main, only changes to files in the artifact-experiment directory are allowed
   - For PRs from hotfix-torch/* branches to main, only changes to files in the artifact-torch directory are allowed
   - For PRs from hotfix-root/* or setup-root/* branches to main, only changes to files outside of the component directories are allowed

2. **Merge to Main**:
   - Merge commits must follow specific conventions (enforced by lint workflows)
   - PRs must come from branches named "dev-<component_name>", "hotfix-<component_name>/<some_other_name>", or "setup-<component_name>/<some_other_name>"
   - Merge commits must include a description that starts with "patch:", "minor:", "major:", or "no-bump:"
   - For hotfixes, use "patch:" in the commit description
   - Non-merge commits are skipped by the linting workflows

3. **Automated Version Bumping**:
   - After a successful merge to main, CI workflows run to verify the code
   - If all checks pass and the bump type is not "no-bump", the component version is automatically bumped based on:
     - Component name (extracted from branch name)
     - Bump type (extracted from commit description)
   - A new git tag is created and pushed
   - If the bump type is "no-bump", the version bump process is skipped

### Contributing to the Project

To contribute to Artifact-ML, follow these steps:

1. **For Regular Development**:
   - Create a feature branch (e.g., `feature/add-login`) from the appropriate dev branch (dev-core, dev-experiment, or dev-torch)
   - Make your changes (only modify files within one component directory)
   - Create a PR to the corresponding `dev-<component_name>` branch
   - Designated reviewers will create a PR from the dev branch to main when features are ready
   - The PR to main should have a title that starts with "patch:", "minor:", "major:", or "no-bump:"
   - Use "no-bump:" for changes that don't require a version bump (e.g., documentation updates)
   - Ensure the PR passes all CI checks
   - When merged to main, the version will be automatically bumped based on the PR title (except for "no-bump:")

2. **For Urgent Hotfixes**:
   - Create a branch named `hotfix-<component_name>/<descriptive-name>` from main (e.g., hotfix-core/fix-critical-bug)
   - Make your changes (only modify files within the specified component directory)
   - Create a PR directly to main with a title that starts with "patch:" or "no-bump:"
   - Use "patch:" for fixes that require a version bump
   - Use "no-bump:" for fixes that don't require a version bump
   - Ensure the PR passes all CI checks
   - When merged with "patch:", the version will be automatically bumped as a patch
   - When merged with "no-bump:", no version bump will occur

3. **For Setup and Configuration**:
   - Create a branch named `setup-<component_name>/<descriptive-name>` from main (e.g., setup-experiment/update-docs)
   - Make your changes (only modify files within the specified component directory)
   - Create a PR directly to main with a title that starts with "no-bump:"
   - Setup branches should always use "no-bump:" as they should not trigger version bumps
   - Ensure the PR passes all CI checks
   - When merged, no version bump will occur

4. **For Monorepo Root Changes**:
   - Create a branch named `hotfix-root/<descriptive-name>` or `setup-root/<descriptive-name>` from main
   - Make your changes (only modify files outside of the artifact-core, artifact-experiment, and artifact-torch directories)
   - Create a PR directly to main with a title that starts with "no-bump:"
   - Root component changes should always use "no-bump:" as they should not trigger version bumps
   - Ensure the PR passes all CI checks
   - When merged, no version bump will occur

## Implementation

The workflows enforcing our CI/CD conventions are powered by shell scripts. These are organized into a modular structure under the `.github/scripts` directory.

All scripts are unit-tested using the [Bats](https://github.com/bats-core/bats-core) framework. Tests are organized in `.github/tests`. Their directory structure mirrors that of `.github/scripts`.

### GitHub Workflows

Our CI/CD pipeline utilizes the following workflows:

- `ci_core.yml` (workflow name: CI_CORE_ON_PUSH), `ci_experiment.yml` (workflow name: CI_EXPERIMENT_ON_PUSH), `ci_torch.yml` (workflow name: CI_TORCH_ON_PUSH) - Runs CI checks when changes are made to files in the respective component directories (artifact-core, artifact-experiment, artifact-torch) on branches other than main and their dev branches
- `ci_dev_core.yml` (workflow name: CI_DEV_CORE), `ci_dev_experiment.yml` (workflow name: CI_DEV_EXPERIMENT), `ci_dev_torch.yml` (workflow name: CI_DEV_TORCH) - Runs CI checks when changes are made to the dev-core, dev-experiment, or dev-torch branches respectively
- `ci_main.yml` (workflow name: CI_MAIN) - Runs CI checks when changes are made to the main branch
- `lint_pr_title_main.yml` (workflow name: LINT_PR_TITLE) - Ensures PR titles to main follow the convention (patch:, minor:, major:, no-bump:) and enforces that PRs from root component branches must use "no-bump:" prefix
- `lint_merge_commit_message.yml` (workflow name: LINT_MERGE_COMMIT_MESSAGE) - Verifies merge commits follow the branch naming convention (dev-<component_name>, hotfix-<component_name>/<some_other_name>, or setup-<component_name>/<some_other_name>)
- `lint_merge_commit_description.yml` (workflow name: LINT_MERGE_COMMIT_DESCRIPTION) - Checks merge commit descriptions for version bump type
- `bump_component_version.yml` (workflow name: BUMP_COMPONENT_VERSION) - Automatically bumps component versions based on commit descriptions (skips for no-bump)
- `enforce_change_dirs_dev_core.yml`, `enforce_change_dirs_dev_experiment.yml`, `enforce_change_dirs_dev_torch.yml` (workflow name: ENFORCE_CHANGE_DIRS) - Ensures PRs to the respective dev branches only modify files in their corresponding directories
- `enforce_change_dirs_main.yml` (workflow name: ENFORCE_CHANGE_DIRS) - Ensures:
  - PRs from dev-core to main only modify files in the artifact-core directory
  - PRs from dev-experiment to main only modify files in the artifact-experiment directory
  - PRs from dev-torch to main only modify files in the artifact-torch directory
  - PRs from hotfix-core/* branches to main only modify files in the artifact-core directory
  - PRs from hotfix-experiment/* branches to main only modify files in the artifact-experiment directory
  - PRs from hotfix-torch/* branches to main only modify files in the artifact-torch directory
  - PRs from hotfix-root/* or setup-root/* branches to main only modify files outside of the component directories
- `enforce_branch_naming.yml` (workflow name: ENFORCE_BRANCH_NAMING) - Ensures that branches being PR'd to main follow the naming convention: `dev-<component>`, `hotfix-<component>/*`, or `setup-<component>/*`

### Scripts

#### Linting Scripts (`.github/scripts/linting/`)

- `check_is_merge_commit.sh` - Checks if a commit is a merge commit
- `detect_bump_pattern.sh` - Detects version bump patterns in text
- `extract_branch_info.sh` - Extracts branch type and component name from a branch name, returning a JSON-formatted string
- `lint_branch_name.sh` - Checks if a branch name follows the required naming convention
- `lint_commit_description.sh` - Lints commit descriptions
- `lint_commit_message.sh` - Lints commit messages for branch naming convention
- `lint_pr_title.sh` - Lints PR titles and enforces that PRs from root component branches must use "no-bump:" prefix
- `lint_merge_commit_description.sh` - Higher-level script that only lints merge commit descriptions
- `lint_merge_commit_message.sh` - Higher-level script that only lints merge commit messages

#### Path Enforcement Scripts (`.github/scripts/enforce_path/`)

- `ensure_changed_files_in_dir.sh` - Ensures all changed files are within a specified directory
- `ensure_changed_files_outside_dirs.sh` - Ensures all changed files are outside specified directories

### Version Bumping Scripts (`.github/scripts/version_bump/`)

- `job.sh` - Main orchestration script that requires no input parameters
- `get_bump_type.sh` - Extracts the bump type from commit description
- `get_component_name.sh` - Extracts component name from merge commit message
- `get_pyproject_path.sh` - Determines the appropriate pyproject.toml path (exits with error if the required pyproject.toml doesn't exist)
- `bump_component_version.sh` - Updates version and generates tag
- `get_component_tag.sh` - Generates a tag name from version and component name
- `identify_new_version.sh` - Calculates the new version number based on semantic versioning rules
- `update_pyproject.sh` - Updates version in pyproject.toml
- `push_version_update.sh` - Handles git operations (commit, tag, push)

#### Version Bump Flow

The version bump process follows this flow:

1. **job.sh** (Main orchestration script):
   - Gets the bump type from commit description using get_bump_type.sh
   - Checks if the bump type is "no-bump" and exits early if it is
   - Gets the component name from the merge commit message using get_component_name.sh
   - Checks if the component is "root" and exits early if it is (root component changes should not trigger version bumps)
   - Gets the appropriate pyproject.toml path using get_pyproject_path.sh
     - If the component's pyproject.toml doesn't exist, the script exits with an error
   - Calls bump_component_version.sh with these parameters

2. **bump_component_version.sh**:
   - Updates the pyproject.toml file with the new version using update_pyproject.sh
   - Gets the full tag name from component name and version using get_component_tag.sh
   - Calls push_version_update.sh to handle git operations

3. **push_version_update.sh**:
   - Adds the file to the staging area
   - Commits the changes with a message
   - Creates a git tag
   - Pushes the changes and tags to the remote repository

### Tests

#### Linting Tests (`.github/tests/linting/`)

- `test_extract_branch_info.bats` - Tests for the branch info extraction script
- `test_lint_branch_name.bats` - Tests for the branch name linting script
- `test_detect_bump_pattern.bats` - Tests for the bump pattern detection script
- `test_is_merge_commit.bats` - Tests for the merge commit check script
- `test_lint_commit_description.bats` - Tests for commit description linting
- `test_lint_commit_message.bats` - Tests for commit message linting
- `test_lint_merge_commit_description.bats` - Tests for the higher-level merge commit description linting
- `test_lint_merge_commit_message.bats` - Tests for the higher-level merge commit message linting
- `test_lint_pr_title.bats` - Tests for PR title linting

#### Version Bump Tests (`.github/tests/version_bump/`)

- `test_get_bump_type.bats` - Tests for extracting bump type from commit descriptions
- `test_get_component_name.bats` - Tests for extracting component name from merge commit messages
- `test_get_component_tag.bats` - Tests for generating tag names
- `test_get_pyproject_path.bats` - Tests for determining pyproject.toml paths
- `test_identify_new_version.bats` - Tests for calculating new version numbers based on semantic versioning
- `test_update_pyproject.bats` - Tests for updating version in pyproject.toml
- `test_push_version_update.bats` - Tests for git operations
- `test_bump_component_version.bats` - Tests for the version bumping process
- `test_job.bats` - Tests for the main orchestration script

#### Test Structure
Unit-tests for the CI/CD scripts follow the pattern:

1. Sets up a fake environment with mocked dependencies.
2. Run the script under test.
3. Verify the script's behavior through assertions.
4. Clean up the test environment.

#### Running the Tests

To run the tests, use the following command (from the monorepo root):

```bash
# Run all tests
bats -r .github/tests

# Run tests for a specific directory
bats -r .github/tests/linting
bats -r .github/tests/version_bump

# Run a specific test file
bats .github/tests/linting/test_lint_pr_title.bats
```

### A Note on Execution Context

All scripts and workflows are designed to run from the repository root.

This means:

- Workflow files (`.github/workflows/*.yml`) execute scripts using paths relative to the repository root (e.g., `.github/scripts/linting/check_is_merge_commit.sh`)
- Scripts reference other scripts using paths relative to the repository root (e.g., `.github/scripts/linting/lint_commit_description.sh`)
- Test files run scripts from the repository root context

This approach follows GitHub Actions' standard execution context, where workflows run from the repository root. It makes the paths more intuitive and consistent, eliminating confusing double references to `.github` in paths.