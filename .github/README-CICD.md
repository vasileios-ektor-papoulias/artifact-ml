# Artifact-ML CI/CD

The present constitues a detailed exposition to the project's dev-ops processes and CI/CD pipelines.

<p align="center">
  <img src="./assets/artifact_ml_logo.svg" width="400" alt="Artifact-ML Logo">
</p>


## Dev-Ops Processes

### Repository Structure and Project Components
Artifact-ML is comprised of three single-purpose subrepos---with independent versioning and release cycles---gathered under a single monorepo. These are:
- `artifact-core`,
- `artifact-experiment`,
- `artifact-torch`.

The project is correspondingly partitioned in the following *components* (providing a way to refer to designated project subdirectories):
- `root` (files in the monorepo root, outside of all subrepo directories),
- `core` (files in the `artifact-core` subrepo),
- `experiment` (files in the `artifact-experiment` subrepo),
- `torch` (files in the `artifact-torch` subrepo).

- The `root` component can only be modified by merging `setup-root/*`/ `hotfix-root/*` directly into main (via PR).
- The subrepo (`core`, `experiment`, `torch`) components can be modified by:
   - merging `setup-<component_name>/*`/ `hotfix-<component_name>/*` directly into main (via PR),
   - merging feautre/ fix branches into `dev-<component_name>` (via PR) and awaiting its periodic merge into main. 

### Branches

- **main**: `dev-<component_name>`
   - Role: The most recent stable release of Artifact-ML.
   - Update:
      - Updated by periodically merging in `dev` branches---resulting in new version releases.
      - Updated by merging in hotfix branches through pull request---resulting in new version releases.
      - Updated by merging in setup branches through pull request---not resulting in new version releases.

- **Development Branches**: `dev-<component_name>`
   - Role: Component-specific development branches used as buffers for recent changes.
   - Update: Updated by merging in feature/ fix branches through pull request. 
   - Examples: `dev-core`, `dev-experiment`, `dev-torch`.

- **Feature/Bug Fix Branches**: `feature/<some_name>` or similar
   - Role: Used for regular development work.
   - Update: Updated by direct pushes.
   - Restrictions: Should only modify files in one component directory (enforced when opening a PR to a given `dev` branch).
   - Example: `feature/add-login`

- **Hotfix Branches**: `hotfix-<component_name>/<some_other_name>`
   - Role: Used for urgent fixes that need to be applied directly to `main`.
   - Update: Updated by direct pushes.
   - Restrictions: Should only modify files in one component directory (enforced when opening a PR to main).
   - Examples: `hotfix-core/fix-critical-bug`, `hotfix-experiment/fix-validation-issue`, `hotfix-torch/fix-model-loading`

- **Setup Branches**: `setup-<component_name>/<some_other_name>`
   - Role: Used for initial setup or configuration changes
   - Update: Updated by direct pushes.
   - Restrictions: 
      - Always use with `no-bump` bump type (as setup changes should not trigger version bumps).
      - Should only modify files in the specified component directory (enforced when opening a PR to main).
   - Examples: `setup-core/initial-config`, `setup-experiment/update-docs`, `setup-torch/add-examples`

### Versioning and PRs to `main`
In line with semantic versioniing, we adopt the following version bump types (bump types for short):

- `patch`: For backwards-compatible bug fixes (including hotfixes)
- `minor`: For backwards-compatible feature additions
- `major`: For backwards-incompatible changes
- `no-bump`: For changes that don't require a version bump

Version bumps occur automatically (via desginated github workflows). To achieve this, PRs targetting `main` must follow a naming convention:

their titles should be prefixed via `<bump_type>:`, (or `<bump_type>(scope):`).

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

Pull requests related to the `root` component (e.g. from `hotfix-root/*` or `setup-root/*`) **must** use the `no-bump` prefix (this is enforced by the relevant workflows).

Periodically, designated contributors open PRs from component `dev` branches to `main`---resulting in associatd version bumps according to the above.

### Contribution Guidelines

To contribute to Artifact-ML, follow these steps:

1. **For Regular Development**:
   - select a component to work on (`core`, `experiment`, `torch`),
   - create a feature branch (e.g., `feature/add-login`) from the appropriate `dev-<component_name>` branch (i.e. `dev-core`, `dev-experiment`, or `dev-torch`),
   - implement your changes (only modify files within the selected component directory),
   - ensure the PR passes all CI checks
   - create a PR to `dev-<component_name>`,
   - designated reviewers will periodically create a PR from `dev-<component_name>` to `main`.

2. **For Urgent Hotfixes**:
   - select a component to work on (`root`, `core`, `experiment`, `torch`),
   - create a branch named `hotfix-<component_name>/<descriptive-name>` from main (e.g., hotfix-core/fix-critical-bug),
   - implement your changes (only modify files within the selected component directory),
   - create a PR directly to main with bump type `patch` or `no-bump` (see the aforementioned PR title convention)
   - ensure the PR passes all CI checks.

3. **For Setup and Configuration**:
   - select a component to work on (`root`, `core`, `experiment`, `torch`),
   - create a branch named `setup-<component_name>/<descriptive-name>` from main (e.g., setup-experiment/update-docs),
   - implement your changes (only modify files within the selected component directory),
   - create a PR directly to main with bump type `no-bump` (see the aforementioned PR title convention),
   - ensure the PR passes all CI checks.

## CI/CD Pipeline

<p align="center">
  <img src="./assets/github_actions.png" width="500" alt="GitHub Actions Logo">
</p>

CI/CD for Artifact-ML relies on GitHub actions.

The github actions workflows powering our CI/CD pipeline delegate to shell scripts.

The latter are organized under the `.github/scripts` directory.

All scripts are unit-tested using the [Bats](https://github.com/bats-core/bats-core) framework.

Tests are organized in `.github/tests`. Their directory structure mirrors that of `.github/scripts`.

### GitHub Actions Workflows

Our CI/CD pipeline utilizes the following workflows:

#### CI Checks (on push)

- `ci_core.yml` (workflow name: CI_CORE_ON_PUSH): runs CI checks when changes are pushed to branches other than `main` and `dev-core` involving files in the `core` component directories,
- `ci_experiment.yml` (workflow name: CI_EXPERIMENT_ON_PUSH): runs CI checks when changes are pushed to branches other than `main` and `dev-experiment` involving files in the `experiment` component directories,
- `ci_torch.yml` (workflow name: CI_TORCH_ON_PUSH): runs CI checks when changes are pushed to branches other than `main` and `dev-torch` involving files in the `torch` component directories,
- `ci_dev_core.yml` (workflow name: CI_DEV_CORE): runs CI checks when changes are pushed to `dev-core` (always merges of `feature`/ `fix` branches through pull request),
- `ci_dev_experiment.yml` (workflow name: CI_DEV_EXPERIMENT): runs CI checks when changes are pushed to `dev-experiment` (always merges of `feature`/ `fix` branches through pull request),
 - `ci_dev_torch.yml` (workflow name: CI_DEV_TORCH): runs CI checks when changes are pushed to `dev-torch` (always merges of `feature`/ `fix` branches through pull request),
- `ci_main.yml` (workflow name: CI_MAIN): runs CI checks when changes are pushed to `main` (always merges of `dev`/ `hotfix`/ `setup` branches through pull request).

#### PR Metadata Validation
- `enforce_source_branch_naming_main.yml` (workflow name: ENFORCE_SOURCE_BRANCH_NAMING): ensures that branches being PR'd to `main` follow the naming convention: `dev-<component>`, `hotfix-<component>/*`, or `setup-<component>/*`
- `enforce_source_branch_naming_dev_core.yml` (workflow name: ENFORCE_SOURCE_BRANCH_NAMING): ensures that branches being PR'd to `dev-core` follow the naming convention: `feature-core/<descriptive_name>`, `fix-core/<descriptive_name>`,
- `enforce_source_branch_naming_dev_experiment.yml` (workflow name: ENFORCE_SOURCE_BRANCH_NAMING): ensures that branches being PR'd to `dev-experiment` follow the naming convention: `feature-experiment/<descriptive_name>`, `fix-experiment/<descriptive_name>`,
- `enforce_source_branch_naming_dev_torch.yml` (workflow name: ENFORCE_SOURCE_BRANCH_NAMING): ensures that branches being PR'd to `dev-torch` follow the naming convention: `feature-torch/<descriptive_name>`, `fix-torch/<descriptive_name>`,
- `lint_pr_title_main.yml` (workflow name: LINT_PR_TITLE): ensures PR titles to `main` follow the appropriate semantic versioning prefix convention (see *Versioning and PRs to `main`*),
- `enforce_change_dirs_main.yml` (workflow name: ENFORCE_CHANGE_DIRS) - Ensures:
  - PRs from `dev-core` to `main` only modify files in the `artifact-core` directory
  - PRs from `dev-experiment` to `main` only modify files in the `artifact-experiment` directory
  - PRs from `dev-torch` to `main`only modify files in the `artifact-torch` directory
  - PRs from `hotfix-core/*` branches to `main` only modify files in the `artifact-core` directory
  - PRs from `hotfix-experiment/*` branches to `main` only modify files in the `artifact-experiment` directory
  - PRs from `hotfix-torch/*` branches to `main` only modify files in the `artifact-torch` directory
  - PRs from `hotfix-root/*` or `setup-root/*` branches to `main` only modify files outside of the subrepo component directories
- `enforce_change_dirs_dev_core.yml` (workflow name: ENFORCE_CHANGE_DIRS): ensures PRs to `dev-core` only modify files in their corresponding directories,
`enforce_change_dirs_dev_experiment.yml` (workflow name: ENFORCE_CHANGE_DIRS): ensures PRs to `dev-experiment` only modify files in their corresponding directories,,
- `enforce_change_dirs_dev_torch.yml` (workflow name: ENFORCE_CHANGE_DIRS): ensures PRs to `dev-torch` only modify files in their corresponding directories,

#### Automatic Version Management (on merge commit push to `main`)
- `lint_merge_commit_message.yml` (workflow name: LINT_MERGE_COMMIT_MESSAGE): validates the message carried by a merge commit pushed to `main`---asserts that the message is of the form "Merge pull request #<`PR_number`> from <`username`>/<`branch-name`>" (or "...<`username`>:<`branch-name`>") where `<branch_name>` is one of the appropriate source branches i.e. `dev-<component_name>`, `hotfix-<component_name>`/`<some_other_name>`, or `setup-<component_name>/<some_other_name>`.
- `lint_merge_commit_description.yml` (workflow name: LINT_MERGE_COMMIT_DESCRIPTION): validates the description carried by a merge commit pushed to `main`---asserts that the description is a valid PR title according to the appropriate semantic versioning prefix convention (see *Versioning and PRs to `main`*),
- `bump_component_version.yml` (workflow name: BUMP_COMPONENT_VERSION): bumps the relevant component version when a merge commit is pushed to `main`---the commit description and message are parsed to identify the relevant component and bump type, the relevant pyproject.toml file is updated and this change is pushed along with a git tag annotating the version change.

### Scripts

#### Execution Context

All scripts are designed to run from the repository root.

This means:

- Workflow files (`.github/workflows/*.yml`) execute scripts using paths relative to the repository root (e.g., `.github/scripts/linting/check_is_merge_commit.sh`)
- Scripts reference other scripts using paths relative to the repository root (e.g., `.github/scripts/linting/lint_commit_description.sh`)
- Test files run scripts from the repository root context

This approach follows GitHub Actions' standard execution context, where workflows run from the repository root. It makes the paths more intuitive and consistent, eliminating confusing double references to `.github` in paths.

#### Linting Scripts (`.github/scripts/linting/`)

- `check_is_merge_commit.sh`:
  - **Given:** the currently checked-out commit (typically `$GITHUB_SHA`/`HEAD`).
  - **Does:** counts parent commits; if >1, it’s a merge commit. Prints the parent count to stdout.
  - **Outcome:** exits `0` for merge commits (multi-parent), `1` otherwise.

- `detect_bump_pattern.sh`:
  - **Given:** a text string (e.g., PR title or commit body).
  - **Does:** lowercases the text and checks if it **starts with** a `bump_type` prefix i.e. `patch:`, `minor:`, `major:`, `no-bump:` or their scoped counterparts e.g. `patch(scope):`.
  - **Outcome:** prints the bump type (`patch` | `minor` | `major` | `no-bump`) to stdout; exits `1` if no valid prefix.

- `extract_branch_info.sh`:
  - **Given:** a branch name following the repository’s branch-naming convention.
  - **Does:** validates the **shape** and parses `branch_type` and `component_name`. Rules:
    - `dev-<component>` *(no trailing `/…` allowed)*
    - `<branch_type>-<component>/<descriptive-name>` for **non-dev** types
  - **Outcome:** prints JSON `{"branch_type":"…","component_name":"…"}` to stdout on success; exits `1` if the branch name doesn’t follow one of the valid shapes.
  - **Examples:**
    - `dev-core` --> `{"branch_type":"dev","component_name":"core"}`
    - `dev-experiment` --> `{"branch_type":"dev","component_name":"experiment"}`
    - `dev-torch` --> `{"branch_type":"dev","component_name":"torch"}`
    - `hotfix-core/fix-ci` --> `{"branch_type":"hotfix","component_name":"core"}`
    - `hotfix-torch/patch-loader-crash` --> `{"branch_type":"hotfix","component_name":"torch"}`
    - `setup-core/seed` --> `{"branch_type":"setup","component_name":"core"}`
    - `setup-experiment/init-config` --> `{"branch_type":"setup","component_name":"experiment"}`
    - `feature-torch/add-dataloader` --> `{"branch_type":"feature","component_name":"torch"}`
    - `feature-core/improve-logging` --> `{"branch_type":"feature","component_name":"core"}`
    - `fix-core/harden-ci` --> `{"branch_type":"fix","component_name":"core"}`
    - `fix-experiment/typo-in-docs` --> `{"branch_type":"fix","component_name":"experiment"}`

- `lint_branch_name.sh`:
  - **Given:** `<branch_name>` and optional space-separated lists:
    - **`<ALLOWED_COMPONENTS>`** (default: `root core experiment torch`)
    - **`<ALLOWED_BRANCH_TYPES>`** (default: `dev hotfix setup`)
  - **Does:**
    1) Calls `extract_branch_info.sh` to **validate the branch shape** and parse `branch_type` + `component_name`. Shape rules:
       - `dev-<component>` *(no trailing `/…` allowed)*
       - `<branch_type>-<component>/<descriptive-name>` for **non-dev** types (e.g., `hotfix`, `setup`; plus any others your extractor supports)
    2) Verifies **`branch_type ∈ ALLOWED_BRANCH_TYPES`** and **`component_name ∈ ALLOWED_COMPONENTS`**.
  - **Outcome:**
    - **Success (`exit 0`)** → prints the parsed JSON to **stdout** (e.g. `{"branch_type":"dev","component_name":"core"}`)
    - **Failure (`exit 1`)** → prints guidance (allowed components/types and example shapes) to **stderr**.

- `lint_pr_title.sh`:
   - **Given:** `"PR Title"` and optionally `[branch_name]`.
   - **Does:** enforces that the title starts with a `bump_type` prefix (`patch:`, `minor:`, `major:`, `no-bump:` or their scoped counterparts e.g. `patch(scope):`). If a `branch_name` is provided and its component parses to `root`, then only `no-bump:` is allowed.
   - **Outcome:** prints the `bump_type` to stdout on success; exits `1` with a clear message if the prefix is missing/invalid or the root rule is violated.

- `lint_commit_description.sh`:
  - **Given:** the **body/description** of the last commit (merge commit in typical PR merges).
  - **Does:** ensures the description begins with a semantic prefix by passing it to `detect_bump_pattern.sh`.
  - **Outcome:** prints the resolved bump type to stdout (`patch` | `minor` | `major` | `no-bump`) and exits `0`; if empty or missing the prefix, prints errors and exits `1`.

- `lint_commit_message.sh`:
  - **Given:** the **subject** of the last commit (expected GitHub merge subject like `Merge pull request #123 from user/branch` or `... user:branch`).
  - **Does:** extracts the `branch` from the subject and validates its naming via `extract_branch_info.sh`.
  - **Outcome:** prints the `component_name` to stdout on success; exits `1` if the subject isn’t a merge format or the branch naming is invalid.

- `lint_merge_commit_description.sh`:
  - **Given:** current commit context (CI).
  - **Does:** confirms the commit is a **merge commit** (`check_is_merge_commit.sh`), then validates the **merge commit description** by invoking `lint_commit_description.sh`.
  - **Outcome:** prints `bump_type` to stdout on success; exits `1` if not a merge commit or the description/prefix validation fails.

- `lint_merge_commit_message.sh`:
  - **Given:** current commit context (CI).
  - **Does:** verifies that the current commit **is a merge commit** (`check_is_merge_commit.sh`), then validates the **merge commit subject** by invoking `lint_commit_message.sh`.
  - **Outcome:** prints the parsed **component_name** to stdout on success; exits `1` if the commit isn’t a merge or if the subject validation fails.



#### Path Enforcement Scripts (`.github/scripts/enforce_path/`)

- `ensure_changed_files_in_dir.sh`:
  - **Given:** `<component_dir>` (repo-root prefix, e.g., `artifact-core`) and `<base_ref>` (e.g., `main`).
  - **Does:** fetches `origin/<base_ref>`, computes `merge-base(origin/<base_ref>, HEAD)`, and diffs `MB..HEAD`; then verifies every changed path **starts with** `<component_dir>/`.
  - **Outcome:** exits `0` if all changed files are under `<component_dir>/`; otherwise exits `1` and lists the offending paths.

- `ensure_changed_files_outside_dirs.sh`:
  - **Given:** `<base_ref>` and one or more `<dir>` prefixes (repo-root, e.g., `docs`, `packages/app`).
  - **Does:** fetches `origin/<base_ref>`, computes `merge-base(origin/<base_ref>, HEAD)`, diffs `MB..HEAD`; then checks that **no** changed path starts with any forbidden `<dir>/` (regex-escaped, trailing slash normalized).
  - **Outcome:** exits `0` if all changes are **outside** the listed directories; otherwise exits `1` and prints the paths that violate the rule.


#### Version Bumping Scripts (`.github/scripts/version_bump/`)


- `get_bump_type.sh`:
  - **Given:** the current commit context (typically the PR merge commit).
  - **Does:** reads the **commit description/body** of the current commit, passes it to `detect_bump_pattern.sh`, and validates that it starts with `patch:` / `minor:` / `major:` / `no-bump:` (or their scoped counterparts e.g. `patch(scope):`).
  - **Outcome:** prints the resolved bump type (`patch` | `minor` | `major` | `no-bump`) to stdout; exits `1` if the description is empty or lacks a valid prefix.

- `get_component_name.sh`:
  - **Given:** the current commit context (expected to be a GitHub **merge commit**).
  - **Does:** parses the **commit subject** (e.g., `Merge pull request #123 from user/branch` or ``Merge pull request #123 from user:branch`), extracts the `branch` portion, then runs `extract_branch_info.sh` to validate branch naming and access the component name (`dev|hotfix|setup`).
  - **Outcome:** prints the **component name** (e.g., `artifact-core`) to stdout; exits `1` if the commit isn’t a merge or the branch naming is invalid.

- `identify_new_version.sh`:
  - **Given:** `<current_version>` (e.g., `1.2.3`) and `<bump_type>` (`patch|minor|major`).
  - **Does:** validates `X.Y.Z` format, splits into `MAJOR.MINOR.PATCH`, and increments per the bump:  
    `patch` → `PATCH+1`; `minor` → `MINOR+1` & `PATCH=0`; `major` → `MAJOR+1` & `MINOR=0` & `PATCH=0`.
  - **Outcome:** prints the **new version** (e.g., `1.3.0`) to stdout; exits `1` on invalid inputs.

- `get_pyproject_path.sh`:
  - **Given:** a **component name** (e.g., `artifact-core`).
  - **Does:** resolves the expected `pyproject.toml` location for that component (e.g., `artifact-core/pyproject.toml`); verifies the file exists.
  - **Outcome:** prints the absolute or repo-relative path to `pyproject.toml` to stdout; exits `1` with an error if it cannot find the required file.

- `update_pyproject.sh`:
  - **Given:** `<pyproject_path>` and `<new_version>`.
  - **Does:** updates the `version = "X.Y.Z"` field inside the given `pyproject.toml` (using a safe in-place edit), preserving file structure and other metadata.
  - **Outcome:** prints the **new version** to stdout (confirmation value) and exits `0`; exits `1` if the file is missing or the version field cannot be updated.

- `get_component_tag.sh`:
  - **Given:** `<component_name>` and `<version>` (e.g., `artifact-core` and `1.3.0`).
  - **Does:** formats a tag string according to your convention (e.g., `artifact-core-v1.3.0`).
  - **Outcome:** prints the **tag name** to stdout; exits `1` if inputs are empty or malformed.

- `push_version_update.sh`:
  - **Given:** the modified repo state, `<tag_name>`, and commit message context.
  - **Does:** stages changes (e.g., `pyproject.toml`), creates a commit, creates/updates the Git tag, and pushes commit + tag to the remote (typically `origin`). Can be gated by CI permissions on forks.
  - **Outcome:** prints a short summary (commit and tag) to stderr/stdout and exits `0`; exits `1` on any git error (e.g., auth, non-fast-forward, missing remote).

- `bump_component_version.sh`:
  - **Given:** `<bump_type>`, `<component_name>`, and optionally an explicit `<pyproject_path>`.
  - **Does:** resolves the `pyproject.toml` (via `get_pyproject_path.sh` if needed), reads the **current version**, computes the **new version** (`identify_new_version.sh`), updates the file (`update_pyproject.sh`), computes the **tag** (`get_component_tag.sh`), and pushes (`push_version_update.sh`).
  - **Outcome:** prints the **new version** and **tag** to stdout (or logs), exits `0` on success; exits `1` if any step fails (resolve, update, tag, or push).

- `job.sh`:
  - **Given:** CI context on a PR merge (or equivalent), with all helper scripts available.
  - **Does:** extracts **bump type** from the merge **description** (`get_bump_type.sh`), extracts **component name** from the merge **subject** (`get_component_name.sh`), derives/locates the component’s `pyproject.toml` (`get_pyproject_path.sh`), and invokes `bump_component_version.sh` to perform the version bump and push a version tag.
  - **Outcome:** performs an end-to-end automated version bump for the component implicated by the PR; exits `0` on success and `1` with actionable errors if inputs or validations fail.


### Tests

Unit-tests for the CI/CD scripts follow the pattern:

1. set up a fake environment with mocked dependencies,
2. run the script under consideration,
3. assert correctness,
4. clean up the test environment.

#### Test Execution

To execute the tests, use the following command (from the monorepo root):

```bash
# Run all tests
bats -r .github/tests

# Run tests for a specific directory
bats -r .github/tests/linting
bats -r .github/tests/version_bump

# Run a specific test file
bats .github/tests/linting/test_lint_pr_title.bats
```