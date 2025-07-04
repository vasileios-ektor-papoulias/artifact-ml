Contributions are welcome!

### For Regular Development

1. Create a feature branch (e.g., `feature/add-login`) from the appropriate dev branch (`dev-core`, `dev-experiment`, or `dev-torch`).
2. Make your changes (only modify files within one component directory).
3. Create a PR to the corresponding dev branch.
4. Designated reviewers will handle merging dev branches to main with appropriate version bump prefixes.

### For Urgent Hotfixes

1. Create a branch named `hotfix-<component_name>/<descriptive-name>` from main (e.g., `hotfix-core/fix-critical-bug`).
2. Make your changes (only modify files within the specified component directory).
3. Create a PR directly to main with a title that starts with "patch:" or "no-bump:".

### For Setup and Configuration

1. Create a branch named `setup-<component_name>/<descriptive-name>` from main (e.g., `setup-experiment/update-docs`).
2. Make your changes (only modify files within the specified component directory).
3. Create a PR directly to main with a title that starts with "no-bump:".

### For Monorepo Root Changes

1. Create a branch named `hotfix-root/<descriptive-name>` or `setup-root/<descriptive-name>` from main.
2. Make your changes (only modify files outside of the artifact-core, artifact-experiment, and artifact-torch directories).
3. Create a PR directly to main with a title that starts with "no-bump:".

For more detailed information about our CI/CD pipeline and contribution guidelines, please consult the relevant [docs](.github/README-CICD.md).