# Contribution Guidelines

**Contributions are welcome!**

To contribute, please consult the guide below.

For a detailed specification of Artifact's **DevOps processes** please consult the relevant [docs](../.github/devops.md).

For a detailed specification of Artifact's **CI/CD pipelines** please consult the relevant [docs](../.github/cicd.md).

<p align="center">
  <img src="../assets/artifact_ml_logo.svg" width="400" alt="Artifact-ML Logo">
</p>


1. **For Regular Development**:
   - select a component to work on (`core`, `experiment`, `torch`),
   - create a feature branch (e.g., `feature-<component_name>/add-login`) based on the appropriate `dev-<component_name>` branch (i.e. `dev-core`, `dev-experiment`, or `dev-torch`),
   - implement your changes (only modify files within the selected component directory),
   - ensure the PR passes all CI checks
   - create a PR to `dev-<component_name>`,
   - designated reviewers will periodically open a PR from `dev-<component_name>` to `main`.

2. **For Urgent Hotfixes**:
   - select a component to work on (`root`, `core`, `experiment`, `torch`),
   - create a branch named `hotfix-<component_name>/<descriptive-name>` based on main (e.g., hotfix-core/fix-critical-bug),
   - implement your changes (only modify files within the selected component directory),
   - open a PR directly to main with bump type `patch` or `no-bump` (see the aforementioned PR title convention).

3. **For Setup and Configuration**:
   - select a component to work on (`root`, `core`, `experiment`, `torch`),
   - create a branch named `setup-<component_name>/<descriptive-name>` based on main (e.g., setup-experiment/update-docs),
   - implement your changes (only modify files within the selected component directory),
   - open a PR directly to main with bump type `no-bump` (see the aforementioned PR title convention).