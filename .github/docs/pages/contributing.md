# Contributing

<p align="center">
  <img src="assets/artifact_ml_logo.svg" width="400" alt="Artifact-ML Logo">
</p>

**Contributions are welcome!**

Artifact-ML is organized into four components: `root`, `core`, `experiment`, `torch`.

## 1) Regular Development
- **Select a component:** `core`, `experiment`, or `torch`.
- **Create a branch** based on `dev-<component>` (e.g., `feature-core/add-foo`).
- **Limit changes** to the selected component’s directory.
- **Open a PR** to `dev-<component>`.  
  *Designated reviewers will periodically merge `dev-<component>` into `main`.*
- **Incorporate reviewer feedback** by addressing comments, pushing updates, and resolving discussions before requesting re-review.

## 2) Urgent Hotfixes
- **Select a component:** `root`, `core`, `experiment`, or `torch`.
- **Create a branch** based on `main` named `hotfix-<component>/<descriptive-name>` (e.g., `hotfix-core/fix-critical-bug`).
- **Limit changes** to the selected component’s directory.
- **Open a PR** to `main` with bump type `patch` or `no-bump`
  *(See: [PR title & versioning rules](devops.md#pr-title-conventions))*.
- **Incorporate reviewer feedback** by addressing comments, pushing updates, and resolving discussions before requesting re-review.

## 3) Setup & Configuration Changes
- **Select a component:** `root`, `core`, `experiment`, or `torch`.
- **Create a branch** based on `main` named `setup-<component>/<descriptive-name>` (e.g., `setup-experiment/update-docs`).
- **Limit changes** to the selected component’s directory.
- **Open a PR** to `main` with bump type `no-bump`
*(See: [PR title & versioning rules](devops.md#pr-title-conventions))*.
- **Incorporate reviewer feedback** by addressing comments, pushing updates, and resolving discussions before requesting re-review.

## 🔗 Relevant Pages

For detailed **PR guidelines** please consult the relevant [docs](pull_requests.md).

For a detailed specification of Artifact's **DevOps processes** please consult the relevant [docs](devops_processes.md).

For a detailed specification of Artifact's **CI/CD pipelines** please consult the relevant [docs](cicd_pipelines.md).