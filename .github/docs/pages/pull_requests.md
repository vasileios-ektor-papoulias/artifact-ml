# Pull Requests

<p align="center">
  <img src="../../assets/artifact_ml_logo.svg" width="400" alt="Artifact-ML Logo">
</p>

## 📌 PR Type & Branch Rules

| Type                     | Naming convention                                   | Target branch            | Description                                               |
|---------------------------|----------------------------------------------------|--------------------------|-----------------------------------------------------------|
| **Feature / Fix**         | `feature-<component>/<name>`<br>`fix-<component>/<name>` | `dev-<component>`        | Regular development work (new features, bug fixes).        |
| **Hotfix**                | `hotfix-<component>/<name>`                         | `main`                   | Urgent fixes that go directly to `main`.                   |
| **Setup / Configuration** | `setup-<component>/<name>`                          | `main` | Project setup or configuration changes.                   |


---

## 🧭 Component

Indicate which component this PR affects

- root
- core
- experiment
- torch

---

## 🏷️ Version Bump Type *(for PRs to main)*

Prefix your PR title accordingly (e.g., "patch: fix bug")
- `patch` — backwards-compatible bug fix  
- `minor` — backwards-compatible feature addition  
- `major` — breaking change  
- `no-bump` — no version bump required

> ⚠️ Root PRs must use `no-bump` (enforced).
> ⚠️ Hotfix PRs must use `patch` (convention).  
> ⚠️ Setup PRs must use `no-bump` (convention).


## 🧪 Description

Please describe the change clearly using the structure below:

- **What**: What does this PR do? (e.g., high-level description of the feature, bug fix, or refactor)  
- **Why**: Why is this change needed? (e.g., motivation, problem solved, value added)  
- **How**: How was it done? (e.g., technical approach, major edits, key decisions)  
- **Impact**: What areas are affected? (e.g., components touched, breaking changes, dependencies)


## ✅ Checklist

- My branch name follows the required naming convention.  
- All CI checks pass.  
- I’ve addressed reviewer feedback.  
- The PR title uses the correct **bump type** prefix (for PRs to main).  
- The changes are limited to the declared component.  
- Documentation or tests were updated if needed.

## 🔗 Relevant Pages

For detailed **contribution guidelines** please consult the relevant [docs](contributing.md).

For a detailed specification of Artifact's **DevOps processes** please consult the relevant [docs](devops_processes.md).

For a detailed specification of Artifact's **CI/CD pipelines** please consult the relevant [docs](cicd_pipelines.md).