# ⚙️ artifact-core

> A unified interface for the declarative computation of diverse validation artifacts in ML experiments.

<p align="center">
  <img src="./assets/artifact_ml_logo.svg" width="400" alt="Artifact-ML Logo">
</p>

[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://artifact-ml.readthedocs.io/en/latest/artifact_core)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/github/license/vasileios-ektor-papoulias/artifact-ml)

[![CI](https://img.shields.io/github/actions/workflow/status/vasileios-ektor-papoulias/artifact-ml/ci_push_main.yml?branch=main&label=CI)](https://github.com/vasileios-ektor-papoulias/artifact-ml/actions/workflows/ci_push_main.yml)
[![Coverage](https://codecov.io/gh/vasileios-ektor-papoulias/artifact-ml/branch/main/graph/badge.svg?flag=core)](https://codecov.io/gh/vasileios-ektor-papoulias/artifact-ml/flags#core)
[![CodeFactor](https://www.codefactor.io/repository/github/vasileios-ektor-papoulias/artifact-ml/badge)](https://www.codefactor.io/repository/github/vasileios-ektor-papoulias/artifact-ml)

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=vasileios-ektor-papoulias_artifact-core&metric=alert_status&branch=main)](https://sonarcloud.io/summary/new_code?id=vasileios-ektor-papoulias_artifact-core&branch=main)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=vasileios-ektor-papoulias_artifact-core&metric=sqale_rating&branch=main)](https://sonarcloud.io/summary/new_code?id=vasileios-ektor-papoulias_artifact-core&branch=main)
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=vasileios-ektor-papoulias_artifact-core&metric=security_rating&branch=main)](https://sonarcloud.io/summary/new_code?id=vasileios-ektor-papoulias_artifact-core&branch=main)
[![Reliability Rating](https://sonarcloud.io/api/project_badges/measure?project=vasileios-ektor-papoulias_artifact-core&metric=reliability_rating&branch=main)](https://sonarcloud.io/summary/new_code?id=vasileios-ektor-papoulias_artifact-core&branch=main)

---

## 📋 Overview

`artifact-core` constitutes the foundation of [**Artifact-ML**](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main).

It provides a unified interface for the declarative computation of diverse validation artifacts in ML experiments.

Its objective is to enable reusable validation workflows by providing the tools to trigger artifacts by name---with zero adapter code.

In line with our [design philosophy](https://artifact-ml.readthedocs.io/en/latest/value_philosophy/), achieving this establishes the foundation for [Artifact-ML’s](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main) broader objective: eliminating imperative glue code in ML experiments at large.

`artifact-core` stands alongside:

- [`artifact-experiment`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-experiment): experiment orchestration and tracking extension.
- [`artifact-torch`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-torch): PyTorch integration for building reusable deep-learning workflows declaratively.

## 🚀 Installation

Clone the [**Artifact-ML monorepo**](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main) by running:

```bash
git clone https://github.com/vasileios-ektor-papoulias/artifact-ml.git
```

Install the `artifact-core` package by running:

```bash
cd artifact-ml/artifact-core

poetry install
```

## 📚 Documentation

Documentation for `artifact-core` is available at [**artifact-core docs**](https://artifact-ml.readthedocs.io/en/latest/artifact-core).

## 🤝 Contributing

Contributions are welcome!

Please consult our [**contribution guidelines document**](https://artifact-ml.readthedocs.io/en/latest/Development/contributing).

## 📄 License

This project is licensed under the [MIT License](https://img.shields.io/github/license/vasileios-ektor-papoulias/artifact-ml).
