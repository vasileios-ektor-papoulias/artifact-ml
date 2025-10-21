# ⚙️ artifact-experiment

> Experiment orchestration and tracking for Artifact-ML.

<p align="center">
  <img src="./assets/artifact_ml_logo.svg" width="400" alt="Artifact-ML Logo">
</p>

[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://artifact-ml.readthedocs.io/en/latest/artifact_experiment)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/github/license/vasileios-ektor-papoulias/artifact-ml)

[![CI](https://img.shields.io/github/actions/workflow/status/vasileios-ektor-papoulias/artifact-ml/ci_push_main.yml?branch=main&label=CI)](https://github.com/vasileios-ektor-papoulias/artifact-ml/actions/workflows/ci_push_main.yml)
[![Coverage](https://codecov.io/gh/vasileios-ektor-papoulias/artifact-ml/branch/main/graph/badge.svg?flag=experiment)](https://codecov.io/gh/vasileios-ektor-papoulias/artifact-ml/flags#experiment)
[![CodeFactor](https://www.codefactor.io/repository/github/vasileios-ektor-papoulias/artifact-ml/badge)](https://www.codefactor.io/repository/github/vasileios-ektor-papoulias/artifact-ml)

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=vasileios-ektor-papoulias_artifact-experiment&metric=alert_status&branch=main)](https://sonarcloud.io/summary/new_code?id=vasileios-ektor-papoulias_artifact-experiment&branch=main)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=vasileios-ektor-papoulias_artifact-experiment&metric=sqale_rating&branch=main)](https://sonarcloud.io/summary/new_code?id=vasileios-ektor-papoulias_artifact-experiment&branch=main)
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=vasileios-ektor-papoulias_artifact-experiment&metric=security_rating&branch=main)](https://sonarcloud.io/summary/new_code?id=vasileios-ektor-papoulias_artifact-experiment&branch=main)
[![Reliability Rating](https://sonarcloud.io/api/project_badges/measure?project=vasileios-ektor-papoulias_artifact-experiment&metric=reliability_rating&branch=main)](https://sonarcloud.io/summary/new_code?id=vasileios-ektor-papoulias_artifact-experiment&branch=main)

---

## 📋 Overview

`artifact-experiment` constitutes the experiment orchestration and tracking extension to Artifact-ML.

It bridges the gap between validation computation and experiment tracking through a core **validation plan** abstraction responsible for the execution and tracking of artifact collections.

It stands alongside:

- [`artifact-core`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-core): Framework foundation providing a flexible uniform interface for the computation of validation artifacts.
- [`artifact-torch`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-torch): PyTorch integration for building reusable deep-learning workflows.

## 🚀 Installation

Clone the [**Artifact-ML monorepo**](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main) by running:

```bash
git clone https://github.com/vasileios-ektor-papoulias/artifact-ml.git
```

Install the `artifact-experiment` package by running:
```bash
cd artifact-ml/artifact-experiment
poetry install
```

## 📚 Documentation

Documentation for `artifact-experiment` is available at [**artifact-experiment docs**](https://artifact-ml.readthedocs.io/en/latest/artifact-experiment).

## 🤝 Contributing

Contributions are welcome!

Please consult our [**contribution guidelines document**](https://artifact-ml.readthedocs.io/en/latest/Development/contributing).

## 📄 License

This project is licensed under the [MIT License](https://img.shields.io/github/license/vasileios-ektor-papoulias/artifact-ml).