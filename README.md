# Artifact-ML

> Reusable ML experiment workflows built declaratively.

<p align="center">
  <img src="docs/assets/artifact_ml_logo.svg" width="500" alt="Artifact-ML Logo">
</p>

[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://artifact-ml.readthedocs.io/en/latest/)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/github/license/vasileios-ektor-papoulias/artifact-ml)

[![CI](https://img.shields.io/github/actions/workflow/status/vasileios-ektor-papoulias/artifact-ml/ci_push_main.yml?branch=main&label=CI)](https://github.com/vasileios-ektor-papoulias/artifact-ml/actions/workflows/ci_push_main.yml)
[![Coverage](https://codecov.io/gh/vasileios-ektor-papoulias/artifact-ml/branch/main/graph/badge.svg)](https://codecov.io/gh/vasileios-ektor-papoulias/artifact-ml/)
[![CodeFactor](https://www.codefactor.io/repository/github/vasileios-ektor-papoulias/artifact-ml/badge)](https://www.codefactor.io/repository/github/vasileios-ektor-papoulias/artifact-ml)

---

## ⚙️ Overview

Artifact-ML eliminates imperative glue code in ML experiments by providing the tools to build **reusable** workflows **declaratively**.

By *reusable*, we refer to workflows that are defined once with the potential to be reused by any model within the same task category.

By *declarative*, we refer to building through expressing high-level intent---rather than catering to implementation details.

For additional context, please refer to our [value proposition](value_proposition.md) and [motivating example](motivating_example.md) docs.

<p align="center">
  <img src="assets/pdf_comparison.png" width="400" alt="PDF Comparison Artifact">
</p>

## 🏗️ Packages

The project comprises three packages:

- [`artifact-core`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-core): foundational interfaces and abstractions for building validation workflows.
- [`artifact-experiment`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-experiment): experiment tracking toolkit supporting popular tracking backends (e.g. [Mlflow](https://mlflow.org/)).
- [`artifact-torch`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-torch): interfaces and abstractions for building deep learning experiments.

## 🚀 Quick Start

Clone the [**Artifact-ML**](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main) monorepo by running:

```bash
git clone https://github.com/vasileios-ektor-papoulias/artifact-ml.git
```

### To install [`artifact-core`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-core) run:

```bash
cd artifact-ml/artifact-core
poetry install
```

### To install [`artifact-experiment`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-experiment) run:

```bash
cd artifact-ml/artifact-experiment
poetry install
```

### To install [`artifact-torch`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-torch) run:

```bash
cd artifact-ml/artifact-torch
poetry install
```

## 📚 Documentation

Documentation for Artifact-ML is available at [**Artifact-ML Docs**](https://artifact-ml.readthedocs.io/en/latest/).

Package-specfic docs are available at:

- [artifact-core docs](https://artifact-ml.readthedocs.io/en/latest/artifact-core)
- [artifact-experiment docs](https://artifact-ml.readthedocs.io/en/latest/artifact-experiment)
- [artifact-torch docs](https://artifact-ml.readthedocs.io/en/latest/artifact-torch)


## 🤝 Contributing

Contributions are welcome!

Please consult our [**contribution guidelines document**](https://artifact-ml.readthedocs.io/en/latest/Development/contributing).


## 📄 License

This project is licensed under the [MIT License](https://img.shields.io/github/license/vasileios-ektor-papoulias/artifact-ml).