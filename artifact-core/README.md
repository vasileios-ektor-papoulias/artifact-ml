# ⚙️ artifact-core

> A unified minimal interface for the computation of heterogeneous validation artifacts in machine learning experiments.

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

This repository serves as the foundation of [**Artifact-ML**](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main).

It stands alongside:

- [`artifact-experiment`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-experiment): The framework's experiment tracking extension.
- [`artifact-torch`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-torch): PyTorch integration for building reusable deep-learning workflows.

`artifact-core` provides a **unified minimal interface** for the computation of heterogeneous validation artifacts in machine learning experiments.

Here, we use the word *minimal* to refer to an interface that is as thin as possible given its purpose.

The goal is to enable declarative experiment orchestration through simple enum-based configuration.

By abstracting away unique parameter requirements (static data specifications, hyperparameters) into framework-managed components, `artifact-core` enables downstream client code (e.g. experiment scripts) to invoke a wide array of validation artifacts using only type enumerations---as opposed to artifact-specific argument profiles.

This design eliminates the need for custom integration code per artifact, enabling generic experiment scripts that scale seamlessly across diverse validation requirements with zero modification/ friction.

## 📚 Usage Sketch

```python
import pandas as pd

from artifact_core.table_comparison import (
    TableComparisonEngine,
    TableComparisonScoreCollectionType,
    TabularDataSpec
)

df_real = pd.read_csv("real_data.csv")

df_synthetic = pd.read_csv("synthetic_data.csv")

data_spec = TabularDataSpec.from_df(
    df=df_real, 
    cat_features=categorical_features, 
    cont_features=continuous_features
)

engine = TableComparisonEngine(resource_spec=data_spec)

dict_js_distance_per_feature = engine.produce_dataset_comparison_score_collection(
    score_collection_type=TableComparisonScoreCollectionType.JS_DISTANCE,
    dataset_real=df_real,
    dataset_synthetic=df_synthetic,
)

dict_js_distance_per_feature
```

<p align="center">
  <img src="assets/js.png" width="350" alt="JS Distance Artifact">
</p>

```python
from artifact_core.table_comparison import (
    TableComparisonPlotType,
)

pca_plot = engine.produce_dataset_comparison_plot(
    plot_type=TableComparisonPlotType.PCA_JUXTAPOSITION,
    dataset_real=df_real,
    dataset_synthetic=df_synthetic,
)

pca_plot
```

<p align="center">
  <img src="assets/pca_comparison_artifact.png" width="1000" alt="PCA Projection Artifact">
</p>

```python
pdf_plot = engine.produce_dataset_comparison_plot(
    plot_type=TableComparisonPlotType.PDF,
    dataset_real=df_real,
    dataset_synthetic=df_synthetic,
)

pdf_plot
```

<p align="center">
  <img src="assets/pdf_comparison_artifact.png" width="1700" alt="PDF Comparison Artifact">
</p>

## 🚀 Installation

Clone the [**Artifact-ML**](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main) monorepo by running:

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
