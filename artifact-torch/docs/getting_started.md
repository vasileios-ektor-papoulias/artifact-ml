# Getting Started  

<p align="center">
  <img src="../assets/artifact_ml_logo.svg" width="200" alt="Artifact-ML Logo">
</p>  

## Installation

Install the latest [`artifact-torch`](https://pypi.org/project/artifact-torch/) release from PyPI by running:

```bash
pip install artifact-torch
```

This pulls in [`artifact-core`](https://pypi.org/project/artifact-core/) and [`artifact-experiment`](https://pypi.org/project/artifact-experiment/) automatically.

## Installing from Source (Development)

To work on `artifact-torch` itself—or to run the bundled [demos](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-torch/demos)—clone the [Artifact-ML monorepo](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main) by running:

```bash
git clone https://github.com/vasileios-ektor-papoulias/artifact-ml.git
```

Then install the package (with dev dependencies) using [Poetry](https://python-poetry.org/):

```bash
cd artifact-ml/artifact-torch

poetry install --with dev
```
