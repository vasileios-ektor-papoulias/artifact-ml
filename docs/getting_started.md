# Getting Started  

<p align="center">
  <img src="../assets/artifact_ml_logo.svg" width="200" alt="Artifact-ML Logo">
</p>  

## Installation

All three Artifact-ML packages are published on PyPI.

### [`artifact-core`](https://pypi.org/project/artifact-core/)

```bash
pip install artifact-core
```

For details consult the [package's docs](https://artifact-ml.readthedocs.io/en/latest/artifact-core).  

### [`artifact-experiment`](https://pypi.org/project/artifact-experiment/)

```bash
pip install artifact-experiment
```

For details consult the [package's docs](https://artifact-ml.readthedocs.io/en/latest/artifact-experiment).  

### [`artifact-torch`](https://pypi.org/project/artifact-torch/)

```bash
pip install artifact-torch
```

For details consult the [package's docs](https://artifact-ml.readthedocs.io/en/latest/artifact-torch).

## Installing from Source (Development)

To work on Artifact-ML itself—or to run the bundled demos—clone the [monorepo](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main) by running:

```bash
git clone https://github.com/vasileios-ektor-papoulias/artifact-ml.git
```

Then install the package you're working on (with dev dependencies) using [Poetry](https://python-poetry.org/), e.g.:

```bash
cd artifact-ml/artifact-core

poetry install --with dev
```

The same applies to `artifact-experiment` and `artifact-torch`: each package directory hosts its own Poetry project.
