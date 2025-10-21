# User Guide

<p align="center">
  <img src="../assets/artifact_ml_logo.svg" width="200" alt="Artifact-ML Logo">
</p>

## Suggested Project Organization

The following template summarizes the various entities that need to be implemented when building a deep learning project with `artifact-torch`:

```
project_root/
├── model/
│   ├── io.py                    # ModelInput/ModelOutput type definitions
│   ├── model.py                 # Framework interface implementation
│   └── architectures/           # Neural network implementations
├── data/
│   ├── dataset.py              # Type-safe dataset implementation
│   └── preprocessing/          # Data transformation pipeline
├── trainer/
│   └── trainer.py              # CustomTrainer extension
├── routines/
│   ├── artifact.py             # Validation routine configuration
│   ├── batch.py                # Batch-level callback routines
│   └── loader.py               # DataLoader-level callback routines
└── config/
    └── configuration files
```

## Implementation Sequence

1. **Define I/O Types**: Establish type contracts for model inputs and outputs
2. **Implement Model Interface**: Extend domain-specific interfaces with your architecture
3. **Configure Data Pipeline**: Implement type-safe dataset and dataloader components
4. **Configure Validation**: Configure validation routines by implementing subclass hooks
5. **Configure Training**: Configure CustomTrainer by implementing subclass hooks
6. **Orchestration**: Create a high-level API for simplified usage (optional)


## End to End Demo Project

For comprehensive usage examples and detailed implementation patterns, refer to our [synthetic tabular data demo project](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-torch/demos/table_comparison).