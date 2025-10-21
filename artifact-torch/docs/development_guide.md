# Development Guide

<p align="center">
  <img src="../assets/artifact_ml_logo.svg" width="200" alt="Artifact-ML Logo">
</p>

## Adding Domain Toolkits

1. **Domain Directory**: Create `domain_name/` in project root
2. **Interface Definition**: Define domain-specific model protocols
3. **Validation Integration**: Implement corresponding validation routines

## Component Extension

**Model Type Contract Development**: Define new `Model`, `ModelInput` and `ModelOutput` contracts in `core/model` for domain-specific data flow patterns, enabling type-safe callback development and static compatibility verification.

**Callback Development**: Place in `libs/components/callbacks/`, inherit from appropriate base classes, implement required hook methods.

**Model Tracker Development**: Extend `ModelTracker[T]` in `libs/components/model_tracking/` with domain-specific best model tracking criteria (e.g., validation loss improvement, custom metric optimization).

**Early Stopping Criteria**: Extend `EarlyStopper[T]` in `libs/components/early_stopping/` with domain-specific termination logic.
