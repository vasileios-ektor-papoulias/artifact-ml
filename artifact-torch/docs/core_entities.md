# Core Entities

<p align="center">
  <img src="../assets/artifact_ml_logo.svg" width="200" alt="Artifact-ML Logo">
</p>

The framework delivers on its objective through coordinated interaction of specialized abstractions across its [four architectural layers](architecture.md):

## **User Implementation Layer**

- **Model Interfaces**: Domain-specific protocols (e.g., `TableSynthesizer`) that define contracts for model integration with the training framework. Researchers extend these interfaces and implement required methods for training and validation.

- **Model I/O Types**: Type-safe contracts using `ModelInput` and `ModelOutput` TypedDict classes that specify exactly what flows through models during training, enabling static type checking and callback compatibility verification.

- **Data Abstractions**: Type-safe wrappers around PyTorch's data primitives with enhanced functionality, including generic `Dataset[T]` wrapper and enhanced `DataLoader` with automatic device management.

```python
class MyModel(TableSynthesizer[MyModelInput, MyModelOutput]):
    def training_step(self, batch: MyModelInput) -> MyModelOutput:
        # Model-specific training logic
        pass
    
    def generate_synthetic_data(self, num_samples: int) -> pd.DataFrame:
        # Domain-specific generation logic
        pass
```

## **User Configuration Layer**

- **CustomTrainer**: Orchestrates the complete training process while providing configuration hooks for declarative customization. Users implement hook methods for optimizer selection, early stopping criteria, and callback configuration while the framework handles training loop execution, device management, and gradient computation.

- **Validation Routines**: Validation workflow executors that integrate into the training pipeline:
  - **BatchRoutine**: callback execution during individual batch processing.
  - **DataLoaderRoutine**: callback execution on prescribed dataloaders (e.g. training/ validation).
  - **ArtifactRoutine**: execution of callbacks injecting validation capabilities provided by `artifact-core`.

```python
class MyTrainer(CustomTrainer[MyModelInput, MyModelOutput]):
    def _get_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=0.001)
    
    def _get_artifact_routine(self) -> MyArtifactRoutine:
        return MyArtifactRoutine.build(validation_data, tracking_client)
```

### **Framework Infrastructure Layer**

- **Callback System**: Type-aware execution hooks that inject custom behavior at specific training points. Callbacks use variance-based type parameters to ensure compatibility with model I/O types through static analysis.

- **Training Infrastructure Components**: Automatic systems that operate behind the scenes:
  - **Device Management**: Automatic tensor placement and device coordination.
  - **In-Memory Caching**: Caching of computed validation scores.
  - **Early Stopping**: Configurable training termination based on validation metrics.
  - **Model Tracking**: State management and best model persistence.

```python
# Framework automatically manages device placement, caching, early stopping
trainer = MyTrainer.build(model=model, train_loader=train_loader)
trainer.train()  # All infrastructure handled automatically
```

## **External Integration Layer**

- **Integration with `artifact-core`**: Automatic validation artifact computation during training through specialized routines that coordinate with the appropriate Artifact-ML [domain toolkit](https://artifact-ml.readthedocs.io/en/latest/artifact-core/domain_toolkits/).

- **Integration with `artifact-experiment`**: Experiment tracking using popular backend services ([MLflow](https://mlflow.org/), [ClearML](https://clear.ml/), [Neptune](https://neptune.ai/)) or simple filesystem/ in-memory caching.