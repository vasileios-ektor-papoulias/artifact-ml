# Core Entities

<p align="center">
  <img src="../assets/artifact_ml_logo.svg" width="200" alt="Artifact-ML Logo">
</p>

## Entities by Layer

[`artifact-torch`](https://github.com/vasileios-ektor-papoulias/artifact-ml/tree/main/artifact-torch) delivers on its objective through the coordinated interaction of specialized abstractions across its [four architectural layers](architecture.md):

### User Implementation Layer

- **Model Interfaces**: Domain-specific protocols (e.g., `TableSynthesizer`) that define contracts for model integration with the training framework. Researchers extend these interfaces and implement required methods for training and validation.

- **Model I/O Types**: Type-safe contracts using `ModelInput` and `ModelOutput` TypedDict classes that specify exactly what flows through models during training, enabling static type checking and callback compatibility verification.

- **Data Abstractions**: Type-safe wrappers around PyTorch's data primitives with enhanced functionality, including generic `Dataset[T]` wrapper and enhanced `DataLoader` with automatic device management.

```python
class MyModel(
  TableSynthesizer[ModelInput, ModelOutput, MyGenerationParams]
  ):
    def forward(self, batch: ModelInput) -> ModelOutput:
        pass
    
    def generate(self, generation_params: MyGenerationParams) -> pd.DataFrame:
        pass
```

### User Configuration Layer

- **CustomTrainer**: Orchestrates the complete training process while providing configuration hooks for declarative customization. Users implement hook methods for optimizer selection, early stopping criteria, and callback configuration while the framework handles training loop execution, device management, and gradient computation.

```python
class MyTrainer(
    CustomTrainer[
        TableSynthesizer[ModelInput, ModelOutput, Any], # Expected model type.
        ModelInput, # Expected forward pass input.
        ModelOutput, # Expected forward pass output.
        ModelTrackingCriterion, # See artifact-torch docs.
        StopperUpdateData # See artifact-torch docs.
    ]
):
    def _get_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.model.parameters(),
            lr=config.lr
            )
    
    def _get_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        return torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.step_size
            )
    
    def _get_early_stopper(self) -> EarlyStopper:
        return EpochBoundStopper(
            n_epochs=config.num_epochs
            )
    
    @staticmethod
    def _get_train_loader_routine(
        data_loader: DataLoader[ModelInputT], 
        tracking_client: Optional[TrackingClient], 
    ) -> Optional[
        DataLoaderRoutine[ModelInputT, ModelOutputT]
        ]:
        return MyDataLoaderRoutine.build(
            data_loader=data_loader, # Artifact-ML type-aware wrapper
            tracking_client=tracking_client
            )
```

- **Validation Routines**: Validation workflow executors that integrate into the training pipeline:
  - **BatchRoutine**: callback execution during individual batch processing.
  - **DataLoaderRoutine**: callback execution on prescribed dataloaders (e.g. training/ validation).
  - **ArtifactRoutine**: execution of callbacks injecting validation capabilities provided by `artifact-core`.

```python
# Works with any neural network fulfilling the IO contract.
# The input contract is contravariant.
# The output contract is covariant.

class MyDataLoaderRoutine(
    DataLoaderRoutine[
        ModelInput, ModelOutput
        ] # Expected IO profile.
    ):
    @staticmethod
    def _get_score_callbacks() -> List[
        DataLoaderScoreCallback[ModelInput, ModelOutput]
        ]:
        return [
            TrainLossCallback(period=config.validation_frequency)
            ]
```


### Framework Infrastructure Layer

- **Callback System**: Type-aware execution hooks that inject custom behavior at specific training points. Callbacks use variance-based type parameters to ensure compatibility with model I/O types through static analysis.

- **Training Infrastructure Components**: Automatic systems that operate behind the scenes:
  - **Device Management**: Automatic tensor placement and device coordination.
  - **In-Memory Caching**: Caching of computed validation scores.
  - **Early Stopping**: Configurable training termination based on validation metrics.
  - **Model Tracking**: State management and best model persistence.

### External Integration Layer

- **Integration with `artifact-core`**: Automatic validation artifact computation during training through specialized routines that coordinate with the appropriate Artifact-ML [domain toolkit](https://artifact-ml.readthedocs.io/en/latest/artifact-core/domain_toolkits/).

- **Integration with `artifact-experiment`**: Experiment tracking using popular backend services (e.g. [MLflow](https://mlflow.org/), [ClearML](https://clear.ml/), [Neptune](https://neptune.ai/)) or simple filesystem/ in-memory caching.