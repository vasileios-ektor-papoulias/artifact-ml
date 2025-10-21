# Core Entities

<p align="center">
  <img src="../assets/artifact_ml_logo.svg" width="200" alt="Artifact-ML Logo">
</p>

The framework achieves project-agnostic training infrastructure through coordinated interaction of specialized abstractions across its [four architectural layers](architecture.md):

## **User Implementation Layer**
Core components that researchers implement to encode their specific research problem logic:

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

**Architecture Role**: These components encode the unique aspects of each research project while conforming to framework contracts that enable infrastructure sharing.

## **User Configuration Layer**
Abstractions that eliminate implementation complexity by requiring only configuration through subclass hooks:

- **CustomTrainer**: Orchestrates the complete training process while providing configuration hooks for domain-specific requirements. Users implement hook methods for optimizer selection, early stopping criteria, and callback configuration while the framework handles training loop execution, device management, and gradient computation.

- **Validation Routines**: Specialized routine configurations that integrate validation into training workflows:
  - **BatchRoutine**: Configures callback execution during individual batch processing
  - **DataLoaderRoutine**: Orchestrates callback execution after processing complete dataloaders
  - **ArtifactRoutine**: Integrates artifact-core validation capabilities into periodic training evaluation

```python
class MyTrainer(CustomTrainer[MyModelInput, MyModelOutput]):
    def _get_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=0.001)
    
    def _get_artifact_routine(self) -> MyArtifactRoutine:
        return MyArtifactRoutine.build(validation_data, tracking_client)
```

**Architecture Role**: These configurations eliminate training infrastructure duplication by providing reusable orchestration logic that adapts to domain-specific requirements through hook methods.

### **Framework Infrastructure Layer**
Concrete framework components that handle training infrastructure automatically:

- **Callback System**: Type-aware execution hooks that inject custom behavior at specific training points. Callbacks use variance-based type parameters to ensure compatibility with model I/O types through static analysis.

- **Training Infrastructure Components**: Automatic systems that operate behind the scenes:
  - **Device Management**: Automatic tensor placement and device coordination
  - **RAM Caching**: Intelligent caching for computed validation scores
  - **Early Stopping**: Configurable training termination based on validation metrics
  - **Model Tracking**: State management and best model persistence

```python
# Framework automatically manages device placement, caching, early stopping
trainer = MyTrainer.build(model=model, train_loader=train_loader)
trainer.train()  # All infrastructure handled automatically
```

**Architecture Role**: These components execute training tasks automatically, freeing researchers from infrastructure concerns while providing comprehensive training capabilities.

## **External Integration Layer**
Seamless connections to the broader Artifact ecosystem:

- **artifact-core Integration**: Automatic validation artifact computation during training through specialized routines that coordinate with domain-specific engines.

- **artifact-experiment Integration**: Experiment tracking and result export to popular backends (MLflow, ClearML, Neptune, filesystem) through unified tracking client interfaces.

**Architecture Role**: These integrations connect training workflows to comprehensive validation and experiment tracking capabilities without requiring additional implementation effort.