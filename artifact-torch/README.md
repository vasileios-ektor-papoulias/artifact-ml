# ⚙️ artifact-torch

> PyTorch integration for the Artifact framework, abstracting training infrastructure to let researchers focus on model innovation

<p align="center">
  <img src="./assets/artifact_ml_logo.svg" width="400" alt="Artifact-ML Logo">
</p>

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/github/license/vasileios-ektor-papoulias/artifact-core)

---

## 📋 Overview

**artifact-torch** provides PyTorch integration for the Artifact framework, delivering a type-safe training infrastructure that automatically integrates with artifact-core's validation capabilities.

The framework abstracts common deep learning engineering patterns—training loops, device management, callback systems, and validation orchestration—allowing researchers to focus on model architecture and domain-specific logic rather than infrastructure concerns.

**Core Value Proposition:**
- **Interface-driven design**: Implement clean contracts rather than complex training infrastructure
- **Automatic validation integration**: Seamless connection to artifact-core's validation ecosystem
- **Type safety throughout**: Full type checking for models, data flow, and component compatibility
- **Domain-specific extensions**: Specialized toolkits for different problem domains

## 🏗️ Architecture

The framework employs a three-layer architecture that enables domain-agnostic infrastructure reuse through standardized interface contracts, allowing diverse domain-specific applications to leverage shared infrastructure components:

**Domain-Specific Components**: Beyond model interfaces, artifact validation routines also contain domain-dependent logic, statically enforced through generic type parameters that demand specific model interface contracts at class definition time. This ensures type-safe integration while maintaining the flexibility for domain-specific validation behaviors across different problem domains.

```mermaid
graph TD
    %% Layer labels
    ImplLabel["<b>User Implementation Layer</b>"]
    ConfigLabel["<b>User Configuration Layer</b>"]
    InfraLabel["<b>Framework Infrastructure Layer</b>"]
    ExternalLabel["<b>External Integration Layer</b>"]
    
    %% Vertical spacing elements
    VerticalSpacer1[" "]
    VerticalSpacer2[" "]
    VerticalSpacer3[" "]
    
    %% Layer groupings with better horizontal distribution
    subgraph impl [" "]
        direction LR
        Model["Model Interface<br/>(Domain-specific logic)"]
        Data["Data Pipeline<br/>(Dataset)"]
    end
    
    subgraph config [" "]
        direction LR
        Trainer["CustomTrainer<br/>(Training orchestration)"]
        ConfigSpacer[" "]
        BatchRoutine["Batch Routine<br/>(Infrastructure)"]
        DataLoaderRoutine["DataLoader Routine<br/>(Infrastructure)"]
        ArtifactRoutine["Artifact Validation Routine<br/>(Domain-specific)"]
    end
    
    subgraph infra [" "]
        direction LR
        Cache["RAM Score Cache<br/>(Caching system)"]
        EarlyStopping["Early Stopping<br/>(Training termination)"]
        ModelTracking["Model Tracking<br/>(State management)"]
        Device["Device Management<br/>(Automatic placement)"]
        BatchCallbacks["Batch Callbacks<br/>(Per-batch execution)"]
        DataLoaderCallbacks["DataLoader Callbacks<br/>(Per-dataloader execution)"]
    end
    
    subgraph external [" "]
        direction LR
        ExternalSpacer1[" "]
        ExternalSpacer2[" "]
        ExternalSpacer3[" "]
        ExternalSpacer4[" "]
        ExternalSpacer5[" "]
        ArtifactCore["artifact-core<br/>(Validation artifacts)"]
        ArtifactExp["artifact-experiment<br/>(Experiment tracking)"]
    end
    
    %% Component connections with optimal ordering
    %% Implementation to Configuration
    Model --> Trainer
    Data --> Trainer
    
    %% Configuration to Infrastructure (left to right order)
    Trainer --> Cache
    Trainer --> EarlyStopping
    Trainer --> ModelTracking
    Trainer --> Device
    
    %% Configuration to Configuration (routine orchestration)
    Trainer --> BatchRoutine
    Trainer --> DataLoaderRoutine
    Trainer --> ArtifactRoutine
    
    %% Infrastructure routines to specific Callback types
    BatchRoutine --> BatchCallbacks
    DataLoaderRoutine --> DataLoaderCallbacks
    
    %% Domain-specific routine to External (direct integration)
    ArtifactRoutine --> ArtifactCore
    ArtifactRoutine --> ArtifactExp
    
    %% Callback systems to External (framework needs)
    BatchCallbacks --> ArtifactCore
    BatchCallbacks --> ArtifactExp
    DataLoaderCallbacks --> ArtifactCore
    DataLoaderCallbacks --> ArtifactExp
    
    %% Style layer labels with positioning
    style ImplLabel fill:#e1f5fe,stroke:#0066cc,stroke-width:3px,color:#000000
    style ConfigLabel fill:#f3e5f5,stroke:#7b1fa2,stroke-width:3px,color:#000000
    style InfraLabel fill:#e8f5e8,stroke:#388e3c,stroke-width:3px,color:#000000
    style ExternalLabel fill:#fff3e0,stroke:#f57c00,stroke-width:3px,color:#000000
    
    %% Style layer boxes
    style impl fill:#ffffff,stroke:#0066cc,stroke-width:2px
    style config fill:#ffffff,stroke:#7b1fa2,stroke-width:2px
    style infra fill:#ffffff,stroke:#388e3c,stroke-width:2px
    style external fill:#ffffff,stroke:#f57c00,stroke-width:2px
    
    %% Clean vertical spacing with extra space for external layer
    ImplLabel --- impl
    ConfigLabel --- config  
    InfraLabel --- infra
    ExternalLabel --- external
    
    impl ~~~ config
    config ~~~ infra
    infra ~~~ VerticalSpacer1
    VerticalSpacer1 ~~~ VerticalSpacer2
    VerticalSpacer2 ~~~ VerticalSpacer3
    VerticalSpacer3 ~~~ external
    
    %% Make all component arrows thicker
    linkStyle 0 stroke-width:3px
    linkStyle 1 stroke-width:3px
    linkStyle 2 stroke-width:3px
    linkStyle 3 stroke-width:3px
    linkStyle 4 stroke-width:3px
    linkStyle 5 stroke-width:3px
    linkStyle 6 stroke-width:3px
    linkStyle 7 stroke-width:3px
    linkStyle 8 stroke-width:3px
    linkStyle 9 stroke-width:3px
    linkStyle 10 stroke-width:3px
    linkStyle 11 stroke-width:3px
    linkStyle 12 stroke-width:3px
    linkStyle 13 stroke-width:3px
    linkStyle 14 stroke-width:3px
    linkStyle 15 stroke-width:3px
    linkStyle 16 stroke-width:3px
    
    %% Hide spacers
    style ExternalSpacer1 fill:transparent,stroke:transparent
    style ExternalSpacer2 fill:transparent,stroke:transparent
    style ExternalSpacer3 fill:transparent,stroke:transparent
    style ExternalSpacer4 fill:transparent,stroke:transparent
    style ExternalSpacer5 fill:transparent,stroke:transparent
    style ConfigSpacer fill:transparent,stroke:transparent
    style VerticalSpacer1 fill:transparent,stroke:transparent
    style VerticalSpacer2 fill:transparent,stroke:transparent
    style VerticalSpacer3 fill:transparent,stroke:transparent
```

### User Implementation Layer
Users implement domain-specific components: model interfaces defining training and generation behavior, data pipelines for loading and processing, and validation routines specifying artifact computation workflows.

### User Configuration Layer
Users extend `CustomTrainer` through subclassing and hook method implementations to configure training behavior—optimization, early stopping, checkpointing, and routine integration—while the framework handles the core orchestration logic.

### Framework Infrastructure Layer
Pure framework components that users don't directly implement: the callback execution system and automatic device management. These provide the underlying infrastructure that powers the user-configured training process.

### External Integration Layer
Automatic connections to the broader Artifact ecosystem: artifact-core for validation artifact computation and artifact-experiment for experiment tracking and result export.

## 🔧 Core Abstractions

### Model Interfaces

**Purpose**: Define contracts for model integration with the training framework.

**Implementation Requirement**: Extend domain-specific interfaces (e.g., `TableSynthesizer` for tabular synthesis) and implement required methods for training and validation.

**Framework Integration**: The trainer uses these interfaces to execute training while validation routines use generation methods for artifact computation.

### Model I/O Types

**Purpose**: Ensure type safety and component compatibility through standardized input/output definitions.

**Design Pattern**: Define `ModelInput` and `ModelOutput` TypedDict classes that specify exactly what flows through your model during training.

**Type Variance Benefits**: I/O types determine callback compatibility—the framework's type system ensures only compatible callbacks can be used with your model.

### CustomTrainer

**Purpose**: Orchestrate the complete training process while providing configuration hooks for domain-specific requirements.

**Framework Responsibilities**: Training loop execution, device management, gradient computation, checkpoint handling, and metric aggregation.

**User Configuration**: Implement hook methods for optimizer selection, early stopping criteria, callback configuration, and validation routine integration.



### Callback System

**Purpose**: Provide extensible hooks for custom behavior injection at specific training points.

**Type Variance Architecture**: Callbacks are model I/O type-aware through variance-based type parameters. The framework uses type variance to enable static type analysis tools to determine which callbacks are compatible with your model: only callbacks compatible with your specific `ModelInput` and `ModelOutput` types can be correctly instantiated.

**Core Callback Types**:

- **Batch Callbacks**: Execute on individual training batches, providing immediate per-batch computations
  - `BatchScoreCallback`: Compute scalar metrics from single batches
  - `BatchArrayCallback`: Generate arrays from single batches  
  - `BatchPlotCallback`: Create visualizations from single batches
  - Collection variants: `BatchScoreCollectionCallback`, `BatchArrayCollectionCallback`, `BatchPlotCollectionCallback`

- **DataLoader Callbacks**: Execute after processing entire dataloaders, aggregating results across all batches
  - `DataLoaderScoreCallback`: Compute metrics by aggregating batch results
  - `DataLoaderArrayCallback`: Generate arrays by combining batch outputs
  - `DataLoaderPlotCallback`: Create visualizations from aggregated data
  - Collection variants: `DataLoaderScoreCollectionCallback`, `DataLoaderArrayCollectionCallback`, `DataLoaderPlotCollectionCallback`

**Type Safety Mechanism**: Through variance-based generics, the framework enables static type analysis to verify that only compatible callbacks can be instantiated with your model's I/O types.

### Routines

**Purpose**: Combine multiple callbacks into standalone execution flows that are injected into the training loop at specific points.

**Architectural Relationship**: Routines operate one abstraction level above callbacks—they orchestrate collections of related callbacks into cohesive execution units rather than executing individual behaviors.

**Type Variance Integration**: Like callbacks, routines use type variance to enable static type analysis to determine compatibility. Static type checkers can verify which routine types are compatible with your model based on I/O type compatibility.

**Core Routine Types**:

- **BatchRoutine**: Combines batch callbacks into execution flows triggered during individual batch processing
  - Configures which batch callbacks to execute and when
  - Provides batch-level cache management and result aggregation
  - Executes during the training loop's batch processing phase

- **DataLoaderRoutine**: Orchestrates dataloader callbacks into flows executed after processing complete dataloaders  
  - Manages multiple callback handler types (scores, arrays, plots, collections)
  - Provides dataloader-level cache management and result aggregation
  - Executes at epoch boundaries or when explicitly triggered

- **ArtifactRoutine**: Integrates artifact-core validation capabilities into periodic training evaluation
  - Orchestrates model-specific actions required for validation plan execution (e.g., synthetic data generation for generative models, prediction generation for classification models)
  - Coordinates artifact computation through artifact-core based on model outputs
  - Manages validation plan execution and result export across different model domains

**Execution Integration**: The `CustomTrainer` integrates these routines at appropriate training loop hooks, ensuring proper execution timing and resource management.

### Data Abstractions

**Purpose**: Provide type-safe wrappers around PyTorch's data primitives with enhanced functionality.

**Components**: Generic `Dataset[T]` wrapper, enhanced `DataLoader` with automatic device management, and `DeviceManager` for placement handling.

**Integration Benefits**: Automatic device management and type-safe data flow through the training pipeline.

## 📁 Implementation Guidelines

### Project Organization

The framework expects a specific project structure that separates concerns and promotes maintainability:

```
project_root/
├── model/
│   ├── io.py                    # ModelInput/ModelOutput type definitions
│   ├── synthesizer.py           # Framework interface implementation
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

### Implementation Sequence

1. **Define I/O Types**: Establish type contracts for model inputs and outputs
2. **Implement Model Interface**: Extend domain-specific interfaces with your architecture
3. **Configure Data Pipeline**: Implement type-safe dataset and dataloader components
4. **Configure Validation**: Define validation routines and artifact generation plans
5. **Configure Training**: Extend CustomTrainer with domain-specific hooks
6. **Orchestration**: Create high-level APIs for simplified usage (optional)

**Detailed Implementation Example**: See the comprehensive tabular VAE demo in `demo/` which demonstrates the complete implementation pattern for tabular data synthesis.

## 🎯 Domain-Specific Toolkits

### Table Comparison

**Scope**: Complete toolkit for tabular data synthesis and evaluation.

**Core Interface**: `TableSynthesizer` protocol defining tabular generation contracts.

**Validation Integration**: `TableComparisonRoutine` with artifact-core's table comparison validation plans.

**Available Artifacts**:
- Distribution analysis (PDF/CDF comparisons)
- Dimensionality reduction visualizations (PCA projections)
- Correlation structure analysis
- Statistical distance metrics (Jensen-Shannon divergence, correlation distance)
- Descriptive statistics comparisons

**Reference Implementation**: The `demo/` directory contains a complete VAE-based tabular synthesizer demonstrating all toolkit components.

### Extension Framework

The architecture supports domain-specific toolkit development through:

1. **Interface Definition**: Create domain-specific model protocols
2. **Validation Integration**: Develop artifact-experiment validation plans (defining which artifacts to compute) and validation routines that execute them (including preparatory work like data generation)

## 🚀 Usage Patterns

### Basic Training Workflow

```python
# Component configuration
model = DomainModel.build(architecture_config)
dataset = TypedDataset(processed_data)
validation_routine = DomainValidationRoutine.build(reference_data, tracking_client)

# Framework orchestration
trainer = DomainTrainer.build(
    model=model,
    train_loader=DataLoader(dataset, batch_size=config.batch_size),
    artifact_routine=validation_routine,
    tracking_client=experiment_tracker
)

# Execution with integrated validation
training_metrics = trainer.train()
```

### Experiment Tracking Integration

The framework integrates with artifact-experiment for automatic result persistence:

```python
from artifact_experiment.libs.tracking.filesystem.client import FilesystemTrackingClient

tracking_client = FilesystemTrackingClient.build(experiment_id="research_experiment")
# All training artifacts automatically saved to structured directories
```

**For comprehensive usage examples and detailed implementation patterns, refer to the demo documentation in `demo/README.md`.**

## 🔧 Framework Extension

### Adding Domain Toolkits

1. **Domain Directory**: Create `domain_name/` in project root
2. **Interface Definition**: Define domain-specific model protocols
3. **Validation Integration**: Implement corresponding validation routines

### Component Extension

**Callback Development**: Place in `libs/components/callbacks/`, inherit from appropriate base classes, implement required hook methods.

**Early Stopping Criteria**: Extend `EarlyStopper[T]` in `libs/components/early_stopping/` with domain-specific termination logic.

**Model Interface Extension**: Define new protocols in domain directories with integration points for trainer and validation systems.

## 🚀 Installation

### Using Poetry (Recommended)

```bash
git clone https://github.com/vasileios-ektor-papoulias/artifact-ml.git
cd artifact-ml/artifact-torch
poetry install
```

### Using Pip

```bash
git clone https://github.com/vasileios-ektor-papoulias/artifact-ml.git
cd artifact-ml/artifact-torch  
pip install .
```

## 🤝 Contributing

Contributions are welcome. Please refer to the [main Artifact-ML contribution guidelines](https://github.com/vasileios-ektor-papoulias/artifact-ml/blob/main/README.md) for development standards and submission procedures.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
