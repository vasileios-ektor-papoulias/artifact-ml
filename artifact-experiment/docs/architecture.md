# Architecture

<p align="center">
  <img src="../assets/artifact_ml_logo.svg" width="200" alt="Artifact-ML Logo">
</p>


## Architectural Layers

### User Specification Layer

The interface through which users declaratively specify validation workflows and experiment configurations.

### Execution Orchestration Layer

The internal coordination system that transforms user specifications into executable validation workflows.

### Backend Integration Layer

The abstraction layer that unifies experiment tracking and management across multiple backend systems.

### External Dependency Layer

The interface that connects the framework to external systems for validation computation and experiment persistence.

## Architecture Diagram

```mermaid
graph TB
    subgraph "User Specification Layer"
        VP[ValidationPlan]
    end
    
    subgraph "Execution Orchestration Layer"  
        AF[Artifact Factories]
        CB[Callbacks]
        CBH[Callback Handlers]
    end
    
    subgraph "Backend Integration Layer"
        RA[Run Adapters]
        AL[Artifact Loggers]
        TC[Tracking Clients]
    end
    
    subgraph "External Dependencies"
        AC["artifact-core<br/>Computation Engine"]
        EB["Experiment Backends<br/>MLflow, ClearML, Neptune"]
    end
    
    VP --> AF
    VP --> CBH
    CBH --> CB
    CBH --> TC
    CB --> TC
    TC --> AL
    TC --> RA
    AL --> RA
    AF --> AC
    RA --> EB
```

