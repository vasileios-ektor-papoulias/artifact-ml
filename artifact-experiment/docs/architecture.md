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
        AP[ArtifactPlan]
    end
    
    subgraph "Execution Orchestration Layer"  
        ACF[ArtifactCallbackFactory]
        CB[Callbacks]
        CBH[CallbackHandlers]
    end
    
    subgraph "Backend Integration Layer"
        TC[TrackingClient]
        TQ[TrackingQueue]
        BLW[BackendLoggingWorker]
        AL["BackendLoggers<br/>(ArtifactLoggers)"]
        RA[RunAdapter]
    end
    
    subgraph "External Dependencies"
        AC["artifact-core<br/>Computation Engine"]
        EB["Experiment Backends<br/>MLflow, ClearML, Neptune"]
    end
    
    AP --> ACF
    AP --> CBH
    CBH --> CB
    ACF --> AC
    CB -->|write results| TQ
    TC -->|owns| TQ
    TC -->|owns| BLW
    BLW -->|consumes| TQ
    BLW -->|holds| AL
    AL --> RA
    RA --> EB
```

