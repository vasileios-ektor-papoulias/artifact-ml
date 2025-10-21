# Architecture

<p align="center">
  <img src="../assets/artifact_ml_logo.svg" width="200" alt="Artifact-ML Logo">
</p>


`artifact-experiment` follows a layered architecture that separates validation specification, execution orchestration, backend integration, and external dependencies:

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

## User Specification Layer
The interface for declaratively specifying validation requirements and experiment configurations.

## Execution Orchestration Layer
The internal workflow coordination system that transforms specifications into executable validation processes.

## Backend Integration Layer
The abstraction boundary that enables unified experiment tracking across multiple backend platforms.

## External Dependencies
External systems that the framework integrates with for validation computation and experiment persistence.