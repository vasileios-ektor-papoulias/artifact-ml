# Architecture

<p align="center">
  <img src="../assets/artifact_ml_logo.svg" width="200" alt="Artifact-ML Logo">
</p>

## Architectural Layers

### User Interaction Layer

The interface through which users engage with the framework’s validation capabilities.

### Framework Infrastructure Layer

The internal computational and management system that executes and maintains artifact workflows.

### External Dependency Layer

The interface that connects the framework to external inputs and configurations required for operation.

Crucially, user-facing interfaces are separated from internal framework infrastructure: users interact primarily with `ArtifactEngine` while the framework handles the complexity of artifact registration, instantiation, and execution through its internal infrastructure components.

## Architecture Diagram

```mermaid
graph TB
    subgraph "User Interaction Layer"
        AE[ArtifactEngine]
    end
    
    subgraph "Framework Infrastructure Layer"
        A[Artifact]
        AREG[ArtifactRegistry]
        AT[ArtifactType]
        AR[ArtifactResources]
        ARS[ArtifactResourceSpec]
        AH[ArtifactHyperparams]
    end
    
    subgraph "External Dependencies"
        Config["Configuration Files"]
        Data["Resource Data"]
    end
    
    %% Dependencies (A uses B)
    AE --> AREG
    AREG --> AT
    AREG --> A
    A --> AR
    A --> ARS
    A --> AH
    AH --> Config
    AR --> Data
```

