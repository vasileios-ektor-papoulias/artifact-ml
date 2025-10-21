# Architecture

<p align="center">
  <img src="../assets/artifact_ml_logo.svg" width="200" alt="Artifact-ML Logo">
</p>

The framework employs a three-layer architecture with standardized interface contracts that enable maximal infrastructure reuse. Beyond abstracting engineering concerns, adherence to these type contracts unlocks instant access to comprehensive validation capabilities. Researchers select a problem domain (e.g., tabular data synthesis), adhere to its contracts and focus solely on architectural innovations. They no longer need to write lengthy experiment scripts with messy logic tailored to individual models. Training infrastructure can be shared. This eliminates implementation overhead and accelerates research velocity.

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
        Model["Model<br/>(Architecture)"]
        Data["Data Pipeline<br/>(Dataset)"]
    end
    
    subgraph config [" "]
        direction LR
        Trainer["CustomTrainer<br/>(Training orchestration)"]
        ConfigSpacer[" "]
        BatchRoutine["Batch Routine<br/>(Batch processing hook)"]
        DataLoaderRoutine["DataLoader Routine<br/>(Data loader processing hook)"]
        ArtifactRoutine["Artifact Validation Routine<br/>(Artifact-ML validation hook)"]
    end
    
    subgraph infra [" "]
        direction LR
        Cache["RAM Score Cache<br/>(Caching system)"]
        EarlyStopping["Early Stopping<br/>(Training termination)"]
        ModelTracking["Model Tracking<br/>(State management)"]
        Device["Device Management<br/>(Automatic placement)"]
        Callbacks["Callbacks<br/>(hook execution atoms)"]
    end
    
    subgraph external [" "]
        direction LR
        ExternalSpacer1[" "]
        ExternalSpacer2[" "]
        ExternalSpacer3[" "]
        ExternalSpacer4[" "]
        ExternalSpacer5[" "]
        ArtifactExp["artifact-experiment<br/>(Experiment tracking)"]
        ArtifactCore["artifact-core<br/>(Validation artifacts)"]
    end
    
    %% Component connections with optimal ordering
    %% Configuration uses Implementation
    Trainer --> Model
    Trainer --> Data
    
    %% Configuration to Infrastructure (left to right order)
    Trainer --> Cache
    Trainer --> EarlyStopping
    Trainer --> ModelTracking
    Trainer --> Device
    Trainer --> Callbacks
    
    %% Configuration to Configuration (routine orchestration)
    Trainer --> BatchRoutine
    Trainer --> DataLoaderRoutine
    Trainer --> ArtifactRoutine
    
    %% Infrastructure routines to Callback System
    BatchRoutine --> Callbacks
    DataLoaderRoutine --> Callbacks
    
    %% Domain-specific routine to External (via experiment tracking)
    ArtifactRoutine --> ArtifactExp
    
    %% Callback System to External (via experiment tracking)
    Callbacks --> ArtifactExp
    
    %% External integration flow
    ArtifactExp --> ArtifactCore
    
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
    
    impl ~~~ config
    config ~~~ infra
    infra ~~~ VerticalSpacer1
    VerticalSpacer1 ~~~ VerticalSpacer2
    VerticalSpacer2 ~~~ VerticalSpacer3
    VerticalSpacer3 ~~~ ExternalLabel
    ExternalLabel --- external
    
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

## User Implementation Layer
The boundary where researchers implement domain-specific logic and model architectures.

## User Configuration Layer
The interface for configuring training behavior and validation workflows through declarative specifications.

## Framework Infrastructure Layer
The automated training infrastructure that operates transparently behind user configurations.

## External Integration Layer
The connection points to external Artifact framework components.