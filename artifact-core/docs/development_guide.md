# Development Guide

<p align="center">
  <img src="../assets/artifact_ml_logo.svg" width="200" alt="Artifact-ML Logo">
</p>

We demonstrate the steps required to create a complete validation toolkit from scratch:

## 1. Define Artifact Types

First, you define enumerations for each type of artifact your engine will support:

```python
from artifact_core.spi.orchestration import ArtifactType

class CustomScoreType(ArtifactType):
    CUSTOM_SCORE = "custom_score"
    
class CustomArrayType(ArtifactType):
    pass
    
class CustomPlotType(ArtifactType):
    pass
    
class CustomScoreCollectionType(ArtifactType):
    pass
    
class CustomArrayCollectionType(ArtifactType):
    pass
    
class CustomPlotCollectionType(ArtifactType):
    pass
```

These enumerations serve as identifiers for the different artifacts that can be computed by your engine.

## 2. Create a Resource Specification

The resource specification defines the structural properties of your validation resources:

```python
from dataclasses import dataclass

from artifact_core.spi.resources import ResourceSpecProtocol

@dataclass
class CustomResourceSpec(ResourceSpecProtocol):
    validation_resource_structural_property: float
```

## 3. Define Resources

Resources are the data objects that artifacts will operate on:

```python
from dataclasses import dataclass

import numpy as np

from artifact_core.spi.resources import ArtifactResources

@dataclass(frozen=True)
class CustomResources(ArtifactResources):
    resource_attribute: np.ndarray
```

## 4. Create Registries

Registries manage the organization and retrieval of artifacts.

Each registry is generic over the resources, the resource specification, the artifact type enum, and the result type---i.e. `ArtifactRegistry[ArtifactResources, ResourceSpecProtocol, ArtifactType, Result]`.

Registries read artifact configurations (keyed by enum member *name*) from configuration files:

```python
import json
import os
from typing import Any, Dict, Mapping

from artifact_core.spi.orchestration import ArtifactRegistry

# Helper function to load configurations
def load_config_section(config_path: str, section: str) -> Dict[str, Dict[str, Any]]:
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            return config.get(section, {})
    return {}

# Path to configuration file
CONFIG_PATH = 'path/to/custom_engine/config/raw.json'

# Create artifact registries
class CustomScoreRegistry(ArtifactRegistry[CustomResources, CustomResourceSpec, CustomScoreType, float]):
    @classmethod
    def _get_artifact_configurations(cls) -> Mapping[str, Mapping[str, Any]]:
        return load_config_section(
            config_path=CONFIG_PATH,
            section='scores'
            )

# Similar registries for other artifact types...
```

## 5. Implement Artifacts

Create concrete artifact implementations.

Artifacts are generic over the resources, the resource specification, the hyperparameters, and the result type---i.e. `Artifact[ArtifactResources, ResourceSpecProtocol, ArtifactHyperparams, Result]`:

```python
from artifact_core.spi.artifact import NO_ARTIFACT_HYPERPARAMS, Artifact

@CustomScoreRegistry.register_artifact(CustomScoreType.CUSTOM_SCORE)
class CustomScore(Artifact[CustomResources, CustomResourceSpec, NO_ARTIFACT_HYPERPARAMS, float]):
    def _validate(self, resources: CustomResources) -> CustomResources:
        if not hasattr(resources, "resource_attribute"):
            raise ValueError("Resources must contain resource_attribute")
        return resources
        
    def _compute(self, resources: CustomResources) -> float:
        return resources.resource_attribute.mean()
```

## 6. Deploy Artifacts through an Artifact Engine

Finally, create the engine that orchestrates the computation of artifacts:

```python
from typing import Type

import numpy as np

from artifact_core.spi.orchestration import ArtifactEngine

class CustomArtifactEngine(ArtifactEngine[
    CustomResources,
    CustomResourceSpec,
    CustomScoreType,
    CustomArrayType,
    CustomPlotType,
    CustomScoreCollectionType,
    CustomArrayCollectionType,
    CustomPlotCollectionType
]):
    @classmethod
    def _get_score_registry(cls) -> Type[CustomScoreRegistry]:
        return CustomScoreRegistry
        
    # Similar methods for other registries...
        
    # Custom methods for your specific use case
    def produce_custom_score(
        self, 
        score_type: CustomScoreType, 
        resource_attribute: np.ndarray
    ) -> float:
        resources = CustomResources(resource_attribute=resource_attribute)
        return self.produce_score(score_type=score_type, resources=resources)
```

Engines are instantiated via the `build()` classmethod, which wires in the appropriate registries automatically:

```python
spec = CustomResourceSpec(validation_resource_structural_property=1.0)

engine = CustomArtifactEngine.build(resource_spec=spec)
```
