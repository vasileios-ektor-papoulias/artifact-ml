# Implementation Guide

<p align="center">
  <img src="../assets/artifact_ml_logo.svg" width="200" alt="Artifact-ML Logo">
</p>

To better understand how the architecture works in practice, we go through the steps required to create a complete validation toolkit from scratch:

## 1. Define Artifact Types

First, you define enumerations for each type of artifact your engine will support:

```python
from enum import Enum
from artifact_core.base.artifact_dependencies import ArtifactType

class CustomScoreType(ArtifactType):
    CUSTOM_SCORE = "accuracy_score"
    
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
from typing import List, Dict, Optional
from artifact_core.base.artifact_dependencies import ResourceSpecProtocol

@dataclass
class CustomResourceSpec(ResourceSpecProtocol):
    validation_resource_structural_property: float
```

## 3. Define Resources

Resources are the data objects that artifacts will operate on:

```python
from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union

from artifact_core.base.artifact_dependencies import ArtifactResources

@dataclass
class CustomResources(ArtifactResources):
    resource_attribute: np.ndarray
```

## 4. Create Registries

Registries manage the organization and retrieval of artifacts:

```python
import json
from matplotlib.figure import Figure
import numpy as np
import os
from typing import Dict, Type, Optional, List

from artifact_core.base.registry import ArtifactRegistry

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
class CustomScoreRegistry(ArtifactRegistry[CustomScoreType, CustomResources, float, CustomResourceSpec]):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return load_config_section(
            config_path=CONFIG_PATH,
            section='scores'
            )

# Similar registries for other artifact types...
```

## 5. Implement Artifacts

Create concrete artifact implementations:

```python
from typing import Any, Optional, Union

from artifact_core.base.artifact import Artifact
from artifact_core.base.artifact_dependencies import NoArtifactHyperparams

@CustomScoreRegistry.register_artifact(CustomScoreType.CUSTOM_SCORE)
class CustomScore(Artifact[CustomResources, float, NoArtifactHyperparams, CustomResourceSpec]):
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
from typing import Type, Dict, Union

from artifact_core.base.engine import ArtifactEngine

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