# Development Guide

<p align="center">
  <img src="../assets/artifact_ml_logo.svg" width="200" alt="Artifact-ML Logo">
</p>


## Creating ValidationPlans for New Domains

Each ArtifactEngine in `artifact-core` should have a corresponding ValidationPlan in `artifact-experiment`. When contributing new artifact types to `artifact-core`, extend `artifact-experiment` with the corresponding validation plan:

```python
from artifact_experiment.base.validation_plan import ValidationPlan

class NewDomainValidationPlan(ValidationPlan[...]):
    @staticmethod
    def _get_callback_factory() -> Type[NewDomainCallbackFactory]:
        return NewDomainCallbackFactory
        
    # Implement required artifact type methods...
```

## Adding New Tracking Backends

To support a new experiment tracking backend:

1. **Create RunAdapter**: Normalize the backend's native run object
2. **Create TrackingClient**: Implement the unified tracking interface  
3. **Implement ArtifactLoggers**: Handle backend-specific artifact export

```python
from artifact_experiment.base.tracking.adapter import RunAdapter
from artifact_experiment.base.tracking.client import TrackingClient
from artifact_experiment.base.tracking.logger import ArtifactLogger

# 1. Create RunAdapter
class MyBackendRunAdapter(RunAdapter[MyNativeRunType]):
    def upload(self, path_source: str, dir_target: str):
        # Implement backend-specific file upload
        pass
    
    def stop(self):
        # Implement run termination logic
        pass

# 2. Create TrackingClient
class MyBackendTrackingClient(TrackingClient[MyBackendRunAdapter]):
    @staticmethod
    def _get_score_logger(run: MyBackendRunAdapter) -> ArtifactLogger[float, MyBackendRunAdapter]:
        return MyBackendScoreLogger(run=run)
    
    # Implement other logger getters...

# 3. Implement ArtifactLoggers  
class MyBackendScoreLogger(ArtifactLogger[float, MyBackendRunAdapter]):
    def log(self, artifact: float, artifact_name: str):
        # Implement backend-specific score logging
        pass
```