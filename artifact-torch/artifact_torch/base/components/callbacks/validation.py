from abc import abstractmethod
from typing import Any, Generic, TypeVar

from artifact_core.base.artifact_dependencies import (
    ArtifactResources,
)
from artifact_experiment.base.callbacks.artifact import ArtifactCallbackResources
from artifact_experiment.base.validation_plan import ValidationPlan

from artifact_torch.base.components.callbacks.periodic import (
    PeriodicCallback,
    PeriodicCallbackResources,
)
from artifact_torch.base.model.base import Model

ModelT = TypeVar("ModelT", bound=Model)
ValidationDataT = TypeVar("ValidationDataT")
ArtifactResourcesT = TypeVar("ArtifactResourcesT", bound=ArtifactResources)
ValidationCallbackResourcesT = TypeVar(
    "ValidationCallbackResourcesT", bound=PeriodicCallbackResources
)


class ValidationCallback(
    PeriodicCallback[ValidationCallbackResourcesT,],
    Generic[ValidationCallbackResourcesT, ArtifactResourcesT],
):
    def __init__(
        self,
        period: int,
        validation_plan: ValidationPlan[Any, Any, Any, Any, Any, Any, ArtifactResourcesT, Any],
    ):
        super().__init__(period=period)
        self._validation_plan = validation_plan

    @abstractmethod
    def _generate_artifact_resources(
        self,
        resources: ValidationCallbackResourcesT,
    ) -> ArtifactResourcesT: ...

    def _compute(self, resources: ValidationCallbackResourcesT):
        artifact_resources = self._generate_artifact_resources(resources=resources)
        artifact_callback_resources = ArtifactCallbackResources(
            artifact_resources=artifact_resources
        )
        self._validation_plan.execute(resources=artifact_callback_resources)
