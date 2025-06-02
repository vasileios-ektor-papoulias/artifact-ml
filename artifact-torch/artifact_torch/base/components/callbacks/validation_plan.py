from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

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
ValidationPlanT = TypeVar("ValidationPlanT", bound=ValidationPlan)
ArtifactResourcesT = TypeVar("ArtifactResourcesT", bound=ArtifactResources)


@dataclass
class ValidationPlanCallbackResources(PeriodicCallbackResources, Generic[ModelT]):
    model: ModelT


class ValidationPlanCallback(
    PeriodicCallback[ValidationPlanCallbackResources],
    Generic[ModelT, ValidationPlanT, ArtifactResourcesT],
):
    def __init__(
        self,
        period: int,
        validation_plan: ValidationPlanT,
    ):
        super().__init__(period=period)
        self._validation_plan = validation_plan

    @abstractmethod
    def _generate_artifact_resources(
        self,
        model: ModelT,
    ) -> ArtifactResourcesT: ...

    def _execute(self, resources: ValidationPlanCallbackResources):
        artifact_resources = self._generate_artifact_resources(model=resources.model)
        artifact_callback_resources = ArtifactCallbackResources(
            artifact_resources=artifact_resources
        )
        self._validation_plan.execute(resources=artifact_callback_resources)
