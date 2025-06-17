from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, Optional, Type, TypeVar

from artifact_core.base.artifact import ResourceSpecProtocol
from artifact_core.base.artifact_dependencies import (
    ArtifactResources,
)
from artifact_experiment.base.callbacks.artifact import ArtifactCallbackResources
from artifact_experiment.base.tracking.client import TrackingClient
from artifact_experiment.base.validation_plan import ValidationPlan
from matplotlib.figure import Figure
from numpy import ndarray

from artifact_torch.base.components.utils.periodic_action_trigger import PeriodicActionTrigger
from artifact_torch.base.model.base import Model

ModelTContr = TypeVar("ModelTContr", bound=Model, contravariant=True)
ArtifactRoutineHyperparamsT = TypeVar(
    "ArtifactRoutineHyperparamsT", bound="ArtifactRoutineHyperparams"
)
ArtifactRoutineDataT = TypeVar("ArtifactRoutineDataT", bound="ArtifactRoutineData")
ArtifactResourcesT = TypeVar("ArtifactResourcesT", bound=ArtifactResources)
ResourceSpecProtocolT = TypeVar("ResourceSpecProtocolT", bound=ResourceSpecProtocol)
ArtifactValidationRoutineT = TypeVar(
    "ArtifactValidationRoutineT", bound="ArtifactValidationRoutine"
)


@dataclass
class ArtifactRoutineHyperparams: ...


@dataclass
class ArtifactRoutineData: ...


class ArtifactValidationRoutine(
    ABC,
    Generic[
        ModelTContr,
        ArtifactRoutineHyperparamsT,
        ArtifactRoutineDataT,
        ArtifactResourcesT,
        ResourceSpecProtocolT,
    ],
):
    def __init__(
        self,
        period: int,
        data: ArtifactRoutineDataT,
        hyperparams: ArtifactRoutineHyperparamsT,
        validation_plan: ValidationPlan[
            Any, Any, Any, Any, Any, Any, ArtifactResourcesT, ResourceSpecProtocolT
        ],
        tracking_client: Optional[TrackingClient] = None,
    ):
        self._period = period
        self._data = data
        self._hyperparams = hyperparams
        self._validation_plan = validation_plan
        self._tracking_client = tracking_client

    @property
    def scores(self) -> Dict[str, float]:
        return self._validation_plan.scores

    @property
    def arrays(self) -> Dict[str, ndarray]:
        return self._validation_plan.arrays

    @property
    def plots(self) -> Dict[str, Figure]:
        return self._validation_plan.plots

    @property
    def score_collections(self) -> Dict[str, Dict[str, float]]:
        return self._validation_plan.score_collections

    @property
    def array_collections(self) -> Dict[str, Dict[str, ndarray]]:
        return self._validation_plan.array_collections

    @property
    def plot_collections(self) -> Dict[str, Dict[str, Figure]]:
        return self._validation_plan.plot_collections

    @classmethod
    @abstractmethod
    def _get_period(cls) -> int: ...

    @classmethod
    @abstractmethod
    def _get_hyperparams(cls) -> ArtifactRoutineHyperparamsT: ...

    @classmethod
    @abstractmethod
    def _get_validation_plan(
        cls,
        artifact_resource_spec: ResourceSpecProtocolT,
        tracking_client: Optional[TrackingClient],
    ) -> ValidationPlan[
        Any, Any, Any, Any, Any, Any, ArtifactResourcesT, ResourceSpecProtocolT
    ]: ...

    @classmethod
    @abstractmethod
    def _export_artifact_resources(
        cls,
        artifact_resources: ArtifactResourcesT,
        n_epochs_elapsed: int,
        tracking_client: TrackingClient,
    ): ...

    @classmethod
    @abstractmethod
    def _generate_artifact_resources(
        cls,
        model: ModelTContr,
        hyperparams: ArtifactRoutineHyperparamsT,
        data: ArtifactRoutineDataT,
    ) -> ArtifactResourcesT: ...

    @classmethod
    def _generate_artifact_callback_resources(
        cls,
        artifact_resources: ArtifactResourcesT,
    ) -> ArtifactCallbackResources[ArtifactResourcesT]:
        return ArtifactCallbackResources(artifact_resources=artifact_resources)

    def clear_cache(self):
        self._validation_plan.clear_cache()

    def execute(self, model: ModelTContr, n_epochs_elapsed: int):
        if PeriodicActionTrigger.should_trigger(step=n_epochs_elapsed, period=self._period):
            artifact_resources = self._generate_artifact_resources(
                model=model, hyperparams=self._hyperparams, data=self._data
            )
            artifact_callback_resources = self._generate_artifact_callback_resources(
                artifact_resources=artifact_resources
            )
            self._validation_plan.execute(resources=artifact_callback_resources)
            if self._tracking_client is not None:
                self._export_artifact_resources(
                    artifact_resources=artifact_resources,
                    n_epochs_elapsed=n_epochs_elapsed,
                    tracking_client=self._tracking_client,
                )

    @classmethod
    def _build(
        cls: Type[ArtifactValidationRoutineT],
        data: ArtifactRoutineDataT,
        artifact_resource_spec: ResourceSpecProtocolT,
        tracking_client: Optional[TrackingClient] = None,
    ) -> ArtifactValidationRoutineT:
        period = cls._get_period()
        validation_plan = cls._get_validation_plan(
            artifact_resource_spec=artifact_resource_spec, tracking_client=tracking_client
        )
        hyperparams = cls._get_hyperparams()
        routine = cls(
            period=period,
            data=data,
            hyperparams=hyperparams,
            validation_plan=validation_plan,
            tracking_client=tracking_client,
        )
        return routine
