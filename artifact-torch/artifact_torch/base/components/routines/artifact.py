from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, Mapping, Optional, Set, Type, TypeVar

from artifact_core.base.artifact import ResourceSpecProtocol
from artifact_core.base.artifact_dependencies import ArtifactResources
from artifact_experiment.base.callbacks.artifact import ArtifactCallbackResources
from artifact_experiment.base.data_split import DataSplit
from artifact_experiment.base.tracking.client import TrackingClient
from artifact_experiment.base.validation_plan import ValidationPlan
from matplotlib.figure import Figure
from numpy import ndarray

from artifact_torch.base.components.utils.periodic_action_trigger import PeriodicActionTrigger
from artifact_torch.base.model.base import Model
from artifact_torch.libs.utils.key_selector import KeySelector

ModelTContr = TypeVar("ModelTContr", bound=Model, contravariant=True)
ArtifactRoutineHyperparamsT = TypeVar(
    "ArtifactRoutineHyperparamsT", bound="ArtifactRoutineHyperparams"
)
ArtifactRoutineDataT = TypeVar("ArtifactRoutineDataT", bound="ArtifactRoutineData")
ArtifactResourcesT = TypeVar("ArtifactResourcesT", bound=ArtifactResources)
ResourceSpecProtocolT = TypeVar("ResourceSpecProtocolT", bound=ResourceSpecProtocol)
ArtifactRoutineT = TypeVar("ArtifactRoutineT", bound="ArtifactRoutine")


@dataclass
class ArtifactRoutineHyperparams: ...


@dataclass
class ArtifactRoutineData: ...


class ArtifactRoutine(
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
        periods: Mapping[DataSplit, int],
        data: Mapping[DataSplit, ArtifactRoutineDataT],
        validation_plans: Mapping[
            DataSplit,
            ValidationPlan[Any, Any, Any, Any, Any, Any, ArtifactResourcesT, ResourceSpecProtocolT],
        ],
        hyperparams: ArtifactRoutineHyperparamsT,
        tracking_client: Optional[TrackingClient] = None,
    ):
        self._periods = periods
        self._data = data
        self._validation_plans = validation_plans
        self._hyperparams = hyperparams
        self._tracking_client = tracking_client
        self._splits = KeySelector.get_common_keys(
            self._periods,
            self._data,
            self._validation_plans,
        )

    @classmethod
    def build(
        cls: Type[ArtifactRoutineT],
        data: Mapping[DataSplit, ArtifactRoutineDataT],
        data_spec: ResourceSpecProtocolT,
        tracking_client: Optional[TrackingClient] = None,
    ) -> ArtifactRoutineT:
        periods = cls._get_periods()
        validation_plans = cls._get_validation_plans(
            artifact_resource_spec=data_spec, tracking_client=tracking_client
        )
        hyperparams = cls._get_hyperparams()
        routine = cls(
            periods=periods,
            data=data,
            validation_plans=validation_plans,
            hyperparams=hyperparams,
            tracking_client=tracking_client,
        )
        return routine

    @property
    def scores(self) -> Dict[str, float]:
        return {
            name: value
            for validation_plan in self._validation_plans.values()
            for name, value in validation_plan.scores.items()
        }

    @property
    def arrays(self) -> Dict[str, ndarray]:
        return {
            name: value
            for validation_plan in self._validation_plans.values()
            for name, value in validation_plan.arrays.items()
        }

    @property
    def plots(self) -> Dict[str, Figure]:
        return {
            name: value
            for validation_plan in self._validation_plans.values()
            for name, value in validation_plan.plots.items()
        }

    @property
    def score_collections(self) -> Dict[str, Dict[str, float]]:
        return {
            name: value
            for validation_plan in self._validation_plans.values()
            for name, value in validation_plan.score_collections.items()
        }

    @property
    def array_collections(self) -> Dict[str, Dict[str, ndarray]]:
        return {
            name: value
            for validation_plan in self._validation_plans.values()
            for name, value in validation_plan.array_collections.items()
        }

    @property
    def plot_collections(self) -> Dict[str, Dict[str, Figure]]:
        return {
            name: value
            for validation_plan in self._validation_plans.values()
            for name, value in validation_plan.plot_collections.items()
        }

    @classmethod
    @abstractmethod
    def _get_periods(cls) -> Mapping[DataSplit, int]: ...

    @classmethod
    @abstractmethod
    def _get_validation_plans(
        cls,
        artifact_resource_spec: ResourceSpecProtocolT,
        tracking_client: Optional[TrackingClient],
    ) -> Mapping[
        DataSplit,
        ValidationPlan[Any, Any, Any, Any, Any, Any, ArtifactResourcesT, ResourceSpecProtocolT],
    ]: ...

    @classmethod
    @abstractmethod
    def _get_hyperparams(cls) -> ArtifactRoutineHyperparamsT: ...

    @classmethod
    @abstractmethod
    def _export_artifact_resources(
        cls,
        artifact_resources: ArtifactResourcesT,
        n_epochs_elapsed: int,
        data_split: DataSplit,
        tracking_client: TrackingClient,
    ): ...

    @classmethod
    @abstractmethod
    def _generate_artifact_resources(
        cls,
        model: ModelTContr,
        hyperparams: ArtifactRoutineHyperparamsT,
        data: Mapping[DataSplit, ArtifactRoutineDataT],
    ) -> Mapping[DataSplit, ArtifactResourcesT]: ...

    def clear_cache(self):
        for validation_plan in self._validation_plans.values():
            validation_plan.clear_cache()

    def execute(self, model: ModelTContr, n_epochs_elapsed: int):
        splits = self._get_active_splits(n_epochs_elapsed=n_epochs_elapsed)
        artifact_resources_by_split = self._generate_artifact_resources(
            model=model, hyperparams=self._hyperparams, data=self._data
        )
        for data_split in splits:
            validation_plan = self._validation_plans[data_split]
            artifact_resources = artifact_resources_by_split[data_split]
            artifact_callback_resources = ArtifactCallbackResources(
                artifact_resources=artifact_resources
            )
            validation_plan.execute(resources=artifact_callback_resources)
            if self._tracking_client is not None:
                self._export_artifact_resources(
                    artifact_resources=artifact_resources,
                    n_epochs_elapsed=n_epochs_elapsed,
                    data_split=data_split,
                    tracking_client=self._tracking_client,
                )

    def _get_active_splits(self, n_epochs_elapsed: int) -> Set[DataSplit]:
        return {
            split
            for split in self._splits
            if PeriodicActionTrigger.should_trigger(
                step=n_epochs_elapsed, period=self._periods[split]
            )
        }
