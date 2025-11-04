from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, Mapping, Optional, Set, Type, TypeVar

from artifact_core.base.artifact import ResourceSpecProtocol
from artifact_core.base.artifact_dependencies import ArtifactResources
from artifact_experiment.base.entities.data_split import DataSplit
from artifact_experiment.base.plans.artifact import ArtifactPlan
from artifact_experiment.base.tracking.client import TrackingClient
from matplotlib.figure import Figure
from numpy import ndarray

from artifact_torch.base.components.utils.periodic_action_trigger import PeriodicActionTrigger
from artifact_torch.base.model.base import Model
from artifact_torch.libs.utils.key_selector import KeySelector

ModelTContr = TypeVar("ModelTContr", bound=Model[Any, Any], contravariant=True)
ArtifactRoutineHyperparamsTCov = TypeVar(
    "ArtifactRoutineHyperparamsTCov", bound="ArtifactRoutineHyperparams", covariant=True
)
ArtifactRoutineDataTContr = TypeVar(
    "ArtifactRoutineDataTContr", bound="ArtifactRoutineData", contravariant=True
)
ArtifactResourcesTContr = TypeVar(
    "ArtifactResourcesTContr", bound=ArtifactResources, contravariant=True
)
ResourceSpecProtocolTContr = TypeVar(
    "ResourceSpecProtocolTContr", bound=ResourceSpecProtocol, contravariant=True
)
ArtifactRoutineT = TypeVar("ArtifactRoutineT", bound="ArtifactRoutine[Any, Any, Any, Any, Any]")


@dataclass
class ArtifactRoutineHyperparams: ...


@dataclass
class ArtifactRoutineData: ...


class ArtifactRoutine(
    ABC,
    Generic[
        ModelTContr,
        ArtifactRoutineHyperparamsTCov,
        ArtifactRoutineDataTContr,
        ArtifactResourcesTContr,
        ResourceSpecProtocolTContr,
    ],
):
    def __init__(
        self,
        artifact_plans: Mapping[
            DataSplit,
            ArtifactPlan[
                ArtifactResourcesTContr, ResourceSpecProtocolTContr, Any, Any, Any, Any, Any, Any
            ],
        ],
        data: Mapping[DataSplit, ArtifactRoutineDataTContr],
        periods: Mapping[DataSplit, int],
        hyperparams: ArtifactRoutineHyperparamsTCov,
        tracking_client: Optional[TrackingClient] = None,
    ):
        self._artifact_plans = artifact_plans
        self._data = data
        self._periods = periods
        self._hyperparams = hyperparams
        self._tracking_client = tracking_client
        self._data_splits = KeySelector.get_common_keys(
            self._periods,
            self._data,
            self._artifact_plans,
        )

    @classmethod
    def build(
        cls: Type[ArtifactRoutineT],
        data: Mapping[DataSplit, ArtifactRoutineDataTContr],
        data_spec: ResourceSpecProtocolTContr,
        tracking_client: Optional[TrackingClient] = None,
    ) -> ArtifactRoutineT:
        periods = {
            data_split: period
            for data_split in DataSplit
            if (period := cls._get_period(data_split=data_split)) is not None
        }

        artifact_plans = {
            data_split: plan
            for data_split in DataSplit
            if (
                plan := cls._get_artifact_plan(
                    artifact_resource_spec=data_spec,
                    data_split=data_split,
                    tracking_client=tracking_client,
                )
            )
            is not None
        }
        hyperparams = cls._get_hyperparams()
        routine = cls(
            artifact_plans=artifact_plans,
            data=data,
            periods=periods,
            hyperparams=hyperparams,
            tracking_client=tracking_client,
        )
        return routine

    @property
    def scores(self) -> Dict[str, float]:
        return {
            name: value
            for artifact_plan in self._artifact_plans.values()
            for name, value in artifact_plan.scores.items()
        }

    @property
    def arrays(self) -> Dict[str, ndarray]:
        return {
            name: value
            for artifact_plan in self._artifact_plans.values()
            for name, value in artifact_plan.arrays.items()
        }

    @property
    def plots(self) -> Dict[str, Figure]:
        return {
            name: value
            for artifact_plan in self._artifact_plans.values()
            for name, value in artifact_plan.plots.items()
        }

    @property
    def score_collections(self) -> Dict[str, Dict[str, float]]:
        return {
            name: value
            for artifact_plan in self._artifact_plans.values()
            for name, value in artifact_plan.score_collections.items()
        }

    @property
    def array_collections(self) -> Dict[str, Dict[str, ndarray]]:
        return {
            name: value
            for artifact_plan in self._artifact_plans.values()
            for name, value in artifact_plan.array_collections.items()
        }

    @property
    def plot_collections(self) -> Dict[str, Dict[str, Figure]]:
        return {
            name: value
            for artifact_plan in self._artifact_plans.values()
            for name, value in artifact_plan.plot_collections.items()
        }

    @classmethod
    @abstractmethod
    def _get_period(cls, data_split: DataSplit) -> Optional[int]: ...

    @classmethod
    @abstractmethod
    def _get_artifact_plan(
        cls,
        artifact_resource_spec: ResourceSpecProtocolTContr,
        data_split: DataSplit,
        tracking_client: Optional[TrackingClient],
    ) -> Optional[
        ArtifactPlan[
            ArtifactResourcesTContr, ResourceSpecProtocolTContr, Any, Any, Any, Any, Any, Any
        ]
    ]: ...

    @classmethod
    @abstractmethod
    def _get_hyperparams(cls) -> ArtifactRoutineHyperparamsTCov: ...

    @classmethod
    @abstractmethod
    def _export_artifact_resources(
        cls,
        artifact_resources: ArtifactResourcesTContr,
        n_epochs_elapsed: int,
        data_split: DataSplit,
        tracking_client: TrackingClient,
    ): ...

    @abstractmethod
    def _generate_artifact_resources(
        self,
        model: ModelTContr,
    ) -> Mapping[DataSplit, ArtifactResourcesTContr]: ...

    def clear_cache(self):
        for validation_plan in self._artifact_plans.values():
            validation_plan.clear_cache()

    def execute(self, model: ModelTContr, n_epochs_elapsed: int):
        data_splits = self._get_active_splits(n_epochs_elapsed=n_epochs_elapsed)
        artifact_resources_by_split = self._generate_artifact_resources(model=model)
        for data_split in data_splits:
            artifact_plan = self._artifact_plans[data_split]
            artifact_resources = artifact_resources_by_split[data_split]
            artifact_plan.execute_artifacts(resources=artifact_resources, data_split=data_split)
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
            for split in self._data_splits
            if PeriodicActionTrigger.should_trigger(
                step=n_epochs_elapsed, period=self._periods[split]
            )
        }
