from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, List, Mapping, Optional, Set, Type, TypeVar

from artifact_core.base.artifact import ResourceSpecProtocol
from artifact_core.base.artifact_dependencies import ArtifactResources
from artifact_experiment.base.components.plans.artifact import (
    ArtifactPlan,
    ArtifactPlanBuildContext,
)
from artifact_experiment.base.entities.data_split import DataSplit
from artifact_experiment.base.tracking.background.tracking_queue import TrackingQueue

from artifact_torch.base.components.callbacks.export import ExportCallback
from artifact_torch.base.components.routines.base import PlanExecutionRoutine, RoutineResources
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
ExportTContr = TypeVar("ExportTContr", contravariant=True)
ArtifactRoutineT = TypeVar(
    "ArtifactRoutineT", bound="ArtifactRoutine[Any, Any, Any, Any, Any, Any]"
)


@dataclass
class ArtifactRoutineHyperparams: ...


@dataclass
class ArtifactRoutineData: ...


class ArtifactRoutine(
    PlanExecutionRoutine[ModelTContr],
    Generic[
        ModelTContr,
        ArtifactRoutineHyperparamsTCov,
        ArtifactRoutineDataTContr,
        ArtifactResourcesTContr,
        ResourceSpecProtocolTContr,
        ExportTContr,
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
        export_callback: Optional[ExportCallback[ExportTContr]],
    ):
        self._data = data
        self._periods = KeySelector.restrict_to_keys(periods, keys_from=data)
        self._artifact_plans = KeySelector.restrict_to_keys(artifact_plans, keys_from=data)
        self._hyperparams = hyperparams
        self._export_callback = export_callback
        super().__init__(plans=list(artifact_plans.values()))

    @classmethod
    def build(
        cls: Type[ArtifactRoutineT],
        data: Mapping[DataSplit, ArtifactRoutineDataTContr],
        data_spec: ResourceSpecProtocolTContr,
        tracking_queue: Optional[TrackingQueue] = None,
    ) -> ArtifactRoutineT:
        plan_build_context = ArtifactPlanBuildContext(
            tracking_queue=tracking_queue, resource_spec=data_spec
        )
        periods = {
            data_split: period
            for data_split in DataSplit
            if (period := cls._get_period(data_split=data_split)) is not None
        }
        artifact_plans = {
            data_split: plan
            for data_split in DataSplit
            if (plan := cls._build_artifact_plan(data_split=data_split, context=plan_build_context))
            is not None
        }
        hyperparams = cls._get_hyperparams()
        export_callback = cls._get_export_callback(tracking_queue=tracking_queue)
        routine = cls(
            artifact_plans=artifact_plans,
            data=data,
            periods=periods,
            hyperparams=hyperparams,
            export_callback=export_callback,
        )
        return routine

    @property
    def _data_splits(self) -> List[DataSplit]:
        return list(self._data.keys())

    @classmethod
    @abstractmethod
    def _get_period(cls, data_split: DataSplit) -> Optional[int]: ...

    @classmethod
    @abstractmethod
    def _get_hyperparams(cls) -> ArtifactRoutineHyperparamsTCov: ...

    @classmethod
    @abstractmethod
    def _get_artifact_plan(
        cls, data_split: DataSplit
    ) -> Optional[
        Type[
            ArtifactPlan[
                ArtifactResourcesTContr, ResourceSpecProtocolTContr, Any, Any, Any, Any, Any, Any
            ]
        ]
    ]: ...

    @abstractmethod
    def _generate_artifact_resources(
        self,
        model: ModelTContr,
    ) -> Mapping[DataSplit, ArtifactResourcesTContr]: ...

    @classmethod
    @abstractmethod
    def _get_export_callback(
        cls, tracking_queue: Optional[TrackingQueue]
    ) -> Optional[ExportCallback[ExportTContr]]: ...

    @classmethod
    @abstractmethod
    def _export_artifact_resources(
        cls,
        artifact_resources: ArtifactResourcesTContr,
        export_callback: ExportCallback[ExportTContr],
        n_epochs_elapsed: int,
        data_split: DataSplit,
    ): ...

    def clear_cache(self):
        for validation_plan in self._artifact_plans.values():
            validation_plan.clear_cache()

    def execute(self, resources: RoutineResources[ModelTContr]):
        data_splits = self._get_active_splits(n_epochs_elapsed=resources.n_epochs_elapsed)
        artifact_resources_by_split = self._generate_artifact_resources(model=resources.model)
        for data_split in data_splits:
            artifact_plan = self._artifact_plans[data_split]
            artifact_resources = artifact_resources_by_split[data_split]
            artifact_plan.execute_artifacts(resources=artifact_resources, data_split=data_split)
            if self._export_callback is not None:
                self._export_artifact_resources(
                    artifact_resources=artifact_resources,
                    export_callback=self._export_callback,
                    n_epochs_elapsed=resources.n_epochs_elapsed,
                    data_split=data_split,
                )

    def _get_active_splits(self, n_epochs_elapsed: int) -> Set[DataSplit]:
        return {
            split
            for split in self._data_splits
            if PeriodicActionTrigger.should_trigger(
                step=n_epochs_elapsed, period=self._periods[split]
            )
        }

    @classmethod
    def _build_artifact_plan(
        cls, data_split: DataSplit, context: ArtifactPlanBuildContext[ResourceSpecProtocolTContr]
    ) -> Optional[
        ArtifactPlan[
            ArtifactResourcesTContr, ResourceSpecProtocolTContr, Any, Any, Any, Any, Any, Any
        ]
    ]:
        plan_class = cls._get_artifact_plan(data_split=data_split)
        if plan_class is not None:
            return plan_class.build(context=context)
