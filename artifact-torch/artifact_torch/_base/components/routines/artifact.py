from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, List, Mapping, Optional, Set, Type, TypeVar

from artifact_core.spi.resources import ArtifactResources, ResourceSpecProtocol
from artifact_experiment.spi.plans import ArtifactPlan, ArtifactPlanBuildContext
from artifact_experiment.tracking import DataSplit
from artifact_experiment.tracking.spi import TrackingQueue

from artifact_torch._base.components.routines.routine import PlanExecutionRoutine, RoutineResources
from artifact_torch._base.model.base import Model
from artifact_torch._utils.collections.key_selector import KeySelector
from artifact_torch._utils.scheduling.periodic_actions import PeriodicActionTrigger

ModelTContr = TypeVar("ModelTContr", bound=Model[Any, Any], contravariant=True)
ArtifactRoutineHyperparamsTCov = TypeVar(
    "ArtifactRoutineHyperparamsTCov", bound="ArtifactRoutineHyperparams", covariant=True
)
ArtifactRoutineDataTContr = TypeVar(
    "ArtifactRoutineDataTContr", bound="ArtifactRoutineData", contravariant=True
)
ArtifactResourcesTCov = TypeVar("ArtifactResourcesTCov", bound=ArtifactResources, covariant=True)
ResourceSpecProtocolTContr = TypeVar(
    "ResourceSpecProtocolTContr", bound=ResourceSpecProtocol, contravariant=True
)
ExportTContr = TypeVar("ExportTContr", contravariant=True)
ArtifactRoutineT = TypeVar("ArtifactRoutineT", bound="ArtifactRoutine[Any, Any, Any, Any, Any]")


@dataclass
class ArtifactRoutineHyperparams: ...


@dataclass
class ArtifactRoutineData: ...


class ArtifactRoutine(
    PlanExecutionRoutine[ModelTContr],
    Generic[
        ModelTContr,
        ArtifactRoutineDataTContr,
        ArtifactRoutineHyperparamsTCov,
        ArtifactResourcesTCov,
        ResourceSpecProtocolTContr,
    ],
):
    def __init__(
        self,
        data: Mapping[DataSplit, ArtifactRoutineDataTContr],
        periods: Mapping[DataSplit, int],
        artifact_plans: Mapping[
            DataSplit,
            ArtifactPlan[
                ArtifactResourcesTCov,
                ResourceSpecProtocolTContr,
                Any,
                Any,
                Any,
                Any,
                Any,
                Any,
                Any,
            ],
        ],
        hyperparams: ArtifactRoutineHyperparamsTCov,
    ):
        periods = KeySelector.restrict_to_keys(periods, keys_from=data)
        artifact_plans = KeySelector.restrict_to_keys(artifact_plans, keys_from=data)
        super().__init__(plans=list(artifact_plans.values()))
        self._data = data
        self._periods = periods
        self._artifact_plans = artifact_plans
        self._hyperparams = hyperparams

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
        routine = cls(
            data=data, periods=periods, artifact_plans=artifact_plans, hyperparams=hyperparams
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
    def _get_artifact_plan(
        cls, data_split: DataSplit
    ) -> Optional[
        Type[
            ArtifactPlan[
                ArtifactResourcesTCov,
                ResourceSpecProtocolTContr,
                Any,
                Any,
                Any,
                Any,
                Any,
                Any,
                Any,
            ]
        ]
    ]: ...

    @classmethod
    @abstractmethod
    def _get_hyperparams(cls) -> ArtifactRoutineHyperparamsTCov: ...

    @abstractmethod
    def _generate_artifact_resources(
        self,
        model: ModelTContr,
    ) -> Mapping[DataSplit, ArtifactResourcesTCov]: ...

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
            ArtifactResourcesTCov, ResourceSpecProtocolTContr, Any, Any, Any, Any, Any, Any, Any
        ]
    ]:
        plan_class = cls._get_artifact_plan(data_split=data_split)
        if plan_class is not None:
            return plan_class.build(context=context)
