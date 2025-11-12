from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Optional, Sequence, Type, TypeVar

from artifact_core._base.orchestration.artifact_type import ArtifactType
from artifact_core._base.primitives import ArtifactResources, ResourceSpecProtocol

from artifact_experiment.base.components.callbacks.artifact import (
    ArtifactArrayCallback,
    ArtifactArrayCollectionCallback,
    ArtifactCallback,
    ArtifactCallbackResources,
    ArtifactPlotCallback,
    ArtifactPlotCollectionCallback,
    ArtifactScoreCallback,
    ArtifactScoreCollectionCallback,
)
from artifact_experiment.base.components.factories.artifact import ArtifactCallbackFactory
from artifact_experiment.base.components.handler_suites.artifact import ArtifactCallbackHandlerSuite
from artifact_experiment.base.components.plans.base import CallbackExecutionPlan, PlanBuildContext
from artifact_experiment.base.entities.data_split import DataSplit
from artifact_experiment.base.tracking.backend.client import TrackingClient
from artifact_experiment.libs.utils.sequence_concatenator import SequenceConcatenator

ResourceSpecProtocolTCov = TypeVar(
    "ResourceSpecProtocolTCov", bound=ResourceSpecProtocol, covariant=True
)


@dataclass(frozen=True)
class ArtifactPlanBuildContext(PlanBuildContext, Generic[ResourceSpecProtocolTCov]):
    resource_spec: ResourceSpecProtocolTCov


ArtifactResourcesTContr = TypeVar(
    "ArtifactResourcesTContr", bound=ArtifactResources, contravariant=True
)
ResourceSpecProtocolTContr = TypeVar(
    "ResourceSpecProtocolTContr", bound=ResourceSpecProtocol, contravariant=True
)
ScoreTypeT = TypeVar("ScoreTypeT", bound=ArtifactType)
ArrayT = TypeVar("ArrayT", bound=ArtifactType)
PlotT = TypeVar("PlotT", bound=ArtifactType)
ScoreCollectionTypeT = TypeVar("ScoreCollectionTypeT", bound=ArtifactType)
ArrayCollectionTypeT = TypeVar("ArrayCollectionTypeT", bound=ArtifactType)
PlotCollectionTypeT = TypeVar("PlotCollectionTypeT", bound=ArtifactType)
ArtifactPlanT = TypeVar("ArtifactPlanT", bound="ArtifactPlan")


class ArtifactPlan(
    CallbackExecutionPlan[
        ArtifactCallbackHandlerSuite[ArtifactResourcesTContr, ResourceSpecProtocolTContr],
        ArtifactCallback[ArtifactResourcesTContr, ResourceSpecProtocolTContr, Any],
        ArtifactCallbackResources[ArtifactResourcesTContr],
        ArtifactPlanBuildContext[ResourceSpecProtocolTContr],
    ],
    Generic[
        ArtifactResourcesTContr,
        ResourceSpecProtocolTContr,
        ScoreTypeT,
        ArrayT,
        PlotT,
        ScoreCollectionTypeT,
        ArrayCollectionTypeT,
        PlotCollectionTypeT,
    ],
):
    @classmethod
    def create(
        cls: Type[ArtifactPlanT],
        resource_spec: ResourceSpecProtocolTContr,
        tracking_client: Optional[TrackingClient] = None,
    ) -> ArtifactPlanT:
        tracking_queue = tracking_client.queue if tracking_client is not None else None
        context = ArtifactPlanBuildContext(
            tracking_queue=tracking_queue, resource_spec=resource_spec
        )
        plan = cls.build(context=context)
        return plan

    @staticmethod
    @abstractmethod
    def _get_callback_factory() -> Type[
        ArtifactCallbackFactory[
            ArtifactResourcesTContr,
            ResourceSpecProtocolTContr,
            ScoreTypeT,
            ArrayT,
            PlotT,
            ScoreCollectionTypeT,
            ArrayCollectionTypeT,
            PlotCollectionTypeT,
        ]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_score_types() -> Sequence[ScoreTypeT]: ...

    @staticmethod
    @abstractmethod
    def _get_array_types() -> Sequence[ArrayT]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_types() -> Sequence[PlotT]: ...

    @staticmethod
    @abstractmethod
    def _get_score_collection_types() -> Sequence[ScoreCollectionTypeT]: ...

    @staticmethod
    @abstractmethod
    def _get_array_collection_types() -> Sequence[ArrayCollectionTypeT]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_collection_types() -> Sequence[PlotCollectionTypeT]: ...

    @staticmethod
    def _get_custom_score_types() -> Sequence[str]:
        return []

    @staticmethod
    def _get_custom_array_types() -> Sequence[str]:
        return []

    @staticmethod
    def _get_custom_plot_types() -> Sequence[str]:
        return []

    @staticmethod
    def _get_custom_score_collection_types() -> Sequence[str]:
        return []

    @staticmethod
    def _get_custom_array_collection_types() -> Sequence[str]:
        return []

    @staticmethod
    def _get_custom_plot_collection_types() -> Sequence[str]:
        return []

    def execute_artifacts(
        self, resources: ArtifactResourcesTContr, data_split: Optional[DataSplit] = None
    ):
        callback_resources = ArtifactCallbackResources[ArtifactResourcesTContr](
            artifact_resources=resources, data_split=data_split
        )
        self.execute(resources=callback_resources)

    @classmethod
    def _get_handler_suite(
        cls,
    ) -> Type[ArtifactCallbackHandlerSuite[ArtifactResourcesTContr, ResourceSpecProtocolTContr]]:
        return ArtifactCallbackHandlerSuite

    @classmethod
    def _get_score_callbacks(
        cls, context: ArtifactPlanBuildContext[ResourceSpecProtocolTContr]
    ) -> Sequence[ArtifactScoreCallback[ArtifactResourcesTContr, ResourceSpecProtocolTContr]]:
        callback_factory = cls._get_callback_factory()
        ls_score_types = SequenceConcatenator.concatenate(
            cls._get_score_types(), cls._get_custom_score_types()
        )
        ls_callbacks = [
            callback_factory.build_score_callback(
                score_type=score_type,
                resource_spec=context.resource_spec,
                writer=context.score_writer,
            )
            for score_type in ls_score_types
        ]
        return ls_callbacks

    @classmethod
    def _get_array_callbacks(
        cls, context: ArtifactPlanBuildContext[ResourceSpecProtocolTContr]
    ) -> Sequence[ArtifactArrayCallback[ArtifactResourcesTContr, ResourceSpecProtocolTContr]]:
        callback_factory = cls._get_callback_factory()
        ls_array_types = SequenceConcatenator.concatenate(
            cls._get_array_types(), cls._get_custom_array_types()
        )
        ls_callbacks = [
            callback_factory.build_array_callback(
                array_type=array_type,
                resource_spec=context.resource_spec,
                writer=context.array_writer,
            )
            for array_type in ls_array_types
        ]
        return ls_callbacks

    @classmethod
    def _get_plot_callbacks(
        cls, context: ArtifactPlanBuildContext[ResourceSpecProtocolTContr]
    ) -> Sequence[ArtifactPlotCallback[ArtifactResourcesTContr, ResourceSpecProtocolTContr]]:
        callback_factory = cls._get_callback_factory()
        ls_plot_types = SequenceConcatenator.concatenate(
            cls._get_plot_types(), cls._get_custom_plot_types()
        )
        ls_callbacks = [
            callback_factory.build_plot_callback(
                plot_type=plot_type, resource_spec=context.resource_spec, writer=context.plot_writer
            )
            for plot_type in ls_plot_types
        ]
        return ls_callbacks

    @classmethod
    def _get_score_collection_callbacks(
        cls, context: ArtifactPlanBuildContext[ResourceSpecProtocolTContr]
    ) -> Sequence[
        ArtifactScoreCollectionCallback[ArtifactResourcesTContr, ResourceSpecProtocolTContr]
    ]:
        callback_factory = cls._get_callback_factory()
        ls_score_collection_types = SequenceConcatenator.concatenate(
            cls._get_score_collection_types(), cls._get_custom_score_collection_types()
        )
        ls_callbacks = [
            callback_factory.build_score_collection_callback(
                score_collection_type=score_collection_type,
                resource_spec=context.resource_spec,
                writer=context.score_collection_writer,
            )
            for score_collection_type in ls_score_collection_types
        ]
        return ls_callbacks

    @classmethod
    def _get_array_collection_callbacks(
        cls, context: ArtifactPlanBuildContext[ResourceSpecProtocolTContr]
    ) -> Sequence[
        ArtifactArrayCollectionCallback[ArtifactResourcesTContr, ResourceSpecProtocolTContr]
    ]:
        callback_factory = cls._get_callback_factory()
        ls_array_collection_types = SequenceConcatenator.concatenate(
            cls._get_array_collection_types(), cls._get_custom_array_collection_types()
        )
        ls_callbacks = [
            callback_factory.build_array_collection_callback(
                array_collection_type=array_collection_type,
                resource_spec=context.resource_spec,
                writer=context.array_collection_writer,
            )
            for array_collection_type in ls_array_collection_types
        ]
        return ls_callbacks

    @classmethod
    def _get_plot_collection_callbacks(
        cls, context: ArtifactPlanBuildContext[ResourceSpecProtocolTContr]
    ) -> Sequence[
        ArtifactPlotCollectionCallback[ArtifactResourcesTContr, ResourceSpecProtocolTContr]
    ]:
        callback_factory = cls._get_callback_factory()
        ls_plot_collection_types = SequenceConcatenator.concatenate(
            cls._get_plot_collection_types(), cls._get_custom_plot_collection_types()
        )
        ls_callbacks = [
            callback_factory.build_plot_collection_callback(
                plot_collection_type=plot_collection_type,
                resource_spec=context.resource_spec,
                writer=context.plot_collection_writer,
            )
            for plot_collection_type in ls_plot_collection_types
        ]
        return ls_callbacks
