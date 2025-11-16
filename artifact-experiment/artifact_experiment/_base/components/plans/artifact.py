from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Optional, Sequence, Type, TypeVar

from artifact_core.shared.collections import SequenceConcatenator
from artifact_core.spi.orchestration import ArtifactType
from artifact_core.spi.resources import ArtifactResources, ResourceSpecProtocol

from artifact_experiment._base.components.callbacks.artifact import (
    ArtifactArrayCallback,
    ArtifactArrayCollectionCallback,
    ArtifactCallback,
    ArtifactCallbackResources,
    ArtifactPlotCallback,
    ArtifactPlotCollectionCallback,
    ArtifactScoreCallback,
    ArtifactScoreCollectionCallback,
)
from artifact_experiment._base.components.callbacks.export import ExportCallback
from artifact_experiment._base.components.factories.artifact import ArtifactCallbackFactory
from artifact_experiment._base.components.handler_suites.artifact import (
    ArtifactCallbackHandlerSuite,
)
from artifact_experiment._base.components.plans.base import CallbackExecutionPlan, PlanBuildContext
from artifact_experiment._base.components.resources.export import ExportCallbackResources
from artifact_experiment._base.primitives.data_split import DataSplit
from artifact_experiment._base.tracking.backend.client import TrackingClient

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
ArrayTypeT = TypeVar("ArrayTypeT", bound=ArtifactType)
PlotTypeT = TypeVar("PlotTypeT", bound=ArtifactType)
ScoreCollectionTypeT = TypeVar("ScoreCollectionTypeT", bound=ArtifactType)
ArrayCollectionTypeT = TypeVar("ArrayCollectionTypeT", bound=ArtifactType)
PlotCollectionTypeT = TypeVar("PlotCollectionTypeT", bound=ArtifactType)
ResourceExportT = TypeVar("ResourceExportT")
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
        ArrayTypeT,
        PlotTypeT,
        ScoreCollectionTypeT,
        ArrayCollectionTypeT,
        PlotCollectionTypeT,
        ResourceExportT,
    ],
):
    def __init__(
        self,
        handler_suite: ArtifactCallbackHandlerSuite[
            ArtifactResourcesTContr, ResourceSpecProtocolTContr
        ],
        context: ArtifactPlanBuildContext[ResourceSpecProtocolTContr],
    ):
        self._handler_suite = handler_suite
        self._context = context
        self._export_callback = self._get_export_callback(context=context)

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
            ArrayTypeT,
            PlotTypeT,
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
    def _get_array_types() -> Sequence[ArrayTypeT]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_types() -> Sequence[PlotTypeT]: ...

    @staticmethod
    @abstractmethod
    def _get_score_collection_types() -> Sequence[ScoreCollectionTypeT]: ...

    @staticmethod
    @abstractmethod
    def _get_array_collection_types() -> Sequence[ArrayCollectionTypeT]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_collection_types() -> Sequence[PlotCollectionTypeT]: ...

    @classmethod
    @abstractmethod
    def _get_export_callback(
        cls, context: ArtifactPlanBuildContext[ResourceSpecProtocolTContr]
    ) -> Optional[ExportCallback[ExportCallbackResources[ResourceExportT]]]: ...

    @classmethod
    @abstractmethod
    def _get_export_resources(
        cls, resources: ArtifactCallbackResources[ArtifactResourcesTContr]
    ) -> ExportCallbackResources[ResourceExportT]: ...

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

    def execute(self, resources: ArtifactCallbackResources[ArtifactResourcesTContr]):
        super().execute(resources=resources)
        self._export_resources(resources=resources)

    def execute_artifacts(
        self, resources: ArtifactResourcesTContr, data_split: Optional[DataSplit] = None
    ):
        callback_resources = ArtifactCallbackResources[ArtifactResourcesTContr](
            artifact_resources=resources, data_split=data_split
        )
        self.execute(resources=callback_resources)

    def _export_resources(self, resources: ArtifactCallbackResources[ArtifactResourcesTContr]):
        export_resources = self._get_export_resources(resources=resources)
        if self._export_callback is not None:
            self._export_callback.execute(resources=export_resources)

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
