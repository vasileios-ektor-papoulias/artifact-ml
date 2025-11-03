from abc import abstractmethod
from typing import Any, Generic, List, Optional, Sequence, Type, TypeVar, Union

from artifact_core.base.artifact_dependencies import ArtifactResources, ResourceSpecProtocol
from artifact_core.base.registry import ArtifactType

from artifact_experiment.base.callbacks.artifact import (
    ArtifactArrayCallback,
    ArtifactArrayCollectionCallback,
    ArtifactCallback,
    ArtifactCallbackHandler,
    ArtifactCallbackResources,
    ArtifactHandlerSuite,
    ArtifactPlotCallback,
    ArtifactPlotCollectionCallback,
    ArtifactScoreCallback,
    ArtifactScoreCollectionCallback,
)
from artifact_experiment.base.entities.data_split import DataSplit
from artifact_experiment.base.plans.base import CallbackExecutionPlan
from artifact_experiment.base.plans.callback_factory import ArtifactCallbackFactory
from artifact_experiment.base.tracking.client import TrackingClient
from artifact_experiment.libs.utils.sequence_concatenator import SequenceConcatenator

ArtifactResourcesT = TypeVar("ArtifactResourcesT", bound=ArtifactResources)
ResourceSpecProtocolT = TypeVar("ResourceSpecProtocolT", bound=ResourceSpecProtocol)
ScoreTypeT = TypeVar("ScoreTypeT", bound=ArtifactType)
ArrayTypeT = TypeVar("ArrayTypeT", bound=ArtifactType)
PlotTypeT = TypeVar("PlotTypeT", bound=ArtifactType)
ScoreCollectionTypeT = TypeVar("ScoreCollectionTypeT", bound=ArtifactType)
ArrayCollectionTypeT = TypeVar("ArrayCollectionTypeT", bound=ArtifactType)
PlotCollectionTypeT = TypeVar("PlotCollectionTypeT", bound=ArtifactType)
ArtifactPlanT = TypeVar("ArtifactPlanT", bound="ArtifactPlan")


class ArtifactPlan(
    CallbackExecutionPlan[
        ArtifactCallbackHandler[
            ArtifactResourcesT,
            ResourceSpecProtocolT,
            Any,
        ],
        ArtifactCallback[ArtifactResourcesT, ResourceSpecProtocolT, Any],
        ArtifactCallbackResources[ArtifactResourcesT],
    ],
    Generic[
        ArtifactResourcesT,
        ResourceSpecProtocolT,
        ScoreTypeT,
        ArrayTypeT,
        PlotTypeT,
        ScoreCollectionTypeT,
        ArrayCollectionTypeT,
        PlotCollectionTypeT,
    ],
):
    def __init__(
        self,
        resource_spec: ResourceSpecProtocolT,
        callback_handlers: ArtifactHandlerSuite[ArtifactResourcesT, ResourceSpecProtocolT],
        data_split: Optional[DataSplit] = None,
        tracking_client: Optional[TrackingClient] = None,
    ):
        self._resource_spec = resource_spec
        super().__init__(
            callback_handlers=callback_handlers,
            data_split=data_split,
            tracking_client=tracking_client,
        )

    @classmethod
    def build(
        cls: Type[ArtifactPlanT],
        resource_spec: ResourceSpecProtocolT,
        data_split: Optional[DataSplit] = None,
        tracking_client: Optional[TrackingClient] = None,
    ) -> ArtifactPlanT:
        score_callbacks = cls._get_score_callbacks(
            resource_spec=resource_spec, data_split=data_split
        )
        array_callbacks = cls._get_array_callbacks(
            resource_spec=resource_spec, data_split=data_split
        )
        plot_callbacks = cls._get_plot_callbacks(resource_spec=resource_spec, data_split=data_split)
        score_collection_callbacks = cls._get_score_collection_callbacks(
            resource_spec=resource_spec, data_split=data_split
        )
        array_collection_callbacks = cls._get_array_collection_callbacks(
            resource_spec=resource_spec, data_split=data_split
        )
        plot_collection_callbacks = cls._get_plot_collection_callbacks(
            resource_spec=resource_spec, data_split=data_split
        )
        callback_handlers = ArtifactHandlerSuite.build(
            score_callbacks=score_callbacks,
            array_callbacks=array_callbacks,
            plot_callbacks=plot_callbacks,
            score_collection_callbacks=score_collection_callbacks,
            array_collection_callbacks=array_collection_callbacks,
            plot_collection_callbacks=plot_collection_callbacks,
            tracking_client=tracking_client,
        )
        plan = cls(
            resource_spec=resource_spec,
            callback_handlers=callback_handlers,
            data_split=data_split,
            tracking_client=tracking_client,
        )
        return plan

    @staticmethod
    @abstractmethod
    def _get_callback_factory() -> Type[
        ArtifactCallbackFactory[
            ArtifactResourcesT,
            ResourceSpecProtocolT,
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

    def execute_artifacts(self, resources: ArtifactResourcesT):
        callback_resources = ArtifactCallbackResources[ArtifactResourcesT](
            artifact_resources=resources
        )
        self.execute(resources=callback_resources)

    @classmethod
    def _get_score_callbacks(
        cls,
        resource_spec: ResourceSpecProtocolT,
        ls_score_types: Optional[List[Union[ScoreTypeT, str]]] = None,
        data_split: Optional[DataSplit] = None,
    ) -> Sequence[ArtifactScoreCallback[ArtifactResourcesT, ResourceSpecProtocolT]]:
        callback_factory = cls._get_callback_factory()
        if ls_score_types is None:
            ls_score_types = SequenceConcatenator.concatenate(
                cls._get_score_types(), cls._get_custom_score_types()
            )
        ls_callbacks = [
            callback_factory.build_score_callback(
                score_type=score_type, resource_spec=resource_spec, data_split=data_split
            )
            for score_type in ls_score_types
        ]
        return ls_callbacks

    @classmethod
    def _get_array_callbacks(
        cls,
        resource_spec: ResourceSpecProtocolT,
        ls_array_types: Optional[List[Union[ArrayTypeT, str]]] = None,
        data_split: Optional[DataSplit] = None,
    ) -> Sequence[ArtifactArrayCallback[ArtifactResourcesT, ResourceSpecProtocolT]]:
        callback_factory = cls._get_callback_factory()
        if ls_array_types is None:
            ls_array_types = SequenceConcatenator.concatenate(
                cls._get_array_types(), cls._get_custom_array_types()
            )
        ls_callbacks = [
            callback_factory.build_array_callback(
                array_type=array_type, resource_spec=resource_spec, data_split=data_split
            )
            for array_type in ls_array_types
        ]
        return ls_callbacks

    @classmethod
    def _get_plot_callbacks(
        cls,
        resource_spec: ResourceSpecProtocolT,
        ls_plot_types: Optional[List[Union[PlotTypeT, str]]] = None,
        data_split: Optional[DataSplit] = None,
    ) -> Sequence[ArtifactPlotCallback[ArtifactResourcesT, ResourceSpecProtocolT]]:
        callback_factory = cls._get_callback_factory()
        if ls_plot_types is None:
            ls_plot_types = SequenceConcatenator.concatenate(
                cls._get_plot_types(), cls._get_custom_plot_types()
            )
        ls_callbacks = [
            callback_factory.build_plot_callback(
                plot_type=plot_type, resource_spec=resource_spec, data_split=data_split
            )
            for plot_type in ls_plot_types
        ]
        return ls_callbacks

    @classmethod
    def _get_score_collection_callbacks(
        cls,
        resource_spec: ResourceSpecProtocolT,
        ls_score_collection_types: Optional[List[Union[ScoreCollectionTypeT, str]]] = None,
        data_split: Optional[DataSplit] = None,
    ) -> Sequence[ArtifactScoreCollectionCallback[ArtifactResourcesT, ResourceSpecProtocolT]]:
        callback_factory = cls._get_callback_factory()
        if ls_score_collection_types is None:
            ls_score_collection_types = SequenceConcatenator.concatenate(
                cls._get_score_collection_types(), cls._get_custom_score_collection_types()
            )
        ls_callbacks = [
            callback_factory.build_score_collection_callback(
                score_collection_type=score_collection_type,
                resource_spec=resource_spec,
                data_split=data_split,
            )
            for score_collection_type in ls_score_collection_types
        ]
        return ls_callbacks

    @classmethod
    def _get_array_collection_callbacks(
        cls,
        resource_spec: ResourceSpecProtocolT,
        ls_array_collection_types: Optional[List[Union[ArrayCollectionTypeT, str]]] = None,
        data_split: Optional[DataSplit] = None,
    ) -> Sequence[ArtifactArrayCollectionCallback[ArtifactResourcesT, ResourceSpecProtocolT]]:
        callback_factory = cls._get_callback_factory()
        if ls_array_collection_types is None:
            ls_array_collection_types = SequenceConcatenator.concatenate(
                cls._get_array_collection_types(), cls._get_custom_array_collection_types()
            )
        ls_callbacks = [
            callback_factory.build_array_collection_callback(
                array_collection_type=array_collection_type,
                resource_spec=resource_spec,
                data_split=data_split,
            )
            for array_collection_type in ls_array_collection_types
        ]
        return ls_callbacks

    @classmethod
    def _get_plot_collection_callbacks(
        cls,
        resource_spec: ResourceSpecProtocolT,
        ls_plot_collection_types: Optional[List[Union[PlotCollectionTypeT, str]]] = None,
        data_split: Optional[DataSplit] = None,
    ) -> Sequence[ArtifactPlotCollectionCallback[ArtifactResourcesT, ResourceSpecProtocolT]]:
        callback_factory = cls._get_callback_factory()
        if ls_plot_collection_types is None:
            ls_plot_collection_types = SequenceConcatenator.concatenate(
                cls._get_plot_collection_types(), cls._get_custom_plot_collection_types()
            )
        ls_callbacks = [
            callback_factory.build_plot_collection_callback(
                plot_collection_type=plot_collection_type,
                resource_spec=resource_spec,
                data_split=data_split,
            )
            for plot_collection_type in ls_plot_collection_types
        ]
        return ls_callbacks
