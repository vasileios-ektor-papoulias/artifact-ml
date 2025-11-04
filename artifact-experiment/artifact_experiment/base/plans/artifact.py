from abc import abstractmethod
from typing import Any, Generic, List, Optional, Sequence, Type, TypeVar, Union

from artifact_core.base.artifact_dependencies import ArtifactResources, ResourceSpecProtocol
from artifact_core.base.registry import ArtifactType

from artifact_experiment.base.callbacks.artifact import (
    ArtifactArrayCallback,
    ArtifactArrayCollectionCallback,
    ArtifactCallback,
    ArtifactCallbackResources,
    ArtifactPlotCallback,
    ArtifactPlotCollectionCallback,
    ArtifactScoreCallback,
    ArtifactScoreCollectionCallback,
)
from artifact_experiment.base.entities.data_split import DataSplit
from artifact_experiment.base.handlers.artifact import (
    ArtifactCallbackHandler,
    ArtifactCallbackHandlerSuite,
)
from artifact_experiment.base.plans.base import CallbackExecutionPlan
from artifact_experiment.base.plans.callback_factory import ArtifactCallbackFactory
from artifact_experiment.base.tracking.client import TrackingClient
from artifact_experiment.libs.utils.sequence_concatenator import SequenceConcatenator

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
ArtifactPlanT = TypeVar("ArtifactPlanT", bound="ArtifactPlan")


class ArtifactPlan(
    CallbackExecutionPlan[
        ArtifactCallbackHandler[
            ArtifactResourcesTContr,
            ResourceSpecProtocolTContr,
            Any,
        ],
        ArtifactCallback[ArtifactResourcesTContr, ResourceSpecProtocolTContr, Any],
        ArtifactCallbackResources[ArtifactResourcesTContr],
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
    ],
):
    def __init__(
        self,
        handler_suite: ArtifactCallbackHandlerSuite[
            ArtifactResourcesTContr, ResourceSpecProtocolTContr
        ],
        resource_spec: ResourceSpecProtocolTContr,
        data_split: Optional[DataSplit] = None,
        tracking_client: Optional[TrackingClient] = None,
    ):
        self._resource_spec = resource_spec
        super().__init__(
            handler_suite=handler_suite,
            data_split=data_split,
            tracking_client=tracking_client,
        )

    @classmethod
    def build(
        cls: Type[ArtifactPlanT],
        resource_spec: ResourceSpecProtocolTContr,
        data_split: Optional[DataSplit] = None,
        tracking_client: Optional[TrackingClient] = None,
    ) -> ArtifactPlanT:
        score_callbacks = cls._build_score_callbacks(resource_spec=resource_spec)
        array_callbacks = cls._build_array_callbacks(resource_spec=resource_spec)
        plot_callbacks = cls._build_plot_callbacks(resource_spec=resource_spec)
        score_collection_callbacks = cls._build_score_collection_callbacks(
            resource_spec=resource_spec
        )
        array_collection_callbacks = cls._build_array_collection_callbacks(
            resource_spec=resource_spec
        )
        plot_collection_callbacks = cls._build_plot_collection_callbacks(
            resource_spec=resource_spec
        )
        handler_suite = ArtifactCallbackHandlerSuite.build(
            score_callbacks=score_callbacks,
            array_callbacks=array_callbacks,
            plot_callbacks=plot_callbacks,
            score_collection_callbacks=score_collection_callbacks,
            array_collection_callbacks=array_collection_callbacks,
            plot_collection_callbacks=plot_collection_callbacks,
            tracking_client=tracking_client,
        )
        plan = cls(
            handler_suite=handler_suite,
            resource_spec=resource_spec,
            data_split=data_split,
            tracking_client=tracking_client,
        )
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
    def _build_score_callbacks(
        cls,
        resource_spec: ResourceSpecProtocolTContr,
        ls_score_types: Optional[List[Union[ScoreTypeT, str]]] = None,
    ) -> Sequence[ArtifactScoreCallback[ArtifactResourcesTContr, ResourceSpecProtocolTContr]]:
        callback_factory = cls._get_callback_factory()
        if ls_score_types is None:
            ls_score_types = SequenceConcatenator.concatenate(
                cls._get_score_types(), cls._get_custom_score_types()
            )
        ls_callbacks = [
            callback_factory.build_score_callback(
                score_type=score_type, resource_spec=resource_spec
            )
            for score_type in ls_score_types
        ]
        return ls_callbacks

    @classmethod
    def _build_array_callbacks(
        cls,
        resource_spec: ResourceSpecProtocolTContr,
        ls_array_types: Optional[List[Union[ArrayTypeT, str]]] = None,
    ) -> Sequence[ArtifactArrayCallback[ArtifactResourcesTContr, ResourceSpecProtocolTContr]]:
        callback_factory = cls._get_callback_factory()
        if ls_array_types is None:
            ls_array_types = SequenceConcatenator.concatenate(
                cls._get_array_types(), cls._get_custom_array_types()
            )
        ls_callbacks = [
            callback_factory.build_array_callback(
                array_type=array_type, resource_spec=resource_spec
            )
            for array_type in ls_array_types
        ]
        return ls_callbacks

    @classmethod
    def _build_plot_callbacks(
        cls,
        resource_spec: ResourceSpecProtocolTContr,
        ls_plot_types: Optional[List[Union[PlotTypeT, str]]] = None,
    ) -> Sequence[ArtifactPlotCallback[ArtifactResourcesTContr, ResourceSpecProtocolTContr]]:
        callback_factory = cls._get_callback_factory()
        if ls_plot_types is None:
            ls_plot_types = SequenceConcatenator.concatenate(
                cls._get_plot_types(), cls._get_custom_plot_types()
            )
        ls_callbacks = [
            callback_factory.build_plot_callback(plot_type=plot_type, resource_spec=resource_spec)
            for plot_type in ls_plot_types
        ]
        return ls_callbacks

    @classmethod
    def _build_score_collection_callbacks(
        cls,
        resource_spec: ResourceSpecProtocolTContr,
        ls_score_collection_types: Optional[List[Union[ScoreCollectionTypeT, str]]] = None,
    ) -> Sequence[
        ArtifactScoreCollectionCallback[ArtifactResourcesTContr, ResourceSpecProtocolTContr]
    ]:
        callback_factory = cls._get_callback_factory()
        if ls_score_collection_types is None:
            ls_score_collection_types = SequenceConcatenator.concatenate(
                cls._get_score_collection_types(), cls._get_custom_score_collection_types()
            )
        ls_callbacks = [
            callback_factory.build_score_collection_callback(
                score_collection_type=score_collection_type, resource_spec=resource_spec
            )
            for score_collection_type in ls_score_collection_types
        ]
        return ls_callbacks

    @classmethod
    def _build_array_collection_callbacks(
        cls,
        resource_spec: ResourceSpecProtocolTContr,
        ls_array_collection_types: Optional[List[Union[ArrayCollectionTypeT, str]]] = None,
    ) -> Sequence[
        ArtifactArrayCollectionCallback[ArtifactResourcesTContr, ResourceSpecProtocolTContr]
    ]:
        callback_factory = cls._get_callback_factory()
        if ls_array_collection_types is None:
            ls_array_collection_types = SequenceConcatenator.concatenate(
                cls._get_array_collection_types(), cls._get_custom_array_collection_types()
            )
        ls_callbacks = [
            callback_factory.build_array_collection_callback(
                array_collection_type=array_collection_type, resource_spec=resource_spec
            )
            for array_collection_type in ls_array_collection_types
        ]
        return ls_callbacks

    @classmethod
    def _build_plot_collection_callbacks(
        cls,
        resource_spec: ResourceSpecProtocolTContr,
        ls_plot_collection_types: Optional[List[Union[PlotCollectionTypeT, str]]] = None,
    ) -> Sequence[
        ArtifactPlotCollectionCallback[ArtifactResourcesTContr, ResourceSpecProtocolTContr]
    ]:
        callback_factory = cls._get_callback_factory()
        if ls_plot_collection_types is None:
            ls_plot_collection_types = SequenceConcatenator.concatenate(
                cls._get_plot_collection_types(), cls._get_custom_plot_collection_types()
            )
        ls_callbacks = [
            callback_factory.build_plot_collection_callback(
                plot_collection_type=plot_collection_type, resource_spec=resource_spec
            )
            for plot_collection_type in ls_plot_collection_types
        ]
        return ls_callbacks
