from abc import ABC, abstractmethod
from typing import Dict, Generic, List, Optional, Type, TypeVar, Union

from artifact_core.base.artifact_dependencies import (
    ArtifactResources,
    ArtifactResult,
    ResourceSpecProtocol,
)
from artifact_core.base.registry import ArtifactType
from matplotlib.figure import Figure
from numpy import ndarray

from artifact_experiment.base.callback_factory import ArtifactCallbackFactory
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
from artifact_experiment.base.callbacks.tracking import (
    ArrayCallbackHandler,
    ArrayCollectionCallbackHandler,
    PlotCallbackHandler,
    PlotCollectionCallbackHandler,
    ScoreCallbackHandler,
    ScoreCollectionCallbackHandler,
    TrackingCallbackHandler,
)
from artifact_experiment.base.tracking.client import TrackingClient

ScoreTypeT = TypeVar("ScoreTypeT", bound=ArtifactType)
ArrayTypeT = TypeVar("ArrayTypeT", bound=ArtifactType)
PlotTypeT = TypeVar("PlotTypeT", bound=ArtifactType)
ScoreCollectionTypeT = TypeVar("ScoreCollectionTypeT", bound=ArtifactType)
ArrayCollectionTypeT = TypeVar("ArrayCollectionTypeT", bound=ArtifactType)
PlotCollectionTypeT = TypeVar("PlotCollectionTypeT", bound=ArtifactType)
ResourceSpecProtocolT = TypeVar("ResourceSpecProtocolT", bound=ResourceSpecProtocol)
ArtifactResourcesT = TypeVar("ArtifactResourcesT", bound=ArtifactResources)
ValidationPlanT = TypeVar("ValidationPlanT", bound="ValidationPlan")


class ValidationPlan(
    ABC,
    Generic[
        ScoreTypeT,
        ArrayTypeT,
        PlotTypeT,
        ScoreCollectionTypeT,
        ArrayCollectionTypeT,
        PlotCollectionTypeT,
        ArtifactResourcesT,
        ResourceSpecProtocolT,
    ],
):
    def __init__(
        self,
        resource_spec: ResourceSpecProtocolT,
        score_handler: ScoreCallbackHandler[
            ArtifactScoreCallback[ArtifactResourcesT, ResourceSpecProtocolT],
            ArtifactCallbackResources[ArtifactResourcesT],
        ],
        array_handler: ArrayCallbackHandler[
            ArtifactArrayCallback[ArtifactResourcesT, ResourceSpecProtocolT],
            ArtifactCallbackResources[ArtifactResourcesT],
        ],
        plot_handler: PlotCallbackHandler[
            ArtifactPlotCallback[ArtifactResourcesT, ResourceSpecProtocolT],
            ArtifactCallbackResources[ArtifactResourcesT],
        ],
        score_collection_handler: ScoreCollectionCallbackHandler[
            ArtifactScoreCollectionCallback[ArtifactResourcesT, ResourceSpecProtocolT],
            ArtifactCallbackResources[ArtifactResourcesT],
        ],
        array_collection_handler: ArrayCollectionCallbackHandler[
            ArtifactArrayCollectionCallback[ArtifactResourcesT, ResourceSpecProtocolT],
            ArtifactCallbackResources[ArtifactResourcesT],
        ],
        plot_collection_handler: PlotCollectionCallbackHandler[
            ArtifactPlotCollectionCallback[ArtifactResourcesT, ResourceSpecProtocolT],
            ArtifactCallbackResources[ArtifactResourcesT],
        ],
    ):
        self._resource_spec = resource_spec
        self._score_handler = score_handler
        self._array_handler = array_handler
        self._plot_handler = plot_handler
        self._score_collection_handler = score_collection_handler
        self._array_collection_handler = array_collection_handler
        self._plot_collection_handler = plot_collection_handler

    @classmethod
    def build(
        cls: Type[ValidationPlanT],
        resource_spec: ResourceSpecProtocolT,
        tracking_client: Optional[TrackingClient] = None,
    ) -> ValidationPlanT:
        score_handler = cls._build_score_handler(
            resource_spec=resource_spec, tracking_client=tracking_client
        )
        array_handler = cls._build_array_handler(
            resource_spec=resource_spec, tracking_client=tracking_client
        )
        plot_handler = cls._build_plot_handler(
            resource_spec=resource_spec, tracking_client=tracking_client
        )
        score_collection_handler = cls._build_score_collection_handler(
            resource_spec=resource_spec, tracking_client=tracking_client
        )
        array_collection_handler = cls._build_array_collection_handler(
            resource_spec=resource_spec, tracking_client=tracking_client
        )
        plot_collection_handler = cls._build_plot_collection_handler(
            resource_spec=resource_spec, tracking_client=tracking_client
        )
        validation_plan = cls(
            resource_spec=resource_spec,
            score_handler=score_handler,
            array_handler=array_handler,
            plot_handler=plot_handler,
            score_collection_handler=score_collection_handler,
            array_collection_handler=array_collection_handler,
            plot_collection_handler=plot_collection_handler,
        )
        return validation_plan

    @property
    def scores(self) -> Dict[str, float]:
        return self._score_handler.active_cache

    @property
    def arrays(self) -> Dict[str, ndarray]:
        return self._array_handler.active_cache

    @property
    def plots(self) -> Dict[str, Figure]:
        return self._plot_handler.active_cache

    @property
    def score_collections(self) -> Dict[str, Dict[str, float]]:
        return self._score_collection_handler.active_cache

    @property
    def array_collections(self) -> Dict[str, Dict[str, ndarray]]:
        return self._array_collection_handler.active_cache

    @property
    def plot_collections(self) -> Dict[str, Dict[str, Figure]]:
        return self._plot_collection_handler.active_cache

    @property
    def tracking_client(self) -> Optional[TrackingClient]:
        return self._score_handler.tracking_client

    @property
    def _ls_handlers(
        self,
    ) -> List[
        TrackingCallbackHandler[
            ArtifactCallback[ArtifactResourcesT, ArtifactResult, ResourceSpecProtocolT],
            ArtifactCallbackResources[ArtifactResourcesT],
            ArtifactResult,
        ]
    ]:
        ls_handlers = [
            self._score_handler,
            self._array_handler,
            self._plot_handler,
            self._score_collection_handler,
            self._array_collection_handler,
            self._plot_collection_handler,
        ]
        return ls_handlers

    @tracking_client.setter
    def tracking_client(self, tracking_client: Optional[TrackingClient]):
        for handler in self._ls_handlers:
            handler.tracking_client = tracking_client

    @property
    def tracking_enabled(self) -> bool:
        return self.tracking_client is not None

    @staticmethod
    @abstractmethod
    def _get_score_types() -> List[ScoreTypeT]: ...

    @staticmethod
    @abstractmethod
    def _get_array_types() -> List[ArrayTypeT]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_types() -> List[PlotTypeT]: ...

    @staticmethod
    @abstractmethod
    def _get_score_collection_types() -> List[ScoreCollectionTypeT]: ...

    @staticmethod
    @abstractmethod
    def _get_array_collection_types() -> List[ArrayCollectionTypeT]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_collection_types() -> List[PlotCollectionTypeT]: ...

    @staticmethod
    @abstractmethod
    def _get_callback_factory() -> Type[
        ArtifactCallbackFactory[
            ScoreTypeT,
            ArrayTypeT,
            PlotTypeT,
            ScoreCollectionTypeT,
            ArrayCollectionTypeT,
            PlotCollectionTypeT,
            ArtifactResourcesT,
            ResourceSpecProtocolT,
        ]
    ]: ...

    @staticmethod
    def _get_custom_score_types() -> List[str]:
        return []

    @staticmethod
    def _get_custom_array_types() -> List[str]:
        return []

    @staticmethod
    def _get_custom_plot_types() -> List[str]:
        return []

    @staticmethod
    def _get_custom_score_collection_types() -> List[str]:
        return []

    @staticmethod
    def _get_custom_array_collection_types() -> List[str]:
        return []

    @staticmethod
    def _get_custom_plot_collection_types() -> List[str]:
        return []

    def execute(self, resources: ArtifactCallbackResources[ArtifactResourcesT]):
        for handler in self._ls_handlers:
            handler.execute(resources=resources)

    def clear_cache(self):
        for handler in self._ls_handlers:
            handler.clear()

    @classmethod
    def _build_score_handler(
        cls,
        resource_spec: ResourceSpecProtocolT,
        ls_score_types: Optional[List[Union[ScoreTypeT, str]]] = None,
        tracking_client: Optional[TrackingClient] = None,
    ) -> ScoreCallbackHandler[
        ArtifactScoreCallback[ArtifactResourcesT, ResourceSpecProtocolT],
        ArtifactCallbackResources[ArtifactResourcesT],
    ]:
        callback_factory = cls._get_callback_factory()
        if ls_score_types is None:
            ls_score_types = cls._get_score_types() + cls._get_custom_score_types()
        ls_callbacks = [
            callback_factory.build_score_callback(
                score_type=score_type, resource_spec=resource_spec
            )
            for score_type in ls_score_types
        ]
        score_handler = ScoreCallbackHandler(
            ls_callbacks=ls_callbacks, tracking_client=tracking_client
        )
        return score_handler

    @classmethod
    def _build_array_handler(
        cls,
        resource_spec: ResourceSpecProtocolT,
        ls_array_types: Optional[List[Union[ArrayTypeT, str]]] = None,
        tracking_client: Optional[TrackingClient] = None,
    ) -> ArrayCallbackHandler[
        ArtifactArrayCallback[ArtifactResourcesT, ResourceSpecProtocolT],
        ArtifactCallbackResources[ArtifactResourcesT],
    ]:
        callback_factory = cls._get_callback_factory()
        if ls_array_types is None:
            ls_array_types = cls._get_array_types() + cls._get_custom_array_types()
        ls_callbacks = [
            callback_factory.build_array_callback(
                array_type=array_type, resource_spec=resource_spec
            )
            for array_type in ls_array_types
        ]
        array_handler = ArrayCallbackHandler(
            ls_callbacks=ls_callbacks, tracking_client=tracking_client
        )
        return array_handler

    @classmethod
    def _build_plot_handler(
        cls,
        resource_spec: ResourceSpecProtocolT,
        ls_plot_types: Optional[List[Union[PlotTypeT, str]]] = None,
        tracking_client: Optional[TrackingClient] = None,
    ) -> PlotCallbackHandler[
        ArtifactPlotCallback[ArtifactResourcesT, ResourceSpecProtocolT],
        ArtifactCallbackResources[ArtifactResourcesT],
    ]:
        callback_factory = cls._get_callback_factory()
        if ls_plot_types is None:
            ls_plot_types = cls._get_plot_types() + cls._get_custom_plot_types()
        ls_callbacks = [
            callback_factory.build_plot_callback(plot_type=plot_type, resource_spec=resource_spec)
            for plot_type in ls_plot_types
        ]
        plot_handler = PlotCallbackHandler(
            ls_callbacks=ls_callbacks, tracking_client=tracking_client
        )
        return plot_handler

    @classmethod
    def _build_score_collection_handler(
        cls,
        resource_spec: ResourceSpecProtocolT,
        ls_score_collection_types: Optional[List[Union[ScoreCollectionTypeT, str]]] = None,
        tracking_client: Optional[TrackingClient] = None,
    ) -> ScoreCollectionCallbackHandler[
        ArtifactScoreCollectionCallback[ArtifactResourcesT, ResourceSpecProtocolT],
        ArtifactCallbackResources[ArtifactResourcesT],
    ]:
        callback_factory = cls._get_callback_factory()
        if ls_score_collection_types is None:
            ls_score_collection_types = (
                cls._get_score_collection_types() + cls._get_custom_score_collection_types()
            )
        ls_callbacks = [
            callback_factory.build_score_collection_callback(
                score_collection_type=score_collection_type, resource_spec=resource_spec
            )
            for score_collection_type in cls._get_score_collection_types()
        ]
        score_collection_handler = ScoreCollectionCallbackHandler(
            ls_callbacks=ls_callbacks, tracking_client=tracking_client
        )
        return score_collection_handler

    @classmethod
    def _build_array_collection_handler(
        cls,
        resource_spec: ResourceSpecProtocolT,
        ls_array_collection_types: Optional[List[Union[ArrayCollectionTypeT, str]]] = None,
        tracking_client: Optional[TrackingClient] = None,
    ) -> ArrayCollectionCallbackHandler[
        ArtifactArrayCollectionCallback[ArtifactResourcesT, ResourceSpecProtocolT],
        ArtifactCallbackResources[ArtifactResourcesT],
    ]:
        callback_factory = cls._get_callback_factory()
        if ls_array_collection_types is None:
            ls_array_collection_types = (
                cls._get_array_collection_types() + cls._get_custom_array_collection_types()
            )
        ls_callbacks = [
            callback_factory.build_array_collection_callback(
                array_collection_type=array_collection_type, resource_spec=resource_spec
            )
            for array_collection_type in ls_array_collection_types
        ]
        array_collection_handler = ArrayCollectionCallbackHandler(
            ls_callbacks=ls_callbacks, tracking_client=tracking_client
        )
        return array_collection_handler

    @classmethod
    def _build_plot_collection_handler(
        cls,
        resource_spec: ResourceSpecProtocolT,
        ls_plot_collection_types: Optional[List[Union[PlotCollectionTypeT, str]]] = None,
        tracking_client: Optional[TrackingClient] = None,
    ) -> PlotCollectionCallbackHandler[
        ArtifactPlotCollectionCallback[ArtifactResourcesT, ResourceSpecProtocolT],
        ArtifactCallbackResources[ArtifactResourcesT],
    ]:
        callback_factory = cls._get_callback_factory()
        if ls_plot_collection_types is None:
            ls_plot_collection_types = (
                cls._get_plot_collection_types() + cls._get_custom_plot_collection_types()
            )
        ls_callbacks = [
            callback_factory.build_plot_collection_callback(
                plot_collection_type=plot_collection_type, resource_spec=resource_spec
            )
            for plot_collection_type in cls._get_plot_collection_types()
        ]
        plot_collection_handler = PlotCollectionCallbackHandler(
            ls_callbacks=ls_callbacks, tracking_client=tracking_client
        )
        return plot_collection_handler
