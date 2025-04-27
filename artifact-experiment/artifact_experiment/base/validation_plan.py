from abc import ABC, abstractmethod
from typing import Dict, Generic, List, Optional, Type, TypeVar

from artifact_core.base.artifact_dependencies import (
    ArtifactResources,
    ResourceSpecProtocol,
)
from artifact_core.base.registry import ArtifactType
from matplotlib.figure import Figure
from numpy import ndarray

from artifact_experiment.base.callbacks.artifact import ArtifactCallback, ArtifactCallbackResources
from artifact_experiment.base.callbacks.factory import ArtifactCallbackFactory
from artifact_experiment.base.callbacks.tracking import (
    ArrayCallbackHandler,
    ArrayCollectionCallbackHandler,
    PlotCallbackHandler,
    PlotCollectionCallbackHandler,
    ScoreCallbackHandler,
    ScoreCollectionCallbackHandler,
)
from artifact_experiment.base.tracking.client import TrackingClient

scoreTypeT = TypeVar("scoreTypeT", bound=ArtifactType)
arrayTypeT = TypeVar("arrayTypeT", bound=ArtifactType)
plotTypeT = TypeVar("plotTypeT", bound=ArtifactType)
scoreCollectionTypeT = TypeVar("scoreCollectionTypeT", bound=ArtifactType)
arrayCollectionTypeT = TypeVar("arrayCollectionTypeT", bound=ArtifactType)
plotCollectionTypeT = TypeVar("plotCollectionTypeT", bound=ArtifactType)
resourceSpecProtocolT = TypeVar("resourceSpecProtocolT", bound=ResourceSpecProtocol)
artifactResourcesT = TypeVar("artifactResourcesT", bound=ArtifactResources)
artifactValidationPlanT = TypeVar("artifactValidationPlanT", bound="ArtifactValidationPlan")


class ArtifactValidationPlan(
    ABC,
    Generic[
        scoreTypeT,
        arrayTypeT,
        plotTypeT,
        scoreCollectionTypeT,
        arrayCollectionTypeT,
        plotCollectionTypeT,
        artifactResourcesT,
        resourceSpecProtocolT,
    ],
):
    def __init__(
        self,
        score_handler: ScoreCallbackHandler[
            ArtifactCallback[artifactResourcesT, float, resourceSpecProtocolT],
            ArtifactCallbackResources[artifactResourcesT],
        ],
        array_handler: ArrayCallbackHandler[
            ArtifactCallback[artifactResourcesT, ndarray, resourceSpecProtocolT],
            ArtifactCallbackResources[artifactResourcesT],
        ],
        plot_handler: PlotCallbackHandler[
            ArtifactCallback[artifactResourcesT, Figure, resourceSpecProtocolT],
            ArtifactCallbackResources[artifactResourcesT],
        ],
        score_collection_handler: ScoreCollectionCallbackHandler[
            ArtifactCallback[artifactResourcesT, Dict[str, float], resourceSpecProtocolT],
            ArtifactCallbackResources[artifactResourcesT],
        ],
        array_collection_handler: ArrayCollectionCallbackHandler[
            ArtifactCallback[artifactResourcesT, Dict[str, ndarray], resourceSpecProtocolT],
            ArtifactCallbackResources[artifactResourcesT],
        ],
        plot_collection_handler: PlotCollectionCallbackHandler[
            ArtifactCallback[artifactResourcesT, Dict[str, Figure], resourceSpecProtocolT],
            ArtifactCallbackResources[artifactResourcesT],
        ],
    ):
        self._score_handler = score_handler
        self._array_handler = array_handler
        self._plot_handler = plot_handler
        self._score_collection_handler = score_collection_handler
        self._array_collection_handler = array_collection_handler
        self._plot_collection_handler = plot_collection_handler

    @classmethod
    def build(
        cls: Type[artifactValidationPlanT],
        resource_spec: resourceSpecProtocolT,
        tracking_client: Optional[TrackingClient] = None,
    ) -> artifactValidationPlanT:
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
            score_handler=score_handler,
            array_handler=array_handler,
            plot_handler=plot_handler,
            score_collection_handler=score_collection_handler,
            array_collection_handler=array_collection_handler,
            plot_collection_handler=plot_collection_handler,
        )
        return validation_plan

    @property
    def scores(self) -> Dict[str, Optional[float]]:
        return self._score_handler.cache

    @property
    def arrays(self) -> Dict[str, Optional[ndarray]]:
        return self._array_handler.cache

    @property
    def plots(self) -> Dict[str, Optional[Figure]]:
        return self._plot_handler.cache

    @property
    def score_collections(self) -> Dict[str, Optional[Dict[str, float]]]:
        return self._score_collection_handler.cache

    @property
    def array_collections(self) -> Dict[str, Optional[Dict[str, ndarray]]]:
        return self._array_collection_handler.cache

    @property
    def plot_collections(self) -> Dict[str, Optional[Dict[str, Figure]]]:
        return self._plot_collection_handler.cache

    def _execute(self, resources: ArtifactCallbackResources[artifactResourcesT]):
        self._score_handler.execute(resources=resources)
        self._array_handler.execute(resources=resources)
        self._plot_handler.execute(resources=resources)
        self._score_collection_handler.execute(resources=resources)
        self._array_collection_handler.execute(resources=resources)
        self._plot_collection_handler.execute(resources=resources)

    @staticmethod
    @abstractmethod
    def _get_score_types() -> List[scoreTypeT]: ...

    @staticmethod
    @abstractmethod
    def _get_array_types() -> List[arrayTypeT]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_types() -> List[plotTypeT]: ...

    @staticmethod
    @abstractmethod
    def _get_score_collection_types() -> List[scoreCollectionTypeT]: ...

    @staticmethod
    @abstractmethod
    def _get_array_collection_types() -> List[arrayCollectionTypeT]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_collection_types() -> List[plotCollectionTypeT]: ...

    @staticmethod
    @abstractmethod
    def _get_callback_factory() -> Type[
        ArtifactCallbackFactory[
            scoreTypeT,
            arrayTypeT,
            plotTypeT,
            scoreCollectionTypeT,
            arrayCollectionTypeT,
            plotCollectionTypeT,
            artifactResourcesT,
            resourceSpecProtocolT,
        ]
    ]: ...

    @classmethod
    def _build_score_handler(
        cls,
        resource_spec: resourceSpecProtocolT,
        ls_score_types: Optional[List[scoreTypeT]] = None,
        tracking_client: Optional[TrackingClient] = None,
    ) -> ScoreCallbackHandler[
        ArtifactCallback[artifactResourcesT, float, resourceSpecProtocolT],
        ArtifactCallbackResources[artifactResourcesT],
    ]:
        callback_factory = cls._get_callback_factory()
        if ls_score_types is None:
            ls_score_types = cls._get_score_types()
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
        resource_spec: resourceSpecProtocolT,
        ls_array_types: Optional[List[arrayTypeT]] = None,
        tracking_client: Optional[TrackingClient] = None,
    ) -> ArrayCallbackHandler[
        ArtifactCallback[artifactResourcesT, ndarray, resourceSpecProtocolT],
        ArtifactCallbackResources[artifactResourcesT],
    ]:
        callback_factory = cls._get_callback_factory()
        if ls_array_types is None:
            ls_array_types = cls._get_array_types()
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
        resource_spec: resourceSpecProtocolT,
        ls_plot_types: Optional[List[plotTypeT]] = None,
        tracking_client: Optional[TrackingClient] = None,
    ) -> PlotCallbackHandler[
        ArtifactCallback[artifactResourcesT, Figure, resourceSpecProtocolT],
        ArtifactCallbackResources[artifactResourcesT],
    ]:
        callback_factory = cls._get_callback_factory()
        if ls_plot_types is None:
            ls_plot_types = cls._get_plot_types()
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
        resource_spec: resourceSpecProtocolT,
        ls_score_collection_types: Optional[List[scoreCollectionTypeT]] = None,
        tracking_client: Optional[TrackingClient] = None,
    ) -> ScoreCollectionCallbackHandler[
        ArtifactCallback[artifactResourcesT, Dict[str, float], resourceSpecProtocolT],
        ArtifactCallbackResources[artifactResourcesT],
    ]:
        callback_factory = cls._get_callback_factory()
        if ls_score_collection_types is None:
            ls_score_collection_types = cls._get_score_collection_types()
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
        resource_spec: resourceSpecProtocolT,
        ls_array_collection_types: Optional[List[arrayCollectionTypeT]] = None,
        tracking_client: Optional[TrackingClient] = None,
    ) -> ArrayCollectionCallbackHandler[
        ArtifactCallback[artifactResourcesT, Dict[str, ndarray], resourceSpecProtocolT],
        ArtifactCallbackResources[artifactResourcesT],
    ]:
        callback_factory = cls._get_callback_factory()
        if ls_array_collection_types is None:
            ls_array_collection_types = cls._get_array_collection_types()
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
        resource_spec: resourceSpecProtocolT,
        ls_plot_collection_types: Optional[List[plotCollectionTypeT]] = None,
        tracking_client: Optional[TrackingClient] = None,
    ) -> PlotCollectionCallbackHandler[
        ArtifactCallback[artifactResourcesT, Dict[str, Figure], resourceSpecProtocolT],
        ArtifactCallbackResources[artifactResourcesT],
    ]:
        callback_factory = cls._get_callback_factory()
        if ls_plot_collection_types is None:
            ls_plot_collection_types = cls._get_plot_collection_types()
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
