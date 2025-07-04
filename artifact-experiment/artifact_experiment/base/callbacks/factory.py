from abc import abstractmethod
from typing import Dict, Generic, Type, TypeVar, Union

from artifact_core.base.artifact_dependencies import (
    ArtifactResources,
    ArtifactResult,
    ResourceSpecProtocol,
)
from artifact_core.base.registry import ArtifactRegistry, ArtifactType
from artifact_experiment.base.callbacks.artifact import (
    ArtifactArrayCallback,
    ArtifactArrayCollectionCallback,
    ArtifactPlotCallback,
    ArtifactPlotCollectionCallback,
    ArtifactScoreCallback,
    ArtifactScoreCollectionCallback,
)
from matplotlib.figure import Figure
from numpy import ndarray

ArtifactTypeT = TypeVar("ArtifactTypeT", bound=ArtifactType)
ScoreTypeT = TypeVar("ScoreTypeT", bound=ArtifactType)
ArrayTypeT = TypeVar("ArrayTypeT", bound=ArtifactType)
PlotTypeT = TypeVar("PlotTypeT", bound=ArtifactType)
ScoreCollectionTypeT = TypeVar("ScoreCollectionTypeT", bound=ArtifactType)
ArrayCollectionTypeT = TypeVar("ArrayCollectionTypeT", bound=ArtifactType)
PlotCollectionTypeT = TypeVar("PlotCollectionTypeT", bound=ArtifactType)
ResourceSpecProtocolT = TypeVar("ResourceSpecProtocolT", bound=ResourceSpecProtocol)
ArtifactResourcesT = TypeVar("ArtifactResourcesT", bound=ArtifactResources)
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)


class ArtifactCallbackFactory(
    Generic[
        ScoreTypeT,
        ArrayTypeT,
        PlotTypeT,
        ScoreCollectionTypeT,
        ArrayCollectionTypeT,
        PlotCollectionTypeT,
        ArtifactResourcesT,
        ResourceSpecProtocolT,
    ]
):
    @staticmethod
    @abstractmethod
    def _get_score_registry() -> Type[
        ArtifactRegistry[ScoreTypeT, ArtifactResourcesT, float, ResourceSpecProtocolT]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_array_registry() -> Type[
        ArtifactRegistry[ArrayTypeT, ArtifactResourcesT, ndarray, ResourceSpecProtocolT]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_registry() -> Type[
        ArtifactRegistry[PlotTypeT, ArtifactResourcesT, Figure, ResourceSpecProtocolT]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_score_collection_registry() -> Type[
        ArtifactRegistry[
            ScoreCollectionTypeT, ArtifactResourcesT, Dict[str, float], ResourceSpecProtocolT
        ]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_array_collection_registry() -> Type[
        ArtifactRegistry[
            ArrayCollectionTypeT, ArtifactResourcesT, Dict[str, ndarray], ResourceSpecProtocolT
        ]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_collection_registry() -> Type[
        ArtifactRegistry[
            PlotCollectionTypeT, ArtifactResourcesT, Dict[str, Figure], ResourceSpecProtocolT
        ]
    ]: ...

    @classmethod
    def build_score_callback(
        cls, score_type: Union[ScoreTypeT, str], resource_spec: ResourceSpecProtocolT
    ) -> ArtifactScoreCallback[ArtifactResourcesT, ResourceSpecProtocolT]:
        registry = cls._get_score_registry()
        artifact = registry.get(artifact_type=score_type, resource_spec=resource_spec)
        key = cls._get_key(artifact_type=score_type)
        callback = ArtifactScoreCallback[ArtifactResourcesT, ResourceSpecProtocolT](
            key=key, artifact=artifact
        )
        return callback

    @classmethod
    def build_array_callback(
        cls, array_type: Union[ArrayTypeT, str], resource_spec: ResourceSpecProtocolT
    ) -> ArtifactArrayCallback[ArtifactResourcesT, ResourceSpecProtocolT]:
        registry = cls._get_array_registry()
        artifact = registry.get(artifact_type=array_type, resource_spec=resource_spec)
        key = cls._get_key(artifact_type=array_type)
        callback = ArtifactArrayCallback[ArtifactResourcesT, ResourceSpecProtocolT](
            key=key, artifact=artifact
        )
        return callback

    @classmethod
    def build_plot_callback(
        cls, plot_type: Union[PlotTypeT, str], resource_spec: ResourceSpecProtocolT
    ) -> ArtifactPlotCallback[ArtifactResourcesT, ResourceSpecProtocolT]:
        registry = cls._get_plot_registry()
        artifact = registry.get(artifact_type=plot_type, resource_spec=resource_spec)
        key = cls._get_key(artifact_type=plot_type)
        callback = ArtifactPlotCallback[ArtifactResourcesT, ResourceSpecProtocolT](
            key=key, artifact=artifact
        )
        return callback

    @classmethod
    def build_score_collection_callback(
        cls,
        score_collection_type: Union[ScoreCollectionTypeT, str],
        resource_spec: ResourceSpecProtocolT,
    ) -> ArtifactScoreCollectionCallback[ArtifactResourcesT, ResourceSpecProtocolT]:
        registry = cls._get_score_collection_registry()
        artifact = registry.get(artifact_type=score_collection_type, resource_spec=resource_spec)
        key = cls._get_key(artifact_type=score_collection_type)
        callback = ArtifactScoreCollectionCallback[ArtifactResourcesT, ResourceSpecProtocolT](
            key=key, artifact=artifact
        )
        return callback

    @classmethod
    def build_array_collection_callback(
        cls,
        array_collection_type: Union[ArrayCollectionTypeT, str],
        resource_spec: ResourceSpecProtocolT,
    ) -> ArtifactArrayCollectionCallback[ArtifactResourcesT, ResourceSpecProtocolT]:
        registry = cls._get_array_collection_registry()
        artifact = registry.get(artifact_type=array_collection_type, resource_spec=resource_spec)
        key = cls._get_key(artifact_type=array_collection_type)
        callback = ArtifactArrayCollectionCallback[ArtifactResourcesT, ResourceSpecProtocolT](
            key=key, artifact=artifact
        )
        return callback

    @classmethod
    def build_plot_collection_callback(
        cls,
        plot_collection_type: Union[PlotCollectionTypeT, str],
        resource_spec: ResourceSpecProtocolT,
    ) -> ArtifactPlotCollectionCallback[ArtifactResourcesT, ResourceSpecProtocolT]:
        registry = cls._get_plot_collection_registry()
        artifact = registry.get(artifact_type=plot_collection_type, resource_spec=resource_spec)
        key = cls._get_key(artifact_type=plot_collection_type)
        callback = ArtifactPlotCollectionCallback[ArtifactResourcesT, ResourceSpecProtocolT](
            key=key, artifact=artifact
        )
        return callback

    @staticmethod
    def _get_key(artifact_type: Union[ArtifactTypeT, str]) -> str:
        if isinstance(artifact_type, str):
            key = artifact_type
        else:
            key = artifact_type.name
        return key
