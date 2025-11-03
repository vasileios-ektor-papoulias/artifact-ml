from abc import abstractmethod
from typing import Dict, Generic, Optional, Type, TypeVar, Union

from artifact_core.base.artifact_dependencies import (
    ArtifactResources,
    ResourceSpecProtocol,
)
from artifact_core.base.registry import ArtifactRegistry, ArtifactType
from matplotlib.figure import Figure
from numpy import ndarray

from artifact_experiment.base.callbacks.artifact import (
    ArtifactArrayCallback,
    ArtifactArrayCollectionCallback,
    ArtifactPlotCallback,
    ArtifactPlotCollectionCallback,
    ArtifactScoreCallback,
    ArtifactScoreCollectionCallback,
)
from artifact_experiment.base.entities.data_split import DataSplit

ArtifactResourcesT = TypeVar("ArtifactResourcesT", bound=ArtifactResources)
ResourceSpecProtocolT = TypeVar("ResourceSpecProtocolT", bound=ResourceSpecProtocol)
ArtifactTypeT = TypeVar("ArtifactTypeT", bound=ArtifactType)
ScoreTypeT = TypeVar("ScoreTypeT", bound=ArtifactType)
ArrayTypeT = TypeVar("ArrayTypeT", bound=ArtifactType)
PlotTypeT = TypeVar("PlotTypeT", bound=ArtifactType)
ScoreCollectionTypeT = TypeVar("ScoreCollectionTypeT", bound=ArtifactType)
ArrayCollectionTypeT = TypeVar("ArrayCollectionTypeT", bound=ArtifactType)
PlotCollectionTypeT = TypeVar("PlotCollectionTypeT", bound=ArtifactType)


class ArtifactCallbackFactory(
    Generic[
        ArtifactResourcesT,
        ResourceSpecProtocolT,
        ScoreTypeT,
        ArrayTypeT,
        PlotTypeT,
        ScoreCollectionTypeT,
        ArrayCollectionTypeT,
        PlotCollectionTypeT,
    ]
):
    @staticmethod
    @abstractmethod
    def _get_score_registry() -> Type[
        ArtifactRegistry[ArtifactResourcesT, ResourceSpecProtocolT, ScoreTypeT, float]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_array_registry() -> Type[
        ArtifactRegistry[ArtifactResourcesT, ResourceSpecProtocolT, ArrayTypeT, ndarray]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_registry() -> Type[
        ArtifactRegistry[ArtifactResourcesT, ResourceSpecProtocolT, PlotTypeT, Figure]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_score_collection_registry() -> Type[
        ArtifactRegistry[
            ArtifactResourcesT, ResourceSpecProtocolT, ScoreCollectionTypeT, Dict[str, float]
        ]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_array_collection_registry() -> Type[
        ArtifactRegistry[
            ArtifactResourcesT,
            ResourceSpecProtocolT,
            ArrayCollectionTypeT,
            Dict[str, ndarray],
        ]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_collection_registry() -> Type[
        ArtifactRegistry[
            ArtifactResourcesT,
            ResourceSpecProtocolT,
            PlotCollectionTypeT,
            Dict[str, Figure],
        ]
    ]: ...

    @classmethod
    def build_score_callback(
        cls,
        score_type: Union[ScoreTypeT, str],
        resource_spec: ResourceSpecProtocolT,
        data_split: Optional[DataSplit] = None,
    ) -> ArtifactScoreCallback[ArtifactResourcesT, ResourceSpecProtocolT]:
        registry = cls._get_score_registry()
        artifact = registry.get(artifact_type=score_type, resource_spec=resource_spec)
        name = cls._get_name(artifact_type=score_type)
        callback = ArtifactScoreCallback[ArtifactResourcesT, ResourceSpecProtocolT](
            name=name, artifact=artifact, data_split=data_split
        )
        return callback

    @classmethod
    def build_array_callback(
        cls,
        array_type: Union[ArrayTypeT, str],
        resource_spec: ResourceSpecProtocolT,
        data_split: Optional[DataSplit] = None,
    ) -> ArtifactArrayCallback[ArtifactResourcesT, ResourceSpecProtocolT]:
        registry = cls._get_array_registry()
        artifact = registry.get(artifact_type=array_type, resource_spec=resource_spec)
        name = cls._get_name(artifact_type=array_type)
        callback = ArtifactArrayCallback[ArtifactResourcesT, ResourceSpecProtocolT](
            name=name, artifact=artifact, data_split=data_split
        )
        return callback

    @classmethod
    def build_plot_callback(
        cls,
        plot_type: Union[PlotTypeT, str],
        resource_spec: ResourceSpecProtocolT,
        data_split: Optional[DataSplit] = None,
    ) -> ArtifactPlotCallback[ArtifactResourcesT, ResourceSpecProtocolT]:
        registry = cls._get_plot_registry()
        artifact = registry.get(artifact_type=plot_type, resource_spec=resource_spec)
        name = cls._get_name(artifact_type=plot_type)
        callback = ArtifactPlotCallback[ArtifactResourcesT, ResourceSpecProtocolT](
            name=name, artifact=artifact, data_split=data_split
        )
        return callback

    @classmethod
    def build_score_collection_callback(
        cls,
        score_collection_type: Union[ScoreCollectionTypeT, str],
        resource_spec: ResourceSpecProtocolT,
        data_split: Optional[DataSplit] = None,
    ) -> ArtifactScoreCollectionCallback[ArtifactResourcesT, ResourceSpecProtocolT]:
        registry = cls._get_score_collection_registry()
        artifact = registry.get(artifact_type=score_collection_type, resource_spec=resource_spec)
        name = cls._get_name(artifact_type=score_collection_type)
        callback = ArtifactScoreCollectionCallback[ArtifactResourcesT, ResourceSpecProtocolT](
            name=name, artifact=artifact, data_split=data_split
        )
        return callback

    @classmethod
    def build_array_collection_callback(
        cls,
        array_collection_type: Union[ArrayCollectionTypeT, str],
        resource_spec: ResourceSpecProtocolT,
        data_split: Optional[DataSplit] = None,
    ) -> ArtifactArrayCollectionCallback[ArtifactResourcesT, ResourceSpecProtocolT]:
        registry = cls._get_array_collection_registry()
        artifact = registry.get(artifact_type=array_collection_type, resource_spec=resource_spec)
        name = cls._get_name(artifact_type=array_collection_type)
        callback = ArtifactArrayCollectionCallback[ArtifactResourcesT, ResourceSpecProtocolT](
            name=name, artifact=artifact, data_split=data_split
        )
        return callback

    @classmethod
    def build_plot_collection_callback(
        cls,
        plot_collection_type: Union[PlotCollectionTypeT, str],
        resource_spec: ResourceSpecProtocolT,
        data_split: Optional[DataSplit] = None,
    ) -> ArtifactPlotCollectionCallback[ArtifactResourcesT, ResourceSpecProtocolT]:
        registry = cls._get_plot_collection_registry()
        artifact = registry.get(artifact_type=plot_collection_type, resource_spec=resource_spec)
        name = cls._get_name(artifact_type=plot_collection_type)
        callback = ArtifactPlotCollectionCallback[ArtifactResourcesT, ResourceSpecProtocolT](
            name=name, artifact=artifact, data_split=data_split
        )
        return callback

    @staticmethod
    def _get_name(artifact_type: Union[ArtifactTypeT, str]) -> str:
        if isinstance(artifact_type, str):
            name = artifact_type
        else:
            name = artifact_type.name
        return name
