from abc import abstractmethod
from typing import Dict, Generic, Optional, Type, TypeVar, Union

from artifact_core.base.artifact_dependencies import (
    ArtifactResources,
    ArtifactResult,
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
from artifact_experiment.base.data_split import DataSplit, DataSplitSuffixAppender

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
        cls,
        score_type: Union[ScoreTypeT, str],
        resource_spec: ResourceSpecProtocolT,
        data_split: Optional[DataSplit] = None,
    ) -> ArtifactScoreCallback[ArtifactResourcesT, ResourceSpecProtocolT]:
        registry = cls._get_score_registry()
        artifact = registry.get(artifact_type=score_type, resource_spec=resource_spec)
        key = cls._get_key(artifact_type=score_type, data_split=data_split)
        callback = ArtifactScoreCallback[ArtifactResourcesT, ResourceSpecProtocolT](
            key=key, artifact=artifact
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
        key = cls._get_key(artifact_type=array_type, data_split=data_split)
        callback = ArtifactArrayCallback[ArtifactResourcesT, ResourceSpecProtocolT](
            key=key, artifact=artifact
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
        key = cls._get_key(artifact_type=plot_type, data_split=data_split)
        callback = ArtifactPlotCallback[ArtifactResourcesT, ResourceSpecProtocolT](
            key=key, artifact=artifact
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
        key = cls._get_key(artifact_type=score_collection_type, data_split=data_split)
        callback = ArtifactScoreCollectionCallback[ArtifactResourcesT, ResourceSpecProtocolT](
            key=key, artifact=artifact
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
        key = cls._get_key(artifact_type=array_collection_type, data_split=data_split)
        callback = ArtifactArrayCollectionCallback[ArtifactResourcesT, ResourceSpecProtocolT](
            key=key, artifact=artifact
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
        key = cls._get_key(artifact_type=plot_collection_type, data_split=data_split)
        callback = ArtifactPlotCollectionCallback[ArtifactResourcesT, ResourceSpecProtocolT](
            key=key, artifact=artifact
        )
        return callback

    @classmethod
    def _get_key(
        cls, artifact_type: Union[ArtifactTypeT, str], data_split: Optional[DataSplit] = None
    ) -> str:
        key = cls._get_name(artifact_type=artifact_type)
        if data_split is not None:
            key = DataSplitSuffixAppender.append_suffix(name=key, data_split=data_split)
        return key

    @staticmethod
    def _get_name(artifact_type: Union[ArtifactTypeT, str]) -> str:
        if isinstance(artifact_type, str):
            name = artifact_type
        else:
            name = artifact_type.name
        return name
