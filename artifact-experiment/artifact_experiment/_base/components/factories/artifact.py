from abc import abstractmethod
from typing import Generic, Optional, Type, TypeVar, Union

from artifact_core.spi.orchestration import ArtifactRegistry, ArtifactType
from artifact_core.spi.resources import ArtifactResources, ResourceSpecProtocol
from artifact_core.typing import (
    Array,
    ArrayCollection,
    Plot,
    PlotCollection,
    Score,
    ScoreCollection,
)

from artifact_experiment._base.components.callbacks.artifact import (
    ArtifactArrayCallback,
    ArtifactArrayCollectionCallback,
    ArtifactPlotCallback,
    ArtifactPlotCollectionCallback,
    ArtifactScoreCallback,
    ArtifactScoreCollectionCallback,
)
from artifact_experiment._base.tracking.background.writer import TrackingQueueWriter

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
        ArtifactRegistry[ArtifactResourcesT, ResourceSpecProtocolT, ScoreTypeT, Score]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_array_registry() -> Type[
        ArtifactRegistry[ArtifactResourcesT, ResourceSpecProtocolT, ArrayTypeT, Array]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_registry() -> Type[
        ArtifactRegistry[ArtifactResourcesT, ResourceSpecProtocolT, PlotTypeT, Plot]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_score_collection_registry() -> Type[
        ArtifactRegistry[
            ArtifactResourcesT, ResourceSpecProtocolT, ScoreCollectionTypeT, ScoreCollection
        ]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_array_collection_registry() -> Type[
        ArtifactRegistry[
            ArtifactResourcesT, ResourceSpecProtocolT, ArrayCollectionTypeT, ArrayCollection
        ]
    ]: ...

    @staticmethod
    @abstractmethod
    def _get_plot_collection_registry() -> Type[
        ArtifactRegistry[
            ArtifactResourcesT, ResourceSpecProtocolT, PlotCollectionTypeT, PlotCollection
        ]
    ]: ...

    @classmethod
    def build_score_callback(
        cls,
        score_type: Union[ScoreTypeT, str],
        resource_spec: ResourceSpecProtocolT,
        writer: Optional[TrackingQueueWriter[Score]] = None,
    ) -> ArtifactScoreCallback[ArtifactResourcesT, ResourceSpecProtocolT]:
        registry = cls._get_score_registry()
        artifact = registry.get(artifact_type=score_type, resource_spec=resource_spec)
        base_key = cls._get_base_key(artifact_type=score_type)
        callback = ArtifactScoreCallback[ArtifactResourcesT, ResourceSpecProtocolT](
            base_key=base_key, artifact=artifact, writer=writer
        )
        return callback

    @classmethod
    def build_array_callback(
        cls,
        array_type: Union[ArrayTypeT, str],
        resource_spec: ResourceSpecProtocolT,
        writer: Optional[TrackingQueueWriter[Array]] = None,
    ) -> ArtifactArrayCallback[ArtifactResourcesT, ResourceSpecProtocolT]:
        registry = cls._get_array_registry()
        artifact = registry.get(artifact_type=array_type, resource_spec=resource_spec)
        base_key = cls._get_base_key(artifact_type=array_type)
        callback = ArtifactArrayCallback[ArtifactResourcesT, ResourceSpecProtocolT](
            base_key=base_key, artifact=artifact, writer=writer
        )
        return callback

    @classmethod
    def build_plot_callback(
        cls,
        plot_type: Union[PlotTypeT, str],
        resource_spec: ResourceSpecProtocolT,
        writer: Optional[TrackingQueueWriter[Plot]] = None,
    ) -> ArtifactPlotCallback[ArtifactResourcesT, ResourceSpecProtocolT]:
        registry = cls._get_plot_registry()
        artifact = registry.get(artifact_type=plot_type, resource_spec=resource_spec)
        base_key = cls._get_base_key(artifact_type=plot_type)
        callback = ArtifactPlotCallback[ArtifactResourcesT, ResourceSpecProtocolT](
            base_key=base_key, artifact=artifact, writer=writer
        )
        return callback

    @classmethod
    def build_score_collection_callback(
        cls,
        score_collection_type: Union[ScoreCollectionTypeT, str],
        resource_spec: ResourceSpecProtocolT,
        writer: Optional[TrackingQueueWriter[ScoreCollection]] = None,
    ) -> ArtifactScoreCollectionCallback[ArtifactResourcesT, ResourceSpecProtocolT]:
        registry = cls._get_score_collection_registry()
        artifact = registry.get(artifact_type=score_collection_type, resource_spec=resource_spec)
        base_key = cls._get_base_key(artifact_type=score_collection_type)
        callback = ArtifactScoreCollectionCallback[ArtifactResourcesT, ResourceSpecProtocolT](
            base_key=base_key, artifact=artifact, writer=writer
        )
        return callback

    @classmethod
    def build_array_collection_callback(
        cls,
        array_collection_type: Union[ArrayCollectionTypeT, str],
        resource_spec: ResourceSpecProtocolT,
        writer: Optional[TrackingQueueWriter[ArrayCollection]] = None,
    ) -> ArtifactArrayCollectionCallback[ArtifactResourcesT, ResourceSpecProtocolT]:
        registry = cls._get_array_collection_registry()
        artifact = registry.get(artifact_type=array_collection_type, resource_spec=resource_spec)
        base_key = cls._get_base_key(artifact_type=array_collection_type)
        callback = ArtifactArrayCollectionCallback[ArtifactResourcesT, ResourceSpecProtocolT](
            base_key=base_key, artifact=artifact, writer=writer
        )
        return callback

    @classmethod
    def build_plot_collection_callback(
        cls,
        plot_collection_type: Union[PlotCollectionTypeT, str],
        resource_spec: ResourceSpecProtocolT,
        writer: Optional[TrackingQueueWriter[PlotCollection]] = None,
    ) -> ArtifactPlotCollectionCallback[ArtifactResourcesT, ResourceSpecProtocolT]:
        registry = cls._get_plot_collection_registry()
        artifact = registry.get(artifact_type=plot_collection_type, resource_spec=resource_spec)
        base_key = cls._get_base_key(artifact_type=plot_collection_type)
        callback = ArtifactPlotCollectionCallback[ArtifactResourcesT, ResourceSpecProtocolT](
            base_key=base_key, artifact=artifact, writer=writer
        )
        return callback

    @staticmethod
    def _get_base_key(artifact_type: Union[ArtifactTypeT, str]) -> str:
        if isinstance(artifact_type, str):
            name = artifact_type
        else:
            name = artifact_type.name
        return name
