from typing import Any, Dict, TypeVar

from artifact_core._base.typing.artifact_result import (
    Array,
    ArrayCollection,
    ArtifactResult,
    Plot,
    PlotCollection,
    Score,
    ScoreCollection,
)
from artifact_core._domains.dataset_comparison.registry import (
    ArtifactType,
    DatasetComparisonArtifactRegistry,
)

from tests._domains.dataset_comparison.dummy.resource_spec import DummyResourceSpec
from tests._domains.dataset_comparison.dummy.resources import DummyDataset

ArtifactTypeT = TypeVar("ArtifactTypeT", bound=ArtifactType)
ArtifactResultT = TypeVar("ArtifactResultT", bound=ArtifactResult)

DummyDatasetComparisonRegistry = DatasetComparisonArtifactRegistry[
    DummyDataset, DummyResourceSpec, ArtifactTypeT, ArtifactResultT
]


class DummyDatasetComparisonScoreType(ArtifactType):
    pass


class DummyDatasetComparisonScoreRegistry(
    DummyDatasetComparisonRegistry[DummyDatasetComparisonScoreType, Score]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}


class DummyDatasetComparisonArrayType(ArtifactType):
    pass


class DummyDatasetComparisonArrayRegistry(
    DummyDatasetComparisonRegistry[DummyDatasetComparisonArrayType, Array]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}


class DummyDatasetComparisonPlotType(ArtifactType):
    pass


class DummyDatasetComparisonPlotRegistry(
    DummyDatasetComparisonRegistry[DummyDatasetComparisonPlotType, Plot]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}


class DummyDatasetComparisonScoreCollectionType(ArtifactType):
    pass


class DummyDatasetComparisonScoreCollectionRegistry(
    DummyDatasetComparisonRegistry[DummyDatasetComparisonScoreCollectionType, ScoreCollection]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}


class DummyDatasetComparisonArrayCollectionType(ArtifactType):
    pass


class DummyDatasetComparisonArrayCollectionRegistry(
    DummyDatasetComparisonRegistry[DummyDatasetComparisonArrayCollectionType, ArrayCollection]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}


class DummyDatasetComparisonPlotCollectionType(ArtifactType):
    pass


class DummyDatasetComparisonPlotCollectionRegistry(
    DummyDatasetComparisonRegistry[DummyDatasetComparisonPlotCollectionType, PlotCollection]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}
