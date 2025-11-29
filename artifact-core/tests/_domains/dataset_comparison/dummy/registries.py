from typing import Any, Dict

from artifact_core._base.typing.artifact_result import (
    Array,
    ArrayCollection,
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


class DummyDatasetComparisonScoreType(ArtifactType):
    pass


class DummyDatasetComparisonScoreRegistry(
    DatasetComparisonArtifactRegistry[
        DummyDataset, DummyResourceSpec, DummyDatasetComparisonScoreType, Score
    ]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}


class DummyDatasetComparisonArray(ArtifactType):
    pass


class DummyDatasetComparisonArrayRegistry(
    DatasetComparisonArtifactRegistry[
        DummyDataset, DummyResourceSpec, DummyDatasetComparisonArray, Array
    ]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}


class DummyDatasetComparisonPlot(ArtifactType):
    pass


class DummyDatasetComparisonPlotRegistry(
    DatasetComparisonArtifactRegistry[
        DummyDataset, DummyResourceSpec, DummyDatasetComparisonPlot, Plot
    ]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}


class DummyDatasetComparisonScoreCollectionType(ArtifactType):
    pass


class DummyDatasetComparisonScoreCollectionRegistry(
    DatasetComparisonArtifactRegistry[
        DummyDataset, DummyResourceSpec, DummyDatasetComparisonScoreCollectionType, ScoreCollection
    ]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}


class DummyDatasetComparisonArrayCollectionType(ArtifactType):
    pass


class DummyDatasetComparisonArrayCollectionRegistry(
    DatasetComparisonArtifactRegistry[
        DummyDataset, DummyResourceSpec, DummyDatasetComparisonArrayCollectionType, ArrayCollection
    ]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}


class DummyDatasetComparisonPlotCollectionType(ArtifactType):
    pass


class DummyDatasetComparisonPlotCollectionRegistry(
    DatasetComparisonArtifactRegistry[
        DummyDataset, DummyResourceSpec, DummyDatasetComparisonPlotCollectionType, PlotCollection
    ]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}
