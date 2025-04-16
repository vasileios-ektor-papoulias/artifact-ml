from typing import Any, Dict

from artifact_core.core.dataset_comparison.registry import (
    ArtifactType,
    DatasetComparisonArtifactRegistry,
)
from matplotlib.figure import Figure
from numpy import ndarray

from tests.core.dataset_comparison.dummy.artifact_dependencies import DummyDataset, DummyDataSpec


class DummyDatasetComparisonScoreType(ArtifactType):
    pass


class DummyDatasetComparisonScoreRegistry(
    DatasetComparisonArtifactRegistry[
        DummyDatasetComparisonScoreType, DummyDataset, float, DummyDataSpec
    ]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}


class DummyDatasetComparisonArrayType(ArtifactType):
    pass


class DummyDatasetComparisonArrayRegistry(
    DatasetComparisonArtifactRegistry[
        DummyDatasetComparisonArrayType, DummyDataset, ndarray, DummyDataSpec
    ]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}


class DummyDatasetComparisonPlotType(ArtifactType):
    pass


class DummyDatasetComparisonPlotRegistry(
    DatasetComparisonArtifactRegistry[
        DummyDatasetComparisonPlotType, DummyDataset, Figure, DummyDataSpec
    ]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}


class DummyDatasetComparisonScoreCollectionType(ArtifactType):
    pass


class DummyDatasetComparisonScoreCollectionRegistry(
    DatasetComparisonArtifactRegistry[
        DummyDatasetComparisonScoreCollectionType, DummyDataset, Dict[str, float], DummyDataSpec
    ]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}


class DummyDatasetComparisonArrayCollectionType(ArtifactType):
    pass


class DummyDatasetComparisonArrayCollectionRegistry(
    DatasetComparisonArtifactRegistry[
        DummyDatasetComparisonArrayCollectionType, DummyDataset, Dict[str, ndarray], DummyDataSpec
    ]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}


class DummyDatasetComparisonPlotCollectionType(ArtifactType):
    pass


class DummyDatasetComparisonPlotCollectionRegistry(
    DatasetComparisonArtifactRegistry[
        DummyDatasetComparisonPlotCollectionType, DummyDataset, Dict[str, Figure], DummyDataSpec
    ]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}
