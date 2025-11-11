from typing import Any, Dict

from artifact_core._core.dataset_comparison.registry import (
    ArtifactType,
    DatasetComparisonArtifactRegistry,
)
from matplotlib.figure import Figure
from numpy import ndarray

from tests.core.dataset_comparison.dummy.artifact_dependencies import (
    DummyDataset,
    DummyResourceSpec,
)


class DummyDatasetComparisonScoreType(ArtifactType):
    pass


class DummyDatasetComparisonScoreRegistry(
    DatasetComparisonArtifactRegistry[
        DummyDatasetComparisonScoreType, DummyDataset, float, DummyResourceSpec
    ]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}


class DummyDatasetComparisonArrayType(ArtifactType):
    pass


class DummyDatasetComparisonArrayRegistry(
    DatasetComparisonArtifactRegistry[
        DummyDatasetComparisonArrayType, DummyDataset, ndarray, DummyResourceSpec
    ]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}


class DummyDatasetComparisonPlotType(ArtifactType):
    pass


class DummyDatasetComparisonPlotRegistry(
    DatasetComparisonArtifactRegistry[
        DummyDatasetComparisonPlotType, DummyDataset, Figure, DummyResourceSpec
    ]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}


class DummyDatasetComparisonScoreCollectionType(ArtifactType):
    pass


class DummyDatasetComparisonScoreCollectionRegistry(
    DatasetComparisonArtifactRegistry[
        DummyDatasetComparisonScoreCollectionType, DummyDataset, Dict[str, float], DummyResourceSpec
    ]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}


class DummyDatasetComparisonArrayCollectionType(ArtifactType):
    pass


class DummyDatasetComparisonArrayCollectionRegistry(
    DatasetComparisonArtifactRegistry[
        DummyDatasetComparisonArrayCollectionType,
        DummyDataset,
        Dict[str, ndarray],
        DummyResourceSpec,
    ]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}


class DummyDatasetComparisonPlotCollectionType(ArtifactType):
    pass


class DummyDatasetComparisonPlotCollectionRegistry(
    DatasetComparisonArtifactRegistry[
        DummyDatasetComparisonPlotCollectionType, DummyDataset, Dict[str, Figure], DummyResourceSpec
    ]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}
