from typing import Any, Dict

from artifact_core._tasks.dataset_comparison.registry import (
    ArtifactType,
    DatasetComparisonArtifactRegistry,
)

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


class DummyDatasetComparisonArray(ArtifactType):
    pass


class DummyDatasetComparisonArrayRegistry(
    DatasetComparisonArtifactRegistry[
        DummyDatasetComparisonArray, DummyDataset, Array, DummyResourceSpec
    ]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}


class DummyDatasetComparisonPlot(ArtifactType):
    pass


class DummyDatasetComparisonPlotRegistry(
    DatasetComparisonArtifactRegistry[
        DummyDatasetComparisonPlot, DummyDataset, Figure, DummyResourceSpec
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
        Dict[str, Array],
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
