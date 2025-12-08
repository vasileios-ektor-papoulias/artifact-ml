from typing import Any, Dict

from artifact_core._base.typing.artifact_result import PlotCollection

from tests._domains.dataset_comparison.dummy.registries.base import DummyDatasetComparisonRegistry
from tests._domains.dataset_comparison.dummy.types.plot_collections import (
    DummyDatasetComparisonPlotCollectionType,
)


class DummyDatasetComparisonPlotCollectionRegistry(
    DummyDatasetComparisonRegistry[DummyDatasetComparisonPlotCollectionType, PlotCollection]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}
