from typing import Any, Dict

from artifact_core._base.typing.artifact_result import Plot

from tests._domains.dataset_comparison.dummy.registries.base import DummyDatasetComparisonRegistry
from tests._domains.dataset_comparison.dummy.types.plots import DummyDatasetComparisonPlotType


class DummyDatasetComparisonPlotRegistry(
    DummyDatasetComparisonRegistry[DummyDatasetComparisonPlotType, Plot]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}
