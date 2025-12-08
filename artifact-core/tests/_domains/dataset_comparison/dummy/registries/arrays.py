from typing import Any, Dict

from artifact_core._base.typing.artifact_result import Array

from tests._domains.dataset_comparison.dummy.registries.base import DummyDatasetComparisonRegistry
from tests._domains.dataset_comparison.dummy.types.arrays import DummyDatasetComparisonArrayType


class DummyDatasetComparisonArrayRegistry(
    DummyDatasetComparisonRegistry[DummyDatasetComparisonArrayType, Array]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}
