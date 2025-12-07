from typing import Any, Dict

from artifact_core._base.typing.artifact_result import ArrayCollection

from tests._domains.dataset_comparison.dummy.registries.base import DummyDatasetComparisonRegistry
from tests._domains.dataset_comparison.dummy.types.array_collections import (
    DummyDatasetComparisonArrayCollectionType,
)


class DummyDatasetComparisonArrayCollectionRegistry(
    DummyDatasetComparisonRegistry[DummyDatasetComparisonArrayCollectionType, ArrayCollection]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}

