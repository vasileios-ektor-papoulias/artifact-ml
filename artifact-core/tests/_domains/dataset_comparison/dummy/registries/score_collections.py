from typing import Any, Dict

from artifact_core._base.typing.artifact_result import ScoreCollection

from tests._domains.dataset_comparison.dummy.registries.base import DummyDatasetComparisonRegistry
from tests._domains.dataset_comparison.dummy.types.score_collections import (
    DummyDatasetComparisonScoreCollectionType,
)


class DummyDatasetComparisonScoreCollectionRegistry(
    DummyDatasetComparisonRegistry[DummyDatasetComparisonScoreCollectionType, ScoreCollection]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}
