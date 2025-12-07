from typing import Any, Dict

from artifact_core._base.typing.artifact_result import Score

from tests._domains.dataset_comparison.dummy.registries.base import DummyDatasetComparisonRegistry
from tests._domains.dataset_comparison.dummy.types.scores import DummyDatasetComparisonScoreType


class DummyDatasetComparisonScoreRegistry(
    DummyDatasetComparisonRegistry[DummyDatasetComparisonScoreType, Score]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "DUMMY_SCORE": {"adjust_scale": True},
        }
