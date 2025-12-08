from typing import Any, Dict

from artifact_core._base.typing.artifact_result import Score

from tests._domains.classification.dummy.registries.base import DummyClassificationRegistry
from tests._domains.classification.dummy.types.scores import DummyClassificationScoreType


class DummyClassificationScoreRegistry(
    DummyClassificationRegistry[DummyClassificationScoreType, Score]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "DUMMY_SCORE": {"weight": 1.0},
        }
