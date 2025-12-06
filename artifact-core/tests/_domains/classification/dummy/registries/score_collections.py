from typing import Any, Dict

from artifact_core._base.typing.artifact_result import ScoreCollection

from tests._domains.classification.dummy.registries.base import DummyClassificationRegistry
from tests._domains.classification.dummy.types.score_collections import (
    DummyClassificationScoreCollectionType,
)


class DummyClassificationScoreCollectionRegistry(
    DummyClassificationRegistry[DummyClassificationScoreCollectionType, ScoreCollection]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}

