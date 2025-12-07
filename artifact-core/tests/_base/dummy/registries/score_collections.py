from typing import Any, Dict

from artifact_core._base.typing.artifact_result import ScoreCollection

from tests._base.dummy.registries.base import DummyArtifactRegistry
from tests._base.dummy.types.score_collections import DummyScoreCollectionType


class DummyScoreCollectionRegistry(
    DummyArtifactRegistry[DummyScoreCollectionType, ScoreCollection]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}
