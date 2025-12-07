from typing import Any, Dict

from artifact_core._base.typing.artifact_result import Array

from tests._domains.classification.dummy.registries.base import DummyClassificationRegistry
from tests._domains.classification.dummy.types.arrays import DummyClassificationArrayType


class DummyClassificationArrayRegistry(
    DummyClassificationRegistry[DummyClassificationArrayType, Array]
):
    @classmethod
    def _get_artifact_configurations(cls) -> Dict[str, Dict[str, Any]]:
        return {}
