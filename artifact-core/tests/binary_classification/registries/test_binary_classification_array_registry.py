from typing import Type

import pytest
from artifact_core._libs.resource_specs.binary_classification.protocol import (
    BinaryClassSpecProtocol,
)
from artifact_core.binary_classification._artifacts.arrays.confusion import ConfusionMatrix
from artifact_core.binary_classification._artifacts.base import BinaryClassificationArray
from artifact_core.binary_classification._registries.arrays import (
    BinaryClassificationArrayRegistry,
)
from artifact_core.binary_classification._types.arrays import BinaryClassificationArrayType


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_type, artifact_class",
    [
        (BinaryClassificationArrayType.CONFUSION_MATRIX, ConfusionMatrix),
    ],
)
def test_get(
    resource_spec: BinaryClassSpecProtocol,
    artifact_type: BinaryClassificationArrayType,
    artifact_class: Type[BinaryClassificationArray],
):
    artifact = BinaryClassificationArrayRegistry.get(
        artifact_type=artifact_type, resource_spec=resource_spec
    )
    assert isinstance(artifact, artifact_class)
    assert artifact.resource_spec == resource_spec
