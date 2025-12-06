from typing import Type

import pytest
from artifact_core._libs.resource_specs.binary_classification.protocol import (
    BinaryClassSpecProtocol,
)
from artifact_core.binary_classification._artifacts.array_collections.confusion import (
    ConfusionMatrixCollection,
)
from artifact_core.binary_classification._artifacts.base import BinaryClassificationArrayCollection
from artifact_core.binary_classification._registries.array_collections import (
    BinaryClassificationArrayCollectionRegistry,
)
from artifact_core.binary_classification._types.array_collections import (
    BinaryClassificationArrayCollectionType,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_type, artifact_class",
    [
        (BinaryClassificationArrayCollectionType.CONFUSION_MATRICES, ConfusionMatrixCollection),
    ],
)
def test_get(
    resource_spec: BinaryClassSpecProtocol,
    artifact_type: BinaryClassificationArrayCollectionType,
    artifact_class: Type[BinaryClassificationArrayCollection],
):
    artifact = BinaryClassificationArrayCollectionRegistry.get(
        artifact_type=artifact_type, resource_spec=resource_spec
    )
    assert isinstance(artifact, artifact_class)
    assert artifact.resource_spec == resource_spec
