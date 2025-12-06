from typing import Type, Union

import pytest
from artifact_core._base.core.hyperparams import ArtifactHyperparams
from artifact_core._base.typing.artifact_result import Score

from tests._domains.classification.dummy.artifacts.scores.dummy import (
    DummyClassificationScore,
    DummyClassificationScoreHyperparams,
)
from tests._domains.classification.dummy.registries.base import DummyClassificationRegistry
from tests._domains.classification.dummy.registries.scores import (
    DummyClassificationScoreRegistry,
)
from tests._domains.classification.dummy.resource_spec import DummyClassSpec
from tests._domains.classification.dummy.types.scores import DummyClassificationScoreType


@pytest.mark.unit
@pytest.mark.parametrize(
    "artifact_registry, artifact_type, resource_spec, "
    + "expected_artifact_class, expected_hyperparams",
    [
        (
            DummyClassificationScoreRegistry,
            DummyClassificationScoreType.DUMMY_SCORE,
            DummyClassSpec(class_names=["cat", "dog"], label_name="target"),
            DummyClassificationScore,
            DummyClassificationScoreHyperparams(weight=1.0),
        ),
        (
            DummyClassificationScoreRegistry,
            DummyClassificationScoreType.DUMMY_SCORE,
            DummyClassSpec(class_names=["a", "b", "c"], label_name="label"),
            DummyClassificationScore,
            DummyClassificationScoreHyperparams(weight=1.0),
        ),
    ],
)
def test_get(
    artifact_registry: Type[DummyClassificationRegistry[DummyClassificationScoreType, Score]],
    artifact_type: Union[DummyClassificationScoreType, str],
    resource_spec: DummyClassSpec,
    expected_artifact_class: Type,
    expected_hyperparams: ArtifactHyperparams,
):
    artifact = artifact_registry.get(artifact_type=artifact_type, resource_spec=resource_spec)
    assert isinstance(artifact, expected_artifact_class)
    assert artifact.resource_spec == resource_spec
    assert artifact.hyperparams == expected_hyperparams
