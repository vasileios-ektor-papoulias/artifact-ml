from typing import Dict

import pytest
from artifact_core._domains.classification.resources import ClassificationArtifactResources
from artifact_core._utils.collections.entity_store import IdentifierType

from tests._domains.classification.conftest import MakeClassificationResults, MakeClassStore
from tests._domains.classification.dummy.artifacts.scores.dummy import (
    DummyClassificationScore,
    DummyClassificationScoreHyperparams,
)
from tests._domains.classification.dummy.resource_spec import DummyClassSpec


@pytest.mark.unit
@pytest.mark.parametrize(
    "hyperparams, resource_spec, id_to_class_idx, id_to_predicted_class_idx, expected",
    [
        (
            DummyClassificationScoreHyperparams(weight=1.0),
            DummyClassSpec(class_names=["cat", "dog"], label_name="target"),
            {0: 0, 1: 1},
            {0: 0, 1: 1},
            1.0,
        ),
        (
            DummyClassificationScoreHyperparams(weight=1.0),
            DummyClassSpec(class_names=["cat", "dog"], label_name="target"),
            {0: 0, 1: 1},
            {0: 0, 1: 0},
            0.5,
        ),
        (
            DummyClassificationScoreHyperparams(weight=2.0),
            DummyClassSpec(class_names=["cat", "dog"], label_name="target"),
            {0: 0, 1: 1},
            {0: 0, 1: 1},
            2.0,
        ),
        (
            DummyClassificationScoreHyperparams(weight=1.0),
            DummyClassSpec(class_names=["cat", "dog"], label_name="target"),
            {0: 0, 1: 1},
            {0: 1, 1: 0},
            0.0,
        ),
    ],
)
def test_compute(
    make_class_store: MakeClassStore,
    make_classification_results: MakeClassificationResults,
    hyperparams: DummyClassificationScoreHyperparams,
    resource_spec: DummyClassSpec,
    id_to_class_idx: Dict[IdentifierType, int],
    id_to_predicted_class_idx: Dict[IdentifierType, int],
    expected: float,
):
    true_class_store = make_class_store(resource_spec, id_to_class_idx)
    classification_results = make_classification_results(resource_spec, id_to_predicted_class_idx)
    artifact_resources = ClassificationArtifactResources(
        true_class_store=true_class_store, classification_results=classification_results
    )
    artifact = DummyClassificationScore(resource_spec=resource_spec, hyperparams=hyperparams)
    result = artifact.compute(resources=artifact_resources)
    assert result == expected
