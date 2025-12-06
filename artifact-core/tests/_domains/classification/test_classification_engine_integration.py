from typing import Dict

import pytest
from artifact_core._utils.collections.entity_store import IdentifierType

from tests._domains.classification.conftest import MakeClassificationResults, MakeClassStore
from tests._domains.classification.dummy.engine.engine import DummyClassificationEngine
from tests._domains.classification.dummy.resource_spec import DummyClassSpec
from tests._domains.classification.dummy.types.scores import DummyClassificationScoreType


@pytest.fixture
def ensure_artifact_registration():
    from tests._domains.classification.dummy.artifacts.scores.dummy import (
        DummyClassificationScore,
        DummyClassificationScoreHyperparams,
    )

    _ = DummyClassificationScore, DummyClassificationScoreHyperparams
    yield


@pytest.mark.unit
@pytest.mark.parametrize(
    "score_type, resource_spec, id_to_class_idx, id_to_predicted_class_idx, expected_result",
    [
        (
            DummyClassificationScoreType.DUMMY_SCORE,
            DummyClassSpec(class_names=["cat", "dog"], label_name="target"),
            {0: 0, 1: 1},
            {0: 0, 1: 1},
            1.0,
        ),
        (
            DummyClassificationScoreType.DUMMY_SCORE,
            DummyClassSpec(class_names=["cat", "dog"], label_name="target"),
            {0: 0, 1: 1},
            {0: 0, 1: 0},
            0.5,
        ),
        (
            DummyClassificationScoreType.DUMMY_SCORE,
            DummyClassSpec(class_names=["cat", "dog"], label_name="target"),
            {0: 0, 1: 1},
            {0: 1, 1: 0},
            0.0,
        ),
        (
            DummyClassificationScoreType.DUMMY_SCORE,
            DummyClassSpec(class_names=["a", "b", "c"], label_name="label"),
            {0: 0, 1: 1, 2: 2},
            {0: 0, 1: 1, 2: 0},
            2 / 3,
        ),
    ],
)
def test_produce_classification_score(
    ensure_artifact_registration,
    make_class_store: MakeClassStore,
    make_classification_results: MakeClassificationResults,
    score_type: DummyClassificationScoreType,
    resource_spec: DummyClassSpec,
    id_to_class_idx: Dict[IdentifierType, int],
    id_to_predicted_class_idx: Dict[IdentifierType, int],
    expected_result: float,
):
    true_class_store = make_class_store(resource_spec, id_to_class_idx)
    classification_results = make_classification_results(resource_spec, id_to_predicted_class_idx)
    engine = DummyClassificationEngine.build(resource_spec=resource_spec)
    result = engine.produce_classification_score(
        score_type=score_type,
        true_class_store=true_class_store,
        classification_results=classification_results,
    )
    assert result == pytest.approx(expected_result)
