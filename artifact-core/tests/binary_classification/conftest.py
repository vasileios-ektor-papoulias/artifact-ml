from collections.abc import Hashable
from typing import Dict

import pytest
from artifact_core._libs.resource_specs.binary_classification.protocol import (
    BinaryClassSpecProtocol,
)
from artifact_core._libs.resource_specs.binary_classification.spec import BinaryClassSpec
from artifact_core._libs.resources.binary_classification.class_store import BinaryClassStore
from artifact_core._libs.resources.binary_classification.classification_results import (
    BinaryClassificationResults,
)


@pytest.fixture
def resource_spec() -> BinaryClassSpecProtocol:
    spec = BinaryClassSpec(
        class_names=["negative", "positive"], positive_class="positive", label_name="target"
    )
    return spec


@pytest.fixture
def true_class_store(resource_spec: BinaryClassSpecProtocol) -> BinaryClassStore:
    id_to_class: Dict[Hashable, str] = {
        0: "positive",
        1: "negative",
        2: "positive",
        3: "negative",
        4: "positive",
    }
    store = BinaryClassStore.from_class_names_and_spec(
        class_spec=resource_spec, id_to_class=id_to_class
    )
    return store


@pytest.fixture
def classification_results(resource_spec: BinaryClassSpecProtocol) -> BinaryClassificationResults:
    id_to_class: Dict[Hashable, str] = {
        0: "positive",
        1: "negative",
        2: "negative",
        3: "negative",
        4: "positive",
    }
    id_to_prob_pos: Dict[Hashable, float] = {
        0: 0.9,
        1: 0.2,
        2: 0.4,
        3: 0.1,
        4: 0.8,
    }
    results = BinaryClassificationResults.from_spec(
        class_spec=resource_spec,
        id_to_class=id_to_class,
        id_to_prob_pos=id_to_prob_pos,
    )
    return results
