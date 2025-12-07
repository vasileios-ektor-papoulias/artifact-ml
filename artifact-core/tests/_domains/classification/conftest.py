from typing import Callable, Mapping

import pytest
from artifact_core._libs.resources.classification.class_store import ClassStore
from artifact_core._libs.resources.classification.classification_results import (
    ClassificationResults,
)
from artifact_core._libs.resources.classification.distribution_store import ClassDistributionStore
from artifact_core._utils.collections.entity_store import IdentifierType

from tests._domains.classification.dummy.resource_spec import DummyClassSpec
from tests._domains.classification.dummy.resources import (
    DummyClassificationResults,
    DummyClassStore,
)

MakeClassStore = Callable[[DummyClassSpec, Mapping[IdentifierType, int]], DummyClassStore]
MakeClassificationResults = Callable[
    [DummyClassSpec, Mapping[IdentifierType, int]], DummyClassificationResults
]


def _make_class_store(
    class_spec: DummyClassSpec,
    id_to_class_idx: Mapping[IdentifierType, int],
) -> DummyClassStore:
    store = ClassStore(class_spec=class_spec)
    store.set_multiple_idx(id_to_class_idx=id_to_class_idx)
    return store


def _make_classification_results(
    class_spec: DummyClassSpec,
    id_to_predicted_class_idx: Mapping[IdentifierType, int],
) -> DummyClassificationResults:
    pred_store = ClassStore(class_spec=class_spec)
    distn_store = ClassDistributionStore(class_spec=class_spec)
    results = ClassificationResults(
        class_spec=class_spec, pred_store=pred_store, distn_store=distn_store
    )
    for identifier, class_idx in id_to_predicted_class_idx.items():
        class_name = class_spec.class_names[class_idx]
        results.set_single(identifier=identifier, predicted_class=class_name)
    return results


@pytest.fixture
def make_class_store() -> MakeClassStore:
    return _make_class_store


@pytest.fixture
def make_classification_results() -> MakeClassificationResults:
    return _make_classification_results
