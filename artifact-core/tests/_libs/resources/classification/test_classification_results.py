from typing import List, Mapping

import numpy as np
import pytest
from artifact_core._libs.resource_specs.classification.spec import ClassSpec
from artifact_core._libs.resources.classification.class_store import ClassStore
from artifact_core._libs.resources.classification.classification_results import (
    ClassificationResults,
)
from artifact_core._libs.resources.classification.distribution_store import ClassDistributionStore
from artifact_core._utils.collections.entity_store import IdentifierType


@pytest.mark.unit
@pytest.mark.parametrize(
    "class_names, label_name",
    [
        (["A", "B", "C"], "target"),
        (["neg", "pos"], "label"),
        (["cat", "dog", "bird", "fish"], "animal"),
    ],
)
def test_build_empty(class_names: List[str], label_name: str):
    spec = ClassSpec(class_names=class_names, label_name=label_name)
    results = ClassificationResults.build_empty(class_spec=spec)
    assert results.n_items == 0
    assert results.label_name == label_name
    assert list(results.class_names) == class_names
    assert results.n_classes == len(class_names)
    assert results.pred_store is not None
    assert results.distn_store is not None


@pytest.mark.unit
@pytest.mark.parametrize(
    "class_names, label_name",
    [
        (["A", "B", "C"], "target"),
        (["neg", "pos"], "label"),
        (["cat", "dog", "bird", "fish"], "animal"),
    ],
)
def test_init(class_names: List[str], label_name: str):
    spec = ClassSpec(class_names=class_names, label_name=label_name)
    pred_store = ClassStore(class_spec=spec)
    distn_store = ClassDistributionStore(class_spec=spec)
    results = ClassificationResults(class_spec=spec, pred_store=pred_store, distn_store=distn_store)
    assert results.n_items == 0
    assert results.label_name == label_name
    assert list(results.class_names) == class_names
    assert results.n_classes == len(class_names)


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_class, expected_len",
    [
        ({}, 0),
        ({0: "A"}, 1),
        ({0: "A", 1: "B"}, 2),
        ({0: "A", 1: "B", 2: "C", 3: "A"}, 4),
    ],
)
def test_len(
    classification_results: ClassificationResults,
    id_to_class: Mapping[IdentifierType, str],
    expected_len: int,
):
    if id_to_class:
        classification_results.set_multiple(id_to_class=id_to_class)
    assert len(classification_results) == expected_len


@pytest.mark.unit
@pytest.mark.parametrize(
    "identifier, predicted_class, logits",
    [
        (0, "B", [1.0, 2.0, 3.0]),
        (5, "A", [3.0, 1.0, 2.0]),
        (99, "C", [0.0, 0.0, 1.0]),
    ],
)
def test_set_single_with_logits(
    classification_results: ClassificationResults,
    identifier: IdentifierType,
    predicted_class: str,
    logits: List[float],
):
    arr_logits = np.array(logits)
    classification_results.set_single(
        identifier=identifier,
        predicted_class=predicted_class,
        logits=arr_logits,
    )
    assert classification_results.n_items == 1
    actual_class = classification_results.get_predicted_class(identifier=identifier)
    assert actual_class == predicted_class
    actual_logits = classification_results.get_logits(identifier=identifier)
    np.testing.assert_array_almost_equal(actual_logits, arr_logits)


@pytest.mark.unit
@pytest.mark.parametrize(
    "identifier, class_name, expected_probs",
    [
        (0, "A", [1.0, 0.0, 0.0]),
        (0, "B", [0.0, 1.0, 0.0]),
        (0, "C", [0.0, 0.0, 1.0]),
        (99, "A", [1.0, 0.0, 0.0]),
    ],
)
def test_set_single_without_logits(
    classification_results: ClassificationResults,
    identifier: IdentifierType,
    class_name: str,
    expected_probs: List[float],
):
    classification_results.set_single(identifier=identifier, predicted_class=class_name)
    actual_class = classification_results.get_predicted_class(identifier=identifier)
    assert actual_class == class_name
    probs = classification_results.get_probs(identifier=identifier)
    np.testing.assert_array_almost_equal(probs, expected_probs)


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_class, logits",
    [
        ({0: "A", 1: "B", 2: "C"}, [1.0, 2.0, 3.0]),
        ({0: "A", 1: "B", 2: "C", 3: "A"}, [0.5, 0.5, 0.5]),
        ({10: "B", 20: "C"}, [1.0, 1.0, 1.0]),
    ],
)
def test_set_multiple_with_logits(
    classification_results: ClassificationResults,
    id_to_class: Mapping[IdentifierType, str],
    logits: List[float],
):
    arr_logits = np.array(logits)
    logits_data = {k: arr_logits for k in id_to_class.keys()}
    classification_results.set_multiple(id_to_class=id_to_class, id_to_logits=logits_data)
    assert classification_results.n_items == len(id_to_class)
    for identifier, expected_class in id_to_class.items():
        actual = classification_results.get_predicted_class(identifier=identifier)
        assert actual == expected_class


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_class",
    [
        {0: "A", 1: "B", 2: "C"},
        {0: "A", 1: "B", 2: "C", 3: "A"},
        {10: "B", 20: "C"},
    ],
)
def test_set_multiple_without_logits(
    classification_results: ClassificationResults,
    id_to_class: Mapping[IdentifierType, str],
):
    classification_results.set_multiple(id_to_class=id_to_class)
    assert classification_results.n_items == len(id_to_class)
    class_names = list(classification_results.class_names)
    for identifier, expected_class in id_to_class.items():
        probs = classification_results.get_probs(identifier=identifier)
        expected_idx = class_names.index(expected_class)
        assert probs[expected_idx] == pytest.approx(expected=1.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_class",
    [
        {0: "A", 1: "B", 2: "C"},
        {0: "A", 1: "B", 2: "C", 3: "A"},
        {10: "B", 20: "C"},
    ],
)
def test_id_to_predicted_class(
    classification_results: ClassificationResults,
    id_to_class: Mapping[IdentifierType, str],
):
    classification_results.set_multiple(id_to_class=id_to_class)
    assert classification_results.id_to_predicted_class == id_to_class


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_class, id_to_class_idx",
    [
        ({0: "A", 1: "B", 2: "C"}, {0: 0, 1: 1, 2: 2}),
        ({0: "A", 1: "B", 2: "C", 3: "A"}, {0: 0, 1: 1, 2: 2, 3: 0}),
        ({10: "B", 20: "C"}, {10: 1, 20: 2}),
    ],
)
def test_id_to_predicted_class_idx(
    classification_results: ClassificationResults,
    id_to_class: Mapping[IdentifierType, str],
    id_to_class_idx: Mapping[IdentifierType, int],
):
    classification_results.set_multiple(id_to_class=id_to_class)
    result = classification_results.id_to_predicted_class_idx
    assert result == id_to_class_idx


@pytest.mark.unit
@pytest.mark.parametrize(
    "identifier, logits",
    [
        (0, [1.0, 2.0, 3.0]),
        (5, [-1.0, 0.0, 1.0]),
    ],
)
def test_id_to_logits_and_probs(
    classification_results: ClassificationResults,
    identifier: IdentifierType,
    logits: List[float],
):
    arr_logits = np.array(logits)
    classification_results.set_single(identifier=identifier, predicted_class="A", logits=arr_logits)

    id_to_logits = classification_results.id_to_logits
    assert identifier in id_to_logits
    np.testing.assert_array_almost_equal(id_to_logits[identifier], arr_logits)

    id_to_probs = classification_results.id_to_probs
    assert identifier in id_to_probs
    assert pytest.approx(expected=sum(id_to_probs[identifier])) == 1.0


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_class",
    [
        {0: "A", 1: "B", 2: "C"},
        {0: "A", 1: "B", 2: "C", 3: "A"},
        {10: "B", 20: "C"},
    ],
)
def test_ids(
    classification_results: ClassificationResults,
    id_to_class: Mapping[IdentifierType, str],
):
    classification_results.set_multiple(id_to_class=id_to_class)
    ids = list(classification_results.ids)
    assert ids == list(id_to_class.keys())


@pytest.mark.unit
@pytest.mark.parametrize(
    "class_names, label_name",
    [
        (["A", "B", "C"], "target"),
        (["neg", "pos"], "label"),
    ],
)
def test_pred_store_and_distn_store(class_names: List[str], label_name: str):
    spec = ClassSpec(class_names=class_names, label_name=label_name)
    pred_store = ClassStore(class_spec=spec)
    distn_store = ClassDistributionStore(class_spec=spec)
    results = ClassificationResults(class_spec=spec, pred_store=pred_store, distn_store=distn_store)
    assert results.pred_store is not None
    assert results.distn_store is not None
    assert results.pred_store == pred_store
    assert results.distn_store == distn_store


@pytest.mark.unit
@pytest.mark.parametrize(
    "identifier, predicted_class",
    [
        (0, "A"),
        (5, "B"),
        (99, "C"),
    ],
)
def test_repr(
    classification_results: ClassificationResults,
    identifier: IdentifierType,
    predicted_class: str,
):
    classification_results.set_single(identifier=identifier, predicted_class=predicted_class)
    repr_str = repr(classification_results)
    assert "ClassificationResults" in repr_str
    label = classification_results.label_name
    assert label in repr_str
    assert "n_items=1" in repr_str
