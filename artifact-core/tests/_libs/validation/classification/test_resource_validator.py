from typing import List, Mapping

import pytest
from artifact_core._libs.resource_specs.classification.spec import ClassSpec
from artifact_core._libs.resources.classification.class_store import ClassStore
from artifact_core._libs.resources.classification.classification_results import (
    ClassificationResults,
)
from artifact_core._libs.validation.classification.resource_validator import (
    ClassificationResourceValidator,
)
from artifact_core._utils.collections.entity_store import IdentifierType


@pytest.mark.unit
@pytest.mark.parametrize(
    "id_to_class",
    [
        {0: "A", 1: "B", 2: "C"},
        {0: "A"},
        {10: "B", 20: "C", 30: "A"},
    ],
)
def test_validate_success(
    true_class_store: ClassStore,
    classification_results: ClassificationResults,
    id_to_class: Mapping[IdentifierType, str],
):
    true_class_store.set_multiple_cat(id_to_class=id_to_class)
    classification_results.set_multiple(id_to_class=id_to_class)
    ClassificationResourceValidator.validate(
        true_class_store=true_class_store, classification_results=classification_results
    )


@pytest.mark.unit
def test_validate_empty_true_raises(
    true_class_store: ClassStore, classification_results: ClassificationResults
):
    classification_results.set_single(identifier=0, predicted_class="A")
    with pytest.raises(ValueError, match="non-empty true_categories"):
        ClassificationResourceValidator.validate(
            true_class_store=true_class_store,
            classification_results=classification_results,
        )


@pytest.mark.unit
def test_validate_empty_results_raises(
    true_class_store: ClassStore, classification_results: ClassificationResults
):
    true_class_store.set_class(identifier=0, class_name="A")
    with pytest.raises(ValueError, match="non-empty classification_results"):
        ClassificationResourceValidator.validate(
            true_class_store=true_class_store,
            classification_results=classification_results,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "true_ids, pred_ids, expected_match",
    [
        ({0: "A", 1: "B"}, {0: "A", 2: "B"}, "missing in predictions.*1"),
        ({0: "A"}, {0: "A", 1: "B"}, "missing in truths.*1"),
        ({0: "A", 1: "B"}, {2: "A", 3: "B"}, "IDs mismatch"),
    ],
)
def test_validate_id_mismatch_raises(
    true_class_store: ClassStore,
    classification_results: ClassificationResults,
    true_ids: Mapping[IdentifierType, str],
    pred_ids: Mapping[IdentifierType, str],
    expected_match: str,
):
    true_class_store.set_multiple_cat(id_to_class=true_ids)
    classification_results.set_multiple(id_to_class=pred_ids)
    with pytest.raises(ValueError, match=expected_match):
        ClassificationResourceValidator.validate(
            true_class_store=true_class_store, classification_results=classification_results
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "true_class_names, pred_class_names",
    [
        (["A", "B", "C"], ["A", "B"]),
        (["A", "B"], ["A", "B", "C"]),
        (["A", "B", "C"], ["X", "Y", "Z"]),
        (["A", "B", "C"], ["C", "B", "A"]),
    ],
)
def test_validate_spec_mismatch_raises(true_class_names: List[str], pred_class_names: List[str]):
    true_spec = ClassSpec(class_names=true_class_names, label_name="target")
    pred_spec = ClassSpec(class_names=pred_class_names, label_name="target")
    true_store = ClassStore.build_empty(class_spec=true_spec)
    true_store.set_class(identifier=0, class_name=true_class_names[0])
    results = ClassificationResults.build_empty(class_spec=pred_spec)
    results.set_single(identifier=0, predicted_class=pred_class_names[0])
    with pytest.raises(ValueError, match="category mismatch"):
        ClassificationResourceValidator.validate(
            true_class_store=true_store, classification_results=results
        )
