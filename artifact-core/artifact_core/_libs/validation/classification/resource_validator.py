from typing import Iterable

from artifact_core._libs.resources.classification.category_store import CategoryStore
from artifact_core._libs.resources.classification.classification_results import (
    ClassificationResults,
)
from artifact_core._libs.resources.tools.entity_store import IdentifierType


class ClassificationResourceValidator:
    @classmethod
    def validate(
        cls,
        true_category_store: CategoryStore,
        classification_results: ClassificationResults,
    ) -> None:
        cls._require_non_empty_true(true_category_store)
        cls._require_non_empty_results(classification_results)
        cls._require_same_ids(true_category_store.ids, classification_results.ids)
        cls._require_compatible_specs(
            true_category_store=true_category_store,
            classification_results=classification_results,
        )

    @staticmethod
    def _require_non_empty_true(
        true_category_store: CategoryStore,
    ) -> None:
        if len(true_category_store) == 0:
            raise ValueError("Expected non-empty true_categories store.")

    @staticmethod
    def _require_non_empty_results(
        classification_results: ClassificationResults,
    ) -> None:
        if len(classification_results) == 0:
            raise ValueError("Expected non-empty classification_results.")

    @staticmethod
    def _require_same_ids(
        ids_true: Iterable[IdentifierType],
        ids_pred: Iterable[IdentifierType],
    ) -> None:
        set_true, set_pred = set(ids_true), set(ids_pred)
        if set_true != set_pred:
            diff_true = set_true - set_pred
            diff_pred = set_pred - set_true
            msgs = []
            if diff_true:
                msgs.append(f"missing in predictions: {diff_true}")
            if diff_pred:
                msgs.append(f"missing in truths: {diff_pred}")
            raise ValueError("IDs mismatch between true and predicted: " + "; ".join(msgs))

    @staticmethod
    def _require_compatible_specs(
        true_category_store: CategoryStore,
        classification_results: ClassificationResults,
    ) -> None:
        spec_true = true_category_store.ls_categories
        spec_pred = classification_results.ls_categories
        if spec_true != spec_pred:
            raise ValueError(
                "Feature-spec category mismatch between ground truth and predictions.\n"
                f"- true  categories: {spec_true}\n"
                f"- pred  categories: {spec_pred}"
            )
