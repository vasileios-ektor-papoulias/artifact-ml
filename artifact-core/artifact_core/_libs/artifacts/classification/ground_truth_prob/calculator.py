from typing import Dict, Iterable

from artifact_core._base.types.artifact_result import Array
from artifact_core._libs.resource_specs.classification.protocol import (
    CategoricalFeatureSpecProtocol,
)
from artifact_core._libs.resources.classification.category_store import CategoryStore
from artifact_core._libs.resources.classification.classification_results import (
    ClassificationResults,
)
from artifact_core._libs.resources.classification.distribution_store import (
    CategoricalDistributionStore,
)
from artifact_core._libs.resources.tools.entity_store import IdentifierType


class GroundTruthProbCalculator:
    @classmethod
    def compute_ground_truth_prob(
        cls,
        classification_results: ClassificationResults[
            CategoricalFeatureSpecProtocol,
            CategoryStore,
            CategoricalDistributionStore,
        ],
        true_category_store: CategoryStore,
        identifier: IdentifierType,
    ) -> float:
        true_idx = cls._get_true_idx(true_category_store, identifier)
        probs = classification_results.get_probs(identifier=identifier)
        cls._validate_index(true_idx, probs, context=f"probs for id={identifier!r}")
        return float(probs[true_idx])

    @classmethod
    def compute_ground_truth_logit(
        cls,
        classification_results: ClassificationResults[
            CategoricalFeatureSpecProtocol,
            CategoryStore,
            CategoricalDistributionStore,
        ],
        true_category_store: CategoryStore,
        identifier: IdentifierType,
    ) -> float:
        true_idx = cls._get_true_idx(true_category_store, identifier)
        logits = classification_results.get_logits(identifier=identifier)
        cls._validate_index(true_idx, logits, context=f"logits for id={identifier!r}")
        return float(logits[true_idx])

    @classmethod
    def compute_id_to_prob_ground_truth(
        cls,
        classification_results: ClassificationResults[
            CategoricalFeatureSpecProtocol,
            CategoryStore,
            CategoricalDistributionStore,
        ],
        true_category_store: CategoryStore,
    ) -> Dict[IdentifierType, float]:
        iter_id = cls._iter_common_ids(classification_results, true_category_store)
        id_to_prob_ground_truth: Dict[IdentifierType, float] = {}
        for identifier in iter_id:
            id_to_prob_ground_truth[identifier] = cls.compute_ground_truth_prob(
                classification_results=classification_results,
                true_category_store=true_category_store,
                identifier=identifier,
            )
        return id_to_prob_ground_truth

    @classmethod
    def compute_id_to_logit_ground_truth(
        cls,
        classification_results: ClassificationResults[
            CategoricalFeatureSpecProtocol,
            CategoryStore,
            CategoricalDistributionStore,
        ],
        true_category_store: CategoryStore,
    ) -> Dict[IdentifierType, float]:
        iter_ids = cls._iter_common_ids(
            classification_results=classification_results, true_category_store=true_category_store
        )
        id_to_logit_ground_truth: Dict[IdentifierType, float] = {}
        for identifier in iter_ids:
            id_to_logit_ground_truth[identifier] = cls.compute_ground_truth_logit(
                classification_results=classification_results,
                true_category_store=true_category_store,
                identifier=identifier,
            )
        return id_to_logit_ground_truth

    @staticmethod
    def _get_true_idx(true_category_store: CategoryStore, identifier: IdentifierType) -> int:
        return true_category_store.get_category_idx(identifier=identifier)

    @staticmethod
    def _validate_index(idx: int, arr: Array, *, context: str) -> None:
        if not (0 <= idx < arr.shape[-1]):
            raise IndexError(
                f"True category index {idx} out of bounds for {context}; array shape={arr.shape}"
            )

    @staticmethod
    def _iter_common_ids(
        classification_results: ClassificationResults[
            CategoricalFeatureSpecProtocol, CategoryStore, CategoricalDistributionStore
        ],
        true_category_store: CategoryStore,
    ) -> Iterable[IdentifierType]:
        cr_ids = set(classification_results.ids)
        gt_ids = set(true_category_store.id_to_category_idx.keys())
        common = cr_ids & gt_ids
        if not common:
            raise KeyError("No common ids between classification_results and true_category_store.")
        return [i for i in classification_results.ids if i in common]
