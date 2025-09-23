from typing import Dict, Iterable

import numpy as np
from artifact_core.libs.resource_spec.categorical.protocol import CategoricalFeatureSpecProtocol
from artifact_core.libs.resources.categorical.category_store.category_store import CategoryStore
from artifact_core.libs.resources.categorical.distribution_store.distribution_store import (
    CategoricalDistributionStore,
)
from artifact_core.libs.resources.classification.classification_results import ClassificationResults
from artifact_core.libs.types.entity_store import IdentifierType


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
        ids: Iterable[IdentifierType] | None = None,
    ) -> Dict[IdentifierType, float]:
        for_id = cls._iter_common_ids(classification_results, true_category_store, ids)
        out: Dict[IdentifierType, float] = {}
        for identifier in for_id:
            out[identifier] = cls.compute_ground_truth_prob(
                classification_results, true_category_store, identifier
            )
        return out

    @classmethod
    def compute_id_to_logit_ground_truth(
        cls,
        classification_results: ClassificationResults[
            CategoricalFeatureSpecProtocol,
            CategoryStore,
            CategoricalDistributionStore,
        ],
        true_category_store: CategoryStore,
        ids: Iterable[IdentifierType] | None = None,
    ) -> Dict[IdentifierType, float]:
        for_id = cls._iter_common_ids(classification_results, true_category_store, ids)
        out: Dict[IdentifierType, float] = {}
        for identifier in for_id:
            out[identifier] = cls.compute_ground_truth_logit(
                classification_results, true_category_store, identifier
            )
        return out

    @staticmethod
    def _get_true_idx(true_category_store: CategoryStore, identifier: IdentifierType) -> int:
        return true_category_store.get_category_idx(identifier=identifier)

    @staticmethod
    def _validate_index(idx: int, arr: np.ndarray, *, context: str) -> None:
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
        ids: Iterable[IdentifierType] | None,
    ) -> Iterable[IdentifierType]:
        if ids is not None:
            ids_list = list(ids)
            missing_cr = [i for i in ids_list if i not in set(classification_results.ids)]
            missing_gt = [
                i for i in ids_list if i not in set(true_category_store.id_to_category_idx)
            ]
            if missing_cr:
                raise KeyError(
                    f"IDs not found in classification_results: "
                    f"{missing_cr[:5]}{'...' if len(missing_cr) > 5 else ''}"
                )
            if missing_gt:
                raise KeyError(
                    f"IDs not found in true_category_store: "
                    f"{missing_gt[:5]}{'...' if len(missing_gt) > 5 else ''}"
                )
            return ids_list
        cr_ids = set(classification_results.ids)
        gt_ids = set(true_category_store.id_to_category_idx.keys())
        common = cr_ids & gt_ids
        if not common:
            raise KeyError("No common ids between classification_results and true_category_store.")
        return [i for i in classification_results.ids if i in common]
