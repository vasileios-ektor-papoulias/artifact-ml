from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd
from artifact_core.libs.implementation.classification.ground_truth_prob.calculator import (
    GroundTruthProbCalculator,
)
from artifact_core.libs.resource_spec.categorical.protocol import CategoricalFeatureSpecProtocol
from artifact_core.libs.resources.categorical.category_store.category_store import CategoryStore
from artifact_core.libs.resources.categorical.distribution_store.distribution_store import (
    CategoricalDistributionStore,
)
from artifact_core.libs.resources.classification.classification_results import ClassificationResults
from artifact_core.libs.types.entity_store import IdentifierType
from artifact_core.libs.utils.descriptive_stats_calculator import (
    DescriptiveStatistic,
    DescriptiveStatsCalculator,
)


class GroundTruthStatsCalculator:
    @classmethod
    def compute(
        cls,
        classification_results: ClassificationResults[
            CategoricalFeatureSpecProtocol,
            CategoryStore,
            CategoricalDistributionStore,
        ],
        true_category_store: CategoryStore,
        stat: DescriptiveStatistic,
        ids: Iterable[IdentifierType] | None = None,
    ) -> float:
        sr_probs = cls._compute_sr_probs(
            classification_results=classification_results,
            true_category_store=true_category_store,
            ids=ids,
        )
        stat_value = DescriptiveStatsCalculator.compute_stat(sr_cts_data=sr_probs, stat=stat)
        return float(stat_value) if pd.notna(stat_value) else float("nan")

    @classmethod
    def compute_multiple(
        cls,
        classification_results: ClassificationResults[
            CategoricalFeatureSpecProtocol,
            CategoryStore,
            CategoricalDistributionStore,
        ],
        true_category_store: CategoryStore,
        stats: Sequence[DescriptiveStatistic],
        ids: Iterable[IdentifierType] | None = None,
    ) -> Dict[DescriptiveStatistic, float]:
        if not stats:
            return {}
        sr_probs = cls._compute_sr_probs(
            classification_results=classification_results,
            true_category_store=true_category_store,
            ids=ids,
        )
        dict_stats = DescriptiveStatsCalculator.compute_dict_stats(
            sr_cts_data=sr_probs, stats=stats
        )
        return dict_stats

    @classmethod
    def _compute_sr_probs(
        cls,
        classification_results: ClassificationResults[
            CategoricalFeatureSpecProtocol,
            CategoryStore,
            CategoricalDistributionStore,
        ],
        true_category_store: CategoryStore,
        ids: Iterable[IdentifierType] | None,
    ) -> pd.Series:
        id_to_prob_true = GroundTruthProbCalculator.compute_id_to_prob_ground_truth(
            classification_results=classification_results,
            true_category_store=true_category_store,
            ids=ids,
        )
        probs = np.asarray([float(v) for v in id_to_prob_true.values()], dtype=float)
        sr_probs = pd.Series(probs, name="prob_true")
        return sr_probs
