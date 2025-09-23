from typing import Dict, Hashable, Iterable, Mapping, Sequence

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
        stats: Sequence[DescriptiveStatistic],
        ids: Iterable[IdentifierType] | None = None,
    ) -> pd.Series:
        id_to_prob_true: Dict[IdentifierType, float] = (
            GroundTruthProbCalculator.compute_id_to_prob_ground_truth(
                classification_results=classification_results,
                true_category_store=true_category_store,
                ids=ids,
            )
        )
        return cls._compute_from_probs(
            id_to_prob_ground_truth=id_to_prob_true,
            stats=stats,
        )

    @classmethod
    def _compute_from_probs(
        cls,
        id_to_prob_ground_truth: Mapping[Hashable, float],
        stats: Sequence[DescriptiveStatistic],
    ) -> pd.Series:
        probs = np.asarray([float(v) for v in id_to_prob_ground_truth.values()], dtype=float)
        sr_probs = pd.Series(probs, name="prob_true")
        sr_stats = DescriptiveStatsCalculator.compute_sr_stats(sr_cts_data=sr_probs, stats=stats)
        return sr_stats
