from typing import Dict, Sequence

import numpy as np
import pandas as pd
from artifact_core._libs.artifacts.classification.ground_truth_prob.calculator import (
    GroundTruthProbCalculator,
)
from artifact_core._libs.artifacts.tools.calculators.descriptive_stats_calculator import (
    DescriptiveStatistic,
    DescriptiveStatsCalculator,
)
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


class GroundTruthProbStatsCalculator:
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
    ) -> float:
        sr_probs = cls._compute_sr_probs(
            classification_results=classification_results, true_category_store=true_category_store
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
    ) -> Dict[DescriptiveStatistic, float]:
        if not stats:
            return {}
        sr_probs = cls._compute_sr_probs(
            classification_results=classification_results, true_category_store=true_category_store
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
    ) -> pd.Series:
        id_to_prob_ground_truth = GroundTruthProbCalculator.compute_id_to_prob_ground_truth(
            classification_results=classification_results, true_category_store=true_category_store
        )
        probs = np.asarray([float(v) for v in id_to_prob_ground_truth.values()], dtype=float)
        sr_probs = pd.Series(probs, name="prob_ground_truth")
        return sr_probs
