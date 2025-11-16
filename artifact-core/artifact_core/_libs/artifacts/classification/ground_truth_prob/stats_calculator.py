from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from artifact_core._libs.artifacts.classification.ground_truth_prob.calculator import (
    GroundTruthProbCalculator,
)
from artifact_core._libs.resource_specs.classification.protocol import (
    ClassSpecProtocol,
)
from artifact_core._libs.resources.classification.class_store import ClassStore
from artifact_core._libs.resources.classification.classification_results import (
    ClassificationResults,
)
from artifact_core._libs.resources.classification.distribution_store import ClassDistributionStore
from artifact_core._libs.tools.calculators.descriptive_stats_calculator import (
    DescriptiveStatistic,
    DescriptiveStatsCalculator,
)


class GroundTruthProbStatsCalculator:
    @classmethod
    def compute(
        cls,
        classification_results: ClassificationResults[
            ClassSpecProtocol,
            ClassStore,
            ClassDistributionStore,
        ],
        true_class_store: ClassStore,
        stat: DescriptiveStatistic,
    ) -> float:
        sr_probs = cls._compute_sr_probs(
            classification_results=classification_results, true_class_store=true_class_store
        )
        stat_value = DescriptiveStatsCalculator.compute_stat(sr_cts_data=sr_probs, stat=stat)
        return float(stat_value) if pd.notna(stat_value) else float("nan")

    @classmethod
    def compute_multiple(
        cls,
        classification_results: ClassificationResults[
            ClassSpecProtocol,
            ClassStore,
            ClassDistributionStore,
        ],
        true_class_store: ClassStore,
        stats: Sequence[DescriptiveStatistic],
    ) -> Mapping[DescriptiveStatistic, float]:
        if not stats:
            return {}
        sr_probs = cls._compute_sr_probs(
            classification_results=classification_results, true_class_store=true_class_store
        )
        dict_stats = DescriptiveStatsCalculator.compute_dict_stats(
            sr_cts_data=sr_probs, stats=stats
        )
        return dict_stats

    @classmethod
    def _compute_sr_probs(
        cls,
        classification_results: ClassificationResults[
            ClassSpecProtocol,
            ClassStore,
            ClassDistributionStore,
        ],
        true_class_store: ClassStore,
    ) -> pd.Series:
        id_to_prob_ground_truth = GroundTruthProbCalculator.compute_id_to_prob_ground_truth(
            classification_results=classification_results, true_class_store=true_class_store
        )
        probs = np.asarray([float(v) for v in id_to_prob_ground_truth.values()], dtype=float)
        sr_probs = pd.Series(probs, name="prob_ground_truth")
        return sr_probs
