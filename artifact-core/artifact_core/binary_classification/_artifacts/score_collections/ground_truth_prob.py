from dataclasses import dataclass
from typing import Dict, Sequence, Type, TypeVar, Union

from artifact_core._base.core.hyperparams import ArtifactHyperparams
from artifact_core._libs.artifacts.classification.ground_truth_prob.stats_calculator import (
    GroundTruthProbStatsCalculator,
)
from artifact_core._libs.resources.binary_classification.class_store import BinaryClassStore
from artifact_core._libs.resources.binary_classification.classification_results import (
    BinaryClassificationResults,
)
from artifact_core._libs.tools.calculators.descriptive_stats_calculator import (
    DescriptiveStatistic,
    DescriptiveStatisticLiteral,
)
from artifact_core.binary_classification._artifacts.base import BinaryClassificationScoreCollection
from artifact_core.binary_classification._registries.score_collections import (
    BinaryClassificationScoreCollectionRegistry,
)
from artifact_core.binary_classification._types.score_collections import (
    BinaryClassificationScoreCollectionType,
)

GroundTruthProbStatsHyperparamsT = TypeVar(
    "GroundTruthProbStatsHyperparamsT", bound="GroundTruthProbStatsHyperparams"
)


@BinaryClassificationScoreCollectionRegistry.register_artifact_hyperparams(
    BinaryClassificationScoreCollectionType.GROUND_TRUTH_PROB_STATS
)
@dataclass(frozen=True)
class GroundTruthProbStatsHyperparams(ArtifactHyperparams):
    stat_types: Sequence[DescriptiveStatistic]

    @classmethod
    def build(
        cls: Type[GroundTruthProbStatsHyperparamsT],
        stat_types: Sequence[Union[DescriptiveStatistic, DescriptiveStatisticLiteral]],
    ) -> GroundTruthProbStatsHyperparamsT:
        ls_resolved = [
            stat_type
            if isinstance(stat_type, DescriptiveStatistic)
            else DescriptiveStatistic[stat_type]
            for stat_type in stat_types
        ]
        hyperparams = cls(stat_types=ls_resolved)
        return hyperparams


@BinaryClassificationScoreCollectionRegistry.register_artifact(
    BinaryClassificationScoreCollectionType.GROUND_TRUTH_PROB_STATS
)
class GroundTruthProbStats(BinaryClassificationScoreCollection[GroundTruthProbStatsHyperparams]):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> Dict[str, float]:
        dict_scores = GroundTruthProbStatsCalculator.compute_multiple(
            stats=self._hyperparams.stat_types,
            true_class_store=true_class_store,
            classification_results=classification_results,
        )
        result = {stat_type.value: stat_value for stat_type, stat_value in dict_scores.items()}
        return result
