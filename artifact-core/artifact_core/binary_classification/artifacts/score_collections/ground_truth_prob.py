from dataclasses import dataclass
from typing import Dict, Sequence, Type, TypeVar, Union

from artifact_core.base.artifact_dependencies import ArtifactHyperparams
from artifact_core.binary_classification.artifacts.base import BinaryClassificationScoreCollection
from artifact_core.binary_classification.registries.score_collections.registry import (
    BinaryClassificationScoreCollectionRegistry,
)
from artifact_core.binary_classification.registries.score_collections.types import (
    BinaryClassificationScoreCollectionType,
)
from artifact_core.libs.implementation.classification.ground_truth_prob.stats_calculator import (
    DescriptiveStatistic,
    GroundTruthProbStatsCalculator,
)
from artifact_core.libs.resources.categorical.category_store.binary import (
    BinaryCategoryStore,
)
from artifact_core.libs.resources.classification.binary_classification_results import (
    BinaryClassificationResults,
)
from artifact_core.libs.utils.calculators.descriptive_stats_calculator import (
    DescriptiveStatisticLiteral,
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
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> Dict[str, float]:
        dict_scores = GroundTruthProbStatsCalculator.compute_multiple(
            stats=self._hyperparams.stat_types,
            true_category_store=true_category_store,
            classification_results=classification_results,
        )
        result = {stat_type.value: stat_value for stat_type, stat_value in dict_scores.items()}
        return result
