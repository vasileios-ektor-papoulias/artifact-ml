from dataclasses import dataclass
from typing import Dict, Sequence, Type, TypeVar, Union

from artifact_core._base.core.hyperparams import ArtifactHyperparams
from artifact_core._libs.artifacts.binary_classification.score_distribution.calculator import (
    BinarySampleSplit,
    DescriptiveStatistic,
    ScoreStatsCalculator,
)
from artifact_core._libs.resources.binary_classification.class_store import BinaryClassStore
from artifact_core._libs.resources.binary_classification.classification_results import (
    BinaryClassificationResults,
)
from artifact_core._libs.tools.calculators.descriptive_stats_calculator import (
    DescriptiveStatisticLiteral,
)
from artifact_core.binary_classification._artifacts.base import BinaryClassificationScoreCollection
from artifact_core.binary_classification._registries.score_collections import (
    BinaryClassificationScoreCollectionRegistry,
)
from artifact_core.binary_classification._types.score_collections import (
    BinaryClassificationScoreCollectionType,
)

ScoreStatsByStatHyperparamsT = TypeVar(
    "ScoreStatsByStatHyperparamsT", bound="ScoreStatsByStatHyperparams"
)


@BinaryClassificationScoreCollectionRegistry.register_artifact_hyperparams(
    BinaryClassificationScoreCollectionType.SCORE_STATS
)
@BinaryClassificationScoreCollectionRegistry.register_artifact_hyperparams(
    BinaryClassificationScoreCollectionType.POSITIVE_CLASS_SCORE_STATS
)
@BinaryClassificationScoreCollectionRegistry.register_artifact_hyperparams(
    BinaryClassificationScoreCollectionType.NEGATIVE_CLASS_SCORE_STATS
)
@dataclass(frozen=True)
class ScoreStatsByStatHyperparams(ArtifactHyperparams):
    stat_types: Sequence[DescriptiveStatistic]

    @classmethod
    def build(
        cls: Type[ScoreStatsByStatHyperparamsT],
        stat_types: Sequence[Union[DescriptiveStatistic, DescriptiveStatisticLiteral]],
    ) -> ScoreStatsByStatHyperparamsT:
        ls_resolved = [
            stat_type
            if isinstance(stat_type, DescriptiveStatistic)
            else DescriptiveStatistic[stat_type]
            for stat_type in stat_types
        ]
        hyperparams = cls(stat_types=ls_resolved)
        return hyperparams


@BinaryClassificationScoreCollectionRegistry.register_artifact(
    BinaryClassificationScoreCollectionType.SCORE_STATS
)
class ScoreStats(BinaryClassificationScoreCollection[ScoreStatsByStatHyperparams]):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> Dict[str, float]:
        dict_scores = ScoreStatsCalculator.compute_by_split(
            id_to_is_pos=true_class_store.id_to_is_positive,
            id_to_prob_pos=classification_results.id_to_prob_pos,
            stats=self._hyperparams.stat_types,
            split=BinarySampleSplit.ALL,
        )
        result = {split_type.name: stat_value for split_type, stat_value in dict_scores.items()}
        return result


@BinaryClassificationScoreCollectionRegistry.register_artifact(
    BinaryClassificationScoreCollectionType.POSITIVE_CLASS_SCORE_STATS
)
class PositiveClassScoreStats(BinaryClassificationScoreCollection[ScoreStatsByStatHyperparams]):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> Dict[str, float]:
        dict_scores = ScoreStatsCalculator.compute_by_split(
            id_to_is_pos=true_class_store.id_to_is_positive,
            id_to_prob_pos=classification_results.id_to_prob_pos,
            stats=self._hyperparams.stat_types,
            split=BinarySampleSplit.POSITIVE,
        )
        result = {split_type.name: stat_value for split_type, stat_value in dict_scores.items()}
        return result


@BinaryClassificationScoreCollectionRegistry.register_artifact(
    BinaryClassificationScoreCollectionType.NEGATIVE_CLASS_SCORE_STATS
)
class NegativeClassScoreStats(BinaryClassificationScoreCollection[ScoreStatsByStatHyperparams]):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> Dict[str, float]:
        dict_scores = ScoreStatsCalculator.compute_by_split(
            id_to_is_pos=true_class_store.id_to_is_positive,
            id_to_prob_pos=classification_results.id_to_prob_pos,
            stats=self._hyperparams.stat_types,
            split=BinarySampleSplit.NEGATIVE,
        )
        result = {split_type.name: stat_value for split_type, stat_value in dict_scores.items()}
        return result
