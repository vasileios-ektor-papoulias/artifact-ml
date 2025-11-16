from dataclasses import dataclass
from typing import Dict, Sequence, Type, TypeVar, Union

from artifact_core._base.core.hyperparams import ArtifactHyperparams
from artifact_core._libs.artifacts.binary_classification.score_distribution.calculator import (
    BinarySampleSplit,
    ScoreStatsCalculator,
)
from artifact_core._libs.artifacts.binary_classification.score_distribution.partitioner import (
    BinarySampleSplitLiteral,
)
from artifact_core._libs.resources.binary_classification.class_store import BinaryClassStore
from artifact_core._libs.resources.binary_classification.classification_results import (
    BinaryClassificationResults,
)
from artifact_core._libs.tools.calculators.descriptive_stats_calculator import (
    DescriptiveStatistic,
)
from artifact_core.binary_classification._artifacts.base import BinaryClassificationScoreCollection
from artifact_core.binary_classification._registries.score_collections import (
    BinaryClassificationScoreCollectionRegistry,
)
from artifact_core.binary_classification._types.score_collections import (
    BinaryClassificationScoreCollectionType,
)

ScoreStatsBySplitHyperparamsT = TypeVar(
    "ScoreStatsBySplitHyperparamsT", bound="ScoreStatsBySplitHyperparams"
)


@BinaryClassificationScoreCollectionRegistry.register_artifact_hyperparams(
    BinaryClassificationScoreCollectionType.SCORE_MEANS,
)
@BinaryClassificationScoreCollectionRegistry.register_artifact_hyperparams(
    BinaryClassificationScoreCollectionType.SCORE_STDS,
)
@BinaryClassificationScoreCollectionRegistry.register_artifact_hyperparams(
    BinaryClassificationScoreCollectionType.SCORE_VARIANCES,
)
@BinaryClassificationScoreCollectionRegistry.register_artifact_hyperparams(
    BinaryClassificationScoreCollectionType.SCORE_MEDIANS,
)
@BinaryClassificationScoreCollectionRegistry.register_artifact_hyperparams(
    BinaryClassificationScoreCollectionType.SCORE_FIRST_QUARTILES,
)
@BinaryClassificationScoreCollectionRegistry.register_artifact_hyperparams(
    BinaryClassificationScoreCollectionType.SCORE_THIRD_QUARTILES,
)
@BinaryClassificationScoreCollectionRegistry.register_artifact_hyperparams(
    BinaryClassificationScoreCollectionType.SCORE_MINIMA,
)
@BinaryClassificationScoreCollectionRegistry.register_artifact_hyperparams(
    BinaryClassificationScoreCollectionType.SCORE_MAXIMA,
)
@dataclass(frozen=True)
class ScoreStatsBySplitHyperparams(ArtifactHyperparams):
    split_types: Sequence[BinarySampleSplit]

    @classmethod
    def build(
        cls: Type[ScoreStatsBySplitHyperparamsT],
        split_types: Sequence[Union[BinarySampleSplit, BinarySampleSplitLiteral]],
    ) -> ScoreStatsBySplitHyperparamsT:
        ls_resolved = [
            split_type
            if isinstance(split_type, BinarySampleSplit)
            else BinarySampleSplit[split_type]
            for split_type in split_types
        ]
        hyperparams = cls(split_types=ls_resolved)
        return hyperparams


@BinaryClassificationScoreCollectionRegistry.register_artifact(
    BinaryClassificationScoreCollectionType.SCORE_MEANS
)
class ScoreMeans(BinaryClassificationScoreCollection[ScoreStatsBySplitHyperparams]):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> Dict[str, float]:
        dict_scores = ScoreStatsCalculator.compute_by_stat(
            id_to_is_pos=true_class_store.id_to_is_positive,
            id_to_prob_pos=classification_results.id_to_prob_pos,
            stat=DescriptiveStatistic.MEAN,
            splits=self._hyperparams.split_types,
        )
        result = {split_type.name: stat_value for split_type, stat_value in dict_scores.items()}
        return result


@BinaryClassificationScoreCollectionRegistry.register_artifact(
    BinaryClassificationScoreCollectionType.SCORE_STDS
)
class ScoreStds(BinaryClassificationScoreCollection[ScoreStatsBySplitHyperparams]):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> Dict[str, float]:
        dict_scores = ScoreStatsCalculator.compute_by_stat(
            id_to_is_pos=true_class_store.id_to_is_positive,
            id_to_prob_pos=classification_results.id_to_prob_pos,
            stat=DescriptiveStatistic.STD,
            splits=self._hyperparams.split_types,
        )
        result = {split_type.name: stat_value for split_type, stat_value in dict_scores.items()}
        return result


@BinaryClassificationScoreCollectionRegistry.register_artifact(
    BinaryClassificationScoreCollectionType.SCORE_VARIANCES
)
class ScoreVariances(BinaryClassificationScoreCollection[ScoreStatsBySplitHyperparams]):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> Dict[str, float]:
        dict_scores = ScoreStatsCalculator.compute_by_stat(
            id_to_is_pos=true_class_store.id_to_is_positive,
            id_to_prob_pos=classification_results.id_to_prob_pos,
            stat=DescriptiveStatistic.VARIANCE,
            splits=self._hyperparams.split_types,
        )
        result = {split_type.name: stat_value for split_type, stat_value in dict_scores.items()}
        return result


@BinaryClassificationScoreCollectionRegistry.register_artifact(
    BinaryClassificationScoreCollectionType.SCORE_MEDIANS
)
class ScoreMedians(BinaryClassificationScoreCollection[ScoreStatsBySplitHyperparams]):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> Dict[str, float]:
        dict_scores = ScoreStatsCalculator.compute_by_stat(
            id_to_is_pos=true_class_store.id_to_is_positive,
            id_to_prob_pos=classification_results.id_to_prob_pos,
            stat=DescriptiveStatistic.MEDIAN,
            splits=self._hyperparams.split_types,
        )
        result = {split_type.name: stat_value for split_type, stat_value in dict_scores.items()}
        return result


@BinaryClassificationScoreCollectionRegistry.register_artifact(
    BinaryClassificationScoreCollectionType.SCORE_FIRST_QUARTILES
)
class ScoreFirstQuartiles(BinaryClassificationScoreCollection[ScoreStatsBySplitHyperparams]):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> Dict[str, float]:
        dict_scores = ScoreStatsCalculator.compute_by_stat(
            id_to_is_pos=true_class_store.id_to_is_positive,
            id_to_prob_pos=classification_results.id_to_prob_pos,
            stat=DescriptiveStatistic.Q1,
            splits=self._hyperparams.split_types,
        )
        result = {split_type.name: stat_value for split_type, stat_value in dict_scores.items()}
        return result


@BinaryClassificationScoreCollectionRegistry.register_artifact(
    BinaryClassificationScoreCollectionType.SCORE_THIRD_QUARTILES
)
class ScoreThirdQuartiles(BinaryClassificationScoreCollection[ScoreStatsBySplitHyperparams]):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> Dict[str, float]:
        dict_scores = ScoreStatsCalculator.compute_by_stat(
            id_to_is_pos=true_class_store.id_to_is_positive,
            id_to_prob_pos=classification_results.id_to_prob_pos,
            stat=DescriptiveStatistic.Q3,
            splits=self._hyperparams.split_types,
        )
        result = {split_type.name: stat_value for split_type, stat_value in dict_scores.items()}
        return result


@BinaryClassificationScoreCollectionRegistry.register_artifact(
    BinaryClassificationScoreCollectionType.SCORE_MINIMA
)
class ScoreMinima(BinaryClassificationScoreCollection[ScoreStatsBySplitHyperparams]):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> Dict[str, float]:
        dict_scores = ScoreStatsCalculator.compute_by_stat(
            id_to_is_pos=true_class_store.id_to_is_positive,
            id_to_prob_pos=classification_results.id_to_prob_pos,
            stat=DescriptiveStatistic.MIN,
            splits=self._hyperparams.split_types,
        )
        result = {split_type.name: stat_value for split_type, stat_value in dict_scores.items()}
        return result


@BinaryClassificationScoreCollectionRegistry.register_artifact(
    BinaryClassificationScoreCollectionType.SCORE_MAXIMA
)
class ScoreMaxima(BinaryClassificationScoreCollection[ScoreStatsBySplitHyperparams]):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> Dict[str, float]:
        dict_scores = ScoreStatsCalculator.compute_by_stat(
            id_to_is_pos=true_class_store.id_to_is_positive,
            id_to_prob_pos=classification_results.id_to_prob_pos,
            stat=DescriptiveStatistic.MAX,
            splits=self._hyperparams.split_types,
        )
        result = {split_type.name: stat_value for split_type, stat_value in dict_scores.items()}
        return result
