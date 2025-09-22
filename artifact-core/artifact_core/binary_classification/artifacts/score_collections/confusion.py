from dataclasses import dataclass
from typing import Dict, Type, TypeVar, Union

from artifact_core.base.artifact_dependencies import ArtifactHyperparams
from artifact_core.binary_classification.artifacts.base import (
    BinaryClassificationScoreCollection,
)
from artifact_core.binary_classification.registries.score_collections.registry import (
    BinaryClassificationScoreCollectionRegistry,
)
from artifact_core.binary_classification.registries.score_collections.types import (
    BinaryClassificationScoreCollectionType,
)
from artifact_core.libs.implementation.binary_classification.confusion.normalized_calculator import (
    ConfusionNormalizationStrategy,
    ConfusionNormalizationStrategyLiteral,
    NormalizedConfusionCalculator,
)
from artifact_core.libs.resources.categorical.category_store.binary import (
    BinaryCategoryStore,
)
from artifact_core.libs.resources.classification.binary_classification_results import (
    BinaryClassificationResults,
)

ConfusionCountsHyperparamsT = TypeVar(
    "ConfusionCountsHyperparamsT",
    bound="ConfusionCountsHyperparams",
)


@BinaryClassificationScoreCollectionRegistry.register_artifact_hyperparams(
    BinaryClassificationScoreCollectionType.CONFUSION_COUNTS
)
@dataclass(frozen=True)
class ConfusionCountsHyperparams(ArtifactHyperparams):
    normalization: ConfusionNormalizationStrategy

    @classmethod
    def build(
        cls: Type[ConfusionCountsHyperparamsT],
        normalization: Union[ConfusionNormalizationStrategy, ConfusionNormalizationStrategyLiteral],
    ) -> ConfusionCountsHyperparamsT:
        normalization = (
            normalization
            if isinstance(normalization, ConfusionNormalizationStrategy)
            else ConfusionNormalizationStrategy[normalization]
        )
        hyperparams = cls(normalization=normalization)
        return hyperparams


@BinaryClassificationScoreCollectionRegistry.register_artifact(
    BinaryClassificationScoreCollectionType.CONFUSION_COUNTS
)
class ConfusionCounts(BinaryClassificationScoreCollection[ConfusionCountsHyperparams]):
    def _evaluate_classification(
        self,
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> Dict[str, float]:
        dict_confusion_counts = (
            NormalizedConfusionCalculator.compute_dict_normalized_confusion_counts(
                true=true_category_store.id_to_category,
                predicted=classification_results.id_to_predicted_category,
                pos_label=self._resource_spec.positive_category,
                neg_label=self._resource_spec.negative_category,
                normalization=self._hyperparams.normalization,
            )
        )
        result: Dict[str, float] = {
            metric_type.value: metric_value
            for metric_type, metric_value in dict_confusion_counts.items()
        }
        return result
