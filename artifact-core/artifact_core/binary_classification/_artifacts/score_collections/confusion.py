from dataclasses import dataclass
from typing import Dict, Type, TypeVar, Union

from artifact_core._base.core.hyperparams import ArtifactHyperparams
from artifact_core._libs.artifacts.binary_classification.confusion.calculator import (
    ConfusionMatrixNormalizationStrategy,
    NormalizedConfusionCalculator,
)
from artifact_core._libs.artifacts.binary_classification.confusion.normalizer import (
    ConfusionNormalizationStrategyLiteral,
)
from artifact_core._libs.resources.binary_classification.class_store import BinaryClassStore
from artifact_core._libs.resources.binary_classification.classification_results import (
    BinaryClassificationResults,
)
from artifact_core.binary_classification._artifacts.base import BinaryClassificationScoreCollection
from artifact_core.binary_classification._registries.score_collections import (
    BinaryClassificationScoreCollectionRegistry,
)
from artifact_core.binary_classification._types.score_collections import (
    BinaryClassificationScoreCollectionType,
)

ConfusionCountsHyperparamsT = TypeVar(
    "ConfusionCountsHyperparamsT", bound="NormalizedConfusionCountsHyperparams"
)


@BinaryClassificationScoreCollectionRegistry.register_artifact_hyperparams(
    BinaryClassificationScoreCollectionType.NORMALIZED_CONFUSION_COUNTS
)
@dataclass(frozen=True)
class NormalizedConfusionCountsHyperparams(ArtifactHyperparams):
    normalization: ConfusionMatrixNormalizationStrategy

    @classmethod
    def build(
        cls: Type[ConfusionCountsHyperparamsT],
        normalization: Union[
            ConfusionMatrixNormalizationStrategy, ConfusionNormalizationStrategyLiteral
        ],
    ) -> ConfusionCountsHyperparamsT:
        normalization = (
            normalization
            if isinstance(normalization, ConfusionMatrixNormalizationStrategy)
            else ConfusionMatrixNormalizationStrategy[normalization]
        )
        hyperparams = cls(normalization=normalization)
        return hyperparams


@BinaryClassificationScoreCollectionRegistry.register_artifact(
    BinaryClassificationScoreCollectionType.NORMALIZED_CONFUSION_COUNTS
)
class NormalizedConfusionCounts(
    BinaryClassificationScoreCollection[NormalizedConfusionCountsHyperparams]
):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> Dict[str, float]:
        dict_confusion_counts = (
            NormalizedConfusionCalculator.compute_dict_normalized_confusion_counts(
                true=true_class_store.id_to_is_positive,
                predicted=classification_results.id_to_predicted_positive,
                normalization=self._hyperparams.normalization,
            )
        )
        result: Dict[str, float] = {
            metric_type.value: metric_value
            for metric_type, metric_value in dict_confusion_counts.items()
        }
        return result
