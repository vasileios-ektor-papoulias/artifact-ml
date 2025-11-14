from dataclasses import dataclass
from typing import Type, TypeVar, Union

from artifact_core._base.core.hyperparams import ArtifactHyperparams
from artifact_core._base.typing.artifact_result import Array
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
from artifact_core.binary_classification._artifacts.base import BinaryClassificationArray
from artifact_core.binary_classification._registries.arrays import BinaryClassificationArrayRegistry
from artifact_core.binary_classification._types.arrays import BinaryClassificationArrayType

ConfusionMatrixHyperparamsT = TypeVar(
    "ConfusionMatrixHyperparamsT", bound="ConfusionMatrixHyperparams"
)


@BinaryClassificationArrayRegistry.register_artifact_hyperparams(
    BinaryClassificationArrayType.CONFUSION_MATRIX
)
@dataclass(frozen=True)
class ConfusionMatrixHyperparams(ArtifactHyperparams):
    normalization: ConfusionMatrixNormalizationStrategy

    @classmethod
    def build(
        cls: Type[ConfusionMatrixHyperparamsT],
        normalization: Union[
            ConfusionMatrixNormalizationStrategy, ConfusionNormalizationStrategyLiteral
        ],
    ) -> ConfusionMatrixHyperparamsT:
        normalization = (
            normalization
            if isinstance(normalization, ConfusionMatrixNormalizationStrategy)
            else ConfusionMatrixNormalizationStrategy[normalization]
        )
        hyperparams = cls(normalization=normalization)
        return hyperparams


@BinaryClassificationArrayRegistry.register_artifact(BinaryClassificationArrayType.CONFUSION_MATRIX)
class ConfusionMatrix(BinaryClassificationArray[ConfusionMatrixHyperparams]):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> Array:
        arr_cm = NormalizedConfusionCalculator.compute_normalized_confusion_matrix(
            true=true_class_store.id_to_is_positive,
            predicted=classification_results.id_to_predicted_positive,
            normalization=self._hyperparams.normalization,
        )
        return arr_cm
