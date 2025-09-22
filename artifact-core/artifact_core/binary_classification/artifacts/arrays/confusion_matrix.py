from dataclasses import dataclass
from typing import Type, TypeVar, Union

import numpy as np

from artifact_core.base.artifact_dependencies import ArtifactHyperparams
from artifact_core.binary_classification.artifacts.base import (
    BinaryClassificationArray,
)
from artifact_core.binary_classification.registries.arrays.registry import (
    BinaryClassificationArrayRegistry,
)
from artifact_core.binary_classification.registries.arrays.types import (
    BinaryClassificationArrayType,
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

ConfusionMatrixHyperparamsT = TypeVar(
    "ConfusionMatrixHyperparamsT",
    bound="ConfusionMatrixHyperparams",
)


@BinaryClassificationArrayRegistry.register_artifact_hyperparams(
    BinaryClassificationArrayType.CONFUSION_MATRIX
)
@dataclass(frozen=True)
class ConfusionMatrixHyperparams(ArtifactHyperparams):
    normalization: ConfusionNormalizationStrategy

    @classmethod
    def build(
        cls: Type[ConfusionMatrixHyperparamsT],
        normalization: Union[ConfusionNormalizationStrategy, ConfusionNormalizationStrategyLiteral],
    ) -> ConfusionMatrixHyperparamsT:
        normalization = (
            normalization
            if isinstance(normalization, ConfusionNormalizationStrategy)
            else ConfusionNormalizationStrategy[normalization]
        )
        hyperparams = cls(normalization=normalization)
        return hyperparams


@BinaryClassificationArrayRegistry.register_artifact(BinaryClassificationArrayType.CONFUSION_MATRIX)
class ConfusionMatrix(BinaryClassificationArray[ConfusionMatrixHyperparams]):
    def _evaluate_classification(
        self,
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> np.ndarray:
        arr_cm = NormalizedConfusionCalculator.compute_normalized_confusion_matrix(
            true=true_category_store.id_to_category,
            predicted=classification_results.id_to_predicted_category,
            pos_label=self._resource_spec.positive_category,
            neg_label=self._resource_spec.negative_category,
            normalization=self._hyperparams.normalization,
        )
        return arr_cm
