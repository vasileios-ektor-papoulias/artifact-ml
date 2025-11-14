from dataclasses import dataclass
from typing import Dict, Sequence, Type, TypeVar, Union

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
from artifact_core.binary_classification._artifacts.base import BinaryClassificationArrayCollection
from artifact_core.binary_classification._registries.array_collections import (
    BinaryClassificationArrayCollectionRegistry,
)
from artifact_core.binary_classification._types.array_collections import (
    BinaryClassificationArrayCollectionType,
)

ConfusionMatrixCollectionHyperparamsT = TypeVar(
    "ConfusionMatrixCollectionHyperparamsT", bound="ConfusionMatrixCollectionHyperparams"
)


@BinaryClassificationArrayCollectionRegistry.register_artifact_hyperparams(
    BinaryClassificationArrayCollectionType.CONFUSION_MATRICES
)
@dataclass(frozen=True)
class ConfusionMatrixCollectionHyperparams(ArtifactHyperparams):
    normalization_types: Sequence[ConfusionMatrixNormalizationStrategy]

    @classmethod
    def build(
        cls: Type[ConfusionMatrixCollectionHyperparamsT],
        normalization_types: Sequence[
            Union[ConfusionMatrixNormalizationStrategy, ConfusionNormalizationStrategyLiteral]
        ],
    ) -> ConfusionMatrixCollectionHyperparamsT:
        normalization_types_resolved = [
            normalization_type
            if isinstance(normalization_type, ConfusionMatrixNormalizationStrategy)
            else ConfusionMatrixNormalizationStrategy[normalization_type]
            for normalization_type in normalization_types
        ]
        hyperparams = cls(normalization_types=normalization_types_resolved)
        return hyperparams


@BinaryClassificationArrayCollectionRegistry.register_artifact(
    BinaryClassificationArrayCollectionType.CONFUSION_MATRICES
)
class ConfusionMatrixCollection(
    BinaryClassificationArrayCollection[ConfusionMatrixCollectionHyperparams]
):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> Dict[str, Array]:
        array_collection = (
            NormalizedConfusionCalculator.compute_confusion_matrix_multiple_normalizations(
                true=true_class_store.id_to_is_positive,
                predicted=classification_results.id_to_predicted_positive,
                normalization_types=self._hyperparams.normalization_types,
            )
        )
        result = {array_type.value: array for array_type, array in array_collection.items()}
        return result
