import numpy as np

from artifact_core.base.artifact_dependencies import NoArtifactHyperparams
from artifact_core.binary_classification.artifacts.base import (
    BinaryClassificationArray,
)
from artifact_core.binary_classification.registries.arrays.registry import (
    BinaryClassificationArrayRegistry,
)
from artifact_core.binary_classification.registries.arrays.types import (
    BinaryClassificationArrayType,
)
from artifact_core.libs.implementation.binary_classification.confusion.calculator import (
    ConfusionCalculator,
)
from artifact_core.libs.resources.categorical.category_store.binary import (
    BinaryCategoryStore,
)
from artifact_core.libs.resources.classification.binary_classification_results import (
    BinaryClassificationResults,
)


@BinaryClassificationArrayRegistry.register_artifact(BinaryClassificationArrayType.CONFUSION_MATRIX)
class ConfusionMatrix(BinaryClassificationArray[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> np.ndarray:
        arr_cm = ConfusionCalculator.compute_confusion_matrix(
            true=true_category_store.id_to_category,
            predicted=classification_results.id_to_predicted_category,
            pos_label=self._resource_spec.positive_category,
            neg_label=self._resource_spec.negative_category,
        )
        return arr_cm
