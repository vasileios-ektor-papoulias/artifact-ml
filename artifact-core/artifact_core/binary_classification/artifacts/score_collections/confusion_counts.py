from typing import Dict

from artifact_core.base.artifact_dependencies import NoArtifactHyperparams
from artifact_core.binary_classification.artifacts.base import (
    BinaryClassificationScoreCollection,
)
from artifact_core.binary_classification.registries.score_collections.registry import (
    BinaryClassificationScoreCollectionRegistry,
)
from artifact_core.binary_classification.registries.score_collections.types import (
    BinaryClassificationScoreCollectionType,
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


@BinaryClassificationScoreCollectionRegistry.register_artifact(
    BinaryClassificationScoreCollectionType.CONFUSION_COUNTS
)
class ConfusionCounts(BinaryClassificationScoreCollection[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> Dict[str, float]:
        dict_confusion_counts = ConfusionCalculator.compute_confusion_counts(
            true=true_category_store.id_to_category,
            predicted=classification_results.id_to_predicted_category,
            pos_label=self._resource_spec.positive_category,
            neg_label=self._resource_spec.negative_category,
        )
        result: Dict[str, float] = {
            metric_type.value: metric_value
            for metric_type, metric_value in dict_confusion_counts.items()
        }
        return result
