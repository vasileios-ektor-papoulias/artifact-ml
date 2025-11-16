from dataclasses import dataclass
from typing import Dict, Sequence, Type, TypeVar, Union

from matplotlib.figure import Figure

from artifact_core._base.core.hyperparams import ArtifactHyperparams
from artifact_core._libs.artifacts.binary_classification.confusion.calculator import (
    ConfusionMatrixNormalizationStrategy,
)
from artifact_core._libs.artifacts.binary_classification.confusion.normalizer import (
    ConfusionNormalizationStrategyLiteral,
)
from artifact_core._libs.artifacts.binary_classification.confusion.plotter import (
    ConfusionMatrixPlotter,
)
from artifact_core._libs.resources.binary_classification.class_store import BinaryClassStore
from artifact_core._libs.resources.binary_classification.classification_results import (
    BinaryClassificationResults,
)
from artifact_core.binary_classification._artifacts.base import BinaryClassificationPlotCollection
from artifact_core.binary_classification._registries.plot_collections import (
    BinaryClassificationPlotCollectionRegistry,
)
from artifact_core.binary_classification._types.plot_collections import (
    BinaryClassificationPlotCollectionType,
)

ConfusionMatrixPlotCollectionHyperparamsT = TypeVar(
    "ConfusionMatrixPlotCollectionHyperparamsT", bound="ConfusionMatrixPlotCollectionHyperparams"
)


@BinaryClassificationPlotCollectionRegistry.register_artifact_hyperparams(
    BinaryClassificationPlotCollectionType.CONFUSION_MATRIX_PLOTS
)
@dataclass(frozen=True)
class ConfusionMatrixPlotCollectionHyperparams(ArtifactHyperparams):
    normalization_types: Sequence[ConfusionMatrixNormalizationStrategy]

    @classmethod
    def build(
        cls: Type[ConfusionMatrixPlotCollectionHyperparamsT],
        normalization_types: Sequence[
            Union[ConfusionMatrixNormalizationStrategy, ConfusionNormalizationStrategyLiteral]
        ],
    ) -> ConfusionMatrixPlotCollectionHyperparamsT:
        normalization_types_resolved = [
            normalization_type
            if isinstance(normalization_type, ConfusionMatrixNormalizationStrategy)
            else ConfusionMatrixNormalizationStrategy[normalization_type]
            for normalization_type in normalization_types
        ]
        hyperparams = cls(normalization_types=normalization_types_resolved)
        return hyperparams


@BinaryClassificationPlotCollectionRegistry.register_artifact(
    BinaryClassificationPlotCollectionType.CONFUSION_MATRIX_PLOTS
)
class ConfusionMatrixPlotCollection(
    BinaryClassificationPlotCollection[ConfusionMatrixPlotCollectionHyperparams]
):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> Dict[str, Figure]:
        plot_collection = ConfusionMatrixPlotter.plot_multiple(
            true=true_class_store.id_to_is_positive,
            predicted=classification_results.id_to_predicted_positive,
            normalization_types=self._hyperparams.normalization_types,
        )
        result = {plot_type.value: plot for plot_type, plot in plot_collection.items()}
        return result
