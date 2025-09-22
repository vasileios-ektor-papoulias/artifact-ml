from dataclasses import dataclass
from typing import Type, TypeVar, Union

from matplotlib.figure import Figure

from artifact_core.base.artifact_dependencies import ArtifactHyperparams
from artifact_core.binary_classification.artifacts.base import (
    BinaryClassificationPlot,
)
from artifact_core.binary_classification.registries.plots.registry import (
    BinaryClassificationPlotRegistry,
)
from artifact_core.binary_classification.registries.plots.types import (
    BinaryClassificationPlotType,
)
from artifact_core.libs.implementation.binary_classification.confusion.normalized_calculator import (
    ConfusionMatrixNormalizationStrategy,
    ConfusionNormalizationStrategyLiteral,
)
from artifact_core.libs.implementation.binary_classification.confusion.plotter import (
    ConfusionMatrixPlotter,
)
from artifact_core.libs.resources.categorical.category_store.binary import (
    BinaryCategoryStore,
)
from artifact_core.libs.resources.classification.binary_classification_results import (
    BinaryClassificationResults,
)

ConfusionMatrixPlotHyperparamsT = TypeVar(
    "ConfusionMatrixPlotHyperparamsT",
    bound="ConfusionMatrixPlotHyperparams",
)


@BinaryClassificationPlotRegistry.register_artifact_hyperparams(
    BinaryClassificationPlotType.CONFUSION_MATRIX
)
@dataclass(frozen=True)
class ConfusionMatrixPlotHyperparams(ArtifactHyperparams):
    normalization: ConfusionMatrixNormalizationStrategy

    @classmethod
    def build(
        cls: Type[ConfusionMatrixPlotHyperparamsT],
        normalization: Union[
            ConfusionMatrixNormalizationStrategy, ConfusionNormalizationStrategyLiteral
        ],
    ) -> ConfusionMatrixPlotHyperparamsT:
        normalization = (
            normalization
            if isinstance(normalization, ConfusionMatrixNormalizationStrategy)
            else ConfusionMatrixNormalizationStrategy[normalization]
        )
        hyperparams = cls(normalization=normalization)
        return hyperparams


@BinaryClassificationPlotRegistry.register_artifact(BinaryClassificationPlotType.CONFUSION_MATRIX)
class ConfusionMatrixPlot(BinaryClassificationPlot[ConfusionMatrixPlotHyperparams]):
    def _evaluate_classification(
        self,
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> Figure:
        plot_cm = ConfusionMatrixPlotter.plot(
            true=true_category_store.id_to_category,
            predicted=classification_results.id_to_predicted_category,
            pos_label=self._resource_spec.positive_category,
            neg_label=self._resource_spec.negative_category,
            normalization=self._hyperparams.normalization,
        )
        return plot_cm
