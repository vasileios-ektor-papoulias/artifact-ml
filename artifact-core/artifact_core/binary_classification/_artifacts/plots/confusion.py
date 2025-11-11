from dataclasses import dataclass
from typing import Type, TypeVar, Union

from matplotlib.figure import Figure

from artifact_core._base.artifact_dependencies import ArtifactHyperparams
from artifact_core._libs.implementation.binary_classification.confusion.calculator import (
    ConfusionMatrixNormalizationStrategy,
)
from artifact_core._libs.implementation.binary_classification.confusion.normalizer import (
    ConfusionNormalizationStrategyLiteral,
)
from artifact_core._libs.implementation.binary_classification.confusion.plotter import (
    ConfusionMatrixPlotter,
)
from artifact_core._libs.resources.categorical.category_store.binary import (
    BinaryCategoryStore,
)
from artifact_core._libs.resources.classification.binary_classification_results import (
    BinaryClassificationResults,
)
from artifact_core.binary_classification._artifacts.base import (
    BinaryClassificationPlot,
)
from artifact_core.binary_classification._registries.plots.registry import (
    BinaryClassificationPlotRegistry,
)
from artifact_core.binary_classification._registries.plots.types import (
    BinaryClassificationPlotType,
)

ConfusionMatrixPlotHyperparamsT = TypeVar(
    "ConfusionMatrixPlotHyperparamsT", bound="ConfusionMatrixPlotHyperparams"
)


@BinaryClassificationPlotRegistry.register_artifact_hyperparams(
    BinaryClassificationPlotType.CONFUSION_MATRIX_PLOT
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


@BinaryClassificationPlotRegistry.register_artifact(
    BinaryClassificationPlotType.CONFUSION_MATRIX_PLOT
)
class ConfusionMatrixPlot(BinaryClassificationPlot[ConfusionMatrixPlotHyperparams]):
    def _evaluate_classification(
        self,
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> Figure:
        plot_cm = ConfusionMatrixPlotter.plot(
            true=true_category_store.id_to_is_positive,
            predicted=classification_results.id_to_predicted_positive,
            normalization=self._hyperparams.normalization,
        )
        return plot_cm
