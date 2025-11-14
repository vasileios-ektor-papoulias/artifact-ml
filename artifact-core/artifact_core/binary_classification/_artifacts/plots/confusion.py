from dataclasses import dataclass
from typing import Type, TypeVar, Union

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
from artifact_core.binary_classification._artifacts.base import BinaryClassificationPlot
from artifact_core.binary_classification._registries.plots import BinaryClassificationPlotRegistry
from artifact_core.binary_classification._types.plots import BinaryClassificationPlotType

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
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> Figure:
        plot_cm = ConfusionMatrixPlotter.plot(
            true=true_class_store.id_to_is_positive,
            predicted=classification_results.id_to_predicted_positive,
            normalization=self._hyperparams.normalization,
        )
        return plot_cm
