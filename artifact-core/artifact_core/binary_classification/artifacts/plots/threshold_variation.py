from matplotlib.figure import Figure

from artifact_core.base.artifact_dependencies import NoArtifactHyperparams
from artifact_core.binary_classification.artifacts.base import BinaryClassificationPlot
from artifact_core.binary_classification.registries.plots.registry import (
    BinaryClassificationPlotRegistry,
)
from artifact_core.binary_classification.registries.plots.types import (
    BinaryClassificationPlotType,
)
from artifact_core.libs.implementation.binary_classification.threshold_variation.plotter import (
    ThresholdVariationCurvePlotter,
    ThresholdVariationCurveType,
)
from artifact_core.libs.resources.categorical.category_store.binary import (
    BinaryCategoryStore,
)
from artifact_core.libs.resources.classification.binary_classification_results import (
    BinaryClassificationResults,
)


@BinaryClassificationPlotRegistry.register_artifact(BinaryClassificationPlotType.ROC)
class ROCPlot(BinaryClassificationPlot[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> Figure:
        return ThresholdVariationCurvePlotter.plot(
            plot_type=ThresholdVariationCurveType.ROC,
            true=true_category_store.id_to_category,
            probs=classification_results.id_to_prob_pos,
            pos_label=self._resource_spec.positive_category,
        )


@BinaryClassificationPlotRegistry.register_artifact(BinaryClassificationPlotType.PR)
class PRPlot(BinaryClassificationPlot[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> Figure:
        return ThresholdVariationCurvePlotter.plot(
            plot_type=ThresholdVariationCurveType.PR,
            true=true_category_store.id_to_category,
            probs=classification_results.id_to_prob_pos,
            pos_label=self._resource_spec.positive_category,
        )


@BinaryClassificationPlotRegistry.register_artifact(BinaryClassificationPlotType.DET)
class DETPlot(BinaryClassificationPlot[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> Figure:
        return ThresholdVariationCurvePlotter.plot(
            plot_type=ThresholdVariationCurveType.DET,
            true=true_category_store.id_to_category,
            probs=classification_results.id_to_prob_pos,
            pos_label=self._resource_spec.positive_category,
        )


@BinaryClassificationPlotRegistry.register_artifact(BinaryClassificationPlotType.TPR_THRESHOLD)
class TPRThresholdPlot(BinaryClassificationPlot[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> Figure:
        return ThresholdVariationCurvePlotter.plot(
            plot_type=ThresholdVariationCurveType.TPR_THRESHOLD,
            true=true_category_store.id_to_category,
            probs=classification_results.id_to_prob_pos,
            pos_label=self._resource_spec.positive_category,
        )


@BinaryClassificationPlotRegistry.register_artifact(
    BinaryClassificationPlotType.PRECISION_THRESHOLD
)
class PrecisionThresholdPlot(BinaryClassificationPlot[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> Figure:
        return ThresholdVariationCurvePlotter.plot(
            plot_type=ThresholdVariationCurveType.PRECISION_THRESHOLD,
            true=true_category_store.id_to_category,
            probs=classification_results.id_to_prob_pos,
            pos_label=self._resource_spec.positive_category,
        )
