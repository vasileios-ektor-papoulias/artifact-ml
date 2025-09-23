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
class ROCCurve(BinaryClassificationPlot[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> Figure:
        return ThresholdVariationCurvePlotter.plot(
            curve_type=ThresholdVariationCurveType.ROC,
            true=true_category_store.id_to_is_positive,
            probs=classification_results.id_to_prob_pos,
        )


@BinaryClassificationPlotRegistry.register_artifact(BinaryClassificationPlotType.PR)
class PRCurve(BinaryClassificationPlot[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> Figure:
        return ThresholdVariationCurvePlotter.plot(
            curve_type=ThresholdVariationCurveType.PR,
            true=true_category_store.id_to_is_positive,
            probs=classification_results.id_to_prob_pos,
        )


@BinaryClassificationPlotRegistry.register_artifact(BinaryClassificationPlotType.DET)
class DETCurve(BinaryClassificationPlot[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> Figure:
        return ThresholdVariationCurvePlotter.plot(
            curve_type=ThresholdVariationCurveType.DET,
            true=true_category_store.id_to_is_positive,
            probs=classification_results.id_to_prob_pos,
        )


@BinaryClassificationPlotRegistry.register_artifact(BinaryClassificationPlotType.TPR_THRESHOLD)
class TPRThresholdCurve(BinaryClassificationPlot[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> Figure:
        return ThresholdVariationCurvePlotter.plot(
            curve_type=ThresholdVariationCurveType.TPR_THRESHOLD,
            true=true_category_store.id_to_is_positive,
            probs=classification_results.id_to_prob_pos,
        )


@BinaryClassificationPlotRegistry.register_artifact(
    BinaryClassificationPlotType.PRECISION_THRESHOLD
)
class PrecisionThresholdCurve(BinaryClassificationPlot[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> Figure:
        return ThresholdVariationCurvePlotter.plot(
            curve_type=ThresholdVariationCurveType.PRECISION_THRESHOLD,
            true=true_category_store.id_to_is_positive,
            probs=classification_results.id_to_prob_pos,
        )
