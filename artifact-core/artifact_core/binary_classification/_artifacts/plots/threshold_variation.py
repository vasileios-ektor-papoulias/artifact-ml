from matplotlib.figure import Figure

from artifact_core._base.core.hyperparams import NoArtifactHyperparams
from artifact_core._libs.artifacts.binary_classification.threshold_variation.plotter import (
    ThresholdVariationCurvePlotter,
    ThresholdVariationCurveType,
)
from artifact_core._libs.resources.binary_classification.class_store import BinaryClassStore
from artifact_core._libs.resources.binary_classification.classification_results import (
    BinaryClassificationResults,
)
from artifact_core.binary_classification._artifacts.base import BinaryClassificationPlot
from artifact_core.binary_classification._registries.plots import BinaryClassificationPlotRegistry
from artifact_core.binary_classification._types.plots import BinaryClassificationPlotType


@BinaryClassificationPlotRegistry.register_artifact(BinaryClassificationPlotType.ROC_CURVE)
class ROCCurve(BinaryClassificationPlot[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> Figure:
        return ThresholdVariationCurvePlotter.plot(
            curve_type=ThresholdVariationCurveType.ROC,
            true=true_class_store.id_to_is_positive,
            probs=classification_results.id_to_prob_pos,
        )


@BinaryClassificationPlotRegistry.register_artifact(BinaryClassificationPlotType.PR_CURVE)
class PRCurve(BinaryClassificationPlot[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> Figure:
        return ThresholdVariationCurvePlotter.plot(
            curve_type=ThresholdVariationCurveType.PR,
            true=true_class_store.id_to_is_positive,
            probs=classification_results.id_to_prob_pos,
        )


@BinaryClassificationPlotRegistry.register_artifact(BinaryClassificationPlotType.DET_CURVE)
class DETCurve(BinaryClassificationPlot[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> Figure:
        return ThresholdVariationCurvePlotter.plot(
            curve_type=ThresholdVariationCurveType.DET,
            true=true_class_store.id_to_is_positive,
            probs=classification_results.id_to_prob_pos,
        )


@BinaryClassificationPlotRegistry.register_artifact(
    BinaryClassificationPlotType.RECALL_THRESHOLD_CURVE
)
class RecallThresholdCurve(BinaryClassificationPlot[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> Figure:
        return ThresholdVariationCurvePlotter.plot(
            curve_type=ThresholdVariationCurveType.RECALL_THRESHOLD,
            true=true_class_store.id_to_is_positive,
            probs=classification_results.id_to_prob_pos,
        )


@BinaryClassificationPlotRegistry.register_artifact(
    BinaryClassificationPlotType.PRECISION_THRESHOLD_CURVE
)
class PrecisionThresholdCurve(BinaryClassificationPlot[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> Figure:
        return ThresholdVariationCurvePlotter.plot(
            curve_type=ThresholdVariationCurveType.PRECISION_THRESHOLD,
            true=true_class_store.id_to_is_positive,
            probs=classification_results.id_to_prob_pos,
        )
