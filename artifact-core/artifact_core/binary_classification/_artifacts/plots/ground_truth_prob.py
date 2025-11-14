from matplotlib.figure import Figure

from artifact_core._base.core.hyperparams import NoArtifactHyperparams
from artifact_core._libs.artifacts.classification.ground_truth_prob.plotter import (
    GroundTruthProbPDFPlotter,
)
from artifact_core._libs.resources.binary_classification.class_store import BinaryClassStore
from artifact_core._libs.resources.binary_classification.classification_results import (
    BinaryClassificationResults,
)
from artifact_core.binary_classification._artifacts.base import BinaryClassificationPlot
from artifact_core.binary_classification._registries.plots import BinaryClassificationPlotRegistry
from artifact_core.binary_classification._types.plots import BinaryClassificationPlotType


@BinaryClassificationPlotRegistry.register_artifact(
    BinaryClassificationPlotType.GROUND_TRUTH_PROB_PDF
)
class GroundTruthProbPDFPlot(BinaryClassificationPlot[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> Figure:
        figure = GroundTruthProbPDFPlotter.plot(
            true_class_store=true_class_store,
            classification_results=classification_results,
        )
        return figure
