from matplotlib.figure import Figure

from artifact_core._base.contracts.hyperparams import NoArtifactHyperparams
from artifact_core._libs.artifacts.classification.ground_truth_prob.plotter import (
    GroundTruthProbPDFPlotter,
)
from artifact_core._libs.resources.binary_classification.category_store import BinaryCategoryStore
from artifact_core._libs.resources.binary_classification.classification_results import (
    BinaryClassificationResults,
)
from artifact_core.binary_classification._artifacts.base import BinaryClassificationPlot
from artifact_core.binary_classification._registries.plots.registry import (
    BinaryClassificationPlotRegistry,
)
from artifact_core.binary_classification._registries.plots.types import BinaryClassificationPlotType


@BinaryClassificationPlotRegistry.register_artifact(
    BinaryClassificationPlotType.GROUND_TRUTH_PROB_PDF
)
class GroundTruthProbPDFPlot(BinaryClassificationPlot[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> Figure:
        figure = GroundTruthProbPDFPlotter.plot(
            true_category_store=true_category_store,
            classification_results=classification_results,
        )
        return figure
