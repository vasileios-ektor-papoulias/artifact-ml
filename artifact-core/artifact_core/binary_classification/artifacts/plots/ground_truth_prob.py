from matplotlib.figure import Figure

from artifact_core.base.artifact_dependencies import NoArtifactHyperparams
from artifact_core.binary_classification.artifacts.base import BinaryClassificationPlot
from artifact_core.binary_classification.registries.plots.registry import (
    BinaryClassificationPlotRegistry,
)
from artifact_core.binary_classification.registries.plots.types import (
    BinaryClassificationPlotType,
)
from artifact_core.libs.implementation.classification.ground_truth_prob.plotter import (
    GroundTruthProbPDFPlotter,
)
from artifact_core.libs.resources.categorical.category_store.binary import (
    BinaryCategoryStore,
)
from artifact_core.libs.resources.classification.binary_classification_results import (
    BinaryClassificationResults,
)


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
