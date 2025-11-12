from matplotlib.figure import Figure

from artifact_core._base.contracts.hyperparams import NoArtifactHyperparams
from artifact_core._libs.artifacts.binary_classification.score_distribution.plotter import (
    ScorePDFPlotter,
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


@BinaryClassificationPlotRegistry.register_artifact(BinaryClassificationPlotType.SCORE_PDF)
class ScoreDistributionPlot(BinaryClassificationPlot[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> Figure:
        plot_cm = ScorePDFPlotter.plot_overlaid(
            id_to_is_pos=true_category_store.id_to_is_positive,
            id_to_prob_pos=classification_results.id_to_prob_pos,
        )
        return plot_cm
