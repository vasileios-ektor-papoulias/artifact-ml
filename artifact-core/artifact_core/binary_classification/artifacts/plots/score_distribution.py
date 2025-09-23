from matplotlib.figure import Figure

from artifact_core.base.artifact_dependencies import NoArtifactHyperparams
from artifact_core.binary_classification.artifacts.base import (
    BinaryClassificationPlot,
)
from artifact_core.binary_classification.registries.plots.registry import (
    BinaryClassificationPlotRegistry,
)
from artifact_core.binary_classification.registries.plots.types import (
    BinaryClassificationPlotType,
)
from artifact_core.libs.implementation.binary_classification.score_distribution.plotter import (
    ScoreDistributionPlotter,
)
from artifact_core.libs.resources.categorical.category_store.binary import (
    BinaryCategoryStore,
)
from artifact_core.libs.resources.classification.binary_classification_results import (
    BinaryClassificationResults,
)


@BinaryClassificationPlotRegistry.register_artifact(
    BinaryClassificationPlotType.SCORE_DISTRIBUTION_PLOT
)
class ScoreDistributionPlot(BinaryClassificationPlot[NoArtifactHyperparams]):
    def _evaluate_classification(
        self,
        true_category_store: BinaryCategoryStore,
        classification_results: BinaryClassificationResults,
    ) -> Figure:
        plot_cm = ScoreDistributionPlotter.plot_overlaid(
            id_to_is_pos=true_category_store.id_to_is_positive,
            id_to_prob_pos=classification_results.id_to_prob_pos,
        )
        return plot_cm
