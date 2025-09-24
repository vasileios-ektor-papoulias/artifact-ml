from typing import Hashable, Mapping

import pandas as pd
from artifact_core.libs.implementation.classification.ground_truth_prob.calculator import (
    GroundTruthProbCalculator,
)
from artifact_core.libs.resource_spec.categorical.protocol import CategoricalFeatureSpecProtocol
from artifact_core.libs.resources.categorical.category_store.category_store import CategoryStore
from artifact_core.libs.resources.categorical.distribution_store.distribution_store import (
    CategoricalDistributionStore,
)
from artifact_core.libs.resources.classification.classification_results import ClassificationResults
from artifact_core.libs.utils.plotters.pdf_plotter import PDFConfig, PDFPlotter
from matplotlib.figure import Figure


class GroundTruthProbPDFPlotter:
    _prob_col_name: str = "P(y=ground_truth)"
    _pdf_config = PDFConfig(
        plot_color="tab:olive",
        gridline_color="grey",
        gridline_style="--",
        axis_font_size="12",
        minor_ax_grid_linewidth=0.2,
        major_ax_grid_linewidth=0.8,
        cts_density_enable_kde=True,
        cts_densitiy_alpha=0.6,
        cts_density_n_bins=70,
    )

    @classmethod
    def plot(
        cls,
        classification_results: ClassificationResults[
            CategoricalFeatureSpecProtocol, CategoryStore, CategoricalDistributionStore
        ],
        true_category_store: CategoryStore,
    ) -> Figure:
        id_to_prob_ground_truth = GroundTruthProbCalculator.compute_id_to_prob_ground_truth(
            classification_results=classification_results, true_category_store=true_category_store
        )
        fig = cls._plot(id_to_prob_ground_truth=id_to_prob_ground_truth)
        return fig

    @classmethod
    def _plot(cls, id_to_prob_ground_truth: Mapping[Hashable, float]) -> Figure:
        ls_probs = [float(v) for v in id_to_prob_ground_truth.values()]
        sr_probs = pd.Series(data=ls_probs)
        fig = PDFPlotter.plot_pdf(
            sr_data=sr_probs, feature_name=cls._prob_col_name, config=cls._pdf_config
        )
        return fig
