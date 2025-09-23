from typing import Hashable, Mapping

import numpy as np
import pandas as pd
from artifact_core.libs.implementation.classification.ground_truth_prob.calculator import (
    GroundTruthProbCalculator,
)
from artifact_core.libs.implementation.tabular.pdf.plotter import PDFPlotter
from artifact_core.libs.resource_spec.categorical.protocol import CategoricalFeatureSpecProtocol
from artifact_core.libs.resources.categorical.category_store.category_store import CategoryStore
from artifact_core.libs.resources.categorical.distribution_store.distribution_store import (
    CategoricalDistributionStore,
)
from artifact_core.libs.resources.classification.classification_results import ClassificationResults
from artifact_core.libs.utils.plot_combiner import PlotCombinationConfig
from matplotlib.figure import Figure


class GroundTruthProbPDFPlotter:
    _prob_col_name: str = "P(y=ground_truth)"

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
        probs = np.asarray([float(v) for v in id_to_prob_ground_truth.values()], dtype=float)
        df = pd.DataFrame({cls._prob_col_name: probs})
        fig = _GroundTruthProbPDFPlotter.get_pdf_plot(
            dataset=df,
            ls_features_order=[cls._prob_col_name],
            ls_cts_features=[cls._prob_col_name],
            ls_cat_features=[],
            cat_unique_map={},
        )
        return fig

    @classmethod
    def _plot(cls, id_to_prob_ground_truth: Mapping[Hashable, float]) -> Figure:
        probs = np.asarray([float(v) for v in id_to_prob_ground_truth.values()], dtype=float)
        df = pd.DataFrame({cls._prob_col_name: probs})

        fig = _GroundTruthProbPDFPlotter.get_pdf_plot(
            dataset=df,
            ls_features_order=[cls._prob_col_name],
            ls_cts_features=[cls._prob_col_name],
            ls_cat_features=[],
            cat_unique_map={},
        )
        return fig


class _GroundTruthProbPDFPlotter(PDFPlotter):
    _plot_color = "tab:green"
    _gridline_color = "grey"
    _gridline_style = "--"
    _axis_font_size = "12"
    _minor_ax_grid_linewidth = 0.2
    _major_ax_grid_linewidth = 0.8
    _cts_density_enable_kde = True
    _cts_densitiy_alpha = 0.6
    _cts_density_n_bins = 70

    _plot_combiner_config = PlotCombinationConfig(
        n_cols=1,
        dpi=150,
        figsize_horizontal_multiplier=6,
        figsize_vertical_multiplier=4,
        tight_layout_rect=(0, 0, 1, 0.95),
        tight_layout_pad=0.1,
        subplots_adjust_hspace=0.1,
        subplots_adjust_wspace=0.1,
        fig_title_fontsize=5,
        include_fig_titles=False,
        combined_title="Ground-Truth Probability PDF",
        combined_title_vertical_position=1,
    )
