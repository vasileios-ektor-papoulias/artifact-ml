from typing import Hashable, Mapping

import pandas as pd
from artifact_core._libs.artifacts.classification.ground_truth_prob.calculator import (
    GroundTruthProbCalculator,
)
from artifact_core._libs.resource_specs.classification.protocol import ClassSpecProtocol
from artifact_core._libs.resources.classification.class_store import ClassStore
from artifact_core._libs.resources.classification.classification_results import (
    ClassificationResults,
)
from artifact_core._libs.resources.classification.distribution_store import (
    ClassDistributionStore,
)
from artifact_core._libs.tools.plotters.pdf_plotter import PDFConfig, PDFPlotter
from matplotlib.figure import Figure


class GroundTruthProbPDFPlotter:
    _prob_col_name: str = "P(y=ground_truth)"
    _pdf_config = PDFConfig(
        plot_color="green",
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
            ClassSpecProtocol, ClassStore, ClassDistributionStore
        ],
        true_class_store: ClassStore,
    ) -> Figure:
        id_to_prob_ground_truth = GroundTruthProbCalculator.compute_id_to_prob_ground_truth(
            classification_results=classification_results, true_class_store=true_class_store
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
