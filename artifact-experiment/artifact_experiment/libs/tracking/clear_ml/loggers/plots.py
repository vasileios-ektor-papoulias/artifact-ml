import os

from matplotlib.figure import Figure

from artifact_experiment.libs.tracking.clear_ml.adapter import ClearMLRunAdapter
from artifact_experiment.libs.tracking.clear_ml.loggers.base import ClearMLArtifactLogger
from artifact_experiment.libs.tracking.clear_ml.readers.plots import ClearMLPlotReader


class ClearMLPlotLogger(ClearMLArtifactLogger[Figure]):
    _series_name: str = "plot"

    def _log(self, path: str, artifact: Figure):
        iteration = self._get_plot_iteration(run=self._run, path=path)
        self._run.log_plot(plot=artifact, title=path, series=self._series_name, iteration=iteration)

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return os.path.join("plots", artifact_name)

    @staticmethod
    def _get_plot_iteration(run: ClearMLRunAdapter, path: str) -> int:
        ls_all_plots = ClearMLPlotReader.get_all_plots(run=run)
        ls_series_metadata = ClearMLPlotReader.get_series_from_path(
            ls_all_plots=ls_all_plots, plot_path=path
        )
        if ls_series_metadata:
            dict_plot_metadata = ls_series_metadata[0]
            iteration = 1 + ClearMLPlotReader.get_plot_iter_from_metadata(
                dict_plot_metadata=dict_plot_metadata
            )
        else:
            iteration = 0
        return iteration
