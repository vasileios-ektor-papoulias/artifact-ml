import os

from matplotlib.figure import Figure

from artifact_experiment.libs.tracking.clear_ml.adapter import ClearMLRunAdapter
from artifact_experiment.libs.tracking.clear_ml.loggers.base import ClearMLArtifactLogger


class ClearMLPlotLogger(ClearMLArtifactLogger[Figure]):
    _series_name: str = "plot"

    def _log(self, path: str, artifact: Figure):
        iteration = self._get_plot_iteration(run=self._run, path=path, series=self._series_name)
        self._run.log_plot(plot=artifact, title=path, series=self._series_name, iteration=iteration)

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return os.path.join("plots", artifact_name)

    @staticmethod
    def _get_plot_iteration(run: ClearMLRunAdapter, path: str, series: str) -> int:
        ls_all_plots = run.get_exported_plots()
        ls_plot_history = [
            dict_plot_metadata
            for dict_plot_metadata in ls_all_plots
            if dict_plot_metadata["metric"] == path
        ]
        if ls_plot_history:
            iteration = 1 + ls_plot_history[0]["iter"]
        else:
            iteration = 0
        return iteration
