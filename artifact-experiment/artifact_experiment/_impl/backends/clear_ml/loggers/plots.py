import os

from artifact_core.typing import Plot

from artifact_experiment._impl.backends.clear_ml.adapter import ClearMLRunAdapter
from artifact_experiment._impl.backends.clear_ml.loggers.artifacts import ClearMLArtifactLogger


class ClearMLPlotLogger(ClearMLArtifactLogger[Plot]):
    _series_name: str = "plot"

    def _append(self, item_path: str, item: Plot):
        iteration = self._get_plot_iteration(run=self._run, path=item_path)
        self._run.log_plot(
            plot=item, title=item_path, series=self._series_name, iteration=iteration
        )

    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return os.path.join("plots", item_name)

    @staticmethod
    def _get_plot_iteration(run: ClearMLRunAdapter, path: str) -> int:
        plot_store = run.get_exported_plots()
        plot_series = plot_store.get(path=path)
        if plot_series:
            plot = plot_series[0]
            iteration = 1 + plot.n_entries
        else:
            iteration = 0
        return iteration
