import os

from artifact_core.typing import PlotCollection

from artifact_experiment._impl.backends.clear_ml.adapter import ClearMLRunAdapter
from artifact_experiment._impl.backends.clear_ml.loggers.artifacts import ClearMLArtifactLogger


class ClearMLPlotCollectionLogger(ClearMLArtifactLogger[PlotCollection]):
    def _append(self, item_path: str, item: PlotCollection):
        iteration = self._get_plot_collection_iteration(run=self._run, path=item_path)
        for plot_name, plot in item.items():
            self._run.log_plot(plot=plot, title=item_path, series=plot_name, iteration=iteration)

    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return os.path.join("plot_collections", item_name)

    @classmethod
    def _get_plot_collection_iteration(cls, run: ClearMLRunAdapter, path: str) -> int:
        plot_store = run.get_exported_plots()
        plot_series = plot_store.get(path=path)
        if plot_series:
            iteration = 1 + max(plot.n_entries for plot in plot_series)
        else:
            iteration = 0
        return iteration
