import os
from typing import Dict

from matplotlib.figure import Figure

from artifact_experiment.libs.tracking.clear_ml.adapter import ClearMLRunAdapter
from artifact_experiment.libs.tracking.clear_ml.loggers.base import ClearMLArtifactLogger


class ClearMLPlotCollectionLogger(ClearMLArtifactLogger[Dict[str, Figure]]):
    def _log(self, path: str, artifact: Dict[str, Figure]):
        iteration = self._get_plot_collection_iteration(run=self._run, path=path)
        for plot_name, plot in artifact.items():
            self._run.log_plot(plot=plot, title=path, series=plot_name, iteration=iteration)

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return os.path.join("plot_collections", artifact_name)

    @staticmethod
    def _get_plot_collection_iteration(run: ClearMLRunAdapter, path: str) -> int:
        ls_all_plots = run.get_exported_plots()
        ls_plot_history = [
            dict_plot_metadata
            for dict_plot_metadata in ls_all_plots
            if dict_plot_metadata["metric"] == path
        ]
        if ls_plot_history:
            iteration = 1 + max(
                dict_plot_metadata["iter"] for dict_plot_metadata in ls_plot_history
            )
        else:
            iteration = 0
        return iteration
