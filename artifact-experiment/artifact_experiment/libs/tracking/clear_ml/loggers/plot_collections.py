import os
from typing import Dict

from matplotlib.figure import Figure

from artifact_experiment.libs.tracking.clear_ml.adapter import ClearMLRunAdapter
from artifact_experiment.libs.tracking.clear_ml.loggers.base import ClearMLArtifactLogger
from artifact_experiment.libs.tracking.clear_ml.readers.plots import ClearMLPlotReader


class ClearMLPlotCollectionLogger(ClearMLArtifactLogger[Dict[str, Figure]]):
    def _append(self, artifact_path: str, artifact: Dict[str, Figure]):
        iteration = self._get_plot_collection_iteration(run=self._run, path=artifact_path)
        for plot_name, plot in artifact.items():
            self._run.log_plot(
                plot=plot, title=artifact_path, series=plot_name, iteration=iteration
            )

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return os.path.join("plot_collections", artifact_name)

    @classmethod
    def _get_plot_collection_iteration(cls, run: ClearMLRunAdapter, path: str) -> int:
        ls_series_metadata = ClearMLPlotReader.get_series_metadata(run=run, plot_path=path)
        if ls_series_metadata:
            iteration = 1 + max(
                ClearMLPlotReader.get_plot_iter_from_metadata(dict_plot_metadata=dict_plot_metadata)
                for dict_plot_metadata in ls_series_metadata
            )
        else:
            iteration = 0
        return iteration
