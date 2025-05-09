import os
from typing import Dict

from matplotlib.figure import Figure

from artifact_experiment.libs.tracking.clear_ml.loggers.base import ClearMLArtifactLogger


class ClearMLPlotCollectionLogger(ClearMLArtifactLogger[Dict[str, Figure]]):
    def _log(self, path: str, artifact: Dict[str, Figure]):
        for plot_name, plot in artifact.items():
            self._run.log_plot(plot=plot, title=path, series=plot_name, iteration=self._iteration)
        self._iteration += 1

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return os.path.join("plot_collections", artifact_name)
