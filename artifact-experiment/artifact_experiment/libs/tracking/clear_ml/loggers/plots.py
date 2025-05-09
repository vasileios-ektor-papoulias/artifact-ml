import os

from matplotlib.figure import Figure

from artifact_experiment.libs.tracking.clear_ml.loggers.base import ClearMLArtifactLogger


class ClearMLPlotLogger(ClearMLArtifactLogger[Figure]):
    def _log(self, path: str, artifact: Figure):
        self._run.log_plot(plot=artifact, title=path, iteration=self._iteration)
        self._iteration += 1

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return os.path.join("plots", artifact_name)
