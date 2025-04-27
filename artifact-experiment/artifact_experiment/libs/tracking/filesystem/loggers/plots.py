from matplotlib.figure import Figure

from artifact_experiment.base.tracking.logger import ArtifactLogger
from artifact_experiment.libs.tracking.filesystem.backend import (
    FilesystemBackend,
    FilesystemExperimentNotSetError,
)
from artifact_experiment.libs.utils.filesystem import (
    IncrementalPathGenerator,
)


class FilesystemPlotLogger(ArtifactLogger[Figure, FilesystemBackend]):
    _fmt: str = "png"
    _dpi: int = 300
    _bbox_inches: str = "tight"

    def __init__(self, backend: FilesystemBackend):
        self._backend = backend

    def _log(self, path: str, artifact: Figure):
        if self._backend.experiment_is_active:
            self._export_plot(dir_path=path, plot=artifact)
        else:
            raise FilesystemExperimentNotSetError("No active experiment.")

    @classmethod
    def _export_plot(cls, dir_path: str, plot: Figure):
        filepath = IncrementalPathGenerator.generate(dir_path=dir_path, fmt=cls._fmt)
        plot.savefig(
            fname=filepath,
            dpi=cls._dpi,
            bbox_inches=cls._bbox_inches,
        )

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return f"plots/{artifact_name}"
