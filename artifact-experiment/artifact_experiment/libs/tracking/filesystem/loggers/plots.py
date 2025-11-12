import os

from artifact_experiment.libs.tracking.filesystem.adapter import InactiveFilesystemRunError
from artifact_experiment.libs.tracking.filesystem.loggers.artifacts import FilesystemArtifactLogger
from artifact_experiment.libs.utils.incremental_path_generator import IncrementalPathGenerator


class FilesystemPlotLogger(FilesystemArtifactLogger[Figure]):
    _fmt: str = "png"
    _dpi: int = 300
    _bbox_inches: str = "tight"

    def _append(self, item_path: str, item: Figure):
        if self._run.is_active:
            self._export_plot(dir_path=item_path, plot=item)
        else:
            raise InactiveFilesystemRunError("Run is inactive")

    @classmethod
    def _export_plot(cls, dir_path: str, plot: Figure):
        filepath = IncrementalPathGenerator.generate(dir_path=dir_path, fmt=cls._fmt)
        plot.savefig(fname=filepath, dpi=cls._dpi, bbox_inches=cls._bbox_inches, format=cls._fmt)

    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return os.path.join("plots", item_name)
