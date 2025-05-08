import os
from typing import Dict

from matplotlib.figure import Figure

from artifact_experiment.libs.tracking.filesystem.adapter import (
    NoActiveFilesystemRunError,
)
from artifact_experiment.libs.tracking.filesystem.loggers.base import FilesystemArtifactLogger
from artifact_experiment.libs.utils.filesystem import (
    IncrementalPathGenerator,
)


class FilesystemPlotCollectionLogger(FilesystemArtifactLogger[Dict[str, Figure]]):
    _fmt: str = "png"
    _dpi: int = 300
    _bbox_inches: str = "tight"

    def _log(self, path: str, artifact: Dict[str, Figure]):
        if self._run.is_active:
            self._export_plot_collection(dir_path=path, plot_collection=artifact)
        else:
            raise NoActiveFilesystemRunError("No active run.")

    @classmethod
    def _export_plot_collection(cls, dir_path: str, plot_collection: Dict[str, Figure]):
        step_path = IncrementalPathGenerator.generate(dir_path=dir_path, fmt=cls._fmt)
        os.makedirs(name=step_path, exist_ok=True)
        for plot_name, plot in plot_collection.items():
            filepath = os.path.join(step_path, plot_name)
            plot.savefig(fname=filepath, dpi=cls._dpi, bbox_inches=cls._bbox_inches)

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return f"plot_collections/{artifact_name}"
