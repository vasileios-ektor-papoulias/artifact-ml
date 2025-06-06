import os
from typing import Dict

from matplotlib.figure import Figure

from artifact_experiment.libs.tracking.filesystem.adapter import (
    InactiveFilesystemRunError,
)
from artifact_experiment.libs.tracking.filesystem.loggers.base import FilesystemArtifactLogger
from artifact_experiment.libs.utils.incremental_path_generator import IncrementalPathGenerator


class FilesystemPlotCollectionLogger(FilesystemArtifactLogger[Dict[str, Figure]]):
    _fmt: str = "png"
    _dpi: int = 300
    _bbox_inches: str = "tight"

    def _append(self, artifact_path: str, artifact: Dict[str, Figure]):
        if self._run.is_active:
            self._export_plot_collection(dir_path=artifact_path, plot_collection=artifact)
        else:
            raise InactiveFilesystemRunError("Run is inactive")

    @classmethod
    def _export_plot_collection(cls, dir_path: str, plot_collection: Dict[str, Figure]):
        step_path = IncrementalPathGenerator.generate(dir_path=dir_path)
        os.makedirs(name=step_path, exist_ok=True)
        for plot_name, plot in plot_collection.items():
            filename = cls._append_extension(filename=plot_name, extension=cls._fmt)
            plot_path = os.path.join(step_path, filename)
            plot.savefig(
                fname=plot_path, dpi=cls._dpi, bbox_inches=cls._bbox_inches, format=cls._fmt
            )

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return os.path.join("plot_collections", artifact_name)

    @staticmethod
    def _append_extension(filename: str, extension: str) -> str:
        if not extension.startswith("."):
            extension = "." + extension
        root, _ = os.path.splitext(filename)
        return root + extension
