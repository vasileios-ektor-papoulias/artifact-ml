import os

from artifact_core.typing import PlotCollection

from artifact_experiment._impl.backends.filesystem.adapter import InactiveFilesystemRunError
from artifact_experiment._impl.backends.filesystem.loggers.artifacts import FilesystemArtifactLogger
from artifact_experiment._utils.filesystem.incremental_paths import IncrementalPathGenerator


class FilesystemPlotCollectionLogger(FilesystemArtifactLogger[PlotCollection]):
    _fmt: str = "png"
    _dpi: int = 300
    _bbox_inches: str = "tight"

    def _append(self, item_path: str, item: PlotCollection):
        if self._run.is_active:
            self._export_plot_collection(dir_path=item_path, plot_collection=item)
        else:
            raise InactiveFilesystemRunError("Run is inactive")

    @classmethod
    def _export_plot_collection(cls, dir_path: str, plot_collection: PlotCollection):
        step_path = IncrementalPathGenerator.generate(dir_path=dir_path)
        os.makedirs(name=step_path, exist_ok=True)
        for plot_name, plot in plot_collection.items():
            filename = cls._append_extension(filename=plot_name, extension=cls._fmt)
            plot_path = os.path.join(step_path, filename)
            plot.savefig(
                fname=plot_path, dpi=cls._dpi, bbox_inches=cls._bbox_inches, format=cls._fmt
            )

    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return os.path.join("plot_collections", item_name)

    @staticmethod
    def _append_extension(filename: str, extension: str) -> str:
        if not extension.startswith("."):
            extension = "." + extension
        root, _ = os.path.splitext(filename)
        return root + extension
