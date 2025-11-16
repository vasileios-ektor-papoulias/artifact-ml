import os
import tempfile
from typing import List

from artifact_core.typing import PlotCollection
from mlflow.entities import FileInfo

from artifact_experiment._base.primitives.file import File
from artifact_experiment._impl.backends.mlflow.adapter import MlflowRunAdapter
from artifact_experiment._impl.backends.mlflow.loggers.artifacts import MlflowArtifactLogger
from artifact_experiment._utils.filesystem.incremental_paths import IncrementalPathFormatter


class MlflowPlotCollectionLogger(MlflowArtifactLogger[PlotCollection]):
    _fmt = "png"

    def _append(self, item_path: str, item: PlotCollection):
        ls_history = self._get_plot_collection_history(
            run=self._run, backend_artifact_path=item_path
        )
        next_step = self._get_next_step_from_history(ls_history=ls_history)
        with tempfile.TemporaryDirectory() as temp_dir:
            collection_local_path = self._get_collection_local_path(
                export_dir=temp_dir, iteration=next_step
            )
            collection_backend_path = self._get_collection_backend_path(
                artifact_path=item_path, iteration=next_step
            )
            for plot_name, plot in item.items():
                plot_local_path = self._get_plot_path_from_collection_path(
                    collection_path=collection_local_path, plot_name=plot_name, fmt=self._fmt
                )
                plot.savefig(fname=plot_local_path, format="png", bbox_inches="tight")
                file = File(path_source=plot_local_path)
                self._run.log_file(backend_dir=collection_backend_path, file=file)

    @staticmethod
    def _get_plot_collection_history(
        run: MlflowRunAdapter, backend_artifact_path: str
    ) -> List[FileInfo]:
        ls_history = run.get_ls_artifact_info(backend_path=backend_artifact_path)
        return ls_history

    @staticmethod
    def _get_next_step_from_history(ls_history: List[FileInfo]) -> int:
        return 1 + len(ls_history)

    @staticmethod
    def _get_plot_path_from_collection_path(collection_path: str, plot_name: str, fmt: str) -> str:
        plot_path = os.path.join(collection_path, plot_name + f".{fmt}")
        return plot_path

    @staticmethod
    def _get_collection_local_path(export_dir: str, iteration: int) -> str:
        collection_local_path = IncrementalPathFormatter.format(
            dir_path=export_dir, next_idx=iteration
        )
        os.makedirs(collection_local_path, exist_ok=True)
        return collection_local_path

    @staticmethod
    def _get_collection_backend_path(artifact_path: str, iteration: int) -> str:
        collection_backend_path = IncrementalPathFormatter.format(
            dir_path=artifact_path, next_idx=iteration
        )
        return collection_backend_path

    @classmethod
    def _get_relative_path(cls, item_name: str) -> str:
        return os.path.join("plot_collections", item_name)
