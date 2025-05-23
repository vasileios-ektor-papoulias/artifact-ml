import os
import tempfile
from typing import Dict, List

from matplotlib.figure import Figure
from mlflow.entities import FileInfo

from artifact_experiment.libs.tracking.mlflow.adapter import (
    MlflowRunAdapter,
)
from artifact_experiment.libs.tracking.mlflow.loggers.base import MlflowArtifactLogger
from artifact_experiment.libs.utils.filesystem import IncrementalPathGenerator


class MLFlowPlotCollectionLogger(MlflowArtifactLogger[Dict[str, Figure]]):
    _fmt = "png"

    def _append(self, artifact_path: str, artifact: Dict[str, Figure]):
        ls_history = self._get_plot_collection_history(
            run=self._run, backend_artifact_path=artifact_path
        )
        next_step = self._get_next_step_from_history(ls_history=ls_history)
        with tempfile.TemporaryDirectory() as temp_dir:
            collection_local_path = self._get_collection_local_path(
                export_dir=temp_dir, iteration=next_step
            )
            collection_backend_path = self._get_collection_backend_path(
                artifact_path=artifact_path, iteration=next_step
            )
            for plot_name, plot in artifact.items():
                plot_local_path = self._get_plot_path_from_collection_path(
                    collection_path=collection_local_path, plot_name=plot_name, fmt=self._fmt
                )
                plot.savefig(fname=plot_local_path, format="png", bbox_inches="tight")
                self._run.upload(
                    backend_path=collection_backend_path,
                    local_path=plot_local_path,
                )

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
        collection_local_path = IncrementalPathGenerator.format_path(
            dir_path=export_dir,
            next_idx=iteration,
        )
        os.makedirs(collection_local_path, exist_ok=True)
        return collection_local_path

    @staticmethod
    def _get_collection_backend_path(artifact_path: str, iteration: int) -> str:
        collection_backend_path = IncrementalPathGenerator.format_path(
            dir_path=artifact_path,
            next_idx=iteration,
        )
        return collection_backend_path

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return os.path.join("plot_collections", artifact_name)
