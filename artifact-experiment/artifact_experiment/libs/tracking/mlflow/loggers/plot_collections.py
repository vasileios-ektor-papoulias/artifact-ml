import io
import os
import tempfile
import zipfile
from typing import Dict, List

from matplotlib.figure import Figure
from mlflow.entities import FileInfo

from artifact_experiment.libs.tracking.mlflow.adapter import (
    MlflowRunAdapter,
)
from artifact_experiment.libs.tracking.mlflow.loggers.base import MlflowArtifactLogger
from artifact_experiment.libs.utils.filesystem import IncrementalPathGenerator


class MLFlowPlotCollectionLogger(MlflowArtifactLogger[Dict[str, Figure]]):
    _fmt = "zip"

    def _log(self, path: str, artifact: Dict[str, Figure]):
        ls_history = self._get_plot_collection_history(run=self._run, path=path)
        next_step = self._get_next_step_from_history(ls_history=ls_history)
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = IncrementalPathGenerator.format_path(
                dir_path=temp_dir,
                next_idx=next_step,
                fmt=self._fmt,
            )
            with zipfile.ZipFile(file=local_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                for name, fig in artifact.items():
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight")
                    buf.seek(0)
                    zf.writestr(f"{name}.png", buf.read())
            self._run.upload(
                backend_path=path,
                local_path=local_path,
            )

    @staticmethod
    def _get_plot_collection_history(run: MlflowRunAdapter, path: str) -> List[FileInfo]:
        ls_history = run.get_ls_artifact_info(backend_path=path)
        return ls_history

    @staticmethod
    def _get_next_step_from_history(ls_history: List[FileInfo]) -> int:
        return 1 + len(ls_history)

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return os.path.join("plot_collections", artifact_name)
