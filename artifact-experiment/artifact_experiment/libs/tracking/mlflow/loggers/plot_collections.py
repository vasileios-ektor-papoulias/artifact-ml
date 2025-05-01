import io
import tempfile
import zipfile
from typing import Dict

from matplotlib.figure import Figure

from artifact_experiment.libs.tracking.mlflow.backend import (
    InactiveMlflowRunError,
    MlflowBackend,
)
from artifact_experiment.libs.tracking.mlflow.loggers.base import MlflowArtifactLogger
from artifact_experiment.libs.utils.filesystem import IncrementalPathGenerator


class MLFlowPlotCollectionLogger(MlflowArtifactLogger[Dict[str, Figure]]):
    _fmt = "zip"

    def __init__(self, backend: MlflowBackend):
        self._backend = backend

    def _log(self, path: str, artifact: Dict[str, Figure]):
        if not self._backend.run_is_active:
            raise InactiveMlflowRunError("No active experiment.")

        client = self._backend.native_client.client
        run_id = self._backend.experiment_id
        remote_dir = path
        with tempfile.TemporaryDirectory() as td:
            archive_path = IncrementalPathGenerator.generate_mlflow(
                client=client,
                run_id=run_id,
                remote_path=remote_dir,
                dir_local=td,
                fmt=self._fmt,
            )
            with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                for name, fig in artifact.items():
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight")
                    buf.seek(0)
                    zf.writestr(f"{name}.png", buf.read())
            client.log_artifact(
                run_id=run_id,
                local_path=archive_path,
                artifact_path=remote_dir,
            )

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return f"plot_collections/{artifact_name}"
