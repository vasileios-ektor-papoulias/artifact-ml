import io
import tempfile
import zipfile
from typing import Dict

from matplotlib.figure import Figure

from artifact_experiment.libs.tracking.mlflow.loggers.base import MlflowArtifactLogger
from artifact_experiment.libs.utils.filesystem import IncrementalPathGenerator


class MLFlowPlotCollectionLogger(MlflowArtifactLogger[Dict[str, Figure]]):
    _fmt = "zip"

    def _log(self, path: str, artifact: Dict[str, Figure]):
        ls_existing_filepaths = [
            str(info.path) for info in self._run.get_ls_artifact_info(backend_path=path)
        ]
        with tempfile.TemporaryDirectory() as td:
            local_path = IncrementalPathGenerator.generate_from_existing_filepaths(
                ls_existing_filepaths=ls_existing_filepaths,
                dir_local=td,
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

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return f"plot_collections/{artifact_name}"
