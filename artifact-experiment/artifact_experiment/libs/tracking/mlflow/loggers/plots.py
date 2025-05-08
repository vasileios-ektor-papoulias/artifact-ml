import tempfile

from matplotlib.figure import Figure

from artifact_experiment.libs.tracking.mlflow.adapter import InactiveMlflowRunError
from artifact_experiment.libs.tracking.mlflow.loggers.base import MlflowArtifactLogger
from artifact_experiment.libs.utils.filesystem import IncrementalPathGenerator


class MLFlowPlotLogger(MlflowArtifactLogger[Figure]):
    _fmt = "png"

    def _log(self, path: str, artifact: Figure):
        if not self._run.is_active:
            raise InactiveMlflowRunError("No active experiment.")
        ls_existing_filepaths = [
            str(info.path) for info in self._run.get_ls_artifact_info(backend_path=path)
        ]
        with tempfile.TemporaryDirectory() as td:
            local_path = IncrementalPathGenerator.generate_from_existing_filepaths(
                ls_existing_filepaths=ls_existing_filepaths,
                dir_local=td,
                fmt=self._fmt,
            )
            artifact.savefig(fname=local_path, format="png", bbox_inches="tight")
            self._run.upload(
                backend_path=path,
                local_path=local_path,
            )

    @classmethod
    def _get_relative_path(cls, artifact_name: str) -> str:
        return f"plots/{artifact_name}"
