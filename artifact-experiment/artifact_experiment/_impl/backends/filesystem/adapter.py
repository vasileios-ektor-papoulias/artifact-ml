import os
import shutil

from artifact_experiment._base.primitives.file import File
from artifact_experiment._base.tracking.backend.adapter import InactiveRunError, RunAdapter
from artifact_experiment._impl.backends.filesystem.native_run import FilesystemRun


class InactiveFilesystemRunError(InactiveRunError):
    pass


class FilesystemRunAdapter(RunAdapter[FilesystemRun]):
    @property
    def experiment_id(self) -> str:
        return self._native_run.experiment_id

    @property
    def run_id(self) -> str:
        return self._native_run.run_id

    @property
    def is_active(self) -> bool:
        return self._native_run.is_active

    @property
    def experiment_dir(self) -> str:
        return self._native_run.experiment_dir

    @property
    def run_dir(self) -> str:
        return self._native_run.run_dir

    def stop(self):
        self._native_run.stop()

    def log_file(self, backend_dir: str, file: File):
        if not self.is_active:
            raise InactiveFilesystemRunError("Cannot log file: run is not active.")
        dir_target_abs = os.path.join(self.run_dir, backend_dir)
        os.makedirs(dir_target_abs, exist_ok=True)
        shutil.copy2(file.path_source, dir_target_abs)

    @classmethod
    def _build_native_run(cls, experiment_id: str, run_id: str) -> FilesystemRun:
        return FilesystemRun(experiment_id=experiment_id, run_id=run_id)
