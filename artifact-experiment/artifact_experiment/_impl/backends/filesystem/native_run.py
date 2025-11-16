import os
from pathlib import Path

from artifact_experiment._base.tracking.backend.adapter import InactiveRunError


class InactiveFilesystemRunError(InactiveRunError):
    pass


class FilesystemRun:
    _open_button_description = "Open Run Dir"

    def __init__(self, experiment_id: str, run_id: str):
        self._experiment_id = experiment_id
        self._start(run_id=run_id)

    @property
    def experiment_id(self) -> str:
        return self._experiment_id

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def is_active(self) -> bool:
        return self._is_active

    @property
    def experiment_dir(self) -> str:
        return os.path.join(str(self._root_dir), self._experiment_id)

    @property
    def run_dir(self) -> str:
        return os.path.join(self.experiment_dir, self._run_id)

    @property
    def _root_dir(self) -> Path:
        return Path.home() / "artifact_ml"

    def stop(self):
        self._is_active = False

    def _start(self, run_id: str):
        self._is_active = True
        self._run_id = run_id
        self._create_run_dir()
        self._print_run_url()

    def _create_run_dir(self):
        os.makedirs(name=self.run_dir, exist_ok=True)

    def _print_run_url(self):
        try:
            from artifact_experiment._utils.ui.dir_open_button import DirectoryOpenButton
        except ImportError:
            print(f"Run directory created at: {self.run_dir}")
        else:
            DirectoryOpenButton(path=self.run_dir, description=self._open_button_description)
