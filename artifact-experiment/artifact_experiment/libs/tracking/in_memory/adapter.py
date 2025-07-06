from typing import Dict

from artifact_experiment.base.tracking.adapter import RunAdapter
from artifact_experiment.libs.tracking.in_memory.native_run import (
    InMemoryNativeRun,
)


class InMemoryTrackingAdapter(RunAdapter[InMemoryNativeRun]):
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
    def uploaded_files(self):
        return self._native_run.uploaded_files

    @classmethod
    def _build_native_run(cls, experiment_id: str, run_id: str) -> InMemoryNativeRun:
        return InMemoryNativeRun(experiment_id=experiment_id, run_id=run_id)

    def _start(self, run_id: str):
        if self.run_id != run_id:
            self._native_run = self._build_native_run(
                experiment_id=self.experiment_id, run_id=run_id
            )
        else:
            self._native_run.is_active = True

    def stop(self):
        self._native_run.is_active = False

    def upload(self, path_source: str, dir_target: str):
        upload_entry = self._format_upload_entry(path_source=path_source, dir_target=dir_target)
        self._native_run.uploaded_files.append(upload_entry)

    @staticmethod
    def _format_upload_entry(path_source: str, dir_target: str) -> Dict[str, str]:
        dict_upload_entry = {"path_source": path_source, "dir_target": dir_target}
        return dict_upload_entry
