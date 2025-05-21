from typing import List

from artifact_experiment.base.tracking.adapter import RunAdapter


class DummyrunClient:
    def __init__(self, experiment_id: str, run_id: str):
        self._is_active = True
        self._experiment_id = experiment_id
        self._run_id = run_id
        self._ls_scores = []

    @property
    def experiment_id(self) -> str:
        return str(self._experiment_id)

    @property
    def run_id(self) -> str:
        return str(self._run_id)

    @property
    def is_active(self) -> bool:
        return self._is_active

    @is_active.setter
    def is_active(self, is_active: bool):
        self._is_active = is_active

    @property
    def ls_scores(self) -> List[float]:
        return self._ls_scores


class DummyTrackingBackend(RunAdapter[DummyrunClient]):
    @property
    def experiment_id(self) -> str:
        return self._native_run.experiment_id

    @property
    def run_id(self) -> str:
        return self._native_run.run_id

    @property
    def is_active(self) -> bool:
        return self._native_run.is_active

    @classmethod
    def _build_native_run(cls, experiment_id: str, run_id: str) -> DummyrunClient:
        return DummyrunClient(experiment_id=experiment_id, run_id=run_id)

    def _start(self, run_id: str):
        if self.run_id != run_id:
            self._native_run = self._build_native_run(
                experiment_id=self.experiment_id, run_id=run_id
            )
        else:
            self._native_run.is_active = True

    def stop(self):
        self._native_run.is_active = False
