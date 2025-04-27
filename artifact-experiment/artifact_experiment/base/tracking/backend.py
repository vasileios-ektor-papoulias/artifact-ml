from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar
from uuid import uuid4

nativeClientT = TypeVar("nativeClientT")


class TrackingBackend(ABC, Generic[nativeClientT]):
    def __init__(self, native_client: nativeClientT):
        self._native_client = native_client

    @property
    @abstractmethod
    def experiment_is_active(self) -> bool: ...

    @property
    @abstractmethod
    def experiment_id(self) -> str: ...

    @property
    @abstractmethod
    def native_client(self) -> nativeClientT: ...

    @abstractmethod
    def _start_experiment(self, experiment_id: str): ...

    @classmethod
    @abstractmethod
    def _stop_experiment(cls, native_client: nativeClientT): ...

    def start_experiment(self, experiment_id: Optional[str] = None) -> str:
        if experiment_id is None:
            experiment_id = self._generate_random_id()
        self._start_experiment(experiment_id=experiment_id)
        return experiment_id

    def stop_experiment(self):
        if self.experiment_is_active:
            self._stop_experiment(native_client=self.native_client)

    @staticmethod
    def _generate_random_id() -> str:
        experiment_id = str(uuid4())
        return experiment_id
