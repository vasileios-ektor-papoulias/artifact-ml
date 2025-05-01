from abc import ABC, abstractmethod
from typing import Generic, Optional, Type, TypeVar
from uuid import uuid4

nativeClientT = TypeVar("nativeClientT")
trackingBackendT = TypeVar("trackingBackendT", bound="TrackingBackend")


class TrackingBackend(ABC, Generic[nativeClientT]):
    def __init__(self, native_client: nativeClientT):
        self._native_client = native_client

    @classmethod
    def build(
        cls: Type[trackingBackendT], experiment_id: str, run_id: Optional[str] = None
    ) -> trackingBackendT:
        if run_id is None:
            run_id = cls._generate_random_id()
        native_client = cls._get_native_client(experiment_id=experiment_id, run_id=run_id)
        backend = cls(native_client=native_client)
        return backend

    @classmethod
    def from_native_client(
        cls: Type[trackingBackendT], native_client: nativeClientT
    ) -> trackingBackendT:
        backend = cls(native_client=native_client)
        return backend

    @property
    def native_client(self) -> nativeClientT:
        return self._native_client

    @property
    @abstractmethod
    def experiment_id(self) -> str: ...

    @property
    @abstractmethod
    def run_id(self) -> str: ...

    @property
    @abstractmethod
    def run_is_active(self) -> bool: ...

    @classmethod
    @abstractmethod
    def _get_native_client(cls, experiment_id: str, run_id: str) -> nativeClientT: ...

    @abstractmethod
    def _start_run(self, run_id: str): ...

    @abstractmethod
    def _stop_run(self): ...

    def start_run(self, run_id: Optional[str] = None) -> str:
        if run_id is None:
            run_id = self._generate_random_id()
        self._start_run(run_id=run_id)
        return run_id

    def stop_run(self):
        if self.run_is_active:
            self._stop_run()

    @staticmethod
    def _generate_random_id() -> str:
        experiment_id = str(uuid4())
        return experiment_id
