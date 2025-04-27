from abc import ABC, abstractmethod
from typing import Generic, Optional, Type, TypeVar
from uuid import uuid4

nativeClientT = TypeVar("nativeClientT")
trackingBackendT = TypeVar("trackingBackendT", bound="TrackingBackend")


class TrackingBackend(ABC, Generic[nativeClientT]):
    def __init__(self, native_client: nativeClientT):
        self._native_client = native_client

    @classmethod
    def build(cls: Type[trackingBackendT], experiment_id: Optional[str] = None) -> trackingBackendT:
        if experiment_id is None:
            experiment_id = cls._generate_random_id()
        native_client = cls._get_native_client(experiment_id=experiment_id)
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
    def experiment_is_active(self) -> bool: ...

    @property
    @abstractmethod
    def experiment_id(self) -> str: ...

    @abstractmethod
    def _start_experiment(self, experiment_id: str): ...

    @classmethod
    @abstractmethod
    def _stop_experiment(cls, native_client: nativeClientT): ...

    @classmethod
    @abstractmethod
    def _get_native_client(cls, experiment_id: str) -> nativeClientT: ...

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
