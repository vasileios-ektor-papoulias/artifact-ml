from abc import ABC, abstractmethod
from typing import Generic, Optional, Type, TypeVar
from uuid import uuid4


class NoNativeClient:
    pass


NO_NATIVE_CLIENT = NoNativeClient()

nativeClientT = TypeVar("nativeClientT")
trackingBackendT = TypeVar("trackingBackendT", bound="TrackingBackend")


class TrackingBackend(ABC, Generic[nativeClientT]):
    def __init__(self, native_client: nativeClientT):
        self._native_client = native_client

    @classmethod
    @abstractmethod
    def build(cls: Type[trackingBackendT]) -> trackingBackendT: ...

    @property
    @abstractmethod
    def experiment_is_active(self) -> bool: ...

    @property
    @abstractmethod
    def experiment_id(self) -> Optional[str]: ...

    @property
    @abstractmethod
    def native_client(self) -> nativeClientT: ...

    @abstractmethod
    def _start_experiment(self, experiment_id: str): ...

    @abstractmethod
    def _stop_experiment(self): ...

    def start_experiment(self, experiment_id: Optional[str] = None) -> str:
        if experiment_id is None:
            experiment_id = str(uuid4())
        self._start_experiment(experiment_id=experiment_id)
        return experiment_id

    def experiment(self):
        if not self.experiment_is_active:
            self._stop_experiment()
