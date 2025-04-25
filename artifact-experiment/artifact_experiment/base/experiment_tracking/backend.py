from abc import ABC, abstractmethod
from typing import Generic, Optional, Type, TypeVar
from uuid import uuid4


class NoNativeClient:
    pass


NO_NATIVE_CLIENT = NoNativeClient()

nativeClientT = TypeVar("nativeClientT")
trackingBackendT = TypeVar("trackingBackendT", bound="ExperimentTrackingBackend")


class ExperimentTrackingBackend(ABC, Generic[nativeClientT]):
    def __init__(self, native_client: nativeClientT):
        self._native_client = native_client

    @classmethod
    @abstractmethod
    def build(cls: Type[trackingBackendT]) -> trackingBackendT: ...

    @property
    @abstractmethod
    def is_active(self) -> bool:
        pass

    @property
    @abstractmethod
    def id(self) -> Optional[str]:
        pass

    @property
    @abstractmethod
    def native_client(self) -> nativeClientT:
        pass

    @abstractmethod
    def _start(self, experiment_id: str):
        pass

    @abstractmethod
    def _stop(self):
        pass

    def start(self, experiment_id: Optional[str] = None) -> str:
        if experiment_id is None:
            experiment_id = str(uuid4())
        self._start(experiment_id=experiment_id)
        return experiment_id

    def stop(self):
        if not self.is_active:
            self._stop()
