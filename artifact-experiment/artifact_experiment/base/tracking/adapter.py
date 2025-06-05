from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Generic, Iterator, Optional, Type, TypeVar
from uuid import uuid4

nativeRunT = TypeVar("nativeRunT")
runAdapterT = TypeVar("runAdapterT", bound="RunAdapter")


class InactiveRunError(Exception):
    pass


class RunAdapter(ABC, Generic[nativeRunT]):
    def __init__(self, native_run: nativeRunT):
        self._native_run = native_run

    @classmethod
    def build(
        cls: Type[runAdapterT], experiment_id: str, run_id: Optional[str] = None
    ) -> runAdapterT:
        if run_id is None:
            run_id = cls._generate_random_id()
        native_run = cls._build_native_run(experiment_id=experiment_id, run_id=run_id)
        run_adapter = cls(native_run=native_run)
        return run_adapter

    @classmethod
    def from_native_run(cls: Type[runAdapterT], native_run: nativeRunT) -> runAdapterT:
        run_adapter = cls(native_run=native_run)
        return run_adapter

    @property
    @abstractmethod
    def experiment_id(self) -> str: ...

    @property
    @abstractmethod
    def run_id(self) -> str: ...

    @property
    @abstractmethod
    def is_active(self) -> bool: ...

    @abstractmethod
    def stop(self): ...

    @abstractmethod
    def upload(self, path_source: str, dir_target: str): ...

    @classmethod
    @abstractmethod
    def _build_native_run(cls, experiment_id: str, run_id: str) -> nativeRunT: ...

    @contextmanager
    def native(self) -> Iterator[nativeRunT]:
        try:
            yield self._native_run
        finally:
            pass

    @staticmethod
    def _generate_random_id() -> str:
        return str(uuid4())
