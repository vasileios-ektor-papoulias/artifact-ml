from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Generic, Iterator, Optional, Type, TypeVar
from uuid import uuid4

NativeRunT = TypeVar("NativeRunT")
RunAdapterT = TypeVar("RunAdapterT", bound="RunAdapter")


class InactiveRunError(Exception):
    def __init__(self, message: str = "Run is inactive"):
        super().__init__(message)


class RunAdapter(ABC, Generic[NativeRunT]):
    def __init__(self, native_run: NativeRunT):
        self._native_run = native_run

    @classmethod
    def build(
        cls: Type[RunAdapterT], experiment_id: str, run_id: Optional[str] = None
    ) -> RunAdapterT:
        if run_id is None:
            run_id = cls._generate_random_id()
        native_run = cls._build_native_run(experiment_id=experiment_id, run_id=run_id)
        run_adapter = cls(native_run=native_run)
        return run_adapter

    @classmethod
    def from_native_run(cls: Type[RunAdapterT], native_run: NativeRunT) -> RunAdapterT:
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

    @classmethod
    @abstractmethod
    def _build_native_run(cls, experiment_id: str, run_id: str) -> NativeRunT: ...

    @contextmanager
    def native(self) -> Iterator[NativeRunT]:
        yield self._native_run

    @staticmethod
    def _generate_random_id() -> str:
        return str(uuid4())
