from abc import ABC, abstractmethod
from typing import Generic, Iterator, TypeVar

from torch.utils.data import Dataset as NativeDataset
from torch.utils.data import IterableDataset as NativeIterableDataset

from artifact_torch.base.model.io import ModelInput

ModelInputT = TypeVar("ModelInputT", bound=ModelInput)


class Dataset(NativeDataset, ABC, Generic[ModelInputT]):
    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> ModelInputT: ...


class IterableDataset(NativeIterableDataset, ABC, Generic[ModelInputT]):
    @abstractmethod
    def __iter__(self) -> Iterator[ModelInputT]: ...
