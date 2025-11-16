from abc import ABC, abstractmethod
from typing import Generic, Iterator, TypeVar

from torch.utils.data import Dataset as NativeDataset
from torch.utils.data import IterableDataset as NativeIterableDataset

from artifact_torch._base.model.io import ModelInput

ModelInputTCov = TypeVar("ModelInputTCov", bound=ModelInput, covariant=True)


class Dataset(NativeDataset, ABC, Generic[ModelInputTCov]):
    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> ModelInputTCov: ...


class IterableDataset(NativeIterableDataset, ABC, Generic[ModelInputTCov]):
    @abstractmethod
    def __iter__(self) -> Iterator[ModelInputTCov]: ...
