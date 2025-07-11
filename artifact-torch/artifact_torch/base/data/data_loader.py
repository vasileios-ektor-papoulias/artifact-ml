from typing import Generic, Iterator, TypeVar, Union

import torch
from torch.utils.data import DataLoader as NativeDataLoader

from artifact_torch.base.data.dataset import Dataset, IterableDataset
from artifact_torch.base.data.device_manager import DeviceManager
from artifact_torch.base.model.io import ModelInput

ModelInputT = TypeVar("ModelInputT", bound=ModelInput)


class DataLoader(NativeDataLoader, Generic[ModelInputT]):
    def __init__(
        self, dataset: Union[Dataset[ModelInputT], IterableDataset[ModelInputT]], *args, **kwargs
    ):
        super().__init__(dataset, *args, **kwargs)
        self._device = torch.device("cpu")

    def __iter__(self) -> Iterator[ModelInputT]:
        for batch in super().__iter__():
            batch = DeviceManager.move_to_device(data=batch, device=self._device)
            yield batch

    @property
    def device(self) -> torch.device:
        return self._device

    @device.setter
    def device(self, device: torch.device):
        self._device = device
