from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

StopperUpdateDataT = TypeVar("StopperUpdateDataT", bound="StopperUpdateData")


@dataclass
class StopperUpdateData:
    n_epochs_elapsed: int


class EarlyStopper(ABC, Generic[StopperUpdateDataT]):
    def __init__(self, max_n_epochs: Optional[int] = None):
        super().__init__()
        self._stopped: bool = False
        self._n_epochs_elapsed: int = 0
        self._max_n_epochs = max_n_epochs

    @property
    def stopped(self) -> bool:
        return self._stopped

    @property
    def n_epochs_elapsed(self) -> int:
        return self._n_epochs_elapsed

    @property
    def max_n_epochs(self) -> Optional[int]:
        return self._max_n_epochs

    @abstractmethod
    def stopping_condition_met(self) -> bool:
        pass

    def update(self, update_data: StopperUpdateDataT):
        self._n_epochs_elapsed = update_data.n_epochs_elapsed
        self._stopped = self.stopping_condition_met()
        self._enforce_max_epochs()

    def _enforce_max_epochs(self) -> None:
        if self._max_n_epochs is not None and self._n_epochs_elapsed >= self._max_n_epochs:
            self._stopped = True
