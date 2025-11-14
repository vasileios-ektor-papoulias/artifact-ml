from abc import abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Generic, List, Optional, TypeVar

from artifact_torch._base.components.early_stopping.stopper import EarlyStopper, StopperUpdateData

PatienceStopperCriterionT = TypeVar("PatienceStopperCriterionT", bound="PatienceStopperCriterion")


@dataclass
class PatienceStopperCriterion: ...


@dataclass
class PatienceStopperUpdateData(StopperUpdateData, Generic[PatienceStopperCriterionT]):
    criterion: PatienceStopperCriterionT


class PatienceStopper(
    EarlyStopper[PatienceStopperUpdateData[PatienceStopperCriterionT]],
    Generic[PatienceStopperCriterionT],
):
    def __init__(self, patience: int, max_n_epochs: Optional[int] = None):
        super().__init__(max_n_epochs=max_n_epochs)
        self._patience = patience
        self._criterion_history = deque[PatienceStopperCriterionT](maxlen=patience + 1)

    @property
    def patience(self) -> int:
        return self._patience

    @property
    def criterion_history(self) -> List[PatienceStopperCriterionT]:
        return list(self._criterion_history)

    @property
    def latest_criterion(self) -> Optional[PatienceStopperCriterionT]:
        if not self._queue_is_empty():
            return self._criterion_history[-1]

    @property
    def earliest_criterion(self) -> Optional[PatienceStopperCriterionT]:
        if not self._queue_is_empty():
            return self._criterion_history[0]

    @abstractmethod
    def stopping_condition_met(self) -> bool:
        pass

    def _update(self, update_data: PatienceStopperUpdateData):
        super()._update(update_data=update_data)
        self._criterion_history.append(update_data.criterion)
        self._current_criterion = update_data.criterion

    def _queue_is_empty(self) -> bool:
        return len(self._criterion_history) == 0

    def _queue_is_full(self) -> bool:
        return len(self._criterion_history) == self._patience + 1
