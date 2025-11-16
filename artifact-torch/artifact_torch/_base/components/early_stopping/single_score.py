from abc import abstractmethod
from typing import List, Optional

from artifact_torch._base.components.early_stopping.patience import PatienceStopper


class SingleScoreStopper(PatienceStopper[float]):
    @property
    def score_history(self) -> List[float]:
        return [criterion for criterion in self.criterion_history]

    @property
    def latest_score(self) -> Optional[float]:
        if self.latest_criterion is not None:
            return self.latest_criterion

    @property
    def earliest_score(self) -> Optional[float]:
        if self.earliest_criterion is not None:
            return self.earliest_criterion

    @property
    def min_score(self) -> Optional[float]:
        if not self._queue_is_empty():
            return min(self.score_history)

    @property
    def max_score(self) -> Optional[float]:
        if not self._queue_is_empty():
            return max(self.score_history)

    @abstractmethod
    def stopping_condition_met(self) -> bool:
        pass
