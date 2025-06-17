from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from artifact_torch.core.components.early_stopping.patience import (
    PatienceStopper,
    PatienceStopperCriterion,
    PatienceStopperUpdateData,
)


@dataclass
class SingleScoreCriterion(PatienceStopperCriterion):
    score: float


SingleScoreUpdateData = PatienceStopperUpdateData[SingleScoreCriterion]


class SingleScoreStopper(PatienceStopper[SingleScoreCriterion]):
    @property
    def score_history(self) -> List[float]:
        return [criterion.score for criterion in self.criterion_history]

    @property
    def latest_score(self) -> Optional[float]:
        if self.latest_criterion is not None:
            return self.latest_criterion.score

    @property
    def earliest_score(self) -> Optional[float]:
        if self.earliest_criterion is not None:
            return self.earliest_criterion.score

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
