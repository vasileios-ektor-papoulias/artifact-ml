from abc import abstractmethod
from dataclasses import dataclass

from artifact_torch.base.components.model_tracking.tracker import (
    ModelTracker,
    ModelTrackingCriterion,
)


@dataclass
class SingleScoreCriterion(ModelTrackingCriterion):
    score: float


class SingleScoreTracker(ModelTracker[SingleScoreCriterion]):
    def __init__(self):
        self.reset()

    @property
    def best_score(self) -> float:
        return self._best_criterion.score

    @staticmethod
    @abstractmethod
    def _new_score_is_best(new_score: float, best_score: float) -> bool: ...

    @staticmethod
    @abstractmethod
    def _get_initial_best_score() -> float: ...

    @classmethod
    def _new_criterion_is_best(
        cls, new_criterion: SingleScoreCriterion, best_criterion: SingleScoreCriterion
    ) -> bool:
        new_criterion_is_best = cls._new_score_is_best(
            new_score=new_criterion.score, best_score=best_criterion.score
        )
        return new_criterion_is_best

    @classmethod
    def _get_initial_best_criterion(cls) -> SingleScoreCriterion:
        initial_best_score = cls._get_initial_best_score()
        initial_best_criterion = SingleScoreCriterion(score=initial_best_score)
        return initial_best_criterion
