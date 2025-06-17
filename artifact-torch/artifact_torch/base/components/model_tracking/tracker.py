from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, TypeVar

from artifact_torch.base.model.base import Model

ModelTrackingCriterionT = TypeVar("ModelTrackingCriterionT", bound="ModelTrackingCriterion")


@dataclass
class ModelTrackingCriterion: ...


@dataclass
class NoTrackingCriterion(ModelTrackingCriterion): ...


class ModelTracker(ABC, Generic[ModelTrackingCriterionT]):
    def __init__(self):
        self.reset()

    @property
    def best_model_state(self) -> Dict[str, Any]:
        return self._best_model_state

    @property
    def best_criterion(self) -> ModelTrackingCriterionT:
        return self._best_criterion

    @property
    def best_epoch(self) -> int:
        return self._best_epoch

    @classmethod
    @abstractmethod
    def _new_criterion_is_best(
        cls, new_criterion: ModelTrackingCriterionT, best_criterion: ModelTrackingCriterionT
    ) -> bool: ...

    @classmethod
    @abstractmethod
    def _get_initial_best_criterion(cls) -> ModelTrackingCriterionT: ...

    def update(self, model: Model[Any, Any], criterion: ModelTrackingCriterionT, epoch: int):
        if self._new_criterion_is_best(
            new_criterion=criterion, best_criterion=self._best_criterion
        ):
            self._best_score = criterion
            self._best_model_state = model.state_dict()
            self._best_epoch = epoch

    def reset(self):
        self._best_model_state = {}
        self._best_criterion = self._get_initial_best_criterion()
        self._best_epoch = 0
