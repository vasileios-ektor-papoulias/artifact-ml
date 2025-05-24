from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np

from artifact_torch.base.model.base import Model


class ModelTracker(ABC):
    def __init__(self, score_key: str):
        self._score_key = score_key
        self.reset()

    @property
    def score_key(self) -> str:
        return self._score_key

    @property
    def best_model_state(self) -> Dict[str, Any]:
        return self._best_model_state

    @property
    def best_score(self) -> float:
        return self._best_score

    @property
    def best_epoch(self) -> int:
        return self._best_epoch

    @staticmethod
    @abstractmethod
    def _new_score_is_best(new_score: float, best_score: float) -> bool: ...

    @staticmethod
    @abstractmethod
    def _get_initial_best_score() -> float: ...

    def update(self, model: Model[Any, Any], score: float, epoch: int):
        if self.__new_score_is_best(new_score=score):
            self._best_score = score
            self._best_model_state = model.state_dict()
            self._best_epoch = epoch

    def reset(self):
        self._best_model_state = {}
        self._best_score = self._get_initial_best_score()
        self._best_epoch = 0

    def __new_score_is_best(self, new_score: float) -> bool:
        if np.isnan(self._best_score):
            return True
        return self._new_score_is_best(new_score=new_score, best_score=self._best_score)
